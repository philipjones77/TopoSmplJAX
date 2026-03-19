from __future__ import annotations

import jax
import jax.numpy as jnp


Array = jnp.ndarray


def _normalize_posedirs(posedirs: Array) -> Array:
    """Return posedirs as (num_pose_basis, V*3)."""
    if posedirs.ndim == 2:
        return posedirs
    if posedirs.ndim == 3:
        num_pose_basis = posedirs.shape[-1]
        return jnp.reshape(posedirs, (-1, num_pose_basis)).T
    raise ValueError(f"Unsupported posedirs rank: {posedirs.ndim}")


def batch_rodrigues(theta: Array, eps: float = 1e-8) -> Array:
    """Convert axis-angle vectors to rotation matrices."""
    angle = jnp.linalg.norm(theta + eps, axis=-1, keepdims=True)
    axis = theta / angle
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = jnp.zeros_like(x)
    k = jnp.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        axis=-1,
    ).reshape(theta.shape[:-1] + (3, 3))
    ident = jnp.broadcast_to(jnp.eye(3), theta.shape[:-1] + (3, 3))
    sin = jnp.sin(angle)[..., None]
    cos = jnp.cos(angle)[..., None]
    return ident + sin * k + (1.0 - cos) * (k @ k)


def blend_shapes(betas: Array, shapedirs: Array) -> Array:
    """Compute per-vertex shape offsets."""
    return jnp.einsum("...b,vcb->...vc", betas, shapedirs)


def vertices2joints(j_regressor: Array, vertices: Array) -> Array:
    """Regress joints from mesh vertices."""
    return jnp.einsum("jv,...vc->...jc", j_regressor, vertices)


def _with_zeros(rot: Array, trans: Array) -> Array:
    n = rot.shape[:-2]
    upper = jnp.concatenate([rot, trans[..., None]], axis=-1)
    bottom = jnp.broadcast_to(jnp.array([0.0, 0.0, 0.0, 1.0], dtype=rot.dtype), n + (1, 4))
    return jnp.concatenate([upper, bottom], axis=-2)


def batch_rigid_transform(rot_mats: Array, joints: Array, parents: Array) -> tuple[Array, Array]:
    """Kinematic tree forward transform."""
    parents = jnp.asarray(parents, dtype=jnp.int32).reshape(-1)
    joints_rel = joints.copy()
    parent_ids = parents[1:]
    joints_rel = joints_rel.at[..., 1:, :].set(joints[..., 1:, :] - joints[..., parent_ids, :])

    transforms = _with_zeros(rot_mats, joints_rel)
    num_joints = int(rot_mats.shape[-3])
    transforms_global = jnp.zeros_like(transforms)
    transforms_global = transforms_global.at[..., 0, :, :].set(transforms[..., 0, :, :])

    def _body(i: int, tg: Array) -> Array:
        parent = parents[i]
        curr = jnp.einsum("...ij,...jk->...ik", tg[..., parent, :, :], transforms[..., i, :, :])
        return tg.at[..., i, :, :].set(curr)

    transforms_global = jax.lax.fori_loop(1, num_joints, _body, transforms_global)
    posed_joints = transforms_global[..., :3, 3]

    joints_h = jnp.concatenate([joints, jnp.ones_like(joints[..., :1])], axis=-1)[..., None]
    rel_joints = transforms_global @ joints_h
    transforms_global = transforms_global.at[..., :3, 3].add(-rel_joints[..., :3, 0])
    return posed_joints, transforms_global


def lbs(
    betas: Array,
    pose: Array,
    v_template: Array,
    shapedirs: Array,
    posedirs: Array,
    j_regressor: Array,
    parents: Array,
    lbs_weights: Array,
    pose2rot: bool = True,
) -> tuple[Array, Array]:
    """Linear blend skinning for SMPL-style models."""
    posedirs = _normalize_posedirs(posedirs)
    batch_size = betas.shape[0]
    v_shaped = v_template[None, ...] + blend_shapes(betas, shapedirs)
    joints = vertices2joints(j_regressor, v_shaped)

    if pose2rot:
        if pose.ndim == 2:
            pose = pose.reshape(batch_size, -1, 3)
        rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
    else:
        if pose.ndim == 4 and pose.shape[-2:] == (3, 3):
            rot_mats = pose
        elif pose.ndim == 3 and pose.shape[-1] == 9:
            rot_mats = pose.reshape(batch_size, -1, 3, 3)
        elif pose.ndim == 2 and (pose.shape[-1] % 9 == 0):
            rot_mats = pose.reshape(batch_size, -1, 3, 3)
        else:
            raise ValueError("pose2rot=False expects pose as (..., J, 3, 3), (..., J, 9), or (..., J*9)")

    pose_feature = (rot_mats[:, 1:, :, :] - jnp.eye(3, dtype=rot_mats.dtype)).reshape(batch_size, -1)
    pose_offsets = (pose_feature @ posedirs).reshape(batch_size, v_template.shape[0], 3)
    v_posed = v_shaped + pose_offsets

    joints_transformed, a_mats = batch_rigid_transform(rot_mats, joints, parents)
    t_mats = jnp.einsum("vj,bjkl->bvkl", lbs_weights, a_mats)
    v_posed_h = jnp.concatenate([v_posed, jnp.ones_like(v_posed[..., :1])], axis=-1)
    verts = jnp.einsum("bvkl,bvl->bvk", t_mats, v_posed_h)[..., :3]
    return verts, joints_transformed
