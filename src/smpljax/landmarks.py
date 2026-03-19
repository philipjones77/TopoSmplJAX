from __future__ import annotations

import jax.numpy as jnp

from .lbs import batch_rodrigues


Array = jnp.ndarray


def rot_mat_to_euler(rot_mats: Array) -> Array:
    sy = jnp.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] + rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return jnp.atan2(-rot_mats[:, 2, 0], sy)


def find_dynamic_lmk_idx_and_bcoords(
    vertices: Array,
    pose: Array,
    dynamic_lmk_faces_idx: Array,
    dynamic_lmk_bary_coords: Array,
    neck_kin_chain: Array,
    pose2rot: bool = True,
) -> tuple[Array, Array]:
    batch_size = vertices.shape[0]
    if pose2rot:
        aa_pose = pose[:, neck_kin_chain, :]
        rot_mats = batch_rodrigues(aa_pose.reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
    else:
        if pose.ndim != 4 or pose.shape[-2:] != (3, 3):
            raise ValueError("pose2rot=False expects pose to be shaped (B, J, 3, 3)")
        rot_mats = pose[:, neck_kin_chain, :, :]

    rel_rot_mat = jnp.broadcast_to(jnp.eye(3, dtype=vertices.dtype), (batch_size, 3, 3))
    for idx in range(rot_mats.shape[1]):
        rel_rot_mat = jnp.einsum("bij,bjk->bik", rot_mats[:, idx], rel_rot_mat)

    y_rot_angle = jnp.round(jnp.clip(-rot_mat_to_euler(rel_rot_mat) * 180.0 / jnp.pi, max=39)).astype(jnp.int32)
    neg_mask = (y_rot_angle < 0).astype(jnp.int32)
    mask = (y_rot_angle < -39).astype(jnp.int32)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle
    y_rot_angle = jnp.clip(y_rot_angle, 0, dynamic_lmk_faces_idx.shape[0] - 1)

    faces_idx = dynamic_lmk_faces_idx[y_rot_angle]
    bary = dynamic_lmk_bary_coords[y_rot_angle]
    if faces_idx.ndim == 1:
        faces_idx = faces_idx[:, None]
    if bary.ndim == 2:
        bary = bary[:, None, :]
    return faces_idx, bary


def vertices2landmarks(vertices: Array, faces: Array, lmk_faces_idx: Array, lmk_bary_coords: Array) -> Array:
    batch_size, num_verts = vertices.shape[:2]
    lmk_faces = faces[lmk_faces_idx.reshape(-1).astype(jnp.int32)].reshape(batch_size, -1, 3)
    lmk_faces = lmk_faces + (jnp.arange(batch_size, dtype=jnp.int32).reshape(-1, 1, 1) * num_verts)
    lmk_vertices = vertices.reshape(-1, 3)[lmk_faces].reshape(batch_size, -1, 3, 3)
    return jnp.einsum("blfi,blf->bli", lmk_vertices, lmk_bary_coords)
