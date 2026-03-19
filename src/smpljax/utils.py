from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np


Array = jnp.ndarray


@dataclass(frozen=True)
class SMPLModelData:
    """Packed arrays needed for SMPL-family forward passes."""

    v_template: Array
    shapedirs: Array
    posedirs: Array
    j_regressor: Array
    parents: Array
    lbs_weights: Array
    num_betas: int | None = None
    num_expression_coeffs: int = 0
    num_body_joints: int | None = None
    num_hand_joints: int = 0
    num_face_joints: int = 0
    model_family: str | None = None
    model_variant: str | None = None
    gender: str | None = None
    pose_mean: Array | None = None
    use_pca: bool = False
    left_hand_components: Array | None = None
    right_hand_components: Array | None = None
    extra_vertex_ids: list[int] | None = None
    joint_mapper: Callable[[Array], Array] | None = None
    faces_tensor: Array | None = None
    lmk_faces_idx: Array | None = None
    lmk_bary_coords: Array | None = None
    dynamic_lmk_faces_idx: Array | None = None
    dynamic_lmk_bary_coords: Array | None = None
    neck_kin_chain: Array | None = None
    use_face_contour: bool = False


def as_jax_array(x: Any, dtype: jnp.dtype | None = None) -> Array:
    if isinstance(x, jnp.ndarray):
        return x.astype(dtype) if dtype is not None else x
    if isinstance(x, np.ndarray):
        return jnp.asarray(x, dtype=dtype)
    return jnp.asarray(x, dtype=dtype)


def normalize_pose_input(pose: Array) -> Array:
    """Ensure pose is shaped (..., J, 3)."""
    pose = as_jax_array(pose)
    if pose.ndim < 2:
        raise ValueError("pose must have at least 2 dims")
    if pose.shape[-1] == 3:
        return pose
    if pose.shape[-1] % 3 != 0:
        raise ValueError("pose last dim must be 3 or multiple of 3")
    return pose.reshape(*pose.shape[:-1], pose.shape[-1] // 3, 3)


def zeros_pose(batch: int, joints: int, dtype: jnp.dtype) -> Array:
    return jnp.zeros((batch, joints, 3), dtype=dtype)


def find_joint_kin_chain(joint_id: int, kinematic_tree: Array) -> list[int]:
    kin_chain: list[int] = []
    curr_idx = int(joint_id)
    parents = np.asarray(kinematic_tree).reshape(-1)
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = int(parents[curr_idx])
    return kin_chain
