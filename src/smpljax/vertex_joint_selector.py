from __future__ import annotations

import jax.numpy as jnp

from .vertex_ids import VERTEX_IDS


Array = jnp.ndarray


def select_vertices(vertices: Array, vertex_ids: list[int] | Array) -> Array:
    """Select semantic vertices from a mesh tensor of shape (..., V, 3)."""
    ids = jnp.asarray(vertex_ids, dtype=jnp.int32)
    return vertices[..., ids, :]


def build_smpl_extra_joint_ids(
    model_type: str,
    use_hands: bool = True,
    use_feet_keypoints: bool = True,
) -> list[int]:
    mt = model_type.lower()
    if mt not in VERTEX_IDS:
        return []
    vid = VERTEX_IDS[mt]

    out = [vid["nose"], vid["reye"], vid["leye"], vid["rear"], vid["lear"]]
    if use_feet_keypoints:
        out.extend([vid["LBigToe"], vid["LSmallToe"], vid["LHeel"], vid["RBigToe"], vid["RSmallToe"], vid["RHeel"]])
    if use_hands:
        out.extend(
            [
                vid["lthumb"],
                vid["lindex"],
                vid["lmiddle"],
                vid["lring"],
                vid["lpinky"],
                vid["rthumb"],
                vid["rindex"],
                vid["rmiddle"],
                vid["rring"],
                vid["rpinky"],
            ]
        )
    return out
