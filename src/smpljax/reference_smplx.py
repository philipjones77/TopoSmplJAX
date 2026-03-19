from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch

from .utils import SMPLModelData


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    num_betas: int
    num_expression: int
    num_body_joints: int
    num_hand_joints: int
    num_face_joints: int


def infer_spec(ref_model: Any, model_type: str) -> ModelSpec:
    m = model_type.lower()
    if m == "smpl":
        return ModelSpec("smpl", 10, 0, int(ref_model.NUM_BODY_JOINTS), 0, 0)
    if m == "smplh":
        return ModelSpec("smplh", 10, 0, int(ref_model.NUM_BODY_JOINTS), int(ref_model.NUM_HAND_JOINTS), 0)
    if m == "smplx":
        return ModelSpec("smplx", 10, 10, int(ref_model.NUM_BODY_JOINTS), int(ref_model.NUM_HAND_JOINTS), 3)
    raise ValueError(f"Unsupported model type: {model_type}")


def to_model_data(ref_model: Any, spec: ModelSpec) -> SMPLModelData:
    shapedirs = ref_model.shapedirs
    if spec.num_expression > 0 and hasattr(ref_model, "expr_dirs"):
        shapedirs = torch.cat([ref_model.shapedirs, ref_model.expr_dirs], dim=-1)

    extra_ids = None
    if hasattr(ref_model, "vertex_joint_selector") and hasattr(ref_model.vertex_joint_selector, "extra_joints_idxs"):
        extra_ids = ref_model.vertex_joint_selector.extra_joints_idxs.detach().cpu().numpy().astype(np.int32).tolist()

    return SMPLModelData(
        v_template=jnp.asarray(ref_model.v_template.detach().cpu().numpy()),
        shapedirs=jnp.asarray(shapedirs.detach().cpu().numpy()),
        posedirs=jnp.asarray(ref_model.posedirs.detach().cpu().numpy()),
        j_regressor=jnp.asarray(ref_model.J_regressor.detach().cpu().numpy()),
        parents=jnp.asarray(ref_model.parents.detach().cpu().numpy()),
        lbs_weights=jnp.asarray(ref_model.lbs_weights.detach().cpu().numpy()),
        num_betas=spec.num_betas,
        num_expression_coeffs=spec.num_expression,
        num_body_joints=spec.num_body_joints,
        num_hand_joints=spec.num_hand_joints,
        num_face_joints=spec.num_face_joints,
        pose_mean=jnp.asarray(ref_model.pose_mean.detach().cpu().numpy()) if hasattr(ref_model, "pose_mean") else None,
        use_pca=False,
        extra_vertex_ids=extra_ids,
        faces_tensor=jnp.asarray(ref_model.faces_tensor.detach().cpu().numpy())
        if hasattr(ref_model, "faces_tensor")
        else None,
        lmk_faces_idx=jnp.asarray(ref_model.lmk_faces_idx.detach().cpu().numpy())
        if hasattr(ref_model, "lmk_faces_idx")
        else None,
        lmk_bary_coords=jnp.asarray(ref_model.lmk_bary_coords.detach().cpu().numpy())
        if hasattr(ref_model, "lmk_bary_coords")
        else None,
        dynamic_lmk_faces_idx=jnp.asarray(ref_model.dynamic_lmk_faces_idx.detach().cpu().numpy())
        if hasattr(ref_model, "dynamic_lmk_faces_idx")
        else None,
        dynamic_lmk_bary_coords=jnp.asarray(ref_model.dynamic_lmk_bary_coords.detach().cpu().numpy())
        if hasattr(ref_model, "dynamic_lmk_bary_coords")
        else None,
        neck_kin_chain=jnp.asarray(ref_model.neck_kin_chain.detach().cpu().numpy()) if hasattr(ref_model, "neck_kin_chain") else None,
        use_face_contour=bool(getattr(ref_model, "use_face_contour", False)),
    )


def sample_inputs(spec: ModelSpec, batch_size: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample: dict[str, np.ndarray] = {
        "betas": (rng.standard_normal((batch_size, spec.num_betas)) * 0.03).astype(np.float32),
        "global_orient": (rng.standard_normal((batch_size, 1, 3)) * 0.05).astype(np.float32),
        "body_pose": (rng.standard_normal((batch_size, spec.num_body_joints, 3)) * 0.05).astype(np.float32),
        "transl": (rng.standard_normal((batch_size, 3)) * 0.01).astype(np.float32),
    }
    if spec.num_expression > 0:
        sample["expression"] = (rng.standard_normal((batch_size, spec.num_expression)) * 0.03).astype(np.float32)
    if spec.num_face_joints > 0:
        sample["jaw_pose"] = (rng.standard_normal((batch_size, 1, 3)) * 0.02).astype(np.float32)
        sample["leye_pose"] = (rng.standard_normal((batch_size, 1, 3)) * 0.01).astype(np.float32)
        sample["reye_pose"] = (rng.standard_normal((batch_size, 1, 3)) * 0.01).astype(np.float32)
    if spec.num_hand_joints > 0:
        sample["left_hand_pose"] = (rng.standard_normal((batch_size, spec.num_hand_joints, 3)) * 0.05).astype(np.float32)
        sample["right_hand_pose"] = (rng.standard_normal((batch_size, spec.num_hand_joints, 3)) * 0.05).astype(np.float32)
    return sample


def ref_forward_kwargs(spec: ModelSpec, sample: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    keys = ["betas", "global_orient", "body_pose", "transl"]
    if spec.num_hand_joints > 0:
        keys.extend(["left_hand_pose", "right_hand_pose"])
    if spec.num_face_joints > 0:
        keys.extend(["jaw_pose", "leye_pose", "reye_pose", "expression"])
    elif spec.num_expression > 0:
        keys.append("expression")
    out = {k: torch.tensor(sample[k]) for k in keys}
    out["return_verts"] = True  # type: ignore[assignment]
    return out
