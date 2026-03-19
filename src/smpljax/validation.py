from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import SMPLModelData


@dataclass(frozen=True)
class ModelSummary:
    num_vertices: int
    num_joints: int
    num_betas: int
    num_expression_coeffs: int
    num_body_joints: int
    num_hand_joints: int
    num_face_joints: int
    has_faces: bool
    has_landmarks: bool
    has_dynamic_landmarks: bool
    use_pca: bool


def summarize_model_data(data: SMPLModelData) -> ModelSummary:
    return ModelSummary(
        num_vertices=int(np.asarray(data.v_template).shape[0]),
        num_joints=int(np.asarray(data.parents).shape[0]),
        num_betas=int(data.num_betas or np.asarray(data.shapedirs).shape[-1]),
        num_expression_coeffs=int(data.num_expression_coeffs or 0),
        num_body_joints=int(data.num_body_joints or max(np.asarray(data.parents).shape[0] - 1, 0)),
        num_hand_joints=int(data.num_hand_joints or 0),
        num_face_joints=int(data.num_face_joints or 0),
        has_faces=data.faces_tensor is not None,
        has_landmarks=data.lmk_faces_idx is not None and data.lmk_bary_coords is not None,
        has_dynamic_landmarks=(
            data.dynamic_lmk_faces_idx is not None and data.dynamic_lmk_bary_coords is not None
        ),
        use_pca=bool(data.use_pca),
    )


def validate_model_data(data: SMPLModelData) -> None:
    issues: list[str] = []

    v_template = np.asarray(data.v_template)
    shapedirs = np.asarray(data.shapedirs)
    posedirs = np.asarray(data.posedirs)
    j_regressor = np.asarray(data.j_regressor)
    parents = np.asarray(data.parents, dtype=np.int32).reshape(-1)
    lbs_weights = np.asarray(data.lbs_weights)
    faces = None if data.faces_tensor is None else np.asarray(data.faces_tensor, dtype=np.int32)

    if v_template.ndim != 2 or v_template.shape[1] != 3:
        issues.append(f"v_template must have shape (V, 3), got {v_template.shape}")
    if shapedirs.ndim != 3 or shapedirs.shape[:2] != v_template.shape:
        issues.append(f"shapedirs must have shape (V, 3, B), got {shapedirs.shape}")
    if posedirs.ndim != 2 or posedirs.shape[1] != int(v_template.shape[0] * 3):
        issues.append(f"posedirs must have shape (P, V*3), got {posedirs.shape}")
    if j_regressor.ndim != 2 or j_regressor.shape[1] != v_template.shape[0]:
        issues.append(f"j_regressor must have shape (J, V), got {j_regressor.shape}")
    if lbs_weights.ndim != 2 or lbs_weights.shape[0] != v_template.shape[0]:
        issues.append(f"lbs_weights must have shape (V, J), got {lbs_weights.shape}")
    if parents.ndim != 1:
        issues.append(f"parents must be 1D, got {parents.shape}")

    num_joints = int(parents.shape[0])
    if j_regressor.ndim == 2 and j_regressor.shape[0] != num_joints:
        issues.append(
            f"parents and j_regressor disagree on joint count: {num_joints} vs {j_regressor.shape[0]}"
        )
    if lbs_weights.ndim == 2 and lbs_weights.shape[1] != num_joints:
        issues.append(
            f"parents and lbs_weights disagree on joint count: {num_joints} vs {lbs_weights.shape[1]}"
        )
    if parents.size > 0 and int(parents[0]) != -1:
        issues.append("parents[0] must be -1")
    if np.any(parents[1:] < 0) or np.any(parents[1:] >= num_joints):
        issues.append("parents[1:] must index valid parent joints")

    num_body = int(data.num_body_joints or max(num_joints - 1, 0))
    num_hand = int(data.num_hand_joints or 0)
    num_face = int(data.num_face_joints or 0)
    if num_body < 0:
        issues.append("num_body_joints must be non-negative")
    if num_hand < 0:
        issues.append("num_hand_joints must be non-negative")
    if num_face < 0:
        issues.append("num_face_joints must be non-negative")
    accounted_joints = 1 + num_body + 2 * num_hand + num_face
    if accounted_joints != num_joints:
        issues.append(
            "joint metadata is inconsistent: "
            f"1 + body({num_body}) + 2*hand({num_hand}) + face({num_face}) = {accounted_joints}, "
            f"but parents has {num_joints} joints"
        )
    if num_face not in (0, 3):
        issues.append(f"num_face_joints must be 0 or 3 for the current runtime, got {num_face}")

    if int(data.num_expression_coeffs or 0) < 0:
        issues.append("num_expression_coeffs must be non-negative")
    if int(data.num_expression_coeffs or 0) > max(shapedirs.shape[-1] - int(data.num_betas or shapedirs.shape[-1]), 0):
        issues.append("num_expression_coeffs exceeds shapedirs capacity")

    if data.use_pca and num_hand > 0:
        if data.left_hand_components is None or data.right_hand_components is None:
            issues.append("use_pca=True requires both left_hand_components and right_hand_components")
    hand_component_width = num_hand * 3 if num_hand > 0 else 0
    for name, comps in (
        ("left_hand_components", data.left_hand_components),
        ("right_hand_components", data.right_hand_components),
    ):
        if comps is None:
            continue
        arr = np.asarray(comps)
        if arr.ndim != 2:
            issues.append(f"{name} must have shape (num_pca_comps, {hand_component_width}), got {arr.shape}")
            continue
        if num_hand == 0:
            issues.append(f"{name} was provided but num_hand_joints is 0")
            continue
        if arr.shape[1] != hand_component_width:
            issues.append(
                f"{name} must have width {hand_component_width} for {num_hand} hand joints, got {arr.shape[1]}"
            )
        if arr.shape[0] <= 0:
            issues.append(f"{name} must include at least one PCA component")

    if faces is not None:
        if faces.ndim != 2 or faces.shape[1] != 3:
            issues.append(f"faces_tensor must have shape (F, 3), got {faces.shape}")
        elif np.any(faces < 0) or np.any(faces >= v_template.shape[0]):
            issues.append("faces_tensor contains out-of-range vertex indices")

    if data.extra_vertex_ids:
        extra = np.asarray(data.extra_vertex_ids, dtype=np.int32)
        if np.any(extra < 0) or np.any(extra >= v_template.shape[0]):
            issues.append("extra_vertex_ids contains out-of-range vertex indices")

    lmk_faces = None if data.lmk_faces_idx is None else np.asarray(data.lmk_faces_idx, dtype=np.int32)
    lmk_bary = None if data.lmk_bary_coords is None else np.asarray(data.lmk_bary_coords)
    if (lmk_faces is None) != (lmk_bary is None):
        issues.append("lmk_faces_idx and lmk_bary_coords must either both be present or both be absent")
    if lmk_faces is not None and lmk_bary is not None:
        if faces is None:
            issues.append("landmark metadata requires faces_tensor")
        if lmk_faces.ndim != 1:
            issues.append(f"lmk_faces_idx must be 1D, got {lmk_faces.shape}")
        if lmk_bary.ndim != 2 or lmk_bary.shape[1] != 3:
            issues.append(f"lmk_bary_coords must have shape (L, 3), got {lmk_bary.shape}")
        if lmk_faces.shape[0] != lmk_bary.shape[0]:
            issues.append("lmk_faces_idx and lmk_bary_coords must have the same length")
        if faces is not None and np.any((lmk_faces < 0) | (lmk_faces >= faces.shape[0])):
            issues.append("lmk_faces_idx contains out-of-range face indices")

    dyn_faces = None if data.dynamic_lmk_faces_idx is None else np.asarray(data.dynamic_lmk_faces_idx, dtype=np.int32)
    dyn_bary = None if data.dynamic_lmk_bary_coords is None else np.asarray(data.dynamic_lmk_bary_coords)
    if (dyn_faces is None) != (dyn_bary is None):
        issues.append(
            "dynamic_lmk_faces_idx and dynamic_lmk_bary_coords must either both be present or both be absent"
        )
    if data.use_face_contour and (dyn_faces is None or dyn_bary is None or data.neck_kin_chain is None):
        issues.append(
            "use_face_contour=True requires dynamic_lmk_faces_idx, dynamic_lmk_bary_coords, and neck_kin_chain"
        )
    if dyn_faces is not None and dyn_bary is not None:
        if lmk_faces is None or lmk_bary is None:
            issues.append("dynamic landmark metadata requires base landmark metadata")
        if faces is None:
            issues.append("dynamic landmark metadata requires faces_tensor")
        if data.neck_kin_chain is None:
            issues.append("dynamic landmark metadata requires neck_kin_chain")
        if dyn_faces.ndim != 1:
            issues.append(f"dynamic_lmk_faces_idx must be 1D, got {dyn_faces.shape}")
        if dyn_bary.ndim != 2 or dyn_bary.shape[1] != 3:
            issues.append(f"dynamic_lmk_bary_coords must have shape (L, 3), got {dyn_bary.shape}")
        if dyn_faces.shape[0] != dyn_bary.shape[0]:
            issues.append("dynamic_lmk_faces_idx and dynamic_lmk_bary_coords must have the same length")
        if faces is not None and np.any((dyn_faces < 0) | (dyn_faces >= faces.shape[0])):
            issues.append("dynamic_lmk_faces_idx contains out-of-range face indices")
    if data.neck_kin_chain is not None:
        neck = np.asarray(data.neck_kin_chain, dtype=np.int32).reshape(-1)
        if neck.ndim != 1:
            issues.append(f"neck_kin_chain must be 1D, got {np.asarray(data.neck_kin_chain).shape}")
        if np.any((neck < 0) | (neck >= num_joints)):
            issues.append("neck_kin_chain contains out-of-range joint indices")

    if issues:
        raise ValueError("Invalid SMPL model data:\n- " + "\n- ".join(issues))
