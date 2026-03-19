from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from smpljax.body_models import ModelOutput, SMPLJAXModel
from smpljax.disk import atomic_write_json
from smpljax.io import load_model
from smpljax.lbs import blend_shapes, vertices2joints
from smpljax.optimized import OptimizedSMPLJAX
from smpljax.utils import SMPLModelData


@dataclass(frozen=True)
class ViewerConfig:
    model_path: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    max_betas: int = 10
    max_expression: int = 10
    joint_limit: float = 1.0
    fps: float = 30.0
    use_optimized_runtime: bool = True
    preset: str = "neutral"
    state_json: Path | None = None
    export_state_json: Path | None = None
    diagnostics_json: Path | None = None
    diagnostics_jsonl: Path | None = None
    diagnostics_every_n_updates: int = 1


@dataclass(frozen=True)
class ViewerState:
    betas: np.ndarray
    expression: np.ndarray | None
    full_pose_aa: np.ndarray
    transl: np.ndarray


@dataclass(frozen=True)
class CameraPreset:
    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    up: tuple[float, float, float]


@dataclass(frozen=True)
class ViewerPreset:
    name: str
    state: ViewerState
    camera: CameraPreset


@dataclass(frozen=True)
class ViewerDiagnostics:
    update_index: int
    preset: str
    use_optimized_runtime: bool
    num_vertices: int
    num_joints: int
    transl_norm: float
    betas_norm: float
    expression_norm: float


def infer_joint_layout(model_like: SMPLJAXModel | OptimizedSMPLJAX) -> tuple[int, int, int, int]:
    data = model_like.data
    total_joints = int(np.asarray(data.parents).shape[0])
    num_face = int(data.num_face_joints or 0)
    num_hand = int(data.num_hand_joints or 0)
    num_body = int(data.num_body_joints or max(total_joints - 1 - num_face - 2 * num_hand, 0))
    return total_joints, num_body, num_hand, num_face


def build_forward_inputs(
    model_like: SMPLJAXModel | OptimizedSMPLJAX,
    state: ViewerState,
) -> dict[str, jnp.ndarray]:
    _, num_body, num_hand, num_face = infer_joint_layout(model_like)
    idx = 0
    global_orient = state.full_pose_aa[:, idx : idx + 1, :]
    idx += 1
    body_pose = state.full_pose_aa[:, idx : idx + num_body, :]
    idx += num_body

    kwargs: dict[str, jnp.ndarray] = {
        "betas": jnp.asarray(state.betas),
        "global_orient": jnp.asarray(global_orient),
        "body_pose": jnp.asarray(body_pose),
        "transl": jnp.asarray(state.transl),
    }
    if state.expression is not None:
        kwargs["expression"] = jnp.asarray(state.expression)
    if num_face > 0:
        kwargs["jaw_pose"] = jnp.asarray(state.full_pose_aa[:, idx : idx + 1, :])
        kwargs["leye_pose"] = jnp.asarray(state.full_pose_aa[:, idx + 1 : idx + 2, :])
        kwargs["reye_pose"] = jnp.asarray(state.full_pose_aa[:, idx + 2 : idx + 3, :])
        idx += num_face
    if num_hand > 0:
        kwargs["left_hand_pose"] = jnp.asarray(state.full_pose_aa[:, idx : idx + num_hand, :])
        kwargs["right_hand_pose"] = jnp.asarray(
            state.full_pose_aa[:, idx + num_hand : idx + 2 * num_hand, :]
        )
    return kwargs


def resolve_runtime(
    config: ViewerConfig,
    model: SMPLJAXModel | None = None,
    data: SMPLModelData | None = None,
    runtime: OptimizedSMPLJAX | None = None,
) -> SMPLJAXModel | OptimizedSMPLJAX:
    if runtime is not None:
        return runtime
    if model is not None:
        return model
    if data is not None:
        if config.use_optimized_runtime:
            return OptimizedSMPLJAX(data=data)
        return SMPLJAXModel(data=data)
    if config.model_path is None:
        raise ValueError("Provide either `model_path` in ViewerConfig or pass `model`/`data`/`runtime`.")
    loaded = load_model(config.model_path)
    if config.use_optimized_runtime:
        return OptimizedSMPLJAX(data=loaded)
    return SMPLJAXModel(data=loaded)


def shape_components(betas: np.ndarray, expression: np.ndarray | None) -> jnp.ndarray:
    betas_jax = jnp.asarray(betas)
    if expression is None:
        return betas_jax
    expr_jax = jnp.asarray(expression, dtype=betas_jax.dtype)
    return jnp.concatenate([betas_jax, expr_jax], axis=-1)


def compute_rest_joints(
    model_like: SMPLJAXModel | OptimizedSMPLJAX,
    state: ViewerState,
) -> np.ndarray:
    shaped = shape_components(betas=state.betas, expression=state.expression)
    shapedirs = model_like.data.shapedirs[..., : shaped.shape[-1]]
    v_shaped = model_like.data.v_template[None, ...] + blend_shapes(shaped, shapedirs)
    joints = vertices2joints(model_like.data.j_regressor, v_shaped)
    return np.asarray(joints, dtype=np.float32)


def parent_relative_joint_positions(
    joints: np.ndarray,
    parents: np.ndarray,
    transl: np.ndarray | None = None,
) -> np.ndarray:
    base = np.asarray(joints, dtype=np.float32)
    out = base.copy()
    for idx in range(1, out.shape[0]):
        out[idx] = base[idx] - base[int(parents[idx])]
    if transl is not None:
        out[0] = out[0] + np.asarray(transl, dtype=np.float32)
    return out


def axis_angles_to_wxyz(axis_angles: np.ndarray) -> np.ndarray:
    axis_angles = np.asarray(axis_angles, dtype=np.float32)
    angles = np.linalg.norm(axis_angles, axis=-1, keepdims=True)
    half_angles = 0.5 * angles
    small = angles < 1e-8
    safe_angles = np.where(small, 1.0, angles)
    xyz = axis_angles * (np.sin(half_angles) / safe_angles)
    xyz = np.where(small, 0.5 * axis_angles, xyz)
    w = np.cos(half_angles)
    return np.concatenate([w, xyz], axis=-1).astype(np.float32)


def make_default_state(
    model_like: SMPLJAXModel | OptimizedSMPLJAX,
    *,
    max_betas: int,
    max_expression: int,
) -> ViewerState:
    total_joints, _, _, _ = infer_joint_layout(model_like)
    num_betas = int(min(max_betas, model_like.data.num_betas or np.asarray(model_like.data.shapedirs).shape[-1]))
    num_expr = int(min(max_expression, model_like.data.num_expression_coeffs or 0))
    return ViewerState(
        betas=np.zeros((1, num_betas), dtype=np.float32),
        expression=np.zeros((1, num_expr), dtype=np.float32) if num_expr > 0 else None,
        full_pose_aa=np.zeros((1, total_joints, 3), dtype=np.float32),
        transl=np.zeros((1, 3), dtype=np.float32),
    )


def _camera_front() -> CameraPreset:
    return CameraPreset(position=(0.0, 0.1, 2.6), focal_point=(0.0, 0.1, 0.0), up=(0.0, 1.0, 0.0))


def _camera_three_quarter() -> CameraPreset:
    return CameraPreset(position=(1.8, 0.5, 2.2), focal_point=(0.0, 0.2, 0.0), up=(0.0, 1.0, 0.0))


def preset_named(
    name: str,
    model_like: SMPLJAXModel | OptimizedSMPLJAX,
    *,
    max_betas: int,
    max_expression: int,
) -> ViewerPreset:
    base = make_default_state(model_like, max_betas=max_betas, max_expression=max_expression)
    pose = base.full_pose_aa.copy()
    transl = base.transl.copy()
    betas = base.betas.copy()
    expression = None if base.expression is None else base.expression.copy()
    key = name.lower()
    if key == "neutral":
        camera = _camera_front()
    elif key == "contrapposto":
        pose[0, 1, 2] = 0.18
        pose[0, 2, 0] = -0.12
        pose[0, 5, 0] = 0.18
        pose[0, 8, 0] = -0.1
        transl[0, 0] = 0.03
        camera = _camera_three_quarter()
    elif key == "stride":
        pose[0, 1, 0] = -0.25
        pose[0, 2, 0] = 0.35
        pose[0, 4, 0] = -0.1
        pose[0, 5, 0] = 0.25
        pose[0, 8, 0] = 0.35
        pose[0, 11, 0] = -0.2
        camera = _camera_three_quarter()
    else:
        raise ValueError(f"Unknown viewer preset: {name}")
    return ViewerPreset(
        name=key,
        state=ViewerState(betas=betas, expression=expression, full_pose_aa=pose, transl=transl),
        camera=camera,
    )


def available_presets() -> tuple[str, ...]:
    return ("neutral", "contrapposto", "stride")


def viewer_state_to_json_dict(state: ViewerState) -> dict[str, object]:
    return {
        "betas": state.betas.tolist(),
        "expression": None if state.expression is None else state.expression.tolist(),
        "full_pose_aa": state.full_pose_aa.tolist(),
        "transl": state.transl.tolist(),
    }


def summarize_viewer_state(
    *,
    state: ViewerState,
    num_vertices: int,
    num_joints: int,
    preset: str,
    use_optimized_runtime: bool,
    update_index: int,
) -> ViewerDiagnostics:
    return ViewerDiagnostics(
        update_index=update_index,
        preset=preset,
        use_optimized_runtime=use_optimized_runtime,
        num_vertices=num_vertices,
        num_joints=num_joints,
        transl_norm=float(np.linalg.norm(state.transl)),
        betas_norm=float(np.linalg.norm(state.betas)),
        expression_norm=(0.0 if state.expression is None else float(np.linalg.norm(state.expression))),
    )


def viewer_state_from_json_dict(payload: dict[str, object]) -> ViewerState:
    expression = payload.get("expression")
    return ViewerState(
        betas=np.asarray(payload["betas"], dtype=np.float32),
        expression=None if expression is None else np.asarray(expression, dtype=np.float32),
        full_pose_aa=np.asarray(payload["full_pose_aa"], dtype=np.float32),
        transl=np.asarray(payload["transl"], dtype=np.float32),
    )


def save_viewer_state(path: Path, state: ViewerState, camera: CameraPreset | None = None) -> None:
    payload: dict[str, object] = {"state": viewer_state_to_json_dict(state)}
    if camera is not None:
        payload["camera"] = {
            "position": list(camera.position),
            "focal_point": list(camera.focal_point),
            "up": list(camera.up),
        }
    atomic_write_json(path, payload)


def load_viewer_state(path: Path) -> tuple[ViewerState, CameraPreset | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    camera_payload = payload.get("camera")
    camera = None
    if camera_payload is not None:
        camera = CameraPreset(
            position=tuple(float(x) for x in camera_payload["position"]),
            focal_point=tuple(float(x) for x in camera_payload["focal_point"]),
            up=tuple(float(x) for x in camera_payload["up"]),
        )
    return viewer_state_from_json_dict(payload["state"]), camera


def camera_from_triplet(camera_position: object) -> CameraPreset | None:
    if not isinstance(camera_position, (list, tuple)) or len(camera_position) != 3:
        return None
    try:
        position = tuple(float(x) for x in camera_position[0])
        focal_point = tuple(float(x) for x in camera_position[1])
        up = tuple(float(x) for x in camera_position[2])
    except Exception:
        return None
    if len(position) != 3 or len(focal_point) != 3 or len(up) != 3:
        return None
    return CameraPreset(position=position, focal_point=focal_point, up=up)


def evaluate_model(
    model_like: SMPLJAXModel | OptimizedSMPLJAX,
    state: ViewerState,
    *,
    return_full_pose: bool = False,
) -> ModelOutput:
    kwargs = build_forward_inputs(model_like=model_like, state=state)
    if isinstance(model_like, OptimizedSMPLJAX):
        prepared = model_like.prepare_inputs(
            batch_size=state.betas.shape[0],
            betas=kwargs["betas"],
            body_pose=kwargs["body_pose"],
            global_orient=kwargs["global_orient"],
            transl=kwargs["transl"],
            expression=kwargs.get("expression"),
            jaw_pose=kwargs.get("jaw_pose"),
            leye_pose=kwargs.get("leye_pose"),
            reye_pose=kwargs.get("reye_pose"),
            left_hand_pose=kwargs.get("left_hand_pose"),
            right_hand_pose=kwargs.get("right_hand_pose"),
            pose2rot=True,
        )
        return model_like.forward(prepared, pose2rot=True, return_full_pose=return_full_pose)
    return model_like(**kwargs, return_full_pose=return_full_pose)
