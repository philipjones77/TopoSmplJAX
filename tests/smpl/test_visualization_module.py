import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from smpljax.body_models import SMPLJAXModel
from smpljax.utils import SMPLModelData
from smpljax.visualization.common import (
    CameraPreset,
    ViewerConfig,
    ViewerState,
    available_presets,
    axis_angles_to_wxyz,
    build_forward_inputs,
    camera_from_triplet,
    compute_rest_joints,
    evaluate_model,
    infer_joint_layout,
    load_viewer_state,
    parent_relative_joint_positions,
    preset_named,
    save_viewer_state,
    summarize_viewer_state,
)
from smpljax.diagnostics import DiagnosticsLogger, diagnostics_payload, write_runtime_diagnostics
from smpljax.optimized import OptimizedSMPLJAX


def _mock_model() -> SMPLJAXModel:
    return SMPLJAXModel(
        data=SMPLModelData(
            v_template=jnp.zeros((5, 3), dtype=jnp.float32),
            shapedirs=jnp.zeros((5, 3, 30), dtype=jnp.float32),
            posedirs=jnp.zeros((486, 15), dtype=jnp.float32),
            j_regressor=jnp.ones((55, 5), dtype=jnp.float32) / 5.0,
            parents=jnp.array([-1] + list(range(54)), dtype=jnp.int32),
            lbs_weights=jnp.ones((5, 55), dtype=jnp.float32) / 55.0,
            num_betas=20,
            num_body_joints=21,
            num_hand_joints=15,
            num_face_joints=3,
            num_expression_coeffs=10,
        )
    )


def test_infer_joint_layout() -> None:
    model = _mock_model()
    total, body, hand, face = infer_joint_layout(model)
    assert (total, body, hand, face) == (55, 21, 15, 3)


def test_build_forward_inputs_splits_pose() -> None:
    model = _mock_model()
    state = ViewerState(
        betas=np.zeros((1, 10), dtype=np.float32),
        expression=np.zeros((1, 10), dtype=np.float32),
        full_pose_aa=np.arange(55 * 3, dtype=np.float32).reshape(1, 55, 3),
        transl=np.zeros((1, 3), dtype=np.float32),
    )
    out = build_forward_inputs(model_like=model, state=state)
    assert out["global_orient"].shape == (1, 1, 3)
    assert out["body_pose"].shape == (1, 21, 3)
    assert out["transl"].shape == (1, 3)
    assert out["jaw_pose"].shape == (1, 1, 3)
    assert out["leye_pose"].shape == (1, 1, 3)
    assert out["reye_pose"].shape == (1, 1, 3)
    assert out["left_hand_pose"].shape == (1, 15, 3)
    assert out["right_hand_pose"].shape == (1, 15, 3)


def test_compute_rest_joints_matches_joint_regressor_mean() -> None:
    model = _mock_model()
    state = ViewerState(
        betas=np.zeros((1, 10), dtype=np.float32),
        expression=None,
        full_pose_aa=np.zeros((1, 55, 3), dtype=np.float32),
        transl=np.zeros((1, 3), dtype=np.float32),
    )
    joints = compute_rest_joints(model, state=state)
    assert joints.shape == (1, 55, 3)
    np.testing.assert_allclose(joints, 0.0, atol=1e-6)


def test_parent_relative_joint_positions_applies_translation_to_root_only() -> None:
    joints = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 5.0, 7.0],
        ],
        dtype=np.float32,
    )
    parents = np.array([-1, 0, 1], dtype=np.int32)
    out = parent_relative_joint_positions(joints, parents=parents, transl=np.array([10.0, 0.0, -1.0]))
    expected = np.array(
        [
            [11.0, 2.0, 2.0],
            [1.0, 2.0, 3.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_axis_angles_to_wxyz_identity_and_pi_rotation() -> None:
    axis_angles = np.array(
        [
            [0.0, 0.0, 0.0],
            [np.pi, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    out = axis_angles_to_wxyz(axis_angles)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_evaluate_model_returns_vertices_and_joints() -> None:
    model = _mock_model()
    state = ViewerState(
        betas=np.zeros((1, 10), dtype=np.float32),
        expression=np.zeros((1, 10), dtype=np.float32),
        full_pose_aa=np.zeros((1, 55, 3), dtype=np.float32),
        transl=np.zeros((1, 3), dtype=np.float32),
    )
    out = evaluate_model(model, state=state, return_full_pose=True)
    assert out.vertices.shape == (1, 5, 3)
    assert out.joints.shape[0] == 1
    assert out.full_pose is not None


def test_presets_and_state_roundtrip() -> None:
    model = _mock_model()
    assert available_presets() == ("neutral", "contrapposto", "stride")
    preset = preset_named("contrapposto", model, max_betas=10, max_expression=10)
    path = Path("results") / "test-viewer-state.json"
    try:
        save_viewer_state(path, preset.state, preset.camera)
        restored_state, restored_camera = load_viewer_state(path)
        np.testing.assert_allclose(restored_state.full_pose_aa, preset.state.full_pose_aa)
        assert restored_camera == preset.camera
    finally:
        if path.exists():
            path.unlink()


def test_camera_from_triplet_parses_camera_positions() -> None:
    camera = camera_from_triplet(((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    assert camera == CameraPreset(
        position=(1.0, 2.0, 3.0),
        focal_point=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
    )


def test_viewer_diagnostics_summary_and_runtime_export() -> None:
    model = _mock_model()
    runtime = OptimizedSMPLJAX(data=model.data)
    runtime.warmup(batch_size=1, pose2rot=True)
    state = ViewerState(
        betas=np.zeros((1, 10), dtype=np.float32),
        expression=np.zeros((1, 10), dtype=np.float32),
        full_pose_aa=np.zeros((1, 55, 3), dtype=np.float32),
        transl=np.zeros((1, 3), dtype=np.float32),
    )
    viewer_diag = summarize_viewer_state(
        state=state,
        num_vertices=5,
        num_joints=55,
        preset="neutral",
        use_optimized_runtime=True,
        update_index=1,
    )
    path = Path("output") / "test-viewer-diagnostics.json"
    try:
        write_runtime_diagnostics(path, runtime=runtime, extra={"viewer": viewer_diag})
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["runtime"]["compile_count"] >= 1
        assert payload["extra"]["viewer"]["num_vertices"] == 5
    finally:
        if path.exists():
            path.unlink()


def test_diagnostics_logger_appends_jsonl() -> None:
    path = Path("output") / "test-viewer-diagnostics.jsonl"
    logger = DiagnosticsLogger(path)
    try:
        logger.append({"event": "viewer_update", "backend": "viser", "value": 1})
        logger.append({"event": "viewer_update", "backend": "pyvista", "value": 2})
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["backend"] == "viser"
        assert json.loads(lines[1])["backend"] == "pyvista"
    finally:
        if path.exists():
            path.unlink()
