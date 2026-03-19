from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from smpljax.body_models import SMPLJAXModel
from smpljax.diagnostics import DiagnosticsLogger, write_runtime_diagnostics
from smpljax.optimized import OptimizedSMPLJAX
from smpljax.utils import SMPLModelData
from smpljax.visualization.common import (
    ViewerConfig,
    summarize_viewer_state,
    available_presets,
    ViewerState,
    load_viewer_state,
    preset_named,
    save_viewer_state,
    axis_angles_to_wxyz,
    compute_rest_joints,
    evaluate_model,
    infer_joint_layout,
    make_default_state,
    parent_relative_joint_positions,
    resolve_runtime,
)


def run_viser_viewer(
    config: ViewerConfig,
    model: SMPLJAXModel | None = None,
    data: SMPLModelData | None = None,
    runtime: OptimizedSMPLJAX | None = None,
) -> None:
    try:
        import viser
        import viser.transforms as tf
    except Exception as exc:
        raise RuntimeError("viser is required. Install with: python -m pip install viser") from exc

    try:
        import trimesh
    except Exception:
        trimesh = None

    runtime_obj = resolve_runtime(config=config, model=model, data=data, runtime=runtime)
    if runtime_obj.data.faces_tensor is None:
        raise ValueError("Model file must include `faces_tensor` (or `f`/`faces`) for visualization.")

    num_verts = int(np.asarray(runtime_obj.data.v_template).shape[0])
    total_joints, _, _, _ = infer_joint_layout(runtime_obj)
    num_betas = int(
        min(config.max_betas, runtime_obj.data.num_betas or np.asarray(runtime_obj.data.shapedirs).shape[-1])
    )
    num_expr = int(min(config.max_expression, runtime_obj.data.num_expression_coeffs or 0))
    parents = np.asarray(runtime_obj.data.parents, dtype=np.int32)

    server = viser.ViserServer(host=config.host, port=config.port)
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/grid", position=(0.0, -1.3, 0.0), plane="xz")
    gui = server.gui
    gui.add_markdown("### smplJAX + viser")

    auto_update = gui.add_checkbox("Auto Update", initial_value=True)
    wireframe = gui.add_checkbox("Wireframe", initial_value=False)
    rgb = gui.add_rgb("Color", initial_value=(220, 170, 120))
    show_controls = gui.add_checkbox("Handles", initial_value=True)
    reset_btn = gui.add_button("Reset Pose/Shape")
    random_shape_btn = gui.add_button("Random Shape")
    random_pose_btn = gui.add_button("Random Pose")
    preset_buttons = {name: gui.add_button(f"Preset: {name}") for name in available_presets()}
    update_btn = gui.add_button("Update Once")

    beta_sliders = [
        gui.add_slider(f"beta_{i}", min=-3.0, max=3.0, step=0.01, initial_value=0.0)
        for i in range(num_betas)
    ]
    expr_sliders = (
        [gui.add_slider(f"expr_{i}", min=-3.0, max=3.0, step=0.01, initial_value=0.0) for i in range(num_expr)]
        if num_expr > 0
        else []
    )
    pose_controls = [
        gui.add_vector3(f"joint_{joint_idx}", initial_value=(0.0, 0.0, 0.0), step=0.01)
        for joint_idx in range(total_joints)
    ]
    transl_control = gui.add_vector3("translation_xyz", initial_value=(0.0, 0.0, 0.0), step=0.01)

    changed = {"value": True}
    update_once_requested = {"value": False}
    update_counter = {"value": 0}
    diagnostics_logger = DiagnosticsLogger(config.diagnostics_jsonl) if config.diagnostics_jsonl is not None else None

    def mark_changed(_: Any = None) -> None:
        changed["value"] = True

    def _apply_state_to_controls(state: ViewerState) -> None:
        for idx, slider in enumerate(beta_sliders):
            slider.value = float(state.betas[0, idx]) if idx < state.betas.shape[1] else 0.0
        for idx, slider in enumerate(expr_sliders):
            if state.expression is None:
                slider.value = 0.0
            else:
                slider.value = float(state.expression[0, idx]) if idx < state.expression.shape[1] else 0.0
        for idx, control in enumerate(pose_controls):
            aa = state.full_pose_aa[0, idx]
            control.value = (float(aa[0]), float(aa[1]), float(aa[2]))
        transl_control.value = tuple(float(x) for x in state.transl[0])
        changed["value"] = True

    for slider in beta_sliders:
        slider.on_update(mark_changed)
    for slider in expr_sliders:
        slider.on_update(mark_changed)
    transl_control.on_update(mark_changed)
    wireframe.on_update(mark_changed)
    rgb.on_update(mark_changed)

    faces = np.asarray(runtime_obj.data.faces_tensor, dtype=np.int32)
    vertices0 = np.asarray(runtime_obj.data.v_template, dtype=np.float32)
    if vertices0.shape != (num_verts, 3):
        vertices0 = np.zeros((num_verts, 3), dtype=np.float32)
    mesh_handle = server.scene.add_mesh_simple(
        "/smpl/mesh",
        vertices=vertices0,
        faces=faces,
        wireframe=wireframe.value,
        color=rgb.value,
    )

    transform_controls: list[Any] = []
    prefixed_joint_names: list[str] = []
    for joint_idx in range(total_joints):
        joint_name = f"joint_{joint_idx}"
        if joint_idx > 0:
            joint_name = f"{prefixed_joint_names[int(parents[joint_idx])]}/{joint_name}"
        prefixed_joint_names.append(joint_name)
        handle = server.scene.add_transform_controls(
            f"/smpl/controls/{joint_name}",
            depth_test=False,
            scale=0.2 * (0.75 ** joint_name.count("/")),
            disable_axes=True,
            disable_sliders=True,
            visible=bool(show_controls.value),
        )
        transform_controls.append(handle)

    @show_controls.on_update
    def _(_: Any) -> None:
        for handle in transform_controls:
            handle.visible = bool(show_controls.value)

    def _set_handle_from_pose(joint_idx: int) -> None:
        transform_controls[joint_idx].wxyz = tuple(
            float(x)
            for x in axis_angles_to_wxyz(np.array([pose_controls[joint_idx].value], dtype=np.float32))[0]
        )

    for joint_idx, pose_control in enumerate(pose_controls):
        def _bind_pose_callback(i: int, control: Any) -> None:
            @control.on_update
            def _(_: Any) -> None:
                _set_handle_from_pose(i)
                changed["value"] = True

        _bind_pose_callback(joint_idx, pose_control)

    for joint_idx, handle in enumerate(transform_controls):
        def _bind_handle_callback(i: int, control_handle: Any) -> None:
            @control_handle.on_update
            def _(_: Any) -> None:
                axis_angle = tf.SO3(control_handle.wxyz).log()
                pose_controls[i].value = (
                    float(axis_angle[0]),
                    float(axis_angle[1]),
                    float(axis_angle[2]),
                )
                changed["value"] = True

        _bind_handle_callback(joint_idx, handle)

    @update_btn.on_click
    def _(_: Any) -> None:
        update_once_requested["value"] = True
        changed["value"] = True

    @reset_btn.on_click
    def _(_: Any) -> None:
        _apply_state_to_controls(
            make_default_state(runtime_obj, max_betas=config.max_betas, max_expression=config.max_expression)
        )

    @random_shape_btn.on_click
    def _(_: Any) -> None:
        rng = np.random.default_rng()
        for slider in beta_sliders:
            slider.value = float(rng.normal(loc=0.0, scale=1.0))
        changed["value"] = True

    @random_pose_btn.on_click
    def _(_: Any) -> None:
        rng = np.random.default_rng()
        for control in pose_controls:
            axis_angle = tf.SO3.sample_uniform(rng).log()
            control.value = (float(axis_angle[0]), float(axis_angle[1]), float(axis_angle[2]))
        changed["value"] = True

    for preset_name, button in preset_buttons.items():
        @button.on_click
        def _(_: Any, name: str = preset_name) -> None:
            preset = preset_named(
                name,
                runtime_obj,
                max_betas=config.max_betas,
                max_expression=config.max_expression,
            )
            _apply_state_to_controls(preset.state)

    vertex_selector = None
    if trimesh is not None and hasattr(server.scene, "add_batched_meshes_trimesh"):
        selector_mesh = trimesh.creation.icosphere(radius=0.003, subdivisions=1)
        selector_mesh.visual.vertex_colors = (255, 0, 0, 255)
        vertex_selector = server.scene.add_batched_meshes_trimesh(
            "/smpl/vertex_selector",
            selector_mesh,
            batched_positions=vertices0,
            batched_wxyzs=((1.0, 0.0, 0.0, 0.0),) * vertices0.shape[0],
        )

        @vertex_selector.on_click
        def _(event: Any) -> None:
            event.client.add_notification(
                f"Clicked on vertex {event.instance_index}",
                body="",
                auto_close=3000,
            )

    def update_mesh() -> None:
        betas = (
            np.array([[slider.value for slider in beta_sliders]], dtype=np.float32)
            if num_betas > 0
            else np.zeros((1, 0), dtype=np.float32)
        )
        expr = (
            np.array([[slider.value for slider in expr_sliders]], dtype=np.float32)
            if num_expr > 0
            else None
        )
        full_pose = np.array(
            [
                [
                    (
                        max(-config.joint_limit, min(config.joint_limit, float(control.value[0]))),
                        max(-config.joint_limit, min(config.joint_limit, float(control.value[1]))),
                        max(-config.joint_limit, min(config.joint_limit, float(control.value[2]))),
                    )
                    for control in pose_controls
                ]
            ],
            dtype=np.float32,
        )
        transl = np.array(
            [[float(transl_control.value[0]), float(transl_control.value[1]), float(transl_control.value[2])]],
            dtype=np.float32,
        )
        state = ViewerState(betas=betas, expression=expr, full_pose_aa=full_pose, transl=transl)
        out = evaluate_model(runtime_obj, state=state, return_full_pose=True)
        if config.export_state_json is not None:
            save_viewer_state(config.export_state_json, state)

        vertices = np.asarray(out.vertices[0], dtype=np.float32)
        mesh_handle.vertices = vertices
        mesh_handle.wireframe = bool(wireframe.value)
        mesh_handle.color = rgb.value

        if vertex_selector is not None:
            vertex_selector.batched_positions = vertices

        rest_joints = compute_rest_joints(runtime_obj, state=state)[0]
        joint_positions = parent_relative_joint_positions(rest_joints, parents=parents, transl=transl[0])
        joint_quats = axis_angles_to_wxyz(np.asarray(out.full_pose[0], dtype=np.float32))
        for joint_idx, handle in enumerate(transform_controls):
            handle.position = tuple(float(x) for x in joint_positions[joint_idx])
            handle.wxyz = tuple(float(x) for x in joint_quats[joint_idx])
        update_counter["value"] += 1
        if update_counter["value"] % max(config.diagnostics_every_n_updates, 1) == 0:
            viewer_diag = summarize_viewer_state(
                state=state,
                num_vertices=int(vertices.shape[0]),
                num_joints=int(out.joints.shape[1]),
                preset=config.preset,
                use_optimized_runtime=isinstance(runtime_obj, OptimizedSMPLJAX),
                update_index=update_counter["value"],
            )
            if isinstance(runtime_obj, OptimizedSMPLJAX) and config.diagnostics_json is not None:
                write_runtime_diagnostics(
                    config.diagnostics_json,
                    runtime=runtime_obj,
                    extra={"viewer": viewer_diag},
                )
            if diagnostics_logger is not None:
                diagnostics_logger.append(
                    {
                        "event": "viewer_update",
                        "backend": "viser",
                        "viewer": viewer_diag,
                        "runtime": (runtime_obj.diagnostics() if isinstance(runtime_obj, OptimizedSMPLJAX) else None),
                    }
                )

    sleep_s = 1.0 / max(config.fps, 1.0)
    if config.state_json is not None:
        state, _ = load_viewer_state(config.state_json)
        _apply_state_to_controls(state)
    else:
        preset = preset_named(
            config.preset,
            runtime_obj,
            max_betas=config.max_betas,
            max_expression=config.max_expression,
        )
        _apply_state_to_controls(preset.state)
    while True:
        if changed["value"] and auto_update.value:
            update_mesh()
            changed["value"] = False
            update_once_requested["value"] = False
        elif changed["value"] and update_once_requested["value"]:
            update_mesh()
            changed["value"] = False
            update_once_requested["value"] = False
        time.sleep(sleep_s)


def _parse_args() -> ViewerConfig:
    parser = argparse.ArgumentParser(description="Run smplJAX visualizer with viser.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-betas", type=int, default=10)
    parser.add_argument("--max-expression", type=int, default=10)
    parser.add_argument("--joint-limit", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--preset", type=str, choices=list(available_presets()), default="neutral")
    parser.add_argument("--state-json", type=Path, default=None)
    parser.add_argument("--export-state-json", type=Path, default=None)
    parser.add_argument("--no-optimized-runtime", action="store_true")
    args = parser.parse_args()
    return ViewerConfig(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        max_betas=args.max_betas,
        max_expression=args.max_expression,
        joint_limit=args.joint_limit,
        fps=args.fps,
        preset=args.preset,
        state_json=args.state_json,
        export_state_json=args.export_state_json,
        use_optimized_runtime=not args.no_optimized_runtime,
    )


def main() -> None:
    run_viser_viewer(_parse_args())


if __name__ == "__main__":
    main()
