import json
from pathlib import Path

import jax.numpy as jnp

from topojax.ad.restart import summarize_mode2_restart_result
from topojax.ad.workflow import initialize_mode2_domain, run_mode2_restart_workflow
from topojax.io.imports import load_gmsh_msh


def test_mode2_polygon_workflow_without_remesh_preserves_boundary_metadata(tmp_path: Path) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.2, 1.0], [1.2, 2.0], [0.0, 2.0]])
    hole = jnp.asarray([[0.45, 0.45], [0.95, 0.45], [0.95, 0.95], [0.45, 0.95]])
    domain = initialize_mode2_domain("polygon", outer_boundary=outer, holes=[hole], target_edge_size=0.18)

    run = run_mode2_restart_workflow(
        domain,
        output_dir=tmp_path,
        prefix="mode2_polygon",
        cycles=1,
        optimization_steps=6,
        optimization_step_size=0.02,
    )

    imported = load_gmsh_msh(run.artifacts["mesh"])
    metrics = json.loads(run.artifacts["metrics"].read_text(encoding="utf-8"))
    phases = json.loads(run.artifacts["phases"].read_text(encoding="utf-8"))
    assert imported.primary_element_kind == "triangle"
    assert any(block.element_kind == "line" for block in imported.extra_element_blocks)
    assert imported.physical_names[(1, 100)] == "outer_boundary"
    assert metrics["metadata_preserved"] == 1.0
    assert len(phases) == 1


def test_mode2_polygon_quad_restart_workflow_exports_phase_history(tmp_path: Path) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.2, 1.0], [1.2, 2.0], [0.0, 2.0]])
    hole = jnp.asarray([[0.45, 0.45], [0.95, 0.45], [0.95, 0.95], [0.45, 0.95]])
    domain = initialize_mode2_domain("polygon-quad", outer_boundary=outer, holes=[hole], target_edge_size=0.2)

    run = run_mode2_restart_workflow(
        domain,
        output_dir=tmp_path,
        prefix="mode2_quad",
        cycles=2,
        optimization_steps=6,
        optimization_step_size=0.015,
        remesh_max_iters=1,
    )

    imported = load_gmsh_msh(run.artifacts["mesh"])
    phases = json.loads(run.artifacts["phases"].read_text(encoding="utf-8"))
    remesh_history = json.loads(run.artifacts["remesh_history"].read_text(encoding="utf-8"))
    summary = summarize_mode2_restart_result(run.result)
    assert imported.primary_element_kind == "quad"
    assert len(phases) == 2
    assert len(remesh_history) == 2
    assert summary["n_cycles"] == 2
    assert summary["n_elements"] == run.result.elements.shape[0]


def test_mode2_tet_restart_workflow_exports_surface_stl(tmp_path: Path) -> None:
    domain = initialize_mode2_domain(
        "sphere-volume",
        center=jnp.asarray([0.5, 0.5, 0.5]),
        radius=0.42,
        nx=7,
        ny=7,
        nz=7,
    )

    run = run_mode2_restart_workflow(
        domain,
        output_dir=tmp_path,
        prefix="mode2_tet",
        cycles=2,
        optimization_steps=4,
        optimization_step_size=0.012,
        remesh_max_iters=1,
        export_stl_surface=True,
    )

    imported = load_gmsh_msh(run.artifacts["mesh"])
    metrics = json.loads(run.artifacts["metrics"].read_text(encoding="utf-8"))
    assert imported.primary_element_kind == "tetra"
    assert run.artifacts["stl"].exists()
    assert metrics["remesh_count"] >= 1.0
