from pathlib import Path

import jax.numpy as jnp

from topojax.ad.workflow import initialize_mode1_domain, run_mode1_workflow
from topojax.io.imports import load_gmsh_msh


def test_mode1_line_workflow_create_export_import(tmp_path: Path) -> None:
    points = jnp.asarray([[0.0, 0.0], [0.2, 0.02], [0.5, -0.03], [0.8, 0.01], [1.0, 0.0]])
    domain = initialize_mode1_domain("line", points=points)
    run = run_mode1_workflow(domain, output_dir=tmp_path, prefix="line", steps=8, step_size=0.02, diagnostics_every=4)
    assert run.artifacts["mesh"].exists()
    imported = load_gmsh_msh(run.artifacts["mesh"], primary_element_kind="line")
    assert imported.primary_element_kind == "line"
    assert imported.topology.elements.shape[1] == 2


def test_mode1_polygon_workflow_and_import_workflow(tmp_path: Path) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.2, 1.0], [1.2, 2.0], [0.0, 2.0]])
    hole = jnp.asarray([[0.45, 0.45], [0.95, 0.45], [0.95, 0.95], [0.45, 0.95]])
    domain = initialize_mode1_domain("polygon", outer_boundary=outer, holes=[hole], target_edge_size=0.18)
    run = run_mode1_workflow(domain, output_dir=tmp_path, prefix="polygon_workflow", steps=10, step_size=0.02, diagnostics_every=5)
    imported_domain = initialize_mode1_domain("import-msh", path=run.artifacts["mesh"])
    rerun = run_mode1_workflow(imported_domain, output_dir=tmp_path / "imported", prefix="polygon_imported", steps=4, step_size=0.02, diagnostics_every=2)
    assert rerun.artifacts["mesh"].exists()


def test_mode1_polygon_quad_workflow(tmp_path: Path) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.2, 1.0], [1.2, 2.0], [0.0, 2.0]])
    hole = jnp.asarray([[0.45, 0.45], [0.95, 0.45], [0.95, 0.95], [0.45, 0.95]])
    domain = initialize_mode1_domain("polygon-quad", outer_boundary=outer, holes=[hole], target_edge_size=0.2)
    run = run_mode1_workflow(domain, output_dir=tmp_path, prefix="polygon_quad_workflow", steps=8, step_size=0.015, diagnostics_every=4)
    imported = load_gmsh_msh(run.artifacts["mesh"])
    assert imported.primary_element_kind == "quad"


def test_mode1_extruded_workflow(tmp_path: Path) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.2, 1.0], [1.2, 2.0], [0.0, 2.0]])
    hole = jnp.asarray([[0.45, 0.45], [0.95, 0.45], [0.95, 0.95], [0.45, 0.95]])
    domain = initialize_mode1_domain("extruded", outer_boundary=outer, holes=[hole], target_edge_size=0.18, height=1.0, layers=3)
    run = run_mode1_workflow(domain, output_dir=tmp_path, prefix="extruded_workflow", steps=6, step_size=0.015, diagnostics_every=3)
    imported = load_gmsh_msh(run.artifacts["mesh"])
    assert imported.primary_element_kind == "tetra"


def test_mode1_box_volume_workflow(tmp_path: Path) -> None:
    domain = initialize_mode1_domain(
        "box-volume",
        bbox_min=jnp.asarray([-1.0, -0.5, 0.0]),
        bbox_max=jnp.asarray([1.0, 0.5, 2.0]),
        nx=7,
        ny=5,
        nz=6,
    )
    run = run_mode1_workflow(
        domain,
        output_dir=tmp_path,
        prefix="box_volume_workflow",
        steps=6,
        step_size=0.012,
        diagnostics_every=3,
        export_stl_surface=True,
    )
    imported = load_gmsh_msh(run.artifacts["mesh"])
    assert imported.primary_element_kind == "tetra"
    assert run.artifacts["stl"].exists()


def test_mode1_sphere_volume_workflow(tmp_path: Path) -> None:
    domain = initialize_mode1_domain(
        "sphere-volume",
        center=jnp.asarray([0.5, 0.5, 0.5]),
        radius=0.42,
        nx=9,
        ny=9,
        nz=9,
    )
    run = run_mode1_workflow(
        domain,
        output_dir=tmp_path,
        prefix="sphere_volume_workflow",
        steps=6,
        step_size=0.012,
        diagnostics_every=3,
    )
    imported = load_gmsh_msh(run.artifacts["mesh"])
    assert imported.primary_element_kind == "tetra"
    assert imported.physical_names[(2, 320)] == "sphere_boundary"
