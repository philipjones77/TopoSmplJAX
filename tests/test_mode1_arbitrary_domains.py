from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from gmshjax.ad.mode1 import export_mode1_artifacts, optimize_mode1_fixed_topology
from gmshjax.io.gmsh_viewer import launch_gmsh_viewer
from gmshjax.io.imports import load_gmsh_msh
from gmshjax.mesh.domains import box_volume_tet_mesh_tagged, extruded_polygon_tet_mesh, implicit_volume_tet_mesh_tagged, polygon_domain_quad_mesh_tagged, polygon_domain_tri_mesh_tagged
from gmshjax.mesh.operators import mesh_quality_energy, quad_mesh_quality_energy, tet_mesh_quality_energy


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    inside = False
    x, y = float(point[0]), float(point[1])
    n = int(polygon.shape[0])
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            x_cross = float(x0 + (y - y0) * (x1 - x0) / max(y1 - y0, 1.0e-12))
            if x < x_cross:
                inside = not inside
    return inside


def test_polygon_domain_tri_mesh_supports_mode1_and_export(tmp_path: Path) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.2, 1.0], [1.2, 2.0], [0.0, 2.0]])
    hole = jnp.asarray([[0.45, 0.45], [0.95, 0.45], [0.95, 0.95], [0.45, 0.95]])
    topo, points, metadata = polygon_domain_tri_mesh_tagged(outer, holes=[hole], target_edge_size=0.18)

    centroids = np.mean(np.asarray(points)[np.asarray(topo.elements)], axis=1)
    outer_np = np.asarray(outer)
    hole_np = np.asarray(hole)
    assert topo.elements.shape[1] == 3
    assert topo.n_nodes == points.shape[0]
    assert all(_point_in_polygon(c, outer_np) and not _point_in_polygon(c, hole_np) for c in centroids)

    distorted = points + 0.03 * jnp.stack([jnp.sin(points[:, 1]), jnp.cos(points[:, 0])], axis=1)
    initial = float(mesh_quality_energy(distorted, topo))
    result = optimize_mode1_fixed_topology(distorted, topo, steps=10, step_size=0.02, diagnostics_every=5)
    final = float(mesh_quality_energy(result.points, topo))
    artifacts = export_mode1_artifacts(
        tmp_path,
        result,
        prefix="polygon",
        extra_element_blocks=metadata.boundary_element_blocks,
        physical_names=metadata.physical_names,
    )
    assert final <= initial + 1.0e-8
    assert artifacts["mesh"].exists()
    text = artifacts["mesh"].read_text(encoding="utf-8")
    assert "$PhysicalNames" in text
    imported = load_gmsh_msh(artifacts["mesh"])
    assert imported.primary_element_kind == "triangle"
    assert any(block.element_kind == "line" for block in imported.extra_element_blocks)
    assert imported.physical_names[(1, 100)] == "outer_boundary"


def test_polygon_domain_quad_mesh_supports_mode1_and_export(tmp_path: Path) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.2, 1.0], [1.2, 2.0], [0.0, 2.0]])
    hole = jnp.asarray([[0.45, 0.45], [0.95, 0.45], [0.95, 0.95], [0.45, 0.95]])
    topo, points, metadata = polygon_domain_quad_mesh_tagged(outer, holes=[hole], target_edge_size=0.2)

    centroids = np.mean(np.asarray(points)[np.asarray(topo.elements)], axis=1)
    assert topo.elements.shape[1] == 4
    assert topo.n_nodes == points.shape[0]
    assert all(_point_in_polygon(c, np.asarray(outer)) and not _point_in_polygon(c, np.asarray(hole)) for c in centroids)

    distorted = points + 0.02 * jnp.stack([jnp.sin(points[:, 1]), jnp.cos(points[:, 0])], axis=1)
    initial = float(quad_mesh_quality_energy(distorted, topo))
    result = optimize_mode1_fixed_topology(distorted, topo, steps=8, step_size=0.015, diagnostics_every=4)
    final = float(quad_mesh_quality_energy(result.points, topo))
    artifacts = export_mode1_artifacts(
        tmp_path,
        result,
        prefix="polygon_quad",
        extra_element_blocks=metadata.boundary_element_blocks,
        physical_names=metadata.physical_names,
    )
    assert final <= initial + 1.0e-8
    imported = load_gmsh_msh(artifacts["mesh"])
    assert imported.primary_element_kind == "quad"
    assert any(block.element_kind == "line" for block in imported.extra_element_blocks)
    assert imported.physical_names[(1, 100)] == "outer_boundary"


def test_implicit_volume_tet_mesh_supports_mode1() -> None:
    def sphere_sdf(x: jnp.ndarray) -> jnp.ndarray:
        center = jnp.asarray([0.5, 0.5, 0.5], dtype=x.dtype)
        return jnp.linalg.norm(x - center[None, :], axis=1) - 0.42

    topo, points, metadata = implicit_volume_tet_mesh_tagged(sphere_sdf, jnp.asarray([0.0, 0.0, 0.0]), jnp.asarray([1.0, 1.0, 1.0]), 9, 9, 9)
    assert topo.elements.shape[1] == 4
    assert points.shape[1] == 3
    assert topo.elements.shape[0] > 0
    assert len(metadata.boundary_element_blocks) == 1

    distorted = points.at[:, 2].set(points[:, 2] + 0.02 * jnp.sin(jnp.pi * points[:, 0]) * jnp.sin(jnp.pi * points[:, 1]))
    initial = float(tet_mesh_quality_energy(distorted, topo))
    result = optimize_mode1_fixed_topology(distorted, topo, steps=8, step_size=0.015, diagnostics_every=4)
    final = float(tet_mesh_quality_energy(result.points, topo))
    assert final <= initial + 1.0e-8


def test_box_volume_tet_mesh_supports_mode1_and_export(tmp_path: Path) -> None:
    topo, points, metadata = box_volume_tet_mesh_tagged(
        jnp.asarray([-1.0, -0.5, 0.0]),
        jnp.asarray([1.0, 0.5, 2.0]),
        7,
        5,
        6,
    )
    distorted = points.at[:, 2].set(points[:, 2] + 0.015 * jnp.sin(jnp.pi * points[:, 0]) * jnp.cos(jnp.pi * points[:, 1]))
    initial = float(tet_mesh_quality_energy(distorted, topo))
    result = optimize_mode1_fixed_topology(distorted, topo, steps=6, step_size=0.012, diagnostics_every=3)
    final = float(tet_mesh_quality_energy(result.points, topo))
    artifacts = export_mode1_artifacts(
        tmp_path,
        result,
        prefix="box_volume",
        extra_element_blocks=metadata.boundary_element_blocks,
        physical_names=metadata.physical_names,
    )
    imported = load_gmsh_msh(artifacts["mesh"])
    assert final <= initial + 1.0e-8
    assert imported.primary_element_kind == "tetra"
    assert imported.physical_names[(2, 500)] == "xmin"
    assert imported.physical_names[(2, 505)] == "zmax"


def test_extruded_polygon_tet_mesh_supports_mode1_and_export(tmp_path: Path) -> None:
    outer = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.2, 1.0], [1.2, 2.0], [0.0, 2.0]])
    hole = jnp.asarray([[0.45, 0.45], [0.95, 0.45], [0.95, 0.95], [0.45, 0.95]])
    topo, points, metadata = extruded_polygon_tet_mesh(outer, holes=[hole], target_edge_size=0.18, height=1.0, layers=3)
    distorted = points.at[:, 2].set(points[:, 2] + 0.02 * jnp.sin(jnp.pi * points[:, 0]) * jnp.sin(jnp.pi * points[:, 1]))
    result = optimize_mode1_fixed_topology(distorted, topo, steps=6, step_size=0.015, diagnostics_every=3)
    artifacts = export_mode1_artifacts(
        tmp_path,
        result,
        prefix="extruded",
        extra_element_blocks=metadata.boundary_element_blocks,
        physical_names=metadata.physical_names,
    )
    imported = load_gmsh_msh(artifacts["mesh"])
    assert imported.primary_element_kind == "tetra"
    assert any(block.element_kind == "triangle" for block in imported.extra_element_blocks)
    assert imported.physical_names[(2, 400)] == "bottom"


def test_launch_gmsh_viewer_invokes_external_viewer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.msh"
    mesh_path.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n", encoding="utf-8")
    called: list[list[str]] = []

    class _DummyProc:
        def wait(self) -> int:
            return 0

    def _fake_popen(cmd: list[str]):
        called.append(cmd)
        return _DummyProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)
    proc = launch_gmsh_viewer(mesh_path, gmsh_executable="gmsh", extra_args=["-nopopup"])
    assert proc is not None
    assert called == [["gmsh", str(mesh_path), "-nopopup"]]
