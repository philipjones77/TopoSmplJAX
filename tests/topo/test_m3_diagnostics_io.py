import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from topojax.ad.mode1 import export_mode1_artifacts, optimize_mode1_fixed_topology
from topojax.io.exports import GmshElementBlock, export_binary_stl, export_gmsh_msh, export_metrics_json, export_snapshot_npz, load_snapshot_npz
from topojax.io.imports import load_gmsh_msh
from topojax.mesh.diagnostics import quad_diagnostics, tet_diagnostics, tri_diagnostics
from topojax.mesh.topology import unit_cube_tet_mesh, unit_square_quad_mesh, unit_square_tri_mesh


def test_diagnostics_return_expected_fields() -> None:
    tri_topo, tri_pts = unit_square_tri_mesh(8, 6)
    quad_topo, quad_pts = unit_square_quad_mesh(8, 6)
    tet_topo, tet_pts = unit_cube_tet_mesh(4, 4, 3)

    d1 = tri_diagnostics(tri_pts, tri_topo.elements)
    d2 = quad_diagnostics(quad_pts, quad_topo.elements)
    d3 = tet_diagnostics(tet_pts, tet_topo.elements)
    for d in [d1, d2, d3]:
        assert d["n_nodes"] > 0
        assert d["n_elements"] > 0
        assert d["mean_icn"] == d["mean_icn"]


def test_snapshot_and_metrics_io_roundtrip(tmp_path: Path) -> None:
    p = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    e = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    npz_path = tmp_path / "snap.npz"
    json_path = tmp_path / "diag.json"
    export_snapshot_npz(npz_path, p, e, metrics={"mean_icn": 0.5})
    export_metrics_json(json_path, {"mean_icn": 0.5, "min_icn": 0.2})
    loaded = load_snapshot_npz(npz_path)
    assert loaded["points"].shape == (3, 2)
    assert loaded["elements"].shape == (1, 3)
    assert "metric_mean_icn" in loaded
    assert json_path.exists()


def test_export_gmsh_msh_triangle_roundtrip_text(tmp_path: Path) -> None:
    p = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    e = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    out_path = tmp_path / "tri_mesh.msh"
    export_gmsh_msh(out_path, p, e, element_entity_tags=jnp.asarray([7], dtype=jnp.int32))

    text = out_path.read_text(encoding="utf-8")
    assert "$MeshFormat" in text
    assert "$Nodes" in text
    assert "$Elements" in text
    assert "1 2 2 7 7 1 2 3" in text
    assert "1 0 0 0" in text


def test_export_gmsh_msh_quad_and_tet_types(tmp_path: Path) -> None:
    quad_topo, quad_pts = unit_square_quad_mesh(3, 3)
    tet_topo, tet_pts = unit_cube_tet_mesh(2, 2, 2)
    quad_path = tmp_path / "quad_mesh.msh"
    tet_path = tmp_path / "tet_mesh.msh"

    export_gmsh_msh(quad_path, quad_pts, quad_topo.elements, element_entity_tags=quad_topo.element_entity_tags)
    export_gmsh_msh(tet_path, tet_pts, tet_topo.elements, element_entity_tags=tet_topo.element_entity_tags)

    quad_text = quad_path.read_text(encoding="utf-8")
    tet_text = tet_path.read_text(encoding="utf-8")
    assert " 3 2 1 1 " in quad_text
    assert " 4 2 1 1 " in tet_text


def test_export_gmsh_msh_triangle_roundtrip_preserves_mixed_block_metadata(tmp_path: Path) -> None:
    points = jnp.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=jnp.float32,
    )
    elements = jnp.asarray([[0, 1, 2], [1, 3, 2]], dtype=jnp.int32)
    path = tmp_path / "mixed_tri.msh"
    block = GmshElementBlock(
        elements=jnp.asarray([[0, 1], [1, 3], [3, 2], [2, 0]], dtype=jnp.int32),
        element_kind="line",
        physical_tags=jnp.asarray([100, 100, 101, 101], dtype=jnp.int32),
        geometrical_tags=jnp.asarray([10, 10, 11, 11], dtype=jnp.int32),
    )
    physical_names = {
        (2, 7): "surface",
        (1, 100): "bottom_and_right",
        (1, 101): "top_and_left",
    }

    export_gmsh_msh(
        path,
        points,
        elements,
        physical_tags=jnp.asarray([7, 7], dtype=jnp.int32),
        geometrical_tags=jnp.asarray([3, 4], dtype=jnp.int32),
        extra_element_blocks=(block,),
        physical_names=physical_names,
    )

    imported = load_gmsh_msh(path, primary_element_kind="triangle")
    assert imported.primary_element_kind == "triangle"
    assert np.array_equal(np.asarray(imported.topology.elements), np.asarray(elements))
    assert np.array_equal(np.asarray(imported.primary_physical_tags), np.asarray([7, 7], dtype=np.int32))
    assert np.array_equal(np.asarray(imported.primary_geometrical_tags), np.asarray([3, 4], dtype=np.int32))
    assert np.array_equal(np.asarray(imported.topology.element_entity_tags), np.asarray([3, 4], dtype=np.int32))
    assert imported.physical_names == physical_names
    assert len(imported.extra_element_blocks) == 1

    extra = imported.extra_element_blocks[0]
    assert extra.element_kind == "line"
    assert np.array_equal(np.asarray(extra.elements), np.asarray(block.elements))
    assert np.array_equal(np.asarray(extra.physical_tags), np.asarray(block.physical_tags))
    assert np.array_equal(np.asarray(extra.geometrical_tags), np.asarray(block.geometrical_tags))


def test_export_gmsh_msh_tetra_roundtrip_preserves_surface_blocks_and_names(tmp_path: Path) -> None:
    points = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    elements = jnp.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=jnp.int32)
    surface_block = GmshElementBlock(
        elements=jnp.asarray([[0, 2, 1], [1, 2, 4], [1, 4, 3]], dtype=jnp.int32),
        element_kind="triangle",
        physical_tags=jnp.asarray([500, 501, 501], dtype=jnp.int32),
        geometrical_tags=jnp.asarray([60, 61, 61], dtype=jnp.int32),
    )
    path = tmp_path / "mixed_tet.msh"
    physical_names = {
        (3, 9): "volume",
        (2, 500): "base",
        (2, 501): "shell",
    }

    export_gmsh_msh(
        path,
        points,
        elements,
        physical_tags=jnp.asarray([9, 9], dtype=jnp.int32),
        geometrical_tags=jnp.asarray([21, 22], dtype=jnp.int32),
        extra_element_blocks=(surface_block,),
        physical_names=physical_names,
    )

    imported = load_gmsh_msh(path, primary_element_kind="tetra")
    assert imported.primary_element_kind == "tetra"
    assert np.array_equal(np.asarray(imported.topology.elements), np.asarray(elements))
    assert np.array_equal(np.asarray(imported.primary_physical_tags), np.asarray([9, 9], dtype=np.int32))
    assert np.array_equal(np.asarray(imported.primary_geometrical_tags), np.asarray([21, 22], dtype=np.int32))
    assert np.array_equal(np.asarray(imported.topology.element_entity_tags), np.asarray([21, 22], dtype=np.int32))
    assert imported.physical_names == physical_names
    assert len(imported.extra_element_blocks) == 1

    extra = imported.extra_element_blocks[0]
    assert extra.element_kind == "triangle"
    assert np.array_equal(np.asarray(extra.elements), np.asarray(surface_block.elements))
    assert np.array_equal(np.asarray(extra.physical_tags), np.asarray(surface_block.physical_tags))
    assert np.array_equal(np.asarray(extra.geometrical_tags), np.asarray(surface_block.geometrical_tags))


def _read_binary_stl_triangle_count(path: Path) -> int:
    data = path.read_bytes()
    assert len(data) >= 84
    return struct.unpack("<I", data[80:84])[0]


def test_export_binary_stl_triangle_quad_and_tet_surface_counts(tmp_path: Path) -> None:
    tri_points = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    tri_elements = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    tri_path = tmp_path / "tri_surface.stl"
    export_binary_stl(tri_path, tri_points, tri_elements)
    assert _read_binary_stl_triangle_count(tri_path) == 1
    assert tri_path.stat().st_size == 84 + 50

    quad_points = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=jnp.float32)
    quad_elements = jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32)
    quad_path = tmp_path / "quad_surface.stl"
    export_binary_stl(quad_path, quad_points, quad_elements)
    assert _read_binary_stl_triangle_count(quad_path) == 2
    assert quad_path.stat().st_size == 84 + 2 * 50

    tet_points = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    tet_elements = jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32)
    tet_path = tmp_path / "tet_surface.stl"
    export_binary_stl(tet_path, tet_points, tet_elements)
    assert _read_binary_stl_triangle_count(tet_path) == 4
    assert tet_path.stat().st_size == 84 + 4 * 50


def test_export_mode1_artifacts_can_include_binary_stl(tmp_path: Path) -> None:
    topo, points = unit_cube_tet_mesh(2, 2, 2)
    result = optimize_mode1_fixed_topology(points, topo, steps=2, step_size=0.01, diagnostics_every=1)
    artifacts = export_mode1_artifacts(tmp_path, result, prefix="tet", export_stl_surface=True)
    assert artifacts["stl"].exists()
    assert _read_binary_stl_triangle_count(artifacts["stl"]) > 0
