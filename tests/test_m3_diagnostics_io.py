from pathlib import Path

import jax.numpy as jnp

from gmshjax.io.exports import export_metrics_json, export_snapshot_npz, load_snapshot_npz
from gmshjax.mesh.diagnostics import quad_diagnostics, tet_diagnostics, tri_diagnostics
from gmshjax.mesh.topology import unit_cube_tet_mesh, unit_square_quad_mesh, unit_square_tri_mesh


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
