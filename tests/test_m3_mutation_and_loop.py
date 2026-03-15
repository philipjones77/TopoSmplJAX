from pathlib import Path

import numpy as np
import jax.numpy as jnp

from gmshjax.io.exports import export_snapshot_npz
from gmshjax.mesh.adaptive import adaptive_remesh_tri
from gmshjax.mesh.mutation import active_elements, make_tri_mesh_buffer, split_triangle, flip_diagonal
from gmshjax.mesh.topology import unit_square_tri_mesh


def test_split_triangle_increases_counts() -> None:
    topo, pts = unit_square_tri_mesh(4, 4)
    buf = make_tri_mesh_buffer(pts, topo.elements, max_nodes=128, max_elements=256)
    n0 = buf.node_count
    e0 = buf.element_count
    out, ok = split_triangle(buf, 0)
    assert ok
    assert out.node_count == n0 + 1
    assert out.element_count == e0 + 2


def test_flip_diagonal_on_two_triangles() -> None:
    points = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=jnp.float32)
    elems = jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
    buf = make_tri_mesh_buffer(points, elems, max_nodes=16, max_elements=16)
    out, ok = flip_diagonal(buf, 0, 1)
    assert ok
    new_e = np.asarray(active_elements(out))
    # After flip there should be edge (1,3) in one of the triangles.
    assert 1 in new_e and 3 in new_e


def test_adaptive_loop_writes_snapshots(tmp_path: Path) -> None:
    topo, pts = unit_square_tri_mesh(12, 8)
    out, hist = adaptive_remesh_tri(
        pts,
        topo.elements,
        max_nodes=4096,
        max_elements=8192,
        max_iters=5,
        target_area=0.01,
        target_mean_icn=0.4,
        snapshot_dir=tmp_path,
    )
    assert len(hist) >= 1
    snaps = sorted(tmp_path.glob("iter_*.npz"))
    assert len(snaps) >= 1
    assert out.node_count >= pts.shape[0]


def test_adaptive_loop_can_record_collapse() -> None:
    topo, pts = unit_square_tri_mesh(4, 4)
    out, hist = adaptive_remesh_tri(
        pts,
        topo.elements,
        max_nodes=128,
        max_elements=256,
        max_iters=1,
        target_area=0.054,
        target_mean_icn=2.0,
        smoothing_steps=0,
    )
    assert len(hist) >= 1
    assert any(step.did_collapse for step in hist)
    assert out.node_count == pts.shape[0]


def test_export_snapshot_npz_roundtrip(tmp_path: Path) -> None:
    p = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    e = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    fp = tmp_path / "snap.npz"
    export_snapshot_npz(fp, p, e, metrics={"foo": 1.5})
    arr = np.load(fp)
    assert arr["points"].shape == (3, 2)
    assert arr["elements"].shape == (1, 3)
    assert float(arr["metric_foo"]) == 1.5
