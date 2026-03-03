from pathlib import Path

import jax.numpy as jnp

from gmshjax.mesh.adaptive_quad import adaptive_remesh_quad
from gmshjax.mesh.generators import project_cube_points_to_sphere
from gmshjax.mesh.mutation_qt import make_quad_mesh_buffer, split_quad
from gmshjax.mesh.topology import unit_cube_tet_mesh, unit_square_quad_mesh


def test_split_quad_increases_counts() -> None:
    topo, pts = unit_square_quad_mesh(5, 4)
    buf = make_quad_mesh_buffer(pts, topo.elements, max_nodes=512, max_elements=512)
    n0 = buf.node_count
    e0 = buf.element_count
    out, ok = split_quad(buf, 0)
    assert ok
    assert out.node_count == n0 + 5
    assert out.element_count == e0 + 3


def test_adaptive_quad_loop_writes_snapshots(tmp_path: Path) -> None:
    topo, pts = unit_square_quad_mesh(12, 8)
    out, hist = adaptive_remesh_quad(
        pts,
        topo.elements,
        max_nodes=5000,
        max_elements=8000,
        max_iters=4,
        target_area=0.02,
        target_mean_icn=0.4,
        snapshot_dir=tmp_path,
    )
    assert len(hist) >= 1
    assert out.node_count >= pts.shape[0]
    assert len(list(tmp_path.glob("iter_*.npz"))) >= 1


def test_cube_to_sphere_projection_radius() -> None:
    topo, pts = unit_cube_tet_mesh(6, 6, 6)
    sph = project_cube_points_to_sphere(pts, radius=1.0, center=jnp.array([0.5, 0.5, 0.5], dtype=pts.dtype))
    r = jnp.linalg.norm(sph - jnp.array([0.5, 0.5, 0.5], dtype=pts.dtype), axis=1)
    assert jnp.max(jnp.abs(r - 1.0)) < 1.0e-4
