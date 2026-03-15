import jax
import numpy as np
import jax.numpy as jnp

from gmshjax.mesh.boundary import (
    BoundaryCurves3D,
    cleanup_uv_tri_mesh,
    evaluate_surface_patch,
    line_segment,
    surface_boundary_constrained_points,
    surface_front_tri_mesh,
    surface_parametric_point_cloud,
    surface_point_cloud,
    surface_transfinite_interpolation,
    uv_triangle_quality_objective,
)
from gmshjax.mesh.mutation import collapse_triangle, make_tri_mesh_buffer, split_triangle
from gmshjax.mesh.mutation_qt import collapse_quad, collapse_tet, make_quad_mesh_buffer, make_tet_mesh_buffer, split_quad, split_tet
from gmshjax.mesh.topology import unit_cube_tet_mesh, unit_square_quad_mesh, unit_square_tri_mesh
from gmshjax.runtime import get_runtime_precision, set_runtime_precision


def _planar_surface_patch(n_u: int, n_v: int) -> BoundaryCurves3D:
    bottom = line_segment(jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.2]), n_u)
    top = line_segment(jnp.array([0.0, 1.0, 0.4]), jnp.array([1.0, 1.0, 0.6]), n_u)
    left = line_segment(bottom[0], top[0], n_v)
    right = line_segment(bottom[-1], top[-1], n_v)
    return BoundaryCurves3D(bottom=bottom, right=right, top=top, left=left)


def test_surface_tfi_matches_boundary_samples() -> None:
    curves = _planar_surface_patch(17, 11)
    grid = surface_transfinite_interpolation(curves)
    assert np.allclose(np.asarray(grid[0]), np.asarray(curves.bottom), atol=1.0e-6)
    assert np.allclose(np.asarray(grid[-1]), np.asarray(curves.top), atol=1.0e-6)
    assert np.allclose(np.asarray(grid[:, 0]), np.asarray(curves.left), atol=1.0e-6)
    assert np.allclose(np.asarray(grid[:, -1]), np.asarray(curves.right), atol=1.0e-6)


def test_surface_boundary_points_and_patch_eval_shapes() -> None:
    curves = _planar_surface_patch(13, 9)
    pts = surface_boundary_constrained_points(curves)
    eval_pts = evaluate_surface_patch(curves, jnp.asarray([[0.0, 0.0], [0.25, 0.5], [1.0, 1.0]], dtype=jnp.float32))
    assert pts.shape == (13 * 9, 3)
    assert eval_pts.shape == (3, 3)
    assert np.allclose(np.asarray(eval_pts[0]), np.asarray(curves.bottom[0]), atol=1.0e-6)
    assert np.allclose(np.asarray(eval_pts[-1]), np.asarray(curves.top[-1]), atol=1.0e-6)


def test_surface_point_cloud_is_irregular_and_on_patch() -> None:
    curves = _planar_surface_patch(15, 12)
    cloud = surface_point_cloud(curves, 6, 5, jitter=0.4, relaxation_steps=2, seed=7)
    structured = surface_transfinite_interpolation(curves).reshape((-1, 3))
    boundary_count = curves.bottom.shape[0] + curves.right.shape[0] + curves.top.shape[0] + curves.left.shape[0] - 4
    assert cloud.shape == (boundary_count + 30, 3)
    assert not np.allclose(np.asarray(cloud[-30:]), np.asarray(structured[:30]), atol=1.0e-3)
    assert float(jnp.min(cloud[:, 2])) >= -1.0e-6
    assert float(jnp.max(cloud[:, 2])) <= 0.600001


def test_surface_parametric_point_cloud_and_front_mesh_shapes() -> None:
    curves = _planar_surface_patch(15, 12)
    uv, pts = surface_parametric_point_cloud(curves, 6, 5, seed=4)
    topo, surf_pts, surf_uv = surface_front_tri_mesh(curves, 6, 5, seed=4)
    topo_base, surf_pts_base, surf_uv_base = surface_front_tri_mesh(curves, 6, 5, seed=4, front_rings=0)
    assert uv.shape[1] == 2
    assert pts.shape[1] == 3
    assert surf_pts.shape[1] == 3
    assert surf_uv.shape[1] == 2
    assert surf_pts.shape[0] == surf_uv.shape[0]
    assert surf_pts.shape[0] >= pts.shape[0]
    assert surf_pts.shape[0] > surf_pts_base.shape[0]
    assert surf_uv.shape[0] > surf_uv_base.shape[0]
    assert topo.elements.shape[1] == 3
    assert topo.edges.shape[1] == 2
    assert topo.n_nodes == surf_pts.shape[0]
    area2 = (surf_uv[topo.elements[:, 1], 0] - surf_uv[topo.elements[:, 0], 0]) * (surf_uv[topo.elements[:, 2], 1] - surf_uv[topo.elements[:, 0], 1]) - (
        surf_uv[topo.elements[:, 1], 1] - surf_uv[topo.elements[:, 0], 1]
    ) * (surf_uv[topo.elements[:, 2], 0] - surf_uv[topo.elements[:, 0], 0])
    assert jnp.all(area2 > 0.0)
    boundary_count = curves.bottom.shape[0] + curves.right.shape[0] + curves.top.shape[0] + curves.left.shape[0] - 4
    expected_boundary_edges = {tuple(sorted((i, (i + 1) % boundary_count))) for i in range(boundary_count)}
    actual_edges = {tuple(sorted((int(a), int(b)))) for a, b in np.asarray(topo.edges).tolist()}
    assert expected_boundary_edges.issubset(actual_edges)


def test_surface_point_cloud_respects_runtime_float64_precision() -> None:
    original = get_runtime_precision()
    try:
        set_runtime_precision("float64")
        curves = _planar_surface_patch(15, 12)
        uv, pts = surface_parametric_point_cloud(curves, 6, 5, seed=4)
        cloud = surface_point_cloud(curves, 6, 5, seed=4, include_boundary=False)
        assert uv.dtype == jnp.float64
        assert pts.dtype == jnp.float64
        assert cloud.dtype == jnp.float64
    finally:
        set_runtime_precision(original)


def test_surface_front_tri_mesh_preserves_boundary_with_front_rings() -> None:
    curves = _planar_surface_patch(15, 12)
    topo, _, surf_uv = surface_front_tri_mesh(curves, 6, 5, seed=4, front_rings=2)
    boundary_count = curves.bottom.shape[0] + curves.right.shape[0] + curves.top.shape[0] + curves.left.shape[0] - 4
    expected_boundary_edges = {tuple(sorted((i, (i + 1) % boundary_count))) for i in range(boundary_count)}
    actual_edges = {tuple(sorted((int(a), int(b)))) for a, b in np.asarray(topo.edges).tolist()}
    assert expected_boundary_edges.issubset(actual_edges)
    area2 = (surf_uv[topo.elements[:, 1], 0] - surf_uv[topo.elements[:, 0], 0]) * (surf_uv[topo.elements[:, 2], 1] - surf_uv[topo.elements[:, 0], 1]) - (
        surf_uv[topo.elements[:, 1], 1] - surf_uv[topo.elements[:, 0], 1]
    ) * (surf_uv[topo.elements[:, 2], 0] - surf_uv[topo.elements[:, 0], 0])
    assert jnp.all(area2 > 0.0)


def test_uv_quality_objective_is_autodiffable() -> None:
    curves = _planar_surface_patch(15, 12)
    topo, _, uv = surface_front_tri_mesh(curves, 6, 5, seed=4)

    def objective(uv_coords: jnp.ndarray) -> jnp.ndarray:
        return uv_triangle_quality_objective(uv_coords, topo.elements)

    grad = jax.grad(objective)(uv)
    assert grad.shape == uv.shape
    assert jnp.all(jnp.isfinite(grad))


def test_cleanup_uv_tri_mesh_improves_quality_objective() -> None:
    curves = _planar_surface_patch(15, 12)
    topo, _, uv = surface_front_tri_mesh(curves, 6, 5, seed=6, cleanup_steps=0, cleanup_flip_passes=0)
    uv_bad = uv.at[-1].add(jnp.asarray([0.04, -0.03], dtype=uv.dtype))
    before = float(uv_triangle_quality_objective(uv_bad, topo.elements))
    boundary_count = curves.bottom.shape[0] + curves.right.shape[0] + curves.top.shape[0] + curves.left.shape[0] - 4
    uv_clean, elements_clean = cleanup_uv_tri_mesh(
        uv_bad,
        topo.elements,
        boundary_count=boundary_count,
        smoothing_steps=4,
        flip_passes=2,
    )
    after = float(uv_triangle_quality_objective(uv_clean, elements_clean))
    assert after <= before + 1.0e-8


def test_surface_patch_is_autodiffable() -> None:
    curves = _planar_surface_patch(17, 11)

    def objective(top_curve: jnp.ndarray) -> jnp.ndarray:
        updated = BoundaryCurves3D(bottom=curves.bottom, right=curves.right, top=top_curve, left=curves.left)
        uv = jnp.asarray([[0.2, 0.3], [0.7, 0.8]], dtype=top_curve.dtype)
        return jnp.sum(evaluate_surface_patch(updated, uv)[:, 2])

    grad = jax.grad(objective)(curves.top)
    assert grad.shape == curves.top.shape
    assert jnp.all(jnp.isfinite(grad))
    assert float(jnp.max(jnp.abs(grad))) > 0.0


def test_surface_point_cloud_is_autodiffable_with_fixed_seed() -> None:
    curves = _planar_surface_patch(15, 12)

    def objective(top_curve: jnp.ndarray) -> jnp.ndarray:
        updated = BoundaryCurves3D(bottom=curves.bottom, right=curves.right, top=top_curve, left=curves.left)
        pts = surface_point_cloud(updated, 5, 4, jitter=0.25, relaxation_steps=1, seed=3, include_boundary=False)
        return jnp.mean(pts[:, 2])

    grad = jax.grad(objective)(curves.top)
    assert grad.shape == curves.top.shape
    assert jnp.all(jnp.isfinite(grad))


def test_collapse_triangle_restores_counts_after_split() -> None:
    topo, pts = unit_square_tri_mesh(4, 4)
    buf = make_tri_mesh_buffer(pts, topo.elements, max_nodes=128, max_elements=256)
    n0 = buf.node_count
    e0 = buf.element_count
    split_buf, ok = split_triangle(buf, 0)
    assert ok
    collapsed, ok = collapse_triangle(split_buf, split_buf.node_count - 1)
    assert ok
    assert collapsed.node_count == n0
    assert collapsed.element_count == e0


def test_collapse_quad_restores_counts_after_split() -> None:
    topo, pts = unit_square_quad_mesh(4, 4)
    buf = make_quad_mesh_buffer(pts, topo.elements, max_nodes=256, max_elements=256)
    n0 = buf.node_count
    e0 = buf.element_count
    split_buf, ok = split_quad(buf, 0)
    assert ok
    collapsed, ok = collapse_quad(split_buf, split_buf.node_count - 1)
    assert ok
    assert collapsed.node_count == n0
    assert collapsed.element_count == e0


def test_collapse_tet_restores_counts_after_split() -> None:
    topo, pts = unit_cube_tet_mesh(2, 2, 2)
    buf = make_tet_mesh_buffer(pts, topo.elements, max_nodes=256, max_elements=256)
    n0 = buf.node_count
    e0 = buf.element_count
    split_buf, ok = split_tet(buf, 0)
    assert ok
    collapsed, ok = collapse_tet(split_buf, split_buf.node_count - 1)
    assert ok
    assert collapsed.node_count == n0
    assert collapsed.element_count == e0
