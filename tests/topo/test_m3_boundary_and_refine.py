import numpy as np
import jax.numpy as jnp

from topojax.mesh.boundary import (
    BoundaryCurves2D,
    boundary_constrained_points,
    line_segment,
    sinusoidal_top_boundary,
    transfinite_interpolation,
)
from topojax.mesh.connectivity_opt import evaluate_edge_flip_candidates, evaluate_laplacian_smoothing_candidates
from topojax.mesh.refine import batched_refinement_step
from topojax.mesh.topology import unit_square_tri_mesh


def _unit_patch_curves(nx: int, ny: int) -> BoundaryCurves2D:
    bottom = line_segment(jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0]), nx)
    top = sinusoidal_top_boundary(0.0, 1.0, 1.0, 0.05, nx)
    left = line_segment(bottom[0], top[0], ny)
    right = line_segment(bottom[-1], top[-1], ny)
    return BoundaryCurves2D(bottom=bottom, right=right, top=top, left=left)


def test_tfi_matches_boundary_samples() -> None:
    nx, ny = 20, 14
    curves = _unit_patch_curves(nx, ny)
    grid = transfinite_interpolation(curves)
    assert np.allclose(np.asarray(grid[0]), np.asarray(curves.bottom), atol=1e-6)
    assert np.allclose(np.asarray(grid[-1]), np.asarray(curves.top), atol=1e-6)
    assert np.allclose(np.asarray(grid[:, 0]), np.asarray(curves.left), atol=1e-6)
    assert np.allclose(np.asarray(grid[:, -1]), np.asarray(curves.right), atol=1e-6)


def test_boundary_constrained_points_shape() -> None:
    curves = _unit_patch_curves(17, 11)
    pts = boundary_constrained_points(curves)
    assert pts.shape == (17 * 11, 2)


def test_batched_refinement_step_fixed_shapes() -> None:
    topo, points = unit_square_tri_mesh(18, 12)
    idx, scores, mids = batched_refinement_step(points, topo.elements, target_area=0.0015, max_candidates=16)
    assert idx.shape == (16,)
    assert scores.shape == (16,)
    assert mids.shape == (16, 2)


def test_edge_flip_candidates_vectorized() -> None:
    points = jnp.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    quads = jnp.asarray([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=jnp.int32)
    scores = evaluate_edge_flip_candidates(points, quads)
    assert scores.shape == (2,)


def test_laplacian_candidate_eval_shapes() -> None:
    topo, points = unit_square_tri_mesh(10, 8)
    n = points.shape[0]
    src = topo.edges[:, 0]
    dst = topo.edges[:, 1]
    neighbor_sum = jnp.zeros_like(points)
    neighbor_sum = neighbor_sum.at[src].add(points[dst])
    neighbor_sum = neighbor_sum.at[dst].add(points[src])
    deg = jnp.zeros((n,), dtype=points.dtype)
    one = jnp.ones((topo.edges.shape[0],), dtype=points.dtype)
    deg = deg.at[src].add(one)
    deg = deg.at[dst].add(one)
    movable = jnp.ones((n,), dtype=bool)
    proposed, disp = evaluate_laplacian_smoothing_candidates(points, neighbor_sum, deg, movable, alpha=0.2)
    assert proposed.shape == points.shape
    assert disp.shape == (n,)
