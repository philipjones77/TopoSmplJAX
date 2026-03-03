"""M3 demo: boundary-constrained mesh + candidate refinement/optimization kernels."""

from __future__ import annotations

import jax.numpy as jnp

from gmshjax.mesh.boundary import (
    BoundaryCurves2D,
    boundary_constrained_points,
    line_segment,
    sinusoidal_top_boundary,
)
from gmshjax.mesh.connectivity_opt import evaluate_edge_flip_candidates
from gmshjax.mesh.refine import batched_refinement_step
from gmshjax.mesh.topology import structured_triangles


def make_curves(nx: int, ny: int) -> BoundaryCurves2D:
    bottom = line_segment(jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0]), nx)
    top = sinusoidal_top_boundary(0.0, 1.0, 1.0, 0.08, nx)
    left = line_segment(bottom[0], top[0], ny)
    right = line_segment(bottom[-1], top[-1], ny)
    return BoundaryCurves2D(bottom=bottom, right=right, top=top, left=left)


def main() -> None:
    nx, ny = 40, 24
    curves = make_curves(nx, ny)
    points = boundary_constrained_points(curves)
    elements = structured_triangles(nx, ny)

    idx, scores, mids = batched_refinement_step(points, elements, target_area=1.5e-3, max_candidates=12)

    # Build small quad patches from candidate triangles for diagonal-flip scoring.
    tri = elements[idx[:4]]
    a = tri[:, 0]
    b = tri[:, 1]
    c = tri[:, 2]
    d = a  # placeholder to keep fixed shape; real patch extraction is next milestone.
    quads = jnp.stack([a, b, c, d], axis=1)
    flip_scores = evaluate_edge_flip_candidates(points, quads)

    print("points_shape:", points.shape)
    print("elements_shape:", elements.shape)
    print("top_refine_idx:", idx)
    print("top_refine_scores:", scores)
    print("new_midpoints_shape:", mids.shape)
    print("flip_scores:", flip_scores)


if __name__ == "__main__":
    main()
