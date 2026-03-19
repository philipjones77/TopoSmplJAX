"""Vectorized candidate evaluators for edge flips and smoothing."""

from __future__ import annotations

import jax.numpy as jnp


def _tri_signed_area(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    ab = b - a
    ac = c - a
    return 0.5 * (ab[..., 0] * ac[..., 1] - ab[..., 1] * ac[..., 0])


def evaluate_edge_flip_candidates(points: jnp.ndarray, quads: jnp.ndarray) -> jnp.ndarray:
    """Score diagonal flip candidates for quad patches.

    Input quads are (n,4) vertex ids [a,b,c,d] with current split (a,c).
    Returns score = post_quality - pre_quality (higher is better), shape (n,).
    """
    p = points[quads]  # (n,4,2)
    a = p[:, 0, :]
    b = p[:, 1, :]
    c = p[:, 2, :]
    d = p[:, 3, :]

    # Current split: (a,b,c) and (a,c,d); candidate split: (a,b,d) and (b,c,d)
    pre = jnp.abs(_tri_signed_area(a, b, c)) + jnp.abs(_tri_signed_area(a, c, d))
    post = jnp.abs(_tri_signed_area(a, b, d)) + jnp.abs(_tri_signed_area(b, c, d))
    # Penalize inverted candidate triangles.
    inv_pen = jnp.minimum(_tri_signed_area(a, b, d), 0.0) + jnp.minimum(_tri_signed_area(b, c, d), 0.0)
    return (post - pre) + 100.0 * inv_pen


def evaluate_laplacian_smoothing_candidates(
    points: jnp.ndarray,
    neighbor_sum: jnp.ndarray,
    neighbor_deg: jnp.ndarray,
    movable_mask: jnp.ndarray,
    alpha: float = 0.2,
) -> jnp.ndarray:
    """Evaluate candidate point displacements from one Laplacian step.

    Returns proposed points and displacement norms (both fixed-shape arrays).
    """
    mean_nbr = neighbor_sum / jnp.maximum(neighbor_deg[:, None], 1.0)
    proposed = points + alpha * (mean_nbr - points)
    proposed = jnp.where(movable_mask[:, None], proposed, points)
    disp = jnp.linalg.norm(proposed - points, axis=1)
    return proposed, disp
