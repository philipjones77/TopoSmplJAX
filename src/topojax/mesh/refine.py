"""Local adaptive refinement kernels with fixed-shape batched updates."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def triangle_area_magnitudes(points: jnp.ndarray, elements: jnp.ndarray) -> jnp.ndarray:
    tri = points[elements]
    a = tri[:, 0, :]
    b = tri[:, 1, :]
    c = tri[:, 2, :]
    ab = b - a
    ac = c - a
    cross2 = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    return 0.5 * jnp.abs(cross2)


def triangle_refinement_priority(points: jnp.ndarray, elements: jnp.ndarray, target_area: float) -> jnp.ndarray:
    """Priority score > 0 means element is larger than target area."""
    areas = triangle_area_magnitudes(points, elements)
    return areas / jnp.maximum(target_area, 1.0e-12) - 1.0


def select_refinement_candidates(priority: jnp.ndarray, k: int) -> jnp.ndarray:
    """Select top-k refinement candidate element indices (fixed output shape)."""
    k = int(k)
    # Keep only positive-priority entries; others become very negative.
    safe = jnp.where(priority > 0.0, priority, -jnp.inf)
    idx = jnp.argsort(safe)[-k:]
    return idx[::-1]


def refinement_midpoints(points: jnp.ndarray, elements: jnp.ndarray, candidate_idx: jnp.ndarray) -> jnp.ndarray:
    """Compute one centroid point per candidate element (fixed-shape batched op)."""
    tri = points[elements[candidate_idx]]
    return jnp.mean(tri, axis=1)


def batched_refinement_step(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    target_area: float,
    max_candidates: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return candidate ids, priorities and candidate centroid inserts.

    This keeps static output shapes for JAX compilation. It does not mutate connectivity yet;
    it is the evaluation/selection kernel used before applying a topology update.
    """
    priority = triangle_refinement_priority(points, elements, target_area)
    idx = select_refinement_candidates(priority, max_candidates)
    mids = refinement_midpoints(points, elements, idx)
    scores = priority[idx]
    return idx, scores, mids
