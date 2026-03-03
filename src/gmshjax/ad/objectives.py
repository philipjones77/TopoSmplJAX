"""AD objectives for mesh quality optimization."""

from __future__ import annotations

import jax.numpy as jnp


def centroid_target_loss(points: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Squared distance between mesh centroid and a target point."""
    c = jnp.mean(points, axis=0)
    d = c - target
    return jnp.dot(d, d)
