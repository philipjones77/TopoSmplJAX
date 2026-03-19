"""Differentiable connectivity surrogates on fixed candidate graphs."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _tri_area2(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])


def _tri_quality(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    e01 = b - a
    e12 = c - b
    e20 = a - c
    area2 = _tri_area2(a, b, c)
    edge2 = jnp.sum(e01 * e01, axis=-1) + jnp.sum(e12 * e12, axis=-1) + jnp.sum(e20 * e20, axis=-1)
    return 2.0 * jnp.sqrt(3.0) * jnp.abs(area2) / jnp.maximum(edge2, 1.0e-12)


def quad_split_candidate_qualities(points: jnp.ndarray, quads: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return per-quad quality for the two diagonal split candidates."""
    p = points[quads]
    a = p[:, 0, :]
    b = p[:, 1, :]
    c = p[:, 2, :]
    d = p[:, 3, :]

    ac_quality = 0.5 * (_tri_quality(a, b, c) + _tri_quality(a, c, d))
    bd_quality = 0.5 * (_tri_quality(a, b, d) + _tri_quality(b, c, d))
    return ac_quality, bd_quality


def soft_quad_diagonal_weights(logits: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
    """Differentiable weights over the two quad diagonal choices."""
    pair_logits = jnp.stack([logits, jnp.zeros_like(logits)], axis=-1)
    return jax.nn.softmax(pair_logits / jnp.maximum(temperature, 1.0e-6), axis=-1)


def soft_quad_connectivity_energy(
    points: jnp.ndarray,
    quads: jnp.ndarray,
    logits: jnp.ndarray,
    *,
    temperature: float = 0.25,
) -> jnp.ndarray:
    """Differentiable surrogate energy over a fixed quad candidate graph."""
    ac_quality, bd_quality = quad_split_candidate_qualities(points, quads)
    weights = soft_quad_diagonal_weights(logits, temperature=temperature)
    mixed_quality = weights[:, 0] * ac_quality + weights[:, 1] * bd_quality
    entropy = -jnp.sum(weights * jnp.log(jnp.maximum(weights, 1.0e-12)), axis=-1)
    return jnp.mean(jax.nn.softplus((0.7 - mixed_quality) * 10.0)) + 1.0e-3 * jnp.mean(entropy)
