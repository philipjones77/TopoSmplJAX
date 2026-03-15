"""Straight-through estimators for discrete connectivity choices."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gmshjax.ad.surrogate import quad_split_candidate_qualities, soft_quad_diagonal_weights


def straight_through_quad_diagonal_weights(logits: jnp.ndarray, temperature: float = 0.25) -> jnp.ndarray:
    """Hard forward quad split choice with soft backward gradients."""
    soft = soft_quad_diagonal_weights(logits, temperature=temperature)
    hard_index = jnp.argmax(soft, axis=-1)
    hard = jax.nn.one_hot(hard_index, 2, dtype=soft.dtype)
    return hard + soft - jax.lax.stop_gradient(soft)


def straight_through_quad_connectivity_energy(
    points: jnp.ndarray,
    quads: jnp.ndarray,
    logits: jnp.ndarray,
    *,
    temperature: float = 0.25,
) -> jnp.ndarray:
    """Use hard split choices in the forward pass and soft gradients in backward."""
    ac_quality, bd_quality = quad_split_candidate_qualities(points, quads)
    weights = straight_through_quad_diagonal_weights(logits, temperature=temperature)
    mixed_quality = weights[:, 0] * ac_quality + weights[:, 1] * bd_quality
    return jnp.mean(jax.nn.softplus((0.7 - mixed_quality) * 10.0))
