"""Manifold parameterization and deformation maps."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class DeformationParams(NamedTuple):
    """Low-dimensional parameters for smooth mesh deformation."""

    translation: jnp.ndarray  # (2,)
    scale: jnp.ndarray  # (2,)
    shear: jnp.ndarray  # (2,)  [xy, yx]
    bend: jnp.ndarray  # (2,)  sinusoidal amplitudes


def apply_deformation(reference_points: jnp.ndarray, params: DeformationParams) -> jnp.ndarray:
    """Map reference points to deformed manifold coordinates.

    The mesh topology remains unchanged; only coordinates are transformed.
    """
    x = reference_points[:, 0]
    y = reference_points[:, 1]

    sx = params.scale[0] * x
    sy = params.scale[1] * y
    shx = params.shear[0] * y
    shy = params.shear[1] * x

    bx = params.bend[0] * jnp.sin(2.0 * jnp.pi * y)
    by = params.bend[1] * jnp.sin(2.0 * jnp.pi * x)

    out_x = sx + shx + bx + params.translation[0]
    out_y = sy + shy + by + params.translation[1]
    return jnp.stack([out_x, out_y], axis=-1)
