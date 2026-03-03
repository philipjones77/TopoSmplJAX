"""Differentiable mesh generators."""

from __future__ import annotations

import jax.numpy as jnp

from gmshjax.runtime import jax_float_dtype


def unit_square_points(nx: int, ny: int, dtype=None) -> jnp.ndarray:
    """Create structured points in [0, 1] x [0, 1] as shape (nx*ny, 2)."""
    if dtype is None:
        dtype = jax_float_dtype()
    xs = jnp.linspace(0.0, 1.0, nx)
    ys = jnp.linspace(0.0, 1.0, ny)
    xs = xs.astype(dtype)
    ys = ys.astype(dtype)
    xx, yy = jnp.meshgrid(xs, ys, indexing="xy")
    return jnp.stack([xx.ravel(), yy.ravel()], axis=-1)


def unit_cube_points(nx: int, ny: int, nz: int, dtype=None) -> jnp.ndarray:
    """Create structured points in [0, 1]^3 as shape (nx*ny*nz, 3)."""
    if dtype is None:
        dtype = jax_float_dtype()
    xs = jnp.linspace(0.0, 1.0, nx)
    ys = jnp.linspace(0.0, 1.0, ny)
    zs = jnp.linspace(0.0, 1.0, nz)
    xs = xs.astype(dtype)
    ys = ys.astype(dtype)
    zs = zs.astype(dtype)
    xx, yy, zz = jnp.meshgrid(xs, ys, zs, indexing="xy")
    return jnp.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)


def project_cube_points_to_sphere(
    points: jnp.ndarray,
    radius: float = 1.0,
    center: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Map cube points to a sphere shell by radial normalization around center."""
    if center is None:
        center = jnp.array([0.5, 0.5, 0.5], dtype=points.dtype)
    d = points - center[None, :]
    n = jnp.linalg.norm(d, axis=1, keepdims=True)
    n = jnp.maximum(n, 1.0e-12)
    return center[None, :] + radius * d / n
