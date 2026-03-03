"""Boundary-driven parameterizations and constrained point placement."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class BoundaryCurves2D(NamedTuple):
    """Counter-clockwise boundary samples along each side of a quadrilateral patch."""

    bottom: jnp.ndarray  # (nx, 2), left->right
    right: jnp.ndarray  # (ny, 2), bottom->top
    top: jnp.ndarray  # (nx, 2), right->left or left->right accepted
    left: jnp.ndarray  # (ny, 2), top->bottom or bottom->top accepted


def line_segment(p0: jnp.ndarray, p1: jnp.ndarray, n: int) -> jnp.ndarray:
    """Sample n points on a line segment."""
    t = jnp.linspace(0.0, 1.0, n)
    return (1.0 - t)[:, None] * p0[None, :] + t[:, None] * p1[None, :]


def sinusoidal_top_boundary(x0: float, x1: float, y0: float, amp: float, n: int) -> jnp.ndarray:
    """Sample a sinusoidal top boundary y = y0 + amp * sin(2*pi*x_norm)."""
    x = jnp.linspace(x0, x1, n)
    xn = (x - x0) / jnp.maximum(x1 - x0, 1.0e-12)
    y = y0 + amp * jnp.sin(2.0 * jnp.pi * xn)
    return jnp.stack([x, y], axis=-1)


def transfinite_interpolation(curves: BoundaryCurves2D) -> jnp.ndarray:
    """Blend boundary curves into an interior grid using TFI.

    Output shape is (ny, nx, 2), with j-index bottom->top and i-index left->right.
    """
    bottom = curves.bottom
    right = curves.right
    top = curves.top
    left = curves.left

    nx = bottom.shape[0]
    ny = left.shape[0]

    # Normalize orientation to bottom->top / left->right conventions.
    if jnp.linalg.norm(top[0] - left[-1]) > jnp.linalg.norm(top[-1] - left[-1]):
        top = top[::-1]
    if jnp.linalg.norm(left[0] - bottom[0]) > jnp.linalg.norm(left[-1] - bottom[0]):
        left = left[::-1]
    if jnp.linalg.norm(right[0] - bottom[-1]) > jnp.linalg.norm(right[-1] - bottom[-1]):
        right = right[::-1]

    u = jnp.linspace(0.0, 1.0, nx)[None, :, None]
    v = jnp.linspace(0.0, 1.0, ny)[:, None, None]

    bottom_uv = jnp.broadcast_to(bottom[None, :, :], (ny, nx, 2))
    top_uv = jnp.broadcast_to(top[None, :, :], (ny, nx, 2))
    left_uv = jnp.broadcast_to(left[:, None, :], (ny, nx, 2))
    right_uv = jnp.broadcast_to(right[:, None, :], (ny, nx, 2))

    p00 = bottom[0]
    p10 = bottom[-1]
    p11 = top[-1]
    p01 = top[0]

    cblend = (
        (1.0 - u) * (1.0 - v) * p00[None, None, :]
        + u * (1.0 - v) * p10[None, None, :]
        + u * v * p11[None, None, :]
        + (1.0 - u) * v * p01[None, None, :]
    )
    sblend = (1.0 - v) * bottom_uv + v * top_uv + (1.0 - u) * left_uv + u * right_uv
    return sblend - cblend


def boundary_constrained_points(curves: BoundaryCurves2D) -> jnp.ndarray:
    """Return flattened points (N,2) from boundary-constrained TFI placement."""
    grid = transfinite_interpolation(curves)
    return grid.reshape((-1, 2))


def smooth_boundary_constrained_points(curves: BoundaryCurves2D, alpha: float = 0.2, steps: int = 10) -> jnp.ndarray:
    """Apply differentiable interior smoothing while keeping boundary fixed."""
    grid = transfinite_interpolation(curves)
    ny, nx, _ = grid.shape

    boundary_mask = jnp.zeros((ny, nx), dtype=bool)
    boundary_mask = boundary_mask.at[0, :].set(True)
    boundary_mask = boundary_mask.at[-1, :].set(True)
    boundary_mask = boundary_mask.at[:, 0].set(True)
    boundary_mask = boundary_mask.at[:, -1].set(True)

    def body(_, x):
        nbr = (
            jnp.roll(x, 1, axis=0)
            + jnp.roll(x, -1, axis=0)
            + jnp.roll(x, 1, axis=1)
            + jnp.roll(x, -1, axis=1)
        ) / 4.0
        y = x + alpha * (nbr - x)
        return jnp.where(boundary_mask[..., None], x, y)

    smoothed = jax.lax.fori_loop(0, steps, body, grid)
    return smoothed.reshape((-1, 2))
