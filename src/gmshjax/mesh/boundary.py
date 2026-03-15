"""Boundary-driven parameterizations and constrained point placement."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gmshjax.runtime import jax_float_dtype

from .topology import MeshTopology, triangle_edges
from .triangulation import delaunay_triangles_2d


class BoundaryCurves2D(NamedTuple):
    """Counter-clockwise boundary samples along each side of a quadrilateral patch."""

    bottom: jnp.ndarray  # (nx, 2), left->right
    right: jnp.ndarray  # (ny, 2), bottom->top
    top: jnp.ndarray  # (nx, 2), right->left or left->right accepted
    left: jnp.ndarray  # (ny, 2), top->bottom or bottom->top accepted


class BoundaryCurves3D(NamedTuple):
    """Boundary samples along each side of a quadrilateral surface patch in 3D."""

    bottom: jnp.ndarray  # (nu, 3), left->right
    right: jnp.ndarray  # (nv, 3), bottom->top
    top: jnp.ndarray  # (nu, 3), right->left or left->right accepted
    left: jnp.ndarray  # (nv, 3), top->bottom or bottom->top accepted


def line_segment(p0: jnp.ndarray, p1: jnp.ndarray, n: int) -> jnp.ndarray:
    """Sample n points on a line segment in 2D or 3D."""
    t = jnp.linspace(0.0, 1.0, n)
    return (1.0 - t)[:, None] * p0[None, :] + t[:, None] * p1[None, :]


def _normalize_patch_boundaries(bottom: jnp.ndarray, right: jnp.ndarray, top: jnp.ndarray, left: jnp.ndarray):
    if jnp.linalg.norm(top[0] - left[-1]) > jnp.linalg.norm(top[-1] - left[-1]):
        top = top[::-1]
    if jnp.linalg.norm(left[0] - bottom[0]) > jnp.linalg.norm(left[-1] - bottom[0]):
        left = left[::-1]
    if jnp.linalg.norm(right[0] - bottom[-1]) > jnp.linalg.norm(right[-1] - bottom[-1]):
        right = right[::-1]
    return bottom, right, top, left


def _sample_curve(curve: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    t = jnp.clip(jnp.asarray(t, dtype=curve.dtype), 0.0, 1.0)
    s = t * float(curve.shape[0] - 1)
    i0 = jnp.floor(s).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, curve.shape[0] - 1)
    w = (s - i0.astype(curve.dtype))[..., None]
    return (1.0 - w) * curve[i0] + w * curve[i1]


def _coons_patch(bottom: jnp.ndarray, right: jnp.ndarray, top: jnp.ndarray, left: jnp.ndarray, u, v) -> jnp.ndarray:
    u = jnp.asarray(u, dtype=bottom.dtype)
    v = jnp.asarray(v, dtype=bottom.dtype)
    bottom_u = _sample_curve(bottom, u)
    top_u = _sample_curve(top, u)
    left_v = _sample_curve(left, v)
    right_v = _sample_curve(right, v)

    p00 = bottom[0]
    p10 = bottom[-1]
    p11 = top[-1]
    p01 = top[0]

    u1 = u[..., None]
    v1 = v[..., None]
    cblend = (1.0 - u1) * (1.0 - v1) * p00 + u1 * (1.0 - v1) * p10 + u1 * v1 * p11 + (1.0 - u1) * v1 * p01
    sblend = (1.0 - v1) * bottom_u + v1 * top_u + (1.0 - u1) * left_v + u1 * right_v
    return sblend - cblend


def _surface_boundary_loop(bottom: jnp.ndarray, right: jnp.ndarray, top: jnp.ndarray, left: jnp.ndarray) -> jnp.ndarray:
    pieces = [bottom]
    if right.shape[0] > 1:
        pieces.append(right[1:])
    if top.shape[0] > 1:
        pieces.append(top[-2::-1])
    if left.shape[0] > 2:
        pieces.append(left[-2:0:-1])
    return jnp.concatenate(pieces, axis=0)


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

    bottom, right, top, left = _normalize_patch_boundaries(bottom, right, top, left)

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


def surface_transfinite_interpolation(curves: BoundaryCurves3D) -> jnp.ndarray:
    """Blend four 3D boundary curves into a quadrilateral surface patch."""
    bottom, right, top, left = _normalize_patch_boundaries(curves.bottom, curves.right, curves.top, curves.left)
    nu = bottom.shape[0]
    nv = left.shape[0]
    uu, vv = jnp.meshgrid(
        jnp.linspace(0.0, 1.0, nu, dtype=bottom.dtype),
        jnp.linspace(0.0, 1.0, nv, dtype=bottom.dtype),
        indexing="xy",
    )
    return _coons_patch(bottom, right, top, left, uu, vv)


def surface_boundary_constrained_points(curves: BoundaryCurves3D) -> jnp.ndarray:
    """Return flattened points from a 3D boundary-constrained surface patch."""
    return surface_transfinite_interpolation(curves).reshape((-1, 3))


def smooth_surface_boundary_constrained_points(
    curves: BoundaryCurves3D,
    alpha: float = 0.2,
    steps: int = 10,
) -> jnp.ndarray:
    """Apply differentiable interior smoothing while keeping the 3D boundary fixed."""
    grid = surface_transfinite_interpolation(curves)
    nv, nu, _ = grid.shape

    boundary_mask = jnp.zeros((nv, nu), dtype=bool)
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
    return smoothed.reshape((-1, 3))


def evaluate_surface_patch(curves: BoundaryCurves3D, uv: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a 3D Coons patch at arbitrary parametric coordinates in [0, 1]^2."""
    bottom, right, top, left = _normalize_patch_boundaries(curves.bottom, curves.right, curves.top, curves.left)
    uv = jnp.asarray(uv, dtype=bottom.dtype)
    return _coons_patch(bottom, right, top, left, uv[..., 0], uv[..., 1])


def surface_point_cloud(
    curves: BoundaryCurves3D,
    n_u: int,
    n_v: int,
    *,
    jitter: float = 0.35,
    relaxation_steps: int = 3,
    seed: int = 0,
    include_boundary: bool = True,
) -> jnp.ndarray:
    """Generate an irregular surface-driven point cloud on a 3D patch."""
    if n_u <= 0 or n_v <= 0:
        raise ValueError("n_u and n_v must be positive")

    key = jax.random.PRNGKey(seed)
    dtype = jax_float_dtype()
    u = (jnp.arange(n_u, dtype=dtype) + 0.5) / float(n_u)
    v = (jnp.arange(n_v, dtype=dtype) + 0.5) / float(n_v)
    uu, vv = jnp.meshgrid(u, v, indexing="xy")
    uv = jnp.stack([uu.reshape((-1,)), vv.reshape((-1,))], axis=-1)

    delta = jax.random.uniform(key, uv.shape, minval=-1.0, maxval=1.0, dtype=uv.dtype)
    scale = jnp.asarray([0.5 / float(n_u), 0.5 / float(n_v)], dtype=uv.dtype)
    eps = jnp.asarray([0.25 / float(n_u), 0.25 / float(n_v)], dtype=uv.dtype)
    uv = jnp.clip(uv + jitter * delta * scale, eps, 1.0 - eps)
    sigma2 = float(max(1.0 / (n_u ** 2), 1.0 / (n_v ** 2)))
    for _ in range(relaxation_steps):
        diff = uv[:, None, :] - uv[None, :, :]
        dist2 = jnp.sum(diff * diff, axis=-1)
        weight = jnp.exp(-dist2 / sigma2) * (1.0 - jnp.eye(uv.shape[0], dtype=uv.dtype))
        repulse = jnp.sum(diff * weight[..., None], axis=1)
        denom = jnp.maximum(jnp.sum(weight, axis=1, keepdims=True), 1.0)
        uv = jnp.clip(uv + 0.08 * repulse / denom, eps, 1.0 - eps)

    pts = evaluate_surface_patch(curves, uv)
    if not include_boundary:
        return pts

    bottom, right, top, left = _normalize_patch_boundaries(curves.bottom, curves.right, curves.top, curves.left)
    boundary = _surface_boundary_loop(bottom, right, top, left)
    return jnp.concatenate([boundary, pts], axis=0)


def surface_parametric_point_cloud(
    curves: BoundaryCurves3D,
    n_u: int,
    n_v: int,
    *,
    jitter: float = 0.35,
    relaxation_steps: int = 3,
    seed: int = 0,
    include_boundary: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return parametric coordinates and 3D points for an irregular surface sample set."""
    if n_u <= 0 or n_v <= 0:
        raise ValueError("n_u and n_v must be positive")

    key = jax.random.PRNGKey(seed)
    dtype = jax_float_dtype()
    u = (jnp.arange(n_u, dtype=dtype) + 0.5) / float(n_u)
    v = (jnp.arange(n_v, dtype=dtype) + 0.5) / float(n_v)
    uu, vv = jnp.meshgrid(u, v, indexing="xy")
    uv = jnp.stack([uu.reshape((-1,)), vv.reshape((-1,))], axis=-1)

    delta = jax.random.uniform(key, uv.shape, minval=-1.0, maxval=1.0, dtype=uv.dtype)
    scale = jnp.asarray([0.5 / float(n_u), 0.5 / float(n_v)], dtype=uv.dtype)
    eps = jnp.asarray([0.25 / float(n_u), 0.25 / float(n_v)], dtype=uv.dtype)
    uv = jnp.clip(uv + jitter * delta * scale, eps, 1.0 - eps)
    sigma2 = float(max(1.0 / (n_u ** 2), 1.0 / (n_v ** 2)))
    for _ in range(relaxation_steps):
        diff = uv[:, None, :] - uv[None, :, :]
        dist2 = jnp.sum(diff * diff, axis=-1)
        weight = jnp.exp(-dist2 / sigma2) * (1.0 - jnp.eye(uv.shape[0], dtype=uv.dtype))
        repulse = jnp.sum(diff * weight[..., None], axis=1)
        denom = jnp.maximum(jnp.sum(weight, axis=1, keepdims=True), 1.0)
        uv = jnp.clip(uv + 0.08 * repulse / denom, eps, 1.0 - eps)

    pts = evaluate_surface_patch(curves, uv)
    if not include_boundary:
        return uv, pts

    bottom, right, top, left = _normalize_patch_boundaries(curves.bottom, curves.right, curves.top, curves.left)
    dtype = bottom.dtype
    boundary_uv = jnp.concatenate(
        [
            jnp.stack([jnp.linspace(0.0, 1.0, bottom.shape[0], dtype=dtype), jnp.zeros((bottom.shape[0],), dtype=dtype)], axis=-1),
            jnp.stack([jnp.ones((right.shape[0] - 1,), dtype=dtype), jnp.linspace(0.0, 1.0, right.shape[0], dtype=dtype)[1:]], axis=-1),
            jnp.stack([jnp.linspace(1.0, 0.0, top.shape[0], dtype=dtype)[1:], jnp.ones((top.shape[0] - 1,), dtype=dtype)], axis=-1),
            jnp.stack([jnp.zeros((left.shape[0] - 2,), dtype=dtype), jnp.linspace(1.0, 0.0, left.shape[0], dtype=dtype)[1:-1]], axis=-1),
        ],
        axis=0,
    )
    boundary = _surface_boundary_loop(bottom, right, top, left)
    return jnp.concatenate([boundary_uv, uv], axis=0), jnp.concatenate([boundary, pts], axis=0)


def uv_triangle_quality_objective(
    uv: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    quality_floor: float = 0.65,
    area_floor: float = 1.0e-5,
) -> jnp.ndarray:
    """Differentiable quality objective for a UV triangle mesh."""
    tri = uv[elements]
    a = tri[:, 0, :]
    b = tri[:, 1, :]
    c = tri[:, 2, :]
    e01 = b - a
    e12 = c - b
    e20 = a - c
    area2 = e01[:, 0] * (c[:, 1] - a[:, 1]) - e01[:, 1] * (c[:, 0] - a[:, 0])
    edge2 = jnp.sum(e01 * e01, axis=1) + jnp.sum(e12 * e12, axis=1) + jnp.sum(e20 * e20, axis=1)
    quality = 2.0 * jnp.sqrt(3.0) * jnp.abs(area2) / jnp.maximum(edge2, 1.0e-12)
    quality_penalty = jnp.mean(jax.nn.softplus((quality_floor - quality) * 10.0))
    area_penalty = jnp.mean(jax.nn.softplus((area_floor - 0.5 * area2) * 50.0))
    edge_reg = 0.05 * jnp.var(jnp.sqrt(jnp.maximum(edge2 / 3.0, 1.0e-12)))
    return quality_penalty + area_penalty + edge_reg


def cleanup_uv_tri_mesh(
    uv: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    boundary_count: int,
    smoothing_steps: int = 6,
    alpha: float = 0.25,
    flip_passes: int = 3,
    protected_edges: set[tuple[int, int]] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Clean up a UV triangle mesh independently of front generation.

    Boundary nodes `[0:boundary_count]` remain fixed while interior nodes are
    Laplacian-smoothed. Connectivity is then improved via local edge flips.
    """
    uv_work = jnp.asarray(uv)
    elems_work = jnp.asarray(elements, dtype=jnp.int32)
    n_nodes = int(uv_work.shape[0])
    constrained_edges = set(protected_edges or _expected_boundary_edges(boundary_count))

    if smoothing_steps > 0 and boundary_count < n_nodes:
        uv_work = _smooth_uv_interior(
            uv_work,
            elems_work,
            boundary_count=boundary_count,
            smoothing_steps=smoothing_steps,
            alpha=alpha,
        )

    elems_np = _improve_triangle_quality(
        np.asarray(uv_work),
        np.asarray(elems_work),
        passes=flip_passes,
        protected_edges=constrained_edges,
    )
    elems_work = jnp.asarray(elems_np, dtype=jnp.int32)

    if smoothing_steps > 1 and boundary_count < n_nodes:
        uv_work = _smooth_uv_interior(
            uv_work,
            elems_work,
            boundary_count=boundary_count,
            smoothing_steps=max(1, smoothing_steps // 2),
            alpha=0.5 * alpha,
        )

    return uv_work, elems_work


def _front_ring_uv(boundary_uv: np.ndarray, rings: int) -> np.ndarray:
    if rings <= 0:
        return np.zeros((0, 2), dtype=boundary_uv.dtype)
    center = np.array([0.5, 0.5], dtype=boundary_uv.dtype)
    out: list[np.ndarray] = []
    for ring in range(1, rings + 1):
        t = 0.14 * ring / float(rings + 1)
        out.append((1.0 - t) * boundary_uv + t * center[None, :])
    return np.concatenate(out, axis=0)


def surface_front_tri_mesh(
    curves: BoundaryCurves3D,
    n_u: int,
    n_v: int,
    *,
    jitter: float = 0.35,
    relaxation_steps: int = 3,
    seed: int = 0,
    front_rings: int = 2,
    cleanup_steps: int = 6,
    cleanup_alpha: float = 0.25,
    cleanup_flip_passes: int = 3,
) -> tuple[MeshTopology, jnp.ndarray, jnp.ndarray]:
    """Build a triangle surface mesh from an irregular parametric point cloud.

    Boundary edges are inserted explicitly after triangulation so optional
    front-ring samples can be added without relying on external triangulation to
    preserve the outer loop.
    """

    uv, points = surface_parametric_point_cloud(
        curves,
        n_u,
        n_v,
        jitter=jitter,
        relaxation_steps=relaxation_steps,
        seed=seed,
        include_boundary=True,
    )
    boundary_count = int(curves.bottom.shape[0] + curves.right.shape[0] + curves.top.shape[0] + curves.left.shape[0] - 4)
    uv_np = np.asarray(uv, dtype=float)
    protected_edges = _expected_boundary_edges(boundary_count)

    if front_rings > 0:
        uv_np, simplices, protected_edges = _triangulate_surface_with_front_rings(
            uv_np,
            boundary_count=boundary_count,
            front_rings=front_rings,
            n_u=n_u,
            n_v=n_v,
        )
    else:
        simplices = _triangulate_with_boundary_ghosts(uv_np, boundary_count=boundary_count, n_u=n_u, n_v=n_v)
        simplices = _recover_boundary_edges(uv_np, simplices, boundary_count)

    uv_clean, elements = cleanup_uv_tri_mesh(
        jnp.asarray(uv_np, dtype=uv.dtype),
        jnp.asarray(simplices, dtype=jnp.int32),
        boundary_count=boundary_count,
        smoothing_steps=cleanup_steps,
        alpha=cleanup_alpha,
        flip_passes=cleanup_flip_passes,
        protected_edges=protected_edges,
    )
    points = evaluate_surface_patch(curves, uv_clean)
    area2 = (uv_clean[elements[:, 1], 0] - uv_clean[elements[:, 0], 0]) * (uv_clean[elements[:, 2], 1] - uv_clean[elements[:, 0], 1]) - (
        uv_clean[elements[:, 1], 1] - uv_clean[elements[:, 0], 1]
    ) * (uv_clean[elements[:, 2], 0] - uv_clean[elements[:, 0], 0])
    oriented = jnp.where(area2[:, None] < 0.0, elements[:, [0, 2, 1]], elements)
    edges = triangle_edges(oriented)
    n_nodes = int(points.shape[0])
    n_elem = int(oriented.shape[0])
    topo = MeshTopology(
        elements=oriented,
        edges=edges,
        node_ids=jnp.arange(n_nodes, dtype=jnp.int32),
        element_ids=jnp.arange(n_elem, dtype=jnp.int32),
        element_entity_tags=jnp.ones((n_elem,), dtype=jnp.int32),
        n_nodes=n_nodes,
    )
    return topo, points, uv_clean


def _boundary_outward_normals(boundary_uv: np.ndarray) -> np.ndarray:
    tol = 1.0e-8
    normals = np.zeros_like(boundary_uv)
    for i, (u, v) in enumerate(boundary_uv.tolist()):
        if abs(u) <= tol and abs(v) <= tol:
            n = np.array([-1.0, -1.0])
        elif abs(u - 1.0) <= tol and abs(v) <= tol:
            n = np.array([1.0, -1.0])
        elif abs(u - 1.0) <= tol and abs(v - 1.0) <= tol:
            n = np.array([1.0, 1.0])
        elif abs(u) <= tol and abs(v - 1.0) <= tol:
            n = np.array([-1.0, 1.0])
        elif abs(v) <= tol:
            n = np.array([0.0, -1.0])
        elif abs(u - 1.0) <= tol:
            n = np.array([1.0, 0.0])
        elif abs(v - 1.0) <= tol:
            n = np.array([0.0, 1.0])
        else:
            n = np.array([-1.0, 0.0])
        normals[i] = n / np.linalg.norm(n)
    return normals


def _as_edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _expected_boundary_edges(boundary_count: int) -> set[tuple[int, int]]:
    return {_as_edge_key(i, (i + 1) % boundary_count) for i in range(boundary_count)}


def _build_edge_to_triangles(elements: np.ndarray) -> dict[tuple[int, int], list[int]]:
    edge_to_tris: dict[tuple[int, int], list[int]] = {}
    for ti, tri in enumerate(elements.tolist()):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_tris.setdefault(_as_edge_key(int(a), int(b)), []).append(ti)
    return edge_to_tris


def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect_strict(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, tol: float = 1.0e-10) -> bool:
    o1 = _orient2d(a0, a1, b0)
    o2 = _orient2d(a0, a1, b1)
    o3 = _orient2d(b0, b1, a0)
    o4 = _orient2d(b0, b1, a1)
    return o1 * o2 < -tol and o3 * o4 < -tol


def _ordered_convex_quad(uv: np.ndarray, vertices: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    uniq = list(dict.fromkeys(vertices))
    if len(uniq) != 4:
        return None
    pts = uv[np.asarray(uniq, dtype=np.int32)]
    center = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = [uniq[int(i)] for i in np.argsort(ang).tolist()]
    cross_sign = 0.0
    for i in range(4):
        p = uv[order[i]]
        q = uv[order[(i + 1) % 4]]
        r = uv[order[(i + 2) % 4]]
        cross = _orient2d(p, q, r)
        if abs(cross) <= 1.0e-10:
            return None
        if cross_sign == 0.0:
            cross_sign = np.sign(cross)
        elif cross * cross_sign <= 0.0:
            return None
    return tuple(order)


def _smooth_uv_interior(
    uv: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    boundary_count: int,
    smoothing_steps: int,
    alpha: float,
) -> jnp.ndarray:
    n_nodes = int(uv.shape[0])
    boundary_mask = jnp.zeros((n_nodes,), dtype=bool).at[:boundary_count].set(True)
    edges = triangle_edges(elements)
    src = edges[:, 0]
    dst = edges[:, 1]
    eps = jnp.asarray([1.0e-4, 1.0e-4], dtype=uv.dtype)
    upper = jnp.asarray([1.0 - 1.0e-4, 1.0 - 1.0e-4], dtype=uv.dtype)
    uv_work = uv
    current_score = float(uv_triangle_quality_objective(uv_work, elements))

    for _ in range(smoothing_steps):
        neighbor_sum = jnp.zeros_like(uv_work)
        neighbor_sum = neighbor_sum.at[src].add(uv_work[dst])
        neighbor_sum = neighbor_sum.at[dst].add(uv_work[src])
        deg = jnp.zeros((n_nodes,), dtype=uv_work.dtype)
        one = jnp.ones((edges.shape[0],), dtype=uv_work.dtype)
        deg = deg.at[src].add(one)
        deg = deg.at[dst].add(one)

        for backoff in range(6):
            trial_alpha = alpha * (0.5 ** backoff)
            prop = uv_work + trial_alpha * (neighbor_sum / jnp.maximum(deg[:, None], 1.0) - uv_work)
            prop = jnp.clip(prop, eps, upper)
            prop = jnp.where(boundary_mask[:, None], uv_work, prop)
            score = float(uv_triangle_quality_objective(prop, elements))
            if score <= current_score + 1.0e-10:
                uv_work = prop
                current_score = score
                break

    return uv_work


def _triangulate_with_boundary_ghosts(
    uv: np.ndarray,
    *,
    boundary_count: int,
    n_u: int,
    n_v: int,
) -> np.ndarray:
    ghost_delta = 0.08 * min(1.0 / max(n_u, 1), 1.0 / max(n_v, 1))
    boundary_uv = uv[:boundary_count]
    ghost_uv = boundary_uv + ghost_delta * _boundary_outward_normals(boundary_uv)
    simplices = delaunay_triangles_2d(jnp.asarray(np.concatenate([uv, ghost_uv], axis=0)))
    simplices = np.asarray(simplices, dtype=np.int32)
    return simplices[np.all(simplices < uv.shape[0], axis=1)]


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    inside = False
    x, y = float(point[0]), float(point[1])
    n = int(polygon.shape[0])
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            x_cross = float(x0 + (y - y0) * (x1 - x0) / max(y1 - y0, 1.0e-12))
            if x < x_cross:
                inside = not inside
    return inside


def _loop_outward_normals(loop_uv: np.ndarray) -> np.ndarray:
    center = np.mean(loop_uv, axis=0)
    delta = loop_uv - center[None, :]
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    return delta / np.maximum(norm, 1.0e-12)


def _bridge_loop_strip(uv: np.ndarray, outer_ids: np.ndarray, inner_ids: np.ndarray) -> np.ndarray:
    tris: list[tuple[int, int, int]] = []
    count = int(outer_ids.shape[0])
    for i in range(count):
        j = (i + 1) % count
        tris.extend(
            [
                _orient_tri_np(uv, (int(outer_ids[i]), int(outer_ids[j]), int(inner_ids[i]))),
                _orient_tri_np(uv, (int(inner_ids[i]), int(outer_ids[j]), int(inner_ids[j]))),
            ]
        )
    return np.asarray(tris, dtype=np.int32)


def _fan_loop_to_point(uv: np.ndarray, loop_ids: np.ndarray, center_id: int) -> np.ndarray:
    tris: list[tuple[int, int, int]] = []
    count = int(loop_ids.shape[0])
    for i in range(count):
        j = (i + 1) % count
        tris.append(_orient_tri_np(uv, (int(loop_ids[i]), int(loop_ids[j]), center_id)))
    return np.asarray(tris, dtype=np.int32)


def _triangulate_surface_with_front_rings(
    uv: np.ndarray,
    *,
    boundary_count: int,
    front_rings: int,
    n_u: int,
    n_v: int,
) -> tuple[np.ndarray, np.ndarray, set[tuple[int, int]]]:
    base_uv = np.asarray(uv, dtype=float)
    boundary_uv = base_uv[:boundary_count]
    front_uv = _front_ring_uv(boundary_uv, front_rings)
    loop_ids: list[np.ndarray] = [np.arange(boundary_count, dtype=np.int32)]

    uv_parts = [boundary_uv]
    front_start = boundary_count
    for ring in range(front_rings):
        start = front_start + ring * boundary_count
        stop = start + boundary_count
        uv_parts.append(front_uv[ring * boundary_count : (ring + 1) * boundary_count])
        loop_ids.append(np.arange(start, stop, dtype=np.int32))

    center_uv = np.mean(uv_parts[-1], axis=0, keepdims=True)
    center_id = boundary_count + front_rings * boundary_count
    uv_parts.append(center_uv)
    uv_all = np.concatenate(uv_parts, axis=0)

    protected_edges = _expected_boundary_edges(boundary_count)
    for loop in loop_ids[1:]:
        protected_edges.update({_as_edge_key(int(loop[i]), int(loop[(i + 1) % boundary_count])) for i in range(boundary_count)})

    strip_elements = [_bridge_loop_strip(uv_all, outer_ids, inner_ids) for outer_ids, inner_ids in zip(loop_ids[:-1], loop_ids[1:])]

    core_global = _fan_loop_to_point(uv_all, loop_ids[-1], center_id)
    all_elements = np.concatenate([*strip_elements, core_global], axis=0) if strip_elements else core_global
    return uv_all, all_elements, protected_edges


def _insert_constrained_edge(
    uv: np.ndarray,
    elements: np.ndarray,
    edge: tuple[int, int],
    *,
    protected_edges: set[tuple[int, int]],
) -> np.ndarray:
    target = _as_edge_key(*edge)
    elems = np.asarray(elements, dtype=np.int32).copy()
    max_iter = max(8, 4 * elems.shape[0])

    for _ in range(max_iter):
        edge_to_tris = _build_edge_to_triangles(elems)
        if target in edge_to_tris:
            return elems

        crossing_edge: tuple[int, int] | None = None
        crossing_tris: list[int] | None = None
        for shared_edge, tri_ids in edge_to_tris.items():
            if len(tri_ids) != 2 or shared_edge in protected_edges:
                continue
            if target[0] in shared_edge or target[1] in shared_edge:
                continue
            if _segments_intersect_strict(uv[target[0]], uv[target[1]], uv[shared_edge[0]], uv[shared_edge[1]]):
                crossing_edge = shared_edge
                crossing_tris = tri_ids
                break

        if crossing_edge is None or crossing_tris is None:
            break

        i, j = crossing_tris
        t1 = elems[i].tolist()
        t2 = elems[j].tolist()
        a, b = crossing_edge
        c = next(v for v in t1 if v not in crossing_edge)
        d = next(v for v in t2 if v not in crossing_edge)
        quad = _ordered_convex_quad(uv, (a, c, b, d))
        if quad is None:
            break

        cand1 = _orient_tri_np(uv, (c, d, a))
        cand2 = _orient_tri_np(uv, (d, c, b))
        if abs(_orient2d(uv[cand1[0]], uv[cand1[1]], uv[cand1[2]])) <= 1.0e-10:
            break
        if abs(_orient2d(uv[cand2[0]], uv[cand2[1]], uv[cand2[2]])) <= 1.0e-10:
            break

        elems[i] = np.asarray(cand1, dtype=np.int32)
        elems[j] = np.asarray(cand2, dtype=np.int32)

    return elems


def _recover_boundary_edges(uv: np.ndarray, elements: np.ndarray, boundary_count: int) -> np.ndarray:
    elems = np.asarray(elements, dtype=np.int32).copy()
    protected_edges = _expected_boundary_edges(boundary_count)
    for edge in sorted(protected_edges):
        elems = _insert_constrained_edge(uv, elems, edge, protected_edges=protected_edges)
    edge_set = set(_build_edge_to_triangles(elems))
    if missing := protected_edges.difference(edge_set):
        raise ValueError(f"Boundary recovery failed for {len(missing)} boundary edges")
    return elems


def _triangle_min_angle(uv: np.ndarray, tri: tuple[int, int, int]) -> float:
    pts = uv[np.asarray(tri, dtype=np.int32)]
    angs: list[float] = []
    for i in range(3):
        a = pts[(i + 1) % 3] - pts[i]
        b = pts[(i + 2) % 3] - pts[i]
        denom = max(np.linalg.norm(a) * np.linalg.norm(b), 1.0e-12)
        cos_theta = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
        angs.append(float(np.arccos(cos_theta)))
    return min(angs)


def _orient_tri_np(uv: np.ndarray, tri: tuple[int, int, int]) -> tuple[int, int, int]:
    a, b, c = tri
    area2 = (uv[b, 0] - uv[a, 0]) * (uv[c, 1] - uv[a, 1]) - (uv[b, 1] - uv[a, 1]) * (uv[c, 0] - uv[a, 0])
    return tri if area2 > 0.0 else (a, c, b)


def _improve_triangle_quality(
    uv: np.ndarray,
    elements: np.ndarray,
    passes: int = 3,
    protected_edges: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    elems = np.asarray(elements, dtype=np.int32).copy()
    protected = protected_edges or set()
    for _ in range(passes):
        changed = False
        edge_to_tris = _build_edge_to_triangles(elems)
        for edge, tri_ids in edge_to_tris.items():
            if len(tri_ids) != 2:
                continue
            if edge in protected:
                continue
            i, j = tri_ids
            t1 = elems[i].tolist()
            t2 = elems[j].tolist()
            a, b = edge
            c = next(v for v in t1 if v not in edge)
            d = next(v for v in t2 if v not in edge)
            if len({a, b, c, d}) != 4:
                continue
            before = min(_triangle_min_angle(uv, tuple(t1)), _triangle_min_angle(uv, tuple(t2)))
            cand1 = _orient_tri_np(uv, (c, d, a))
            cand2 = _orient_tri_np(uv, (d, c, b))
            after = min(_triangle_min_angle(uv, cand1), _triangle_min_angle(uv, cand2))
            if after > before + 1.0e-6:
                elems[i] = np.asarray(cand1, dtype=np.int32)
                elems[j] = np.asarray(cand2, dtype=np.int32)
                changed = True
        if not changed:
            break
    return elems
