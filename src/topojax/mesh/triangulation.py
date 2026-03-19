"""Pure in-repo 2D triangulation helpers."""

from __future__ import annotations

import jax.numpy as jnp


def _orient2d(pa: jnp.ndarray, pb: jnp.ndarray, pc: jnp.ndarray) -> float:
    return float((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]))


def _triangle_area2(points: jnp.ndarray, tri: tuple[int, int, int]) -> float:
    return _orient2d(points[tri[0]], points[tri[1]], points[tri[2]])


def _normalize_triangle(points: jnp.ndarray, tri: tuple[int, int, int]) -> tuple[int, int, int] | None:
    area2 = _triangle_area2(points, tri)
    if abs(area2) <= 1.0e-12:
        return None
    return tri if area2 > 0.0 else (tri[0], tri[2], tri[1])


def _circumcircle_contains(points: jnp.ndarray, tri: tuple[int, int, int], point_id: int, tol: float = 1.0e-12) -> bool:
    ax = float(points[tri[0], 0] - points[point_id, 0])
    ay = float(points[tri[0], 1] - points[point_id, 1])
    bx = float(points[tri[1], 0] - points[point_id, 0])
    by = float(points[tri[1], 1] - points[point_id, 1])
    cx = float(points[tri[2], 0] - points[point_id, 0])
    cy = float(points[tri[2], 1] - points[point_id, 1])

    det = (ax * ax + ay * ay) * (bx * cy - by * cx)
    det -= (bx * bx + by * by) * (ax * cy - ay * cx)
    det += (cx * cx + cy * cy) * (ax * by - ay * bx)

    orient = _triangle_area2(points, tri)
    if orient > 0.0:
        return det > tol
    return det < -tol


def _super_triangle(points: jnp.ndarray) -> jnp.ndarray:
    lo = jnp.min(points, axis=0)
    hi = jnp.max(points, axis=0)
    center = 0.5 * (lo + hi)
    span = float(jnp.max(hi - lo))
    scale = max(span, 1.0)
    return jnp.asarray(
        [
            [center[0] - 20.0 * scale, center[1] - 4.0 * scale],
            [center[0], center[1] + 20.0 * scale],
            [center[0] + 20.0 * scale, center[1] - 4.0 * scale],
        ],
        dtype=points.dtype,
    )


def delaunay_triangles_2d(points: jnp.ndarray) -> jnp.ndarray:
    """Triangulate a 2D point set using Bowyer-Watson with JAX arrays.

    The implementation uses Python control flow for the discrete topology edits,
    but all geometric data is stored in JAX arrays. It is intended as a runtime
    triangulator for meshing, not as a differentiable primitive.
    """
    pts = jnp.asarray(points)
    n_points = int(pts.shape[0])
    if n_points < 3:
        return jnp.zeros((0, 3), dtype=jnp.int32)

    lo = jnp.min(pts, axis=0)
    hi = jnp.max(pts, axis=0)
    eps = 1.0e-9 * max(float(jnp.max(hi - lo)), 1.0)
    perturb = eps * jnp.stack(
        [jnp.arange(n_points, dtype=pts.dtype), jnp.arange(n_points - 1, -1, -1, dtype=pts.dtype)],
        axis=1,
    )
    work_pts = pts + perturb
    ext_pts = jnp.concatenate([work_pts, _super_triangle(work_pts)], axis=0)
    super_ids = (n_points, n_points + 1, n_points + 2)
    triangles: list[tuple[int, int, int]] = [super_ids]

    for point_id in range(n_points):
        bad_tris = [tri for tri in triangles if _circumcircle_contains(ext_pts, tri, point_id)]
        edge_use: dict[tuple[int, int], int] = {}
        for tri in bad_tris:
            for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = tuple(sorted(edge))
                edge_use[key] = edge_use.get(key, 0) + 1

        triangles = [tri for tri in triangles if tri not in bad_tris]
        polygon = [edge for edge, count in edge_use.items() if count == 1]
        for edge in polygon:
            cand = _normalize_triangle(ext_pts, (edge[0], edge[1], point_id))
            if cand is not None:
                triangles.append(cand)

    final_tris = [tri for tri in triangles if all(v < n_points for v in tri)]
    final_tris = [tri for tri in final_tris if abs(_triangle_area2(ext_pts, tri)) > 1.0e-12]
    if not final_tris:
        return jnp.zeros((0, 3), dtype=jnp.int32)
    return jnp.asarray(final_tris, dtype=jnp.int32)