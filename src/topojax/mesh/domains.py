"""Discrete domain meshing helpers for fixed-topology workflows."""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy as np

from topojax.io.exports import GmshElementBlock
from topojax.mesh.generators import unit_cube_points
from topojax.mesh.topology import MeshTopology, mesh_topology_from_points_and_elements, structured_tetrahedra, tet_edges, triangle_edges
from topojax.mesh.triangulation import delaunay_triangles_2d
from topojax.runtime import jax_float_dtype


class DomainMeshMetadata(NamedTuple):
    boundary_element_blocks: tuple[GmshElementBlock, ...]
    physical_names: dict[tuple[int, int], str]


def _as_closed_loop(loop: jnp.ndarray) -> np.ndarray:
    arr = np.asarray(loop, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        raise ValueError("Each polygon loop must have shape (n, 2) with n >= 3")
    if np.linalg.norm(arr[0] - arr[-1]) <= 1.0e-12:
        arr = arr[:-1]
    if arr.shape[0] < 3:
        raise ValueError("Each polygon loop must contain at least 3 distinct vertices")
    return arr


def _loop_signed_area(loop: np.ndarray) -> float:
    shifted = np.roll(loop, -1, axis=0)
    return 0.5 * float(np.sum(loop[:, 0] * shifted[:, 1] - shifted[:, 0] * loop[:, 1]))


def _ensure_orientation(loop: np.ndarray, *, ccw: bool) -> np.ndarray:
    area = _loop_signed_area(loop)
    if ccw and area < 0.0:
        return loop[::-1].copy()
    return loop[::-1].copy() if (not ccw and area > 0.0) else loop


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


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1.0e-16:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def _sample_loop(loop: np.ndarray, edge_size: float) -> np.ndarray:
    pts: list[np.ndarray] = []
    n = int(loop.shape[0])
    for i in range(n):
        a = loop[i]
        b = loop[(i + 1) % n]
        length = float(np.linalg.norm(b - a))
        subdivisions = max(1, int(np.ceil(length / max(edge_size, 1.0e-8))))
        for step in range(subdivisions):
            t = step / subdivisions
            pts.append((1.0 - t) * a + t * b)
    out = np.asarray(pts, dtype=float)
    if out.shape[0] == 0:
        raise ValueError("Failed to sample polygon loop")
    return out


def _as_edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _expected_loop_edges(loop_ranges: list[tuple[int, int]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for start, count in loop_ranges:
        for i in range(count):
            a = start + i
            b = start + ((i + 1) % count)
            edges.add(_as_edge_key(a, b))
    return edges


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


def _orient_tri(points: np.ndarray, tri: tuple[int, int, int]) -> tuple[int, int, int]:
    a, b, c = tri
    return tri if _orient2d(points[a], points[b], points[c]) > 0.0 else (a, c, b)


def _ordered_convex_quad(points: np.ndarray, vertices: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    uniq = list(dict.fromkeys(vertices))
    if len(uniq) != 4:
        return None
    pts = points[np.asarray(uniq, dtype=np.int32)]
    center = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = [uniq[int(i)] for i in np.argsort(ang).tolist()]
    cross_sign = 0.0
    for i in range(4):
        p = points[order[i]]
        q = points[order[(i + 1) % 4]]
        r = points[order[(i + 2) % 4]]
        cross = _orient2d(p, q, r)
        if abs(cross) <= 1.0e-10:
            return None
        if cross_sign == 0.0:
            cross_sign = np.sign(cross)
        elif cross * cross_sign <= 0.0:
            return None
    return tuple(order)


def _insert_constrained_edge(
    points: np.ndarray,
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
            if _segments_intersect_strict(points[target[0]], points[target[1]], points[shared_edge[0]], points[shared_edge[1]]):
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
        if _ordered_convex_quad(points, (a, c, b, d)) is None:
            break

        cand1 = _orient_tri(points, (c, d, a))
        cand2 = _orient_tri(points, (d, c, b))
        if abs(_orient2d(points[cand1[0]], points[cand1[1]], points[cand1[2]])) <= 1.0e-10:
            break
        if abs(_orient2d(points[cand2[0]], points[cand2[1]], points[cand2[2]])) <= 1.0e-10:
            break
        elems[i] = np.asarray(cand1, dtype=np.int32)
        elems[j] = np.asarray(cand2, dtype=np.int32)

    return elems


def _polygon_domain_contains(point: np.ndarray, outer: np.ndarray, holes: list[np.ndarray]) -> bool:
    if not _point_in_polygon(point, outer):
        return False
    return not any(_point_in_polygon(point, hole) for hole in holes)


def _compact_triangle_mesh(points: np.ndarray, elements: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if elements.shape[0] == 0:
        raise ValueError("No elements remain after polygon meshing")
    used = np.unique(elements.reshape(-1))
    remap = -np.ones((points.shape[0],), dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)
    return points[used], remap[elements], remap


def _build_polygon_domain_tri_mesh(
    outer_boundary: jnp.ndarray,
    *,
    holes: list[jnp.ndarray] | None,
    target_edge_size: float | None,
    interior_relaxation: float,
) -> tuple[MeshTopology, jnp.ndarray, list[np.ndarray], list[tuple[int, int]], np.ndarray]:
    outer = _ensure_orientation(_as_closed_loop(outer_boundary), ccw=True)
    hole_loops = [_ensure_orientation(_as_closed_loop(loop), ccw=False) for loop in (holes or [])]

    all_loop_points = [outer, *hole_loops]
    bbox_lo = np.min(np.concatenate(all_loop_points, axis=0), axis=0)
    bbox_hi = np.max(np.concatenate(all_loop_points, axis=0), axis=0)
    bbox_span = float(np.max(bbox_hi - bbox_lo))
    if target_edge_size is None:
        outer_lengths = np.linalg.norm(np.roll(outer, -1, axis=0) - outer, axis=1)
        target_edge_size = max(float(np.mean(outer_lengths)) * 0.5, bbox_span / 40.0, 1.0e-3)

    sampled_loops = [_sample_loop(outer, target_edge_size)]
    sampled_loops.extend(_sample_loop(loop, target_edge_size) for loop in hole_loops)

    loop_ranges: list[tuple[int, int]] = []
    point_parts: list[np.ndarray] = []
    cursor = 0
    for loop in sampled_loops:
        loop_ranges.append((cursor, int(loop.shape[0])))
        point_parts.append(loop)
        cursor += int(loop.shape[0])

    xs = np.arange(bbox_lo[0] + 0.5 * target_edge_size, bbox_hi[0], target_edge_size)
    ys = np.arange(bbox_lo[1] + 0.5 * target_edge_size, bbox_hi[1], target_edge_size)
    interior_points: list[np.ndarray] = []
    clearance = max(0.15, interior_relaxation) * target_edge_size
    for x in xs.tolist():
        for y in ys.tolist():
            point = np.asarray([x, y], dtype=float)
            if not _polygon_domain_contains(point, sampled_loops[0], sampled_loops[1:]):
                continue
            near_boundary = False
            for loop in sampled_loops:
                for i in range(loop.shape[0]):
                    a = loop[i]
                    b = loop[(i + 1) % loop.shape[0]]
                    if _point_segment_distance(point, a, b) < clearance:
                        near_boundary = True
                        break
                if near_boundary:
                    break
            if not near_boundary:
                interior_points.append(point)

    if interior_points:
        point_parts.append(np.asarray(interior_points, dtype=float))
    points_np = np.concatenate(point_parts, axis=0)
    raw_elements = np.asarray(delaunay_triangles_2d(jnp.asarray(points_np, dtype=jax_float_dtype())), dtype=np.int32)
    if raw_elements.shape[0] == 0:
        raise ValueError("Polygon triangulation failed to produce any triangles")

    centroids = np.mean(points_np[raw_elements], axis=1)
    keep = np.asarray([_polygon_domain_contains(c, sampled_loops[0], sampled_loops[1:]) for c in centroids], dtype=bool)
    elements_np = raw_elements[keep]

    protected_edges = _expected_loop_edges(loop_ranges)
    for edge in sorted(protected_edges):
        elements_np = _insert_constrained_edge(points_np, elements_np, edge, protected_edges=protected_edges)

    centroids = np.mean(points_np[elements_np], axis=1)
    keep = np.asarray([_polygon_domain_contains(c, sampled_loops[0], sampled_loops[1:]) for c in centroids], dtype=bool)
    elements_np = elements_np[keep]
    edge_set = set(_build_edge_to_triangles(elements_np))
    if missing := protected_edges.difference(edge_set):
        raise ValueError(f"Polygon boundary recovery failed for {len(missing)} constrained edges")

    points_np, elements_np, remap = _compact_triangle_mesh(points_np, elements_np)
    area2 = (points_np[elements_np[:, 1], 0] - points_np[elements_np[:, 0], 0]) * (points_np[elements_np[:, 2], 1] - points_np[elements_np[:, 0], 1]) - (
        points_np[elements_np[:, 1], 1] - points_np[elements_np[:, 0], 1]
    ) * (points_np[elements_np[:, 2], 0] - points_np[elements_np[:, 0], 0])
    elements_np = np.where(area2[:, None] < 0.0, elements_np[:, [0, 2, 1]], elements_np)

    points = jnp.asarray(points_np, dtype=jax_float_dtype())
    elements = jnp.asarray(elements_np, dtype=jnp.int32)
    edges = triangle_edges(elements)
    n_nodes = int(points.shape[0])
    n_elem = int(elements.shape[0])
    topology = MeshTopology(
        elements=elements,
        edges=edges,
        node_ids=jnp.arange(n_nodes, dtype=jnp.int32),
        element_ids=jnp.arange(n_elem, dtype=jnp.int32),
        element_entity_tags=jnp.ones((n_elem,), dtype=jnp.int32),
        n_nodes=n_nodes,
    )
    return topology, points, sampled_loops, loop_ranges, remap


def _line_block(elements: np.ndarray, physical_tag: int, geometrical_tag: int, name: str) -> tuple[GmshElementBlock, tuple[tuple[int, int], str]]:
    block = GmshElementBlock(
        elements=jnp.asarray(elements, dtype=jnp.int32),
        element_kind="line",
        physical_tags=jnp.full((elements.shape[0],), physical_tag, dtype=jnp.int32),
        geometrical_tags=jnp.full((elements.shape[0],), geometrical_tag, dtype=jnp.int32),
    )
    return block, ((1, physical_tag), name)


def _triangle_block(elements: np.ndarray, physical_tag: int, geometrical_tag: int, name: str) -> tuple[GmshElementBlock, tuple[tuple[int, int], str]]:
    block = GmshElementBlock(
        elements=jnp.asarray(elements, dtype=jnp.int32),
        element_kind="triangle",
        physical_tags=jnp.full((elements.shape[0],), physical_tag, dtype=jnp.int32),
        geometrical_tags=jnp.full((elements.shape[0],), geometrical_tag, dtype=jnp.int32),
    )
    return block, ((2, physical_tag), name)


def _polygon_boundary_metadata(loop_ranges: list[tuple[int, int]], remap: np.ndarray) -> DomainMeshMetadata:
    blocks: list[GmshElementBlock] = []
    physical_names: dict[tuple[int, int], str] = {}
    for loop_index, (start, count) in enumerate(loop_ranges):
        edges = []
        for i in range(count):
            a = int(remap[start + i])
            b = int(remap[start + ((i + 1) % count)])
            if a >= 0 and b >= 0:
                edges.append([a, b])
        if not edges:
            continue
        physical_tag = 100 + loop_index
        geometrical_tag = 10 + loop_index
        name = "outer_boundary" if loop_index == 0 else f"hole_{loop_index}_boundary"
        block, (key, label) = _line_block(np.asarray(edges, dtype=np.int32), physical_tag, geometrical_tag, name)
        blocks.append(block)
        physical_names[key] = label
    return DomainMeshMetadata(boundary_element_blocks=tuple(blocks), physical_names=physical_names)


def _triangle_to_quad_points_and_elements(points: np.ndarray, elements: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    edge_keys = sorted(_build_edge_to_triangles(elements))
    edge_to_midpoint: dict[tuple[int, int], int] = {}
    point_parts: list[np.ndarray] = [np.asarray(points, dtype=float)]

    midpoint_coords = []
    for edge in edge_keys:
        midpoint_index = points.shape[0] + len(midpoint_coords)
        edge_to_midpoint[edge] = midpoint_index
        midpoint_coords.append(0.5 * (points[edge[0]] + points[edge[1]]))
    if midpoint_coords:
        point_parts.append(np.asarray(midpoint_coords, dtype=float))

    centroid_offset = sum(part.shape[0] for part in point_parts)
    centroids = np.mean(points[elements], axis=1)
    point_parts.append(np.asarray(centroids, dtype=float))
    quad_points = np.concatenate(point_parts, axis=0)

    quads: list[tuple[int, int, int, int]] = []
    for tri_index, tri in enumerate(elements.tolist()):
        a, b, c = _orient_tri(points, (int(tri[0]), int(tri[1]), int(tri[2])))
        m_ab = edge_to_midpoint[_as_edge_key(a, b)]
        m_bc = edge_to_midpoint[_as_edge_key(b, c)]
        m_ca = edge_to_midpoint[_as_edge_key(c, a)]
        centroid = centroid_offset + tri_index
        for quad_vertices in ((a, m_ab, centroid, m_ca), (b, m_bc, centroid, m_ab), (c, m_ca, centroid, m_bc)):
            ordered = _ordered_convex_quad(quad_points, quad_vertices)
            quads.append(ordered if ordered is not None else quad_vertices)

    return quad_points, np.asarray(quads, dtype=np.int32), edge_to_midpoint


def _polygon_quad_boundary_metadata(loop_ranges: list[tuple[int, int]], remap: np.ndarray, edge_to_midpoint: dict[tuple[int, int], int]) -> DomainMeshMetadata:
    blocks: list[GmshElementBlock] = []
    physical_names: dict[tuple[int, int], str] = {}
    for loop_index, (start, count) in enumerate(loop_ranges):
        edges = []
        for i in range(count):
            a = int(remap[start + i])
            b = int(remap[start + ((i + 1) % count)])
            if a < 0 or b < 0:
                continue
            midpoint = edge_to_midpoint[_as_edge_key(a, b)]
            edges.extend(([a, midpoint], [midpoint, b]))
        if not edges:
            continue
        physical_tag = 100 + loop_index
        geometrical_tag = 10 + loop_index
        name = "outer_boundary" if loop_index == 0 else f"hole_{loop_index}_boundary"
        block, (key, label) = _line_block(np.asarray(edges, dtype=np.int32), physical_tag, geometrical_tag, name)
        blocks.append(block)
        physical_names[key] = label
    return DomainMeshMetadata(boundary_element_blocks=tuple(blocks), physical_names=physical_names)


def _tet_boundary_faces(elements: np.ndarray) -> np.ndarray:
    face_map: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for tet in np.asarray(elements, dtype=np.int32).tolist():
        a, b, c, d = tet
        faces = [(a, c, b), (a, b, d), (a, d, c), (b, c, d)]
        for face in faces:
            face_map.setdefault(tuple(sorted(face)), []).append(face)
    boundary = [entries[0] for entries in face_map.values() if len(entries) == 1]
    return np.asarray(boundary, dtype=np.int32) if boundary else np.zeros((0, 3), dtype=np.int32)


def _structured_box_boundary_faces(nx: int, ny: int, nz: int) -> dict[str, np.ndarray]:
    def idx(i: int, j: int, k: int) -> int:
        return (k * ny + j) * nx + i

    def face_tris(corners: tuple[int, int, int, int]) -> list[list[int]]:
        a, b, c, d = corners
        return [[a, b, c], [a, c, d]]

    faces = {name: [] for name in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}

    for j in range(ny - 1):
        for k in range(nz - 1):
            faces["xmin"].extend(face_tris((idx(0, j, k), idx(0, j + 1, k), idx(0, j + 1, k + 1), idx(0, j, k + 1))))
            faces["xmax"].extend(face_tris((idx(nx - 1, j, k), idx(nx - 1, j, k + 1), idx(nx - 1, j + 1, k + 1), idx(nx - 1, j + 1, k))))

    for i in range(nx - 1):
        for k in range(nz - 1):
            faces["ymin"].extend(face_tris((idx(i, 0, k), idx(i, 0, k + 1), idx(i + 1, 0, k + 1), idx(i + 1, 0, k))))
            faces["ymax"].extend(face_tris((idx(i, ny - 1, k), idx(i + 1, ny - 1, k), idx(i + 1, ny - 1, k + 1), idx(i, ny - 1, k + 1))))

    for i in range(nx - 1):
        for j in range(ny - 1):
            faces["zmin"].extend(face_tris((idx(i, j, 0), idx(i + 1, j, 0), idx(i + 1, j + 1, 0), idx(i, j + 1, 0))))
            faces["zmax"].extend(face_tris((idx(i, j, nz - 1), idx(i, j + 1, nz - 1), idx(i + 1, j + 1, nz - 1), idx(i + 1, j, nz - 1))))

    return {name: np.asarray(entries, dtype=np.int32) for name, entries in faces.items()}


def polygon_domain_tri_mesh(
    outer_boundary: jnp.ndarray,
    *,
    holes: list[jnp.ndarray] | None = None,
    target_edge_size: float | None = None,
    interior_relaxation: float = 0.35,
) -> tuple[MeshTopology, jnp.ndarray]:
    """Create a fixed-topology triangle mesh for an arbitrary 2D polygon domain.

    The geometry stage is discrete and uses point sampling plus constrained-edge
    recovery. The resulting topology is suitable for Mode 1 fixed-topology AD.
    """
    topology, points, _, _, _ = _build_polygon_domain_tri_mesh(
        outer_boundary,
        holes=holes,
        target_edge_size=target_edge_size,
        interior_relaxation=interior_relaxation,
    )
    return topology, points


def polygon_domain_tri_mesh_tagged(
    outer_boundary: jnp.ndarray,
    *,
    holes: list[jnp.ndarray] | None = None,
    target_edge_size: float | None = None,
    interior_relaxation: float = 0.35,
) -> tuple[MeshTopology, jnp.ndarray, DomainMeshMetadata]:
    topology, points, _, loop_ranges, remap = _build_polygon_domain_tri_mesh(
        outer_boundary,
        holes=holes,
        target_edge_size=target_edge_size,
        interior_relaxation=interior_relaxation,
    )
    return topology, points, _polygon_boundary_metadata(loop_ranges, remap)


def polygon_domain_quad_mesh(
    outer_boundary: jnp.ndarray,
    *,
    holes: list[jnp.ndarray] | None = None,
    target_edge_size: float | None = None,
    interior_relaxation: float = 0.35,
) -> tuple[MeshTopology, jnp.ndarray]:
    """Create a conforming quad mesh for a polygonal domain via tri-to-quad subdivision."""
    tri_topology, tri_points, _, _, _ = _build_polygon_domain_tri_mesh(
        outer_boundary,
        holes=holes,
        target_edge_size=target_edge_size,
        interior_relaxation=interior_relaxation,
    )
    quad_points_np, quad_elements_np, _ = _triangle_to_quad_points_and_elements(np.asarray(tri_points), np.asarray(tri_topology.elements))
    quad_points = jnp.asarray(quad_points_np, dtype=jax_float_dtype())
    quad_elements = jnp.asarray(quad_elements_np, dtype=jnp.int32)
    return mesh_topology_from_points_and_elements(quad_points, quad_elements), quad_points


def polygon_domain_quad_mesh_tagged(
    outer_boundary: jnp.ndarray,
    *,
    holes: list[jnp.ndarray] | None = None,
    target_edge_size: float | None = None,
    interior_relaxation: float = 0.35,
) -> tuple[MeshTopology, jnp.ndarray, DomainMeshMetadata]:
    tri_topology, tri_points, _, loop_ranges, remap = _build_polygon_domain_tri_mesh(
        outer_boundary,
        holes=holes,
        target_edge_size=target_edge_size,
        interior_relaxation=interior_relaxation,
    )
    quad_points_np, quad_elements_np, edge_to_midpoint = _triangle_to_quad_points_and_elements(np.asarray(tri_points), np.asarray(tri_topology.elements))
    quad_points = jnp.asarray(quad_points_np, dtype=jax_float_dtype())
    quad_elements = jnp.asarray(quad_elements_np, dtype=jnp.int32)
    metadata = _polygon_quad_boundary_metadata(loop_ranges, remap, edge_to_midpoint)
    return mesh_topology_from_points_and_elements(quad_points, quad_elements), quad_points, metadata


def box_volume_tet_mesh(
    bbox_min: jnp.ndarray,
    bbox_max: jnp.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[MeshTopology, jnp.ndarray]:
    """Create a tetrahedral mesh for an axis-aligned box from an exact structured fill."""
    if min(nx, ny, nz) < 2:
        raise ValueError("nx, ny, nz must each be at least 2")

    dtype = jax_float_dtype()
    lo = jnp.asarray(bbox_min, dtype=dtype)
    hi = jnp.asarray(bbox_max, dtype=dtype)
    unit_pts = unit_cube_points(nx, ny, nz, dtype=dtype)
    points = lo[None, :] + unit_pts * (hi - lo)[None, :]
    elements = structured_tetrahedra(nx, ny, nz)
    return mesh_topology_from_points_and_elements(points, elements), points


def box_volume_tet_mesh_tagged(
    bbox_min: jnp.ndarray,
    bbox_max: jnp.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[MeshTopology, jnp.ndarray, DomainMeshMetadata]:
    topology, points = box_volume_tet_mesh(bbox_min, bbox_max, nx, ny, nz)
    side_specs = [
        (500, 60, "xmin"),
        (501, 61, "xmax"),
        (502, 62, "ymin"),
        (503, 63, "ymax"),
        (504, 64, "zmin"),
        (505, 65, "zmax"),
    ]
    blocks: list[GmshElementBlock] = []
    physical_names: dict[tuple[int, int], str] = {}
    boundary_faces = _structured_box_boundary_faces(nx, ny, nz)
    for physical_tag, geometrical_tag, name in side_specs:
        faces = boundary_faces[name]
        if faces.shape[0] == 0:
            continue
        block, (key, label) = _triangle_block(faces, physical_tag, geometrical_tag, name)
        blocks.append(block)
        physical_names[key] = label
    return topology, points, DomainMeshMetadata(boundary_element_blocks=tuple(blocks), physical_names=physical_names)


def implicit_volume_tet_mesh(
    level_set_fn: Callable[[jnp.ndarray], jnp.ndarray],
    bbox_min: jnp.ndarray,
    bbox_max: jnp.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[MeshTopology, jnp.ndarray]:
    """Create a tetrahedral mesh for a 3D implicit domain from a structured background grid."""
    if min(nx, ny, nz) < 2:
        raise ValueError("nx, ny, nz must each be at least 2")

    dtype = jax_float_dtype()
    lo = jnp.asarray(bbox_min, dtype=dtype)
    hi = jnp.asarray(bbox_max, dtype=dtype)
    unit_pts = unit_cube_points(nx, ny, nz, dtype=dtype)
    points = lo[None, :] + unit_pts * (hi - lo)[None, :]
    elements = structured_tetrahedra(nx, ny, nz)

    centroids = jnp.mean(points[elements], axis=1)
    phi = jnp.asarray(level_set_fn(centroids), dtype=points.dtype)
    keep = np.asarray(phi <= 0.0, dtype=bool)
    elements_np = np.asarray(elements, dtype=np.int32)[keep]
    if elements_np.shape[0] == 0:
        raise ValueError("Implicit volume mesher produced no tetrahedra inside the domain")

    used = np.unique(elements_np.reshape(-1))
    remap = -np.ones((points.shape[0],), dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)
    compact_points = np.asarray(points)[used]
    compact_elements = remap[elements_np]

    points_out = jnp.asarray(compact_points, dtype=points.dtype)
    elements_out = jnp.asarray(compact_elements, dtype=jnp.int32)
    edges = tet_edges(elements_out)
    n_nodes = int(points_out.shape[0])
    n_elem = int(elements_out.shape[0])
    topology = MeshTopology(
        elements=elements_out,
        edges=edges,
        node_ids=jnp.arange(n_nodes, dtype=jnp.int32),
        element_ids=jnp.arange(n_elem, dtype=jnp.int32),
        element_entity_tags=jnp.ones((n_elem,), dtype=jnp.int32),
        n_nodes=n_nodes,
    )
    return topology, points_out


def implicit_volume_tet_mesh_tagged(
    level_set_fn: Callable[[jnp.ndarray], jnp.ndarray],
    bbox_min: jnp.ndarray,
    bbox_max: jnp.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[MeshTopology, jnp.ndarray, DomainMeshMetadata]:
    topology, points = implicit_volume_tet_mesh(level_set_fn, bbox_min, bbox_max, nx, ny, nz)
    faces = _tet_boundary_faces(np.asarray(topology.elements))
    block, (key, name) = _triangle_block(faces, physical_tag=300, geometrical_tag=30, name="implicit_boundary")
    return topology, points, DomainMeshMetadata(boundary_element_blocks=(block,), physical_names={key: name})


def sphere_volume_tet_mesh(
    center: jnp.ndarray,
    radius: float,
    nx: int,
    ny: int,
    nz: int,
    *,
    padding: float = 0.0,
) -> tuple[MeshTopology, jnp.ndarray]:
    """Create a tetrahedral mesh for a sphere via the implicit-volume initializer."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if padding < 0.0:
        raise ValueError("padding must be non-negative")

    dtype = jax_float_dtype()
    center_arr = jnp.asarray(center, dtype=dtype)
    span = jnp.full((3,), float(radius + padding), dtype=dtype)
    bbox_min = center_arr - span
    bbox_max = center_arr + span

    def sphere_level_set(points: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(points - center_arr[None, :], axis=1) - float(radius)

    return implicit_volume_tet_mesh(sphere_level_set, bbox_min, bbox_max, nx, ny, nz)


def sphere_volume_tet_mesh_tagged(
    center: jnp.ndarray,
    radius: float,
    nx: int,
    ny: int,
    nz: int,
    *,
    padding: float = 0.0,
) -> tuple[MeshTopology, jnp.ndarray, DomainMeshMetadata]:
    """Create a tetrahedral sphere mesh and tag its extracted boundary triangles."""
    topology, points = sphere_volume_tet_mesh(center, radius, nx, ny, nz, padding=padding)
    faces = _tet_boundary_faces(np.asarray(topology.elements))
    block, (key, name) = _triangle_block(faces, physical_tag=320, geometrical_tag=32, name="sphere_boundary")
    return topology, points, DomainMeshMetadata(boundary_element_blocks=(block,), physical_names={key: name})


def _orient_tet(points: np.ndarray, tet: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    a, b, c, d = tet
    pa, pb, pc, pd = points[[a, b, c, d]]
    det = np.dot(pb - pa, np.cross(pc - pa, pd - pa))
    return tet if det > 0.0 else (a, c, b, d)


def extruded_polygon_tet_mesh(
    outer_boundary: jnp.ndarray,
    *,
    holes: list[jnp.ndarray] | None = None,
    target_edge_size: float | None = None,
    height: float = 1.0,
    layers: int = 4,
) -> tuple[MeshTopology, jnp.ndarray, DomainMeshMetadata]:
    if layers < 1:
        raise ValueError("layers must be at least 1")
    base_topology, base_points, base_meta = polygon_domain_tri_mesh_tagged(
        outer_boundary,
        holes=holes,
        target_edge_size=target_edge_size,
    )
    z_levels = np.linspace(0.0, height, layers + 1)
    base_np = np.asarray(base_points)
    n_base = int(base_np.shape[0])

    point_layers = [np.concatenate([base_np, np.full((n_base, 1), z, dtype=base_np.dtype)], axis=1) for z in z_levels.tolist()]
    points_np = np.concatenate(point_layers, axis=0)

    tets: list[tuple[int, int, int, int]] = []
    for layer in range(layers):
        offset0 = layer * n_base
        offset1 = (layer + 1) * n_base
        for tri in np.asarray(base_topology.elements).tolist():
            a0, b0, c0 = [offset0 + int(v) for v in tri]
            a1, b1, c1 = [offset1 + int(v) for v in tri]
            tets.extend(
                [
                    _orient_tet(points_np, (a0, b0, c0, c1)),
                    _orient_tet(points_np, (a0, b0, b1, c1)),
                    _orient_tet(points_np, (a0, a1, b1, c1)),
                ]
            )

    elements = jnp.asarray(np.asarray(tets, dtype=np.int32), dtype=jnp.int32)
    points = jnp.asarray(points_np, dtype=jax_float_dtype())
    edges = tet_edges(elements)
    n_nodes = int(points.shape[0])
    n_elem = int(elements.shape[0])
    topology = MeshTopology(
        elements=elements,
        edges=edges,
        node_ids=jnp.arange(n_nodes, dtype=jnp.int32),
        element_ids=jnp.arange(n_elem, dtype=jnp.int32),
        element_entity_tags=jnp.ones((n_elem,), dtype=jnp.int32),
        n_nodes=n_nodes,
    )

    boundary_blocks: list[GmshElementBlock] = []
    physical_names: dict[tuple[int, int], str] = {}
    bottom_faces = np.asarray(base_topology.elements)[:, [0, 2, 1]]
    top_faces = np.asarray(base_topology.elements) + layers * n_base
    for faces, physical_tag, geometrical_tag, name in [
        (bottom_faces, 400, 40, "bottom"),
        (top_faces, 401, 41, "top"),
    ]:
        block, (key, label) = _triangle_block(np.asarray(faces, dtype=np.int32), physical_tag, geometrical_tag, name)
        boundary_blocks.append(block)
        physical_names[key] = label

    for block_index, block in enumerate(base_meta.boundary_element_blocks):
        edge_array = np.asarray(block.elements, dtype=np.int32)
        wall_faces: list[list[int]] = []
        for layer in range(layers):
            offset0 = layer * n_base
            offset1 = (layer + 1) * n_base
            for a, b in edge_array.tolist():
                a0, b0 = offset0 + int(a), offset0 + int(b)
                a1, b1 = offset1 + int(a), offset1 + int(b)
                wall_faces.extend(([a0, b0, a1], [a1, b0, b1]))
        phys_tag = 410 + block_index
        geom_tag = 50 + block_index
        name = list(base_meta.physical_names.values())[block_index].replace("_boundary", "_wall")
        wall_block, (key, label) = _triangle_block(np.asarray(wall_faces, dtype=np.int32), phys_tag, geom_tag, name)
        boundary_blocks.append(wall_block)
        physical_names[key] = label

    return topology, points, DomainMeshMetadata(boundary_element_blocks=tuple(boundary_blocks), physical_names=physical_names)
