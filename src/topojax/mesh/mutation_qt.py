"""Fixed-capacity mutation buffers/operators for quad and tet meshes."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from topojax.runtime import jax_float_dtype


class QuadMeshBuffer(NamedTuple):
    points: jnp.ndarray  # (max_nodes, 2)
    elements: jnp.ndarray  # (max_elements, 4)
    active_nodes: jnp.ndarray
    active_elements: jnp.ndarray
    node_count: int
    element_count: int


class TetMeshBuffer(NamedTuple):
    points: jnp.ndarray  # (max_nodes, 3)
    elements: jnp.ndarray  # (max_elements, 4)
    active_nodes: jnp.ndarray
    active_elements: jnp.ndarray
    node_count: int
    element_count: int


def _trim_tail(mask: jnp.ndarray, count: int) -> int:
    count_out = count
    while count_out > 0 and not bool(mask[count_out - 1]):
        count_out -= 1
    return count_out


def _order_quad(points: jnp.ndarray, verts: list[int]) -> list[int]:
    pts = np.asarray(points[jnp.asarray(verts, dtype=jnp.int32)])
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(ang)
    ordered = [verts[int(i)] for i in order.tolist()]
    pts2 = np.asarray(points[jnp.asarray(ordered, dtype=jnp.int32)])
    x = pts2[:, 0]
    y = pts2[:, 1]
    area2 = float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    if area2 < 0.0:
        ordered = ordered[::-1]
    return ordered


def _orient_tet(points: jnp.ndarray, tet: list[int]) -> list[int]:
    a, b, c, d = tet
    pa = np.asarray(points[a])
    pb = np.asarray(points[b])
    pc = np.asarray(points[c])
    pd = np.asarray(points[d])
    det = float(np.linalg.det(np.stack([pb - pa, pc - pa, pd - pa], axis=1)))
    return [a, c, b, d] if det < 0.0 else tet


def make_quad_mesh_buffer(points: jnp.ndarray, elements: jnp.ndarray, max_nodes: int, max_elements: int) -> QuadMeshBuffer:
    points = jnp.asarray(points, dtype=jax_float_dtype())
    elements = jnp.asarray(elements, dtype=jnp.int32)
    n = int(points.shape[0])
    m = int(elements.shape[0])
    if n > max_nodes or m > max_elements:
        raise ValueError("Initial quad mesh exceeds capacity.")
    pbuf = jnp.zeros((max_nodes, 2), dtype=points.dtype).at[:n].set(points)
    ebuf = jnp.zeros((max_elements, 4), dtype=elements.dtype).at[:m].set(elements)
    nmask = jnp.zeros((max_nodes,), dtype=bool).at[:n].set(True)
    emask = jnp.zeros((max_elements,), dtype=bool).at[:m].set(True)
    return QuadMeshBuffer(pbuf, ebuf, nmask, emask, n, m)


def make_tet_mesh_buffer(points: jnp.ndarray, elements: jnp.ndarray, max_nodes: int, max_elements: int) -> TetMeshBuffer:
    points = jnp.asarray(points, dtype=jax_float_dtype())
    elements = jnp.asarray(elements, dtype=jnp.int32)
    n = int(points.shape[0])
    m = int(elements.shape[0])
    if n > max_nodes or m > max_elements:
        raise ValueError("Initial tet mesh exceeds capacity.")
    pbuf = jnp.zeros((max_nodes, 3), dtype=points.dtype).at[:n].set(points)
    ebuf = jnp.zeros((max_elements, 4), dtype=elements.dtype).at[:m].set(elements)
    nmask = jnp.zeros((max_nodes,), dtype=bool).at[:n].set(True)
    emask = jnp.zeros((max_elements,), dtype=bool).at[:m].set(True)
    return TetMeshBuffer(pbuf, ebuf, nmask, emask, n, m)


def active_quad_elements(buf: QuadMeshBuffer) -> jnp.ndarray:
    return buf.elements[buf.active_elements]


def active_tet_elements(buf: TetMeshBuffer) -> jnp.ndarray:
    return buf.elements[buf.active_elements]


def active_quad_points(buf: QuadMeshBuffer) -> jnp.ndarray:
    return buf.points[buf.active_nodes]


def active_tet_points(buf: TetMeshBuffer) -> jnp.ndarray:
    return buf.points[buf.active_nodes]


def split_quad(buf: QuadMeshBuffer, element_slot: int) -> tuple[QuadMeshBuffer, bool]:
    """Split one quad into 4 quads by adding edge midpoints and center."""
    if element_slot < 0 or element_slot >= buf.element_count or not bool(buf.active_elements[element_slot]):
        return buf, False
    if buf.node_count + 5 > buf.points.shape[0] or buf.element_count + 3 > buf.elements.shape[0]:
        return buf, False

    a, b, c, d = [int(v) for v in np.asarray(buf.elements[element_slot]).tolist()]
    p = buf.points
    m_ab = 0.5 * (p[a] + p[b])
    m_bc = 0.5 * (p[b] + p[c])
    m_cd = 0.5 * (p[c] + p[d])
    m_da = 0.5 * (p[d] + p[a])
    cen = 0.25 * (p[a] + p[b] + p[c] + p[d])

    n0 = int(buf.node_count)
    n1, n2, n3, n4 = n0, n0 + 1, n0 + 2, n0 + 3
    nc = n0 + 4
    e1 = int(buf.element_count)
    e2 = e1 + 1
    e3 = e1 + 2

    points = (
        buf.points.at[n1].set(m_ab)
        .at[n2].set(m_bc)
        .at[n3].set(m_cd)
        .at[n4].set(m_da)
        .at[nc].set(cen)
    )
    nmask = buf.active_nodes.at[n1].set(True).at[n2].set(True).at[n3].set(True).at[n4].set(True).at[nc].set(True)
    elems = (
        buf.elements.at[element_slot].set(jnp.array([a, n1, nc, n4], dtype=buf.elements.dtype))
        .at[e1].set(jnp.array([n1, b, n2, nc], dtype=buf.elements.dtype))
        .at[e2].set(jnp.array([nc, n2, c, n3], dtype=buf.elements.dtype))
        .at[e3].set(jnp.array([n4, nc, n3, d], dtype=buf.elements.dtype))
    )
    emask = buf.active_elements.at[e1].set(True).at[e2].set(True).at[e3].set(True)
    return QuadMeshBuffer(points, elems, nmask, emask, nc + 1, e3 + 1), True


def split_tet(buf: TetMeshBuffer, element_slot: int) -> tuple[TetMeshBuffer, bool]:
    """Split one tetrahedron into 4 tetrahedra by inserting centroid."""
    if element_slot < 0 or element_slot >= buf.element_count or not bool(buf.active_elements[element_slot]):
        return buf, False
    if buf.node_count + 1 > buf.points.shape[0] or buf.element_count + 3 > buf.elements.shape[0]:
        return buf, False

    a, b, c, d = [int(v) for v in np.asarray(buf.elements[element_slot]).tolist()]
    cen = 0.25 * (buf.points[a] + buf.points[b] + buf.points[c] + buf.points[d])
    nid = int(buf.node_count)
    e1 = int(buf.element_count)
    e2 = e1 + 1
    e3 = e1 + 2

    points = buf.points.at[nid].set(cen)
    nmask = buf.active_nodes.at[nid].set(True)
    elems = (
        buf.elements.at[element_slot].set(jnp.array([a, b, c, nid], dtype=buf.elements.dtype))
        .at[e1].set(jnp.array([a, b, nid, d], dtype=buf.elements.dtype))
        .at[e2].set(jnp.array([a, nid, c, d], dtype=buf.elements.dtype))
        .at[e3].set(jnp.array([nid, b, c, d], dtype=buf.elements.dtype))
    )
    emask = buf.active_elements.at[e1].set(True).at[e2].set(True).at[e3].set(True)
    return TetMeshBuffer(points, elems, nmask, emask, nid + 1, e3 + 1), True


def collapse_quad(buf: QuadMeshBuffer, node_id: int) -> tuple[QuadMeshBuffer, bool]:
    """Collapse a 4-quad fan around one center node back to a single quad."""
    if node_id < 0 or node_id >= buf.node_count or not bool(buf.active_nodes[node_id]):
        return buf, False

    slots: list[int] = []
    boundary: list[int] = []
    for slot in range(int(buf.element_count)):
        if not bool(buf.active_elements[slot]):
            continue
        quad = [int(v) for v in np.asarray(buf.elements[slot]).tolist()]
        if node_id in quad:
            slots.append(slot)
            boundary.extend(v for v in quad if v != node_id)

    counts: dict[int, int] = {}
    for vertex in boundary:
        counts[vertex] = counts.get(vertex, 0) + 1
    corners = sorted(vertex for vertex, freq in counts.items() if freq == 1)
    midpoints = sorted(vertex for vertex, freq in counts.items() if freq == 2)
    if len(slots) != 4 or len(corners) != 4:
        return buf, False

    keep = min(slots)
    quad = _order_quad(buf.points, corners)
    elems = buf.elements.at[keep].set(jnp.asarray(quad, dtype=buf.elements.dtype))
    emask = buf.active_elements
    for slot in slots:
        if slot != keep:
            emask = emask.at[slot].set(False)
    nmask = buf.active_nodes.at[node_id].set(False)
    for midpoint in midpoints:
        nmask = nmask.at[midpoint].set(False)
    return (
        QuadMeshBuffer(
            points=buf.points,
            elements=elems,
            active_nodes=nmask,
            active_elements=emask,
            node_count=_trim_tail(nmask, buf.node_count),
            element_count=_trim_tail(emask, buf.element_count),
        ),
        True,
    )


def collapse_tet(buf: TetMeshBuffer, node_id: int) -> tuple[TetMeshBuffer, bool]:
    """Collapse a 4-tet fan around one center node back to a single tetrahedron."""
    if node_id < 0 or node_id >= buf.node_count or not bool(buf.active_nodes[node_id]):
        return buf, False

    slots: list[int] = []
    boundary: list[int] = []
    for slot in range(int(buf.element_count)):
        if not bool(buf.active_elements[slot]):
            continue
        tet = [int(v) for v in np.asarray(buf.elements[slot]).tolist()]
        if node_id in tet:
            slots.append(slot)
            boundary.extend(v for v in tet if v != node_id)

    unique = sorted(set(boundary))
    if len(slots) != 4 or len(unique) != 4:
        return buf, False

    keep = min(slots)
    tet = _orient_tet(buf.points, unique)
    elems = buf.elements.at[keep].set(jnp.asarray(tet, dtype=buf.elements.dtype))
    emask = buf.active_elements
    for slot in slots:
        if slot != keep:
            emask = emask.at[slot].set(False)
    nmask = buf.active_nodes.at[node_id].set(False)
    return (
        TetMeshBuffer(
            points=buf.points,
            elements=elems,
            active_nodes=nmask,
            active_elements=emask,
            node_count=_trim_tail(nmask, buf.node_count),
            element_count=_trim_tail(emask, buf.element_count),
        ),
        True,
    )
