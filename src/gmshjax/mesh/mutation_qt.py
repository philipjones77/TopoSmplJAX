"""Fixed-capacity mutation buffers/operators for quad and tet meshes."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from gmshjax.runtime import jax_float_dtype

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
