"""Connectivity mutation operators with fixed-capacity buffers (triangle meshes)."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from gmshjax.runtime import jax_float_dtype

class TriMeshBuffer(NamedTuple):
    points: jnp.ndarray  # (max_nodes, 2)
    elements: jnp.ndarray  # (max_elements, 3)
    active_nodes: jnp.ndarray  # (max_nodes,)
    active_elements: jnp.ndarray  # (max_elements,)
    node_count: int
    element_count: int


def make_tri_mesh_buffer(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    max_nodes: int,
    max_elements: int,
) -> TriMeshBuffer:
    """Initialize fixed-capacity triangle mesh buffers."""
    points = jnp.asarray(points, dtype=jax_float_dtype())
    elements = jnp.asarray(elements, dtype=jnp.int32)
    n = int(points.shape[0])
    m = int(elements.shape[0])
    if n > max_nodes or m > max_elements:
        raise ValueError("Initial mesh exceeds fixed-capacity buffer.")

    pbuf = jnp.zeros((max_nodes, points.shape[1]), dtype=points.dtype).at[:n].set(points)
    ebuf = jnp.zeros((max_elements, elements.shape[1]), dtype=elements.dtype).at[:m].set(elements)
    nmask = jnp.zeros((max_nodes,), dtype=bool).at[:n].set(True)
    emask = jnp.zeros((max_elements,), dtype=bool).at[:m].set(True)
    return TriMeshBuffer(pbuf, ebuf, nmask, emask, n, m)


def active_points(buffer: TriMeshBuffer) -> jnp.ndarray:
    return buffer.points[buffer.active_nodes]


def active_elements(buffer: TriMeshBuffer) -> jnp.ndarray:
    return buffer.elements[buffer.active_elements]


def split_triangle(buffer: TriMeshBuffer, element_slot: int) -> tuple[TriMeshBuffer, bool]:
    """Split one active triangle into 3 by inserting centroid node."""
    if element_slot < 0 or element_slot >= buffer.element_count:
        return buffer, False
    if not bool(buffer.active_elements[element_slot]):
        return buffer, False
    if buffer.node_count + 1 > buffer.points.shape[0]:
        return buffer, False
    if buffer.element_count + 2 > buffer.elements.shape[0]:
        return buffer, False

    tri = np.asarray(buffer.elements[element_slot]).tolist()
    a, b, c = [int(v) for v in tri]
    p = buffer.points
    new_node = jnp.mean(p[jnp.array([a, b, c], dtype=jnp.int32)], axis=0)
    nid = int(buffer.node_count)
    e1 = int(buffer.element_count)
    e2 = int(buffer.element_count + 1)

    points = buffer.points.at[nid].set(new_node)
    nodes_mask = buffer.active_nodes.at[nid].set(True)
    elems = (
        buffer.elements.at[element_slot]
        .set(jnp.array([a, b, nid], dtype=buffer.elements.dtype))
        .at[e1]
        .set(jnp.array([b, c, nid], dtype=buffer.elements.dtype))
        .at[e2]
        .set(jnp.array([c, a, nid], dtype=buffer.elements.dtype))
    )
    elems_mask = buffer.active_elements.at[e1].set(True).at[e2].set(True)
    out = TriMeshBuffer(
        points=points,
        elements=elems,
        active_nodes=nodes_mask,
        active_elements=elems_mask,
        node_count=nid + 1,
        element_count=e2 + 1,
    )
    return out, True


def _orient_tri(p: jnp.ndarray, tri: list[int]) -> list[int]:
    a, b, c = tri
    pa = p[a]
    pb = p[b]
    pc = p[c]
    det = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
    if float(det) < 0.0:
        return [a, c, b]
    return tri


def flip_diagonal(buffer: TriMeshBuffer, elem_i: int, elem_j: int) -> tuple[TriMeshBuffer, bool]:
    """Flip shared edge between two active triangles."""
    if elem_i < 0 or elem_j < 0 or elem_i >= buffer.element_count or elem_j >= buffer.element_count:
        return buffer, False
    if not bool(buffer.active_elements[elem_i]) or not bool(buffer.active_elements[elem_j]):
        return buffer, False

    t1 = [int(v) for v in np.asarray(buffer.elements[elem_i]).tolist()]
    t2 = [int(v) for v in np.asarray(buffer.elements[elem_j]).tolist()]
    s = sorted(set(t1).intersection(set(t2)))
    if len(s) != 2:
        return buffer, False
    o1 = [v for v in t1 if v not in s]
    o2 = [v for v in t2 if v not in s]
    if len(o1) != 1 or len(o2) != 1:
        return buffer, False

    u = int(o1[0])
    v = int(o2[0])
    s0, s1 = int(s[0]), int(s[1])
    nt1 = _orient_tri(buffer.points, [u, s0, v])
    nt2 = _orient_tri(buffer.points, [u, v, s1])

    elems = (
        buffer.elements.at[elem_i]
        .set(jnp.array(nt1, dtype=buffer.elements.dtype))
        .at[elem_j]
        .set(jnp.array(nt2, dtype=buffer.elements.dtype))
    )
    return (
        TriMeshBuffer(
            points=buffer.points,
            elements=elems,
            active_nodes=buffer.active_nodes,
            active_elements=buffer.active_elements,
            node_count=buffer.node_count,
            element_count=buffer.element_count,
        ),
        True,
    )
