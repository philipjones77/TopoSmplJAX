"""Static mesh topology builders and containers."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .generators import unit_cube_points, unit_square_points


class MeshTopology(NamedTuple):
    """Connectivity and immutable ids that stay fixed while node coordinates move."""

    elements: jnp.ndarray  # (n_elem, 3) triangle indices
    edges: jnp.ndarray  # (n_edge, 2) undirected edge indices
    node_ids: jnp.ndarray  # (n_nodes,)
    element_ids: jnp.ndarray  # (n_elem,)
    element_entity_tags: jnp.ndarray  # (n_elem,) geometric entity tag
    n_nodes: int


def structured_triangles(nx: int, ny: int) -> jnp.ndarray:
    """Build 2-triangle-per-cell connectivity on an nx-by-ny node grid."""
    tris: list[list[int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = n00 + 1
            n01 = n00 + nx
            n11 = n01 + 1
            tris.append([n00, n10, n11])
            tris.append([n00, n11, n01])
    return jnp.asarray(tris, dtype=jnp.int32)


def structured_quads(nx: int, ny: int) -> jnp.ndarray:
    """Build 1-quad-per-cell connectivity on an nx-by-ny node grid."""
    quads: list[list[int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = n00 + 1
            n01 = n00 + nx
            n11 = n01 + 1
            quads.append([n00, n10, n11, n01])
    return jnp.asarray(quads, dtype=jnp.int32)


def structured_tetrahedra(nx: int, ny: int, nz: int) -> jnp.ndarray:
    """Split each structured cube into 5 tetrahedra with consistent orientation."""
    tets: list[list[int]] = []

    def idx(i: int, j: int, k: int) -> int:
        return (k * ny + j) * nx + i

    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                v000 = idx(i, j, k)
                v100 = idx(i + 1, j, k)
                v010 = idx(i, j + 1, k)
                v110 = idx(i + 1, j + 1, k)
                v001 = idx(i, j, k + 1)
                v101 = idx(i + 1, j, k + 1)
                v011 = idx(i, j + 1, k + 1)
                v111 = idx(i + 1, j + 1, k + 1)
                tets.extend(
                    [
                        [v000, v100, v110, v111],
                        [v000, v110, v010, v111],
                        [v000, v010, v011, v111],
                        [v000, v011, v001, v111],
                        [v000, v001, v101, v111],
                    ]
                )
    return jnp.asarray(tets, dtype=jnp.int32)


def triangle_edges(elements: jnp.ndarray) -> jnp.ndarray:
    """Return unique undirected edges extracted from triangles."""
    e01 = elements[:, [0, 1]]
    e12 = elements[:, [1, 2]]
    e20 = elements[:, [2, 0]]
    edges = jnp.concatenate([e01, e12, e20], axis=0)
    edges = jnp.sort(edges, axis=1)
    edges = jnp.unique(edges, axis=0)
    return edges.astype(jnp.int32)


def quad_edges(elements: jnp.ndarray) -> jnp.ndarray:
    """Return unique undirected edges extracted from quads."""
    e01 = elements[:, [0, 1]]
    e12 = elements[:, [1, 2]]
    e23 = elements[:, [2, 3]]
    e30 = elements[:, [3, 0]]
    edges = jnp.concatenate([e01, e12, e23, e30], axis=0)
    edges = jnp.sort(edges, axis=1)
    edges = jnp.unique(edges, axis=0)
    return edges.astype(jnp.int32)


def tet_edges(elements: jnp.ndarray) -> jnp.ndarray:
    """Return unique undirected edges extracted from tetrahedra."""
    e01 = elements[:, [0, 1]]
    e02 = elements[:, [0, 2]]
    e03 = elements[:, [0, 3]]
    e12 = elements[:, [1, 2]]
    e13 = elements[:, [1, 3]]
    e23 = elements[:, [2, 3]]
    edges = jnp.concatenate([e01, e02, e03, e12, e13, e23], axis=0)
    edges = jnp.sort(edges, axis=1)
    edges = jnp.unique(edges, axis=0)
    return edges.astype(jnp.int32)


def unit_square_tri_mesh(nx: int, ny: int) -> tuple[MeshTopology, jnp.ndarray]:
    """Create fixed topology + initial node coordinates on the unit square."""
    points = unit_square_points(nx, ny)
    elements = structured_triangles(nx, ny)
    edges = triangle_edges(elements)
    n_nodes = nx * ny
    n_elem = elements.shape[0]
    topo = MeshTopology(
        elements=elements,
        edges=edges,
        node_ids=jnp.arange(n_nodes, dtype=jnp.int32),
        element_ids=jnp.arange(n_elem, dtype=jnp.int32),
        element_entity_tags=jnp.ones((n_elem,), dtype=jnp.int32),
        n_nodes=n_nodes,
    )
    return topo, points


def unit_square_quad_mesh(nx: int, ny: int) -> tuple[MeshTopology, jnp.ndarray]:
    """Create fixed quad topology + initial node coordinates on the unit square."""
    points = unit_square_points(nx, ny)
    elements = structured_quads(nx, ny)
    edges = quad_edges(elements)
    n_nodes = nx * ny
    n_elem = elements.shape[0]
    topo = MeshTopology(
        elements=elements,
        edges=edges,
        node_ids=jnp.arange(n_nodes, dtype=jnp.int32),
        element_ids=jnp.arange(n_elem, dtype=jnp.int32),
        element_entity_tags=jnp.ones((n_elem,), dtype=jnp.int32),
        n_nodes=n_nodes,
    )
    return topo, points


def unit_cube_tet_mesh(nx: int, ny: int, nz: int) -> tuple[MeshTopology, jnp.ndarray]:
    """Create fixed tet topology + initial node coordinates on the unit cube."""
    points = unit_cube_points(nx, ny, nz)
    elements = structured_tetrahedra(nx, ny, nz)
    edges = tet_edges(elements)
    n_nodes = nx * ny * nz
    n_elem = elements.shape[0]
    topo = MeshTopology(
        elements=elements,
        edges=edges,
        node_ids=jnp.arange(n_nodes, dtype=jnp.int32),
        element_ids=jnp.arange(n_elem, dtype=jnp.int32),
        element_entity_tags=jnp.ones((n_elem,), dtype=jnp.int32),
        n_nodes=n_nodes,
    )
    return topo, points
