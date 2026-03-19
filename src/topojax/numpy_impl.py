"""NumPy implementation of core mesh operations."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from topojax.runtime import numpy_float_dtype

_C_TRI = 2.0 / np.sqrt(3.0)


class NumpyMeshTopology(NamedTuple):
    elements: np.ndarray
    edges: np.ndarray
    node_ids: np.ndarray
    element_ids: np.ndarray
    element_entity_tags: np.ndarray
    n_nodes: int


class NumpyDeformationParams(NamedTuple):
    translation: np.ndarray
    scale: np.ndarray
    shear: np.ndarray
    bend: np.ndarray


def unit_square_points(nx: int, ny: int, dtype=None) -> np.ndarray:
    if dtype is None:
        dtype = numpy_float_dtype()
    xs = np.linspace(0.0, 1.0, nx, dtype=dtype)
    ys = np.linspace(0.0, 1.0, ny, dtype=dtype)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel()], axis=-1)


def unit_cube_points(nx: int, ny: int, nz: int, dtype=None) -> np.ndarray:
    if dtype is None:
        dtype = numpy_float_dtype()
    xs = np.linspace(0.0, 1.0, nx, dtype=dtype)
    ys = np.linspace(0.0, 1.0, ny, dtype=dtype)
    zs = np.linspace(0.0, 1.0, nz, dtype=dtype)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)


def structured_triangles(nx: int, ny: int) -> np.ndarray:
    tris: list[list[int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = n00 + 1
            n01 = n00 + nx
            n11 = n01 + 1
            tris.append([n00, n10, n11])
            tris.append([n00, n11, n01])
    return np.asarray(tris, dtype=np.int32)


def structured_quads(nx: int, ny: int) -> np.ndarray:
    quads: list[list[int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = n00 + 1
            n01 = n00 + nx
            n11 = n01 + 1
            quads.append([n00, n10, n11, n01])
    return np.asarray(quads, dtype=np.int32)


def structured_tetrahedra(nx: int, ny: int, nz: int) -> np.ndarray:
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
    return np.asarray(tets, dtype=np.int32)


def triangle_edges(elements: np.ndarray) -> np.ndarray:
    e01 = elements[:, [0, 1]]
    e12 = elements[:, [1, 2]]
    e20 = elements[:, [2, 0]]
    edges = np.concatenate([e01, e12, e20], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges.astype(np.int32)


def quad_edges(elements: np.ndarray) -> np.ndarray:
    e01 = elements[:, [0, 1]]
    e12 = elements[:, [1, 2]]
    e23 = elements[:, [2, 3]]
    e30 = elements[:, [3, 0]]
    edges = np.concatenate([e01, e12, e23, e30], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges.astype(np.int32)


def tet_edges(elements: np.ndarray) -> np.ndarray:
    e01 = elements[:, [0, 1]]
    e02 = elements[:, [0, 2]]
    e03 = elements[:, [0, 3]]
    e12 = elements[:, [1, 2]]
    e13 = elements[:, [1, 3]]
    e23 = elements[:, [2, 3]]
    edges = np.concatenate([e01, e02, e03, e12, e13, e23], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges.astype(np.int32)


def unit_square_tri_mesh(nx: int, ny: int, dtype=None) -> tuple[NumpyMeshTopology, np.ndarray]:
    points = unit_square_points(nx, ny, dtype=dtype)
    elements = structured_triangles(nx, ny)
    edges = triangle_edges(elements)
    n_nodes = nx * ny
    n_elem = elements.shape[0]
    topo = NumpyMeshTopology(
        elements=elements,
        edges=edges,
        node_ids=np.arange(n_nodes, dtype=np.int32),
        element_ids=np.arange(n_elem, dtype=np.int32),
        element_entity_tags=np.ones((n_elem,), dtype=np.int32),
        n_nodes=n_nodes,
    )
    return topo, points


def unit_square_quad_mesh(nx: int, ny: int, dtype=None) -> tuple[NumpyMeshTopology, np.ndarray]:
    points = unit_square_points(nx, ny, dtype=dtype)
    elements = structured_quads(nx, ny)
    edges = quad_edges(elements)
    n_nodes = nx * ny
    n_elem = elements.shape[0]
    topo = NumpyMeshTopology(
        elements=elements,
        edges=edges,
        node_ids=np.arange(n_nodes, dtype=np.int32),
        element_ids=np.arange(n_elem, dtype=np.int32),
        element_entity_tags=np.ones((n_elem,), dtype=np.int32),
        n_nodes=n_nodes,
    )
    return topo, points


def unit_cube_tet_mesh(nx: int, ny: int, nz: int, dtype=None) -> tuple[NumpyMeshTopology, np.ndarray]:
    points = unit_cube_points(nx, ny, nz, dtype=dtype)
    elements = structured_tetrahedra(nx, ny, nz)
    edges = tet_edges(elements)
    n_nodes = nx * ny * nz
    n_elem = elements.shape[0]
    topo = NumpyMeshTopology(
        elements=elements,
        edges=edges,
        node_ids=np.arange(n_nodes, dtype=np.int32),
        element_ids=np.arange(n_elem, dtype=np.int32),
        element_entity_tags=np.ones((n_elem,), dtype=np.int32),
        n_nodes=n_nodes,
    )
    return topo, points


def apply_deformation(points: np.ndarray, params: NumpyDeformationParams) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    sx = params.scale[0] * x
    sy = params.scale[1] * y
    shx = params.shear[0] * y
    shy = params.shear[1] * x
    bx = params.bend[0] * np.sin(2.0 * np.pi * y)
    by = params.bend[1] * np.sin(2.0 * np.pi * x)
    out_x = sx + shx + bx + params.translation[0]
    out_y = sy + shy + by + params.translation[1]
    return np.stack([out_x, out_y], axis=-1)


def triangle_signed_areas(points: np.ndarray, elements: np.ndarray) -> np.ndarray:
    det, _, _, _, _ = _triangle_jacobian_terms(points, elements)
    return 0.5 * det


def _triangle_jacobian_terms(points: np.ndarray, elements: np.ndarray):
    tri = points[elements]
    a = tri[:, 0, :]
    b = tri[:, 1, :]
    c = tri[:, 2, :]
    e1 = b - a
    e2 = c - a
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    fro2 = np.sum(e1 * e1, axis=1) + np.sum(e2 * e2, axis=1)
    l01 = np.linalg.norm(e1, axis=1)
    l02 = np.linalg.norm(e2, axis=1)
    l12 = np.linalg.norm(c - b, axis=1)
    return det, fro2, l01, l02, l12


def triangle_icn(points: np.ndarray, elements: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    det, fro2, _, _, _ = _triangle_jacobian_terms(points, elements)
    return 2.0 * det / np.maximum(fro2, eps)


def triangle_ige(points: np.ndarray, elements: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    det, _, l01, l02, l12 = _triangle_jacobian_terms(points, elements)
    inv_terms = (1.0 / np.maximum(l01 * l02, eps)) + (1.0 / np.maximum(l01 * l12, eps)) + (
        1.0 / np.maximum(l02 * l12, eps)
    )
    return _C_TRI * det * (inv_terms / 3.0)


def quad_icn(points: np.ndarray, elements: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    q = points[elements]
    dxi = 0.25 * (-q[:, 0, :] + q[:, 1, :] + q[:, 2, :] - q[:, 3, :])
    deta = 0.25 * (-q[:, 0, :] - q[:, 1, :] + q[:, 2, :] + q[:, 3, :])
    det = dxi[:, 0] * deta[:, 1] - dxi[:, 1] * deta[:, 0]
    fro2 = np.sum(dxi * dxi, axis=1) + np.sum(deta * deta, axis=1)
    return 2.0 * det / np.maximum(fro2, eps)


def quad_ige(points: np.ndarray, elements: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    q = points[elements]
    dxi = 0.25 * (-q[:, 0, :] + q[:, 1, :] + q[:, 2, :] - q[:, 3, :])
    deta = 0.25 * (-q[:, 0, :] - q[:, 1, :] + q[:, 2, :] + q[:, 3, :])
    det = dxi[:, 0] * deta[:, 1] - dxi[:, 1] * deta[:, 0]
    l1 = np.linalg.norm(dxi, axis=1)
    l2 = np.linalg.norm(deta, axis=1)
    return det / np.maximum(l1 * l2, eps)


def tet_icn(points: np.ndarray, elements: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    t = points[elements]
    a = t[:, 0, :]
    b = t[:, 1, :]
    c = t[:, 2, :]
    d = t[:, 3, :]
    e1 = b - a
    e2 = c - a
    e3 = d - a
    det = (
        e1[:, 0] * (e2[:, 1] * e3[:, 2] - e2[:, 2] * e3[:, 1])
        - e1[:, 1] * (e2[:, 0] * e3[:, 2] - e2[:, 2] * e3[:, 0])
        + e1[:, 2] * (e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0])
    )
    fro2 = np.sum(e1 * e1, axis=1) + np.sum(e2 * e2, axis=1) + np.sum(e3 * e3, axis=1)
    signed_pow = np.sign(det) * np.power(np.maximum(np.abs(det), eps), 2.0 / 3.0)
    return 3.0 * signed_pow / np.maximum(fro2, eps)


def tet_ige(points: np.ndarray, elements: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    t = points[elements]
    a = t[:, 0, :]
    b = t[:, 1, :]
    c = t[:, 2, :]
    d = t[:, 3, :]
    e1 = b - a
    e2 = c - a
    e3 = d - a
    det = (
        e1[:, 0] * (e2[:, 1] * e3[:, 2] - e2[:, 2] * e3[:, 1])
        - e1[:, 1] * (e2[:, 0] * e3[:, 2] - e2[:, 2] * e3[:, 0])
        + e1[:, 2] * (e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0])
    )
    l01 = np.linalg.norm(b - a, axis=1)
    l02 = np.linalg.norm(c - a, axis=1)
    l03 = np.linalg.norm(d - a, axis=1)
    l12 = np.linalg.norm(c - b, axis=1)
    l13 = np.linalg.norm(d - b, axis=1)
    l23 = np.linalg.norm(d - c, axis=1)
    terms = (
        1.0 / np.maximum(l01 * l23 * l02, eps)
        + 1.0 / np.maximum(l01 * l23 * l03, eps)
        + 1.0 / np.maximum(l01 * l23 * l12, eps)
        + 1.0 / np.maximum(l01 * l23 * l13, eps)
        + 1.0 / np.maximum(l02 * l13 * l01, eps)
        + 1.0 / np.maximum(l02 * l13 * l03, eps)
        + 1.0 / np.maximum(l02 * l13 * l12, eps)
        + 1.0 / np.maximum(l02 * l13 * l23, eps)
        + 1.0 / np.maximum(l03 * l12 * l01, eps)
        + 1.0 / np.maximum(l03 * l12 * l02, eps)
        + 1.0 / np.maximum(l03 * l12 * l13, eps)
        + 1.0 / np.maximum(l03 * l12 * l23, eps)
    )
    return np.sqrt(2.0) * det * (terms / 12.0)


def edge_lengths(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    p0 = points[edges[:, 0]]
    p1 = points[edges[:, 1]]
    return np.linalg.norm(p1 - p0, axis=1)


def mesh_quality_energy(points: np.ndarray, topology: NumpyMeshTopology) -> np.ndarray:
    areas = triangle_signed_areas(points, topology.elements)
    lens = edge_lengths(points, topology.edges)
    icn = triangle_icn(points, topology.elements)
    ige = triangle_ige(points, topology.elements)
    area_term = np.var(np.abs(areas))
    edge_term = np.var(lens)
    fold_penalty = np.mean(np.log1p(np.exp(-areas * 100.0)))
    quality_barrier = np.mean(np.log1p(np.exp((0.2 - icn) * 20.0)) + np.log1p(np.exp((0.2 - ige) * 20.0)))
    return area_term + edge_term + 1.0e-3 * fold_penalty + 1.0e-3 * quality_barrier
