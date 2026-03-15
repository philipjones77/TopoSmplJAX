"""Differentiable operators over a fixed-topology mesh."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .topology import MeshTopology


_C_TRI = 2.0 / jnp.sqrt(3.0)
_C_TET = jnp.sqrt(2.0)


def line_element_lengths(points: jnp.ndarray, elements: jnp.ndarray) -> jnp.ndarray:
    seg = points[elements]
    return jnp.linalg.norm(seg[:, 1, :] - seg[:, 0, :], axis=1)


def _triangle_jacobian_terms(points: jnp.ndarray, elements: jnp.ndarray):
    tri = points[elements]  # (n_elem, 3, 2)
    a = tri[:, 0, :]
    b = tri[:, 1, :]
    c = tri[:, 2, :]
    e1 = b - a
    e2 = c - a
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    fro2 = jnp.sum(e1 * e1, axis=1) + jnp.sum(e2 * e2, axis=1)
    l01 = jnp.linalg.norm(e1, axis=1)
    l02 = jnp.linalg.norm(e2, axis=1)
    l12 = jnp.linalg.norm(c - b, axis=1)
    return det, fro2, l01, l02, l12


def triangle_signed_areas(points: jnp.ndarray, elements: jnp.ndarray) -> jnp.ndarray:
    """Signed triangle area for each element."""
    det, _, _, _, _ = _triangle_jacobian_terms(points, elements)
    return 0.5 * det


def triangle_icn(points: jnp.ndarray, elements: jnp.ndarray, eps: float = 1.0e-12) -> jnp.ndarray:
    """Inverse condition number proxy for triangles (Gmsh-style Jacobian metric)."""
    det, fro2, _, _, _ = _triangle_jacobian_terms(points, elements)
    return 2.0 * det / jnp.maximum(fro2, eps)


def triangle_ige(points: jnp.ndarray, elements: jnp.ndarray, eps: float = 1.0e-12) -> jnp.ndarray:
    """Inverse gradient error proxy for triangles (Gmsh-style Jacobian metric)."""
    det, _, l01, l02, l12 = _triangle_jacobian_terms(points, elements)
    inv_terms = (1.0 / jnp.maximum(l01 * l02, eps)) + (1.0 / jnp.maximum(l01 * l12, eps)) + (
        1.0 / jnp.maximum(l02 * l12, eps)
    )
    return _C_TRI * det * (inv_terms / 3.0)


def quad_icn(points: jnp.ndarray, elements: jnp.ndarray, eps: float = 1.0e-12) -> jnp.ndarray:
    """Inverse condition number proxy at quad center using bilinear Jacobian."""
    q = points[elements]  # (n_elem, 4, 2)
    dxi = 0.25 * (-q[:, 0, :] + q[:, 1, :] + q[:, 2, :] - q[:, 3, :])
    deta = 0.25 * (-q[:, 0, :] - q[:, 1, :] + q[:, 2, :] + q[:, 3, :])
    det = dxi[:, 0] * deta[:, 1] - dxi[:, 1] * deta[:, 0]
    fro2 = jnp.sum(dxi * dxi, axis=1) + jnp.sum(deta * deta, axis=1)
    return 2.0 * det / jnp.maximum(fro2, eps)


def quad_ige(points: jnp.ndarray, elements: jnp.ndarray, eps: float = 1.0e-12) -> jnp.ndarray:
    """Inverse gradient error proxy at quad center."""
    q = points[elements]
    dxi = 0.25 * (-q[:, 0, :] + q[:, 1, :] + q[:, 2, :] - q[:, 3, :])
    deta = 0.25 * (-q[:, 0, :] - q[:, 1, :] + q[:, 2, :] + q[:, 3, :])
    det = dxi[:, 0] * deta[:, 1] - dxi[:, 1] * deta[:, 0]
    l1 = jnp.linalg.norm(dxi, axis=1)
    l2 = jnp.linalg.norm(deta, axis=1)
    return det / jnp.maximum(l1 * l2, eps)


def tet_icn(points: jnp.ndarray, elements: jnp.ndarray, eps: float = 1.0e-12) -> jnp.ndarray:
    """Inverse condition number proxy for tetrahedra."""
    t = points[elements]  # (n_elem, 4, 3)
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
    fro2 = jnp.sum(e1 * e1, axis=1) + jnp.sum(e2 * e2, axis=1) + jnp.sum(e3 * e3, axis=1)
    signed_pow = jnp.sign(det) * jnp.power(jnp.maximum(jnp.abs(det), eps), 2.0 / 3.0)
    return 3.0 * signed_pow / jnp.maximum(fro2, eps)


def tet_ige(points: jnp.ndarray, elements: jnp.ndarray, eps: float = 1.0e-12) -> jnp.ndarray:
    """Inverse gradient error proxy for tetrahedra (edge-length averaged)."""
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
    l01 = jnp.linalg.norm(b - a, axis=1)
    l02 = jnp.linalg.norm(c - a, axis=1)
    l03 = jnp.linalg.norm(d - a, axis=1)
    l12 = jnp.linalg.norm(c - b, axis=1)
    l13 = jnp.linalg.norm(d - b, axis=1)
    l23 = jnp.linalg.norm(d - c, axis=1)
    terms = (
        1.0 / jnp.maximum(l01 * l23 * l02, eps)
        + 1.0 / jnp.maximum(l01 * l23 * l03, eps)
        + 1.0 / jnp.maximum(l01 * l23 * l12, eps)
        + 1.0 / jnp.maximum(l01 * l23 * l13, eps)
        + 1.0 / jnp.maximum(l02 * l13 * l01, eps)
        + 1.0 / jnp.maximum(l02 * l13 * l03, eps)
        + 1.0 / jnp.maximum(l02 * l13 * l12, eps)
        + 1.0 / jnp.maximum(l02 * l13 * l23, eps)
        + 1.0 / jnp.maximum(l03 * l12 * l01, eps)
        + 1.0 / jnp.maximum(l03 * l12 * l02, eps)
        + 1.0 / jnp.maximum(l03 * l12 * l13, eps)
        + 1.0 / jnp.maximum(l03 * l12 * l23, eps)
    )
    return _C_TET * det * (terms / 12.0)


def edge_lengths(points: jnp.ndarray, edges: jnp.ndarray) -> jnp.ndarray:
    """Euclidean edge lengths."""
    p0 = points[edges[:, 0]]
    p1 = points[edges[:, 1]]
    return jnp.linalg.norm(p1 - p0, axis=1)


def graph_laplacian_step(points: jnp.ndarray, edges: jnp.ndarray, alpha: float = 0.1) -> jnp.ndarray:
    """One explicit Laplacian smoothing step."""
    n = points.shape[0]
    src = edges[:, 0]
    dst = edges[:, 1]

    neighbor_sum = jnp.zeros_like(points)
    neighbor_sum = neighbor_sum.at[src].add(points[dst])
    neighbor_sum = neighbor_sum.at[dst].add(points[src])

    deg = jnp.zeros((n,), dtype=points.dtype)
    one = jnp.ones((edges.shape[0],), dtype=points.dtype)
    deg = deg.at[src].add(one)
    deg = deg.at[dst].add(one)

    deg = jnp.maximum(deg, 1.0)
    mean_nbr = neighbor_sum / deg[:, None]
    return points + alpha * (mean_nbr - points)


def mesh_quality_energy(points: jnp.ndarray, topology: MeshTopology) -> jnp.ndarray:
    """Simple smooth energy: area variance + edge-length variance + fold penalty."""
    areas = triangle_signed_areas(points, topology.elements)
    lens = edge_lengths(points, topology.edges)
    icn = triangle_icn(points, topology.elements)
    ige = triangle_ige(points, topology.elements)

    area_term = jnp.var(jnp.abs(areas))
    edge_term = jnp.var(lens)
    # Differentiable barrier against element inversion.
    fold_penalty = jnp.mean(jax.nn.softplus(-areas * 100.0))
    quality_barrier = jnp.mean(jax.nn.softplus((0.2 - icn) * 20.0) + jax.nn.softplus((0.2 - ige) * 20.0))
    return area_term + edge_term + 1.0e-3 * fold_penalty + 1.0e-3 * quality_barrier


def quad_mesh_quality_energy(points: jnp.ndarray, topology: MeshTopology) -> jnp.ndarray:
    """Smooth quality energy for fixed-topology quad meshes."""
    q = points[topology.elements]
    a = q[:, 0, :]
    b = q[:, 1, :]
    c = q[:, 2, :]
    d = q[:, 3, :]
    area1 = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    area2 = 0.5 * ((c[:, 0] - a[:, 0]) * (d[:, 1] - a[:, 1]) - (c[:, 1] - a[:, 1]) * (d[:, 0] - a[:, 0]))
    areas = area1 + area2
    lens = edge_lengths(points, topology.edges)
    icn = quad_icn(points, topology.elements)
    ige = quad_ige(points, topology.elements)

    area_term = jnp.var(jnp.abs(areas))
    edge_term = jnp.var(lens)
    fold_penalty = jnp.mean(jax.nn.softplus(-areas * 100.0))
    quality_barrier = jnp.mean(jax.nn.softplus((0.2 - icn) * 20.0) + jax.nn.softplus((0.2 - ige) * 20.0))
    return area_term + edge_term + 1.0e-3 * fold_penalty + 1.0e-3 * quality_barrier


def tet_mesh_quality_energy(points: jnp.ndarray, topology: MeshTopology) -> jnp.ndarray:
    """Smooth quality energy for fixed-topology tetrahedral meshes."""
    t = points[topology.elements]
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
    volumes = det / 6.0
    lens = edge_lengths(points, topology.edges)
    icn = tet_icn(points, topology.elements)
    ige = tet_ige(points, topology.elements)

    volume_term = jnp.var(jnp.abs(volumes))
    edge_term = jnp.var(lens)
    fold_penalty = jnp.mean(jax.nn.softplus(-volumes * 100.0))
    quality_barrier = jnp.mean(jax.nn.softplus((0.1 - icn) * 20.0) + jax.nn.softplus((0.1 - ige) * 20.0))
    return volume_term + edge_term + 1.0e-3 * fold_penalty + 1.0e-3 * quality_barrier


def line_mesh_quality_energy(points: jnp.ndarray, topology: MeshTopology) -> jnp.ndarray:
    """Smooth quality energy for fixed-topology line meshes."""
    lengths = line_element_lengths(points, topology.elements)
    src = topology.edges[:, 0]
    dst = topology.edges[:, 1]
    neighbor_sum = jnp.zeros_like(points)
    neighbor_sum = neighbor_sum.at[src].add(points[dst])
    neighbor_sum = neighbor_sum.at[dst].add(points[src])
    deg = jnp.zeros((points.shape[0],), dtype=points.dtype)
    one = jnp.ones((topology.edges.shape[0],), dtype=points.dtype)
    deg = deg.at[src].add(one)
    deg = deg.at[dst].add(one)
    mean_nbr = neighbor_sum / jnp.maximum(deg[:, None], 1.0)
    lap = mean_nbr - points
    length_term = jnp.var(lengths)
    smooth_term = jnp.mean(jnp.sum(lap * lap, axis=1))
    collapse_penalty = jnp.mean(jax.nn.softplus((0.05 - lengths) * 40.0))
    return length_term + 0.1 * smooth_term + 1.0e-3 * collapse_penalty
