"""Diagnostics helpers for mesh quality and evolution."""

from __future__ import annotations

from typing import TypedDict

import jax.numpy as jnp

from topojax.mesh.adaptive_quad import quad_area_magnitudes
from topojax.mesh.operators import line_element_lengths, line_mesh_quality_energy, quad_icn, quad_ige, tet_icn, tet_ige, triangle_icn, triangle_ige
from topojax.mesh.refine import triangle_area_magnitudes
from topojax.mesh.topology import mesh_topology_from_points_and_elements


class MeshStats(TypedDict):
    n_nodes: int
    n_elements: int
    mean_icn: float
    min_icn: float
    mean_ige: float
    min_ige: float
    max_area_or_volume_proxy: float


def line_diagnostics(points: jnp.ndarray, elements: jnp.ndarray) -> dict[str, float | int]:
    lengths = line_element_lengths(points, elements)
    topo = mesh_topology_from_points_and_elements(points, elements)
    topo_energy = line_mesh_quality_energy(points, topo)
    return {
        "n_nodes": int(points.shape[0]),
        "n_elements": int(elements.shape[0]),
        "mean_edge_length": float(jnp.mean(lengths)),
        "min_edge_length": float(jnp.min(lengths)),
        "max_edge_length": float(jnp.max(lengths)),
        "line_energy": float(topo_energy),
    }


def tri_diagnostics(points: jnp.ndarray, elements: jnp.ndarray) -> MeshStats:
    icn = triangle_icn(points, elements)
    ige = triangle_ige(points, elements)
    area = triangle_area_magnitudes(points, elements)
    return MeshStats(
        n_nodes=int(points.shape[0]),
        n_elements=int(elements.shape[0]),
        mean_icn=float(jnp.mean(icn)),
        min_icn=float(jnp.min(icn)),
        mean_ige=float(jnp.mean(ige)),
        min_ige=float(jnp.min(ige)),
        max_area_or_volume_proxy=float(jnp.max(area)),
    )


def quad_diagnostics(points: jnp.ndarray, elements: jnp.ndarray) -> MeshStats:
    icn = quad_icn(points, elements)
    ige = quad_ige(points, elements)
    area = quad_area_magnitudes(points, elements)
    return MeshStats(
        n_nodes=int(points.shape[0]),
        n_elements=int(elements.shape[0]),
        mean_icn=float(jnp.mean(icn)),
        min_icn=float(jnp.min(icn)),
        mean_ige=float(jnp.mean(ige)),
        min_ige=float(jnp.min(ige)),
        max_area_or_volume_proxy=float(jnp.max(area)),
    )


def tet_diagnostics(points: jnp.ndarray, elements: jnp.ndarray) -> MeshStats:
    icn = tet_icn(points, elements)
    ige = tet_ige(points, elements)
    # Volume proxy via determinant magnitude from ICN scaling fallback.
    vol_proxy = jnp.abs(icn)
    return MeshStats(
        n_nodes=int(points.shape[0]),
        n_elements=int(elements.shape[0]),
        mean_icn=float(jnp.mean(icn)),
        min_icn=float(jnp.min(icn)),
        mean_ige=float(jnp.mean(ige)),
        min_ige=float(jnp.min(ige)),
        max_area_or_volume_proxy=float(jnp.max(vol_proxy)),
    )
