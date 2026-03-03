"""Diagnostics helpers for mesh quality and evolution."""

from __future__ import annotations

from typing import TypedDict

import jax.numpy as jnp

from gmshjax.mesh.adaptive_quad import quad_area_magnitudes
from gmshjax.mesh.operators import quad_icn, quad_ige, tet_icn, tet_ige, triangle_icn, triangle_ige
from gmshjax.mesh.refine import triangle_area_magnitudes


class MeshStats(TypedDict):
    n_nodes: int
    n_elements: int
    mean_icn: float
    min_icn: float
    mean_ige: float
    min_ige: float
    max_area_or_volume_proxy: float


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
