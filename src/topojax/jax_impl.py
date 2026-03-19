"""JAX-native implementation wrappers and dtype helpers."""

from __future__ import annotations

import jax.numpy as jnp

from topojax.ad.pipeline import build_model_parametric_quality_value_and_grad
from topojax.mesh.factory import make_unit_square_model
from topojax.mesh.manifold import DeformationParams, apply_deformation
from topojax.mesh.operators import (
    edge_lengths,
    mesh_quality_energy,
    quad_icn,
    quad_ige,
    tet_icn,
    tet_ige,
    triangle_icn,
    triangle_ige,
    triangle_signed_areas,
)
from topojax.mesh.topology import unit_cube_tet_mesh, unit_square_quad_mesh, unit_square_tri_mesh
from topojax.runtime import jax_float_dtype


def as_jax_array(x, dtype=None):
    """Convert inputs to JAX arrays with project-consistent dtype."""
    if dtype is None:
        dtype = jax_float_dtype()
    return jnp.asarray(x, dtype=dtype)


def default_deformation_params(dtype=None) -> DeformationParams:
    if dtype is None:
        dtype = jax_float_dtype()
    return DeformationParams(
        translation=jnp.array([0.0, 0.0], dtype=dtype),
        scale=jnp.array([1.0, 1.0], dtype=dtype),
        shear=jnp.array([0.0, 0.0], dtype=dtype),
        bend=jnp.array([0.0, 0.0], dtype=dtype),
    )


__all__ = [
    "DeformationParams",
    "apply_deformation",
    "as_jax_array",
    "build_model_parametric_quality_value_and_grad",
    "default_deformation_params",
    "edge_lengths",
    "make_unit_square_model",
    "mesh_quality_energy",
    "quad_icn",
    "quad_ige",
    "tet_icn",
    "tet_ige",
    "triangle_icn",
    "triangle_ige",
    "triangle_signed_areas",
    "unit_cube_tet_mesh",
    "unit_square_quad_mesh",
    "unit_square_tri_mesh",
]
