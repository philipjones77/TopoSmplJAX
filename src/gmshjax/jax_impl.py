"""JAX-native implementation wrappers and dtype helpers."""

from __future__ import annotations

import jax.numpy as jnp

from gmshjax.ad.pipeline import build_model_parametric_quality_value_and_grad
from gmshjax.mesh.factory import make_unit_square_model
from gmshjax.mesh.manifold import DeformationParams, apply_deformation
from gmshjax.mesh.operators import (
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
from gmshjax.mesh.topology import unit_cube_tet_mesh, unit_square_quad_mesh, unit_square_tri_mesh


def as_jax_array(x, dtype=jnp.float32):
    """Convert inputs to JAX arrays with project-consistent dtype."""
    return jnp.asarray(x, dtype=dtype)


def default_deformation_params(dtype=jnp.float32) -> DeformationParams:
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
