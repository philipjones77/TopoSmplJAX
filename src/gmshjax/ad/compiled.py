"""JAX-compiled objective builders for fixed-topology meshes."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gmshjax.mesh.operators import line_mesh_quality_energy, mesh_quality_energy, quad_mesh_quality_energy, tet_mesh_quality_energy
from gmshjax.mesh.topology import MeshTopology


def _quality_energy(points: jnp.ndarray, topology: MeshTopology) -> jnp.ndarray:
    order = int(topology.elements.shape[1])
    if order == 2:
        return line_mesh_quality_energy(points, topology)
    if order == 3:
        return mesh_quality_energy(points, topology)
    if order == 4 and points.shape[1] == 2:
        return quad_mesh_quality_energy(points, topology)
    if order == 4 and points.shape[1] == 3:
        return tet_mesh_quality_energy(points, topology)
    raise ValueError("Unsupported topology for compiled quality evaluation")


def build_quality_value_and_grad(topology: MeshTopology):
    """Compile value/grad for moving node coordinates with fixed topology."""

    def energy_fn(points: jnp.ndarray) -> jnp.ndarray:
        return _quality_energy(points, topology)

    return jax.jit(jax.value_and_grad(energy_fn))
