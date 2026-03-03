"""JAX-compiled objective builders for fixed-topology meshes."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gmshjax.mesh.operators import mesh_quality_energy
from gmshjax.mesh.topology import MeshTopology


def build_quality_value_and_grad(topology: MeshTopology):
    """Compile value/grad for moving node coordinates with fixed topology."""

    def energy_fn(points: jnp.ndarray) -> jnp.ndarray:
        return mesh_quality_energy(points, topology)

    return jax.jit(jax.value_and_grad(energy_fn))
