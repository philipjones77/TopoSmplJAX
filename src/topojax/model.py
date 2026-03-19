"""Gmsh-inspired model containers for JAX workflows.

Design mirrors key Gmsh ideas:
- immutable node/element ids
- element-to-entity ownership
- topology fixed, node coordinates movable
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from topojax.mesh.topology import MeshTopology


class GEntity(NamedTuple):
    """Minimal geometric entity descriptor (dim/tag)."""

    dim: int
    tag: int


class MeshState(NamedTuple):
    """Dynamic nodal coordinates for a fixed topology."""

    points: jnp.ndarray  # (n_nodes, 2)


class MeshModel(NamedTuple):
    """Static model data + current dynamic state."""

    topology: MeshTopology
    entities: tuple[GEntity, ...]
    state: MeshState


def update_points(model: MeshModel, points: jnp.ndarray) -> MeshModel:
    """Return a new model with updated coordinates but same topology."""
    return MeshModel(topology=model.topology, entities=model.entities, state=MeshState(points=points))
