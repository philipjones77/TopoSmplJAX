"""Model factories for common meshes."""

from __future__ import annotations

from topojax.mesh.topology import unit_square_tri_mesh
from topojax.model import GEntity, MeshModel, MeshState


def make_unit_square_model(nx: int, ny: int) -> MeshModel:
    """Create a Gmsh-inspired model for a single surface entity."""
    topology, points = unit_square_tri_mesh(nx, ny)
    entities = (GEntity(dim=2, tag=1),)
    return MeshModel(topology=topology, entities=entities, state=MeshState(points=points))
