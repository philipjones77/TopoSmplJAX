"""Catalog of mesh movement modes supported or planned in GmshJAX."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple


class MeshMovementMode(str, Enum):
    """High-level modes for moving and optionally remeshing a mesh."""

    FIXED_TOPOLOGY = "fixed-topology-ad"
    REMESH_RESTART = "remesh-restart"
    SOFT_CONNECTIVITY = "soft-connectivity-surrogate"
    STRAIGHT_THROUGH = "straight-through-connectivity"
    FULLY_DYNAMIC = "fully-dynamic-remeshing"


class MeshMovementModeSpec(NamedTuple):
    mode: MeshMovementMode
    summary: str
    connectivity: str
    geometry_motion: bool
    autodiff_status: str
    implementation_status: str


MESH_MOVEMENT_MODES: tuple[MeshMovementModeSpec, ...] = (
    MeshMovementModeSpec(
        mode=MeshMovementMode.FIXED_TOPOLOGY,
        summary="Move node coordinates while keeping connectivity fixed.",
        connectivity="static",
        geometry_motion=True,
        autodiff_status="full",
        implementation_status="implemented",
    ),
    MeshMovementModeSpec(
        mode=MeshMovementMode.REMESH_RESTART,
        summary="Optimize with fixed topology, remesh discretely, then restart on the new mesh.",
        connectivity="piecewise static between discrete remesh phases",
        geometry_motion=True,
        autodiff_status="phase-local only",
        implementation_status="implemented",
    ),
    MeshMovementModeSpec(
        mode=MeshMovementMode.SOFT_CONNECTIVITY,
        summary="Optimize soft connectivity weights over a fixed candidate graph.",
        connectivity="soft over a fixed candidate set",
        geometry_motion=True,
        autodiff_status="surrogate",
        implementation_status="experimental",
    ),
    MeshMovementModeSpec(
        mode=MeshMovementMode.STRAIGHT_THROUGH,
        summary="Use hard connectivity choices in the forward pass and soft gradients in backward.",
        connectivity="hard choices over a fixed candidate set",
        geometry_motion=True,
        autodiff_status="straight-through surrogate",
        implementation_status="experimental",
    ),
    MeshMovementModeSpec(
        mode=MeshMovementMode.FULLY_DYNAMIC,
        summary="Allow full remeshing while geometry is moving inside a single optimization process.",
        connectivity="fully dynamic",
        geometry_motion=True,
        autodiff_status="aspirational",
        implementation_status="planned",
    ),
)


def get_mesh_movement_modes() -> tuple[MeshMovementModeSpec, ...]:
    """Return the ordered mesh movement mode catalog."""

    return MESH_MOVEMENT_MODES


def get_mesh_movement_mode(mode: MeshMovementMode | str) -> MeshMovementModeSpec:
    """Look up one mesh movement mode specification."""

    needle = MeshMovementMode(mode)
    for spec in MESH_MOVEMENT_MODES:
        if spec.mode == needle:
            return spec
    raise KeyError(f"Unknown mesh movement mode: {mode}")