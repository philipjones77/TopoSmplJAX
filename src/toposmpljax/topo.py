"""Topo backend adapters for the combined TopoSmplJAX namespace."""

from __future__ import annotations

from typing import Any

from topojax.ad.modes import MeshMovementMode
from topojax.rf77 import (
    RandomFields77ModeBridge,
    build_mode1_randomfields77_bridge,
    build_mode2_randomfields77_bridge,
    build_mode3_randomfields77_bridge,
    build_mode4_randomfields77_bridge,
    build_mode5_randomfields77_bridge,
)

from .mesh_repair import MeshRepairResult, RepairBackend, repair_topo_mesh_for_printing


def build_mode_bridge(source: Any, mode: MeshMovementMode | str, **kwargs: Any) -> RandomFields77ModeBridge:
    normalized = MeshMovementMode(mode)
    if normalized == MeshMovementMode.FIXED_TOPOLOGY:
        return build_mode1_randomfields77_bridge(source, **kwargs)
    if normalized == MeshMovementMode.REMESH_RESTART:
        return build_mode2_randomfields77_bridge(source, **kwargs)
    if normalized == MeshMovementMode.SOFT_CONNECTIVITY:
        if hasattr(source, "points") and hasattr(source, "topology"):
            return build_mode3_randomfields77_bridge(source.points, source.topology, metadata=getattr(source, "metadata", None), **kwargs)
        raise TypeError("Mode 3 topo bridge expects a source with `.points` and `.topology`.")
    if normalized == MeshMovementMode.STRAIGHT_THROUGH:
        if hasattr(source, "points") and hasattr(source, "topology"):
            return build_mode4_randomfields77_bridge(source.points, source.topology, metadata=getattr(source, "metadata", None), **kwargs)
        raise TypeError("Mode 4 topo bridge expects a source with `.points` and `.topology`.")
    if normalized == MeshMovementMode.FULLY_DYNAMIC:
        if hasattr(source, "points") and hasattr(source, "topology"):
            return build_mode5_randomfields77_bridge(source.points, source.topology, metadata=getattr(source, "metadata", None), **kwargs)
        raise TypeError("Mode 5 topo bridge expects a source with `.points` and `.topology`.")
    raise KeyError(f"Unsupported topo mode: {mode}")


def repair_print_mesh(
    points: Any,
    elements: Any,
    *,
    element_kind: str | None = None,
    backend: RepairBackend = "auto",
    **kwargs: Any,
) -> MeshRepairResult:
    return repair_topo_mesh_for_printing(points, elements, element_kind=element_kind, backend=backend, **kwargs)
