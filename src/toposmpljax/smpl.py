"""SMPL backend adapters for the combined TopoSmplJAX namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from topojax.ad.modes import MeshMovementMode
from topojax.mesh.topology import mesh_topology_from_points_and_elements
from topojax.rf77 import (
    RandomFields77ModeBridge,
    build_mode1_randomfields77_bridge,
    build_mode3_randomfields77_bridge,
    build_mode4_randomfields77_bridge,
    build_mode5_randomfields77_bridge,
)
from smpljax.body_models import SMPLJAXModel
from smpljax.mesh_export import export_posed_mesh, export_template_mesh
from smpljax.optimized import ForwardInputs, OptimizedSMPLJAX

from .mesh_repair import MeshRepairResult, RepairBackend, repair_smpl_mesh_for_printing


ModelLike = SMPLJAXModel | OptimizedSMPLJAX


def _template_points(model: ModelLike) -> jnp.ndarray:
    mesh = export_template_mesh(model)
    return jnp.asarray(mesh["nodes"])


def _topology(model: ModelLike):
    mesh = export_template_mesh(model)
    return mesh_topology_from_points_and_elements(jnp.asarray(mesh["nodes"]), jnp.asarray(mesh["faces"], dtype=jnp.int32))


def _posed_points_fn(model: ModelLike):
    def geometry_fn(params: Mapping[str, Any] | None = None) -> jnp.ndarray:
        if params is None:
            return _template_points(model)
        mesh = export_posed_mesh(model, params if isinstance(params, ForwardInputs) else dict(params))
        return jnp.asarray(mesh["nodes"])

    return geometry_fn


def build_mode_bridge(
    model: ModelLike,
    mode: MeshMovementMode | str,
    *,
    params: Mapping[str, Any] | ForwardInputs | None = None,
    **kwargs: Any,
) -> RandomFields77ModeBridge:
    normalized = MeshMovementMode(mode)
    points = _template_points(model) if params is None else jnp.asarray(export_posed_mesh(model, params)["nodes"])
    topology = _topology(model)
    geometry_fn = _posed_points_fn(model)

    if normalized == MeshMovementMode.FIXED_TOPOLOGY:
        return build_mode1_randomfields77_bridge(
            type("SmplDomain", (), {"points": points, "topology": topology, "metadata": None})(),
            geometry_fn=geometry_fn,
            builder_options={"backend": "smpl", **kwargs.pop("builder_options", {})},
            **kwargs,
        )
    if normalized == MeshMovementMode.REMESH_RESTART:
        return build_mode3_randomfields77_bridge(
            points,
            topology,
            geometry_fn=geometry_fn,
            builder_options={"backend": "smpl", "review_note": "smpl backend does not implement remesh-restart; exported as review stub"},
            **kwargs,
        )
    if normalized == MeshMovementMode.SOFT_CONNECTIVITY:
        return build_mode3_randomfields77_bridge(
            points,
            topology,
            geometry_fn=geometry_fn,
            **kwargs,
        )
    if normalized == MeshMovementMode.STRAIGHT_THROUGH:
        return build_mode4_randomfields77_bridge(
            points,
            topology,
            geometry_fn=geometry_fn,
            **kwargs,
        )
    if normalized == MeshMovementMode.FULLY_DYNAMIC:
        return build_mode5_randomfields77_bridge(
            points,
            topology,
            geometry_fn=geometry_fn,
            **kwargs,
        )
    raise KeyError(f"Unsupported smpl mode: {mode}")


def repair_print_mesh(
    model: ModelLike,
    *,
    params: Mapping[str, Any] | ForwardInputs | None = None,
    batch_index: int = 0,
    backend: RepairBackend = "auto",
    **kwargs: Any,
) -> MeshRepairResult:
    return repair_smpl_mesh_for_printing(
        model,
        params=params,
        batch_index=batch_index,
        backend=backend,
        **kwargs,
    )
