"""High-level Mode 1 domain initialization and workflow helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp

from gmshjax.ad.mode1 import Mode1OptimizationResult, export_mode1_artifacts, optimize_mode1_fixed_topology
from gmshjax.io.gmsh_viewer import launch_gmsh_viewer
from gmshjax.io.imports import load_gmsh_msh
from gmshjax.mesh.domains import (
    DomainMeshMetadata,
    box_volume_tet_mesh_tagged,
    extruded_polygon_tet_mesh,
    implicit_volume_tet_mesh_tagged,
    polygon_domain_quad_mesh_tagged,
    polygon_domain_tri_mesh_tagged,
)
from gmshjax.mesh.topology import MeshTopology, polyline_mesh, unit_interval_line_mesh


class Mode1Domain(NamedTuple):
    topology: MeshTopology
    points: jnp.ndarray
    metadata: DomainMeshMetadata | None = None


class Mode1WorkflowRun(NamedTuple):
    domain: Mode1Domain
    result: Mode1OptimizationResult
    artifacts: dict[str, Path]


def initialize_mode1_domain(kind: str, **kwargs: Any) -> Mode1Domain:
    """Initialize a supported Mode 1 domain."""
    if kind == "line":
        if "points" in kwargs:
            topo, points = polyline_mesh(kwargs["points"], closed=bool(kwargs.get("closed", False)))
        else:
            topo, points = unit_interval_line_mesh(int(kwargs.get("n", 16)))
        return Mode1Domain(topology=topo, points=points)
    if kind == "polygon":
        topo, points, metadata = polygon_domain_tri_mesh_tagged(
            kwargs["outer_boundary"],
            holes=kwargs.get("holes"),
            target_edge_size=kwargs.get("target_edge_size"),
            interior_relaxation=float(kwargs.get("interior_relaxation", 0.35)),
        )
        return Mode1Domain(topo, points, metadata)
    if kind == "polygon-quad":
        topo, points, metadata = polygon_domain_quad_mesh_tagged(
            kwargs["outer_boundary"],
            holes=kwargs.get("holes"),
            target_edge_size=kwargs.get("target_edge_size"),
            interior_relaxation=float(kwargs.get("interior_relaxation", 0.35)),
        )
        return Mode1Domain(topo, points, metadata)
    if kind == "extruded":
        topo, points, metadata = extruded_polygon_tet_mesh(
            kwargs["outer_boundary"],
            holes=kwargs.get("holes"),
            target_edge_size=kwargs.get("target_edge_size"),
            height=float(kwargs.get("height", 1.0)),
            layers=int(kwargs.get("layers", 4)),
        )
        return Mode1Domain(topo, points, metadata)
    if kind == "box-volume":
        topo, points, metadata = box_volume_tet_mesh_tagged(
            kwargs["bbox_min"],
            kwargs["bbox_max"],
            int(kwargs["nx"]),
            int(kwargs["ny"]),
            int(kwargs["nz"]),
        )
        return Mode1Domain(topo, points, metadata)
    if kind == "implicit-volume":
        topo, points, metadata = implicit_volume_tet_mesh_tagged(
            kwargs["level_set_fn"],
            kwargs["bbox_min"],
            kwargs["bbox_max"],
            int(kwargs["nx"]),
            int(kwargs["ny"]),
            int(kwargs["nz"]),
        )
        return Mode1Domain(topo, points, metadata)
    if kind == "import-msh":
        imported = load_gmsh_msh(kwargs["path"], primary_element_kind=kwargs.get("primary_element_kind"))
        metadata = DomainMeshMetadata(
            boundary_element_blocks=imported.extra_element_blocks,
            physical_names=imported.physical_names,
        )
        return Mode1Domain(imported.topology, imported.points, metadata)
    raise ValueError("Unsupported Mode 1 domain kind")


def run_mode1_workflow(
    domain: Mode1Domain,
    *,
    output_dir: str | Path,
    prefix: str = "mode1",
    steps: int = 80,
    step_size: float = 0.03,
    movable_mask: jnp.ndarray | None = None,
    diagnostics_every: int = 10,
    launch_gmsh: bool = False,
    gmsh_executable: str = "gmsh",
    gmsh_extra_args: list[str] | None = None,
) -> Mode1WorkflowRun:
    """Run the end-to-end Mode 1 optimize-export-view workflow."""
    result = optimize_mode1_fixed_topology(
        domain.points,
        domain.topology,
        steps=steps,
        step_size=step_size,
        movable_mask=movable_mask,
        diagnostics_every=diagnostics_every,
    )
    artifacts = export_mode1_artifacts(
        output_dir,
        result,
        prefix=prefix,
        extra_element_blocks=None if domain.metadata is None else domain.metadata.boundary_element_blocks,
        physical_names=None if domain.metadata is None else domain.metadata.physical_names,
    )
    if launch_gmsh:
        launch_gmsh_viewer(artifacts["mesh"], gmsh_executable=gmsh_executable, extra_args=gmsh_extra_args)
    return Mode1WorkflowRun(domain=domain, result=result, artifacts=artifacts)
