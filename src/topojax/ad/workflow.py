"""High-level Mode 1 and Mode 2 workflow helpers."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from topojax.ad.mode1 import Mode1OptimizationResult, export_mode1_artifacts, optimize_mode1_fixed_topology
from topojax.ad.restart import (
    RestartOptimizationResult,
    export_mode2_artifacts,
    optimize_remesh_restart_quad,
    optimize_remesh_restart_tet,
    optimize_remesh_restart_tri,
)
from topojax.ad.workflow_common import MeshWorkflowDomain, initialize_workflow_domain
from topojax.io.gmsh_viewer import launch_gmsh_viewer
from topojax.mesh.adaptive_quad import quad_area_magnitudes
from topojax.mesh.adaptive_tet import tet_volume_magnitudes
from topojax.mesh.operators import triangle_signed_areas


Mode1Domain = MeshWorkflowDomain
Mode2Domain = MeshWorkflowDomain


class Mode1WorkflowRun(NamedTuple):
    domain: Mode1Domain
    result: Mode1OptimizationResult
    artifacts: dict[str, Path]


class Mode2WorkflowRun(NamedTuple):
    domain: Mode2Domain
    result: RestartOptimizationResult
    artifacts: dict[str, Path]


def initialize_mode1_domain(kind: str, **kwargs) -> Mode1Domain:
    """Initialize a supported Mode 1 domain."""
    return initialize_workflow_domain(kind, **kwargs)


def initialize_mode2_domain(kind: str, **kwargs) -> Mode2Domain:
    """Initialize a supported Mode 2 domain."""
    return initialize_workflow_domain(kind, **kwargs)


def run_mode1_workflow(
    domain: Mode1Domain,
    *,
    output_dir: str | Path,
    prefix: str = "mode1",
    steps: int = 80,
    step_size: float = 0.03,
    movable_mask: jnp.ndarray | None = None,
    diagnostics_every: int = 10,
    export_stl_surface: bool = False,
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
        export_stl_surface=export_stl_surface,
        extra_element_blocks=None if domain.metadata is None else domain.metadata.boundary_element_blocks,
        physical_names=None if domain.metadata is None else domain.metadata.physical_names,
    )
    if launch_gmsh:
        launch_gmsh_viewer(artifacts["mesh"], gmsh_executable=gmsh_executable, extra_args=gmsh_extra_args)
    return Mode1WorkflowRun(domain=domain, result=result, artifacts=artifacts)


def _default_restart_target(domain: Mode2Domain) -> tuple[str, float]:
    order = int(domain.topology.elements.shape[1])
    dim = int(domain.points.shape[1])
    if order == 3:
        mean_area = float(jnp.mean(jnp.abs(triangle_signed_areas(domain.points, domain.topology.elements))))
        return "target_area", max(0.8 * mean_area, 1.0e-8)
    if order == 4 and dim == 2:
        mean_area = float(jnp.mean(quad_area_magnitudes(domain.points, domain.topology.elements)))
        return "target_area", max(0.8 * mean_area, 1.0e-8)
    if order == 4 and dim == 3:
        mean_volume = float(jnp.mean(tet_volume_magnitudes(domain.points, domain.topology.elements)))
        return "target_volume", max(0.8 * mean_volume, 1.0e-8)
    raise ValueError("Mode 2 currently supports triangle, quad, and tetra domains only")


def run_mode2_restart_workflow(
    domain: Mode2Domain,
    *,
    output_dir: str | Path,
    prefix: str = "mode2",
    cycles: int = 2,
    optimization_steps: int = 60,
    optimization_step_size: float = 0.03,
    max_nodes: int | None = None,
    max_elements: int | None = None,
    remesh_max_iters: int | None = None,
    target_area: float | None = None,
    target_volume: float | None = None,
    target_mean_icn: float | None = None,
    smoothing_alpha: float | None = None,
    smoothing_steps: int = 2,
    movable_mask: jnp.ndarray | None = None,
    export_stl_surface: bool = False,
    launch_gmsh: bool = False,
    gmsh_executable: str = "gmsh",
    gmsh_extra_args: list[str] | None = None,
) -> Mode2WorkflowRun:
    """Run the end-to-end Mode 2 remesh-restart workflow."""
    n_nodes = int(domain.points.shape[0])
    n_elements = int(domain.topology.elements.shape[0])
    max_nodes_local = max_nodes if max_nodes is not None else max(4 * n_nodes, n_nodes + 32)
    max_elements_local = max_elements if max_elements is not None else max(4 * n_elements, n_elements + 64)

    order = int(domain.topology.elements.shape[1])
    dim = int(domain.points.shape[1])
    _, inferred_target = _default_restart_target(domain)
    physical_names = None
    extra_element_blocks = None

    if order == 3:
        result = optimize_remesh_restart_tri(
            domain.points,
            domain.topology.elements,
            cycles=cycles,
            optimization_steps=optimization_steps,
            optimization_step_size=optimization_step_size,
            max_nodes=max_nodes_local,
            max_elements=max_elements_local,
            remesh_max_iters=4 if remesh_max_iters is None else remesh_max_iters,
            target_area=inferred_target if target_area is None else target_area,
            target_mean_icn=0.50 if target_mean_icn is None else target_mean_icn,
            smoothing_alpha=0.15 if smoothing_alpha is None else smoothing_alpha,
            smoothing_steps=smoothing_steps,
            movable_mask=movable_mask,
        )
    elif order == 4 and dim == 2:
        result = optimize_remesh_restart_quad(
            domain.points,
            domain.topology.elements,
            cycles=cycles,
            optimization_steps=optimization_steps,
            optimization_step_size=optimization_step_size,
            max_nodes=max_nodes_local,
            max_elements=max_elements_local,
            remesh_max_iters=4 if remesh_max_iters is None else remesh_max_iters,
            target_area=inferred_target if target_area is None else target_area,
            target_mean_icn=0.55 if target_mean_icn is None else target_mean_icn,
            smoothing_alpha=0.15 if smoothing_alpha is None else smoothing_alpha,
            smoothing_steps=smoothing_steps,
            movable_mask=movable_mask,
        )
    elif order == 4 and dim == 3:
        result = optimize_remesh_restart_tet(
            domain.points,
            domain.topology.elements,
            cycles=cycles,
            optimization_steps=optimization_steps,
            optimization_step_size=optimization_step_size,
            max_nodes=max_nodes_local,
            max_elements=max_elements_local,
            remesh_max_iters=3 if remesh_max_iters is None else remesh_max_iters,
            target_volume=inferred_target if target_volume is None else target_volume,
            target_mean_icn=0.35 if target_mean_icn is None else target_mean_icn,
            smoothing_alpha=0.12 if smoothing_alpha is None else smoothing_alpha,
            smoothing_steps=smoothing_steps,
            movable_mask=movable_mask,
        )
    else:
        raise ValueError("Mode 2 currently supports triangle, quad, and tetra domains only")

    if not any(phase.remeshed for phase in result.phases) and domain.metadata is not None:
        extra_element_blocks = domain.metadata.boundary_element_blocks
        physical_names = domain.metadata.physical_names

    artifacts = export_mode2_artifacts(
        output_dir,
        result,
        prefix=prefix,
        export_stl_surface=export_stl_surface,
        extra_element_blocks=extra_element_blocks,
        physical_names=physical_names,
    )
    if launch_gmsh:
        launch_gmsh_viewer(artifacts["mesh"], gmsh_executable=gmsh_executable, extra_args=gmsh_extra_args)
    return Mode2WorkflowRun(domain=domain, result=result, artifacts=artifacts)
