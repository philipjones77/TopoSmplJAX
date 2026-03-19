"""Restart-style workflows that alternate fixed-topology AD with remeshing."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from topojax.ad._common import cached_build, coerce_runtime_points, fit_node_mask, mesh_topology_metrics, topology_cache_key
from topojax.ad.compiled import build_quality_value_and_grad
from topojax.io.exports import GmshElementBlock, export_binary_stl, export_gmsh_msh, export_metrics_json, export_snapshot_npz
from topojax.mesh.adaptive import AdaptiveHistory, adaptive_remesh_tri
from topojax.mesh.adaptive_quad import QuadAdaptiveHistory, adaptive_remesh_quad
from topojax.mesh.adaptive_tet import TetAdaptiveHistory, adaptive_remesh_tet
from topojax.mesh.mutation import active_elements, active_points
from topojax.mesh.mutation_qt import active_quad_elements, active_quad_points, active_tet_elements, active_tet_points
from topojax.mesh.topology import MeshTopology, quad_edges, tet_edges, triangle_edges
 

_RESTART_SCAN_CACHE: OrderedDict[tuple[object, ...], object] = OrderedDict()


class RestartPhase(NamedTuple):
    cycle: int
    start_energy: float
    final_energy: float
    n_nodes: int
    n_elements: int
    remeshed: bool


class RestartTriOptimizationResult(NamedTuple):
    points: jnp.ndarray
    elements: jnp.ndarray
    phases: list[RestartPhase]
    remesh_histories: list[list[AdaptiveHistory]]


class RestartQuadOptimizationResult(NamedTuple):
    points: jnp.ndarray
    elements: jnp.ndarray
    phases: list[RestartPhase]
    remesh_histories: list[list[QuadAdaptiveHistory]]


class RestartTetOptimizationResult(NamedTuple):
    points: jnp.ndarray
    elements: jnp.ndarray
    phases: list[RestartPhase]
    remesh_histories: list[list[TetAdaptiveHistory]]


RestartOptimizationResult = RestartTriOptimizationResult | RestartQuadOptimizationResult | RestartTetOptimizationResult


def _build_fixed_topology_scan(
    topology: MeshTopology,
    *,
    steps: int,
):
    key = ("restart_scan", topology_cache_key(topology), int(steps))

    def _build():
        value_and_grad = build_quality_value_and_grad(topology)

        @jax.jit
        def run(points: jnp.ndarray, step_size, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            pts0 = coerce_runtime_points(points)
            step_size_local = jnp.asarray(step_size, dtype=pts0.dtype)
            mask_local = jnp.asarray(mask, dtype=pts0.dtype)

            def body(pts: jnp.ndarray, _):
                value, grad = value_and_grad(pts)
                grad = grad * mask_local[:, None]
                next_pts = pts - step_size_local * grad
                return next_pts, value

            final_points, losses = jax.lax.scan(body, pts0, xs=None, length=steps)
            return final_points, losses

        return run

    return cached_build(_RESTART_SCAN_CACHE, key, _build)


def triangle_topology_from_elements(elements: jnp.ndarray, n_nodes: int | None = None) -> MeshTopology:
    elems = jnp.asarray(elements, dtype=jnp.int32)
    if elems.shape[0] == 0:
        raise ValueError("elements must be non-empty")
    if n_nodes is None:
        n_nodes = int(jnp.max(elems)) + 1
    n_elem = int(elems.shape[0])
    return MeshTopology(
        elements=elems,
        edges=triangle_edges(elems),
        node_ids=jnp.arange(n_nodes, dtype=jnp.int32),
        element_ids=jnp.arange(n_elem, dtype=jnp.int32),
        element_entity_tags=jnp.ones((n_elem,), dtype=jnp.int32),
        n_nodes=n_nodes,
    )


def quad_topology_from_elements(elements: jnp.ndarray, n_nodes: int | None = None) -> MeshTopology:
    elems = jnp.asarray(elements, dtype=jnp.int32)
    if elems.shape[0] == 0:
        raise ValueError("elements must be non-empty")
    if n_nodes is None:
        n_nodes = int(jnp.max(elems)) + 1
    n_elem = int(elems.shape[0])
    return MeshTopology(
        elements=elems,
        edges=quad_edges(elems),
        node_ids=jnp.arange(n_nodes, dtype=jnp.int32),
        element_ids=jnp.arange(n_elem, dtype=jnp.int32),
        element_entity_tags=jnp.ones((n_elem,), dtype=jnp.int32),
        n_nodes=n_nodes,
    )


def tet_topology_from_elements(elements: jnp.ndarray, n_nodes: int | None = None) -> MeshTopology:
    elems = jnp.asarray(elements, dtype=jnp.int32)
    if elems.shape[0] == 0:
        raise ValueError("elements must be non-empty")
    if n_nodes is None:
        n_nodes = int(jnp.max(elems)) + 1
    n_elem = int(elems.shape[0])
    return MeshTopology(
        elements=elems,
        edges=tet_edges(elems),
        node_ids=jnp.arange(n_nodes, dtype=jnp.int32),
        element_ids=jnp.arange(n_elem, dtype=jnp.int32),
        element_entity_tags=jnp.ones((n_elem,), dtype=jnp.int32),
        n_nodes=n_nodes,
    )


def optimize_points_fixed_topology(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    steps: int = 80,
    step_size: float = 0.03,
    movable_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Optimize node coordinates while keeping topology fixed."""
    pts = coerce_runtime_points(points)
    mask = fit_node_mask(movable_mask, int(pts.shape[0]))
    if mask is None:
        mask = jnp.ones((pts.shape[0],), dtype=bool)
    optimizer = _build_fixed_topology_scan(topology, steps=steps)
    return optimizer(pts, step_size, mask)


def optimize_remesh_restart_tri(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    cycles: int = 2,
    optimization_steps: int = 60,
    optimization_step_size: float = 0.03,
    max_nodes: int,
    max_elements: int,
    remesh_max_iters: int = 4,
    target_area: float = 1.0e-3,
    target_mean_icn: float = 0.50,
    smoothing_alpha: float = 0.15,
    smoothing_steps: int = 2,
    movable_mask: jnp.ndarray | None = None,
) -> RestartTriOptimizationResult:
    """Alternate fixed-topology AD optimization with discrete triangle remeshing."""
    if cycles <= 0:
        raise ValueError("cycles must be positive")

    pts = coerce_runtime_points(points)
    elems = jnp.asarray(elements, dtype=jnp.int32)
    phases: list[RestartPhase] = []
    remesh_histories: list[list[AdaptiveHistory]] = []
    current_mask = fit_node_mask(movable_mask, pts.shape[0])

    for cycle in range(cycles):
        topo = triangle_topology_from_elements(elems, n_nodes=pts.shape[0])
        value_and_grad = build_quality_value_and_grad(topo)
        start_energy = float(value_and_grad(pts)[0])
        pts, losses = optimize_points_fixed_topology(
            pts,
            topo,
            steps=optimization_steps,
            step_size=optimization_step_size,
            movable_mask=current_mask,
        )
        final_energy = float(losses[-1]) if losses.size else start_energy
        remeshed = cycle < cycles - 1

        if remeshed:
            buffer, history = adaptive_remesh_tri(
                pts,
                elems,
                max_nodes=max_nodes,
                max_elements=max_elements,
                max_iters=remesh_max_iters,
                target_area=target_area,
                target_mean_icn=target_mean_icn,
                smoothing_alpha=smoothing_alpha,
                smoothing_steps=smoothing_steps,
                movable_mask=current_mask,
            )
            remesh_histories.append(history)
            pts = active_points(buffer)
            elems = active_elements(buffer)
            current_mask = fit_node_mask(current_mask, pts.shape[0])
        else:
            remesh_histories.append([])

        phases.append(
            RestartPhase(
                cycle=cycle,
                start_energy=start_energy,
                final_energy=final_energy,
                n_nodes=int(pts.shape[0]),
                n_elements=int(elems.shape[0]),
                remeshed=remeshed,
            )
        )

    return RestartTriOptimizationResult(points=pts, elements=elems, phases=phases, remesh_histories=remesh_histories)


def optimize_remesh_restart_quad(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    cycles: int = 2,
    optimization_steps: int = 60,
    optimization_step_size: float = 0.03,
    max_nodes: int,
    max_elements: int,
    remesh_max_iters: int = 4,
    target_area: float = 2.0e-3,
    target_mean_icn: float = 0.55,
    smoothing_alpha: float = 0.15,
    smoothing_steps: int = 2,
    movable_mask: jnp.ndarray | None = None,
) -> RestartQuadOptimizationResult:
    if cycles <= 0:
        raise ValueError("cycles must be positive")

    pts = coerce_runtime_points(points)
    elems = jnp.asarray(elements, dtype=jnp.int32)
    phases: list[RestartPhase] = []
    remesh_histories: list[list[QuadAdaptiveHistory]] = []
    current_mask = fit_node_mask(movable_mask, pts.shape[0])

    for cycle in range(cycles):
        topo = quad_topology_from_elements(elems, n_nodes=pts.shape[0])
        value_and_grad = build_quality_value_and_grad(topo)
        start_energy = float(value_and_grad(pts)[0])
        pts, losses = optimize_points_fixed_topology(
            pts,
            topo,
            steps=optimization_steps,
            step_size=optimization_step_size,
            movable_mask=current_mask,
        )
        final_energy = float(losses[-1]) if losses.size else start_energy
        remeshed = cycle < cycles - 1

        if remeshed:
            buffer, history = adaptive_remesh_quad(
                pts,
                elems,
                max_nodes=max_nodes,
                max_elements=max_elements,
                max_iters=remesh_max_iters,
                target_area=target_area,
                target_mean_icn=target_mean_icn,
                smoothing_alpha=smoothing_alpha,
                smoothing_steps=smoothing_steps,
                movable_mask=current_mask,
            )
            remesh_histories.append(history)
            pts = active_quad_points(buffer)
            elems = active_quad_elements(buffer)
            current_mask = fit_node_mask(current_mask, pts.shape[0])
        else:
            remesh_histories.append([])

        phases.append(RestartPhase(cycle, start_energy, final_energy, int(pts.shape[0]), int(elems.shape[0]), remeshed))

    return RestartQuadOptimizationResult(points=pts, elements=elems, phases=phases, remesh_histories=remesh_histories)


def optimize_remesh_restart_tet(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    cycles: int = 2,
    optimization_steps: int = 50,
    optimization_step_size: float = 0.02,
    max_nodes: int,
    max_elements: int,
    remesh_max_iters: int = 3,
    target_volume: float = 8.0e-4,
    target_mean_icn: float = 0.35,
    smoothing_alpha: float = 0.12,
    smoothing_steps: int = 2,
    movable_mask: jnp.ndarray | None = None,
) -> RestartTetOptimizationResult:
    if cycles <= 0:
        raise ValueError("cycles must be positive")

    pts = coerce_runtime_points(points)
    elems = jnp.asarray(elements, dtype=jnp.int32)
    phases: list[RestartPhase] = []
    remesh_histories: list[list[TetAdaptiveHistory]] = []
    current_mask = fit_node_mask(movable_mask, pts.shape[0])

    for cycle in range(cycles):
        topo = tet_topology_from_elements(elems, n_nodes=pts.shape[0])
        value_and_grad = build_quality_value_and_grad(topo)
        start_energy = float(value_and_grad(pts)[0])
        pts, losses = optimize_points_fixed_topology(
            pts,
            topo,
            steps=optimization_steps,
            step_size=optimization_step_size,
            movable_mask=current_mask,
        )
        final_energy = float(losses[-1]) if losses.size else start_energy
        remeshed = cycle < cycles - 1

        if remeshed:
            buffer, history = adaptive_remesh_tet(
                pts,
                elems,
                max_nodes=max_nodes,
                max_elements=max_elements,
                max_iters=remesh_max_iters,
                target_volume=target_volume,
                target_mean_icn=target_mean_icn,
                smoothing_alpha=smoothing_alpha,
                smoothing_steps=smoothing_steps,
                movable_mask=current_mask,
            )
            remesh_histories.append(history)
            pts = active_tet_points(buffer)
            elems = active_tet_elements(buffer)
            current_mask = fit_node_mask(current_mask, pts.shape[0])
        else:
            remesh_histories.append([])

        phases.append(RestartPhase(cycle, start_energy, final_energy, int(pts.shape[0]), int(elems.shape[0]), remeshed))

    return RestartTetOptimizationResult(points=pts, elements=elems, phases=phases, remesh_histories=remesh_histories)


def restart_result_topology(result: RestartOptimizationResult) -> MeshTopology:
    """Construct a final fixed-topology view from a restart result."""
    points = jnp.asarray(result.points)
    elements = jnp.asarray(result.elements, dtype=jnp.int32)
    order = int(elements.shape[1])
    dim = int(points.shape[1])
    if order == 3:
        return triangle_topology_from_elements(elements, n_nodes=points.shape[0])
    if order == 4 and dim == 2:
        return quad_topology_from_elements(elements, n_nodes=points.shape[0])
    if order == 4 and dim == 3:
        return tet_topology_from_elements(elements, n_nodes=points.shape[0])
    raise ValueError("Unsupported restart result topology")


def summarize_mode2_restart_result(result: RestartOptimizationResult) -> dict[str, float | int]:
    """Return a compact scalar summary of a restart workflow."""
    topology = restart_result_topology(result)
    final_metrics = mesh_topology_metrics(result.points, topology)
    remesh_count = int(sum(1 for phase in result.phases if phase.remeshed))
    return {
        "n_cycles": len(result.phases),
        "remesh_count": remesh_count,
        "final_energy": float(result.phases[-1].final_energy),
        "start_energy": float(result.phases[0].start_energy),
        "n_nodes": int(result.points.shape[0]),
        "n_elements": int(result.elements.shape[0]),
        **final_metrics,
    }


def _phase_history_arrays(phases: list[RestartPhase]) -> dict[str, np.ndarray]:
    return {
        "cycle": np.asarray([phase.cycle for phase in phases], dtype=np.int32),
        "start_energy": np.asarray([phase.start_energy for phase in phases], dtype=float),
        "final_energy": np.asarray([phase.final_energy for phase in phases], dtype=float),
        "n_nodes": np.asarray([phase.n_nodes for phase in phases], dtype=np.int32),
        "n_elements": np.asarray([phase.n_elements for phase in phases], dtype=np.int32),
        "remeshed": np.asarray([phase.remeshed for phase in phases], dtype=bool),
    }


def _history_dicts(history_group: list[AdaptiveHistory] | list[QuadAdaptiveHistory] | list[TetAdaptiveHistory]) -> list[dict[str, float | int | bool]]:
    return [{field: getattr(entry, field) for field in entry._fields} for entry in history_group]


def export_mode2_artifacts(
    output_dir: str | Path,
    result: RestartOptimizationResult,
    *,
    prefix: str = "mode2",
    export_stl_surface: bool = False,
    extra_element_blocks: tuple[GmshElementBlock, ...] | None = None,
    physical_names: dict[tuple[int, int], str] | None = None,
) -> dict[str, Path]:
    """Export final Mode 2 mesh artifacts and restart-phase diagnostics."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topology = restart_result_topology(result)
    summary = summarize_mode2_restart_result(result)
    summary["metadata_preserved"] = int(bool(extra_element_blocks or physical_names))

    snap_path = out_dir / f"{prefix}_final_snapshot.npz"
    json_path = out_dir / f"{prefix}_final_metrics.json"
    msh_path = out_dir / f"{prefix}_final_mesh.msh"
    hist_path = out_dir / f"{prefix}_phase_history.npz"
    phase_json_path = out_dir / f"{prefix}_phases.json"
    remesh_json_path = out_dir / f"{prefix}_remesh_history.json"
    stl_path = out_dir / f"{prefix}_final_surface.stl"

    export_snapshot_npz(snap_path, result.points, result.elements, metrics=summary)
    export_metrics_json(json_path, summary)
    export_gmsh_msh(
        msh_path,
        result.points,
        result.elements,
        element_entity_tags=topology.element_entity_tags,
        extra_element_blocks=extra_element_blocks,
        physical_names=physical_names,
    )
    np.savez(hist_path, **_phase_history_arrays(result.phases))
    with phase_json_path.open("w", encoding="utf-8") as f:
        json.dump([{field: getattr(phase, field) for field in phase._fields} for phase in result.phases], f, indent=2)
    with remesh_json_path.open("w", encoding="utf-8") as f:
        json.dump([_history_dicts(group) for group in result.remesh_histories], f, indent=2)

    artifacts = {
        "snapshot": snap_path,
        "metrics": json_path,
        "mesh": msh_path,
        "history": hist_path,
        "phases": phase_json_path,
        "remesh_history": remesh_json_path,
    }
    if export_stl_surface:
        export_binary_stl(stl_path, result.points, result.elements)
        artifacts["stl"] = stl_path
    return artifacts
