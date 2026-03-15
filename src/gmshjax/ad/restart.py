"""Restart-style workflows that alternate fixed-topology AD with remeshing."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from gmshjax.ad.compiled import build_quality_value_and_grad
from gmshjax.mesh.adaptive import AdaptiveHistory, adaptive_remesh_tri
from gmshjax.mesh.adaptive_quad import QuadAdaptiveHistory, adaptive_remesh_quad
from gmshjax.mesh.adaptive_tet import TetAdaptiveHistory, adaptive_remesh_tet
from gmshjax.mesh.mutation import active_elements, active_points
from gmshjax.mesh.mutation_qt import active_quad_elements, active_quad_points, active_tet_elements, active_tet_points
from gmshjax.mesh.topology import MeshTopology, quad_edges, tet_edges, triangle_edges
from gmshjax.runtime import jax_float_dtype


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


def _fit_movable_mask(movable_mask: jnp.ndarray | None, n_nodes: int) -> jnp.ndarray | None:
    if movable_mask is None:
        return None
    mask = jnp.asarray(movable_mask, dtype=bool)
    if mask.shape[0] >= n_nodes:
        return mask[:n_nodes]
    pad = jnp.ones((n_nodes - mask.shape[0],), dtype=bool)
    return jnp.concatenate([mask, pad], axis=0)


def _coerce_restart_points(points: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(points, dtype=jax_float_dtype())


def _build_fixed_topology_scan(
    topology: MeshTopology,
    *,
    steps: int,
    step_size: float,
    movable_mask: jnp.ndarray | None,
):
    value_and_grad = build_quality_value_and_grad(topology)

    @jax.jit
    def run(points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        pts0 = _coerce_restart_points(points)
        step_size_local = jnp.asarray(step_size, dtype=pts0.dtype)
        mask = _fit_movable_mask(movable_mask, pts0.shape[0])
        if mask is not None:
            mask = mask.astype(pts0.dtype)

        def body(pts: jnp.ndarray, _):
            value, grad = value_and_grad(pts)
            if mask is not None:
                grad = grad * mask[:, None]
            next_pts = pts - step_size_local * grad
            return next_pts, value

        final_points, losses = jax.lax.scan(body, pts0, xs=None, length=steps)
        return final_points, losses

    return run


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
    optimizer = _build_fixed_topology_scan(
        topology,
        steps=steps,
        step_size=step_size,
        movable_mask=movable_mask,
    )
    return optimizer(points)


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

    pts = _coerce_restart_points(points)
    elems = jnp.asarray(elements, dtype=jnp.int32)
    phases: list[RestartPhase] = []
    remesh_histories: list[list[AdaptiveHistory]] = []
    current_mask = _fit_movable_mask(movable_mask, pts.shape[0])

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
            current_mask = _fit_movable_mask(current_mask, pts.shape[0])
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

    pts = _coerce_restart_points(points)
    elems = jnp.asarray(elements, dtype=jnp.int32)
    phases: list[RestartPhase] = []
    remesh_histories: list[list[QuadAdaptiveHistory]] = []
    current_mask = _fit_movable_mask(movable_mask, pts.shape[0])

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
            current_mask = _fit_movable_mask(current_mask, pts.shape[0])
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

    pts = _coerce_restart_points(points)
    elems = jnp.asarray(elements, dtype=jnp.int32)
    phases: list[RestartPhase] = []
    remesh_histories: list[list[TetAdaptiveHistory]] = []
    current_mask = _fit_movable_mask(movable_mask, pts.shape[0])

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
            current_mask = _fit_movable_mask(current_mask, pts.shape[0])
        else:
            remesh_histories.append([])

        phases.append(RestartPhase(cycle, start_energy, final_energy, int(pts.shape[0]), int(elems.shape[0]), remeshed))

    return RestartTetOptimizationResult(points=pts, elements=elems, phases=phases, remesh_histories=remesh_histories)