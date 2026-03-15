"""Mode 1 helpers: fixed-topology AD optimization, diagnostics, IO, and benchmarks."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gmshjax.ad.compiled import build_quality_value_and_grad
from gmshjax.io.exports import GmshElementBlock, export_gmsh_msh, export_metrics_json, export_snapshot_npz
from gmshjax.mesh.diagnostics import line_diagnostics, quad_diagnostics, tet_diagnostics, tri_diagnostics
from gmshjax.mesh.topology import MeshTopology
from gmshjax.runtime import jax_float_dtype


class Mode1StepDiagnostics(NamedTuple):
    step: int
    energy: float
    grad_norm: float
    metrics: dict[str, float | int]


class Mode1OptimizationResult(NamedTuple):
    points: jnp.ndarray
    topology: MeshTopology
    energy_history: jnp.ndarray
    grad_norm_history: jnp.ndarray
    step_diagnostics: tuple[Mode1StepDiagnostics, ...]


class Mode1BenchmarkResult(NamedTuple):
    first_call_ms: float
    steady_state_ms_per_step: float
    final_energy: float
    final_grad_norm: float
    steps: int


def _mask_like(points: jnp.ndarray, movable_mask: jnp.ndarray | None) -> jnp.ndarray:
    if movable_mask is None:
        return jnp.ones((points.shape[0],), dtype=points.dtype)
    mask = jnp.asarray(movable_mask, dtype=bool)
    return mask.astype(points.dtype)


def _coerce_mode1_points(points: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(points, dtype=jax_float_dtype())


def _build_mode1_scan(
    topology: MeshTopology,
    *,
    steps: int,
    step_size: float,
    movable_mask: jnp.ndarray | None,
    record_points: bool,
):
    value_and_grad = build_quality_value_and_grad(topology)

    @jax.jit
    def run(points: jnp.ndarray):
        pts0 = _coerce_mode1_points(points)
        mask = _mask_like(pts0, movable_mask)
        step_size_local = jnp.asarray(step_size, dtype=pts0.dtype)

        def body(pts: jnp.ndarray, _):
            value, grad = value_and_grad(pts)
            masked_grad = grad * mask[:, None]
            next_pts = pts - step_size_local * masked_grad
            if record_points:
                return next_pts, (next_pts, value, jnp.linalg.norm(masked_grad))
            return next_pts, (value, jnp.linalg.norm(masked_grad))

        final_points, outputs = jax.lax.scan(body, pts0, xs=None, length=steps)
        return final_points, outputs

    return run


def _diagnostic_step_indices(steps: int, diagnostics_every: int) -> tuple[int, ...]:
    if steps <= 0 or diagnostics_every <= 0:
        return ()
    indices = {0, steps - 1}
    indices.update(step - 1 for step in range(diagnostics_every, steps + 1, diagnostics_every))
    return tuple(sorted(index for index in indices if 0 <= index < steps))


def build_mode1_optimizer(
    topology: MeshTopology,
    *,
    steps: int = 80,
    step_size: float = 0.03,
    movable_mask: jnp.ndarray | None = None,
):
    """Build a pure-JAX fixed-topology optimizer for mode 1.

    The returned callable is jitted and executes the optimization loop through
    `jax.lax.scan`, so the hot path contains no Python step loop.
    """
    run = _build_mode1_scan(
        topology,
        steps=steps,
        step_size=step_size,
        movable_mask=movable_mask,
        record_points=False,
    )

    def wrapped(points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        final_points, outputs = run(points)
        energy_history, grad_norm_history = outputs
        return final_points, energy_history, grad_norm_history

    return wrapped


def _topology_metrics(points: jnp.ndarray, topology: MeshTopology) -> dict[str, float | int]:
    order = int(topology.elements.shape[1])
    dim = int(points.shape[1])
    if order == 2:
        return line_diagnostics(points, topology.elements)
    if order == 3:
        return tri_diagnostics(points, topology.elements)
    if order == 4 and dim == 2:
        return quad_diagnostics(points, topology.elements)
    if order == 4 and dim == 3:
        return tet_diagnostics(points, topology.elements)
    raise ValueError("Unsupported topology for mode 1 diagnostics")


def optimize_mode1_fixed_topology(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    steps: int = 80,
    step_size: float = 0.03,
    movable_mask: jnp.ndarray | None = None,
    diagnostics_every: int = 10,
) -> Mode1OptimizationResult:
    """Run fixed-topology AD optimization with diagnostics collection."""
    optimizer = build_mode1_optimizer(topology, steps=steps, step_size=step_size, movable_mask=movable_mask)
    pts0 = _coerce_mode1_points(points)
    pts, energy_history, grad_norm_history = optimizer(pts0)
    step_diagnostics: list[Mode1StepDiagnostics] = []

    if diagnostics_every > 0:
        history_optimizer = _build_mode1_scan(
            topology,
            steps=steps,
            step_size=step_size,
            movable_mask=movable_mask,
            record_points=True,
        )
        _, outputs = history_optimizer(pts0)
        points_history, _, _ = outputs
        for step in _diagnostic_step_indices(steps, diagnostics_every):
            snap = points_history[step]
            metrics = _topology_metrics(snap, topology)
            step_diagnostics.append(
                Mode1StepDiagnostics(
                    step=step,
                    energy=float(energy_history[step]),
                    grad_norm=float(grad_norm_history[step]),
                    metrics=metrics,
                )
            )

    return Mode1OptimizationResult(
        points=pts,
        topology=topology,
        energy_history=energy_history,
        grad_norm_history=grad_norm_history,
        step_diagnostics=tuple(step_diagnostics),
    )


def export_mode1_artifacts(
    output_dir: str | Path,
    result: Mode1OptimizationResult,
    *,
    prefix: str = "mode1",
    extra_element_blocks: tuple[GmshElementBlock, ...] | None = None,
    physical_names: dict[tuple[int, int], str] | None = None,
) -> dict[str, Path]:
    """Export final mode-1 snapshot, metrics, and mesh artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_metrics = _topology_metrics(result.points, result.topology)
    final_metrics = {
        **final_metrics,
        "final_energy": float(result.energy_history[-1]),
        "final_grad_norm": float(result.grad_norm_history[-1]),
        "n_steps": int(result.energy_history.shape[0]),
    }
    snap_path = out_dir / f"{prefix}_final_snapshot.npz"
    json_path = out_dir / f"{prefix}_final_metrics.json"
    msh_path = out_dir / f"{prefix}_final_mesh.msh"
    hist_path = out_dir / f"{prefix}_history.npz"

    export_snapshot_npz(snap_path, result.points, result.topology.elements, metrics=final_metrics)
    export_metrics_json(json_path, final_metrics)
    export_gmsh_msh(
        msh_path,
        result.points,
        result.topology.elements,
        element_entity_tags=result.topology.element_entity_tags,
        extra_element_blocks=extra_element_blocks,
        physical_names=physical_names,
    )
    np.savez(
        hist_path,
        energy_history=np.asarray(result.energy_history),
        grad_norm_history=np.asarray(result.grad_norm_history),
    )
    return {
        "snapshot": snap_path,
        "metrics": json_path,
        "mesh": msh_path,
        "history": hist_path,
    }


def benchmark_mode1_fixed_topology(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    steps: int = 50,
    step_size: float = 0.03,
    movable_mask: jnp.ndarray | None = None,
) -> Mode1BenchmarkResult:
    """Benchmark compile cost and steady-state cost for mode 1."""
    pts = _coerce_mode1_points(points)
    optimizer = build_mode1_optimizer(topology, steps=steps, step_size=step_size, movable_mask=movable_mask)

    t0 = perf_counter()
    _, energy_history, grad_norm_history = optimizer(pts)
    _ = jax.block_until_ready((energy_history, grad_norm_history))
    t1 = perf_counter()

    t2 = perf_counter()
    _, energy_history_steady, grad_norm_history_steady = optimizer(pts)
    _ = jax.block_until_ready((energy_history_steady, grad_norm_history_steady))
    t3 = perf_counter()

    return Mode1BenchmarkResult(
        first_call_ms=(t1 - t0) * 1.0e3,
        steady_state_ms_per_step=((t3 - t2) / max(steps, 1)) * 1.0e3,
        final_energy=float(energy_history_steady[-1]),
        final_grad_norm=float(grad_norm_history_steady[-1]),
        steps=steps,
    )


def summarize_mode1_result(result: Mode1OptimizationResult) -> dict[str, Any]:
    """Return a compact scalar summary of a mode-1 optimization run."""
    final_metrics = _topology_metrics(result.points, result.topology)
    return {
        "final_energy": float(result.energy_history[-1]),
        "final_grad_norm": float(result.grad_norm_history[-1]),
        "steps": int(result.energy_history.shape[0]),
        **final_metrics,
    }