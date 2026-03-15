from pathlib import Path

import matplotlib
import jax.numpy as jnp
import pyvista as pv

from gmshjax.ad.mode1 import benchmark_mode1_fixed_topology, export_mode1_artifacts, optimize_mode1_fixed_topology, summarize_mode1_result
from gmshjax.mesh.topology import unit_cube_tet_mesh, unit_square_quad_mesh, unit_square_tri_mesh
from gmshjax.runtime import get_runtime_precision, set_runtime_precision
from gmshjax.visualization import build_mode1_viper_payload, build_pyvista_dataset, plot_mode1_matplotlib, plot_mode1_pyvista, plot_mode1_viper

matplotlib.use("Agg")
pv.OFF_SCREEN = True


def _distort(points: jnp.ndarray) -> jnp.ndarray:
    if points.shape[1] == 2:
        x = points[:, 0]
        y = points[:, 1]
        return points.at[:, 1].set(y + 0.08 * jnp.sin(2.0 * jnp.pi * x) * x * (1.0 - x))
    x = points[:, 0]
    z = points[:, 2]
    return points.at[:, 2].set(z + 0.05 * x * (1.0 - x))


def test_mode1_optimization_collects_diagnostics_for_tri_quad_tet() -> None:
    for topo, points in [unit_square_tri_mesh(10, 8), unit_square_quad_mesh(10, 8), unit_cube_tet_mesh(3, 3, 3)]:
        result = optimize_mode1_fixed_topology(_distort(points), topo, steps=12, step_size=0.02, diagnostics_every=4)
        assert result.energy_history.shape == (12,)
        assert result.grad_norm_history.shape == (12,)
        assert len(result.step_diagnostics) >= 3
        assert float(result.energy_history[-1]) <= float(result.energy_history[0]) + 1.0e-8
        summary = summarize_mode1_result(result)
        assert summary["n_nodes"] == topo.n_nodes
        assert summary["n_elements"] == topo.elements.shape[0]


def test_mode1_export_writes_artifacts(tmp_path: Path) -> None:
    topo, points = unit_square_tri_mesh(8, 6)
    result = optimize_mode1_fixed_topology(_distort(points), topo, steps=8, step_size=0.02, diagnostics_every=4)
    artifacts = export_mode1_artifacts(tmp_path, result, prefix="tri")
    assert artifacts["snapshot"].exists()
    assert artifacts["metrics"].exists()
    assert artifacts["mesh"].exists()
    assert artifacts["history"].exists()


def test_mode1_benchmark_returns_positive_timings() -> None:
    topo, points = unit_square_tri_mesh(12, 10)
    result = benchmark_mode1_fixed_topology(_distort(points), topo, steps=5)
    assert result.first_call_ms > 0.0
    assert result.steady_state_ms_per_step > 0.0
    assert result.steps == 5


def test_mode1_matplotlib_and_pyvista_visualization_backends() -> None:
    topo, points = unit_square_tri_mesh(8, 6)
    result = optimize_mode1_fixed_topology(_distort(points), topo, steps=4, step_size=0.02, diagnostics_every=2)

    fig = plot_mode1_matplotlib(result.points, topo)
    assert fig is not None

    dataset = build_pyvista_dataset(result.points, topo)
    assert dataset.n_points == topo.n_nodes
    plotter = plot_mode1_pyvista(result.points, topo, show=False)
    assert plotter is not None
    plotter.close()


def test_mode1_viper_payload_and_missing_backend() -> None:
    topo, points = unit_square_quad_mesh(6, 5)
    payload = build_mode1_viper_payload(points, topo)
    assert payload["element_order"] == 4
    assert payload["point_dim"] == 2
    try:
        plot_mode1_viper(points, topo)
    except ModuleNotFoundError as exc:
        assert "viper is not installed" in str(exc)
    else:
        raise AssertionError("Expected missing viper backend in this environment")


def test_mode1_respects_runtime_float32_precision() -> None:
    original = get_runtime_precision()
    try:
        set_runtime_precision("float32")
        topo, points = unit_square_tri_mesh(8, 6)
        result = optimize_mode1_fixed_topology(_distort(points), topo, steps=4, step_size=0.02, diagnostics_every=2)
        bench = benchmark_mode1_fixed_topology(_distort(points), topo, steps=2)
        assert result.points.dtype == jnp.float32
        assert result.energy_history.dtype == jnp.float32
        assert isinstance(bench.final_energy, float)
    finally:
        set_runtime_precision(original)


def test_mode1_respects_runtime_float64_precision() -> None:
    original = get_runtime_precision()
    try:
        set_runtime_precision("float64")
        topo, points = unit_square_tri_mesh(8, 6)
        result = optimize_mode1_fixed_topology(_distort(points), topo, steps=4, step_size=0.02, diagnostics_every=2)
        bench = benchmark_mode1_fixed_topology(_distort(points), topo, steps=2)
        assert result.points.dtype == jnp.float64
        assert result.energy_history.dtype == jnp.float64
        assert isinstance(bench.final_grad_norm, float)
    finally:
        set_runtime_precision(original)
