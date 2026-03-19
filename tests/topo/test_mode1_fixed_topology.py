from pathlib import Path
import sys
import types

import jax.numpy as jnp
import pytest

from topojax.ad.mode1 import (
    benchmark_mode1_fixed_topology,
    build_mode1_optimizer,
    collect_mode1_jax_diagnostics,
    export_mode1_artifacts,
    optimize_mode1_fixed_topology,
    summarize_mode1_result,
)
from topojax.ad.compiled import build_quality_value_and_grad
from topojax.mesh.topology import polyline_mesh, unit_cube_tet_mesh, unit_square_quad_mesh, unit_square_tri_mesh
from topojax.runtime import get_runtime_precision, set_runtime_precision
from topojax.visualization import build_mode1_viper_payload, build_pyvista_dataset, plot_mode1_matplotlib, plot_mode1_pyvista, plot_mode1_viper


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


def test_build_mode1_optimizer_cache_stable_for_same_shapes() -> None:
    topo, points = unit_square_tri_mesh(8, 6)
    optimizer = build_mode1_optimizer(topo, steps=4, step_size=0.02)
    _ = optimizer(points)
    _ = optimizer(_distort(points))
    if hasattr(optimizer, "_cache_size"):
        assert optimizer._cache_size() == 1


def test_mode1_compiled_builders_reuse_cached_runs_for_same_topology() -> None:
    topo, _ = unit_square_tri_mesh(8, 6)
    value_and_grad_a = build_quality_value_and_grad(topo)
    value_and_grad_b = build_quality_value_and_grad(topo)
    optimizer_a = build_mode1_optimizer(topo, steps=4, step_size=0.02)
    optimizer_b = build_mode1_optimizer(topo, steps=4, step_size=0.05)
    assert value_and_grad_a is value_and_grad_b
    assert optimizer_a._compiled_run is optimizer_b._compiled_run


def test_collect_mode1_jax_diagnostics_reports_compiled_cache_state() -> None:
    topo, points = unit_square_tri_mesh(8, 6)
    diagnostics = collect_mode1_jax_diagnostics(points, topo, steps=4, step_size=0.02)
    assert diagnostics.runtime_precision == get_runtime_precision()
    assert diagnostics.point_shape == tuple(int(v) for v in points.shape)
    assert diagnostics.element_shape == tuple(int(v) for v in topo.elements.shape)
    assert diagnostics.point_dim == 2
    assert diagnostics.element_order == 3
    assert diagnostics.point_dtype in {"float32", "float64"}
    if diagnostics.value_and_grad_cache_size is not None:
        assert diagnostics.value_and_grad_cache_size >= 1
    if diagnostics.optimizer_cache_size is not None:
        assert diagnostics.optimizer_cache_size >= 1


def test_mode1_matplotlib_visualization_backend() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    topo, points = unit_square_tri_mesh(8, 6)
    result = optimize_mode1_fixed_topology(_distort(points), topo, steps=4, step_size=0.02, diagnostics_every=2)

    fig = plot_mode1_matplotlib(result.points, topo)
    assert fig is not None
    fig.canvas.draw()


def test_mode1_pyvista_dataset_supports_line_tri_and_tet(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakePolyData:
        def __init__(self, points, faces=None, lines=None):
            self.points = points
            self.faces = faces
            self.lines = lines
            self.n_points = int(points.shape[0])

    class _FakeUnstructuredGrid:
        def __init__(self, cells, celltypes, points):
            self.cells = cells
            self.celltypes = celltypes
            self.points = points
            self.n_points = int(points.shape[0])

    class _FakePlotter:
        def __init__(self, off_screen):
            self.off_screen = off_screen
            self.meshes = []
            self.titles = []

        def add_mesh(self, dataset, **kwargs):
            self.meshes.append((dataset, kwargs))

        def add_title(self, title):
            self.titles.append(title)

        def close(self):
            return None

    fake_pv = types.SimpleNamespace(
        PolyData=_FakePolyData,
        UnstructuredGrid=_FakeUnstructuredGrid,
        CellType=types.SimpleNamespace(TETRA=10),
        Plotter=_FakePlotter,
    )
    monkeypatch.setitem(sys.modules, "pyvista", fake_pv)

    line_topo, line_points = polyline_mesh(jnp.asarray([[0.0, 0.0], [0.4, 0.1], [1.0, 0.0]], dtype=jnp.float32))
    tri_topo, tri_points = unit_square_tri_mesh(4, 3)
    tet_topo, tet_points = unit_cube_tet_mesh(2, 2, 2)

    line_dataset = build_pyvista_dataset(line_points, line_topo)
    assert line_dataset.n_points == line_topo.n_nodes
    assert line_dataset.lines is not None

    tri_dataset = build_pyvista_dataset(tri_points, tri_topo)
    assert tri_dataset.n_points == tri_topo.n_nodes
    assert tri_dataset.faces is not None

    tet_dataset = build_pyvista_dataset(tet_points, tet_topo)
    assert tet_dataset.n_points == tet_topo.n_nodes
    assert tet_dataset.celltypes.shape[0] == tet_topo.elements.shape[0]

    plotter = plot_mode1_pyvista(line_points, line_topo, show=False)
    assert plotter.off_screen is True
    assert len(plotter.meshes) == 1
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


def test_mode1_viper_dispatches_to_plot_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    topo, points = unit_square_tri_mesh(4, 3)
    captured: dict[str, object] = {}

    def _plot_mesh(points_arg, elements_arg, *, title):
        captured["points"] = points_arg
        captured["elements"] = elements_arg
        captured["title"] = title
        return "ok"

    monkeypatch.setitem(sys.modules, "viper", types.SimpleNamespace(plot_mesh=_plot_mesh))
    result = plot_mode1_viper(points, topo, title="Tri Mesh")
    assert result == "ok"
    assert captured["title"] == "Tri Mesh"
    assert len(captured["points"]) == topo.n_nodes
    assert len(captured["elements"]) == topo.elements.shape[0]


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
