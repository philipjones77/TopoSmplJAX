import jax.numpy as jnp

from topojax.ad.restart import optimize_points_fixed_topology, optimize_remesh_restart_tri
from topojax.mesh.operators import mesh_quality_energy
from topojax.mesh.topology import unit_square_tri_mesh
from topojax.runtime import get_runtime_precision, set_runtime_precision


def test_optimize_points_fixed_topology_reduces_energy() -> None:
    topo, points = unit_square_tri_mesh(10, 8)
    x = points[:, 0]
    y = points[:, 1]
    distorted = points.at[:, 1].set(y + 0.08 * jnp.sin(2.0 * jnp.pi * x))
    initial = float(mesh_quality_energy(distorted, topo))
    optimized, losses = optimize_points_fixed_topology(distorted, topo, steps=30, step_size=0.02)
    final = float(mesh_quality_energy(optimized, topo))
    assert losses.size == 30
    assert final < initial


def test_optimize_remesh_restart_tri_runs_restart_workflow() -> None:
    topo, points = unit_square_tri_mesh(10, 8)
    x = points[:, 0]
    y = points[:, 1]
    distorted = points.at[:, 1].set(y + 0.10 * jnp.sin(2.0 * jnp.pi * x) * x * (1.0 - x))

    result = optimize_remesh_restart_tri(
        distorted,
        topo.elements,
        cycles=2,
        optimization_steps=20,
        optimization_step_size=0.02,
        max_nodes=2000,
        max_elements=4000,
        remesh_max_iters=2,
        target_area=0.008,
        target_mean_icn=0.55,
        smoothing_alpha=0.15,
        smoothing_steps=1,
    )

    assert len(result.phases) == 2
    assert result.phases[0].final_energy <= result.phases[0].start_energy + 1.0e-8
    assert result.phases[0].remeshed
    assert result.phases[1].final_energy <= result.phases[1].start_energy + 1.0e-8
    assert result.elements.shape[0] >= topo.elements.shape[0]
    assert result.points.shape[0] >= topo.n_nodes


def test_restart_optimizer_respects_runtime_float64_precision() -> None:
    original = get_runtime_precision()
    try:
        set_runtime_precision("float64")
        topo, points = unit_square_tri_mesh(8, 6)
        x = points[:, 0]
        distorted = points.at[:, 1].set(points[:, 1] + 0.05 * jnp.sin(2.0 * jnp.pi * x))
        optimized, losses = optimize_points_fixed_topology(distorted, topo, steps=5, step_size=0.02)
        assert optimized.dtype == jnp.float64
        assert losses.dtype == jnp.float64
    finally:
        set_runtime_precision(original)