import jax
import jax.numpy as jnp

from gmshjax.ad.restart import optimize_remesh_restart_quad, optimize_remesh_restart_tet, quad_topology_from_elements, tet_topology_from_elements
from gmshjax.ad.straight_through import straight_through_quad_connectivity_energy, straight_through_quad_diagonal_weights
from gmshjax.ad.surrogate import soft_quad_connectivity_energy, soft_quad_diagonal_weights
from gmshjax.mesh.operators import quad_mesh_quality_energy, tet_mesh_quality_energy
from gmshjax.mesh.topology import unit_cube_tet_mesh, unit_square_quad_mesh


def test_quad_fixed_topology_energy_and_restart_workflow() -> None:
    topo, points = unit_square_quad_mesh(8, 6)
    x = points[:, 0]
    y = points[:, 1]
    distorted = points.at[:, 1].set(y + 0.08 * jnp.sin(2.0 * jnp.pi * x) * x * (1.0 - x))
    initial = float(quad_mesh_quality_energy(distorted, topo))

    result = optimize_remesh_restart_quad(
        distorted,
        topo.elements,
        cycles=2,
        optimization_steps=20,
        optimization_step_size=0.02,
        max_nodes=2000,
        max_elements=2000,
        remesh_max_iters=2,
        target_area=0.03,
        target_mean_icn=0.55,
        smoothing_alpha=0.15,
        smoothing_steps=1,
    )

    final_topo = quad_topology_from_elements(result.elements, n_nodes=result.points.shape[0])
    final = float(quad_mesh_quality_energy(result.points, final_topo))
    assert len(result.phases) == 2
    assert result.phases[0].final_energy <= result.phases[0].start_energy + 1.0e-8
    assert final < initial


def test_tet_fixed_topology_energy_and_restart_workflow() -> None:
    topo, points = unit_cube_tet_mesh(3, 3, 3)
    distorted = points.at[:, 2].set(points[:, 2] + 0.05 * points[:, 0] * (1.0 - points[:, 0]))
    initial = float(tet_mesh_quality_energy(distorted, topo))

    result = optimize_remesh_restart_tet(
        distorted,
        topo.elements,
        cycles=2,
        optimization_steps=15,
        optimization_step_size=0.015,
        max_nodes=4000,
        max_elements=4000,
        remesh_max_iters=1,
        target_volume=0.03,
        target_mean_icn=0.25,
        smoothing_alpha=0.1,
        smoothing_steps=1,
    )

    final_topo = tet_topology_from_elements(result.elements, n_nodes=result.points.shape[0])
    final = float(tet_mesh_quality_energy(result.points, final_topo))
    assert len(result.phases) == 2
    assert result.phases[0].final_energy <= result.phases[0].start_energy + 1.0e-8
    assert final < initial


def test_soft_quad_connectivity_surrogate_is_differentiable() -> None:
    topo, points = unit_square_quad_mesh(6, 5)
    logits = jnp.zeros((topo.elements.shape[0],), dtype=points.dtype)

    def objective(theta: jnp.ndarray) -> jnp.ndarray:
        return soft_quad_connectivity_energy(points, topo.elements, theta, temperature=0.3)

    grad = jax.grad(objective)(logits)
    weights = soft_quad_diagonal_weights(logits, temperature=0.3)
    assert grad.shape == logits.shape
    assert jnp.all(jnp.isfinite(grad))
    assert weights.shape == (topo.elements.shape[0], 2)
    assert jnp.allclose(jnp.sum(weights, axis=1), 1.0, atol=1.0e-6)


def test_straight_through_quad_connectivity_surrogate_has_gradients() -> None:
    topo, points = unit_square_quad_mesh(6, 5)
    logits = jnp.linspace(-0.5, 0.5, topo.elements.shape[0], dtype=points.dtype)

    def objective(theta: jnp.ndarray) -> jnp.ndarray:
        return straight_through_quad_connectivity_energy(points, topo.elements, theta, temperature=0.25)

    grad = jax.grad(objective)(logits)
    weights = straight_through_quad_diagonal_weights(logits, temperature=0.25)
    assert grad.shape == logits.shape
    assert jnp.all(jnp.isfinite(grad))
    assert weights.shape == (topo.elements.shape[0], 2)
    assert jnp.allclose(weights, jnp.round(weights), atol=1.0e-6)
    assert jnp.allclose(jnp.sum(weights, axis=1), 1.0, atol=1.0e-6)
