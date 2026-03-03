import numpy as np
import jax
import jax.numpy as jnp

from gmshjax.ad.pipeline import build_model_parametric_quality_value_and_grad, default_param_vector
from gmshjax.mesh.factory import make_unit_square_model
from gmshjax.mesh.topology import unit_cube_tet_mesh, unit_square_quad_mesh, unit_square_tri_mesh
from gmshjax.numpy_impl import unit_square_tri_mesh as np_unit_square_tri_mesh


def test_static_tri_generator_is_deterministic() -> None:
    topo_a, pts_a = unit_square_tri_mesh(16, 12)
    topo_b, pts_b = unit_square_tri_mesh(16, 12)
    assert np.array_equal(np.asarray(topo_a.elements), np.asarray(topo_b.elements))
    assert np.array_equal(np.asarray(topo_a.edges), np.asarray(topo_b.edges))
    assert np.allclose(np.asarray(pts_a), np.asarray(pts_b))


def test_static_quad_and_tet_counts() -> None:
    topo_q, pts_q = unit_square_quad_mesh(11, 7)
    topo_t, pts_t = unit_cube_tet_mesh(6, 5, 4)
    assert pts_q.shape == (77, 2)
    assert topo_q.elements.shape == ((11 - 1) * (7 - 1), 4)
    assert pts_t.shape == (6 * 5 * 4, 3)
    assert topo_t.elements.shape[0] == 5 * (6 - 1) * (5 - 1) * (4 - 1)
    assert topo_t.elements.shape[1] == 4


def test_numpy_and_jax_tri_topology_identical() -> None:
    topo_jax, _ = unit_square_tri_mesh(9, 8)
    topo_np, _ = np_unit_square_tri_mesh(9, 8, dtype=np.float32)
    assert np.array_equal(np.asarray(topo_jax.elements), topo_np.elements)
    assert np.array_equal(np.asarray(topo_jax.edges), topo_np.edges)


def test_jax_pipeline_dtype_is_jax_dtype() -> None:
    model = make_unit_square_model(10, 10)
    value_and_grad = build_model_parametric_quality_value_and_grad(model)
    theta = default_param_vector()
    value, grad = value_and_grad(theta)
    assert isinstance(value, jax.Array)
    assert isinstance(grad, jax.Array)
    assert grad.dtype == jnp.asarray(theta).dtype
