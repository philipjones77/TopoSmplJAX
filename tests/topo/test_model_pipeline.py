import jax.numpy as jnp

from topojax.ad.pipeline import (
    build_model_parametric_quality_value_and_grad,
    default_param_vector,
)
from topojax.mesh.factory import make_unit_square_model
from topojax.mesh.manifold import DeformationParams, apply_deformation


def test_model_has_immutable_ids_and_entity_tags() -> None:
    model = make_unit_square_model(5, 4)
    topo = model.topology
    assert topo.node_ids.shape == (20,)
    assert topo.element_ids.shape[0] == topo.elements.shape[0]
    assert topo.element_entity_tags.shape[0] == topo.elements.shape[0]
    assert int(topo.node_ids[0]) == 0
    assert int(topo.element_ids[0]) == 0


def test_deformation_preserves_node_count_and_connectivity_shape() -> None:
    model = make_unit_square_model(8, 6)
    theta = default_param_vector()
    params = (
        theta.at[0].set(0.1)
        .at[1].set(-0.2)
        .at[4].set(0.05)
        .at[6].set(0.02)
    )
    deformed = apply_deformation(
        model.state.points,
        DeformationParams(
            translation=params[0:2],
            scale=params[2:4],
            shear=params[4:6],
            bend=params[6:8],
        ),
    )
    assert deformed.shape == model.state.points.shape
    assert model.topology.elements.shape[1] == 3


def test_parametric_pipeline_is_differentiable() -> None:
    model = make_unit_square_model(12, 10)
    value_and_grad = build_model_parametric_quality_value_and_grad(model)
    theta = default_param_vector()
    value, grad = value_and_grad(theta)
    assert jnp.isfinite(value)
    assert grad.shape == theta.shape
    assert jnp.all(jnp.isfinite(grad))


def test_jit_cache_stable_for_same_shapes() -> None:
    model = make_unit_square_model(12, 10)
    value_and_grad = build_model_parametric_quality_value_and_grad(model)
    theta0 = default_param_vector()
    theta1 = theta0.at[0].set(0.12).at[6].set(0.04)

    _ = value_and_grad(theta0)
    _ = value_and_grad(theta1)

    if hasattr(value_and_grad, "_cache_size"):
        assert value_and_grad._cache_size() == 1
