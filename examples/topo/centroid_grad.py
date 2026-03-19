"""Example: compile once and optimize a fixed-topology moving mesh."""

import jax.numpy as jnp

from topojax.ad.pipeline import build_model_parametric_quality_value_and_grad, default_param_vector
from topojax.mesh.factory import make_unit_square_model


model = make_unit_square_model(16, 16)
value_and_grad = build_model_parametric_quality_value_and_grad(model)
theta = default_param_vector()

# Same compiled function call shape each step -> no recompilation.
for _ in range(20):
    value, grad = value_and_grad(theta)
    theta = theta - 0.2 * grad

print("final energy:", value)
print("grad norm:", jnp.linalg.norm(grad))
