"""Benchmark: JAX compile cost vs steady-state evaluation cost."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from topojax.ad.pipeline import build_model_parametric_quality_value_and_grad, default_param_vector
from topojax.mesh.factory import make_unit_square_model


def main() -> None:
    model = make_unit_square_model(64, 64)
    value_and_grad = build_model_parametric_quality_value_and_grad(model)
    theta = default_param_vector()

    # First call includes compilation.
    t0 = time.perf_counter()
    value, grad = value_and_grad(theta)
    _ = jax.block_until_ready((value, grad))
    t1 = time.perf_counter()

    # Steady-state calls reuse compiled executable.
    n_steps = 200
    t2 = time.perf_counter()
    for _ in range(n_steps):
        value, grad = value_and_grad(theta)
        theta = theta - 0.05 * grad
    _ = jax.block_until_ready((value, grad))
    t3 = time.perf_counter()

    first_ms = (t1 - t0) * 1e3
    steady_ms = ((t3 - t2) / n_steps) * 1e3

    print(f"first_call_ms={first_ms:.3f}")
    print(f"steady_state_ms_per_step={steady_ms:.3f}")
    print(f"final_value={float(value):.6e}")
    print(f"final_grad_norm={float(jnp.linalg.norm(grad)):.6e}")


if __name__ == "__main__":
    main()
