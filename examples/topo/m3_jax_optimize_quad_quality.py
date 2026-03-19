"""JAX optimization demo for 2D quad mesh quality."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from topojax.mesh.operators import quad_icn
from topojax.mesh.topology import unit_square_quad_mesh


def main() -> None:
    topo, points = unit_square_quad_mesh(28, 20)
    # Introduce a smooth distortion to optimize away.
    x = points[:, 0]
    y = points[:, 1]
    points = points.at[:, 1].set(y + 0.10 * jnp.sin(2.0 * jnp.pi * x))

    def loss_fn(p: jnp.ndarray) -> jnp.ndarray:
        q = quad_icn(p, topo.elements)
        return jnp.mean((0.8 - q) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    p = points
    for _ in range(80):
        val, grad = value_and_grad(p)
        p = p - 0.03 * grad

    q0 = quad_icn(points, topo.elements)
    q1 = quad_icn(p, topo.elements)
    print("initial_mean_icn:", float(jnp.mean(q0)))
    print("final_mean_icn:", float(jnp.mean(q1)))
    print("final_loss:", float(val))


if __name__ == "__main__":
    main()
