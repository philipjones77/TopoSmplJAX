"""M3 completion demo: adaptive split/flip/smooth loop with snapshot export."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from topojax.mesh.adaptive import adaptive_remesh_tri
from topojax.mesh.boundary import BoundaryCurves2D, boundary_constrained_points, line_segment, sinusoidal_top_boundary
from topojax.mesh.topology import structured_triangles


def make_curves(nx: int, ny: int) -> BoundaryCurves2D:
    bottom = line_segment(jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0]), nx)
    top = sinusoidal_top_boundary(0.0, 1.0, 1.0, 0.10, nx)
    left = line_segment(bottom[0], top[0], ny)
    right = line_segment(bottom[-1], top[-1], ny)
    return BoundaryCurves2D(bottom=bottom, right=right, top=top, left=left)


def main() -> None:
    nx, ny = 28, 18
    curves = make_curves(nx, ny)
    points = boundary_constrained_points(curves)
    elements = structured_triangles(nx, ny)

    out_dir = Path("results") / "m3_adaptive_demo"
    buffer, history = adaptive_remesh_tri(
        points,
        elements,
        max_nodes=8000,
        max_elements=16000,
        max_iters=12,
        target_area=0.0015,
        target_mean_icn=0.45,
        smoothing_alpha=0.2,
        smoothing_steps=2,
        snapshot_dir=out_dir,
    )

    print("iterations:", len(history))
    print("final_nodes:", buffer.node_count)
    print("final_elements:", buffer.element_count)
    if history:
        last = history[-1]
        print("last_mean_icn:", last.mean_icn)
        print("last_max_area:", last.max_area)
    print("snapshots:", str(out_dir))


if __name__ == "__main__":
    main()
