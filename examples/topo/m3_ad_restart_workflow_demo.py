"""Alternate fixed-topology AD optimization with discrete remeshing and restart."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from topojax.ad.restart import optimize_remesh_restart_tri
from topojax.io.exports import export_snapshot_npz
from topojax.mesh.topology import unit_square_tri_mesh


def main() -> None:
    topo, points = unit_square_tri_mesh(18, 14)
    x = points[:, 0]
    y = points[:, 1]
    distorted = points.at[:, 1].set(y + 0.10 * jnp.sin(2.0 * jnp.pi * x) * x * (1.0 - x))

    result = optimize_remesh_restart_tri(
        distorted,
        topo.elements,
        cycles=3,
        optimization_steps=50,
        optimization_step_size=0.025,
        max_nodes=5000,
        max_elements=10000,
        remesh_max_iters=3,
        target_area=0.0025,
        target_mean_icn=0.55,
        smoothing_alpha=0.15,
        smoothing_steps=2,
    )

    out_dir = Path("results") / "m3_ad_restart_workflow_demo"
    export_snapshot_npz(out_dir / "final_state.npz", result.points, result.elements)

    for phase in result.phases:
        print(
            "cycle:",
            phase.cycle,
            "start_energy:",
            phase.start_energy,
            "final_energy:",
            phase.final_energy,
            "n_nodes:",
            phase.n_nodes,
            "n_elements:",
            phase.n_elements,
            "remeshed:",
            phase.remeshed,
        )
    print("output:", str(out_dir / "final_state.npz"))


if __name__ == "__main__":
    main()