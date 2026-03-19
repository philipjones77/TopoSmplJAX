"""Mode 1 fixed-topology demo with diagnostics, IO export, and visualization backends."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from topojax.ad.mode1 import benchmark_mode1_fixed_topology, export_mode1_artifacts, optimize_mode1_fixed_topology, summarize_mode1_result
from topojax.mesh.topology import unit_square_tri_mesh
from topojax.visualization import build_mode1_viper_payload, plot_mode1_matplotlib, plot_mode1_pyvista, plot_mode1_viper


def main() -> None:
    topo, points = unit_square_tri_mesh(20, 16)
    x = points[:, 0]
    y = points[:, 1]
    distorted = points.at[:, 1].set(y + 0.10 * jnp.sin(2.0 * jnp.pi * x) * x * (1.0 - x))

    result = optimize_mode1_fixed_topology(distorted, topo, steps=60, step_size=0.025, diagnostics_every=15)
    bench = benchmark_mode1_fixed_topology(distorted, topo, steps=30)
    out_dir = Path("results") / "m1_fixed_topology_mode_demo"
    artifacts = export_mode1_artifacts(out_dir, result)

    fig = plot_mode1_matplotlib(result.points, topo, title="Mode 1 Matplotlib")
    fig.savefig(out_dir / "mode1_matplotlib.png", dpi=150, bbox_inches="tight")

    plotter = plot_mode1_pyvista(result.points, topo, title="Mode 1 PyVista", show=False)
    plotter.screenshot(str(out_dir / "mode1_pyvista.png"))
    plotter.close()

    payload = build_mode1_viper_payload(result.points, topo, title="Mode 1 Viper")
    (out_dir / "mode1_viper_payload.json").write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")

    try:
        plot_mode1_viper(result.points, topo, title="Mode 1 Viper")
        viper_status = "invoked"
    except Exception as exc:
        viper_status = str(exc)

    print("summary:", summarize_mode1_result(result))
    print("benchmark_first_call_ms:", bench.first_call_ms)
    print("benchmark_steady_state_ms_per_step:", bench.steady_state_ms_per_step)
    print("artifacts:", {k: str(v) for k, v in artifacts.items()})
    print("viper_status:", viper_status)


if __name__ == "__main__":
    main()
