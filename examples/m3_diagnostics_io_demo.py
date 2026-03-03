"""Diagnostics + IO demo for quad workflow snapshots."""

from __future__ import annotations

from pathlib import Path

from gmshjax.io.exports import export_metrics_json, load_snapshot_npz
from gmshjax.mesh.adaptive_quad import adaptive_remesh_quad
from gmshjax.mesh.diagnostics import quad_diagnostics
from gmshjax.mesh.topology import unit_square_quad_mesh


def main() -> None:
    topo, pts = unit_square_quad_mesh(14, 10)
    out_dir = Path("results") / "m3_diag_io_demo"
    buf, hist = adaptive_remesh_quad(
        pts,
        topo.elements,
        max_nodes=5000,
        max_elements=9000,
        max_iters=3,
        target_area=0.01,
        target_mean_icn=0.45,
        snapshot_dir=out_dir,
    )
    stats = quad_diagnostics(buf.points[: buf.node_count], buf.elements[: buf.element_count])
    export_metrics_json(out_dir / "final_metrics.json", stats)
    first = load_snapshot_npz(out_dir / "iter_0000.npz")
    print("history_len:", len(hist))
    print("first_snapshot_points:", first["points"].shape)
    print("final_mean_icn:", stats["mean_icn"])


if __name__ == "__main__":
    main()
