"""M3 demo for 2D quads: adaptive split/smooth with snapshots."""

from __future__ import annotations

from pathlib import Path

from gmshjax.mesh.adaptive_quad import adaptive_remesh_quad
from gmshjax.mesh.topology import unit_square_quad_mesh


def main() -> None:
    topo, pts = unit_square_quad_mesh(20, 14)
    out_dir = Path("results") / "m3_quad_demo"
    buf, hist = adaptive_remesh_quad(
        pts,
        topo.elements,
        max_nodes=12000,
        max_elements=18000,
        max_iters=10,
        target_area=0.0018,
        target_mean_icn=0.50,
        snapshot_dir=out_dir,
    )
    print("iterations:", len(hist))
    print("final_nodes:", buf.node_count)
    print("final_elements:", buf.element_count)
    print("snapshots:", str(out_dir))


if __name__ == "__main__":
    main()
