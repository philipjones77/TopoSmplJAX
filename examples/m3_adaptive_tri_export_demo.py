"""Run adaptive triangle remeshing and export each snapshot to Gmsh MSH."""

from __future__ import annotations

from pathlib import Path

from gmshjax.io.exports import export_gmsh_msh, load_snapshot_npz
from gmshjax.mesh.adaptive import adaptive_remesh_tri
from gmshjax.mesh.topology import unit_square_tri_mesh


def main() -> None:
    topo, pts = unit_square_tri_mesh(12, 8)
    out_dir = Path("results") / "m3_adaptive_tri_export_demo"
    out, hist = adaptive_remesh_tri(
        pts,
        topo.elements,
        max_nodes=4096,
        max_elements=8192,
        max_iters=4,
        target_area=0.01,
        target_mean_icn=0.4,
        snapshot_dir=out_dir,
    )

    snapshot_count = 0
    for npz_path in sorted(out_dir.glob("iter_*.npz")):
        snap = load_snapshot_npz(npz_path)
        msh_path = npz_path.with_suffix(".msh")
        export_gmsh_msh(msh_path, snap["points"], snap["elements"])
        snapshot_count += 1

    export_gmsh_msh(out_dir / "final_mesh.msh", out.points[: out.node_count], out.elements[: out.element_count])
    print("history_len:", len(hist))
    print("snapshot_count:", snapshot_count)
    print("final_msh_exists:", (out_dir / "final_mesh.msh").exists())


if __name__ == "__main__":
    main()