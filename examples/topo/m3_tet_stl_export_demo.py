"""Export a tetrahedral mesh boundary as a binary STL surface."""

from __future__ import annotations

from pathlib import Path

from topojax.io.exports import export_binary_stl
from topojax.mesh.topology import unit_cube_tet_mesh


def main() -> None:
    topo, points = unit_cube_tet_mesh(4, 4, 4)
    out_path = Path("results") / "m3_tet_stl_export_demo" / "cube_surface.stl"
    export_binary_stl(out_path, points, topo.elements)
    print("stl_exists:", out_path.exists())
    print("triangle_count_estimate:", 12 * (4 - 1) * (4 - 1))


if __name__ == "__main__":
    main()
