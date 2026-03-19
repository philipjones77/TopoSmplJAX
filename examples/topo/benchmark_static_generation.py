"""Benchmark: static mesh generation throughput (NumPy vs JAX)."""

from __future__ import annotations

import time

from topojax.mesh.topology import unit_cube_tet_mesh, unit_square_quad_mesh, unit_square_tri_mesh
from topojax.numpy_impl import (
    unit_cube_tet_mesh as np_unit_cube_tet_mesh,
    unit_square_quad_mesh as np_unit_square_quad_mesh,
    unit_square_tri_mesh as np_unit_square_tri_mesh,
)


def _bench(label: str, fn, repeats: int = 20) -> None:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    dt = (time.perf_counter() - start) / repeats
    print(f"{label:28s} avg={dt * 1e3:8.3f} ms")


def main() -> None:
    print("Static mesh generation benchmark")
    _bench("JAX tri  (64x64)", lambda: unit_square_tri_mesh(64, 64))
    _bench("NumPy tri(64x64)", lambda: np_unit_square_tri_mesh(64, 64))
    _bench("JAX quad (64x64)", lambda: unit_square_quad_mesh(64, 64))
    _bench("NumPy quad(64x64)", lambda: np_unit_square_quad_mesh(64, 64))
    _bench("JAX tet  (20x20x20)", lambda: unit_cube_tet_mesh(20, 20, 20), repeats=5)
    _bench("NumPy tet(20x20x20)", lambda: np_unit_cube_tet_mesh(20, 20, 20), repeats=5)


if __name__ == "__main__":
    main()
