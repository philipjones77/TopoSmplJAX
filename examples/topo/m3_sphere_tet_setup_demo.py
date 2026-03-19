"""3D setup demo: static tetra mesh projected onto a sphere."""

from __future__ import annotations

import jax.numpy as jnp

from topojax.mesh.generators import project_cube_points_to_sphere
from topojax.mesh.operators import tet_icn, tet_ige
from topojax.mesh.topology import unit_cube_tet_mesh


def main() -> None:
    topo, pts = unit_cube_tet_mesh(10, 10, 10)
    sph = project_cube_points_to_sphere(pts, radius=1.0, center=jnp.array([0.5, 0.5, 0.5], dtype=pts.dtype))
    icn = tet_icn(sph, topo.elements)
    ige = tet_ige(sph, topo.elements)
    print("sphere_points_shape:", sph.shape)
    print("tet_elements_shape:", topo.elements.shape)
    print("mean_icn:", float(jnp.mean(icn)))
    print("mean_ige:", float(jnp.mean(ige)))


if __name__ == "__main__":
    main()
