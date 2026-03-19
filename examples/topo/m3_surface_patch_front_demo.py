"""Build a 3D surface patch, triangulate it, and export the result as Gmsh MSH."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from topojax.io.exports import export_gmsh_msh
from topojax.mesh.boundary import BoundaryCurves3D, line_segment, surface_front_tri_mesh


def _surface_patch(n_u: int, n_v: int) -> BoundaryCurves3D:
    bottom = line_segment(jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.25]), n_u)
    top = line_segment(jnp.array([0.0, 1.0, 0.35]), jnp.array([1.0, 1.0, 0.65]), n_u)
    left = line_segment(bottom[0], top[0], n_v)
    right = line_segment(bottom[-1], top[-1], n_v)
    return BoundaryCurves3D(bottom=bottom, right=right, top=top, left=left)


def main() -> None:
    out_dir = Path("results") / "m3_surface_patch_front_demo"
    curves = _surface_patch(21, 17)
    topo, points, uv = surface_front_tri_mesh(curves, 10, 8, jitter=0.28, relaxation_steps=3, seed=5)
    export_gmsh_msh(out_dir / "surface_patch_front.msh", points, topo.elements, element_entity_tags=topo.element_entity_tags)
    print("n_nodes:", topo.n_nodes)
    print("n_elements:", topo.elements.shape[0])
    print("uv_shape:", uv.shape)
    print("msh_exists:", (out_dir / "surface_patch_front.msh").exists())


if __name__ == "__main__":
    main()