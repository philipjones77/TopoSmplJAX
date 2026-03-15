"""Mode 1 demo for arbitrary 2D and 3D meshes with Gmsh viewing."""

from __future__ import annotations

import argparse

import jax.numpy as jnp

from gmshjax.ad.workflow import initialize_mode1_domain, run_mode1_workflow


def _polygon_case():
    outer = jnp.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.2, 1.0],
            [1.2, 2.0],
            [0.0, 2.0],
        ]
    )
    hole = jnp.asarray(
        [
            [0.45, 0.45],
            [0.95, 0.45],
            [0.95, 0.95],
            [0.45, 0.95],
        ]
    )
    domain = initialize_mode1_domain("polygon", outer_boundary=outer, holes=[hole], target_edge_size=0.18)
    x = domain.points[:, 0]
    y = domain.points[:, 1]
    distorted = domain.points + 0.04 * jnp.stack([jnp.sin(2.0 * jnp.pi * y), jnp.sin(jnp.pi * x)], axis=1)
    return domain._replace(points=distorted), "polygon"


def _polygon_quad_case():
    outer = jnp.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.2, 1.0],
            [1.2, 2.0],
            [0.0, 2.0],
        ]
    )
    hole = jnp.asarray(
        [
            [0.45, 0.45],
            [0.95, 0.45],
            [0.95, 0.95],
            [0.45, 0.95],
        ]
    )
    domain = initialize_mode1_domain("polygon-quad", outer_boundary=outer, holes=[hole], target_edge_size=0.2)
    x = domain.points[:, 0]
    y = domain.points[:, 1]
    distorted = domain.points + 0.025 * jnp.stack([jnp.sin(1.5 * jnp.pi * y), jnp.sin(jnp.pi * x)], axis=1)
    return domain._replace(points=distorted), "polygon_quad"


def _sphere_level_set(points: jnp.ndarray) -> jnp.ndarray:
    center = jnp.asarray([0.5, 0.5, 0.5], dtype=points.dtype)
    return jnp.linalg.norm(points - center[None, :], axis=1) - 0.42


def _volume_case():
    domain = initialize_mode1_domain(
        "implicit-volume",
        level_set_fn=_sphere_level_set,
        bbox_min=jnp.asarray([0.0, 0.0, 0.0]),
        bbox_max=jnp.asarray([1.0, 1.0, 1.0]),
        nx=11,
        ny=11,
        nz=11,
    )
    x = domain.points[:, 0]
    y = domain.points[:, 1]
    z = domain.points[:, 2]
    distorted = domain.points.at[:, 2].set(z + 0.03 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y))
    return domain._replace(points=distorted), "volume"


def _box_volume_case():
    domain = initialize_mode1_domain(
        "box-volume",
        bbox_min=jnp.asarray([-1.0, -0.5, 0.0]),
        bbox_max=jnp.asarray([1.0, 0.5, 2.0]),
        nx=9,
        ny=7,
        nz=8,
    )
    x = domain.points[:, 0]
    y = domain.points[:, 1]
    z = domain.points[:, 2]
    distorted = domain.points.at[:, 2].set(z + 0.025 * jnp.sin(0.5 * jnp.pi * x) * jnp.cos(jnp.pi * y))
    return domain._replace(points=distorted), "box_volume"


def _extruded_case():
    outer = jnp.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.2, 1.0],
            [1.2, 2.0],
            [0.0, 2.0],
        ]
    )
    hole = jnp.asarray(
        [
            [0.45, 0.45],
            [0.95, 0.45],
            [0.95, 0.95],
            [0.45, 0.95],
        ]
    )
    domain = initialize_mode1_domain("extruded", outer_boundary=outer, holes=[hole], target_edge_size=0.18, height=1.0, layers=4)
    x = domain.points[:, 0]
    y = domain.points[:, 1]
    z = domain.points[:, 2]
    distorted = domain.points.at[:, 2].set(z + 0.025 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y))
    return domain._replace(points=distorted), "extruded"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mode 1 arbitrary-domain demo with optional Gmsh viewing")
    parser.add_argument("--kind", choices=["polygon", "polygon-quad", "extruded", "volume", "box-volume"], default="polygon")
    parser.add_argument("--output-dir", default="results/m1_polygon_volume_gmsh_demo")
    parser.add_argument("--launch-gmsh", action="store_true")
    parser.add_argument("--gmsh-executable", default="gmsh")
    args = parser.parse_args()

    if args.kind == "polygon":
        domain, prefix = _polygon_case()
    elif args.kind == "polygon-quad":
        domain, prefix = _polygon_quad_case()
    elif args.kind == "extruded":
        domain, prefix = _extruded_case()
    elif args.kind == "box-volume":
        domain, prefix = _box_volume_case()
    else:
        domain, prefix = _volume_case()
    run = run_mode1_workflow(
        domain,
        output_dir=args.output_dir,
        prefix=prefix,
        steps=60,
        step_size=0.02,
        diagnostics_every=15,
        launch_gmsh=args.launch_gmsh,
        gmsh_executable=args.gmsh_executable,
    )

    print("summary:", {"final_energy": float(run.result.energy_history[-1]), "steps": int(run.result.energy_history.shape[0])})
    print("mesh:", run.artifacts["mesh"])


if __name__ == "__main__":
    main()
