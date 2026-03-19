"""Benchmark mode-1 fixed-topology optimization for tri, quad, and tet meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topojax.ad.mode1 import benchmark_mode1_fixed_topology
from topojax.mesh.topology import unit_cube_tet_mesh, unit_square_quad_mesh, unit_square_tri_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark mode-1 fixed-topology optimization kernels.")
    parser.add_argument("--kind", choices=["tri", "quad", "tet"], default="tri")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.kind == "tri":
        topo, points = unit_square_tri_mesh(48, 36)
    elif args.kind == "quad":
        topo, points = unit_square_quad_mesh(48, 36)
    else:
        topo, points = unit_cube_tet_mesh(10, 10, 8)

    result = benchmark_mode1_fixed_topology(points, topo, steps=args.steps)
    payload = {
        "kind": args.kind,
        "steps": result.steps,
        "first_call_ms": result.first_call_ms,
        "steady_state_ms_per_step": result.steady_state_ms_per_step,
        "final_energy": result.final_energy,
        "final_grad_norm": result.final_grad_norm,
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    for key, value in payload.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
