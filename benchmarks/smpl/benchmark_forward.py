from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp

from smpljax.body_models import SMPLJAXModel
from smpljax.utils import SMPLModelData


def _build_synthetic_model(
    *,
    num_verts: int,
    num_joints: int,
    num_betas: int,
    dtype: jnp.dtype,
) -> SMPLJAXModel:
    body_joints = max(num_joints - 1, 0)
    return SMPLJAXModel(
        data=SMPLModelData(
            v_template=jnp.zeros((num_verts, 3), dtype=dtype),
            shapedirs=jnp.zeros((num_verts, 3, num_betas), dtype=dtype),
            posedirs=jnp.zeros((body_joints * 9, num_verts * 3), dtype=dtype),
            j_regressor=jnp.ones((num_joints, num_verts), dtype=dtype) / max(num_verts, 1),
            parents=jnp.array([-1] + list(range(num_joints - 1)), dtype=jnp.int32),
            lbs_weights=jnp.ones((num_verts, num_joints), dtype=dtype) / max(num_joints, 1),
            num_betas=num_betas,
            num_body_joints=body_joints,
        )
    )


def _runtime_metadata() -> dict[str, str]:
    devices = jax.devices()
    default = devices[0] if devices else None
    return {
        "jax_backend": jax.default_backend(),
        "device_kind": getattr(default, "device_kind", "unknown") if default is not None else "unknown",
        "platform": getattr(default, "platform", "unknown") if default is not None else "unknown",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline SMPLJAX forward runtime.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--num-verts", type=int, default=64)
    parser.add_argument("--num-joints", type=int, default=24)
    parser.add_argument("--num-betas", type=int, default=10)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    model = _build_synthetic_model(
        num_verts=args.num_verts,
        num_joints=args.num_joints,
        num_betas=args.num_betas,
        dtype=dtype,
    )
    betas = jnp.zeros((args.batch_size, args.num_betas), dtype=dtype)
    body_pose = jnp.zeros((args.batch_size, max(args.num_joints - 1, 0), 3), dtype=dtype)
    global_orient = jnp.zeros((args.batch_size, 1, 3), dtype=dtype)

    t0 = time.perf_counter()
    out = model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        pose2rot=True,
    )
    _ = out.vertices.block_until_ready()
    compile_plus_first_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(args.iters):
        out = model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            pose2rot=True,
        )
        _ = out.vertices.block_until_ready()
    runtime_s = (time.perf_counter() - t0) / max(args.iters, 1)

    payload = {
        "benchmark": "forward",
        "runtime": "baseline",
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "num_verts": args.num_verts,
        "num_joints": args.num_joints,
        "num_betas": args.num_betas,
        "iters": args.iters,
        "compile_plus_first_s": compile_plus_first_s,
        "runtime_s": runtime_s,
        **_runtime_metadata(),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps([payload], indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
