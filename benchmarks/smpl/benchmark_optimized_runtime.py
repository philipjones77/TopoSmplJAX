from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax.numpy as jnp

from smpljax.optimized import CachePolicy, OptimizedSMPLJAX
from smpljax.utils import SMPLModelData


def _build_synthetic_model(
    *,
    num_verts: int,
    num_joints: int,
    num_betas: int,
    dtype: jnp.dtype,
    cache_policy: CachePolicy,
) -> OptimizedSMPLJAX:
    body_joints = max(num_joints - 1, 0)
    return OptimizedSMPLJAX(
        data=SMPLModelData(
            v_template=jnp.zeros((num_verts, 3), dtype=dtype),
            shapedirs=jnp.zeros((num_verts, 3, num_betas), dtype=dtype),
            posedirs=jnp.zeros((body_joints * 9, num_verts * 3), dtype=dtype),
            j_regressor=jnp.ones((num_joints, num_verts), dtype=dtype) / max(num_verts, 1),
            parents=jnp.array([-1] + list(range(num_joints - 1)), dtype=jnp.int32),
            lbs_weights=jnp.ones((num_verts, num_joints), dtype=dtype) / max(num_joints, 1),
            num_betas=num_betas,
            num_body_joints=body_joints,
        ),
        dtype=dtype,
        cache_policy=cache_policy,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark optimized SMPLJAX forward runtime.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-batch-size", type=int, default=None)
    parser.add_argument("--fixed-padded-batch-size", type=int, default=None)
    parser.add_argument("--forbid-new-compiles", action="store_true")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--num-verts", type=int, default=64)
    parser.add_argument("--num-joints", type=int, default=24)
    parser.add_argument("--num-betas", type=int, default=10)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    policy = CachePolicy(
        dtype=dtype,
        fixed_padded_batch_size=args.fixed_padded_batch_size,
        forbid_new_compiles=args.forbid_new_compiles,
    )
    model = _build_synthetic_model(
        num_verts=args.num_verts,
        num_joints=args.num_joints,
        num_betas=args.num_betas,
        dtype=dtype,
        cache_policy=policy,
    )

    warmup_s = 0.0
    warmup_batch_size = args.warmup_batch_size
    if warmup_batch_size is not None:
        t0 = time.perf_counter()
        model.warmup(batch_size=warmup_batch_size, pose2rot=True)
        warmup_s = time.perf_counter() - t0

    inputs = model.prepare_inputs(batch_size=args.batch_size, pose2rot=True)
    t0 = time.perf_counter()
    out = model.forward(inputs, pose2rot=True)
    _ = out.vertices.block_until_ready()
    compile_plus_first_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(args.iters):
        out = model.forward(inputs, pose2rot=True)
        _ = out.vertices.block_until_ready()
    steady_state_s = (time.perf_counter() - t0) / max(args.iters, 1)

    diagnostics = model.diagnostics()
    payload = {
        "benchmark": "optimized_runtime",
        "runtime": "optimized",
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "num_verts": args.num_verts,
        "num_joints": args.num_joints,
        "num_betas": args.num_betas,
        "iters": args.iters,
        "compile_plus_first_s": compile_plus_first_s,
        "steady_state_s": steady_state_s,
        "compile_count": diagnostics.compile_count,
        "cache_hits": diagnostics.cache_hits,
        "cache_misses": diagnostics.cache_misses,
        "compiled_entries": diagnostics.compiled_entries,
        "warmup_batch_size": warmup_batch_size,
        "warmup_s": warmup_s,
        "fixed_padded_batch_size": diagnostics.fixed_padded_batch_size,
        "forbid_new_compiles": diagnostics.forbid_new_compiles,
        "last_input_bytes": diagnostics.last_input_bytes,
        "last_output_bytes": diagnostics.last_output_bytes,
        "jax_backend": diagnostics.jax_backend,
        "device_kind": diagnostics.device_kind,
        "platform": diagnostics.platform,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps([payload], indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
