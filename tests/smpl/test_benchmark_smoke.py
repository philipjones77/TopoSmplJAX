import time

import jax.numpy as jnp

from smpljax.optimized import CachePolicy, OptimizedSMPLJAX
from smpljax.utils import SMPLModelData


def test_optimized_runtime_smoke_benchmark() -> None:
    data = SMPLModelData(
        v_template=jnp.zeros((64, 3), dtype=jnp.float32),
        shapedirs=jnp.zeros((64, 3, 5), dtype=jnp.float32),
        posedirs=jnp.zeros((207, 64 * 3), dtype=jnp.float32),
        j_regressor=jnp.ones((24, 64), dtype=jnp.float32) / 64.0,
        parents=jnp.array([-1] + list(range(23)), dtype=jnp.int32),
        lbs_weights=jnp.ones((64, 24), dtype=jnp.float32) / 24.0,
        num_body_joints=23,
    )
    model = OptimizedSMPLJAX(data=data)
    inp = model.prepare_inputs(batch_size=2)

    t0 = time.perf_counter()
    out = model.forward(inp, pose2rot=True)
    _ = out.vertices.block_until_ready()
    first_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = model.forward(inp, pose2rot=True)
    _ = out.vertices.block_until_ready()
    second_t = time.perf_counter() - t0

    assert model.compile_count == 1
    assert first_t > 0.0
    assert second_t > 0.0


def test_fixed_32_runtime_smoke_benchmark() -> None:
    data = SMPLModelData(
        v_template=jnp.zeros((64, 3), dtype=jnp.float32),
        shapedirs=jnp.zeros((64, 3, 5), dtype=jnp.float32),
        posedirs=jnp.zeros((207, 64 * 3), dtype=jnp.float32),
        j_regressor=jnp.ones((24, 64), dtype=jnp.float32) / 64.0,
        parents=jnp.array([-1] + list(range(23)), dtype=jnp.int32),
        lbs_weights=jnp.ones((64, 24), dtype=jnp.float32) / 24.0,
        num_body_joints=23,
    )
    model = OptimizedSMPLJAX(
        data=data,
        cache_policy=CachePolicy(fixed_padded_batch_size=32, forbid_new_compiles=True),
    )
    _ = model.warmup(batch_size=32, pose2rot=True)
    inp = model.prepare_inputs(batch_size=16)

    t0 = time.perf_counter()
    out = model.forward(inp, pose2rot=True)
    _ = out.vertices.block_until_ready()
    steady_t = time.perf_counter() - t0

    assert model.compile_count == 1
    assert inp.padded_batch_size == 32
    assert steady_t > 0.0
