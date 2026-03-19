# Benchmark Results (Version 1.0.0)

This page documents the intended published benchmark surface for `smplJAX`.

## Methodology

- Synthetic models are used so benchmarks are reproducible without licensed assets.
- Benchmarks are run with `PYTHONPATH=src`.
- Baseline and optimized runtime modes are reported separately.
- Compile time and steady-state runtime are both tracked.
- Optimized runtime results should include cache diagnostics where possible.

## Published Record Fields

- `benchmark`
- `runtime`
- `dtype`
- `batch_size`
- `num_verts`
- `num_joints`
- `num_betas`
- `iters`
- `compile_s` or `compile_plus_first_s`
- `runtime_s` or `steady_state_s`
- `speedup`
- `compile_count`
- `cache_hits`
- `cache_misses`
- `compiled_entries`
- `warmup_batch_size`
- `warmup_s`
- `fixed_padded_batch_size`
- `forbid_new_compiles`
- `last_input_bytes`
- `last_output_bytes`
- `jax_backend`
- `device_kind`
- `platform`

## Example Commands

```powershell
$env:PYTHONPATH='src'
python benchmarks/smpl/benchmark_forward.py `
  --batch-size 16 `
  --iters 100 `
  --output-json outputs/smpl/benchmarks/forward.json

python benchmarks/smpl/benchmark_optimized_runtime.py `
  --batch-size 32 `
  --iters 100 `
  --output-json outputs/smpl/benchmarks/optimized.json

python benchmarks/smpl/benchmark_optimized_runtime.py `
  --batch-size 16 `
  --warmup-batch-size 32 `
  --fixed-padded-batch-size 32 `
  --forbid-new-compiles `
  --iters 100 `
  --output-json outputs/smpl/benchmarks/optimized-fixed32.json
```

## Notes

- Numeric results depend on device, OS, JAX backend, and dtype configuration.
- `float64` benchmark runs should be executed with `JAX_ENABLE_X64=1`.
- Cross-platform reporting should include the OS and Python/JAX versions in the surrounding report or CI artifact metadata.
- The optimized runtime is expected to minimize recompiles for value-only changes with fixed shapes.
- Fixed padded `32` runs are the reference no-new-compile benchmark for deployment-style usage.
- Benchmark workflow artifacts are published by `.github/workflows/benchmarks.yml`.
