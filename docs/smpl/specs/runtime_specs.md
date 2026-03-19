# Runtime Specs (Version 1.0.0)

## Dtype
- Default runtime dtype: `float32`.
- Optional runtime dtype: `float64` (policy-controlled).
- `float64` execution depends on JAX x64 support being enabled; otherwise JAX may downcast to `float32`.
- CI should exercise both `JAX_ENABLE_X64=0` and `JAX_ENABLE_X64=1`.

## Cache Bounds
- Compile cache: bounded LRU (`max_compiled`).
- IO cache: hard-capped at 2 models in memory.
- Optional fixed padded batch capacity can reserve a single compile shape ahead of time.

## No-recompile Contract
- Value-only changes (e.g., `transl`) with fixed shapes should reuse compiled functions.
- Shape/bucket changes may compile new variants up to cache limits.
- Diagnostics should expose compile count, cache hits/misses, and current compiled-entry count.
- Diagnostics should expose compile-key history, warmup coverage, and explicit strict-mode readiness.
- Diagnostics should expose compile-key evictions when cache pressure removes older compiled variants.
- `fixed_padded_batch_size` allows smaller later batches to reuse the same padded compile shape.
- `forbid_new_compiles=True` converts unexpected new compile keys into runtime errors.

## AD Contract
- Forward path supports JAX autodiff in optimized runtime.
- Benchmark and diagnostics workflows should preserve this property; instrumentation must not replace JAX ops in the forward path.

## Platform Contract
- Windows and Linux are supported targets for core runtime and docs workflows.
- Runtime diagnostics should expose backend/device metadata and JAX/Python version context for deployment reports.
