# Implementation Overview (Version 1.0.0)

## Modules
- `smpljax.io`: model parsing + normalization + bounded IO cache.
- `smpljax.lbs`: core linear blend skinning math.
- `smpljax.body_models`: baseline model interface.
- `smpljax.optimized`: JIT-focused runtime with cache policy + diagnostics.
- `smpljax.visualization`: optional `viser` and `pyvista` integration.

## Runtime Modes
- Baseline mode (`SMPLJAXModel`) for straightforward usage.
- Optimized mode (`OptimizedSMPLJAX`) for iterative optimization and repeated inference.

## Performance Controls
- `CachePolicy.max_compiled`: bounds compiled function count.
- `CachePolicy.batch_buckets`: limits shape variants and recompiles.
- `CachePolicy.dtype`: controls global numeric dtype.

## Cross-Platform Notes
- Python entry points and path handling are compatible with Windows/Linux.
- Optional visualization stack remains isolated from core runtime.
