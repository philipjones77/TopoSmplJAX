# API Contract (Version 1.0.0)

## Core APIs
- `create(model_path)`
- `create_uncached(model_path)`
- `create_optimized(model_path, ...)`
- `create_runtime(model_path, mode=...)`
- `SMPLJAXModel`
- `OptimizedSMPLJAX`

## Behavioral Guarantees
- Core forward paths are autodiff-compatible.
- Runtime caching is bounded and observable through diagnostics.
- IO model cache is capped to 2 models in memory.
- Non-cached model loading remains available for workflows that need strict reload semantics.
- Optimized runtime supports controlled recompiles through fixed-shape execution and optional batch bucketing.
- Formal model IO diagnostics are available through `describe_model(...)` and validator tooling.

## Compatibility
- Supported model containers: `.npz`, `.pkl`.
- Supported OS targets: Windows, Linux.
