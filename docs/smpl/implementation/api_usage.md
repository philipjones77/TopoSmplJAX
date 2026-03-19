# API Usage (Version 1.0.0)

## Minimal API

Baseline:

```python
from smpljax import create

model = create("private_data/smpl/models/validated/smplx/MODEL_NAME.npz")
out = model(betas=betas, body_pose=body_pose, global_orient=global_orient)
```

Uncached baseline:

```python
from smpljax import create_uncached

model = create_uncached("private_data/smpl/models/validated/smplx/MODEL_NAME.npz")
```

Optimized runtime:

```python
from smpljax import CachePolicy, create_optimized

runtime = create_optimized(
    "private_data/smpl/models/validated/smplx/MODEL_NAME.npz",
    cache_policy=CachePolicy(max_compiled=4, batch_buckets=(1, 8, 16, 32, 64)),
)
inputs = runtime.prepare_inputs(batch_size=8)
out = runtime.forward(inputs, pose2rot=True)
diag = runtime.diagnostics()
```

Strict no-recompile window with fixed upfront padding:

```python
from smpljax import CachePolicy, create_optimized

runtime = create_optimized(
    "private_data/smpl/models/validated/smplx/MODEL_NAME.npz",
    cache_policy=CachePolicy(
        fixed_padded_batch_size=32,
        forbid_new_compiles=False,
    ),
)

# Warm once at the padded capacity, then reuse for smaller batches.
runtime.warmup(batch_size=32, pose2rot=True)

runtime.policy  # fixed padded capacity remains 32
```

Benchmark-style fixed `32` capacity flow:

```python
runtime = create_optimized(
    "private_data/smpl/models/validated/smplx/MODEL_NAME.npz",
    cache_policy=CachePolicy(
        fixed_padded_batch_size=32,
        forbid_new_compiles=True,
    ),
)
runtime.warmup(batch_size=32, pose2rot=True)
inputs = runtime.prepare_inputs(batch_size=16)
out = runtime.forward(inputs, pose2rot=True)
```

Unified factory:

```python
from smpljax import create_runtime

plain = create_runtime("private_data/smpl/models/validated/smplx/MODEL_NAME.npz", mode="plain")
uncached = create_runtime("private_data/smpl/models/validated/smplx/MODEL_NAME.npz", mode="uncached")
optimized = create_runtime("private_data/smpl/models/validated/smplx/MODEL_NAME.npz", mode="optimized")
```

Mesh export surface:

```python
template = optimized.export_template_mesh()
posed = optimized.export_posed_mesh(
    {
        "betas": betas,
        "body_pose": body_pose,
        "global_orient": global_orient,
        "transl": transl,
    }
)

ct_template = optimized.export_ct_mesh_payload_template()
ct_posed = optimized.export_ct_mesh_payload_pose(
    {
        "betas": betas,
        "body_pose": body_pose,
        "global_orient": global_orient,
        "transl": transl,
    }
)
```

## Diagnostics

Model IO:

```python
from smpljax import describe_model

summary = describe_model("private_data/smpl/models/validated/smplx/MODEL_NAME.npz")
```

Runtime cache and memory-ish diagnostics:

```python
diag = optimized.diagnostics()
optimized.assert_strict_ready()
```

Structured diagnostics export:

```python
from smpljax import write_runtime_diagnostics

write_runtime_diagnostics("output/runtime.json", runtime=optimized)
```

Diagnostics now include:

IO cache diagnostics:

```python
from smpljax import io_cache_diagnostics

stats = io_cache_diagnostics()
```

Viewer workflows can also emit:

## Contracts
