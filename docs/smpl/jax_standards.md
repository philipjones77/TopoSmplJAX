# JAX Standards (Version 1.0.0)

## JIT Policy
- Compile on stable-shape signatures only.
- Distinguish value updates from shape changes.
- Reuse compiled functions through bounded runtime cache.

## AD Policy
- Preserve pure-JAX ops in forward path to keep `grad`/`value_and_grad` valid.
- Do not insert NumPy transforms in differentiable hot paths.
- Keep loss/grad step jitted where possible.

## Shape Policy
- Bucket batch sizes for stable compile keys.
- Pad to bucket size and slice outputs to actual batch size.
- Keep `pose2rot` mode explicit and fixed per compiled function.

## Dtype Policy
- Default: `float32`.
- Optional: `float64` via runtime policy.
- Keep integer indices (`parents`, faces, ids) as `int32`.

## Diagnostics Policy
- Record and expose:
  - compile count
  - cache hits/misses
  - compiled entries
  - model/input/output byte estimates
