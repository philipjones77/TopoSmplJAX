# Release Policy (Version 1.0.0)

## Versioning
- Semantic versioning with current baseline `1.0.0`.

## Requirements for Minor/patch releases
- Tests passing on supported environments.
- Updated docs for any changed runtime policy.
- Benchmark smoke run for optimized runtime.

## Runtime Policy Changes
- Any change to cache policy, dtype defaults, or platform behavior must include:
  - documentation updates
  - tests covering expected behavior
