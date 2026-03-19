# Standards (Version 1.0.0)

## Runtime
- Use `OptimizedSMPLJAX` for long-running loops and optimization workflows.
- Use `CachePolicy` to control dtype, compile-cache size, and batch bucketing.
- Keep compile-cache bounded (`max_compiled`) to protect memory.

## Memory and Caching
- IO model cache is hard-capped to 2 models in memory.
- Compile cache uses bounded LRU eviction.
- Collect diagnostics regularly in long-running services.

## Cross-Platform
- Support Windows and Linux as first-class runtime targets.
- Avoid shell-specific assumptions in docs and scripts; include both patterns where needed.
- Use portable Python entry points (`python -m ...`) when possible.

## Interfaces
- Inputs should be shape-stable and dtype-consistent for JIT reuse.
- Public APIs should remain backward-compatible within `1.x`.

## Documentation
- Any runtime policy changes must update:
  - `project_overview.md`
  - `architecture.md`
  - `jax_standards.md`
  - `implementation/dependency_registry.md`
