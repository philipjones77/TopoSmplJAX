# JAX Standards (Version 1.0.0)

- Use `jax.lax` / vectorized operations for iterative kernels in hot paths.
- Keep compile keys limited and predictable.
- Separate setup/IO from runtime compute.
- Ensure all optimization-loop data remains in JAX arrays.
- Track compile/cache diagnostics in long-lived processes.
