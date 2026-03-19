# smplJAX Architecture (Version 1.0.0)

High-level architecture layers:
1. Model IO and schema normalization (`.npz` / `.pkl`).
2. Core LBS math (JAX-native, autodiff-safe).
3. Runtime orchestration:
   - baseline API (`SMPLJAXModel`)
   - optimized API (`OptimizedSMPLJAX`) with policy-controlled caching.
4. Optional visualization integration (`viser`).
5. Tests, parity harness, and benchmarks.

Repository structure authority is defined in `docs/smpl/documentation_governance.md`.
For standards and platform policy, see `docs/smpl/standards.md` and `docs/smpl/jax_standards.md`.
