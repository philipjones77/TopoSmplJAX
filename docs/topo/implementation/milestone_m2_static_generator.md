# Milestone M2: Functional Static Mesh Generator

Status: implemented in code and covered by tests.

## Objective

Provide a reliable static mesh generation baseline with:

1. Deterministic structured generation.
2. Shared NumPy and JAX implementations.
3. JAX compile-once operator pipeline for moving coordinates on fixed topology.

## Delivered

1. Generators:
   - `unit_square_tri_mesh(nx, ny)`
   - `unit_square_quad_mesh(nx, ny)`
   - `unit_cube_tet_mesh(nx, ny, nz)`
2. Quality metrics in NumPy and JAX:
   - Triangle/Quad/Tet `ICN`, `IGE`
3. AD pipeline:
   - `build_model_parametric_quality_value_and_grad(model)`
4. Benchmarks:
   - `examples/topo/benchmark_static_generation.py`
   - `examples/topo/benchmark_jax_quality_pipeline.py`
5. Notebook:
   - `notebooks/milestone_static_generator_bench.ipynb`

## Acceptance Criteria

1. Static generation is deterministic for fixed `(nx, ny, nz)`.
2. NumPy and JAX connectivity outputs are identical.
3. NumPy and JAX forward quality metrics match within tolerance.
4. JAX first-call cost is higher than steady-state calls (compile-once behavior).

## Next Milestone (M3)

From structured static meshes to geometry-driven meshing:

1. Boundary curve/surface parameterization API (JAX-compatible).
2. Structured boundary-constrained point placement.
3. Local adaptive refinement operators with fixed-shape batched updates.
4. Edge-flip/smoothing candidate evaluation kernels (vectorized and differentiable where possible).

## M3 Progress

Implemented:

1. Boundary API and TFI-based constrained placement in `mesh/boundary.py`.
2. Batched refinement candidate kernel in `mesh/refine.py`.
3. Vectorized edge-flip and smoothing candidate evaluators in `mesh/connectivity_opt.py`.
4. M3 tests in `tests/topo/test_m3_boundary_and_refine.py`.
5. Demo script in `examples/topo/m3_boundary_refine_demo.py`.

M3 completion delivered:

1. Connectivity mutation operators:
   - `split_triangle`
   - `flip_diagonal`
   - fixed-capacity triangle buffers
2. Adaptive remeshing loop:
   - iterative score/split/flip/smooth
   - quality and area stopping targets
3. Snapshot export each iteration:
   - `.npz` debug outputs for visualization and replay
