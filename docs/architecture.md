# Architecture

GmshJAX is an AD-oriented mesh creation stack built around a static-topology, dynamic-coordinate model.

## Core Model

- mesh topology is represented with fixed-shape arrays for node ids, element ids, and connectivity
- coordinates live in mutable runtime state so geometry can be optimized with `jit`, `vmap`, and `grad`
- Gmsh-style entity ownership is preserved through dimension and tag descriptors attached to runtime objects and element groups

## Runtime Layers

- `src/gmshjax/model.py`: mesh model, state, and entity definitions
- `src/gmshjax/mesh/topology.py`: topology builders and fixed-shape mesh substrates
- `src/gmshjax/mesh/generators.py` and `src/gmshjax/mesh/factory.py`: structured generators and model factory helpers
- `src/gmshjax/mesh/boundary.py`: boundary parameterization and constrained point placement
- `src/gmshjax/mesh/refine.py`, `src/gmshjax/mesh/mutation.py`, and `src/gmshjax/mesh/mutation_qt.py`: refinement candidates and in-place style fixed-capacity mutation buffers encoded functionally
- `src/gmshjax/mesh/adaptive.py`, `src/gmshjax/mesh/adaptive_quad.py`, and `src/gmshjax/mesh/adaptive_tet.py`: adaptive remeshing loops and history objects
- `src/gmshjax/ad/`: objective compilation and parameterized quality pipelines
- `src/gmshjax/io/exports.py`: export hooks for downstream tooling

## Execution Split

- `src/gmshjax/numpy_impl.py`: NumPy forward/reference implementation substrate
- `src/gmshjax/jax_impl.py`: JAX-native implementation helpers and compiled pathways
- shared mesh logic aims to keep forward behavior aligned across the NumPy and JAX layers

## User-Facing Surface

The package root re-exports the intended public surface from `gmshjax.__init__`, including:

- mesh model and topology types such as `MeshModel`, `MeshState`, `MeshTopology`, and `GEntity`
- static generators such as `unit_square_tri_mesh`, `unit_square_quad_mesh`, and `unit_cube_tet_mesh`
- adaptive entry points such as `adaptive_remesh_tri`, `adaptive_remesh_quad`, and `adaptive_remesh_tet`
- diagnostic and quality helpers such as `tri_diagnostics`, `quad_diagnostics`, `tet_diagnostics`, `triangle_icn`, `quad_icn`, and `tet_icn`
- AD pipeline builders such as `build_quality_value_and_grad`, `build_parametric_quality_value_and_grad`, and `build_model_parametric_quality_value_and_grad`

## Artifacts and Validation

- `tests/` covers generators, boundary handling, refinement, mutation, diagnostics, and adaptive workflows
- `examples/` and `notebooks/` demonstrate end-to-end usage patterns
- `results/` currently stores generated demo and benchmark outputs as a retained legacy artifact namespace
