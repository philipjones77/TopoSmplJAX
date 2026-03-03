# GmshJAX

AD Mesh Creation for JAX.

Gmsh-inspired architecture in JAX:
- `MVertex`/`MElement` style immutable ids are represented as static arrays in topology.
- `GEntity` ownership is preserved through element-to-entity tags.
- Topology is static, while node coordinates are dynamic and differentiable.

## Layout

- `src/gmshjax/`: differentiable mesh creation runtime.
- `src/gmshjax/numpy_impl.py`: NumPy forward implementation.
- `src/gmshjax/jax_impl.py`: JAX-native implementation and helpers.
- `src/gmshjax/mesh/boundary.py`: boundary parameterization and constrained placement.
- `src/gmshjax/mesh/refine.py`: batched local refinement candidate kernels.
- `src/gmshjax/mesh/connectivity_opt.py`: vectorized edge-flip/smoothing candidate evaluators.
- `src/gmshjax/mesh/mutation.py`: fixed-capacity split/flip connectivity mutation.
- `src/gmshjax/mesh/adaptive.py`: adaptive remeshing loop with stopping criteria.
- `src/gmshjax/mesh/mutation_qt.py`: fixed-capacity quad/tet mutation buffers.
- `src/gmshjax/mesh/adaptive_quad.py`: adaptive remeshing loop for 2D quads.
- `tests/`: unit and integration tests.
- `tools/`: utility scripts for local workflows.
- `docs/`: design and implementation notes.
- `examples/`, `notebooks/`: usage and experiments.
- `results/`: generated experiment outputs.

## Install (editable)

```powershell
python -m pip install -e .[dev]
```

## Tests

```powershell
python -m pytest tests -q
```

## Notes

- This repository is focused on AD-friendly mesh construction and optimization in JAX.
- Geometry and meshing APIs are designed to work with `jit`, `vmap`, and `grad`.
- Fixed connectivity and stable shapes let JAX compile once and reuse as the manifold deforms.
- NumPy and JAX layers share the same mesh logic so forward computations match closely.
- Static generators currently available: structured 2D triangle/quad and 3D tetrahedral meshes.
- M3 candidate kernels are available for boundary-constrained meshing and adaptive refinement scoring.
- M3 complete path is available: split/flip mutation + adaptive loop + snapshot export.
- 2D quad adaptive loop is included; 3D sphere setups are supported via cube-to-sphere projection.
