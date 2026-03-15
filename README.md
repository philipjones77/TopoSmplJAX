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
- `docs/`: governed documentation tree with architecture, implementation, theory, and status material.
- `contracts/`: binding public API and runtime guarantees.
- `examples/`, `notebooks/`: usage and experiments.
- `results/`: generated experiment outputs.

Documentation entry points:

- `docs/project_overview.md`
- `docs/architecture.md`
- `docs/documentation_governance.md`
- `docs/status/mode1_2d_3d_status.md`
- `contracts/api_contract.md`

## Install (editable)

```powershell
python -m pip install -e .[dev]
```

Visualization extras:

```powershell
python -m pip install -e .[dev,viz]
```

## Tests

```powershell
python -m pytest tests -q
```

## Notes

- This repository is focused on AD-friendly mesh construction and optimization in JAX.
- Geometry and meshing APIs are designed to work with `jit`, `vmap`, and `grad`.
- Fixed connectivity and stable shapes let JAX compile once and reuse as the manifold deforms.
- Mode 1 fixed-topology workflows now have dedicated optimization, diagnostics/IO export, benchmarking, and visualization helpers.
- Mode 1 now supports first-class line meshes, arbitrary 2D polygon-domain triangles and quads, exact box-volume tetrahedral initialization, 3D implicit-volume tetrahedral initialization, 3D polygon extrusion to tetrahedra, supported `.msh` import, and a single high-level create-optimize-export-view workflow.
- Tagged domain metadata is exported through Gmsh line and triangle boundary blocks, with Gmsh used only as a viewer.
- The repo virtual environment can be prepared for Mode 1 visualization tests with `matplotlib` and `pyvista` installed as optional dependencies.
- NumPy and JAX layers share the same mesh logic so forward computations match closely.
- Static generators currently available: structured 2D triangle/quad and 3D tetrahedral meshes.
- M3 candidate kernels are available for boundary-constrained meshing and adaptive refinement scoring.
- M3 complete path is available: split/flip mutation + adaptive loop + snapshot export.
- 2D quad adaptive loop is included; 3D sphere setups are supported via cube-to-sphere projection.
