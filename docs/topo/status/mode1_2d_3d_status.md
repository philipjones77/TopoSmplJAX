# Mode 1 2D And 3D Status

Status: active
Date: 2026-03-15

## Scope Of This Status Note

This note closes out the current practical Mode 1 expansion work for 1D, 2D, and 3D fixed-topology workflows.

Mode 1 in this repository means:

- initialize or import a mesh discretely
- keep connectivity fixed during optimization
- optimize node coordinates with JAX-compatible fixed-topology objectives
- export diagnostics and mesh artifacts after optimization

It does not mean full original Gmsh parity or exact autodiff through topology mutation.

## Completed Practical Coverage

The current Mode 1 surface now includes:

- 1D line meshes through `polyline_mesh` and `unit_interval_line_mesh`
- arbitrary 2D polygon-domain triangle meshes with optional polygonal holes
- arbitrary 2D polygon-domain conforming quad meshes via tri-to-quad subdivision
- exact structured tetrahedral fills for axis-aligned boxes
- implicit-volume tetrahedral initialization from level-set filtering
- 3D polygon extrusion to tetrahedra
- supported Gmsh MSH 2.2 import for line, triangle, quad, and tetra meshes
- a single high-level create or import, optimize, export, and optional viewer-launch workflow
- tagged boundary export through Gmsh line and triangle element blocks with physical names

## Validation State

The following validation points have been verified during this work:

- the repo virtual environment now includes the optional visualization dependencies needed by `tests/topo/test_mode1_fixed_topology.py`
- `tests/topo/test_mode1_fixed_topology.py` passes in the repo virtual environment
- the focused non-visualization Mode 1 slice has previously passed for arbitrary domains, workflow coverage, and diagnostics or export coverage

Recommended verification commands:

```powershell
c:\dev\TopoJAX\.venv\Scripts\python.exe -m pip install matplotlib>=3.8 pyvista>=0.43
$env:PYTHONPATH='src'
c:\dev\TopoJAX\.venv\Scripts\python.exe -m pytest tests/topo/test_mode1_fixed_topology.py -vv
c:\dev\TopoJAX\.venv\Scripts\python.exe -m pytest tests/topo/test_mode1_arbitrary_domains.py tests/topo/test_mode1_workflow.py tests/topo/test_m3_diagnostics_io.py -q
```

## Remaining Known Boundaries

Mode 1 is now in good shape for practical fixed-topology use, but these boundaries remain intentional:

- no exact reverse-mode gradients through split, collapse, flip, triangulation, or remeshing events
- no CAD-kernel-level geometry modeling or classification parity with original Gmsh
- no hex, prism, or pyramid Mode 1 domain initialization path
- no general-purpose body-fitted 3D unstructured mesher beyond the current structured, implicit, and extrusion-based initializers

## Next Implementation Plan

The next tranche should focus on strengthening the bridge from practical fixed-topology workflows to dynamic workflows.

### Near-Term Priorities

- add one fresh-process validation path or task wrapper for the standard Mode 1 regression slice to avoid terminal-state ambiguity during long runs
- add `docs/topo/specs/` material for mesh topology classes and domain-initializer semantics that are now effectively first-class runtime objects
- expand import and export coverage with more mixed-block round-trip tests and stricter metadata assertions
- add at least one additional practical 3D initializer if needed, but only if it fits the fixed-topology contract cleanly

### Medium-Term Priorities

- build mesh-to-mesh transfer primitives needed by restart and future dynamic workflows
- extend surrogate local connectivity work beyond quad diagonal choices
- unify mode dispatch and controller logic so Mode 1 through Mode 5 selection is exposed more coherently

### Explicit Recommendation

Do not spend the next cycle chasing broad Gmsh feature parity inside Mode 1. The better technical path is to treat the current Mode 1 coverage as the stable fixed-topology baseline and move new effort into transfer, controller, and surrogate infrastructure.


