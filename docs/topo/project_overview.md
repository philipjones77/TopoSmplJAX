# Project Overview

TopoJAX is a JAX-first mesh creation and mesh adaptation workspace that keeps topology static while allowing coordinates, objectives, and deformation parameters to remain differentiable.

## Goals

- provide AD-friendly mesh generation and adaptation building blocks in JAX
- preserve Gmsh-inspired topology and entity concepts in an array-first runtime
- support both NumPy reference execution and JAX-compiled/autodiff execution
- keep exported meshes, diagnostics, and adaptive workflows reproducible through stable entry points

## Repository Shape

- `src/topojax/`: runtime implementation, model types, mutation kernels, and AD pipeline code
- `tests/topo/`: regression and integration tests for generators, refinement, mutation, and adaptive loops
- `examples/topo/`: runnable demos for boundary refinement, diagnostics, and adaptive remeshing workflows
- `notebooks/`: exploratory workflow notebooks
- `tools/topo/`: TopoJAX-owned tooling
- `tools/common/`: shared repository tooling
- `docs/topo/`: architectural, implementation, theory, and status documentation
- `contracts/topo/`: binding API and runtime guarantees
- `outputs/topo/`: retained demo and benchmark outputs

## Documentation Model

The repository uses `docs/topo/documentation_governance.md` as the placement standard for new documentation.

- use `docs/topo/specs/` for semantic mesh and object definitions
- use `contracts/topo/` for runtime and API obligations
- use `docs/topo/implementation/` for workflow notes, implementation mapping, and methodology
- use `docs/topo/status/` for roadmaps, release checks, and current-state summaries

## Current State

The source tree and example workflows are already established, and the governed documentation structure is now in active use.

Current runtime emphasis:

- Mode 1 fixed-topology workflows now cover practical 1D, 2D, and 3D create or import, optimize, export, and view paths
- restart, surrogate, and dynamic mode planning are treated as the next layer on top of that fixed-topology baseline

The root `docs/` folder is now only a namespace entry point for `docs/common/`, `docs/topo/`, and `docs/smpl/`.
