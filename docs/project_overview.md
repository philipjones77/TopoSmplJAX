# Project Overview

GmshJAX is a JAX-first mesh creation and mesh adaptation workspace that keeps topology static while allowing coordinates, objectives, and deformation parameters to remain differentiable.

## Goals

- provide AD-friendly mesh generation and adaptation building blocks in JAX
- preserve Gmsh-inspired topology and entity concepts in an array-first runtime
- support both NumPy reference execution and JAX-compiled/autodiff execution
- keep exported meshes, diagnostics, and adaptive workflows reproducible through stable entry points

## Repository Shape

- `src/gmshjax/`: runtime implementation, model types, mutation kernels, and AD pipeline code
- `tests/`: regression and integration tests for generators, refinement, mutation, and adaptive loops
- `examples/`: runnable demos for boundary refinement, diagnostics, and adaptive remeshing workflows
- `notebooks/`: exploratory workflow notebooks
- `tools/`: repository tooling and maintenance scripts
- `docs/`: architectural, implementation, theory, and status documentation
- `contracts/`: binding API and runtime guarantees
- `results/`: legacy artifact output retained while demos and validation flows still write there

## Documentation Model

The repository uses `docs/documentation_governance.md` as the placement standard for new documentation.

- use `docs/specs/` for semantic mesh and object definitions
- use `contracts/` for runtime and API obligations
- use `docs/implementation/` for workflow notes, implementation mapping, and methodology
- use `docs/status/` for roadmaps, release checks, and current-state summaries

## Current State

The source tree and example workflows are already established, and the governed documentation structure is now in active use.

Current runtime emphasis:

- Mode 1 fixed-topology workflows now cover practical 1D, 2D, and 3D create or import, optimize, export, and view paths
- restart, surrogate, and dynamic mode planning are treated as the next layer on top of that fixed-topology baseline

Existing root-level docs such as `docs/overview.md` and `docs/milestone_m2_static_generator.md` remain in place as legacy documents until they are revised and moved deliberately.
