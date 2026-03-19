# Mode 2 Remesh-Restart Roadmap

Status: active
Date: 2026-03-17

## Purpose

This document defines Mode 2, `remesh-restart`, as the practical piecewise-differentiable workflow that sits between Mode 1 fixed-topology optimization and the more experimental surrogate or dynamic modes.

## Definition

Mode 2 is a phased optimization workflow with this structure:

1. build or receive one concrete mesh topology
2. optimize geometry while that topology is fixed
3. stop the differentiable phase
4. run a discrete remeshing operator
5. restart optimization on the new topology

The full process is not globally differentiable across remeshing events. Differentiability applies only inside each fixed-topology phase.

## Current Runtime Surface

The current implemented Mode 2 surface is centered on:

- `optimize_points_fixed_topology`
- `optimize_remesh_restart_tri`
- `optimize_remesh_restart_quad`
- `optimize_remesh_restart_tet`
- `initialize_mode2_domain`
- `run_mode2_restart_workflow`
- `export_mode2_artifacts`
- `summarize_mode2_restart_result`
- `RestartPhase`
- `RestartTriOptimizationResult`
- `RestartQuadOptimizationResult`
- `RestartTetOptimizationResult`

Primary implementation modules:

- `src/topojax/ad/restart.py`
- `src/topojax/mesh/adaptive.py`
- `src/topojax/mesh/adaptive_quad.py`
- `src/topojax/mesh/adaptive_tet.py`

## Current Guarantees

Mode 2 currently guarantees:

- fixed-topology optimization within each phase
- explicit discrete remesh boundaries between phases
- support for triangle, quad, and tetra restart workflows
- phase summaries recording start energy, final energy, mesh size, and whether remeshing occurred
- artifact export for final mesh, per-phase history, summary metrics, and optional STL surface output
- one high-level workflow entry point comparable to Mode 1 for supported restart domains

## Current Boundaries

Mode 2 currently does not yet provide:

- explicit transfer objects for arbitrary user state across remesh phases
- preservation of geometric entity tags and boundary metadata through restart remeshing
- one unified public driver that coordinates Mode 1 and Mode 2 under the same workflow surface

## TODO List

### Near-Term

- add stricter tests for tri, quad, and tet restart workflows covering high-level workflow dispatch and artifact contents
- document the expected sizing and quality parameters for tri, quad, and tet restart workflows

### Metadata and Interchange

- preserve or explicitly reconstruct boundary element blocks and physical-name metadata across restart remesh phases where feasible
- define what entity-tag stability means after local split, collapse, or reconnection events
- add import or export round-trip coverage for restart outputs once Mode 2 artifact writers exist

### State Transfer

- define a restart-state container for point-associated and element-associated fields
- implement simple point-field transfer primitives that can survive a remesh-restart boundary
- keep transfer explicit rather than pretending remesh events are end-to-end differentiable

### API and Control

- unify Mode 1 and Mode 2 entry points behind a higher-level driver without hiding the discrete restart boundary
- define common diagnostics naming between Mode 1 optimization steps and Mode 2 restart phases
- expose clearer mode-selection guidance in docs and examples so users know when to choose fixed-topology versus remesh-restart

### Medium-Term

- connect Mode 2 restart-state transfer to the planned Mode 5 transition infrastructure
- add controller logic that decides when a Mode 1 run should escalate into a Mode 2 remesh-restart cycle
- add at least one end-to-end example for each of tri, quad, and tet restart workflows with exportable outputs

## Recommendation

Treat Mode 2 as the production-safe discrete fallback above Mode 1:

- use Mode 1 when fixed topology is acceptable and clean autodiff matters most
- use Mode 2 when mesh quality cannot be recovered without topology change, but the workflow can tolerate explicit restart boundaries

Do not describe Mode 2 as differentiating through remeshing. The accurate claim is phase-local autodiff with explicit discrete remesh and restart steps.
