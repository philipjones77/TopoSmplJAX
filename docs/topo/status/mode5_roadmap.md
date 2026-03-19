# Mode 5 Prototype Roadmap

Status: active planning
Date: 2026-03-15

## Purpose

This document defines the prototype implementation roadmap for mode 5, `fully-dynamic-remeshing`.

Mode 5 is the aspirational workflow where complete remeshing is allowed while geometry is moving inside one optimization process. In the current repository, this is a planning target rather than a supported runtime guarantee.

## Target Outcome

The prototype target is not "differentiate through exact Gmsh-style remeshing".

The target is a practical hybrid system with these properties:

- geometry can move continuously during optimization
- remeshing decisions can occur inside the same high-level optimization loop
- topology changes are represented through approximations or surrogate state rather than naive reverse-mode through exact remeshing
- state transfer between mesh states is explicit and testable

## Explicit Non-Goals

The first prototype does not try to deliver:

- exact reverse-mode gradients through split, collapse, flip, or full retriangulation events
- full parity with original Gmsh meshing functionality across all geometry/entity types
- unrestricted CAD reprojection and reclassification on arbitrary geometry kernels
- production-grade dynamic remeshing for all triangle, quad, tet, prism, pyramid, and hex workflows
- solver-coupled PDE state transfer beyond basic mesh-to-mesh field projection primitives

## Approximation Strategy

Mode 5 will be approached through layered approximations rather than one monolithic remeshing operator.

### Layer A: Fixed-Topology Baseline

Keep the existing mode-1 and mode-2 stack as the stable optimization base:

- fixed-topology AD for geometry movement
- remesh-restart as the current production-safe discrete fallback

### Layer B: Fixed Candidate Graph Surrogates

Extend mode-3 and mode-4 style surrogates from quads to triangles and local tet reconnection patches:

- soft weights over local connectivity alternatives
- straight-through hard forward choices with surrogate backward gradients

This layer provides a differentiable proxy for local topology adaptation while keeping candidate sets fixed.

### Layer C: Mesh-to-Mesh State Transfer

Introduce explicit transfer operators between old and new meshes:

- point-associated field interpolation
- element-associated field projection or redistribution
- restart-safe state containers for moving between topology phases

This is required before dynamic remeshing can be meaningfully embedded in one optimization process.

### Layer D: Dynamic Controller

Add a unified driver that can choose among:

- stay on fixed topology
- use surrogate local topology updates
- trigger discrete remeshing with transfer and restart

The controller should be driven by quality, distortion, size-field, or surrogate-confidence thresholds.

### Layer E: Relaxed Mode-5 Prototype

The first true mode-5 prototype should treat remeshing as a controlled hybrid process:

- continuous geometry motion
- continuous surrogate topology variables where possible
- explicit discrete re-meshing events when controller thresholds are crossed
- state transfer after each discrete event

This is still not exact AD through full remeshing, but it is the nearest practical route to a dynamic in-loop remeshing workflow.

## Milestones

## M5.1: Metric and Field Infrastructure

Goal:

- add background size/metric fields that can drive remeshing decisions while geometry moves

Code targets:

- `src/topojax/mesh/fields.py`
- `src/topojax/mesh/metrics.py`
- `src/topojax/ad/controller.py`

Acceptance signals:

- fixed-shape metric evaluation in JAX
- tests for isotropic size fields and anisotropic placeholders
- examples showing geometry motion coupled to size-field evaluation

## M5.2: Mesh-to-Mesh Transfer

Goal:

- transfer state consistently across remeshing phases

Code targets:

- `src/topojax/transfer/point_fields.py`
- `src/topojax/transfer/element_fields.py`
- `src/topojax/transfer/state.py`

Acceptance signals:

- point-field interpolation tests across triangle and quad remeshes
- element-field redistribution tests for simple scalar fields
- restart examples preserving state after remeshing

## M5.3: Local Surrogate Expansion

Goal:

- extend experimental surrogate topology layers beyond quad diagonal choices

Code targets:

- `src/topojax/ad/surrogate.py`
- `src/topojax/ad/straight_through.py`
- `src/topojax/mesh/connectivity_opt.py`

Acceptance signals:

- triangle edge-flip surrogate candidates
- local tet reconnection surrogate candidates
- gradient tests for soft and straight-through variants

## M5.4: Unified Mode Driver

Goal:

- expose one controller that coordinates modes 1 through 5

Code targets:

- `src/topojax/ad/driver.py`
- `src/topojax/ad/modes.py`
- `src/topojax/ad/controller.py`

Acceptance signals:

- one public entry point for selecting workflow mode
- thresholds for switching between fixed, surrogate, and discrete remesh behavior
- regression tests verifying mode dispatch and restart behavior

## M5.5: Relaxed Dynamic Prototype

Goal:

- build the first end-to-end mode-5 prototype for moving 2D triangle meshes

Code targets:

- `src/topojax/ad/dynamic.py`
- `examples/topo/m5_dynamic_tri_prototype.py`
- `tests/topo/test_mode5_dynamic_proto.py`

Acceptance signals:

- moving geometry with in-loop controller decisions
- at least one surrogate-local topology phase and one discrete remesh-transfer phase
- stable example run with documented limitations

## Suggested Build Order

1. M5.2 mesh-to-mesh transfer
2. M5.3 local surrogate expansion
3. M5.4 unified mode driver
4. M5.1 metric and field infrastructure
5. M5.5 relaxed dynamic prototype

Reason:

- transfer and surrogate infrastructure are immediate dependencies for any credible dynamic prototype
- the driver is needed before the mode-5 workflow can be exercised coherently
- metric fields matter, but they are more useful after the transition and controller layers exist

## Current Gap vs Original Gmsh

Relative to original Gmsh-style dynamic meshing workflows, the most important missing pieces are:

- robust field-driven remeshing infrastructure
- entity-aware reprojection and classification during topology changes
- state transfer across mesh changes
- broad local reconnection families beyond the currently implemented operators
- controller logic for mixing continuous motion and remeshing

## Exit Criteria For "Prototype Complete"

The first mode-5 prototype is complete when all of the following are true:

- there is one documented and tested 2D dynamic workflow
- geometry motion occurs continuously within an optimization loop
- local surrogate connectivity updates are exercised in that loop
- at least one discrete remesh-transfer-restart event occurs in the same example
- limitations are documented explicitly rather than hidden behind mode names

## Related Documents

- `docs/topo/objects/mesh_movement_modes.md`
- `docs/topo/theory/mesh_movement_modes.md`
- `docs/topo/implementation/mesh_movement_modes.md`
