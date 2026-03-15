# Mesh Movement Modes Implementation

This note maps the five mesh movement modes onto the current GmshJAX implementation.

## Mode 1: Fixed Topology AD

Primary code paths:

- `src/gmshjax/ad/compiled.py`
- `src/gmshjax/ad/pipeline.py`
- `src/gmshjax/mesh/operators.py`

This mode is the production baseline. Node coordinates move, while topology is held fixed.

Current initialization scope includes first-class 1D lines, arbitrary 2D polygon domains, extruded polygonal 3D domains, implicit-volume tetrahedral backgrounds, and imported fixed meshes. The concrete initializers are cataloged in `docs/objects/mode1_domain_initializers.md`.

## Mode 2: Remesh Restart

Primary code paths:

- `src/gmshjax/ad/restart.py`
- `src/gmshjax/mesh/adaptive.py`
- `src/gmshjax/mesh/adaptive_quad.py`
- `src/gmshjax/mesh/adaptive_tet.py`

This mode is implemented for triangle, quad, and tet workflows. It performs fixed-topology optimization, then a separate discrete remesh step, then restarts optimization on the new topology.

## Mode 3: Soft Connectivity Surrogate

Primary code paths:

- `src/gmshjax/ad/surrogate.py`

Current implementation scope is a fixed quad candidate graph with differentiable weights over diagonal choices.

## Mode 4: Straight-Through Connectivity

Primary code paths:

- `src/gmshjax/ad/straight_through.py`

Current implementation scope is also quad diagonal choice, but with hard forward decisions and surrogate backward gradients.

## Mode 5: Fully Dynamic Remeshing

Current status: aspirational.

There is no production implementation yet. The intended target is a workflow where complete remeshing is allowed while geometry is moving, without reducing the system to phase-local fixed-topology solves. For now, GmshJAX should treat this mode as a planning target rather than a supported runtime guarantee.

The concrete prototype plan, milestones, code targets, non-goals, and approximation strategy are tracked in `docs/status/mode5_roadmap.md`.
