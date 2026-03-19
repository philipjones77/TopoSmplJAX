# Mesh Movement Modes Implementation

This note maps the five mesh movement modes onto the current TopoJAX implementation.

## Mode 1: Fixed Topology AD

Primary code paths:

- `src/topojax/ad/compiled.py`
- `src/topojax/ad/pipeline.py`
- `src/topojax/mesh/operators.py`

This mode is the production baseline. Node coordinates move, while topology is held fixed.

Current initialization scope includes first-class 1D lines, arbitrary 2D polygon domains, extruded polygonal 3D domains, implicit-volume tetrahedral backgrounds, and imported fixed meshes. The concrete initializers are cataloged in `docs/topo/objects/mode1_domain_initializers.md`.

## Mode 2: Remesh Restart

Primary code paths:

- `src/topojax/ad/restart.py`
- `src/topojax/mesh/adaptive.py`
- `src/topojax/mesh/adaptive_quad.py`
- `src/topojax/mesh/adaptive_tet.py`

This mode is implemented for triangle, quad, and tet workflows. It performs fixed-topology optimization, then a separate discrete remesh step, then restarts optimization on the new topology.

The practical runtime surface now also includes:

- a shared workflow-domain initializer through `initialize_mode2_domain`
- a high-level restart runner through `run_mode2_restart_workflow`
- final-mesh and phase-history export through `export_mode2_artifacts`
- compact restart summaries through `summarize_mode2_restart_result`

The concrete current-state definition, boundaries, and TODO list for this mode are tracked in `docs/topo/status/mode2_roadmap.md`.

## Mode 3: Soft Connectivity Surrogate

Primary code paths:

- `src/topojax/ad/surrogate.py`

Current implementation scope is a fixed quad candidate graph with differentiable weights over diagonal choices.

## Mode 4: Straight-Through Connectivity

Primary code paths:

- `src/topojax/ad/straight_through.py`

Current implementation scope is also quad diagonal choice, but with hard forward decisions and surrogate backward gradients.

## Mode 5: Fully Dynamic Remeshing

Current status: aspirational.

There is no production implementation yet. The intended target is a workflow where complete remeshing is allowed while geometry is moving, without reducing the system to phase-local fixed-topology solves. For now, TopoJAX should treat this mode as a planning target rather than a supported runtime guarantee.

The concrete prototype plan, milestones, code targets, non-goals, and approximation strategy are tracked in `docs/topo/status/mode5_roadmap.md`.
