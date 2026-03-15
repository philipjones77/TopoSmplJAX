# Mesh Movement Modes

This document is the canonical mode catalog for mesh movement and remeshing workflows in GmshJAX.

## Five Modes

1. `fixed-topology-ad`
   Move node coordinates while keeping connectivity fixed.
   AD status: full.
   Status: implemented.
   Initialization scope: 1D line meshes, structured templates, arbitrary polygonal 2D domains, selected arbitrary 3D domains, and imported fixed meshes.

2. `remesh-restart`
   Optimize with fixed topology, remesh as a separate discrete step, then restart on the new mesh.
   AD status: phase-local only.
   Status: implemented.

3. `soft-connectivity-surrogate`
   Optimize soft connectivity weights over a fixed candidate graph.
   AD status: surrogate.
   Status: experimental.

4. `straight-through-connectivity`
   Use hard connectivity choices in the forward pass and soft gradients in backward.
   AD status: straight-through surrogate.
   Status: experimental.

5. `fully-dynamic-remeshing`
   Allow complete remeshing while geometry is moving inside a single optimization process.
   AD status: aspirational.
   Status: planned.

## Authority

This file is a catalog of runtime-facing workflow modes. Theory belongs in `docs/theory/mesh_movement_modes.md`. Engineering mapping belongs in `docs/implementation/mesh_movement_modes.md`.
