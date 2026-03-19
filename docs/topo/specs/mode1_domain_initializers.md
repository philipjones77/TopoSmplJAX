# Mode 1 Domain Initializer Semantics

Status: active
Date: 2026-03-17

## Scope

This document defines the semantic contract for Mode 1 domain initialization and import helpers.

## Shared Contract

A Mode 1 domain initializer is any public entry point that returns a point array and a `MeshTopology` suitable for fixed-topology optimization.

These initializers must:

- return a fixed node count and fixed element connectivity after initialization
- return point coordinates whose leading dimension matches `topology.n_nodes`
- return primary elements compatible with the Mode 1 quality objective for that topology kind
- preserve exported or imported entity tags when tagged metadata is part of the initializer surface
- avoid in-run topology mutation once initialization has completed

Tagged initializers may additionally return `DomainMeshMetadata`, which carries:

- `boundary_element_blocks`: extra Gmsh-compatible lower-dimensional blocks for export
- `physical_names`: stable `(dimension, physical_tag) -> name` mappings for those blocks

## Workflow Initializer Kinds

`initialize_mode1_domain` is the high-level dispatcher. The current supported `kind` values are:

- `line`
  Builds an open or closed polyline, or a unit-interval line mesh when explicit points are not provided.

- `polygon`
  Builds a triangle mesh for a polygonal domain with optional polygonal holes.

- `polygon-quad`
  Builds a conforming quad mesh for a polygonal domain by subdividing a recovered triangle mesh.

- `box-volume`
  Builds a structured tetrahedral fill for an axis-aligned box.

- `implicit-volume`
  Builds a tetrahedral mesh by filtering a structured tet background grid with a user level-set function.

- `sphere-volume`
  Builds a tetrahedral sphere mesh by specializing the implicit-volume contract to a sphere signed-distance field inside a computed bounding box.

- `extruded`
  Builds a tetrahedral mesh by extruding a polygon-domain triangle mesh through a fixed layer count.

- `import-msh`
  Imports a supported fixed mesh from a Gmsh MSH 2.2 file and reconstructs the corresponding Mode 1 topology and metadata.

## Semantics By Family

Line and 2D polygon initializers:

- produce line, triangle, or quad primary elements in 2D coordinates
- may export tagged line boundary blocks for outer loops and hole loops

3D volume initializers:

- produce tetrahedral primary elements in 3D coordinates
- may export tagged triangle boundary blocks for surfaces such as box sides, extruded walls, or implicit boundaries

Imported MSH initializers:

- support line, triangle, quad, and tetra primary blocks
- preserve physical-name mappings and non-primary element blocks that are relevant to later export or viewing workflows

## Non-Goals

These initializers are practical fixed-topology entry points. They do not imply:

- CAD-kernel parity with original Gmsh
- reverse-mode differentiation through topology mutation or remeshing
- broad support for hex, prism, or pyramid primary elements
