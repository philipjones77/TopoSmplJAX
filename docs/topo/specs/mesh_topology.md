# Mesh Topology Spec

Status: active
Date: 2026-03-17

## Scope

This document defines the semantic contract for `MeshTopology` and for the canonical topology-construction helpers that feed fixed-topology workflows.

## `MeshTopology`

`MeshTopology` is the repository's immutable connectivity container for a single fixed mesh state.

Fields:

- `elements`: rank-2 `int32` array of shape `(n_elements, element_order)` containing node indices for each element
- `edges`: rank-2 `int32` array of shape `(n_edges, 2)` containing unique undirected edges induced by `elements`
- `node_ids`: rank-1 `int32` array of shape `(n_nodes,)` containing stable node identifiers
- `element_ids`: rank-1 `int32` array of shape `(n_elements,)` containing stable element identifiers
- `element_entity_tags`: rank-1 `int32` array of shape `(n_elements,)` containing the owning geometric entity tag for each primary element
- `n_nodes`: Python `int` equal to the number of mesh nodes addressable by `elements`

## Invariants

- `elements`, `edges`, `node_ids`, `element_ids`, and `element_entity_tags` are shape-stable for the life of a fixed-topology optimization run.
- Every entry in `elements` must be an integer node index in `[0, n_nodes)`.
- `node_ids` and `element_ids` are stable identifiers, not transient counters that may be renumbered during Mode 1 optimization.
- `edges` are undirected, unique, and sorted pairwise.
- `element_entity_tags.shape == (n_elements,)`.
- `n_nodes == points.shape[0]` for any point array paired with the topology.

## Supported Element Signatures

The current fixed-topology builders support these primary element signatures:

- line meshes: `(n_elements, 2)` with 2D or 3D point coordinates
- triangle meshes: `(n_elements, 3)` with 2D point coordinates
- quad meshes: `(n_elements, 4)` with 2D point coordinates
- tetrahedral meshes: `(n_elements, 4)` with 3D point coordinates

The point dimension and element arity jointly determine the supported topology kind for `mesh_topology_from_points_and_elements`.

## Construction Semantics

`mesh_topology_from_points_and_elements(points, elements, element_entity_tags=...)` is the canonical constructor for externally supplied fixed meshes.

It must:

- preserve the provided node ordering
- preserve the provided element ordering
- derive `edges` deterministically from the element connectivity
- default `element_entity_tags` to ones when explicit tags are not supplied
- raise `ValueError` for unsupported point-dimension and element-arity combinations

## Canonical Builder Roles

These helpers build `MeshTopology` instances intended for public fixed-topology workflows:

- `polyline_mesh`
- `unit_interval_line_mesh`
- `unit_square_tri_mesh`
- `unit_square_quad_mesh`
- `unit_cube_tet_mesh`
- the arbitrary-domain builders in `topojax.mesh.domains`

All of them must return point arrays and topologies that are mutually compatible with Mode 1 optimization and export helpers.
