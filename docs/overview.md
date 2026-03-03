# AD Mesh Creation for JAX

## Scope

- Differentiable mesh construction primitives.
- JAX-native objective functions for adaptation.
- Export hooks for external meshing and solver pipelines.
- Gmsh-style model separation: static topology, dynamic coordinates.
- Dual backend support: NumPy (forward/reference) and JAX (compiled/autodiff).

## Gmsh Mapping

- `MVertex` concept: immutable `node_ids` with mutable coordinates in `MeshState.points`.
- `MElement` concept: immutable `element_ids` and element connectivity in `MeshTopology.elements`.
- `GEntity` concept: `dim/tag` descriptors with per-element `element_entity_tags`.
- Jacobian-quality spirit (`qualityMeasuresJacobian.cpp`): `ICN`/`IGE` metrics and smooth inversion penalties.

## Initial Roadmap

1. Add differentiable boundary parameterizations.
2. Extend from structured static generators to unstructured generation.
3. Add quality metrics with gradient checks.
