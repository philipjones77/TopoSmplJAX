"""RandomFields77 mesh-bridge adapters and mode-specific review stubs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.metadata
from typing import Any, Callable, Mapping, Sequence

import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import numpy as np

from topojax.ad._common import topology_cache_key
from topojax.ad.modes import MeshMovementMode, MeshMovementModeSpec, get_mesh_movement_mode
from topojax.io.exports import GmshElementBlock
from topojax.mesh.domains import DomainMeshMetadata
from topojax.mesh.operators import line_element_lengths, triangle_signed_areas
from topojax.mesh.topology import MeshTopology
from topojax.visualization import _points3


GeometryFn = Callable[[Mapping[str, Any] | None], jnp.ndarray]
GeodesicFn = Callable[[jnp.ndarray, MeshTopology], jnp.ndarray]
GraphWeightFn = Callable[[jnp.ndarray, jnp.ndarray, str], jnp.ndarray]
OperatorHook = Callable[[jnp.ndarray, MeshTopology], jnp.ndarray]


def _builder_version() -> str | None:
    try:
        return importlib.metadata.version("topojax")
    except importlib.metadata.PackageNotFoundError:
        return None


def _element_family(points: jnp.ndarray, topology: MeshTopology) -> str:
    order = int(topology.elements.shape[1])
    dim = int(points.shape[1])
    if order == 2:
        return "line"
    if order == 3:
        return "triangle"
    if order == 4 and dim == 2:
        return "quad"
    if order == 4 and dim == 3:
        return "tetra"
    raise ValueError("Unsupported point/element combination")


def _geometry_dim(family: str) -> int:
    if family == "line":
        return 1
    if family in ("triangle", "quad"):
        return 2
    if family == "tetra":
        return 3
    raise ValueError(f"Unsupported element family: {family}")


def _topology_id(topology: MeshTopology) -> str:
    payload = repr(topology_cache_key(topology)).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _topology_fixed_for_mode(mode: MeshMovementMode) -> bool:
    return mode == MeshMovementMode.FIXED_TOPOLOGY


def _boundary_blocks(metadata: DomainMeshMetadata | None, primary_dim: int) -> tuple[GmshElementBlock, ...]:
    if metadata is None:
        return ()
    blocks: list[GmshElementBlock] = []
    for block in metadata.boundary_element_blocks:
        dim = _geometry_dim(block.element_kind)
        if dim < primary_dim:
            blocks.append(block)
    return tuple(blocks)


def _concat_boundary_facets(blocks: Sequence[GmshElementBlock]) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
    if not blocks:
        return None, None
    first_width = int(np.asarray(blocks[0].elements).shape[1])
    if any(int(np.asarray(block.elements).shape[1]) != first_width for block in blocks):
        return None, None
    facets = jnp.concatenate([jnp.asarray(block.elements, dtype=jnp.int32) for block in blocks], axis=0)
    tags_list = []
    for block in blocks:
        if block.physical_tags is not None:
            tags_list.append(jnp.asarray(block.physical_tags, dtype=jnp.int32))
        elif block.geometrical_tags is not None:
            tags_list.append(jnp.asarray(block.geometrical_tags, dtype=jnp.int32))
        else:
            tags_list.append(jnp.ones((int(np.asarray(block.elements).shape[0]),), dtype=jnp.int32))
    return facets, jnp.concatenate(tags_list, axis=0)


def _count_tag_usage(tags: np.ndarray) -> dict[int, int]:
    if tags.size == 0:
        return {}
    uniq, counts = np.unique(tags.astype(np.int32, copy=False), return_counts=True)
    return {int(tag): int(count) for tag, count in zip(uniq, counts, strict=True)}


def _physical_groups(
    topology: MeshTopology,
    metadata: DomainMeshMetadata | None,
    *,
    primary_dim: int,
) -> dict[str, Any] | None:
    if metadata is None or not metadata.physical_names:
        return None

    grouped_counts: dict[tuple[int, int], int] = {}
    primary_counts = _count_tag_usage(np.asarray(topology.element_entity_tags))
    for tag, count in primary_counts.items():
        grouped_counts[(primary_dim, tag)] = grouped_counts.get((primary_dim, tag), 0) + count
    for block in metadata.boundary_element_blocks:
        dim = _geometry_dim(block.element_kind)
        tags = block.physical_tags if block.physical_tags is not None else block.geometrical_tags
        if tags is None:
            continue
        for tag, count in _count_tag_usage(np.asarray(tags)).items():
            grouped_counts[(dim, tag)] = grouped_counts.get((dim, tag), 0) + count

    out: dict[str, Any] = {}
    for key, name in sorted(metadata.physical_names.items(), key=lambda item: item[1]):
        out[name] = {
            "dim": int(key[0]),
            "tag": int(key[1]),
            "count": int(grouped_counts.get(key, 0)),
        }
    return out


def _boundary_groups(
    metadata: DomainMeshMetadata | None,
    *,
    primary_dim: int,
) -> dict[str, Any] | None:
    if metadata is None or not metadata.physical_names:
        return None
    counts: dict[tuple[int, int], int] = {}
    for block in _boundary_blocks(metadata, primary_dim):
        dim = _geometry_dim(block.element_kind)
        tags = block.physical_tags if block.physical_tags is not None else block.geometrical_tags
        if tags is None:
            continue
        for tag, count in _count_tag_usage(np.asarray(tags)).items():
            counts[(dim, tag)] = counts.get((dim, tag), 0) + count

    out: dict[str, Any] = {}
    for key, name in sorted(metadata.physical_names.items(), key=lambda item: item[1]):
        if key[0] >= primary_dim:
            continue
        out[name] = {
            "dim": int(key[0]),
            "tag": int(key[1]),
            "count": int(counts.get(key, 0)),
        }
    return out or None


def _edge_weights(points: jnp.ndarray, edges: jnp.ndarray, weight_mode: str) -> jnp.ndarray:
    if weight_mode == "unit":
        return jnp.ones((edges.shape[0],), dtype=points.dtype)
    lengths = jnp.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    if weight_mode == "distance":
        return lengths
    if weight_mode == "inverse_distance":
        return 1.0 / jnp.maximum(lengths, jnp.asarray(1.0e-12, dtype=points.dtype))
    raise ValueError(f"Unsupported weight_mode: {weight_mode}")


def _graph_laplacian_dense(points: jnp.ndarray, edges: jnp.ndarray, weight_mode: str) -> jnp.ndarray:
    n = int(points.shape[0])
    src = edges[:, 0]
    dst = edges[:, 1]
    weights = _edge_weights(points, edges, weight_mode)
    mat = jnp.zeros((n, n), dtype=points.dtype)
    mat = mat.at[src, src].add(weights)
    mat = mat.at[dst, dst].add(weights)
    mat = mat.at[src, dst].add(-weights)
    mat = mat.at[dst, src].add(-weights)
    return mat


def _graph_laplacian_sparse(points: jnp.ndarray, edges: jnp.ndarray, weight_mode: str) -> BCOO:
    src = jnp.asarray(edges[:, 0], dtype=jnp.int32)
    dst = jnp.asarray(edges[:, 1], dtype=jnp.int32)
    weights = _edge_weights(points, edges, weight_mode)
    indices = jnp.concatenate(
        [
            jnp.stack([src, src], axis=1),
            jnp.stack([dst, dst], axis=1),
            jnp.stack([src, dst], axis=1),
            jnp.stack([dst, src], axis=1),
        ],
        axis=0,
    )
    data = jnp.concatenate([weights, weights, -weights, -weights], axis=0)
    out = BCOO((data, indices), shape=(int(points.shape[0]), int(points.shape[0])))
    return out.sum_duplicates()


def _quad_signed_areas(points: jnp.ndarray, elements: jnp.ndarray) -> jnp.ndarray:
    q = points[elements]
    a = q[:, 0, :]
    b = q[:, 1, :]
    c = q[:, 2, :]
    d = q[:, 3, :]
    area1 = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    area2 = 0.5 * ((c[:, 0] - a[:, 0]) * (d[:, 1] - a[:, 1]) - (c[:, 1] - a[:, 1]) * (d[:, 0] - a[:, 0]))
    return area1 + area2


def _tet_signed_volumes(points: jnp.ndarray, elements: jnp.ndarray) -> jnp.ndarray:
    t = points[elements]
    a = t[:, 0, :]
    b = t[:, 1, :]
    c = t[:, 2, :]
    d = t[:, 3, :]
    e1 = b - a
    e2 = c - a
    e3 = d - a
    det = (
        e1[:, 0] * (e2[:, 1] * e3[:, 2] - e2[:, 2] * e3[:, 1])
        - e1[:, 1] * (e2[:, 0] * e3[:, 2] - e2[:, 2] * e3[:, 0])
        + e1[:, 2] * (e2[:, 0] * e3[:, 1] - e2[:, 1] * e3[:, 0])
    )
    return det / 6.0


def _cell_measures(points: jnp.ndarray, topology: MeshTopology) -> jnp.ndarray:
    family = _element_family(points, topology)
    if family == "line":
        return line_element_lengths(points, topology.elements)
    if family == "triangle":
        return jnp.abs(triangle_signed_areas(points, topology.elements))
    if family == "quad":
        return jnp.abs(_quad_signed_areas(points, topology.elements))
    if family == "tetra":
        return jnp.abs(_tet_signed_volumes(points, topology.elements))
    raise ValueError(f"Unsupported family: {family}")


def _mass_matrix(points: jnp.ndarray, topology: MeshTopology) -> jnp.ndarray:
    measures = _cell_measures(points, topology)
    order = int(topology.elements.shape[1])
    contrib = jnp.repeat(measures / order, order)
    node_mass = jnp.zeros((int(points.shape[0]),), dtype=points.dtype)
    node_mass = node_mass.at[topology.elements.reshape(-1)].add(contrib)
    return jnp.diag(node_mass)


def _normalize_mode(mode: MeshMovementMode | str) -> MeshMovementMode:
    return MeshMovementMode(mode)


def _pyvista_payload(points: jnp.ndarray, topology: MeshTopology) -> dict[str, Any]:
    pts3 = _points3(points)
    family = _element_family(points, topology)
    order = int(topology.elements.shape[1])
    dataset_kind = "unstructured_grid" if family == "tetra" else "polydata"
    return {
        "dataset_kind": dataset_kind,
        "points": pts3,
        "cells": jnp.asarray(topology.elements, dtype=jnp.int32),
        "cell_kind": family,
        "cell_order": order,
        "edge_index": jnp.asarray(topology.edges, dtype=jnp.int32),
    }


def _gmsh_native_payload(
    points: jnp.ndarray,
    topology: MeshTopology,
    metadata: DomainMeshMetadata | None,
) -> dict[str, Any]:
    blocks = []
    if metadata is not None:
        for block in metadata.boundary_element_blocks:
            blocks.append(
                {
                    "elements": jnp.asarray(block.elements, dtype=jnp.int32),
                    "element_kind": block.element_kind,
                    "physical_tags": None if block.physical_tags is None else jnp.asarray(block.physical_tags, dtype=jnp.int32),
                    "geometrical_tags": None
                    if block.geometrical_tags is None
                    else jnp.asarray(block.geometrical_tags, dtype=jnp.int32),
                }
            )
    return {
        "points": jnp.asarray(points),
        "elements": jnp.asarray(topology.elements, dtype=jnp.int32),
        "element_entity_tags": jnp.asarray(topology.element_entity_tags, dtype=jnp.int32),
        "element_kind": _element_family(points, topology),
        "extra_element_blocks": blocks,
        "physical_names": None if metadata is None else dict(metadata.physical_names),
    }


def _mode_contract(spec: MeshMovementModeSpec) -> dict[str, Any]:
    base = {
        "mode": spec.mode.value,
        "summary": spec.summary,
        "connectivity": spec.connectivity,
        "autodiff_status": spec.autodiff_status,
        "implementation_status": spec.implementation_status,
    }
    if spec.mode == MeshMovementMode.FIXED_TOPOLOGY:
        base["hook_points"] = [
            "geometry_fn(params) -> nodes",
            "geodesic_fn(points, topology) -> dense geodesic matrix",
            "operator_hooks['mass'|'stiffness'|'laplacian'](points, topology)",
        ]
    elif spec.mode == MeshMovementMode.REMESH_RESTART:
        base["hook_points"] = [
            "geometry_fn(params) -> nodes for phase-local exports",
            "builder_options['restart_phases'] for per-phase review",
            "operator_hooks for phase-local operators after each remesh",
        ]
    elif spec.mode == MeshMovementMode.SOFT_CONNECTIVITY:
        base["hook_points"] = [
            "builder_options['candidate_graph'] for the fixed connectivity candidate set",
            "builder_options['soft_weights'] or external surrogate state",
            "operator_hooks to expose surrogate-weighted operators",
        ]
    elif spec.mode == MeshMovementMode.STRAIGHT_THROUGH:
        base["hook_points"] = [
            "builder_options['candidate_graph'] for forward connectivity choices",
            "builder_options['forward_state'] for hard decisions",
            "builder_options['backward_surrogate'] for surrogate-gradient review",
        ]
    else:
        base["hook_points"] = [
            "builder_options['dynamic_remesh_fn'] for topology mutation",
            "builder_options['state_transfer_fn'] for field transfer across remeshes",
            "builder_options['controller_fn'] for adaptive mode and remesh control",
        ]
    return base


@dataclass
class RandomFields77ModeBridge:
    """Mode-aware adapter exposing a common RF77 mesh/operator API."""

    points: jnp.ndarray
    topology: MeshTopology
    mode: MeshMovementMode
    metadata: DomainMeshMetadata | None = None
    mesh_source: str = "topojax"
    builder_options: dict[str, Any] | None = None
    geometry_fn: GeometryFn | None = None
    geodesic_fn: GeodesicFn | None = None
    graph_weight_fn: GraphWeightFn | None = None
    operator_hooks: Mapping[str, OperatorHook] | None = None
    contract: dict[str, Any] | None = None
    _operator_cache: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def _mode_spec(self) -> MeshMovementModeSpec:
        return get_mesh_movement_mode(self.mode)

    def _resolve_points(self, params: Mapping[str, Any] | None = None) -> jnp.ndarray:
        if self.geometry_fn is None:
            return jnp.asarray(self.points)
        return jnp.asarray(self.geometry_fn(params))

    def topology_id(self) -> str:
        return _topology_id(self.topology)

    def shape_signature(self) -> dict[str, Any]:
        family = _element_family(self.points, self.topology)
        primary_dim = _geometry_dim(family)
        facets, _ = _concat_boundary_facets(_boundary_blocks(self.metadata, primary_dim))
        return {
            "mode": self.mode.value,
            "topology_id": self.topology_id(),
            "nodes": tuple(int(v) for v in self.points.shape),
            "cells": tuple(int(v) for v in self.topology.elements.shape),
            "edges": tuple(int(v) for v in self.topology.edges.shape),
            "facets": None if facets is None else tuple(int(v) for v in facets.shape),
            "embedding_dim": int(self.points.shape[1]),
            "geometry_dim": primary_dim,
            "element_family": family,
        }

    def physical_groups(self) -> dict[str, Any] | None:
        family = _element_family(self.points, self.topology)
        return _physical_groups(self.topology, self.metadata, primary_dim=_geometry_dim(family))

    def boundary_tags(self) -> dict[str, Any] | None:
        family = _element_family(self.points, self.topology)
        return _boundary_groups(self.metadata, primary_dim=_geometry_dim(family))

    def edge_index(self) -> jnp.ndarray:
        return jnp.asarray(self.topology.edges, dtype=jnp.int32)

    def _graph_weights(self, points: jnp.ndarray, weight_mode: str) -> jnp.ndarray:
        if self.graph_weight_fn is not None:
            return jnp.asarray(self.graph_weight_fn(points, self.edge_index(), weight_mode))
        return _edge_weights(points, self.edge_index(), weight_mode)

    def graph_laplacian_dense(self, weight_mode: str = "unit") -> jnp.ndarray:
        key = f"laplacian_dense:{weight_mode}"
        if key not in self._operator_cache:
            points = self._resolve_points()
            self._operator_cache[key] = {
                "kind": "dense",
                "weight_mode": weight_mode,
                "matrix": _graph_laplacian_dense(points, self.edge_index(), weight_mode),
            }
        return self._operator_cache[key]["matrix"]

    def graph_laplacian_sparse(self, weight_mode: str = "unit") -> BCOO:
        key = f"laplacian_sparse:{weight_mode}"
        if key not in self._operator_cache:
            points = self._resolve_points()
            self._operator_cache[key] = {
                "kind": "sparse",
                "weight_mode": weight_mode,
                "matrix": _graph_laplacian_sparse(points, self.edge_index(), weight_mode),
            }
        return self._operator_cache[key]["matrix"]

    def stiffness_matrix(self) -> jnp.ndarray:
        if self.operator_hooks is not None and "stiffness" in self.operator_hooks:
            return jnp.asarray(self.operator_hooks["stiffness"](self._resolve_points(), self.topology))
        key = "stiffness"
        if key not in self._operator_cache:
            self._operator_cache[key] = {
                "kind": "dense",
                "matrix": self.graph_laplacian_dense(weight_mode="inverse_distance"),
            }
        return self._operator_cache[key]["matrix"]

    def mass_matrix(self) -> jnp.ndarray:
        if self.operator_hooks is not None and "mass" in self.operator_hooks:
            return jnp.asarray(self.operator_hooks["mass"](self._resolve_points(), self.topology))
        key = "mass"
        if key not in self._operator_cache:
            self._operator_cache[key] = {
                "kind": "dense",
                "matrix": _mass_matrix(self._resolve_points(), self.topology),
            }
        return self._operator_cache[key]["matrix"]

    def operator_apply(self, x, operator: str = "laplacian"):
        vec = jnp.asarray(x)
        if self.operator_hooks is not None and operator in self.operator_hooks:
            mat = jnp.asarray(self.operator_hooks[operator](self._resolve_points(), self.topology))
            return mat @ vec
        if operator == "laplacian":
            return self.graph_laplacian_dense(weight_mode="unit") @ vec
        if operator == "mass":
            return self.mass_matrix() @ vec
        if operator == "stiffness":
            return self.stiffness_matrix() @ vec
        raise ValueError(f"Unsupported operator: {operator}")

    def cached_operator_state(self) -> dict[str, Any]:
        return {key: dict(value) for key, value in self._operator_cache.items()}

    def to_randomfields77_graph_payload(self, weight_mode: str = "unit") -> dict[str, Any]:
        points = self._resolve_points()
        return {
            "topology_id": self.topology_id(),
            "mode": self.mode.value,
            "nodes": points,
            "edge_index": self.edge_index(),
            "edge_weight": self._graph_weights(points, weight_mode),
            "graph_laplacian": self.graph_laplacian_dense(weight_mode=weight_mode),
            "weight_mode": weight_mode,
        }

    def _mesh_payload(self, points: jnp.ndarray) -> dict[str, Any]:
        family = _element_family(points, self.topology)
        primary_dim = _geometry_dim(family)
        boundary_blocks = _boundary_blocks(self.metadata, primary_dim)
        facets, facet_tags = _concat_boundary_facets(boundary_blocks)
        geodesic_matrix = None if self.geodesic_fn is None else jnp.asarray(self.geodesic_fn(points, self.topology))
        return {
            "nodes": jnp.asarray(points),
            "cells": jnp.asarray(self.topology.elements, dtype=jnp.int32),
            "cell_tags": jnp.asarray(self.topology.element_entity_tags, dtype=jnp.int32),
            "facets": facets,
            "facet_tags": facet_tags,
            "geodesic_matrix": geodesic_matrix,
            "invariants": ["topology_id", "shape_signature"] if _topology_fixed_for_mode(self.mode) else ["mode_contract"],
            "mesh_storage": {
                "canonical_format": "rf77.mesh_payload.v1",
                "mesh_source": self.mesh_source,
                "mesh_builder": "topojax",
                "mesh_builder_version": _builder_version(),
                "mesh_builder_options": None if self.builder_options is None else dict(self.builder_options),
                "runtime_backends": ["numpy", "jax"],
            },
            "metadata": {
                "physical_groups": self.physical_groups(),
                "boundary_groups": self.boundary_tags(),
                "element_family": family,
                "geometry_dim": primary_dim,
                "embedding_dim": int(points.shape[1]),
                "topology_fixed": _topology_fixed_for_mode(self.mode),
            },
        }

    def to_randomfields77_mesh_payload(self) -> dict[str, Any]:
        return self._mesh_payload(self._resolve_points())

    def to_randomfields77_dynamic_mesh_state(self, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._mesh_payload(self._resolve_points(params))

    def batch_to_randomfields77_dynamic_mesh_state(self, params_batch) -> dict[str, Any]:
        if self.geometry_fn is None:
            batch_size = 1 if params_batch is None else len(params_batch)
            nodes = jnp.broadcast_to(
                jnp.asarray(self.points),
                (batch_size, int(self.points.shape[0]), int(self.points.shape[1])),
            )
            out = self.to_randomfields77_mesh_payload()
            out["nodes"] = nodes
            return out
        if isinstance(params_batch, Mapping):
            leaves = [jnp.asarray(value) for value in params_batch.values()]
            if not leaves:
                raise ValueError("params_batch mapping must not be empty")
            batch_size = int(leaves[0].shape[0])
            params_seq = [{key: jnp.asarray(value[index]) for key, value in params_batch.items()} for index in range(batch_size)]
        else:
            params_seq = list(params_batch)
            batch_size = len(params_seq)
        nodes = jnp.stack([self._resolve_points(params) for params in params_seq], axis=0)
        out = self.to_randomfields77_mesh_payload()
        out["nodes"] = nodes
        return out

    def to_randomfields77_common_dataset(self, values, *, name: str = "values", location: str = "nodes") -> dict[str, Any]:
        return {
            "topology_id": self.topology_id(),
            "mode": self.mode.value,
            "location": location,
            "name": name,
            "values": jnp.asarray(values),
            "mesh": self.to_randomfields77_mesh_payload(),
        }

    def to_randomfields77_common_graph(self, *, values=None, weight_mode: str = "unit") -> dict[str, Any]:
        payload = self.to_randomfields77_graph_payload(weight_mode=weight_mode)
        payload["values"] = None if values is None else jnp.asarray(values)
        return payload

    def to_pyvista_payload(self) -> dict[str, Any]:
        return _pyvista_payload(self._resolve_points(), self.topology)

    def to_gmsh_native_payload(self) -> dict[str, Any]:
        return _gmsh_native_payload(self._resolve_points(), self.topology, self.metadata)

    def mode_contract(self) -> dict[str, Any]:
        return dict(self.contract or _mode_contract(self._mode_spec()))

    def runtime_report(self) -> dict[str, Any]:
        spec = self._mode_spec()
        supports_dynamic_geometry = self.geometry_fn is not None
        supports_geodesics = self.geodesic_fn is not None
        supports_batched_export = True
        return {
            "mode": spec.mode.value,
            "mode_summary": spec.summary,
            "implementation_status": spec.implementation_status,
            "autodiff_status": spec.autodiff_status,
            "topology_id": self.topology_id(),
            "shape_signature": self.shape_signature(),
            "backend_support": ["numpy", "jax"],
            "ad_support": {
                "geometry_motion": spec.geometry_motion,
                "autodiff_status": spec.autodiff_status,
                "topology_fixed": _topology_fixed_for_mode(self.mode),
            },
            "batching_support": {
                "supports_batched_export": supports_batched_export,
                "supports_dynamic_geometry": supports_dynamic_geometry,
            },
            "capability_flags": {
                "supports_numpy": True,
                "supports_jax": True,
                "supports_batched_export": supports_batched_export,
                "supports_dynamic_geometry": supports_dynamic_geometry,
                "supports_geodesics": supports_geodesics,
                "supports_graph_export": True,
                "supports_pyvista": True,
                "supports_gmsh": True,
            },
            "mode_contract": self.mode_contract(),
        }


def build_randomfields77_bridge(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    mode: MeshMovementMode | str,
    metadata: DomainMeshMetadata | None = None,
    mesh_source: str | None = None,
    builder_options: dict[str, Any] | None = None,
    geometry_fn: GeometryFn | None = None,
    geodesic_fn: GeodesicFn | None = None,
    graph_weight_fn: GraphWeightFn | None = None,
    operator_hooks: Mapping[str, OperatorHook] | None = None,
    contract: dict[str, Any] | None = None,
) -> RandomFields77ModeBridge:
    normalized_mode = _normalize_mode(mode)
    return RandomFields77ModeBridge(
        points=jnp.asarray(points),
        topology=topology,
        mode=normalized_mode,
        metadata=metadata,
        mesh_source=mesh_source or f"topojax:{normalized_mode.value}",
        builder_options=builder_options,
        geometry_fn=geometry_fn,
        geodesic_fn=geodesic_fn,
        graph_weight_fn=graph_weight_fn,
        operator_hooks=operator_hooks,
        contract=contract or _mode_contract(get_mesh_movement_mode(normalized_mode)),
    )


def build_mode1_randomfields77_bridge(
    source,
    *,
    geometry_fn: GeometryFn | None = None,
    geodesic_fn: GeodesicFn | None = None,
    graph_weight_fn: GraphWeightFn | None = None,
    operator_hooks: Mapping[str, OperatorHook] | None = None,
    builder_options: dict[str, Any] | None = None,
) -> RandomFields77ModeBridge:
    if hasattr(source, "domain") and hasattr(source, "result"):
        points = source.result.points
        topology = source.result.topology
        metadata = source.domain.metadata
        mesh_source = "topojax:mode1-workflow"
    else:
        points = source.points
        topology = source.topology
        metadata = source.metadata
        mesh_source = "topojax:mode1-domain"
    return build_randomfields77_bridge(
        points,
        topology,
        mode=MeshMovementMode.FIXED_TOPOLOGY,
        metadata=metadata,
        mesh_source=mesh_source,
        builder_options=builder_options,
        geometry_fn=geometry_fn,
        geodesic_fn=geodesic_fn,
        graph_weight_fn=graph_weight_fn,
        operator_hooks=operator_hooks,
    )


def build_mode2_randomfields77_bridge(
    source,
    *,
    geometry_fn: GeometryFn | None = None,
    geodesic_fn: GeodesicFn | None = None,
    graph_weight_fn: GraphWeightFn | None = None,
    operator_hooks: Mapping[str, OperatorHook] | None = None,
    builder_options: dict[str, Any] | None = None,
) -> RandomFields77ModeBridge:
    local_builder_options = {} if builder_options is None else dict(builder_options)
    if hasattr(source, "domain") and hasattr(source, "result"):
        points = source.result.points
        if int(source.result.elements.shape[1]) == 3:
            from topojax.ad.restart import triangle_topology_from_elements

            topology = triangle_topology_from_elements(source.result.elements, n_nodes=int(points.shape[0]))
        elif int(source.result.elements.shape[1]) == 4 and int(points.shape[1]) == 2:
            from topojax.ad.restart import quad_topology_from_elements

            topology = quad_topology_from_elements(source.result.elements, n_nodes=int(points.shape[0]))
        else:
            from topojax.ad.restart import tet_topology_from_elements

            topology = tet_topology_from_elements(source.result.elements, n_nodes=int(points.shape[0]))
        metadata = source.domain.metadata if not any(phase.remeshed for phase in source.result.phases) else None
        local_builder_options["restart_phases"] = [phase._asdict() for phase in source.result.phases]
        mesh_source = "topojax:mode2-workflow"
    else:
        points = source.points
        topology = source.topology
        metadata = source.metadata
        mesh_source = "topojax:mode2-domain"
    return build_randomfields77_bridge(
        points,
        topology,
        mode=MeshMovementMode.REMESH_RESTART,
        metadata=metadata,
        mesh_source=mesh_source,
        builder_options=local_builder_options,
        geometry_fn=geometry_fn,
        geodesic_fn=geodesic_fn,
        graph_weight_fn=graph_weight_fn,
        operator_hooks=operator_hooks,
    )


def build_mode3_randomfields77_bridge(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    metadata: DomainMeshMetadata | None = None,
    geometry_fn: GeometryFn | None = None,
    geodesic_fn: GeodesicFn | None = None,
    graph_weight_fn: GraphWeightFn | None = None,
    operator_hooks: Mapping[str, OperatorHook] | None = None,
    candidate_graph=None,
    soft_weights=None,
    builder_options: dict[str, Any] | None = None,
) -> RandomFields77ModeBridge:
    local_options = {} if builder_options is None else dict(builder_options)
    local_options.setdefault("candidate_graph", candidate_graph)
    local_options.setdefault("soft_weights", soft_weights)
    local_options.setdefault("review_status", "stub")
    return build_randomfields77_bridge(
        points,
        topology,
        mode=MeshMovementMode.SOFT_CONNECTIVITY,
        metadata=metadata,
        mesh_source="topojax:mode3-stub",
        builder_options=local_options,
        geometry_fn=geometry_fn,
        geodesic_fn=geodesic_fn,
        graph_weight_fn=graph_weight_fn,
        operator_hooks=operator_hooks,
    )


def build_mode4_randomfields77_bridge(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    metadata: DomainMeshMetadata | None = None,
    geometry_fn: GeometryFn | None = None,
    geodesic_fn: GeodesicFn | None = None,
    graph_weight_fn: GraphWeightFn | None = None,
    operator_hooks: Mapping[str, OperatorHook] | None = None,
    candidate_graph=None,
    forward_state=None,
    backward_surrogate=None,
    builder_options: dict[str, Any] | None = None,
) -> RandomFields77ModeBridge:
    local_options = {} if builder_options is None else dict(builder_options)
    local_options.setdefault("candidate_graph", candidate_graph)
    local_options.setdefault("forward_state", forward_state)
    local_options.setdefault("backward_surrogate", backward_surrogate)
    local_options.setdefault("review_status", "stub")
    return build_randomfields77_bridge(
        points,
        topology,
        mode=MeshMovementMode.STRAIGHT_THROUGH,
        metadata=metadata,
        mesh_source="topojax:mode4-stub",
        builder_options=local_options,
        geometry_fn=geometry_fn,
        geodesic_fn=geodesic_fn,
        graph_weight_fn=graph_weight_fn,
        operator_hooks=operator_hooks,
    )


def build_mode5_randomfields77_bridge(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    metadata: DomainMeshMetadata | None = None,
    geometry_fn: GeometryFn | None = None,
    geodesic_fn: GeodesicFn | None = None,
    graph_weight_fn: GraphWeightFn | None = None,
    operator_hooks: Mapping[str, OperatorHook] | None = None,
    dynamic_remesh_fn=None,
    state_transfer_fn=None,
    controller_fn=None,
    builder_options: dict[str, Any] | None = None,
) -> RandomFields77ModeBridge:
    local_options = {} if builder_options is None else dict(builder_options)
    local_options.setdefault("dynamic_remesh_fn", dynamic_remesh_fn)
    local_options.setdefault("state_transfer_fn", state_transfer_fn)
    local_options.setdefault("controller_fn", controller_fn)
    local_options.setdefault("review_status", "stub")
    return build_randomfields77_bridge(
        points,
        topology,
        mode=MeshMovementMode.FULLY_DYNAMIC,
        metadata=metadata,
        mesh_source="topojax:mode5-stub",
        builder_options=local_options,
        geometry_fn=geometry_fn,
        geodesic_fn=geodesic_fn,
        graph_weight_fn=graph_weight_fn,
        operator_hooks=operator_hooks,
    )
