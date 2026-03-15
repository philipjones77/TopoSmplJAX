"""Adaptive remeshing loop for triangle meshes."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from gmshjax.io.exports import export_snapshot_npz
from gmshjax.mesh.mutation import (
    TriMeshBuffer,
    active_elements,
    active_points,
    collapse_triangle,
    flip_diagonal,
    make_tri_mesh_buffer,
    split_triangle,
)
from gmshjax.mesh.operators import triangle_icn
from gmshjax.mesh.refine import triangle_area_magnitudes, triangle_refinement_priority
from gmshjax.mesh.topology import triangle_edges


class AdaptiveHistory(NamedTuple):
    iteration: int
    n_nodes: int
    n_elements: int
    mean_icn: float
    min_icn: float
    max_area: float
    did_split: bool
    did_flip: bool
    did_collapse: bool


def _neighbor_accum(points: jnp.ndarray, edges: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    n = points.shape[0]
    src = edges[:, 0]
    dst = edges[:, 1]
    neighbor_sum = jnp.zeros_like(points)
    neighbor_sum = neighbor_sum.at[src].add(points[dst])
    neighbor_sum = neighbor_sum.at[dst].add(points[src])
    deg = jnp.zeros((n,), dtype=points.dtype)
    one = jnp.ones((edges.shape[0],), dtype=points.dtype)
    deg = deg.at[src].add(one)
    deg = deg.at[dst].add(one)
    return neighbor_sum, deg


def _smooth_active_points(
    buffer: TriMeshBuffer,
    alpha: float,
    steps: int,
    movable_mask: jnp.ndarray | None,
) -> TriMeshBuffer:
    points = buffer.points
    elems = active_elements(buffer)
    edges = triangle_edges(elems)
    n = int(buffer.node_count)

    local_mask = jnp.zeros((points.shape[0],), dtype=bool).at[:n].set(True)
    if movable_mask is not None:
        local_mask = jnp.logical_and(local_mask, movable_mask)

    for _ in range(steps):
        neighbor_sum, deg = _neighbor_accum(points[:n], edges)
        mean_nbr = neighbor_sum / jnp.maximum(deg[:, None], 1.0)
        proposal = points[:n] + alpha * (mean_nbr - points[:n])
        updated = jnp.where(local_mask[:n, None], proposal, points[:n])
        points = points.at[:n].set(updated)

    return TriMeshBuffer(points, buffer.elements, buffer.active_nodes, buffer.active_elements, buffer.node_count, buffer.element_count)


def _find_first_shared_edge_pair(elements: np.ndarray) -> tuple[int, int] | None:
    edge_to_elem: dict[tuple[int, int], int] = {}
    for ei, t in enumerate(elements.tolist()):
        a, b, c = [int(v) for v in t]
        for e in [(a, b), (b, c), (c, a)]:
            ee = tuple(sorted(e))
            if ee in edge_to_elem:
                return edge_to_elem[ee], ei
            edge_to_elem[ee] = ei
    return None


def _collapse_candidate_node(buffer: TriMeshBuffer, target_area: float, min_nodes: int) -> int | None:
    if buffer.node_count <= min_nodes:
        return None

    points = np.asarray(active_points(buffer))
    elements = np.asarray(active_elements(buffer))
    collapse_limit = 0.35 * target_area

    for node_id in range(buffer.node_count - 1, min_nodes - 1, -1):
        incident_areas: list[float] = []
        incident_count = 0
        for tri in elements.tolist():
            if node_id not in tri:
                continue
            incident_count += 1
            pa, pb, pc = points[np.asarray(tri, dtype=np.int32)]
            area = 0.5 * abs((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]))
            incident_areas.append(float(area))
        if incident_count == 3 and incident_areas and max(incident_areas) <= collapse_limit:
            return node_id
    return None


def adaptive_remesh_tri(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    max_nodes: int,
    max_elements: int,
    max_iters: int = 30,
    target_area: float = 1.0e-3,
    target_mean_icn: float = 0.50,
    smoothing_alpha: float = 0.15,
    smoothing_steps: int = 2,
    movable_mask: jnp.ndarray | None = None,
    snapshot_dir: str | Path | None = None,
) -> tuple[TriMeshBuffer, list[AdaptiveHistory]]:
    """Run adaptive split/flip/smooth loop with fixed-capacity buffers."""
    buffer = make_tri_mesh_buffer(points, elements, max_nodes=max_nodes, max_elements=max_elements)
    min_nodes = int(buffer.node_count)
    history: list[AdaptiveHistory] = []
    snap_root = Path(snapshot_dir) if snapshot_dir is not None else None

    for it in range(max_iters):
        pts = active_points(buffer)
        elems = active_elements(buffer)

        icn = triangle_icn(pts, elems)
        areas = triangle_area_magnitudes(pts, elems)
        mean_icn = float(jnp.mean(icn))
        min_icn = float(jnp.min(icn))
        max_area = float(jnp.max(areas))

        if snap_root is not None:
            export_snapshot_npz(
                snap_root / f"iter_{it:04d}.npz",
                pts,
                elems,
                metrics={
                    "mean_icn": mean_icn,
                    "min_icn": min_icn,
                    "max_area": max_area,
                    "n_nodes": float(buffer.node_count),
                    "n_elements": float(buffer.element_count),
                },
            )

        if mean_icn >= target_mean_icn and max_area <= target_area:
            history.append(
                AdaptiveHistory(it, buffer.node_count, buffer.element_count, mean_icn, min_icn, max_area, False, False, False)
            )
            break

        priority = triangle_refinement_priority(pts, elems, target_area=target_area)
        split_idx = int(jnp.argmax(priority))
        did_split = priority[split_idx] > 0.0
        if did_split:
            buffer, did_split = split_triangle(buffer, split_idx)

        did_flip = False
        pair = _find_first_shared_edge_pair(np.asarray(active_elements(buffer)))
        if pair is not None:
            buffer, did_flip = flip_diagonal(buffer, pair[0], pair[1])

        did_collapse = False
        collapse_node = _collapse_candidate_node(buffer, target_area=target_area, min_nodes=min_nodes)
        if collapse_node is not None:
            buffer, did_collapse = collapse_triangle(buffer, collapse_node)

        buffer = _smooth_active_points(buffer, alpha=smoothing_alpha, steps=smoothing_steps, movable_mask=movable_mask)
        history.append(
            AdaptiveHistory(
                it,
                buffer.node_count,
                buffer.element_count,
                mean_icn,
                min_icn,
                max_area,
                did_split,
                did_flip,
                did_collapse,
            )
        )

    return buffer, history
