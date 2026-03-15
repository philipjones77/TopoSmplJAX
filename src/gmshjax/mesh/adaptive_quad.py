"""Adaptive remeshing loop for 2D quad meshes."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from gmshjax.io.exports import export_snapshot_npz
from gmshjax.mesh.mutation_qt import (
    QuadMeshBuffer,
    active_quad_elements,
    active_quad_points,
    collapse_quad,
    make_quad_mesh_buffer,
    split_quad,
)
from gmshjax.mesh.operators import quad_icn
from gmshjax.mesh.topology import quad_edges


class QuadAdaptiveHistory(NamedTuple):
    iteration: int
    n_nodes: int
    n_elements: int
    mean_icn: float
    min_icn: float
    max_area: float
    did_split: bool
    did_collapse: bool


def quad_area_magnitudes(points: jnp.ndarray, elements: jnp.ndarray) -> jnp.ndarray:
    q = points[elements]
    a = q[:, 0, :]
    b = q[:, 1, :]
    c = q[:, 2, :]
    d = q[:, 3, :]
    area1 = 0.5 * jnp.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    area2 = 0.5 * jnp.abs((c[:, 0] - a[:, 0]) * (d[:, 1] - a[:, 1]) - (c[:, 1] - a[:, 1]) * (d[:, 0] - a[:, 0]))
    return area1 + area2


def quad_refinement_priority(points: jnp.ndarray, elements: jnp.ndarray, target_area: float) -> jnp.ndarray:
    areas = quad_area_magnitudes(points, elements)
    return areas / jnp.maximum(target_area, 1.0e-12) - 1.0


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


def _collapse_candidate_node(buf: QuadMeshBuffer, target_area: float, min_nodes: int) -> int | None:
    if buf.node_count <= min_nodes:
        return None

    elems = active_quad_elements(buf)
    pts = active_quad_points(buf)
    collapse_limit = 0.35 * target_area
    for node_id in range(buf.node_count - 1, min_nodes - 1, -1):
        incident = elems == node_id
        incident_mask = jnp.any(incident, axis=1)
        incident_count = int(jnp.sum(incident_mask))
        if incident_count != 4:
            continue
        local_quads = elems[incident_mask]
        local_areas = quad_area_magnitudes(pts, local_quads)
        if float(jnp.max(local_areas)) <= collapse_limit:
            return node_id
    return None


def adaptive_remesh_quad(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    max_nodes: int,
    max_elements: int,
    max_iters: int = 20,
    target_area: float = 2.0e-3,
    target_mean_icn: float = 0.55,
    smoothing_alpha: float = 0.15,
    smoothing_steps: int = 2,
    movable_mask: jnp.ndarray | None = None,
    snapshot_dir: str | Path | None = None,
) -> tuple[QuadMeshBuffer, list[QuadAdaptiveHistory]]:
    buf = make_quad_mesh_buffer(points, elements, max_nodes=max_nodes, max_elements=max_elements)
    min_nodes = int(buf.node_count)
    hist: list[QuadAdaptiveHistory] = []
    snap_root = Path(snapshot_dir) if snapshot_dir is not None else None

    for it in range(max_iters):
        pts = active_quad_points(buf)
        elems = active_quad_elements(buf)
        icn = quad_icn(pts, elems)
        areas = quad_area_magnitudes(pts, elems)
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
                    "n_nodes": float(buf.node_count),
                    "n_elements": float(buf.element_count),
                },
            )

        if mean_icn >= target_mean_icn and max_area <= target_area:
            hist.append(QuadAdaptiveHistory(it, buf.node_count, buf.element_count, mean_icn, min_icn, max_area, False, False))
            break

        priority = quad_refinement_priority(pts, elems, target_area=target_area)
        split_idx = int(jnp.argmax(priority))
        did_split = priority[split_idx] > 0.0
        if did_split:
            buf, did_split = split_quad(buf, split_idx)

        did_collapse = False
        collapse_node = _collapse_candidate_node(buf, target_area=target_area, min_nodes=min_nodes)
        if collapse_node is not None:
            buf, did_collapse = collapse_quad(buf, collapse_node)

        # Smoothing on active nodes.
        for _ in range(smoothing_steps):
            pts_full = buf.points
            n = int(buf.node_count)
            edges = quad_edges(active_quad_elements(buf))
            nbr, deg = _neighbor_accum(pts_full[:n], edges)
            prop = pts_full[:n] + smoothing_alpha * (nbr / jnp.maximum(deg[:, None], 1.0) - pts_full[:n])
            if movable_mask is not None:
                prop = jnp.where(movable_mask[:n, None], prop, pts_full[:n])
            pts_full = pts_full.at[:n].set(prop)
            buf = QuadMeshBuffer(
                points=pts_full,
                elements=buf.elements,
                active_nodes=buf.active_nodes,
                active_elements=buf.active_elements,
                node_count=buf.node_count,
                element_count=buf.element_count,
            )

        hist.append(QuadAdaptiveHistory(it, buf.node_count, buf.element_count, mean_icn, min_icn, max_area, did_split, did_collapse))

    return buf, hist
