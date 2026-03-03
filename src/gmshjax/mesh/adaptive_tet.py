"""Adaptive remeshing loop for 3D tetrahedral meshes."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp

from gmshjax.io.exports import export_snapshot_npz
from gmshjax.mesh.mutation_qt import (
    TetMeshBuffer,
    active_tet_elements,
    active_tet_points,
    make_tet_mesh_buffer,
    split_tet,
)
from gmshjax.mesh.operators import tet_icn
from gmshjax.mesh.topology import tet_edges


class TetAdaptiveHistory(NamedTuple):
    iteration: int
    n_nodes: int
    n_elements: int
    mean_icn: float
    min_icn: float
    max_volume: float
    did_split: bool


def tet_volume_magnitudes(points: jnp.ndarray, elements: jnp.ndarray) -> jnp.ndarray:
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
    return jnp.abs(det) / 6.0


def tet_refinement_priority(points: jnp.ndarray, elements: jnp.ndarray, target_volume: float) -> jnp.ndarray:
    vol = tet_volume_magnitudes(points, elements)
    return vol / jnp.maximum(target_volume, 1.0e-12) - 1.0


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


def adaptive_remesh_tet(
    points: jnp.ndarray,
    elements: jnp.ndarray,
    *,
    max_nodes: int,
    max_elements: int,
    max_iters: int = 15,
    target_volume: float = 8.0e-4,
    target_mean_icn: float = 0.35,
    smoothing_alpha: float = 0.12,
    smoothing_steps: int = 2,
    movable_mask: jnp.ndarray | None = None,
    snapshot_dir: str | Path | None = None,
) -> tuple[TetMeshBuffer, list[TetAdaptiveHistory]]:
    buf = make_tet_mesh_buffer(points, elements, max_nodes=max_nodes, max_elements=max_elements)
    hist: list[TetAdaptiveHistory] = []
    snap_root = Path(snapshot_dir) if snapshot_dir is not None else None

    for it in range(max_iters):
        pts = active_tet_points(buf)
        elems = active_tet_elements(buf)
        icn = tet_icn(pts, elems)
        vol = tet_volume_magnitudes(pts, elems)
        mean_icn = float(jnp.mean(icn))
        min_icn = float(jnp.min(icn))
        max_vol = float(jnp.max(vol))

        if snap_root is not None:
            export_snapshot_npz(
                snap_root / f"iter_{it:04d}.npz",
                pts,
                elems,
                metrics={
                    "mean_icn": mean_icn,
                    "min_icn": min_icn,
                    "max_volume": max_vol,
                    "n_nodes": float(buf.node_count),
                    "n_elements": float(buf.element_count),
                },
            )

        if mean_icn >= target_mean_icn and max_vol <= target_volume:
            hist.append(TetAdaptiveHistory(it, buf.node_count, buf.element_count, mean_icn, min_icn, max_vol, False))
            break

        prio = tet_refinement_priority(pts, elems, target_volume=target_volume)
        split_idx = int(jnp.argmax(prio))
        did_split = bool(float(prio[split_idx]) > 0.0)
        if did_split:
            buf, did_split = split_tet(buf, split_idx)

        for _ in range(smoothing_steps):
            pfull = buf.points
            n = int(buf.node_count)
            edges = tet_edges(active_tet_elements(buf))
            nbr, deg = _neighbor_accum(pfull[:n], edges)
            prop = pfull[:n] + smoothing_alpha * (nbr / jnp.maximum(deg[:, None], 1.0) - pfull[:n])
            if movable_mask is not None:
                prop = jnp.where(movable_mask[:n, None], prop, pfull[:n])
            pfull = pfull.at[:n].set(prop)
            buf = TetMeshBuffer(pfull, buf.elements, buf.active_nodes, buf.active_elements, buf.node_count, buf.element_count)

        hist.append(TetAdaptiveHistory(it, buf.node_count, buf.element_count, mean_icn, min_icn, max_vol, did_split))

    return buf, hist
