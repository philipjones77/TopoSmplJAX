"""Mesh export adapters and simple interchange formats."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

import jax.numpy as jnp
import json
import numpy as np


_MSH_ELEMENT_TYPES: dict[str, int] = {
    "line": 1,
    "triangle": 2,
    "quad": 3,
    "tetra": 4,
}


class GmshElementBlock(NamedTuple):
    elements: jnp.ndarray | np.ndarray
    element_kind: str
    physical_tags: jnp.ndarray | np.ndarray | None = None
    geometrical_tags: jnp.ndarray | np.ndarray | None = None


def as_array(points: jnp.ndarray):
    """Return array-like object for downstream export integrations."""
    return points


def _as_export_points(points) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError("points must have shape (n_points, 2) or (n_points, 3)")
    if arr.shape[1] == 2:
        zeros = np.zeros((arr.shape[0], 1), dtype=arr.dtype)
        arr = np.concatenate([arr, zeros], axis=1)
    return arr


def _as_export_elements(elements) -> np.ndarray:
    arr = np.asarray(elements)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3, 4):
        raise ValueError("elements must have shape (n_elements, 2), (n_elements, 3), or (n_elements, 4)")
    return arr.astype(np.int64, copy=False)


def _as_entity_tags(element_count: int, tags, *, name: str) -> np.ndarray:
    if tags is None:
        return np.ones((element_count,), dtype=np.int64)
    tags = np.asarray(tags)
    if tags.shape != (element_count,):
        raise ValueError(f"{name} must have shape (n_elements,)")
    return tags.astype(np.int64, copy=False)


def _infer_msh_element_type(points: np.ndarray, elements: np.ndarray, element_kind: str | None) -> int:
    if element_kind is not None:
        if element_kind not in _MSH_ELEMENT_TYPES:
            raise ValueError("element_kind must be one of: line, triangle, quad, tetra")
        return _MSH_ELEMENT_TYPES[element_kind]
    if elements.shape[1] == 2:
        return _MSH_ELEMENT_TYPES["line"]
    if elements.shape[1] == 3:
        return _MSH_ELEMENT_TYPES["triangle"]
    if points.shape[1] == 2:
        return _MSH_ELEMENT_TYPES["quad"]
    return _MSH_ELEMENT_TYPES["tetra"]


def _as_element_block(
    points: np.ndarray,
    elements,
    *,
    element_kind: str | None,
    physical_tags,
    geometrical_tags,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    elems = _as_export_elements(elements)
    element_type = _infer_msh_element_type(points, elems, element_kind)
    phys = _as_entity_tags(elems.shape[0], physical_tags, name="physical_tags")
    geom = _as_entity_tags(elems.shape[0], geometrical_tags, name="geometrical_tags")
    return elems, element_type, phys, geom


def export_gmsh_msh(
    path: str | Path,
    points,
    elements,
    *,
    element_entity_tags=None,
    physical_tags=None,
    geometrical_tags=None,
    element_kind: str | None = None,
    extra_element_blocks: Sequence[GmshElementBlock] | None = None,
    physical_names: Mapping[tuple[int, int], str] | None = None,
) -> None:
    """Write an ASCII Gmsh MSH 2.2 file for mixed-dimensional mesh blocks.

    The primary block is described by `points` and `elements`. Optional extra
    blocks can be used for tagged boundary entities such as lines or triangles.
    """
    raw_points = np.asarray(points)
    pts = _as_export_points(points)
    primary_geom = geometrical_tags if geometrical_tags is not None else element_entity_tags
    primary_phys = physical_tags if physical_tags is not None else primary_geom
    primary_block = _as_element_block(
        raw_points,
        elements,
        element_kind=element_kind,
        physical_tags=primary_phys,
        geometrical_tags=primary_geom,
    )
    blocks = [primary_block]
    blocks.extend(
        _as_element_block(
            raw_points,
            block.elements,
            element_kind=block.element_kind,
            physical_tags=block.physical_tags,
            geometrical_tags=block.geometrical_tags,
        )
        for block in (extra_element_blocks or ())
    )
    total_elements = int(sum(block[0].shape[0] for block in blocks))

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        if physical_names:
            f.write("$PhysicalNames\n")
            f.write(f"{len(physical_names)}\n")
            for (dim, tag), name in sorted(physical_names.items()):
                f.write(f'{int(dim)} {int(tag)} "{name}"\n')
            f.write("$EndPhysicalNames\n")
        f.write("$Nodes\n")
        f.write(f"{pts.shape[0]}\n")
        for node_id, xyz in enumerate(pts, start=1):
            f.write(f"{node_id} {xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g}\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        f.write(f"{total_elements}\n")
        element_id = 1
        for elems, element_type, phys_tags, geom_tags in blocks:
            for phys_tag, geom_tag, conn in zip(phys_tags, geom_tags, elems, strict=True):
                node_ids = " ".join(str(int(node_index) + 1) for node_index in conn)
                f.write(f"{element_id} {element_type} 2 {int(phys_tag)} {int(geom_tag)} {node_ids}\n")
                element_id += 1
        f.write("$EndElements\n")


def export_snapshot_npz(
    path: str | Path,
    points,
    elements,
    metrics: Mapping[str, float] | None = None,
) -> None:
    """Write a compact mesh snapshot for debugging/visualization."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "points": np.asarray(points),
        "elements": np.asarray(elements),
    }
    if metrics:
        for k, v in metrics.items():
            data[f"metric_{k}"] = np.asarray(v)
    np.savez(p, **data)


def load_snapshot_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load snapshot arrays from `.npz`."""
    arr = np.load(Path(path))
    return {k: arr[k] for k in arr.files}


def export_metrics_json(path: str | Path, metrics: Mapping[str, float]) -> None:
    """Write scalar diagnostics as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2, sort_keys=True)
