"""Mesh export adapters and simple interchange formats."""

from __future__ import annotations

from pathlib import Path
import struct
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

_STL_FACET_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute_byte_count", "<u2"),
    ]
)


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
    inferred_kind = _infer_element_kind(points, elements, element_kind)
    return _MSH_ELEMENT_TYPES[inferred_kind]


def _infer_element_kind(points: np.ndarray, elements: np.ndarray, element_kind: str | None) -> str:
    if element_kind is not None:
        if element_kind not in _MSH_ELEMENT_TYPES:
            raise ValueError("element_kind must be one of: line, triangle, quad, tetra")
        return element_kind
    if elements.shape[1] == 2:
        return "line"
    if elements.shape[1] == 3:
        return "triangle"
    if points.shape[1] == 2:
        return "quad"
    return "tetra"


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


def _drop_degenerate_triangles(points: np.ndarray, triangles: np.ndarray, *, tol: float = 1.0e-12) -> np.ndarray:
    if triangles.size == 0:
        return triangles
    tri_points = points[triangles]
    doubled_area_norm = np.linalg.norm(
        np.cross(tri_points[:, 1] - tri_points[:, 0], tri_points[:, 2] - tri_points[:, 0]),
        axis=1,
    )
    return triangles[doubled_area_norm > tol]


def _extract_tetra_boundary_triangles(points: np.ndarray, elements: np.ndarray) -> np.ndarray:
    face_specs = (
        (0, 1, 2, 3),
        (0, 3, 1, 2),
        (0, 2, 3, 1),
        (1, 3, 2, 0),
    )
    counts: dict[tuple[int, int, int], int] = {}
    oriented_faces: dict[tuple[int, int, int], np.ndarray] = {}

    for tet in elements:
        tet_points = points[tet]
        for a, b, c, opposite in face_specs:
            face = np.asarray([tet[a], tet[b], tet[c]], dtype=np.int64)
            pa, pb, pc, pd = tet_points[[a, b, c, opposite]]
            normal = np.cross(pb - pa, pc - pa)
            if float(np.dot(normal, pd - pa)) > 0.0:
                face = face[[0, 2, 1]]
            key = tuple(sorted(int(v) for v in face))
            counts[key] = counts.get(key, 0) + 1
            oriented_faces[key] = face

    boundary = [oriented_faces[key] for key, count in counts.items() if count == 1]
    if not boundary:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(boundary, dtype=np.int64)


def _surface_triangles(points: np.ndarray, elements: np.ndarray, *, element_kind: str) -> np.ndarray:
    if element_kind == "line":
        raise ValueError("STL export requires triangle, quad, or tetra elements")
    if element_kind == "triangle":
        triangles = elements
    elif element_kind == "quad":
        triangles = np.concatenate([elements[:, [0, 1, 2]], elements[:, [0, 2, 3]]], axis=0)
    elif element_kind == "tetra":
        triangles = _extract_tetra_boundary_triangles(points, elements)
    else:
        raise ValueError("element_kind must be one of: triangle, quad, tetra")
    return _drop_degenerate_triangles(points, triangles)


def _triangle_normals(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    if triangles.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    tri_points = points[triangles]
    normals = np.cross(tri_points[:, 1] - tri_points[:, 0], tri_points[:, 2] - tri_points[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    safe_lengths = np.where(lengths > 0.0, lengths, 1.0)
    return (normals / safe_lengths).astype(np.float32, copy=False)


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


def export_binary_stl(
    path: str | Path,
    points,
    elements,
    *,
    element_kind: str | None = None,
    header: bytes | str = b"TopoJAX binary STL",
) -> None:
    """Write a binary STL surface mesh from triangle, quad, or tetra connectivity."""
    raw_points = np.asarray(points)
    pts = _as_export_points(points).astype(np.float32, copy=False)
    elems = _as_export_elements(elements)
    kind = _infer_element_kind(raw_points, elems, element_kind)
    triangles = _surface_triangles(pts, elems, element_kind=kind)
    normals = _triangle_normals(pts, triangles)
    facets = np.zeros((triangles.shape[0],), dtype=_STL_FACET_DTYPE)
    facets["normal"] = normals
    facets["vertices"] = pts[triangles].astype(np.float32, copy=False)

    header_bytes = header.encode("ascii", errors="ignore") if isinstance(header, str) else header
    header_bytes = header_bytes[:80].ljust(80, b"\0")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        f.write(header_bytes)
        f.write(struct.pack("<I", int(triangles.shape[0])))
        f.write(facets.tobytes())


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
