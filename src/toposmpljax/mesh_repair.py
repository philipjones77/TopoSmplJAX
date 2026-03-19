"""Shared mesh-repair helpers for 3D-printable surface preparation."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Literal

import numpy as np

from topojax.io.exports import export_binary_stl
from smpljax.mesh_export import export_posed_mesh, export_template_mesh


RepairBackend = Literal["auto", "pymeshfix", "meshlabserver", "fallback"]


@dataclass(frozen=True)
class MeshRepairResult:
    vertices: np.ndarray
    faces: np.ndarray
    backend_requested: RepairBackend
    backend_used: str
    watertight: bool
    boundary_edge_count: int
    component_count: int
    input_vertex_count: int
    input_face_count: int
    output_vertex_count: int
    output_face_count: int
    holes_filled: int
    removed_degenerate_faces: int
    removed_duplicate_faces: int
    removed_small_components: int
    notes: tuple[str, ...] = ()

    def metadata(self) -> dict[str, Any]:
        return {
            "backend_requested": self.backend_requested,
            "backend_used": self.backend_used,
            "watertight": self.watertight,
            "boundary_edge_count": self.boundary_edge_count,
            "component_count": self.component_count,
            "input_vertex_count": self.input_vertex_count,
            "input_face_count": self.input_face_count,
            "output_vertex_count": self.output_vertex_count,
            "output_face_count": self.output_face_count,
            "holes_filled": self.holes_filled,
            "removed_degenerate_faces": self.removed_degenerate_faces,
            "removed_duplicate_faces": self.removed_duplicate_faces,
            "removed_small_components": self.removed_small_components,
            "notes": list(self.notes),
        }


def _as_vertices(vertices: Any) -> np.ndarray:
    arr = np.asarray(vertices, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("vertices must have shape (n_vertices, 3)")
    return arr


def _as_faces(faces: Any) -> np.ndarray:
    arr = np.asarray(faces, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("faces must have shape (n_faces, 3)")
    return arr


def _component_labels(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if faces.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    parent = np.arange(faces.shape[0], dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    vertex_to_faces: dict[int, list[int]] = {}
    for face_id, face in enumerate(faces):
        for vertex_id in face:
            vertex_to_faces.setdefault(int(vertex_id), []).append(face_id)

    for face_ids in vertex_to_faces.values():
        anchor = face_ids[0]
        for face_id in face_ids[1:]:
            union(anchor, face_id)

    roots = np.asarray([find(i) for i in range(faces.shape[0])], dtype=np.int64)
    unique_roots, labels = np.unique(roots, return_inverse=True)
    return labels, unique_roots


def _compact_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if faces.size == 0:
        return np.zeros((0, 3), dtype=vertices.dtype), np.zeros((0, 3), dtype=np.int64)
    used_vertices = np.unique(faces.reshape(-1))
    remap = np.full((vertices.shape[0],), -1, dtype=np.int64)
    remap[used_vertices] = np.arange(used_vertices.shape[0], dtype=np.int64)
    return vertices[used_vertices], remap[faces]


def _canonical_face_rows(faces: np.ndarray) -> np.ndarray:
    return np.sort(faces, axis=1)


def _remove_invalid_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    if faces.size == 0:
        return vertices, faces, 0, 0
    in_range = np.all((faces >= 0) & (faces < vertices.shape[0]), axis=1)
    faces = faces[in_range]
    canonical = _canonical_face_rows(faces)
    nondegenerate_mask = (canonical[:, 0] != canonical[:, 1]) & (canonical[:, 1] != canonical[:, 2])
    removed_degenerate = int(faces.shape[0] - int(np.sum(nondegenerate_mask)))
    faces = faces[nondegenerate_mask]
    canonical = canonical[nondegenerate_mask]
    if canonical.size == 0:
        return np.zeros((0, 3), dtype=vertices.dtype), np.zeros((0, 3), dtype=np.int64), removed_degenerate, 0
    _, unique_index = np.unique(canonical, axis=0, return_index=True)
    keep = np.zeros((faces.shape[0],), dtype=bool)
    keep[np.asarray(unique_index, dtype=np.int64)] = True
    removed_duplicates = int(faces.shape[0] - int(np.sum(keep)))
    compact_vertices, compact_faces = _compact_mesh(vertices, faces[keep])
    return compact_vertices, compact_faces, removed_degenerate, removed_duplicates


def _edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for face in faces:
        for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(edge[0]), int(edge[1]))))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _boundary_loops(faces: np.ndarray) -> list[list[int]]:
    counts = _edge_counts(faces)
    boundary_edges = [edge for edge, count in counts.items() if count == 1]
    if not boundary_edges:
        return []

    adjacency: dict[int, list[int]] = {}
    edge_set = set(boundary_edges)
    for a, b in boundary_edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return []

    visited_edges: set[tuple[int, int]] = set()
    loops: list[list[int]] = []
    for edge in boundary_edges:
        key = tuple(sorted(edge))
        if key in visited_edges:
            continue
        start, nxt = edge
        loop = [start, nxt]
        visited_edges.add(key)
        prev = start
        current = nxt
        while True:
            candidates = adjacency[current]
            next_vertex = candidates[0] if candidates[0] != prev else candidates[1]
            sorted_edge = tuple(sorted((current, next_vertex)))
            if next_vertex == start:
                visited_edges.add(sorted_edge)
                break
            if sorted_edge in visited_edges:
                return []
            loop.append(next_vertex)
            visited_edges.add(sorted_edge)
            prev, current = current, next_vertex
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _fill_boundary_holes(vertices: np.ndarray, faces: np.ndarray, *, max_hole_edges: int | None) -> tuple[np.ndarray, int]:
    loops = _boundary_loops(faces)
    if not loops:
        return faces, 0

    patches: list[np.ndarray] = []
    holes_filled = 0
    for loop in loops:
        if max_hole_edges is not None and len(loop) > max_hole_edges:
            continue
        root = int(loop[0])
        for idx in range(1, len(loop) - 1):
            tri = np.asarray([root, loop[idx], loop[idx + 1]], dtype=np.int64)
            tri_points = vertices[tri]
            if np.linalg.norm(np.cross(tri_points[1] - tri_points[0], tri_points[2] - tri_points[0])) > 1.0e-12:
                patches.append(tri)
        holes_filled += 1
    if not patches:
        return faces, 0
    return np.concatenate([faces, np.asarray(patches, dtype=np.int64)], axis=0), holes_filled


def _keep_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    remove_small_components: bool,
    min_component_faces: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if faces.size == 0:
        return vertices, faces, 0
    labels, roots = _component_labels(faces)
    component_count = int(roots.shape[0])
    if component_count <= 1:
        return vertices, faces, 0
    counts = np.bincount(labels, minlength=component_count)
    if remove_small_components:
        keep_components = np.flatnonzero(counts >= max(1, int(min_component_faces)))
        if keep_components.size == 0:
            keep_components = np.asarray([int(np.argmax(counts))], dtype=np.int64)
    else:
        keep_components = np.arange(component_count, dtype=np.int64)
    keep_mask = np.isin(labels, keep_components)
    compact_vertices, compact_faces = _compact_mesh(vertices, faces[keep_mask])
    removed_components = int(component_count - int(keep_components.shape[0]))
    return compact_vertices, compact_faces, removed_components


def _fallback_repair(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    fill_holes: bool,
    remove_small_components: bool,
    min_component_faces: int,
    max_hole_edges: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    vertices, faces, removed_degenerate, removed_duplicates = _remove_invalid_faces(vertices, faces)
    vertices, faces, removed_small_components = _keep_components(
        vertices,
        faces,
        remove_small_components=remove_small_components,
        min_component_faces=min_component_faces,
    )
    holes_filled = 0
    if fill_holes and faces.size:
        faces, holes_filled = _fill_boundary_holes(vertices, faces, max_hole_edges=max_hole_edges)
        vertices, faces, extra_degenerate, extra_duplicates = _remove_invalid_faces(vertices, faces)
        removed_degenerate += extra_degenerate
        removed_duplicates += extra_duplicates
    return vertices, faces, {
        "backend_used": "fallback",
        "holes_filled": holes_filled,
        "removed_degenerate_faces": removed_degenerate,
        "removed_duplicate_faces": removed_duplicates,
        "removed_small_components": removed_small_components,
    }


def _repair_with_pymeshfix(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    join_components: bool,
    remove_small_components: bool,
) -> tuple[np.ndarray, np.ndarray]:
    from pymeshfix import MeshFix  # type: ignore

    meshfix = MeshFix(vertices.copy(), faces.copy())
    meshfix.repair(joincomp=join_components, remove_smallest_components=remove_small_components)
    repaired_vertices = np.asarray(meshfix.points, dtype=np.float64)
    repaired_faces = np.asarray(meshfix.faces, dtype=np.int64)
    return repaired_vertices, repaired_faces


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n")
        for face in faces:
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            _, x, y, z, *_ = line.split()
            vertices.append([float(x), float(y), float(z)])
            continue
        if line.startswith("f "):
            parts = line.split()[1:]
            if len(parts) < 3:
                continue
            face = []
            for part in parts[:3]:
                face.append(int(part.split("/")[0]) - 1)
            faces.append(face)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _repair_with_meshlabserver(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    meshlabserver_executable: str,
    meshlab_script_path: str | Path | None,
    meshlab_script_xml: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="toposmpljax-repair-") as tmp_dir:
        tmp = Path(tmp_dir)
        input_obj = tmp / "input.obj"
        output_obj = tmp / "output.obj"
        script_path: Path | None = None
        if meshlab_script_path is not None:
            script_path = Path(meshlab_script_path)
        elif meshlab_script_xml is not None:
            script_path = tmp / "repair.mlx"
            script_path.write_text(meshlab_script_xml, encoding="utf-8")
        else:
            raise ValueError("meshlabserver repair requires meshlab_script_path or meshlab_script_xml")

        _write_obj(input_obj, vertices, faces)
        command = [meshlabserver_executable, "-i", str(input_obj), "-o", str(output_obj), "-s", str(script_path)]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        if not output_obj.exists():
            stderr = completed.stderr.strip()
            raise RuntimeError(f"meshlabserver did not produce output mesh: {stderr}")
        return _read_obj(output_obj)


def repair_triangle_mesh(
    vertices: Any,
    faces: Any,
    *,
    backend: RepairBackend = "auto",
    fill_holes: bool = True,
    join_components: bool = False,
    remove_small_components: bool = True,
    min_component_faces: int = 8,
    max_hole_edges: int | None = 128,
    require_watertight: bool = False,
    meshlabserver_executable: str = "meshlabserver",
    meshlab_script_path: str | Path | None = None,
    meshlab_script_xml: str | None = None,
) -> MeshRepairResult:
    """Repair a triangle surface mesh for downstream 3D-print workflows."""
    input_vertices = _as_vertices(vertices)
    input_faces = _as_faces(faces)

    notes: list[str] = []
    chosen_backend = backend
    if backend == "auto":
        if importlib.util.find_spec("pymeshfix") is not None:
            chosen_backend = "pymeshfix"
        elif shutil.which(meshlabserver_executable) is not None and (meshlab_script_path is not None or meshlab_script_xml is not None):
            chosen_backend = "meshlabserver"
        else:
            chosen_backend = "fallback"

    if chosen_backend == "pymeshfix":
        repaired_vertices, repaired_faces = _repair_with_pymeshfix(
            input_vertices,
            input_faces,
            join_components=join_components,
            remove_small_components=remove_small_components,
        )
        repaired_vertices, repaired_faces, removed_degenerate, removed_duplicates = _remove_invalid_faces(
            repaired_vertices, repaired_faces
        )
        repaired_vertices, repaired_faces, removed_small_components_count = _keep_components(
            repaired_vertices,
            repaired_faces,
            remove_small_components=remove_small_components,
            min_component_faces=min_component_faces,
        )
        holes_filled = 0
        backend_used = "pymeshfix"
        notes.append("repaired with pymeshfix MeshFix backend")
    elif chosen_backend == "meshlabserver":
        repaired_vertices, repaired_faces = _repair_with_meshlabserver(
            input_vertices,
            input_faces,
            meshlabserver_executable=meshlabserver_executable,
            meshlab_script_path=meshlab_script_path,
            meshlab_script_xml=meshlab_script_xml,
        )
        repaired_vertices, repaired_faces, removed_degenerate, removed_duplicates = _remove_invalid_faces(
            repaired_vertices, repaired_faces
        )
        repaired_vertices, repaired_faces, removed_small_components_count = _keep_components(
            repaired_vertices,
            repaired_faces,
            remove_small_components=remove_small_components,
            min_component_faces=min_component_faces,
        )
        holes_filled = 0
        backend_used = "meshlabserver"
        notes.append("repaired with meshlabserver filter script")
    elif chosen_backend == "fallback":
        repaired_vertices, repaired_faces, stats = _fallback_repair(
            input_vertices,
            input_faces,
            fill_holes=fill_holes,
            remove_small_components=remove_small_components,
            min_component_faces=min_component_faces,
            max_hole_edges=max_hole_edges,
        )
        holes_filled = int(stats["holes_filled"])
        removed_degenerate = int(stats["removed_degenerate_faces"])
        removed_duplicates = int(stats["removed_duplicate_faces"])
        removed_small_components_count = int(stats["removed_small_components"])
        backend_used = str(stats["backend_used"])
        notes.append("used built-in cleanup and boundary-hole triangulation fallback")
    else:
        raise ValueError("backend must be one of: auto, pymeshfix, meshlabserver, fallback")

    boundary_edge_count = int(sum(1 for count in _edge_counts(repaired_faces).values() if count == 1))
    _, roots = _component_labels(repaired_faces)
    component_count = int(roots.shape[0])
    watertight = repaired_faces.size > 0 and boundary_edge_count == 0 and component_count <= 1
    if require_watertight and not watertight:
        raise RuntimeError(
            f"Mesh repair did not produce a watertight mesh using backend {backend_used}. "
            f"Remaining boundary edges: {boundary_edge_count}"
        )
    if not watertight:
        notes.append("mesh still has open boundaries after repair")

    return MeshRepairResult(
        vertices=repaired_vertices,
        faces=repaired_faces,
        backend_requested=backend,
        backend_used=backend_used,
        watertight=watertight,
        boundary_edge_count=boundary_edge_count,
        component_count=component_count,
        input_vertex_count=int(input_vertices.shape[0]),
        input_face_count=int(input_faces.shape[0]),
        output_vertex_count=int(repaired_vertices.shape[0]),
        output_face_count=int(repaired_faces.shape[0]),
        holes_filled=holes_filled,
        removed_degenerate_faces=removed_degenerate,
        removed_duplicate_faces=removed_duplicates,
        removed_small_components=removed_small_components_count,
        notes=tuple(notes),
    )


def _surface_triangles_from_gmsh(points: np.ndarray, elements: np.ndarray, *, element_kind: str | None) -> np.ndarray:
    if elements.ndim != 2:
        raise ValueError("elements must have shape (n_elements, k)")
    if element_kind == "line" or elements.shape[1] == 2:
        raise ValueError("line meshes cannot be repaired into printable surfaces")
    if element_kind == "triangle" or elements.shape[1] == 3:
        return elements.astype(np.int64, copy=False)
    if element_kind == "quad" or (element_kind is None and elements.shape[1] == 4 and points.ndim == 2 and points.shape[1] == 2):
        return np.concatenate([elements[:, [0, 1, 2]], elements[:, [0, 2, 3]]], axis=0).astype(np.int64, copy=False)
    if element_kind == "tetra" or (element_kind is None and elements.shape[1] == 4):
        face_specs = (
            (0, 1, 2, 3),
            (0, 3, 1, 2),
            (0, 2, 3, 1),
            (1, 3, 2, 0),
        )
        oriented_faces: dict[tuple[int, int, int], np.ndarray] = {}
        counts: dict[tuple[int, int, int], int] = {}
        for tet in elements.astype(np.int64, copy=False):
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
        return np.asarray(boundary, dtype=np.int64)
    raise ValueError("unsupported element_kind for printable surface extraction")


def repair_topo_mesh_for_printing(
    points: Any,
    elements: Any,
    *,
    element_kind: str | None = None,
    backend: RepairBackend = "auto",
    **kwargs: Any,
) -> MeshRepairResult:
    vertices = np.asarray(points, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] not in (2, 3):
        raise ValueError("points must have shape (n_points, 2) or (n_points, 3)")
    if vertices.shape[1] == 2:
        vertices = np.concatenate([vertices, np.zeros((vertices.shape[0], 1), dtype=vertices.dtype)], axis=1)
    faces = _surface_triangles_from_gmsh(vertices, np.asarray(elements, dtype=np.int64), element_kind=element_kind)
    return repair_triangle_mesh(vertices, faces, backend=backend, **kwargs)


def repair_smpl_mesh_for_printing(
    model: Any,
    *,
    params: Any | None = None,
    batch_index: int = 0,
    backend: RepairBackend = "auto",
    **kwargs: Any,
) -> MeshRepairResult:
    payload = export_template_mesh(model) if params is None else export_posed_mesh(model, params)
    vertices = np.asarray(payload["nodes"])
    if vertices.ndim == 3:
        if not 0 <= batch_index < vertices.shape[0]:
            raise IndexError("batch_index out of range for batched SMPL mesh export")
        vertices = vertices[batch_index]
    faces = np.asarray(payload["faces"], dtype=np.int64)
    return repair_triangle_mesh(vertices, faces, backend=backend, **kwargs)


def export_repaired_stl(path: str | Path, result: MeshRepairResult, *, header: bytes | str = b"TopoSmplJAX repaired STL") -> None:
    export_binary_stl(path, result.vertices, result.faces, element_kind="triangle", header=header)
