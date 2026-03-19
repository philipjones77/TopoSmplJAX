"""Visualization helpers for fixed-topology mode-1 workflows."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from topojax.mesh.topology import MeshTopology


def _points3(points: jnp.ndarray) -> np.ndarray:
    arr = np.asarray(points)
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)], axis=1)
    return arr


def _tet_boundary_faces(elements: np.ndarray) -> np.ndarray:
    face_counts: dict[tuple[int, int, int], int] = {}
    face_owner: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for tet in elements.tolist():
        faces = [
            (tet[0], tet[1], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]),
            (tet[1], tet[2], tet[3]),
        ]
        for face in faces:
            key = tuple(sorted(face))
            face_counts[key] = face_counts.get(key, 0) + 1
            face_owner[key] = tuple(face)
    boundary = [face_owner[key] for key, count in face_counts.items() if count == 1]
    return np.asarray(boundary, dtype=np.int32)


def _pyvista_lines(elements: np.ndarray) -> np.ndarray:
    counts = np.full((elements.shape[0], 1), 2, dtype=np.int32)
    return np.hstack([counts, elements]).reshape(-1)


def build_pyvista_dataset(points: jnp.ndarray, topology: MeshTopology):
    """Build a PyVista dataset for lines, triangles, quads, or tetrahedra."""
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyvista is not installed; install pyvista to enable this backend") from exc

    pts = _points3(points)
    elems = np.asarray(topology.elements, dtype=np.int32)
    order = int(elems.shape[1])
    dim = int(np.asarray(points).shape[1])

    if order == 2:
        return pv.PolyData(pts, lines=_pyvista_lines(elems))
    if order == 3:
        faces = np.hstack([np.full((elems.shape[0], 1), 3, dtype=np.int32), elems]).reshape(-1)
        return pv.PolyData(pts, faces)
    if order == 4 and dim == 2:
        faces = np.hstack([np.full((elems.shape[0], 1), 4, dtype=np.int32), elems]).reshape(-1)
        return pv.PolyData(pts, faces)
    if order == 4 and dim == 3:
        cell_sizes = np.full((elems.shape[0], 1), 4, dtype=np.int32)
        cells = np.hstack([cell_sizes, elems]).reshape(-1)
        celltypes = np.full((elems.shape[0],), pv.CellType.TETRA, dtype=np.uint8)
        return pv.UnstructuredGrid(cells, celltypes, pts)
    raise ValueError("Unsupported topology for PyVista visualization")


def plot_mode1_matplotlib(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    title: str = "Mode 1 Fixed-Topology Mesh",
):
    """Return a Matplotlib figure for a mode-1 mesh state."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    pts = np.asarray(points)
    edges = np.asarray(topology.edges, dtype=np.int32)
    order = int(topology.elements.shape[1])
    dim = int(pts.shape[1])

    if dim == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        segs = pts[edges]
        ax.add_collection(LineCollection(segs, colors="black", linewidths=0.8))
        ax.scatter(pts[:, 0], pts[:, 1], s=8, c="tab:blue")
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        return fig

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    pts3 = _points3(points)
    if order == 3:
        poly = Poly3DCollection(pts3[np.asarray(topology.elements, dtype=np.int32)], alpha=0.25, facecolor="tab:blue", edgecolor="black")
        ax.add_collection3d(poly)
    elif order == 4:
        faces = _tet_boundary_faces(np.asarray(topology.elements, dtype=np.int32))
        poly = Poly3DCollection(pts3[faces], alpha=0.25, facecolor="tab:blue", edgecolor="black")
        ax.add_collection3d(poly)
    lines = pts3[edges]
    ax.add_collection3d(Line3DCollection(lines, colors="black", linewidths=0.6))
    ax.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2], s=8, c="tab:red")
    ax.set_title(title)
    return fig


def plot_mode1_pyvista(
    points: jnp.ndarray,
    topology: MeshTopology,
    *,
    title: str = "Mode 1 Fixed-Topology Mesh",
    show: bool = False,
):
    """Build a PyVista plotter for a mode-1 mesh state."""
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyvista is not installed; install pyvista to enable this backend") from exc

    dataset = build_pyvista_dataset(points, topology)
    plotter = pv.Plotter(off_screen=not show)
    plotter.add_mesh(dataset, show_edges=True, color="lightblue")
    plotter.add_title(title)
    if show:
        plotter.show()
    return plotter


def build_mode1_viper_payload(points: jnp.ndarray, topology: MeshTopology, *, title: str = "Mode 1 Fixed-Topology Mesh") -> dict[str, Any]:
    """Build a backend-neutral payload for a viper-style viewer."""
    return {
        "title": title,
        "points": np.asarray(points).tolist(),
        "elements": np.asarray(topology.elements, dtype=np.int32).tolist(),
        "edges": np.asarray(topology.edges, dtype=np.int32).tolist(),
        "element_order": int(topology.elements.shape[1]),
        "point_dim": int(np.asarray(points).shape[1]),
    }


def plot_mode1_viper(points: jnp.ndarray, topology: MeshTopology, *, title: str = "Mode 1 Fixed-Topology Mesh"):
    """Invoke an installed `viper` backend if available.

    The adapter performs capability detection because the package API is not part
    of TopoJAX. If the installed package does not expose a recognized entry point,
    the caller can still use `build_mode1_viper_payload` directly.
    """
    payload = build_mode1_viper_payload(points, topology, title=title)
    try:
        import viper  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("viper is not installed; install viper to enable this backend") from exc

    if hasattr(viper, "plot_mesh"):
        return viper.plot_mesh(payload["points"], payload["elements"], title=payload["title"])
    if hasattr(viper, "show"):
        return viper.show(payload)
    raise RuntimeError("Installed viper package does not expose a recognized mesh visualization entry point")
