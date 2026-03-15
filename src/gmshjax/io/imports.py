"""Mesh import helpers for fixed-topology workflows."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from gmshjax.io.exports import GmshElementBlock
from gmshjax.mesh.topology import MeshTopology, mesh_topology_from_points_and_elements
from gmshjax.runtime import jax_float_dtype


_ELEMENT_TYPE_TO_KIND = {
    1: "line",
    2: "triangle",
    3: "quad",
    4: "tetra",
}

_ELEMENT_KIND_DIM = {
    "line": 1,
    "triangle": 2,
    "quad": 2,
    "tetra": 3,
}


class ImportedGmshMesh(NamedTuple):
    points: jnp.ndarray
    topology: MeshTopology
    primary_element_kind: str
    primary_physical_tags: jnp.ndarray
    primary_geometrical_tags: jnp.ndarray
    extra_element_blocks: tuple[GmshElementBlock, ...]
    physical_names: dict[tuple[int, int], str]


def load_gmsh_msh(path: str | Path, *, primary_element_kind: str | None = None) -> ImportedGmshMesh:
    """Load a Gmsh MSH 2.2 mesh file into a fixed-topology representation.

    This parser supports the line, triangle, quad, and tetrahedral blocks used by GmshJAX.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    i = 0
    points_xyz: np.ndarray | None = None
    physical_names: dict[tuple[int, int], str] = {}
    grouped: dict[str, dict[str, list]] = {}

    while i < len(lines):
        line = lines[i].strip()
        if line == "$PhysicalNames":
            count = int(lines[i + 1].strip())
            for j in range(count):
                raw = lines[i + 2 + j].strip()
                dim_str, tag_str, name = raw.split(" ", 2)
                physical_names[(int(dim_str), int(tag_str))] = name.strip().strip('"')
            i += count + 3
            continue
        if line == "$Nodes":
            count = int(lines[i + 1].strip())
            pts = np.zeros((count, 3), dtype=float)
            for j in range(count):
                fields = lines[i + 2 + j].split()
                node_id = int(fields[0]) - 1
                pts[node_id] = [float(fields[1]), float(fields[2]), float(fields[3])]
            points_xyz = pts
            i += count + 3
            continue
        if line == "$Elements":
            count = int(lines[i + 1].strip())
            for j in range(count):
                fields = lines[i + 2 + j].split()
                elem_type = int(fields[1])
                if elem_type not in _ELEMENT_TYPE_TO_KIND:
                    continue
                kind = _ELEMENT_TYPE_TO_KIND[elem_type]
                n_tags = int(fields[2])
                tags = [int(v) for v in fields[3 : 3 + n_tags]]
                conn = [int(v) - 1 for v in fields[3 + n_tags :]]
                entry = grouped.setdefault(kind, {"elements": [], "physical": [], "geometrical": []})
                entry["elements"].append(conn)
                entry["physical"].append(tags[0] if n_tags >= 1 else 1)
                entry["geometrical"].append(tags[1] if n_tags >= 2 else entry["physical"][-1])
            i += count + 3
            continue
        i += 1

    if points_xyz is None:
        raise ValueError("MSH file does not contain a $Nodes section")
    if not grouped:
        raise ValueError("MSH file does not contain supported element blocks")

    if primary_element_kind is None:
        primary_element_kind = max(grouped, key=lambda kind: _ELEMENT_KIND_DIM[kind])
    if primary_element_kind not in grouped:
        raise ValueError(f"Requested primary_element_kind '{primary_element_kind}' not found in mesh")

    primary = grouped[primary_element_kind]
    primary_elems = np.asarray(primary["elements"], dtype=np.int32)
    primary_phys = jnp.asarray(primary["physical"], dtype=jnp.int32)
    primary_geom = jnp.asarray(primary["geometrical"], dtype=jnp.int32)

    point_dim = 3
    if primary_element_kind in ("line", "triangle", "quad") and np.allclose(points_xyz[:, 2], 0.0):
        point_dim = 2
    points = jnp.asarray(points_xyz[:, :point_dim], dtype=jax_float_dtype())
    topology = mesh_topology_from_points_and_elements(points, jnp.asarray(primary_elems, dtype=jnp.int32), element_entity_tags=primary_geom)

    extra_blocks: list[GmshElementBlock] = []
    for kind, entry in grouped.items():
        if kind == primary_element_kind:
            continue
        extra_blocks.append(
            GmshElementBlock(
                elements=jnp.asarray(np.asarray(entry["elements"], dtype=np.int32), dtype=jnp.int32),
                element_kind=kind,
                physical_tags=jnp.asarray(entry["physical"], dtype=jnp.int32),
                geometrical_tags=jnp.asarray(entry["geometrical"], dtype=jnp.int32),
            )
        )

    return ImportedGmshMesh(
        points=points,
        topology=topology,
        primary_element_kind=primary_element_kind,
        primary_physical_tags=primary_phys,
        primary_geometrical_tags=primary_geom,
        extra_element_blocks=tuple(extra_blocks),
        physical_names=physical_names,
    )
