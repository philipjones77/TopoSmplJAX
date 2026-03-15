"""Launch the external Gmsh GUI as a viewer-only mesh inspector."""

from __future__ import annotations

from pathlib import Path
import subprocess


def launch_gmsh_viewer(
    mesh_path: str | Path,
    *,
    gmsh_executable: str = "gmsh",
    wait: bool = False,
    extra_args: list[str] | None = None,
):
    """Launch Gmsh to inspect an exported mesh file.

    This helper is viewer-only: it does not use Gmsh for geometry creation or meshing.
    """
    path = Path(mesh_path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file does not exist: {path}")
    cmd = [gmsh_executable, str(path), *(extra_args or [])]
    proc = subprocess.Popen(cmd)
    if wait:
        proc.wait()
    return proc
