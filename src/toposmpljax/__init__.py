"""Combined namespace for the TopoJAX and smplJAX backends."""

from .backends import BackendName, BackendSpec, build_mode_bridge, get_backend_mode_report, get_backend_spec, get_backends, repair_print_mesh
from .mesh_repair import (
    MeshRepairResult,
    RepairBackend,
    export_repaired_stl,
    repair_topo_mesh_for_printing,
    repair_smpl_mesh_for_printing,
    repair_triangle_mesh,
)

__all__ = [
    "BackendName",
    "BackendSpec",
    "MeshRepairResult",
    "RepairBackend",
    "build_mode_bridge",
    "export_repaired_stl",
    "get_backend_mode_report",
    "get_backend_spec",
    "get_backends",
    "repair_topo_mesh_for_printing",
    "repair_print_mesh",
    "repair_smpl_mesh_for_printing",
    "repair_triangle_mesh",
]
