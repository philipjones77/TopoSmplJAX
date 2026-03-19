from pathlib import Path

import jax.numpy as jnp
import numpy as np

from toposmpljax import export_repaired_stl, repair_topo_mesh_for_printing, repair_smpl_mesh_for_printing, repair_triangle_mesh
from smpljax.body_models import SMPLJAXModel
from smpljax.utils import SMPLModelData


def _smpl_model() -> SMPLJAXModel:
    num_joints = 2
    return SMPLJAXModel(
        data=SMPLModelData(
            v_template=jnp.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=jnp.float32,
            ),
            shapedirs=jnp.zeros((4, 3, 1), dtype=jnp.float32),
            posedirs=jnp.zeros((9, 12), dtype=jnp.float32),
            j_regressor=jnp.ones((num_joints, 4), dtype=jnp.float32) / 4.0,
            parents=jnp.array([-1, 0], dtype=jnp.int32),
            lbs_weights=jnp.ones((4, num_joints), dtype=jnp.float32) / float(num_joints),
            num_body_joints=1,
            model_family="smpl",
            model_variant="demo",
            gender="neutral",
            faces_tensor=jnp.array([[0, 1, 2], [0, 3, 1], [0, 2, 3]], dtype=jnp.int32),
        )
    )


def test_repair_triangle_mesh_fallback_closes_simple_hole_and_drops_noise() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [10.0, 10.0, 10.0],
            [10.2, 10.0, 10.0],
            [10.0, 10.2, 10.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [0, 2, 2],
            [0, 3, 1],
            [4, 5, 6],
        ],
        dtype=np.int64,
    )

    result = repair_triangle_mesh(vertices, faces, backend="fallback", min_component_faces=2)

    assert result.backend_used == "fallback"
    assert result.watertight is True
    assert result.boundary_edge_count == 0
    assert result.holes_filled == 1
    assert result.removed_degenerate_faces >= 1
    assert result.removed_duplicate_faces >= 1
    assert result.removed_small_components == 1
    assert result.faces.shape == (4, 3)


def test_repair_topo_mesh_for_printing_extracts_tet_boundary() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    elements = np.asarray([[0, 1, 2, 3]], dtype=np.int64)

    result = repair_topo_mesh_for_printing(points, elements, backend="fallback", element_kind="tetra")

    assert result.watertight is True
    assert result.faces.shape == (4, 3)


def test_repair_smpl_mesh_for_printing_uses_shared_pipeline() -> None:
    model = _smpl_model()

    result = repair_smpl_mesh_for_printing(model, backend="fallback")

    assert result.output_vertex_count == 4
    assert result.faces.shape == (4, 3)
    assert result.watertight is True


def test_export_repaired_stl_writes_binary_surface(tmp_path: Path) -> None:
    model = _smpl_model()
    result = repair_smpl_mesh_for_printing(model, backend="fallback")
    out_path = tmp_path / "repaired.stl"

    export_repaired_stl(out_path, result)

    assert out_path.exists()
    data = out_path.read_bytes()
    assert len(data) == 84 + 4 * 50
