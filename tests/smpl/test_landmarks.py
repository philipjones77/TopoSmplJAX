import jax.numpy as jnp
import numpy as np

from smpljax.landmarks import find_dynamic_lmk_idx_and_bcoords, vertices2landmarks
from smpljax.vertex_joint_selector import build_smpl_extra_joint_ids


def test_vertices2landmarks_barycentric() -> None:
    vertices = jnp.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=jnp.float32)
    faces = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    lmk_faces_idx = jnp.array([[0]], dtype=jnp.int32)
    lmk_bary = jnp.array([[[0.25, 0.25, 0.5]]], dtype=jnp.float32)
    out = vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary)
    np.testing.assert_allclose(np.asarray(out[0, 0]), np.array([0.25, 0.5, 0.0], dtype=np.float32))


def test_build_smplx_extra_joint_ids_length() -> None:
    ids = build_smpl_extra_joint_ids("smplx", use_hands=True, use_feet_keypoints=True)
    assert len(ids) == 21


def test_dynamic_landmark_lookup_zero_pose_selects_zero_bin() -> None:
    vertices = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    pose = jnp.zeros((2, 3, 3), dtype=jnp.float32)
    dynamic_faces_idx = jnp.arange(79, dtype=jnp.int32)
    dynamic_bary = jnp.stack([jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)] * 79, axis=0)
    neck_chain = jnp.array([0], dtype=jnp.int32)

    faces_idx, bary = find_dynamic_lmk_idx_and_bcoords(
        vertices=vertices,
        pose=pose,
        dynamic_lmk_faces_idx=dynamic_faces_idx,
        dynamic_lmk_bary_coords=dynamic_bary,
        neck_kin_chain=neck_chain,
    )

    np.testing.assert_array_equal(np.asarray(faces_idx), np.zeros((2, 1), dtype=np.int32))
    np.testing.assert_allclose(
        np.asarray(bary),
        np.tile(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32), (2, 1, 1)),
    )
