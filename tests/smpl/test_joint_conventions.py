import jax.numpy as jnp
import numpy as np

from smpljax.body_models import SMPLJAXModel
from smpljax.utils import SMPLModelData


def _build_base_model_data() -> SMPLModelData:
    num_verts = 4
    num_joints = 3
    return SMPLModelData(
        v_template=jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        ),
        shapedirs=jnp.zeros((num_verts, 3, 1), dtype=jnp.float32),
        posedirs=jnp.zeros(((num_joints - 1) * 9, num_verts * 3), dtype=jnp.float32),
        j_regressor=jnp.ones((num_joints, num_verts), dtype=jnp.float32) / num_verts,
        parents=jnp.array([-1, 0, 1], dtype=jnp.int32),
        lbs_weights=jnp.ones((num_verts, num_joints), dtype=jnp.float32) / num_joints,
        num_body_joints=2,
    )


def test_joint_augmentation_order_extra_then_landmarks() -> None:
    data = _build_base_model_data()
    data = SMPLModelData(
        **{
            **data.__dict__,
            "extra_vertex_ids": [1],
            "faces_tensor": jnp.array([[0, 1, 2]], dtype=jnp.int32),
            "lmk_faces_idx": jnp.array([0], dtype=jnp.int32),
            "lmk_bary_coords": jnp.array([[0.5, 0.5, 0.0]], dtype=jnp.float32),
        }
    )
    model = SMPLJAXModel(data=data)
    out = model(
        betas=jnp.zeros((1, 1), dtype=jnp.float32),
        body_pose=jnp.zeros((1, 2, 3), dtype=jnp.float32),
        global_orient=jnp.zeros((1, 1, 3), dtype=jnp.float32),
    )

    base_joint_count = 3
    # Joint layout: base joints, extra vertex joints, then landmarks.
    extra_joint = np.asarray(out.joints[0, base_joint_count])
    landmark_joint = np.asarray(out.joints[0, base_joint_count + 1])
    np.testing.assert_allclose(extra_joint, np.asarray(out.vertices[0, 1]), rtol=1e-6, atol=1e-6)
    v = np.asarray(out.vertices[0])
    expected_landmark = 0.5 * v[0] + 0.5 * v[1]
    np.testing.assert_allclose(landmark_joint, expected_landmark, rtol=1e-6, atol=1e-6)


def test_joint_augmentation_with_dynamic_landmarks_adds_count() -> None:
    data = _build_base_model_data()
    data = SMPLModelData(
        **{
            **data.__dict__,
            "faces_tensor": jnp.array([[0, 1, 2]], dtype=jnp.int32),
            "lmk_faces_idx": jnp.array([0], dtype=jnp.int32),
            "lmk_bary_coords": jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
            "dynamic_lmk_faces_idx": jnp.array([0] * 79, dtype=jnp.int32),
            "dynamic_lmk_bary_coords": jnp.array([[0.0, 1.0, 0.0]] * 79, dtype=jnp.float32),
            "neck_kin_chain": jnp.array([0], dtype=jnp.int32),
            "use_face_contour": True,
        }
    )
    model = SMPLJAXModel(data=data)
    out = model(
        betas=jnp.zeros((1, 1), dtype=jnp.float32),
        body_pose=jnp.zeros((1, 2, 3), dtype=jnp.float32),
        global_orient=jnp.zeros((1, 1, 3), dtype=jnp.float32),
    )
    # base joints + static landmark + dynamic landmark
    assert out.joints.shape == (1, 5, 3)
