import jax.numpy as jnp
import numpy as np

from smpljax.body_models import SMPLJAXModel
from smpljax.lbs import batch_rodrigues
from smpljax.utils import SMPLModelData


def _model() -> SMPLJAXModel:
    num_verts = 6
    num_joints = 4
    return SMPLJAXModel(
        data=SMPLModelData(
            v_template=jnp.zeros((num_verts, 3), dtype=jnp.float32),
            shapedirs=jnp.zeros((num_verts, 3, 2), dtype=jnp.float32),
            posedirs=jnp.zeros(((num_joints - 1) * 9, num_verts * 3), dtype=jnp.float32),
            j_regressor=jnp.ones((num_joints, num_verts), dtype=jnp.float32) / num_verts,
            parents=jnp.array([-1, 0, 1, 2], dtype=jnp.int32),
            lbs_weights=jnp.ones((num_verts, num_joints), dtype=jnp.float32) / num_joints,
            num_body_joints=num_joints - 1,
            num_hand_joints=0,
            num_face_joints=0,
        )
    )


def test_pose2rot_false_matches_axis_angle_path() -> None:
    model = _model()
    betas = jnp.array([[0.1, -0.2]], dtype=jnp.float32)
    global_orient_aa = jnp.array([[[0.02, -0.01, 0.03]]], dtype=jnp.float32)
    body_pose_aa = jnp.array(
        [[[0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, -0.03]]], dtype=jnp.float32
    )

    out_aa = model(
        betas=betas,
        body_pose=body_pose_aa,
        global_orient=global_orient_aa,
        pose2rot=True,
    )

    global_orient_rm = batch_rodrigues(global_orient_aa.reshape(-1, 3)).reshape(1, 1, 3, 3)
    body_pose_rm = batch_rodrigues(body_pose_aa.reshape(-1, 3)).reshape(1, 3, 3, 3)
    out_rm = model(
        betas=betas,
        body_pose=body_pose_rm,
        global_orient=global_orient_rm,
        pose2rot=False,
    )

    np.testing.assert_allclose(np.asarray(out_rm.vertices), np.asarray(out_aa.vertices), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(out_rm.joints), np.asarray(out_aa.joints), rtol=1e-5, atol=1e-5)


def test_pose2rot_false_rejects_pca_path() -> None:
    model = SMPLJAXModel(
        data=SMPLModelData(
            v_template=jnp.zeros((6, 3), dtype=jnp.float32),
            shapedirs=jnp.zeros((6, 3, 1), dtype=jnp.float32),
            posedirs=jnp.zeros((459, 18), dtype=jnp.float32),
            j_regressor=jnp.ones((52, 6), dtype=jnp.float32) / 6.0,
            parents=jnp.array([-1] + list(range(51)), dtype=jnp.int32),
            lbs_weights=jnp.ones((6, 52), dtype=jnp.float32) / 52.0,
            num_body_joints=21,
            num_hand_joints=15,
            use_pca=True,
            left_hand_components=jnp.zeros((6, 45), dtype=jnp.float32),
            right_hand_components=jnp.zeros((6, 45), dtype=jnp.float32),
        )
    )
    try:
        _ = model(
            betas=jnp.zeros((1, 1), dtype=jnp.float32),
            body_pose=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (1, 21, 3, 3)),
            global_orient=jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (1, 1, 3, 3)),
            left_hand_pose=jnp.zeros((1, 6), dtype=jnp.float32),
            right_hand_pose=jnp.zeros((1, 6), dtype=jnp.float32),
            pose2rot=False,
        )
    except ValueError as e:
        assert "use_pca=True with pose2rot=False" in str(e)
    else:
        raise AssertionError("Expected ValueError for pose2rot=False with PCA hand input")
