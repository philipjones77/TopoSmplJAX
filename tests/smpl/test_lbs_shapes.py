import jax.numpy as jnp

from smpljax.body_models import SMPLJAXModel
from smpljax.utils import SMPLModelData


def test_forward_shape_tiny_model() -> None:
    batch_size = 2
    num_verts = 5
    num_joints = 3
    num_betas = 2

    model = SMPLJAXModel(
        data=SMPLModelData(
            v_template=jnp.zeros((num_verts, 3)),
            shapedirs=jnp.zeros((num_verts, 3, num_betas)),
            posedirs=jnp.zeros(((num_joints - 1) * 9, num_verts * 3)),
            j_regressor=jnp.ones((num_joints, num_verts)) / num_verts,
            parents=jnp.array([-1, 0, 1], dtype=jnp.int32),
            lbs_weights=jnp.ones((num_verts, num_joints)) / num_joints,
            num_body_joints=num_joints - 1,
        )
    )

    betas = jnp.zeros((batch_size, num_betas))
    body_pose = jnp.zeros((batch_size, num_joints - 1, 3))
    global_orient = jnp.zeros((batch_size, 1, 3))
    out = model(betas=betas, body_pose=body_pose, global_orient=global_orient)
    assert out.vertices.shape == (batch_size, num_verts, 3)
    assert out.joints.shape == (batch_size, num_joints, 3)


def test_forward_shape_with_expression_and_hands() -> None:
    batch_size = 1
    num_verts = 6
    num_joints = 55
    num_betas = 10
    num_expr = 10

    model = SMPLJAXModel(
        data=SMPLModelData(
            v_template=jnp.zeros((num_verts, 3)),
            shapedirs=jnp.zeros((num_verts, 3, num_betas + num_expr)),
            posedirs=jnp.zeros(((num_joints - 1) * 9, num_verts * 3)),
            j_regressor=jnp.ones((num_joints, num_verts)) / num_verts,
            parents=jnp.array([-1] + list(range(0, num_joints - 1)), dtype=jnp.int32),
            lbs_weights=jnp.ones((num_verts, num_joints)) / num_joints,
            num_body_joints=21,
            num_hand_joints=15,
            num_face_joints=3,
        )
    )

    out = model(
        betas=jnp.zeros((batch_size, num_betas)),
        expression=jnp.zeros((batch_size, num_expr)),
        body_pose=jnp.zeros((batch_size, 21, 3)),
        global_orient=jnp.zeros((batch_size, 1, 3)),
        jaw_pose=jnp.zeros((batch_size, 1, 3)),
        leye_pose=jnp.zeros((batch_size, 1, 3)),
        reye_pose=jnp.zeros((batch_size, 1, 3)),
        left_hand_pose=jnp.zeros((batch_size, 15, 3)),
        right_hand_pose=jnp.zeros((batch_size, 15, 3)),
        return_full_pose=True,
    )

    assert out.vertices.shape == (batch_size, num_verts, 3)
    assert out.joints.shape == (batch_size, num_joints, 3)
    assert out.full_pose is not None
    assert out.full_pose.shape == (batch_size, 55, 3)


def test_forward_with_hand_pca_and_extra_joint_selection() -> None:
    batch_size = 1
    num_verts = 7
    num_joints = 52
    num_betas = 10
    num_pca = 6

    model = SMPLJAXModel(
        data=SMPLModelData(
            v_template=jnp.zeros((num_verts, 3)),
            shapedirs=jnp.zeros((num_verts, 3, num_betas)),
            posedirs=jnp.zeros(((num_joints - 1) * 9, num_verts * 3)),
            j_regressor=jnp.ones((num_joints, num_verts)) / num_verts,
            parents=jnp.array([-1] + list(range(0, num_joints - 1)), dtype=jnp.int32),
            lbs_weights=jnp.ones((num_verts, num_joints)) / num_joints,
            num_body_joints=21,
            num_hand_joints=15,
            num_face_joints=0,
            use_pca=True,
            left_hand_components=jnp.zeros((num_pca, 45)),
            right_hand_components=jnp.zeros((num_pca, 45)),
            extra_vertex_ids=[0, 1],
        )
    )

    out = model(
        betas=jnp.zeros((batch_size, num_betas)),
        body_pose=jnp.zeros((batch_size, 21, 3)),
        global_orient=jnp.zeros((batch_size, 1, 3)),
        left_hand_pose=jnp.zeros((batch_size, num_pca)),
        right_hand_pose=jnp.zeros((batch_size, num_pca)),
        return_full_pose=True,
    )

    assert out.full_pose is not None
    assert out.full_pose.shape == (batch_size, num_joints, 3)
    assert out.joints.shape == (batch_size, num_joints + 2, 3)
