import jax.numpy as jnp

from smpljax.body_models import SMPLJAXModel
from smpljax.utils import SMPLModelData


def _base_model() -> SMPLJAXModel:
    return SMPLJAXModel(
        data=SMPLModelData(
            v_template=jnp.zeros((6, 3), dtype=jnp.float32),
            shapedirs=jnp.zeros((6, 3, 2), dtype=jnp.float32),
            posedirs=jnp.zeros((27, 18), dtype=jnp.float32),
            j_regressor=jnp.ones((4, 6), dtype=jnp.float32) / 6.0,
            parents=jnp.array([-1, 0, 1, 2], dtype=jnp.int32),
            lbs_weights=jnp.ones((6, 4), dtype=jnp.float32) / 4.0,
            num_body_joints=3,
        )
    )


def test_expression_rejected_for_model_without_expression_coeffs() -> None:
    model = _base_model()
    try:
        _ = model(
            betas=jnp.zeros((1, 2), dtype=jnp.float32),
            body_pose=jnp.zeros((1, 3, 3), dtype=jnp.float32),
            expression=jnp.zeros((1, 1), dtype=jnp.float32),
        )
    except ValueError as exc:
        assert "does not support expression coefficients" in str(exc)
    else:
        raise AssertionError("Expected expression validation failure")


def test_face_pose_rejected_for_model_without_face_joints() -> None:
    model = _base_model()
    try:
        _ = model(
            betas=jnp.zeros((1, 2), dtype=jnp.float32),
            body_pose=jnp.zeros((1, 3, 3), dtype=jnp.float32),
            jaw_pose=jnp.zeros((1, 1, 3), dtype=jnp.float32),
        )
    except ValueError as exc:
        assert "no face joints" in str(exc)
    else:
        raise AssertionError("Expected face-pose validation failure")


def test_transl_batch_must_match_betas_batch() -> None:
    model = _base_model()
    try:
        _ = model(
            betas=jnp.zeros((2, 2), dtype=jnp.float32),
            body_pose=jnp.zeros((2, 3, 3), dtype=jnp.float32),
            transl=jnp.zeros((1, 3), dtype=jnp.float32),
        )
    except ValueError as exc:
        assert "transl batch size must match betas batch size" in str(exc)
    else:
        raise AssertionError("Expected translation batch-size validation failure")
