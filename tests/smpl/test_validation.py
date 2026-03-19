import numpy as np
import jax.numpy as jnp

from smpljax.utils import SMPLModelData
from smpljax.validation import summarize_model_data, validate_model_data


def _valid_model() -> SMPLModelData:
    return SMPLModelData(
        v_template=jnp.zeros((4, 3), dtype=jnp.float32),
        shapedirs=jnp.zeros((4, 3, 2), dtype=jnp.float32),
        posedirs=jnp.zeros((18, 12), dtype=jnp.float32),
        j_regressor=jnp.zeros((3, 4), dtype=jnp.float32),
        parents=jnp.array([-1, 0, 1], dtype=jnp.int32),
        lbs_weights=jnp.ones((4, 3), dtype=jnp.float32) / 3.0,
        num_betas=2,
        num_body_joints=2,
    )


def test_validate_model_data_accepts_valid_model() -> None:
    model = _valid_model()
    validate_model_data(model)
    summary = summarize_model_data(model)
    assert summary.num_vertices == 4
    assert summary.num_joints == 3


def test_validate_model_data_rejects_inconsistent_joint_metadata() -> None:
    model = SMPLModelData(
        **{
            **_valid_model().__dict__,
            "num_body_joints": 1,
            "num_hand_joints": 1,
        }
    )
    try:
        validate_model_data(model)
    except ValueError as exc:
        assert "joint metadata is inconsistent" in str(exc)
    else:
        raise AssertionError("Expected validation failure for inconsistent joint metadata")


def test_validate_model_data_rejects_out_of_range_faces() -> None:
    model = SMPLModelData(
        **{
            **_valid_model().__dict__,
            "faces_tensor": np.array([[0, 1, 10]], dtype=np.int32),
        }
    )
    try:
        validate_model_data(model)
    except ValueError as exc:
        assert "faces_tensor contains out-of-range vertex indices" in str(exc)
    else:
        raise AssertionError("Expected validation failure for invalid face indices")


def test_validate_model_data_rejects_bad_pca_component_width() -> None:
    model = SMPLModelData(
        **{
            **_valid_model().__dict__,
            "parents": jnp.array([-1, 0, 1, 2, 3, 4, 5], dtype=jnp.int32),
            "j_regressor": jnp.zeros((7, 4), dtype=jnp.float32),
            "lbs_weights": jnp.ones((4, 7), dtype=jnp.float32) / 7.0,
            "posedirs": jnp.zeros((54, 12), dtype=jnp.float32),
            "num_body_joints": 2,
            "num_hand_joints": 1,
            "num_face_joints": 2,
            "use_pca": True,
            "left_hand_components": np.zeros((6, 2), dtype=np.float32),
            "right_hand_components": np.zeros((6, 3), dtype=np.float32),
        }
    )
    try:
        validate_model_data(model)
    except ValueError as exc:
        assert "num_face_joints must be 0 or 3" in str(exc)
        assert "left_hand_components must have width 3" in str(exc)
    else:
        raise AssertionError("Expected validation failure for malformed PCA metadata")


def test_validate_model_data_rejects_pca_components_without_hand_joints() -> None:
    model = SMPLModelData(
        **{
            **_valid_model().__dict__,
            "left_hand_components": np.zeros((6, 0), dtype=np.float32),
        }
    )
    try:
        validate_model_data(model)
    except ValueError as exc:
        assert "left_hand_components was provided but num_hand_joints is 0" in str(exc)
    else:
        raise AssertionError("Expected validation failure for stray PCA metadata")


def test_validate_model_data_rejects_use_face_contour_without_dynamic_metadata() -> None:
    model = SMPLModelData(
        **{
            **_valid_model().__dict__,
            "faces_tensor": np.array([[0, 1, 2]], dtype=np.int32),
            "lmk_faces_idx": np.array([0], dtype=np.int32),
            "lmk_bary_coords": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            "use_face_contour": True,
        }
    )
    try:
        validate_model_data(model)
    except ValueError as exc:
        assert "use_face_contour=True requires dynamic_lmk_faces_idx" in str(exc)
    else:
        raise AssertionError("Expected validation failure for incomplete face contour metadata")


def test_validate_model_data_rejects_dynamic_landmarks_without_base_landmarks() -> None:
    model = SMPLModelData(
        **{
            **_valid_model().__dict__,
            "faces_tensor": np.array([[0, 1, 2]], dtype=np.int32),
            "dynamic_lmk_faces_idx": np.array([0], dtype=np.int32),
            "dynamic_lmk_bary_coords": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            "neck_kin_chain": np.array([1], dtype=np.int32),
        }
    )
    try:
        validate_model_data(model)
    except ValueError as exc:
        assert "dynamic landmark metadata requires base landmark metadata" in str(exc)
    else:
        raise AssertionError("Expected validation failure for partial dynamic landmark metadata")


def test_validate_model_data_rejects_out_of_range_landmark_and_neck_indices() -> None:
    model = SMPLModelData(
        **{
            **_valid_model().__dict__,
            "faces_tensor": np.array([[0, 1, 2]], dtype=np.int32),
            "lmk_faces_idx": np.array([5], dtype=np.int32),
            "lmk_bary_coords": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            "dynamic_lmk_faces_idx": np.array([2], dtype=np.int32),
            "dynamic_lmk_bary_coords": np.array([[0.5, 0.5, 0.0]], dtype=np.float32),
            "neck_kin_chain": np.array([4], dtype=np.int32),
        }
    )
    try:
        validate_model_data(model)
    except ValueError as exc:
        text = str(exc)
        assert "lmk_faces_idx contains out-of-range face indices" in text
        assert "dynamic_lmk_faces_idx contains out-of-range face indices" in text
        assert "neck_kin_chain contains out-of-range joint indices" in text
    else:
        raise AssertionError("Expected validation failure for out-of-range landmark metadata")
