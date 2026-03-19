import numpy as np
import pickle

from smpljax.io import clear_io_cache, io_cache_diagnostics, load_model, load_model_npz, load_model_pkl


class _FakeSparse:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def toarray(self):
        return self._arr


def test_load_model_npz_reads_extended_metadata(tmp_path) -> None:
    out_path = tmp_path / "model.npz"
    num_joints = 36
    np.savez(
        out_path,
        v_template=np.zeros((4, 3), dtype=np.float32),
        shapedirs=np.zeros((4, 3, 2), dtype=np.float32),
        posedirs=np.zeros((18, 12), dtype=np.float32),
        J_regressor=np.zeros((num_joints, 4), dtype=np.float32),
        weights=np.ones((4, num_joints), dtype=np.float32) / float(num_joints),
        parents=np.array([-1] + list(range(num_joints - 1)), dtype=np.int32),
        num_body_joints=np.array(2, dtype=np.int32),
        num_hand_joints=np.array(15, dtype=np.int32),
        num_face_joints=np.array(3, dtype=np.int32),
        use_pca=np.array(True),
        left_hand_components=np.zeros((6, 45), dtype=np.float32),
        right_hand_components=np.zeros((6, 45), dtype=np.float32),
        faces_tensor=np.array([[0, 1, 2]], dtype=np.int32),
        lmk_faces_idx=np.array([0], dtype=np.int32),
        lmk_bary_coords=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        dynamic_lmk_faces_idx=np.array([0] * 79, dtype=np.int32),
        dynamic_lmk_bary_coords=np.array([[0.0, 1.0, 0.0]] * 79, dtype=np.float32),
        neck_kin_chain=np.array([0], dtype=np.int32),
        use_face_contour=np.array(True),
        extra_vertex_ids=np.array([1, 2], dtype=np.int32),
    )
    model = load_model_npz(out_path)

    assert model.num_body_joints == 2
    assert model.num_hand_joints == 15
    assert model.num_face_joints == 3
    assert model.use_pca is True
    assert model.left_hand_components is not None
    assert model.right_hand_components is not None
    assert model.faces_tensor is not None
    assert model.lmk_faces_idx is not None
    assert model.dynamic_lmk_faces_idx is not None
    assert model.neck_kin_chain is not None
    assert model.use_face_contour is True
    assert model.extra_vertex_ids == [1, 2]


def test_load_model_pkl_supports_aliases_and_sparse_like(tmp_path) -> None:
    out_path = tmp_path / "model.pkl"
    payload = {
        "v_template": np.zeros((4, 3), dtype=np.float32),
        "shapedirs": np.zeros((4, 3, 2), dtype=np.float32),
        "posedirs": np.zeros((4, 3, 18), dtype=np.float32),
        "j_regressor": _FakeSparse(np.zeros((3, 4), dtype=np.float32)),
        "lbs_weights": np.ones((4, 3), dtype=np.float32) / 3.0,
        "kintree_table": np.array([[0, 0, 1], [0, 1, 2]], dtype=np.int32),
        "f": np.array([[0, 1, 2]], dtype=np.int32),
    }
    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    model = load_model_pkl(out_path)
    assert model.j_regressor.shape == (3, 4)
    assert model.lbs_weights.shape == (4, 3)
    assert int(np.asarray(model.parents)[0]) == -1
    assert model.faces_tensor is not None


def test_load_model_dispatches_by_extension(tmp_path) -> None:
    npz_path = tmp_path / "m.npz"
    np.savez(
        npz_path,
        v_template=np.zeros((2, 3), dtype=np.float32),
        shapedirs=np.zeros((2, 3, 1), dtype=np.float32),
        posedirs=np.zeros((9, 6), dtype=np.float32),
        J_regressor=np.zeros((2, 2), dtype=np.float32),
        weights=np.ones((2, 2), dtype=np.float32) / 2.0,
        parents=np.array([-1, 0], dtype=np.int32),
    )
    model = load_model(npz_path)
    assert model.num_body_joints == 1

    bad_path = tmp_path / "m.txt"
    bad_path.write_text("x", encoding="utf-8")
    try:
        _ = load_model(bad_path)
    except ValueError as e:
        assert "Unsupported model file type" in str(e)
    else:
        raise AssertionError("Expected ValueError for unsupported extension")


def test_load_model_can_bypass_cache(tmp_path) -> None:
    npz_path = tmp_path / "m.npz"
    np.savez(
        npz_path,
        v_template=np.zeros((2, 3), dtype=np.float32),
        shapedirs=np.zeros((2, 3, 1), dtype=np.float32),
        posedirs=np.zeros((9, 6), dtype=np.float32),
        J_regressor=np.zeros((2, 2), dtype=np.float32),
        weights=np.ones((2, 2), dtype=np.float32) / 2.0,
        parents=np.array([-1, 0], dtype=np.int32),
    )
    clear_io_cache()
    model0 = load_model(npz_path, use_cache=False)
    model1 = load_model(npz_path, use_cache=False)
    assert model0 is not model1
    stats = io_cache_diagnostics()
    assert stats.entries == 0
    assert stats.hits == 0
    assert stats.misses == 0
