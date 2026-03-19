import numpy as np

from smpljax.io import clear_io_cache, io_cache_diagnostics, load_model_cached


def _write_model(path):
    np.savez(
        path,
        v_template=np.zeros((2, 3), dtype=np.float32),
        shapedirs=np.zeros((2, 3, 1), dtype=np.float32),
        posedirs=np.zeros((9, 6), dtype=np.float32),
        J_regressor=np.zeros((2, 2), dtype=np.float32),
        weights=np.ones((2, 2), dtype=np.float32) / 2.0,
        parents=np.array([-1, 0], dtype=np.int32),
    )


def test_io_cache_hard_cap_two_models(tmp_path) -> None:
    clear_io_cache()
    p1 = tmp_path / "m1.npz"
    p2 = tmp_path / "m2.npz"
    p3 = tmp_path / "m3.npz"
    _write_model(p1)
    _write_model(p2)
    _write_model(p3)

    _ = load_model_cached(p1)
    _ = load_model_cached(p2)
    d = io_cache_diagnostics()
    assert d.entries == 2

    _ = load_model_cached(p3)
    d = io_cache_diagnostics()
    assert d.entries == 2
    assert len(d.keys) == 2


def test_io_cache_hit_miss_tracking(tmp_path) -> None:
    clear_io_cache()
    p1 = tmp_path / "m1.npz"
    _write_model(p1)

    _ = load_model_cached(p1)
    _ = load_model_cached(p1)
    d = io_cache_diagnostics()
    assert d.misses >= 1
    assert d.hits >= 1
