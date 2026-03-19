import numpy as np

from smpljax import CachePolicy, OptimizedSMPLJAX, SMPLJAXModel, create_runtime


def _write_model(path) -> None:
    np.savez(
        path,
        v_template=np.zeros((2, 3), dtype=np.float32),
        shapedirs=np.zeros((2, 3, 1), dtype=np.float32),
        posedirs=np.zeros((9, 6), dtype=np.float32),
        J_regressor=np.zeros((2, 2), dtype=np.float32),
        weights=np.ones((2, 2), dtype=np.float32) / 2.0,
        parents=np.array([-1, 0], dtype=np.int32),
    )


def test_create_runtime_plain_and_uncached_return_baseline(tmp_path) -> None:
    path = tmp_path / "model.npz"
    _write_model(path)

    plain = create_runtime(path, mode="plain")
    uncached = create_runtime(path, mode="uncached")

    assert isinstance(plain, SMPLJAXModel)
    assert isinstance(uncached, SMPLJAXModel)
    assert plain.data.num_body_joints == 1
    assert uncached.data.num_body_joints == 1


def test_create_runtime_optimized_returns_optimized_runtime(tmp_path) -> None:
    path = tmp_path / "model.npz"
    _write_model(path)

    runtime = create_runtime(
        path,
        mode="optimized",
        cache_policy=CachePolicy(enable_batch_bucketing=False, max_compiled=2),
    )

    assert isinstance(runtime, OptimizedSMPLJAX)
    inp = runtime.prepare_inputs(batch_size=1)
    out = runtime.forward(inp, pose2rot=True)
    assert out.vertices.shape == (1, 2, 3)


def test_create_runtime_rejects_unknown_mode(tmp_path) -> None:
    path = tmp_path / "model.npz"
    _write_model(path)

    try:
        _ = create_runtime(path, mode="bad")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "Unsupported runtime mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported runtime mode")
