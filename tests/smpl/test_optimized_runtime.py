import jax
import jax.numpy as jnp
import numpy as np

from smpljax.body_models import SMPLJAXModel
from smpljax.optimized import CachePolicy, OptimizedSMPLJAX
from smpljax.utils import SMPLModelData


def _array_dtype(dtype=jnp.dtype | np.dtype) -> np.dtype:
    if dtype == jnp.float64 and bool(jax.config.read("jax_enable_x64")):
        return np.float64
    return np.float32


def _model_data(dtype=jnp.float32) -> SMPLModelData:
    num_verts = 8
    num_joints = 4
    arr_dtype = _array_dtype(dtype)
    return SMPLModelData(
        v_template=np.zeros((num_verts, 3), dtype=arr_dtype),
        shapedirs=np.zeros((num_verts, 3, 3), dtype=arr_dtype),
        posedirs=np.zeros(((num_joints - 1) * 9, num_verts * 3), dtype=arr_dtype),
        j_regressor=np.ones((num_joints, num_verts), dtype=arr_dtype) / num_verts,
        parents=jnp.array([-1, 0, 1, 2], dtype=jnp.int32),
        lbs_weights=np.ones((num_verts, num_joints), dtype=arr_dtype) / num_joints,
        num_body_joints=num_joints - 1,
    )


def test_optimized_matches_reference_forward() -> None:
    data = _model_data()
    ref = SMPLJAXModel(data=data)
    opt = OptimizedSMPLJAX(data=data, cache_policy=CachePolicy(enable_batch_bucketing=False))
    inp = opt.prepare_inputs(batch_size=2)
    inp = inp.__class__(
        **{
            **inp.__dict__,
            "betas": jnp.array([[0.1, -0.2, 0.3], [0.0, 0.1, -0.1]], dtype=jnp.float32),
            "global_orient": jnp.array([[[0.02, 0.01, -0.03]], [[0.0, 0.0, 0.0]]], dtype=jnp.float32),
        }
    )
    out_opt = opt.forward(inp, pose2rot=True)
    out_ref = ref(
        betas=inp.betas,
        body_pose=inp.body_pose,
        global_orient=inp.global_orient,
        transl=inp.transl,
    )
    np.testing.assert_allclose(np.asarray(out_opt.vertices), np.asarray(out_ref.vertices), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out_opt.joints), np.asarray(out_ref.joints), rtol=1e-6, atol=1e-6)


def test_optimized_dtype_consistency() -> None:
    data = _model_data(dtype=jnp.float32)
    opt = OptimizedSMPLJAX(data=data, dtype=jnp.float32, cache_policy=CachePolicy(enable_batch_bucketing=False))
    inp = opt.prepare_inputs(batch_size=1)
    out = opt.forward(inp, pose2rot=True)
    assert out.vertices.dtype == jnp.float32
    assert out.joints.dtype == jnp.float32


def test_optimized_float64_policy_matches_jax_x64_setting() -> None:
    data = _model_data(dtype=jnp.float32)
    requested_dtype = jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32
    opt = OptimizedSMPLJAX(data=data, dtype=requested_dtype, cache_policy=CachePolicy(enable_batch_bucketing=False))
    inp = opt.prepare_inputs(batch_size=1)
    out = opt.forward(inp, pose2rot=True)
    x64_enabled = bool(jax.config.read("jax_enable_x64"))
    expected_dtype = jnp.float64 if x64_enabled else jnp.float32
    assert out.vertices.dtype == expected_dtype
    assert out.joints.dtype == expected_dtype


def test_optimized_compile_cache_reuse() -> None:
    opt = OptimizedSMPLJAX(data=_model_data(), cache_policy=CachePolicy(enable_batch_bucketing=False))
    inp = opt.prepare_inputs(batch_size=1)
    _ = opt.forward(inp, pose2rot=True)
    _ = opt.forward(inp, pose2rot=True)
    assert opt.compile_count == 1
    inp_rm = opt.prepare_inputs(batch_size=1, pose2rot=False)
    _ = opt.forward(inp_rm, pose2rot=False)
    assert opt.compile_count == 2


def test_value_changes_no_recompile_same_shapes() -> None:
    opt = OptimizedSMPLJAX(data=_model_data(), cache_policy=CachePolicy(enable_batch_bucketing=False))
    inp0 = opt.prepare_inputs(batch_size=1)
    _ = opt.forward(inp0, pose2rot=True)
    c0 = opt.compile_count
    inp1 = inp0.__class__(**{**inp0.__dict__, "transl": jnp.array([[1.0, -2.0, 0.5]], dtype=jnp.float32)})
    _ = opt.forward(inp1, pose2rot=True)
    assert opt.compile_count == c0


def test_optimized_autodiff_over_betas() -> None:
    opt = OptimizedSMPLJAX(data=_model_data(), cache_policy=CachePolicy(enable_batch_bucketing=False))
    inp = opt.prepare_inputs(batch_size=1)

    def loss_fn(betas):
        local = inp.__class__(**{**inp.__dict__, "betas": betas})
        out = opt.forward(local, pose2rot=True)
        return jnp.sum(out.vertices**2)

    grad = jax.grad(loss_fn)(inp.betas)
    assert grad.shape == inp.betas.shape
    assert jnp.all(jnp.isfinite(grad))


def test_translation_optimization_loop_no_recompile_after_warmup() -> None:
    opt = OptimizedSMPLJAX(data=_model_data(), cache_policy=CachePolicy(enable_batch_bucketing=False))
    inp = opt.prepare_inputs(batch_size=1)

    def loss_fn(transl):
        local = inp.__class__(**{**inp.__dict__, "transl": transl})
        out = opt.forward(local, pose2rot=True)
        return jnp.mean(out.vertices**2)

    val_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    transl = inp.transl
    _ = val_and_grad(transl)  # warmup compile
    c0 = opt.compile_count
    for _ in range(5):
        loss, grad = val_and_grad(transl)
        transl = transl - 1e-2 * grad
        _ = loss.block_until_ready()
    assert opt.compile_count == c0


def test_fixed_padded_batch_size_reuses_single_compile_across_smaller_batches() -> None:
    opt = OptimizedSMPLJAX(
        data=_model_data(),
        cache_policy=CachePolicy(enable_batch_bucketing=False, fixed_padded_batch_size=8),
    )
    inp2 = opt.prepare_inputs(batch_size=2)
    _ = opt.forward(inp2, pose2rot=True)
    c0 = opt.compile_count
    inp5 = opt.prepare_inputs(batch_size=5)
    _ = opt.forward(inp5, pose2rot=True)
    assert opt.compile_count == c0
    assert inp2.padded_batch_size == 8
    assert inp5.padded_batch_size == 8


def test_fixed_padded_batch_size_rejects_larger_batches() -> None:
    opt = OptimizedSMPLJAX(
        data=_model_data(),
        cache_policy=CachePolicy(enable_batch_bucketing=False, fixed_padded_batch_size=4),
    )
    try:
        _ = opt.prepare_inputs(batch_size=5)
    except ValueError as exc:
        assert "exceeds fixed_padded_batch_size" in str(exc)
    else:
        raise AssertionError("Expected fixed padded batch capacity failure")


def test_forbid_new_compiles_blocks_new_shapes_after_warmup() -> None:
    warm = OptimizedSMPLJAX(
        data=_model_data(),
        cache_policy=CachePolicy(enable_batch_bucketing=False, fixed_padded_batch_size=4),
    )
    _ = warm.warmup(batch_size=1, pose2rot=True)
    opt = OptimizedSMPLJAX(
        data=_model_data(),
        cache_policy=CachePolicy(
            enable_batch_bucketing=False,
            fixed_padded_batch_size=4,
            forbid_new_compiles=True,
        ),
    )
    opt._compiled = warm._compiled.copy()  # exercise policy after a warmup population
    inp = opt.prepare_inputs(batch_size=3)
    _ = opt.forward(inp, pose2rot=True)
    try:
        _ = opt.forward(inp, pose2rot=False)
    except RuntimeError as exc:
        assert "forbid_new_compiles=True" in str(exc)
    else:
        raise AssertionError("Expected forbid_new_compiles failure for a new compile key")


def test_warmup_populates_compile_cache_and_returns_inputs() -> None:
    opt = OptimizedSMPLJAX(
        data=_model_data(),
        cache_policy=CachePolicy(enable_batch_bucketing=False, fixed_padded_batch_size=8),
    )
    inputs = opt.warmup(batch_size=3, pose2rot=True, return_full_pose=True)
    assert inputs.actual_batch_size == 3
    assert inputs.padded_batch_size == 8
    assert opt.compile_count == 1
    diag = opt.diagnostics()
    assert diag.compiled_entries == 1
    assert diag.warmup_keys == ((True, True, 8),)
    assert diag.expected_warmup_keys == ()
    assert diag.missing_warmup_keys == ()


def test_warmup_can_seed_cache_for_strict_runtime() -> None:
    opt = OptimizedSMPLJAX(
        data=_model_data(),
        cache_policy=CachePolicy(
            enable_batch_bucketing=False,
            fixed_padded_batch_size=8,
            forbid_new_compiles=True,
        ),
    )
    inputs = opt.warmup(batch_size=8, pose2rot=True)
    assert inputs.padded_batch_size == 8
    assert opt.compile_count == 1
    followup = opt.prepare_inputs(batch_size=3)
    _ = opt.forward(followup, pose2rot=True)
    assert opt.compile_count == 1
    diag = opt.diagnostics()
    assert diag.strict_mode_ready is True
    assert diag.fixed_padded_batch_size == 8
    assert diag.missing_resident_keys == ()
    assert diag.compile_history[0].source == "warmup"
    assert diag.compile_history[0].compiled is True
    assert diag.compile_history[-1].cache_hit is True
    assert diag.jax_backend
    assert diag.jax_version
    assert diag.python_version
    opt.assert_strict_ready()


def test_diagnostics_reports_not_ready_without_warmup_for_strict_mode() -> None:
    opt = OptimizedSMPLJAX(
        data=_model_data(),
        cache_policy=CachePolicy(
            enable_batch_bucketing=False,
            fixed_padded_batch_size=8,
            forbid_new_compiles=True,
        ),
    )
    diag = opt.diagnostics()
    assert diag.strict_mode_ready is False
    assert "missing warmup targets" in diag.strict_mode_reason


def test_assert_strict_ready_rejects_missing_expected_targets() -> None:
    opt = OptimizedSMPLJAX(
        data=_model_data(),
        cache_policy=CachePolicy(
            enable_batch_bucketing=False,
            fixed_padded_batch_size=8,
            forbid_new_compiles=True,
        ),
    )
    expected = opt.expected_warmup_keys(
        padded_batch_size=8,
        pose2rot_options=(True, False),
        return_full_pose_options=(False,),
    )
    opt.set_expected_warmup_keys(expected)
    _ = opt.warmup(batch_size=8, pose2rot=True)
    try:
        opt.assert_strict_ready()
    except RuntimeError as exc:
        assert "missing warmup targets" in str(exc)
    else:
        raise AssertionError("Expected strict readiness failure for missing warmup mode")


def test_compile_key_eviction_history_is_recorded() -> None:
    opt = OptimizedSMPLJAX(
        data=_model_data(),
        cache_policy=CachePolicy(enable_batch_bucketing=False, max_compiled=1),
    )
    _ = opt.forward(opt.prepare_inputs(batch_size=1), pose2rot=True)
    _ = opt.forward(opt.prepare_inputs(batch_size=1, pose2rot=False), pose2rot=False)
    diag = opt.diagnostics()
    assert diag.eviction_count == 1
    assert diag.evicted_keys == ((True, False, 1),)


def test_strict_readiness_fails_if_required_key_was_evicted() -> None:
    opt = OptimizedSMPLJAX(
        data=_model_data(),
        cache_policy=CachePolicy(
            enable_batch_bucketing=False,
            fixed_padded_batch_size=8,
            forbid_new_compiles=True,
            max_compiled=1,
        ),
    )
    _ = opt.warmup(batch_size=8, pose2rot=True)
    _ = opt.forward(opt.prepare_inputs(batch_size=8), pose2rot=True, return_full_pose=True, allow_new_compile=True)
    diag = opt.diagnostics()
    assert diag.strict_mode_ready is False
    assert diag.eviction_count == 1
    assert diag.missing_resident_keys == ((True, False, 8),)
    assert "evicted" in diag.strict_mode_reason


def test_clear_caches_resets_history_and_warmup_state() -> None:
    opt = OptimizedSMPLJAX(
        data=_model_data(),
        cache_policy=CachePolicy(enable_batch_bucketing=False, fixed_padded_batch_size=8),
    )
    _ = opt.warmup(batch_size=8, pose2rot=True)
    opt.clear_caches()
    diag = opt.diagnostics()
    assert diag.compiled_entries == 0
    assert diag.cache_hits == 0
    assert diag.cache_misses == 0
    assert diag.compile_history == ()
    assert diag.warmup_keys == ()
    assert diag.expected_warmup_keys == ()
    assert diag.evicted_keys == ()
