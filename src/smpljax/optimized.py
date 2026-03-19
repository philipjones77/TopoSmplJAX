from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import platform
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .body_models import ModelOutput
from .landmarks import find_dynamic_lmk_idx_and_bcoords, vertices2landmarks
from .lbs import lbs
from .utils import SMPLModelData, as_jax_array


Array = jnp.ndarray


@dataclass(frozen=True)
class CachePolicy:
    dtype: str | jnp.dtype = jnp.float32
    max_compiled: int = 4
    batch_buckets: tuple[int, ...] = (1, 8, 16, 32, 64)
    enable_batch_bucketing: bool = True
    fixed_padded_batch_size: int | None = None
    forbid_new_compiles: bool = False


@dataclass(frozen=True)
class ForwardInputs:
    betas: Array
    body_pose: Array
    global_orient: Array
    transl: Array
    expression: Array
    jaw_pose: Array
    leye_pose: Array
    reye_pose: Array
    left_hand_pose: Array
    right_hand_pose: Array
    actual_batch_size: int
    padded_batch_size: int


@dataclass(frozen=True)
class CompileEvent:
    pose2rot: bool
    return_full_pose: bool
    padded_batch_size: int
    source: str
    compiled: bool
    cache_hit: bool


@dataclass(frozen=True)
class WarmupCoverage:
    expected_keys: tuple[tuple[bool, bool, int], ...]
    covered_keys: tuple[tuple[bool, bool, int], ...]
    missing_keys: tuple[tuple[bool, bool, int], ...]
    ready: bool


@dataclass(frozen=True)
class RuntimeDiagnostics:
    dtype: str
    compile_count: int
    cache_hits: int
    cache_misses: int
    compiled_entries: int
    compiled_keys: tuple[tuple[bool, bool, int], ...]
    model_bytes: int
    last_input_bytes: int
    last_output_bytes: int
    batch_buckets: tuple[int, ...]
    fixed_padded_batch_size: int | None
    forbid_new_compiles: bool
    strict_mode_ready: bool
    strict_mode_reason: str
    warmup_keys: tuple[tuple[bool, bool, int], ...]
    expected_warmup_keys: tuple[tuple[bool, bool, int], ...]
    missing_warmup_keys: tuple[tuple[bool, bool, int], ...]
    missing_resident_keys: tuple[tuple[bool, bool, int], ...]
    compile_history: tuple[CompileEvent, ...]
    evicted_keys: tuple[tuple[bool, bool, int], ...]
    eviction_count: int
    jax_backend: str
    device_kind: str
    platform: str
    jax_version: str
    python_version: str
    x64_enabled: bool


def _resolve_dtype(dtype: str | jnp.dtype) -> jnp.dtype:
    if isinstance(dtype, str):
        if dtype.lower() == "float64":
            return jnp.float64
        return jnp.float32
    if dtype == jnp.float64:
        return jnp.float64
    return jnp.float32


def _array_nbytes(x: object | None) -> int:
    if x is None:
        return 0
    try:
        return int(np.asarray(x).nbytes)
    except Exception:
        return 0


def _cast_model_data(data: SMPLModelData, dtype: jnp.dtype) -> SMPLModelData:
    def _cast_opt(x):
        if x is None:
            return None
        if np.issubdtype(np.asarray(x).dtype, np.integer):
            return as_jax_array(x, dtype=jnp.int32)
        return as_jax_array(x, dtype=dtype)

    return SMPLModelData(
        v_template=as_jax_array(data.v_template, dtype=dtype),
        shapedirs=as_jax_array(data.shapedirs, dtype=dtype),
        posedirs=as_jax_array(data.posedirs, dtype=dtype),
        j_regressor=as_jax_array(data.j_regressor, dtype=dtype),
        parents=as_jax_array(data.parents, dtype=jnp.int32),
        lbs_weights=as_jax_array(data.lbs_weights, dtype=dtype),
        num_betas=data.num_betas,
        num_expression_coeffs=data.num_expression_coeffs,
        num_body_joints=data.num_body_joints,
        num_hand_joints=data.num_hand_joints,
        num_face_joints=data.num_face_joints,
        model_family=data.model_family,
        model_variant=data.model_variant,
        gender=data.gender,
        pose_mean=_cast_opt(data.pose_mean),
        use_pca=data.use_pca,
        left_hand_components=_cast_opt(data.left_hand_components),
        right_hand_components=_cast_opt(data.right_hand_components),
        extra_vertex_ids=data.extra_vertex_ids,
        joint_mapper=data.joint_mapper,
        faces_tensor=_cast_opt(data.faces_tensor),
        lmk_faces_idx=_cast_opt(data.lmk_faces_idx),
        lmk_bary_coords=_cast_opt(data.lmk_bary_coords),
        dynamic_lmk_faces_idx=_cast_opt(data.dynamic_lmk_faces_idx),
        dynamic_lmk_bary_coords=_cast_opt(data.dynamic_lmk_bary_coords),
        neck_kin_chain=_cast_opt(data.neck_kin_chain),
        use_face_contour=data.use_face_contour,
    )


class OptimizedSMPLJAX:
    """JAX-native compiled runtime with bounded caches and diagnostics."""

    def __init__(self, data: SMPLModelData, dtype: str | jnp.dtype | None = None, cache_policy: CachePolicy | None = None):
        policy = cache_policy or CachePolicy()
        if dtype is not None:
            policy = CachePolicy(
                dtype=dtype,
                max_compiled=policy.max_compiled,
                batch_buckets=policy.batch_buckets,
                enable_batch_bucketing=policy.enable_batch_bucketing,
                fixed_padded_batch_size=policy.fixed_padded_batch_size,
                forbid_new_compiles=policy.forbid_new_compiles,
            )
        self.policy = policy
        self.dtype = _resolve_dtype(policy.dtype)
        self.data = _cast_model_data(data, dtype=self.dtype)
        self._compiled: OrderedDict[tuple[bool, bool, int], Callable[..., tuple[Array, Array, Array]]] = OrderedDict()
        self._compile_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_input_bytes = 0
        self._last_output_bytes = 0
        self._model_bytes = self._estimate_model_bytes(self.data)
        self._compile_history: list[CompileEvent] = []
        self._warmup_keys: set[tuple[bool, bool, int]] = set()
        self._expected_warmup_keys: set[tuple[bool, bool, int]] = set()
        self._evicted_keys: list[tuple[bool, bool, int]] = []

    def export_template_mesh(self) -> dict[str, object]:
        from .mesh_export import export_template_mesh

        return export_template_mesh(self)

    def export_posed_mesh(self, params: ForwardInputs | dict[str, object]) -> dict[str, object]:
        from .mesh_export import export_posed_mesh

        return export_posed_mesh(self, params)

    def export_ct_mesh_payload_template(self) -> dict[str, object]:
        from .mesh_export import export_ct_mesh_payload_template

        return export_ct_mesh_payload_template(self)

    def export_ct_mesh_payload_pose(self, params: ForwardInputs | dict[str, object]) -> dict[str, object]:
        from .mesh_export import export_ct_mesh_payload_pose

        return export_ct_mesh_payload_pose(self, params)

    def to_randomfields77_static_domain_payload(self) -> dict[str, object]:
        from .mesh_export import to_randomfields77_static_domain_payload

        return to_randomfields77_static_domain_payload(self)

    def to_randomfields77_dynamic_mesh_state(self, params: ForwardInputs | dict[str, object]) -> dict[str, object]:
        from .mesh_export import to_randomfields77_dynamic_mesh_state

        return to_randomfields77_dynamic_mesh_state(self, params)

    @property
    def compile_count(self) -> int:
        return self._compile_count

    def _estimate_model_bytes(self, data: SMPLModelData) -> int:
        total = 0
        for k, v in data.__dict__.items():
            if k in {"joint_mapper", "extra_vertex_ids"}:
                if k == "extra_vertex_ids" and v is not None:
                    total += np.asarray(v, dtype=np.int32).nbytes
                continue
            total += _array_nbytes(v)
        return total

    def _num_joints(self) -> int:
        return int(np.asarray(self.data.parents).shape[0])

    def _bucket(self, batch_size: int) -> int:
        if self.policy.fixed_padded_batch_size is not None:
            fixed = int(self.policy.fixed_padded_batch_size)
            if batch_size > fixed:
                raise ValueError(
                    f"batch_size {batch_size} exceeds fixed_padded_batch_size {fixed}; "
                    "increase the fixed capacity to avoid recompiles within a larger range"
                )
            return fixed
        if not self.policy.enable_batch_bucketing:
            return batch_size
        for b in self.policy.batch_buckets:
            if batch_size <= b:
                return b
        return int(self.policy.batch_buckets[-1])

    def prepare_inputs(
        self,
        batch_size: int,
        betas: Array | None = None,
        body_pose: Array | None = None,
        global_orient: Array | None = None,
        transl: Array | None = None,
        expression: Array | None = None,
        jaw_pose: Array | None = None,
        leye_pose: Array | None = None,
        reye_pose: Array | None = None,
        left_hand_pose: Array | None = None,
        right_hand_pose: Array | None = None,
        pose2rot: bool = True,
    ) -> ForwardInputs:
        padded = self._bucket(batch_size)
        num_betas = int(self.data.num_betas or np.asarray(self.data.shapedirs).shape[-1])
        num_expr = int(self.data.num_expression_coeffs or 0)
        num_body = int(self.data.num_body_joints or max(self._num_joints() - 1, 0))
        num_hand = int(self.data.num_hand_joints or 0)

        def _zeros_pose(joints: int) -> Array:
            if pose2rot:
                return jnp.zeros((padded, joints, 3), dtype=self.dtype)
            return jnp.broadcast_to(jnp.eye(3, dtype=self.dtype), (padded, joints, 3, 3))

        def _pad_first_dim(x: Array | None, target_shape: tuple[int, ...]) -> Array:
            if x is None:
                return jnp.zeros(target_shape, dtype=self.dtype)
            x = as_jax_array(x, dtype=self.dtype)
            if x.shape[0] == padded:
                return x
            if x.shape[0] > padded:
                return x[:padded]
            pad_width = [(0, padded - x.shape[0])] + [(0, 0)] * (x.ndim - 1)
            return jnp.pad(x, pad_width)

        return ForwardInputs(
            betas=_pad_first_dim(betas, (padded, num_betas)),
            body_pose=_pad_first_dim(body_pose, _zeros_pose(num_body).shape) if body_pose is not None else _zeros_pose(num_body),
            global_orient=_pad_first_dim(global_orient, _zeros_pose(1).shape)
            if global_orient is not None
            else _zeros_pose(1),
            transl=_pad_first_dim(transl, (padded, 3)),
            expression=_pad_first_dim(expression, (padded, num_expr)),
            jaw_pose=_pad_first_dim(jaw_pose, _zeros_pose(1).shape) if jaw_pose is not None else _zeros_pose(1),
            leye_pose=_pad_first_dim(leye_pose, _zeros_pose(1).shape) if leye_pose is not None else _zeros_pose(1),
            reye_pose=_pad_first_dim(reye_pose, _zeros_pose(1).shape) if reye_pose is not None else _zeros_pose(1),
            left_hand_pose=_pad_first_dim(left_hand_pose, _zeros_pose(num_hand).shape)
            if left_hand_pose is not None
            else _zeros_pose(num_hand),
            right_hand_pose=_pad_first_dim(right_hand_pose, _zeros_pose(num_hand).shape)
            if right_hand_pose is not None
            else _zeros_pose(num_hand),
            actual_batch_size=batch_size,
            padded_batch_size=padded,
        )

    def _compile(self, pose2rot: bool, return_full_pose: bool) -> Callable[..., tuple[Array, Array, Array]]:
        data = self.data
        num_hand = int(data.num_hand_joints or 0)
        num_face = int(data.num_face_joints or 0)
        num_expr = int(data.num_expression_coeffs or 0)
        extra_ids = jnp.asarray(data.extra_vertex_ids or [], dtype=jnp.int32)
        has_extra = extra_ids.size > 0
        has_lmk = data.faces_tensor is not None and data.lmk_faces_idx is not None and data.lmk_bary_coords is not None
        has_dyn = (
            has_lmk
            and data.use_face_contour
            and data.dynamic_lmk_faces_idx is not None
            and data.dynamic_lmk_bary_coords is not None
            and data.neck_kin_chain is not None
        )

        faces = data.faces_tensor
        lmk_faces_idx0 = data.lmk_faces_idx
        lmk_bary0 = data.lmk_bary_coords
        dyn_faces = data.dynamic_lmk_faces_idx
        dyn_bary = data.dynamic_lmk_bary_coords
        neck_chain = data.neck_kin_chain

        def _forward(
            betas: Array,
            body_pose: Array,
            global_orient: Array,
            transl: Array,
            expression: Array,
            jaw_pose: Array,
            leye_pose: Array,
            reye_pose: Array,
            left_hand_pose: Array,
            right_hand_pose: Array,
        ) -> tuple[Array, Array, Array]:
            shape_components = jnp.concatenate([betas, expression], axis=-1) if num_expr > 0 else betas
            pieces = [global_orient, body_pose]
            if num_face > 0:
                pieces.extend([jaw_pose, leye_pose, reye_pose])
            if num_hand > 0:
                pieces.extend([left_hand_pose, right_hand_pose])
            full_pose = jnp.concatenate(pieces, axis=(-2 if pose2rot else -3))
            if pose2rot and data.pose_mean is not None:
                pm = data.pose_mean.reshape(-1, 3) if data.pose_mean.ndim == 1 else data.pose_mean
                full_pose = full_pose + pm[None, :, :]

            shapedirs = data.shapedirs[..., : shape_components.shape[-1]]
            verts, joints = lbs(
                betas=shape_components,
                pose=full_pose,
                v_template=data.v_template,
                shapedirs=shapedirs,
                posedirs=data.posedirs,
                j_regressor=data.j_regressor,
                parents=data.parents,
                lbs_weights=data.lbs_weights,
                pose2rot=pose2rot,
            )
            joints = joints + transl[:, None, :]
            verts = verts + transl[:, None, :]
            if has_extra:
                joints = jnp.concatenate([joints, verts[..., extra_ids, :]], axis=-2)
            if has_lmk:
                lmk_faces_idx = jnp.repeat(lmk_faces_idx0[None, ...], verts.shape[0], axis=0)
                lmk_bary = jnp.repeat(lmk_bary0[None, ...], verts.shape[0], axis=0)
                if has_dyn:
                    dfi, dbc = find_dynamic_lmk_idx_and_bcoords(
                        vertices=verts,
                        pose=full_pose,
                        dynamic_lmk_faces_idx=dyn_faces,
                        dynamic_lmk_bary_coords=dyn_bary,
                        neck_kin_chain=neck_chain,
                        pose2rot=pose2rot,
                    )
                    lmk_faces_idx = jnp.concatenate([lmk_faces_idx, dfi], axis=1)
                    lmk_bary = jnp.concatenate([lmk_bary, dbc], axis=1)
                lmk = vertices2landmarks(verts, faces, lmk_faces_idx, lmk_bary)
                joints = jnp.concatenate([joints, lmk], axis=-2)
            if data.joint_mapper is not None:
                joints = data.joint_mapper(joints)
            fp = full_pose if return_full_pose else jnp.zeros((1,), dtype=verts.dtype)
            return verts, joints, fp

        self._compile_count += 1
        return jax.jit(_forward)

    def _cache_get_or_compile(
        self,
        key: tuple[bool, bool, int],
        *,
        allow_new_compile: bool = False,
        source: str = "forward",
    ) -> Callable[..., tuple[Array, Array, Array]]:
        if key in self._compiled:
            self._cache_hits += 1
            fn = self._compiled.pop(key)
            self._compiled[key] = fn
            self._compile_history.append(
                CompileEvent(
                    pose2rot=key[0],
                    return_full_pose=key[1],
                    padded_batch_size=key[2],
                    source=source,
                    compiled=False,
                    cache_hit=True,
                )
            )
            return fn
        if self.policy.forbid_new_compiles and not allow_new_compile:
            raise RuntimeError(
                "A new compiled shape/mode was requested while forbid_new_compiles=True. "
                "Warm the runtime with the desired padded shape first, or relax the policy."
            )
        self._cache_misses += 1
        fn = self._compile(pose2rot=key[0], return_full_pose=key[1])
        self._compiled[key] = fn
        self._compile_history.append(
            CompileEvent(
                pose2rot=key[0],
                return_full_pose=key[1],
                padded_batch_size=key[2],
                source=source,
                compiled=True,
                cache_hit=False,
            )
        )
        while len(self._compiled) > int(self.policy.max_compiled):
            evicted_key, _ = self._compiled.popitem(last=False)
            self._evicted_keys.append(evicted_key)
        return fn

    def forward(
        self,
        inputs: ForwardInputs,
        pose2rot: bool = True,
        return_full_pose: bool = False,
        *,
        allow_new_compile: bool = False,
    ) -> ModelOutput:
        key = (pose2rot, return_full_pose, int(inputs.padded_batch_size))
        fn = self._cache_get_or_compile(
            key,
            allow_new_compile=allow_new_compile,
            source=("warmup" if allow_new_compile else "forward"),
        )
        self._last_input_bytes = sum(
            _array_nbytes(x)
            for x in [
                inputs.betas,
                inputs.body_pose,
                inputs.global_orient,
                inputs.transl,
                inputs.expression,
                inputs.jaw_pose,
                inputs.leye_pose,
                inputs.reye_pose,
                inputs.left_hand_pose,
                inputs.right_hand_pose,
            ]
        )
        verts, joints, full_pose = fn(
            inputs.betas,
            inputs.body_pose,
            inputs.global_orient,
            inputs.transl,
            inputs.expression,
            inputs.jaw_pose,
            inputs.leye_pose,
            inputs.reye_pose,
            inputs.left_hand_pose,
            inputs.right_hand_pose,
        )
        if inputs.actual_batch_size < inputs.padded_batch_size:
            verts = verts[: inputs.actual_batch_size]
            joints = joints[: inputs.actual_batch_size]
            if return_full_pose:
                full_pose = full_pose[: inputs.actual_batch_size]
        self._last_output_bytes = _array_nbytes(verts) + _array_nbytes(joints) + _array_nbytes(full_pose)
        return ModelOutput(vertices=verts, joints=joints, full_pose=(full_pose if return_full_pose else None))

    def warmup(
        self,
        *,
        batch_size: int,
        pose2rot: bool = True,
        return_full_pose: bool = False,
        betas: Array | None = None,
        body_pose: Array | None = None,
        global_orient: Array | None = None,
        transl: Array | None = None,
        expression: Array | None = None,
        jaw_pose: Array | None = None,
        leye_pose: Array | None = None,
        reye_pose: Array | None = None,
        left_hand_pose: Array | None = None,
        right_hand_pose: Array | None = None,
    ) -> ForwardInputs:
        inputs = self.prepare_inputs(
            batch_size=batch_size,
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            pose2rot=pose2rot,
        )
        out = self.forward(
            inputs,
            pose2rot=pose2rot,
            return_full_pose=return_full_pose,
            allow_new_compile=True,
        )
        _ = out.vertices.block_until_ready()
        self._warmup_keys.add((pose2rot, return_full_pose, int(inputs.padded_batch_size)))
        return inputs

    def expected_warmup_keys(
        self,
        *,
        padded_batch_size: int | None = None,
        pose2rot_options: tuple[bool, ...] = (True,),
        return_full_pose_options: tuple[bool, ...] = (False,),
    ) -> tuple[tuple[bool, bool, int], ...]:
        target_batch = int(
            self.policy.fixed_padded_batch_size
            if padded_batch_size is None and self.policy.fixed_padded_batch_size is not None
            else padded_batch_size or 1
        )
        return tuple(
            sorted(
                {
                    (pose2rot, return_full_pose, target_batch)
                    for pose2rot in pose2rot_options
                    for return_full_pose in return_full_pose_options
                }
            )
        )

    def set_expected_warmup_keys(
        self,
        keys: tuple[tuple[bool, bool, int], ...],
    ) -> tuple[tuple[bool, bool, int], ...]:
        self._expected_warmup_keys = set(keys)
        return tuple(sorted(self._expected_warmup_keys))

    def warmup_coverage(
        self,
        expected_keys: tuple[tuple[bool, bool, int], ...] | None = None,
    ) -> WarmupCoverage:
        if expected_keys is None:
            if self._expected_warmup_keys:
                expected = tuple(sorted(self._expected_warmup_keys))
            elif self.policy.forbid_new_compiles and self.policy.fixed_padded_batch_size is not None:
                expected = self.expected_warmup_keys(
                    padded_batch_size=int(self.policy.fixed_padded_batch_size),
                )
            else:
                expected = ()
        else:
            expected = tuple(sorted(expected_keys))
        warm = set(self._warmup_keys)
        covered = tuple(key for key in expected if key in warm)
        missing = tuple(key for key in expected if key not in warm)
        return WarmupCoverage(
            expected_keys=expected,
            covered_keys=covered,
            missing_keys=missing,
            ready=len(missing) == 0,
        )

    def _runtime_metadata(self) -> dict[str, str | bool]:
        devices = jax.devices()
        default = devices[0] if devices else None
        return {
            "jax_backend": jax.default_backend(),
            "device_kind": getattr(default, "device_kind", "unknown") if default is not None else "unknown",
            "platform": getattr(default, "platform", "unknown") if default is not None else "unknown",
            "jax_version": getattr(jax, "__version__", "unknown"),
            "python_version": platform.python_version(),
            "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        }

    def _strict_mode_status(self) -> tuple[bool, str]:
        if not self.policy.forbid_new_compiles:
            return False, "strict compile enforcement is disabled"
        if self.policy.fixed_padded_batch_size is None:
            return False, "strict compile enforcement requires fixed_padded_batch_size"
        coverage = self.warmup_coverage()
        if not coverage.expected_keys:
            return False, "no expected warmup targets are configured"
        if not coverage.ready:
            return False, f"missing warmup targets: {coverage.missing_keys}"
        resident_missing = tuple(key for key in coverage.expected_keys if key not in self._compiled)
        if resident_missing:
            return False, f"required compiled targets were evicted: {resident_missing}"
        return True, f"warmup recorded for targets {coverage.expected_keys}"

    def assert_strict_ready(
        self,
        expected_keys: tuple[tuple[bool, bool, int], ...] | None = None,
    ) -> None:
        if expected_keys is not None:
            self.set_expected_warmup_keys(expected_keys)
        ready, reason = self._strict_mode_status()
        if not ready:
            raise RuntimeError(f"Strict runtime is not ready: {reason}")

    def diagnostics(self) -> RuntimeDiagnostics:
        strict_mode_ready, strict_mode_reason = self._strict_mode_status()
        metadata = self._runtime_metadata()
        coverage = self.warmup_coverage()
        return RuntimeDiagnostics(
            dtype=str(self.dtype),
            compile_count=self._compile_count,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            compiled_entries=len(self._compiled),
            compiled_keys=tuple(self._compiled.keys()),
            model_bytes=self._model_bytes,
            last_input_bytes=self._last_input_bytes,
            last_output_bytes=self._last_output_bytes,
            batch_buckets=tuple(self.policy.batch_buckets),
            fixed_padded_batch_size=self.policy.fixed_padded_batch_size,
            forbid_new_compiles=self.policy.forbid_new_compiles,
            strict_mode_ready=strict_mode_ready,
            strict_mode_reason=strict_mode_reason,
            warmup_keys=tuple(sorted(self._warmup_keys)),
            expected_warmup_keys=coverage.expected_keys,
            missing_warmup_keys=coverage.missing_keys,
            missing_resident_keys=tuple(key for key in coverage.expected_keys if key not in self._compiled),
            compile_history=tuple(self._compile_history),
            evicted_keys=tuple(self._evicted_keys),
            eviction_count=len(self._evicted_keys),
            jax_backend=str(metadata["jax_backend"]),
            device_kind=str(metadata["device_kind"]),
            platform=str(metadata["platform"]),
            jax_version=str(metadata["jax_version"]),
            python_version=str(metadata["python_version"]),
            x64_enabled=bool(metadata["x64_enabled"]),
        )

    def clear_caches(self) -> None:
        self._compiled.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._compile_history.clear()
        self._warmup_keys.clear()
        self._expected_warmup_keys.clear()
        self._evicted_keys.clear()
