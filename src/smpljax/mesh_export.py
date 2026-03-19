from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .body_models import SMPLJAXModel
from .optimized import ForwardInputs, OptimizedSMPLJAX
from .utils import SMPLModelData, as_jax_array


MeshPayload = dict[str, Any]
ModelLike = SMPLJAXModel | OptimizedSMPLJAX


def _require_faces(data: SMPLModelData) -> jax.Array:
    if data.faces_tensor is None:
        raise ValueError("Mesh export requires faces_tensor with fixed triangle topology.")
    return as_jax_array(data.faces_tensor, dtype=jnp.int32)


def _infer_model_family(data: SMPLModelData) -> str:
    if data.model_family:
        return str(data.model_family).lower()
    if int(data.num_face_joints or 0) > 0:
        return "smplx"
    if int(data.num_hand_joints or 0) > 0:
        return "smplh"
    return "smpl"


def _infer_model_variant(data: SMPLModelData) -> str:
    if data.model_variant:
        return str(data.model_variant)
    if data.gender:
        return str(data.gender)
    return "unknown"


def _topology_id(data: SMPLModelData, faces: jax.Array) -> str:
    digest = hashlib.sha1(np.asarray(faces, dtype=np.int32).tobytes()).hexdigest()[:16]
    return f"{_infer_model_family(data)}:{int(data.v_template.shape[0])}:{int(faces.shape[0])}:{digest}"


def _device_kind(arr: jax.Array) -> str:
    try:
        devices = tuple(arr.devices())
    except Exception:
        devices = ()
    if not devices:
        devices = tuple(jax.devices())
    default = devices[0] if devices else None
    return getattr(default, "device_kind", "unknown") if default is not None else "unknown"


def _platform(arr: jax.Array) -> str:
    try:
        devices = tuple(arr.devices())
    except Exception:
        devices = ()
    if not devices:
        devices = tuple(jax.devices())
    default = devices[0] if devices else None
    return getattr(default, "platform", "unknown") if default is not None else "unknown"


def _metadata(
    data: SMPLModelData,
    *,
    nodes: jax.Array,
    faces: jax.Array,
    output_kind: str,
    topology_id: str,
) -> dict[str, Any]:
    return {
        "model_family": _infer_model_family(data),
        "model_variant": _infer_model_variant(data),
        "gender": data.gender or "unknown",
        "dtype": str(nodes.dtype),
        "jax_backend": jax.default_backend(),
        "device_kind": _device_kind(nodes),
        "platform": _platform(nodes),
        "shape_signature": {
            "nodes": tuple(int(x) for x in nodes.shape),
            "faces": tuple(int(x) for x in faces.shape),
        },
        "output_kind": output_kind,
        "fixed_topology": True,
        "topology_contract": "pose and shape updates preserve vertex count and face connectivity",
        "topology_id": topology_id,
        "n_vertices": int(faces.max()) + 1 if int(data.v_template.shape[0]) == 0 else int(data.v_template.shape[0]),
        "n_faces": int(faces.shape[0]),
    }


def _batch_size_from_params(params: Mapping[str, Any]) -> int:
    explicit = params.get("batch_size")
    if explicit is not None:
        return int(explicit)
    for key in (
        "betas",
        "body_pose",
        "global_orient",
        "transl",
        "expression",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "left_hand_pose",
        "right_hand_pose",
    ):
        value = params.get(key)
        if value is not None:
            return int(as_jax_array(value).shape[0])
    raise ValueError("Unable to infer batch size from params. Provide `batch_size` or one batched input array.")


def _export_payload(data: SMPLModelData, *, nodes: jax.Array, output_kind: str) -> MeshPayload:
    faces = _require_faces(data)
    topology_id = _topology_id(data, faces)
    return {
        "nodes": as_jax_array(nodes),
        "faces": faces,
        "topology_id": topology_id,
        "n_vertices": int(data.v_template.shape[0]),
        "n_faces": int(faces.shape[0]),
        "metadata": _metadata(data, nodes=as_jax_array(nodes), faces=faces, output_kind=output_kind, topology_id=topology_id),
    }


def export_template_mesh(model: ModelLike) -> MeshPayload:
    return _export_payload(model.data, nodes=as_jax_array(model.data.v_template), output_kind="template")


def export_posed_mesh(model: ModelLike, params: ForwardInputs | Mapping[str, Any]) -> MeshPayload:
    if isinstance(model, OptimizedSMPLJAX):
        pose2rot = True
        if isinstance(params, ForwardInputs):
            inputs = params
        else:
            pose2rot = bool(params.get("pose2rot", True))
            prepare_kwargs = {k: v for k, v in params.items() if k not in {"batch_size", "pose2rot"}}
            inputs = model.prepare_inputs(
                batch_size=_batch_size_from_params(params),
                pose2rot=pose2rot,
                **prepare_kwargs,
            )
        out = model.forward(inputs, pose2rot=pose2rot)
        return _export_payload(model.data, nodes=out.vertices, output_kind="posed")
    if not isinstance(params, Mapping):
        raise TypeError("SMPLJAXModel export_posed_mesh expects a parameter mapping.")
    pose2rot = bool(params.get("pose2rot", True))
    forward_kwargs = {k: v for k, v in params.items() if k not in {"pose2rot", "batch_size"}}
    out = model(**forward_kwargs, pose2rot=pose2rot)
    return _export_payload(model.data, nodes=out.vertices, output_kind="posed")


def export_ct_mesh_payload_template(model: ModelLike) -> MeshPayload:
    mesh = export_template_mesh(model)
    return {
        "nodes": mesh["nodes"],
        "cells": mesh["faces"],
        "faces": mesh["faces"],
        "topology_id": mesh["topology_id"],
        "n_vertices": mesh["n_vertices"],
        "n_faces": mesh["n_faces"],
        "metadata": mesh["metadata"],
    }


def export_ct_mesh_payload_pose(model: ModelLike, params: ForwardInputs | Mapping[str, Any]) -> MeshPayload:
    mesh = export_posed_mesh(model, params)
    return {
        "nodes": mesh["nodes"],
        "cells": mesh["faces"],
        "faces": mesh["faces"],
        "topology_id": mesh["topology_id"],
        "n_vertices": mesh["n_vertices"],
        "n_faces": mesh["n_faces"],
        "metadata": mesh["metadata"],
    }


def to_randomfields77_static_domain_payload(model: ModelLike) -> MeshPayload:
    return export_ct_mesh_payload_template(model)


def to_randomfields77_dynamic_mesh_state(
    model: ModelLike, params: ForwardInputs | Mapping[str, Any]
) -> MeshPayload:
    return export_ct_mesh_payload_pose(model, params)
