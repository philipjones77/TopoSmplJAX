from __future__ import annotations

import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import SMPLModelData, as_jax_array
from .validation import ModelSummary, summarize_model_data, validate_model_data


@dataclass(frozen=True)
class IOCacheDiagnostics:
    entries: int
    hits: int
    misses: int
    keys: tuple[str, ...]


_IO_CACHE: OrderedDict[tuple[str, int, int], SMPLModelData] = OrderedDict()
_IO_HITS = 0
_IO_MISSES = 0
_IO_MAX_MODELS = 2


def _dense_if_sparse_like(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    if hasattr(x, "todense"):
        return np.asarray(x.todense())
    return np.asarray(x)


def _pick(d: dict[str, Any], *keys: str, required: bool = False) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    if required:
        raise KeyError(f"Missing required keys. Tried: {keys}")
    return None


def _extract_parents(d: dict[str, Any]) -> np.ndarray:
    parents = _pick(d, "parents", required=False)
    if parents is not None:
        parents = np.asarray(parents, dtype=np.int32).reshape(-1)
        if parents.size > 0:
            parents[0] = -1
        return parents

    kintree = _pick(d, "kintree_table", required=True)
    kintree = np.asarray(kintree)
    if kintree.ndim != 2:
        raise ValueError("kintree_table must be 2D")
    if kintree.shape[0] >= 2:
        parents = np.asarray(kintree[0], dtype=np.int32).reshape(-1)
    else:
        parents = np.asarray(kintree.reshape(-1), dtype=np.int32)
    if parents.size > 0:
        parents[0] = -1
    return parents


def _normalize_posedirs(posedirs: np.ndarray) -> np.ndarray:
    posedirs = np.asarray(posedirs)
    if posedirs.ndim == 2:
        return posedirs
    if posedirs.ndim == 3:
        num_pose_basis = posedirs.shape[-1]
        return np.reshape(posedirs, (-1, num_pose_basis)).T
    raise ValueError(f"Unsupported posedirs rank: {posedirs.ndim}")


def _normalize_extra_vertex_ids(v: Any) -> list[int] | None:
    if v is None:
        return None
    arr = np.asarray(v).reshape(-1)
    return [int(x) for x in arr.tolist()]


def _to_model_data(d: dict[str, Any]) -> SMPLModelData:
    v_template = _pick(d, "v_template", required=True)
    shapedirs = _pick(d, "shapedirs", required=True)
    posedirs = _pick(d, "posedirs", required=True)
    j_reg = _pick(d, "J_regressor", "j_regressor", required=True)
    weights = _pick(d, "weights", "lbs_weights", required=True)

    parents = _extract_parents(d)
    num_joints = int(parents.shape[0])
    num_body_joints = max(num_joints - 1, 0)

    lhand = _pick(d, "left_hand_components", "hands_componentsl", required=False)
    rhand = _pick(d, "right_hand_components", "hands_componentsr", required=False)

    model = SMPLModelData(
        v_template=as_jax_array(np.asarray(v_template)),
        shapedirs=as_jax_array(np.asarray(shapedirs)),
        posedirs=as_jax_array(_normalize_posedirs(np.asarray(posedirs))),
        j_regressor=as_jax_array(_dense_if_sparse_like(j_reg)),
        parents=as_jax_array(parents),
        lbs_weights=as_jax_array(np.asarray(weights)),
        num_betas=int(np.asarray(shapedirs).shape[-1]),
        num_expression_coeffs=int(_pick(d, "num_expression_coeffs", required=False) or 0),
        num_body_joints=int(_pick(d, "num_body_joints", required=False) or num_body_joints),
        num_hand_joints=int(_pick(d, "num_hand_joints", required=False) or 0),
        num_face_joints=int(_pick(d, "num_face_joints", required=False) or 0),
        model_family=str(_pick(d, "model_family", "model_type", required=False))
        if _pick(d, "model_family", "model_type", required=False) is not None
        else None,
        model_variant=str(_pick(d, "model_variant", required=False))
        if _pick(d, "model_variant", required=False) is not None
        else None,
        gender=str(_pick(d, "gender", required=False)) if _pick(d, "gender", required=False) is not None else None,
        pose_mean=as_jax_array(np.asarray(_pick(d, "pose_mean", required=False)))
        if _pick(d, "pose_mean", required=False) is not None
        else None,
        use_pca=bool(_pick(d, "use_pca", required=False) or False),
        left_hand_components=as_jax_array(np.asarray(lhand)) if lhand is not None else None,
        right_hand_components=as_jax_array(np.asarray(rhand)) if rhand is not None else None,
        extra_vertex_ids=_normalize_extra_vertex_ids(_pick(d, "extra_vertex_ids", required=False)),
        faces_tensor=as_jax_array(np.asarray(_pick(d, "faces_tensor", "f", "faces", required=False)))
        if _pick(d, "faces_tensor", "f", "faces", required=False) is not None
        else None,
        lmk_faces_idx=as_jax_array(np.asarray(_pick(d, "lmk_faces_idx", required=False)))
        if _pick(d, "lmk_faces_idx", required=False) is not None
        else None,
        lmk_bary_coords=as_jax_array(np.asarray(_pick(d, "lmk_bary_coords", required=False)))
        if _pick(d, "lmk_bary_coords", required=False) is not None
        else None,
        dynamic_lmk_faces_idx=as_jax_array(np.asarray(_pick(d, "dynamic_lmk_faces_idx", required=False)))
        if _pick(d, "dynamic_lmk_faces_idx", required=False) is not None
        else None,
        dynamic_lmk_bary_coords=as_jax_array(np.asarray(_pick(d, "dynamic_lmk_bary_coords", required=False)))
        if _pick(d, "dynamic_lmk_bary_coords", required=False) is not None
        else None,
        neck_kin_chain=as_jax_array(np.asarray(_pick(d, "neck_kin_chain", required=False)))
        if _pick(d, "neck_kin_chain", required=False) is not None
        else None,
        use_face_contour=bool(_pick(d, "use_face_contour", required=False) or False),
    )
    validate_model_data(model)
    return model


def load_model_npz(path: str | Path) -> SMPLModelData:
    with np.load(path, allow_pickle=True) as data:
        d = {k: data[k] for k in data.files}
    return _to_model_data(d)


def load_model_pkl(path: str | Path) -> SMPLModelData:
    with Path(path).open("rb") as f:
        d = pickle.load(f, encoding="latin1")
    if not isinstance(d, dict):
        raise ValueError("Expected pickle model payload to be a dict-like object")
    return _to_model_data(d)


def load_model_uncached(path: str | Path) -> SMPLModelData:
    p = Path(path)
    if p.suffix.lower() == ".npz":
        return load_model_npz(p)
    if p.suffix.lower() == ".pkl":
        return load_model_pkl(p)
    raise ValueError(f"Unsupported model file type: {p.suffix}. Expected .npz or .pkl")


def load_model(path: str | Path, use_cache: bool = True, max_entries: int = 2) -> SMPLModelData:
    if use_cache:
        return load_model_cached(path, max_entries=max_entries)
    return load_model_uncached(path)


def describe_model(path: str | Path, use_cache: bool = False, max_entries: int = 2) -> ModelSummary:
    return summarize_model_data(load_model(path, use_cache=use_cache, max_entries=max_entries))


def _cache_key(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return (str(path.resolve()).lower(), int(st.st_mtime_ns), int(st.st_size))


def load_model_cached(path: str | Path, max_entries: int = 2) -> SMPLModelData:
    global _IO_HITS, _IO_MISSES
    p = Path(path)
    key = _cache_key(p)
    if key in _IO_CACHE:
        _IO_HITS += 1
        item = _IO_CACHE.pop(key)
        _IO_CACHE[key] = item
        return item

    _IO_MISSES += 1
    model = load_model_uncached(p)

    _IO_CACHE[key] = model
    cap = min(_IO_MAX_MODELS, max(1, int(max_entries)))
    while len(_IO_CACHE) > cap:
        _IO_CACHE.popitem(last=False)
    return model


def io_cache_diagnostics() -> IOCacheDiagnostics:
    return IOCacheDiagnostics(
        entries=len(_IO_CACHE),
        hits=_IO_HITS,
        misses=_IO_MISSES,
        keys=tuple(k[0] for k in _IO_CACHE.keys()),
    )


def clear_io_cache() -> None:
    global _IO_HITS, _IO_MISSES
    _IO_CACHE.clear()
    _IO_HITS = 0
    _IO_MISSES = 0
