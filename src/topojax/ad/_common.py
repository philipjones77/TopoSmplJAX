"""Internal helpers for AD dtype coercion and compiled-function caching."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Callable, TypeVar

import jax.numpy as jnp
import numpy as np

from topojax.mesh.diagnostics import line_diagnostics, quad_diagnostics, tet_diagnostics, tri_diagnostics
from topojax.mesh.topology import MeshTopology
from topojax.runtime import jax_float_dtype


_K = TypeVar("_K")
_V = TypeVar("_V")
_CACHE_LIMIT = 32


def coerce_runtime_points(points: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(points, dtype=jax_float_dtype())


def fit_node_mask(movable_mask: jnp.ndarray | None, n_nodes: int) -> jnp.ndarray | None:
    if movable_mask is None:
        return None
    mask = jnp.asarray(movable_mask, dtype=bool)
    if mask.shape[0] >= n_nodes:
        return mask[:n_nodes]
    pad = jnp.ones((n_nodes - mask.shape[0],), dtype=bool)
    return jnp.concatenate([mask, pad], axis=0)


def array_cache_key(array) -> tuple[str, tuple[int, ...], str] | None:
    if array is None:
        return None
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
    return str(arr.dtype), tuple(int(v) for v in arr.shape), digest


def topology_cache_key(topology: MeshTopology) -> tuple[object, ...]:
    return (
        int(topology.n_nodes),
        array_cache_key(topology.elements),
        array_cache_key(topology.edges),
        array_cache_key(topology.node_ids),
        array_cache_key(topology.element_ids),
        array_cache_key(topology.element_entity_tags),
    )


def cached_build(cache: OrderedDict[_K, _V], key: _K, builder: Callable[[], _V]) -> _V:
    if key in cache:
        value = cache.pop(key)
        cache[key] = value
        return value
    value = builder()
    cache[key] = value
    while len(cache) > _CACHE_LIMIT:
        cache.popitem(last=False)
    return value


def mesh_topology_metrics(points: jnp.ndarray, topology: MeshTopology) -> dict[str, float | int]:
    order = int(topology.elements.shape[1])
    dim = int(points.shape[1])
    if order == 2:
        return line_diagnostics(points, topology.elements)
    if order == 3:
        return tri_diagnostics(points, topology.elements)
    if order == 4 and dim == 2:
        return quad_diagnostics(points, topology.elements)
    if order == 4 and dim == 3:
        return tet_diagnostics(points, topology.elements)
    raise ValueError("Unsupported topology for diagnostics")
