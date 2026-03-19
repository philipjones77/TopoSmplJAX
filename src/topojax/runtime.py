"""Runtime precision controls for NumPy/JAX."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

_PRECISION = "float32"


def set_runtime_precision(precision: str = "float32") -> None:
    """Set global runtime precision: `float32` or `float64`.

    Call early in process startup before heavy JAX tracing/compilation.
    """
    global _PRECISION
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")
    _PRECISION = precision
    jax.config.update("jax_enable_x64", precision == "float64")


def get_runtime_precision() -> str:
    return _PRECISION


def jax_float_dtype():
    return jnp.float64 if _PRECISION == "float64" else jnp.float32


def numpy_float_dtype():
    return np.float64 if _PRECISION == "float64" else np.float32
