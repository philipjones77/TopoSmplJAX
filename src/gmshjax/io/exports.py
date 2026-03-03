"""Mesh export adapters (initial stubs)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import jax.numpy as jnp
import json
import numpy as np


def as_array(points: jnp.ndarray):
    """Return array-like object for downstream export integrations."""
    return points


def export_snapshot_npz(
    path: str | Path,
    points,
    elements,
    metrics: Mapping[str, float] | None = None,
) -> None:
    """Write a compact mesh snapshot for debugging/visualization."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "points": np.asarray(points),
        "elements": np.asarray(elements),
    }
    if metrics:
        for k, v in metrics.items():
            data[f"metric_{k}"] = np.asarray(v)
    np.savez(p, **data)


def load_snapshot_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load snapshot arrays from `.npz`."""
    arr = np.load(Path(path))
    return {k: arr[k] for k in arr.files}


def export_metrics_json(path: str | Path, metrics: Mapping[str, float]) -> None:
    """Write scalar diagnostics as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2, sort_keys=True)
