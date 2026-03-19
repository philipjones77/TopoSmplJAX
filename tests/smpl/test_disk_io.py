from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from smpljax.disk import atomic_copy2, atomic_write_csv, atomic_write_json, atomic_write_npz


def _scratch_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / "output" / "test_disk_io"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_atomic_write_json_and_copy() -> None:
    root = _scratch_dir()
    src = root / "src.json"
    dst = root / "dst.json"
    atomic_write_json(src, {"a": 1, "b": True})
    atomic_copy2(src, dst)
    assert json.loads(dst.read_text(encoding="utf-8")) == {"a": 1, "b": True}
    shutil.rmtree(root)


def test_atomic_write_csv() -> None:
    root = _scratch_dir()
    path = root / "table.csv"
    atomic_write_csv(path, [{"b": 2, "a": 1}, {"a": 3, "b": 4}])
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    shutil.rmtree(root)


def test_atomic_write_npz() -> None:
    root = _scratch_dir()
    path = root / "model.npz"
    atomic_write_npz(path, values=np.asarray([1, 2, 3], dtype=np.int32))
    with np.load(path, allow_pickle=True) as payload:
        assert payload["values"].tolist() == [1, 2, 3]
    shutil.rmtree(root)
