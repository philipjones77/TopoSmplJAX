from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _atomic_replace(path: Path, write_fn: Callable[[Path], None]) -> None:
    path = ensure_parent_dir(path)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}.",
        suffix=f"{path.suffix}.tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    target = Path(path)
    _atomic_replace(target, lambda tmp_path: tmp_path.write_text(text, encoding=encoding))


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def atomic_write_csv(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    rows = list(records)
    fieldnames = sorted({key for row in rows for key in row.keys()})

    def _write(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    _atomic_replace(Path(path), _write)


def atomic_write_npz(path: str | Path, **arrays: Any) -> None:
    target = Path(path)
    import numpy as np

    def _write(tmp_path: Path) -> None:
        with tmp_path.open("wb") as f:
            np.savez(f, **arrays)

    _atomic_replace(target, _write)


def atomic_copy2(src: str | Path, dst: str | Path) -> None:
    import shutil

    source = Path(src)

    def _write(tmp_path: Path) -> None:
        shutil.copy2(source, tmp_path)

    _atomic_replace(Path(dst), _write)
