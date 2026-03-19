from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from .disk import atomic_write_json, ensure_parent_dir


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def diagnostics_payload(*, runtime: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"runtime": _to_jsonable(runtime.diagnostics())}
    if extra:
        payload["extra"] = _to_jsonable(extra)
    return payload


def write_runtime_diagnostics(
    path: str | Path,
    *,
    runtime: Any,
    extra: dict[str, Any] | None = None,
) -> None:
    atomic_write_json(path, diagnostics_payload(runtime=runtime, extra=extra))


class DiagnosticsLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = ensure_parent_dir(path)

    def append(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(_to_jsonable(event), sort_keys=True))
            f.write("\n")
