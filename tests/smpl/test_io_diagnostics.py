import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from smpljax import describe_model


def _write_model(path) -> None:
    np.savez(
        path,
        v_template=np.zeros((2, 3), dtype=np.float32),
        shapedirs=np.zeros((2, 3, 1), dtype=np.float32),
        posedirs=np.zeros((9, 6), dtype=np.float32),
        J_regressor=np.zeros((2, 2), dtype=np.float32),
        weights=np.ones((2, 2), dtype=np.float32) / 2.0,
        parents=np.array([-1, 0], dtype=np.int32),
    )


def test_describe_model_returns_structured_summary(tmp_path) -> None:
    path = tmp_path / "model.npz"
    _write_model(path)
    summary = describe_model(path)
    assert summary.num_vertices == 2
    assert summary.num_joints == 2
    assert summary.has_faces is False


def test_validate_model_tool_supports_json_output(tmp_path) -> None:
    path = tmp_path / "model.npz"
    _write_model(path)
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    out = subprocess.run(
        [sys.executable, "tools/smpl/validate_model.py", "--input", str(path), "--json"],
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload["num_vertices"] == 2
    assert payload["num_joints"] == 2
