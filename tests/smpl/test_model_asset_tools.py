import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_minimal_smpl_pkl(path: Path) -> None:
    num_joints = 24
    payload = {
        "v_template": np.zeros((4, 3), dtype=np.float32),
        "shapedirs": np.zeros((4, 3, 2), dtype=np.float32),
        "posedirs": np.zeros(((num_joints - 1) * 9, 12), dtype=np.float32),
        "J_regressor": np.zeros((num_joints, 4), dtype=np.float32),
        "weights": np.ones((4, num_joints), dtype=np.float32) / num_joints,
        "kintree_table": np.vstack(
            [
                np.array([-1] + list(range(num_joints - 1)), dtype=np.int32),
                np.array([-1] + list(range(num_joints - 1)), dtype=np.int32),
            ]
        ),
        "f": np.array([[0, 1, 2]], dtype=np.int32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_convert_tool_can_write_to_validated_root() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    base = Path("output") / "test-model-assets"
    input_path = base / "minimal.pkl"
    validated_root = base / "validated"
    _write_minimal_smpl_pkl(input_path)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "tools/smpl/convert_smplx_npz.py",
                "--input",
                str(input_path),
                "--model-type",
                "smpl",
                "--validated-root",
                str(validated_root),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        output_path = validated_root / "smpl" / "minimal.npz"
        summary_path = validated_root / "smpl" / "minimal.summary.json"
        assert output_path.exists()
        assert summary_path.exists()
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["validated"] is True
        assert payload["model_type"] == "smpl"
    finally:
        for path in sorted(base.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()


def test_validate_tool_can_promote_validated_model() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    base = Path("output") / "test-model-promote"
    source = base / "source.npz"
    promote_dir = base / "validated"
    base.mkdir(parents=True, exist_ok=True)
    np.savez(
        source,
        v_template=np.zeros((4, 3), dtype=np.float32),
        shapedirs=np.zeros((4, 3, 2), dtype=np.float32),
        posedirs=np.zeros((207, 12), dtype=np.float32),
        J_regressor=np.zeros((24, 4), dtype=np.float32),
        weights=np.ones((4, 24), dtype=np.float32) / 24.0,
        kintree_table=np.vstack(
            [
                np.array([-1] + list(range(23)), dtype=np.int32),
                np.array([-1] + list(range(23)), dtype=np.int32),
            ]
        ),
        faces_tensor=np.array([[0, 1, 2]], dtype=np.int32),
        num_body_joints=np.array(23, dtype=np.int32),
        num_hand_joints=np.array(0, dtype=np.int32),
        num_face_joints=np.array(0, dtype=np.int32),
        num_expression_coeffs=np.array(0, dtype=np.int32),
    )
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "tools/smpl/validate_model.py",
                "--input",
                str(source),
                "--promote-dir",
                str(promote_dir),
                "--json",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        promoted = promote_dir / "source.npz"
        summary = promote_dir / "source.summary.json"
        assert promoted.exists()
        assert summary.exists()
        payload = json.loads(summary.read_text(encoding="utf-8"))
        assert payload["validated"] is True
        assert payload["input_path"].endswith("source.npz")
    finally:
        for path in sorted(base.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
