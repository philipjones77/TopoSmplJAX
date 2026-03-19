from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _write_minimal_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        v_template=np.zeros((3, 3), dtype=np.float32),
        shapedirs=np.zeros((3, 3, 2), dtype=np.float32),
        posedirs=np.zeros((18, 9), dtype=np.float32),
        J_regressor=np.zeros((1, 3), dtype=np.float32),
        weights=np.ones((3, 1), dtype=np.float32),
        parents=np.array([-1], dtype=np.int32),
        f=np.array([[0, 1, 2]], dtype=np.int32),
    )


def test_list_validated_models_json_output() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture_root = repo_root / "tests" / "output" / "test_list_validated_models"
    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    validated_root = fixture_root / "validated"
    model_path = validated_root / "smplx" / "demo_model.npz"
    summary_path = model_path.with_suffix(".summary.json")
    _write_minimal_model(model_path)
    summary_path.write_text(
        json.dumps(
            {
                "validated": True,
                "input_path": str(model_path),
                "num_vertices": 3,
                "num_joints": 1,
                "num_betas": 2,
                "num_expression_coeffs": 0,
                "num_body_joints": 0,
                "num_hand_joints": 0,
                "num_face_joints": 0,
                "has_faces": True,
                "has_landmarks": False,
                "has_dynamic_landmarks": False,
                "use_pca": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "smpl" / "list_validated_models.py"),
            "--validated-root",
            str(validated_root),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(repo_root / "src")},
    )

    payload = json.loads(result.stdout)
    assert payload["count"] == 1
    assert payload["models"][0]["relative_path"] == "smplx\\demo_model.npz" or payload["models"][0]["relative_path"] == "smplx/demo_model.npz"
    assert payload["models"][0]["validated"] is True
    assert payload["models"][0]["has_summary_json"] is True

    shutil.rmtree(fixture_root)
