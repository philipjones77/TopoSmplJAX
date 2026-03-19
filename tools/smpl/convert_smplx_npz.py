"""Convert official SMPL-family model files into an NPZ used by smpljax.

Usage:
    python tools/smpl/convert_smplx_npz.py --input path/to/SMPLX_*.pkl --output model.npz
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from smpljax.disk import atomic_write_json, atomic_write_npz
from smpljax.io import load_model_uncached
from smpljax.validation import summarize_model_data


def _dense_if_sparse(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return x


def _find_joint_kin_chain(joint_id: int, parents: np.ndarray) -> np.ndarray:
    chain = []
    curr = int(joint_id)
    while curr != -1:
        chain.append(curr)
        curr = int(parents[curr])
    return np.asarray(chain, dtype=np.int32)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_validated_root() -> Path:
    return _repo_root() / "private_data" / "smpl" / "models" / "validated"


def _canonical_output_path(validated_root: Path, model_type: str, input_path: Path) -> Path:
    return validated_root / model_type / f"{input_path.stem}.npz"


def _summary_payload(
    *,
    summary,
    input_path: Path,
    output_path: Path,
    model_type: str,
) -> dict[str, object]:
    return {
        "validated": True,
        "model_type": model_type,
        "input_path": str(input_path),
        "output_path": str(output_path),
        **asdict(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validated-root", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--model-type", choices=["smpl", "smplh", "smplx"], default="smplx")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    output_path = args.output
    validated_root = args.validated_root or _default_validated_root()
    if output_path is None:
        output_path = _canonical_output_path(validated_root, args.model_type, args.input)
    with args.input.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    packed = {
        "v_template": np.asarray(data["v_template"], dtype=np.float32),
        "shapedirs": np.asarray(data["shapedirs"], dtype=np.float32),
        "posedirs": np.asarray(data["posedirs"], dtype=np.float32),
        "J_regressor": np.asarray(_dense_if_sparse(data["J_regressor"]), dtype=np.float32),
        "weights": np.asarray(data["weights"], dtype=np.float32),
        "kintree_table": np.asarray(data["kintree_table"], dtype=np.int32),
    }
    if "f" in data:
        packed["faces_tensor"] = np.asarray(data["f"], dtype=np.int32)
    if "lmk_faces_idx" in data:
        packed["lmk_faces_idx"] = np.asarray(data["lmk_faces_idx"], dtype=np.int32)
    if "lmk_bary_coords" in data:
        packed["lmk_bary_coords"] = np.asarray(data["lmk_bary_coords"], dtype=np.float32)
    if "dynamic_lmk_faces_idx" in data:
        packed["dynamic_lmk_faces_idx"] = np.asarray(data["dynamic_lmk_faces_idx"], dtype=np.int32)
    if "dynamic_lmk_bary_coords" in data:
        packed["dynamic_lmk_bary_coords"] = np.asarray(data["dynamic_lmk_bary_coords"], dtype=np.float32)
    if args.model_type == "smplx":
        packed["num_body_joints"] = np.asarray(21, dtype=np.int32)
        packed["num_hand_joints"] = np.asarray(15, dtype=np.int32)
        packed["num_face_joints"] = np.asarray(3, dtype=np.int32)
        packed["num_expression_coeffs"] = np.asarray(10, dtype=np.int32)
        packed["use_face_contour"] = np.asarray(False)
        if "hands_componentsl" in data:
            packed["left_hand_components"] = np.asarray(data["hands_componentsl"], dtype=np.float32)
        if "hands_componentsr" in data:
            packed["right_hand_components"] = np.asarray(data["hands_componentsr"], dtype=np.float32)
        parents = np.asarray(data["kintree_table"], dtype=np.int32)[0]
        packed["neck_kin_chain"] = _find_joint_kin_chain(12, parents)
    elif args.model_type == "smplh":
        packed["num_body_joints"] = np.asarray(21, dtype=np.int32)
        packed["num_hand_joints"] = np.asarray(15, dtype=np.int32)
        packed["num_face_joints"] = np.asarray(0, dtype=np.int32)
        packed["num_expression_coeffs"] = np.asarray(0, dtype=np.int32)
        if "hands_componentsl" in data:
            packed["left_hand_components"] = np.asarray(data["hands_componentsl"], dtype=np.float32)
        if "hands_componentsr" in data:
            packed["right_hand_components"] = np.asarray(data["hands_componentsr"], dtype=np.float32)
    else:
        packed["num_body_joints"] = np.asarray(23, dtype=np.int32)
        packed["num_hand_joints"] = np.asarray(0, dtype=np.int32)
        packed["num_face_joints"] = np.asarray(0, dtype=np.int32)
        packed["num_expression_coeffs"] = np.asarray(0, dtype=np.int32)
    atomic_write_npz(output_path, **packed)
    print(f"Wrote {output_path}")
    if not args.skip_validation:
        model = load_model_uncached(output_path)
        summary = summarize_model_data(model)
        print(
            "Validated output: "
            f"verts={summary.num_vertices} joints={summary.num_joints} "
            f"body={summary.num_body_joints} hand={summary.num_hand_joints} face={summary.num_face_joints}"
        )
        summary_path = args.summary_json or output_path.with_suffix(".summary.json")
        atomic_write_json(
            summary_path,
            _summary_payload(
                summary=summary,
                input_path=args.input,
                output_path=output_path,
                model_type=args.model_type,
            ),
        )
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
