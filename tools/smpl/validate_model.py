from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

from smpljax.disk import atomic_copy2, atomic_write_json
from smpljax.io import load_model_uncached
from smpljax.validation import summarize_model_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a SMPL-family model package for smplJAX.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--promote-dir", type=Path, default=None)
    args = parser.parse_args()

    model = load_model_uncached(args.input)
    summary = summarize_model_data(model)
    payload = {
        "validated": True,
        "input_path": str(args.input),
        **asdict(summary),
    }
    if args.summary_json is not None:
        atomic_write_json(args.summary_json, payload)
    if args.json:
        import json

        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"validated: {args.input}")
        print(
            "summary: "
            f"verts={summary.num_vertices} joints={summary.num_joints} "
            f"betas={summary.num_betas} expr={summary.num_expression_coeffs} "
            f"body={summary.num_body_joints} hand={summary.num_hand_joints} face={summary.num_face_joints}"
        )
        print(
            "features: "
            f"faces={summary.has_faces} landmarks={summary.has_landmarks} "
            f"dynamic_landmarks={summary.has_dynamic_landmarks} use_pca={summary.use_pca}"
        )
    if args.promote_dir is not None:
        args.promote_dir.mkdir(parents=True, exist_ok=True)
        dest = args.promote_dir / args.input.name
        if args.input.resolve() != dest.resolve():
            atomic_copy2(args.input, dest)
        summary_path = dest.with_suffix(".summary.json")
        atomic_write_json(summary_path, payload)
        print(f"promoted: {dest}")
        print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
