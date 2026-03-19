from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def _default_validated_root() -> Path:
    return Path(__file__).resolve().parents[2] / "private_data" / "smpl" / "models" / "validated"


def _load_summary_or_none(summary_path: Path) -> dict[str, object] | None:
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid summary JSON payload in {summary_path}")
    return payload


def _collect_entries(validated_root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for model_path in sorted(validated_root.rglob("*.npz")):
        summary_path = model_path.with_suffix(".summary.json")
        summary = _load_summary_or_none(summary_path)
        if summary is None:
            from smpljax.io import describe_model

            summary_payload = asdict(describe_model(model_path))
            summary = {
                "validated": False,
                "input_path": str(model_path),
                **summary_payload,
            }
        entries.append(
            {
                "path": str(model_path),
                "relative_path": str(model_path.relative_to(validated_root)),
                "model_family": model_path.parent.name,
                "has_summary_json": summary_path.exists(),
                **summary,
            }
        )
    return entries


def _print_human(entries: list[dict[str, object]], validated_root: Path) -> None:
    print(f"validated_root: {validated_root}")
    if not entries:
        print("models: 0")
        return

    print(f"models: {len(entries)}")
    for entry in entries:
        print(
            "model: "
            f"{entry['relative_path']} "
            f"family={entry['model_family']} "
            f"verts={entry['num_vertices']} joints={entry['num_joints']} "
            f"betas={entry['num_betas']} expr={entry['num_expression_coeffs']} "
            f"body={entry['num_body_joints']} hand={entry['num_hand_joints']} face={entry['num_face_joints']} "
            f"faces={entry['has_faces']} landmarks={entry['has_landmarks']} "
            f"dynamic_landmarks={entry['has_dynamic_landmarks']} use_pca={entry['use_pca']} "
            f"summary_json={entry['has_summary_json']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="List canonical validated SMPL-family model assets.")
    parser.add_argument("--validated-root", type=Path, default=_default_validated_root())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    entries = _collect_entries(args.validated_root)
    payload = {
        "validated_root": str(args.validated_root),
        "count": len(entries),
        "models": entries,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    _print_human(entries, args.validated_root)


if __name__ == "__main__":
    main()
