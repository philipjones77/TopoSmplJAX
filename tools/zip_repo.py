from __future__ import annotations

import argparse
import datetime as dt
import re
import zipfile
from pathlib import Path


ZIP_PREFIX = "TopoSmplJAX"
BUNDLE_DIRNAME = "_bundles"

ALLOWED_DIRS = {
    "src",
    "docs",
    "contracts",
    "benchmarks",
    "examples",
    "experiments",
    "tests",
    "tools",
    "outputs",
    "common",
}

ALLOWED_FILES = {
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "pyproject.toml",
    "TopoJAX.code-workspace",
    "py.typed",
}

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "_bundles",
    "build",
    "dist",
    "private_data",
    "results",
    ".idea",
    ".vscode",
}

EXCLUDE_EXTS = {
    ".zip",
    ".whl",
    ".tar",
    ".gz",
    ".7z",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".pkl",
    ".npy",
    ".npz",
    ".pt",
    ".bin",
    ".pyc",
    ".pyo",
}


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts.intersection(EXCLUDE_DIRS):
        return True
    if path.suffix.lower() in EXCLUDE_EXTS:
        return True
    if path.name == ".env":
        return True
    return False


def should_include_outputs_file(path: Path) -> bool:
    return path.name in {"README.md", ".gitkeep"}


def default_zip_name(date_str: str | None) -> str:
    if date_str is None:
        date_str = dt.date.today().isoformat()
    return f"{ZIP_PREFIX}_source_{date_str}.zip"


def validate_zip_name(out_path: Path) -> None:
    pattern = rf"^{re.escape(ZIP_PREFIX)}_source_(\d{{4}}-\d{{2}}-\d{{2}})\.zip$"
    match = re.match(pattern, out_path.name)
    if not match:
        raise SystemExit(
            f"error: output zip name must match "
            f"'{ZIP_PREFIX}_source_YYYY-MM-DD.zip' (got '{out_path.name}')"
        )
    try:
        dt.date.fromisoformat(match.group(1))
    except ValueError as exc:
        raise SystemExit(
            f"error: output zip name must use a valid date; got '{match.group(1)}'"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the repository into a source zip under _bundles/.")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--date", help="Date stamp for the zip (YYYY-MM-DD). Defaults to today.")
    parser.add_argument(
        "--out",
        help=f"Output zip path. Filename must be '{ZIP_PREFIX}_source_YYYY-MM-DD.zip'.",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = (repo / BUNDLE_DIRNAME / default_zip_name(args.date)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validate_zip_name(out_path)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(ALLOWED_FILES):
            path = repo / rel
            if path.is_file() and not should_skip(path):
                zf.write(path, rel)

        for rel_dir in sorted(ALLOWED_DIRS):
            base = repo / rel_dir
            if not base.exists():
                continue
            for path in sorted(base.rglob("*")):
                if path.is_dir():
                    continue
                if should_skip(path):
                    continue
                if rel_dir == "outputs" and not should_include_outputs_file(path):
                    continue
                zf.write(path, path.relative_to(repo).as_posix())

    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
