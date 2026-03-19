"""Run the standard Mode 1 regression slice in fresh Python processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _default_targets() -> list[list[str]]:
    return [
        ["tests/topo/test_mode1_fixed_topology.py", "-q"],
        ["tests/topo/test_mode1_arbitrary_domains.py", "-q"],
        ["tests/topo/test_mode1_workflow.py", "-q"],
        ["tests/topo/test_m3_diagnostics_io.py", "-q"],
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standard Mode 1 regression slice in fresh Python processes.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for subprocess test runs.")
    parser.add_argument("pytest_args", nargs="*", help="Optional extra pytest arguments appended to each subprocess run.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]

    targets = _default_targets()
    for index, target_args in enumerate(targets, start=1):
        cmd = [args.python, "-m", "pytest", *target_args, *args.pytest_args]
        print(f"[mode1-regression {index}/{len(targets)}] {' '.join(cmd)}", flush=True)
        completed = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
