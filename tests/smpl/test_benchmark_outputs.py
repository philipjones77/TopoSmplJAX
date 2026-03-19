import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_OUTPUT_ROOT = REPO_ROOT / "tests" / "output" / "test_benchmark_outputs"


def _run_script(script: str, output_root: Path, *extra_args: str) -> dict:
    out_json = output_root / f"{Path(script).stem}.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [sys.executable, script, "--iters", "2", "--output-json", str(out_json), *extra_args],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and payload
    return payload[0]


def _make_output_root(name: str) -> Path:
    root = TEST_OUTPUT_ROOT / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_forward_benchmark_emits_json() -> None:
    output_root = _make_output_root("forward")
    record = _run_script("benchmarks/smpl/benchmark_forward.py", output_root)
    assert record["benchmark"] == "forward"
    assert record["runtime"] == "baseline"
    assert "jax_backend" in record
    assert "device_kind" in record
    shutil.rmtree(output_root)


def test_optimized_benchmark_emits_json() -> None:
    output_root = _make_output_root("optimized")
    record = _run_script("benchmarks/smpl/benchmark_optimized_runtime.py", output_root)
    assert record["benchmark"] == "optimized_runtime"
    assert record["runtime"] == "optimized"
    assert "fixed_padded_batch_size" in record
    assert "forbid_new_compiles" in record
    shutil.rmtree(output_root)


def test_optimized_benchmark_emits_fixed_batch_policy_json() -> None:
    output_root = _make_output_root("optimized_fixed")
    record = _run_script(
        "benchmarks/smpl/benchmark_optimized_runtime.py",
        output_root,
        "--batch-size",
        "16",
        "--warmup-batch-size",
        "32",
        "--fixed-padded-batch-size",
        "32",
        "--forbid-new-compiles",
    )
    assert record["benchmark"] == "optimized_runtime"
    assert record["fixed_padded_batch_size"] == 32
    assert record["forbid_new_compiles"] is True
    assert record["warmup_batch_size"] == 32
    shutil.rmtree(output_root)
