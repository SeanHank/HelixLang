"""Tests for doc/34 P1: validation benchmarks.

Runs the validation benchmarks and verifies they produce valid results.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

VALIDATION_DIR = Path(__file__).parent.parent / "validation"
BENCHMARKS_DIR = VALIDATION_DIR / "benchmarks"


def _collect_benchmarks() -> list[Path]:
    """Collect all benchmark directories that have a run.py."""
    dirs = []
    for d in sorted(BENCHMARKS_DIR.iterdir()):
        if d.is_dir() and (d / "run.py").exists():
            dirs.append(d)
    return dirs


BENCHMARK_PATHS = _collect_benchmarks()


@pytest.mark.parametrize(
    "bench_dir",
    BENCHMARK_PATHS,
    ids=[d.name for d in BENCHMARK_PATHS],
)
def test_benchmark_runs(bench_dir: Path) -> None:
    """Each benchmark run.py executes successfully and returns valid JSON."""
    result = subprocess.run(
        [sys.executable, str(bench_dir / "run.py")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Benchmark {bench_dir.name} failed:\n{result.stderr[:500]}"
    )

    data = json.loads(result.stdout)
    assert "id" in data, "Missing 'id' field"
    assert "status" in data, "Missing 'status' field"
    assert data["status"] in ("PASS", "SKIP"), (
        f"Benchmark {bench_dir.name} status: {data['status']}"
    )


class TestBenchmarkYAML:
    """Each benchmark has a valid benchmark.yaml."""

    def test_all_benchmarks_have_yaml(self) -> None:
        for d in BENCHMARK_PATHS:
            yaml_path = d / "benchmark.yaml"
            assert yaml_path.exists(), f"Missing benchmark.yaml in {d.name}"

    def test_yaml_has_required_fields(self) -> None:
        for d in BENCHMARK_PATHS:
            content = (d / "benchmark.yaml").read_text()
            assert "id:" in content, f"{d.name}: missing id"
            assert "name:" in content, f"{d.name}: missing name"
            assert "layer:" in content, f"{d.name}: missing layer"
            assert "reference:" in content, f"{d.name}: missing reference"


class TestValidationFramework:
    """The validation framework itself is well-formed."""

    def test_run_all_exists(self) -> None:
        assert (VALIDATION_DIR / "run_all.py").exists()

    def test_readme_exists(self) -> None:
        assert (VALIDATION_DIR / "README.md").exists()

    def test_results_dir_exists(self) -> None:
        results_dir = VALIDATION_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        assert results_dir.is_dir()
