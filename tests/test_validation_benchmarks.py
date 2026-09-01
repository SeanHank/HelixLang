"""Tests for doc/34 P1: validation benchmarks.

Runs the validation benchmarks and verifies they produce valid results.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

VALIDATION_DIR = Path(__file__).parent.parent / "validation"
BENCHMARKS_DIR = VALIDATION_DIR / "benchmarks"

# Allow importing run_all.py / schema.py from the validation package.
sys.path.insert(0, str(VALIDATION_DIR))


def _collect_benchmarks() -> list[Path]:
    """Collect all benchmark directories that have a run.py."""
    dirs = []
    for d in sorted(BENCHMARKS_DIR.iterdir()):
        if d.is_dir() and (d / "run.py").exists():
            dirs.append(d)
    return dirs


BENCHMARK_PATHS = _collect_benchmarks()


# Per-benchmark subprocess timeout in seconds.
#
# Most benchmarks are quick module smoke tests and run well under a 60 s cap.
# `11_performance_comparison` is a deliberately compute-intensive benchmark that
# times 100 solver runs on both e_coli_core and iML1515 (a ~2.7k-reaction GEM).
# Under heavy concurrent xdist load it can legitimately exceed the generic cap,
# so its hard timeout is intentionally removed (None = no timeout). The benchmark
# itself still has an internal wall-clock budget for the solve loops, so it will
# not hang indefinitely — it just is not killed by the test harness.
_BENCHMARK_TIMEOUTS: dict[str, float | None] = {
    "11_performance_comparison": None,
}


def _benchmark_timeout(name: str) -> float | None:
    return _BENCHMARK_TIMEOUTS.get(name, 60)


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
        timeout=_benchmark_timeout(bench_dir.name),
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
            assert "level:" in content, f"{d.name}: missing level (doc/41 §3)"

    def test_yaml_level_values_valid(self) -> None:
        """Every benchmark declares one canonical L0–L5 level (doc/41 §3)."""
        for d in BENCHMARK_PATHS:
            content = (d / "benchmark.yaml").read_text()
            m = re.search(r"^level:\s*(\S+)", content, flags=re.MULTILINE)
            assert m, f"{d.name}: missing level value"
            assert m.group(1) in ("L0", "L1", "L2", "L3", "L4", "L5"), (
                f"{d.name}: invalid level {m.group(1)!r}"
            )


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

    def test_merge_metadata_fills_layer_level(self) -> None:
        """benchmark.yaml is the single source of layer/level (doc/41 §3)."""
        import run_all
        merged = run_all.merge_metadata([{"id": "03_ecoli_fba"}])
        assert merged[0]["layer"] == "metabolism"
        assert merged[0]["level"] == "L4"

    def test_report_has_level_column_and_skip_counts(self) -> None:
        """Report shows per-level counts; SKIP is a success, not a failure."""
        import run_all
        rep = run_all.generate_report([
            {"id": "a", "status": "SKIP", "reason": "offline"},
            {"id": "b", "status": "PASS", "layer": "metabolism", "level": "L2"},
        ])
        assert "| Level |" in rep
        assert "L2×1" in rep
        assert "| Skipped | 1 |" in rep

    def test_no_level_gate_violations(self) -> None:
        """doc/42 Phase A: every benchmark satisfies its declared L0–L5 gate.

        This is the CI enforcement of the level gate (doc/41 §3.2 Rule 5):
        L2 needs a golden_hash, L3 a reference.doi, L4 an experimental range,
        L5 an external clinical dataset. Any violation is a hard failure so the
        taxonomy stays honest.
        """
        import json as _json

        import run_all
        from schema import EvidenceChain

        results_dir = VALIDATION_DIR / "results"
        results = []
        for rf in sorted(results_dir.glob("*.json")):
            try:
                results.append(_json.loads(rf.read_text()))
            except (_json.JSONDecodeError, OSError):
                continue
        # Results must exist (generated by validation/run_all.py) for this gate.
        assert len(results) >= 82, (
            f"Expected ≥82 persisted results, found {len(results)}. "
            "Run `python validation/run_all.py --parallel 4` first."
        )
        merged = run_all.merge_metadata(results)
        violations: list[str] = []
        for r in merged:
            violations.extend(EvidenceChain.from_dict(r).level_gate_violations())
        assert not violations, (
            f"{len(violations)} level-gate violation(s):\n  "
            + "\n  ".join(violations)
        )

