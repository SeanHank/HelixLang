#!/usr/bin/env python3
"""Run all validation benchmarks and generate a report."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"
RESULTS_DIR = Path(__file__).parent / "results"
REPORT_PATH = Path(__file__).parent / "report.md"


def run_benchmark(benchmark_dir: Path) -> dict:
    """Run a single benchmark and return its result dict."""
    run_py = benchmark_dir / "run.py"
    if not run_py.exists():
        return {"id": benchmark_dir.name, "status": "SKIP", "error": "no run.py"}

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, str(run_py)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            return {
                "id": benchmark_dir.name,
                "status": "FAIL",
                "error": result.stderr[:500],
                "runtime_seconds": elapsed,
            }
        data = json.loads(result.stdout)
        data["runtime_seconds"] = elapsed
        return data
    except subprocess.TimeoutExpired:
        return {
            "id": benchmark_dir.name,
            "status": "TIMEOUT",
            "error": "exceeded 120s",
            "runtime_seconds": 120.0,
        }
    except Exception as e:
        return {"id": benchmark_dir.name, "status": "ERROR", "error": str(e)}


def generate_report(results: list[dict]) -> str:
    """Generate a markdown report from benchmark results."""
    lines = [
        "# HelixLang Validation Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n",
        "| # | Benchmark | Status | Runtime |",
        "|---|-----------|--------|---------|",
    ]
    passed = sum(1 for r in results if r.get("status") == "PASS")
    for r in results:
        status = r.get("status", "ERROR")
        rt = f"{r.get('runtime_seconds', 0):.3f}s"
        lines.append(f"| {r['id']} | {r['id']} | {status} | {rt} |")
    lines.append(f"\n**{passed}/{len(results)} benchmarks passed.**\n")
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    benchmarks = sorted(BENCHMARKS_DIR.iterdir())
    benchmarks = [b for b in benchmarks if b.is_dir()]

    results = []
    for bench in benchmarks:
        print(f"Running {bench.name}...", file=sys.stderr)
        result = run_benchmark(bench)
        results.append(result)

        # Save individual result
        out = RESULTS_DIR / f"{bench.name}.json"
        out.write_text(json.dumps(result, indent=2))

    # Generate report
    report = generate_report(results)
    REPORT_PATH.write_text(report)
    print(report)

    # Exit code: 0 if all pass
    failed = sum(1 for r in results if r.get("status") != "PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
