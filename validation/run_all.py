#!/usr/bin/env python3
"""Run all validation benchmarks, save results, and generate report.

Single entry point for the full validation pipeline:
  1. Run every benchmark in benchmarks/
  2. Save full JSON to results/
  3. Normalize each result via EvidenceChain.from_dict()
  4. Generate validation/report.md with complete evidence chains

Usage:
    python validation/run_all.py              # run + report
    python validation/run_all.py --report-only  # report from existing results/
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import EvidenceChain

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"
RESULTS_DIR = Path(__file__).parent / "results"
REPORT_PATH = Path(__file__).parent / "report.md"


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(benchmark_dir: Path) -> dict:
    """Run a single benchmark and return its result dict."""
    run_py = benchmark_dir / "run.py"
    if not run_py.exists():
        return {"id": benchmark_dir.name, "status": "SKIP", "error": "no run.py"}

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-B", str(run_py)],
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


# ── Report generator (uses schema.py as single source of truth) ──────────────

def generate_report(results: list[dict]) -> str:
    """Generate markdown report from results using EvidenceChain normalization."""
    chains: list[EvidenceChain] = []
    for r in results:
        chains.append(EvidenceChain.from_dict(r))

    pass_count = sum(1 for c in chains if c.status == "PASS")
    fail_count = sum(1 for c in chains if c.status == "FAIL")
    skip_count = sum(1 for c in chains if c.status == "SKIP")
    total = len(chains)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    rows: list[str] = []
    failures: list[str] = []
    for c in chains:
        if c.status == "PASS":
            icon = "✅ PASS"
        elif c.status == "SKIP":
            icon = "⏭ SKIP"
        else:
            icon = "❌ FAIL"
            err_msg = c.error.message or str(c.error)
            failures.append(f"### {c.benchmark_id}\n\n- Status: {c.status}\n- Error: {err_msg}\n")

        evidence = c.fmt_evidence()
        rows.append(f"| {c.benchmark_id} | {c.name} | {c.layer} | {evidence} | {icon} |")

    lines = [
        "# HelixLang Validation Report",
        "",
        f"Generated: {now}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Benchmarks | **{pass_count}/{total}** pass |",
        f"| Failures | {fail_count} |",
        f"| Skipped | {skip_count} |",
        "",
        "## Evidence Chains",
        "",
        "| # | Benchmark | Layer | Reference → Expected → Actual → Error | Status |",
        "|---|-----------|-------|---------------------------------------|--------|",
    ]
    lines.extend(rows)
    lines.append("")
    lines.append(f"**{pass_count}/{total} benchmarks passed.**")

    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        lines.extend(failures)

    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    report_only = "--report-only" in sys.argv

    if report_only:
        # Load existing results
        if not RESULTS_DIR.exists():
            print(f"Error: results dir not found: {RESULTS_DIR}", file=sys.stderr)
            sys.exit(1)
        results = []
        for rf in sorted(RESULTS_DIR.glob("*.json")):
            try:
                results.append(json.loads(rf.read_text()))
            except (json.JSONDecodeError, OSError):
                results.append({"id": rf.stem, "status": "FAIL", "error": "unreadable"})
    else:
        # Run all benchmarks
        RESULTS_DIR.mkdir(exist_ok=True)
        benchmarks = sorted(b for b in BENCHMARKS_DIR.iterdir() if b.is_dir())

        results = []
        for bench in benchmarks:
            print(f"Running {bench.name}...", file=sys.stderr)
            result = run_benchmark(bench)
            results.append(result)

            # Save individual result
            out = RESULTS_DIR / f"{bench.name}.json"
            out.write_text(json.dumps(result, indent=2))

            status = result.get("status", "?")
            icon = "✓" if status == "PASS" else "✗"
            print(f"  {icon} {bench.name}: {status}")

    # Generate report
    report = generate_report(results)
    REPORT_PATH.write_text(report)
    print(f"\nReport → {REPORT_PATH}")

    # Print summary
    passed = sum(1 for r in results if r.get("status") == "PASS")
    failed = sum(1 for r in results if r.get("status") != "PASS")
    print(f"Summary: {len(results)} total, {passed} PASS, {failed} FAIL")

    # Exit code: 0 if all pass
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
