#!/usr/bin/env python3
"""Run all validation benchmarks, save results, and generate report.

Single entry point for the full validation pipeline:
  1. Run every benchmark in benchmarks/
  2. Save full JSON to results/
  3. Normalize each result via EvidenceChain.from_dict()
  4. Generate validation/report.md with complete evidence chains

Usage:
    python validation/run_all.py              # run + report
    python validation/run_all.py --parallel   # parallel benchmark dispatch (doc/39 O9)
    python validation/run_all.py --parallel 4 # exactly 4 workers
    python validation/run_all.py --report-only  # report from existing results/
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import VALID_LEVELS, EvidenceChain

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"
RESULTS_DIR = Path(__file__).parent / "results"
REPORT_PATH = Path(__file__).parent / "report.md"


def _load_yaml(path: Path) -> dict:
    """Load a benchmark.yaml, tolerating parse failures (returns {})."""
    try:
        import yaml
        data = yaml.safe_load(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(benchmark_dir: Path) -> dict:
    """Run a single benchmark and return its result dict."""
    run_py = benchmark_dir / "run.py"
    if not run_py.exists():
        return {"id": benchmark_dir.name, "status": "SKIP", "error": "no run.py"}

    t0 = time.perf_counter()
    try:
        # Each benchmark runs as its own subprocess, so dispatch stays
        # deterministic in output regardless of thread scheduling.
        result = subprocess.run(
            [sys.executable, "-B", str(run_py)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            # A benchmark can legitimately exit nonzero for SKIP (e.g. an
            # unavailable external tool, doc/41 §2.2: cannot run → skip).
            # Honor its JSON status when present so a skip never downgrades
            # to FAIL in the gate summary.
            try:
                data = json.loads(result.stdout)
                if isinstance(data, dict) and data.get("status") == "SKIP":
                    data["runtime_seconds"] = elapsed
                    return data
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
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


def run_benchmarks(benchmarks: list[Path], workers: int = 1) -> list[dict]:
    """Run benchmarks, optionally in parallel (doc/39 O9-part-2).

    Each benchmark already runs in its own subprocess, so parallelism uses a
    thread pool that releases the GIL while the subprocess runs — safe to use
    on any platform, with output ordered identically to the serial run.
    """
    if workers <= 1 or len(benchmarks) <= 1:
        return [run_benchmark(b) for b in benchmarks]
    results: list[dict] = [None] * len(benchmarks)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=min(workers, len(benchmarks))) as pool:
        futures = {pool.submit(run_benchmark, b): i
                   for i, b in enumerate(benchmarks)}
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
    return results


def merge_metadata(results: list[dict]) -> list[dict]:
    """Merge declarative benchmark.yaml metadata (layer/level/name) into results.

    The yaml is the single source of truth for ``layer`` and the new canonical
    ``level`` (doc/41 §2.4, §3): fixes the historically empty Layer column for
    71/75 rows and the run.py/yaml double-writes (10_whole_cell run.py:157,
    30_pk run.py:83 emit non-yaml layer values).
    """
    meta: dict[str, dict] = {}
    for d in sorted(BENCHMARKS_DIR.iterdir()):
        if d.is_dir():
            meta[d.name] = _load_yaml(d / "benchmark.yaml")

    merged: list[dict] = []
    for r in results:
        bid = r.get("id", "")
        y = meta.get(bid, {})
        out = dict(r)
        if not out.get("layer") and y.get("layer"):
            out["layer"] = y["layer"]
        if y.get("layer"):
            out["layer"] = y["layer"]
        if not out.get("name") and y.get("name"):
            out["name"] = y["name"]
        if y.get("level"):
            out["level"] = y["level"]
        # Backfill reference.doi from the declarative yaml so the L3 level gate
        # (and the report reference column) is satisfied even when run.py does
        # not repeat the citation (01/02/…).
        ref = out.get("reference")
        if y.get("reference_doi"):
            if isinstance(ref, dict):
                if not ref.get("doi"):
                    ref = dict(ref)
                    ref["doi"] = y["reference_doi"]
            else:
                ref = {"doi": y["reference_doi"]}
            out["reference"] = ref
        merged.append(out)
    return merged


# ── Report generator (uses schema.py as single source of truth) ──────────────

def generate_report(results: list[dict]) -> tuple[str, int]:
    """Generate markdown report from results using EvidenceChain normalization.

    Returns ``(report_markdown, gate_violation_count)`` so the caller can fail
    CI when a level-gate violation occurs (doc/42 VD-3: gate violations are a
    hard failure, not informational).
    """
    chains: list[EvidenceChain] = []
    for r in results:
        chains.append(EvidenceChain.from_dict(r))

    pass_count = sum(1 for c in chains if c.status == "PASS")
    fail_count = sum(1 for c in chains if c.status in ("FAIL", "TIMEOUT", "ERROR"))
    skip_count = sum(1 for c in chains if c.status == "SKIP")
    total = len(chains)

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    rows: list[str] = []
    failures: list[str] = []
    gate_warnings: list[str] = []
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
        gate_warnings.extend(c.level_gate_violations())
        rows.append(
            f"| {c.benchmark_id} | {c.name} | {c.layer} | {c.level} | "
            f"{evidence} | {icon} |"
        )

    level_counts = {lv: sum(1 for c in chains if c.level == lv) for lv in VALID_LEVELS}

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
        "| Validation levels | " + " · ".join(
            f"{lv}×{level_counts[lv]}" for lv in VALID_LEVELS
        ) + " |",
        "| Level-gate violations | " + (f"{len(gate_warnings)}" if gate_warnings else "0") + " |",
        "",
        "## Evidence Chains",
        "",
        "| # | Benchmark | Layer | Level | Reference → Expected → Actual → Error | Status |",
        "|---|-----------|-------|-------|---------------------------------------|--------|",
    ]
    lines.extend(rows)
    lines.append("")
    lines.append(f"**{pass_count}/{total} benchmarks passed.**")

    if gate_warnings:
        lines.append("")
        lines.append("## Level-Gate Violations (doc/41 §3.2 Rule 5 — hard failure)")
        lines.append("")
        lines.extend(f"- {w}" for w in gate_warnings)

    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        lines.extend(failures)

    return "\n".join(lines) + "\n", len(gate_warnings)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    report_only = "--report-only" in sys.argv
    workers = 1
    if "--parallel" in sys.argv:
        i = sys.argv.index("--parallel")
        if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
            workers = int(sys.argv[i + 1])
        else:
            workers = os.cpu_count() or 1

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

        results = run_benchmarks(benchmarks, workers=workers)
        for result in results:
            bid = result.get("id", "?")
            status = result.get("status", "?")
            icon = "✓" if status == "PASS" else "✗"
            print(f"  {icon} {bid}: {status}", file=sys.stderr)

    # Merge declarative benchmark.yaml metadata (layer/level/name, doc/41 §3)
    results = merge_metadata(results)
    # Persist each merged result so --report-only reproduces the same output
    for r in results:
        if r.get("id"):
            (RESULTS_DIR / f"{r['id']}.json").write_text(json.dumps(r, indent=2))

    # Generate report
    report, gate_violations = generate_report(results)
    REPORT_PATH.write_text(report)
    print(f"\nReport → {REPORT_PATH}")

    # Print summary
    passed = sum(1 for r in results if r.get("status") == "PASS")
    skipped = sum(1 for r in results if r.get("status") == "SKIP")
    failed = sum(1 for r in results if r.get("status") in ("FAIL", "TIMEOUT", "ERROR"))
    print(f"Summary: {len(results)} total, {passed} PASS, {skipped} SKIP, {failed} FAIL")
    if gate_violations:
        print(f"Level-gate violations: {gate_violations} (doc/42 VD-3 — hard failure)",
              file=sys.stderr)

    # Exit code: nonzero if a benchmark failed OR a level-gate violation
    # occurred.  SKIP is a success (doc/41 §2.2 — an unavailable external
    # artefact must never fail the gate); level-gate violations are a hard
    # CI failure (doc/42 VD-3).
    sys.exit(1 if (failed or gate_violations) else 0)


if __name__ == "__main__":
    main()
