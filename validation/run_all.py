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
from schema import EvidenceChain, LEVEL_NAMES, VALID_LEVELS

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

def generate_report(results: list[dict]) -> str:
    """Generate markdown report from results using EvidenceChain normalization."""
    chains: list[EvidenceChain] = []
    for r in results:
        chains.append(EvidenceChain.from_dict(r))

    pass_count = sum(1 for c in chains if c.status == "PASS")
    fail_count = sum(1 for c in chains if c.status in ("FAIL", "TIMEOUT", "ERROR"))
    skip_count = sum(1 for c in chains if c.status == "SKIP")
    total = len(chains)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

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
        "| Level-gate warnings | " + (f"{len(gate_warnings)}" if gate_warnings else "0") + " |",
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
        lines.append("## Level-Gate Warnings (doc/41 §3.2 Rule 5 — informational)")
        lines.append("")
        lines.extend(f"- {w}" for w in gate_warnings)

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

            status = result.get("status", "?")
            icon = "✓" if status == "PASS" else "✗"
            print(f"  {icon} {bench.name}: {status}")

    # Merge declarative benchmark.yaml metadata (layer/level/name, doc/41 §3)
    results = merge_metadata(results)
    # Persist each merged result so --report-only reproduces the same output
    for r in results:
        if r.get("id"):
            (RESULTS_DIR / f"{r['id']}.json").write_text(json.dumps(r, indent=2))

    # Generate report
    report = generate_report(results)
    REPORT_PATH.write_text(report)
    print(f"\nReport → {REPORT_PATH}")

    # Print summary
    passed = sum(1 for r in results if r.get("status") == "PASS")
    skipped = sum(1 for r in results if r.get("status") == "SKIP")
    failed = sum(1 for r in results if r.get("status") in ("FAIL", "TIMEOUT", "ERROR"))
    print(f"Summary: {len(results)} total, {passed} PASS, {skipped} SKIP, {failed} FAIL")

    # Exit code: 0 unless a benchmark actually failed (SKIP is a success,
    # doc/41 §2.2 — an unavailable external artefact must never fail the gate).
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
