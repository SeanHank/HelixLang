#!/usr/bin/env python3
"""Comprehensive performance profiling harness (doc/37 §3.5).

Profiles the full VM pipeline: compile, GRN, dispatch, and snapshot phases,
comparing Python-dispatch vs accelerated execution.  Outputs a structured
JSON report plus a human-readable summary.

Usage::
    python benchmarks/bench_profile.py [--ticks 1000] [--json report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _build_program(n_genes: int = 8, ticks: int = 500) -> object:
    """Construct a moderately-sized GRN program with regulatory feedback."""
    from helixlang.core.parser import parse_source
    from helixlang.core.semantic import SemanticAnalyzer

    lines: list[str] = []
    for i in range(1, n_genes + 1):
        lines.append(f"#gene name=g{i}")
        lines.append("ATG GCT GGT GCT TAA")
        lines.append("#end")
    for i in range(1, n_genes):
        lines.append(f"#regulate g{i} -> g{i + 1} strength=+0.4")
    lines.append(f"#config ticks={ticks}")
    prog = parse_source("\n".join(lines) + "\n")
    SemanticAnalyzer(prog).check()
    return prog


def run_profile(ticks: int, json_out: str | None) -> None:
    from helixlang.core.performance import VMProfiler

    print(f"Building program (8 genes, {ticks} ticks)...")
    prog = _build_program(n_genes=8, ticks=ticks)

    profiler = VMProfiler(enable_tracemalloc=False)

    print("Profiling...")
    t0 = time.perf_counter()
    result = profiler.profile(prog, max_ticks=ticks, use_accel=True)
    t_elapsed = time.perf_counter() - t0

    report = result.to_dict()
    report["wall_clock_seconds"] = round(t_elapsed, 3)

    if json_out:
        Path(json_out).write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON report written to {json_out}")

    print("\n=== Performance Profile ===")
    print(f"  compile_time_ms     : {report['compile_time_ms']:.2f}")
    print(f"  vm_run_time_ms      : {report['vm_run_time_ms']:.2f}")
    print(f"  total_time_ms       : {report['total_time_ms']:.2f}")
    print(f"  ticks_executed      : {report['ticks_executed']}")
    print(f"  ticks_per_sec       : {report['ticks_per_sec']:.1f}")
    print(f"  trace_entries       : {report['trace_entries']}")
    print(f"  snapshot_interval   : {report['snapshot_interval']}")
    print(f"  peak_memory_bytes   : {report['peak_memory_bytes']}")
    print(f"  acceleration        : {'native C' if report['accel_used'] else 'python'}")
    print("  component_times     :")
    for k, v in report["component_times"].items():
        print(f"      {k:16s}: {v:.3f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=500,
                        help="number of simulation ticks (default 500)")
    parser.add_argument("--json", type=str, default=None,
                        help="write structured JSON report to this path")
    args = parser.parse_args()
    try:
        run_profile(args.ticks, args.json)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
