#!/usr/bin/env python3
"""Generate golden outputs for all 45 validation benchmarks.

Runs each benchmark, saves the JSON output, and computes SHA256 hashes.
- Timing fields excluded
- Floats rounded to 6 decimal places
- Lists sorted for deterministic comparison
- Performance-only benchmarks (11) are marked but not hashed
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
GOLDENS_DIR = Path(__file__).resolve().parent
MANIFEST: dict[str, dict] = {}

# Benchmarks that are purely performance measurements (throughput varies per run)
_PERFORMANCE_ONLY = {"11_performance_comparison"}


def _make_deterministic(d, float_prec=6):
    """Recursively make JSON-serializable data deterministic for hashing."""
    if isinstance(d, dict):
        return {
            k: _make_deterministic(v, float_prec)
            for k, v in d.items()
            if k not in {"runtime_seconds", "ms", "speedup_ratio"}
            and not (isinstance(k, str) and k.endswith("_ms"))
            and not (isinstance(k, str) and "timing" in k.lower() and isinstance(v, (int, float)))
        }
    elif isinstance(d, list):
        processed = [_make_deterministic(v, float_prec) for v in d]
        try:
            return sorted(processed, key=lambda x: json.dumps(x, sort_keys=True, default=str))
        except TypeError:
            return processed
    elif isinstance(d, float):
        return round(d, float_prec)
    return d


def _load_benchmark(run_path: Path):
    spec = importlib.util.spec_from_file_location("benchmark", run_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_all() -> dict:
    results = {}
    for bench_dir in sorted(BENCHMARKS_DIR.iterdir()):
        if not bench_dir.is_dir():
            continue
        run_py = bench_dir / "run.py"
        if not run_py.exists():
            continue
        bench_id = bench_dir.name
        print(f"  Running {bench_id}...")
        try:
            mod = _load_benchmark(run_py)
            output = mod.run()
            status = output.get("status", "UNKNOWN")
            if bench_id in _PERFORMANCE_ONLY:
                MANIFEST[bench_id] = {
                    "sha256": "skipped (performance-only)",
                    "status": status,
                    "note": "throughput values vary per run; correctness verified internally",
                }
                results[bench_id] = status
                print(f"    {bench_id}: {status} (performance-only, no golden hash)")
            else:
                stable = _make_deterministic(copy.deepcopy(output))
                canonical = json.dumps(stable, sort_keys=True, indent=2, default=str)
                sha = hashlib.sha256(canonical.encode()).hexdigest()
                golden_path = GOLDENS_DIR / f"{bench_id}.golden.json"
                golden_path.write_text(json.dumps(output, sort_keys=True, indent=2, default=str))
                hash_path = GOLDENS_DIR / f"{bench_id}.sha256"
                hash_path.write_text(f"{sha}  {bench_id}.golden.json\n")
                MANIFEST[bench_id] = {"sha256": sha, "status": status}
                results[bench_id] = status
                print(f"    {bench_id}: {status} (sha256={sha[:16]}...)")
        except Exception as e:
            MANIFEST[bench_id] = {"sha256": "", "status": "ERROR", "error": str(e)}
            results[bench_id] = "ERROR"
            print(f"    {bench_id}: ERROR - {e}")

    manifest_path = GOLDENS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(MANIFEST, sort_keys=True, indent=2) + "\n")
    return results


if __name__ == "__main__":
    print("Generating golden outputs for all 45 benchmarks...")
    results = generate_all()
    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    print(f"\n{passed}/{total} benchmarks passed")
    print(f"Goldens saved to: {GOLDENS_DIR}")
    sys.exit(0 if passed == total else 1)
