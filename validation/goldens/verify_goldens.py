#!/usr/bin/env python3
"""Verify benchmark outputs match golden SHA256 hashes."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
GOLDENS_DIR = Path(__file__).resolve().parent
_PERFORMANCE_ONLY = {"11_performance_comparison"}


def _make_deterministic(d, float_prec=6):
    """Recursively make JSON-serializable data deterministic for hashing."""
    if isinstance(d, dict):
        return {
            k: _make_deterministic(v, float_prec)
            for k, v in d.items()
            if k not in {"runtime_seconds", "ms", "speedup_ratio", "timestamp"}
            and not (isinstance(k, str) and k.endswith(("_ms", "_seconds", "_ratio")))
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


def verify_all() -> dict[str, bool]:
    results = {}
    for bench_dir in sorted(BENCHMARKS_DIR.iterdir()):
        if not bench_dir.is_dir():
            continue
        run_py = bench_dir / "run.py"
        if not run_py.exists():
            continue
        bench_id = bench_dir.name

        if bench_id in _PERFORMANCE_ONLY:
            print(f"  {bench_id}: SKIP (performance-only)")
            results[bench_id] = True
            continue

        golden_hash_path = GOLDENS_DIR / f"{bench_id}.sha256"
        if not golden_hash_path.exists():
            print(f"  {bench_id}: NO GOLDEN (skipped)")
            results[bench_id] = False
            continue

        expected_hash = golden_hash_path.read_text().strip().split()[0]
        try:
            mod = _load_benchmark(run_py)
            output = mod.run()
            stable = _make_deterministic(copy.deepcopy(output))
            canonical = json.dumps(stable, sort_keys=True, indent=2, default=str)
            actual_hash = hashlib.sha256(canonical.encode()).hexdigest()
            match = actual_hash == expected_hash
            results[bench_id] = match
            status = "MATCH" if match else "MISMATCH"
            print(f"  {bench_id}: {status}")
        except Exception as e:
            results[bench_id] = False
            print(f"  {bench_id}: ERROR - {e}")

    return results


if __name__ == "__main__":
    print("Verifying benchmark outputs against golden hashes...")
    results = verify_all()
    matched = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{matched}/{total} golden hashes match (or skipped)")
    sys.exit(0 if matched == total else 1)
