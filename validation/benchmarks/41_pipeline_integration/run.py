#!/usr/bin/env python3
"""Benchmark 41: Pipeline Integration — import all pipeline modules."""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.plugins.apps import full_pipeline
        checks["import_full_pipeline"] = True

        from helixlang.plugins.apps import gem_pipeline
        checks["import_gem_pipeline"] = True

        from helixlang.plugins.apps import population_calibration
        checks["import_population_calibration"] = True

        from helixlang.plugins.apps import virtual_cell_bench
        checks["import_virtual_cell_bench"] = True

        assert hasattr(full_pipeline, "run_full_pipeline"), (
            "full_pipeline should have run_full_pipeline"
        )
        checks["full_pipeline_has_run_function"] = True

        assert hasattr(gem_pipeline, "GemPipelineResult"), (
            "gem_pipeline should have GemPipelineResult"
        )
        checks["gem_pipeline_has_result_class"] = True

        assert hasattr(population_calibration, "PopulationCalibration"), (
            "population_calibration should have PopulationCalibration"
        )
        checks["population_calibration_has_class"] = True

        assert hasattr(virtual_cell_bench, "VirtualCellBench"), (
            "virtual_cell_bench should have VirtualCellBench"
        )
        checks["virtual_cell_bench_has_class"] = True

        details["modules"] = [
            "full_pipeline",
            "gem_pipeline",
            "population_calibration",
            "virtual_cell_bench",
        ]

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "41_pipeline_integration",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "41_pipeline_integration",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
