#!/usr/bin/env python3
"""Benchmark 36: Omics Integration — expression inference and spatial omics."""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.omics._spatial_omics import (
            ExpressionMatrix,
            SpatialAtlas,
            from_arrays,
        )
        from helixlang.omics.expression_inference import (
            ExpressionModel,
            ExpressionState,
            hill_function,
        )

        checks["import_omics_modules"] = True

        required_classes = {
            "ExpressionModel": ExpressionModel,
            "ExpressionState": ExpressionState,
            "ExpressionMatrix": ExpressionMatrix,
            "SpatialAtlas": SpatialAtlas,
        }
        for name, cls in required_classes.items():
            assert cls is not None, f"{name} should be importable"
        checks["expression_model_classes_exist"] = True

        h0 = hill_function(0.0, 1.0, 1.0)
        details["hill_0_1_1"] = h0
        assert abs(h0 - 0.0) < 1e-9, f"hill(0,1,1) should be 0.0, got {h0}"
        checks["hill_function_zero"] = True

        h1 = hill_function(1.0, 1.0, 1.0)
        details["hill_1_1_1"] = h1
        assert abs(h1 - 0.5) < 1e-9, f"hill(1,1,1) should be 0.5, got {h1}"
        checks["hill_function_half_max"] = True

        model = ExpressionModel()
        model.promoter_strength["geneA"] = 0.8
        model.promoter_strength["geneB"] = 1.0
        model.rbs_strength["geneA"] = 0.9
        model.rbs_strength["geneB"] = 1.0
        model.tf_effects["geneA"] = [("TF1", 0.5, 2.0, "activation")]
        details["expression_model_created"] = True

        em = from_arrays(
            genes=["g1", "g2", "g3"],
            cells=["c1", "c2", "c3"],
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        )
        assert em.shape == (3, 3), f"Shape should be (3,3), got {em.shape}"
        norm = em.normalized(method="max")
        assert norm.values[0][0] == 1.0 / 7.0, "Normalization should divide by max"
        details["expression_matrix_shape"] = em.shape

        atlas = SpatialAtlas(
            spots=[(0.0, 0.0, 0.0), (10.0, 10.0, 0.0)],
            state_ids=[0, 1],
        )
        assert atlas.state_at(1.0, 1.0) == 0
        assert atlas.state_at(9.0, 9.0) == 1
        details["spatial_atlas_works"] = True

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "36_omics_integration",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "36_omics_integration",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
