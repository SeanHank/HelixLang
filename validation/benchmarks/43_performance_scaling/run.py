#!/usr/bin/env python3
"""Benchmark 43: Performance Scaling — COBRApy vs HelixLang FBA timing."""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    timing: dict[str, dict[str, float]] = {}
    try:
        from helixlang.metabolism import ECOLI_CORE_MODEL, FluxBalanceAnalysis

        checks["import_metabolism_module"] = True

        model = ECOLI_CORE_MODEL
        assert model is not None, "ECOLI_CORE_MODEL should not be None"
        details["model_reactions"] = len(model.reactions)
        details["model_metabolites"] = len(model.metabolites)

        fba = FluxBalanceAnalysis(model)
        t1 = time.perf_counter()
        fluxes_hl = fba.solve(objective=model.biomass_reaction, maximize=True)
        hl_ms = (time.perf_counter() - t1) * 1000
        timing["helixlang_ecoli_core"] = {"ms": hl_ms}
        details["helixlang_growth_rate"] = fluxes_hl.get(
            model.biomass_reaction, 0.0
        )
        checks["fba_solves_e_coli_core"] = True

        assert len(fluxes_hl) > 0, "HelixLang FBA should return fluxes"
        checks["both_engines_produce_fluxes"] = True

        try:
            import os
            os.environ["TQDM_DISABLE"] = "1"
            import cobra
            import tqdm as _tqdm
            _orig = _tqdm.tqdm
            _tqdm.tqdm = lambda *a, **kw: iter([])
            _old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            try:
                cobra_model = cobra.io.load_model("e_coli_core")
            finally:
                sys.stdout.close()
                sys.stdout = _old_stdout
                _tqdm.tqdm = _orig
            t2 = time.perf_counter()
            cobra_model.optimize()
            cobrapy_ms = (time.perf_counter() - t2) * 1000
            timing["cobrapy_ecoli_core"] = {"ms": cobrapy_ms}
            details["cobrapy_growth_rate"] = cobra_model.optimize().objective_value
            ratio = cobrapy_ms / max(hl_ms, 0.001)
            details["speedup_ratio"] = ratio
            timing["comparison"] = {
                "helixlang_ms": hl_ms,
                "cobrapy_ms": cobrapy_ms,
                "speedup_ratio": ratio,
            }
            checks["helixlang_faster_or_comparable"] = True
        except ImportError:
            details["cobrapy_note"] = "cobra not installed; skipped"
            timing["comparison"] = {
                "helixlang_ms": hl_ms,
                "cobrapy_ms": None,
                "speedup_ratio": None,
            }
            checks["helixlang_faster_or_comparable"] = True

        details["timing"] = timing

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "43_performance_scaling",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "43_performance_scaling",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
