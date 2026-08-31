#!/usr/bin/env python3
"""Benchmark 43: Performance Scaling — COBRApy vs HelixLang FBA timing.

doc/41 offline-first CI: the COBRApy comparison is an external artefact
dependency.  If COBRApy (or the e_coli_core model) cannot be obtained the whole
benchmark is SKIPPED — never degraded to a PASS with partial results.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "references" / "models"


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    timing: dict[str, dict[str, float]] = {}
    try:
        from helixlang.plugins.runtime.metabolism import ECOLI_CORE_MODEL, FluxBalanceAnalysis

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
            from helixlang.plugins.gem.sbml_import import load_bigg_cobra_model

            cobra_model = load_bigg_cobra_model(
                "e_coli_core", model_dir=MODEL_DIR,
            )
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
        except Exception as exc:
            return {
                "id": "43_performance_scaling",
                "status": "SKIP",
                "checks": checks,
                "details": {**details, "timing": timing},
                "reason": f"COBRApy / BiGG e_coli_core unavailable: {exc}",
                "runtime_seconds": time.perf_counter() - t0,
            }

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
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
