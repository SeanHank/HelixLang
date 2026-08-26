#!/usr/bin/env python3
"""Benchmark 04: E. coli iML1515 genome-scale FBA.

Tries to download the iML1515 model from BiGG via COBRApy,
runs FBA with GLC uptake = 10, and compares growth rate against
the expected ~0.871 h⁻¹.  Skips gracefully if the model is
unavailable (network error or COBRApy not installed).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REF_DIR = Path(__file__).resolve().parents[2] / "references"
EXPECTED_GROWTH = 0.871
TOLERANCE = 0.05  # 5 % relative


def run() -> dict:
    t0 = time.perf_counter()

    # ── Step 1: try to load iML1515 via COBRApy ────────────────────────
    try:
        import io
        import os
        os.environ["TQDM_DISABLE"] = "1"
        import tqdm as _tqdm_mod
        _orig_tqdm = _tqdm_mod.tqdm
        _tqdm_mod.tqdm = lambda *a, **kw: _orig_tqdm(*a, **{**kw, "disable": True})

        import cobra

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            cobra_model = cobra.io.load_model("iML1515")
        finally:
            sys.stdout = old_stdout
    except ImportError:
        return {
            "id": "04_iml1515_fba",
            "status": "SKIP",
            "reason": "COBRApy not installed",
            "runtime_seconds": time.perf_counter() - t0,
        }
    except Exception as exc:
        return {
            "id": "04_iml1515_fba",
            "status": "SKIP",
            "reason": f"Could not download iML1515 from BiGG: {exc}",
            "runtime_seconds": time.perf_counter() - t0,
        }

    try:
        # ── Step 2: set GLC uptake = 10 and solve with COBRApy ──────────
        cobra_model.solver = "glpk"
        ex_rxn = cobra_model.reactions.get_by_id("EX_glc__D_e")
        ex_rxn.lower_bound = -10.0
        ex_rxn.upper_bound = 0.0
        cobra_sol = cobra_model.optimize()
        cobra_growth = float(cobra_sol.objective_value)
        cobra_fluxes = {r.id: float(cobra_sol.fluxes[r.id])
                        for r in cobra_model.reactions}

        # Save reference fluxes
        REF_DIR.mkdir(parents=True, exist_ok=True)
        ref_path = REF_DIR / "iml1515_fluxes.json"
        with open(ref_path, "w") as fh:
            json.dump(cobra_fluxes, fh, indent=2)

        # ── Step 3: convert to HelixLang and run FBA ───────────────────
        from helixlang.metabolism import FluxBalanceAnalysis, _from_cobra_model

        helix_model = _from_cobra_model(cobra_model)
        fba = FluxBalanceAnalysis(helix_model)
        fba.set_uptake("glc_e", 10.0)
        helix_fluxes = fba.solve(objective="biomass")
        helix_growth = helix_fluxes.get(helix_model.biomass_reaction, 0.0)

        # ── Step 4: compare ────────────────────────────────────────────
        growth_rel_err = (abs(helix_growth - cobra_growth) / cobra_growth
                          if cobra_growth > 0 else float("inf"))
        passed = growth_rel_err <= TOLERANCE

        return {
            "id": "04_iml1515_fba",
            "status": "PASS" if passed else "FAIL",
            "reference": {
                "source": "BiGG iML1515 via COBRApy",
                "growth_rate": cobra_growth,
                "expected_approx": EXPECTED_GROWTH,
                "n_reactions": len(cobra_model.reactions),
                "n_metabolites": len(cobra_model.metabolites),
                "glc_uptake": 10.0,
            },
            "helixlang": {
                "growth_rate": helix_growth,
                "model_reactions": len(helix_model.reactions),
            },
            "comparison": {
                "growth_rate_rel_error": growth_rel_err,
                "growth_rate_tolerance": TOLERANCE,
            },
            "runtime_seconds": time.perf_counter() - t0,
        }

    except Exception as exc:
        return {
            "id": "04_iml1515_fba",
            "status": "FAIL",
            "error": str(exc),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
