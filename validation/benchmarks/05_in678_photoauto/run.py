#!/usr/bin/env python3
"""Benchmark 05: Synechocystis PCC 6803 iJN678 photoautotrophic FBA.

Downloads iJN678 from BiGG via COBRApy, sets photoautotrophic conditions
(block glucose, enable photon + CO2 uptake), converts to HelixLang, runs FBA,
and compares growth rate and fluxes against the COBRApy solution.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

REF_DIR = Path(__file__).resolve().parents[2] / "references"
MODEL_DIR = REF_DIR / "models"


def _pearson_r(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=True))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx < 1e-15 or sy < 1e-15:
        return 0.0
    return cov / (sx * sy)


def run() -> dict:
    t0 = time.perf_counter()
    try:
        os.environ["TQDM_DISABLE"] = "1"
        import tqdm as _tqdm_mod
        _orig_tqdm = _tqdm_mod.tqdm
        _tqdm_mod.tqdm = lambda *a, **kw: _orig_tqdm(*a, **{**kw, "disable": True})

        from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis, _from_cobra_model
        from helixlang.plugins.gem.sbml_import import load_bigg_cobra_model

        # Load iJN678 (vendored copy first; doc/41 — SKIP if unavailable)
        try:
            cobra_model = load_bigg_cobra_model(
                "iJN678", model_dir=MODEL_DIR,
            )
        except Exception as exc:
            return {
                "id": "05_in678_photoauto",
                "status": "SKIP",
                "reference": {
                    "source": "BiGG iJN678 via COBRApy (photoautotrophic)",
                },
                "reason": f"BiGG iJN678 unavailable: {exc}",
                "runtime_seconds": time.perf_counter() - t0,
            }

        n_rxn = len(cobra_model.reactions)
        n_met = len(cobra_model.metabolites)

        # Set photoautotrophic conditions:
        # - Block glucose uptake (EX_glc__D_e: lb=0)
        # - Enable photon uptake (EX_photon_e: lb=-1000, ub=1000)
        # - Enable CO2 uptake (EX_co2_e: lb=-1000)
        cobra_model.reactions.get_by_id("EX_glc__D_e").lower_bound = 0.0
        cobra_model.reactions.get_by_id("EX_photon_e").lower_bound = -1000.0
        cobra_model.reactions.get_by_id("EX_photon_e").upper_bound = 1000.0
        cobra_model.reactions.get_by_id("EX_co2_e").lower_bound = -1000.0

        # COBRApy reference solution
        cobra_model.solver = "glpk"
        cobra_sol = cobra_model.optimize()
        cobra_growth = float(cobra_sol.objective_value)
        cobra_fluxes = {r.id: float(cobra_sol.fluxes[r.id])
                        for r in cobra_model.reactions}

        # Save reference
        REF_DIR.mkdir(parents=True, exist_ok=True)
        ref_path = REF_DIR / "ijn678_photoauto_fluxes.json"
        with open(ref_path, "w") as fh:
            json.dump(cobra_fluxes, fh, indent=2)

        # Convert to HelixLang and run FBA
        helix_model = _from_cobra_model(cobra_model)
        fba = FluxBalanceAnalysis(helix_model)

        # Set same photoautotrophic constraints in HelixLang
        fba.set_uptake("glc_e", 0.0)
        fba.set_uptake("photon_e", 1000.0)
        fba.set_uptake("co2_e", 1000.0)

        helix_fluxes = fba.solve(objective="biomass")
        helix_growth = helix_fluxes.get(helix_model.biomass_reaction, 0.0)

        # Compare
        growth_rel_err = (abs(helix_growth - cobra_growth) / cobra_growth
                          if cobra_growth > 0 else float("inf"))
        growth_pass = growth_rel_err <= 0.05

        # Pearson r for active fluxes (|flux| > 1.0 in either solution)
        common_rids = sorted(
            set(cobra_fluxes.keys()) & set(helix_fluxes.keys()),
        )
        active_rids = [r for r in common_rids
                       if abs(cobra_fluxes[r]) > 1.0
                       or abs(helix_fluxes.get(r, 0.0)) > 1.0]
        cobrapy_active = [cobra_fluxes[r] for r in active_rids]
        helix_active = [helix_fluxes.get(r, 0.0) for r in active_rids]
        pearson_r = _pearson_r(cobrapy_active, helix_active)

        return {
            "id": "05_in678_photoauto",
            "status": "PASS" if growth_pass else "FAIL",
            "reference": {
                "source": "BiGG iJN678 via COBRApy (photoautotrophic)",
                "growth_rate": cobra_growth,
                "n_reactions": n_rxn,
                "n_metabolites": n_met,
                "glucose_uptake": 0.0,
                "photon_uptake": 1000.0,
                "co2_uptake": 1000.0,
            },
            "helixlang": {
                "growth_rate": helix_growth,
                "model_reactions": len(helix_model.reactions),
            },
            "comparison": {
                "growth_rate_rel_error": growth_rel_err,
                "growth_rate_tolerance": 0.05,
                "pearson_r_active_fluxes": pearson_r,
                "n_active_fluxes": len(active_rids),
                "note": "Growth rate is primary metric; Pearson r for active fluxes reported (FBA degenerate optima expected)",
            },
            "runtime_seconds": time.perf_counter() - t0,
        }

    except Exception as exc:
        return {
            "id": "05_in678_photoauto",
            "status": "FAIL",
            "error": str(exc),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
