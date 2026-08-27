#!/usr/bin/env python3
"""Benchmark 48: Innate immune dynamics.

Validates the innate immune ODE model:
  - Cytokine kinetics (IL-6 production/clearance under infection)
  - Neutrophil mobilisation via G-CSF/IL-6 axis
  - CRP response driven by IL-6
  - Cortisol-mediated immunosuppression

Reference: Chrousos 1995 (HPA axis);.systems biology of innate immunity.
"""
from __future__ import annotations

import json
import math
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    try:
        from helixlang.human.immune import (
            CytokinePool,
            CRPDriver,
            InnateImmuneModel,
            ImmuneCellPopulation,
            create_immune_model,
        )

        checks: dict[str, bool] = {}
        details: dict[str, float | dict] = {}

        # ── Test 1: Cytokine production under infection ──────────────────────
        pool = CytokinePool()
        il6_0 = pool.il6
        # Apply pathogen signal
        pool.pathogen_signal = 1.0
        for _ in range(24):  # 24 hours at dt=1h
            pool.step(1.0)
        il6_24 = pool.il6
        checks["il6_increases_under_infection"] = il6_24 > il6_0
        details["il6_baseline"] = il6_0
        details["il6_24h_infection"] = il6_24

        # ── Test 2: TNF-alpha also increases ─────────────────────────────────
        pool2 = CytokinePool()
        tnf_0 = pool2.tnf_alpha
        pool2.pathogen_signal = 1.0
        for _ in range(24):
            pool2.step(1.0)
        checks["tnf_increases_under_infection"] = pool2.tnf_alpha > tnf_0
        details["tnf_24h"] = pool2.tnf_alpha

        # ── Test 3: IL-10 (anti-inflammatory) increases with pro-inflammatory ─
        il10_before = pool2.il10
        pool2.pathogen_signal = 1.0
        for _ in range(24):
            pool2.step(1.0)
        checks["il10_induced_by_proinflammatory"] = pool2.il10 > il10_before
        details["il10_24h"] = pool2.il10

        # ── Test 4: Neutrophil mobilisation via IL-6 ─────────────────────────
        pop = ImmuneCellPopulation()
        neut_0 = pop.neutrophils
        for _ in range(48):
            pop.step(1.0, il6=il6_24, tnf=pool2.tnf_alpha)
        checks["neutrophils_increased"] = pop.neutrophils > neut_0
        details["neutrophils_0"] = neut_0
        details["neutrophils_48h"] = pop.neutrophils

        # ── Test 5: WBC total is positive ────────────────────────────────────
        wbc = pop.get_wbc_total()
        checks["wbc_total_positive"] = wbc > 0
        details["wbc_total"] = wbc

        # ── Test 6: CRP response to IL-6 ─────────────────────────────────────
        crp = CRPDriver()
        crp_0 = crp.crp_mg_l
        for _ in range(72):  # 72 hours
            crp.step(1.0, il6_pg_ml=il6_24)
        checks["crp_increases_with_il6"] = crp.crp_mg_l > crp_0
        details["crp_0"] = crp_0
        details["crp_72h"] = crp.crp_mg_l
        # CRP should be physiologically bounded
        checks["crp_bounded"] = 0.0 <= crp.crp_mg_l <= 200.0

        # ── Test 7: Cortisol suppression ─────────────────────────────────────
        model_suppressed, crp_suppressed = create_immune_model(
            infection_severity=0.5,
            cortisol_level=30.0,  # High cortisol -> suppression
        )
        model_normal, crp_normal = create_immune_model(
            infection_severity=0.5,
            cortisol_level=12.0,  # Normal cortisol
        )
        # Run both for 48h
        for _ in range(48):
            model_suppressed.step(1.0)
            model_normal.step(1.0)
        # High cortisol should suppress IL-6 (anti-inflammatory)
        il6_suppressed = model_suppressed.get_il6()
        il6_normal = model_normal.get_il6()
        checks["cortisol_suppresses_il6"] = il6_suppressed <= il6_normal
        details["il6_suppressed_48h"] = il6_suppressed
        details["il6_normal_48h"] = il6_normal

        # ── Test 8: No infection -> baseline stays low ────────────────────────
        model_quiet, _ = create_immune_model(infection_severity=0.0)
        for _ in range(48):
            model_quiet.step(1.0)
        checks["no_infection_low_il6"] = model_quiet.get_il6() < 1.0
        details["il6_no_infection_48h"] = model_quiet.get_il6()

        all_pass = all(checks.values())

        return {
            "id": "48_immune_dynamics",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": "Chrousos GP 1995, N Engl J Med 332:1351-1362",
                "doi": "10.1056/NEJM199505183322007",
                "note": "Innate immune cytokine/cell dynamics validated against known qualitative behavior",
            },
            "runtime_seconds": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "id": "48_immune_dynamics",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
