#!/usr/bin/env python3
"""Benchmark 77: Innate immune fidelity (doc/40 Phase A — G1/G4/G8/G9).

Validates the literature-grounded upgrades to the innate immune model:
  - G1 type-I IFN antiviral loop (IFNPool): rises with pathogen, suppresses
    effective pathogen signal via saturating Hill kinetics (L5 Pawelek 2012).
  - G8/G9 CRP v2 (CRPDriver): saturating Hill IL-6→CRP with ~6 h lag
    compartment and widened sepsis ceiling (~1000 mg/L) (L9 Sproston & Ashworth
    2018: severe sepsis CRP >= 400 mg/L; peak lags IL-6).
  - G4 Friberg granulopoiesis (ImmuneCellPopulation): 4-compartment transit
    chain; chemotherapy-style drug kill on proliferating precursors drives a
    myelosuppressive ANC reduction (L4 Friberg 2002).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys_path = str(Path(__file__).resolve().parents[3])
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)


def run() -> dict:
    t0 = time.perf_counter()
    try:
        from helixlang.plugins.human.immune import (
            CRPDriver,
            IFNPool,
            ImmuneCellPopulation,
        )

        checks: dict[str, bool] = {}
        details: dict[str, float] = {}

        # ── G1: type-I IFN antiviral loop ──────────────────────────────────
        ifn = IFNPool()
        for _ in range(24):
            ifn.step(1.0, 0.9)
        details["ifn_24h_high_signal"] = ifn.ifn_alpha_beta
        checks["ifn_rises_with_pathogen"] = ifn.ifn_alpha_beta > 0.5
        # Antiviral suppression: effective pathogen below the raw signal.
        eff = ifn.effective_pathogen(0.9)
        details["ifn_effective_pathogen"] = eff
        checks["ifn_suppresses_effective_pathogen"] = eff < 0.9 * 0.9

        # ── G8/G9: CRP v2 saturating production + sepsis ceiling + lag ─────
        crp = CRPDriver()
        # Strong, sustained IL-6 must push CRP into the severe-sepsis band.
        for _ in range(72):
            crp.step(1.0, il6_pg_ml=80.0)
        details["crp_72h_high_il6"] = crp.crp_mg_l
        checks["crp_reaches_sepsis_level_under_high_il6"] = crp.crp_mg_l >= 300.0
        # APR panel responds (procalcitonin rises with IL-6).
        details["pct_72h"] = crp.pct_ng_ml
        checks["crp_apr_panel_responds"] = crp.pct_ng_ml > 1.0

        # Lag: piecewise IL-6 pulse yields a delayed, still-rising CRP — the
        # lag compartment means CRP does not saturate instantly within the
        # first hours of a strong stimulus.
        crp2 = CRPDriver()
        for _ in range(12):
            crp2.step(1.0, il6_pg_ml=250.0)
        crp_12h = crp2.crp_mg_l
        details["crp_12h_high_il6"] = crp_12h
        checks["crp_lag_compartment_delays_response"] = crp_12h < 400.0

        # ── G4: Friberg granulopoiesis — drug-kill drives myelosuppression ─
        healthy = ImmuneCellPopulation()
        for _ in range(96):
            healthy.step(1.0, il6=1.0, tnf=1.0)
        healthy_anc = healthy.neutrophils
        details["healthy_anc"] = healthy_anc

        myelosuppressed = ImmuneCellPopulation()
        myelosuppressed.friberg_drug_kill = 0.85  # chemo-like kill
        nadir = myelosuppressed.neutrophils
        for _ in range(24 * 10):
            myelosuppressed.step(1.0, il6=1.0, tnf=1.0)
            nadir = min(nadir, myelosuppressed.neutrophils)
        details["chemo_anc_nadir"] = nadir
        checks["friberg_drug_kill_reduces_neutrophils"] = nadir < healthy_anc * 0.6

        all_pass = all(checks.values())
        return {
            "id": "77_immune_ifn_crp_friberg",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": "doc/40 L5 (Pawelek 2012), L9 (Sproston & Ashworth 2018), L4 (Friberg et al. 2002)",
                "doi": "10.1371/journal.pcbi.1002588; 10.3389/fimmu.2018.00754; 10.1200/JCO.2002.20.23.4713",
                "note": "IFN antiviral loop, IL-6→CRP saturating kinetics with ~6h lag and widened sepsis range, and Friberg transit-chain granulopoiesis",
            },
            "runtime_seconds": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "id": "77_immune_ifn_crp_friberg",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
