#!/usr/bin/env python3
"""Benchmark 59: Hematology myelosuppression."""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "59_hematology"}
    try:
        from helixlang.plugins.human.hematology_model import (
            FribergLineage,
            ErythropoiesisModel,
            HematologySystem,
            MyelosuppressionParams,
            LineageConfig,
            create_hematology_system,
        )

        checks: dict[str, bool] = {}
        details: dict[str, object] = {}

        # 1. Import all classes
        checks["import_all_classes"] = True

        # 2. create_hematology_system() returns HematologySystem
        heme = create_hematology_system()
        checks["create_returns_HematologySystem"] = isinstance(heme, HematologySystem)

        # 3. HematologySystem has step() method
        checks["hematology_has_step"] = hasattr(heme, "step") and callable(heme.step)

        # 4. Step once with no drug exposure, neutrophil count within 20% of baseline
        baseline_anc = heme.neutrophils.count()
        snap = heme.step(dt_h=24.0, exposures={})
        anc_after = snap["anc_x10e3_ul"]
        pct_change = abs(anc_after - baseline_anc) / baseline_anc if baseline_anc > 0 else 0.0
        checks["no_drug_anc_stable"] = pct_change <= 0.20
        details["baseline_anc"] = round(baseline_anc, 4)
        details["anc_after_24h_no_drug"] = round(anc_after, 4)
        details["anc_pct_change"] = round(pct_change, 4)

        # 5. MyelosuppressionParams with Emax=0.9 instantiates
        drug_params = MyelosuppressionParams(
            drug_name="test_cytotox", emax=0.9, ec50_mg_l=0.1, hill=1.0
        )
        checks["myelosuppression_params_instantiate"] = True
        details["drug_effect_at_ec50"] = round(drug_params.effect_at(0.1), 4)

        # 6. Step 7 times (7 days) with high drug exposure, neutrophils should decrease
        heme2 = create_hematology_system()
        anc_start = heme2.neutrophils.count()
        heme2.register_myelosuppressant(
            MyelosuppressionParams(drug_name="cytotox", emax=0.9, ec50_mg_l=0.05)
        )
        for day in range(7):
            heme2.step(dt_h=24.0, exposures={"cytotox": 0.5})
        anc_end = heme2.neutrophils.count()
        checks["drug_exposure_anc_decreased"] = anc_end < anc_start
        details["anc_start_drug"] = round(anc_start, 4)
        details["anc_end_after_7d_drug"] = round(anc_end, 4)

        # 7. ErythropoiesisModel: step works, RBC (Hb) stays positive
        rbc = ErythropoiesisModel()
        hb_after = rbc.step(dt_h=24.0, chemo_inhibition=0.0)
        checks["erythropoiesis_step_positive"] = hb_after > 0
        details["hemoglobin_after_24h"] = round(hb_after, 4)

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        results.update({
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": "Friberg semimechanistic myelosuppression model",
                "authors": "Friberg LE et al.",
                "year": 2002,
                "journal": "J Pharmacokinet Pharmacodyn",
                "doi": "10.1023/A:1020492211773",
                "note": "Friberg LE et al. 2002, J Pharmacokinet Pharmacodyn 29:411-428. Semi-mechanistic model of drug-induced myelosuppression: drug exposure decreases proliferating cells with time-delayed neutrophil nadir.",
            },
            "experimental_comparison": {
                "anc_stable_no_drug_pct_change": {
                    "reference_min": 0.0,
                    "reference_max": 0.20,
                    "tolerance": 0.20,
                    "unit": "fraction of baseline",
                    "note": "No drug exposure → ANC stays within 20% of baseline (homeostatic steady state).",
                },
                "anc_decreases_under_cytotoxic_drug": {
                    "reference_min": 1.0,
                    "reference_max": 1.0,
                    "tolerance": 0.0,
                    "unit": "boolean",
                    "note": "Repeated cytotoxic exposure produces neutrophil decline (drug-induced neutropenia).",
                },
            },
            "runtime_seconds": elapsed,
        })
    except Exception as e:
        results.update({
            "status": "FAIL",
            "checks": {},
            "details": {"error": str(e)},
            "reference": {
                "source": "Friberg semimechanistic myelosuppression model",
                "authors": "Friberg LE et al.",
                "year": 2002,
                "doi": "10.1023/A:1020492211773",
                "note": "Friberg LE et al. 2002, J Pharmacokinet Pharmacodyn 29:411-428.",
            },
            "runtime_seconds": time.perf_counter() - t0,
        })
    return results


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
