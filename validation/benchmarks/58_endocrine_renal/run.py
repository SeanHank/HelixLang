#!/usr/bin/env python3
"""Benchmark 58: Endocrine + renal ODEs."""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "58_endocrine_renal"}
    try:
        from helixlang.human.endocrine import (
            InsulinGlucoseAxis,
            HPAAxis,
            HPTAxis,
            EndocrineSystem,
            create_endocrine,
        )
        from helixlang.human.renal_model import (
            ckd_epi_2021,
            RenalFunctionModel,
        )

        checks: dict[str, bool] = {}
        details: dict[str, object] = {}

        # 1. Import all classes
        checks["import_all_classes"] = True

        # 2. create_endocrine() returns EndocrineSystem
        endocrine = create_endocrine()
        checks["create_endocrine_returns_EndocrineSystem"] = isinstance(
            endocrine, EndocrineSystem
        )

        # 3. InsulinGlucoseAxis: step once, glucose positive
        ig = InsulinGlucoseAxis()
        ig.step(dt_h=1.0)
        checks["insulin_glucose_positive"] = ig.glucose_mg_dl > 0
        details["glucose_after_step"] = round(ig.glucose_mg_dl, 4)

        # 4. HPAAxis: step 24 times (24h), cortisol positive
        hpa = HPAAxis()
        for _ in range(24):
            hpa.step(dt_h=1.0)
        checks["hpa_cortisol_positive"] = hpa.cortisol_ug_dl > 0
        details["cortisol_after_24h"] = round(hpa.cortisol_ug_dl, 4)

        # 5. HPTAxis: step once, TSH non-negative
        hpt = HPTAxis()
        hpt.step(dt_h=1.0)
        checks["hpt_tsh_non_negative"] = hpt.tsh_miul >= 0
        details["tsh_after_step"] = round(hpt.tsh_miul, 4)

        # 6. ckd_epi_2021: 40yo male, creatinine 1.0 → eGFR ~90-120
        egfr_male = ckd_epi_2021(1.0, 40.0, is_female=False)
        checks["ckd_epi_male_in_range"] = 90.0 <= egfr_male <= 120.0
        details["egfr_40yo_male_cr1"] = round(egfr_male, 2)

        # 7. ckd_epi_2021: 70yo female, creatinine 2.0 → eGFR 20-60
        egfr_female = ckd_epi_2021(2.0, 70.0, is_female=True)
        checks["ckd_epi_female_in_range"] = 20.0 <= egfr_female <= 60.0
        details["egfr_70yo_female_cr2"] = round(egfr_female, 2)

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        results.update({
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": (
                "Inker LA et al. 2021, N Engl J Med 385:2031-2043 (CKD-EPI 2021); "
                "Bergman RN et al. 1979, Am J Physiol 236:E667 (minimal model)"
            ),
            "runtime_seconds": elapsed,
        })
    except Exception as e:
        results.update({
            "status": "FAIL",
            "checks": {},
            "details": {"error": str(e)},
            "reference": (
                "Inker LA et al. 2021, N Engl J Med 385:2031-2043 (CKD-EPI 2021); "
                "Bergman RN et al. 1979, Am J Physiol 236:E667 (minimal model)"
            ),
            "runtime_seconds": time.perf_counter() - t0,
        })
    return results


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
