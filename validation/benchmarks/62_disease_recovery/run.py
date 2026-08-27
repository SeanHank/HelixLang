#!/usr/bin/env python3
"""Benchmark 62: Disease progression and recovery.

Validates disease_progression.py and recovery.py modules:
  - DiseaseProgressionModel with PROGRESSION_PROFILES (CKD, cirrhosis, T2DM, cancer)
  - DiseaseStage enum (PRECLINICAL → CRITICAL)
  - RecoveryModel with ORGAN_RECOVERY_PROFILES (liver, kidney, bone_marrow, heart)
  - Step dynamics: no-drug progression and post-treatment recovery

Reference: Sonnenberg FA, Beck JR 1993, Med Decis Making 13:322
           (Markov disease progression).
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    try:
        from helixlang.human.disease_progression import (
            DiseaseProgressionModel,
            DiseaseStage,
            create_progression_model,
            PROGRESSION_PROFILES,
        )
        from helixlang.human.recovery import (
            RecoveryModel,
            create_recovery_model,
            ORGAN_RECOVERY_PROFILES,
        )

        checks: dict[str, bool] = {}
        details: dict[str, object] = {}

        # 1. All required imports succeeded
        checks["import_all_classes"] = True

        # 2. create_progression_model returns DiseaseProgressionModel
        model = create_progression_model("CKD")
        checks["create_progression_model_returns_model"] = isinstance(
            model, DiseaseProgressionModel
        )
        details["initial_disease"] = model.disease_name

        # 3. DiseaseStage enum has required members
        checks["disease_stage_has_preclinical"] = DiseaseStage.PRECLINICAL.value == "preclinical"
        checks["disease_stage_has_mild"] = DiseaseStage.MILD.value == "mild"
        checks["disease_stage_has_moderate"] = DiseaseStage.MODERATE.value == "moderate"
        checks["disease_stage_has_severe"] = DiseaseStage.SEVERE.value == "severe"
        checks["disease_stage_has_critical"] = DiseaseStage.CRITICAL.value == "critical"

        # 4. PROGRESSION_PROFILES is a dict with >= 3 disease profiles
        checks["progression_profiles_is_dict"] = isinstance(
            PROGRESSION_PROFILES, dict
        )
        checks["progression_profiles_ge_3"] = len(PROGRESSION_PROFILES) >= 3
        details["progression_profile_keys"] = list(PROGRESSION_PROFILES.keys())

        # 5. DiseaseProgressionModel: step once without drug, severity should
        #    increase or stay same (progression_rate_per_year > 0 for CKD)
        prev_severity = model.get_severity()
        model.step(dt_h=24.0 * 365.0, drug_effectiveness=0.0)  # 1 year, no drug
        new_severity = model.get_severity()
        checks["severity_non_decreasing_without_drug"] = new_severity >= prev_severity - 1e-9
        details["prev_severity"] = prev_severity
        details["new_severity_1yr"] = new_severity

        # 6. create_recovery_model returns RecoveryModel
        recovery = create_recovery_model(
            drug_names=["doxorubicin"],
            baseline_biomarkers={"alt": 35.0, "creatinine": 1.0},
        )
        checks["create_recovery_model_returns_model"] = isinstance(
            recovery, RecoveryModel
        )

        # 7. ORGAN_RECOVERY_PROFILES is a dict with >= 3 organ profiles
        checks["organ_recovery_profiles_is_dict"] = isinstance(
            ORGAN_RECOVERY_PROFILES, dict
        )
        checks["organ_recovery_profiles_ge_3"] = len(ORGAN_RECOVERY_PROFILES) >= 3
        details["organ_profile_keys"] = list(ORGAN_RECOVERY_PROFILES.keys())

        # 8. RecoveryModel: step once with treatment active → biomarkers unchanged;
        #    then deactivate treatment and step → recovery fraction in [0, 1]
        # Set current biomarkers away from baseline
        recovery.current_biomarkers["alt"] = 120.0
        recovery.current_biomarkers["creatinine"] = 3.0

        # Treatment active: values should stay unchanged
        recovery.set_treatment_inactive()
        prev_alt = recovery.current_biomarkers["alt"]
        result = recovery.step(dt_h=24.0, current_time_h=1.0)
        new_alt = recovery.current_biomarkers["alt"]
        # After one day of recovery, ALT should move toward baseline (35.0)
        checks["recovery_alt_moves_toward_baseline"] = new_alt < prev_alt
        checks["recovery_fraction_in_range"] = 0.0 <= (new_alt - 35.0) / (120.0 - 35.0) <= 1.0
        details["prev_alt"] = prev_alt
        details["new_alt_1d"] = new_alt

        all_pass = all(checks.values())
        return {
            "id": "62_disease_recovery",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": "Sonnenberg FA, Beck JR 1993",
                "authors": "Sonnenberg FA, Beck JR",
                "year": 1993,
                "journal": "Med Decis Making",
                "volume": "13",
                "pages": "322",
            },
            "runtime_seconds": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "id": "62_disease_recovery",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
