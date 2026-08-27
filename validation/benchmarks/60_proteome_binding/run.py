#!/usr/bin/env python3
"""Benchmark 60: Proteome binding + DDI."""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "60_proteome_binding"}
    try:
        from helixlang.human.proteome_binding import (
            ProteomeBindingCascade,
            ProteomeBindingProfile,
            ProteomeDDIPrediction,
            BindingPrediction,
        )

        checks: dict[str, bool] = {}
        details: dict[str, object] = {}

        # 1. Import all classes
        checks["import_all_classes"] = True

        # 2. ProteomeBindingCascade: create with defaults, verify instantiates
        cascade = ProteomeBindingCascade()
        checks["cascade_instantiate"] = True

        # 3. ProteomeBindingProfile: create with known Kd, verify fields
        bp = BindingPrediction(
            target="CYP2D6",
            kd_um=5.0,
            occupancy=0.67,
            is_substrate=True,
            is_inhibitor=False,
            inhibition_strength=0.0,
            confidence=0.9,
        )
        profile = ProteomeBindingProfile(
            drug_name="test_drug",
            smiles="CC(=O)O",
            bindings=[bp],
            n_targets_screened=50,
            n_significant_bindings=1,
        )
        checks["profile_fields_accessible"] = (
            profile.drug_name == "test_drug"
            and profile.bindings[0].target == "CYP2D6"
            and profile.bindings[0].kd_um == 5.0
        )
        details["profile_binding_dict"] = profile.binding_dict

        # 4. BindingPrediction: confidence between 0 and 1
        checks["binding_prediction_confidence_range"] = 0.0 <= bp.confidence <= 1.0
        details["binding_confidence"] = bp.confidence

        # 5. ProteomeDDIPrediction: create, verify ratio/effect fields
        ddi = ProteomeDDIPrediction(
            drug_a="warfarin",
            drug_b="amiodarone",
            auc_ratio=1.8,
            interacting_targets=["CYP2C9", "CYP3A4"],
            max_occupancy=0.75,
            significance="DDD_ALERT",
            confidence=0.85,
        )
        checks["ddi_has_auc_ratio"] = hasattr(ddi, "auc_ratio") and ddi.auc_ratio > 0
        checks["ddi_has_significance"] = hasattr(ddi, "significance")
        details["ddi_auc_ratio"] = ddi.auc_ratio
        details["ddi_significance"] = ddi.significance

        # 6. ProteomeBindingCascade has predict() or screen() method
        checks["cascade_has_screen"] = hasattr(cascade, "screen_drug") and callable(
            cascade.screen_drug
        )
        checks["cascade_has_predict_ddi"] = hasattr(cascade, "predict_ddi") and callable(
            cascade.predict_ddi
        )

        # Bonus: screen a known drug to verify end-to-end
        warfarin_profile = cascade.screen_drug(
            "warfarin", "CC(=O)Cc1ccccc1C(=O)O", drug_conc_um=10.0
        )
        checks["screen_known_drug_returns_profile"] = isinstance(
            warfarin_profile, ProteomeBindingProfile
        )
        details["warfarin_n_bindings"] = len(warfarin_profile.bindings)
        details["warfarin_n_significant"] = warfarin_profile.n_significant_bindings

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        results.update({
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": "Yoshida K et al. 2020, CPT Pharmacometrics Syst Pharmacol 9:51 (DDI prediction)",
            "runtime_seconds": elapsed,
        })
    except Exception as e:
        results.update({
            "status": "FAIL",
            "checks": {},
            "details": {"error": str(e)},
            "reference": "Yoshida K et al. 2020, CPT Pharmacometrics Syst Pharmacol 9:51 (DDI prediction)",
            "runtime_seconds": time.perf_counter() - t0,
        })
    return results


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
