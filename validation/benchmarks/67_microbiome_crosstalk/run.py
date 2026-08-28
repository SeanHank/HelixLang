#!/usr/bin/env python3
"""Benchmark 67: Microbiome + organ crosstalk.

Validates gut microbiome compartment modeling and organ-organ
crosstalk coupling signals.

Reference:
  Spanogiannopoulos P et al. 2016, Nat Rev Pharmacol 16:135 (microbiome drug metabolism)
  Qi Z et al. 2017, PLoS Comput Biol 13:e1005416 (organ crosstalk)
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.plugins.human.microbiome import (
            MicrobiomeCompartment,
            MicrobialSpecies,
            MicrobiomeState,
        )
        from helixlang.plugins.human.organ_crosstalk import (
            OrganCrosstalk,
            apply_crosstalk,
            create_crosstalk,
        )
        checks["import_all_classes"] = True

        # --- Check 2: MicrobiomeCompartment instantiation ---
        comp = MicrobiomeCompartment(healthy_composition=True)
        assert hasattr(comp, "state"), "MicrobiomeCompartment should have a state attribute"
        assert isinstance(comp.state, MicrobiomeState)
        checks["microbiome_compartment_instantiates"] = True
        details["n_species"] = len(comp._species)

        # --- Check 3: MicrobialSpecies fields accessible ---
        sp = MicrobialSpecies(
            name="TestSpecies",
            abundance=0.15,
            reactions=["test_reaction"],
            growth_rate_h=0.1,
        )
        assert sp.name == "TestSpecies"
        assert sp.abundance == 0.15
        assert sp.growth_rate_h == 0.1
        checks["microbial_species_fields_accessible"] = True

        # --- Check 4: MicrobiomeState has step method (via compartment) ---
        state = MicrobiomeState()
        assert state.total_biomass == 1.0
        assert state.diversity_index == 1.0
        # MicrobiomeCompartment.step() advances microbiome state
        comp.step(dt_h=1.0)
        checks["microbiome_compartment_step_works"] = True
        details["scfa_after_1h"] = round(comp.state.scfa_total_mM, 4)

        # --- Check 5: create_crosstalk returns OrganCrosstalk ---
        ct = create_crosstalk()
        assert isinstance(ct, OrganCrosstalk), (
            f"create_crosstalk should return OrganCrosstalk, got {type(ct).__name__}"
        )
        checks["create_crosstalk_returns_organ_crosstalk"] = True

        # --- Check 6: OrganCrosstalk has apply_crosstalk function ---
        assert callable(apply_crosstalk), "apply_crosstalk should be callable"
        checks["organ_crosstalk_has_apply"] = True

        # --- Check 7: apply_crosstalk modifies params ---
        ct_result = apply_crosstalk(
            ct,
            glucose_mg_dl=250.0,   # hyperglycemic
            egfr=30.0,             # low kidney function
            cortisol_ug_dl=30.0,   # high cortisol
            albumin_g_dl=2.5,      # low albumin (hepatic dysfunction)
            inr=2.0,               # elevated INR
            il6_pg_ml=20.0,        # high inflammation
            tnf_pg_ml=50.0,
            phosphate_mg_dl=6.0,   # hyperphosphatemia
        )
        assert isinstance(ct_result, OrganCrosstalk)
        # Hyperglycemia should increase CV risk
        checks["apply_crosstalk_cv_risk_elevated"] = ct_result.cv_risk_multiplier > 1.0
        # Low eGFR should elevate anemia risk
        checks["apply_crosstalk_anemia_risk_elevated"] = ct_result.anemia_risk_multiplier > 1.0
        # High cortisol should suppress immune
        checks["apply_crosstalk_immune_suppressed"] = ct_result.immune_suppression_from_cortisol > 0.0
        # Liver dysfunction should reduce clearance
        checks["apply_crosstalk_clearance_reduced"] = ct_result.clearance_modifier_from_liver < 1.0
        details["cv_risk_multiplier"] = round(ct_result.cv_risk_multiplier, 4)
        details["anemia_risk_multiplier"] = round(ct_result.anemia_risk_multiplier, 4)
        details["immune_suppression"] = round(ct_result.immune_suppression_from_cortisol, 4)
        details["clearance_modifier"] = round(ct_result.clearance_modifier_from_liver, 4)
        details["child_pugh_score"] = round(ct_result.child_pugh_score, 4)

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "67_microbiome_crosstalk",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "microbiome": "Spanogiannopoulos P et al. 2016, Nat Rev Pharmacol 16:135",
                "organ_crosstalk": "Qi Z et al. 2017, PLoS Comput Biol 13:e1005416",
            },
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "67_microbiome_crosstalk",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
