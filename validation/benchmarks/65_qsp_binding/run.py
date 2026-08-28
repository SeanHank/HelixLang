#!/usr/bin/env python3
"""Benchmark 65: QSP binding models.

Validates mass-action, TMDD, and competitive binding models
for mechanistic pharmacodynamic binding.

Reference:
  Mager DE, Jusko WJ 2001, J Pharmacokinet Pharmacodyn 28:507 (TMDD)
"""
from __future__ import annotations

import json
import math
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.plugins.human.qsp_binding import (
            CompetitiveBinding,
            MassActionBinding,
            QSPBindingModel,
            QSPBindingSystem,
            TMDDBinding,
            create_qsp_binding,
        )
        checks["import_all_classes"] = True

        # --- Check 2: MassActionBinding occupancy at [D]=100 nM, Kd=10 nM ---
        # occupancy = D / (Kd + D) = 100 / (10 + 100) ≈ 0.909
        mab = MassActionBinding(kd_nM=10.0, target_receptor_total=100.0)
        occ = mab.compute_occupancy(drug_conc_nM=100.0)
        expected_occ = 100.0 / (10.0 + 100.0)
        checks["mass_action_occupancy_at_100nM"] = abs(occ - expected_occ) < 0.01
        details["mass_action_occupancy"] = round(occ, 6)
        details["expected_occupancy"] = round(expected_occ, 6)

        # --- Check 3: MassActionBinding occupancy at [D]=0 ---
        occ_zero = mab.compute_occupancy(drug_conc_nM=0.0)
        checks["mass_action_occupancy_at_0nM"] = abs(occ_zero) < 1e-10
        details["mass_action_occupancy_zero"] = occ_zero

        # --- Check 4: TMDDBinding step ---
        tmdd = TMDDBinding(kss_nM=5.0, kint=0.02, ksyn=0.5, kdeg=0.1)
        r_before = tmdd.r_total_nM
        # Add drug input for one step
        tmdd.step(dt_h=1.0, drug_input_rate_nM_h=10.0)
        checks["tmdd_step_works"] = tmdd.c_free_nM >= 0.0 and tmdd.r_total_nM >= 0.0
        details["tmdd_c_free_after_1h"] = round(tmdd.c_free_nM, 6)
        details["tmdd_r_total_before"] = r_before
        details["tmdd_r_total_after"] = round(tmdd.r_total_nM, 6)

        # --- Check 5: CompetitiveBinding instantiation ---
        cb = CompetitiveBinding(kd_agonist_nM=10.0, ki_antagonist_nM=5.0, emax=1.0)
        # Verify effect decreases with antagonist
        eff_no_antag = cb.compute_effect(agonist_conc_nM=10.0, antagonist_conc_nM=0.0)
        eff_with_antag = cb.compute_effect(agonist_conc_nM=10.0, antagonist_conc_nM=10.0)
        checks["competitive_binding_instantiates"] = True
        checks["competitive_antagonist_reduces_effect"] = eff_with_antag < eff_no_antag
        details["competitive_effect_no_antag"] = round(eff_no_antag, 6)
        details["competitive_effect_with_antag"] = round(eff_with_antag, 6)

        # --- Check 6: create_qsp_binding returns QSPBindingSystem ---
        qsp = create_qsp_binding()
        assert isinstance(qsp, QSPBindingSystem), (
            f"create_qsp_binding should return QSPBindingSystem, got {type(qsp).__name__}"
        )
        checks["create_qsp_binding_returns_system"] = True
        details["n_models"] = len(qsp.models)
        details["model_names"] = list(qsp.models.keys())

        # --- Check 7: QSPBindingSystem has step() method ---
        checks["qsp_system_has_step"] = callable(getattr(qsp, "step", None))

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "65_qsp_binding",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "tmdd": "Mager DE, Jusko WJ 2001, J Pharmacokinet Pharmacodyn 28:507",
                "qss_tmdd": "Gibiansky L, Gibiansky E 2014, J Pharmacokinet Pharmacodyn 41:275",
                "schild": "Schild HO 1949, Br J Pharmacol 4:277",
            },
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "65_qsp_binding",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
