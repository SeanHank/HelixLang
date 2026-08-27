#!/usr/bin/env python3
"""Benchmark 57: PBPK pharmacokinetics."""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "57_pbpk_pharmacokinetics"}
    try:
        from helixlang.human.pharmacokinetics import (
            PBPKConfig,
            PBPKModel,
            PBPKResult,
            _trapezoid,
            _terminal_half_life,
        )
        from helixlang.human.drug import Drug, DrugMolecule, IV
        from helixlang.human.physiology import create_default_physiology

        checks: dict[str, bool] = {}
        details: dict[str, object] = {}

        # 1. Import PBPKModel, PBPKConfig, PBPKResult
        checks["import_pbpk_classes"] = True

        # 2. Create PBPKConfig with default parameters
        config = PBPKConfig()
        config.validate()
        checks["pbpk_config_default"] = config.dt_min == 1.0
        details["config_dt_min"] = config.dt_min
        details["config_total_time_h"] = config.total_time_h

        # 3. Create PBPKModel with config — verify it has step()
        molecule = DrugMolecule(
            name="test_drug", molecular_weight_da=300.0, smiles="", formula="C15H20O5"
        )
        drug = Drug(
            molecule=molecule,
            dose_mg=100.0,
            dosing_interval_h=24.0,
            route=IV,
            duration_days=1.0,
            bioavailability=1.0,
            volume_distribution_l=50.0,
            clearance_ml_per_min=100.0,
            half_life_h=6.0,
            renal_fraction=0.5,
        )
        physiology = create_default_physiology()
        cfg = PBPKConfig(dt_min=5.0, total_time_h=1.0)
        model = PBPKModel(drug=drug, physiology=physiology, config=cfg)
        checks["pbpk_model_has_step"] = hasattr(model, "step") and callable(model.step)
        checks["pbpk_model_has_run"] = hasattr(model, "run") and callable(model.run)

        # 4. Test _trapezoid: y=[0,1,1,0] x=[0,1,2,3] → AUC = 2.0
        auc_trap = _trapezoid([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 1.0, 0.0])
        checks["trapezoid_auc"] = abs(auc_trap - 2.0) < 1e-9
        details["trapezoid_auc_computed"] = auc_trap
        details["trapezoid_auc_expected"] = 2.0

        # 5. Test _terminal_half_life: exponential decay → positive half-life
        import math

        times_exp = [float(i) for i in range(20)]
        conc_exp = [10.0 * math.exp(-0.1 * t) for t in times_exp]
        hl = _terminal_half_life(times_exp, conc_exp, fallback_h=6.0)
        checks["terminal_half_life_positive"] = hl > 0
        details["terminal_half_life_h"] = hl

        # 6. Run a short simulation (10 time steps), verify non-negative concentrations
        cfg_short = PBPKConfig(dt_min=1.0, total_time_h=10.0 / 60.0 * 60.0)
        model_short = PBPKModel(
            drug=drug, physiology=create_default_physiology(), config=cfg_short
        )
        for _ in range(10):
            model_short.step(dt_min=1.0)
        concs = model_short.get_concentrations()
        non_negative = all(v >= 0.0 for v in concs.values())
        checks["simulation_non_negative"] = non_negative
        details["final_concentrations"] = {k: round(v, 6) for k, v in concs.items()}

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        results.update({
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": "Jones HM, Rowland-Yeo K 2013, CPT Pharmacometrics Syst Pharmacol 2:e737 (PBPK)",
            "runtime_seconds": elapsed,
        })
    except Exception as e:
        results.update({
            "status": "FAIL",
            "checks": {},
            "details": {"error": str(e)},
            "reference": "Jones HM, Rowland-Yeo K 2013, CPT Pharmacometrics Syst Pharmacol 2:e737 (PBPK)",
            "runtime_seconds": time.perf_counter() - t0,
        })
    return results


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
