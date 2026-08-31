#!/usr/bin/env python3
"""Benchmark 30: PBPK IV bolus simulation."""
from __future__ import annotations

import json
import sys
import time

from helixlang.plugins.human.drug import IV, Drug, DrugMolecule
from helixlang.plugins.human.pharmacokinetics import PBPKConfig, PBPKModel
from helixlang.plugins.human.physiology import create_default_physiology


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "30_pk_simulation"}
    try:
        # 1. Create a Drug with known parameters (IV bolus)
        molecule = DrugMolecule(
            name="test_drug_iv",
            molecular_weight_da=300.0,
            smiles="",
            formula="C15H20O5",
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
        assert drug.route == IV, f"Route should be IV, got {drug.route!r}"
        problems = drug.validate()
        assert not problems, f"Drug validation failed: {problems}"

        # 2. Create PBPKModel and run for 24 hours
        physiology = create_default_physiology()
        config = PBPKConfig(dt_min=5.0, total_time_h=24.0)
        model = PBPKModel(drug=drug, physiology=physiology, config=config)
        result = model.run()

        assert len(result.time_h) > 0, "No time points generated"
        assert len(result.central_concentration) > 0, "No concentrations generated"
        assert result.time_h[-1] == 24.0, f"Expected end at 24h, got {result.time_h[-1]}"

        # 3. Cmax occurs at t=0 for IV bolus
        c_max = result.c_max
        t_max = result.t_max
        assert c_max > 0, f"Cmax should be >0, got {c_max}"
        assert t_max == 0.0, f"For IV bolus, t_max should be 0, got {t_max}"

        # Initial concentration equals Dose * F / Vc (central/plasma compartment volume)
        actual_c0 = result.central_concentration[0]
        expected_c0 = drug.dose_mg * drug.bioavailability / config.plasma_volume_l
        assert abs(actual_c0 - expected_c0) / expected_c0 < 0.05, (
            f"C0 mismatch: expected ~{expected_c0:.2f}, got {actual_c0:.2f}"
        )

        # 4. Concentration decreases monotonically after Cmax
        # For IV bolus with no absorption phase, check from index 1 onward
        concs = result.central_concentration
        monotonically_decreasing = True
        for i in range(1, len(concs)):
            if concs[i] > concs[i - 1] + 1e-12:
                monotonically_decreasing = False
                break
        assert monotonically_decreasing, "Concentration should decrease monotonically after Cmax"

        # 5. AUC > 0
        assert result.auc > 0, f"AUC should be >0, got {result.auc}"

        # Additional: terminal half-life should be reasonable
        assert result.half_life_h > 0, "Half-life should be positive"

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "layer": "human",
            "name": "PBPK IV bolus simulation",
            "reference": {
                "source": "IV bolus pharmacokinetics (one-compartment model)",
                "authors": "Rowland & Tozer",
                "year": 2011,
                "journal": "Clinical Pharmacokinetics and Pharmacodynamics",
                "note": "C0 = Dose*F/Vd, monoexponential decay for IV bolus",
            },
            "expected": {
                "metric": "c0_concentration",
                "value": expected_c0,
                "tolerance": 0.05,
                "unit": "mg/L",
            },
            "actual": {
                "value": actual_c0,
            },
            "error": {
                "abs_error": abs(actual_c0 - expected_c0),
                "rel_error": abs(actual_c0 - expected_c0) / expected_c0 if expected_c0 > 0 else 0.0,
                "passed": True,
            },
            "reproducibility": {
                "deterministic": True,
                "environment": f"Python {sys.version.split()[0]}",
            },
            "checks": {
                "creates_valid_drug": True,
                "runs_pbpk_model_for_24h": True,
                "cmax_at_t0_for_iv_bolus": True,
                "concentration_decreases_monotonically_after_cmax": True,
                "auc_positive": True,
            },
            "details": {
                "c_max": round(c_max, 4),
                "t_max": t_max,
                "c0_expected": round(expected_c0, 4),
                "c0_actual": round(actual_c0, 4),
                "auc": round(result.auc, 4),
                "half_life_h": round(result.half_life_h, 4),
                "n_time_points": len(result.time_h),
                "final_concentration": round(concs[-1], 6),
            },
            "runtime_seconds": elapsed,
        })
    except Exception as e:
        results.update({
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        })
    return results


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
