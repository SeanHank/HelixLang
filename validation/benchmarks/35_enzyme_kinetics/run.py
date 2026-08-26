#!/usr/bin/env python3
"""Benchmark 35: Enzyme Kinetics — kcat and Km prediction for known enzymes."""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.kinetics.kcat_predictor import (
            KcatPredictor,
            predict_kcat,
        )
        from helixlang.kinetics.km_estimator import (
            KmEstimator,
            estimate_km,
        )
        checks["import_kinetics_modules"] = True

        kcat = predict_kcat(
            reaction_id="HEX1",
            ec_number="2.7.1.1",
            substrate="glucose",
            target_organism="Escherichia coli",
        )
        details["kcat_hex1"] = kcat.kcat_value
        details["kcat_source"] = kcat.source
        assert kcat.kcat_value > 0, "kcat should be > 0"
        checks["predict_kcat_positive"] = True

        assert kcat.kcat_value < 10000, f"kcat {kcat.kcat_value} should be < 10000 s^-1"
        checks["predict_kcat_in_range"] = True

        km_hex1 = estimate_km(
            reaction_id="HEX1",
            substrate="glucose",
            target_organism="Escherichia coli",
        )
        details["km_hex1"] = km_hex1
        assert km_hex1 > 0, "Km should be > 0"
        checks["predict_km_positive"] = True

        assert km_hex1 < 100, f"Km {km_hex1} should be < 100 mM"
        checks["predict_km_in_range"] = True

        predictor = KcatPredictor(target_organism="Escherichia coli")
        test_enzymes = [
            ("PFK", "2.7.1.11", "fructose-6-phosphate"),
            ("CS", "2.3.3.1", "acetyl-CoA"),
            ("ENO", "4.2.1.11", "phosphoenolpyruvate"),
            ("PYK", "2.7.1.40", "phosphoenolpyruvate"),
        ]
        kcat_values = {}
        for rxn_id, ec, sub in test_enzymes:
            pred = predictor.predict(rxn_id, ec_number=ec, substrate=sub)
            kcat_values[rxn_id] = pred.kcat_value
            assert pred.kcat_value > 0, f"kcat for {rxn_id} should be > 0"
        details["additional_kcats"] = kcat_values

        estimator = KmEstimator(target_organism="Escherichia coli")
        km_values = {}
        for sub in ["glucose", "pyruvate", "ATP", "acetyl-CoA"]:
            km = estimator.estimate("test_rxn", substrate=sub)
            km_values[sub] = km
            assert km > 0, f"Km for {sub} should be > 0"
            assert km < 100, f"Km for {sub} should be < 100 mM"
        details["additional_kms"] = km_values

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())

        # Experimental comparison against published enzyme kinetics data
        published_kcats = {
            "PFK": {"kcat_s1": 300.0, "range": [200, 400], "ref": "Krebs & Bridger 1980; Bar-Even et al. 2011"},
            "CS":  {"kcat_s1": 70.0,  "range": [50, 100],  "ref": "Krebs & Bridger 1980; BRENDA EC 2.3.3.1"},
            "ENO": {"kcat_s1": 200.0, "range": [150, 300], "ref": "BRENDA EC 4.2.1.11; Panchaud et al. 2009"},
            "PYK": {"kcat_s1": 400.0, "range": [300, 500], "ref": "BRENDA EC 2.7.1.40; Bar-Even et al. 2011"},
        }
        published_kms = {
            "glucose":    {"km_mM": 0.1,  "range": [0.05, 0.3],  "ref": "BRENDA EC 2.7.1.1; Lostia et al. 2022"},
            "ATP":        {"km_mM": 0.3,  "range": [0.1, 1.0],   "ref": "BRENDA EC 2.7.1.1; Newsholme & Crabtree 1980"},
            "acetyl-CoA": {"km_mM": 0.05, "range": [0.01, 0.2],  "ref": "BRENDA EC 2.3.3.1; Krebs & Bridger 1980"},
        }
        exp_comparison = {"published_kcats": {}, "published_kms": {}}
        for rxn_id, ref_data in published_kcats.items():
            pred = kcat_values.get(rxn_id, 0)
            in_range = ref_data["range"][0] <= pred <= ref_data["range"][1]
            exp_comparison["published_kcats"][rxn_id] = {
                "predicted": pred,
                "published": ref_data["kcat_s1"],
                "range": ref_data["range"],
                "within_range": in_range,
                "ref": ref_data["ref"],
            }
        for sub, ref_data in published_kms.items():
            pred = km_values.get(sub, 0)
            in_range = ref_data["range"][0] <= pred <= ref_data["range"][1]
            exp_comparison["published_kms"][sub] = {
                "predicted": pred,
                "published": ref_data["km_mM"],
                "range": ref_data["range"],
                "within_range": in_range,
                "ref": ref_data["ref"],
            }

        return {
            "id": "35_enzyme_kinetics",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "experimental_comparison": exp_comparison,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "35_enzyme_kinetics",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
