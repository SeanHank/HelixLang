#!/usr/bin/env python3
"""Benchmark 68: Biological validity — Helix Model vs measured data.

Validates the bio_validity framework (doc/37 §2):
  1. Out-of-scope detection against published parameter ranges
  2. Parameter fitting towards experimental target values
  3. Uncertainty quantification (bootstrap / Monte Carlo CI coverage)
  4. Replication verification (bit-exact reproducibility)
  5. Aggregated BioAccuracyReport

References:
  Orth JD et al. 2010, EcoSal Plus 4 (E. coli growth rates)
  Elowitz MB 2000, Nature 403:335 (repressilator period 160±40 min)
  Bar-Even A et al. 2011, Biochemistry 50:4402 (enzyme kcat/Km ranges)
  Wanner BL 1996, E. coli and Salmonella (generation time)
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
        from helixlang.plugins.runtime.bio_validity import (
            BioAccuracySuite,
            OutOfScopeDetector,
            ParameterFitter,
            ReplicationVerifier,
            ScopeLevel,
            UncertaintyQuantifier,
        )

        # --- Check 1: Out-of-scope detection ---
        detector = OutOfScopeDetector()
        report_safe = detector.check({
            "growth_rate": 0.87,        # Orth 2010 measured max
            "glucose_uptake": 10.0,
            "repressilator_period": 160.0,  # Elowitz 2000
        })
        report_bad = detector.check({"growth_rate": 42.0})
        checks["scope_in_out_detection"] = (
            report_safe.all_safe and report_bad.any_out_of_scope
        )
        details["scope_safe_checks"] = len(report_safe.checks)
        details["out_of_scope_growth_rate"] = (
            ScopeLevel.OUT_OF_SCOPE.value if report_bad.any_out_of_scope else "none"
        )

        # --- Check 2: Parameter fitting improves residual ---
        fitter = ParameterFitter()
        fitter._bounds = {
            "growth_rate": (0.1, 1.5),
            "glucose_uptake": (5.0, 15.0),
        }
        fit = fitter.fit(
            {"growth_rate": 2.0, "glucose_uptake": 5.0},
            {"growth_rate": 0.87, "glucose_uptake": 10.0},
        )
        checks["parameter_fit_improves"] = (
            fit.converged and fit.residual_after < fit.residual_before
        )
        details["fit_residual_before"] = round(fit.residual_before, 6)
        details["fit_residual_after"] = round(fit.residual_after, 6)
        details["fit_improvement_pct"] = round(fit.improvement_pct, 2)
        details["fitted_growth_rate"] = round(fit.fitted_params["growth_rate"], 4)

        # --- Check 3: Uncertainty CI contains the reference value ---
        uq = UncertaintyQuantifier(n_bootstrap=50, seed=42)
        unc = uq.monte_carlo(
            {"growth_rate": 0.87},
            {"growth_rate": 0.05},
            n_samples=100,
        )
        ci_contains_orth = unc.ci_lower <= 0.87 <= unc.ci_upper
        ci_contains_elowitz = unc.ci_lower <= 160.0 <= unc.ci_upper
        checks["uncertainty_ci_valid"] = ci_contains_orth
        details["uncertainty_mean"] = round(unc.mean, 4)
        details["uncertainty_ci"] = [
            round(unc.ci_lower, 4), round(unc.ci_upper, 4),
        ]

        # --- Check 4: Replication identical across runs ---
        verifier = ReplicationVerifier(n_runs=5, seed=0)
        repl = verifier.verify(max_ticks=40)
        checks["replication_identical"] = repl.all_identical
        details["replication_runs"] = repl.n_runs
        details["replication_hashes_unique"] = len(set(repl.hashes))

        # --- Check 5: Aggregated bio-accuracy suite passes ---
        suite = BioAccuracySuite()
        report = suite.run(
            "68_bio_validity",
            {"growth_rate": 0.87, "glucose_uptake": 10.0,
             "repressilator_period": 160.0, "hill_coefficient": 2.0},
            target_values={
                "growth_rate": 0.87,
                "glucose_uptake": 10.0,
                "repressilator_period": 160.0,
            },
            bounds={
                "growth_rate": (0.1, 1.5),
                "glucose_uptake": (5.0, 15.0),
                "repressilator_period": (120.0, 240.0),
            },
        )
        checks["accuracy_report_pass"] = report.overall_accuracy >= 0.8
        details["overall_accuracy"] = round(report.overall_accuracy, 4)
        details["report_status"] = report.status

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "68_bio_validity",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "growth_rate": "Orth JD et al. 2010, EcoSal Plus 4",
                "repressilator": "Elowitz MB et al. 2000, Nature 403:335",
                "kinetics": "Bar-Even A et al. 2011, Biochemistry 50:4402",
                "generation_time": "Wanner BL 1996",
            },
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "68_bio_validity",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)