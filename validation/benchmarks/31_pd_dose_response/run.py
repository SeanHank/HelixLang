#!/usr/bin/env python3
"""Benchmark 31: Hill equation dose-response curve."""
from __future__ import annotations

import json
import sys
import time

from helixlang.human.pharmacodynamics import hill_equation


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "31_pd_dose_response"}
    try:
        ec50 = 10.0
        n = 1.0
        emax = 1.0

        # 1. hill_equation(C=0, ec50, n) -> effect = 0
        effect_at_zero = hill_equation(0.0, ec50, emax, n)
        assert effect_at_zero == 0.0, (
            f"Effect at C=0 should be 0.0, got {effect_at_zero}"
        )

        # Also test negative concentration returns e0
        effect_at_neg = hill_equation(-5.0, ec50, emax, n)
        assert effect_at_neg == 0.0, (
            f"Effect at C<0 should be 0.0, got {effect_at_neg}"
        )

        # 2. hill_equation(C=ec50, ec50, n=1) -> effect = 0.5
        effect_at_ec50 = hill_equation(ec50, ec50, emax, n)
        expected_half = 0.5
        assert abs(effect_at_ec50 - expected_half) < 1e-10, (
            f"Effect at C=EC50 should be {expected_half}, got {effect_at_ec50}"
        )

        # 3. hill_equation(C>>ec50) -> effect approaches emax
        effect_at_high = hill_equation(1000.0 * ec50, ec50, emax, n)
        assert effect_at_high > 0.99, (
            f"Effect at C=1000*EC50 should approach {emax}, got {effect_at_high}"
        )
        assert effect_at_high <= emax, (
            f"Effect should not exceed emax={emax}, got {effect_at_high}"
        )

        # 4. Dose-response is monotonically increasing
        test_concentrations = [0.0, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
        effects = [hill_equation(c, ec50, emax, n) for c in test_concentrations]
        monotonic = True
        for i in range(1, len(effects)):
            if effects[i] < effects[i - 1] - 1e-12:
                monotonic = False
                break
        assert monotonic, "Dose-response should be monotonically increasing"

        # Verify with non-unit Hill coefficient (steepness)
        n_steep = 2.0
        effect_steep_at_ec50 = hill_equation(ec50, ec50, emax, n_steep)
        assert abs(effect_steep_at_ec50 - 0.5) < 1e-10, (
            f"Effect at C=EC50 with n=2 should still be 0.5, got {effect_steep_at_ec50}"
        )

        # Verify with e0 (baseline effect)
        e0 = 0.2
        effect_with_e0 = hill_equation(0.0, ec50, emax, n, e0=e0)
        assert effect_with_e0 == e0, (
            f"Effect at C=0 with e0 should be {e0}, got {effect_with_e0}"
        )
        effect_high_with_e0 = hill_equation(1000.0 * ec50, ec50, emax, n, e0=e0)
        assert effect_high_with_e0 > e0, (
            f"High concentration effect should exceed e0={e0}"
        )

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "checks": {
                "hill_at_zero_concentration_is_zero": True,
                "hill_at_ec50_is_half_max": True,
                "hill_at_high_concentration_approaches_emax": True,
                "dose_response_monotonically_increasing": True,
            },
            "details": {
                "ec50": ec50,
                "hill_n": n,
                "effect_at_zero": effect_at_zero,
                "effect_at_ec50": round(effect_at_ec50, 10),
                "effect_at_1000x_ec50": round(effect_at_high, 10),
                "effects_curve": [round(e, 6) for e in effects],
                "steep_hill_at_ec50": round(effect_steep_at_ec50, 10),
                "effect_with_baseline_at_zero": effect_with_e0,
                "effect_with_baseline_at_high": round(effect_high_with_e0, 6),
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
