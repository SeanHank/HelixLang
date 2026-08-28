#!/usr/bin/env python3
"""Benchmark 38: Ecosystem — Lotka-Volterra predator-prey dynamics."""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.plugins.apps.ecosystem import (
            lotka_volterra_conserved,
            lotka_volterra_step,
        )

        checks["import_ecosystem_module"] = True

        alpha = 1.1
        beta = 0.4
        delta = 0.1
        gamma = 0.4
        prey, predator = 10.0, 5.0

        prey, predator = lotka_volterra_step(
            prey, predator, alpha, beta, delta, gamma, dt=1.0, substeps=16,
        )
        details["after_1_step"] = {"prey": prey, "predator": predator}
        checks["lotka_volterra_step_runs"] = True

        prey, predator = 10.0, 5.0
        history_prey = [prey]
        history_pred = [predator]
        for _ in range(50):
            prey, predator = lotka_volterra_step(
                prey, predator, alpha, beta, delta, gamma,
                dt=1.0, substeps=16,
            )
            history_prey.append(prey)
            history_pred.append(predator)

        assert all(p > 0 for p in history_prey), "Prey went to zero"
        assert all(p > 0 for p in history_pred), "Predator went to zero"
        checks["both_populations_positive"] = True

        prey_range = max(history_prey) - min(history_prey)
        pred_range = max(history_pred) - min(history_pred)
        details["prey_range"] = prey_range
        details["predator_range"] = pred_range
        assert prey_range > 0.5, "Prey population should oscillate"
        assert pred_range > 0.5, "Predator population should oscillate"
        checks["oscillatory_behavior"] = True

        V_init = lotka_volterra_conserved(
            10.0, 5.0, alpha, beta, delta, gamma,
        )
        V_final = lotka_volterra_conserved(
            prey, predator, alpha, beta, delta, gamma,
        )
        details["V_initial"] = V_init
        details["V_final"] = V_final
        rel_diff = abs(V_final - V_init) / max(abs(V_init), 1e-10)
        details["V_relative_diff"] = rel_diff
        assert rel_diff < 0.15, (
            f"Conserved quantity drifted too much: {rel_diff:.4f}"
        )
        checks["conserved_quantity_near_initial"] = True

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "38_ecosystem_lotka_volterra",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "38_ecosystem_lotka_volterra",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
