#!/usr/bin/env python3
"""Benchmark 33: Disease ODE model simulation."""
from __future__ import annotations

import json
import sys
import time

from helixlang.plugins.human.disease_ode_models import (
    CancerODE,
    CardiovascularODE,
    HematologicalODE,
    HepaticODE,
    MetabolicT2DODE,
    NeurologicalODE,
    RenalODE,
    create_disease_model,
)


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "33_disease_ode"}
    try:
        # 1. create_disease_model("metabolic_t2d") returns an ODE model
        model = create_disease_model("metabolic_t2d")
        assert model is not None, "create_disease_model returned None"
        assert hasattr(model, "step"), "Model should have a step() method"

        # 2. Model has a step() method
        assert callable(getattr(model, "step", None)), "step() should be callable"

        # 3. Running 365 daily steps produces a trajectory
        dt_h = 24.0  # daily steps
        beta_trajectory: list[float] = []
        hepatic_glr_trajectory: list[float] = []

        # Record initial state
        if hasattr(model, "beta_cell_function"):
            beta_trajectory.append(model.beta_cell_function)
        if hasattr(model, "hepatic_glr"):
            hepatic_glr_trajectory.append(model.hepatic_glr)

        for day in range(365):
            # Simulate with moderate glucose (T2D patient)
            glucose = 160.0 + 20.0 * (day / 365.0)
            insulin = 8.0
            drug_eff = 0.3  # partial metformin effect
            model.step(dt_h, glucose_mg_dl=glucose, insulin_uuml=insulin,
                       drug_effectiveness=drug_eff)

            if hasattr(model, "beta_cell_function"):
                beta_trajectory.append(model.beta_cell_function)
            if hasattr(model, "hepatic_glr"):
                hepatic_glr_trajectory.append(model.hepatic_glr)

        # Verify trajectory was produced
        assert len(beta_trajectory) > 1, "No beta cell trajectory"
        assert len(hepatic_glr_trajectory) > 1, "No hepatic GLR trajectory"

        # 4. HbA1c (proxied by beta cell function) starts near baseline and changes
        initial_beta = beta_trajectory[0]
        final_beta = beta_trajectory[-1]
        assert abs(initial_beta - 1.0) < 0.5, (
            f"Initial beta cell should be near 1.0, got {initial_beta}"
        )
        # Over 365 days of glucotoxicity, beta cell function should decline
        assert final_beta < initial_beta, (
            f"Beta cell function should decline over 365 days: "
            f"start={initial_beta:.4f}, end={final_beta:.4f}"
        )

        # 5. At least 5 disease models are available
        model_names_tested = []
        test_cases = [
            ("cardiovascular", CardiovascularODE),
            ("metabolic_t2d", MetabolicT2DODE),
            ("cancer", CancerODE),
            ("renal", RenalODE),
            ("hepatic", HepaticODE),
            ("neurological", NeurologicalODE),
            ("hematological", HematologicalODE),
        ]
        for name, expected_type in test_cases:
            m = create_disease_model(name)
            assert isinstance(m, expected_type), (
                f"create_disease_model({name!r}) returned {type(m).__name__}, "
                f"expected {expected_type.__name__}"
            )
            assert hasattr(m, "step"), f"{name} model should have step()"
            model_names_tested.append(name)

        assert len(model_names_tested) >= 5, (
            f"Expected >=5 disease models, tested {len(model_names_tested)}"
        )

        # Verify step advances the model (apply a perturbation to break equilibrium)
        cv_model = create_disease_model("cardiovascular")
        cv_model.hypertension_severity = 0.5
        initial_map = cv_model.map_mmhg
        for _ in range(24):
            cv_model.step(1.0)
        assert abs(cv_model.map_mmhg - initial_map) > 0.01, (
            "Cardiovascular model should change state after perturbation"
        )

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "reference": {
                "source": "Minimal model of glucose-insulin dynamics; T2D beta-cell glucotoxicity progression",
                "doi": "10.1152/ajpendo.1981.240.4.E480",
                "note": "Bergman RN et al. 1981 minimal model of glucose disappearance, Am J Physiol 240:E480-E490; qualitative T2D beta-cell decline used here.",
            },
            "checks": {
                "create_disease_model_returns_ode_model": True,
                "model_has_step_method": True,
                "running_365_daily_steps_produces_trajectory": True,
                "hba1c_starts_near_baseline_and_changes": True,
                "at_least_5_disease_models_available": True,
            },
            "details": {
                "models_tested": model_names_tested,
                "initial_beta_cell_function": round(initial_beta, 6),
                "final_beta_cell_function": round(final_beta, 6),
                "trajectory_length": len(beta_trajectory),
                "initial_hepatic_glr": round(hepatic_glr_trajectory[0], 6),
                "final_hepatic_glr": round(hepatic_glr_trajectory[-1], 6),
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
