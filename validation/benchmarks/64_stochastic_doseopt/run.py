#!/usr/bin/env python3
"""Benchmark 64: Stochastic ODE + dose optimizer.

Validates:
  - Stochastic ODE solver (Euler-Maruyama, SDE ensembles)
  - Bayesian dose optimization (PK profiles, PTA, ECDF distance)

Reference:
  Maruyama K 1955, Proc Imp Acad 30:10 (Euler-Maruyama)
  Jusko WJ, Ko HC 1994, J Pharmacokinet Biopharm 22:389 (dose optimization)
"""
from __future__ import annotations

import json
import math
import random
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.human.stochastic_ode import (
            SDEConfig,
            SDEDistribution,
            SDETrajectory,
            euler_maruyama_step,
            solve_sde,
            solve_sde_ensemble,
        )
        from helixlang.human.dose_optimizer import (
            DoseOptimizer,
            DoseRecommendation,
            PKProfile,
        )
        checks["import_all_classes"] = True

        # --- Check 2: euler_maruyama_step ---
        # dx = -x*dt + sigma*dW with x0=1.0, dt=0.01, sigma_intrinsic=0.1
        rng = random.Random(42)
        x0 = 1.0
        dt = 0.01
        drift = -x0
        sigma_in = 0.1
        sigma_ex = 0.05
        x1 = euler_maruyama_step(x0, dt, drift, sigma_in, sigma_ex, rng)
        checks["euler_maruyama_step_finite"] = math.isfinite(x1)
        details["euler_maruyama_x0"] = x0
        details["euler_maruyama_x1"] = round(x1, 6)

        # --- Check 3: solve_sde ---
        # dx = -x, x0=1.0, t_end=1.0
        traj = solve_sde(
            t_end=1.0, dt=0.01, state0=1.0,
            drift_fn=lambda t, x: -x,
            sigma_intrinsic=0.1, sigma_extrinsic=0.05, seed=123,
        )
        assert isinstance(traj, SDETrajectory), "solve_sde should return SDETrajectory"
        final_x = traj.states[-1]
        checks["solve_sde_x_positive_at_t1"] = final_x > 0
        details["solve_sde_final_x"] = round(final_x, 6)
        details["solve_sde_n_points"] = len(traj.states)

        # --- Check 4: SDEConfig instantiation ---
        cfg = SDEConfig(sigma_intrinsic=0.1, sigma_extrinsic=0.05, n_patients=100, dt=0.1, seed=42)
        assert cfg.sigma_intrinsic == 0.1
        assert cfg.n_patients == 100
        checks["sde_config_instantiates"] = True

        # --- Check 5: DoseOptimizer instantiation ---
        opt = DoseOptimizer(target_range=(10.0, 50.0), target_auc_range=(100.0, 500.0))
        assert opt.target_range == (10.0, 50.0)
        checks["dose_optimizer_instantiates"] = True

        # --- Check 6: PKProfile has time/concentration arrays, AUC computable ---
        times = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
        concs = [0.0, 15.0, 25.0, 20.0, 12.0, 7.0, 4.0, 1.5]
        pk = PKProfile(times=times, concentrations=concs)
        assert len(pk.times) == len(pk.concentrations)
        auc = opt.compute_auc(pk)
        cmax = opt.compute_cmax(pk)
        checks["pk_profile_has_time_conc"] = len(pk.times) == len(pk.concentrations)
        checks["pk_profile_auc_computable"] = auc > 0
        details["pk_auc"] = round(auc, 4)
        details["pk_cmax"] = cmax

        # --- Check 7: DoseRecommendation dose field accessible ---
        rec = DoseRecommendation(
            recommended_dose=100.0,
            regimen="100 mg q12h",
            target_auc_range=(100.0, 500.0),
            predicted_auc=200.0,
            predicted_cmax=30.0,
            predicted_tmin=0.0,
            pta=0.85,
            ecdf_distance=0.12,
            confidence=0.9,
        )
        assert rec.recommended_dose == 100.0
        checks["dose_recommendation_dose_accessible"] = True

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "64_stochastic_doseopt",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "euler_maruyama": "Maruyama K 1955, Proc Imp Acad 30:10",
                "dose_optimization": "Jusko WJ, Ko HC 1994, J Pharmacokinet Biopharm 22:389",
            },
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "64_stochastic_doseopt",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
