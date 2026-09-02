#!/usr/bin/env python3
"""Benchmark 23b: Stochastic — Gillespie SSA telegraph model Fano factor.

Analytical reference: Peccoud & Ycart 1995 — the two-state (telegraph)
promoter Fano factor is F = 1 + b * k_on / (k_on + k_off) * 1/(1 + γ/(k_on+k_off))
where b = burst_size, γ = degradation_rate.  The Gillespie SSA must produce
an observed Fano within 30% of this analytic value for 2000 replicates.
"""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "23b_stochastic"}
    try:
        from helixlang.plugins.runtime.stochastic import (
            TelegraphPromoter,
            gillespie_telegraph,
            telegraph_fano_factor,
        )

        # Telegraph model parameters
        k_on = 0.5     # 1/min
        k_off = 0.3    # 1/min
        burst = 5.0
        gamma = 0.14   # 1/min (mRNA degradation)

        # ── Analytical Fano factor ───────────────────────────────────
        theoretical_fano = telegraph_fano_factor(k_on, k_off, burst, gamma)
        assert theoretical_fano > 1.0, "Bursty promoter should have Fano > 1"

        # ── TelegraphPromoter object ─────────────────────────────────
        tp = TelegraphPromoter(k_on=k_on, k_off=k_off, burst_size=burst,
                               degradation_rate=gamma)
        tp_fano = tp.fano_factor()
        tp_ok = abs(tp_fano - theoretical_fano) < 1e-10

        # ── Gillespie SSA ────────────────────────────────────────────
        ssa = gillespie_telegraph(
            k_on=k_on,
            k_off=k_off,
            burst_size=burst,
            degradation_rate=gamma,
            t_max=500.0,
            n_replicates=2000,
            seed=42,
        )
        observed_fano = ssa["fano"]

        # Fano > 1 (super-Poissonian)
        fano_super_poisson = observed_fano > 1.0

        # Within 30% of theoretical
        if theoretical_fano > 0:
            rel_err = abs(observed_fano - theoretical_fano) / theoretical_fano
        else:
            rel_err = float("inf")
        fano_close = rel_err < 0.30

        all_ok = tp_ok and fano_super_poisson and fano_close

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS" if all_ok else "FAIL",
            "checks": {
                "telegraph_promoter_fano_matches": tp_ok,
                "fano_greater_than_one": fano_super_poisson,
                "fano_within_30pct_of_theory": fano_close,
            },
            "details": {
                "fano_theoretical": round(theoretical_fano, 4),
                "fano_observed_ssa": round(observed_fano, 4),
                "fano_relative_error": round(rel_err, 4),
                "fano_promoter_object": round(tp_fano, 4),
                "ssa_mean": round(ssa["mean"], 2),
                "ssa_variance": round(ssa["variance"], 2),
                "n_replicates": 2000,
                "t_max": 500.0,
            },
            "reference": {
                "source": "Peccoud & Ycart 1995 — telegraph promoter Fano factor",
                "authors": "Peccoud J, Ycart B",
                "year": 1995,
                "journal": "Theoretical Population Biology",
                "note": f"F = 1 + b·k_on/(k_on+k_off)·1/(1+γ/(k_on+k_off)); "
                        f"theoretical={round(theoretical_fano, 4)}",
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
