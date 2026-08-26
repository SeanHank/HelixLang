#!/usr/bin/env python3
"""Benchmark 08: Population dynamics — exponential growth + competition.

Validates against analytical solutions:
  - Growth:  N(t) = N_0 * 2^(t / t_d)  with t_d = 20 ticks
  - Competition: fast species dominates exponentially
"""
from __future__ import annotations

import json
import math
import sys
import time

# ── Analytical reference ────────────────────────────────────────────
# Energy-budget physics (per tick = 1 min):
#   energy_intake  = 5.0e7 ATP
#   metabolic_cost = 1.0e7 ATP
#   net gain       = 4.0e7 ATP / tick
#   division_threshold = 1.8e9 ATP
#   newborn energy     = 1.0e9 ATP  (after first division halves energy)
#   ticks to divide    = (1.8e9 - 1.0e9) / 4.0e7 = 20
# → doubling time t_d = 20 ticks
TICKS_PER_DIVISION = (1.8e9 - 1.0e9) / (5.0e7 - 1.0e7)  # 20.0


def _analytical_N(n0: int, t: int, td: float = TICKS_PER_DIVISION) -> float:
    """Continuous exponential: N(t) = N_0 * 2^(t/t_d)."""
    return n0 * (2.0 ** (t / td))


def _analytical_competition(n0: int, t: int,
                            td_fast: float, td_slow: float) -> tuple[float, float]:
    """Analytical N_fast(t), N_slow(t) for two non-interacting species."""
    nf = n0 * (2.0 ** (t / td_fast))
    ns = n0 * (2.0 ** (t / td_slow))
    return nf, ns


# ── Peak finder (reused by all benchmarks) ──────────────────────────
def _find_peaks(vals: list[float], start_idx: int = 0) -> list[int]:
    """Return indices of local maxima."""
    peaks: list[int] = []
    for i in range(start_idx + 1, len(vals) - 1):
        if vals[i] > vals[i - 1] and vals[i] >= vals[i + 1]:
            peaks.append(i)
    return peaks


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "08_population_dynamics"}
    try:
        from helixlang.population import (
            CellPopulation,
            PopulationCell,
            PopulationConfig,
            SpeciesParams,
        )

        # ════════════════════════════════════════════════════════════════
        # Part A: exponential growth in excess glucose (constant intake)
        # ════════════════════════════════════════════════════════════════
        init_count = 10
        initial = [
            PopulationCell(id=i, x=i % 10, y=i // 10, energy=2.0e9)
            for i in range(init_count)
        ]
        config = PopulationConfig(
            max_size=2000,
            grid_width=50,
            grid_height=50,
            signaling_enabled=False,
            division_threshold=1.8e9,
        )
        pop = CellPopulation(initial, config=config, seed=42)
        n_ticks = 50

        sizes: list[int] = [init_count]
        for _ in range(n_ticks):
            pop.step()
            alive = sum(1 for c in pop.cells if c.alive)
            sizes.append(alive)

        n_final = sizes[-1]

        # ── Validation 1: growth curve within factor 2 of analytical ──
        n_analytical = _analytical_N(init_count, n_ticks)
        growth_ratio = n_final / n_analytical if n_analytical > 0 else 0.0
        growth_ok = 0.5 < growth_ratio < 2.0

        # ── Validation 2: doubling time 15-25 ticks ──────────────────
        if n_final > init_count and n_ticks > 0:
            observed_doublings = math.log2(n_final / init_count)
            observed_rate = observed_doublings / n_ticks
            observed_doubling_time = (1.0 / observed_rate
                                      if observed_rate > 0 else float("inf"))
        else:
            observed_rate = 0.0
            observed_doubling_time = float("inf")
        doubling_ok = 15.0 <= observed_doubling_time <= 25.0

        # ── Trajectory summary (every 5 ticks) ───────────────────────
        trajectory = [
            {"tick": i * 5, "N": sizes[i * 5]}
            for i in range(n_ticks // 5 + 1)
            if i * 5 < len(sizes)
        ]
        # Add final tick if not already present
        if n_ticks % 5 != 0:
            trajectory.append({"tick": n_ticks, "N": sizes[-1]})

        # ════════════════════════════════════════════════════════════════
        # Part B: competition between two species
        # ════════════════════════════════════════════════════════════════
        # Species A (fast): intake=5e7, cost=1e7, net=4e7 → td ≈ 20
        # Species B (slow): intake=2e7, cost=1e7, net=1e7 → td ≈ 80
        sp_fast = SpeciesParams(energy_intake=5.0e7, metabolic_cost=1.0e7)
        sp_slow = SpeciesParams(energy_intake=2.0e7, metabolic_cost=1.0e7)
        comp_cells = (
            [PopulationCell(id=i, x=0, y=0, energy=2.0e9, species="fast")
             for i in range(5)]
            + [PopulationCell(id=i + 5, x=0, y=0, energy=2.0e9, species="slow")
               for i in range(5)]
        )
        comp_config = PopulationConfig(
            max_size=2000,
            grid_width=20,
            grid_height=20,
            signaling_enabled=False,
            division_threshold=1.8e9,
            species_params={"fast": sp_fast, "slow": sp_slow},
        )
        comp_pop = CellPopulation(comp_cells, config=comp_config, seed=99)
        comp_trajectory: list[dict[str, object]] = []
        for tick in range(101):
            if tick % 10 == 0:
                fc = sum(1 for c in comp_pop.cells
                         if c.alive and c.species == "fast")
                sc = sum(1 for c in comp_pop.cells
                         if c.alive and c.species == "slow")
                comp_trajectory.append({
                    "tick": tick, "fast": fc, "slow": sc,
                })
            if tick < 100:
                comp_pop.step()

        fast_count = sum(1 for c in comp_pop.cells
                         if c.alive and c.species == "fast")
        slow_count = sum(1 for c in comp_pop.cells
                         if c.alive and c.species == "slow")
        total = fast_count + slow_count
        fast_frac = fast_count / total if total > 0 else 0.0

        # ── Validation 3: fast species > 70% after 100 ticks ─────────
        competition_ok = fast_frac > 0.70

        # ── Analytical comparison for competition ─────────────────────
        td_fast = TICKS_PER_DIVISION
        td_slow = (1.8e9 - 1.0e9) / (2.0e7 - 1.0e7)  # 80 ticks
        ana_fast, ana_slow = _analytical_competition(5, 100, td_fast, td_slow)
        ana_total = ana_fast + ana_slow
        ana_fast_frac = ana_fast / ana_total if ana_total > 0 else 0.0

        passed = growth_ok and doubling_ok and competition_ok

        # ════════════════════════════════════════════════════════════════
        # Assemble result
        # ════════════════════════════════════════════════════════════════
        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS" if passed else "FAIL",
            "validation": {
                "growth_curve_factor2": growth_ok,
                "doubling_time_15_25": doubling_ok,
                "fast_species_dominance": competition_ok,
            },
            "growth": {
                "N_initial": init_count,
                "N_final_helixlang": n_final,
                "N_final_analytical": round(n_analytical, 1),
                "growth_ratio": round(growth_ratio, 4),
                "observed_doubling_time_ticks": round(observed_doubling_time, 2),
                "expected_doubling_time_ticks": TICKS_PER_DIVISION,
                "trajectory": trajectory,
            },
            "competition": {
                "fast_count": fast_count,
                "slow_count": slow_count,
                "fast_fraction": round(fast_frac, 4),
                "analytical_fast_fraction": round(ana_fast_frac, 4),
                "threshold": 0.70,
                "trajectory": comp_trajectory,
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
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
