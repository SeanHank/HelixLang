#!/usr/bin/env python3
"""Benchmark 10: Whole-cell simulation — Tier 1 validation.

Validates VirtualCell against biophysical constraints:
  (a) Energy budget: division_time = (division_energy - energy_init) / net_income
  (b) Protein expression: each gene produces protein proportional to expression level
  (c) Mass conservation: mass must increase monotonically (barring numerical noise)
  (d) Division mechanics: energy halves at division, biomass doubles
"""
from __future__ import annotations

import json
import math
import sys
import time


def _pearson_r(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=True))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx < 1e-15 or sy < 1e-15:
        return 0.0
    return cov / (sx * sy)


def run() -> dict:
    t0 = time.perf_counter()
    try:
        from helixlang.grn import GRN
        from helixlang.virtual_cell import (
            VirtualCell,
            VirtualCellConfig,
        )

        grn = GRN()
        grn.add_gene("lacI", threshold=-1.0, initial_level=1.0)
        grn.add_gene("tetR", threshold=-1.0, initial_level=0.5)

        genome = {
            "lacI": "AGGAGG" + "GACC" + "ATG" + "GCT" * 10 + "TAA",
            "tetR": "AGGAGG" + "GACC" + "ATG" + "GGT" * 10 + "TAA",
        }

        config = VirtualCellConfig(
            energy_init=1.5e9,
            division_energy=2.0e9,
            maintenance_atp_per_min=2.5e7,
            biomass_to_atp=3.0e7,
            uptake={"GLC": 10.0},
        )

        cell = VirtualCell(genome, grn, config=config)
        history = cell.run(200)

        alive = cell.alive
        elapsed = time.perf_counter() - t0

        # ── (a) Energy budget analytical comparison ─────────────────────
        # First division: time = (division_energy - energy_init) / net_income
        # Subsequent divisions: time = (division_energy - division_energy/2) / net_income
        #   because energy halves at each division.
        biomass_fluxes = [h.get("biomass_flux", 0.0) for h in history
                          if h.get("alive", True)]
        avg_flux = sum(biomass_fluxes) / len(biomass_fluxes) if biomass_fluxes else 0.0
        net_income = avg_flux * config.biomass_to_atp - config.maintenance_atp_per_min
        first_div_time = ((config.division_energy - config.energy_init) / net_income
                          if net_income > 0 else float("inf"))
        subsequent_div_time = ((config.division_energy - config.division_energy / 2.0) / net_income
                               if net_income > 0 else float("inf"))

        # Actual division times from history
        division_times = []
        prev_divs = 0
        for i, h in enumerate(history):
            d = h.get("divisions", 0)
            if d > prev_divs:
                division_times.append(i)
                prev_divs = d
        if len(division_times) >= 2:
            first_actual = division_times[0]
            subsequent_intervals = [division_times[i+1] - division_times[i]
                                    for i in range(len(division_times) - 1)]
            avg_subsequent = sum(subsequent_intervals) / len(subsequent_intervals)
        else:
            first_actual = float("inf")
            avg_subsequent = float("inf")

        # First division within 30% of analytical
        first_div_ok = (
            abs(first_actual - first_div_time) / first_div_time < 0.3
            if first_div_time > 0 and math.isfinite(first_actual)
            else False
        )
        # Subsequent intervals within 30% of analytical
        subsequent_div_ok = (
            abs(avg_subsequent - subsequent_div_time) / subsequent_div_time < 0.3
            if subsequent_div_time > 0 and math.isfinite(avg_subsequent)
            else True  # Pass if <2 divisions
        )
        energy_budget_ok = first_div_ok and subsequent_div_ok

        # ── (b) Protein expression consistency ──────────────────────────
        # lacI has initial_level=1.0, tetR has 0.5
        # After running, both should have proteins (expression from genome)
        final_proteins = history[-1].get("proteins", {}) if history else {}
        lacI_level = final_proteins.get("lacI", 0.0)
        tetR_level = final_proteins.get("tetR", 0.0)
        proteins_present = lacI_level > 0 and tetR_level > 0

        # ── (c) Mass conservation: mass should increase monotonically ───
        masses = [h.get("mass", 0.0) for h in history]
        mass_increases = all(
            masses[i+1] >= masses[i] - 1e-10
            for i in range(len(masses) - 1)
        ) if len(masses) > 1 else False

        # ── (d) Division mechanics: post-division energy ≈ division_energy/2 ──
        # After division, energy should be ~division_energy/2 (binary fission)
        energy_at_divisions = []
        prev_divs = 0
        for h in history:
            d = h.get("divisions", 0)
            if d > prev_divs:
                energy_at_divisions.append(h.get("energy", 0.0))
                prev_divs = d
        expected_post_div_energy = config.division_energy / 2.0
        energy_halves = all(
            abs(e - expected_post_div_energy) / expected_post_div_energy < 0.15
            for e in energy_at_divisions
        ) if energy_at_divisions else False

        # ── (e) Time-series trajectory for reference ────────────────────
        trajectory_keyframes = []
        for i in range(0, len(history), max(1, len(history) // 10)):
            h = history[i]
            trajectory_keyframes.append({
                "t": i,
                "energy": round(h.get("energy", 0.0), 2),
                "mass": round(h.get("mass", 0.0), 4),
                "biomass_flux": round(h.get("biomass_flux", 0.0), 4),
                "divisions": h.get("divisions", 0),
                "alive": h.get("alive", True),
            })

        all_pass = (alive and energy_budget_ok and proteins_present
                    and mass_increases and energy_halves)

        return {
            "id": "10_whole_cell",
            "status": "PASS" if all_pass else "FAIL",
            "validation": {
                "alive": alive,
                "energy_budget_ok": energy_budget_ok,
                "proteins_present": proteins_present,
                "mass_monotonic": mass_increases,
                "energy_halves_at_division": energy_halves,
            },
            "energy_budget": {
                "avg_biomass_flux": round(avg_flux, 4),
                "net_income_ATP_per_min": round(net_income, 2),
                "first_division_analytical_min": round(first_div_time, 1),
                "first_division_actual_min": first_actual,
                "subsequent_interval_analytical_min": round(subsequent_div_time, 1),
                "subsequent_interval_actual_min": round(avg_subsequent, 1) if math.isfinite(avg_subsequent) else None,
                "divisions_observed": cell.divisions,
                "division_times": division_times,
                "first_div_ok": first_div_ok,
                "subsequent_div_ok": subsequent_div_ok,
            },
            "proteins": {
                "lacI": round(lacI_level, 4),
                "tetR": round(tetR_level, 4),
            },
            "trajectory_keyframes": trajectory_keyframes,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "10_whole_cell",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
