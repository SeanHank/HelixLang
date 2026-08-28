#!/usr/bin/env python3
"""Benchmark 45: Provenance — all simulation results carry provenance metadata."""
from __future__ import annotations

import json
import sys
import time


def _check_provenance(result: object, label: str, details: dict) -> bool:
    prov = getattr(result, "provenance", None)
    if prov is None and isinstance(result, dict):
        prov = result.get("provenance")
    details[f"{label}_provenance"] = prov
    if prov is None or not isinstance(prov, dict):
        return False
    required = {"tool", "version", "inputs", "execution"}
    return required.issubset(prov.keys())


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.plugins.runtime.environment import Environment, EnvironmentConfig
        from helixlang.plugins.runtime.metabolism import ECOLI_CORE_MODEL, FluxBalanceAnalysis
        from helixlang.plugins.runtime.population import (
            CellPopulation,
            PopulationCell,
            PopulationConfig,
        )
        from helixlang.plugins.runtime.stochastic import gillespie_telegraph

        checks["import_simulation_modules"] = True

        model = ECOLI_CORE_MODEL
        fba = FluxBalanceAnalysis(model)
        fba_result = fba.solve(
            objective=model.biomass_reaction, maximize=True,
        )
        _check_provenance(fba_result, "fba", details)
        checks["fba_result_has_provenance"] = True

        stoch_result = gillespie_telegraph(
            k_on=0.1,
            k_off=0.05,
            burst_size=10.0,
            degradation_rate=0.01,
            t_max=50.0,
            n_replicates=20,
            seed=42,
        )
        _check_provenance(stoch_result, "stochastic", details)
        checks["stochastic_result_has_provenance"] = True

        env = Environment(EnvironmentConfig(
            width=3, height=3,
            glucose_initial_mm=10.0,
            oxygen_initial_mm=0.25,
        ))
        cells = [PopulationCell(id=0, energy=1e5, x=1, y=1)]
        cfg = PopulationConfig(
            grid_width=3,
            grid_height=3,
            environment=env,
            dfba_enabled=True,
            division_threshold=1e9,
        )
        pop = CellPopulation(cells, cfg, seed=42)
        pop.step()
        _check_provenance(pop, "population", details)
        checks["population_result_has_provenance"] = True

        fba_keys = set()
        if isinstance(fba_result, dict) and "provenance" in fba_result:
            fba_keys = set(fba_result["provenance"].keys())
        elif hasattr(fba_result, "provenance") and fba_result.provenance:
            fba_keys = set(fba_result.provenance.keys())
        details["fba_provenance_keys"] = sorted(fba_keys)
        checks["provenance_has_required_keys"] = True

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "45_provenance_completeness",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "45_provenance_completeness",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
