#!/usr/bin/env python3
"""Benchmark 44: Determinism — same seed produces identical output for all backends."""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.environment import Environment, EnvironmentConfig
        from helixlang.population import (
            CellPopulation,
            PopulationCell,
            PopulationConfig,
        )
        from helixlang.stochastic import gillespie_telegraph

        checks["import_simulation_modules"] = True

        def _run_stochastic(seed: int) -> dict:
            result = gillespie_telegraph(
                k_on=0.1,
                k_off=0.05,
                burst_size=10.0,
                degradation_rate=0.01,
                t_max=100.0,
                n_replicates=50,
                seed=seed,
            )
            return {k: round(v, 4) for k, v in result.items()}

        stoch_a = _run_stochastic(42)
        stoch_b = _run_stochastic(42)
        details["stochastic_a"] = stoch_a
        details["stochastic_b"] = stoch_b
        assert stoch_a == stoch_b, (
            f"Stochastic determinism failed: {stoch_a} != {stoch_b}"
        )
        checks["stochastic_gillespie_deterministic"] = True

        def _run_evolution(seed: int) -> dict:
            import random

            from helixlang.evolution import EvolutionaryPopulation, EvolutionConfig
            cfg = EvolutionConfig(
                mutation_rate=0.01,
                population_size=20,
                generations=3,
                selection_coefficient=2.0,
            )
            rng = random.Random(seed)
            pop = EvolutionaryPopulation(
                initial_dna="ACGT" * 10,
                config=cfg,
                target_dna="ACGT" * 10,
                fitness_method="hamming",
                rng=rng,
            )
            pop.evolve()
            stats = pop.get_generation_stats()
            return {
                "final_generation": stats[-1].get("generation", 0)
                if stats else 0,
                "final_mean_fitness": round(
                    stats[-1].get("mean_fitness", 0.0), 4
                ) if stats else 0.0,
                "n_generations": len(stats),
                "seed": seed,
            }

        evo_a = _run_evolution(42)
        evo_b = _run_evolution(42)
        details["evolution_a"] = evo_a
        details["evolution_b"] = evo_b
        assert evo_a == evo_b, (
            f"Evolution determinism failed: {evo_a} != {evo_b}"
        )
        checks["evolution_deterministic"] = True

        def _run_population(seed: int) -> dict:
            env = Environment(EnvironmentConfig(
                width=4, height=4,
                glucose_initial_mm=10.0,
                oxygen_initial_mm=0.25,
            ))
            cells = [
                PopulationCell(id=0, energy=1e5, x=2, y=2),
                PopulationCell(id=1, energy=1e5, x=2, y=3),
            ]
            cfg = PopulationConfig(
                grid_width=4,
                grid_height=4,
                environment=env,
                dfba_enabled=True,
                division_threshold=1e9,
            )
            pop = CellPopulation(cells, cfg, seed=seed)
            for _ in range(5):
                pop.step()
            alive = sum(1 for c in pop.cells if c.alive)
            total_energy = sum(c.energy for c in pop.cells if c.alive)
            return {"alive": alive, "total_energy": total_energy, "seed": seed}

        pop_a = _run_population(42)
        pop_b = _run_population(42)
        details["population_a"] = pop_a
        details["population_b"] = pop_b
        assert pop_a == pop_b, (
            f"Population determinism failed: {pop_a} != {pop_b}"
        )
        checks["population_deterministic"] = True

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "44_determinism_all_backends",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "44_determinism_all_backends",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
