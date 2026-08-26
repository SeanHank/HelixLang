#!/usr/bin/env python3
"""Benchmark 23a: Evolution — mutation, fitness, Wright-Fisher population."""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "23a_evolution"}
    try:
        from helixlang.evolution import (
            EvolutionaryPopulation,
            EvolutionConfig,
            calculate_fitness,
            mutate,
        )

        dna = "ATGCGTACGATCGATCGATCGATCGATCGATCGATCG" * 3  # 108 nt
        target = dna

        # ── Mutation ─────────────────────────────────────────────────
        mutated, mutations = mutate(
            dna, mutation_rate=0.05, indel_rate=0.01, rng=None
        )
        mut_count = len(mutations)
        mut_reasonable = mut_count > 0

        # ── Fitness ──────────────────────────────────────────────────
        fit_original = calculate_fitness(dna, target, method="hamming")
        fit_ok = abs(fit_original - 1.0) < 1e-9

        # Hamming fitness of mutated should be < 1.0
        fit_mutated = calculate_fitness(mutated, target, method="hamming")
        fit_mutated_lower = fit_mutated < 1.0

        # ── Wright-Fisher population ─────────────────────────────────
        cfg = EvolutionConfig(
            mutation_rate=0.01,
            indel_rate=0.001,
            population_size=50,
            generations=20,
            selection_coefficient=0.1,
        )
        pop = EvolutionaryPopulation(
            initial_dna=dna,
            config=cfg,
            target_dna=target,
            fitness_method="hamming",
        )
        pop.evolve(20)
        best = pop.best_individual()
        evo_ok = best is not None and best.fitness > 0
        stats = pop.get_generation_stats()

        all_ok = mut_reasonable and fit_ok and fit_mutated_lower and evo_ok

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS" if all_ok else "FAIL",
            "validation": {
                "mutations_produced": mut_reasonable,
                "fitness_original_is_one": fit_ok,
                "fitness_mutated_is_lower": fit_mutated_lower,
                "population_evolved": evo_ok,
            },
            "mutation": {
                "count": mut_count,
                "original_length": len(dna),
                "mutated_length": len(mutated),
            },
            "fitness": {
                "original": round(fit_original, 4),
                "mutated": round(fit_mutated, 4),
                "best_after_evolution": round(best.fitness, 4) if best else None,
            },
            "population": {
                "generations_run": pop.generation,
                "final_size": len(pop.individuals),
                "diversity": round(pop.get_diversity(), 4),
                "fitness_trajectory": [
                    {"gen": s["generation"], "mean": round(s["mean_fitness"], 4)}
                    for s in stats[::5]
                ] + [{"gen": stats[-1]["generation"],
                       "mean": round(stats[-1]["mean_fitness"], 4)}],
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
