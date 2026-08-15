"""Spatial range-expansion evolution tests (doc/18-programmable-cell-population-simulation.md §13 Design 1).

Verification goals (Bosshard et al. 2020, BMC Genomics 21:232):
- The genotype -> phenotype map places the first half of the genome's
  bits onto the nutrient-uptake / membrane-transport gain and the second
  half onto the division-rate gain (the trait classes that adapt during
  E. coli range expansion).
- Colony fitness is monotone in uptake gain: faster uptake -> more
  divisions -> a larger colony radius at the same tick budget, and the
  cost term keeps runaway uptake selected against.
- Dual-loop selection (truncation + mutation/recombination on the outer
  loop, spatial colonization score on the inner loop) raises mean and
  max fitness over generations.
- The inner-sim seed is genome-derived, so the fitness landscape is
  stationary: a genotype always receives the same random stream.
"""
from __future__ import annotations

import random

from helixlang.apps.spatial_evolution import (
    SpatialEvolution,
    SpatialEvolutionConfig,
    evaluate,
    phenotype_of,
    random_genome,
)
from helixlang.evolution import mutate


def test_phenotype_map_ranges() -> None:
    # all-A genome: first-half bits all 0 -> g_uptake floor 0.8
    low, _ = phenotype_of("A" * 30)
    # all-T genome: first-half bits all 1 -> g_uptake ceiling 1.8
    high, _ = phenotype_of("T" * 30)
    assert low == 0.8
    assert high == 1.8
    # second half drives the division gain only
    g_up, g_div = phenotype_of(("T" * 15) + ("A" * 15))
    assert g_up == 1.8
    assert g_div == 0.9


def test_fitness_monotone_in_uptake_gain() -> None:
    cfg = SpatialEvolutionConfig(seed=1)
    lo = evaluate("A" * 30, cfg, seed=9)
    hi = evaluate("T" * 30, cfg, seed=9)
    assert hi["fitness"] > lo["fitness"]
    assert hi["radius_sites"] >= lo["radius_sites"]


def test_stationary_landscape() -> None:
    """The same genome scored twice gets the same fitness (genome-derived
    inner-sim seed, so selection is heritable across generations)."""
    cfg = SpatialEvolutionConfig(seed=7)
    genome = random_genome(30, random.Random(3))
    a = evaluate(genome, cfg, seed=None)
    b = evaluate(genome, cfg, seed=None)
    assert a["fitness"] == b["fitness"]


def test_selection_raises_fitness() -> None:
    cfg = SpatialEvolutionConfig(
        generations=15,
        population_size=12,
        genome_length_nt=30,
        substitution_rate=0.03,
        seed=42,
    )
    evo = SpatialEvolution(cfg)
    rows = evo.run()
    assert len(rows) == cfg.generations
    first, last = rows[0], rows[-1]
    assert last["mean_fitness"] > first["mean_fitness"]
    assert last["max_fitness"] >= first["max_fitness"]
    # the population's mean uptake gain drifts up as transport/membrane
    # gain is selected (Bosshard 2020)
    assert last["mean_uptake_gain"] >= first["mean_uptake_gain"]


def test_deterministic_with_seed() -> None:
    def run() -> list[dict[str, float]]:
        cfg = SpatialEvolutionConfig(generations=6, population_size=8,
                                     seed=11)
        return SpatialEvolution(cfg).run()

    assert run() == run()


def test_mutation_reuses_evolution_primitive() -> None:
    rng = random.Random(5)
    genome = random_genome(30, rng)
    child, events = mutate(genome, mutation_rate=0.5, rng=rng)
    assert events or child != genome or len(genome) != 30
