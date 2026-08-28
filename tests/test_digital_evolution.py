"""Digital evolution tests: Avida-style signal-task selection (S3).

Verification goals:
- Instruction genomes execute to a bit sequence; fitness is the
  exponential of the number of target bits reproduced (multiplicative
  landscape).
- Selection (Wright-Fisher + mutation) increases population fitness over
  generations, reproducing the small-effect beneficial-mutation dynamics
  of digital organisms (Lenski 2003 Nature 423:139; Ofria & Wilke 2004).
- A pure-drift control does not improve fitness (neutral evolution).
- The empirical mutation spectrum matches the configured per-instruction
  substitution/insertion/deletion rates.
- Beyond a critical mutation rate the population undergoes an error
  catastrophe (Eigen 1971), so mean fitness is lower than at an
  intermediate rate.
"""
from __future__ import annotations

import random

from helixlang.plugins.apps.digital_evolution import (
    DigitalEvolution,
    DigitalEvolutionConfig,
    execute,
    fitness_of,
    mutate_genome,
)

TARGET = (1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1)


def test_execute_and_fitness() -> None:
    # a genome that emits the target one bit at a time
    emitter = []
    for bit in TARGET:
        emitter.append("MOV1" if bit else "MOV0")
        emitter.append("OUT")
    assert execute(tuple(emitter)) == list(TARGET)
    assert fitness_of(tuple(emitter), TARGET) == 2 ** len(TARGET)
    # a genome emitting only the first bit reproduces 1 bit
    assert fitness_of(("MOV1", "OUT"), TARGET) == 2.0
    # a genome emitting nothing reproduces 0 bits
    assert fitness_of(("HALT",), TARGET) == 1.0


def test_selection_increases_fitness() -> None:
    cfg = DigitalEvolutionConfig(
        population_size=200,
        genome_length=12,
        target=TARGET,
        substitution_rate=0.01,
        insertion_rate=0.001,
        deletion_rate=0.001,
        generations=300,
        seed=3,
    )
    evo = DigitalEvolution(cfg)
    initial_max = evo.max_fitness()
    initial_mean = evo.mean_fitness()
    evo.run()
    # the fittest organism strictly improves (at least one more correct bit)
    assert evo.max_fitness() > initial_max
    assert evo.mean_fitness() > initial_mean
    assert evo.history[-1]["max_fitness"] == evo.max_fitness()


def test_neutral_drift_does_not_improve() -> None:
    cfg = DigitalEvolutionConfig(
        population_size=200,
        genome_length=12,
        target=TARGET,
        substitution_rate=0.01,
        insertion_rate=0.001,
        deletion_rate=0.001,
        selection_enabled=False,
        generations=200,
        seed=3,
    )
    evo = DigitalEvolution(cfg)
    initial_mean = evo.mean_fitness()
    evo.run()
    # drift alone must not systematically improve the mean fitness
    assert evo.mean_fitness() < 2.0 * initial_mean
    # and it stays far below a selected population on the same task
    selected = DigitalEvolutionConfig(
        population_size=200,
        genome_length=12,
        target=TARGET,
        substitution_rate=0.01,
        insertion_rate=0.001,
        deletion_rate=0.001,
        generations=200,
        seed=3,
    )
    evo_sel = DigitalEvolution(selected)
    evo_sel.run()
    assert evo_sel.mean_fitness() > evo.mean_fitness()


def test_mutation_spectrum_matches_rates() -> None:
    # each event type is measured in isolation so indel-induced frame
    # shifts do not inflate the substitution count
    length = 12
    subs = DigitalEvolutionConfig(
        genome_length=length, substitution_rate=0.05,
        insertion_rate=0.0, deletion_rate=0.0, seed=11,
    )
    ins = DigitalEvolutionConfig(
        genome_length=length, substitution_rate=0.0,
        insertion_rate=0.02, deletion_rate=0.0, seed=11,
    )
    dels = DigitalEvolutionConfig(
        genome_length=length, substitution_rate=0.0,
        insertion_rate=0.0, deletion_rate=0.05, seed=11,
    )
    spec_s = DigitalEvolution(subs).substitution_spectrum(samples=500)
    spec_i = DigitalEvolution(ins).substitution_spectrum(samples=500)
    spec_d = DigitalEvolution(dels).substitution_spectrum(samples=2000)
    assert abs(spec_s["substitution"] - length * 0.05) < length * 0.02
    assert abs(spec_i["insertion"] - length * 0.02) < length * 0.015
    assert abs(spec_d["deletion"] - 0.05) < 0.02


def test_mutate_genome_changes_length_and_content() -> None:
    rng = random.Random(1)
    genome = ("MOV1", "OUT", "NOP", "HALT")
    changed = False
    for _ in range(2000):
        mutant = mutate_genome(genome, 0.0, 0.5, 0.5, rng)
        if mutant != genome:
            changed = True
            break
    assert changed


def test_error_catastrophe_at_high_mutation() -> None:
    low = DigitalEvolutionConfig(
        population_size=150,
        genome_length=12,
        target=TARGET,
        substitution_rate=0.02,
        insertion_rate=0.0,
        deletion_rate=0.0,
        generations=200,
        seed=5,
    )
    high = DigitalEvolutionConfig(
        population_size=150,
        genome_length=12,
        target=TARGET,
        substitution_rate=0.6,
        insertion_rate=0.0,
        deletion_rate=0.0,
        generations=200,
        seed=5,
    )
    evo_low = DigitalEvolution(low)
    evo_low.run()
    evo_high = DigitalEvolution(high)
    evo_high.run()
    # the intermediate-rate population adapts; the high-rate population
    # sits at the random baseline (error catastrophe, Eigen 1971)
    assert evo_low.mean_fitness() > 2.0 * evo_high.mean_fitness()
