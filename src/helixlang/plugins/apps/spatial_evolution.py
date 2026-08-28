"""Spatial range-expansion evolution: dual-loop adaptation on the lattice.

Design 1 of the population roadmap (doc/18-programmable-cell-population-simulation.md §13 Design 1): the outer loop
evolves a population of *DNA genotypes*; the inner loop evaluates each
genotype as a *spatial colonizer* on the :class:`~helixlang.plugins.runtime.population.
CellPopulation3D` lattice and scores it by the Bosshard et al. 2020
fitness proxy:

    fitness = colony_radius_sites * core_survival - metabolic_cost

Bosshard et al. 2020 (BMC Genomics 21:232) sequenced 10 mutator E. coli
lines over ~1650 generations of colony expansion on hard agar and found
beneficial mutations enriched in *transporter and membrane-structure
genes* (nutrient uptake) that left a heritable, spatially-expanding
phenotype.  Colony size is therefore the natural proxy of adaptation in
a range-expansion experiment, which is exactly the phenotype we evolve:

    - the inner spatial simulation reads the genotype's uptake and
      division modifiers (map of the genome onto transport/membrane
      traits),
    - faster uptake -> more divisions -> larger colony radius at the
      same tick budget,
    - fitness rewards both the radial extent (space captured) and the
      survival of the colony core (Bosshard's "core survival").

Genetics reuse the :mod:`helixlang.plugins.runtime.evolution` primitives verbatim
(``mutate``/``recombine``), so the mutation spectrum, transition bias
and recombination model are identical to the classic pipeline.

The default parameters are small enough to be unit-test friendly:
``generations=10``, ``population_size=10``, ``colonization_ticks=25``,
``inner_population_size=40`` runs in well under a second per generation.
"""
from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass

from helixlang.plugins.runtime.evolution import mutate, recombine
from helixlang.plugins.runtime.population import (
    CellPopulation3D,
    PopulationCell,
    PopulationConfig,
)

_NT_BITS = {"A": (0, 0), "C": (0, 1), "G": (1, 0), "T": (1, 1)}
_NT_BITS_UPPER = {
    **{k: v for k, v in _NT_BITS.items()},
    **{k.lower(): v for k, v in _NT_BITS.items()},
}


@dataclass(slots=True)
class SpatialEvolutionConfig:
    """Outer evolution loop + inner spatial colonization parameters.

    Args:
        generations: outer-loop generations.
        population_size: number of genotypes alive per generation.
        genome_length_nt: genotype length in nucleotides (3 nt = one
            codon; the two phenotype halves each span half the genome).
        substitution_rate: per-nt substitution probability (reused from
            :func:`helixlang.plugins.runtime.evolution.mutate`).
        indel_rate: per-nt indel probability (reused from
            :func:`helixlang.plugins.runtime.evolution.mutate`).
        recombination_rate: probability a child is a crossover of two
            selected parents (:func:`helixlang.plugins.runtime.evolution.recombine`).
        selection_fraction: truncation-selection fraction kept each
            generation.
        metabolic_cost: quadratic penalty on uptake gain, ``cost *
            (g_uptake - 1)^2``, so runaway substrate acquisition is
            selectively discouraged (Bosshard: over-expression carries a
            fitness cost).
        seed: master RNG seed (per-evaluation seeds are derived).
        grid_width, grid_height: inner lattice dimensions (µm-edge
            sites).
        colonization_ticks: inner colonization duration per evaluation.
        inner_population_size: cells seeded at the inner lattice centre.
        energy_intake: inner rich-medium ATP intake per tick, scaled by
            the genotype's uptake gain.
        base_division_threshold: inner division energy (ATP); the
            genotype's division gain lowers it.
        signaling: inner AI-2 signaling on/off.
    """

    generations: int = 10
    population_size: int = 10
    genome_length_nt: int = 30
    substitution_rate: float = 0.05
    indel_rate: float = 0.0
    recombination_rate: float = 0.2
    selection_fraction: float = 0.25
    metabolic_cost: float = 0.05
    seed: int | None = None
    grid_width: int = 24
    grid_height: int = 24
    colonization_ticks: int = 25
    inner_population_size: int = 40
    energy_intake: float = 5.0e7
    base_division_threshold: float = 1.8e9
    signaling: bool = True


def random_genome(length_nt: int,
                  rng: random.Random) -> str:
    """Uniform-random DNA genotype of the given length."""
    return "".join(rng.choice("ACGT") for _ in range(length_nt))


def _nt_bits(dna: str) -> list[int]:
    """Map each nucleotide onto 2 bits (A=00, C=01, G=10, T=11)."""
    bits: list[int] = []
    for ch in dna:
        b = _NT_BITS_UPPER[ch]
        bits.extend(b)
    return bits


def phenotype_of(genome: str) -> tuple[float, float]:
    """Genotype -> (uptake gain, division gain).

    The genome is decoded to a bit vector (2 bits/nt).  The first half
    of the bits drive the *nutrient uptake / membrane-transport* gain
    ``g_uptake`` in ``[0.8, 1.8]``, the second half drive the *division
    rate* gain ``g_division`` in ``[0.9, 1.1]``.  This is the model-level
    analogue of Bosshard et al. 2020, where adaptation fixes mutations in
    transport/membrane genes that raise the spatial growth rate.
    """
    bits = _nt_bits(genome)
    if not bits:
        return 1.0, 1.0
    mid = len(bits) // 2
    first = bits[:mid]
    second = bits[mid:]
    mean_first = sum(first) / len(first)
    mean_second = sum(second) / len(second)
    g_uptake = 0.8 + 1.0 * mean_first
    g_division = 0.9 + 0.2 * mean_second
    return g_uptake, g_division


def _seed_block(config: PopulationConfig,
                n: int) -> list[PopulationCell]:
    """Pack ``n`` cells as a centred colony block."""
    w, h = config.grid_width, config.grid_height
    side = 1
    while side * side < n:
        side += 1
    off_x = max(0, (w - side) // 2)
    off_y = max(0, (h - side) // 2)
    cells: list[PopulationCell] = []
    for i in range(n):
        cells.append(PopulationCell(
            id=i,
            x=min(off_x + i % side, w - 1),
            y=min(off_y + (i // side) % side, h - 1),
        ))
    return cells


def evaluate(genome: str,
             config: SpatialEvolutionConfig,
             seed: int | None = None) -> dict:
    """Run the inner spatial colonization and score the genotype.

    Returns a dict with ``radius_sites``, ``survival`` and ``fitness``.
    """
    g_uptake, g_division = phenotype_of(genome)
    inner = PopulationConfig(
        max_size=max(config.inner_population_size * 4, 100),
        grid_width=config.grid_width,
        grid_height=config.grid_height,
        division_threshold=config.base_division_threshold / g_division,
        energy_intake=config.energy_intake * g_uptake,
        signaling_enabled=config.signaling,
        mechanics="shoving",
    )
    cells = _seed_block(inner, config.inner_population_size)
    pop = CellPopulation3D(cells, inner, seed=seed)
    start = sum(1 for c in cells if c.alive)
    for _ in range(config.colonization_ticks):
        pop.step()
    alive = [c for c in pop.cells if c.alive]
    survival = len(alive) / max(1, start)
    if not alive:
        return {"radius_sites": 0.0, "survival": 0.0, "fitness": 0.0}
    cx = sum(c.x for c in alive) / len(alive)
    cy = sum(c.y for c in alive) / len(alive)
    radius = max(math.hypot(c.x - cx, c.y - cy) for c in alive)
    cost = config.metabolic_cost * (g_uptake - 1.0) ** 2
    fitness = radius * survival - cost
    return {
        "radius_sites": radius,
        "survival": survival,
        "fitness": max(0.0, fitness),
    }


class SpatialEvolution:
    """Dual-loop spatial range-expansion evolution.

    The outer Wright-Fisher-style loop truncation-selects the top
    ``selection_fraction`` genotypes, then reproduces the population
    from them with mutation (and optional recombination).  Every child's
    fitness is re-measured by its own spatial colonization run on the
    next generation, so selection acts on the *spatial phenotype* rather
    than on the genome directly -- the selective signal that Bosshard et
    al. 2020 measured at the population scale.
    """

    def __init__(self, config: SpatialEvolutionConfig | None = None):
        self.config = config or SpatialEvolutionConfig()
        self.rng = random.Random(self.config.seed)
        self.generation = 0
        self.population: list[str] = []
        self.history: list[dict[str, float]] = []
        self._best: dict[str, float] = {"mean_fitness": 0.0, "max_fitness": 0.0}

    # -------- generation life-cycle --------

    def _evaluate(self, genomes: list[str]) -> list[dict]:
        # The inner-sim seed is derived from the *genome* (plus a fixed
        # master offset), not the generation, so a genotype receives the
        # same random stream whenever it is scored: the fitness landscape
        # is stationary and selection heritable (an offspring scored next
        # generation competes under identical environmental draws).
        base = (self.config.seed if self.config.seed is not None else 0)
        evals: list[dict] = []
        for genome in genomes:
            seed = base + zlib.crc32(genome.encode("ascii")) % (2 ** 31)
            evals.append(evaluate(genome, self.config, seed=seed))
        return evals

    def _next_generation(self, scored: list[tuple[str, dict]]) -> None:
        cfg = self.config
        n_keep = max(1, int(cfg.population_size * cfg.selection_fraction))
        ranked = sorted(scored, key=lambda item: item[1]["fitness"],
                        reverse=True)
        elites = [g for g, _ in ranked[:n_keep]]
        # fill the population from the selected (fittest) genotypes
        offspring: list[str] = []
        while len(offspring) < cfg.population_size:
            parent = self.rng.choice(elites)
            child = parent
            if (cfg.recombination_rate > 0.0
                    and self.rng.random() < cfg.recombination_rate
                    and len(elites) > 1):
                other = self.rng.choice(elites)
                child = recombine(parent, other, 1.0, self.rng)
            child, _ = mutate(
                child,
                mutation_rate=cfg.substitution_rate,
                indel_rate=cfg.indel_rate,
                rng=self.rng,
            )
            # keep genotype length fixed
            if len(child) != cfg.genome_length_nt:
                child = (child + "A" * cfg.genome_length_nt)[
                    :cfg.genome_length_nt]
            offspring.append(child)
        self.population = offspring

    def step(self) -> dict[str, float]:
        """Evolve one generation; returns the per-generation row."""
        if not self.population:
            self.population = [
                random_genome(self.config.genome_length_nt, self.rng)
                for _ in range(self.config.population_size)
            ]
        evals = self._evaluate(self.population)
        scored = list(zip(self.population, evals, strict=True))
        mean_fitness = sum(e["fitness"] for e in evals) / max(1, len(evals))
        max_fitness = max(e["fitness"] for e in evals)
        best = max(scored, key=lambda item: item[1]["fitness"])[1]
        row = {
            "generation": self.generation,
            "mean_fitness": mean_fitness,
            "max_fitness": max_fitness,
            "best_radius_sites": best["radius_sites"],
            "best_survival": best["survival"],
            "mean_uptake_gain": sum(
                phenotype_of(g)[0] for g in self.population)
                / max(1, len(self.population)),
        }
        self.history.append(row)
        self._best = row
        self._next_generation(scored)
        self.generation += 1
        return row

    def run(self) -> list[dict[str, float]]:
        """Run ``generations`` full generations and return the history."""
        for _ in range(self.config.generations):
            self.step()
        return self.history

    # -------- queries --------

    def mean_fitness(self) -> float:
        return self._best["mean_fitness"]

    def max_fitness(self) -> float:
        return self._best["max_fitness"]

    def best_genome(self) -> str:
        """Fittest genome of the most recently scored generation."""
        cfg = self.config
        base = (cfg.seed if cfg.seed is not None else 0)
        best: tuple[str, dict] | None = None
        for genome in self.population:
            seed = base + zlib.crc32(genome.encode("ascii")) % (2 ** 31)
            ev = evaluate(genome, cfg, seed=seed)
            if best is None or ev["fitness"] > best[1]["fitness"]:
                best = (genome, ev)
        return best[0] if best else ""
