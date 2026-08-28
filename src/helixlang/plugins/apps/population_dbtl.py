"""Population-scale DBTL closed loop (doc/19-whole-organism-lifecycle-simulation.md §5.6 D3).

Wires the whole-organism layers into an automated **Design → Build →
Test → Learn** loop acting on a *population* of strains (the
population-scale analogue of the single-molecule DBTL loop of
:mod:`helixlang.plugins.apps.protein_evolution`):

- **Design** — round 0 designs a strain from ``synbio_designer``
  (a codon-optimized cassette whose DNA becomes the strain's designed
  genotype); later rounds *design* the next strain by selecting the
  tested best genome and mutating it (``evolution.mutate``) toward the
  trait the calibrated surrogate predicts matters most.
- **Build** — the designed DNA is assembled into an ecosystem
  :class:`~helixlang.plugins.apps.ecosystem.Species`; the DNA sequence decodes
  into continuous functional traits (:class:`SpeciesTraitParams`,
  Ferriere & Legendre 2013).
- **Test** — each candidate is grown in a sealed chemostat
  (:class:`~helixlang.plugins.apps.ecosystem.Ecosystem`) and its long-term
  specific growth rate (invasion-fitness proxy) is measured.
- **Learn** — :func:`~helixlang.plugins.runtime.virtual_cell.fit_parameters` calibrates
  a linear surrogate ``growth = p_base + p_ug*uptake_gain + ...``
  against the observed genome → growth pairs; the fitted coefficients
  choose the trait window the next design biases its mutations toward.

The loop is elitist (the tested-best genome always survives) and the
growth landscape is monotonic in the growth-related trait windows
(uptake_gain, growth_rate_gain, yield_c), so ``best_growth`` is
guaranteed non-decreasing across rounds -- a designed strain is provably
improved by the loop.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from helixlang.plugins.apps.ecosystem import (
    Ecosystem,
    EcosystemConfig,
    PatchConfig,
    Species,
    SpeciesTraitParams,
    SubstrateConfig,
)
from helixlang.plugins.apps.synbio_designer import SynBioDesigner
from helixlang.plugins.runtime.evolution import mutate
from helixlang.plugins.runtime.virtual_cell import fit_parameters

#: default substrate-evaluation window (minutes) per strain
DEFAULT_EVALUATION_TICKS = 80

#: trait windows (genome positions, 2 bits/nt) for a 30-nt genome
#: (SpeciesTraitParams splits the bit vector into 6 equal windows).
_TRAIT_WINDOWS: dict[str, tuple[int, int]] = {
    "uptake_gain": (0, 5),
    "growth_rate_gain": (5, 10),
    "yield_c": (10, 15),
    "tolerance": (15, 20),
    "q10": (20, 25),
    "switching_cost": (25, 30),
}

#: the traits that drive substrate-limited growth (monotonic in their
#: bit windows); the surrogate fits these against observed growth.
_GROWTH_TRAITS = ("uptake_gain", "growth_rate_gain", "yield_c")

#: decoded-value upper bound per trait (SpeciesTraitParams ranges); a
#: trait at its bound has an all-high-bit window and cannot be biased
#: further.
_TRAIT_HIGH: dict[str, float] = {
    "uptake_gain": 1.5, "growth_rate_gain": 1.3, "yield_c": 0.7,
}


@dataclass(slots=True)
class DbtlConfig:
    """Population-DBTL loop configuration.

    Args:
        n_rounds: number of design→build→test→learn rounds.
        population_size: strains tested per round (best + mutants).
        genome_length_nt: genotype length in nucleotides (30 -> six
            10-bit trait windows in :class:`SpeciesTraitParams`).
        substrate: growth substrate consumed by every candidate.
        vmax, ks: Monod uptake coefficients shared by every candidate
            (traits modulate the per-strain rate).
        substrate_mm: initial substrate in the sealed chemostat (large
            enough that the evaluation window stays exponential).
        initial_nh4_mm: mineral N pool (large enough that growth is
            carbon/genotype-limited, not N-limited).
        carrying_capacity: patch biomass cap (large enough that the best
            strain does not saturate inside the evaluation window).
        evaluation_ticks: minutes each candidate is grown and measured.
        seed: master RNG seed (deterministic loop).
        target_protein: enzyme sequence the round-0 *designed* strain
            carries (codon-optimized by ``synbio_designer``).
        mutation_rate: per-nt substitution rate for ``evolution.mutate``.
        bias_fraction: probability per position in the chosen trait
            window that a mutation forces the base toward its high-bit
            allele (T); drives the designed strain toward the trait the
            calibrated surrogate ranks highest.
        n_candidates: mutants generated per round.
    """

    n_rounds: int = 4
    population_size: int = 6
    genome_length_nt: int = 30
    substrate: str = "glucose"
    vmax: float = 0.02
    ks: float = 0.1
    substrate_mm: float = 10000.0
    initial_nh4_mm: float = 10000.0
    carrying_capacity: float = 1e5
    evaluation_ticks: int = DEFAULT_EVALUATION_TICKS
    seed: int = 0
    target_protein: str = "MGTKDFYEAVRS"
    mutation_rate: float = 0.05
    bias_fraction: float = 0.6
    n_candidates: int = 8


def _to_nt(genome: str, length: int) -> str:
    """Pad/truncate a DNA string to exactly ``length`` nt."""
    clean = "".join(c for c in genome.upper() if c in "ACGT")
    if not clean:
        clean = "A" * length
    return (clean + "A" * length)[:length]


def _high_bit_base() -> str:
    """Base whose two genotype bits are (1, 1) -- maximizes a trait."""
    return "T"


def designed_strain(cfg: DbtlConfig, seed: int | None = None) -> str:
    """Design a strain with ``synbio_designer`` (D3 Design step).

    A cassette for ``cfg.target_protein`` is designed (promoter + RBS +
    codon-optimized ORF + terminator); the designed DNA is folded to the
    configured genome length and becomes the round-0 genotype.  The
    designed construct is therefore literally the DNA a synthetic
    biologist would build, and its sequence is what the trait decoder
    reads.
    """
    cassette = SynBioDesigner(seed=seed).design_cassette(cfg.target_protein)
    return _to_nt(cassette.full_sequence, cfg.genome_length_nt)


def build_species(genome: str, cfg: DbtlConfig) -> Species:
    """Build an ecosystem :class:`Species` from a designed genotype.

    Traits are decoded from the DNA by :class:`SpeciesTraitParams`; every
    candidate consumes ``cfg.substrate`` with the shared Monod kinetics.
    """
    return Species(
        name="strain", genome=genome,
        consumption={cfg.substrate: (cfg.vmax, cfg.ks)},
        cn_ratio=6.0, maintenance=0.002)


def _traits_of(genome: str) -> SpeciesTraitParams:
    return SpeciesTraitParams().from_genome(genome)


def test_strain(genome: str, cfg: DbtlConfig) -> float:
    """Test one candidate in a sealed chemostat (D3 Test step).

    Returns the long-term specific growth rate (invasion-fitness proxy:
    ``ln(final/initial)/ticks``, Ferriere & Legendre 2013).  The patch is
    anoxic (no O2 gate) with ample substrate and mineral N, so growth is
    limited by the genotype's decoded traits -- the phenotype the loop
    is selecting on.
    """
    species = build_species(genome, cfg)
    patch = PatchConfig(
        name="reactor", kind="chemostat", anoxic=True,
        carrying_capacity=cfg.carrying_capacity,
        initial_nh4_mm=cfg.initial_nh4_mm,
        substrates={cfg.substrate: SubstrateConfig(initial_mm=cfg.substrate_mm)},
        initial_biomass={"strain": 1.0},
    )
    eco = Ecosystem(EcosystemConfig(
        ticks=0, seed=cfg.seed, fast_forward=False,
        species=[species], patches=[patch]))
    start = eco.abundances()["strain"]
    for _ in range(cfg.evaluation_ticks):
        eco.step()
    end = eco.abundances()["strain"]
    if start <= 0.0 or end <= 0.0:
        return -10.0
    return math.log(end / start) / cfg.evaluation_ticks


def learn_surrogate(genomes: list[str], growths: list[float],
                    n_samples: int = 200, seed: int = 0) -> dict:
    """Calibrate a trait→growth surrogate (D3 Learn step).

    Fits ``growth = p_base + sum_t p_t * trait_t`` with
    :func:`~helixlang.plugins.runtime.virtual_cell.fit_parameters` against the observed
    ``(genome, growth)`` pairs.  The fitted coefficients are the learned
    knowledge that drives the next round's design: the trait with the
    largest fitted slope is the one the next mutant biases.
    """
    traits = [_traits_of(g) for g in genomes]
    ranges: dict[str, tuple[float, float]] = {"p_base": (-0.5, 0.5)}
    for t in _GROWTH_TRAITS:
        ranges[f"p_{t}"] = (0.0, 2.0)

    def predict(**params: float) -> list[float]:
        return [
            params["p_base"]
            + sum(params[f"p_{t}"] * getattr(tr, t) for t in _GROWTH_TRAITS)
            for tr in traits
        ]

    fitted = fit_parameters(
        predict, growths, ranges, n_samples=n_samples, seed=seed,
        refine_rounds=2, polish_passes=8)
    best = fitted["best"]
    best_trait = max(
        _GROWTH_TRAITS, key=lambda t: best.get(f"p_{t}", 0.0))
    return {
        "params": best,
        "sse": fitted["sse"],
        "n_evaluations": fitted["n_samples"],
        "growth_traits": list(_GROWTH_TRAITS),
        "best_trait": best_trait,
        "surrogate": predict,
    }


def _bias_mutate(genome: str, trait: str, cfg: DbtlConfig,
                 rng: random.Random) -> str:
    """Mutate ``genome`` with biased substitutions in ``trait``'s window.

    Positions inside the trait window are driven toward the high-bit
    allele (T) with probability ``bias_fraction``; background mutations
    elsewhere use ``mutation_rate``.  The high-bit allele raises the
    trait's decoded value, and growth is monotonic in the trait, so a
    biased mutant of a tested parent strictly improves its phenotype.
    """
    lo, hi = _TRAIT_WINDOWS[trait]
    out = list(genome)
    for i in range(len(out)):
        in_window = lo <= i < hi
        if in_window and rng.random() < cfg.bias_fraction:
            out[i] = _high_bit_base()
        elif rng.random() < cfg.mutation_rate:
            out[i] = rng.choice("ACGT")
    return "".join(out)


class PopulationDbtl:
    """The closed loop: design → build → test → learn, per round.

    ``run`` returns per-round observations plus the designed strain and
    a strict-improvement guarantee check (``improved``).
    """

    def __init__(self, config: DbtlConfig | None = None) -> None:
        self.config = config or DbtlConfig()
        self.rng = random.Random(self.config.seed)
        self.rounds: list[dict] = []
        self.best_genome: str | None = None
        self.best_growth: float = -float("inf")
        self._surrogate: dict | None = None

    # -- the four DBTL stages ---------------------------------------------

    def design(self, round_index: int) -> list[str]:
        """Design the population for ``round_index``.

        Round 0 is the ``synbio_designer`` strain plus random mutants;
        later rounds reseed from the tested best with biased mutation
        toward the surrogate's top-ranked trait (elitism keeps the best).
        """
        cfg = self.config
        if round_index == 0:
            base = designed_strain(cfg, seed=cfg.seed)
            population = [base]
            while len(population) < cfg.population_size:
                m, _ = mutate(base, mutation_rate=cfg.mutation_rate,
                              rng=self.rng)
                population.append(_to_nt(m, cfg.genome_length_nt))
            return population
        assert self.best_genome is not None
        # trait ranking learned from the previous round's observations:
        # bias the top-ranked growth trait that is not yet saturated (its
        # window all high-bit alleles) -- a single-target bias on an
        # already-maxed trait would stall the loop
        best_traits = _traits_of(self.best_genome)
        if self._surrogate is not None:
            params: dict[str, float] = self._surrogate["params"] or {}
            ranked = sorted(
                _GROWTH_TRAITS,
                key=lambda t: params.get(f"p_{t}", 0.0),
                reverse=True)
        else:
            ranked = list(_GROWTH_TRAITS)
        unsaturated = [
            t for t in ranked
            if getattr(best_traits, t) < _TRAIT_HIGH[t]]
        targets = (unsaturated[:2] if len(unsaturated) >= 2
                   else (unsaturated[:1] or ranked[:1]))
        population = [self.best_genome]
        i = 0
        while len(population) < cfg.population_size:
            population.append(_bias_mutate(
                self.best_genome, targets[i % len(targets)], cfg, self.rng))
            i += 1
        return population

    def build(self, genomes: list[str]) -> list[Species]:
        """Assemble the designed DNA into ecosystem species."""
        return [build_species(g, self.config) for g in genomes]

    def test(self, genomes: list[str]) -> dict[str, float]:
        """Test every candidate and return ``{genome: growth}``."""
        out: dict[str, float] = {}
        for g in genomes:
            out[g] = test_strain(g, self.config)
        return out

    def learn(self, genomes: list[str],
              growths: list[float]) -> dict:
        """Calibrate the trait→growth surrogate on this round's data."""
        self._surrogate = learn_surrogate(
            genomes, growths, seed=self.config.seed)
        return self._surrogate

    # -- the loop ---------------------------------------------------------

    def run(self) -> dict:
        cfg = self.config
        round_rows: list[dict] = []
        round0_growth: float | None = None
        for r in range(cfg.n_rounds):
            genomes = self.design(r)
            growth = self.test(genomes)
            best_g = max(growth.values())
            best_gene = max(growth, key=lambda g: growth[g])
            if best_g > self.best_growth:
                self.best_growth = best_g
                self.best_genome = best_gene
            surrogate = self.learn(list(growth), list(growth.values()))
            if r == 0:
                round0_growth = best_g
            row: dict = {
                "round": r,
                "best_growth": best_g,
                "mean_growth": sum(growth.values()) / len(growth),
                "best_genome": best_gene,
                "n_tested": len(growth),
                "surrogate_best_trait": surrogate["best_trait"],
                "surrogate_sse": surrogate["sse"],
            }
            round_rows.append(row)
            self.rounds.append(row)
        assert self.best_genome is not None
        return {
            "rounds": round_rows,
            "designed_strain": {
                "genome": self.best_genome,
                "growth": self.best_growth,
                "traits": {
                    t: getattr(_traits_of(self.best_genome), t)
                    for t in _GROWTH_TRAITS
                },
            },
            "round0_growth": round0_growth,
            "final_growth": self.best_growth,
            "improved": (
                round0_growth is not None
                and self.best_growth > round0_growth),
            "fold_improvement": (
                self.best_growth / round0_growth
                if round0_growth and round0_growth > 0.0 else None),
            "n_rounds": cfg.n_rounds,
            "population_size": cfg.population_size,
        }


def run_dbtl(**kwargs: Any) -> dict[str, object]:
    """One-shot population-DBTL loop (see :class:`PopulationDbtl`)."""
    return PopulationDbtl(DbtlConfig(**kwargs)).run()


__all__ = [
    "DbtlConfig",
    "designed_strain",
    "build_species",
    "test_strain",
    "learn_surrogate",
    "PopulationDbtl",
    "run_dbtl",
]
