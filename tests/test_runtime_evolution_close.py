"""Branch-closure tests for helixlang.plugins.runtime.evolution.

Drives the remaining uncovered lines/branches: the numpy-optional
fallbacks, batch-mutation edge cases, recombination guards, fitness
method edge cases, and the remaining dN/dS interpretation buckets.
"""
from __future__ import annotations

import importlib
import random
import sys

import pytest

import helixlang.plugins.runtime.evolution as evolution
from helixlang.plugins.runtime.evolution import (
    EvolutionaryPopulation,
    EvolutionConfig,
    Individual,
    _aa_of_codon,
    calculate_fitness,
    dnds_codeml,
    dnds_ratio,
    fitness_landscape,
    mutate_batch,
    recombine,
    select,
)


class TestNumpyOptionalFallbacks:
    def test_module_loads_without_numpy(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "numpy", None)
        mod = importlib.reload(evolution)
        assert mod._HAS_NUMPY is False
        assert mod.mutate_batch(
            [mod.Individual("ACGT", 0.5, 0)], 0.1, 0.1, 3.0, random.Random(0)
        )
        monkeypatch.undo()
        importlib.reload(evolution)
        assert evolution._HAS_NUMPY is True

    def test_mutate_batch_pure_python_fallback(self, monkeypatch) -> None:
        monkeypatch.setattr(evolution, "_HAS_NUMPY", False)
        inds = [Individual("ACGT", 0.5, 0), Individual("ACGT", 0.5, 0)]
        out = mutate_batch(inds, 0.5, 0.5, 3.0, random.Random(1))
        assert len(out) == 2

    def test_select_pure_python_fallback_clamps(self, monkeypatch) -> None:
        monkeypatch.setattr(evolution, "_HAS_NUMPY", False)
        inds = [Individual("A" * 10, f, 0) for f in (1.0, 0.0, 0.0, 0.0)]
        sel = select(inds, 4.0, random.Random(0))
        assert len(sel) == 4

    def test_hamming_pure_python_loop(self, monkeypatch) -> None:
        monkeypatch.setattr(evolution, "_HAS_NUMPY", False)
        assert calculate_fitness("ACGT", "ACGT", method="hamming") == 1.0


class TestMutateBatchEdges:
    def _ind(self, dna: str) -> Individual:
        return Individual(dna, 0.5, 0)

    def test_empty_population(self) -> None:
        assert mutate_batch([], 0.1, 0.1, 3.0, random.Random(0)) == []

    def test_all_empty_dnas(self) -> None:
        out = mutate_batch(
            [self._ind(""), self._ind("")], 0.1, 0.1, 3.0, random.Random(0)
        )
        assert out == [("", []), ("", [])]

    def test_empty_dna_in_numpy_path(self) -> None:
        out = mutate_batch(
            [self._ind(""), self._ind("ACGT")],
            0.5, 0.5, 3.0, random.Random(3),
        )
        assert out[0] == ("", [])

    def test_numpy_path_insertions_and_deletions(self) -> None:
        inds = [self._ind("ACGT" * 30)]
        new_dna, muts = mutate_batch(
            inds, 0.01, 0.9, 3.0, random.Random(5)
        )[0]
        assert any(m.startswith("ins@") for m in muts)
        assert any(m.startswith("del@") for m in muts)
        assert new_dna


class TestRecombineEdges:
    def test_default_rng(self) -> None:
        assert recombine("ACGT", "ACGT", 0.0) == "ACGT"

    def test_empty_parent1(self) -> None:
        assert recombine("", "ACGT", 1.0) == "ACGT"

    def test_empty_parent2(self) -> None:
        assert recombine("ACGT", "", 1.0) == "ACGT"

    def test_short_parents_no_crossover(self) -> None:
        assert recombine("A", "C", 1.0) == "A"


class TestCalculateFitnessEdges:
    def test_oracle_method(self) -> None:
        f = calculate_fitness(
            "ATGGCGCCG", "ATGGCGCCG",
            method="oracle", oracle="blosum62",
        )
        assert isinstance(f, float)

    def test_oracle_requires_target(self) -> None:
        with pytest.raises(ValueError):
            calculate_fitness("ATGGCGCCG", method="oracle", oracle="blosum62")

    def test_both_empty_hamming(self) -> None:
        assert calculate_fitness("", "", method="hamming") == 1.0

    def test_one_empty_hamming_python_path(self) -> None:
        assert calculate_fitness("", "ACGT", method="hamming") == 0.0

    def test_empty_gc(self) -> None:
        assert calculate_fitness("", method="gc") == 0.0


class TestFitnessLandscapeEdges:
    def test_empty_dna(self) -> None:
        assert fitness_landscape("") == {}

    def test_default_target(self) -> None:
        landscape = fitness_landscape("ACGT")
        assert set(landscape) == {0, 1, 2, 3}

    def test_out_of_range_positions_skipped(self) -> None:
        landscape = fitness_landscape("ACGT", positions=[0, 5, -3])
        assert set(landscape) == {0}


class TestDndsBranches:
    def test_aa_of_codon_unknown(self) -> None:
        assert _aa_of_codon("NNN") == "X"

    def test_dnds_weak_purifying(self) -> None:
        result = dnds_ratio("GTCACGTACCGA", "GTTAAGTGTCGA")
        assert result["interpretation"] == "weak purifying selection"

    def test_dnds_positive(self) -> None:
        result = dnds_ratio("CAGCCTCCCTCA", "CGGGGTGCATCG")
        assert result["interpretation"] == "positive selection"

    def test_codeml_ambiguous_codon_skipped(self) -> None:
        result = dnds_codeml("ATGNNNTAA", "ATGAAATAA")
        assert result["dNdS"] >= 0.0

    def test_codeml_multiposition_pair_fallback(self) -> None:
        result = dnds_codeml("CCC", "AAA")
        assert result["method"] == "M0"

    def test_codeml_weak_purifying(self) -> None:
        result = dnds_codeml("ATAACAACGATAGCA", "CTAAAGACAATTACA")
        assert "weak purifying" in result["interpretation"]

    def test_codeml_neutral(self) -> None:
        result = dnds_codeml("TACTCGTTGGCGTATTGT", "AACTTGTTGGCCCAGTGT")
        assert "neutral" in result["interpretation"]

    def test_codeml_positive_lrt_significant(self) -> None:
        result = dnds_codeml("CTAGGTGAGTCATGGAAA", "GGAGTGGCCACTCGGACA")
        assert "positive selection" in result["interpretation"]
        assert "LRT significant" in result["interpretation"]


class TestEvolutionaryPopulationEdges:
    def _empty_pop(self) -> EvolutionaryPopulation:
        return EvolutionaryPopulation(
            "ACGT",
            config=EvolutionConfig(population_size=0),
            fitness_method="gc",
        )

    def test_empty_population_records_stats(self) -> None:
        pop = self._empty_pop()
        assert pop.history[-1]["population_size"] == 0
        assert pop.mean_fitness() == 0.0
        assert pop.best_individual() is None
        assert pop.get_fitness_landscape() == {}

    def test_odd_population_recombination_residual(self) -> None:
        pop = EvolutionaryPopulation(
            "ACGTACGT",
            config=EvolutionConfig(
                population_size=3,
                recombination_rate=1.0,
                mutation_rate=0.0,
            ),
            target_dna="ACGTACGT",
            fitness_method="hamming",
            rng=random.Random(0),
        )
        pop.step()
        assert len(pop.individuals) == 3
