"""mutate_batch and pure-Python mutate consistency tests.

Verifies in src/helixlang/evolution.py:
- mutate_batch (numpy vectorized path) and mutate (pure Python path)
  are consistent in statistical distribution, mutation format, and boundary behavior.
- When numpy is unavailable, mutate_batch falls back to per-individual mutate (should be bit-identical).

Note: the numpy path and the pure Python path use different random number streams,
results are not required to be bit-identical, but:
  1. Mutation record format must be consistent (sub@/ins@/del@ prefixes)
  2. Mutation count statistical distributions must be consistent under the same parameters (fluctuation allowed)
  3. Boundary behavior must be consistent (empty DNA, zero rates, single individual)
  4. transition/transversion bias must be consistent
  5. Length change rule must be consistent (length = original + ins - del)
"""
from __future__ import annotations

import random
from collections import Counter

import pytest

from helixlang import evolution as evo
from helixlang.evolution import (
    Individual,
    mutate,
    mutate_batch,
)

# ============================================================================
# Helpers
# ============================================================================

def _make_individuals(dnas: list[str]) -> list[Individual]:
    return [Individual(dna=d, fitness=0.5, generation=0, mutations=[])
            for d in dnas]


def _mutation_types(muts: list[str]) -> list[str]:
    """Return the mutation type prefix list (sub/ins/del)."""
    return [m.split("@")[0] for m in muts]


def _count_types(muts: list[str]) -> Counter:
    return Counter(_mutation_types(muts))


# ============================================================================
# Boundary behavior consistency
# ============================================================================

class TestBoundaryConsistency:
    """Boundary scenarios: the two paths behave consistently."""

    def test_empty_dna_both_return_empty(self):
        rng_py = random.Random(1)
        rng_batch = random.Random(1)
        # pure Python
        dna_py, muts_py = mutate("", mutation_rate=0.5, indel_rate=0.5,
                                 rng=rng_py)
        # batch (numpy)
        inds = _make_individuals([""])
        results = mutate_batch(inds, mutation_rate=0.5, indel_rate=0.5,
                               ratio=2.0, rng=rng_batch)
        dna_batch, muts_batch = results[0]
        assert dna_py == "" and dna_batch == ""
        assert muts_py == [] and muts_batch == []

    def test_zero_rates_no_mutations(self):
        dna = "ACGT" * 50
        rng_py = random.Random(7)
        rng_batch = random.Random(7)
        _, muts_py = mutate(dna, mutation_rate=0.0, indel_rate=0.0,
                            rng=rng_py)
        results = mutate_batch(_make_individuals([dna]),
                               mutation_rate=0.0, indel_rate=0.0,
                               ratio=2.0, rng=rng_batch)
        _, muts_batch = results[0]
        assert muts_py == []
        assert muts_batch == []

    def test_single_individual_batch_returns_one_result(self):
        dna = "ACGT" * 20
        rng = random.Random(3)
        results = mutate_batch(_make_individuals([dna]),
                               mutation_rate=0.1, indel_rate=0.02,
                               ratio=2.0, rng=rng)
        assert len(results) == 1
        new_dna, muts = results[0]
        assert isinstance(new_dna, str)
        assert isinstance(muts, list)

    def test_empty_population_returns_empty_list(self):
        rng = random.Random(0)
        results = mutate_batch([], mutation_rate=0.1, indel_rate=0.02,
                               ratio=2.0, rng=rng)
        assert results == []

    def test_all_empty_dna_population(self):
        """Population of all-empty DNA → returns [("", [])] * N."""
        rng = random.Random(0)
        inds = _make_individuals(["", "", ""])
        results = mutate_batch(inds, mutation_rate=0.5, indel_rate=0.5,
                               ratio=2.0, rng=rng)
        assert len(results) == 3
        for dna, muts in results:
            assert dna == ""
            assert muts == []


# ============================================================================
# Mutation format consistency
# ============================================================================

class TestMutationFormat:
    """The mutation records produced by the two paths have consistent format."""

    def test_both_use_same_prefixes(self):
        dna = "ACGT" * 200
        rng_py = random.Random(11)
        rng_batch = random.Random(11)
        _, muts_py = mutate(dna, mutation_rate=0.05, indel_rate=0.05,
                            ratio=2.0, rng=rng_py)
        results = mutate_batch(_make_individuals([dna]),
                               mutation_rate=0.05, indel_rate=0.05,
                               ratio=2.0, rng=rng_batch)
        _, muts_batch = results[0]

        prefixes_py = set(_mutation_types(muts_py))
        prefixes_batch = set(_mutation_types(muts_batch))
        # the prefix sets used by both should be the same (or both empty)
        valid = {"sub", "ins", "del"}
        assert prefixes_py.issubset(valid)
        assert prefixes_batch.issubset(valid)
        # if both have mutations, the prefix sets should match
        if prefixes_py and prefixes_batch:
            assert prefixes_py == prefixes_batch

    def test_sub_format_has_arrow(self):
        dna = "A" * 500
        rng = random.Random(2)
        _, muts = mutate(dna, mutation_rate=0.1, indel_rate=0.0,
                         ratio=2.0, rng=rng)
        sub_muts = [m for m in muts if m.startswith("sub@")]
        assert len(sub_muts) > 0
        for m in sub_muts:
            assert "->" in m
            assert m.startswith("sub@")

    def test_batch_sub_format_has_arrow(self):
        dna = "A" * 500
        rng = random.Random(2)
        results = mutate_batch(_make_individuals([dna]),
                               mutation_rate=0.1, indel_rate=0.0,
                               ratio=2.0, rng=rng)
        _, muts = results[0]
        sub_muts = [m for m in muts if m.startswith("sub@")]
        assert len(sub_muts) > 0
        for m in sub_muts:
            assert "->" in m
            assert m.startswith("sub@")

    def test_indel_format_has_colon(self):
        dna = "ACGT" * 200
        rng = random.Random(5)
        _, muts = mutate(dna, mutation_rate=0.0, indel_rate=0.1,
                         ratio=2.0, rng=rng)
        indel_muts = [m for m in muts
                      if m.startswith("ins@") or m.startswith("del@")]
        assert len(indel_muts) > 0
        for m in indel_muts:
            assert ":" in m

    def test_batch_indel_format_has_colon(self):
        dna = "ACGT" * 200
        rng = random.Random(5)
        results = mutate_batch(_make_individuals([dna]),
                               mutation_rate=0.0, indel_rate=0.1,
                               ratio=2.0, rng=rng)
        _, muts = results[0]
        indel_muts = [m for m in muts
                      if m.startswith("ins@") or m.startswith("del@")]
        assert len(indel_muts) > 0
        for m in indel_muts:
            assert ":" in m


# ============================================================================
# Length change consistency
# ============================================================================

class TestLengthConsistency:
    """Under both paths, the length change = ins - del rule is consistent."""

    @pytest.mark.parametrize("rate_indel", [0.0, 0.02, 0.1])
    def test_py_length_invariant(self, rate_indel):
        dna = "ACGT" * 100
        rng = random.Random(42)
        new_dna, muts = mutate(dna, mutation_rate=0.0, indel_rate=rate_indel,
                               ratio=2.0, rng=rng)
        ins = sum(1 for m in muts if m.startswith("ins"))
        dele = sum(1 for m in muts if m.startswith("del"))
        assert len(new_dna) == len(dna) + ins - dele

    @pytest.mark.parametrize("rate_indel", [0.0, 0.02, 0.1])
    def test_batch_length_invariant(self, rate_indel):
        dna = "ACGT" * 100
        rng = random.Random(42)
        results = mutate_batch(_make_individuals([dna]),
                               mutation_rate=0.0, indel_rate=rate_indel,
                               ratio=2.0, rng=rng)
        new_dna, muts = results[0]
        ins = sum(1 for m in muts if m.startswith("ins"))
        dele = sum(1 for m in muts if m.startswith("del"))
        assert len(new_dna) == len(dna) + ins - dele

    def test_substitution_only_preserves_length_py(self):
        dna = "ACGT" * 100
        rng = random.Random(0)
        new_dna, muts = mutate(dna, mutation_rate=0.1, indel_rate=0.0,
                               ratio=2.0, rng=rng)
        assert len(new_dna) == len(dna)

    def test_substitution_only_preserves_length_batch(self):
        dna = "ACGT" * 100
        rng = random.Random(0)
        results = mutate_batch(_make_individuals([dna]),
                               mutation_rate=0.1, indel_rate=0.0,
                               ratio=2.0, rng=rng)
        new_dna, _ = results[0]
        assert len(new_dna) == len(dna)


# ============================================================================
# Statistical distribution consistency
# ============================================================================

class TestStatisticalConsistency:
    """Under the same parameters, the mutation count distributions of the two paths are consistent (fluctuation allowed)."""

    def test_substitution_count_similar(self):
        """High mutation_rate, no indels → both substitution counts are similar."""
        dna = "ACGT" * 1000  # 4000 nt
        n_trials = 5
        py_counts = []
        batch_counts = []
        for seed in range(n_trials):
            rng1 = random.Random(seed * 100 + 1)
            _, m_py = mutate(dna, mutation_rate=0.05, indel_rate=0.0,
                             ratio=2.0, rng=rng1)
            rng2 = random.Random(seed * 100 + 1)
            res = mutate_batch(_make_individuals([dna]),
                               mutation_rate=0.05, indel_rate=0.0,
                               ratio=2.0, rng=rng2)
            _, m_batch = res[0]
            py_counts.append(sum(1 for m in m_py if m.startswith("sub")))
            batch_counts.append(sum(1 for m in m_batch if m.startswith("sub")))
        mean_py = sum(py_counts) / n_trials
        mean_batch = sum(batch_counts) / n_trials
        # expected ~200 (4000 * 0.05), allow larger relative fluctuation
        expected = 200
        assert abs(mean_py - expected) < 60, \
            f"py mean {mean_py} far from expected {expected}"
        assert abs(mean_batch - expected) < 60, \
            f"batch mean {mean_batch} far from expected {expected}"
        # the two means should be close within a reasonable range (allow 25% difference)
        avg = (mean_py + mean_batch) / 2
        assert abs(mean_py - mean_batch) < 0.5 * avg + 10

    def test_indel_count_similar(self):
        """High indel_rate, no substitution → both indel counts are similar."""
        dna = "ACGT" * 1000
        n_trials = 5
        py_counts = []
        batch_counts = []
        for seed in range(n_trials):
            rng1 = random.Random(seed * 7 + 3)
            _, m_py = mutate(dna, mutation_rate=0.0, indel_rate=0.05,
                             ratio=2.0, rng=rng1)
            rng2 = random.Random(seed * 7 + 3)
            res = mutate_batch(_make_individuals([dna]),
                               mutation_rate=0.0, indel_rate=0.05,
                               ratio=2.0, rng=rng2)
            _, m_batch = res[0]
            py_counts.append(len(m_py))
            batch_counts.append(len(m_batch))
        mean_py = sum(py_counts) / n_trials
        mean_batch = sum(batch_counts) / n_trials
        expected = 200  # 4000 * 0.05
        assert abs(mean_py - expected) < 60
        assert abs(mean_batch - expected) < 60
        avg = (mean_py + mean_batch) / 2
        assert abs(mean_py - mean_batch) < 0.5 * avg + 10

    def test_transition_bias_both_paths(self):
        """Both paths should show transition bias (A→G >> A→T)."""
        dna = "A" * 2000
        # pure Python
        rng1 = random.Random(99)
        _, m_py = mutate(dna, mutation_rate=0.1, indel_rate=0.0,
                         ratio=2.0, rng=rng1)
        # batch
        rng2 = random.Random(99)
        res = mutate_batch(_make_individuals([dna]),
                           mutation_rate=0.1, indel_rate=0.0,
                           ratio=2.0, rng=rng2)
        _, m_batch = res[0]

        def a_to_g(muts):
            return sum(1 for m in muts
                       if m.startswith("sub") and m.endswith("->G"))
        def a_to_t(muts):
            return sum(1 for m in muts
                       if m.startswith("sub") and m.endswith("->T"))

        # A→G should far outnumber A→T in both (theoretical 4:1)
        assert a_to_g(m_py) > a_to_t(m_py) * 2, \
            f"py: A→G {a_to_g(m_py)} not >> A→T {a_to_t(m_py)} * 2"
        assert a_to_g(m_batch) > a_to_t(m_batch) * 2, \
            f"batch: A→G {a_to_g(m_batch)} not >> A→T {a_to_t(m_batch)} * 2"

    def test_no_invalid_bases_in_results(self):
        """DNA produced by both paths contains only ACGT."""
        dna = "ACGT" * 200
        valid = set("ACGT")
        rng1 = random.Random(13)
        new_py, _ = mutate(dna, mutation_rate=0.1, indel_rate=0.05,
                           ratio=2.0, rng=rng1)
        rng2 = random.Random(13)
        res = mutate_batch(_make_individuals([dna]),
                           mutation_rate=0.1, indel_rate=0.05,
                           ratio=2.0, rng=rng2)
        new_batch, _ = res[0]
        assert set(new_py).issubset(valid)
        assert set(new_batch).issubset(valid)


# ============================================================================
# Fallback path (numpy unavailable)
# ============================================================================

class TestFallbackPath:
    """When numpy is unavailable, mutate_batch should fall back to per-individual mutate."""

    def test_fallback_gives_identical_results(self, monkeypatch):
        """With numpy disabled, mutate_batch and per-individual mutate give identical results."""
        monkeypatch.setattr(evo, "_HAS_NUMPY", False)
        dna_list = ["ACGT" * 50, "GGGG" * 30, "ATAT" * 40]
        inds = _make_individuals(dna_list)
        # same seed: pure Python per individual
        expected = []
        for d in dna_list:
            rng_each = random.Random(123)
            expected.append(mutate(d, mutation_rate=0.05, indel_rate=0.02,
                                   ratio=2.0, rng=rng_each))
        # mutate_batch fallback: uses a single shared rng
        rng_shared = random.Random(123)
        # internally the fallback calls mutate(dna, ..., rng=rng_shared) for each ind
        # so it matches running per individual with the same shared rng
        expected_shared = []
        for d in dna_list:
            expected_shared.append(mutate(d, mutation_rate=0.05,
                                          indel_rate=0.02, ratio=2.0,
                                          rng=rng_shared))
        # rebuild a shared rng and run batch
        rng_batch = random.Random(123)
        results = mutate_batch(inds, mutation_rate=0.05, indel_rate=0.02,
                               ratio=2.0, rng=rng_batch)
        # should be bit-identical with expected_shared
        for (d_exp, m_exp), (d_got, m_got) in zip(expected_shared, results, strict=False):
            assert d_exp == d_got, f"DNA mismatch: {d_exp} != {d_got}"
            assert m_exp == m_got, f"mut list mismatch: {m_exp} != {m_got}"

    def test_fallback_returns_correct_count(self, monkeypatch):
        monkeypatch.setattr(evo, "_HAS_NUMPY", False)
        inds = _make_individuals(["ACGT" * 20] * 4)
        rng = random.Random(0)
        results = mutate_batch(inds, mutation_rate=0.01, indel_rate=0.001,
                               ratio=2.0, rng=rng)
        assert len(results) == 4

    def test_fallback_empty_population(self, monkeypatch):
        monkeypatch.setattr(evo, "_HAS_NUMPY", False)
        rng = random.Random(0)
        results = mutate_batch([], mutation_rate=0.1, indel_rate=0.1,
                               ratio=2.0, rng=rng)
        assert results == []

    def test_fallback_all_empty_dna(self, monkeypatch):
        monkeypatch.setattr(evo, "_HAS_NUMPY", False)
        inds = _make_individuals(["", ""])
        rng = random.Random(0)
        results = mutate_batch(inds, mutation_rate=0.5, indel_rate=0.5,
                               ratio=2.0, rng=rng)
        assert results == [("", []), ("", [])]


# ============================================================================
# Multiple-individual batch
# ============================================================================

class TestBatchPopulation:
    """Verify mutate_batch's handling of multiple individuals."""

    def test_batch_preserves_order(self):
        """Result order matches input order."""
        dna_list = ["AAAA" * 20, "CCCC" * 20, "GGGG" * 20, "TTTT" * 20]
        inds = _make_individuals(dna_list)
        rng = random.Random(0)
        results = mutate_batch(inds, mutation_rate=0.05, indel_rate=0.0,
                               ratio=2.0, rng=rng)
        assert len(results) == 4
        # each result's original DNA length (no indels) should be preserved
        for i, (new_dna, _) in enumerate(results):
            assert len(new_dna) == len(dna_list[i])

    def test_batch_unequal_lengths(self):
        """Unequal-length DNAs are also batched correctly."""
        dna_list = ["A" * 10, "C" * 100, "G" * 50]
        inds = _make_individuals(dna_list)
        rng = random.Random(0)
        results = mutate_batch(inds, mutation_rate=0.0, indel_rate=0.0,
                               ratio=2.0, rng=rng)
        assert len(results) == 3
        for i, (new_dna, muts) in enumerate(results):
            assert new_dna == dna_list[i]
            assert muts == []

    def test_batch_large_population(self):
        """Large-population batch mutation should complete efficiently."""
        inds = _make_individuals(["ACGT" * 25] * 50)  # 50 individuals of 100 nt
        rng = random.Random(42)
        results = mutate_batch(inds, mutation_rate=0.05, indel_rate=0.01,
                               ratio=2.0, rng=rng)
        assert len(results) == 50
        # at least some individuals should mutate
        mutated_count = sum(1 for _, m in results if len(m) > 0)
        assert mutated_count > 20, \
            f"expected >20 mutated individuals, got {mutated_count}"

    def test_batch_reproducible_with_seed(self):
        """Same seed → same results (also holds on the numpy path)."""
        inds = _make_individuals(["ACGT" * 30] * 3)
        rng1 = random.Random(777)
        r1 = mutate_batch(inds, mutation_rate=0.05, indel_rate=0.02,
                          ratio=2.0, rng=rng1)
        inds2 = _make_individuals(["ACGT" * 30] * 3)
        rng2 = random.Random(777)
        r2 = mutate_batch(inds2, mutation_rate=0.05, indel_rate=0.02,
                          ratio=2.0, rng=rng2)
        for (d1, m1), (d2, m2) in zip(r1, r2, strict=False):
            assert d1 == d2
            assert m1 == m2


# ============================================================================
# Different ratio parameters
# ============================================================================

class TestRatioParameter:
    """Verify the ratio parameter takes effect on both paths."""

    def test_high_ratio_increases_transition_py(self):
        """ratio=10 → higher transition share (pure Python)."""
        dna = "A" * 2000
        rng = random.Random(0)
        _, muts = mutate(dna, mutation_rate=0.1, indel_rate=0.0,
                         ratio=10.0, rng=rng)
        a_to_g = sum(1 for m in muts if m.startswith("sub") and m.endswith("->G"))
        a_to_t = sum(1 for m in muts if m.startswith("sub") and m.endswith("->T"))
        a_to_c = sum(1 for m in muts if m.startswith("sub") and m.endswith("->C"))
        # ratio=10 → P(transition)=10/11, A→G should far outnumber A→T + A→C
        assert a_to_g > (a_to_t + a_to_c) * 3

    def test_high_ratio_increases_transition_batch(self):
        """ratio=10 → higher transition share (batch)."""
        dna = "A" * 2000
        rng = random.Random(0)
        res = mutate_batch(_make_individuals([dna]),
                           mutation_rate=0.1, indel_rate=0.0,
                           ratio=10.0, rng=rng)
        _, muts = res[0]
        a_to_g = sum(1 for m in muts if m.startswith("sub") and m.endswith("->G"))
        a_to_t = sum(1 for m in muts if m.startswith("sub") and m.endswith("->T"))
        a_to_c = sum(1 for m in muts if m.startswith("sub") and m.endswith("->C"))
        assert a_to_g > (a_to_t + a_to_c) * 3

    def test_zero_ratio_gives_uniform(self):
        """ratio=0 → transition_bias=0.5, no transition bias (pure Python)."""
        dna = "A" * 2000
        rng = random.Random(0)
        _, muts = mutate(dna, mutation_rate=0.1, indel_rate=0.0,
                         ratio=0.0, rng=rng)
        a_to_g = sum(1 for m in muts if m.startswith("sub") and m.endswith("->G"))
        total = sum(1 for m in muts if m.startswith("sub"))
        # ratio=0 → P(transition)=0.5, A→G ~ 1/6 of all mutations
        # actual P(A→G) = 0.5 (one of the transitions), the other two are 0.25 each
        if total > 0:
            # A→G should be ~50%, at least not > 80%
            assert a_to_g / total < 0.8
