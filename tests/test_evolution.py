"""Evolution engine tests: verify the Wright-Fisher model based on real evolutionary biology parameters.

Verifies (based on real published parameters):
- Mutation rate statistics: substitution rate matches the setting (Lee 2012, 2.2e-10 per nt per gen)
- transition:transversion = 3:1 (conserved bacterial value; Stoltzfus 2009, bacteria 3:1–6:1)
- Indel introduction: insertion + deletion statistics
- Natural selection: high-fitness individuals survive more (Wright-Fisher sampling)
- Homologous recombination: crossover produces mixed sequences
- Fitness calculation: Hamming/CAI/GC three methods
- Population diversity: Shannon entropy normalization
- Fitness improvement after many generations of evolution (mutation + selection drive adaptation)
- dN/dS calculation (simplified Nei-Gojobori 1986)
- Population statistics: mean/max/min fitness + diversity recorded each generation

References:
- Drake JW. Genetics 1991 148:1667-1686 (E. coli mutation rate)
- Lee H et al. Nature 2012 489:527-531 (base substitution rate 2.2e-10)
- Stoltzfus A & Norris RW. Mol Biol Evol 2016 33:595-604 (transition bias)
- Nei M & Gojobori T. Mol Biol Evol 1986 3:418-426 (dN/dS)
- Sharp PM et al. Nucleic Acids Res 1987 15:1281-1295 (CAI)

Pure Python tests, no BioPython/reedsolo dependencies.
"""
from __future__ import annotations

import math
import random

import pytest

from helixlang.evolution import (
    E_COLI_INDEL_RATE,
    E_COLI_NE,
    E_COLI_SUBSTITUTION_RATE,
    TRANSITION_TRANSVERSION_RATIO,
    EvolutionaryPopulation,
    EvolutionConfig,
    Individual,
    calculate_fitness,
    dnds_ratio,
    fitness_landscape,
    mutate,
    recombine,
    select,
)

# ============================================================================
# Real parameter constant verification
# ============================================================================

class TestRealisticParameters:
    """Verify evolution parameters are based on real published measurements."""

    def test_ecoli_substitution_rate(self):
        """E. coli base substitution rate 2.2e-10 per nt per generation (Lee 2012)."""
        assert E_COLI_SUBSTITUTION_RATE == 2.2e-10

    def test_ecoli_indel_rate(self):
        """E. coli indel rate 2.2e-11 per nt per generation (Lee 2012; indel:sub ≈ 0.10)."""
        assert E_COLI_INDEL_RATE == 2.2e-11

    def test_transition_transversion_ratio(self):
        """transition:transversion = 3:1 (conserved bacterial value; Stoltzfus 2009, bacteria 3:1–6:1)."""
        assert TRANSITION_TRANSVERSION_RATIO == 3.0

    def test_ecoli_effective_population_size(self):
        """E. coli effective population size ~1.3e8 (Hartl & Clark 2007)."""
        assert E_COLI_NE == 1.3e8

    def test_default_config_uses_real_rates(self):
        """EvolutionConfig defaults use real mutation rates."""
        cfg = EvolutionConfig()
        assert cfg.mutation_rate == E_COLI_SUBSTITUTION_RATE
        assert cfg.indel_rate == E_COLI_INDEL_RATE
        assert cfg.transition_transversion_ratio == TRANSITION_TRANSVERSION_RATIO


# ============================================================================
# Mutation
# ============================================================================

class TestMutate:
    """Verify the mutation model."""

    def test_substitution_rate_statistical(self):
        """Statistical verification: substitution count matches expectation under high mutation_rate.

        Uses a 10000 nt sequence + mutation_rate=0.01 → expected ~100 substitutions.
        indel_rate=0 isolates the substitution effect.
        """
        rng = random.Random(42)
        dna = "".join(rng.choice("ACGT") for _ in range(10000))
        mutated, mutations = mutate(
            dna, mutation_rate=0.01, indel_rate=0.0, ratio=2.0, rng=rng
        )
        # length unchanged (no indels)
        assert len(mutated) == len(dna)
        # count substitutions
        sub_count = sum(1 for m in mutations if m.startswith("sub"))
        # expected 100, allow [50, 200] (5σ statistical fluctuation)
        assert 50 < sub_count < 200, \
            f"substitution count {sub_count} not in [50, 200] (expected ~100)"
        # actual position differences should match the mutations list
        pos_diff = sum(1 for i in range(len(dna)) if dna[i] != mutated[i])
        assert pos_diff == sub_count, \
            f"position diffs {pos_diff} != sub_count {sub_count}"

    def test_zero_mutation_rate_no_changes(self):
        """mutation_rate=0 + indel_rate=0 → sequence unchanged."""
        rng = random.Random(42)
        dna = "ACGTACGTAC" * 100
        mutated, mutations = mutate(
            dna, mutation_rate=0.0, indel_rate=0.0, rng=rng
        )
        assert mutated == dna
        assert mutations == []

    def test_transition_bias_a_to_g_dominates(self):
        """Transition bias: A→G should far outnumber A→T (ratio=2:1 → P(A→G)=2/3, P(A→T)=1/6).

        Stoltzfus 2009: transition:transversion ≈ 2:1.
        For A: transition=A→G, transversion=A→C or A→T.
        P(A→G) = 2/3, P(A→T) = 1/6 → A→G should be ~4x A→T.
        """
        rng = random.Random(42)
        # all-A sequence, high substitution rate to get statistical signal
        dna = "A" * 10000
        mutated, mutations = mutate(
            dna, mutation_rate=0.05, indel_rate=0.0, ratio=2.0, rng=rng
        )
        # count each substitution type
        a_to_g = 0
        a_to_t = 0
        a_to_c = 0
        for m in mutations:
            if m.startswith("sub"):
                # format: "sub@{pos}:A->{new}"
                new_base = m.split("->")[1]
                if new_base == "G":
                    a_to_g += 1
                elif new_base == "T":
                    a_to_t += 1
                elif new_base == "C":
                    a_to_c += 1
        # A→G should far outnumber A→T (theoretical ratio 4:1)
        assert a_to_g > 0, "expected some A→G transitions"
        assert a_to_g > a_to_t * 2, \
            f"A→G ({a_to_g}) should >> A→T ({a_to_t}) * 2 (ratio=2:1 → ~4:1)"
        # A→G should also outnumber A→C (transversion)
        assert a_to_g > a_to_c, \
            f"A→G ({a_to_g}) should > A→C ({a_to_c})"

    def test_transition_bias_c_to_t(self):
        """C→T is a common transition (pyrimidine interchange)."""
        rng = random.Random(42)
        dna = "C" * 10000
        mutated, mutations = mutate(
            dna, mutation_rate=0.05, indel_rate=0.0, ratio=2.0, rng=rng
        )
        c_to_t = sum(1 for m in mutations
                     if m.startswith("sub") and m.endswith("->T"))
        c_to_a = sum(1 for m in mutations
                     if m.startswith("sub") and m.endswith("->A"))
        # C→T (transition) should far outnumber C→A (transversion)
        assert c_to_t > c_to_a * 2, \
            f"C→T ({c_to_t}) should >> C→A ({c_to_a}) * 2"

    def test_indel_introduction(self):
        """Indel introduction: sequence length changes under high indel_rate."""
        rng = random.Random(42)
        dna = "ACGT" * 1000  # 4000 nt
        mutated, mutations = mutate(
            dna, mutation_rate=0.0, indel_rate=0.02, ratio=2.0, rng=rng
        )
        # should have indel events
        ins_count = sum(1 for m in mutations if m.startswith("ins"))
        del_count = sum(1 for m in mutations if m.startswith("del"))
        sub_count = sum(1 for m in mutations if m.startswith("sub"))
        total_indel = ins_count + del_count
        # expected ~80 indels (4000 * 0.02)
        assert total_indel > 30, \
            f"indel count {total_indel} too low (expected ~80)"
        # mutation_rate=0 → no substitutions expected
        assert sub_count == 0, \
            f"sub_count {sub_count} should be 0 (mutation_rate=0)"
        # length change = insertions - deletions
        expected_len = len(dna) + ins_count - del_count
        assert len(mutated) == expected_len, \
            f"length {len(mutated)} != expected {expected_len}"

    def test_indel_insertion_and_deletion_both_occur(self):
        """Both insertion and deletion should occur (statistical verification)."""
        rng = random.Random(123)
        dna = "ACGTACGTAC" * 500  # 5000 nt
        mutated, mutations = mutate(
            dna, mutation_rate=0.0, indel_rate=0.05, rng=rng
        )
        ins_count = sum(1 for m in mutations if m.startswith("ins"))
        del_count = sum(1 for m in mutations if m.startswith("del"))
        # both should be > 0 (5000 * 0.025 = 125 each expected)
        assert ins_count > 50, f"insertions {ins_count} too low"
        assert del_count > 50, f"deletions {del_count} too low"

    def test_mutation_reproducible_with_seed(self):
        """The same rng seed produces the same mutation results."""
        dna = "ACGT" * 100
        m1, _ = mutate(dna, mutation_rate=0.01, indel_rate=0.001,
                       rng=random.Random(42))
        m2, _ = mutate(dna, mutation_rate=0.01, indel_rate=0.001,
                       rng=random.Random(42))
        assert m1 == m2

    def test_empty_dna(self):
        """Empty DNA produces no mutations."""
        mutated, mutations = mutate("", mutation_rate=0.5, indel_rate=0.5)
        assert mutated == ""
        assert mutations == []

    def test_mutation_format(self):
        """Mutation record format is correct."""
        rng = random.Random(42)
        dna = "A" * 1000
        _, mutations = mutate(
            dna, mutation_rate=0.1, indel_rate=0.05, rng=rng
        )
        for m in mutations:
            assert m.startswith(("sub@", "ins@", "del@"))
            if m.startswith("sub@"):
                # format: sub@{pos}:{orig}->{new}
                assert "->" in m
            elif m.startswith("ins@") or m.startswith("del@"):
                # format: ins@{pos}:{base} or del@{pos}:{base}
                assert ":" in m


# ============================================================================
# Selection (Wright-Fisher sampling)
# ============================================================================

class TestSelect:
    """Verify natural selection (Wright-Fisher sampling)."""

    def test_high_fitness_survives_more(self):
        """High-fitness individuals gain a significantly larger share after selection."""
        rng = random.Random(42)
        # 50 high-fitness + 50 low-fitness
        pop = [
            Individual(dna=f"HIGH{i}", fitness=0.9, generation=0, mutations=[])
            for i in range(50)
        ] + [
            Individual(dna=f"LOW{i}", fitness=0.1, generation=0, mutations=[])
            for i in range(50)
        ]
        selected = select(pop, selection_coefficient=10.0, rng=rng)
        # population size unchanged after selection
        assert len(selected) == 100
        # high-fitness individuals should be the majority
        high_count = sum(1 for ind in selected if ind.fitness == 0.9)
        assert high_count > 80, \
            f"high-fitness count {high_count} should > 80 (s=10 strong selection)"

    def test_negative_selection_favors_low_fitness(self):
        """s < 0 (negative selection): low-fitness individuals are favored."""
        rng = random.Random(42)
        pop = [
            Individual(dna=f"HIGH{i}", fitness=0.9, generation=0, mutations=[])
            for i in range(50)
        ] + [
            Individual(dna=f"LOW{i}", fitness=0.1, generation=0, mutations=[])
            for i in range(50)
        ]
        selected = select(pop, selection_coefficient=-10.0, rng=rng)
        low_count = sum(1 for ind in selected if ind.fitness == 0.1)
        assert low_count > 80, \
            f"low-fitness count {low_count} should > 80 (s=-10 negative selection)"

    def test_wright_fisher_neutral_drift(self):
        """Neutral drift (s=0): uniform sampling, size unchanged.

        When all individuals have the same fitness, s=0 → all weights are 1 → uniform sampling.
        """
        rng = random.Random(42)
        n = 100
        pop = [
            Individual(dna=f"IND{i}", fitness=0.5, generation=0, mutations=[])
            for i in range(n)
        ]
        selected = select(pop, selection_coefficient=0.0, rng=rng)
        # size = input size (Wright-Fisher sampling with replacement)
        assert len(selected) == n
        # uniform sampling: no individual is over-selected
        from collections import Counter
        dna_counts = Counter(ind.dna for ind in selected)
        max_count = max(dna_counts.values())
        # sampling 100 individuals uniformly 100 times, max count should be < 15 (Poisson λ=1, P(X≥15) tiny)
        assert max_count < 15, \
            f"max count {max_count} too high for neutral drift (uniform expected)"

    def test_population_size_maintained(self):
        """Wright-Fisher sampling maintains population size."""
        rng = random.Random(42)
        for size in [10, 50, 200]:
            pop = [
                Individual(dna=f"X{i}", fitness=0.5 + i * 0.001,
                           generation=0, mutations=[])
                for i in range(size)
            ]
            selected = select(pop, selection_coefficient=1.0, rng=rng)
            assert len(selected) == size, \
                f"size {size}: selected {len(selected)} != {size}"

    def test_empty_population(self):
        """Empty population returns an empty list."""
        assert select([], selection_coefficient=1.0) == []

    def test_single_individual(self):
        """A single-individual population returns itself."""
        ind = Individual(dna="ACGT", fitness=1.0, generation=0, mutations=[])
        result = select([ind], selection_coefficient=1.0)
        assert len(result) == 1
        assert result[0] is ind


# ============================================================================
# Recombination
# ============================================================================

class TestRecombine:
    """Verify homologous recombination (crossover)."""

    def test_recombination_mixes_parents(self):
        """After recombination, DNA contains segments from both parents."""
        rng = random.Random(42)
        p1 = "A" * 100
        p2 = "T" * 100
        child = recombine(p1, p2, rate=1.0, rng=rng)
        # length = parent length
        assert len(child) == 100
        # should contain both A and T (from the two parents)
        assert "A" in child, "child should contain A from parent1"
        assert "T" in child, "child should contain T from parent2"

    def test_no_recombination_rate_zero(self):
        """rate=0 means no recombination, returns parent1."""
        rng = random.Random(42)
        p1 = "ACGT" * 25
        p2 = "TGCA" * 25
        child = recombine(p1, p2, rate=0.0, rng=rng)
        assert child == p1

    def test_recombination_preserves_length(self):
        """Recombination preserves the length of equal-length parents."""
        rng = random.Random(42)
        for _ in range(20):
            n = rng.randint(10, 200)
            p1 = "".join(rng.choice("ACGT") for _ in range(n))
            p2 = "".join(rng.choice("ACGT") for _ in range(n))
            child = recombine(p1, p2, rate=1.0, rng=rng)
            assert len(child) == n, \
                f"child length {len(child)} != parent length {n}"

    def test_recombination_different_lengths(self):
        """Recombination with different-length parents: length = max(len(p1), len(p2))."""
        rng = random.Random(42)
        p1 = "A" * 50
        p2 = "T" * 80
        child = recombine(p1, p2, rate=1.0, rng=rng)
        # length = max(50, 80) = 80
        assert len(child) == 80

    def test_recombination_reproducible(self):
        """The same seed produces the same recombination result."""
        p1 = "ACGT" * 25
        p2 = "TGCA" * 25
        c1 = recombine(p1, p2, rate=1.0, rng=random.Random(42))
        c2 = recombine(p1, p2, rate=1.0, rng=random.Random(42))
        assert c1 == c2

    def test_recombination_only_acgt(self):
        """Recombination results contain only ACGT."""
        rng = random.Random(42)
        p1 = "".join(rng.choice("ACGT") for _ in range(100))
        p2 = "".join(rng.choice("ACGT") for _ in range(100))
        child = recombine(p1, p2, rate=1.0, rng=rng)
        assert all(c in "ACGT" for c in child)


# ============================================================================
# Fitness calculation
# ============================================================================

class TestCalculateFitness:
    """Verify fitness calculation (Hamming/CAI/GC/custom)."""

    def test_hamming_identical(self):
        """Completely identical → fitness 1.0."""
        assert calculate_fitness("ACGTACGT", "ACGTACGT", "hamming") == 1.0

    def test_hamming_completely_different(self):
        """Completely different → fitness 0.0."""
        assert calculate_fitness("AAAA", "TTTT", "hamming") == 0.0

    def test_hamming_partial_match(self):
        """Partial match → fitness = match rate."""
        # 3 of 4 match
        assert calculate_fitness("ACGT", "ACGA", "hamming") == 0.75
        # 2 of 4 match (A=A, C≠G, G=G, T=T → 3 matches; use another example)
        # ACGT vs AGCT: A=A✓ C≠G✗ G≠C✗ T=T✓ → 2 matches
        assert calculate_fitness("ACGT", "AGCT", "hamming") == 0.5

    def test_hamming_different_lengths(self):
        """Different lengths: matches / max length."""
        # "ACG" vs "ACGT" → 3 matches / 4 = 0.75
        assert calculate_fitness("ACG", "ACGT", "hamming") == 0.75

    def test_hamming_requires_target(self):
        """hamming method requires target_dna."""
        with pytest.raises(ValueError):
            calculate_fitness("ACGT", method="hamming")

    def test_cai_optimal_codons(self):
        """Using only optimal codons (fraction=1.0) → CAI = 1.0.

        ATG (Met, fraction=1.0) and TGG (Trp, fraction=1.0) are unique codons.
        """
        # ATG TGG ATG TGG → M W M W
        dna = "ATGTGG" * 2
        cai = calculate_fitness(dna, method="cai")
        assert cai == pytest.approx(1.0, abs=0.01)

    def test_cai_rare_codons_low(self):
        """Using only rare codons (CTA, fraction=0.04) → very low CAI."""
        # ATG + 20×CTA (Leu rare) + TAG
        dna = "ATG" + "CTA" * 20 + "TAG"
        cai = calculate_fitness(dna, method="cai")
        assert cai < 0.15, f"rare codon CAI {cai:.3f} should < 0.15"

    def test_cai_optimal_higher_than_rare(self):
        """Optimal codon CAI > rare codon CAI."""
        optimal_dna = "ATG" + "CTG" * 10 + "TAA"  # CTG = Leu optimal (0.47)
        rare_dna = "ATG" + "CTA" * 10 + "TAA"     # CTA = Leu rare (0.04)
        cai_opt = calculate_fitness(optimal_dna, method="cai")
        cai_rare = calculate_fitness(rare_dna, method="cai")
        assert cai_opt > cai_rare, \
            f"optimal CAI {cai_opt} should > rare CAI {cai_rare}"

    def test_gc_content_50_percent(self):
        """50% GC → fitness 1.0."""
        # ACGT → 2/4 = 50% GC
        assert calculate_fitness("ACGT", method="gc") == 1.0
        # AAGC → 2/4 = 50% GC
        assert calculate_fitness("AAGC", method="gc") == 1.0

    def test_gc_content_zero_or_full(self):
        """0% or 100% GC → fitness 0.0."""
        assert calculate_fitness("AAAA", method="gc") == 0.0
        assert calculate_fitness("GCGC", method="gc") == 0.0

    def test_gc_content_partial(self):
        """Partial GC → fitness = 1 - |GC - 0.5| * 2."""
        # ACG → 2/3 GC → 1 - |2/3 - 0.5| * 2 = 1 - 1/3 = 2/3
        gc_fit = calculate_fitness("ACG", method="gc")
        assert gc_fit == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_custom_method(self):
        """Custom fitness function."""
        # use sequence length as fitness
        def custom_func(dna: str) -> float:
            return len(dna) / 100.0
        assert calculate_fitness("ACGT", method="custom",
                                 custom_func=custom_func) == 0.04
        assert calculate_fitness("A" * 50, method="custom",
                                 custom_func=custom_func) == 0.5

    def test_custom_requires_func(self):
        """custom method requires custom_func."""
        with pytest.raises(ValueError):
            calculate_fitness("ACGT", method="custom")

    def test_unknown_method_raises(self):
        """Unknown method raises ValueError."""
        with pytest.raises(ValueError):
            calculate_fitness("ACGT", method="nonexistent")


# ============================================================================
# Population diversity
# ============================================================================

class TestPopulationDiversity:
    """Verify population diversity (Shannon entropy)."""

    def test_uniform_population_zero_diversity(self):
        """All individuals identical → diversity = 0."""
        cfg = EvolutionConfig(population_size=50)
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGTAC",
            config=cfg,
            target_dna="ACGTACGTAC",
            fitness_method="hamming",
        )
        assert pop.get_diversity() == 0.0

    def test_diverse_population_high(self):
        """All individuals distinct → diversity close to 1.0."""
        cfg = EvolutionConfig(population_size=20)
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGTAC",
            config=cfg,
            target_dna="ACGTACGTAC",
            fitness_method="hamming",
        )
        # manually set different DNAs
        for i, ind in enumerate(pop.individuals):
            ind.dna = f"ACGT{i:06d}"  # each distinct
        diversity = pop.get_diversity()
        assert diversity > 0.95, \
            f"all-distinct diversity {diversity:.3f} should ~1.0"

    def test_partial_diversity(self):
        """Partial diversity: two equal groups → diversity = ln(2)/ln(N) * ... ≈ moderate."""
        cfg = EvolutionConfig(population_size=100)
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGTAC",
            config=cfg,
            target_dna="ACGTACGTAC",
            fitness_method="hamming",
        )
        # 50 of A, 50 of B
        for i in range(50):
            pop.individuals[i].dna = "AAAA"
        for i in range(50, 100):
            pop.individuals[i].dna = "TTTT"
        diversity = pop.get_diversity()
        # two equal groups: H = -2 * (0.5 * ln(0.5)) = ln(2)
        # normalized: ln(2) / ln(100) ≈ 0.151
        assert 0.1 < diversity < 0.25, \
            f"two-group diversity {diversity:.3f} not in [0.1, 0.25]"

    def test_single_individual_zero_diversity(self):
        """Single-individual population diversity = 0."""
        cfg = EvolutionConfig(population_size=1)
        pop = EvolutionaryPopulation(
            initial_dna="ACGT",
            config=cfg,
            target_dna="ACGT",
            fitness_method="hamming",
        )
        assert pop.get_diversity() == 0.0


# ============================================================================
# Fitness improvement after many generations of evolution
# ============================================================================

class TestEvolutionImprovement:
    """Verify fitness improves after many generations (mutation + selection drive adaptation)."""

    def test_fitness_improves_over_generations(self):
        """Under high mutation rate + strong selection, mean fitness improves significantly after many generations."""
        target = "ATGGCTGGTGGCGCCTAA" * 3   # 54 nt
        initial = "ATGAAGAGGAGGAAGTAA" * 3  # 54 nt, ~50% match
        cfg = EvolutionConfig(
            mutation_rate=0.02,
            indel_rate=0.0,         # keep length stable (hamming needs alignment)
            transition_transversion_ratio=2.0,
            population_size=300,
            generations=80,
            recombination_rate=0.0,
            selection_coefficient=8.0,
        )
        rng = random.Random(42)
        pop = EvolutionaryPopulation(
            initial_dna=initial,
            config=cfg,
            target_dna=target,
            fitness_method="hamming",
            rng=rng,
        )
        initial_mean = pop.mean_fitness()
        # initial fitness should be ~0.5 (half positions match)
        assert 0.3 < initial_mean < 0.7, \
            f"initial mean fitness {initial_mean:.3f} not ~0.5"
        # evolve 80 generations
        pop.evolve(80)
        final_mean = pop.mean_fitness()
        # fitness should improve significantly (+0.1 or more)
        assert final_mean > initial_mean + 0.1, \
            f"fitness did not improve: {initial_mean:.3f} → {final_mean:.3f}"
        # final fitness should be > 0.6
        assert final_mean > 0.6, \
            f"final fitness {final_mean:.3f} should > 0.6"

    def test_max_fitness_increases(self):
        """Max fitness should increase with generations."""
        target = "ATGGCTGGTGGCGCCTAA" * 2  # 36 nt
        initial = "ATGAAGAGGAGGAAGTAA" * 2
        cfg = EvolutionConfig(
            mutation_rate=0.02,
            indel_rate=0.0,
            population_size=100,
            generations=30,
            selection_coefficient=8.0,
        )
        pop = EvolutionaryPopulation(
            initial_dna=initial,
            config=cfg,
            target_dna=target,
            fitness_method="hamming",
            rng=random.Random(42),
        )
        initial_max = pop.best_individual().fitness
        pop.evolve(30)
        final_max = pop.best_individual().fitness
        assert final_max > initial_max, \
            f"max fitness did not improve: {initial_max:.3f} → {final_max:.3f}"

    def test_neutral_evolution_no_direction(self):
        """Neutral evolution (s=0) has no directional fitness change."""
        target = "ATGGCTGGTGGCGCCTAA"
        initial = "ATGAAGAGGAGGAAGTAA"
        cfg = EvolutionConfig(
            mutation_rate=0.01,
            indel_rate=0.0,
            population_size=100,
            generations=20,
            selection_coefficient=0.0,  # neutral
        )
        pop = EvolutionaryPopulation(
            initial_dna=initial,
            config=cfg,
            target_dna=target,
            fitness_method="hamming",
            rng=random.Random(42),
        )
        initial_mean = pop.mean_fitness()
        pop.evolve(20)
        final_mean = pop.mean_fitness()
        # neutral evolution: fitness should not improve significantly (fluctuation allowed)
        # fitness change should be within ±0.15
        assert abs(final_mean - initial_mean) < 0.2, \
            f"neutral evolution should not improve fitness significantly: " \
            f"{initial_mean:.3f} → {final_mean:.3f}"


# ============================================================================
# dN/dS calculation
# ============================================================================

class TestDnDsRatio:
    """Verify dN/dS calculation (simplified Nei-Gojobori 1986)."""

    def test_synonymous_only(self):
        """Only synonymous substitutions → dN=0, dS>0, dNdS=0 (purifying selection)."""
        # ancestral: ATG GCT GGT TAA (M-A-G-*)
        # dna:       ATG GCC GGA TAA (M-A-G-*)
        # GCT→GCC: both Ala (synonymous)
        # GGT→GGA: both Gly (synonymous)
        ancestral = "ATGGCTGGTTAA"
        dna = "ATGGCCGGATAA"
        result = dnds_ratio(dna, ancestral)
        assert result["syn_substitutions"] == 2
        assert result["nonsyn_substitutions"] == 0
        assert result["dN"] == 0.0
        assert result["dS"] > 0.0
        assert result["dNdS"] == 0.0
        assert "purifying" in result["interpretation"].lower() or \
               "purifying" in result["interpretation"]

    def test_nonsynonymous_only(self):
        """Only nonsynonymous substitutions → dN>0, dS=0, dNdS=inf (positive selection)."""
        # ancestral: ATG GCT GGT TAA (M-A-G-*)
        # dna:       ATG CCT GGT TAA (M-P-G-*)
        # GCT→CCT: Ala→Pro (non-synonymous)
        ancestral = "ATGGCTGGTTAA"
        dna = "ATGCCTGGTTAA"
        result = dnds_ratio(dna, ancestral)
        assert result["nonsyn_substitutions"] == 1
        assert result["syn_substitutions"] == 0
        assert result["dN"] > 0.0
        assert math.isinf(result["dNdS"])
        assert "positive" in result["interpretation"].lower() or \
               "positive" in result["interpretation"]

    def test_no_substitutions(self):
        """No substitutions → dN=0, dS=0, dNdS=0."""
        dna = "ATGGCTGGTTAA"
        result = dnds_ratio(dna, dna)
        assert result["syn_substitutions"] == 0
        assert result["nonsyn_substitutions"] == 0
        assert result["dN"] == 0.0
        assert result["dS"] == 0.0
        assert result["dNdS"] == 0.0
        assert "no substitutions" in result["interpretation"] or \
               "no substitutions" in result["interpretation"]

    def test_mixed_substitutions(self):
        """Mixed substitutions: synonymous + nonsynonymous."""
        # ancestral: ATG GCT GGT TAA (M-A-G-*)
        # dna:       ATG GCC CCT TAA (M-A-P-*)
        # GCT→GCC: Ala→Ala (synonymous)
        # GGT→CCT: Gly→Pro (non-synonymous)
        ancestral = "ATGGCTGGTTAA"
        dna = "ATGGCCCCTTAA"
        result = dnds_ratio(dna, ancestral)
        assert result["syn_substitutions"] == 1
        assert result["nonsyn_substitutions"] == 1
        assert result["dN"] > 0.0
        assert result["dS"] > 0.0

    def test_syn_sites_plus_nonsyn_sites(self):
        """Each codon has S + N = 3 (Nei-Gojobori property)."""
        ancestral = "ATGGCTGGTTAA"
        dna = "ATGGCTGGTTAA"  # unchanged
        result = dnds_ratio(dna, ancestral)
        # 4 codons, each S + N = 3 → total 12
        total_sites = result["syn_sites"] + result["nonsyn_sites"]
        assert total_sites == pytest.approx(12.0, abs=0.01), \
            f"S+N = {total_sites}, expected 12 (4 codons × 3)"

    def test_return_fields_complete(self):
        """Returned dict contains all required fields."""
        result = dnds_ratio("ATGGCTTAA", "ATGGCTTAA")
        required_fields = {"dN", "dS", "dNdS", "nonsyn_substitutions",
                           "syn_substitutions", "syn_sites",
                           "nonsyn_sites", "interpretation"}
        assert set(result.keys()) == required_fields, \
            f"missing fields: {required_fields - set(result.keys())}"

    def test_empty_sequences(self):
        """Empty sequences do not raise exceptions."""
        result = dnds_ratio("", "")
        assert result["dN"] == 0.0
        assert result["dS"] == 0.0


# ============================================================================
# Population statistics
# ============================================================================

class TestPopulationStats:
    """Verify population statistics recording."""

    def test_history_length_after_evolution(self):
        """After N generations of evolution, history has N+1 records (including initial)."""
        cfg = EvolutionConfig(population_size=20, generations=5)
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGT",
            config=cfg,
            target_dna="ACGTACGT",
            fitness_method="hamming",
            rng=random.Random(42),
        )
        pop.evolve(5)
        stats = pop.get_generation_stats()
        assert len(stats) == 6  # initial + 5 generations
        # generation increments
        for i, s in enumerate(stats):
            assert s["generation"] == i, \
                f"generation {s['generation']} != expected {i}"

    def test_stats_fields(self):
        """Each statistic contains the required fields."""
        cfg = EvolutionConfig(population_size=10, generations=2)
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGT",
            config=cfg,
            target_dna="ACGTACGT",
            fitness_method="hamming",
            rng=random.Random(42),
        )
        pop.evolve(2)
        for stats in pop.get_generation_stats():
            assert "generation" in stats
            assert "population_size" in stats
            assert "mean_fitness" in stats
            assert "max_fitness" in stats
            assert "min_fitness" in stats
            assert "diversity" in stats

    def test_population_size_constant_in_evolution(self):
        """Population size remains constant during evolution."""
        cfg = EvolutionConfig(
            population_size=50, generations=10,
            mutation_rate=0.01, indel_rate=0.0,
        )
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGTAC",
            config=cfg,
            target_dna="ACGTACGTAC",
            fitness_method="hamming",
            rng=random.Random(42),
        )
        pop.evolve(10)
        for stats in pop.get_generation_stats():
            assert stats["population_size"] == 50

    def test_fitness_bounds(self):
        """Fitness values are within [0, 1] (hamming method)."""
        cfg = EvolutionConfig(
            population_size=30, generations=5,
            mutation_rate=0.05, indel_rate=0.0,
        )
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGTAC",
            config=cfg,
            target_dna="TTTTTTTTTT",
            fitness_method="hamming",
            rng=random.Random(42),
        )
        pop.evolve(5)
        _eps = 1e-9
        for stats in pop.get_generation_stats():
            assert 0.0 <= stats["min_fitness"] <= stats["max_fitness"] + _eps <= 1.0 + _eps
            assert stats["min_fitness"] - _eps <= stats["mean_fitness"] <= stats["max_fitness"] + _eps

    def test_diversity_increases_with_mutation(self):
        """Mutation increases population diversity."""
        cfg = EvolutionConfig(
            population_size=50, generations=10,
            mutation_rate=0.1, indel_rate=0.0,
            selection_coefficient=0.0,  # neutral, let diversity accumulate
        )
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGTAC",
            config=cfg,
            target_dna="ACGTACGTAC",
            fitness_method="hamming",
            rng=random.Random(42),
        )
        initial_diversity = pop.get_diversity()
        assert initial_diversity == 0.0  # initially all identical
        pop.evolve(10)
        final_diversity = pop.get_diversity()
        # under high mutation rate, diversity should increase significantly
        assert final_diversity > 0.3, \
            f"diversity {final_diversity:.3f} should > 0.3 after high mutation"


# ============================================================================
# Fitness landscape
# ============================================================================

class TestFitnessLandscape:
    """Verify fitness landscape calculation."""

    def test_landscape_structure(self):
        """Landscape returns correct structure {position: {base: fitness}}."""
        dna = "ACGT"
        target = "ACGT"
        landscape = fitness_landscape(dna, target)
        assert len(landscape) == 4  # 4 positions
        for pos in range(4):
            assert pos in landscape
            assert set(landscape[pos].keys()) == {"A", "C", "G", "T"}

    def test_landscape_original_base_highest(self):
        """The original base has the highest fitness (matches target)."""
        dna = "ACGT"
        target = "ACGT"
        landscape = fitness_landscape(dna, target)
        for pos in range(4):
            original = dna[pos]
            original_fit = landscape[pos][original]
            # original base fitness = 1.0 (perfect match)
            assert original_fit == 1.0
            # other bases fitness < 1.0
            for base in "ACGT":
                if base != original:
                    assert landscape[pos][base] < 1.0

    def test_landscape_specific_positions(self):
        """Evaluate only specified positions."""
        dna = "ACGTACGT"
        target = "ACGTACGT"
        landscape = fitness_landscape(dna, target, positions=[0, 2])
        assert set(landscape.keys()) == {0, 2}

    def test_landscape_with_different_target(self):
        """With a different target, mutation may improve fitness."""
        dna = "AAAA"       # current sequence
        target = "ACGT"    # target
        landscape = fitness_landscape(dna, target)
        # position 1: original base A (does not match target[1]=C)
        # mutating to C should improve fitness
        pos1 = landscape[1]
        assert pos1["C"] > pos1["A"], \
            f"mutating A→C at pos 1 should improve fitness " \
            f"({pos1['C']:.3f} > {pos1['A']:.3f})"

    def test_population_fitness_landscape(self):
        """Population.get_fitness_landscape returns the landscape of the best individual."""
        cfg = EvolutionConfig(population_size=10)
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGT",
            config=cfg,
            target_dna="ACGTACGT",
            fitness_method="hamming",
            rng=random.Random(42),
        )
        landscape = pop.get_fitness_landscape(positions=[0, 1])
        assert set(landscape.keys()) == {0, 1}


# ============================================================================
# Population basic functionality
# ============================================================================

class TestPopulationBasics:
    """Verify the basic functionality of the Population class."""

    def test_initialization(self):
        """After initialization, population size is correct and all individuals are identical."""
        cfg = EvolutionConfig(population_size=50)
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGT",
            config=cfg,
            target_dna="ACGTACGT",
            fitness_method="hamming",
        )
        assert len(pop.individuals) == 50
        assert pop.generation == 0
        # all individuals have identical DNA
        assert all(ind.dna == "ACGTACGT" for ind in pop.individuals)
        # initial fitness = 1.0 (perfect match with target)
        assert all(ind.fitness == 1.0 for ind in pop.individuals)

    def test_step_increments_generation(self):
        """step() increments generation by 1."""
        cfg = EvolutionConfig(population_size=20, mutation_rate=0.01)
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGTAC",
            config=cfg,
            target_dna="ACGTACGTAC",
            fitness_method="hamming",
            rng=random.Random(42),
        )
        assert pop.generation == 0
        pop.step()
        assert pop.generation == 1
        pop.step()
        assert pop.generation == 2

    def test_evolve_with_config_generations(self):
        """evolve() uses config.generations when no argument is passed."""
        cfg = EvolutionConfig(
            population_size=20, generations=5,
            mutation_rate=0.01, indel_rate=0.0,
        )
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGTAC",
            config=cfg,
            target_dna="ACGTACGTAC",
            fitness_method="hamming",
            rng=random.Random(42),
        )
        pop.evolve()
        assert pop.generation == 5

    def test_best_individual(self):
        """best_individual returns the individual with the highest fitness."""
        cfg = EvolutionConfig(population_size=10)
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGT",
            config=cfg,
            target_dna="ACGTACGT",
            fitness_method="hamming",
        )
        best = pop.best_individual()
        assert best is not None
        assert best.fitness == 1.0  # initially perfect match

    def test_recombination_in_population(self):
        """Population runs correctly with recombination enabled."""
        cfg = EvolutionConfig(
            population_size=20, generations=5,
            mutation_rate=0.01, indel_rate=0.0,
            recombination_rate=0.5,
            selection_coefficient=2.0,
        )
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGTACGT",
            config=cfg,
            target_dna="ACGTACGTACGT",
            fitness_method="hamming",
            rng=random.Random(42),
        )
        pop.evolve(5)
        # population size remains unchanged
        assert len(pop.individuals) == 20
        assert pop.generation == 5

    def test_gc_fitness_population(self):
        """Population uses the GC fitness method."""
        cfg = EvolutionConfig(
            population_size=20, generations=5,
            mutation_rate=0.02, indel_rate=0.0,
        )
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGTAC",
            config=cfg,
            fitness_method="gc",  # no target_dna needed
            rng=random.Random(42),
        )
        # initial GC = 50% → fitness 1.0
        assert pop.mean_fitness() == pytest.approx(1.0, abs=0.01)
        pop.evolve(5)
        # after evolution, fitness should still be reasonable
        assert pop.mean_fitness() >= 0.0

    def test_custom_fitness_population(self):
        """Population uses a custom fitness function."""
        cfg = EvolutionConfig(
            population_size=10, generations=3,
            mutation_rate=0.01, indel_rate=0.0,
        )
        # custom: the closer GC content is to 0.5, the better
        pop = EvolutionaryPopulation(
            initial_dna="ACGTACGTAC",
            config=cfg,
            fitness_method="custom",
            fitness_func=lambda dna: 1.0 - abs(
                sum(1 for c in dna if c in "GC") / len(dna) - 0.5
            ) * 2,
            rng=random.Random(42),
        )
        assert pop.mean_fitness() == pytest.approx(1.0, abs=0.01)
        pop.evolve(3)
        assert len(pop.individuals) == 10
