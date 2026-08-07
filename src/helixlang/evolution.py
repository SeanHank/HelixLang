"""Evolution engine: mutation + selection + drift + recombination.

Based on real evolutionary biology data:
- E. coli genome mutation rate 1e-3 per genome per generation (Drake
  1991 Genetics 148:1667-1686)
- Base substitution rate ~2.2e-10 per nt per generation (Lee 2012
  Nature 489:527-531)
- transition:transversion ~ 3:1 (conservative bacterial value;
  Stoltzfus 2009 review, bacteria 3:1-6:1)
- indel rate ~2.2e-11 per nt per generation (about 1/10 of
  substitution)
- Synonymous/nonsynonymous substitution ratio dN/dS (Ka/Ks)
- Effective population size Ne (E. coli ~1.3e8, Arabidopsis ~4e5)
- Fitness landscape model

Module structure:
    EvolutionConfig    evolution parameters (mutation rates, population
                       size, selection coefficient, etc.)
    Individual         a single individual (DNA + fitness + mutation
                       history)
    Population         Wright-Fisher population
                       (mutation->selection->drift->recombination)
                       (= EvolutionaryPopulation alias, see below)

    mutate              introduce substitution/indel into DNA
                        (transition bias)
    select              natural selection (fitness-proportional
                        Wright-Fisher sampling)
    recombine           homologous recombination (crossover)
    calculate_fitness   fitness calculation (Hamming/CAI/GC/custom)
    fitness_landscape   fitness landscape (mutation effect at each
                        position)
    dnds_ratio         dN/dS (nonsynonymous/synonymous substitution
                       ratio, Nei-Gojobori simplified)

Pure Python, no external dependencies. CAI/translation uses
bio_data.ECOLI_CODON_USAGE.

References:
- Drake JW. Genetics 1991 148:1667-1686 (E. coli mutation rate)
- Lee H et al. Nature 2012 489:527-531 (base substitution rate 2.2e-10)
- Stoltzfus A & Norris RW. Mol Biol Evol 2016 33:595-604 (transition
  bias)
- Nei M & Gojobori T. Mol Biol Evol 1986 3:418-426 (dN/dS)
- Sharp PM et al. Nucleic Acids Res 1987 15:1281-1295 (CAI)
"""
from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

# numpy optional (degrades to pure Python if missing)
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from helixlang.seq_utils import gc_content as _gc_content

# ============================================================================
# Real evolution parameters (paper measurements)
# ============================================================================

# E. coli base substitution rate (Lee 2012 Nature 489:527-531)
# ~2.2e-10 per base per generation (genome ~4.6 Mb -> ~1e-3
# substitutions/genome per generation, consistent with Drake 1991)
E_COLI_SUBSTITUTION_RATE = 2.2e-10

# indel rate (Lee 2012; indel:substitution ~ 0.01-0.10, conservative
# 0.10)
# indels are 10x rarer than substitutions (real bacterial values
# 10-100x, conservative 10x)
E_COLI_INDEL_RATE = 2.2e-11

# transition:transversion ~ 3:1 (conservative bacterial value)
# Stoltzfus 2009 review: eukaryotes ~2:1, bacteria 3:1-6:1 (affected by
# CpG/methylation), conservative 3.0
TRANSITION_TRANSVERSION_RATIO = 3.0

# effective population size Ne (Hartl & Clark 2007 Principles of
# Population Genetics)
E_COLI_NE = 1.3e8      # E. coli
ARABIDOPSIS_NE = 4e5   # Arabidopsis
HUMAN_NE = 1e4         # human
DROSOPHILA_NE = 1e6    # fruit fly

# mutation spectrum: A->G / C->T are the most common transitions
# purine interconversions (A<->G) and pyrimidine interconversions
# (C<->T) make up ~2/3 of substitutions
_TRANSITIONS: dict[str, str] = {"A": "G", "G": "A", "C": "T", "T": "C"}
_TRANSVERSIONS: dict[str, tuple[str, str]] = {
    "A": ("C", "T"), "G": ("C", "T"),
    "C": ("A", "G"), "T": ("A", "G"),
}


# ============================================================================
# Configuration and individuals
# ============================================================================

@dataclass(slots=True)
class EvolutionConfig:
    """Evolution parameters (based on real paper defaults).

    mutation_rate: base substitution rate per nt per generation
                   default 2.2e-10 (Lee 2012, E. coli)
    indel_rate:   indel rate per nt per generation
                   default 2.2e-11 (Lee 2012;
                   indel:substitution ~ 0.10, conservative bacterial
                   value)
    transition_transversion_ratio: transition:transversion ratio
                   default 3.0 (conservative bacterial value; Stoltzfus
                   2009, bacteria 3:1-6:1)
    population_size: population size (N for Wright-Fisher)
    generations:     number of evolution generations
    recombination_rate: sexual recombination rate (0=asexual,
                        1=recombine every generation)
    selection_coefficient: selection coefficient s
                  s > 0: positive selection (high fitness preferred)
                  s = 0: neutral drift
                  s < 0: negative selection (low fitness preferred)
    """
    mutation_rate: float = E_COLI_SUBSTITUTION_RATE
    indel_rate: float = E_COLI_INDEL_RATE
    transition_transversion_ratio: float = TRANSITION_TRANSVERSION_RATIO
    population_size: int = 1000
    generations: int = 100
    recombination_rate: float = 0.0
    selection_coefficient: float = 0.01


@dataclass(slots=True)
class Individual:
    """A single individual.

    dna:        DNA sequence (ACGT only)
    fitness:    fitness (usually [0, 1])
    generation: birth generation (0 = initial population)
    mutations:  mutation history (list of strings, e.g. "sub@5:A->G")
    """
    dna: str
    fitness: float
    generation: int
    mutations: list[str] = field(default_factory=list)


# ============================================================================
# Mutation
# ============================================================================

def _substitute(base: str, rng: random.Random,
                transition_bias: float) -> str:
    """Substitute a base with transition/transversion bias.

    transition_bias: probability of choosing a transition.
        ratio = 2:1 -> transition_bias = 2/3
        ratio = 6:1 -> transition_bias = 6/7
    """
    if rng.random() < transition_bias:
        return _TRANSITIONS[base]
    else:
        return rng.choice(_TRANSVERSIONS[base])


def mutate(dna: str,
           mutation_rate: float = E_COLI_SUBSTITUTION_RATE,
           indel_rate: float = E_COLI_INDEL_RATE,
           ratio: float = TRANSITION_TRANSVERSION_RATIO,
           rng: random.Random | None = None
           ) -> tuple[str, list[str]]:
    """Introduce mutations into a DNA sequence.

    Each base is treated independently:
    - insert a random base with probability indel_rate/2 (keeping the
      original base)
    - delete with probability indel_rate/2
    - substitute with probability mutation_rate
      (transition:transversion = ratio:1)
    - otherwise keep it

    Args:
        dna:           the original DNA sequence
        mutation_rate: per-nt substitution probability
        indel_rate:    per-nt indel probability (insertion + deletion
                       each account for half)
        ratio:         transition:transversion ratio (e.g. 2.0 = 2:1)
        rng:           random number generator (None creates a new one)

    Returns:
        (mutated_dna, mutation_list)
        mutation_list element formats:
            "sub@{pos}:{orig}->{new}"   substitution
            "ins@{pos}:{base}"          insertion
            "del@{pos}:{base}"          deletion

    With real parameters (mutation_rate=2.2e-10), a 1 Mb sequence
    expects ~2.2e-4 substitutions per generation.
    For statistical validation it is recommended to raise mutation_rate
    (e.g. 0.01).
    """
    if rng is None:
        rng = random.Random()
    if not dna:
        return "", []
    # P(transition | substitution) = ratio / (ratio + 1)
    transition_bias = ratio / (ratio + 1.0) if ratio > 0 else 0.5
    bases = "ACGT"
    out: list[str] = []
    mutations: list[str] = []
    n = len(dna)
    i = 0
    while i < n:
        base = dna[i]
        r = rng.random()
        if r < indel_rate / 2:
            # insertion: insert a random base before the current
            # position, then keep the original base
            new_base = rng.choice(bases)
            out.append(new_base)
            out.append(base)
            mutations.append(f"ins@{i}:{new_base}")
            i += 1
        elif r < indel_rate:
            # deletion: skip the current base
            mutations.append(f"del@{i}:{base}")
            i += 1
        elif r < indel_rate + mutation_rate:
            # substitution (transition/transversion bias)
            new_base = _substitute(base, rng, transition_bias)
            out.append(new_base)
            mutations.append(f"sub@{i}:{base}->{new_base}")
            i += 1
        else:
            out.append(base)
            i += 1
    return "".join(out), mutations


# ============================================================================
# Batch mutation (numpy-vectorized random number generation)
# ============================================================================

def _mutate_with_randoms(dna: str,
                         r_array: np.ndarray,
                         np_rng: np.random.Generator,
                         mutation_rate: float,
                         indel_rate: float,
                         ratio: float) -> tuple[str, list[str]]:
    """Mutate a single DNA using pre-generated random arrays (numpy
    vectorized path).

    Semantically consistent with :func:`mutate`, but the random numbers
    are generated in bulk via numpy (``r_array`` is a 1D array of
    ``[0,1)`` values with length >= ``len(dna)``), reducing the per-call
    overhead of Python-side ``rng.random()``. ``np_rng`` is used to pick
    bases for insertion/substitution.
    """
    if not dna:
        return "", []
    transition_bias = ratio / (ratio + 1.0) if ratio > 0 else 0.5
    bases = "ACGT"
    out: list[str] = []
    mutations: list[str] = []
    n = len(dna)
    i = 0
    while i < n:
        base = dna[i]
        r = float(r_array[i])
        if r < indel_rate / 2:
            # insertion
            new_base = bases[int(np_rng.integers(0, 4))]
            out.append(new_base)
            out.append(base)
            mutations.append(f"ins@{i}:{new_base}")
            i += 1
        elif r < indel_rate:
            # deletion
            mutations.append(f"del@{i}:{base}")
            i += 1
        elif r < indel_rate + mutation_rate:
            # substitution (transition/transversion bias)
            if float(np_rng.random()) < transition_bias:
                new_base = _TRANSITIONS[base]
            else:
                tv = _TRANSVERSIONS[base]
                new_base = tv[int(np_rng.integers(0, 2))]
            out.append(new_base)
            mutations.append(f"sub@{i}:{base}->{new_base}")
            i += 1
        else:
            out.append(base)
            i += 1
    return "".join(out), mutations


def mutate_batch(individuals: list[Individual],
                 mutation_rate: float,
                 indel_rate: float,
                 ratio: float,
                 rng: random.Random) -> list[tuple[str, list[str]]]:
    """Mutate a population in batch (numpy-vectorized random number
    generation).

    Generates an (N x L) random number matrix in one shot with numpy,
    then applies :func:`_mutate_with_randoms` to each individual. When
    numpy is unavailable, falls back to per-individual :func:`mutate`.

    Returns ``[(mutated_dna, mutation_list), ...]``, length = number of
    input individuals.
    """
    n = len(individuals)
    if n == 0:
        return []

    if not _HAS_NUMPY:
        # pure Python fallback
        results: list[tuple[str, list[str]]] = []
        for ind in individuals:
            results.append(mutate(
                ind.dna,
                mutation_rate=mutation_rate,
                indel_rate=indel_rate,
                ratio=ratio,
                rng=rng,
            ))
        return results

    # numpy path: derive a numpy seed from the Python rng, generate the
    # random matrix in bulk
    np_seed = rng.randrange(2 ** 32)
    np_rng = np.random.default_rng(np_seed)

    # build the random matrix from the max DNA length (unequal lengths
    # still work; extra random numbers are ignored)
    max_len = max(len(ind.dna) for ind in individuals)
    if max_len == 0:
        return [("", []) for _ in individuals]

    # generate the (N x max_len) random number matrix in one shot
    r_matrix = np_rng.random((n, max_len))

    results = []
    for i, ind in enumerate(individuals):
        L = len(ind.dna)
        new_dna, muts = _mutate_with_randoms(
            ind.dna,
            r_matrix[i, :L] if L > 0 else r_matrix[i, :0],
            np_rng,
            mutation_rate,
            indel_rate,
            ratio,
        )
        results.append((new_dna, muts))
    return results


# ============================================================================
# Selection (Wright-Fisher sampling)
# ============================================================================

def select(population: list[Individual],
           selection_coefficient: float,
           rng: random.Random | None = None) -> list[Individual]:
    """Natural selection (fitness-proportional sampling) - Wright-Fisher
    model.

    The probability that each individual is chosen as a parent of the
    next generation is proportional to its relative fitness:
        w_i = 1 + s * (fitness_i - mean_fitness)

    - s > 0: positive selection, high-fitness individuals preferred
      (natural selection)
    - s = 0: neutral drift (w_i = 1, uniform sampling)
    - s < 0: negative selection, low-fitness individuals preferred

    Balancing selection can be implemented with a custom fitness
    function (e.g. making intermediate fitness highest).

    Sampling with replacement; return size = input population size.

    Args:
        population:            the current population
        selection_coefficient: selection coefficient s
        rng:                   random number generator

    Returns:
        list of selected individuals (length = len(population),
        may contain duplicate references)
    """
    if rng is None:
        rng = random.Random()
    n = len(population)
    if n == 0:
        return []
    if n == 1:
        return [population[0]]
    # compute relative fitness weights (vectorized: numpy computes all
    # weights at once)
    if _HAS_NUMPY:
        fits = np.fromiter(
            (ind.fitness for ind in population), dtype=float, count=n
        )
        mean_fit = float(fits.mean())
        weights = 1.0 + selection_coefficient * (fits - mean_fit)
        # ensure non-negative (extreme s can produce negative values)
        weights = np.where(weights <= 0.0, 1e-10, weights)
        probs = weights / weights.sum()
        # use numpy random.choice instead of a pure Python sampling loop
        # derive a numpy seed from the Python rng to stay reproducible
        np_seed = rng.randrange(2 ** 32)
        np_rng = np.random.default_rng(np_seed)
        selected_idx = np_rng.choice(n, size=n, replace=True, p=probs)
        return [population[int(i)] for i in selected_idx]
    # pure Python fallback
    mean_fit = sum(ind.fitness for ind in population) / n
    s = selection_coefficient
    weights_py: list[float] = []
    for ind in population:
        w = 1.0 + s * (ind.fitness - mean_fit)
        # ensure non-negative (extreme s can produce negative values)
        if w <= 0.0:
            w = 1e-10
        weights_py.append(w)
    # Wright-Fisher sampling with replacement
    selected_idx_py = rng.choices(range(n), weights=weights_py, k=n)
    return [population[i] for i in selected_idx_py]


# ============================================================================
# Recombination
# ============================================================================

def recombine(parent1: str, parent2: str,
              rate: float,
              rng: random.Random | None = None) -> str:
    """Homologous recombination (crossover).

    Performs 1-3 point crossover on two equal-length (or similar) DNA
    sequences with probability rate.
    rate = 0 -> returns parent1
    rate = 1 -> always recombine

    Args:
        parent1, parent2: parent DNA
        rate:             recombination probability (0..1)
        rng:              random number generator

    Returns:
        recombined DNA (length = max(len(p1), len(p2)); recombined
        segments come from min(len(p1), len(p2)), the tail comes from
        the longer parent)
    """
    if rng is None:
        rng = random.Random()
    if not parent1:
        return parent2
    if not parent2:
        return parent1
    if rate <= 0 or rng.random() > rate:
        return parent1
    # recombined segment length = shorter parent
    n = min(len(parent1), len(parent2))
    if n < 2:
        return parent1
    # 1-3 crossover points (weights favor fewer crossovers)
    n_crossover = rng.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
    n_crossover = min(n_crossover, n - 1)
    # crossover point positions (no overlap)
    points = sorted(rng.sample(range(1, n), n_crossover))
    # build alternating segments
    bounds = [0] + points + [n]
    out: list[str] = []
    src = parent1
    for k in range(len(bounds) - 1):
        out.append(src[bounds[k]:bounds[k + 1]])
        # switch source
        src = parent2 if src is parent1 else parent1
    # tail: remainder of the longer parent
    longer = parent1 if len(parent1) >= len(parent2) else parent2
    if len(longer) > n:
        out.append(longer[n:])
    return "".join(out)


# ============================================================================
# Fitness calculation
# ============================================================================

def calculate_fitness(dna: str,
                      target_dna: str | None = None,
                      method: str = "hamming",
                      custom_func: Callable[[str], float] | None = None,
                      ) -> float:
    """Fitness calculation.

    method:
        "hamming": Hamming similarity to target_dna (match rate)
                   requires target_dna.
                   fitness = matches / max(len)
        "cai":     E. coli CAI (codon adaptation index, Sharp 1987)
                   range [0, 1], 1.0 = all optimal codons
        "gc":      closeness of GC content to 0.5 (1.0 = 50% GC,
                   0.0 = 0% or 100%)
        "custom":  custom function custom_func(dna) -> float

    Args:
        dna:         the DNA to evaluate
        target_dna:  target sequence (required for the hamming method)
        method:      fitness calculation method
        custom_func: custom fitness function (required when
                     method="custom")

    Returns:
        fitness value (usually [0, 1])
    """
    if method == "hamming":
        if target_dna is None:
            raise ValueError("hamming method requires target_dna")
        if len(dna) == 0 and len(target_dna) == 0:
            return 1.0
        max_len = max(len(dna), len(target_dna))
        if max_len == 0:
            return 1.0
        min_len = min(len(dna), len(target_dna))
        # vectorized Hamming distance calculation (numpy path)
        if _HAS_NUMPY and min_len > 0:
            a = np.frombuffer(dna[:min_len].encode("ascii"), dtype=np.uint8)
            b = np.frombuffer(
                target_dna[:min_len].encode("ascii"), dtype=np.uint8
            )
            matches = int(np.count_nonzero(a == b))
        else:
            matches = sum(
                1 for i in range(min_len) if dna[i] == target_dna[i]
            )
        return matches / max_len
    elif method == "cai":
        # use biocodec's CAI implementation (pure Python, depends on
        # ECOLI_CODON_USAGE)
        from helixlang.biocodec import codon_adaptation_index_full
        return codon_adaptation_index_full(dna)
    elif method == "gc":
        if not dna:
            return 0.0
        gc = _gc_content(dna)
        # 0.5 GC -> 1.0; 0.0 or 1.0 -> 0.0
        return 1.0 - abs(gc - 0.5) * 2.0
    elif method == "custom":
        if custom_func is None:
            raise ValueError("custom method requires custom_func")
        return float(custom_func(dna))
    else:
        raise ValueError(f"unknown method {method!r}; "
                         f"available: hamming, cai, gc, custom")


# ============================================================================
# Fitness landscape
# ============================================================================

def fitness_landscape(dna: str,
                      target: str | None = None,
                      positions: list[int] | None = None,
                      ) -> dict[int, dict[str, float]]:
    """Compute the fitness landscape (mutation effect at each
    position).

    For each specified position, try mutating it to A/C/G/T and compute
    the fitness of the mutated sequence (hamming method).

    Returns:
        {position: {base: fitness}}
        where base includes the original base (its fitness = fitness of
        the original sequence)

    Args:
        dna:       the original DNA
        target:    target sequence (None = use dna itself, so all
                   mutations lower fitness)
        positions: positions to evaluate (None = all positions)
    """
    if not dna:
        return {}
    if target is None:
        target = dna
    if positions is None:
        positions = list(range(len(dna)))
    landscape: dict[int, dict[str, float]] = {}
    for pos in positions:
        if pos < 0 or pos >= len(dna):
            continue
        original = dna[pos]
        per_pos: dict[str, float] = {}
        for base in "ACGT":
            if base == original:
                mutated = dna
            else:
                mutated = dna[:pos] + base + dna[pos + 1:]
            per_pos[base] = calculate_fitness(mutated, target, method="hamming")
        landscape[pos] = per_pos
    return landscape


# ============================================================================
# dN/dS (nonsynonymous/synonymous substitution ratio)
# ============================================================================

def _aa_of_codon(codon: str) -> str:
    """Translate a single codon using ECOLI_CODON_USAGE (no BioPython
    dependency)."""
    from helixlang.bio_data import ECOLI_CODON_USAGE
    if codon in ECOLI_CODON_USAGE:
        return ECOLI_CODON_USAGE[codon][0]
    return "X"


def dnds_ratio(dna: str, ancestral: str,
               method: str = "nei_gojobori") -> dict:
    """Compute dN/dS (nonsynonymous/synonymous substitution ratio).

    Steps:
    1. Align dna and ancestral by codon
    2. For each ancestral codon, count synonymous sites S and
       nonsynonymous sites N (examining the 3 possible mutations at each
       position, classified by whether they change the amino acid)
    3. Count observed synonymous substitutions Sd and nonsynonymous
       substitutions Nd
    4. dS = Sd / S_total, dN = Nd / N_total
    5. dN/dS = dN / dS

    Interpretation (biological meaning of dN/dS):
        dN/dS << 1: strong purifying selection (most nonsynonymous
                    mutations are deleterious)
        dN/dS ~ 1:  neutral evolution
        dN/dS > 1:  positive selection (nonsynonymous mutations are
                    beneficial)

    Args:
        dna:       derived sequence
        ancestral: ancestral sequence
        method:    "nei_gojobori" (default; Nei-Gojobori 1986 site
                   counting) or "codeml" / "m0" (M0 one-ratio
                   codon-substitution maximum-likelihood fit,
                   :func:`dnds_codeml`)

    Returns:
        dict containing dN, dS, dNdS, substitution counts, site counts,
        and interpretation
    """
    if method in ("codeml", "m0"):
        return dnds_codeml(dna, ancestral)
    if method != "nei_gojobori":
        raise ValueError(
            f"unknown method {method!r}; available: nei_gojobori, codeml, m0"
        )
    if not dna or not ancestral:
        return {
            "dN": 0.0, "dS": 0.0, "dNdS": 0.0,
            "nonsyn_substitutions": 0, "syn_substitutions": 0,
            "syn_sites": 0.0, "nonsyn_sites": 0.0,
            "interpretation": "no data (empty sequence)",
        }
    # align by codon (multiple of 3 of the shorter length)
    n = min(len(dna), len(ancestral)) // 3 * 3
    n_codon = n // 3

    S_total = 0.0   # total synonymous sites
    N_total = 0.0   # total nonsynonymous sites
    Sd = 0          # observed synonymous substitutions
    Nd = 0          # observed nonsynonymous substitutions

    for i in range(n_codon):
        c1 = ancestral[i * 3:(i + 1) * 3].upper()
        c2 = dna[i * 3:(i + 1) * 3].upper()
        aa1 = _aa_of_codon(c1)

        # count synonymous/nonsynonymous sites of c1
        # for each position j, examine the 3 possible mutations and
        # count synonymous mutations
        syn_count = 0
        for j in range(3):
            orig = c1[j]
            for new_base in "ACGT":
                if new_base == orig:
                    continue
                mut_codon = c1[:j] + new_base + c1[j + 1:]
                if _aa_of_codon(mut_codon) == aa1:
                    syn_count += 1
        # each position contributes (syn_mutations / 3) synonymous sites
        S_codon = syn_count / 3.0
        N_codon = 3.0 - S_codon
        S_total += S_codon
        N_total += N_codon

        # count actual substitutions
        if c1 != c2:
            aa2 = _aa_of_codon(c2)
            if aa1 == aa2:
                Sd += 1
            else:
                Nd += 1

    # dS, dN (normalized by sites)
    dS = Sd / S_total if S_total > 0 else 0.0
    dN = Nd / N_total if N_total > 0 else 0.0

    # dN/dS
    if dS == 0.0:
        if dN == 0.0:
            dNdS = 0.0
        else:
            dNdS = float("inf")
    else:
        dNdS = dN / dS

    # interpretation
    if Sd == 0 and Nd == 0:
        interpretation = "no substitutions observed"
    elif dN == 0.0 and Sd > 0:
        interpretation = "purifying selection (dN=0)"
    elif math.isinf(dNdS):
        interpretation = "positive selection (dS=0 but dN>0)"
    elif dNdS < 0.5:
        interpretation = "strong purifying selection"
    elif dNdS < 1.0:
        interpretation = "weak purifying selection"
    elif dNdS <= 1.5:
        interpretation = "neutral evolution"
    else:
        interpretation = "positive selection"

    return {
        "dN": dN,
        "dS": dS,
        "dNdS": dNdS,
        "nonsyn_substitutions": Nd,
        "syn_substitutions": Sd,
        "syn_sites": S_total,
        "nonsyn_sites": N_total,
        "interpretation": interpretation,
    }


# ============================================================================
# M0 one-ratio codon-substitution model (Goldman & Yang 1994)
# ============================================================================
# Optional maximum-likelihood dN/dS estimate for a two-sequence
# comparison.  The codon-substitution process is the classical M0 model
# with equal codon frequencies and a transition/transversion rate ratio
# kappa; the nonsynonymous/synonymous rate ratio omega is fitted by ML
# jointly with the branch length t.  A first-hit (Poisson) approximation
# gives each site's probability from the competing single-step rates out
# of the ancestral codon, avoiding a 61x61 matrix exponentiation in pure
# Python.

_TS_TRANSITIONS = {
    ("A", "G"), ("G", "A"), ("C", "T"), ("T", "C"),
}


def _codon_mutation_table() -> dict[str, list[tuple[str, bool, bool]]]:
    """Per sense-codon single-step mutations.

    Returns ``{codon: [(target_codon, is_transition, is_synonymous)]}``
    for every sense codon and each of its 3x3 single-base changes that
    lands on a sense codon.
    """
    table: dict[str, list[tuple[str, bool, bool]]] = {}
    for i in range(64):
        a = "TCAG"[i // 16]
        b = "TCAG"[(i // 4) % 4]
        c = "TCAG"[i % 4]
        codon = a + b + c
        aa_i = _aa_of_codon(codon)
        if aa_i == "X":
            continue
        out: list[tuple[str, bool, bool]] = []
        for pos in range(3):
            for nb in "TCAG":
                if nb == codon[pos]:
                    continue
                mut = codon[:pos] + nb + codon[pos + 1:]
                aa_j = _aa_of_codon(mut)
                if aa_j == "X":
                    continue
                is_ts = (codon[pos], nb) in _TS_TRANSITIONS
                out.append((mut, is_ts, aa_j == aa_i))
        table[codon] = out
    return table


_CODON_MUTATIONS = _codon_mutation_table()


def dnds_codeml(dna: str, ancestral: str,
                kappa: float = 3.0) -> dict:
    """M0 one-ratio codon-substitution dN/dS (Goldman & Yang 1994
    Mol Biol Evol 11:725-736).

    Fits the nonsynonymous/synonymous rate ratio ``omega`` and the
    branch length ``t`` (expected substitutions per codon) by maximum
    likelihood under the M0 codon-substitution process (equal codon
    frequencies, transition/transversion ratio ``kappa``; default 3, the
    mammalian ts/tv ratio).  A first-hit Poisson approximation per site
    keeps the fit dependency-free:

    - identical site:      P = exp(-L_i(omega) * t)
    - single-step change:  P = q_ij(omega) * t * exp(-L_i(omega) * t)

    where ``L_i(omega)`` is the total rate out of the ancestral codon
    and ``q_ij(omega)`` the ancestral->derived rate (kappa for
    transitions, x1 for transversions, x1 synonymous / x omega
    nonsynonymous).

    Args:
        dna:       derived sequence
        ancestral: ancestral sequence
        kappa:     transition/transversion rate ratio

    Returns:
        dict with M0 ``omega`` (= dN/dS), branch length ``t``, lnL,
        likelihood-ratio test vs the neutral (omega=1) model, and the
        same count-based keys/interpretation as :func:`dnds_ratio`
    """
    n = min(len(dna), len(ancestral)) // 3 * 3
    sites: list[tuple[str, str]] = []
    for i in range(n // 3):
        c1 = ancestral[i * 3:i * 3 + 3].upper()
        c2 = dna[i * 3:i * 3 + 3].upper()
        if c1 in _CODON_MUTATIONS and c2 in _CODON_MUTATIONS:
            sites.append((c1, c2))

    empty = {
        "method": "M0", "omega": 1.0, "t": 0.0,
        "dN": 0.0, "dS": 0.0, "dNdS": 0.0,
        "nonsyn_substitutions": 0, "syn_substitutions": 0,
        "syn_sites": 0.0, "nonsyn_sites": 0.0,
        "lnL": 0.0, "lnL_neutral": 0.0,
        "lrt_stat": 0.0, "lrt_p": 1.0,
        "interpretation": "no data (empty sequence)",
    }
    if not sites:
        return empty

    # per-codon rate summary at omega and kappa
    def total_rate(c1: str, omega: float) -> float:
        total = 0.0
        for _target, is_ts, is_syn in _CODON_MUTATIONS[c1]:
            q = kappa if is_ts else 1.0
            if not is_syn:
                q *= omega
            total += q
        return total

    # precompute the pair rates for the observed sites
    pair_q: dict[tuple[str, str], float] = {}
    pair_nonsyn: dict[tuple[str, str], bool] = {}
    for c1, c2 in sites:
        if c1 == c2:
            continue
        found = False
        for target, is_ts, is_syn in _CODON_MUTATIONS[c1]:
            if target == c2:
                pair_q[(c1, c2)] = kappa if is_ts else 1.0
                pair_nonsyn[(c1, c2)] = not is_syn
                found = True
                break
        if not found:
            # differs at >= 2 positions: model as one nonsynonymous
            # transversion-equivalent step (conservative fallback)
            pair_q[(c1, c2)] = 1.0
            pair_nonsyn[(c1, c2)] = True

    def log_likelihood(omega: float, t: float) -> float:
        ll = 0.0
        for c1, c2 in sites:
            if c1 == c2:
                ll -= total_rate(c1, omega) * t
            else:
                q = pair_q.get((c1, c2), 1.0)
                if pair_nonsyn.get((c1, c2), True):
                    q *= omega
                ll += math.log(q) + math.log(t) - total_rate(c1, omega) * t
        return ll

    # bounded golden-section maximization (stdlib)
    def _maximize(func: Callable[[float], float], lo: float, hi: float,
                  tol: float = 1e-7, max_iter: int = 200) -> tuple[float, float]:
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, fd = func(c), func(d)
        best_x, best_f = (c, fc) if fc >= fd else (d, fd)
        for _ in range(max_iter):
            if fc > fd:
                b, d = d, c
                c = b - gr * (b - a)
                fd, fc = fc, func(c)
            else:
                a, c = c, d
                d = a + gr * (b - a)
                fc, fd = fd, func(d)
            if fc >= fd:
                if fc > best_f:
                    best_x, best_f = c, fc
            elif fd > best_f:
                best_x, best_f = d, fd
            if b - a < tol:
                break
        return best_x, best_f

    # nested fit: outer omega in [0, 5], inner t in [0, 10]
    def fit_for_omega(omega: float) -> float:
        t_best, ll = _maximize(lambda t: log_likelihood(omega, t), 0.0, 10.0)
        return ll

    omega_best, _ = _maximize(fit_for_omega, 0.0, 5.0)
    t_best, _ = _maximize(lambda t: log_likelihood(omega_best, t), 0.0, 10.0)
    # one coordinate ascent round
    omega_best, _ = _maximize(lambda w: log_likelihood(w, t_best), 0.0, 5.0)
    t_best, _ = _maximize(lambda t: log_likelihood(omega_best, t), 0.0, 10.0)
    ll_alt = log_likelihood(omega_best, t_best)

    # no-substitution case: likelihood maximized at t=0, omega
    # unidentifiable -> report the neutral M0 and note no signal
    if not any(c1 != c2 for c1, c2 in sites):
        empty.update({
            "interpretation": "no substitutions observed (M0)",
            "omega": 1.0,
            "t": 0.0,
            "dS": 0.0,
        })
        return empty

    # neutral model (omega = 1)
    t_neutral, ll_neutral = _maximize(
        lambda t: log_likelihood(1.0, t), 0.0, 10.0)

    # likelihood-ratio test, df = 1 (chi-square survival via erfc)
    lrt = 2.0 * max(0.0, ll_alt - ll_neutral)
    p_value = math.erfc(math.sqrt(lrt / 2.0)) if lrt > 0 else 1.0

    # count-based site/substitution summary (Nei-Gojobori counts) so the
    # return shape matches dnds_ratio()
    count = dnds_ratio(dna, ancestral, method="nei_gojobori")

    if omega_best < 0.5:
        interpretation = "strong purifying selection (M0 omega<0.5)"
    elif omega_best < 1.0:
        interpretation = "weak purifying selection (M0 omega<1)"
    elif omega_best <= 1.5:
        interpretation = "neutral evolution (M0 omega≈1)"
    else:
        interpretation = "positive selection (M0 omega>1.5)"
    if p_value < 0.05 and omega_best > 1.0:
        interpretation += "; LRT significant vs neutral model"

    return {
        "method": "M0",
        "omega": omega_best,
        "t": t_best,
        "dN": omega_best * count["dS"],
        "dS": count["dS"],
        "dNdS": omega_best,
        "nonsyn_substitutions": count["nonsyn_substitutions"],
        "syn_substitutions": count["syn_substitutions"],
        "syn_sites": count["syn_sites"],
        "nonsyn_sites": count["nonsyn_sites"],
        "lnL": ll_alt,
        "lnL_neutral": ll_neutral,
        "lrt_stat": lrt,
        "lrt_p": p_value,
        "kappa": kappa,
        "interpretation": interpretation,
    }


# ============================================================================
# Population (Wright-Fisher model)
# ============================================================================

class EvolutionaryPopulation:
    """Wright-Fisher population: mutation -> selection -> drift ->
    recombination.

    Note: this class was originally named ``Population``; to avoid name
    collision with the multicellular spatial population ``CellPopulation``
    (also originally called ``Population``) in ``helixlang.population``,
    it was renamed to ``EvolutionaryPopulation``. The backward-compatible
    alias ``Population = EvolutionaryPopulation`` is at the end of the
    module (DeprecationWarning is noted in the comment there).

    Usage example:
        cfg = EvolutionConfig(mutation_rate=0.01, population_size=100,
                              generations=50, selection_coefficient=5.0)
        pop = EvolutionaryPopulation(initial_dna="ACGT"*10, config=cfg,
                                     target_dna="ACGT"*10,
                                     fitness_method="hamming")
        pop.evolve()
        stats = pop.get_generation_stats()

    Attributes:
        individuals: list of individuals in the current population
        generation:  current generation (0 = initial)
        history:     list of per-generation statistics dicts
    """

    def __init__(self,
                 initial_dna: str,
                 config: EvolutionConfig | None = None,
                 target_dna: str | None = None,
                 fitness_method: str = "hamming",
                 fitness_func: Callable[[str], float] | None = None,
                 rng: random.Random | None = None,
                 ) -> None:
        """Initialize the population.

        Args:
            initial_dna:    initial DNA sequence (all individuals start
                            identical)
            config:         evolution parameters (None = default
                            EvolutionConfig)
            target_dna:     target sequence (required for the hamming
                            method)
            fitness_method: fitness calculation method
                            (hamming/cai/gc/custom)
            fitness_func:   custom fitness function (used when
                            method="custom")
            rng:            random number generator (None = create new)
        """
        self.config = config if config is not None else EvolutionConfig()
        self.target_dna = target_dna
        self.fitness_method = fitness_method
        self.fitness_func = fitness_func
        self._rng = rng if rng is not None else random.Random()
        self.generation = 0
        self.history: list[dict] = []
        # initialize the population (all individuals start with the same
        # DNA)
        initial_fit = self._compute_fitness(initial_dna)
        self.individuals: list[Individual] = [
            Individual(
                dna=initial_dna,
                fitness=initial_fit,
                generation=0,
                mutations=[],
            )
            for _ in range(self.config.population_size)
        ]
        self._record_stats()

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _compute_fitness(self, dna: str) -> float:
        """Compute DNA fitness according to the configuration."""
        if self.fitness_method == "custom":
            return calculate_fitness(
                dna, self.target_dna, self.fitness_method, self.fitness_func
            )
        return calculate_fitness(
            dna, self.target_dna, self.fitness_method
        )

    def _record_stats(self) -> None:
        """Record the current generation statistics to history."""
        fits = [ind.fitness for ind in self.individuals]
        n = len(fits)
        if n == 0:
            stats = {
                "generation": self.generation,
                "population_size": 0,
                "mean_fitness": 0.0,
                "max_fitness": 0.0,
                "min_fitness": 0.0,
                "diversity": 0.0,
            }
        else:
            stats = {
                "generation": self.generation,
                "population_size": n,
                "mean_fitness": sum(fits) / n,
                "max_fitness": max(fits),
                "min_fitness": min(fits),
                "diversity": self.get_diversity(),
            }
        self.history.append(stats)

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def step(self) -> None:
        """Run one generation of evolution (mutation -> selection ->
        drift -> recombination).

        Wright-Fisher model:
        1. introduce mutations into each individual's DNA
        2. sample N with replacement proportionally to fitness
           (selection + drift)
        3. if recombination_rate > 0, pair up randomly and do
           crossover
        """
        cfg = self.config
        rng = self._rng
        next_gen = self.generation + 1

        # 1. mutation: each individual produces a mutated offspring
        # large populations use the numpy batch random path; small
        # populations use pure Python mutate()
        pop_size = len(self.individuals)
        if _HAS_NUMPY and pop_size > 100:
            batch_results = mutate_batch(
                self.individuals,
                mutation_rate=cfg.mutation_rate,
                indel_rate=cfg.indel_rate,
                ratio=cfg.transition_transversion_ratio,
                rng=rng,
            )
            mutated: list[Individual] = []
            for ind, (new_dna, muts) in zip(self.individuals, batch_results, strict=False):
                new_fit = self._compute_fitness(new_dna)
                mutated.append(Individual(
                    dna=new_dna,
                    fitness=new_fit,
                    generation=next_gen,
                    mutations=ind.mutations + muts,
                ))
        else:
            mutated = []
            for ind in self.individuals:
                new_dna, muts = mutate(
                    ind.dna,
                    mutation_rate=cfg.mutation_rate,
                    indel_rate=cfg.indel_rate,
                    ratio=cfg.transition_transversion_ratio,
                    rng=rng,
                )
                new_fit = self._compute_fitness(new_dna)
                mutated.append(Individual(
                    dna=new_dna,
                    fitness=new_fit,
                    generation=next_gen,
                    mutations=ind.mutations + muts,
                ))

        # 2. selection + drift (Wright-Fisher sampling with
        #    replacement)
        selected = select(mutated, cfg.selection_coefficient, rng)

        # 3. recombination (sexual reproduction, random pairing and
        #    crossover)
        if cfg.recombination_rate > 0 and len(selected) >= 2:
            recombined: list[Individual] = []
            indices = list(range(len(selected)))
            rng.shuffle(indices)
            i = 0
            while i + 1 < len(indices):
                p1 = selected[indices[i]]
                p2 = selected[indices[i + 1]]
                # each parent pair produces 2 offspring (keeps
                # population size)
                child1_dna = recombine(
                    p1.dna, p2.dna, cfg.recombination_rate, rng
                )
                child2_dna = recombine(
                    p2.dna, p1.dna, cfg.recombination_rate, rng
                )
                child1_fit = self._compute_fitness(child1_dna)
                child2_fit = self._compute_fitness(child2_dna)
                recombined.append(Individual(
                    dna=child1_dna,
                    fitness=child1_fit,
                    generation=next_gen,
                    mutations=p1.mutations,  # simplified: inherit the
                                             # parent's mutation history
                ))
                recombined.append(Individual(
                    dna=child2_dna,
                    fitness=child2_fit,
                    generation=next_gen,
                    mutations=p2.mutations,
                ))
                i += 2
            # odd count: keep the last one as is
            if i < len(indices):
                recombined.append(selected[indices[i]])
            self.individuals = recombined
        else:
            self.individuals = selected

        self.generation = next_gen
        self._record_stats()

    def evolve(self, generations: int | None = None) -> None:
        """Run multiple generations of evolution.

        Args:
            generations: number of evolution generations (None = use
                         config.generations)
        """
        n = generations if generations is not None else self.config.generations
        for _ in range(n):
            self.step()

    def get_fitness_landscape(self,
                              positions: list[int] | None = None,
                              ) -> dict[int, dict[str, float]]:
        """Return the fitness landscape of the current best individual.

        Args:
            positions: positions to evaluate (None = all positions)

        Returns:
            {position: {base: fitness}}
        """
        if not self.individuals:
            return {}
        best = max(self.individuals, key=lambda ind: ind.fitness)
        return fitness_landscape(best.dna, self.target_dna, positions)

    def get_diversity(self) -> float:
        """Return population diversity (Shannon entropy, normalized to
        [0, 1]).

        For all distinct DNA sequences in the population, compute the
        frequency p_i, Shannon entropy H = -sum(p_i * ln(p_i)),
        normalized: H / ln(N), N = population size.

        0.0 = all individuals identical (no diversity)
        1.0 = all individuals distinct (maximum diversity)
        """
        n = len(self.individuals)
        if n <= 1:
            return 0.0
        counts: Counter[str] = Counter(ind.dna for ind in self.individuals)
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log(p)
        max_entropy = math.log(n)
        if max_entropy <= 0:
            return 0.0
        return entropy / max_entropy

    def get_generation_stats(self) -> list[dict]:
        """Return the per-generation statistics list.

        Each element contains: generation, population_size, mean_fitness,
        max_fitness, min_fitness, diversity
        """
        return list(self.history)

    def best_individual(self) -> Individual | None:
        """Return the individual with the highest fitness in the current
        population."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda ind: ind.fitness)

    def mean_fitness(self) -> float:
        """Return the mean fitness of the current population."""
        if not self.individuals:
            return 0.0
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)


# ============================================================================
# Backward-compatible alias (deprecated)
# ============================================================================
# ``Population`` was originally the name of this module's Wright-Fisher
# evolution population class, now renamed to ``EvolutionaryPopulation``
# to distinguish it from ``helixlang.population.CellPopulation``. This
# alias is kept for backward compatibility; new code should use
# ``EvolutionaryPopulation``.
#
# DeprecationWarning: the ``Population`` alias is deprecated and may be
# removed in a future version. Please use ``EvolutionaryPopulation``
# instead.
Population = EvolutionaryPopulation
