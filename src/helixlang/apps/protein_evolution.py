"""ML-guided directed evolution of the GB1 IgG-binding domain (S8).

Closed-loop "design - build - test - learn" (DBTL) protein engineering
in miniature.  The fitness *landscape* is the weighted-BLOSUM62
chemical-tolerance model of the GB1 domain (Streptococcal protein G
B1, 56 residues) with the known IgG-binding interface window weighted
harder (Wu et al. 2016 Nature 532:58 DMS of GB1 residues 39-54; Olson
et al. 2017 Protein Sci 26:1521).  The *oracle* is a zero-shot fitness
predictor scored against the wild type, exactly as EVOLVEpro (Jiang et
al. 2025 Science 387:eadr6006), MULTI-evolve (Tran et al. 2025 Science
388:aea1820) and FSFP (Otalora Ottó et al. Nat Commun 2024 15:5566)
drive their mutagenesis rounds: ``ESM2Oracle`` pseudo-likelihood
(Frazer et al. 2021 PNAS zero-shot protocol) when transformers/torch
are installed, else the :class:`BLOSUMOracle` baseline.

:func:`guided_directed_evolution` starts from a deliberately
*crippled* GB1 variant (worst-BLOSUM substitutions at interface
positions), then runs rounds of

1. **design**   - single-residue mutant library around the current best
2. **build**    - ``rank_variants`` with the oracle vs the wild type
3. **test**     - top-K screened on the GB1 landscape
4. **learn**    - best screened variant becomes the next parent

and reports the cumulative-best fitness trajectory against an
oracle-free random-screening baseline, plus the Spearman rank
correlation between oracle predictions and landscape fitness over the
round-1 library (the standard ProteinGym alignment metric).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from helixlang.protein_fitness import (
    AA20,
    BLOSUM62,
    BLOSUMOracle,
    ESM2Oracle,
    FitnessOracle,
    oracle_score,
    rank_variants,
)

#: GB1 domain wild type (56 residues, standard sequence).
GB1_WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

#: IgG-binding interface window measured in the GB1 DMS (Wu et al. 2016);
#: 0-indexed residues 38..53 == 1-indexed 39..54.
GB1_INTERFACE = frozenset(range(38, 54))

#: landscape weight for interface positions (chemical-tolerance hotspots)
GB1_INTERFACE_WEIGHT = 2.0
GB1_BACKGROUND_WEIGHT = 0.4

_MUTATION_POOL = tuple(aa for aa in AA20)
_BLOSUM_MIN = {aa: min(row.values()) for aa, row in BLOSUM62.items()}


def _validate(sequence: str, name: str) -> None:
    if not sequence:
        raise ValueError(f"{name} must be a non-empty protein sequence")
    for i, aa in enumerate(sequence):
        if aa not in BLOSUM62:
            raise ValueError(f"invalid amino acid {aa!r} at {name} position {i}")


def gbi_landscape(variant: str, reference: str = GB1_WT) -> float:
    """Weighted-BLOSUM62 fitness of a GB1 variant in [0, 1].

    ``identity == 1.0``; interface (binding-contact) positions carry
    five times the weight of background positions, so damage there is
    proportionally more deleterious -- the single-site tolerance
    profile measured on the real GB1 domain.
    """
    if len(reference) != len(variant):
        raise ValueError("reference and variant must have equal length")
    _validate(reference, "reference")
    _validate(variant, "variant")
    raw = best = worst = 0.0
    for i, (r, v) in enumerate(zip(reference, variant, strict=True)):
        w = GB1_INTERFACE_WEIGHT if i in GB1_INTERFACE else GB1_BACKGROUND_WEIGHT
        raw += w * BLOSUM62[r][v]
        best += w * BLOSUM62[r][r]
        worst += w * _BLOSUM_MIN[r]
    if best == worst:
        return 1.0
    return max(0.0, min(1.0, (raw - worst) / (best - worst)))


def make_oracle(prefer_esm: bool = True) -> tuple[FitnessOracle, str]:
    """Build the fitness oracle: ESM-2 when available, else BLOSUM62.

    Returns ``(oracle, name)`` where ``name`` is ``"esm2"`` or
    ``"blosum62"``.  This is the EVOLVEpro-style zero-shot predictor
    with a graceful dependency fallback.
    """
    if prefer_esm:
        esm = ESM2Oracle()
        if esm.available:  # pragma: no cover - requires optional extra
            return esm, "esm2"
    return BLOSUMOracle(), "blosum62"


def make_crippled(reference: str = GB1_WT, n_sites: int = 8,
                  seed: int = 1) -> str:
    """Cripple ``n_sites`` interface positions to their worst BLOSUM residue."""
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    rng = random.Random(seed)
    positions = list(GB1_INTERFACE)
    rng.shuffle(positions)
    out = list(reference)
    for i in positions[:n_sites]:
        aa = reference[i]
        worst = min(BLOSUM62[aa].values())
        candidates = [c for c, s in BLOSUM62[aa].items() if s == worst]
        out[i] = rng.choice(candidates)
    return "".join(out)


def _single_mutant(sequence: str, rng: random.Random) -> str:
    """One random single-residue substitution."""
    i = rng.randrange(len(sequence))
    aa = sequence[i]
    replacement = rng.choice([c for c in _MUTATION_POOL if c != aa])
    return sequence[:i] + replacement + sequence[i + 1:]


def _ranks(values: list[float]) -> list[float]:
    """Average ranks of a list (ties share the mean rank)."""
    ordered = sorted(values)
    n = len(values)
    rank_of: dict[float, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1] == ordered[i]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            rank_of[ordered[k]] = avg
        i = j + 1
    return [rank_of[v] for v in values]


def spearman_rank_correlation(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation between two equal-length samples."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("xs and ys must have equal length >= 2")
    rx = _ranks(xs)
    ry = _ranks(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den_x = sum((a - mx) ** 2 for a in rx)
    den_y = sum((b - my) ** 2 for b in ry)
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / math.sqrt(den_x * den_y)


def oracle_vs_landscape_spearman(
    variants: list[str],
    reference: str = GB1_WT,
    oracle: FitnessOracle | str | None = None,
) -> float:
    """Spearman rank correlation between oracle predictions and the GB1
    landscape over a panel of variants (the ProteinGym alignment metric)."""
    predicted = [oracle_score(reference, v, oracle) for v in variants]
    measured = [gbi_landscape(v, reference) for v in variants]
    return spearman_rank_correlation(predicted, measured)


@dataclass(slots=True)
class DirectedEvolutionResult:
    """Per-round trajectories of a guided-evolution campaign."""

    rounds: int
    library_size: int
    top_k: int
    oracle_name: str
    initial_fitness: float
    guided_cumulative_best: list[float]
    baseline_cumulative_best: list[float]
    final_best_sequence: str
    spearman_rho: float

    @property
    def guided_gain(self) -> float:
        """Fitness recovered by the guided campaign (absolute)."""
        return self.guided_cumulative_best[-1] - self.initial_fitness

    @property
    def baseline_gain(self) -> float:
        """Fitness recovered by the random baseline."""
        return self.baseline_cumulative_best[-1] - self.initial_fitness

    @property
    def guided_recovery(self) -> float:
        """Final cumulative-best fitness of the guided campaign."""
        return self.guided_cumulative_best[-1]


def _run_policy(
    start: str,
    rounds: int,
    library_size: int,
    top_k: int,
    guided: bool,
    oracle: FitnessOracle | str | None,
    seed: int,
) -> tuple[list[float], list[float], str]:
    """Run one selection policy from the same crippled start."""
    rng = random.Random(seed)
    parent = start
    best = gbi_landscape(start)
    cumulative = [best]
    round1_scores: list[float] = []
    for rnd in range(rounds):
        library = [_single_mutant(parent, rng) for _ in range(library_size)]
        if rnd == 0:
            round1_scores = [gbi_landscape(v) for v in library]
        if guided:
            ranked = rank_variants(GB1_WT, library, oracle)
            chosen = [v for v, _ in ranked[:top_k]]
        else:
            chosen = rng.sample(library, top_k)
        fits = [gbi_landscape(v) for v in chosen]
        best_round = max(fits)
        parent = chosen[fits.index(best_round)]
        best = max(best, best_round)
        cumulative.append(best)
    return cumulative, round1_scores, parent


def guided_directed_evolution(
    rounds: int = 8,
    library_size: int = 60,
    top_k: int = 5,
    oracle: FitnessOracle | str | None = None,
    n_crippled_sites: int = 8,
    cripple_seed: int = 1,
    seed: int = 7,
) -> DirectedEvolutionResult:
    """Run oracle-guided vs random-screening evolution on GB1.

    Both policies start from the same crippled GB1 variant and draw
    identical round-1 libraries (same ``seed``); the guided policy
    ranks the library with the oracle against the wild type and screens
    the top-K, the baseline screens a random top-K.  The cumulative-best
    landscape fitness of each campaign and the round-1 oracle-vs-
    landscape Spearman correlation are returned.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if top_k > library_size:
        raise ValueError("top_k must be <= library_size")
    effective_oracle, name = make_oracle() if oracle is None else (oracle, _oracle_name(oracle))
    start = make_crippled(GB1_WT, n_crippled_sites, cripple_seed)
    initial = gbi_landscape(start)
    guided, round1_scores, best_seq = _run_policy(
        start, rounds, library_size, top_k, True, effective_oracle, seed)
    baseline, _, _ = _run_policy(
        start, rounds, library_size, top_k, False, effective_oracle, seed)
    rho = spearman_rank_correlation(
        [oracle_score(GB1_WT, v, effective_oracle) for v in
         _round1_variants(start, seed, library_size)],
        round1_scores,
    )
    return DirectedEvolutionResult(
        rounds=rounds,
        library_size=library_size,
        top_k=top_k,
        oracle_name=name,
        initial_fitness=initial,
        guided_cumulative_best=guided,
        baseline_cumulative_best=baseline,
        final_best_sequence=best_seq,
        spearman_rho=rho,
    )


def _round1_variants(start: str, seed: int, library_size: int) -> list[str]:
    """The identical round-1 library drawn by both policies."""
    rng = random.Random(seed)
    return [_single_mutant(start, rng) for _ in range(library_size)]


def _oracle_name(oracle: FitnessOracle | str | None) -> str:
    if isinstance(oracle, str):
        return oracle
    return type(oracle).__name__.removesuffix("Oracle").lower()


__all__ = [
    "GB1_WT",
    "GB1_INTERFACE",
    "DirectedEvolutionResult",
    "gbi_landscape",
    "make_crippled",
    "make_oracle",
    "guided_directed_evolution",
    "oracle_vs_landscape_spearman",
    "spearman_rank_correlation",
]
