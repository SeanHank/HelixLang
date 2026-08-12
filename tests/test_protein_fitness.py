"""PLM fitness oracle tests (T2.4, P8).

Verification goals:
- BLOSUM62 raw / normalized scoring: identity = 1.0, conservative
  substitutions score above disruptive ones, normalization stays in
  [0, 1].
- BLOSUMOracle ranks a DMS-style panel sensibly (synonymous / conservative
  variants outrank disruptive ones).
- rank_variants returns a sorted panel with scores.
- The optional ESM-2 oracle degrades cleanly (available False, clear error)
  when transformers/torch are absent, and scores when present.
- evolution.calculate_fitness(method="oracle") routes through the oracle
  and rejects missing wild-type / bad names.

References:
- Henikoff S & Henikoff JG. PNAS 1992 89:10915-10919 (BLOSUM matrices)
- Frazer J et al. PNAS 2021 118:e2012055118 (zero-shot variant effect)
- Notin P et al. Nat Commun 2024 15:5566 (ProteinGym / FSFP)
- Jiang W et al. Science 2025 387:eadr6006 (EVOLVEpro)
"""
from __future__ import annotations

import pytest

from helixlang.evolution import calculate_fitness
from helixlang.protein_fitness import (
    AA20,
    BLOSUM62,
    BLOSUMOracle,
    ESM2Oracle,
    blosum62_normalized,
    blosum62_raw,
    dna_fitness,
    oracle_score,
    protein_to_dna,
    rank_variants,
)

#: a short model peptide (e.g. a lacI N-terminal helix fragment)
WT = "MVNPVTLYDVAEYAGVSYQTVSRVVNQASHVSAKTREKVEAAMAELNYIPNRVAQQLAGKQSLLIGV"


def test_blosum62_matrix_is_20x20_and_symmetric() -> None:
    assert len(BLOSUM62) == 20
    assert tuple(BLOSUM62) == AA20
    for aa, row in BLOSUM62.items():
        assert len(row) == 20
        for bb, v in row.items():
            assert v == BLOSUM62[bb][aa]  # symmetric


def test_blosum62_raw_identity_is_max() -> None:
    assert blosum62_raw(WT, WT) == sum(BLOSUM62[a][a] for a in WT)


def test_blosum62_normalized_identity() -> None:
    assert blosum62_normalized(WT, WT) == 1.0


def test_blosum62_normalized_bounds() -> None:
    # any variant must land in [0, 1]
    for v in (WT[1:] + "G", WT[:10] + "W" + WT[11:], "M" * len(WT)):
        s = blosum62_normalized(WT, v)
        assert 0.0 <= s <= 1.0


def test_conservative_beats_disruptive() -> None:
    # I->L is a conservative (BLOSUM +2) substitution; I->W is disruptive
    pos = WT.index("I")
    conservative = WT[:pos] + "L" + WT[pos + 1:]
    disruptive = WT[:pos] + "W" + WT[pos + 1:]
    assert blosum62_raw(WT, conservative) > blosum62_raw(WT, disruptive)
    assert (blosum62_normalized(WT, conservative)
            > blosum62_normalized(WT, disruptive))


def test_blosum_oracle_available_and_scoring() -> None:
    oracle = BLOSUMOracle()
    assert oracle.available
    assert oracle.score(WT, WT) == 1.0
    assert 0.0 <= oracle.score(WT, WT[1:] + "G") < 1.0


def test_rank_variants_sorted() -> None:
    variants = [
        WT,  # identity -> best
        "W" * len(WT),  # disruptive
        WT[:5] + "G" + WT[6:],  # single conservative-ish change
    ]
    ranked = rank_variants(WT, variants)
    assert ranked[0][0] == WT
    assert ranked[0][1] == 1.0
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_oracle_score_name_dispatch() -> None:
    assert oracle_score(WT, WT) == 1.0
    assert oracle_score(WT, WT, oracle="blosum62") == 1.0
    with pytest.raises(ValueError):
        oracle_score(WT, WT, oracle="nope")


def test_oracle_score_invalid_aa_raises() -> None:
    with pytest.raises(ValueError):
        blosum62_raw(WT, WT + "Z")


def test_blosum62_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        blosum62_raw(WT, WT[:-1])


def test_dna_fitness_translates_and_scores() -> None:
    wt_dna = protein_to_dna(WT)
    assert dna_fitness(wt_dna, wt_dna, "blosum62") == 1.0
    # silent (synonymous) codon change -> same protein -> fitness 1.0
    syn_dna = protein_to_dna(WT)  # deterministic back-translation
    assert syn_dna == wt_dna
    # a single amino-acid swap (WT[6]='L') scored below identity
    mut = WT[:6] + "W" + WT[7:]
    mut_dna = protein_to_dna(mut)
    assert dna_fitness(mut_dna, wt_dna, "blosum62") < 1.0


def test_protein_to_dna_roundtrip_translation() -> None:
    from helixlang.dna_codec import translate_dna

    dna = protein_to_dna(WT)
    assert translate_dna(dna) == WT


def test_calculate_fitness_oracle_method() -> None:
    wt_dna = protein_to_dna(WT)
    assert calculate_fitness(wt_dna, wt_dna, method="oracle",
                             oracle="blosum62") == 1.0
    assert calculate_fitness(wt_dna, wt_dna, method="oracle",
                             oracle=BLOSUMOracle()) == 1.0


def test_calculate_fitness_oracle_requires_target() -> None:
    with pytest.raises(ValueError):
        calculate_fitness(protein_to_dna(WT), method="oracle",
                          oracle="blosum62")


def test_calculate_fitness_unknown_method_still_raises() -> None:
    with pytest.raises(ValueError):
        calculate_fitness("ACGT", method="nope")


# ============================================================================
# Optional ESM-2 oracle (skip cleanly when transformers/torch absent)
# ============================================================================

def test_esm2_oracle_degrades_gracefully() -> None:
    oracle = ESM2Oracle()
    if not oracle.available:  # pragma: no cover - depends on environment
        with pytest.raises(RuntimeError):
            oracle.pseudo_log_likelihood(WT)
        with pytest.raises(RuntimeError):
            oracle_score(WT, WT, oracle="esm2")
    else:
        assert oracle.available
        pll = oracle.pseudo_log_likelihood(WT)
        assert pll < 0.0  # log-likelihoods are negative
        assert oracle.score(WT, WT) >= 0.0  # wt - wt == 0 or tiny float
