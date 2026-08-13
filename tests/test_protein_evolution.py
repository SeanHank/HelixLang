"""Tests for ML-guided directed evolution of GB1 (S8).

Verification goals:
- gbi_landscape: identity = 1.0, bounds [0, 1], interface damage is
  disproportionately deleterious (Wu et al. 2016 GB1 DMS weighting).
- make_crippled: exactly n_sites interface substitutions to worst
  BLOSUM residues; fitness drops well below 1.
- make_oracle: ESM-2 preferred when available, else BLOSUM62 fallback;
  the returned oracle always scores.
- guided_directed_evolution: oracle-guided top-K screening recovers
  fitness from a crippled start far better than random screening
  (EVOLVEpro/FSFP closed-loop DBTL), is deterministic under a fixed
  seed, and reports the round-1 oracle-vs-landscape Spearman alignment
  (ProteinGym metric).
"""
from __future__ import annotations

import pytest

from helixlang.apps.protein_evolution import (
    GB1_WT,
    DirectedEvolutionResult,
    gbi_landscape,
    guided_directed_evolution,
    make_crippled,
    make_oracle,
    oracle_vs_landscape_spearman,
    spearman_rank_correlation,
)


def test_gbi_landscape_identity_is_optimal() -> None:
    assert gbi_landscape(GB1_WT) == pytest.approx(1.0)
    variant = GB1_WT[:9] + "A" + GB1_WT[10:]
    assert 0.0 <= gbi_landscape(variant) <= 1.0


def test_gbi_landscape_interface_damage_is_worse() -> None:
    # D at interface index 39 (position 40) vs non-interface index 21
    assert GB1_WT[39] == "D" and GB1_WT[21] == "D"
    iface = GB1_WT[:39] + "A" + GB1_WT[40:]
    backg = GB1_WT[:21] + "A" + GB1_WT[22:]
    drop_iface = 1.0 - gbi_landscape(iface)
    drop_backg = 1.0 - gbi_landscape(backg)
    assert drop_iface > 3.0 * drop_backg


def test_gbi_landscape_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        gbi_landscape(GB1_WT[:-1])


def test_make_crippled_damages_interface_sites() -> None:
    crippled = make_crippled(n_sites=8, seed=1)
    changed = [i for i in range(len(GB1_WT)) if crippled[i] != GB1_WT[i]]
    assert len(changed) == 8
    assert all(i in frozenset(range(38, 54)) for i in changed)
    assert gbi_landscape(crippled) < 0.9
    with pytest.raises(ValueError):
        make_crippled(n_sites=0)


def test_make_oracle_falls_back_to_blosum62() -> None:
    oracle, name = make_oracle()
    assert oracle.available
    assert name in ("esm2", "blosum62")
    assert oracle.score(GB1_WT, GB1_WT) >= 0.0
    _, name2 = make_oracle(prefer_esm=False)
    assert name2 == "blosum62"


def test_guided_directed_evolution_beats_random_baseline() -> None:
    res = guided_directed_evolution()
    assert res.guided_gain > res.baseline_gain + 0.05
    assert res.guided_gain > 0.10
    assert res.baseline_gain < res.guided_gain


def test_guided_evolution_recovers_wild_type() -> None:
    res = guided_directed_evolution()
    start = make_crippled(seed=1)
    mismatches_start = sum(a != b for a, b in zip(GB1_WT, start, strict=True))
    mismatches_final = sum(
        a != b for a, b in zip(GB1_WT, res.final_best_sequence, strict=True)
    )
    assert mismatches_final < mismatches_start
    assert res.guided_recovery > 0.85


def test_guided_evolution_is_deterministic() -> None:
    a = guided_directed_evolution()
    b = guided_directed_evolution()
    assert a.guided_cumulative_best == b.guided_cumulative_best
    assert a.baseline_cumulative_best == b.baseline_cumulative_best
    assert a.final_best_sequence == b.final_best_sequence


def test_oracle_spearman_alignment_with_landscape() -> None:
    res = guided_directed_evolution()
    assert res.spearman_rho > 0.5
    assert isinstance(res, DirectedEvolutionResult)


def test_top_k_validation() -> None:
    with pytest.raises(ValueError):
        guided_directed_evolution(top_k=0)
    with pytest.raises(ValueError):
        guided_directed_evolution(top_k=100, library_size=60)


def test_spearman_rank_correlation_basic() -> None:
    assert spearman_rank_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert spearman_rank_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        spearman_rank_correlation([1.0], [1.0])


def test_oracle_vs_landscape_spearman_public() -> None:
    import random
    rng = random.Random(5)
    from helixlang.apps.protein_evolution import _single_mutant
    variants = [_single_mutant(GB1_WT, rng) for _ in range(40)]
    rho = oracle_vs_landscape_spearman(variants)
    assert rho > 0.4
