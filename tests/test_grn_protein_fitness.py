"""Tests for grn.py and protein_fitness.py — covering remaining uncovered branches."""
from __future__ import annotations

import pytest

from helixlang.plugins.runtime.grn import (
    GRN,
    ContinuousGRNResult,
    _resample,
    grn_derivatives,
    hill,
)
from helixlang.plugins.runtime.protein_fitness import (
    BLOSUMOracle,
    ESM2Oracle,
    _validate,
    blosum62_normalized,
    oracle_score,
)

# ── grn: hill(kd=0) line 73 ────────────────────────────────────────────


def test_hill_kd_zero():
    assert hill(x=1.0, n=2.0, kd=0.0) == 1.0


def test_hill_x_negative():
    assert hill(x=-1.0, n=2.0, kd=1.0) == 0.0


# ── grn: grn_derivatives levels mismatch line 152 ──────────────────────


def test_grn_derivatives_levels_mismatch():
    g = GRN()
    g.add_gene("a", threshold=0.5, initial_level=0.5)
    g.add_gene("b", threshold=0.5, initial_level=0.5)
    with pytest.raises(ValueError, match="levels must match"):
        grn_derivatives(g, levels=[0.5])


# ── grn: set_level line 259 ────────────────────────────────────────────


def test_grn_set_level():
    g = GRN()
    g.add_gene("x", threshold=0.5)
    g.set_level("x", 0.8)
    assert g.nodes["x"].level == pytest.approx(0.8)


def test_grn_set_level_clamp():
    g = GRN()
    g.add_gene("x", threshold=0.5, initial_level=0.5)
    g.set_level("x", 2.0)
    assert g.nodes["x"].level == pytest.approx(1.0)
    g.set_level("x", -1.0)
    assert g.nodes["x"].level == pytest.approx(0.0)


def test_grn_set_level_unknown():
    g = GRN()
    g.set_level("nonexistent", 0.5)


# ── grn: noise skip branch line 353 ────────────────────────────────────


def test_grn_step_accel_noise_skip():
    from helixlang.plugins.runtime.stochastic import TelegraphPromoter

    g = GRN(noise_enabled=True, noise_seed=42)
    tp = TelegraphPromoter(k_on=0.5, k_off=0.5, burst_size=1.0)
    g.add_gene("a", threshold=0.5, initial_level=0.5, noise=tp)
    g.add_gene("b", threshold=0.5, initial_level=0.5)
    g.add_edge("a", "b", weight=1.0)
    g.nodes["b"].noise = None
    g.step_accel(prefer="python")
    assert 0.0 <= g.nodes["b"].level <= 1.0


# ── grn: _resample n_points=1 line 551 ─────────────────────────────────


def test_resample_single_point():
    times = [0.0, 1.0, 2.0]
    ys = [[0.0], [0.5], [1.0]]
    fs = [[1.0], [1.0], [1.0]]
    out_t, out_y = _resample(times, ys, fs, n_points=1)
    assert out_t == times
    assert out_y == ys


# ── grn: ContinuousGRNResult.at() single point line 596 ────────────────


def test_cgrn_at_single_point():
    r = ContinuousGRNResult(names=["gene1"], times=[5.0], levels=[[0.75]])
    result = r.at(10.0)
    assert result["gene1"] == pytest.approx(0.75)


# ── grn: _rebuild_incoming guard line 267 ──────────────────────────────


def test_grn_step_rebuild_incoming():
    g = GRN()
    g.add_gene("a", threshold=0.5, initial_level=1.0)
    g.add_gene("b", threshold=0.5, initial_level=0.0)
    g.add_edge("a", "b", weight=1.0)
    g.step()
    e = type(g.edges[0])(source="b", target="a", weight=0.5)
    g.edges.append(e)
    g.step()
    assert 0.0 <= g.nodes["a"].level <= 1.0


# ── protein_fitness: _validate errors lines 115, 118 ───────────────────


def test_validate_empty():
    with pytest.raises(ValueError, match="non-empty"):
        _validate("", "ref")


def test_validate_invalid_aa():
    with pytest.raises(ValueError, match="invalid amino acid"):
        _validate("AXZ", "ref")


# ── protein_fitness: blosum62_normalized length mismatch line 143 ──────


def test_blosum62_normalized_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        blosum62_normalized("ACDEF", "ACDE")


# ── protein_fitness: ESM2Oracle error paths ─────────────────────────────


def test_esm2_score_length_mismatch():
    esm = ESM2Oracle()
    with pytest.raises(ValueError, match="equal length"):
        esm.score("ACD", "ACDE")


# ── protein_fitness: oracle_score with esm2 ────────────────────────────


def test_oracle_score_esm2_dispatch():
    esm = ESM2Oracle()
    if esm.available:
        s = oracle_score("ACDE", "ACDF", oracle="esm2")
        assert isinstance(s, float)
    else:
        with pytest.raises(RuntimeError, match="esm2 oracle"):
            oracle_score("ACDE", "ACDF", oracle="esm2")


# ── protein_fitness: BLOSUMOracle happy path ────────────────────────────


def test_blosum_oracle_score():
    oracle = BLOSUMOracle()
    assert oracle.available is True
    s = oracle.score("ACDEF", "ACDEF")
    assert s == pytest.approx(1.0)
