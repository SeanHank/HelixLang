"""Tests for small high-coverage runtime modules: stochastic, morphology_3d,
seq_utils, and lsystem.
"""
from __future__ import annotations

import pytest

from helixlang.plugins.runtime.morphology_3d import (
    LSystem3D,
    Point3D,
    _normalize,
    rotate_vector,
)
from helixlang.plugins.runtime.seq_utils import gc_content, max_homopolymer, reverse_complement
from helixlang.plugins.runtime.stochastic import (
    TelegraphPromoter,
    fano_to_noise_std,
    gillespie_telegraph,
    telegraph_fano_factor,
)

# ── stochastic: error branches (lines 141, 168) ────────────────────────


def test_fano_to_noise_std_bad_scale():
    with pytest.raises(ValueError, match="expression_scale"):
        fano_to_noise_std(fano=1.5, mean=100.0, decay=0.5, expression_scale=0.0)


def test_gillespie_telegraph_bad_tmax():
    with pytest.raises(ValueError, match="t_max"):
        gillespie_telegraph(0.1, 0.2, 1.0, 0.14, t_max=-1.0)


def test_fano_to_noise_std_valid():
    val = fano_to_noise_std(fano=1.5, mean=100.0, decay=0.9, expression_scale=100.0)
    assert val > 0.0


def test_gillespie_telegraph_valid():
    res = gillespie_telegraph(0.1, 0.2, 1.0, 0.14, t_max=100.0, n_replicates=50, seed=42)
    assert res["mean"] >= 0.0
    assert res["fano"] >= 1.0


def test_telegraph_fano_factor_constitutive():
    fano = telegraph_fano_factor(k_on=10.0, k_off=0.01, burst_size=0.001, degradation_rate=0.14)
    assert fano == pytest.approx(1.0, abs=0.01)


def test_telegraph_promoter_properties():
    tp = TelegraphPromoter(k_on=0.5, k_off=0.5, burst_size=5.0)
    assert tp.transcription_rate == pytest.approx(2.5)
    assert tp.on_fraction == pytest.approx(0.5)
    assert tp.fano_factor() > 1.0


# ── morphology_3d: zero-vector normalize (line 75) + pen_up branch ─────


def test_normalize_zero_vector():
    result = _normalize(Point3D(0.0, 0.0, 0.0))
    assert result.x == 0.0
    assert result.y == 0.0
    assert result.z == 0.0


def test_normalize_nonzero():
    result = _normalize(Point3D(3.0, 0.0, 0.0))
    assert result.x == pytest.approx(1.0)


def test_pen_up_no_lines():
    ls = LSystem3D(axiom="F", rules={}, angle=22.5, step=1.0)
    lines_pen_down = ls.draw(0)
    assert len(lines_pen_down) == 1
    assert ls.draw(0) is not None


def test_rotate_vector_identity():
    v = Point3D(1.0, 0.0, 0.0)
    axis = Point3D(0.0, 0.0, 1.0)
    result = rotate_vector(v, axis, 0.0)
    assert result.x == pytest.approx(1.0)
    assert result.y == pytest.approx(0.0)


def test_3d_system_draw():
    ls = LSystem3D(axiom="F", rules={"F": "F+F"}, angle=90.0, step=1.0)
    lines = ls.draw(2)
    assert len(lines) > 0


def test_3d_system_get_bounds():
    ls = LSystem3D(axiom="F", rules={"F": "F+F"}, angle=90.0, step=1.0)
    bounds = ls.get_bounds(2)
    assert "min" in bounds
    assert "max" in bounds


# ── seq_utils: max_homopolymer empty (line 108) ────────────────────────


def test_max_homopolymer_empty():
    assert max_homopolymer("") == 0


def test_max_homopolymer_single():
    assert max_homopolymer("A") == 1


def test_max_homopolymer_runs():
    assert max_homopolymer("AAATTTGGG") == 3


def test_gc_content_and_reverse_complement():
    assert gc_content("ATGC") == pytest.approx(0.5)
    assert reverse_complement("ATGC") == "GCAT"


# ── lsystem: state_length (line 62) ────────────────────────────────────


def test_lsystem_state_length():
    from helixlang.plugins.runtime.lsystem import LSystem

    ls = LSystem(axiom="FX", rules={"X": "X+YF"})
    assert ls.state_length() == 2
    ls.iterate()
    assert ls.state_length() > 2
