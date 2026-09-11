"""Tests for helixlang.plugins.human.dose_optimizer.

Covers the Bayesian dose optimization loop, PTA/ECDF distance, compute_tmin,
and the MAP estimate with degenerate denom_slope.
"""
from __future__ import annotations

import math

import pytest

from helixlang.plugins.human.dose_optimizer import (
    DoseOptimizer,
    PKProfile,
)


def _simple_pk(dose, ke_modifier=1.0, vd_modifier=1.0):
    vd = 50.0 * vd_modifier
    ke = 0.15 * ke_modifier
    times = [0.0, 2.0, 6.0, 12.0]
    concs = [(dose / vd) * math.exp(-ke * t) for t in times]
    return PKProfile(times=times, concentrations=concs)


class _SimplePKGenerator:
    def generate(self, dose, ke_modifier=1.0, vd_modifier=1.0):
        return _simple_pk(dose, ke_modifier=ke_modifier, vd_modifier=vd_modifier)


class _LegacyPKGenerator:
    """Generator whose ``generate`` accepts no modifier kwargs (TypeError path)."""

    def generate(self, dose):
        return _simple_pk(dose, ke_modifier=1.0, vd_modifier=1.0)


# ── compute_auc ──────────────────────────────────────────────────────────


def test_compute_auc():
    pk = PKProfile(times=[0.0, 2.0], concentrations=[10.0, 5.0])
    opt = DoseOptimizer()
    auc = opt.compute_auc(pk)
    assert auc == pytest.approx(15.0)


# ── compute_cmax ─────────────────────────────────────────────────────────


def test_compute_cmax_empty():
    opt = DoseOptimizer()
    assert opt.compute_cmax(PKProfile(times=[], concentrations=[])) == 0.0


def test_compute_cmax():
    pk = PKProfile(times=[0.0, 1.0], concentrations=[3.0, 9.0])
    assert DoseOptimizer().compute_cmax(pk) == 9.0


# ── compute_tmin ─────────────────────────────────────────────────────────


def test_compute_tmin_below_threshold():
    pk = PKProfile(
        times=[0.0, 2.0, 6.0, 12.0],
        concentrations=[20.0, 5.0, 8.0, 25.0],
    )
    opt = DoseOptimizer(target_range=(10.0, 50.0))
    tmin = opt.compute_tmin(pk)
    assert tmin == pytest.approx(6.0)


def test_compute_tmin_above_threshold():
    pk = PKProfile(times=[0.0, 12.0], concentrations=[20.0, 30.0])
    opt = DoseOptimizer(target_range=(10.0, 50.0))
    assert opt.compute_tmin(pk) == pytest.approx(0.0)


def test_compute_tmin_all_below():
    pk = PKProfile(times=[0.0, 5.0, 10.0], concentrations=[1.0, 2.0, 0.5])
    opt = DoseOptimizer(target_range=(10.0, 50.0))
    assert opt.compute_tmin(pk) == pytest.approx(10.0)


# ── pta ──────────────────────────────────────────────────────────────────


def test_pta_empty():
    opt = DoseOptimizer()
    assert opt.pta([], 100.0) == 0.0


def test_pta_partial():
    opt = DoseOptimizer()
    assert opt.pta([50.0, 100.0, 150.0], 100.0) == pytest.approx(2 / 3)


def test_pta_all_above():
    opt = DoseOptimizer()
    assert opt.pta([200.0, 300.0], 100.0) == 1.0


# ── ecdf_distance ────────────────────────────────────────────────────────


def test_ecdf_distance_empty_pred():
    opt = DoseOptimizer()
    assert opt.ecdf_distance([], [1.0, 2.0]) == float("inf")


def test_ecdf_distance_empty_target():
    opt = DoseOptimizer()
    assert opt.ecdf_distance([1.0], []) == float("inf")


def test_ecdf_distance_perfect():
    opt = DoseOptimizer()
    assert opt.ecdf_distance([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)


def test_ecdf_distance_divergent():
    opt = DoseOptimizer()
    d = opt.ecdf_distance([100.0], [1.0])
    assert d > 0.0


# ── recommend_dose ───────────────────────────────────────────────────────


def test_recommend_dose_basic():
    opt = DoseOptimizer(target_range=(10.0, 50.0), target_auc_range=(100.0, 500.0))
    rec = opt.recommend_dose(
        dose_range=(100, 500),
        pk_generator=_SimplePKGenerator(),
        n_simulations=5,
        population_size=3,
    )
    assert 100.0 <= rec.recommended_dose <= 500.0
    assert rec.pta >= 0.0
    assert rec.ecdf_distance >= 0.0
    assert "mg" in rec.regimen


def test_recommend_dose_legacy_generator_fallback():
    """Generator that does not accept modifier kwargs uses the TypeError fallback."""
    opt = DoseOptimizer(target_range=(10.0, 50.0), target_auc_range=(100.0, 500.0))
    rec = opt.recommend_dose(
        dose_range=(100, 500),
        pk_generator=_LegacyPKGenerator(),
        n_simulations=4,
        population_size=2,
    )
    assert 100.0 <= rec.recommended_dose <= 500.0


# ── bayesian_map_estimate ────────────────────────────────────────────────


def test_bayesian_map_no_data():
    est = DoseOptimizer().bayesian_map_estimate([], [])
    assert est.n_observations == 0
    assert est.ke == pytest.approx(0.15)


def test_bayesian_map_typical():
    times = [0.5, 1.0, 2.0, 4.0, 8.0]
    vd, ke = 50.0, 0.15
    concs = [(100.0 / vd) * math.exp(-ke * t) for t in times]
    est = DoseOptimizer().bayesian_map_estimate(times, concs)
    assert est.n_observations == 5
    assert 0.01 <= est.ke <= 1.0
    assert est.vd >= 5.0
    assert est.posterior_variance >= 0.0


def test_bayesian_map_degenerate_slope():
    times = [1.0, 1.0, 1.0]
    concs = [5.0, 5.0, 5.0]
    est = DoseOptimizer().bayesian_map_estimate(
        times, concs, prior_ke=0.15, prior_vd=50.0,
        prior_var_ke=0.01, prior_var_vd=100.0, noise_var=0.01,
    )
    assert est.ke == pytest.approx(0.15)


# ── dose optimizer constructor defaults ──────────────────────────────────


def test_constructor_defaults():
    opt = DoseOptimizer()
    assert opt.target_range == (10.0, 50.0)
    assert opt.target_auc_range == (100.0, 500.0)
    assert opt.dosing_interval == 12.0
