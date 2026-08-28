"""Omics-level data calibration tests (S10.1 direction A).

Verification goals:
- negative-binomial count noise reproduces the DESeq2 variance structure
  ``Var = mu + dispersion * mu^2``; log-normal noise is multiplicative.
- ``fit_parameters`` weighted multi-observation calibration (inverse-
  variance weights, Karr 2012 / DESeq2 style) recovers coupling constants
  of a two-readout model within 5% and validates weight shape.
- The CRISPRi perturb-seq calibration -> prediction benchmark (VCC 2025
  protocol) predicts held-out perturbations: MAE well below the WT
  baseline, high response correlation, high DE sign agreement.

References:
- Virtual Cell Challenge 2025 (MAE_k / DES/PDS metrics; CRISPRi library,
  ~83% knockdown efficacy)
- Love et al. 2014 (DESeq2): negative-binomial count model
- Karr et al. 2012 Cell 150:389 (weighted whole-cell parameter estimation)
"""
from __future__ import annotations

import math
import random

import pytest

from helixlang.plugins.apps.omics_calibration import (
    OmicsCalibrationBenchmark,
    de_sign_agreement,
    generate_perturb_seq_data,
    inverse_variance_weights,
    log_normal_noise,
    negative_binomial_noise,
    response_correlation,
    run_omics_calibration_benchmark,
    vcc_mae,
)
from helixlang.plugins.runtime.virtual_cell import fit_parameters

# ============================================================================
# Noise models
# ============================================================================

def test_negative_binomial_variance_structure() -> None:
    mu, disp, n = 50.0, 0.05, 4000
    rng = random.Random(0)
    xs = [negative_binomial_noise(mu, rng, disp) for _ in range(n)]
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    assert mean == pytest.approx(mu, rel=0.1)
    # Var = mu + dispersion * mu^2 (DESeq2)
    assert var == pytest.approx(mu + disp * mu * mu, rel=0.15)


def test_log_normal_noise_is_multiplicative() -> None:
    rng = random.Random(1)
    vals = log_normal_noise([10.0, 100.0, 1000.0], rng, sigma=0.1)
    # multiplicative noise: log of ratio is roughly N(0, sigma^2)
    logs = [math.log(v / ref) for v, ref in zip(vals, [10.0, 100.0, 1000.0], strict=True)]
    assert math.isclose(sum(logs) / len(logs), 0.0, abs_tol=0.3)


# ============================================================================
# Weighted multi-observation fit_parameters (the calibration harness)
# ============================================================================

def test_weighted_fit_recovers_two_readouts() -> None:
    """Recover (a, b) of y = (a*t, 1000*b*t) with inverse-variance weights.

    The two readouts have very different scales; unweighted SSE would be
    dominated by the large-scale readout.  With inverse-variance weights
    the fitter recovers both constants within 5%.
    """
    t = list(range(10))
    a_true, b_true = 3.0, 5.0
    observed = [a_true * ti for ti in t] + [1000.0 * b_true * ti for ti in t]
    weights = inverse_variance_weights(
        [[a_true * ti + 1.0 for ti in t], [1000.0 * b_true * ti + 1.0
                                           for ti in t]])

    def predict(a: float, b: float) -> list[float]:
        return [a * ti for ti in t] + [1000.0 * b * ti for ti in t]

    fit = fit_parameters(predict, observed, {"a": (1.0, 6.0), "b": (1.0, 8.0)},
                         n_samples=300, seed=0, refine_rounds=4,
                         weights=weights)
    assert abs(fit["best"]["a"] - a_true) / a_true < 0.05
    assert abs(fit["best"]["b"] - b_true) / b_true < 0.05


def test_fit_parameters_weights_length_checked() -> None:
    with pytest.raises(ValueError):
        fit_parameters(lambda x: [x, 2.0 * x], [1.0, 2.0],
                       {"x": (0.0, 3.0)}, weights=[1.0])


def test_fit_parameters_unit_weights_matches_default() -> None:
    """weights=None is exactly the unweighted objective (backward compat)."""
    obs = [1.0, 4.0, 9.0]

    def predict(c: float) -> list[float]:
        return [c * v for v in obs]

    a = fit_parameters(predict, obs, {"c": (0.0, 3.0)},
                       n_samples=200, seed=0, refine_rounds=3)
    b = fit_parameters(predict, obs, {"c": (0.0, 3.0)},
                       n_samples=200, seed=0, refine_rounds=3,
                       weights=[1.0, 1.0, 1.0])
    assert a["best"]["c"] == pytest.approx(b["best"]["c"])


# ============================================================================
# Synthetic perturb-seq data
# ============================================================================

def test_perturb_seq_data_shapes_and_wt_baseline() -> None:
    data = generate_perturb_seq_data(n_genes=50, n_perturbations=12,
                                     n_tf=4, seed=3)
    assert len(data["genes"]) == 50
    assert len(data["perturbations"]) == 12
    assert len(data["design"]) == 50 and len(data["design"][0]) == 4
    assert len(data["mean"]) == 50 and len(data["mean"][0]) == 12
    assert len(data["counts"]) == 50 and len(data["counts"][0]) == 12
    # the WT column is the perturbation-0 baseline
    assert data["perturbations"][0] == "WT"
    assert all(a == 1.0 for a in data["activities"][0])
    # knockdown rows have exactly one regulator reduced to kd
    kd_row = data["activities"][1]
    assert min(kd_row) == pytest.approx(data["kd"])


def test_inverse_variance_weights_decay_with_count() -> None:
    w = inverse_variance_weights([[10.0, 100.0, 1000.0]])
    assert w[0] > w[1] > w[2]
    assert len(w) == 3


# ============================================================================
# VCC-style scoring
# ============================================================================

def test_vcc_mae_formula() -> None:
    assert vcc_mae([1.0, 2.0, 3.0], [1.0, 4.0, 3.0]) == pytest.approx(2 / 3)


def test_response_correlation() -> None:
    assert response_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) > 0.99
    assert response_correlation([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]) < -0.99


def test_de_sign_agreement() -> None:
    # up-regulated gene predicted up, down-regulated predicted up -> 0.5
    pred = [2.0, 2.0]
    truth = [3.0, 0.5]
    wt = [1.0, 1.0]
    assert de_sign_agreement(pred, truth, wt) == pytest.approx(0.5)


# ============================================================================
# Benchmark (VCC 2025 calibrate-then-predict protocol)
# ============================================================================

def test_omics_benchmark_predicts_held_out_perturbations() -> None:
    result = run_omics_calibration_benchmark(
        n_genes=160, n_perturbations=20, n_train=12, seed=0, fit_seed=1,
    )
    assert result["passed"]
    assert result["mae_improvement_vs_baseline"] > 0.5
    assert result["response_correlation"] > 0.9
    assert result["de_sign_agreement"] > 0.9
    assert len(result["holdout_perturbations"]) == 8


def test_omics_benchmark_reports_coupling() -> None:
    b = OmicsCalibrationBenchmark(n_genes=120, n_perturbations=20,
                                  n_train=12, seed=2, fit_seed=1)
    r = b.score()
    assert set(r["truth_coupling"]) == {"response_gain", "hill_n"}
    assert set(r["fitted_coupling"]) == {"response_gain", "hill_n"}
    assert all(math.isfinite(v) for v in r["fitted_coupling"].values())
