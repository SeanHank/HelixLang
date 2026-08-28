"""Two-state (telegraph) promoter noise model unit tests."""
import math

import pytest

from helixlang.plugins.runtime.grn import GRN, decay_from_half_life_ticks
from helixlang.plugins.runtime.stochastic import (
    TelegraphPromoter,
    fano_to_noise_std,
    gillespie_telegraph,
    telegraph_fano_factor,
)


# -- telegraph_fano_factor (Peccoud & Ycart 1995; Jones et al. 2014) --
def test_fano_constitutive_limit_is_one():
    """k_on >> k_off, gamma -> Poisson (Fano == 1)."""
    fano = telegraph_fano_factor(
        k_on=1e6, k_off=1.0, burst_size=5.0, degradation_rate=0.14)
    assert fano == pytest.approx(1.0, rel=1e-3)


def test_fano_bursty_promoter_above_one():
    """A switching promoter is super-Poissonian (Fano > 1)."""
    fano = telegraph_fano_factor(
        k_on=2.0, k_off=20.0, burst_size=4.0, degradation_rate=0.5)
    assert fano > 1.0


def test_fano_grows_with_burst_size():
    fano_small = telegraph_fano_factor(
        k_on=2.0, k_off=20.0, burst_size=1.0, degradation_rate=0.5)
    fano_large = telegraph_fano_factor(
        k_on=2.0, k_off=20.0, burst_size=8.0, degradation_rate=0.5)
    assert fano_large > fano_small


def test_fano_negative_rates_raise():
    with pytest.raises(ValueError):
        telegraph_fano_factor(-1.0, 1.0, 1.0, 0.1)
    with pytest.raises(ValueError):
        telegraph_fano_factor(1.0, 1.0, 1.0, 0.0)


def test_fano_zero_switching_is_poisson():
    assert telegraph_fano_factor(0.0, 0.0, 3.0, 0.5) == pytest.approx(1.0)


# -- TelegraphPromoter --
def test_transcription_rate_is_burst_times_koff():
    p = TelegraphPromoter(k_on=2.0, k_off=20.0, burst_size=4.0)
    assert p.transcription_rate == pytest.approx(80.0)


def test_on_fraction():
    p = TelegraphPromoter(k_on=2.0, k_off=20.0, burst_size=4.0)
    assert p.on_fraction == pytest.approx(2.0 / 22.0)


def test_fano_factor_matches_function():
    p = TelegraphPromoter(k_on=2.0, k_off=20.0, burst_size=4.0)
    assert p.fano_factor() == pytest.approx(
        telegraph_fano_factor(2.0, 20.0, 4.0, 0.14))


def test_default_degradation_is_ecoli_mrna_half_life():
    # 5 min mRNA half-life -> gamma = ln 2 / 5
    p = TelegraphPromoter(k_on=1.0, k_off=1.0, burst_size=1.0)
    assert p.degradation_rate == pytest.approx(math.log(2) / 5.0, rel=1e-2)


# -- fano_to_noise_std --
def test_noise_std_scales_with_fano():
    std1 = fano_to_noise_std(2.0, 0.5, 0.9)
    std2 = fano_to_noise_std(4.0, 0.5, 0.9)
    assert std2 > std1
    assert std2 == pytest.approx(std1 * math.sqrt(2.0))


def test_noise_std_zero_mean_keeps_stationary_variance():
    # var_eta = (1 - decay^2) * fano * mean / scale
    decay = 0.9
    fano, mean, scale = 3.0, 0.4, 100.0
    std = fano_to_noise_std(fano, mean, decay, scale)
    assert std ** 2 == pytest.approx(
        (1 - decay ** 2) * fano * mean / scale)


def test_noise_std_invalid_args():
    with pytest.raises(ValueError):
        fano_to_noise_std(0.9, 0.5, 0.9)   # fano < 1
    with pytest.raises(ValueError):
        fano_to_noise_std(2.0, -1.0, 0.9)  # mean < 0
    with pytest.raises(ValueError):
        fano_to_noise_std(2.0, 0.5, 1.0)   # decay >= 1


# -- Gillespie SSA reference --
def test_gillespie_matches_analytic_fano_within_sampling():
    """The exact SSA reproduces the closed-form Fano factor within
    sampling error (k_on=2, k_off=20, b=4, gamma=0.5)."""
    k_on, k_off, b, gamma = 2.0, 20.0, 4.0, 0.5
    analytic = telegraph_fano_factor(k_on, k_off, b, gamma)
    ssa = gillespie_telegraph(
        k_on, k_off, b, gamma, t_max=4000.0, n_replicates=4000, seed=7)
    assert ssa["fano"] == pytest.approx(analytic, rel=0.15)
    assert ssa["fano"] > 1.0


def test_gillespie_constitutive_poisson():
    """Constitutive promoter: SSA Fano ~ 1."""
    ssa = gillespie_telegraph(
        1e4, 1.0, 2.0, 0.5, t_max=4000.0, n_replicates=2000, seed=3)
    assert ssa["fano"] == pytest.approx(1.0, rel=0.2)


def test_gillespie_reproducible_with_seed():
    a = gillespie_telegraph(2.0, 20.0, 4.0, 0.5, 100.0,
                            n_replicates=100, seed=11)
    b = gillespie_telegraph(2.0, 20.0, 4.0, 0.5, 100.0,
                            n_replicates=100, seed=11)
    assert a == b


# -- GRN integration: noise preserves the deterministic mean --
def test_grn_noise_preserves_deterministic_mean():
    """With telegraph noise enabled the steady-state mean matches the
    deterministic path while the variance reproduces Fano*mean/scale."""
    tprom = TelegraphPromoter(0.2, 2.0, 1.0, 0.14, expression_scale=100.0)
    noisy = GRN(noise_enabled=True, noise_seed=5)
    noisy.add_gene("g", threshold=0.0, initial_level=0.5, decay=0.9,
                   noise=tprom)
    det = GRN(noise_enabled=False)
    det.add_gene("g", threshold=0.0, initial_level=0.5, decay=0.9)

    for _ in range(500):
        noisy.step()
        det.step()

    assert noisy.nodes["g"].level == pytest.approx(
        det.nodes["g"].level, abs=0.05)
    assert 0.0 < noisy.nodes["g"].level < 1.0


def test_grn_noise_disabled_is_deterministic():
    """Noise disabled -> bit-identical to the pre-noise recurrence."""
    a = GRN()
    b = GRN(noise_enabled=False)
    for grn in (a, b):
        grn.add_gene("x", threshold=0.5, decay=decay_from_half_life_ticks(110))
        grn.add_edge("x", "x", 0.8)
    for _ in range(10):
        assert a.step() == b.step()
