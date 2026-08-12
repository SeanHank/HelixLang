"""Tests for dynamic flux balance analysis (Mahadevan et al. 2002)."""
import math

import pytest

from helixlang.environment import Environment, EnvironmentConfig
from helixlang.metabolism import (
    ECOLI_CORE_MODEL,
    DynamicFBAConfig,
    DynamicFluxBalance,
    FluxBalanceAnalysis,
)


@pytest.fixture()
def fba() -> DynamicFluxBalance:
    cfg = DynamicFBAConfig(
        dt_h=0.1,
        initial_biomass_gdw=0.05,
        initial_glucose_mm=10.0,
        max_glucose_uptake=10.0,
        glucose_half_saturation_mm=0.1,
    )
    return DynamicFluxBalance(config=cfg)


def _lp_growth(glucose_uptake: float) -> float:
    fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
    fba.set_uptake("GLC", glucose_uptake)
    return fba.solve()["BIOMASS"]


# ---------------------------------------------------------------------------
# substrate availability
# ---------------------------------------------------------------------------

def test_uptake_bound_michaelis_menten(fba: DynamicFluxBalance) -> None:
    cfg = fba.config
    vmax = cfg.max_glucose_uptake
    ks = cfg.glucose_half_saturation_mm
    # saturating -> v_max
    assert fba.uptake_bound(100000.0) == pytest.approx(vmax, rel=1e-4)
    # at half-saturation -> v_max / 2
    assert fba.uptake_bound(ks) == pytest.approx(vmax / 2.0, rel=1e-6)
    # no substrate -> no uptake
    assert fba.uptake_bound(0.0) == 0.0


def test_uptake_bound_monotonic_in_substrate(fba: DynamicFluxBalance) -> None:
    assert (fba.uptake_bound(0.2) > fba.uptake_bound(0.1)
            > fba.uptake_bound(0.05))


# ---------------------------------------------------------------------------
# batch dynamics
# ---------------------------------------------------------------------------

def test_log_linear_growth_while_glucose_saturating(fba: DynamicFluxBalance) -> None:
    h = fba.run(duration_h=2.0)
    # during the saturating phase the growth rate is ~ constant
    saturating = [e for e in h if e["glucose"] > 1.0]
    assert len(saturating) > 3
    mu_max = _lp_growth(10.0)
    for e in saturating:
        assert e["growth_rate"] == pytest.approx(mu_max, rel=0.2)
    # exponential: ln X linear in time
    ts = [e["time"] for e in saturating]
    lns = [math.log(e["biomass"]) for e in saturating]
    slope = (lns[-1] - lns[0]) / (ts[-1] - ts[0])
    mid = (ts[0] + ts[-1]) / 2.0
    i = min(range(len(ts)), key=lambda k: abs(ts[k] - mid))
    ln_mid = math.log(fba.history[i]["biomass"])
    assert ln_mid == pytest.approx(
        lns[0] + slope * (mid - ts[0]), rel=0.15)


def test_glucose_depletion_arrests_growth(fba: DynamicFluxBalance) -> None:
    h = fba.run(duration_h=8.0)
    last = h[-1]
    assert last["glucose"] < 0.05
    mu_max = max(e["growth_rate"] for e in h)
    assert mu_max > 0.5
    # growth collapses once glucose is exhausted (no glyoxylate shunt in
    # the reduced core model, so acetate cannot re-feed biomass)
    assert last["growth_rate"] < 0.01 * mu_max
    # biomass grows substantially but saturates instead of running away
    assert last["biomass"] > 5.0 * fba.config.initial_biomass_gdw
    assert last["biomass"] < 100.0 * fba.config.initial_biomass_gdw


def test_mass_balance_closes(fba: DynamicFluxBalance) -> None:
    """Produced biomass ~ yield x consumed glucose (LP steady state)."""
    mu = _lp_growth(10.0)
    vg = 10.0
    yield_biomass_per_glucose = mu / vg
    h = fba.run(duration_h=8.0)
    last = h[-1]
    X0 = fba.config.initial_biomass_gdw
    S0 = fba.config.initial_glucose_mm
    predicted = X0 + yield_biomass_per_glucose * (S0 - last["glucose"])
    assert last["biomass"] == pytest.approx(predicted, rel=0.06)


def test_more_uptake_grows_faster() -> None:
    fast = DynamicFluxBalance(config=DynamicFBAConfig(
        dt_h=0.1, initial_glucose_mm=10.0, max_glucose_uptake=10.0))
    slow = DynamicFluxBalance(config=DynamicFBAConfig(
        dt_h=0.1, initial_glucose_mm=10.0, max_glucose_uptake=2.0))
    fast.run(duration_h=1.0)
    slow.run(duration_h=1.0)
    assert fast.last()["biomass"] > slow.last()["biomass"]
    assert fast.last()["growth_rate"] > slow.last()["growth_rate"]


def test_high_half_saturation_slows_depletion() -> None:
    low_ks = DynamicFluxBalance(config=DynamicFBAConfig(
        dt_h=0.1, initial_glucose_mm=10.0, glucose_half_saturation_mm=0.05))
    high_ks = DynamicFluxBalance(config=DynamicFBAConfig(
        dt_h=0.1, initial_glucose_mm=10.0, glucose_half_saturation_mm=5.0))
    low_ks.run(duration_h=2.0)
    high_ks.run(duration_h=2.0)
    # a larger Ks keeps the uptake bound lower for longer -> less glucose
    # consumed over the same window
    assert low_ks.last()["glucose"] < high_ks.last()["glucose"]


def test_no_glucose_no_growth(fba: DynamicFluxBalance) -> None:
    fba.set_state(glucose_mm=0.0)
    h = fba.run(duration_h=2.0)
    for e in h:
        assert e["growth_rate"] == pytest.approx(0.0, abs=1e-9)
    assert fba.last()["biomass"] == pytest.approx(
        fba.config.initial_biomass_gdw, abs=1e-12)


def test_byproduct_co2_tracks_growth(fba: DynamicFluxBalance) -> None:
    fba.run(duration_h=3.0)
    # fully respiring model secretes CO2 that accumulates monotonically
    co2s = [e["co2"] for e in fba.history]
    assert co2s == sorted(co2s)
    assert co2s[-1] > 0.0
    assert fba.last()["acetate"] == pytest.approx(0.0, abs=1e-12)


def test_run_duration_honors_horizon(fba: DynamicFluxBalance) -> None:
    dt = fba.config.dt_h
    fba.run(duration_h=2.0)
    assert len(fba.history) == pytest.approx(2.0 / dt)
    assert fba.last()["time"] == pytest.approx(2.0, rel=1e-6)


def test_step_appends_history_entry(fba: DynamicFluxBalance) -> None:
    assert fba.history == []
    entry = fba.step()
    assert len(fba.history) == 1
    for key in ("time", "biomass", "glucose", "growth_rate",
                "glucose_uptake", "co2", "acetate", "lactate"):
        assert key in entry
    assert entry["time"] == pytest.approx(fba.config.dt_h)


def test_growth_rate_property(fba: DynamicFluxBalance) -> None:
    assert fba.growth_rate == 0.0
    fba.step()
    assert fba.growth_rate == fba.last()["growth_rate"]


# ---------------------------------------------------------------------------
# state control
# ---------------------------------------------------------------------------

def test_reset_restores_initial_state(fba: DynamicFluxBalance) -> None:
    fba.run(duration_h=1.0)
    assert fba.biomass_gdw > fba.config.initial_biomass_gdw
    fba.reset()
    assert fba.history == []
    assert fba.biomass_gdw == pytest.approx(fba.config.initial_biomass_gdw)
    assert fba.glucose_mm == pytest.approx(fba.config.initial_glucose_mm)
    assert fba.time_h == pytest.approx(0.0)


def test_set_state_overrides(fba: DynamicFluxBalance) -> None:
    fba.set_state(biomass_gdw=0.5, glucose_mm=3.0, acetate_mm=1.0)
    assert fba.biomass_gdw == pytest.approx(0.5)
    assert fba.glucose_mm == pytest.approx(3.0)
    assert fba.byproducts_mm["acetate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# environment coupling
# ---------------------------------------------------------------------------

def test_update_from_environment_reads_glucose() -> None:
    env = Environment(EnvironmentConfig(width=11, height=11,
                                        glucose_initial_mm=2.5, flow_rate=0.0))
    d = DynamicFluxBalance(config=DynamicFBAConfig(initial_glucose_mm=10.0))
    d.update_from_environment(env, x=5, y=5)
    assert d.glucose_mm == pytest.approx(2.5)
    # default reads the lattice centre
    d2 = DynamicFluxBalance(config=DynamicFBAConfig(initial_glucose_mm=10.0))
    d2.update_from_environment(env)
    assert d2.glucose_mm == pytest.approx(2.5)


def test_apply_to_environment_creates_and_deposits_acetate() -> None:
    env = Environment(EnvironmentConfig(width=11, height=11, flow_rate=0.0))
    assert "acetate" not in env.fields
    d = DynamicFluxBalance(config=DynamicFBAConfig())
    d.set_state(acetate_mm=0.7)
    d.apply_to_environment(env, x=3, y=4)
    assert "acetate" in env.fields
    assert env.fields["acetate"].get(3, 4) == pytest.approx(0.7)
    # depositing again accumulates on the same field
    d.apply_to_environment(env, x=3, y=4)
    assert env.fields["acetate"].get(3, 4) == pytest.approx(1.4)


def test_environment_glucose_feeds_batch() -> None:
    """Growth is slower when the coupled environment holds less glucose."""
    env = Environment(EnvironmentConfig(width=11, height=11,
                                        glucose_initial_mm=0.3, flow_rate=0.0))
    d = DynamicFluxBalance(config=DynamicFBAConfig(
        dt_h=0.1, initial_glucose_mm=10.0, max_glucose_uptake=10.0))
    d.update_from_environment(env)
    assert d.glucose_mm < 1.0
    d.run(duration_h=1.0)
    assert d.last()["biomass"] < 0.2
