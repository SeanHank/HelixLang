"""Genome-scale dFBA + dAMN-style metabolic proxy tests (T3.2, gap G8).

Verification goals:
- DynamicFluxBalance.bound_override: a state-dependent bound hook changes
  the batch trajectory without touching the integration loop
  (transcriptomics-guided dynamic bounds, MiMICS 2024 / Mahadevan 2002).
- MetabolicProxy (dAMN 2025 surrogate): polynomial regression on sampled
  FBA solutions predicts biomass / byproduct fluxes from uptake bounds
  with bounded error, and irreversible fluxes never go negative.

References:
- Mahadevan et al. 2002 (dynamic FBA, static optimization approach)
- Monk et al. 2017 (iML1515 genome-scale E. coli reconstruction)
- dAMN 2025 (neural-mechanistic surrogate of FBA for dynamic simulation)
- Walsh et al. 2024 MiMICS (transcriptomics-guided metabolic switching)
"""
from __future__ import annotations

import pytest

from helixlang.metabolism import (
    DynamicFBAConfig,
    DynamicFluxBalance,
    FluxBalanceAnalysis,
    MetabolicProxy,
    _poly_features,
)

from helixlang.metabolism import ECOLI_CORE_MODEL


def _batch(**kw):
    return DynamicFBAConfig(dt_h=0.25, **kw)


# ============================================================================
# Dynamic bound overrides
# ============================================================================

def test_bound_override_slows_batch() -> None:
    def override(t_h, dfba):
        # after 1 h, cap glucose uptake at half the MM-derived bound
        if t_h > 1.0:
            return {"EX_glc": dfba.uptake_bound(dfba.glucose_mm) * 0.5}
        return {}

    limited = DynamicFluxBalance(ECOLI_CORE_MODEL, _batch(),
                                 bound_override=override)
    limited.run(duration_h=3.0)
    baseline = DynamicFluxBalance(ECOLI_CORE_MODEL, _batch())
    baseline.run(duration_h=3.0)
    assert limited.history[-1]["biomass"] < baseline.history[-1]["biomass"]


def test_bound_override_called_each_step() -> None:
    calls: list[float] = []

    def override(t_h, dfba):
        calls.append(t_h)
        return {}

    dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, _batch(),
                              bound_override=override)
    dfba.run(duration_h=1.0)
    assert len(calls) == len(dfba.history)
    # the hook fires at the pre-step time, one dt before the recorded entry
    assert calls == [e["time"] - dfba.config.dt_h for e in dfba.history]


def test_bound_override_exchange_and_reaction() -> None:
    def override(t_h, dfba):
        return {"EX_glc": 2.0, "PGI": 1.5}

    dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, _batch(),
                              bound_override=override)
    dfba.step()
    assert dfba.fba.uptake_limits["GLC"] == 2.0
    assert dfba.fba.model.reactions["PGI"].upper_bound == 1.5


def test_bound_override_unknown_reaction_ignored() -> None:
    def override(t_h, dfba):
        return {"NOT_A_REACTION": 9.0}

    dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, _batch(),
                              bound_override=override)
    dfba.step()  # no raise
    assert dfba.history


# ============================================================================
# MetabolicProxy (dAMN-style surrogate)
# ============================================================================

def test_proxy_defaults() -> None:
    proxy = MetabolicProxy(ECOLI_CORE_MODEL)
    assert "GLC" in proxy.features
    assert "BIOMASS" in proxy.outputs
    assert proxy.degree == 2


def test_proxy_no_features_raises() -> None:
    from helixlang.metabolism import MetabolicModel

    empty = MetabolicModel()
    with pytest.raises(ValueError):
        MetabolicProxy(empty)


def test_proxy_predict_before_fit_raises() -> None:
    proxy = MetabolicProxy(ECOLI_CORE_MODEL)
    proxy.coeffs = {}
    with pytest.raises(RuntimeError):
        proxy.predict({"GLC": 5.0})


def test_proxy_biomass_monotonic_in_glucose() -> None:
    proxy = MetabolicProxy(ECOLI_CORE_MODEL, degree=2).fit(n_samples=120,
                                                           seed=0)
    p0 = proxy.predict({"GLC": 0.0})["BIOMASS"]
    p10 = proxy.predict({"GLC": 10.0})["BIOMASS"]
    assert p10 > p0
    assert p0 >= 0.0  # clamped: irreversible biomass never negative


def test_proxy_predict_vector_and_dict_match() -> None:
    proxy = MetabolicProxy(ECOLI_CORE_MODEL, degree=2).fit(n_samples=80,
                                                           seed=2)
    vec = proxy.predict([8.0])
    dct = proxy.predict({"GLC": 8.0})
    assert vec == dct
    with pytest.raises(ValueError):
        proxy.predict({"GLC": 1.0, "Extra": 2.0})


def test_proxy_rmse_bounded() -> None:
    proxy = MetabolicProxy(ECOLI_CORE_MODEL, degree=2).fit(n_samples=150,
                                                           seed=3)
    rmse = proxy.rmse(n_holdout=40, seed=4)
    # biomass is essentially linear in glucose -> tight fit
    assert rmse["BIOMASS"] < 0.05


def test_proxy_outputs_cover_byproducts() -> None:
    proxy = MetabolicProxy(ECOLI_CORE_MODEL).fit(n_samples=30, seed=0)
    pred = proxy.predict({"GLC": 10.0})
    assert set(pred) == set(proxy.outputs)
    assert all(v >= 0.0 for v in pred.values())


def test_poly_features() -> None:
    assert _poly_features([2.0], 1) == [1.0, 2.0]
    assert _poly_features([2.0], 2) == [1.0, 2.0, 4.0]
    # 2 features, degree 2 -> 1 + 2 + 3 = 6 monomials
    assert len(_poly_features([1.0, 2.0], 2)) == 6
    assert _poly_features([1.0, 2.0], 2)[-1] == 4.0  # x1^2
