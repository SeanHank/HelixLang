"""Branch-completion tests for helixlang.plugins.human.bayesian_fitter."""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from helixlang.plugins.human.bayesian_fitter import (
    BayesianFitResult,
    BayesianFitter,
    _hill,
    _log_likelihood,
)


def _load_blocking_imports(module_path: str, blocked: set[str]):
    import builtins

    spec = importlib.util.find_spec(module_path)
    throwaway = importlib.util.spec_from_file_location(
        "_no_" + module_path.rsplit(".", 1)[-1], spec.origin)
    old = sys.modules.get(throwaway.name)
    sys.modules[throwaway.name] = mod = importlib.util.module_from_spec(throwaway)
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in blocked:
            raise ImportError(f"{name} blocked")
        return real_import(name, *a, **k)

    builtins.__import__ = fake_import
    try:
        throwaway.loader.exec_module(mod)  # type: ignore[union-attr]
    finally:
        builtins.__import__ = real_import
        if old is None:
            sys.modules.pop(throwaway.name, None)
        else:
            sys.modules[throwaway.name] = old
    return mod


class TestHill:
    def test_zero_half_returns_one(self):
        assert _hill(5.0, 0.0, 2.0) == 1.0
        assert _hill(5.0, -1.0, 2.0) == 1.0

    def test_positive_half(self):
        assert _hill(0.3, 0.3, 2.0) == pytest.approx(0.5)


class TestLogLikelihood:
    def test_unknown_key_skipped(self):
        ll = _log_likelihood([0.0] * 432, {"bogus_channel": 1.0})
        assert ll == 0.0

    def test_zero_sigma_skipped(self):
        obs = {"il6_pg_ml": 100.0}
        ll = _log_likelihood(
            [0.0] * 432, obs, sigma={"il6_pg_ml": 0.0})
        assert ll == 0.0

    def test_negative_sigma_skipped(self):
        obs = {"il6_pg_ml": 100.0}
        ll = _log_likelihood(
            [0.0] * 432, obs, sigma={"il6_pg_ml": -2.0})
        assert ll == 0.0

    def test_full_scoring_negative(self):
        obs = {"il6_pg_ml": 100.0}
        ll = _log_likelihood([0.0] * 432, obs, sigma={"il6_pg_ml": 1.0})
        assert ll < 0.0


class TestMapParams:
    def _result(self):
        return BayesianFitResult(
            backend="emcee",
            param_names=["log_96", "log_97", "plain"],
            chains=[[], [], []],
            map_estimate=[1.0, 3.0, 0.5],
            median=[1.0, 3.0, 0.5],
            ci90_lower=[0.0, 0.0, 0.0],
            ci90_upper=[2.0, 4.0, 1.0],
        )

    def test_map_updates_log_params(self):
        ps = self._result().map_params()
        assert ps.to_list()[96] == pytest.approx(np.exp(1.0))
        assert ps.to_list()[97] == pytest.approx(np.exp(3.0))

    def test_map_skips_non_log_names(self):
        from helixlang.plugins.human.patient_params import PatientParameterSet
        baseline = PatientParameterSet().to_list()
        ps = self._result().map_params()
        vec = ps.to_list()
        assert vec[96] == pytest.approx(np.exp(1.0))
        assert vec[97] == pytest.approx(np.exp(3.0))
        for idx in (0, 7, 100):
            assert vec[idx] == baseline[idx]


class TestMaxLikelihoodNoNumpy:
    def test_restart_without_numpy(self):
        mod = _load_blocking_imports(
            "helixlang.plugins.human.bayesian_fitter", {"numpy"})
        obs = {"il6_pg_ml": 90.0, "tnf_pg_ml": 40.0,
               "neutrophils": 8.0, "igg_titer": 19.0}
        fitter = mod.BayesianFitter("emcee", obs, seed=5)
        best, ll = fitter._max_likelihood()
        assert len(best) == len(fitter.param_indices)
        assert isinstance(ll, float)


class TestNoNumpyBackends:
    def test_pymc_raises_without_numpy(self):
        mod = _load_blocking_imports(
            "helixlang.plugins.human.bayesian_fitter", {"numpy"})
        with pytest.raises(RuntimeError):
            mod.BayesianFitter(
                "pymc", {"il6_pg_ml": 10.0})._fit_pymc(10, 10, 1, 1.0)

    def test_emcee_raises_without_numpy(self):
        mod = _load_blocking_imports(
            "helixlang.plugins.human.bayesian_fitter", {"numpy"})
        with pytest.raises(RuntimeError):
            mod.BayesianFitter(
                "emcee", {"il6_pg_ml": 10.0})._fit_emcee(8, 10, 1.0)


class TestExplicitParams:
    def test_constructor_with_param_indices(self):
        f = BayesianFitter("emcee", {"il6_pg_ml": 10.0},
                           param_indices=[96, 101])
        assert f.param_names == ["log_96", "log_101"]


class TestPymcUnknownChannel:
    def test_unknown_observed_channel_skipped(self):
        obs = {"il6_pg_ml": 90.0, "unmapped_channel": 5.0}
        f = BayesianFitter("pymc", obs, seed=2)
        r = f.fit(draws=10, tune=10, n_chains=1)
        assert r.backend == "pymc"
        assert r.converged


class TestEmceeNonFinite:
    def test_lnprob_nonfinite_rejected(self, monkeypatch):
        import emcee

        captured = {}

        class FakeSampler:
            def __init__(self, n_walkers, ndim, lnprob, *a, **k):
                self.ndim = ndim
                self.lnprob = lnprob

            def run_mcmc(self, pos, steps, progress=False):
                captured["nonfinite"] = self.lnprob(
                    np.full(self.ndim, np.nan))
                captured["finite"] = self.lnprob(
                    np.full(self.ndim, 0.0))

            def get_chain(self, flat=True):
                return np.zeros((16, self.ndim))

        monkeypatch.setattr(emcee, "EnsembleSampler", FakeSampler)
        obs = {"il6_pg_ml": 90.0, "tnf_pg_ml": 40.0,
               "neutrophils": 8.0, "igg_titer": 19.0}
        f = BayesianFitter("emcee", obs, seed=5)
        r = f.fit(n_walkers=14, n_steps=3, stimulus=1.0)
        assert captured["nonfinite"] == -np.inf
        assert np.isfinite(captured["finite"])
        assert r.backend == "emcee"
