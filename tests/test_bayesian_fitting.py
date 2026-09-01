"""Tests for doc/40 Phase H — Bayesian patient immune calibration.

Covers the 432-parameter IIRABM-style parameter set
(:mod:`helixlang.plugins.human.patient_params`) and the Bayesian re-fitter
(:mod:`helixlang.plugins.human.bayesian_fitter`) with both declared backends:
pymc (HMC/NUTS) and emcee (ensemble).  Backends are never silently chosen —
an unknown backend must raise.
"""
from __future__ import annotations

import math

import pytest

from helixlang.plugins.human.bayesian_fitter import (
    BayesianFitter,
    BayesianFitResult,
    forward_observables,
    posterior_virtual_population,
)
from helixlang.plugins.human.patient_params import (
    N_PARAMS,
    PatientParameterSet,
    PatientParameterTable,
    DOMAIN_SLICES,
    nominal_params,
)


class TestPatientParameterSet:
    def test_432_params_and_domains(self):
        assert N_PARAMS == 432
        total = sum(hi - lo for lo, hi in DOMAIN_SLICES.values())
        assert total == 432
        assert set(DOMAIN_SLICES) >= {
            "proliferation", "cytokine", "affinity", "migration",
            "complement", "pd1", "apr",
        }

    def test_nominal_length_and_positive(self):
        p = nominal_params()
        assert len(p) == 432
        assert all(v > 0 for v in p)

    def test_named_access(self):
        ps = PatientParameterSet()
        assert ps.get("il6_production") == pytest.approx(12.0)
        assert ps.get("cd4_prolif") == pytest.approx(0.06)
        ps.set("cd4_prolif", 0.09)
        assert ps.get("cd4_prolif") == pytest.approx(0.09)

    def test_patient_variant_deterministic(self):
        a = PatientParameterSet.patient_variant(seed=1, patient_idx=2)
        b = PatientParameterSet.patient_variant(seed=1, patient_idx=2)
        assert [round(x, 12) for x in a.to_list()] == \
               [round(x, 12) for x in b.to_list()]
        base = nominal_params()
        assert sum(a.to_list()) != pytest.approx(sum(base))

    def test_table(self):
        t = PatientParameterTable.from_seed(5, seed=3)
        assert len(t) == 5
        assert t.get("patient_2").size == 432


class TestForwardModel:
    def test_deterministic_observables(self):
        a = forward_observables([0.0] * 432, 1.0)
        b = forward_observables([0.0] * 432, 1.0)
        assert a == b

    def test_observable_keys(self):
        obs = forward_observables(nominal_params(), 1.0)
        assert set(obs) >= {"il6_pg_ml", "tnf_pg_ml", "neutrophils", "igg_titer"}
        # At nominal params + stimulus, immune channels should be non-zero.
        assert obs["il6_pg_ml"] > 0
        assert obs["neutrophils"] > 0


class TestBayesianFitter:
    def test_unknown_backend_raises(self):
        # No silent fallback: an unknown backend is an error, never a default.
        with pytest.raises(ValueError):
            BayesianFitter("nuts", {"il6_pg_ml": 10.0})

    def test_forward_observed_roundtrip(self):
        obs = forward_observables(nominal_params(), 1.0)
        fitter = BayesianFitter("emcee", obs, seed=11)
        assert fitter.backend == "emcee"
        pred = forward_observables(nominal_params(), 1.0)
        assert set(obs) == set(pred)
        assert obs == pred

    @pytest.mark.parametrize("backend", ["emcee"])
    def test_emcee_fit_produces_result(self, backend, tmp_path):
        obs = {"il6_pg_ml": 90.0, "tnf_pg_ml": 40.0,
               "neutrophils": 8.0, "igg_titer": 19.0}
        f = BayesianFitter(backend, obs, seed=5)
        r = f.fit(n_walkers=14, n_steps=60, stimulus=1.0)
        assert isinstance(r, BayesianFitResult)
        assert r.backend == "emcee"
        assert len(r.median) == len(r.param_names)
        assert len(r.ci90_lower) == len(r.param_names)
        assert len(r.ci90_upper) == len(r.param_names)
        # cis are ordered (lower <= median <= upper) per parameter
        for lo, med, hi in zip(r.ci90_lower, r.median, r.ci90_upper, strict=True):
            assert lo <= med <= hi

    @pytest.mark.parametrize("backend", ["emcee", "pymc"])
    def test_backends_consistent_contract(self, backend, tmp_path):
        obs = {"il6_pg_ml": 110.0, "tnf_pg_ml": 50.0,
               "neutrophils": 9.0, "igg_titer": 20.0}
        draws = 40 if backend == "pymc" else 40
        tune = 40
        f = BayesianFitter(backend, obs, seed=9)
        r = f.fit(draws=draws, tune=tune, n_chains=1,
                  n_walkers=14, n_steps=50, stimulus=1.0)
        assert isinstance(r, BayesianFitResult)
        assert r.converged
        assert len(r.chains[0]) > 1

    def test_posterior_virtual_population(self):
        obs = {"il6_pg_ml": 80.0, "tnf_pg_ml": 35.0,
               "neutrophils": 7.5, "igg_titer": 17.0}
        f = BayesianFitter("emcee", obs, seed=3)
        r = f.fit(n_walkers=12, n_steps=40, stimulus=1.0)
        pop = posterior_virtual_population(r, n=4, seed=2)
        assert len(pop) == 4
        for ps in pop:
            assert ps.size == 432
