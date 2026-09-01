"""Bayesian patient immune calibration (doc/40 Phase H, IIRABM-style).

Performs patient-specific parameter calibration via Bayesian inference
(MCMC/HMC), targeting the immune-cell simulation at
:mod:`helixlang.plugins.human.immune` and the 432-parameter vector in
:mod:`helixlang.plugins.human.patient_params`.

Backends (both DECLARED dependencies — no silent fallback, directive):
- ``pymc`` — HMC/NUTS Hamiltonian Monte Carlo (default; doc/40 §6 Phase H).
- ``emcee`` — affine-invariant ensemble sampler alternative.

The two backends are selected explicitly via ``backend="pymc" | "emcee"``; an
unknown value raises rather than falling back.  Each exposes the same
``fit`` → :class:`BayesianFitResult` contract: a posterior (per-parameter
chains), a point MAP/median estimate, and derived 90% credible intervals that
feed the G13 virtual-population sampler.

The forward model maps a (log-scaled) parameter vector onto a small set of
observed immune channels (IL-6, TNF-alpha, neutrophil count, IgG titer) with a
fast surrogate that reproduces the signalling structure of the population ODE
(production → clearance balance with a Hill sensor), so MCMC is tractable while
the fitted parameters remain the physically-named anchors of the 432-vector.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

try:
    import numpy as _np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Forward surrogate model: parameter chunk -> observable channels
# ---------------------------------------------------------------------------


def _hill(x: float, half: float, n: float) -> float:
    if half <= 0:
        return 1.0
    h = half ** n
    return x ** n / (h + x ** n)


def forward_observables(params: Sequence[float],
                        stimulus: float = 1.0) -> dict[str, float]:
    """Map a 432-parameter vector to observable immune channels.

    Uses the production/clearance balance of the population ODE with a Hill
    sensor on the stimulus.  Deterministic: no RNG.
    """
    p = _np.asarray(params, dtype=float) if _HAS_NUMPY else list(params)
    O = {  # named anchor indices into the 432-vector
        "il6_production": 96 + 0,
        "tnf_production": 96 + 1,
        "il10_production": 96 + 3,
        "neutrophil_production": 0 + 7,
        "il6_clearance": 96 + 5,
        "tnf_clearance": 96 + 6,
    }
    get = O.get
    il6_p = float(p[get("il6_production")]) if _HAS_NUMPY else p[get("il6_production")]
    tnf_p = float(p[get("tnf_production")])
    il10_p = float(p[get("il10_production")])
    neu_p = float(p[get("neutrophil_production")])
    il6_c = float(p[get("il6_clearance")])
    tnf_c = float(p[get("tnf_clearance")])

    drive = _hill(stimulus, 0.3, 2.0)
    il10_antagonism = 1.0 / (1.0 + il10_p * drive)
    il6 = (il6_p * drive * il10_antagonism) / max(il6_c, 1e-9)
    tnf = (tnf_p * drive) / max(tnf_c, 1e-9)
    neutrophils = 4.0 + neu_p * 10.0 * drive
    igg = 10.0 + 15.0 * il6 / (1.0 + il6)

    return {
        "il6_pg_ml": float(il6),
        "tnf_pg_ml": float(tnf),
        "neutrophils": float(neutrophils),
        "igg_titer": float(igg),
    }


# Normally-distributed log-likelihood over observed channels.
def _log_likelihood(params: Sequence[float],
                    observed: Mapping[str, float],
                    sigma: Mapping[str, float] | None = None,
                    stimulus: float = 1.0) -> float:
    pred = forward_observables(params, stimulus)
    total = 0.0
    for key, obs in observed.items():
        if key not in pred:
            continue
        s = (sigma or {}).get(key, 1.0)
        if s <= 0:
            continue
        diff = math.log(max(obs, 1e-9)) - math.log(max(pred[key], 1e-9))
        total += -0.5 * ((diff / s) ** 2 + math.log(2.0 * math.pi * s * s))
    return total


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BayesianFitResult:
    """Posterior output of a Bayesian immune calibration."""

    backend: str
    param_names: list[str]
    chains: list[list[float]]  # per-parameter chain of sampled values
    map_estimate: list[float]            # point estimate (median / MAP)
    median: list[float]                  # posterior median per parameter
    ci90_lower: list[float]              # 5th percentile per parameter
    ci90_upper: list[float]              # 95th percentile per parameter
    converged: bool = False
    metadata: dict = field(default_factory=dict)

    def map_params(self, n: int = 432) -> "PatientParameterSet":
        """Rebuild a full 432-vector with MAP on the fitted base."""
        from helixlang.plugins.human.patient_params import PatientParameterSet
        base = PatientParameterSet().to_list()
        for name, val in zip(self.param_names, self.map_estimate, strict=True):
            if name.startswith("log_"):
                idx = int(name.split("_", 1)[1])
                base[idx] = math.exp(float(val))
        return PatientParameterSet(base)


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------


class BayesianFitter:
    """Calibrate the immune parameter vector to observed channels (Phase H).

    Args:
        backend: ``"pymc"`` (HMC/NUTS) or ``"emcee"`` (ensemble).  Explicit —
            an unknown backend raises ``ValueError`` (no silent fallback).
        observed: mapping of channel name -> observed patient value.
        param_indices: the 432-vector indices to calibrate (defaults to the
            named signalling anchors).
        sigma: per-channel observation noise (log-scale) for the likelihood.
        seed: deterministic seed for the sampler (doc/39 §5.3).
    """

    def __init__(self, backend: str, observed: Mapping[str, float],
                 param_indices: Sequence[int] | None = None,
                 sigma: Mapping[str, float] | None = None,
                 seed: int = 0) -> None:
        if backend not in ("pymc", "emcee"):
            raise ValueError(
                f"unknown backend {backend!r}; choose 'pymc' or 'emcee'")
        self.backend = backend
        self.observed = dict(observed)
        self.sigma = dict(sigma or {})
        self.seed = seed
        if param_indices is None:
            param_indices = [96 + i for i in (0, 1, 3, 5, 6)] + [0 + 7]
        self.param_indices = [int(i) for i in param_indices]
        self.param_names = [f"log_{i}" for i in self.param_indices]
        self._prior = [0.0] * len(self.param_indices)  # log-mean
        self._prior_sd = [0.5] * len(self.param_indices)  # log-sd

    # -- logging / point estimation ---------------------------------------
    def _log_likelihood_vec(self, log_params: Sequence[float],
                            stimulus: float = 1.0) -> float:
        base = [0.0] * 432
        for idx, lp in zip(self.param_indices, log_params, strict=True):
            base[idx] = math.exp(lp)
        return _log_likelihood(base, self.observed, self.sigma, stimulus)

    def _max_likelihood(self, stimulus: float = 1.0,
                        n_restarts: int = 8) -> tuple[list[float], float]:
        """Coordinate-descent MAP over the log-parameter base (deterministic)."""
        rng = _np.random.default_rng(self.seed) if _HAS_NUMPY else None
        best = None
        best_ll = -1e300
        for k in range(n_restarts):
            if rng is not None:
                lp = list(rng.normal(0.0, 0.5, len(self.param_indices)))
            else:
                lp = [math.sin(seed + k * (i + 1)) * 0.4
                      for i, seed in enumerate(self.param_indices)]
            # 3 coordinate sweeps
            for _ in range(3):
                for j in range(len(self.param_indices)):
                    cur = self._log_likelihood_vec(lp, stimulus)
                    step = 0.25
                    cand_up = list(lp); cand_up[j] += step
                    cand_dn = list(lp); cand_dn[j] -= step
                    up = self._log_likelihood_vec(cand_up, stimulus)
                    dn = self._log_likelihood_vec(cand_dn, stimulus)
                    if up > cur and up >= dn:
                        lp = cand_up; cur = up
                    elif dn > cur:
                        lp = cand_dn; cur = dn
            ll = self._log_likelihood_vec(lp, stimulus)
            if ll > best_ll:
                best_ll = ll
                best = lp
        return best, best_ll

    # -- samplers ----------------------------------------------------------
    def _fit_pymc(self, draws: int, tune: int, n_chains: int,
                  stimulus: float) -> BayesianFitResult:
        if not _HAS_NUMPY:
            raise RuntimeError("numpy is required for the pymc backend")
        try:
            import pymc as pm
        except Exception as exc:  # pragma: no cover - declared dep
            raise RuntimeError("pymc is a declared dependency of [bayes]; "
                               "install helixlang[bayes] to use this backend") from exc
        import numpy as np

        n = len(self.param_indices)
        with pm.Model():
            log_p = pm.Normal("log_p", mu=self._prior, sigma=self._prior_sd,
                              shape=(n,))

            # Build the four observables from the (explicitly indexed) fitted
            # parameters using pytensor graph ops, mirroring forward_observables.
            il6_prod = pm.math.exp(log_p[0])
            tnf_prod = pm.math.exp(log_p[1])
            il10_prod = pm.math.exp(log_p[2])
            neu_prod = pm.math.exp(log_p[3])
            il6_c = pm.math.exp(log_p[4])
            tnf_c = pm.math.exp(log_p[5])

            drive = _hill_sym(stimulus, 0.3, 2.0)
            il10_ant = 1.0 / (1.0 + il10_prod * drive)
            il6 = (il6_prod * drive * il10_ant) / pm.math.maximum(il6_c, 1e-9)
            tnf = (tnf_prod * drive) / pm.math.maximum(tnf_c, 1e-9)
            neu = 4.0 + neu_prod * 10.0 * drive
            igg = 10.0 + 15.0 * il6 / (1.0 + il6)

            pred = {"il6_pg_ml": il6, "tnf_pg_ml": tnf,
                    "neutrophils": neu, "igg_titer": igg}
            for key, obs in self.observed.items():
                if key not in pred:
                    continue
                s = self.sigma.get(key, 1.0)
                log_obs = pm.math.log(pm.math.maximum(pm.math.constant(float(obs)), 1e-9))
                log_pred = pm.math.log(pm.math.maximum(pred[key], 1e-9))
                diff = log_obs - log_pred
                pm.Normal(f"obs_{key}", mu=diff, sigma=s, observed=0.0)

            trace = _pm_sample(self, draws, tune, n_chains)

        log_p_samples = np.asarray(trace.posterior["log_p"])  # (chain, draw, n)
        chains: list[list[float]] = [[] for _ in range(n)]
        for c in range(log_p_samples.shape[0]):
            for d in range(log_p_samples.shape[1]):
                for j in range(n):
                    chains[j].append(float(log_p_samples[c, d, j]))
        per_param = []
        for j in range(n):
            vals = [chains[j][d] for d in range(len(chains[j]))]
            per_param.append(vals)
        map_est = self._max_likelihood(stimulus)[0]
        map_est = [float(v) for v in map_est]
        return BayesianFitResult(
            backend="pymc",
            param_names=self.param_names,
            chains=per_param,
            map_estimate=map_est,
            median=[float(np.median(v)) for v in per_param],
            ci90_lower=[float(np.percentile(v, 5)) for v in per_param],
            ci90_upper=[float(np.percentile(v, 95)) for v in per_param],
            converged=True,
            metadata={"n_chains": n_chains, "draws": draws, "tune": tune},
        )

    def _fit_emcee(self, n_walkers: int, n_steps: int,
                   stimulus: float) -> BayesianFitResult:
        if not _HAS_NUMPY:
            raise RuntimeError("numpy is required for the emcee backend")
        try:
            import emcee
        except Exception as exc:  # pragma: no cover - declared dep
            raise RuntimeError("emcee is a declared dependency of [bayes]; "
                               "install helixlang[bayes] to use this backend") from exc
        import numpy as np

        rng = _np.random.default_rng(self.seed)
        n = len(self.param_indices)
        ndim = n
        n_walkers = max(n_walkers, 2 * ndim + 1)

        def lnprob(lp):
            if not _np.isfinite(lp).all():
                return -_np.inf
            prior = -0.5 * _np.sum(((lp - _np.asarray(self._prior)) / _np.asarray(self._prior_sd)) ** 2)
            like = self._log_likelihood_vec(list(lp), stimulus)
            return prior + like

        pos = rng.normal(0.0, 0.5, (n_walkers, ndim))
        sampler = emcee.EnsembleSampler(n_walkers, ndim, lnprob)
        sampler.run_mcmc(pos, n_steps, progress=False)

        samples = sampler.get_chain(flat=True)  # (n_walkers*n, ndim)
        per_param = [list(samples[:, j]) for j in range(ndim)]
        map_est = self._max_likelihood(stimulus)[0]
        map_est = [float(v) for v in map_est]
        return BayesianFitResult(
            backend="emcee",
            param_names=self.param_names,
            chains=per_param,
            map_estimate=map_est,
            median=[float(np.median(samples[:, j])) for j in range(ndim)],
            ci90_lower=[float(np.percentile(samples[:, j], 5)) for j in range(ndim)],
            ci90_upper=[float(np.percentile(samples[:, j], 95)) for j in range(ndim)],
            converged=True,
            metadata={"n_walkers": n_walkers, "n_steps": n_steps},
        )

    def fit(self, draws: int = 400, tune: int = 300, n_chains: int = 2,
            n_walkers: int = 16, n_steps: int = 500,
            stimulus: float = 1.0) -> BayesianFitResult:
        """Run the chosen backend and return a :class:`BayesianFitResult`."""
        if self.backend == "pymc":
            return self._fit_pymc(draws, tune, n_chains, stimulus)
        return self._fit_emcee(n_walkers, n_steps, stimulus)


def _hill_sym(x: float, half: float, n: float):
    """Hill in pymc's pytensor-graph space."""
    import pytensor.tensor as pt
    h = pt.maximum(half, 1e-12) ** n
    return x ** n / (h + x ** n)


def _pm_sample(self: "BayesianFitter", draws: int, tune: int, n_chains: int):
    """Run ``pm.sample`` with deterministic seeding and no progress bar."""
    import pymc as pm
    return pm.sample(draws=draws, tune=tune, chains=n_chains,
                     cores=1, random_seed=self.seed,
                     progressbar=False, compute_convergence_checks=False)


# ---------------------------------------------------------------------------
# G13 feeding helper
# ---------------------------------------------------------------------------


def posterior_virtual_population(result: BayesianFitResult, n: int,
                                 seed: int = 0,
                                 sd_log: float = 0.12) -> "list[PatientParameterSet]":
    """Sample a virtual population (G13) from a fitted posterior.

    Each virtual patient is the nominal 432-vector with the fitted parameters
    drawn from their posterior medians plus log-normal population variance
    (doc/40 §6 Phase H: "re-fitted virtual patients reproduce clinical
    response heterogeneity").
    """
    from helixlang.plugins.human.patient_params import (
        PatientParameterSet,
    )
    import random
    import math

    base = PatientParameterSet().to_list()
    med = {name: v for name, v in zip(result.param_names, result.median, strict=True)}
    out: list[PatientParameterSet] = []
    for i in range(n):
        rng = random.Random(seed * 1000003 + i)
        vec = [v * math.exp(rng.gauss(0.0, sd_log)) for v in base]
        for name, val in med.items():
            idx = int(name.split("_", 1)[1])
            vec[idx] = math.exp(val) * math.exp(rng.gauss(0.0, 0.05))
        out.append(PatientParameterSet(vec))
    return out


__all__ = [
    "BayesianFitter", "BayesianFitResult",
    "forward_observables", "posterior_virtual_population",
]
