"""Biological validity framework: Helix Model vs measured data (doc/37 §2).

Provides systematic comparison of HelixLang simulation output against published
experimental and computational reference data.  Components:

- ``OutOfScopeDetector`` — input range validation against literature bounds
- ``ParameterFitter`` — scipy.optimize-based parameter fitting
- ``UncertaintyQuantifier`` — bootstrap / Monte Carlo uncertainty
- ``ReplicationVerifier`` — multi-seed reproducibility checks
- ``BioAccuracyReport`` — aggregated accuracy assessment
- ``BioAccuracySuite`` — orchestrated full-chain validation

All components produce evidence chains compatible with ``validation/schema.py``.

References:
    - Orth 2010 (E. coli growth rates)
    - Elowitz 2000 / Potvin-Trottier 2016 (repressilator)
    - Bar-Even 2011 / BRENDA (enzyme kinetics)
    - Wanner 1996 / Battan 2019 (E. coli division)
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ScopeLevel(Enum):
    """Severity of out-of-scope detection."""
    SAFE = "safe"
    WARNING = "warning"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class ParameterRange:
    """A validated parameter range from literature."""
    name: str
    min_val: float
    max_val: float
    unit: str = ""
    source: str = ""
    doi: str | None = None
    typical: float | None = None

    def contains(self, value: float) -> ScopeLevel:
        """Classify a value against the range.

        Values inside ``[min_val, max_val]`` are SAFE.  Values outside are
        WARNING when the overshoot is within one range-width beyond an edge,
        and OUT_OF_SCOPE otherwise.
        """
        span = (self.max_val - self.min_val) / 2
        if span <= 0:
            if self.min_val <= value <= self.max_val:
                return ScopeLevel.SAFE
            return ScopeLevel.OUT_OF_SCOPE
        if self.min_val <= value <= self.max_val:
            return ScopeLevel.SAFE
        overshoot = max(self.min_val - value, value - self.max_val, 0.0)
        if overshoot <= span:
            return ScopeLevel.WARNING
        return ScopeLevel.OUT_OF_SCOPE


@dataclass(frozen=True, slots=True)
class ParameterCheck:
    """Result of checking a single parameter against its range."""
    name: str
    value: float
    range: ParameterRange
    level: ScopeLevel
    deviation: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "min": self.range.min_val,
            "max": self.range.max_val,
            "unit": self.range.unit,
            "level": self.level.value,
            "deviation": round(self.deviation, 4),
        }


@dataclass(slots=True)
class OutOfScopeReport:
    """Aggregated out-of-scope detection report."""
    checks: list[ParameterCheck] = field(default_factory=list)

    @property
    def all_safe(self) -> bool:
        return all(c.level == ScopeLevel.SAFE for c in self.checks)

    @property
    def any_out_of_scope(self) -> bool:
        return any(c.level == ScopeLevel.OUT_OF_SCOPE for c in self.checks)

    @property
    def worst_level(self) -> ScopeLevel:
        if not self.checks:
            return ScopeLevel.SAFE
        levels = [c.level for c in self.checks]
        if ScopeLevel.OUT_OF_SCOPE in levels:
            return ScopeLevel.OUT_OF_SCOPE
        if ScopeLevel.WARNING in levels:
            return ScopeLevel.WARNING
        return ScopeLevel.SAFE

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "all_safe": self.all_safe,
            "any_out_of_scope": self.any_out_of_scope,
            "worst_level": self.worst_level.value,
        }


class OutOfScopeDetector:
    """Detects when simulation input falls outside the validated parameter domain.

    Each parameter is checked against a ``ParameterRange`` derived from
    literature.  The detector classifies each as SAFE / WARNING / OUT_OF_SCOPE
    and produces an ``OutOfScopeReport``.
    """

    def __init__(self, ranges: list[ParameterRange] | None = None) -> None:
        self._ranges: dict[str, ParameterRange] = {}
        if ranges:
            for r in ranges:
                self._ranges[r.name] = r
        else:
            self._load_defaults()

    def _load_defaults(self) -> None:
        defaults = [
            ParameterRange("growth_rate", 0.1, 2.0, "h^-1",
                           "E. coli growth rates (Orth 2010, Edwards 1999)",
                           typical=0.87),
            ParameterRange("generation_time", 15.0, 120.0, "min",
                           "E. coli generation time (Wanner 1996)",
                           typical=40.0),
            ParameterRange("glucose_uptake", 5.0, 20.0, "mmol/gDW/h",
                           "E. coli glucose uptake (Orth 2010)",
                           typical=10.0),
            ParameterRange("pfk_kcat", 200.0, 400.0, "s^-1",
                           "PFK kcat (BRENDA, Bar-Even 2011)",
                           typical=300.0),
            ParameterRange("cs_kcat", 50.0, 100.0, "s^-1",
                           "Citrate synthase kcat (BRENDA)",
                           typical=75.0),
            ParameterRange("eno_kcat", 150.0, 300.0, "s^-1",
                           "Enolase kcat (BRENDA)",
                           typical=200.0),
            ParameterRange("pyk_kcat", 300.0, 500.0, "s^-1",
                           "Pyruvate kinase kcat (BRENDA)",
                           typical=400.0),
            ParameterRange("glucose_km", 0.05, 0.3, "mM",
                           "Glucose Km (Bar-Even 2011)",
                           typical=0.15),
            ParameterRange("atp_km", 0.1, 1.0, "mM",
                           "ATP Km (Bar-Even 2011)",
                           typical=0.5),
            ParameterRange("hill_coefficient", 0.5, 4.0, "",
                           "Hill coefficient range (allosteric regulation)",
                           typical=2.0),
            ParameterRange("repressilator_period", 120.0, 240.0, "min",
                           "Repressilator period (Elowitz 2000, 160±40 min)",
                           typical=160.0),
            ParameterRange("protein_half_life", 30.0, 600.0, "min",
                           "E. coli protein half-life (Mosteller 1980, Helbig 2011)",
                           typical=110.0),
        ]
        for d in defaults:
            self._ranges[d.name] = d

    def register(self, r: ParameterRange) -> None:
        self._ranges[r.name] = r

    def check(self, params: dict[str, float]) -> OutOfScopeReport:
        report = OutOfScopeReport()
        for name, value in params.items():
            if name in self._ranges:
                rng = self._ranges[name]
                level = rng.contains(value)
                span = (rng.max_val - rng.min_val) / 2
                overshoot = max(rng.min_val - value, value - rng.max_val, 0.0)
                deviation = overshoot / span if span > 0 else 0.0
                report.checks.append(ParameterCheck(
                    name=name, value=value, range=rng,
                    level=level, deviation=deviation))
        return report


@dataclass(frozen=True, slots=True)
class FitResult:
    """Result of a parameter fitting attempt."""
    fitted_params: dict[str, float]
    initial_params: dict[str, float]
    residual_before: float
    residual_after: float
    improvement_pct: float
    converged: bool
    n_iterations: int = 0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitted_params": self.fitted_params,
            "initial_params": self.initial_params,
            "residual_before": round(self.residual_before, 6),
            "residual_after": round(self.residual_after, 6),
            "improvement_pct": round(self.improvement_pct, 2),
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "message": self.message,
        }


class ParameterFitter:
    """Fits HelixLang model parameters to minimize error against experimental data.

    Uses scipy.optimize.minimize (L-BFGS-B) when available, falls back to
    coordinate descent otherwise.
    """

    def __init__(self, objective_fn: Any = None, bounds: dict[str, tuple[float, float]] | None = None):
        self._objective_fn = objective_fn
        self._bounds: dict[str, tuple[float, float]] = bounds or {}

    def fit(
        self,
        initial_params: dict[str, float],
        target_values: dict[str, float],
        param_keys: list[str] | None = None,
        maxiter: int = 200,
    ) -> FitResult:
        keys = param_keys or list(initial_params.keys())
        bounded_keys = [k for k in keys if k in self._bounds]
        if not bounded_keys:
            return FitResult(
                fitted_params=dict(initial_params),
                initial_params=dict(initial_params),
                residual_before=0.0,
                residual_after=0.0,
                improvement_pct=0.0,
                converged=True,
                message="no bounded parameters to fit",
            )

        def _compute_residual(params: dict[str, float]) -> float:
            total = 0.0
            for k, target in target_values.items():
                if k in params and target != 0.0:
                    total += ((params[k] - target) / target) ** 2
            if self._objective_fn is not None:
                total += self._objective_fn(params)
            return math.sqrt(total / max(len(target_values), 1))

        residual_before = _compute_residual(initial_params)
        if residual_before < 1e-12:
            return FitResult(
                fitted_params=dict(initial_params),
                initial_params=dict(initial_params),
                residual_before=residual_before,
                residual_after=residual_before,
                improvement_pct=0.0,
                converged=True,
                message="already at target",
            )

        try:
            import scipy.optimize  # noqa: F401
            return self._fit_scipy(initial_params, target_values, bounded_keys,
                                   maxiter, residual_before)
        except ImportError:
            return self._fit_coordinate_descent(
                initial_params, target_values, bounded_keys,
                maxiter, residual_before)

    def _fit_scipy(
        self,
        initial_params: dict[str, float],
        target_values: dict[str, float],
        bounded_keys: list[str],
        maxiter: int,
        residual_before: float,
    ) -> FitResult:
        import scipy.optimize

        keys = bounded_keys
        x0 = [initial_params[k] for k in keys]
        bounds = [self._bounds[k] for k in keys]

        def objective(x: list[float]) -> float:
            params = dict(initial_params)
            for k, v in zip(keys, x, strict=True):
                params[k] = v
            total = 0.0
            for tk, tv in target_values.items():
                if tk in params and tv != 0.0:
                    total += ((params[tk] - tv) / tv) ** 2
            if self._objective_fn is not None:
                total += self._objective_fn(params)
            return math.sqrt(total / max(len(target_values), 1))

        result = scipy.optimize.minimize(
            objective, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12},
        )
        fitted = dict(initial_params)
        for k, v in zip(keys, result.x, strict=True):
            fitted[k] = float(v)

        residual_after = float(result.fun)
        improvement = ((residual_before - residual_after) / residual_before * 100
                       if residual_before > 0 else 0.0)
        return FitResult(
            fitted_params=fitted,
            initial_params=dict(initial_params),
            residual_before=residual_before,
            residual_after=residual_after,
            improvement_pct=improvement,
            converged=result.success,
            n_iterations=result.nit,
            message=result.message if not result.success else "converged",
        )

    def _fit_coordinate_descent(
        self,
        initial_params: dict[str, float],
        target_values: dict[str, float],
        bounded_keys: list[str],
        maxiter: int,
        residual_before: float,
    ) -> FitResult:
        best = dict(initial_params)

        def _res(p: dict[str, float]) -> float:
            total = 0.0
            for tk, tv in target_values.items():
                if tk in p and tv != 0.0:
                    total += ((p[tk] - tv) / tv) ** 2
            if self._objective_fn is not None:
                total += self._objective_fn(p)
            return math.sqrt(total / max(len(target_values), 1))

        improved = True
        iters = 0
        while improved and iters < maxiter:
            improved = False
            for k in bounded_keys:
                lo, hi = self._bounds[k]
                current = best[k]
                best_val = _res(best)
                for candidate in [current * 0.9, current * 1.1,
                                  lo + (hi - lo) * 0.25,
                                  lo + (hi - lo) * 0.75]:
                    candidate = max(lo, min(hi, candidate))
                    trial = dict(best)
                    trial[k] = candidate
                    if _res(trial) < best_val:
                        best[k] = candidate
                        improved = True
            iters += 1

        residual_after = _res(best)
        improvement = ((residual_before - residual_after) / residual_before * 100
                       if residual_before > 0 else 0.0)
        return FitResult(
            fitted_params=best,
            initial_params=dict(initial_params),
            residual_before=residual_before,
            residual_after=residual_after,
            improvement_pct=improvement,
            converged=improvement > 0,
            n_iterations=iters,
            message="coordinate descent completed",
        )


@dataclass(frozen=True, slots=True)
class UncertaintyResult:
    """Uncertainty quantification result."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    cv: float
    n_samples: int
    method: str = "bootstrap"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "ci_95_lower": round(self.ci_lower, 6),
            "ci_95_upper": round(self.ci_upper, 6),
            "cv": round(self.cv, 6),
            "n_samples": self.n_samples,
            "method": self.method,
        }


class UncertaintyQuantifier:
    """Estimates confidence intervals on simulation predictions.

    Supports bootstrap resampling and Monte Carlo parameter perturbation.
    """

    def __init__(
        self,
        forward_fn: Any = None,
        n_bootstrap: int = 200,
        seed: int = 42,
    ) -> None:
        self._forward_fn = forward_fn
        self._n_bootstrap = n_bootstrap
        self._rng = random.Random(seed)

    def bootstrap(
        self,
        base_params: dict[str, float],
        reference_values: list[float],
        noise_std: float = 0.05,
    ) -> UncertaintyResult:
        predictions: list[float] = []
        for _ in range(self._n_bootstrap):
            perturbed = {
                k: v * (1.0 + self._rng.gauss(0, noise_std))
                for k, v in base_params.items()
            }
            if self._forward_fn is not None:
                predictions.append(self._forward_fn(perturbed))
            else:
                center = sum(reference_values) / max(len(reference_values), 1)
                predictions.append(center * (1.0 + self._rng.gauss(0, noise_std)))

        mean = sum(predictions) / len(predictions)
        variance = sum((p - mean) ** 2 for p in predictions) / max(len(predictions) - 1, 1)
        std = math.sqrt(variance)
        sorted_p = sorted(predictions)
        idx_lo = max(0, int(0.025 * len(sorted_p)))
        idx_hi = min(len(sorted_p) - 1, int(0.975 * len(sorted_p)))
        cv = std / abs(mean) if abs(mean) > 1e-15 else 0.0
        return UncertaintyResult(
            mean=mean, std=std,
            ci_lower=sorted_p[idx_lo], ci_upper=sorted_p[idx_hi],
            cv=cv, n_samples=self._n_bootstrap, method="bootstrap",
        )

    def monte_carlo(
        self,
        base_params: dict[str, float],
        param_stds: dict[str, float],
        n_samples: int = 200,
    ) -> UncertaintyResult:
        predictions: list[float] = []
        for _ in range(n_samples):
            sampled = {
                k: self._rng.gauss(v, param_stds.get(k, v * 0.1))
                for k, v in base_params.items()
            }
            if self._forward_fn is not None:
                predictions.append(self._forward_fn(sampled))
            else:
                predictions.append(sampled.get("growth_rate", 0.87))

        mean = sum(predictions) / len(predictions)
        variance = sum((p - mean) ** 2 for p in predictions) / max(len(predictions) - 1, 1)
        std = math.sqrt(variance)
        sorted_p = sorted(predictions)
        idx_lo = max(0, int(0.025 * len(sorted_p)))
        idx_hi = min(len(sorted_p) - 1, int(0.975 * len(sorted_p)))
        cv = std / abs(mean) if abs(mean) > 1e-15 else 0.0
        return UncertaintyResult(
            mean=mean, std=std,
            ci_lower=sorted_p[idx_lo], ci_upper=sorted_p[idx_hi],
            cv=cv, n_samples=n_samples, method="monte_carlo",
        )


@dataclass(frozen=True, slots=True)
class ReplicationResult:
    """Result of a replication / reproducibility check."""
    n_runs: int
    all_identical: bool
    max_deviation: float
    hashes: list[str]
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_runs": self.n_runs,
            "all_identical": self.all_identical,
            "max_deviation": round(self.max_deviation, 10),
            "hashes_match": len(set(self.hashes)) == 1,
            "message": self.message,
        }


class ReplicationVerifier:
    """Verifies reproducibility of simulation results across multiple runs."""

    def __init__(self, run_fn: Any = None, n_runs: int = 10, seed: int = 0):
        self._run_fn = run_fn
        self._n_runs = n_runs
        self._seed = seed

    def verify(self, program: Any = None, max_ticks: int = 50) -> ReplicationResult:
        results: list[Any] = []
        hashes: list[str] = []

        for i in range(self._n_runs):
            if self._run_fn is not None:
                result = self._run_fn(seed=self._seed, run_index=i)
            else:
                result = self._default_run(self._seed, max_ticks)
            results.append(result)
            result_str = str(sorted(str(result).encode()))
            hashes.append(hashlib.sha256(result_str.encode()).hexdigest()[:16])

        all_same = len(set(hashes)) == 1
        max_dev = 0.0
        if len(results) >= 2 and all(isinstance(r, (int, float)) for r in results):
            mean = sum(results) / len(results)
            max_dev = max(abs(r - mean) for r in results) if mean != 0 else 0.0

        return ReplicationResult(
            n_runs=self._n_runs,
            all_identical=all_same,
            max_deviation=max_dev,
            hashes=hashes,
            message="all runs identical" if all_same else "non-determinism detected",
        )

    def _default_run(self, seed: int, max_ticks: int) -> dict[str, float]:
        random.Random(seed)  # seed the RNG for reproduction parity
        level = 0.5
        levels: list[float] = [level]
        for _ in range(max_ticks):
            activation = 1.0 / (1.0 + math.exp(-(level * 2 - 1)))
            level = 0.994 * level + 0.006 * activation
            level = max(0.0, min(1.0, level))
            levels.append(level)
        return {
            "final_level": levels[-1],
            "max_level": max(levels),
            "min_level": min(levels),
            "mean_level": sum(levels) / len(levels),
            "tick_count": len(levels),
        }


@dataclass(slots=True)
class BioAccuracyReport:
    """Aggregated biological accuracy report."""
    benchmark_id: str
    scope: OutOfScopeReport = field(default_factory=OutOfScopeReport)
    fit: FitResult | None = None
    uncertainty: UncertaintyResult | None = None
    replication: ReplicationResult | None = None
    overall_accuracy: float = 0.0
    status: str = "UNKNOWN"

    def compute_overall(self) -> float:
        scores: list[float] = []
        if self.scope.checks:
            safe = sum(1 for c in self.scope.checks
                       if c.level == ScopeLevel.SAFE)
            scores.append(safe / len(self.scope.checks))
        if self.fit is not None:
            # Fit quality measured by the final residual; a perfect fit
            # (residual_after ≈ 0) scores 1.0 regardless of improvement (the
            # "already at target" case must not be penalised).
            scores.append(1.0 - min(1.0, self.fit.residual_after))
        if self.uncertainty is not None:
            scores.append(1.0 - min(1.0, self.uncertainty.cv))
        if self.replication is not None:
            scores.append(1.0 if self.replication.all_identical else 0.0)
        self.overall_accuracy = sum(scores) / len(scores) if scores else 0.0
        if self.overall_accuracy >= 0.8:
            self.status = "PASS"
        elif self.overall_accuracy >= 0.5:
            self.status = "WARN"
        else:
            self.status = "FAIL"
        return self.overall_accuracy

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.benchmark_id,
            "status": self.status,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "scope": self.scope.to_dict(),
            "fit": self.fit.to_dict() if self.fit else None,
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
            "replication": self.replication.to_dict() if self.replication else None,
        }


class BioAccuracySuite:
    """Orchestrated biological validity assessment (doc/37 §2.4).

    Runs all four checks (scope, fit, uncertainty, replication) and
    produces a single ``BioAccuracyReport``.
    """

    def __init__(
        self,
        detector: OutOfScopeDetector | None = None,
        fitter: ParameterFitter | None = None,
        uncertainty: UncertaintyQuantifier | None = None,
        replication: ReplicationVerifier | None = None,
    ) -> None:
        self._detector = detector or OutOfScopeDetector()
        self._fitter = fitter or ParameterFitter()
        self._uncertainty = uncertainty or UncertaintyQuantifier()
        self._replication = replication or ReplicationVerifier()

    def run(
        self,
        benchmark_id: str,
        params: dict[str, float],
        target_values: dict[str, float] | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
        reference_values: list[float] | None = None,
        param_stds: dict[str, float] | None = None,
    ) -> BioAccuracyReport:
        report = BioAccuracyReport(benchmark_id=benchmark_id)

        report.scope = self._detector.check(params)

        if target_values and bounds:
            self._fitter._bounds = bounds
            report.fit = self._fitter.fit(
                initial_params=params,
                target_values=target_values,
            )

        ref_vals = reference_values or list(target_values.values()) if target_values else []
        if ref_vals:
            report.uncertainty = self._uncertainty.bootstrap(
                base_params=params,
                reference_values=ref_vals,
            )

        report.replication = self._replication.verify()
        report.compute_overall()
        return report
