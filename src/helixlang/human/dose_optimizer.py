"""Dose Optimizer: Individualized dose optimization (doc/32 §5).

Provides Bayesian dose optimization using simulated PK profiles with:
- Probability of Target Attainment (PTA)
- ECDF distance method
- Bayesian MAP estimation for individual PK parameters

Literature:
- Chotsiri (CPT:PSP 2025): ECDF distance, PTA methods
- PopPK + MIPD (PMC 2025): Bayesian MIPD best practices
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PKProfile:
    """Pharmacokinetic profile (concentration vs time)."""

    times: list[float]
    concentrations: list[float]


@dataclass(frozen=True)
class DoseRecommendation:
    """Result of dose optimization."""

    recommended_dose: float
    regimen: str
    target_auc_range: tuple[float, float]
    predicted_auc: float
    predicted_cmax: float
    predicted_tmin: float
    pta: float
    ecdf_distance: float
    confidence: float = 0.8


@dataclass(frozen=True)
class BayesianEstimate:
    """Bayesian MAP estimate of individual PK parameters."""

    ke: float
    vd: float
    cl: float
    posterior_variance: float
    n_observations: int


class DoseOptimizer:
    """Bayesian dose optimization using simulated PK profiles.

    Usage:
        optimizer = DoseOptimizer(target_range=(10.0, 50.0))
        recommendation = optimizer.recommend_dose(
            dose_range=(100, 500),
            pk_profile=simulated_pk,
        )
    """

    def __init__(
        self,
        target_range: tuple[float, float] = (10.0, 50.0),
        target_auc_range: tuple[float, float] = (100.0, 500.0),
        dosing_interval_hours: float = 12.0,
    ) -> None:
        self.target_range = target_range
        self.target_auc_range = target_auc_range
        self.dosing_interval = dosing_interval_hours

    def compute_auc(self, pk: PKProfile) -> float:
        """Compute AUC using trapezoidal rule."""
        auc = 0.0
        for i in range(1, len(pk.times)):
            dt = pk.times[i] - pk.times[i - 1]
            c_avg = (pk.concentrations[i] + pk.concentrations[i - 1]) / 2.0
            auc += c_avg * dt
        return auc

    def compute_cmax(self, pk: PKProfile) -> float:
        """Compute Cmax from PK profile."""
        return max(pk.concentrations) if pk.concentrations else 0.0

    def compute_tmin(self, pk: PKProfile) -> float:
        """Compute time below minimum effective concentration."""
        min_conc = self.target_range[0]
        t_below = 0.0
        for i in range(1, len(pk.times)):
            dt = pk.times[i] - pk.times[i - 1]
            if pk.concentrations[i] < min_conc:
                t_below += dt
        return t_below

    def pta(
        self,
        predicted_aucs: list[float],
        target_auc: float,
    ) -> float:
        """Probability of target attainment.

        Fraction of predicted AUCs above the target threshold.
        """
        if not predicted_aucs:
            return 0.0
        n_above = sum(1 for a in predicted_aucs if a >= target_auc)
        return n_above / len(predicted_aucs)

    def ecdf_distance(
        self,
        predicted_conc: list[float],
        target_conc: list[float],
    ) -> float:
        """ECDF distance between predicted and target concentration distributions.

        Lower distance = better match.
        """
        if not predicted_conc or not target_conc:
            return float("inf")

        all_vals = sorted(set(predicted_conc + target_conc))
        max_dist = 0.0

        for val in all_vals:
            ecdf_pred = sum(1 for c in predicted_conc if c <= val) / len(predicted_conc)
            ecdf_target = sum(1 for c in target_conc if c <= val) / len(target_conc)
            dist = abs(ecdf_pred - ecdf_target)
            max_dist = max(max_dist, dist)

        return max_dist

    def recommend_dose(
        self,
        dose_range: tuple[float, float],
        pk_generator: object,
        n_simulations: int = 100,
        population_size: int = 50,
    ) -> DoseRecommendation:
        """Find optimal dose within range using population PTA + ECDF distance.

        Simulates a population of virtual patients with inter-individual
        variability (IIV) for each dose candidate and computes PTA as the
        fraction of patients achieving the target AUC.

        Args:
            dose_range: (min_dose, max_dose) to explore
            pk_generator: object with generate(dose, ke_modifier, vd_modifier) → PKProfile
            n_simulations: number of dose candidates to evaluate
            population_size: number of virtual patients per dose (IIV sampling)

        Returns:
            DoseRecommendation with optimal dose and metrics
        """
        import random
        rng = random.Random(42)
        min_dose, max_dose = dose_range
        best_dose = min_dose
        best_score = float("inf")
        best_pta = 0.0
        best_ecdf = float("inf")
        best_auc = 0.0
        best_cmax = 0.0

        for i in range(n_simulations):
            dose = min_dose + (max_dose - min_dose) * i / max(n_simulations - 1, 1)

            # simulate population with IIV (log-normal CV ~30% for ke/vd)
            pop_aucs: list[float] = []
            pop_concs: list[float] = []
            for _ in range(population_size):
                ke_mod = rng.lognormvariate(0.0, 0.30)
                vd_mod = rng.lognormvariate(0.0, 0.30)
                try:
                    pk = pk_generator.generate(dose, ke_modifier=ke_mod, vd_modifier=vd_mod)  # type: ignore[attr-defined]
                except TypeError:
                    pk = pk_generator.generate(dose)  # type: ignore[attr-defined]
                pop_aucs.append(self.compute_auc(pk))
                pop_concs.extend(pk.concentrations)

            auc_mean = sum(pop_aucs) / len(pop_aucs) if pop_aucs else 0.0
            cmax_mean = max(pop_concs) if pop_concs else 0.0

            pta_val = self.pta(pop_aucs, self.target_auc_range[0])

            target_conc = [
                self.target_range[0] + (self.target_range[1] - self.target_range[0]) * j / 100
                for j in range(101)
            ]
            ecdf_dist = self.ecdf_distance(pop_concs, target_conc)

            score = ecdf_dist - 0.5 * pta_val

            if score < best_score:
                best_score = score
                best_dose = dose
                best_pta = pta_val
                best_ecdf = ecdf_dist
                best_auc = auc_mean
                best_cmax = cmax_mean

        return DoseRecommendation(
            recommended_dose=best_dose,
            regimen=f"{best_dose:.0f} mg q{self.dosing_interval:.0f}h",
            target_auc_range=self.target_auc_range,
            predicted_auc=best_auc,
            predicted_cmax=best_cmax,
            predicted_tmin=0.0,
            pta=best_pta,
            ecdf_distance=best_ecdf,
        )

    def bayesian_map_estimate(
        self,
        observed_times: list[float],
        observed_concs: list[float],
        prior_ke: float = 0.15,
        prior_vd: float = 50.0,
        prior_var_ke: float = 0.01,
        prior_var_vd: float = 100.0,
        noise_var: float = 0.01,
    ) -> BayesianEstimate:
        """Bayesian MAP estimation of individual PK parameters.

        Uses 1-compartment IV model:
            C(t) = (D/Vd) * exp(-ke * t)
        Linearized: ln(C) = ln(D/Vd) - ke * t
        MAP estimate with Gaussian prior (conjugate update).

        The MAP posterior mean for slope (ke) is:
            mu_post = (prior_precision * prior_ke + data_precision * data_slope) /
                      (prior_precision + data_precision)

        The MAP posterior mean for intercept (ln(Vd)) is:
            mu_post = (prior_precision * ln(prior_vd) + data_precision * data_intercept) /
                      (prior_precision + data_precision)
        """
        if not observed_times or not observed_concs:
            return BayesianEstimate(
                ke=prior_ke, vd=prior_vd,
                cl=prior_ke * prior_vd,
                posterior_variance=prior_var_ke + prior_var_vd,
                n_observations=0,
            )

        safe_concs = [max(c, 1e-10) for c in observed_concs]
        log_concs = [math.log(c) for c in safe_concs]

        n = len(observed_times)
        sum_t = sum(observed_times)
        sum_t2 = sum(t**2 for t in observed_times)
        sum_y = sum(log_concs)
        sum_ty = sum(t * y for t, y in zip(observed_times, log_concs, strict=True))

        # data precision per observation
        # --- ke (slope) MAP estimate ---
        # OLS slope: beta = (n*Σty - Σt*Σy) / (n*Σt² - (Σt)²)
        denom_slope = n * sum_t2 - sum_t ** 2
        if abs(denom_slope) < 1e-15:
            ke_est = prior_ke
            post_var_ke = prior_var_ke
        else:
            ols_slope = (n * sum_ty - sum_t * sum_y) / denom_slope
            ols_slope_precision = abs(denom_slope) / (n * max(noise_var, 1e-10))
            prior_prec_ke = 1.0 / max(prior_var_ke, 1e-10)
            post_var_ke = 1.0 / (prior_prec_ke + ols_slope_precision)
            ke_post_mean = post_var_ke * (prior_prec_ke * prior_ke + ols_slope_precision * (-ols_slope))
            ke_est = max(0.01, ke_post_mean)

        # --- intercept (ln(D/Vd)) MAP estimate ---
        # OLS intercept: alpha = (Σy - beta*Σt) / n
        ols_intercept = (sum_y + ols_slope * sum_t) / n if denom_slope > 1e-15 else math.log(prior_vd)
        intercept_precision = n / max(noise_var, 1e-10)
        prior_prec_vd = 1.0 / max(prior_var_vd, 1e-10)
        post_var_vd = 1.0 / (prior_prec_vd + intercept_precision)
        intercept_post_mean = post_var_vd * (prior_prec_vd * math.log(prior_vd) + intercept_precision * ols_intercept)
        vd_est = max(5.0, math.exp(intercept_post_mean))
        cl_est = ke_est * vd_est

        total_var = post_var_ke + post_var_vd

        return BayesianEstimate(
            ke=ke_est,
            vd=vd_est,
            cl=cl_est,
            posterior_variance=total_var,
            n_observations=n,
        )
