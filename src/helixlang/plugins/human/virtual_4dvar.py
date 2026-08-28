"""Virtual 4D-Var Data Assimilation: individual prediction without TDM (doc/32 §7.2).

Uses ALL observable clinical outputs (ALT, creatinine, WBC, glucose, cortisol)
as constraints on hidden PK/PD parameters, analogous to weather forecasting
data assimilation. Enables individual-level prediction without TDM.

Literature:
- CAR-T QSP data assimilation (PMC 2025): R² > 0.96 for individual PK
- CURE4TCR (ScienceDirect 2025): in silico TCR signal PK/PD
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import minimize


@dataclass(frozen=True)
class Observation:
    """A single clinical observation at a specific time."""

    time: float
    variable: str
    value: float
    noise_variance: float = 1.0


@dataclass(frozen=True)
class AssimilationResult:
    """Result of 4D-Var data assimilation."""

    estimated_state: dict[str, float]
    cost_function: float
    convergence: bool
    n_iterations: int
    state_trajectory: list[dict[str, float]]


# physiologically plausible bounds for each PK/PD parameter
_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "ke": (0.01, 2.0),
    "vd": (5.0, 500.0),
    "dose": (50.0, 5000.0),
    "alt": (5.0, 500.0),
    "creatinine": (0.3, 15.0),
    "wbc": (500.0, 30000.0),
    "drug_effect": (0.0, 1.0),
}


class Virtual4DVar:
    """Virtual 4D-Var data assimilation for individual PK prediction.

    Finds the optimal initial state x₀ that minimizes:
        J(x₀) = Σ wᵢ(y_model(tᵢ) - y_obs(tᵢ))²/σᵢ² + λ(x₀ - x_prior)ᵀ P⁻¹ (x₀ - x_prior)

    Uses L-BFGS-B (scipy.optimize.minimize) with box bounds for physiological
    parameter constraints.

    The forward model may be supplied as a callable
    ``forward_model(state, times) -> list[dict]`` returning one observation
    dict per time point (keys = observable variable names); when omitted, a
    built-in 1-compartment PK + linear PD surrogate is used.
    """

    def __init__(
        self,
        forward_model: object | None = None,
        prior_state: dict[str, float] | None = None,
        prior_covariance: dict[str, float] | None = None,
        regularization: float = 1.0,
        parameter_names: list[str] | None = None,
    ) -> None:
        self.forward_model = forward_model
        self.prior_state = prior_state or {}
        self.prior_covariance = prior_covariance or {}
        self.regularization = regularization
        self.parameter_names = (
            list(parameter_names) if parameter_names else sorted(_PARAM_BOUNDS)
        )

    def _forward_simulate(
        self, initial_state: dict[str, float], times: list[float]
    ) -> list[dict[str, float]]:
        """Run the supplied forward model, or the built-in PK/PD surrogate."""
        if callable(self.forward_model):
            return [dict(point) for point in self.forward_model(initial_state, times)]

        trajectory = []
        ke = initial_state.get("ke", 0.15)
        vd = initial_state.get("vd", 50.0)
        dose = initial_state.get("dose", 500.0)
        base_alt = initial_state.get("alt", 25.0)
        base_creatinine = initial_state.get("creatinine", 0.8)
        base_wbc = initial_state.get("wbc", 7000.0)
        drug_effect = initial_state.get("drug_effect", 0.01)

        for t in times:
            concentration = (dose / vd) * math.exp(-ke * t)
            alt = base_alt + drug_effect * concentration * 0.5
            creatinine = base_creatinine + drug_effect * concentration * 0.01
            wbc = base_wbc * (1 - drug_effect * concentration * 0.001)

            trajectory.append({
                "concentration": concentration,
                "alt": alt,
                "creatinine": creatinine,
                "wbc": max(100.0, wbc),
                "time": t,
            })

        return trajectory

    def _cost_and_gradient(
        self,
        x: list[float],
        keys: list[str],
        observations: list[Observation],
        times: list[float],
    ) -> tuple[float, list[float]]:
        """Compute cost function J(x₀) and its gradient via central differences."""
        state = {k: v for k, v in zip(keys, x, strict=True)}

        trajectory = self._forward_simulate(state, times)
        time_idx = {t: i for i, t in enumerate(times)}

        data_cost = 0.0
        for obs in observations:
            idx = time_idx.get(obs.time, 0)
            model_val = trajectory[idx].get(obs.variable, 0.0)
            data_cost += ((model_val - obs.value) ** 2) / max(obs.noise_variance, 1e-10)

        prior_cost = 0.0
        for i, key in enumerate(keys):
            if key in self.prior_state:
                prior_var = self.prior_covariance.get(key, 1.0)
                prior_cost += ((x[i] - self.prior_state[key]) ** 2) / max(prior_var, 1e-10)

        total_cost = data_cost + self.regularization * prior_cost

        # central-difference gradient
        eps = 1e-5
        grad = [0.0] * len(keys)
        for j in range(len(keys)):
            x_plus = list(x)
            x_plus[j] += eps
            state_plus = {k: v for k, v in zip(keys, x_plus, strict=True)}
            traj_plus = self._forward_simulate(state_plus, times)

            x_minus = list(x)
            x_minus[j] -= eps
            state_minus = {k: v for k, v in zip(keys, x_minus, strict=True)}
            traj_minus = self._forward_simulate(state_minus, times)

            cost_plus = 0.0
            cost_minus = 0.0
            for obs in observations:
                idx = time_idx.get(obs.time, 0)
                obs_var = max(obs.noise_variance, 1e-10)
                mp = traj_plus[idx].get(obs.variable, 0.0)
                mm = traj_minus[idx].get(obs.variable, 0.0)
                cost_plus += ((mp - obs.value) ** 2) / obs_var
                cost_minus += ((mm - obs.value) ** 2) / obs_var

            prior_plus = 0.0
            prior_minus = 0.0
            for i, key in enumerate(keys):
                if key in self.prior_state:
                    prior_var = self.prior_covariance.get(key, 1.0)
                    dp = x_plus[i] - self.prior_state[key]
                    dm = x_minus[i] - self.prior_state[key]
                    prior_plus += (dp ** 2) / prior_var
                    prior_minus += (dm ** 2) / prior_var

            total_plus = cost_plus + self.regularization * prior_plus
            total_minus = cost_minus + self.regularization * prior_minus
            grad[j] = (total_plus - total_minus) / (2 * eps)

        return total_cost, grad

    def assimilate(
        self,
        observations: list[Observation],
        max_iterations: int = 200,
        tolerance: float = 1e-8,
    ) -> AssimilationResult:
        """Find optimal initial state using L-BFGS-B (scipy.optimize.minimize).

        Box bounds enforce physiological feasibility of each parameter.
        """
        keys = sorted(set(self.parameter_names) | set(self.prior_state.keys()))
        if not keys:
            return AssimilationResult(
                estimated_state={},
                cost_function=0.0,
                convergence=True,
                n_iterations=0,
                state_trajectory=[],
            )
        default_state = {
            "ke": 0.15, "vd": 50.0, "dose": 500.0,
            "alt": 25.0, "creatinine": 0.8, "wbc": 7000.0,
            "drug_effect": 0.01,
        }
        x0 = [self.prior_state.get(k, default_state.get(k, 1.0)) for k in keys]
        bounds = [_PARAM_BOUNDS.get(k, (1e-6, 1e6)) for k in keys]
        times = sorted({obs.time for obs in observations})

        result = minimize(
            fun=lambda x: self._cost_and_gradient(x, keys, observations, times),
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            jac=True,
            options={"maxiter": max_iterations, "ftol": tolerance, "gtol": 1e-6},
        )

        optimized_state = {k: v for k, v in zip(keys, result.x, strict=True)}
        trajectory = self._forward_simulate(optimized_state, times)

        return AssimilationResult(
            estimated_state=optimized_state,
            cost_function=result.fun,
            convergence=result.success,
            n_iterations=result.nit,
            state_trajectory=trajectory,
        )
