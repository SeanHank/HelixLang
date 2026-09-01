"""Stochastic ODE solver: Euler-Maruyama for population distributions (doc/32 §8.2).

Extends deterministic ODE solvers with stochastic noise terms to predict the
FULL DISTRIBUTION of drug response, not just the mean. This transforms
"irreducible individual variation" into "predictable population heterogeneity."

Literature:
- END-nSDE (PLOS Comp Bio 2025): intrinsic + extrinsic noise, R² 71.2% → 82.8%
- SDEs in gene regulation (2025): further improvement to R² 84.3%
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SDEConfig:
    """Configuration for stochastic ODE simulation."""

    sigma_intrinsic: float = 0.1
    sigma_extrinsic: float = 0.05
    n_patients: int = 500
    dt: float = 0.1
    seed: int | None = None


@dataclass(frozen=True)
class SDETrajectory:
    """Single stochastic trajectory."""

    times: list[float]
    states: list[float]


@dataclass(frozen=True)
class SDEDistribution:
    """Population distribution from SDE ensemble."""

    times: list[float]
    trajectories: list[list[float]]
    means: list[float]
    stds: list[float]
    percentiles: dict[float, list[float]]
    extreme_events: dict[str, float]


def euler_maruyama_step(
    state: float,
    dt: float,
    drift: float,
    sigma_intrinsic: float,
    sigma_extrinsic: float,
    rng: random.Random,
) -> float:
    """Single Euler-Maruyama step.

    dX = drift * dt + sigma * dW
    where sigma = sigma_intrinsic * sqrt(|X|) + sigma_extrinsic * |X|
    """
    sigma = sigma_intrinsic * math.sqrt(max(abs(state), 1e-10)) + sigma_extrinsic * abs(state)
    dW = rng.gauss(0, math.sqrt(dt))
    return state + drift * dt + sigma * dW


def solve_sde(
    t_end: float,
    dt: float,
    state0: float,
    drift_fn: Callable[[float, float], float],
    sigma_intrinsic: float = 0.1,
    sigma_extrinsic: float = 0.05,
    seed: int | None = None,
) -> SDETrajectory:
    """Solve a scalar SDE using Euler-Maruyama.

    Args:
        t_end: end time
        dt: time step
        state0: initial state
        drift_fn: function(t, state) → drift coefficient
        sigma_intrinsic: intrinsic noise strength (√state scaling)
        sigma_extrinsic: extrinsic noise strength (linear scaling)
        seed: random seed for reproducibility

    Returns:
        SDETrajectory with times and states
    """
    rng = random.Random(seed)
    times = [0.0]
    states = [state0]
    t = 0.0
    state = state0

    while t < t_end - 1e-10:
        drift = drift_fn(t, state)
        state = euler_maruyama_step(
            state, dt, drift, sigma_intrinsic, sigma_extrinsic, rng
        )
        state = max(0.0, state)
        t += dt
        times.append(t)
        states.append(state)

    return SDETrajectory(times=times, states=states)


def _solve_sde_worker(
    t_end: float,
    dt: float,
    state0: float,
    drift_fn: Callable[[float, float], float],
    sigma_intrinsic: float,
    sigma_extrinsic: float,
    seed: int,
) -> list[float]:
    """Top-level worker: solve one SDE trajectory for a distinct seed.

    Module-scope (picklable) so a ``spawn`` pool can run trajectories of an
    ensemble in parallel.  Each trajectory's RNG is independent, so workers'
    order does not change the per-trajectory result and the assembled ensemble
    is identical to the single-process loop.
    """
    traj = solve_sde(
        t_end, dt, state0, drift_fn,
        sigma_intrinsic, sigma_extrinsic,
        seed=seed,
    )
    return traj.states


def solve_sde_ensemble(
    t_end: float,
    dt: float,
    state0: float,
    drift_fn: Callable[[float, float], float],
    config: SDEConfig | None = None,
    *,
    workers: int = 1,
) -> SDEDistribution:
    """Run ensemble of SDE simulations to compute population distribution.

    Args:
        t_end: end time
        dt: time step
        state0: initial state (same for all patients)
        drift_fn: function(t, state) → drift coefficient
        config: SDE configuration (noise levels, n_patients, seed)
        workers: number of processes over which to parallelize the ensemble
            (1 = single-process).  Requires ``drift_fn`` to be picklable when
            ``workers > 1``.  Results are identical for any ``workers`` value.

    Returns:
        SDEDistribution with statistics across ensemble
    """
    if config is None:
        config = SDEConfig()

    base_seed = config.seed if config.seed is not None else 42
    seeds = [base_seed + i for i in range(config.n_patients)]

    if workers <= 1 or config.n_patients <= 1:
        trajectories = [
            _solve_sde_worker(
                t_end, dt, state0, drift_fn,
                config.sigma_intrinsic, config.sigma_extrinsic, seed,
            )
            for seed in seeds
        ]
    else:
        import multiprocessing as _mp

        args = [
            (t_end, dt, state0, drift_fn,
             config.sigma_intrinsic, config.sigma_extrinsic, s)
            for s in seeds
        ]
        ctx = _mp.get_context("spawn")
        with ctx.Pool(processes=max(2, int(workers))) as pool:
            trajectories = pool.starmap(_solve_sde_worker, args)

    n_times = len(trajectories[0]) if trajectories else 0
    times = [i * dt for i in range(n_times)]

    means = []
    stds = []
    for j in range(n_times):
        vals = [traj[j] for traj in trajectories]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        means.append(mu)
        stds.append(math.sqrt(var))

    percentile_levels = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    percentiles: dict[float, list[float]] = {p: [] for p in percentile_levels}
    for j in range(n_times):
        sorted_vals = sorted(traj[j] for traj in trajectories)
        for p in percentile_levels:
            idx = int(p * (len(sorted_vals) - 1))
            percentiles[p].append(sorted_vals[idx])

    final_vals = [traj[-1] for traj in trajectories] if trajectories else []
    extreme_events = {}
    if final_vals:
        threshold_low = means[-1] * 0.3 if means else 0.1
        threshold_high = means[-1] * 3.0 if means else 10.0
        extreme_events["p_below_30pct_mean"] = sum(
            1 for v in final_vals if v < threshold_low
        ) / len(final_vals)
        extreme_events["p_above_300pct_mean"] = sum(
            1 for v in final_vals if v > threshold_high
        ) / len(final_vals)

    return SDEDistribution(
        times=times,
        trajectories=trajectories,
        means=means,
        stds=stds,
        percentiles=percentiles,
        extreme_events=extreme_events,
    )
