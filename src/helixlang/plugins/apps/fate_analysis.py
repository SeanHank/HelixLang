"""Cell-fate decision analysis: bistability scan + stochastic switching (S7).

A two-gene mutual-repression *toggle switch* (Gardner, Cantor & Collins
2000 Nature 403:339-342) is the canonical model of a binary cell-fate
decision: two differentiated attractors (fate A vs fate B) separated by
an unstable boundary state.  This module supplies three complementary
views of the decision machinery:

1. :func:`bistability_scan` -- deterministic bifurcation analysis.  For
   each repression strength ``w`` the fixed points of the toggle map
   ``a* = S(-w * S(-w * a*))`` (``S`` = sigmoid) are located by
   root-finding and classified stable/unstable by the local map slope.
   Below the saddle-node the circuit is monostable; above it two stable
   fates coexist (Gardner 2000; Ozbudak et al. 2004 Science 306:1378).

2. :func:`switching_rate` -- stochastic Monte-Carlo.  Telegraph-promoter
   noise (Peccoud & Ycart 1995; ``TelegraphPromoter``/``fano_to_noise_std``
   from :mod:`helixlang.plugins.runtime.grn`) drives spontaneous fate flips.  A shared
   translation-resource term (Goetz et al. 2025, resource-competition
   destabilization of the toggle) throttles both gene products
   simultaneously, collapsing the barrier and amplifying switching.

3. :func:`critical_slowing_down` -- near the bifurcation the return rate
   to the (monostable) fixed point vanishes: the lag-1 autocorrelation of
   a single long noisy trajectory approaches 1 (Scheffer et al. 2009
   Nature 461:53; "critical slowing down" as an early-warning signal).
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from helixlang.plugins.runtime.grn import GRN, fano_to_noise_std, sigmoid

A_GENE = "a"
B_GENE = "b"

_DEFAULT_INITIAL = (0.9, 0.1)
_FIXED_POINT_GRID_STEP = 1.0 / 2000.0
_FIXED_POINT_TOL = 1e-4


@dataclass(frozen=True, slots=True)
class StableState:
    """One stationary fate of the toggle, ``(a, b)`` normalized levels."""

    a: float
    b: float


@dataclass(slots=True)
class BifurcationPoint:
    """Fixed-point structure of the toggle at one repression strength."""

    parameter: float
    stable_states: list[StableState]
    unstable_states: list[StableState]

    @property
    def n_stable_states(self) -> int:
        """Number of stable fates (1 monostable, 2 bistable)."""
        return len(self.stable_states)

    @property
    def is_bistable(self) -> bool:
        """True when two differentiated fates coexist."""
        return len(self.stable_states) == 2


def make_toggle_grn(
    repression_strength: float,
    decay: float = 0.5,
    a0: float = 0.5,
    b0: float = 0.5,
) -> GRN:
    """Build the deterministic Gardner-2000 toggle as a :class:`GRN`.

    Two mutually inhibitory genes with ``threshold = 0`` (constitutive
    half-max basal) and negative edge weights ``-w``, matching the
    continuous GRN integration used across the library.
    """
    grn = GRN()
    grn.add_gene(A_GENE, threshold=0.0, initial_level=a0, decay=decay)
    grn.add_gene(B_GENE, threshold=0.0, initial_level=b0, decay=decay)
    grn.add_edge(A_GENE, B_GENE, -repression_strength)
    grn.add_edge(B_GENE, A_GENE, -repression_strength)
    return grn


def _toggle_map_a(a: float, w: float) -> float:
    """Fixed-point map for the A level: ``a* = S(-w S(-w a*))``."""
    return sigmoid(-w * sigmoid(-w * a))


def _fixed_points(w: float) -> tuple[list[StableState], list[StableState]]:
    """Locate stable and unstable fixed points of the toggle map."""
    grid = [i * _FIXED_POINT_GRID_STEP for i in range(1, 2000)]
    values = [x - _toggle_map_a(x, w) for x in grid]
    roots: list[float] = []
    for i in range(len(values) - 1):
        if values[i] * values[i + 1] <= 0.0:
            lo, hi = grid[i], grid[i + 1]
            flo = values[i]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                fmid = mid - _toggle_map_a(mid, w)
                if fmid * flo <= 0.0:
                    hi = mid
                else:
                    lo = mid
                    flo = fmid
            roots.append(0.5 * (lo + hi))
    uniq: list[float] = []
    for x in roots:
        if not uniq or abs(x - uniq[-1]) > _FIXED_POINT_TOL:
            uniq.append(x)
    stable: list[StableState] = []
    unstable: list[StableState] = []
    for a in uniq:
        slope = (
            _toggle_map_a(a + 1e-6, w) - _toggle_map_a(a - 1e-6, w)
        ) / 2e-6
        state = StableState(a=a, b=sigmoid(-w * a))
        if abs(slope) < 1.0:
            stable.append(state)
        else:
            unstable.append(state)
    return stable, unstable


def bistability_scan(
    w_values: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
) -> list[BifurcationPoint]:
    """Bifurcation scan over repression strength.

    Returns one :class:`BifurcationPoint` per ``w``; ``n_stable_states``
    is 1 below the saddle-node and 2 above it, and the stable
    ``(a, b)`` levels trace the classic supercritical pitchfork /
    saddle-node birth of the two fates.
    """
    results: list[BifurcationPoint] = []
    for w in w_values:
        stable, unstable = _fixed_points(w)
        results.append(
            BifurcationPoint(parameter=w, stable_states=stable, unstable_states=unstable)
        )
    return results


def fate(a: float, b: float) -> str:
    """Classify a state as the ``"a"`` or ``"b"`` fate by dominance."""
    return A_GENE if a >= b else B_GENE


def simulate_toggle_trajectories(
    w: float,
    n_trajectories: int = 600,
    n_ticks: int = 200,
    noise_fano: float = 2.0,
    expression_scale: float = 1000.0,
    decay: float = 0.5,
    resource_strength: float = 0.0,
    initial_state: tuple[float, float] = _DEFAULT_INITIAL,
    seed: int | None = 0,
) -> list[tuple[float, float]]:
    """Monte-Carlo trajectories of the noisy toggle.

    The recurrence mirrors the discrete GRN update
    ``level' = decay*level + (1-decay)*raw + eta`` with telegraph
    Fano-matched Gaussian noise (:func:`fano_to_noise_std`).  A shared
    resource pool throttles both production terms by
    ``1 / (1 + resource_strength*(a+b))`` (Goetz et al. 2025): high
    simultaneous demand squeezes the loop gain toward the bifurcation.

    Returns the ``(a, b)`` final levels of each trajectory.
    """
    if resource_strength < 0.0:
        raise ValueError("resource_strength must be >= 0")
    rng = random.Random(seed)
    a0, b0 = initial_state
    finals: list[tuple[float, float]] = []
    for _ in range(n_trajectories):
        a, b = a0, b0
        for _ in range(n_ticks):
            pool = 1.0 + resource_strength * (a + b)
            a_raw = sigmoid(-w * b) / pool
            b_raw = sigmoid(-w * a) / pool
            eta_a = fano_to_noise_std(noise_fano, a_raw, decay, expression_scale)
            eta_b = fano_to_noise_std(noise_fano, b_raw, decay, expression_scale)
            a = max(0.0, min(1.0, decay * a + (1.0 - decay) * a_raw + rng.gauss(0.0, eta_a)))
            b = max(0.0, min(1.0, decay * b + (1.0 - decay) * b_raw + rng.gauss(0.0, eta_b)))
        finals.append((a, b))
    return finals


def switching_rate(
    w: float,
    resource_strength: float = 0.0,
    initial_fate: str = A_GENE,
    n_trajectories: int = 600,
    n_ticks: int = 200,
    noise_fano: float = 2.0,
    expression_scale: float = 1000.0,
    decay: float = 0.5,
    seed: int | None = 0,
) -> float:
    """Fraction of noisy trajectories that end in the opposite fate.

    With no resource competition a deep bistable toggle is locked into
    its initial fate; as ``resource_strength`` rises the shared pool
    throttles both genes, the effective loop gain drops toward the
    saddle-node and telegraph noise flips the fate more often.
    """
    initial_state = (
        (0.9, 0.1) if initial_fate == A_GENE else (0.1, 0.9)
    )
    finals = simulate_toggle_trajectories(
        w=w,
        n_trajectories=n_trajectories,
        n_ticks=n_ticks,
        noise_fano=noise_fano,
        expression_scale=expression_scale,
        decay=decay,
        resource_strength=resource_strength,
        initial_state=initial_state,
        seed=seed,
    )
    switched = sum(1 for a, b in finals if fate(a, b) != initial_fate)
    return switched / len(finals)


def critical_slowing_down(
    w: float,
    n_ticks: int = 2000,
    noise_fano: float = 2.0,
    expression_scale: float = 1000.0,
    decay: float = 0.5,
    seed: int | None = 1,
) -> float:
    """Lag-1 autocorrelation of ``a(t)`` from one long noisy trajectory.

    On the monostable side of the bifurcation the return rate to the
    fixed point falls to zero as ``w -> w*``, so the autocorrelation
    approaches 1: an early-warning signal of an imminent fate
    decision (Scheffer et al. 2009).
    """
    rng = random.Random(seed)
    a, b = 0.6, 0.4
    series: list[float] = []
    for _ in range(n_ticks):
        a_raw = sigmoid(-w * b)
        b_raw = sigmoid(-w * a)
        eta_a = fano_to_noise_std(noise_fano, a_raw, decay, expression_scale)
        eta_b = fano_to_noise_std(noise_fano, b_raw, decay, expression_scale)
        a = max(0.0, min(1.0, decay * a + (1.0 - decay) * a_raw + rng.gauss(0.0, eta_a)))
        b = max(0.0, min(1.0, decay * b + (1.0 - decay) * b_raw + rng.gauss(0.0, eta_b)))
        series.append(a)
    mu = sum(series) / len(series)
    var = sum((x - mu) ** 2 for x in series) / (len(series) - 1)
    if var == 0.0:
        return 1.0
    lag1 = sum(
        (series[i] - mu) * (series[i + 1] - mu) for i in range(len(series) - 1)
    ) / (len(series) - 1)
    return lag1 / var


def run_fate_analysis(
    w_values: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
    scan_w: float = 7.0,
    switch_resources: tuple[float, ...] = (0.0, 0.5, 1.0),
    seed: int | None = 0,
) -> dict[str, object]:
    """Run the full fate-decision pipeline and summarize it as a dict."""
    scan = bistability_scan(w_values)
    switch_rates = {
        r: switching_rate(scan_w, resource_strength=r, seed=seed)
        for r in switch_resources
    }
    slowing_down = {
        w: critical_slowing_down(w, seed=seed) for w in (3.0, 5.0, 5.3, 5.5)
    }
    return {
        "bifurcation": [
            {
                "w": p.parameter,
                "n_stable_states": p.n_stable_states,
                "stable_states": [(s.a, s.b) for s in p.stable_states],
            }
            for p in scan
        ],
        "switching_rates": switch_rates,
        "critical_slowing_down": slowing_down,
    }
