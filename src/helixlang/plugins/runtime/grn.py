"""Gene regulatory network (GRN): sigmoid / Hill concentration
threshold model.

Each tick updates each gene's expression level; genes with level > 0.5
are triggered to execute their ORF.

Models:
- Sigmoid threshold model (default, unchanged)::
      activation = sigmoid(sum(w_i * level_i) - threshold)
      level' = decay * level + (1 - decay) * activation
- Optional per-gene Hill kinetics (activator-saturating)::
      activation = inputs^n / (kd^n + inputs^n),  inputs >= 0
  with kd the half-maximal effector concentration and n the Hill
  coefficient. Real measured Kd values (e.g. lacI repressor Kd ~ 0.1 nM,
  Oehler 1990 EMBO J) can be supplied per gene via ``kd=``.
- Optional per-gene telegraph (two-state promoter) intrinsic noise
  (:mod:`helixlang.plugins.runtime.stochastic`): when the GRN is created with
  ``noise_enabled=True`` and a gene carries a ``noise=``
  :class:`~helixlang.plugins.runtime.stochastic.TelegraphPromoter`, each tick adds
  zero-mean Gaussian noise whose steady-state variance reproduces the
  exact two-state Fano factor (Peccoud & Ycart 1995; Jones et al.
  2014).  The deterministic default (noise disabled) is unchanged.

Genes without an explicit ``decay=`` default to
``decay_from_half_life_ticks(110)`` (~0.994 per tick), the E. coli
median protein half-life of ~110 min with one tick per minute
(Mosteller 1980 J Biol Chem, Helbig 2011 Proteomics 11).
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from helixlang.api.units import (
    PROTEIN_HALF_LIFE_MEDIAN_TICKS,
)
from helixlang.api.units import (
    decay_from_half_life_ticks as units_decay_from_half_life_ticks,
)
from helixlang.plugins.runtime.stochastic import (
    TelegraphPromoter,
    fano_to_noise_std,
)


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def hill(x: float, n: float, kd: float) -> float:
    """Hill activation function.

    ``x^n / (kd^n + x^n)``, half-max at ``x = kd``; returns 0 for
    non-positive input (pure activator, no ligand-binding side effect).

    Args:
        x:  effector concentration (regulatory input)
        n:  Hill coefficient (cooperativity)
        kd: dissociation constant at half-maximum activation
    """
    if x <= 0:
        return 0.0
    xn: float = x ** n
    kdn: float = kd ** n
    if kdn <= 0:  # kd == 0 -> unit step at any positive input
        return 1.0
    return xn / (kdn + xn)


def decay_from_half_life_ticks(half_life_ticks: float) -> float:
    """Per-tick decay coefficient from a protein half-life.

    Re-exported from :mod:`helixlang.core.units` (the physical unit system);
    see :func:`helixlang.core.units.decay_from_half_life_ticks` for the full
    documentation.

    E. coli protein half-lives are ~60-600 min (median ~110 min;
    Mosteller 1980, Helbig 2011), so with one tick per minute a typical
    decay is ~0.994.
    """
    return units_decay_from_half_life_ticks(half_life_ticks)


def rate_constant_from_decay(decay: float) -> float:
    """First-order degradation rate constant (1/min) for a per-tick decay.

    ``k = -ln(decay)``.  The discrete recurrence
    ``level' = decay*level + (1-decay)*activation`` is an explicit Euler
    step (dt = 1 min) of the continuous ODE ``dL/dt = k*(activation - L)``
    to first order in ``(1 - decay)`` (equal to first order in k), and the
    two share the exact same fixed points ``L* = activation``.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be in (0, 1)")
    return -math.log(decay)


def _activation_raw(node: GeneNode, inputs: float) -> float:
    """Raw activation term of a gene given its summed regulatory input.

    Shared by the discrete recurrence (:meth:`GRN.step`) and the
    continuous-time ODE (:func:`grn_derivatives`) so both models use
    byte-for-byte the same activation function (sigmoid threshold or
    Hill kinetics).
    """
    if node.hill_n is not None:
        kd = node.kd if node.kd is not None else node.threshold
        return hill(inputs, node.hill_n, kd)
    return sigmoid(inputs - node.threshold)


def grn_derivatives(grn: GRN,
                    levels: list[float] | None = None) -> tuple[list[str], Callable[[float, list[float]], list[float]], list[float]]:
    """Continuous-time ODE form of a GRN (Alon 2007; GRN_modeler 2025).

    Builds the first-order mass-action model

        dL_i/dt = k_i * (activation_i(inputs_i) - L_i)

    where ``k_i = -ln(decay_i)`` (1/min), so the ODE fixed points are
    exactly the fixed points of the discrete recurrence (:func:`GRN.step`
    is an Euler step of this ODE with dt = 1 min).  Gene order is
    deterministic (insertion order).

    Args:
        grn: the GRN to convert.
        levels: optional numeric initial state (length must equal the
            number of genes); defaults to the current ``GeneNode.level``
            values.

    Returns:
        ``(names, rhs, y0)`` where ``rhs(t, y) -> list[float]`` is the
        derivative and ``y0`` the initial state.  The returned ``names``
        list indexes the state vector.
    """
    names = list(grn.nodes.keys())
    idx = {name: i for i, name in enumerate(names)}
    edges = grn.edges
    nodes = grn.nodes
    default_decay = grn._default_decay
    if levels is None:
        y0 = [nodes[n].level for n in names]
    else:
        if len(levels) != len(names):
            raise ValueError("levels must match the number of genes")
        y0 = list(levels)

    def rhs(_t: float, y: list[float]) -> list[float]:
        inputs: list[float] = [0.0] * len(names)
        for e in edges:
            inputs[idx[e.target]] += e.weight * y[idx[e.source]]
        out: list[float] = []
        for i, name in enumerate(names):
            node = nodes[name]
            act = _activation_raw(node, inputs[i])
            decay = node.decay if node.decay is not None else default_decay
            out.append(rate_constant_from_decay(decay) * (act - y[i]))
        return out

    return names, rhs, y0


@dataclass(slots=True)
class GeneNode:
    name: str
    threshold: float
    level: float = 0.0
    decay: float | None = None
    hill_n: float | None = None
    kd: float | None = None
    noise: TelegraphPromoter | None = None


@dataclass(slots=True)
class Edge:
    source: str
    target: str
    weight: float


class GRN:
    """Hybrid GRN model with discrete ticks + sigmoid/Hill thresholds.

    ``DECAY`` is the default per-tick decay coefficient used when a gene
    has no per-gene ``decay=``: derived from the E. coli median protein
    half-life of ~110 min (Mosteller 1980, Helbig 2011), i.e. ~0.994 per
    tick.  Per-gene decay should be derived from measured half-lives via
    :func:`decay_from_half_life_ticks`.
    """

    DECAY = units_decay_from_half_life_ticks(PROTEIN_HALF_LIFE_MEDIAN_TICKS)

    def __init__(self, noise_enabled: bool = False,
                 noise_seed: int | None = None) -> None:
        self.nodes: dict[str, GeneNode] = {}
        self.edges: list[Edge] = []
        # Per-target incoming-edge index (target -> edges), so ``step()`` is
        # O(N + E) instead of scanning all edges per node (O(N·E)).
        self._incoming: dict[str, list[Edge]] = {}
        self._edge_count = 0
        # Genes without an explicit ``decay=`` default to the E. coli
        # median protein half-life (Mosteller 1980, Helbig 2011).
        self._default_decay = self.DECAY
        # Telegraph-model intrinsic noise (T1.4): disabled by default so
        # the deterministic recurrence (and every existing test) is
        # unchanged; when enabled, per-gene ``noise=`` params inject
        # zero-mean noise matching the two-state Fano factor.
        self.noise_enabled = noise_enabled
        self._noise_rng = random.Random(noise_seed)

    def add_gene(self, name: str, threshold: float,
                 initial_level: float = 0.0,
                 decay: float | None = None,
                 hill_n: float | None = None,
                 kd: float | None = None,
                 noise: TelegraphPromoter | None = None) -> None:
        """Add a gene node.

        Args:
            name: gene name
            threshold: activation threshold for the sigmoid path
                (ignored when ``hill_n`` is set and ``kd`` is given)
            initial_level: starting expression level in [0, 1]
            decay: per-gene decay coefficient (default: :attr:`DECAY`);
                derive from a measured half-life with
                :func:`decay_from_half_life_ticks`
            hill_n: optional Hill coefficient; when set, activation uses
                Hill kinetics instead of the sigmoid
            kd: dissociation constant for Hill activation (default:
                ``threshold`` when omitted)
            noise: optional :class:`TelegraphPromoter` two-state
                promoter noise; only applied when the GRN has
                ``noise_enabled=True``
        """
        self.nodes[name] = GeneNode(name, threshold, initial_level,
                                    decay, hill_n, kd, noise)
        self._incoming.setdefault(name, [])

    def add_edge(self, source: str, target: str, weight: float) -> None:
        self.edges.append(Edge(source, target, weight))
        self._incoming.setdefault(target, []).append(self.edges[-1])
        self._edge_count += 1

    def _rebuild_incoming(self) -> None:
        """Rebuild the incoming-edge index after direct ``edges`` mutation."""
        self._incoming = {}
        for e in self.edges:
            self._incoming.setdefault(e.target, []).append(e)
        self._edge_count = len(self.edges)

    def set_level(self, name: str, level: float) -> None:
        if name in self.nodes:
            self.nodes[name].level = max(0.0, min(1.0, level))

    def step(self) -> list[str]:
        """Advance one tick, returning the names of the genes triggered this tick (level > 0.5)."""
        # In-place weight updates (OP_REGULATE) reuse the same Edge objects, so
        # the index stays valid; guard against external direct ``edges`` appends.
        if self._edge_count != len(self.edges):
            self._rebuild_incoming()
        incoming = self._incoming
        nodes = self.nodes
        noise_enabled = self.noise_enabled
        noise_rng = self._noise_rng
        new_levels: dict[str, float] = {}
        for name, node in nodes.items():
            inputs: float = 0
            for e in incoming.get(name, ()):
                inputs += e.weight * nodes[e.source].level
            raw = _activation_raw(node, inputs)
            decay = node.decay if node.decay is not None else self._default_decay
            # Decay + new input (not doubled, to avoid self-excitation of genes without inputs)
            blended = decay * node.level + (1 - decay) * raw
            if noise_enabled and node.noise is not None:
                # Telegraph (two-state promoter) intrinsic noise: add
                # zero-mean Gaussian noise so the stationary variance
                # equals Fano * mean / expression_scale (Peccoud & Ycart
                # 1995; Jones 2014), keeping the deterministic mean
                # trajectory unchanged.
                fano = node.noise.fano_factor()
                std = fano_to_noise_std(
                    fano, max(0.0, blended), decay,
                    node.noise.expression_scale)
                blended += noise_rng.gauss(0.0, std)
            new_levels[name] = max(0.0, min(1.0, blended))
        for name, lvl in new_levels.items():
            self.nodes[name].level = lvl
        return [n for n, l in new_levels.items() if l > 0.5]

    def step_accel(self, prefer: str | None = None) -> list[str]:
        """Advance one tick via the isolated hot-loop kernel (doc/36 §4.1 P1).

        Produces results identical to :meth:`step` for the noiseless sigmoid
        threshold path (no ``hill_n``, no telegraph noise) — a pure speed
        switch, never a fidelity switch (§3ξ.5).  The kernel is selected by the
        ``_accel`` loader (native > numpy > python) with the optional ``prefer``
        override.  Returns the triggered gene names, matching :meth:`step`.

        Raises:
            ValueError: if the graph uses Hill kinetics or telegraph noise, which
                the equivalent-fidelity kernel does not mirror.
        """
        from helixlang.api.accel import grn_step as accel_step

        if self.noise_enabled:
            raise ValueError(
                "step_accel requires noise disabled (telegraph noise is not "
                "part of the equivalent-fidelity kernel; use step())")
        names = list(self.nodes)
        index = {n: i for i, n in enumerate(names)}
        for name in names:
            node = self.nodes[name]
            if node.hill_n is not None:
                raise ValueError(
                    f"step_accel does not mirror Hill kinetics on node {name!r}; "
                    "use step()")

        levels = [self.nodes[n].level for n in names]
        src = [index[e.source] for e in self.edges]
        dst = [index[e.target] for e in self.edges]
        weights = [e.weight for e in self.edges]
        decays = [self.nodes[n].decay for n in names]
        thresholds = [self.nodes[n].threshold for n in names]
        default_decay = self._default_decay

        new_levels, _trig = accel_step(
            levels, src, dst, weights, decays, thresholds, default_decay,
        )
        for name, lvl in zip(names, new_levels, strict=True):
            self.nodes[name].level = lvl
        return [n for n, l in zip(names, new_levels, strict=True) if l > 0.5]


# ============================================================================
# Continuous-time solvers (T2.2, gap G6)
#
# The discrete recurrence is an Euler step (dt = 1 min) of
# ``dL/dt = k*(activation - L)`` (see :func:`grn_derivatives`).  These
# integrators solve that ODE at continuous time so the same circuit can
# be compared against COPASI/Tellurium-class benchmarks (GRN_modeler
# 2025).  Pure-Python RK4 and adaptive Dormand-Prince RK45 are always
# available; scipy's ``solve_ivp`` is used when ``method="scipy"`` and
# scipy is installed (optional extra).
# ============================================================================

#: Dormand-Prince (1980) / Fehlberg-style embedded RK5(4) tableau.
_DOPRI5_C = (0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0)
_DOPRI5_A = (
    (),
    (1/5,),
    (3/40, 9/40),
    (44/45, -56/15, 32/9),
    (19372/6561, -25360/2187, 64448/6561, -212/729),
    (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656),
    (35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84),
)
#: 5th-order solution weights.
_DOPRI5_B5 = (35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0)
#: embedded 4th-order error estimator weights.
_DOPRI5_B4 = (5179/57600, 0.0, 7571/16695, 393/640, -92097/339200,
              187/2100, 1/40)


def integrate_ode(rhs: Callable[[float, list[float]], list[float]], y0: list[float], t_span: tuple[float, float],
                  n_points: int = 1000, method: str = "rk45",
                  dt0: float = 0.01, atol: float = 1e-8, rtol: float = 1e-6,
                  max_steps: int = 1_000_000) -> tuple[list[float], list[list[float]]]:
    """Integrate ``dy/dt = rhs(t, y)`` over ``t_span``.

    Args:
        rhs: ``f(t, y) -> list[float]``.
        y0: initial state vector.
        t_span: ``(t_start, t_end)`` in minutes.
        n_points: number of uniformly sampled output points.
        method: ``"rk4"`` (fixed-step), ``"rk45"`` (pure-Python adaptive
            Dormand-Prince) or ``"scipy"`` (``scipy.integrate.solve_ivp``
            RK45 when scipy is installed).
        dt0: initial step (rk45 only).
        atol, rtol: absolute/relative tolerances (rk45).
        max_steps: safety cap on integrator steps (rk45).

    Returns:
        ``(times, y_history)`` sampled on ``n_points`` uniform times;
        ``times`` spans ``t_span``.
    """
    if t_span[1] <= t_span[0]:
        raise ValueError("t_span must be increasing")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    t0, t_end = t_span
    if method == "rk4":
        dt = (t_end - t0) / (n_points - 1)
        times = [t0 + dt * i for i in range(n_points)]
        ys: list[list[float]] = [list(y0)]
        y = list(y0)
        for i in range(1, n_points):
            y = _rk4_step(rhs, times[i - 1], y, dt)
            ys.append(y)
        return times, ys
    if method == "rk45":
        times, ys, fs = _dopri5(rhs, t0, list(y0), t_end, dt0, atol, rtol,
                                max_steps)
        return _resample(times, ys, fs, n_points)
    if method == "scipy":
        try:
            from scipy.integrate import solve_ivp
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "method='scipy' requires scipy; use 'rk4' or 'rk45'") from exc
        sol = solve_ivp(rhs, t_span, list(y0), method="RK45",
                        t_eval=[t0 + (t_end - t0) * i / (n_points - 1)
                                for i in range(n_points)],
                        rtol=rtol, atol=atol)
        return list(sol.t), [list(row) for row in sol.y.T]
    raise ValueError("method must be 'rk4', 'rk45' or 'scipy'")


def _rk4_step(rhs: Callable[[float, list[float]], list[float]], t: float, y: list[float],
              dt: float) -> list[float]:
    n = len(y)
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * dt, [y[i] + 0.5 * dt * k1[i] for i in range(n)])
    k3 = rhs(t + 0.5 * dt, [y[i] + 0.5 * dt * k2[i] for i in range(n)])
    k4 = rhs(t + dt, [y[i] + dt * k3[i] for i in range(n)])
    return [y[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
            for i in range(n)]


def _dopri5(rhs: Callable[[float, list[float]], list[float]], t0: float, y0: list[float], t_end: float,
            dt0: float, atol: float, rtol: float,
            max_steps: int) -> tuple[list[float], list[list[float]], list[list[float]]]:
    """Adaptive Dormand-Prince RK5(4) integrator with step control.

    Returns ``(times, ys, fs)`` with the accepted points and their
    derivatives ``f(t, y)`` (used for cubic-Hermite dense output).
    """
    t = t0
    y = list(y0)
    dt = max(dt0, 1e-12)
    n = len(y)
    times = [t0]
    ys: list[list[float]] = [list(y0)]
    fs: list[list[float]] = [rhs(t0, y0)]
    steps = 0
    while t < t_end:
        if steps >= max_steps:
            break
        if t + dt > t_end:
            dt = t_end - t
        k1 = rhs(t, y)
        k2 = rhs(t + _DOPRI5_C[1] * dt,
                 [y[i] + dt * _DOPRI5_A[1][0] * k1[i] for i in range(n)])
        k3 = rhs(t + _DOPRI5_C[2] * dt,
                 [y[i] + dt * (_DOPRI5_A[2][0] * k1[i]
                               + _DOPRI5_A[2][1] * k2[i]) for i in range(n)])
        k4 = rhs(t + _DOPRI5_C[3] * dt,
                 [y[i] + dt * (_DOPRI5_A[3][0] * k1[i]
                               + _DOPRI5_A[3][1] * k2[i]
                               + _DOPRI5_A[3][2] * k3[i]) for i in range(n)])
        k5 = rhs(t + _DOPRI5_C[4] * dt,
                 [y[i] + dt * (_DOPRI5_A[4][0] * k1[i]
                               + _DOPRI5_A[4][1] * k2[i]
                               + _DOPRI5_A[4][2] * k3[i]
                               + _DOPRI5_A[4][3] * k4[i]) for i in range(n)])
        k6 = rhs(t + _DOPRI5_C[5] * dt,
                 [y[i] + dt * (_DOPRI5_A[5][0] * k1[i]
                               + _DOPRI5_A[5][1] * k2[i]
                               + _DOPRI5_A[5][2] * k3[i]
                               + _DOPRI5_A[5][3] * k4[i]
                               + _DOPRI5_A[5][4] * k5[i]) for i in range(n)])
        k7 = rhs(t + _DOPRI5_C[6] * dt,
                 [y[i] + dt * (_DOPRI5_A[6][0] * k1[i]
                               + _DOPRI5_A[6][1] * k2[i]
                               + _DOPRI5_A[6][2] * k3[i]
                               + _DOPRI5_A[6][3] * k4[i]
                               + _DOPRI5_A[6][4] * k5[i]
                               + _DOPRI5_A[6][5] * k6[i]) for i in range(n)])
        ks = (k1, k2, k3, k4, k5, k6, k7)
        y5 = [y[i] + dt * sum(_DOPRI5_B5[j] * ks[j][i] for j in range(7))
              for i in range(n)]
        y4 = [y[i] + dt * sum(_DOPRI5_B4[j] * ks[j][i] for j in range(7))
              for i in range(n)]
        err = 0.0
        for i in range(n):
            denom = atol + rtol * abs(y5[i])
            e = abs(y5[i] - y4[i]) / denom if denom > 0 else abs(y5[i] - y4[i])
            if e > err:
                err = e
        if err <= 1.0:
            t += dt
            y = y5
            times.append(t)
            ys.append(list(y))
            fs.append(rhs(t, y))
            steps += 1
            if err == 0.0:
                dt *= 5.0
            else:
                dt *= min(5.0, 0.9 * err ** -0.2)
        else:
            dt *= max(0.1, 0.9 * err ** -0.2)
            if dt < 1e-14:
                dt = 1e-14
    return times, ys, fs


def _resample(times: list[float], ys: list[list[float]],
              fs: list[list[float]],
              n_points: int) -> tuple[list[float], list[list[float]]]:
    """Cubic-Hermite dense output on uniformly spaced times.

    Uses the ODE derivative at each accepted point (the DOPRI5 dense
    output is cubic Hermite, error O(dt^4)), which keeps the sampled
    trajectory accurate even across large adaptive steps.
    """
    if n_points <= 1:
        return times, ys
    n = len(ys[0])
    t_start, t_end = times[0], times[-1]
    out_t = [t_start + (t_end - t_start) * i / (n_points - 1)
             for i in range(n_points)]
    out_y: list[list[float]] = []
    j = 0
    for tt in out_t:
        while j < len(times) - 2 and times[j + 1] < tt:
            j += 1
        t0, t1 = times[j], times[j + 1]
        dt = t1 - t0
        u = 0.0 if dt == 0 else (tt - t0) / dt
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1
        h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2
        h11 = u ** 3 - u ** 2
        y0, y1 = ys[j], ys[j + 1]
        f0, f1 = fs[j], fs[j + 1]
        out_y.append([h00 * y0[i] + h10 * dt * f0[i]
                      + h01 * y1[i] + h11 * dt * f1[i] for i in range(n)])
    return out_t, out_y


class ContinuousGRNResult:
    """Output of :func:`integrate_grn` on a GRN.

    Holds the gene names, integration times (minutes) and the level
    matrix (``levels[k][i]`` = level of gene ``i`` at time ``times[k]``).
    """

    def __init__(self, names: list[str], times: list[float],
                 levels: list[list[float]]) -> None:
        self.names = list(names)
        self.times = list(times)
        self.levels = [list(row) for row in levels]

    def final(self) -> dict[str, float]:
        """Final level of every gene (last sampled time)."""
        return {name: self.levels[-1][i] for i, name in enumerate(self.names)}

    def at(self, t_min: float) -> dict[str, float]:
        """Levels at the nearest sampled time (linear interpolation)."""
        n = len(self.times)
        if n == 1:
            row = self.levels[0]
        else:
            i = min(n - 1, max(0, int((t_min - self.times[0])
                                       / (self.times[-1] - self.times[0])
                                       * (n - 1))))
            row = self.levels[i]
        return {name: row[k] for k, name in enumerate(self.names)}

    def triggered(self, threshold: float = 0.5) -> list[str]:
        """Genes with final level above ``threshold`` (active)."""
        final = self.final()
        return [name for name in self.names if final[name] > threshold]

    def trajectory(self, name: str) -> list[float]:
        """Level time course of one gene."""
        i = self.names.index(name)
        return [row[i] for row in self.levels]


def integrate_grn(grn: GRN, t_span: tuple[float, float],
                  n_points: int = 1000, method: str = "rk45",
                  atol: float = 1e-8, rtol: float = 1e-6,
                  levels: list[float] | None = None) -> ContinuousGRNResult:
    """Integrate a GRN's continuous-time ODE (T2.2, gap G6).

    Solves ``dL/dt = k*(activation - L)`` from :func:`grn_derivatives`
    over ``t_span`` minutes and returns a :class:`ContinuousGRNResult`.
    Use ``method="rk45"`` (default, pure-Python adaptive
    Dormand-Prince), ``"rk4"`` (fixed step) or ``"scipy"`` (scipy
    ``solve_ivp``, optional extra) to cross-check against COPASI/
    Tellurium-class solvers (GRN_modeler 2025).

    Example — a toggle switch from two initial conditions settles on two
    distinct stable fixed points (Gardner 2000; see tests).

    Args:
        grn: the GRN to simulate.
        t_span: integration window ``(t_start, t_end)`` in minutes.
        n_points: number of uniformly sampled output points.
        method: ``"rk4"``, ``"rk45"`` (default) or ``"scipy"``.
        atol, rtol: integrator tolerances (rk45/scipy).
        levels: optional initial state; defaults to current levels.

    Returns:
        a :class:`ContinuousGRNResult` with the level time courses.
    """
    names, rhs, y0 = grn_derivatives(grn, levels)
    times, ys = integrate_ode(rhs, y0, t_span, n_points=n_points,
                              method=method, atol=atol, rtol=rtol)
    return ContinuousGRNResult(names, times, ys)
