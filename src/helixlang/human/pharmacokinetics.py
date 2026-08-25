"""Physiology-based pharmacokinetic (PBPK) compartmental model (doc/27).

Couples :class:`~helixlang.human.drug.Drug` regimen specifications with
:class:`~helixlang.human.physiology.HumanPhysiology` perfusion data to
simulate drug disposition over a six-compartment whole-body structure:
one central plasma pool plus liver, kidney, brain, muscle, and adipose
tissue compartments. Compartment geometry defaults to the
literature-anchored volumes exposed through :class:`PBPKConfig`; organ
blood flows come from the physiology specification (Guyton & Hall 2016).

Each tissue behaves as a well-stirred, perfusion-limited compartment
(Rowland & Tozer 2020), with the central plasma pool doubling as the
arterial supply because no separate lung compartment is modeled (classic
venous-equilibrated reduction):

    dCi/dt  = (Qi/Vi)*(Cc - Ci*Rati) - CLi*Ci/Vi
    Cv      = sum(Qi*Ci*Rati)/Qtotal            (mixed venous return)
    dCc/dt  = sum(Qi*Ci*Rati)/Vc - Qtotal*Cc/Vc + Input(t)

where ``Qi`` are organ perfusion rates (L/h), ``Vi`` compartment volumes
(L), ``Rati`` tissue:plasma partition ratios (default 1.0), ``CLi`` organ
clearances allocated between renal (kidney) and hepatic (liver) pathways
according to ``Drug.renal_fraction``, and ``Input(t)`` is the route-specific
dosing contribution to dCc/dt (mg/L/h):

- ``oral`` (and other trans-membrane routes): first-order gut absorption
  beginning at ``t_lag`` and following
  ``D*ka*F*exp(-ka*(t - t_lag))/Vgut``, with ``Vgut`` defaulting to the
  plasma volume, which recovers the classical first-order profile
- ``iv``: instantaneous bolus applied as the initial central concentration
- ``iv_infusion``: zero-order input ``(D/T_inf)/Vc`` while ``t`` lies
  within the infusion window ``[0, T_inf]``

Integration uses :func:`scipy.integrate.solve_ivp` (stiff-capable LSODA)
when SciPy is installed and falls back to stability-safe explicit Euler
sub-stepping otherwise, mirroring the graceful-degradation strategy of
:mod:`helixlang.human.drug`. Summary endpoints in :class:`PBPKResult` —
AUC by the trapezoidal rule, Cmax/tmax from the sampled profile, and
terminal half-life from log-linear regression of the terminal phase —
follow standard non-compartmental analysis practice. Concentrations are
reported in mg/L, volumes in L, flows and clearances in L/h
(``Drug.clearance_ml_per_min`` converts with the factor 0.06).

References:
- Rowland M & Tozer TN. Clinical Pharmacokinetics and Pharmacodynamics,
  Concepts and Applications, 5th ed. 2020.
- Guyton AC, Hall JE. Textbook of Medical Physiology, 14th ed. 2016.
- Nestorov I. Whole-body physiologically based pharmacokinetic models.
  J Pharmacokinet Biopharm 2003;30:479-497.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

try:
    from scipy.integrate import solve_ivp

    _HAS_SCIPY = True
except ImportError:
    solve_ivp = None
    _HAS_SCIPY = False

from helixlang.human.drug import (
    INTRAMUSCULAR,
    IV,
    IV_INFUSION,
    ORAL,
    SUBCUTANEOUS,
    Drug,
)
from helixlang.human.physiology import HumanPhysiology

LN_2 = math.log(2.0)

ML_PER_MIN_TO_L_PER_H = 0.06

ORGAN_NAMES: tuple[str, ...] = ("liver", "kidney", "brain", "muscle", "adipose")

DEFAULT_FLOW_FRACTIONS: dict[str, float] = {
    "liver": 0.25,
    "kidney": 0.20,
    "brain": 0.125,
    "muscle": 0.125,
    "adipose": 0.04,
}

FIRST_ORDER_ROUTES: frozenset[str] = frozenset({ORAL, SUBCUTANEOUS, INTRAMUSCULAR})

SUPPORTED_ROUTES: tuple[str, ...] = (ORAL, IV, IV_INFUSION, SUBCUTANEOUS, INTRAMUSCULAR)

DEFAULT_LAG_TIME_H = 0.0
DEFAULT_INFUSION_DURATION_H = 1.0
MIN_TERMINAL_CONCENTRATION = 1e-12
EULER_SAFETY = 0.4


@dataclass(slots=True)
class PBPKConfig:
    """Numerical and anatomical settings for a :class:`PBPKModel` run.

    ``dt_min`` is the integration/output sampling interval (minutes),
    ``total_time_h`` the simulated horizon (hours); the ``*_volume_l``
    fields give compartment volumes in liters anchored to the reference
    adult of :mod:`helixlang.human.physiology`.
    """

    dt_min: float = 1.0
    total_time_h: float = 24.0
    plasma_volume_l: float = 3.0
    liver_volume_l: float = 1.5
    kidney_volume_l: float = 0.3
    brain_volume_l: float = 1.4
    muscle_volume_l: float = 24.0
    adipose_volume_l: float = 15.0

    def validate(self) -> None:
        """Raise ``ValueError`` when any simulation parameter is invalid."""
        for name in (
            "dt_min", "plasma_volume_l", "liver_volume_l",
            "kidney_volume_l", "brain_volume_l", "muscle_volume_l",
            "adipose_volume_l",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.total_time_h < 0.0:
            raise ValueError("total_time_h must be non-negative")


@dataclass
class PBPKResult:
    """Output of a complete :meth:`PBPKModel.run` simulation.

    ``time_h`` is the sampling grid (hours) aligned with every profile;
    ``concentrations`` maps organ name to its profile (mg/L);
    ``central_concentration`` is the plasma profile; ``auc``, ``c_max``,
    ``t_max``, ``clearance_total_l_per_h``, ``volume_distribution_vd_l``,
    and ``half_life_h`` summarize exposure, peak, and disposition, with
    ``half_life_h`` falling back to ``Drug.half_life_h`` when terminal
    regression is not credible.
    """

    time_h: list[float]
    concentrations: dict[str, list[float]]
    central_concentration: list[float]
    auc: float = 0.0
    c_max: float = 0.0
    t_max: float = 0.0
    clearance_total_l_per_h: float = 0.0
    volume_distribution_vd_l: float = 0.0
    half_life_h: float = 0.0


def _trapezoid(times: list[float], values: list[float]) -> float:
    """Integrate ``values`` over ``times`` with the trapezoidal rule."""
    return sum(
        (v2 + v1) * (t2 - t1) * 0.5
        for t1, t2, v1, v2 in zip(
            times[:-1], times[1:], values[:-1], values[1:], strict=True
        )
    )


def _least_squares_slope(x_values: list[float], y_values: list[float]) -> float:
    """Return the ordinary least-squares slope of ``y_values`` vs ``x_values``."""
    n = len(x_values)
    if n < 2:
        return 0.0
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    s_xx = sum((x - mean_x) ** 2 for x in x_values)
    if s_xx <= 0.0:
        return 0.0
    s_xy = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(x_values, y_values, strict=True)
    )
    return s_xy / s_xx


def _terminal_half_life(times: list[float], concentrations: list[float],
                        fallback_h: float) -> float:
    """Estimate terminal half-life by regressing ln(C) over the terminal phase.

    Uses the second half of the detectable profile points and returns
    ``fallback_h`` when fewer than three usable points exist or the slope
    is not a credible decline.
    """
    usable = [(t, math.log(c)) for t, c in zip(times, concentrations, strict=True)
              if c > MIN_TERMINAL_CONCENTRATION]
    if len(usable) < 4:
        return fallback_h
    tail = usable[len(usable) // 2:]
    if len(tail) < 3:
        tail = usable[-3:]
    slope = _least_squares_slope(
        [point[0] for point in tail],
        [point[1] for point in tail],
    )
    if slope >= -1e-9:
        return fallback_h
    return LN_2 / -slope


class PBPKModel:
    """Six-compartment PBPK simulator driven by a drug and a physiology.

    The state vector is ordered ``[central, liver, kidney, brain, muscle,
    adipose]`` in mg/L. Organ perfusion comes from ``physiology.organs``
    (falling back to :data:`DEFAULT_FLOW_FRACTIONS` of cardiac output for
    missing organs), volumes from :class:`PBPKConfig`, and total clearance
    from ``Drug.clearance_ml_per_min`` split between kidney (renal
    fraction) and liver. ``Qtotal`` equals the summed modeled organ
    effluent so the circuit conserves mass exactly; territories without a
    PK compartment sit outside the modeled loop.
    """

    def __init__(
        self,
        drug: Drug,
        physiology: HumanPhysiology,
        config: PBPKConfig | None = None,
        *,
        gut_volume_l: float | None = None,
        lag_time_h: float = DEFAULT_LAG_TIME_H,
        infusion_duration_h: float = DEFAULT_INFUSION_DURATION_H,
        partition_ratios: dict[str, float] | None = None,
    ) -> None:
        """Prepare the model without integrating anything.

        Args:
            drug: dosing regimen and ADME parameters.
            physiology: donor physiology supplying cardiac output and
                per-organ blood flows.
            config: compartment geometry and solver settings; a fresh
                :class:`PBPKConfig` when omitted.
            gut_volume_l: normalizing volume (L) for first-order absorption
                input; defaults to the plasma volume so the oral profile
                reduces to the classical ``D*ka*F*exp(-ka*t)/Vc`` drive.
            lag_time_h: absorption lag before first-order input begins.
            infusion_duration_h: zero-order infusion window length (hours).
            partition_ratios: tissue:plasma partition ratio overrides keyed
                by organ name (default 1.0).

        Raises:
            ValueError: on an invalid drug specification, route, or
                numerical parameter.
        """
        problems = drug.validate()
        if problems:
            raise ValueError(f"invalid drug specification: {'; '.join(problems)}")
        if drug.route not in SUPPORTED_ROUTES:
            raise ValueError(
                f"route {drug.route!r} has no PBPK input model; "
                f"supported routes: {', '.join(SUPPORTED_ROUTES)}"
            )
        self.config = config if config is not None else PBPKConfig()
        self.config.validate()
        if gut_volume_l is not None and gut_volume_l <= 0.0:
            raise ValueError("gut_volume_l must be positive")
        if lag_time_h < 0.0:
            raise ValueError("lag_time_h must be non-negative")
        if infusion_duration_h <= 0.0:
            raise ValueError("infusion_duration_h must be positive")

        self.drug = drug
        self.physiology = physiology
        default_gut = float(self.config.plasma_volume_l)
        self.gut_volume_l = (
            default_gut if gut_volume_l is None else float(gut_volume_l)
        )
        self.lag_time_h = float(lag_time_h)
        self.infusion_duration_h = float(infusion_duration_h)
        overrides = partition_ratios or {}
        self.partition_ratios: dict[str, float] = {
            name: float(overrides.get(name, 1.0)) for name in ORGAN_NAMES
        }

        co_l_per_h = physiology.cardiac_output_ml_per_min * ML_PER_MIN_TO_L_PER_H
        if co_l_per_h <= 0.0:
            raise ValueError("cardiac output must translate to positive flow")
        self.cardiac_output_l_per_h = co_l_per_h
        self.organ_flows_l_per_h = self._resolve_flows()
        self.q_total_l_per_h = sum(self.organ_flows_l_per_h.values())
        self.organ_volumes_l = {
            name: getattr(self.config, f"{name}_volume_l") for name in ORGAN_NAMES
        }
        self.volume_distribution_vd_l = drug.volume_distribution_l
        self.cl_total_l_per_h = drug.clearance_ml_per_min * ML_PER_MIN_TO_L_PER_H
        self.organ_clearances_l_per_h = self._allocate_clearances()

        self._time_h = 0.0
        self._state: list[float] = self._initial_state()
        self._rhs = self._build_odes()

    def _resolve_flows(self) -> dict[str, float]:
        """Resolve per-organ perfusion rates (L/h) from the physiology."""
        flows: dict[str, float] = {}
        for name in ORGAN_NAMES:
            organ = self.physiology.organs.get(name)
            if organ is not None and organ.blood_flow_ml_per_min > 0.0:
                flows[name] = (
                    organ.blood_flow_ml_per_min * ML_PER_MIN_TO_L_PER_H
                )
            else:
                flows[name] = (
                    self.cardiac_output_l_per_h * DEFAULT_FLOW_FRACTIONS[name]
                )
        return flows

    def _allocate_clearances(self) -> dict[str, float]:
        """Split total clearance between eliminating organ compartments."""
        renal_share = min(max(self.drug.renal_fraction, 0.0), 1.0)
        clearances = {name: 0.0 for name in ORGAN_NAMES}
        clearances["kidney"] = self.cl_total_l_per_h * renal_share
        clearances["liver"] = self.cl_total_l_per_h * (1.0 - renal_share)
        return clearances

    def _initial_state(self) -> list[float]:
        """Build the starting concentration vector for the chosen route."""
        state = [0.0] * (len(ORGAN_NAMES) + 1)
        if self.drug.route == IV:
            state[0] = (
                self.drug.dose_mg * self.drug.bioavailability
                / self.config.plasma_volume_l
            )
        return state

    def _build_odes(self) -> Callable[[float, list[float]], list[float]]:
        """Build and return the right-hand-side function for the ODE system.

        The callable maps ``(t_h, state)`` to ``d[state]/dt_h`` per the
        module mass balance and is directly compatible with
        :func:`scipy.integrate.solve_ivp`.
        """
        flows = [self.organ_flows_l_per_h[name] for name in ORGAN_NAMES]
        volumes = [self.organ_volumes_l[name] for name in ORGAN_NAMES]
        ratios = [self.partition_ratios[name] for name in ORGAN_NAMES]
        clearances = [self.organ_clearances_l_per_h[name] for name in ORGAN_NAMES]
        vc = self.config.plasma_volume_l
        q_total = self.q_total_l_per_h
        n_organs = len(ORGAN_NAMES)

        def rhs(t_h: float, state: list[float]) -> list[float]:
            cc = state[0]
            recirculation = sum(
                flows[i] * state[i + 1] * ratios[i] for i in range(n_organs)
            )
            derivatives = [0.0] * (n_organs + 1)
            for i in range(n_organs):
                derivatives[i + 1] = (
                    (flows[i] / volumes[i])
                    * (cc - state[i + 1] * ratios[i])
                    - clearances[i] * state[i + 1] / volumes[i]
                )
            derivatives[0] = (
                recirculation / vc
                - q_total * cc / vc
                + self._dose_input(t_h)
            )
            return derivatives

        return rhs

    def _dose_input(self, t_h: float) -> float:
        """Return the route-specific additive contribution to dCc/dt (mg/L/h).

        First-order routes yield zero before ``lag_time_h`` and then
        ``D*ka*F*exp(-ka*(t - t_lag))/Vgut``; ``iv_infusion`` yields
        ``(D/T_inf)/Vc`` inside the infusion window; ``iv`` bolus
        contributes nothing because it is applied through the initial
        condition.
        """
        route = self.drug.route
        dose_mg = self.drug.dose_mg
        if route in FIRST_ORDER_ROUTES:
            if t_h < self.lag_time_h:
                return 0.0
            elapsed = t_h - self.lag_time_h
            absorbed_flux = (
                dose_mg
                * self.drug.bioavailability
                * self.drug.absorption_rate_h
                * math.exp(-self.drug.absorption_rate_h * elapsed)
            )
            return absorbed_flux / self.gut_volume_l
        if route == IV_INFUSION:
            if 0.0 <= t_h <= self.infusion_duration_h:
                infusion_rate = (
                    dose_mg * self.drug.bioavailability / self.infusion_duration_h
                )
                return infusion_rate / self.config.plasma_volume_l
            return 0.0
        return 0.0

    def get_concentrations(self) -> dict[str, float]:
        """Return the current concentration snapshot (mg/L) per compartment."""
        snapshot = {"central": self._state[0]}
        for i, name in enumerate(ORGAN_NAMES):
            snapshot[name] = self._state[i + 1]
        return snapshot

    def _max_rate_constant_per_h(self) -> float:
        """Largest first-order rate constant of the compartment network."""
        candidates = [self.q_total_l_per_h / self.config.plasma_volume_l]
        for name in ORGAN_NAMES:
            volume = self.organ_volumes_l[name]
            rate = self.organ_flows_l_per_h[name] / volume
            rate += self.organ_clearances_l_per_h[name] / volume
            candidates.append(rate)
        return max(candidates)

    def _euler_advance(
        self, state: list[float], t_h: float, span_h: float
    ) -> tuple[list[float], float]:
        """Advance ``state`` by ``span_h`` with stability-safe Euler sub-steps.

        The span is subdivided so the fastest compartment exchange stays
        inside the explicit Euler stability region.
        """
        max_substep = EULER_SAFETY / self._max_rate_constant_per_h()
        n_substeps = max(1, math.ceil(span_h / max_substep))
        substep_h = span_h / n_substeps
        state = list(state)
        for _ in range(n_substeps):
            derivatives = self._rhs(t_h, state)
            state = [
                y + substep_h * dy
                for y, dy in zip(state, derivatives, strict=True)
            ]
            t_h += substep_h
        return state, t_h

    def step(self, dt_min: float) -> dict[str, float]:
        """Advance the model by ``dt_min`` minutes with explicit Euler steps.

        The span is subdivided into sub-steps small enough to keep the
        fastest compartment exchange inside the Euler stability region.
        Returns the resulting concentration snapshot; repeated calls
        continue from the current state, independently of :meth:`run`.

        Raises:
            ValueError: when ``dt_min`` is not strictly positive.
        """
        if dt_min <= 0.0:
            raise ValueError("dt_min must be positive")
        self._state, self._time_h = self._euler_advance(
            self._state, self._time_h, dt_min / 60.0
        )
        return self.get_concentrations()

    def _sample_grid(self) -> list[float]:
        """Build the output sampling grid covering the configured horizon."""
        dt_h = self.config.dt_min / 60.0
        if self.config.total_time_h <= 0.0 or dt_h <= 0.0:
            return [0.0]
        n_steps = max(1, int(round(self.config.total_time_h / dt_h)))
        grid = [i * dt_h for i in range(n_steps + 1)]
        grid[-1] = self.config.total_time_h
        return grid

    def _euler_trajectory(self, grid: list[float]) -> list[list[float]]:
        """Integrate with stability-safe explicit Euler steps along ``grid``."""
        state = list(self._state)
        rows: list[list[float]] = [list(state)]
        for i in range(1, len(grid)):
            state, _ = self._euler_advance(
                state, grid[i - 1], grid[i] - grid[i - 1]
            )
            rows.append(list(state))
        return rows

    def _solver_max_step_h(self) -> float:
        """Bound internal solver steps around dosing discontinuities.

        Adaptive integrators accelerate through quiescent intervals and
        can jump clean over a lag onset or infusion off-switch, so the
        maximum step is limited to a fraction of the interval preceding
        each input switching time.
        """
        horizon = self.config.total_time_h
        switches: list[float] = []
        if self.drug.route == IV_INFUSION:
            switches.append(self.infusion_duration_h)
        if self.drug.route in FIRST_ORDER_ROUTES and self.lag_time_h > 0.0:
            switches.append(self.lag_time_h)
        interior = [s for s in switches if 0.0 < s < horizon]
        if not interior:
            return horizon
        return min(interior) / 8.0

    def _solve_trajectory(self, grid: list[float]) -> list[list[float]]:
        """Integrate the ODE system over ``grid``, preferring SciPy LSODA."""
        y0 = list(self._state)
        if _HAS_SCIPY and len(grid) > 1:
            solution = solve_ivp(
                self._rhs,
                (grid[0], grid[-1]),
                y0,
                t_eval=grid,
                method="LSODA",
                rtol=1e-7,
                atol=1e-9,
                max_step=self._solver_max_step_h(),
            )
            if solution.success and solution.y.shape[1] == len(grid):
                return [[float(v) for v in column] for column in solution.y.T]
        return self._euler_trajectory(grid)

    def run(self) -> PBPKResult:
        """Integrate the full configured horizon and summarize PK endpoints.

        Resets the state to the route-specific initial condition,
        integrates (SciPy LSODA when available, stability-safe Euler
        otherwise) over the uniform :attr:`PBPKConfig.dt_min` grid, and
        leaves the model positioned at the final time point.
        """
        self._time_h = 0.0
        self._state = self._initial_state()
        grid = self._sample_grid()
        rows = self._solve_trajectory(grid)

        times = list(grid)
        central = [row[0] for row in rows]
        concentrations = {
            name: [row[i + 1] for row in rows]
            for i, name in enumerate(ORGAN_NAMES)
        }

        self._time_h = times[-1]
        self._state = list(rows[-1])

        c_max = max(central)
        t_max = times[central.index(c_max)]
        half_life = _terminal_half_life(times, central, self.drug.half_life_h)

        return PBPKResult(
            time_h=times,
            concentrations=concentrations,
            central_concentration=central,
            auc=_trapezoid(times, central),
            c_max=c_max,
            t_max=t_max,
            clearance_total_l_per_h=self.cl_total_l_per_h,
            volume_distribution_vd_l=self.volume_distribution_vd_l,
            half_life_h=half_life,
        )


__all__ = [
    "DEFAULT_FLOW_FRACTIONS",
    "FIRST_ORDER_ROUTES",
    "ML_PER_MIN_TO_L_PER_H",
    "ORGAN_NAMES",
    "PBPKConfig",
    "PBPKModel",
    "PBPKResult",
    "SUPPORTED_ROUTES",
]
