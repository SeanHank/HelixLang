"""Lattice-Boltzmann D2Q9 BGK flow solver (doc/18-programmable-cell-population-simulation.md §13 Design 6, Level 2).

A BGK (Bhatnagar-Gross-Krook) lattice Boltzmann solver on the D2Q9
velocity stencil.  It provides the *solver* flow field for the
population:

- occupied lattice sites (the cells) act as no-slip obstacles through
  full-node bounce-back, so the flow is displaced around the colony
  (iDynoMiCS-style flow-field + bacterial colonies; Cockx et al. 2024),
- the macroscopic velocity is fed back to the environment as a
  :class:`~helixlang.flow.FlowField` driving substrate advection and
  cell drift,
- the momentum exchange across each bounce-back link gives the force
  the flow exerts on the obstacle cells.

Physics notes:

- The D2Q9 stencil with weights ``w = (4/9, 1/9 x4, 1/36 x4)`` and
  lattice sound speed ``cs = 1/sqrt(3)`` recovers the incompressible
  Navier-Stokes equations in the low-Mach limit (Frisch 1986; Succi
  *The Lattice Boltzmann Equation*, Oxford 2001; Krüger et al. *The
  Lattice Boltzmann Method*, Springer 2017).
- BGK relaxation ``f' = f - omega (f - feq)`` with the kinematic
  viscosity ``nu = (1/omega - 1/2)/3``; ``omega -> 2`` is the
  no-viscosity limit.
- Full-node bounce-back implements no-slip walls/obstacles; the
  momentum-exchange method (Ladd 1994 J Fluid Mech 271:285-309)
  computes the force on each obstacle node.

All quantities are dimensionless lattice units; ``u = 1`` is one site
per LBM step.  The solver is **numpy-only** (a single step is a few
array operations; the pure-Python path would be too slow for the
10^4-cell performance gate).
"""
from __future__ import annotations

import numpy as np

from helixlang.flow import FlowField

#: D2Q9 velocity directions (0 = rest, 1-4 = axes, 5-8 = diagonals)
VELOCITIES: tuple[tuple[int, int], ...] = (
    (0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
    (1, 1), (-1, 1), (-1, -1), (1, -1),
)
#: D2Q9 quadrature weights
WEIGHTS: tuple[float, ...] = (
    4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
)
#: index of the opposite direction
OPPOSITE: tuple[int, int, int, int, int, int, int, int, int] = (
    0, 3, 4, 1, 2, 7, 8, 5, 6,
)
#: collision pairs that bounce back into each other (non-rest)
_BOUNCE_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 3), (2, 4), (5, 7), (6, 8),
)


def equilibrium(rho: np.ndarray, u: np.ndarray, v: np.ndarray
                ) -> np.ndarray:
    """D2Q9 Maxwell-Boltzmann equilibrium distributions ``f_eq[9]``."""
    u2 = u * u + v * v
    feq = np.empty((9,) + rho.shape, dtype=float)
    feq[0] = WEIGHTS[0] * rho * (1.0 - 1.5 * u2)
    for i in range(1, 9):
        ex, ey = VELOCITIES[i]
        cu = 3.0 * (ex * u + ey * v)
        feq[i] = WEIGHTS[i] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * u2)
    return feq


class LatticeBoltzmann:
    """D2Q9 BGK flow solver over a ``width x height`` lattice.

    Args:
        width, height: lattice dimensions
        omega: BGK relaxation rate in (0, 2) (kinematic viscosity
            ``(1/omega - 1/2)/3``; 1.0 = moderate diffusion)
        inlet_velocity: uniform inflow speed (sites/step) at x = 0, or a
            numpy (H,) array for a shaped (e.g. parabolic) inlet profile
        inlet_density, outlet_density: prescribed equilibrium densities
            at x = 0 and x = width - 1 (open channel).  ``inlet_density
            > outlet_density`` is a constant pressure difference that
            drives Poiseuille flow; ``inlet_velocity`` may additionally
            impose a shaped profile.
        closed: when True every edge is a no-slip bounce-back wall
            (closed box for mass-conservation tests).
        periodic_x: when True the x edges wrap (no inlet/outlet); pairs
            with ``body_force`` for the periodic-channel benchmark.
        body_force: uniform acceleration ``(fx, fy)`` applied every step
            (Guo 2002 discrete forcing) — the body-force equivalent of a
            constant pressure difference along the channel.
    """

    def __init__(
        self,
        width: int,
        height: int,
        omega: float = 1.0,
        inlet_velocity: float | np.ndarray = 0.0,
        inlet_density: float = 1.001,
        outlet_density: float = 0.999,
        closed: bool = False,
        periodic_x: bool = False,
        body_force: tuple[float, float] | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("lattice dimensions must be > 0")
        if not 0.0 < omega < 2.0:
            raise ValueError("omega must be in (0, 2)")
        self.width = width
        self.height = height
        self.omega = float(omega)
        self.closed = bool(closed)
        self.periodic_x = bool(periodic_x)
        self.inlet_velocity = inlet_velocity
        self.inlet_density = float(inlet_density)
        self.outlet_density = float(outlet_density)
        self.body_force = body_force
        # full-node distribution functions f[i, y, x]
        self.f: np.ndarray = np.zeros((9, height, width), dtype=float)
        rho0 = np.ones((height, width), dtype=float)
        self.f[:] = equilibrium(rho0, np.zeros_like(rho0),
                                 np.zeros_like(rho0))
        # macroscopic fields
        self.rho: np.ndarray = rho0.copy()
        self.u: np.ndarray = np.zeros((height, width), dtype=float)
        self.v: np.ndarray = np.zeros((height, width), dtype=float)
        # momentum-exchange force on obstacles (per site, per step)
        self.force_x: np.ndarray = np.zeros((height, width), dtype=float)
        self.force_y: np.ndarray = np.zeros((height, width), dtype=float)
        self._occupancy: np.ndarray = np.zeros((height, width), dtype=bool)
        self.tick = 0
        self._update_macroscopic()

    # -- configuration --
    def set_occupancy(self, mask: np.ndarray | list[list[bool]]) -> None:
        """Mark occupied (solid) sites; they bounce the flow back.

        ``mask`` is indexed ``[y][x]``; the wall rows are always solid.
        """
        occ = np.asarray(mask, dtype=bool)
        if occ.shape != (self.height, self.width):
            raise ValueError(
                f"occupancy mask must be ({self.height}, {self.width})")
        self._occupancy = occ.copy()

    @property
    def solid(self) -> np.ndarray:
        """True at solid nodes (walls + obstacles)."""
        solid = self._occupancy.copy()
        solid[0, :] = True
        solid[-1, :] = True
        if self.closed:
            solid[:, 0] = True
            solid[:, -1] = True
        return solid
    def total_mass(self) -> float:
        """Total density (sum over sites); conserved in a closed box."""
        return float(self.rho.sum())

    def velocity_fields(self) -> tuple[np.ndarray, np.ndarray]:
        """(u, v) macroscopic velocities (sites/step), indexed [y][x]."""
        return self.u, self.v

    def flow_field(self, substeps: int = 1) -> FlowField:
        """The solver velocity field as a :class:`~helixlang.flow.FlowField`.

        ``substeps`` is how many LBM ticks run per population tick; the
        per-step velocities (sites/step) are scaled by it so the field
        is in the population's sites-per-tick units.  Fluid velocities
        are first spread into solid/obstacle cells (bounce-back nodes),
        so a cell sitting on its own no-slip site still feels the local
        current rather than the solver's spurious node velocity.
        """
        scale = float(max(1, int(substeps)))
        u = self._spread_velocity(self.u, self.v)[0] * scale
        v = self._spread_velocity(self.u, self.v)[1] * scale
        return FlowField(
            self.width, self.height,
            u.tolist(), v.tolist(),
        )

    def _spread_velocity(
        self, u: np.ndarray, v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (u, v) copies with solid-node velocities interpolated.

        Solid cells (walls + obstacles) take the mean velocity of their
        fluid neighbours, iterated a few passes so several-cell-thick
        colonies are also covered; fully-enclosed solid cells keep the
        solver value (near zero).
        """
        fluid = ~self.solid
        if fluid.all():
            return u, v
        u_out = np.array(u)
        v_out = np.array(v)
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (1, -1), (-1, 1), (1, 1))
        for _ in range(4):
            nsum = np.zeros_like(u_out)
            vsum = np.zeros_like(v_out)
            cnt = np.zeros_like(u_out, dtype=float)
            for dx, dy in offsets:
                fs = np.roll(np.roll(fluid, dy, axis=0), dx, axis=1)
                if not fs.any():
                    continue
                nsum += np.where(fs, np.roll(np.roll(u_out, dy, axis=0), dx, axis=1), 0.0)
                vsum += np.where(fs, np.roll(np.roll(v_out, dy, axis=0), dx, axis=1), 0.0)
                cnt += fs
            solid_open = (~fluid) & (cnt > 0)
            if not solid_open.any():
                break
            u_out = np.where(solid_open, nsum / np.maximum(cnt, 1e-12), u_out)
            v_out = np.where(solid_open, vsum / np.maximum(cnt, 1e-12), v_out)
        return u_out, v_out

    def _update_macroscopic(self) -> None:
        rho = self.f.sum(axis=0)
        self.rho = rho
        u = np.zeros_like(rho)
        v = np.zeros_like(rho)
        for i in range(9):
            ex, ey = VELOCITIES[i]
            if ex:
                u += ex * self.f[i]
            if ey:
                v += ey * self.f[i]
        self.u = u / rho
        self.v = v / rho

    # -- main step --
    def step(self) -> None:
        """One collision + streaming + boundary tick."""
        # collide (BGK relaxation toward local equilibrium, fluid nodes only;
        # solid nodes are set purely by streaming + bounce-back)
        f = self.f.copy()
        fluid = ~self.solid
        feq = equilibrium(self.rho, self.u, self.v)
        f[:, fluid] += self.omega * (feq[:, fluid] - f[:, fluid])
        # stream: x is periodic (wraps), y is not (walls at rows 0/H-1;
        # the boundary rows are left for the bounce-back step below, so the
        # two walls never exchange populations through the wrap).
        for i in range(1, 9):
            ex, ey = VELOCITIES[i]
            if ex:
                f[i] = np.roll(f[i], shift=ex, axis=1)
            if ey == 1:
                f[i][1:] = f[i][:-1]
            elif ey == -1:
                f[i][:-1] = f[i][1:]

        # body force (Guo 2002 discrete forcing): acts as a uniform
        # pressure gradient along the periodic channel.
        if self.body_force is not None:
            self._apply_body_force(f)

        solid = self.solid

        # momentum exchange at the solid nodes (Ladd 1994): the force on a
        # solid node is the sum over its fluid-facing links i of
        # ``e_i (f_i + f_opp)``.  Only links whose far end is fluid
        # contribute, so interior obstacle nodes (all-solid neighbourhood)
        # contribute nothing and the hydrostatic equilibrium offset
        # cancels over closed surfaces — leaving the physical drag.
        fx = np.zeros((self.height, self.width), dtype=float)
        fy = np.zeros((self.height, self.width), dtype=float)
        for i in range(9):
            ex, ey = VELOCITIES[i]
            if ex == 0 and ey == 0:
                continue
            neighbor = np.roll(solid, shift=(-ey, -ex), axis=(0, 1))
            fluid_link = solid & ~neighbor
            pair = f[i] + f[OPPOSITE[i]]
            if ex:
                fx[fluid_link] -= ex * pair[fluid_link]
            if ey:
                fy[fluid_link] -= ey * pair[fluid_link]
        self.force_x = fx
        self.force_y = fy

        # full-node bounce-back: reflect the streamed populations at
        # solid nodes (no-slip walls + obstacles).
        for i, opp in _BOUNCE_PAIRS:
            f_i_solid = f[i][solid]
            f[i][solid] = f[opp][solid]
            f[opp][solid] = f_i_solid

        # inlet / outlet (open channel only)
        if not self.closed and not self.periodic_x:
            self._apply_inlet(f)
            self._apply_outlet(f)

        self.f = f
        self._update_macroscopic()
        self.tick += 1

    def _apply_body_force(self, f: np.ndarray) -> None:
        """Guo 2002 forcing: add ``(1 - w/2) w_i C_i . F`` to each link."""
        if self.body_force is None:
            return
        fx, fy = self.body_force
        g = 1.0 - 0.5 * self.omega
        rho = np.sum(f, axis=0)
        u = np.zeros_like(rho)
        v = np.zeros_like(rho)
        for i in range(9):
            ex, ey = VELOCITIES[i]
            if ex:
                u += ex * f[i]
            if ey:
                v += ey * f[i]
        u /= rho
        v /= rho
        fluid = ~self.solid
        for i in range(9):
            ex, ey = VELOCITIES[i]
            eu = ex * u + ey * v
            term = (3.0 * (ex - u) * fx + 3.0 * (ey - v) * fy
                    + 9.0 * eu * (ex * fx + ey * fy))
            f[i][fluid] += g * WEIGHTS[i] * term[fluid]

    def _apply_inlet(self, f: np.ndarray) -> None:
        """Prescribe equilibrium at x = 0 (density + optional profile)."""
        vel = self.inlet_velocity
        if isinstance(vel, (int, float)):
            profile = np.full(self.height, float(vel), dtype=float)
        else:
            profile = np.asarray(vel, dtype=float)
        rho_in = np.full(self.height, self.inlet_density, dtype=float)
        v_in = np.zeros_like(rho_in)
        # fill the x=0 column from the 1-D equilibrium slices
        f[:, :, 0] = equilibrium(
            rho_in[:, None], profile[:, None], v_in[:, None])[:, :, 0]

    def _apply_outlet(self, f: np.ndarray) -> None:
        """Prescribe equilibrium at x = width - 1 at the outlet density."""
        rho_out = np.full(self.height, self.outlet_density, dtype=float)
        u_out = np.zeros_like(rho_out)
        f[:, :, -1] = equilibrium(
            rho_out[:, None], u_out[:, None], u_out[:, None])[:, :, -1]

    def run(self, steps: int) -> None:
        """Advance ``steps`` ticks."""
        for _ in range(steps):
            self.step()


__all__ = [
    "VELOCITIES", "WEIGHTS", "OPPOSITE",
    "LatticeBoltzmann", "equilibrium",
]
