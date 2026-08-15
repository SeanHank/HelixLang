"""Lattice-Boltzmann D3Q19 BGK flow solver (doc/18-programmable-cell-population-simulation.md §13 Design 6, Level 2 3D).

The 3D counterpart of :mod:`helixlang.apps.lattice_boltzmann` (D2Q9).
It solves the incompressible Navier-Stokes equations on the D3Q19
velocity stencil over a ``width x height x depth`` volume:

- occupied lattice sites (the cells) act as no-slip obstacles through
  full-node bounce-back, so the flow is displaced around the colony
  in three dimensions,
- the macroscopic velocity field is fed back to the population as a
  :class:`~helixlang.flow.FlowField3D` driving 3D substrate advection
  and cell drift,
- the momentum exchange across each bounce-back link gives the 3D force
  the flow exerts on the obstacle cells.

Physics notes (same dimensionless lattice units as the 2D solver):

- D3Q19: one rest node, six axis links and twelve diagonal links with
  weights ``(1/3, 1/18 x6, 1/36 x12)`` and ``cs^2 = 1/3``; recovers the
  incompressible Navier-Stokes equations in the low-Mach limit (Frisch
  1986; Krüger et al. *The Lattice Boltzmann Method*, Springer 2017).
- BGK relaxation with ``nu = (1/omega - 1/2)/3``; full-node bounce-back
  implements no-slip walls/obstacles; momentum exchange (Ladd 1994)
  gives the force on each obstacle node.
- Boundary planes follow the BFL 0.8 *by-lattice equilibrium* scheme:
  the inlet/outlet planes are overwritten with the local equilibrium
  distributions (the 2D full-column overwrite is the degenerate special
  case), and the wall planes are mid-grid bounce-back no-slip surfaces.

Streaming uses a precomputed slice table (one bulk memmove per
direction) instead of ``np.roll``, so opposite walls never exchange
populations through the wrap; the x axis wraps only when
``periodic_x`` is set (np.roll, the same as the 2D solver).
"""
from __future__ import annotations

import numpy as np

from helixlang.flow import FlowField3D

#: D3Q19 velocity directions: 0 = rest, 1-6 = axes, 7-18 = diagonals.
VELOCITIES: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
    (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
)
#: D3Q19 quadrature weights.
WEIGHTS: tuple[float, ...] = (
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
)
#: index of the opposite direction (e_opp = -e_i); 9 non-rest pairs.
OPPOSITE: tuple[int, ...] = (
    0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15,
)


def equilibrium(rho: np.ndarray, u: np.ndarray, v: np.ndarray,
                w: np.ndarray) -> np.ndarray:
    """D3Q19 Maxwell-Boltzmann equilibrium distributions ``f_eq[19]``."""
    u2 = u * u + v * v + w * w
    feq = np.empty((19,) + rho.shape, dtype=float)
    feq[0] = WEIGHTS[0] * rho * (1.0 - 1.5 * u2)
    for i in range(1, 19):
        ex, ey, ez = VELOCITIES[i]
        cu = 3.0 * (ex * u + ey * v + ez * w)
        feq[i] = WEIGHTS[i] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * u2)
    return feq


def _shift_3d(arr: np.ndarray, dz: int, dy: int, dx: int) -> np.ndarray:
    """Shift ``arr`` along (z, y, x) by (dz, dy, dx), zero-filling edges.

    The wrap-free analogue of ``np.roll`` used for neighbour lookups and
    velocity spreading, so opposite walls never exchange through the
    wrap.
    """
    d, h, w = arr.shape
    out = np.zeros_like(arr)
    dz_dst = slice(max(0, dz), min(d, d + dz)) if dz else slice(None)
    dz_src = slice(max(0, -dz), min(d, d - dz)) if dz else slice(None)
    dy_dst = slice(max(0, dy), min(h, h + dy)) if dy else slice(None)
    dy_src = slice(max(0, -dy), min(h, h - dy)) if dy else slice(None)
    dx_dst = slice(max(0, dx), min(w, w + dx)) if dx else slice(None)
    dx_src = slice(max(0, -dx), min(w, w - dx)) if dx else slice(None)
    out[dz_dst, dy_dst, dx_dst] = arr[dz_src, dy_src, dx_src]
    return out


class LatticeBoltzmann3D:
    """D3Q19 BGK flow solver over a ``width x height x depth`` volume.

    Args:
        width, height, depth: lattice dimensions (x, y, z)
        omega: BGK relaxation rate in (0, 2) (kinematic viscosity
            ``(1/omega - 1/2)/3``; 1.0 = moderate diffusion)
        inlet_velocity: uniform inflow speed (sites/step) at x = 0, or a
            numpy ``(depth, height)`` array for a shaped (e.g. parabolic)
            inlet profile
        inlet_density, outlet_density: prescribed equilibrium densities
            at x = 0 and x = width - 1 (open channel); ``inlet_density
            > outlet_density`` is a constant pressure difference that
            drives 3D duct Poiseuille flow.
        closed: when True every face is a no-slip bounce-back wall
            (closed box for mass-conservation tests).
        periodic_x: when True the x faces wrap (no inlet/outlet); pairs
            with ``body_force`` for the periodic-channel benchmark.
        body_force: uniform acceleration ``(fx, fy, fz)`` applied every
            step (Guo 2002 discrete forcing).
    """

    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        omega: float = 1.0,
        inlet_velocity: float | np.ndarray = 0.0,
        inlet_density: float = 1.001,
        outlet_density: float = 0.999,
        closed: bool = False,
        periodic_x: bool = False,
        body_force: tuple[float, float, float] | None = None,
    ) -> None:
        if width <= 0 or height <= 0 or depth <= 0:
            raise ValueError("lattice dimensions must be > 0")
        if not 0.0 < omega < 2.0:
            raise ValueError("omega must be in (0, 2)")
        self.width = width
        self.height = height
        self.depth = depth
        self.omega = float(omega)
        self.closed = bool(closed)
        self.periodic_x = bool(periodic_x)
        self.inlet_velocity = inlet_velocity
        self.inlet_density = float(inlet_density)
        self.outlet_density = float(outlet_density)
        self.body_force = body_force
        # full-node distribution functions f[i, z, y, x]
        self.f: np.ndarray = np.zeros((19, depth, height, width), dtype=float)
        rho0 = np.ones((depth, height, width), dtype=float)
        z = np.zeros_like(rho0)
        self.f[:] = equilibrium(rho0, z, z, z)
        # macroscopic fields
        self.rho: np.ndarray = rho0.copy()
        self.u: np.ndarray = np.zeros_like(rho0)
        self.v: np.ndarray = np.zeros_like(rho0)
        self.w: np.ndarray = np.zeros_like(rho0)
        # momentum-exchange force on obstacles (per site, per step)
        self.force_x: np.ndarray = np.zeros_like(rho0)
        self.force_y: np.ndarray = np.zeros_like(rho0)
        self.force_z: np.ndarray = np.zeros_like(rho0)
        self._occupancy: np.ndarray = np.zeros_like(rho0, dtype=bool)
        self.tick = 0
        self._stream_table = self._build_stream_table()
        self._update_macroscopic()

    # -- configuration --
    @staticmethod
    def _build_stream_table() -> list[
            tuple[slice, slice, slice, slice, slice, slice]]:
        """Precomputed source/target slices for each streamed direction.

        For direction ``i`` the stream is the single bulk assignment
        ``f[i][dz_dst, dy_dst, dx_dst] = f[i][dz_src, dy_src, dx_src]``
        with ``target = source + e_i`` (numpy copies through a temporary
        when source and target overlap, so the in-place memmove is
        safe).  ``periodic_x`` replaces the x pair with ``np.roll``.
        """
        table: list[tuple[slice, slice, slice, slice, slice, slice]] = []
        for i in range(1, 19):
            ex, ey, ez = VELOCITIES[i]
            dz_dst, dz_src = _axis_slices(ez)
            dy_dst, dy_src = _axis_slices(ey)
            dx_dst, dx_src = _axis_slices(ex)
            table.append((dz_dst, dz_src, dy_dst, dy_src,
                          dx_dst, dx_src))
        return table

    def set_occupancy(self, mask: np.ndarray | list) -> None:
        """Mark occupied (solid) sites; they bounce the flow back.

        ``mask`` is indexed ``[z][y][x]``; the y/z wall planes are always
        solid.
        """
        occ = np.asarray(mask, dtype=bool)
        if occ.shape != (self.depth, self.height, self.width):
            raise ValueError(
                f"occupancy mask must be ({self.depth}, {self.height}, "
                f"{self.width})")
        self._occupancy = occ.copy()

    @property
    def solid(self) -> np.ndarray:
        """True at solid nodes (walls + obstacles), indexed [z][y][x]."""
        solid = self._occupancy.copy()
        solid[:, 0, :] = True
        solid[:, -1, :] = True
        solid[0, :, :] = True
        solid[-1, :, :] = True
        if self.closed:
            solid[:, :, 0] = True
            solid[:, :, -1] = True
        return solid

    def total_mass(self) -> float:
        """Total density (sum over sites); conserved in a closed box."""
        return float(self.rho.sum())

    def velocity_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(u, v, w) macroscopic velocities (sites/step), indexed [z][y][x]."""
        return self.u, self.v, self.w

    def flow_field(self, substeps: int = 1) -> FlowField3D:
        """The solver velocity field as a :class:`~helixlang.flow.FlowField3D`.

        ``substeps`` is how many LBM ticks run per population tick; the
        per-step velocities (sites/step) are scaled by it so the field is
        in the population's sites-per-tick units.  Fluid velocities are
        first spread into solid/obstacle cells (bounce-back nodes), so a
        cell sitting on its own no-slip site still feels the local
        current.
        """
        scale = float(max(1, int(substeps)))
        u, v, w = self._spread_velocity_3d(self.u, self.v, self.w)
        return FlowField3D(
            self.width, self.height, self.depth,
            (u * scale).tolist(),
            (v * scale).tolist(),
            (w * scale).tolist(),
        )

    def _spread_velocity_3d(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(u, v, w) copies with solid-node velocities interpolated.

        Solid cells (walls + obstacles) take the mean velocity of their
        26 fluid neighbours, iterated a few passes so several-cell-thick
        colonies are also covered; fully-enclosed solid cells keep the
        solver value (near zero).
        """
        fluid = ~self.solid
        if fluid.all():
            return u, v, w
        u_out = np.array(u)
        v_out = np.array(v)
        w_out = np.array(w)
        offsets = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1)
                   for dx in (-1, 0, 1) if (dz, dy, dx) != (0, 0, 0)]
        for _ in range(4):
            nsum = np.zeros_like(u_out)
            vsum = np.zeros_like(v_out)
            wsum = np.zeros_like(w_out)
            cnt = np.zeros_like(u_out, dtype=float)
            for dz, dy, dx in offsets:
                fs = _shift_3d(fluid, dz, dy, dx)
                if not fs.any():
                    continue
                nsum += np.where(fs, _shift_3d(u_out, dz, dy, dx), 0.0)
                vsum += np.where(fs, _shift_3d(v_out, dz, dy, dx), 0.0)
                wsum += np.where(fs, _shift_3d(w_out, dz, dy, dx), 0.0)
                cnt += fs
            solid_open = (~fluid) & (cnt > 0)
            if not solid_open.any():
                break
            u_out = np.where(solid_open, nsum / np.maximum(cnt, 1e-12),
                             u_out)
            v_out = np.where(solid_open, vsum / np.maximum(cnt, 1e-12),
                             v_out)
            w_out = np.where(solid_open, wsum / np.maximum(cnt, 1e-12),
                             w_out)
        return u_out, v_out, w_out

    def _update_macroscopic(self) -> None:
        rho = self.f.sum(axis=0)
        self.rho = rho
        u = np.zeros_like(rho)
        v = np.zeros_like(rho)
        w = np.zeros_like(rho)
        for i in range(19):
            ex, ey, ez = VELOCITIES[i]
            if ex:
                u += ex * self.f[i]
            if ey:
                v += ey * self.f[i]
            if ez:
                w += ez * self.f[i]
        self.u = u / rho
        self.v = v / rho
        self.w = w / rho

    def _stream(self, f: np.ndarray) -> None:
        """Move every non-rest distribution by one step (index-table)."""
        if self.periodic_x:
            for i in range(1, 19):
                ex, _, _ = VELOCITIES[i]
                if ex:
                    f[i] = np.roll(f[i], shift=ex, axis=2)
                dz_dst, dz_src, dy_dst, dy_src, _, _ = self._stream_table[i - 1]
                f[i][dz_dst, dy_dst, :] = f[i][dz_src, dy_src, :]
        else:
            for i in range(1, 19):
                (dz_dst, dz_src, dy_dst, dy_src,
                 dx_dst, dx_src) = self._stream_table[i - 1]
                f[i][dz_dst, dy_dst, dx_dst] = f[i][dz_src, dy_src, dx_src]

    # -- main step --
    def step(self) -> None:
        """One collision + streaming + boundary tick."""
        # collide (BGK relaxation toward local equilibrium, fluid nodes only)
        f = self.f.copy()
        fluid = ~self.solid
        feq = equilibrium(self.rho, self.u, self.v, self.w)
        f[:, fluid] += self.omega * (feq[:, fluid] - f[:, fluid])
        # stream
        self._stream(f)

        # body force (Guo 2002 discrete forcing)
        if self.body_force is not None:
            self._apply_body_force(f)

        solid = self.solid

        # momentum exchange at the solid nodes (Ladd 1994): the force on
        # a solid node is the sum over its fluid-facing links i of
        # ``e_i (f_i + f_opp)``.  Only links whose far end is fluid
        # contribute, so interior obstacle nodes (all-solid
        # neighbourhood) contribute nothing and the hydrostatic
        # equilibrium offset cancels over closed surfaces.
        fx = np.zeros_like(self.rho)
        fy = np.zeros_like(self.rho)
        fz = np.zeros_like(self.rho)
        for i in range(1, 19):
            ex, ey, ez = VELOCITIES[i]
            # neighbour lookup wraps every axis (as the 2D solver does), so
            # links that point outside the domain face the opposite face's
            # nodes: wall rows land on wall rows and self-exclude, exactly
            # like the 2D solver's ``np.roll`` neighbourhood.
            neighbor = np.roll(np.roll(np.roll(
                solid, -ez, axis=0), -ey, axis=1), -ex, axis=2)
            fluid_link = solid & ~neighbor
            pair = f[i] + f[OPPOSITE[i]]
            if ex:
                fx[fluid_link] -= ex * pair[fluid_link]
            if ey:
                fy[fluid_link] -= ey * pair[fluid_link]
            if ez:
                fz[fluid_link] -= ez * pair[fluid_link]
        self.force_x = fx
        self.force_y = fy
        self.force_z = fz

        # full-node bounce-back: reflect the streamed populations at
        # solid nodes (no-slip walls + obstacles).
        for i in range(1, 19):
            opp = OPPOSITE[i]
            if i >= opp:
                continue
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
        fx, fy, fz = self.body_force
        g = 1.0 - 0.5 * self.omega
        rho = np.sum(f, axis=0)
        u = np.zeros_like(rho)
        v = np.zeros_like(rho)
        w = np.zeros_like(rho)
        for i in range(19):
            ex, ey, ez = VELOCITIES[i]
            if ex:
                u += ex * f[i]
            if ey:
                v += ey * f[i]
            if ez:
                w += ez * f[i]
        u /= rho
        v /= rho
        w /= rho
        fluid = ~self.solid
        for i in range(1, 19):
            ex, ey, ez = VELOCITIES[i]
            eu = ex * u + ey * v + ez * w
            term = (3.0 * (ex - u) * fx + 3.0 * (ey - v) * fy
                    + 3.0 * (ez - w) * fz
                    + 9.0 * eu * (ex * fx + ey * fy + ez * fz))
            f[i][fluid] += g * WEIGHTS[i] * term[fluid]

    def _apply_inlet(self, f: np.ndarray) -> None:
        """Prescribe equilibrium at x = 0 (density + optional profile)."""
        vel = self.inlet_velocity
        if isinstance(vel, (int, float)):
            profile = np.full((self.depth, self.height), float(vel),
                              dtype=float)
        else:
            profile = np.asarray(vel, dtype=float)
        rho_in = np.full((self.depth, self.height), self.inlet_density,
                         dtype=float)
        v_in = np.zeros_like(rho_in)
        w_in = np.zeros_like(rho_in)
        # fill the x=0 plane from the 2-D equilibrium slices
        f[:, :, :, 0] = equilibrium(rho_in, profile, v_in, w_in)

    def _apply_outlet(self, f: np.ndarray) -> None:
        """Prescribe equilibrium at x = width - 1 at the outlet density."""
        rho_out = np.full((self.depth, self.height), self.outlet_density,
                          dtype=float)
        u_out = np.zeros_like(rho_out)
        f[:, :, :, -1] = equilibrium(rho_out, u_out, u_out, u_out)

    def run(self, steps: int) -> None:
        """Advance ``steps`` ticks."""
        for _ in range(steps):
            self.step()


def _axis_slices(e: int) -> tuple[slice, slice]:
    """(target, source) slices moving a plane one step along an axis.

    ``e > 0``: target[1:] = source[:-1] (data moves toward +axis);
    ``e < 0``: target[:-1] = source[1:]; ``e == 0``: identity.
    """
    if e > 0:
        return slice(1, None), slice(None, -1)
    if e < 0:
        return slice(None, -1), slice(1, None)
    return slice(None), slice(None)


__all__ = [
    "VELOCITIES", "WEIGHTS", "OPPOSITE",
    "LatticeBoltzmann3D", "equilibrium",
]
