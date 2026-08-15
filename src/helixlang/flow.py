"""Analytical flow fields (doc/18-programmable-cell-population-simulation.md §13 Design 6, Level 1).

A :class:`FlowField` is a 2D velocity field over the simulation
lattice, in **lattice sites per tick** (one site edge = 10 µm, one
tick = 1 min; see :mod:`helixlang.units`).  It drives:

- advective substrate transport (:meth:`ConcentrationField.advect`,
  upwind differencing, CFL sub-stepped),
- cell drift (``PopulationConfig.flow``).

Level 1 ships analytic profiles only (no solver):

- :func:`channel_poiseuille`: pressure-driven flow in a straight
  channel, parabolic ``u(y) = u_peak * 4 y' (1 - y')`` with
  ``u_peak = 1.5 * u_mean`` and no-slip at the walls — the exact
  steady-state of the Stokes / Navier-Stokes equations
  (Hagen-Poiseuille; cf. the Level-2 LBM acceptance gate comparing the
  solver's profile to this parabola).
- :func:`stagnant`: zero velocity everywhere (default).

``mean_velocity_um_s`` is the physical centreline cross-section mean
in µm/s; the conversion to lattice sites per tick is
``u * DIFFUSION_DT_S / LATTICE_SPACING_UM``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from helixlang.units import DIFFUSION_DT_S, LATTICE_SPACING_UM

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is an optional extra
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

#: flow directions accepted by :func:`channel_poiseuille`
FLOW_DIRECTIONS = frozenset({"E", "W", "N", "S"})

#: flow directions accepted by :func:`channel_poiseuille_3d`
#: (E/W along x, N/S along y, U/D along z)
FLOW_DIRECTIONS_3D = frozenset({"E", "W", "N", "S", "U", "D"})


def um_s_to_sites_per_tick(velocity_um_s: float) -> float:
    """Convert a physical flow velocity (µm/s) to lattice sites per tick."""
    return velocity_um_s * DIFFUSION_DT_S / LATTICE_SPACING_UM


@dataclass(slots=True)
class FlowField:
    """2D velocity field over a ``width x height`` lattice.

    Velocities are in lattice sites per tick (positive x = east,
    positive y = north), indexed ``[y][x]`` to match the concentration
    grids.  ``u`` = x-velocity, ``v`` = y-velocity.
    """

    width: int
    height: int
    u: list[list[float]]
    v: list[list[float]]
    _u_arr: object = field(init=False, repr=False, default=None)
    _v_arr: object = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if len(self.u) != self.height or len(self.v) != self.height:
            raise ValueError("velocity rows must match the field height")
        if any(len(row) != self.width for row in self.u + self.v):
            raise ValueError("velocity columns must match the field width")

    def velocity(self, x: int, y: int) -> tuple[float, float]:
        """(u, v) in sites/tick at lattice position (x, y); zero outside."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0, 0.0
        return self.u[y][x], self.v[y][x]

    def max_magnitude(self) -> float:
        """Peak |velocity| in sites/tick (drives the CFL substep count)."""
        peak = 0.0
        for row_u, row_v in zip(self.u, self.v, strict=True):
            for u, v in zip(row_u, row_v, strict=True):
                peak = max(peak, math.hypot(u, v))
        return peak

    def arrays(self) -> tuple[object, object]:
        """Numpy (U, V) arrays (lazy; ``None`` when numpy is absent)."""
        if not _HAS_NUMPY:
            return None, None
        if getattr(self, "_u_arr", None) is None:
            self._u_arr = np.asarray(self.u, dtype=float)
            self._v_arr = np.asarray(self.v, dtype=float)
        return self._u_arr, self._v_arr


def stagnant(width: int, height: int) -> FlowField:
    """Zero-velocity field (no flow)."""
    zeros = [[0.0] * width for _ in range(height)]
    return FlowField(width, height, zeros, [row[:] for row in zeros])


def channel_poiseuille(
    width: int,
    height: int,
    mean_velocity_um_s: float,
    direction: str = "E",
) -> FlowField:
    """Pressure-driven (Poiseuille) channel flow.

    No-slip walls with a parabolic cross-section profile whose
    centreline peak is exactly ``1.5 x mean``:

        u(y') = u_peak * 4 y' (1 - y'),   y' = (y + 0.5) / H

    ``u_peak = 1.5 * u_mean`` in lattice sites per tick.  The profile is
    zero at both walls and maximal at the channel centre; it is the
    analytical steady state that the Level-2 LBM solver is checked
    against (doc/18-programmable-cell-population-simulation.md gate 1).

    Args:
        width, height: lattice dimensions (sites)
        mean_velocity_um_s: cross-section mean velocity, µm/s
        direction: channel axis, one of ``E`` (left-to-right), ``W``,
            ``N`` (bottom-to-top) or ``S``.

    Returns:
        the flow field, indexed [y][x].
    """
    if width <= 0 or height <= 0:
        raise ValueError("flow field dimensions must be > 0")
    if mean_velocity_um_s < 0.0:
        raise ValueError("mean_velocity_um_s must be >= 0")
    if direction not in FLOW_DIRECTIONS:
        raise ValueError(
            f"direction: expected one of {sorted(FLOW_DIRECTIONS)}, "
            f"got {direction!r}")
    mean = um_s_to_sites_per_tick(mean_velocity_um_s)
    peak = 1.5 * mean
    sign = 1.0
    if direction in ("W", "S"):
        sign = -1.0

    if direction in ("E", "W"):
        # flow along x, profile across y
        u_rows: list[list[float]] = []
        v_rows: list[list[float]] = [[0.0] * width for _ in range(height)]
        for y in range(height):
            yp = (y + 0.5) / height
            vel = sign * peak * 4.0 * yp * (1.0 - yp)
            u_rows.append([vel] * width)
        return FlowField(width, height, u_rows, v_rows)

    # flow along y, profile across x
    u_rows = [[0.0] * width for _ in range(height)]
    v_rows = []
    for _y in range(height):
        row_v: list[float] = []
        for x in range(width):
            xp = (x + 0.5) / width
            row_v.append(sign * peak * 4.0 * xp * (1.0 - xp))
        v_rows.append(row_v)
    return FlowField(width, height, u_rows, v_rows)


# ============================================================================
# 3D flow fields (Design 6 Level 2 3D extension, doc/18-programmable-cell-population-simulation.md §13)
# ============================================================================
@dataclass(slots=True)
class FlowField3D:
    """A 3D velocity field over a ``width x height x depth`` lattice.

    Velocities are in lattice sites per tick (positive x = east,
    positive y = north, positive z = up), indexed ``[z][y][x]`` to match
    the concentration volumes.  ``u``/``v``/``w`` are the x/y/z
    components.  Drives 3D substrate advection
    (:meth:`~helixlang.environment.ConcentrationField3D.advect_3d`) and
    3D cell drift.
    """

    width: int
    height: int
    depth: int
    u: list[list[list[float]]]
    v: list[list[list[float]]]
    w: list[list[list[float]]]
    _u_arr: object = field(init=False, repr=False, default=None)
    _v_arr: object = field(init=False, repr=False, default=None)
    _w_arr: object = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if (len(self.u) != self.depth or len(self.v) != self.depth
                or len(self.w) != self.depth):
            raise ValueError("velocity planes must match the field depth")
        for plane_u, plane_v, plane_w in zip(self.u, self.v, self.w,
                                             strict=True):
            if (len(plane_u) != self.height or len(plane_v) != self.height
                    or len(plane_w) != self.height):
                raise ValueError("velocity rows must match the field height")
            if any(len(row) != self.width
                   for row in plane_u + plane_v + plane_w):
                raise ValueError("velocity columns must match the field width")

    def velocity(self, x: int, y: int, z: int = 0) -> tuple[float, float, float]:
        """(u, v, w) in sites/tick at lattice position (x, y, z); zero outside."""
        if not (0 <= x < self.width and 0 <= y < self.height
                and 0 <= z < self.depth):
            return 0.0, 0.0, 0.0
        return self.u[z][y][x], self.v[z][y][x], self.w[z][y][x]

    def max_magnitude(self) -> float:
        """Peak |velocity| in sites/tick (drives the CFL substep count)."""
        peak = 0.0
        for plane_u, plane_v, plane_w in zip(self.u, self.v, self.w,
                                             strict=True):
            for row_u, row_v, row_w in zip(plane_u, plane_v, plane_w,
                                           strict=True):
                for u, v, w in zip(row_u, row_v, row_w, strict=True):
                    peak = max(peak, math.hypot(u, v, w))
        return peak

    def arrays(self) -> tuple[object, object, object]:
        """Numpy (U, V, W) arrays (lazy; ``None`` when numpy is absent)."""
        if not _HAS_NUMPY:
            return None, None, None
        if getattr(self, "_u_arr", None) is None:
            self._u_arr = np.asarray(self.u, dtype=float)
            self._v_arr = np.asarray(self.v, dtype=float)
            self._w_arr = np.asarray(self.w, dtype=float)
        return self._u_arr, self._v_arr, self._w_arr


def stagnant_3d(width: int, height: int, depth: int) -> FlowField3D:
    """Zero-velocity 3D field (no flow)."""
    zeros = [[[0.0] * width for _ in range(height)] for _ in range(depth)]
    return FlowField3D(width, height, depth, zeros,
                       [plane[:] for plane in zeros],
                       [plane[:] for plane in zeros])


def _duct_profile(dim1: int, dim2: int, terms: int = 24) -> list[list[float]]:
    """Unnormalised 3D duct (Boussinesq) Poiseuille profile, ``[dim2][dim1]``.

    The series solution of pressure-driven laminar flow in a rectangular
    duct (four no-slip walls; Whittaker / Boussinesq double sine series):

        u(y, z) = sum_{m,n odd} sin(m pi y'/H) sin(n pi z'/D)
                  / (m n (m^2/H^2 + n^2/D^2))

    with ``y' = (y + 0.5) / dim1``, ``z' = (z + 0.5) / dim2`` at the site
    centres.  The leading constants are omitted -- the caller normalises
    to the requested mean velocity.  ``terms`` sine harmonics give the
    profile to far better than the 2 % gate.
    """
    if _HAS_NUMPY:
        idx1 = np.arange(dim1, dtype=float)
        idx2 = np.arange(dim2, dtype=float)
        yy = (idx1 + 0.5) / dim1
        zz = (idx2 + 0.5) / dim2
        prof = np.zeros((dim2, dim1))
        for m in range(1, terms + 1, 2):
            sy = np.sin(m * np.pi * yy)
            for n in range(1, terms + 1, 2):
                prof += np.outer(np.sin(n * np.pi * zz), sy) / (
                    m * n * (m * m / (dim1 * dim1) + n * n / (dim2 * dim2)))
        result: list[list[float]] = prof.tolist()
        return result
    prof_list = [[0.0] * dim1 for _ in range(dim2)]
    for i in range(dim1):
        y_pos = (i + 0.5) / dim1
        for j in range(dim2):
            z_pos = (j + 0.5) / dim2
            for m in range(1, terms + 1, 2):
                sm = math.sin(m * math.pi * y_pos)
                for n in range(1, terms + 1, 2):
                    prof_list[j][i] += sm * math.sin(n * math.pi * z_pos) / (
                        m * n * (m * m / (dim1 * dim1)
                                 + n * n / (dim2 * dim2)))
    return prof_list


def _scale_to_mean(profile: list[list[float]],
                   mean_sites: float) -> list[list[float]]:
    """Scale a ``[dim2][dim1]`` profile so its mean equals ``mean_sites``."""
    n = len(profile) * (len(profile[0]) if profile else 0)
    if n == 0:
        return profile
    raw_mean = sum(sum(row) for row in profile) / n
    if raw_mean <= 0.0:
        return profile
    scale = mean_sites / raw_mean
    return [[value * scale for value in row] for row in profile]


def channel_poiseuille_3d(
    width: int,
    height: int,
    depth: int,
    mean_velocity_um_s: float,
    direction: str = "E",
) -> FlowField3D:
    """3D pressure-driven (Poiseuille) duct flow.

    The analytic steady state of the Level-2 3D LBM solver gate
    (doc/18-programmable-cell-population-simulation.md §13): pressure-driven laminar flow in a rectangular duct
    with four no-slip walls, using the Boussinesq double sine series
    (:func:`_duct_profile`).  For a square duct the centreline peak is
    ~2.096 x the cross-section mean (vs 1.5 in 2D -- the corners add
    viscous drag).  Zero at all four walls; constant along the flow axis.

    Args:
        width, height, depth: lattice dimensions (sites)
        mean_velocity_um_s: cross-section mean velocity, µm/s
        direction: channel axis, one of ``E`` (along +x), ``W``, ``N``
            (along +y), ``S``, ``U`` (along +z) or ``D``.

    Returns:
        the flow field, indexed [z][y][x] with the velocity along the
        channel axis nonzero.
    """
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError("flow field dimensions must be > 0")
    if mean_velocity_um_s < 0.0:
        raise ValueError("mean_velocity_um_s must be >= 0")
    if direction not in FLOW_DIRECTIONS_3D:
        raise ValueError(
            f"direction: expected one of {sorted(FLOW_DIRECTIONS_3D)}, "
            f"got {direction!r}")
    mean = um_s_to_sites_per_tick(mean_velocity_um_s)
    sign = -1.0 if direction in ("W", "S", "D") else 1.0

    if direction in ("E", "W"):
        profile = _scale_to_mean(_duct_profile(height, depth),
                                 sign * mean)          # [z][y]
        u = [[[plane[y] for x in range(width)] for y in range(height)]
             for plane in profile]
        v = [[[0.0] * width for _ in range(height)] for _ in range(depth)]
        w = [[[0.0] * width for _ in range(height)] for _ in range(depth)]
        return FlowField3D(width, height, depth, u, v, w)

    if direction in ("N", "S"):
        profile = _scale_to_mean(_duct_profile(width, depth),
                                 sign * mean)          # [z][x]
        u = [[[0.0] * width for _ in range(height)] for _ in range(depth)]
        v = [[[plane[x] for x in range(width)] for _ in range(height)]
             for plane in profile]
        w = [[[0.0] * width for _ in range(height)] for _ in range(depth)]
        return FlowField3D(width, height, depth, u, v, w)

    # U / D: flow along z, profile across (x, y) -> [y][x]
    profile = _scale_to_mean(_duct_profile(width, height), sign * mean)
    u = [[[0.0] * width for _ in range(height)] for _ in range(depth)]
    v = [[[0.0] * width for _ in range(height)] for _ in range(depth)]
    w = [[[profile[y][x] for x in range(width)] for y in range(height)]
         for _ in range(depth)]
    return FlowField3D(width, height, depth, u, v, w)


__all__ = [
    "FLOW_DIRECTIONS",
    "FLOW_DIRECTIONS_3D",
    "um_s_to_sites_per_tick",
    "FlowField",
    "FlowField3D",
    "stagnant",
    "stagnant_3d",
    "channel_poiseuille",
    "channel_poiseuille_3d",
]
