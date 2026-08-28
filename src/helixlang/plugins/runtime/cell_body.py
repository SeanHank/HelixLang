"""Rod-shaped cell bodies with Hertzian contact and Stokes drag
(doc/18-programmable-cell-population-simulation.md §13 Design 6, Level 3).

A :class:`CellBody` is a 2D spherocylinder — a line segment of length
``length_um`` capped by hemispheres of radius ``diameter_um/2`` — in
continuous micrometre coordinates.  It replaces the point-particle +
occupancy-site cell model:

- rod-rod contact uses the exact nearest distance between the two
  centreline segments, then a Hertzian normal repulsion
  ``F = k * overlap^1.5`` (Hertz 1882; ``k`` is the ``contact_stiffness``
  config key);
- rod-wall contact is the same capsule/plane distance;
- drag follows Stokes' law: a contact force ``F`` moves a rod by
  ``F / (6*pi*mu*r)`` µm per tick, so stiff contacts separate
  quasi-statically without oscillating (iDynoMiCS 2.0-style overdamped
  relaxation, Cockx et al. 2024).

Units: lengths in µm, forces in pN, viscosity in mPa·s, time in ticks
(1 tick = 1 min).  ``6*pi*mu*r`` with ``mu = 1 mPa·s`` (water) is a drag
coefficient in pN·min/µm.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helixlang.plugins.runtime.flow import FlowField


def capsule_radius(diameter_um: float) -> float:
    """Radius of a capsule rod's hemispherical caps (µm)."""
    return 0.5 * max(0.0, diameter_um)


def rod_volume_um3(length_um: float, diameter_um: float) -> float:
    """Spherocylinder volume ``pi r^2 L + 4/3 pi r^3`` (µm³)."""
    r = capsule_radius(diameter_um)
    return math.pi * r * r * length_um + (4.0 / 3.0) * math.pi * r ** 3


def hertzian_force(stiffness: float, overlap: float) -> float:
    """Hertzian normal repulsion (pN): ``k * overlap^1.5``, zero apart."""
    if overlap <= 0.0:
        return 0.0
    return stiffness * overlap * math.sqrt(overlap)


def stokes_drag_coefficient(radius_um: float,
                            viscosity_mpas: float) -> float:
    """Stokes drag coefficient ``6*pi*mu*r`` in pN·min/µm.

    ``viscosity_mpas`` is in mPa·s (water ~= 1.0).  1 mPa·s = 1e-3
    pN·s/µm² = 1e-3/60 pN·min/µm², so with r in µm the coefficient is
    in pN·min/µm — a force in pN divided by it is a displacement in µm
    per 1-minute tick.
    """
    mu = max(0.0, viscosity_mpas) * 1e-3 / 60.0
    return 6.0 * math.pi * mu * max(0.0, radius_um)


def _axis(angle: float) -> tuple[float, float]:
    return math.cos(angle), math.sin(angle)


def _closest_point_on_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float,
) -> tuple[float, float]:
    """Closest point on segment AB to point P (Ericson, RTCD)."""
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        return ax, ay
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = min(1.0, max(0.0, t))
    return ax + t * abx, ay + t * aby


def _segment_distance_sq(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> tuple[float, float, float, float, float]:
    """Squared distance between segments AB and CD (Ericson, RTCD).

    Returns ``(sq_dist, p1x, p1y, p2x, p2y)`` with the closest points.
    """
    d1x = bx - ax
    d1y = by - ay
    d2x = dx - cx
    d2y = dy - cy
    rx = ax - cx
    ry = ay - cy
    a = d1x * d1x + d1y * d1y
    e = d2x * d2x + d2y * d2y
    f = d2x * rx + d2y * ry
    if a <= 1e-12 and e <= 1e-12:
        return rx * rx + ry * ry, ax, ay, cx, cy
    if a <= 1e-12:
        p2x, p2y = _closest_point_on_segment(ax, ay, cx, cy, dx, dy)
        return (p2x - ax) ** 2 + (p2y - ay) ** 2, ax, ay, p2x, p2y
    c = d1x * rx + d1y * ry
    b = d1x * d2x + d1y * d2y
    denom = a * e - b * b
    if denom > 1e-12:
        s = min(1.0, max(0.0, (b * f - c * e) / denom))
    else:
        s = 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = min(1.0, max(0.0, -c / a))
    elif t > 1.0:
        t = 1.0
        s = min(1.0, max(0.0, (b - c) / a))
    p1x = ax + s * d1x
    p1y = ay + s * d1y
    p2x = cx + t * d2x
    p2y = cy + t * d2y
    return (p1x - p2x) ** 2 + (p1y - p2y) ** 2, p1x, p1y, p2x, p2y


@dataclass(slots=True)
class CellBody:
    """A rod (spherocylinder) body in continuous µm coordinates.

    Args:
        x, y: centre of the centreline segment (µm)
        length_um: centreline length, not counting the caps (µm)
        diameter_um: rod diameter (µm)
        angle: axial orientation in radians (0 = along +x)
        vx, vy: velocity in µm/tick (drift bookkeeping; set by the
            contact + drag solver each tick)
    """

    x: float
    y: float
    length_um: float
    diameter_um: float = 1.0
    angle: float = 0.0
    vx: float = 0.0
    vy: float = 0.0

    def axis(self) -> tuple[float, float]:
        """Unit vector along the rod axis."""
        return _axis(self.angle)

    def endpoints(self) -> tuple[float, float, float, float]:
        """Cap-centre endpoints ``(x1, y1, x2, y2)`` of the centreline."""
        ax, ay = self.axis()
        s = self.length_um / 2.0
        return self.x - ax * s, self.y - ay * s, self.x + ax * s, self.y + ay * s

    def volume_um3(self) -> float:
        """Capsule volume (µm³)."""
        return rod_volume_um3(self.length_um, self.diameter_um)

    def lattice(self, width: int, height: int, spacing: float
                ) -> tuple[int, int]:
        """Nearest lattice site, clamped to the grid."""
        sx = max(0, min(width - 1, int(round(self.x / spacing))))
        sy = max(0, min(height - 1, int(round(self.y / spacing))))
        return sx, sy

    def sites(self, width: int, height: int, spacing: float
              ) -> list[tuple[int, int]]:
        """Lattice sites covered by the capsule (for LBM occupancy)."""
        r = capsule_radius(self.diameter_um)
        ax, ay = self.axis()
        s = self.length_um / 2.0
        x1 = self.x - ax * s
        y1 = self.y - ay * s
        x2 = self.x + ax * s
        y2 = self.y + ay * s
        x0 = max(0, int((min(x1, x2) - r) // spacing))
        x1i = min(width - 1, int((max(x1, x2) + r) // spacing))
        y0 = max(0, int((min(y1, y2) - r) // spacing))
        y1i = min(height - 1, int((max(y1, y2) + r) // spacing))
        out: list[tuple[int, int]] = []
        for sy in range(y0, y1i + 1):
            for sx in range(x0, x1i + 1):
                px = (sx + 0.5) * spacing
                py = (sy + 0.5) * spacing
                qx, qy = _closest_point_on_segment(px, py, x1, y1, x2, y2)
                if (px - qx) ** 2 + (py - qy) ** 2 <= r * r:
                    out.append((sx, sy))
        return out


def rod_rod_contact(
    a: CellBody, b: CellBody,
) -> tuple[float, float, float] | None:
    """``(overlap, nx, ny)`` between two rods, or ``None`` apart.

    ``overlap > 0`` when the two capsules intersect; ``(nx, ny)`` is the
    unit normal pointing from ``b`` toward ``a``.
    """
    ax1, ay1, ax2, ay2 = a.endpoints()
    bx1, by1, bx2, by2 = b.endpoints()
    sq, p1x, p1y, p2x, p2y = _segment_distance_sq(
        ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
    dist = math.sqrt(sq)
    overlap = (capsule_radius(a.diameter_um)
               + capsule_radius(b.diameter_um)) - dist
    if overlap <= 0.0:
        return None
    if dist > 1e-9:
        nx = (p1x - p2x) / dist
        ny = (p1y - p2y) / dist
    else:
        axn, ayn = a.axis()
        bxn, byn = b.axis()
        nx = axn - bxn
        ny = ayn - byn
        norm = math.hypot(nx, ny)
        if norm <= 1e-9:
            nx, ny = 1.0, 0.0
        else:
            nx /= norm
            ny /= norm
    return overlap, nx, ny


def rod_wall_contacts(
    body: CellBody,
    x_max: float,
    y_max: float,
    x_min: float = 0.0,
    y_min: float = 0.0,
) -> list[tuple[float, float, float]]:
    """``[(overlap, nx, ny), ...]`` for each penetrated box wall.

    The normal points back into the box.
    """
    r = capsule_radius(body.diameter_um)
    x1, y1, x2, y2 = body.endpoints()
    minx = min(x1, x2) - r
    maxx = max(x1, x2) + r
    miny = min(y1, y2) - r
    maxy = max(y1, y2) + r
    out: list[tuple[float, float, float]] = []
    if minx < x_min:
        out.append((x_min - minx, 1.0, 0.0))
    if maxx > x_max:
        out.append((maxx - x_max, -1.0, 0.0))
    if miny < y_min:
        out.append((y_min - miny, 0.0, 1.0))
    if maxy > y_max:
        out.append((maxy - y_max, 0.0, -1.0))
    return out


def resolve_rod_contacts(
    bodies: list[CellBody],
    drag: float,
    stiffness: float,
    x_max: float,
    y_max: float,
    iterations: int = 40,
    tolerance_um: float = 1e-4,
) -> None:
    """Push overlapping rods apart along their contact normals.

    Quasi-static (overdamped) relaxation: each contact displacement is
    ``min(overlap/2, hertzian_force(stiffness, overlap)/drag)`` so a
    stiff contact separates in one iteration while a soft one relaxes
    geometrically.  Repeated (Jacobi) sweeps drive every overlap below
    ``tolerance_um``.
    """
    n = len(bodies)
    if n == 0:
        return
    for _ in range(max(1, iterations)):
        dxs = [0.0] * n
        dys = [0.0] * n
        for i in range(n):
            for j in range(i + 1, n):
                contact = rod_rod_contact(bodies[i], bodies[j])
                if contact is None:
                    continue
                overlap, nx, ny = contact
                if drag > 0.0:
                    push = min(overlap / 2.0,
                               hertzian_force(stiffness, overlap) / drag)
                else:
                    push = overlap / 2.0
                if push <= 0.0:
                    continue
                dxs[i] += nx * push
                dys[i] += ny * push
                dxs[j] -= nx * push
                dys[j] -= ny * push
        for i in range(n):
            for overlap, nx, ny in rod_wall_contacts(
                    bodies[i], x_max, y_max):
                if drag > 0.0:
                    push = min(overlap,
                               hertzian_force(stiffness, overlap) / drag)
                else:
                    push = overlap
                if push <= 0.0:
                    continue
                dxs[i] += nx * push
                dys[i] += ny * push
        max_move = 0.0
        for i in range(n):
            bodies[i].x += dxs[i]
            bodies[i].y += dys[i]
            bodies[i].vx = dxs[i]
            bodies[i].vy = dys[i]
            max_move = max(max_move, abs(dxs[i]), abs(dys[i]))
        if max_move < tolerance_um:
            break


def advect_rods(bodies: list[CellBody], flow: FlowField | None,
                spacing: float) -> None:
    """Drift rods with the local flow velocity (quasi-static Stokes drag).

    ``flow`` is in lattice sites per tick; ``spacing`` converts to
    µm/tick.  ``None`` (no flow) leaves the rods where they are.
    """
    if flow is None:
        return
    for b in bodies:
        sx = int(b.x // spacing)
        sy = int(b.y // spacing)
        u, v = flow.velocity(sx, sy)
        b.x += u * spacing
        b.y += v * spacing


def divide_rod(body: CellBody, rng: random.Random,
               epsilon: float = 0.05) -> tuple[CellBody, CellBody]:
    """Split a rod at its centre into two half-length daughters.

    The daughters' centrelines lie back-to-back along the parent axis
    (one on each side of the parent centre); their angles are jittered
    by ``+/- epsilon`` radians so the two halves do not stay exactly
    aligned.
    """
    ax, ay = body.axis()
    q = body.length_um / 4.0
    a = CellBody(
        x=body.x - ax * q, y=body.y - ay * q,
        length_um=body.length_um / 2.0, diameter_um=body.diameter_um,
        angle=body.angle + rng.uniform(-epsilon, epsilon))
    b = CellBody(
        x=body.x + ax * q, y=body.y + ay * q,
        length_um=body.length_um / 2.0, diameter_um=body.diameter_um,
        angle=body.angle + rng.uniform(-epsilon, epsilon))
    return a, b


__all__ = [
    "CellBody",
    "capsule_radius",
    "rod_volume_um3",
    "hertzian_force",
    "stokes_drag_coefficient",
    "rod_rod_contact",
    "rod_wall_contacts",
    "resolve_rod_contacts",
    "advect_rods",
    "divide_rod",
]
