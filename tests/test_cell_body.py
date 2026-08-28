"""Level-3 rod-cell body + contact + LBM coupling tests
(doc/18-programmable-cell-population-simulation.md §13 Design 6, gates 3-5).

Acceptance gates:

3. **Obstacle flow** (LBM integration): the flow field is spread into
   solid/obstacle cells so a rod sitting on its own no-slip site feels
   the local current, and a rod drifts downstream under the Level-2
   LBM flow (one-way coupling: flow drags cells).
4. **Nutrient boundary layer** (covered by the Level-1 advection +
   environment tests in ``test_flow.py``).
5. **Rods do not overlap**: two growing rods pushed together relax to
   zero overlap under Hertzian contact + Stokes drag, and the two
   half-length daughters of a fission do not penetrate (``divide_cell``
   rod branch).

Also covered: capsule geometry/volume, segment-distance contact
normals, wall contact, drag coefficients, ``#sim``/``#config`` wiring
(``lbm=true``, ``flow=channel_poiseuille``, ``mechanics=contact``).
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.plugins.runtime.cell_body import (
    CellBody,
    capsule_radius,
    hertzian_force,
    resolve_rod_contacts,
    rod_rod_contact,
    rod_volume_um3,
    rod_wall_contacts,
    stokes_drag_coefficient,
)
from helixlang.plugins.runtime.population import (
    CellPopulation,
    PopulationCell,
    PopulationConfig,
    divide_cell,
)
from helixlang.sim_runtime import (
    _build_population_config,
    _seed_cells,
    run,
)


def parse(src: str):
    return Parser(list(Lexer(src).tokens())).parse()


def _make_rod(x: float, y: float, length_um: float = 4.0,
              diameter_um: float = 1.0, angle: float = 0.0) -> CellBody:
    return CellBody(x=x, y=y, length_um=length_um,
                    diameter_um=diameter_um, angle=angle)


# ============================================================================
# Geometry
# ============================================================================
def test_capsule_radius_and_volume() -> None:
    assert capsule_radius(1.0) == pytest.approx(0.5)
    assert capsule_radius(-2.0) == pytest.approx(0.0)
    # V = pi r^2 L + 4/3 pi r^3  (length-2, diameter-1 rod)
    r = 0.5
    expected = math.pi * r * r * 2.0 + (4.0 / 3.0) * math.pi * r ** 3
    assert rod_volume_um3(2.0, 1.0) == pytest.approx(expected)
    # a point-like rod (L=0) is a sphere
    assert rod_volume_um3(0.0, 1.0) == pytest.approx((4.0 / 3.0) * math.pi * r ** 3)


def test_rod_axis_endpoints_and_volume() -> None:
    rod = _make_rod(10.0, 20.0, length_um=4.0)
    assert rod.axis() == (1.0, 0.0)
    assert rod.endpoints() == (8.0, 20.0, 12.0, 20.0)
    assert rod.volume_um3() == pytest.approx(rod_volume_um3(4.0, 1.0))
    up = _make_rod(0.0, 0.0, angle=math.pi / 2.0)
    assert up.axis() == pytest.approx((0.0, 1.0))


def test_lattice_clamps_and_sites_cover_capsule() -> None:
    rod = _make_rod(205.0, 125.0, length_um=4.0)
    assert rod.lattice(40, 40, 10.0) == (20, 12)          # round(x/10)
    assert _make_rod(-5.0, 5.0).lattice(10, 10, 10.0) == (0, 0)   # clamp
    # a long rod (spans > 2 sites at 10 µm spacing) covers several sites
    long = _make_rod(205.0, 125.0, length_um=20.0)
    sites = long.sites(40, 40, 10.0)
    assert len(sites) >= 3
    assert (20, 12) in sites
    # every reported site lies within the capsule's bounding box
    for sx, sy in sites:
        px = (sx + 0.5) * 10.0
        py = (sy + 0.5) * 10.0
        assert long.x - 10.5 <= px <= long.x + 10.5
        assert long.y - 0.5 <= py <= long.y + 0.5


# ============================================================================
# Contact
# ============================================================================
def test_hertzian_force_zero_apart_and_scales() -> None:
    assert hertzian_force(1e3, 0.0) == 0.0
    assert hertzian_force(1e3, -0.5) == 0.0
    assert hertzian_force(1e3, 0.5) == pytest.approx(1e3 * 0.5 * math.sqrt(0.5))
    assert hertzian_force(2e3, 0.5) == pytest.approx(2.0 * hertzian_force(1e3, 0.5))


def test_stokes_drag_coefficient_numeric() -> None:
    # 6*pi*mu*r, mu = 1 mPa·s = 1e-3/60 pN·min/µm²
    c = stokes_drag_coefficient(0.5, 1.0)
    assert c == pytest.approx(6.0 * math.pi * 1e-3 / 60.0 * 0.5)
    assert stokes_drag_coefficient(1.0, 1.0) == pytest.approx(2.0 * c)


def test_rod_rod_contact_overlap_and_normal() -> None:
    # side-by-side parallel rods, centres 0.8 apart -> overlap 0.2,
    # normal from b toward a is +y (b below a)
    a = _make_rod(10.0, 10.0)
    b = _make_rod(10.0, 9.2)
    contact = rod_rod_contact(a, b)
    assert contact is not None
    overlap, nx, ny = contact
    assert overlap == pytest.approx(1.0 - 0.8)
    assert nx == pytest.approx(0.0)
    assert ny == pytest.approx(1.0)


def test_rod_rod_contact_none_when_apart() -> None:
    a = _make_rod(0.0, 0.0)
    b = _make_rod(3.0, 3.0)
    assert rod_rod_contact(a, b) is None


def test_rod_rod_contact_collinear_fallback() -> None:
    # exactly collinear rods: closest points coincide -> axis-difference
    # fallback (here zero, so +x)
    a = _make_rod(0.0, 0.0)
    b = _make_rod(0.3, 0.0)
    contact = rod_rod_contact(a, b)
    assert contact is not None
    overlap, nx, ny = contact
    assert overlap > 0.0
    assert nx == pytest.approx(1.0) and ny == pytest.approx(0.0)


def test_rod_rod_contact_cross() -> None:
    # cross of two +-long rods centred on the same point
    a = _make_rod(0.0, 0.0, length_um=2.0)
    b = _make_rod(0.0, 0.0, length_um=2.0, angle=math.pi / 2.0)
    assert rod_rod_contact(a, b) is not None


def test_rod_wall_contacts_all_four_walls() -> None:
    rod = _make_rod(0.8, 0.4, length_um=4.0)   # spans x in [-1.2, 2.8], y-caps below 0
    out = rod_wall_contacts(rod, x_max=10.0, y_max=10.0)
    by_n = {(round(nx, 2), round(ny, 2)): ov for ov, nx, ny in out}
    assert (1.0, 0.0) in by_n and by_n[(1.0, 0.0)] > 0.0   # left wall
    assert (0.0, 1.0) in by_n and by_n[(0.0, 1.0)] > 0.0   # bottom wall
    far = _make_rod(9.2, 9.6, length_um=4.0)  # near the top-right corner
    by_n = {(round(nx, 2), round(ny, 2)): ov for ov, nx, ny in
            rod_wall_contacts(far, x_max=10.0, y_max=10.0)}
    assert (-1.0, 0.0) in by_n and by_n[(-1.0, 0.0)] > 0.0  # right wall
    assert (0.0, -1.0) in by_n and by_n[(0.0, -1.0)] > 0.0  # top wall
    # a rod at the centre of a 10x10 box touches no wall
    assert rod_wall_contacts(_make_rod(5.0, 5.0, length_um=1.0),
                             10.0, 10.0) == []


# ------------------------------------------------------------------ gate 5
def test_gate5_resolve_overlapping_rods_to_zero() -> None:
    """Two overlapping parallel rods relax to zero overlap."""
    drag = stokes_drag_coefficient(0.5, 1.0)
    bodies = [
        _make_rod(10.0, 10.0),
        _make_rod(10.0, 9.4),   # 0.6 apart, overlap 0.4
    ]
    resolve_rod_contacts(bodies, drag, stiffness=1.0e3,
                         x_max=100.0, y_max=100.0)
    contact = rod_rod_contact(bodies[0], bodies[1])
    assert contact is None or contact[0] < 1e-4
    # centres separated to ~1.0 (the diameter)
    assert bodies[0].y - bodies[1].y == pytest.approx(1.0, abs=1e-3)


def test_gate5_resolve_collinear_rods_end_to_end() -> None:
    """Collinear growing rods relax end-to-end (centres 2r + L apart)."""
    drag = stokes_drag_coefficient(0.5, 1.0)
    bodies = [
        _make_rod(10.0, 10.0, length_um=2.0),
        _make_rod(10.3, 10.0, length_um=2.0),
    ]
    resolve_rod_contacts(bodies, drag, stiffness=1.0e3,
                         x_max=100.0, y_max=100.0)
    contact = rod_rod_contact(bodies[0], bodies[1])
    assert contact is None or contact[0] < 1e-4
    assert abs(bodies[1].x - bodies[0].x) == pytest.approx(3.0, abs=1e-3)


def test_gate5_divide_cell_daughters_do_not_penetrate() -> None:
    """Fission of a length-8 rod yields two non-overlapping halves with
    conserved volume (adder control on the body)."""
    cfg = PopulationConfig(grid_width=40, grid_height=40,
                           cell_shape="rod", cell_length_um=4.0,
                           cell_diameter_um=1.0, mechanics="contact",
                           division_threshold=1e9)
    parent = PopulationCell(id=0, x=20, y=20, energy=1e9)
    parent.body = _make_rod(205.0, 205.0, length_um=8.0)
    a, b = divide_cell(parent, cfg, random.Random(3))
    assert a.body is not None and b.body is not None
    # daughters sit back-to-back on the parent axis (parent centre 205, 205
    # is preserved as the midpoint; the axial +/- epsilon jitter cancels)
    assert (a.body.x + b.body.x) / 2.0 == pytest.approx(205.0, abs=1e-6)
    assert (a.body.y + b.body.y) / 2.0 == pytest.approx(205.0, abs=1e-6)
    # adder control: fission halves the length back to the birth length
    assert a.body.length_um == pytest.approx(4.0)
    assert b.body.length_um == pytest.approx(4.0)
    # back-to-back fission starts slightly overlapping (the two cap
    # hemispheres share the fission point); the contact solver separates
    # them so the halves do not penetrate (gate 5)
    drag = stokes_drag_coefficient(0.5, 1.0)
    resolve_rod_contacts([a.body, b.body], drag, stiffness=1.0e3,
                         x_max=400.0, y_max=400.0)
    contact = rod_rod_contact(a.body, b.body)
    assert contact is None or contact[0] < 1e-4
    # lattice coordinates are re-derived from the bodies and stay in bounds
    assert 0 <= a.x < 40 and 0 <= b.x < 40


# ============================================================================
# LBM coupling (gates 3 & 5 integration)
# ============================================================================
def test_lbm_flow_field_spreads_into_solid_cells() -> None:
    """flow_field() fills obstacle cells with the local fluid velocity so
    a cell on its own no-slip site still feels the current."""
    from helixlang.plugins.apps.lattice_boltzmann import LatticeBoltzmann

    lbm = LatticeBoltzmann(width=24, height=24, omega=1.0)
    lbm.set_occupancy([[False] * 24 for _ in range(24)])
    lbm.run(3000)
    body = _make_rod(55.0, 125.0, length_um=4.0)
    mask = np.zeros((24, 24), dtype=bool)
    for sx, sy in body.sites(24, 24, 10.0):
        mask[sy][sx] = True
    lbm.set_occupancy(mask)
    lbm.run(500)
    u = np.asarray(lbm.u)
    assert u[12, 5] <= 0.0 or u[12, 5] < 1e-5   # raw solid node is ~0/negative
    f = lbm.flow_field()
    # spread flow now gives the rod's centre site a positive (E) current
    assert f.velocity(5, 12)[0] > 1e-5


def test_gate5_lbm_drag_advects_rod_downstream() -> None:
    """A single rod under the Level-2 LBM flow drifts east (one-way
    coupling: flow drags cells)."""
    from helixlang.plugins.apps.lattice_boltzmann import LatticeBoltzmann

    cfg = PopulationConfig(
        grid_width=24, grid_height=24,
        cell_shape="rod", cell_length_um=4.0, cell_diameter_um=1.0,
        mechanics="contact", contact_stiffness=2.0e3,
        lbm=LatticeBoltzmann(width=24, height=24, omega=1.0),
        flow_substeps=1, division_threshold=1e9, energy_intake=1.0,
    )
    lbm = cfg.lbm
    lbm.set_occupancy([[False] * 24 for _ in range(24)])
    lbm.run(3000)
    cells = [PopulationCell(id=0, x=5, y=12, energy=1e9)]
    cells[0].body = _make_rod(55.0, 125.0, length_um=4.0)
    pop = CellPopulation(cells, cfg)
    x0 = pop.cells[0].body.x
    for _ in range(30):
        pop.step()
    x1 = pop.cells[0].body.x
    assert x1 > x0 + 1e-3


def test_population_rejects_contact_without_rods() -> None:
    with pytest.raises(ValueError, match="mechanics=contact requires "
                                         "cell_shape=rod"):
        CellPopulation([], PopulationConfig(mechanics="contact"))


# ============================================================================
# #sim / #config wiring
# ============================================================================
def test_sim_lbm_true_builds_solver() -> None:
    prog = parse("#sim lbm=true relaxation_omega=1.2")
    cfg = _build_population_config(prog)
    assert cfg.lbm is not None and cfg.lbm.omega == pytest.approx(1.2)
    assert cfg.cell_shape is None
    # lbm default substeps = 1; #sim lbm=false keeps the default path
    assert cfg.flow_substeps == 1
    assert _build_population_config(parse("#sim lbm=false")).lbm is None


def test_sim_flow_channel_poiseuille_builds_field() -> None:
    prog = parse("#sim flow=channel_poiseuille direction=E "
                 "mean_velocity_um_s=50")
    cfg = _build_population_config(prog)
    assert cfg.flow is not None
    assert cfg.lbm is None
    # centreline velocity = 1.5 x mean; mean 50 µm/s = 300 sites/tick
    u_mid = cfg.flow.velocity(5, 16)[0]
    assert u_mid == pytest.approx(1.5 * 300.0, rel=0.01)
    assert cfg.flow.velocity(5, 0)[0] < u_mid  # slower at the walls


def test_sim_flow_and_lbm_mutually_exclusive() -> None:
    from helixlang.core.errors import SimConfigError

    with pytest.raises(SimConfigError, match="mutually exclusive"):
        _build_population_config(
            parse("#sim flow=channel_poiseuille #sim lbm=true"))


def test_sim_invalid_flow_and_shape_rejected() -> None:
    from helixlang.core.errors import SimConfigError

    with pytest.raises(SimConfigError, match="'flow'"):
        _build_population_config(parse("#sim flow=spiral"))
    with pytest.raises(SimConfigError, match="'cell_shape'"):
        _build_population_config(parse("#config cell_shape=sphere"))


def test_sim_rod_seeds_continuous_bodies() -> None:
    cfg = _build_population_config(
        parse("#config cell_shape=rod length_um=4.0 diameter_um=1.0"))
    cells = _seed_cells(cfg, 4)
    assert len(cells) == 4
    assert all(c.body is not None for c in cells)
    assert cells[0].body.length_um == pytest.approx(4.0)
    assert cells[0].body.diameter_um == pytest.approx(1.0)
    # bodies sit on the grid centres (µm); nearest-site projection matches
    assert abs(cells[0].x - cells[0].body.x / 10.0) <= 0.5


def test_run_rod_biofilm_with_lbm_end_to_end() -> None:
    """The full population backend run with rod + contact + LBM produces
    per-tick rows without errors (regression wiring test)."""
    src = """
#config backend=population
#config population_size=16 grid_width=20 grid_height=20
#config cell_shape=rod length_um=4.0 diameter_um=1.0
#config mechanics=contact contact_stiffness=2.0e3
#sim lbm=true relaxation_omega=1.5 lbm_substeps=5
#config seed=0
#config ticks=6
#config output=alive_count
"""
    result = run(parse(src))
    assert result is not None and result.backend == "population"
    assert len(result.rows) == 6
    assert result.rows[-1]["alive_count"] > 0
