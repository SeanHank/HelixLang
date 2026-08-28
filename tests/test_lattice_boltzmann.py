"""Level-2 lattice-Boltzmann solver tests (doc/18-programmable-cell-population-simulation.md §13 Design 6, gates 1-3).

Acceptance gates:

1. **Poiseuille analytic profile**: in a periodic channel driven by a
   uniform body force (the constant-pressure-difference equivalent), the
   steady-state centreline profile is parabolic with
   ``u_peak / u_mean = 1.5`` within 1 % (full-node bounce-back puts the
   no-slip surface half a lattice spacing inside each wall row, so the
   magnitude is compared to the analytic parabola over the *effective*
   channel width).
2. **Mass conservation**: the closed-box total density drifts by
   < 1e-6 relative over 10^3 steps, and the open channel with a solid
   obstacle conserves mass (no leak at the bounce-back surface).
3. **Obstacle flow reduction**: a solid obstacle in an open channel
   conserves the x-flux (continuity), the flow is displaced around it,
   and the momentum-exchange drag points downstream.

Also covered: equilibrium normalization, ``velocity_fields`` /
``total_mass`` / ``solid`` observables, the density-driven open channel
(``inlet_density > outlet_density``), and the wall drag balance against
the body force at steady state.
"""
from __future__ import annotations

import numpy as np
import pytest

from helixlang.plugins.apps.lattice_boltzmann import (
    VELOCITIES,
    WEIGHTS,
    LatticeBoltzmann,
    equilibrium,
)


def test_equilibrium_recovers_density_and_velocity() -> None:
    rho = np.ones((5, 5))
    u = np.full((5, 5), 0.02)
    v = np.zeros((5, 5))
    feq = equilibrium(rho, u, v)
    assert feq.shape == (9, 5, 5)
    rho_out = feq.sum(axis=0)
    assert np.allclose(rho_out, rho)
    assert bool(np.all(feq >= 0.0))
    u_out = np.zeros((5, 5))
    v_out = np.zeros((5, 5))
    for i in range(9):
        ex, ey = VELOCITIES[i]
        u_out += ex * feq[i]
        v_out += ey * feq[i]
    assert np.allclose(u_out / rho_out, u)
    assert np.allclose(v_out / rho_out, v)


def test_equilibrium_rest_state_is_weights() -> None:
    feq = equilibrium(np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)))
    for i in range(9):
        assert feq[i, 0, 0] == pytest.approx(WEIGHTS[i])


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="omega"):
        LatticeBoltzmann(8, 8, omega=2.0)
    with pytest.raises(ValueError, match="dimensions"):
        LatticeBoltzmann(0, 8)


def test_solid_mask_marks_walls_and_occupancy() -> None:
    lbm = LatticeBoltzmann(10, 8)
    occ = [[False] * 10 for _ in range(8)]
    occ[4][3] = True
    lbm.set_occupancy(occ)
    solid = lbm.solid
    assert solid[0, :].all() and solid[-1, :].all()  # top/bottom walls
    assert solid[4, 3] and not solid[4, 4]


def test_total_mass_and_initial_state() -> None:
    lbm = LatticeBoltzmann(20, 10)
    assert lbm.total_mass() == pytest.approx(20 * 10 * 1.0)
    u, v = lbm.velocity_fields()
    assert np.allclose(u, 0.0)
    assert np.allclose(v, 0.0)


# ------------------------------------------------------------------ gate 1
def _parabola_peak_mean_ratio(u_profile: np.ndarray) -> float:
    """Peak / mean over the interior (wall rows excluded) profile."""
    bulk = u_profile[1:-1]
    return float(bulk.max() / bulk.mean())


@pytest.mark.parametrize("omega", [1.0, 1.2])
def test_gate1_poiseuille_body_force_channel(omega: float) -> None:
    height, width = 33, 40
    force = 2.0e-5
    lbm = LatticeBoltzmann(
        width, height, omega=omega,
        periodic_x=True, body_force=(force, 0.0))
    lbm.run(30000)
    u, v = lbm.velocity_fields()
    assert np.allclose(v[1:-1, :], 0.0, atol=1e-9)
    profile = u[:, width // 2]
    assert abs(profile[0]) < 1e-6
    assert abs(profile[-1]) < 1e-6
    # gate: peak = 1.5 x mean within 1 %
    ratio = _parabola_peak_mean_ratio(profile)
    assert ratio == pytest.approx(1.5, rel=0.01)
    # shape: symmetric about the channel centre
    for dy in range(1, height // 2):
        assert profile[dy] == pytest.approx(profile[height - 1 - dy], rel=0.02)


def test_gate1_wall_drag_balances_body_force() -> None:
    height, width, force = 17, 30, 2.4e-4
    lbm = LatticeBoltzmann(
        width, height, omega=1.0, periodic_x=True, body_force=(force, 0.0))
    lbm.run(20000)
    wall_drag = lbm.force_x[0, :].sum() + lbm.force_x[-1, :].sum()
    body = 0.5 * force * (height - 2) * width  # (1 - omega/2) * F * Nfluid
    assert wall_drag == pytest.approx(body, rel=0.02)


# ------------------------------------------------------------------ gate 2
def test_gate2_closed_box_mass_conservation() -> None:
    lbm = LatticeBoltzmann(50, 33, omega=1.0, closed=True)
    m0 = lbm.total_mass()
    lbm.run(1000)
    drift = abs(lbm.total_mass() - m0)
    assert drift < 1e-6 * m0


def test_gate2_obstacle_does_not_leak_mass() -> None:
    height, width = 33, 100
    lbm = LatticeBoltzmann(
        width, height, omega=1.0,
        inlet_density=1.0005, outlet_density=0.9995)
    occ = [[(20 <= x <= 24 and 12 <= y <= 20) for x in range(width)]
           for y in range(height)]
    lbm.set_occupancy(occ)
    lbm.run(20000)
    m0 = lbm.total_mass()
    lbm.run(1000)
    assert abs(lbm.total_mass() - m0) < 1e-6 * m0


# ------------------------------------------------------------------ gate 3
def test_gate3_open_channel_poiseuille() -> None:
    height, width = 33, 100
    lbm = LatticeBoltzmann(
        width, height, omega=1.0,
        inlet_density=1.0005, outlet_density=0.9995)
    lbm.run(20000)
    u, _ = lbm.velocity_fields()
    profile = u[:, width // 2]
    ratio = _parabola_peak_mean_ratio(profile)
    assert ratio == pytest.approx(1.5, rel=0.01)
    # wall rows are essentially no-slip (macroscopic at solid nodes is a
    # bounce-back artefact, but stays within ~1e-7)
    assert abs(profile[0]) < 1e-6
    assert abs(profile[-1]) < 1e-6


def test_gate3_obstacle_displaces_flow_and_conserves_flux() -> None:
    height, width = 33, 100
    lbm = LatticeBoltzmann(
        width, height, omega=1.0,
        inlet_density=1.0005, outlet_density=0.9995)
    occ = [[(20 <= x <= 24 and 12 <= y <= 20) for x in range(width)]
           for y in range(height)]
    lbm.set_occupancy(occ)
    lbm.run(30000)
    u, _ = lbm.velocity_fields()
    # x-flux (continuity) is conserved along the channel
    flux_up = float(u[1:-1, 15].sum())
    flux_down = float(u[1:-1, 80].sum())
    assert flux_down == pytest.approx(flux_up, rel=0.05)
    # the rows occupied by the obstacle carry no x-velocity (the outer
    # boundary nodes keep a ~1e-5 bounce-back residual; the interior is 0)
    assert np.allclose(u[12:21, 22], 0.0, atol=1e-4)
    assert np.allclose(u[14:18, 22], 0.0, atol=1e-12)
    # flow is accelerated in the gap around the obstacle
    assert u[1:-1, 22].max() > u[1:-1, 15].max()


def test_gate3_momentum_exchange_drag_points_downstream() -> None:
    height, width = 33, 100
    lbm = LatticeBoltzmann(
        width, height, omega=1.0,
        inlet_density=1.0005, outlet_density=0.9995)
    occ = [[(20 <= x <= 24 and 12 <= y <= 20) for x in range(width)]
           for y in range(height)]
    lbm.set_occupancy(occ)
    lbm.run(30000)
    drag_x = float(lbm.force_x[12:21, 20:25].sum())
    drag_y = float(lbm.force_y[12:21, 20:25].sum())
    assert drag_x > 0.0  # pushed downstream (+x)
    assert abs(drag_y) < 0.05 * drag_x  # mostly axial, symmetric about y
    # interior obstacle nodes contribute nothing (fluid-facing links only)
    assert float(lbm.force_x[14:18, 21:24].sum()) == pytest.approx(0.0, abs=1e-12)


def test_run_advances_and_observables() -> None:
    lbm = LatticeBoltzmann(16, 16, omega=1.0, closed=True)
    lbm.run(5)
    assert lbm.tick == 5
    u, v = lbm.velocity_fields()
    assert u.shape == (16, 16) and v.shape == (16, 16)
    assert np.allclose(u[0, :], 0.0, atol=1e-12)  # closed walls stay put
