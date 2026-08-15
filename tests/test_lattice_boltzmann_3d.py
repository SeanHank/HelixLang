"""Level-2 3D lattice-Boltzmann solver tests (doc/18-programmable-cell-population-simulation.md §13 Design 6, gate 7).

3D counterpart of :mod:`test_lattice_boltzmann` for the D3Q19 BGK solver
``LatticeBoltzmann3D``.

Acceptance gates:

1. **Rectangular-duct analytic profile** (Boussinesq series): in a periodic
   channel driven by a uniform body force, the steady-state cross-section
   profile matches ``channel_poiseuille_3d`` (whose ``y' = (y + 0.5)/H``
   sampling places the no-slip surfaces half a lattice spacing outside the
   wall rows, exactly where full-node bounce-back puts them) with a
   per-point correlation > 0.99 and ``u_peak / u_mean = 2.096`` within 1 %
   (vs 1.5 in 2D - the corners add viscous drag).
2. **Mass conservation**: the closed-box total density drifts by < 1e-6
   over 10^3 steps, and an open channel with a solid obstacle conserves
   mass (no leak at the bounce-back surface).
3. **Obstacle**: the x-flux is conserved along the channel (continuity),
   the flow is displaced around the obstacle (no velocity inside), and the
   momentum-exchange drag points downstream with a ~zero transverse
   component.

Also covered: D3Q19 equilibrium normalisation, ``velocity_fields`` /
``total_mass`` / ``solid`` observables, the density-driven open channel,
the wall-drag / body-force balance, and the ``flow_field`` /
``FlowField3D`` conversion with velocity spreading into solid cells.
"""
from __future__ import annotations

import numpy as np
import pytest

from helixlang.apps.lattice_boltzmann_3d import (
    VELOCITIES,
    WEIGHTS,
    LatticeBoltzmann3D,
    equilibrium,
)
from helixlang.flow import FlowField3D, channel_poiseuille_3d


def _duct_peak_mean_ratio(profile: np.ndarray) -> float:
    """Peak / mean over the interior (wall planes excluded) cross-section."""
    return float(profile.max() / profile.mean())


def test_equilibrium_recovers_density_and_velocity() -> None:
    rho = np.ones((4, 5, 6))
    u = np.full_like(rho, 0.02)
    v = np.full_like(rho, 0.01)
    w = np.zeros_like(rho)
    feq = equilibrium(rho, u, v, w)
    assert feq.shape == (19, 4, 5, 6)
    rho_out = feq.sum(axis=0)
    assert np.allclose(rho_out, rho)
    assert bool(np.all(feq >= 0.0))
    u_out = np.zeros_like(rho)
    v_out = np.zeros_like(rho)
    w_out = np.zeros_like(rho)
    for i in range(19):
        ex, ey, ez = VELOCITIES[i]
        u_out += ex * feq[i]
        v_out += ey * feq[i]
        w_out += ez * feq[i]
    assert np.allclose(u_out / rho_out, u)
    assert np.allclose(v_out / rho_out, v)
    assert np.allclose(w_out / rho_out, w)


def test_equilibrium_rest_state_is_weights() -> None:
    feq = equilibrium(np.ones((1, 1, 1)), np.zeros((1, 1, 1)),
                      np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))
    for i in range(19):
        assert feq[i, 0, 0, 0] == pytest.approx(WEIGHTS[i])


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="omega"):
        LatticeBoltzmann3D(8, 8, 8, omega=2.0)
    with pytest.raises(ValueError, match="dimensions"):
        LatticeBoltzmann3D(0, 8, 8)


def test_solid_mask_marks_walls_and_occupancy() -> None:
    lbm = LatticeBoltzmann3D(10, 8, 6)
    occ = np.zeros((6, 8, 10), dtype=bool)
    occ[3, 4, 3] = True
    lbm.set_occupancy(occ)
    solid = lbm.solid
    assert solid[0, :, :].all() and solid[-1, :, :].all()  # up/down walls
    assert solid[:, 0, :].all() and solid[:, -1, :].all()  # north/south walls
    assert solid[3, 4, 3] and not solid[3, 4, 4]


def test_set_occupancy_checks_shape() -> None:
    lbm = LatticeBoltzmann3D(10, 8, 6)
    with pytest.raises(ValueError, match="occupancy mask"):
        lbm.set_occupancy(np.ones((8, 10), dtype=bool))


def test_total_mass_and_initial_state() -> None:
    lbm = LatticeBoltzmann3D(20, 10, 10)
    assert lbm.total_mass() == pytest.approx(20 * 10 * 10 * 1.0)
    u, v, w = lbm.velocity_fields()
    assert np.allclose(u, 0.0)
    assert np.allclose(v, 0.0)
    assert np.allclose(w, 0.0)


# ---------------------------------------------------------------- gate 1
@pytest.mark.parametrize("omega", [1.0, 1.2])
def test_gate1_poiseuille_body_force_channel(omega: float) -> None:
    size, force = 21, 1.0e-4
    lbm = LatticeBoltzmann3D(
        size, size, size, omega=omega, periodic_x=True,
        body_force=(force, 0.0, 0.0))
    lbm.run(1500)
    u, v, w = lbm.velocity_fields()
    # the flow is purely axial (no cross-flow builds up)
    assert np.allclose(v[1:-1, :, :], 0.0, atol=1e-5)
    assert np.allclose(w[:, 1:-1, :], 0.0, atol=1e-5)
    # wall planes are effectively no-slip (bounce-back residual only)
    assert np.allclose(u[:, 0, :], 0.0, atol=1e-6)
    assert np.allclose(u[:, -1, :], 0.0, atol=1e-6)
    assert np.allclose(u[0, :, :], 0.0, atol=1e-6)
    assert np.allclose(u[-1, :, :], 0.0, atol=1e-6)
    # gate (a): interior peak = 2.096 x mean within 1 %
    profile = u[1:-1, 1:-1, size // 2]
    ratio = _duct_peak_mean_ratio(profile)
    assert ratio == pytest.approx(2.096, rel=0.01)
    # the cross-section matches the analytic duct series: correlation > 0.99
    ref = np.asarray(
        channel_poiseuille_3d(size, size, size, 1.0, "E").u)
    ref_profile = ref[1:-1, 1:-1, size // 2]
    corr = float(np.corrcoef(profile.ravel(), ref_profile.ravel())[0, 1])
    assert corr > 0.99
    # symmetric about the centre in both transverse directions
    h_mid, d_mid = profile.shape[0] - 1, profile.shape[1] - 1
    for dz in range(1, profile.shape[0]):
        assert profile[dz, :] == pytest.approx(profile[h_mid - dz, :],
                                               rel=0.02)
    for dy in range(1, profile.shape[1]):
        assert profile[:, dy] == pytest.approx(profile[:, d_mid - dy],
                                               rel=0.02)


def test_gate1_wall_drag_balances_body_force() -> None:
    height, depth, width, force, omega = 15, 15, 15, 2.4e-4, 1.0
    lbm = LatticeBoltzmann3D(
        width, height, depth, omega=omega,
        periodic_x=True, body_force=(force, 0.0, 0.0))
    lbm.run(1200)
    fx = lbm.force_x
    wall_drag = (fx[:, 0, :].sum() + fx[:, -1, :].sum()
                 + fx[0, :, :].sum() + fx[-1, :, :].sum())
    # (1 - omega/2) * F * N_fluid, Guo 2002 forcing on every fluid node
    body = 0.5 * force * (height - 2) * (depth - 2) * width
    assert wall_drag == pytest.approx(body, rel=0.02)


# ---------------------------------------------------------------- gate 2
def test_gate2_closed_box_mass_conservation() -> None:
    lbm = LatticeBoltzmann3D(15, 15, 15, omega=1.0, closed=True)
    m0 = lbm.total_mass()
    lbm.run(1000)
    drift = abs(lbm.total_mass() - m0)
    assert drift < 1e-6 * m0


def test_gate2_obstacle_does_not_leak_mass() -> None:
    height, depth, width = 15, 15, 40
    lbm = LatticeBoltzmann3D(
        width, height, depth, omega=1.0,
        inlet_density=1.0005, outlet_density=0.9995)
    occ = np.zeros((depth, height, width), dtype=bool)
    occ[6:10, 5:10, 18:22] = True
    lbm.set_occupancy(occ)
    lbm.run(1500)
    m0 = lbm.total_mass()
    lbm.run(500)
    assert abs(lbm.total_mass() - m0) < 1e-6 * m0


# ---------------------------------------------------------------- gate 3
def test_gate3_open_channel_duct() -> None:
    height, depth, width = 17, 17, 40
    lbm = LatticeBoltzmann3D(
        width, height, depth, omega=1.0,
        inlet_density=1.0005, outlet_density=0.9995)
    lbm.run(1500)
    u, v, w = lbm.velocity_fields()
    # developed duct profile at mid-channel
    profile = u[1:-1, 1:-1, width // 2]
    ratio = _duct_peak_mean_ratio(profile)
    assert ratio == pytest.approx(2.096, rel=0.01)
    assert np.allclose(v[1:-1, :, width // 2], 0.0, atol=1e-5)
    assert np.allclose(w[:, 1:-1, width // 2], 0.0, atol=1e-5)
    # wall planes are no-slip at mid-channel (the inlet/outlet columns
    # carry a prescribed-equilibrium overwrite residual of ~1e-5)
    x = width // 2
    assert np.allclose(u[:, 0, x], 0.0, atol=1e-6)
    assert np.allclose(u[:, -1, x], 0.0, atol=1e-6)
    assert np.allclose(u[0, :, x], 0.0, atol=1e-6)
    assert np.allclose(u[-1, :, x], 0.0, atol=1e-6)


def test_gate3_obstacle_displaces_flow_and_conserves_flux() -> None:
    height, depth, width = 15, 15, 50
    lbm = LatticeBoltzmann3D(
        width, height, depth, omega=1.0,
        inlet_density=1.0005, outlet_density=0.9995)
    occ = np.zeros((depth, height, width), dtype=bool)
    occ[5:10, 5:10, 22:28] = True
    lbm.set_occupancy(occ)
    lbm.run(2000)
    u, _, _ = lbm.velocity_fields()
    # x-flux (continuity) is conserved along the channel
    flux_up = float(u[1:-1, 1:-1, 15].sum())
    flux_down = float(u[1:-1, 1:-1, 42].sum())
    assert flux_down == pytest.approx(flux_up, rel=0.05)
    # the obstacle interior carries no x-velocity
    assert np.allclose(u[5:10, 5:10, 22:28], 0.0, atol=1e-4)
    assert np.allclose(u[6:9, 6:9, 25], 0.0, atol=1e-12)


def test_gate3_momentum_exchange_drag_points_downstream() -> None:
    height, depth, width = 15, 15, 50
    lbm = LatticeBoltzmann3D(
        width, height, depth, omega=1.0,
        inlet_density=1.0005, outlet_density=0.9995)
    occ = np.zeros((depth, height, width), dtype=bool)
    occ[5:10, 5:10, 22:28] = True
    lbm.set_occupancy(occ)
    lbm.run(2000)
    drag_x = float(lbm.force_x[5:10, 5:10, 22:28].sum())
    drag_y = float(lbm.force_y[5:10, 5:10, 22:28].sum())
    drag_z = float(lbm.force_z[5:10, 5:10, 22:28].sum())
    assert drag_x > 0.0  # pushed downstream (+x)
    assert abs(drag_y) < 0.05 * drag_x  # symmetric about y
    assert abs(drag_z) < 0.05 * drag_x  # symmetric about z
    # interior obstacle nodes contribute nothing (fluid-facing links only)
    assert float(lbm.force_x[6:9, 6:9, 25].sum()) == pytest.approx(0.0,
                                                                   abs=1e-12)


def test_flow_field_spreads_into_solid_cells() -> None:
    height, depth, width = 15, 15, 20
    lbm = LatticeBoltzmann3D(
        width, height, depth, omega=1.0,
        periodic_x=True, body_force=(1.0e-4, 0.0, 0.0))
    occ = np.zeros((depth, height, width), dtype=bool)
    occ[6:13, 4:11, 9:11] = True  # 7 x 7 x 2 block (deeper than 4 passes)
    lbm.set_occupancy(occ)
    lbm.run(1200)
    u, _, _ = lbm.velocity_fields()
    field = lbm.flow_field(substeps=3)
    assert isinstance(field, FlowField3D)
    assert (field.width, field.height, field.depth) == (width, height, depth)
    u_arr = np.asarray(field.u)
    # fluid velocities scale by the substep count (sites/tick units)
    assert u_arr[7, 7, 5] == pytest.approx(3.0 * u[7, 7, 5], rel=1e-12)
    # solid cells inherit the local current (spread from fluid neighbours)
    assert abs(u_arr[6, 6, 9]) > 0.0
    assert abs(u_arr[7, 7, 10]) > 0.0
    # cells deeper than the spread radius keep the solver value (near zero)
    assert abs(u_arr[9, 7, 9]) < 1e-3


def test_run_advances_and_observables() -> None:
    lbm = LatticeBoltzmann3D(12, 12, 12, omega=1.0, closed=True)
    lbm.run(5)
    assert lbm.tick == 5
    u, v, w = lbm.velocity_fields()
    assert u.shape == (12, 12, 12) and v.shape == (12, 12, 12)
    assert w.shape == (12, 12, 12)
    assert np.allclose(u[0, :, :], 0.0, atol=1e-12)  # closed walls stay put
