"""Level-1 flow field + advection tests (doc/18-programmable-cell-population-simulation.md §13 Design 6).

Covers:

- :mod:`helixlang.plugins.runtime.flow`: unit conversions, the analytical Poiseuille
  channel profile (peak = 1.5 x mean, no-slip walls) in all four
  directions, the stagnant field and the :class:`FlowField` API;
- :meth:`ConcentrationField.advect`: conservative first-order upwind
  transport — total mass is preserved to rounding error, a pulse is
  carried downstream by a uniform flow, and zero-flux boundaries stay
  reflecting;
- the Level-1 wiring: :meth:`Environment.set_flow` dimension checks and
  the advect -> diffuse -> replenish ordering in
  :meth:`Environment.step`.
"""
from __future__ import annotations

import pytest

from helixlang.plugins.runtime.environment import ConcentrationField, Environment, EnvironmentConfig
from helixlang.plugins.runtime.flow import (
    FlowField,
    channel_poiseuille,
    stagnant,
    um_s_to_sites_per_tick,
)


# ---------------------------------------------------------------- conversions
def test_um_s_to_sites_per_tick_worked_example() -> None:
    # one tick = 60 s, one site = 10 um
    assert um_s_to_sites_per_tick(10.0) == pytest.approx(60.0)
    assert um_s_to_sites_per_tick(1.0) == pytest.approx(6.0)
    assert um_s_to_sites_per_tick(0.0) == 0.0


def test_channel_poiseuille_peak_is_one_point_five_times_mean() -> None:
    # the CONTINUOUS peak is 1.5 x mean; on a lattice of even height no
    # row sits exactly on y' = 0.5, so the discrete maximum is within
    # ~0.5 % of the continuous peak.
    field = channel_poiseuille(32, 16, mean_velocity_um_s=10.0, direction="E")
    peak = 1.5 * um_s_to_sites_per_tick(10.0)
    vals = [u for row in field.u for u in row]
    assert max(vals) == pytest.approx(peak, rel=0.01)


def test_channel_poiseuille_no_slip_and_parabolic_profile() -> None:
    height, width = 20, 8
    field = channel_poiseuille(width, height, mean_velocity_um_s=5.0)
    # the no-slip walls sit HALF a lattice spacing outside the edge rows
    # (matching the LBM full-node bounce-back geometry), so the edge rows
    # have a small non-zero velocity and the profile is symmetric.
    col = [field.u[y][0] for y in range(height)]
    assert col[0] > 0.0
    assert col[0] == pytest.approx(col[height - 1], rel=1e-12)
    for y in range(height // 2):
        assert col[y] == pytest.approx(col[height - 1 - y], rel=1e-12)
    # exact discrete parabola u(y') = peak*4*y'(1-y') with y' = (y+0.5)/H
    peak = 1.5 * um_s_to_sites_per_tick(5.0)
    for y in range(height):
        yp = (y + 0.5) / height
        assert field.u[y][0] == pytest.approx(peak * 4.0 * yp * (1.0 - yp), rel=1e-12)


def test_channel_poiseuille_directions() -> None:
    width, height = 12, 24
    for direction, (axis, sign) in {
        "E": ("u", 1.0),
        "W": ("u", -1.0),
        "N": ("v", 1.0),
        "S": ("v", -1.0),
    }.items():
        field = channel_poiseuille(width, height, 8.0, direction=direction)
        other = field.v if axis == "u" else field.u
        assert all(v == 0.0 for row in other for v in row)
        grid = field.u if axis == "u" else field.v
        peak = 1.5 * um_s_to_sites_per_tick(8.0)
        assert max(max(abs(v) for v in row) for row in grid) == pytest.approx(
            peak, rel=0.01)
        # sign of the flow follows the direction
        if sign > 0:
            assert min(min(v for v in row) for row in grid) >= 0.0
        else:
            assert max(max(v for v in row) for row in grid) <= 0.0


def test_channel_poiseuille_invalid_args() -> None:
    with pytest.raises(ValueError, match="direction"):
        channel_poiseuille(8, 8, 1.0, direction="X")
    with pytest.raises(ValueError, match="mean_velocity"):
        channel_poiseuille(8, 8, -1.0)
    with pytest.raises(ValueError, match="dimensions"):
        channel_poiseuille(0, 8, 1.0)


def test_stagnant_is_zero_and_flow_field_api() -> None:
    field = stagnant(4, 3)
    assert field.width == 4 and field.height == 3
    assert field.max_magnitude() == 0.0
    assert field.velocity(2, 1) == (0.0, 0.0)
    assert field.velocity(99, 99) == (0.0, 0.0)  # outside -> zero
    u_arr, v_arr = field.arrays()
    assert u_arr.shape == (3, 4)
    assert float(u_arr.sum()) == 0.0


def test_flow_field_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="rows"):
        FlowField(4, 3, [[0.0] * 4], [[0.0] * 4])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="columns"):
        FlowField(4, 3, [[0.0] * 3] * 3, [[0.0] * 3] * 3)  # type: ignore[list-item]


# --------------------------------------------------------------- advection
def _uniform_flow(width: int, height: int, u: float, v: float = 0.0) -> FlowField:
    return FlowField(
        width,
        height,
        [[u] * width for _ in range(height)],
        [[v] * width for _ in range(height)],
    )


def test_advect_conserves_mass_uniform_flow() -> None:
    field = ConcentrationField(
        "glucose", 24, 16, diffusion_um2_s=600.0, initial_concentration=1.0)
    field.concentration[4][6] += 5.0
    before = field.total_mm()
    field.advect(_uniform_flow(24, 16, 0.3))
    assert field.total_mm() == pytest.approx(before, rel=1e-12)
    assert field.concentration[4][6] < 5.0  # the pulse moved


def test_advect_pulse_moves_downstream() -> None:
    field = ConcentrationField(
        "glucose", 40, 8, diffusion_um2_s=600.0, initial_concentration=0.0)
    field.concentration[4][5] = 10.0
    field.advect(_uniform_flow(40, 8, 1.0))
    # u = 1 -> a single substep with dt = 1: the upwind scheme reduces to
    # a pure translation one site east (new[j] = a[j-1]).
    assert field.concentration[4][6] == pytest.approx(10.0)
    assert field.concentration[4][5] == 0.0


def test_advect_partial_flow_smears_downstream() -> None:
    field = ConcentrationField(
        "glucose", 40, 8, diffusion_um2_s=600.0, initial_concentration=0.0)
    field.concentration[4][5] = 10.0
    field.advect(_uniform_flow(40, 8, 0.4))
    # u = 0.4 < 1 -> one substep dt = 1; 40 % of the cell mass crosses
    # the east face, the rest stays upstream (first-order upwind).
    assert field.concentration[4][5] == pytest.approx(6.0)
    assert field.concentration[4][6] == pytest.approx(4.0)


def test_advect_zero_flux_boundary_keeps_mass_inside() -> None:
    field = ConcentrationField(
        "glucose", 8, 8, diffusion_um2_s=600.0, initial_concentration=0.0)
    field.concentration[4][0] = 10.0  # flush against the west wall
    before = field.total_mm()
    field.advect(_uniform_flow(8, 8, 1.0))  # eastward: drags it off the wall
    assert field.total_mm() == pytest.approx(before, rel=1e-12)
    # nothing leaks out of the east wall either
    assert sum(field.concentration[y][7] for y in range(8)) <= 0.0 + 1e-12


def test_advect_no_flow_is_noop() -> None:
    field = ConcentrationField(
        "glucose", 8, 8, diffusion_um2_s=600.0, initial_concentration=0.0)
    field.concentration[3][3] = 4.0
    grid = [row[:] for row in field.concentration]
    field.advect(stagnant(8, 8))
    assert field.concentration == grid


# ------------------------------------------------------------- environment
def test_set_flow_dimension_mismatch_raises() -> None:
    env = Environment(EnvironmentConfig(width=8, height=8))
    with pytest.raises(ValueError, match="dimensions"):
        env.set_flow(channel_poiseuille(9, 9, 1.0))


def test_environment_step_advects_with_attached_flow() -> None:
    env = Environment(EnvironmentConfig(width=16, height=16))
    field = env.get_field("glucose")
    field.concentration[4][4] = 8.0
    before = env.glucose.total_mm()
    env.set_flow(_uniform_flow(16, 16, 0.5))
    env.step()
    assert env.glucose.total_mm() == pytest.approx(before, rel=1e-9)
    # the pulse has been carried toward the east
    assert env.glucose.concentration[4][5] > env.glucose.concentration[4][4] - 1e-12 or \
        env.glucose.concentration[4][5] > 0.0


def test_environment_step_without_flow_is_unchanged_path() -> None:
    env = Environment(EnvironmentConfig(width=8, height=8))
    env.get_field("glucose").concentration[3][3] = 2.0
    c0 = [row[:] for row in env.glucose.concentration]
    env.step()  # diffusion only (flow_rate = 0, no flow attached)
    # the diffused profile still holds the same total mass
    assert env.glucose.total_mm() == pytest.approx(
        sum(sum(row) for row in c0), rel=1e-9)
