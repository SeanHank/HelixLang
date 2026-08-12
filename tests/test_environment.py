"""Extracellular environment (glucose/O2 fields, Monod kinetics,
crowding effective diffusion) unit tests."""
import math

import pytest

from helixlang.environment import (
    BULK_GLUCOSE_MM,
    BULK_OXYGEN_MM,
    CROMICS_CRITICAL_VOLUME_FRACTION,
    GLUCOSE_DIFFUSION_UM2_S,
    GLUCOSE_HALF_SATURATION_MM,
    OXYGEN_DIFFUSION_UM2_S,
    SITE_VOLUME_L,
    ConcentrationField,
    Environment,
    EnvironmentConfig,
    atp_yield,
    crowding_diffusion_factor,
    michaelis_menten_rate,
    molecules_per_site,
    monod_uptake,
)
from helixlang.units import ATP_PER_GLUCOSE


# -- Monod / Michaelis-Menten kinetics --
def test_monod_half_max_at_ks():
    assert monod_uptake(2.0, GLUCOSE_HALF_SATURATION_MM,
                        GLUCOSE_HALF_SATURATION_MM) == pytest.approx(1.0)


def test_monod_saturation():
    assert monod_uptake(1.0, 100.0, 0.1) == pytest.approx(1.0, abs=1e-3)
    assert monod_uptake(1.0, 0.0, 0.1) == pytest.approx(0.0)


def test_monod_invalid_args():
    with pytest.raises(ValueError):
        monod_uptake(-1.0, 0.5, 0.1)
    with pytest.raises(ValueError):
        monod_uptake(1.0, -0.5, 0.1)
    with pytest.raises(ValueError):
        monod_uptake(1.0, 0.5, 0.0)


def test_michaelis_menten_is_monod_alias():
    assert michaelis_menten_rate(3.0, 0.05, 0.05) == pytest.approx(
        monod_uptake(3.0, 0.05, 0.05))


# -- molecule counts / energy yield --
def test_molecules_per_site_1mm():
    # 1 mM in a (10 um)^3 site = 1e-12 L -> ~6e8 molecules
    n = molecules_per_site(1.0)
    assert n == pytest.approx(1e-3 * 6.022e23 * SITE_VOLUME_L, rel=1e-9)
    assert 6e8 < n < 6.05e8


def test_molecules_per_site_scales_linearly():
    assert molecules_per_site(0.5) == pytest.approx(
        molecules_per_site(1.0) / 2)


def test_atp_yield_38_per_glucose():
    assert atp_yield(10.0) == pytest.approx(10.0 * ATP_PER_GLUCOSE)


def test_atp_yield_from_1mm_site():
    """A full 1 mM site oxidized aerobically yields ~2.3e10 ATP."""
    assert atp_yield(molecules_per_site(1.0)) == pytest.approx(
        1e-3 * 6.022e23 * SITE_VOLUME_L * ATP_PER_GLUCOSE)


# -- crowding effective diffusion (CROMICS) --
def test_crowding_factor_empty_medium_is_one():
    assert crowding_diffusion_factor(0.0) == pytest.approx(1.0)


def test_crowding_factor_free_volume():
    assert crowding_diffusion_factor(0.25) == pytest.approx(0.75)


def test_crowding_factor_close_packing_zero():
    assert crowding_diffusion_factor(0.999) == pytest.approx(0.001)


def test_crowding_critical_threshold_value():
    assert CROMICS_CRITICAL_VOLUME_FRACTION == pytest.approx(0.14)


def test_crowding_factor_invalid_volume_fraction():
    with pytest.raises(ValueError):
        crowding_diffusion_factor(-0.1)
    with pytest.raises(ValueError):
        crowding_diffusion_factor(1.0)


# -- ConcentrationField diffusion --
def test_field_diffusion_conserves_mass():
    f = ConcentrationField("s", 41, 41, GLUCOSE_DIFFUSION_UM2_S)
    f.add(20, 20, 100.0)
    before = f.total_mm()
    f.diffuse()
    assert f.total_mm() == pytest.approx(before, rel=1e-9)


def test_field_diffusion_gaussian_variance():
    """A point source spreads to E[r^2] = 4Dt (D_phys in um^2/s over a
    10 um lattice, one 1-min tick).  Grid 401x401 so the Neumann
    boundaries never reach the Gaussian tail (sigma ~ 38 sites)."""
    size = 401
    f = ConcentrationField("s", size, size, GLUCOSE_DIFFUSION_UM2_S)
    f.add(size // 2, size // 2, 1000.0)
    f.diffuse()
    mass = f.total_mm()
    cy = cx = size // 2
    var = sum(
        f.concentration[i][j] * ((i - cy) ** 2 + (j - cx) ** 2)
        for i in range(size) for j in range(size)
    ) / mass
    # on-lattice D for glucose at 10 um: 600 um^2/s * 60 s / 100 um^2
    d_lattice = GLUCOSE_DIFFUSION_UM2_S * 60.0 / 100.0
    assert var == pytest.approx(4.0 * d_lattice, rel=1e-3)


def test_field_negative_never_occurs():
    f = ConcentrationField("s", 8, 8, OXYGEN_DIFFUSION_UM2_S)
    f.add(0, 0, 1.0)
    f.diffuse()
    assert min(min(row) for row in f.concentration) >= 0.0


def test_field_deplete_clamps():
    f = ConcentrationField("s", 4, 4, 0.0, initial_concentration=1.0)
    removed = f.deplete(0, 0, 5.0)
    assert removed == pytest.approx(1.0)
    assert f.get(0, 0) == pytest.approx(0.0)


# -- Environment --
def test_environment_init_fields():
    env = Environment(EnvironmentConfig(width=10, height=10))
    assert env.glucose.get(0, 0) == pytest.approx(BULK_GLUCOSE_MM)
    assert env.oxygen.get(0, 0) == pytest.approx(BULK_OXYGEN_MM)


def test_environment_flow_replenishes_to_bulk():
    env = Environment(EnvironmentConfig(
        width=8, height=8, flow_rate=0.1,
        bulk_glucose_mm=1.0, glucose_initial_mm=0.0))
    env.step()
    assert env.glucose.get(0, 0) == pytest.approx(0.1)
    env.step()
    assert env.glucose.get(0, 0) == pytest.approx(0.19)


def test_environment_local_uptake_half_max_at_ks():
    env = Environment(EnvironmentConfig(
        width=8, height=8, bulk_glucose_mm=GLUCOSE_HALF_SATURATION_MM,
        glucose_initial_mm=GLUCOSE_HALF_SATURATION_MM))
    assert env.local_uptake(0, 0) == pytest.approx(0.5)


def test_environment_steps_diffuse_and_replenish():
    env = Environment(EnvironmentConfig(width=16, height=16))
    env.glucose.deplete(8, 8, env.glucose.get(8, 8))
    env.step()
    assert env.tick == 1


def test_environment_invalid_config():
    with pytest.raises(ValueError):
        Environment(EnvironmentConfig(width=0, height=8))
    with pytest.raises(ValueError):
        Environment(EnvironmentConfig(flow_rate=1.5))


def test_add_field_dimension_mismatch_raises():
    env = Environment(EnvironmentConfig(width=8, height=8))
    bad = ConcentrationField("extra", 4, 4, 100.0)
    with pytest.raises(ValueError):
        env.add_field("extra", bad)


def test_environment_chemostat_steady_state():
    """With flow, a depleted field converges toward the bulk concentration."""
    env = Environment(EnvironmentConfig(
        width=16, height=16, flow_rate=0.05,
        bulk_glucose_mm=1.0, glucose_initial_mm=0.0))
    for _ in range(200):
        env.step()
    assert env.glucose.get(8, 8) == pytest.approx(1.0, abs=1e-2)


def test_diffusion_coefficient_sanity():
    assert GLUCOSE_DIFFUSION_UM2_S == pytest.approx(600.0)
    assert OXYGEN_DIFFUSION_UM2_S == pytest.approx(2500.0)
    assert math.log(2) > 0
