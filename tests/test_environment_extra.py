"""Additional coverage for environment.py uncovered branches:
3D fields, advection, scalar stepping, flow routing, edge cases."""
import math

import pytest

import helixlang.plugins.runtime.environment as env
from helixlang.plugins.runtime.environment import (
    ClimateTable,
    ConcentrationField,
    ConcentrationField3D,
    DiurnalForcing,
    Environment,
    EnvironmentConfig,
    ScalarField,
    molecules_per_site,
)
from helixlang.plugins.runtime.flow import (
    channel_poiseuille,
    channel_poiseuille_3d,
    stagnant,
    stagnant_3d,
)


def test_molecules_per_site_negative_raises():
    with pytest.raises(ValueError):
        molecules_per_site(-0.1)


def test_concentration_field_get_out_of_bounds():
    c = ConcentrationField("g", 3, 3, 600.0, 1.0)
    assert c.get(-1, 0) == 0.0
    assert c.get(3, 0) == 0.0
    assert c.get(0, 5) == 0.0


def test_concentration_field_set_negative_clamps_and_oob_ignored():
    c = ConcentrationField("g", 3, 3, 600.0, 1.0)
    c.set(1, 1, -5.0)
    assert c.get(1, 1) == 0.0
    c.set(99, 99, 3.0)
    assert c.get(0, 0) == 1.0


def test_concentration_field_set_all_and_add():
    c = ConcentrationField("g", 3, 3, 600.0, 1.0)
    c.set_all(2.5)
    assert c.snapshot() == [[2.5] * 3 for _ in range(3)]
    c.add(0, 0, 1.0)
    assert c.get(0, 0) == 3.5
    c.add(99, 99, 10.0)
    c.add(1, 1, -20.0)
    assert c.get(1, 1) == 0.0


def test_concentration_field_deplete_zero_amount_or_oob():
    c = ConcentrationField("g", 3, 3, 600.0, 5.0)
    assert c.deplete(1, 1, 0.0) == 0.0
    assert c.deplete(1, 1, -3.0) == 0.0
    assert c.deplete(99, 99, 1.0) == 0.0
    c.deplete(1, 1, 10.0)
    assert c.get(1, 1) == 0.0


def test_concentration_field_diffuse_zero_d_and_single_site():
    c = ConcentrationField("g", 3, 3, 0.0, 2.0)
    c.diffuse()
    assert all(row == [2.0] * 3 for row in c.snapshot())
    single = ConcentrationField("g", 1, 1, 600.0, 2.0)
    single.diffuse()
    assert single.snapshot() == [[2.0]]
    single.deplete(0, 0, 0.5)
    assert single.get(0, 0) == 1.5


def test_concentration_field3d_ops():
    c = ConcentrationField3D("g", 4, 4, 4, 600.0, 1.0)
    assert c.get(0, 0, 0) == 1.0
    assert c.get(99, 0, 0) == 0.0
    c.set(1, 1, 1, 3.0)
    assert c.get(1, 1, 1) == 3.0
    c.set(1, 1, 1, -5.0)
    assert c.get(1, 1, 1) == 0.0
    c.set(99, 1, 1, 9.0)
    c.add(2, 2, 2, 2.0)
    assert c.get(2, 2, 2) == 3.0
    c.add(99, 0, 0, 5.0)
    assert c.deplete(2, 2, 2, 0.0) == 0.0
    assert c.deplete(99, 0, 0, 3.0) == 0.0
    removed = c.deplete(2, 2, 2, 1.0)
    assert removed == 1.0
    assert c.get(2, 2, 2) == 2.0
    snap = c.snapshot()
    assert len(snap) == 4 and len(snap[0]) == 4
    layer = c.layer(0)
    assert len(layer) == 4
    with pytest.raises(ValueError):
        c.layer(99)
    assert c.total_mm() > 0.0


def test_concentration_field3d_diffuse_and_total():
    c = ConcentrationField3D("g", 4, 4, 4, 600.0, 0.0)
    c.set(2, 2, 2, 5.0)
    before = c.total_mm()
    c.diffuse()
    after = c.total_mm()
    assert math.isclose(before, after, rel_tol=1e-6)


def test_concentration_field3d_advect_3d():
    c = ConcentrationField3D("g", 6, 6, 6, 600.0, 0.0)
    c.set(1, 1, 1, 4.0)
    flow = stagnant_3d(6, 6, 6)
    c.advect_3d(flow)
    assert c.get(1, 1, 1) == 4.0


def test_environment_set_flow_3d_routing_and_step():
    cfg = EnvironmentConfig(width=8, height=8)
    e = Environment(cfg)
    flow2 = stagnant(8, 8)
    e.set_flow(flow2)
    assert e.flow is flow2
    flow3 = stagnant_3d(8, 8, 8)
    e.set_flow(flow3)
    assert e.flow3d is flow3
    assert e.flow is None
    e.step()
    assert e.tick == 1


def test_environment_set_flow_3d_dimension_mismatch():
    e = Environment(EnvironmentConfig(width=8, height=8))
    flow3 = stagnant_3d(4, 4, 4)
    with pytest.raises(ValueError):
        e.set_flow(flow3)


def test_environment_add_field_mismatch():
    e = Environment(EnvironmentConfig(width=8, height=8))
    with pytest.raises(ValueError):
        e.add_field("bad", ConcentrationField("x", 4, 4, 600.0))
    good = ConcentrationField("x", 8, 8, 600.0, 1.0)
    e.add_field("x", good)
    assert e.get_field("x") is good
    assert e.substrate_at(0, 0, "x") == 1.0
    assert e.substrate_at(99, 99) == 0.0


def test_environment_add_scalar_and_query():
    e = Environment(EnvironmentConfig(width=4, height=4))
    f = ScalarField("temp", 4, 4, "temperature", 25.0)
    e.add_scalar("temp", f)
    assert e.get_scalar("temp") is f
    assert e.scalar_at("temp", 0, 0) == 25.0
    assert e.scalar_at("temp", 99, 99) == 0.0
    e.step()
    assert e.tick == 1


def test_scalar_field_step_with_forcing_and_diffusion():
    f = ScalarField("temp", 4, 4, "temperature", 0.0,
                    forcing=DiurnalForcing(20.0, 5.0),
                    diffusion_um2_s=600.0)
    f.step(0)
    assert f.get(0, 0) > 0.0
    f.set(1, 1, 100.0)
    f.step(1)
    assert f.mean() > 0.0
    f.add(0, 0, 10.0)
    assert f.get(0, 0) > 0.0
    f.set_all(-5.0)
    assert f.get(0, 0) == -5.0
    f.snapshot()
    assert f._d_lattice() > 0.0


def test_scalar_field_diffuse_single_site():
    f = ScalarField("x", 1, 1, "pH", 0.0, diffusion_um2_s=600.0)
    f._diffuse()
    assert f.get(0, 0) == 0.0


def test_scalar_field_get_set_oob():
    f = ScalarField("x", 3, 3, "pH", 1.0)
    assert f.get(99, 99) == 0.0
    f.set(99, 99, 5.0)
    assert f.get(0, 0) == 1.0
    f.add(99, 99, 1.0)
    assert f.get(0, 0) == 1.0


def test_scalar_field_no_forcing_static_with_diffusion_zero():
    f = ScalarField("x", 3, 3, "pH", 1.0, forcing=None,
                    diffusion_um2_s=0.0)
    f.step(0)
    assert f.get(0, 0) == 1.0


def test_climate_table_interpolation_midpoint():
    ct = ClimateTable([0, 10, 20], [0.0, 10.0, 20.0])
    assert ct(5) == 5.0
    assert ct(-1) == 0.0
    assert ct(25) == 20.0
    assert ct(10) == 10.0
    with pytest.raises(ValueError):
        ClimateTable([], [])
    with pytest.raises(ValueError):
        ClimateTable([1, 2], [1.0])


def test_environment_local_uptake_branches():
    e = Environment(EnvironmentConfig(width=3, height=3,
                                      glucose_initial_mm=1.0,
                                      oxygen_initial_mm=0.2))
    g = e.local_uptake(1, 1, "glucose")
    o = e.local_uptake(1, 1, "oxygen")
    custom_field = ConcentrationField("custom", 3, 3, 600.0, 1.0)
    e.add_field("custom", custom_field)
    other = e.local_uptake(1, 1, "custom", v_max=2.0)
    assert g > 0.0 and o > 0.0 and other > 0.0
    explicit = e.local_uptake(1, 1, "glucose", half_saturation=0.5)
    assert explicit > 0.0


def test_environment_replenish_skips_unknown_field():
    cfg = EnvironmentConfig(width=4, height=4, flow_rate=0.5,
                            bulk_glucose_mm=2.0, bulk_oxygen_mm=0.4)
    e = Environment(cfg)
    e.add_field("custom", ConcentrationField("custom", 4, 4, 600.0, 0.0))
    e.step()
    assert e.glucose.get(0, 0) > 1.0


def test_environment_invalid_flow_rate():
    cfg = EnvironmentConfig(width=4, height=4, flow_rate=2.0)
    with pytest.raises(ValueError):
        Environment(cfg)


def test_environment_step_flow2d_and_scalars():
    e = Environment(EnvironmentConfig(width=6, height=6))
    e.set_flow(stagnant(6, 6))
    f = ScalarField("light", 6, 6, "light", 0.0,
                    forcing=DiurnalForcing(100.0, 50.0))
    e.add_scalar("light", f)
    e.step()
    assert e.tick == 1
    assert e.scalar_at("light", 0, 0) > 0.0


def test_flow_upwind_pure_python(monkeypatch):
    monkeypatch.setattr(env, "_HAS_NUMPY", False)
    c = ConcentrationField("g", 5, 5, 600.0, 1.0)
    c.set(0, 0, 3.0)
    flow = channel_poiseuille(5, 5, 100.0, "E")
    c.advect(flow)
    assert c.total_mm() > 0.0


def test_flow_laplacian_pure_python(monkeypatch):
    monkeypatch.setattr(env, "_HAS_NUMPY", False)
    c = ConcentrationField("g", 5, 5, 600.0, 0.0)
    c.set(2, 2, 5.0)
    c.diffuse()
    assert c.get(2, 2) >= 0.0


def test_flow_laplacian_3d_pure_python(monkeypatch):
    monkeypatch.setattr(env, "_HAS_NUMPY", False)
    c = ConcentrationField3D("g", 4, 4, 4, 600.0, 0.0)
    c.set(2, 2, 2, 5.0)
    c.diffuse()
    assert c.get(2, 2, 2) >= 0.0


def test_flow_upwind_3d_pure_python(monkeypatch):
    monkeypatch.setattr(env, "_HAS_NUMPY", False)
    c = ConcentrationField3D("g", 6, 6, 6, 600.0, 0.0)
    c.set(1, 1, 1, 4.0)
    flow = channel_poiseuille_3d(6, 6, 6, 100.0, "E")
    c.advect_3d(flow)
    assert c.total_mm() > 0.0


def test_molecules_per_site_positive():
    assert molecules_per_site(0.0) == 0.0
    assert molecules_per_site(1.0) > 0.0


def test_concentration_field_invalid_dimensions():
    with pytest.raises(ValueError):
        ConcentrationField("g", 0, 3, 600.0)
    with pytest.raises(ValueError):
        ConcentrationField("g", 3, -1, 600.0)


def test_environment_set_flow_2d_dimension_mismatch():
    e = Environment(EnvironmentConfig(width=8, height=8))
    with pytest.raises(ValueError):
        e.set_flow(stagnant(4, 4))


def test_concentration_field3d_diffuse_zero_d():
    c = ConcentrationField3D("g", 3, 3, 3, 0.0, 2.0)
    c.diffuse()
    assert c.get(0, 0, 0) == 2.0


def test_climate_table_loop_fallback_single():
    ct = ClimateTable([5], [7.0])
    assert ct(3) == 7.0
    assert ct(10) == 7.0


def test_advect_numpy_east_flow(monkeypatch):
    monkeypatch.setattr(env, "_HAS_NUMPY", True)
    c = ConcentrationField("g", 6, 6, 600.0, 0.0)
    c.set(1, 1, 4.0)
    flow = channel_poiseuille(6, 6, 100.0, "E")
    c.advect(flow)
    assert c.total_mm() > 0.0


def test_advect_numpy_west_north_south_flow(monkeypatch):
    monkeypatch.setattr(env, "_HAS_NUMPY", True)
    for direction in ("W", "N", "S"):
        c = ConcentrationField("g", 6, 6, 600.0, 0.0)
        c.set(3, 3, 4.0)
        flow = channel_poiseuille(6, 6, 100.0, direction)
        c.advect(flow)
        assert c.total_mm() > 0.0


def test_advect_3d_numpy_all_directions(monkeypatch):
    monkeypatch.setattr(env, "_HAS_NUMPY", True)
    for direction in ("E", "W", "N", "S", "U", "D"):
        c = ConcentrationField3D("g", 6, 6, 6, 600.0, 0.0)
        c.set(3, 3, 3, 4.0)
        flow = channel_poiseuille_3d(6, 6, 6, 100.0, direction)
        c.advect_3d(flow)
        assert c.total_mm() > 0.0


def test_flow_upwind_pure_python_negative_velocities(monkeypatch):
    monkeypatch.setattr(env, "_HAS_NUMPY", False)
    c = ConcentrationField("g", 5, 5, 600.0, 1.0)
    flow = channel_poiseuille(5, 5, 100.0, "W")
    c.advect(flow)
    assert c.total_mm() > 0.0


def test_flow_upwind_pure_python_north_south(monkeypatch):
    monkeypatch.setattr(env, "_HAS_NUMPY", False)
    for direction in ("N", "S"):
        c = ConcentrationField("g", 5, 5, 600.0, 1.0)
        c.set(2, 2, 4.0)
        flow = channel_poiseuille(5, 5, 100.0, direction)
        c.advect(flow)
        assert c.total_mm() > 0.0


def test_concentration_field3d_invalid_depth():
    with pytest.raises(ValueError):
        ConcentrationField3D("g", 1, 1, 0, 600.0)


def test_scalar_field_diffuse_zero_d():
    f = ScalarField("x", 3, 3, "pH", 1.0, diffusion_um2_s=0.0)
    f._diffuse()
    assert f.get(0, 0) == 1.0


def test_environment_add_scalar_dimension_mismatch():
    e = Environment(EnvironmentConfig(width=4, height=4))
    with pytest.raises(ValueError):
        e.add_scalar("temp", ScalarField("t", 2, 2, "temperature"))


def test_environment_step_flow3d_with_3d_field(monkeypatch):
    monkeypatch.setattr(env, "_HAS_NUMPY", True)
    e = Environment(EnvironmentConfig(width=6, height=6))
    e.flow3d = stagnant_3d(6, 6, 6)
    e.flow = None
    field3d = ConcentrationField3D("g3d", 6, 6, 6, 600.0, 0.0)
    e.add_field("g3d", field3d)
    e.step()
    assert e.tick == 1
