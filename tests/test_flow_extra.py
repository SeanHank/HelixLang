"""Flow field additional-coverage tests (3D fields, validation, numpy-free paths)."""
import pytest

import helixlang.plugins.runtime.flow as flow
from helixlang.plugins.runtime.flow import (
    FlowField,
    FlowField3D,
    channel_poiseuille_3d,
    stagnant_3d,
    um_s_to_sites_per_tick,
)


def _zeros3d(width, height, depth):
    return [[[0.0] * width for _ in range(height)] for _ in range(depth)]


def test_flow_field_arrays_no_numpy(monkeypatch):
    monkeypatch.setattr(flow, "_HAS_NUMPY", False)
    f = FlowField(3, 3, [[0.0] * 3] * 3, [[0.0] * 3] * 3)
    assert f.arrays() == (None, None)


def test_flow_field3d_invalid_shapes():
    _z = _zeros3d
    with pytest.raises(ValueError):
        FlowField3D(3, 3, 3, _z(3, 3, 3), _z(3, 3, 3), _z(3, 3, 2))
    with pytest.raises(ValueError):
        FlowField3D(3, 3, 3, _z(3, 2, 3), _z(3, 3, 3), _z(3, 3, 3))
    with pytest.raises(ValueError):
        FlowField3D(3, 3, 3, _z(4, 3, 3), _z(3, 3, 3), _z(3, 3, 3))


def test_flow_field3d_velocity_out_of_bounds():
    _z = _zeros3d
    f = FlowField3D(3, 3, 3, _z(3, 3, 3), _z(3, 3, 3), _z(3, 3, 3))
    assert f.velocity(0, 0, 0) == (0.0, 0.0, 0.0)
    assert f.velocity(99, 0, 0) == (0.0, 0.0, 0.0)
    assert f.velocity(0, 5, 0) == (0.0, 0.0, 0.0)
    assert f.velocity(0, 0, 9) == (0.0, 0.0, 0.0)


def test_flow_field3d_max_magnitude():
    _z = _zeros3d
    u = _z(3, 3, 3)
    u[0][0][0] = 3.0
    v = _z(3, 3, 3)
    v[0][0][0] = 4.0
    f = FlowField3D(3, 3, 3, u, v, _z(3, 3, 3))
    assert f.max_magnitude() == 5.0


def test_flow_field3d_arrays_no_numpy(monkeypatch):
    monkeypatch.setattr(flow, "_HAS_NUMPY", False)
    f = stagnant_3d(3, 3, 3)
    assert f.arrays() == (None, None, None)


def test_stagnant_3d_zero():
    f = stagnant_3d(4, 4, 4)
    assert f.max_magnitude() == 0.0


def test_channel_poiseuille_3d_invalid_args():
    with pytest.raises(ValueError):
        channel_poiseuille_3d(0, 4, 4, 10.0)
    with pytest.raises(ValueError):
        channel_poiseuille_3d(4, -1, 4, 10.0)
    with pytest.raises(ValueError):
        channel_poiseuille_3d(4, 4, 0, 10.0)
    with pytest.raises(ValueError):
        channel_poiseuille_3d(4, 4, 4, -5.0)
    with pytest.raises(ValueError):
        channel_poiseuille_3d(4, 4, 4, 10.0, direction="Z")


def test_channel_poiseuille_3d_directions():
    for direction in ("E", "W", "N", "S", "U", "D"):
        f = channel_poiseuille_3d(6, 6, 6, 50.0, direction)
        assert f.max_magnitude() > 0.0
    assert channel_poiseuille_3d(6, 6, 6, 0.0).max_magnitude() == 0.0


def test_duct_profile_pure_python(monkeypatch):
    monkeypatch.setattr(flow, "_HAS_NUMPY", False)
    f = channel_poiseuille_3d(6, 6, 6, 50.0, "E")
    assert f.max_magnitude() > 0.0
    f2 = channel_poiseuille_3d(6, 6, 6, 50.0, "N")
    assert f2.max_magnitude() > 0.0
    f3 = channel_poiseuille_3d(6, 6, 6, 50.0, "U")
    assert f3.max_magnitude() > 0.0


def test_scale_to_mean_empty_or_zero_profile():
    assert flow._scale_to_mean([], 5.0) == []
    assert flow._scale_to_mean([[0.0, 0.0], [0.0, 0.0]], 5.0) == [
        [0.0, 0.0], [0.0, 0.0]]
    scaled = flow._scale_to_mean([[1.0, 2.0], [3.0, 4.0]], 10.0)
    assert scaled[0][0] != 1.0


def test_um_s_conversion_sanity():
    assert um_s_to_sites_per_tick(0.0) == 0.0
    assert um_s_to_sites_per_tick(600.0) > 0.0
