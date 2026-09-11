"""Additional coverage for Gray-Scott reaction-diffusion (from_preset, _lap,
pure-python clamp branches)."""
import pytest

import helixlang.plugins.runtime.reaction_diffusion as rd
from helixlang.plugins.runtime.reaction_diffusion import GrayScott


def test_from_preset():
    gs = GrayScott.from_preset("mitosis", n=12)
    assert gs.n == 12
    with pytest.raises(ValueError):
        GrayScott.from_preset("not-a-preset")


def test_seed_small_n_guard():
    gs = GrayScott(n=5, seed=1)
    assert gs.n == 5


def test_lap_static():
    field = [[0.0, 1.0, 0.0],
             [1.0, 5.0, 1.0],
             [0.0, 1.0, 0.0]]
    # neighbors sum = 1+1+1+1 = 4, center 5 -> (4 - 5*4)*0.25 = -4.0
    assert GrayScott._lap(field, 1, 1) == (4.0 - 20.0) * 0.25


def test_step_py_clamps_low_u(monkeypatch):
    monkeypatch.setattr(rd, "_HAS_NUMPY", False)
    gs = GrayScott(n=6, seed=1)
    gs.u[2][2] = -0.5
    gs.step()
    assert gs.u[2][2] >= 0.0
    assert all(0.0 <= v <= 1.0 for row in gs.u for v in row)


def test_step_py_clamps_high_u(monkeypatch):
    monkeypatch.setattr(rd, "_HAS_NUMPY", False)
    gs = GrayScott(n=6, seed=1)
    # force a large positive Laplacian at the centre (low centre, high
    # neighbours) so the updated value exceeds 1.0 and hits the ceiling clamp
    for i in range(6):
        for j in range(6):
            gs.u[i][j] = 0.9
            gs.v[i][j] = 0.0
    gs.u[3][3] = 0.1
    for (i, j) in [(2, 3), (4, 3), (3, 2), (3, 4)]:
        gs.u[i][j] = 10.0
    gs.step()
    assert gs.u[3][3] <= 1.0
    assert gs.u[3][3] >= 0.0


def test_step_py_clamps_low_v(monkeypatch):
    monkeypatch.setattr(rd, "_HAS_NUMPY", False)
    gs = GrayScott(n=6, seed=1)
    gs.v[2][2] = -0.9
    gs.step()
    assert gs.v[2][2] >= 0.0


def test_step_py_clamps_high_v(monkeypatch):
    monkeypatch.setattr(rd, "_HAS_NUMPY", False)
    gs = GrayScott(n=6, seed=1)
    gs.v[2][2] = 9.0
    gs.step()
    assert gs.v[2][2] <= 1.0
