"""Gray-Scott reaction-diffusion unit tests."""
import numpy as np
import pytest

from helixlang import reaction_diffusion
from helixlang.reaction_diffusion import GrayScott


def test_init():
    gs = GrayScott(n=16)
    assert gs.n == 16
    assert len(gs.u) == 16
    assert len(gs.v) == 16
    # Initially, most of U is 1.0
    assert gs.u[0][0] == pytest.approx(1.0)
    # Initially, most of V is 0.0
    assert gs.v[0][0] == pytest.approx(0.0)


def test_step_changes_field():
    gs = GrayScott(n=16)
    v_before = sum(sum(row) for row in gs.v)
    gs.step()
    v_after = sum(sum(row) for row in gs.v)
    # After one step, total V should change
    assert v_after != pytest.approx(v_before)


def test_values_clamped():
    gs = GrayScott(n=16)
    for _ in range(50):
        gs.step()
    for row in gs.u:
        for v in row:
            assert 0.0 <= v <= 1.0
    for row in gs.v:
        for v in row:
            assert 0.0 <= v <= 1.0


def test_emit():
    gs = GrayScott(n=16)
    gs.emit(5, 5, 0.5)
    assert gs.v[5][5] >= 0.5


def test_emit_out_of_bounds():
    """Out-of-bounds emit should be safely ignored."""
    gs = GrayScott(n=16)
    gs.emit(-1, 100, 1.0)  # should not raise an exception
    gs.emit(100, -1, 1.0)


def test_total_v():
    gs = GrayScott(n=16)
    assert gs.total_v() > 0  # seed injected V


def test_numpy_backend_used_when_available():
    """The vectorized backend is active when numpy is installed."""
    gs = GrayScott(n=16)
    if reaction_diffusion._HAS_NUMPY:
        assert isinstance(gs.u, np.ndarray)
    else:
        assert isinstance(gs.u, list)


def test_pure_python_backend_matches_numpy(monkeypatch):
    """Pure-Python scratch fallback must produce identical fields to numpy."""
    gs_np = GrayScott(n=16, seed=3)  # numpy backend (field = ndarray)
    monkeypatch.setattr(reaction_diffusion, "_HAS_NUMPY", False)
    gs_py = GrayScott(n=16, seed=3)  # pure-Python fallback (field = lists)
    assert not isinstance(gs_py.u, np.ndarray)
    assert isinstance(gs_py.v, list)
    for _ in range(50):
        gs_py.step()
        gs_np.step()
    assert np.array_equal(gs_py.u, gs_np.u)
    assert np.array_equal(gs_py.v, gs_np.v)


def test_pure_python_backend_matches_reference(monkeypatch):
    """Pure-Python path reproduces the classic copy-based algorithm exactly."""
    def reference_step(gs: GrayScott) -> None:
        n = gs.n
        nu = [row[:] for row in gs.u]
        nv = [row[:] for row in gs.v]
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                lu = (gs.u[i - 1][j] + gs.u[i + 1][j] + gs.u[i][j - 1]
                      + gs.u[i][j + 1] - 4.0 * gs.u[i][j]) * 0.25
                lv = (gs.v[i - 1][j] + gs.v[i + 1][j] + gs.v[i][j - 1]
                      + gs.v[i][j + 1] - 4.0 * gs.v[i][j]) * 0.25
                uij = gs.u[i][j]
                vij = gs.v[i][j]
                uvv = uij * vij * vij
                nu[i][j] = uij + (gs.Du * lu - uvv + gs.F * (1 - uij))
                nv[i][j] = vij + (gs.Dv * lv + uvv - (gs.F + gs.k) * vij)
        for i in range(n):
            for j in range(n):
                nu[i][j] = min(1.0, max(0.0, nu[i][j]))
                nv[i][j] = min(1.0, max(0.0, nv[i][j]))
        gs.u, gs.v = nu, nv

    monkeypatch.setattr(reaction_diffusion, "_HAS_NUMPY", False)
    gs = GrayScott(n=16, seed=5)
    ref = GrayScott(n=16, seed=5)
    for _ in range(40):
        gs.step()
        reference_step(ref)
    assert gs.u == ref.u and gs.v == ref.v


def test_emit_survives_buffer_swap(monkeypatch):
    """Border injections must not be lost across scratch-buffer swaps."""
    monkeypatch.setattr(reaction_diffusion, "_HAS_NUMPY", False)
    gs = GrayScott(n=16, seed=1)
    gs.emit(0, 0, 0.9)
    gs.emit(15, 15, 0.7)
    gs.emit(7, 0, 0.5)
    for _ in range(8):
        gs.step()
    assert gs.v[0][0] >= 0.9
    assert gs.v[15][15] >= 0.7
    assert gs.v[7][0] >= 0.5
