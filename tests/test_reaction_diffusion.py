"""Gray-Scott reaction-diffusion unit tests."""
import pytest

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
