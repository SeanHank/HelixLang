"""GRN unit tests."""
import pytest

from helixlang.grn import GRN, sigmoid


def test_sigmoid_basic():
    assert sigmoid(0) == pytest.approx(0.5)
    assert sigmoid(100) > 0.99
    assert sigmoid(-100) < 0.01


def test_sigmoid_stable():
    """Large inputs should not overflow."""
    assert 0.0 <= sigmoid(1000) <= 1.0
    assert 0.0 <= sigmoid(-1000) <= 1.0


def test_constitutive_gene_triggered():
    """No input + negative threshold -> constitutive expression."""
    grn = GRN()
    grn.add_gene("g", threshold=-1.0, initial_level=1.0)
    triggered = grn.step()
    assert "g" in triggered


def test_repressed_gene_not_triggered():
    """Strong inhibitory input -> not triggered."""
    grn = GRN()
    grn.add_gene("src", threshold=-1.0, initial_level=1.0)
    grn.add_gene("tgt", threshold=0.3, initial_level=0.0)
    grn.add_edge("src", "tgt", -0.9)
    # After many ticks, tgt should still not exceed 0.5
    for _ in range(20):
        grn.step()
    assert grn.nodes["tgt"].level < 0.5


def test_activated_gene_triggered():
    """Strong activating input -> triggered."""
    grn = GRN()
    grn.add_gene("src", threshold=-1.0, initial_level=1.0)
    grn.add_gene("tgt", threshold=0.3, initial_level=0.0)
    grn.add_edge("src", "tgt", 0.9)
    for _ in range(20):
        grn.step()
    assert grn.nodes["tgt"].level > 0.5


def test_toggle_switch_converges():
    """Two mutually inhibitory genes -> converge to steady state (a symmetric
    point under a simple sigmoid; true bistability needs Hill cooperativity,
    which is beyond the prototype's scope)."""
    grn = GRN()
    grn.add_gene("ci", threshold=0.0, initial_level=0.8)
    grn.add_gene("cro", threshold=0.0, initial_level=0.2)
    grn.add_edge("ci", "cro", -1.0)
    grn.add_edge("cro", "ci", -1.0)
    levels = []
    for _ in range(50):
        grn.step()
        levels.append((grn.nodes["ci"].level, grn.nodes["cro"].level))
    # Level changes over the last 10 ticks should be < 0.01 (convergence)
    for i in range(-10, -1):
        d_ci = abs(levels[i + 1][0] - levels[i][0])
        d_cro = abs(levels[i + 1][1] - levels[i][1])
        assert d_ci < 0.01
        assert d_cro < 0.01
    # Both genes should be in [0, 1]
    assert 0.0 <= grn.nodes["ci"].level <= 1.0
    assert 0.0 <= grn.nodes["cro"].level <= 1.0


def test_level_clamped():
    grn = GRN()
    grn.add_gene("g", threshold=0.0, initial_level=0.0)
    # Should not exceed [0, 1]
    for _ in range(100):
        grn.step()
    assert 0.0 <= grn.nodes["g"].level <= 1.0
