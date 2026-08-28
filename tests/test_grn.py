"""GRN unit tests."""
import pytest

from helixlang.plugins.runtime.grn import GRN, decay_from_half_life_ticks, hill, sigmoid


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
    # Physical decay (~0.994/tick) converges slowly; 500 ticks >> 110-min
    # half-life, so tgt reaches its ~0.6 steady state.
    for _ in range(500):
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


def test_hill_activation():
    """Hill kinetics: half-max at x = kd, 0 below, saturation above."""
    assert hill(0.0, 2, 1.0) == 0.0
    assert hill(-1.0, 2, 1.0) == 0.0
    assert hill(1.0, 2, 1.0) == pytest.approx(0.5)
    assert hill(3.0, 2, 1.0) == pytest.approx(9.0 / 10.0)
    assert hill(10.0, 2, 1.0) > 0.99


def test_hill_half_max_at_kd():
    """Hill activation is 0.5 when input == kd."""
    assert hill(5.0, 4, 5.0) == pytest.approx(0.5)
    assert hill(0.1, 1, 0.1) == pytest.approx(0.5)


def test_decay_from_half_life():
    """decay_from_half_life_ticks halves the level at its half-life."""
    decay = decay_from_half_life_ticks(10.0)
    assert decay == pytest.approx(0.5 ** 0.1)
    # after half_life_ticks ticks, level * decay**n == level/2
    level = 0.8
    for _ in range(10):
        level *= decay
    assert level == pytest.approx(0.4)
    # 1 tick half-life -> decay 0.5
    assert decay_from_half_life_ticks(1.0) == pytest.approx(0.5)
    # long half-life -> decay close to 1 (slow degradation)
    assert decay_from_half_life_ticks(110.0) == pytest.approx(0.9937, abs=1e-3)
    with pytest.raises(ValueError):
        decay_from_half_life_ticks(0.0)


def test_per_gene_decay_used():
    """Per-gene decay overrides the universal DECAY."""
    grn = GRN()
    # gene with per-gene decay 0.5 (fast decay) vs default 0.7
    grn.add_gene("fast", threshold=-1.0, initial_level=1.0, decay=0.5)
    grn.add_gene("slow", threshold=-1.0, initial_level=1.0)
    for _ in range(3):
        grn.step()
    assert grn.nodes["fast"].level < grn.nodes["slow"].level


def test_hill_gene_kinetics_steady_state():
    """Hill path: steady-state expression matches Hill prediction."""
    grn = GRN()
    # constitutive source held at level 1 (no decay)
    grn.add_gene("src", threshold=-1.0, initial_level=1.0, decay=1.0)
    # target with Hill kinetics: kd=0.5, n=3
    grn.add_gene("tgt", threshold=0.0, initial_level=0.0, hill_n=3, kd=0.5)
    grn.add_edge("src", "tgt", 1.0)
    for _ in range(700):
        grn.step()
    # input = 1.0 -> activation = 1^3/(0.5^3 + 1^3) = 0.8889
    act = hill(1.0, 3, 0.5)
    assert act == pytest.approx(1.0 / (0.125 + 1.0))
    # steady state level = activation (decay absorbs)
    assert grn.nodes["tgt"].level == pytest.approx(act, abs=0.02)
    assert grn.nodes["tgt"].level > 0.5


# ------------------------------------------------------------------ #
# Physical units (calibrated decay is now the universal default)
# ------------------------------------------------------------------ #
def _pure_decay_grn():
    """A GRN whose single gene never activates (pure exponential decay)."""
    grn = GRN()
    grn.add_gene("g", threshold=1e3, initial_level=1.0)
    return grn


def test_default_decay_comes_from_110_min_half_life():
    """The universal DECAY is derived from the 110-min E. coli protein
    half-life (Mosteller 1980, Helbig 2011)."""
    assert GRN().DECAY == pytest.approx(decay_from_half_life_ticks(110.0))


def test_default_decay_halves_at_110_ticks():
    """One tick = one minute -> level halves after ~110 ticks."""
    grn = _pure_decay_grn()
    for _ in range(110):
        grn.step()
    assert grn.nodes["g"].level == pytest.approx(0.5, abs=0.02)


def test_default_decay_slow_over_first_ticks():
    """Calibrated decay is ~0.994/tick, so very little is lost in one tick."""
    grn = _pure_decay_grn()
    grn.step()
    assert grn.nodes["g"].level == pytest.approx(
        decay_from_half_life_ticks(110.0), abs=1e-4)


def test_per_gene_decay_overrides_universal_default():
    """An explicit per-gene decay= wins over the universal default."""
    grn = GRN()
    grn.add_gene("g", threshold=1e3, initial_level=1.0, decay=0.5)
    for _ in range(3):
        grn.step()
    assert grn.nodes["g"].level == pytest.approx(0.5 ** 3)
