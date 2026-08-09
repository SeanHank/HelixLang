"""Cell dataclass unit tests.

Covers src/helixlang/cell.py: construction, energy, division, protein storage,
slots, color, morphology_points, and alive/dead state transitions.
"""
from __future__ import annotations

import pytest

from helixlang.cell import (
    DEFAULT_MEMBRANE_PERMEABILITY,
    DIRECTIONS,
    INITIAL_CELL_ENERGY,
    MAX_MEMBRANE_PERMEABILITY,
    MOVE_ENERGY_COST,
    Cell,
)
from helixlang.units import energy_to_atp

# ============================================================================
# Construction and default values
# ============================================================================

class TestCellConstruction:
    """Verify Cell construction and default values."""

    def test_default_construction(self):
        c = Cell()
        assert c.name == "cell-0"
        assert c.x == 0
        assert c.y == 0
        assert c.energy == 100
        assert c.proteins == {}
        assert c.alive is True
        assert c.color == (255, 255, 255)
        assert c.age == 0
        assert c.divisions == 0
        assert c.morphology_points == [(0.0, 0.0)]
        assert c.membrane_permeability == DEFAULT_MEMBRANE_PERMEABILITY
        assert c.membrane_permeability == MAX_MEMBRANE_PERMEABILITY

    def test_default_slots_length(self):
        c = Cell()
        assert len(c.slots) == 256
        assert all(s is None for s in c.slots)

    def test_custom_construction(self):
        c = Cell(name="mycell", x=3, y=7, energy=50,
                 color=(10, 20, 30))
        assert c.name == "mycell"
        assert c.x == 3
        assert c.y == 7
        assert c.energy == 50
        assert c.color == (10, 20, 30)

    def test_slots_independent_between_instances(self):
        a = Cell()
        b = Cell()
        a.slots[0] = "X"
        assert b.slots[0] is None

    def test_proteins_independent_between_instances(self):
        a = Cell()
        b = Cell()
        a.add_protein(1)
        assert 1 in a.proteins
        assert 1 not in b.proteins

    def test_morphology_points_independent(self):
        a = Cell()
        b = Cell()
        a.morphology_points.append((1.0, 1.0))
        assert b.morphology_points == [(0.0, 0.0)]


# ============================================================================
# Protein storage
# ============================================================================

class TestProteinStorage:
    """Verify add_protein / consume_protein."""

    def test_add_protein_default_amount(self):
        c = Cell()
        c.add_protein(3)
        assert c.proteins[3] == 1.0

    def test_add_protein_custom_amount(self):
        c = Cell()
        c.add_protein(3, amount=2.5)
        assert c.proteins[3] == 2.5

    def test_add_protein_accumulates(self):
        c = Cell()
        c.add_protein(3, amount=1.0)
        c.add_protein(3, amount=0.5)
        assert c.proteins[3] == pytest.approx(1.5)

    def test_add_protein_different_kinds(self):
        c = Cell()
        c.add_protein(1)
        c.add_protein(2)
        assert set(c.proteins.keys()) == {1, 2}

    def test_consume_protein_partial(self):
        c = Cell()
        c.add_protein(3, amount=2.0)
        consumed = c.consume_protein(3, amount=0.5)
        assert consumed == pytest.approx(0.5)
        assert c.proteins[3] == pytest.approx(1.5)

    def test_consume_protein_exact(self):
        c = Cell()
        c.add_protein(3, amount=1.0)
        consumed = c.consume_protein(3, amount=1.0)
        assert consumed == pytest.approx(1.0)
        assert 3 not in c.proteins  # 0 -> removed

    def test_consume_protein_more_than_available(self):
        c = Cell()
        c.add_protein(3, amount=0.3)
        consumed = c.consume_protein(3, amount=1.0)
        assert consumed == pytest.approx(0.3)
        assert 3 not in c.proteins

    def test_consume_protein_missing_kind(self):
        c = Cell()
        consumed = c.consume_protein(99, amount=1.0)
        assert consumed == 0.0
        assert 99 not in c.proteins

    def test_consume_protein_default_amount(self):
        c = Cell()
        c.add_protein(3, amount=2.0)
        consumed = c.consume_protein(3)
        assert consumed == pytest.approx(1.0)
        assert c.proteins[3] == pytest.approx(1.0)


# ============================================================================
# Energy and movement
# ============================================================================

class TestEnergyAndMove:
    """Verify move / consume_energy / feed."""

    def test_move_north(self):
        c = Cell()
        c.move(0)  # N = (0, -1)
        assert c.x == 0
        assert c.y == -1
        assert c.energy == 99

    def test_move_east(self):
        c = Cell()
        c.move(1)  # E = (1, 0)
        assert c.x == 1
        assert c.y == 0
        assert c.energy == 99

    def test_move_south(self):
        c = Cell()
        c.move(2)  # S = (0, 1)
        assert c.x == 0
        assert c.y == 1
        assert c.energy == 99

    def test_move_west(self):
        c = Cell()
        c.move(3)  # W = (-1, 0)
        assert c.x == -1
        assert c.y == 0
        assert c.energy == 99

    def test_move_wraps_direction_modulo_4(self):
        c = Cell()
        c.move(4)  # 4 % 4 = 0 -> N
        assert c.x == 0
        assert c.y == -1
        c.move(7)  # 7 % 4 = 3 -> W
        assert c.x == -1

    def test_move_zero_energy_no_decrement(self):
        c = Cell(energy=0)
        c.move(0)
        assert c.x == 0
        assert c.y == -1
        assert c.energy == 0  # no underflow

    def test_move_decrements_each_step(self):
        c = Cell(energy=10)
        for _ in range(5):
            c.move(1)
        assert c.x == 5
        assert c.energy == 5

    def test_consume_energy_success(self):
        c = Cell(energy=100)
        assert c.consume_energy(30) is True
        assert c.energy == 70

    def test_consume_energy_insufficient(self):
        c = Cell(energy=5)
        assert c.consume_energy(10) is False
        assert c.energy == 5  # unchanged

    def test_consume_energy_exact(self):
        c = Cell(energy=10)
        assert c.consume_energy(10) is True
        assert c.energy == 0

    def test_consume_energy_default_n(self):
        c = Cell(energy=5)
        assert c.consume_energy() is True
        assert c.energy == 4

    def test_feed_default_amount(self):
        c = Cell(energy=50)
        c.feed()
        assert c.energy == 60

    def test_feed_custom_amount(self):
        c = Cell(energy=50)
        c.feed(25)
        assert c.energy == 75

    def test_feed_can_exceed_default(self):
        c = Cell(energy=100)
        c.feed(50)
        assert c.energy == 150


# ============================================================================
# Membrane permeability
# ============================================================================

class TestMembranePermeability:
    """Verify the membrane permeability model scales nutrient intake."""

    def test_set_permeability(self):
        c = Cell()
        c.set_membrane_permeability(200)
        assert c.membrane_permeability == 200

    def test_set_permeability_clamps_low(self):
        c = Cell()
        c.set_membrane_permeability(-10)
        assert c.membrane_permeability == 0

    def test_set_permeability_clamps_high(self):
        c = Cell()
        c.set_membrane_permeability(999)
        assert c.membrane_permeability == MAX_MEMBRANE_PERMEABILITY

    def test_feed_impermeable_gains_nothing(self):
        c = Cell(energy=50)
        c.set_membrane_permeability(0)
        c.feed(10)
        assert c.energy == 50

    def test_feed_fully_permeable_gains_full_amount(self):
        c = Cell(energy=50)
        c.feed(10)
        assert c.energy == 60

    def test_feed_half_permeable_gains_half(self):
        c = Cell(energy=50)
        c.set_membrane_permeability(128)
        c.feed(10)
        assert c.energy == 50 + round(10 * 128 / MAX_MEMBRANE_PERMEABILITY)

    def test_feed_scales_with_permeability(self):
        low, high = Cell(energy=0), Cell(energy=0)
        low.set_membrane_permeability(64)
        high.set_membrane_permeability(255)
        low.feed(20)
        high.feed(20)
        assert low.energy < high.energy


# ============================================================================
# Division
# ============================================================================

class TestDivision:
    """Verify the divide threshold and behavior."""

    def test_divide_success_halves_energy(self):
        c = Cell(energy=100)
        assert c.divide() is True
        assert c.energy == 50  # 100 // 2
        assert c.divisions == 1

    def test_divide_odd_energy_floors(self):
        c = Cell(energy=99)
        assert c.divide() is True
        assert c.energy == 49  # 99 // 2
        assert c.divisions == 1

    def test_divide_threshold_energy_2(self):
        c = Cell(energy=2)
        assert c.divide() is True
        assert c.energy == 1
        assert c.divisions == 1

    def test_divide_fails_below_threshold(self):
        c = Cell(energy=1)
        assert c.divide() is False
        assert c.energy == 1
        assert c.divisions == 0

    def test_divide_fails_at_zero(self):
        c = Cell(energy=0)
        assert c.divide() is False
        assert c.energy == 0
        assert c.divisions == 0

    def test_multiple_divisions_increment_counter(self):
        c = Cell(energy=100)
        c.divide()
        c.divide()
        c.divide()
        assert c.divisions == 3
        # 100 -> 50 -> 25 -> 12
        assert c.energy == 12

    def test_division_chain_until_below_threshold(self):
        c = Cell(energy=100)
        count = 0
        while c.divide():
            count += 1
        # 100->50->25->12->6->3->1(fail) -> 6 successes
        assert count == 6
        assert c.energy == 1


# ============================================================================
# Alive/dead state
# ============================================================================

class TestAliveDead:
    """Verify alive/dead state transitions."""

    def test_default_alive(self):
        c = Cell()
        assert c.alive is True

    def test_die_sets_alive_false(self):
        c = Cell()
        c.die()
        assert c.alive is False

    def test_die_idempotent(self):
        c = Cell()
        c.die()
        c.die()
        assert c.alive is False

    def test_dead_cell_can_still_manipulate_state(self):
        # die() only flips a flag and does not lock other operations
        c = Cell()
        c.die()
        c.add_protein(1)
        c.feed(5)
        assert c.proteins[1] == 1.0
        assert c.energy == 105


# ============================================================================
# Morphology, color, slots
# ============================================================================

class TestMorphologyColorSlots:
    """Verify morphology_points / color / slots."""

    def test_default_morphology_has_origin(self):
        c = Cell()
        assert c.morphology_points == [(0.0, 0.0)]

    def test_morphology_points_mutable(self):
        c = Cell()
        c.morphology_points.append((1.0, 2.0))
        assert (1.0, 2.0) in c.morphology_points

    def test_default_color_white(self):
        c = Cell()
        assert c.color == (255, 255, 255)

    def test_color_mutable(self):
        c = Cell()
        c.color = (10, 20, 30)
        assert c.color == (10, 20, 30)

    def test_slots_default_none(self):
        c = Cell()
        for i in range(256):
            assert c.slots[i] is None

    def test_slots_writable(self):
        c = Cell()
        c.slots[0] = "value"
        c.slots[255] = 42
        assert c.slots[0] == "value"
        assert c.slots[255] == 42

    def test_slots_index_out_of_range_raises(self):
        c = Cell()
        with pytest.raises(IndexError):
            _ = c.slots[256]


# ============================================================================
# dump
# ============================================================================

class TestDump:
    """Verify the dump string."""

    def test_dump_returns_string(self):
        c = Cell()
        s = c.dump()
        assert isinstance(s, str)
        assert "Cell(" in s
        assert "cell-0" in s
        assert "energy=100" in s
        assert "alive=True" in s

    def test_dump_reflects_state(self):
        c = Cell(name="abc", x=5, y=-3, energy=42)
        c.die()
        s = c.dump()
        assert "abc" in s
        assert "pos=(5,-3)" in s
        assert "energy=42" in s
        assert "alive=False" in s

    def test_dump_includes_proteins(self):
        c = Cell()
        c.add_protein(7, amount=2.0)
        s = c.dump()
        assert "proteins=" in s
        assert "7" in s


# ============================================================================
# DIRECTIONS constant
# ============================================================================

class TestDirectionsConstant:
    """Verify the DIRECTIONS constant."""

    def test_directions_count(self):
        assert len(DIRECTIONS) == 4

    def test_directions_values(self):
        assert DIRECTIONS[0] == (0, -1)  # N
        assert DIRECTIONS[1] == (1, 0)   # E
        assert DIRECTIONS[2] == (0, 1)   # S
        assert DIRECTIONS[3] == (-1, 0)  # W


# ============================================================================
# Calibrated mode (doc/gameplay-units-upgrade.md §7 Tier 2)
# ============================================================================

class TestCalibratedCell:
    """Verify calibrated=True exposes physical-energy metadata."""

    def test_calibrated_defaults_match_gameplay(self):
        c = Cell(calibrated=True)
        assert c.energy == INITIAL_CELL_ENERGY == 100.0
        assert c.membrane_permeability == DEFAULT_MEMBRANE_PERMEABILITY

    def test_energy_atp_gameplay_is_plain_float(self):
        assert Cell().energy_atp == pytest.approx(INITIAL_CELL_ENERGY)

    def test_energy_atp_calibrated_counts_atp(self):
        # 100 energy units = 100 * 10^7 = 10^9 ATP
        assert Cell(calibrated=True).energy_atp == pytest.approx(1e9)

    def test_energy_atp_matches_conversion_fn(self):
        c = Cell(calibrated=True, energy=42.0)
        assert c.energy_atp == pytest.approx(energy_to_atp(42.0))

    def test_starved_cell_runs_out_on_schedule(self):
        c = Cell()
        for _ in range(int(INITIAL_CELL_ENERGY / MOVE_ENERGY_COST)):
            c.move(0)
        assert c.energy == 0.0
        c.move(0)
        assert c.energy == 0.0
