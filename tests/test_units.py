"""Physical unit-system and runtime-default consistency tests.

Every runtime constant is anchored to a physical unit (SBML Level 3 /
whole-cell convention): energy in ATP molecules, signals in µM, space in
µm, time in minutes/seconds, diffusion in µm²/s.
"""
import pytest

import helixlang.units as units
from helixlang.units import (
    AI2_DIFFUSION_UM2_S,
    DIFFUSION_DT_S,
    LATTICE_SPACING_UM,
    PROTEIN_HALF_LIFE_MEDIAN_TICKS,
    decay_from_half_life_ticks,
    decay_to_half_life_ticks,
    diffusion_lattice_to_dx,
    diffusion_to_lattice,
    ticks_to_min,
)


# -- conversion dimension checks --
def test_diffusion_to_lattice_worked_example():
    # D = 1e-6 cm^2/s = 100 um^2/s (AI-2), dt = 60 s, dx = 10 um
    assert diffusion_to_lattice(100.0, 60.0, 10.0) == pytest.approx(60.0)
    # the declared lattice edge is recovered from the on-lattice D
    assert diffusion_lattice_to_dx(100.0, 60.0, 60.0) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        diffusion_to_lattice(100.0, 60.0, 0.0)
    with pytest.raises(ValueError):
        diffusion_lattice_to_dx(100.0, 60.0, 0.0)


def test_decay_conversions():
    # 110 min half-life -> ~0.994 per tick (1 tick = 1 min)
    assert decay_from_half_life_ticks(110.0) == pytest.approx(0.9937, abs=1e-3)
    # the inverse recovers the half-life
    assert decay_to_half_life_ticks(
        decay_from_half_life_ticks(110.0)) == pytest.approx(110.0)
    with pytest.raises(ValueError):
        decay_from_half_life_ticks(0.0)
    with pytest.raises(ValueError):
        decay_to_half_life_ticks(0.0)
    with pytest.raises(ValueError):
        decay_to_half_life_ticks(1.0)


def test_ticks_to_min_trivial():
    assert ticks_to_min(5) == pytest.approx(5.0)


def test_module_exports():
    assert units.TIME_TICK_MIN == 1.0
    assert units.TIME_TICK_S == 60.0
    assert units.LATTICE_SPACING_UM == 10.0
    assert units.ATP_PER_GLUCOSE == 38
    assert units.PROTEIN_HALF_LIFE_MEDIAN_TICKS == 110.0
    assert units.AI2_DIFFUSION_UM2_S == 100.0
    assert units.DIFFUSION_DT_S == 60.0


# -- runtime defaults are physically consistent (the unified unit system) --
def test_default_diffusion_is_physical_and_stable_on_lattice():
    from helixlang.population import SIGNAL_DIFFUSION_UM2_S
    assert SIGNAL_DIFFUSION_UM2_S == AI2_DIFFUSION_UM2_S
    # D_lattice = 60 at the declared 10 um / 60 s lattice
    assert diffusion_to_lattice(
        SIGNAL_DIFFUSION_UM2_S, DIFFUSION_DT_S, LATTICE_SPACING_UM
    ) == pytest.approx(60.0)


def test_default_grn_decay_from_median_half_life():
    from helixlang.grn import GRN
    assert GRN.DECAY == pytest.approx(
        decay_from_half_life_ticks(PROTEIN_HALF_LIFE_MEDIAN_TICKS))
    assert GRN.DECAY == pytest.approx(0.994, abs=1e-3)


def test_default_cell_energy_is_atp_molecule_count():
    from helixlang.cell import INITIAL_CELL_ENERGY
    assert INITIAL_CELL_ENERGY == pytest.approx(1e9)  # newborn cell, ~10^9 ATP


def test_default_division_threshold_gives_20_tick_doubling():
    from helixlang.population import (
        DIVISION_ENERGY_THRESHOLD,
        ENERGY_INTAKE_PER_STEP,
        METABOLIC_COST_PER_STEP,
        POPULATION_CELL_INITIAL_ENERGY,
    )
    net = ENERGY_INTAKE_PER_STEP - METABOLIC_COST_PER_STEP
    # (threshold - initial) / net == 20 ticks (Neidhardt 1996 doubling)
    assert (DIVISION_ENERGY_THRESHOLD - POPULATION_CELL_INITIAL_ENERGY
            ) / net == pytest.approx(20.0)


def test_default_quorum_threshold_and_emission_in_um():
    from helixlang.population import (
        QUORUM_SIGNAL_THRESHOLD,
        SIGNAL_EMISSION_PER_STEP,
    )
    assert QUORUM_SIGNAL_THRESHOLD == pytest.approx(10.0)   # 10 uM AI-2
    assert SIGNAL_EMISSION_PER_STEP == pytest.approx(2.0)   # 2 uM per tick
