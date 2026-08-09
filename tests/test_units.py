"""Calibration registry and unit-conversion tests (doc/gameplay-units-upgrade.md §7, Tier 1)."""
import importlib

import pytest

import helixlang.units as units
from helixlang.units import (
    AI2_DIFFUSION_UM2_S,
    CALIBRATED,
    DIFFUSION_DT_S,
    LATTICE_SPACING_UM,
    PROTEIN_HALF_LIFE_MEDIAN_TICKS,
    decay_from_half_life_ticks,
    decay_to_half_life_ticks,
    diffusion_lattice_to_dx,
    diffusion_to_lattice,
    energy_to_atp,
    signal_to_um,
    ticks_to_min,
)


def _resolve(name: str):
    """Resolve a dotted registry name like ``cell.INITIAL_CELL_ENERGY`` to its value."""
    parts = name.split(".")
    obj = importlib.import_module(f"helixlang.{parts[0]}")
    for p in parts[1:]:
        obj = getattr(obj, p)
    return obj


# -- conversion dimension checks (§3.2) --
def test_signal_to_um_conversion():
    assert signal_to_um(5.0) == pytest.approx(10.0)   # 5.0 lattice = 10 uM AI-2


def test_energy_to_atp_conversion():
    assert energy_to_atp(1.0) == pytest.approx(1e7)   # 1 unit = 10^7 ATP
    assert energy_to_atp(100.0) == pytest.approx(1e9)  # newborn cell = 10^9 ATP


def test_diffusion_to_lattice_worked_example():
    # D = 1e-6 cm^2/s = 100 um^2/s, dt = 60 s, dx = 10 um
    assert diffusion_to_lattice(100.0, 60.0, 10.0) == pytest.approx(60.0)
    # legacy on-lattice 0.1 implies a colony-scale patch
    assert diffusion_lattice_to_dx(100.0, 60.0, 0.1) == pytest.approx(244.9, rel=1e-2)
    with pytest.raises(ValueError):
        diffusion_to_lattice(100.0, 60.0, 0.0)
    with pytest.raises(ValueError):
        diffusion_lattice_to_dx(100.0, 60.0, 0.0)


def test_decay_conversions():
    # calibrated: 110 min half-life -> ~0.994 per tick (1 tick = 1 min)
    assert decay_from_half_life_ticks(110.0) == pytest.approx(0.9937, abs=1e-3)
    # legacy 0.7 halves the level every ~2 ticks
    assert decay_to_half_life_ticks(0.7) == pytest.approx(1.943, rel=1e-2)
    with pytest.raises(ValueError):
        decay_from_half_life_ticks(0.0)
    with pytest.raises(ValueError):
        decay_to_half_life_ticks(0.0)
    with pytest.raises(ValueError):
        decay_to_half_life_ticks(1.0)


def test_ticks_to_min_trivial():
    assert ticks_to_min(5) == pytest.approx(5.0)


# -- registry invariants (§7 Tier 1) --
def test_calibrated_entries_resolve_to_live_constants():
    """Every CALIBRATED name must resolve to a real module/class attribute."""
    for name in CALIBRATED:
        assert _resolve(name) is not None, f"{name} does not resolve"


def test_calibrated_legacy_default_matches_current_constant():
    """legacy_default must equal today's constant — calibration never changes defaults."""
    for name, cal in CALIBRATED.items():
        assert _resolve(name) == cal.legacy_default, (
            f"{name} legacy_default {cal.legacy_default} != current "
            f"constant {_resolve(name)}"
        )


def test_calibrated_entries_have_citation_and_unit():
    for name, cal in CALIBRATED.items():
        assert cal.unit, f"{name} missing unit"
        assert cal.citation, f"{name} missing citation"


def test_calibrated_conversion_fns_are_consistent():
    """Where a conversion_fn exists, it maps the legacy default to the declared physics."""
    q = CALIBRATED["population.QUORUM_SIGNAL_THRESHOLD"]
    assert q.conversion_fn(q.legacy_default) == pytest.approx(10.0)  # 10 uM

    e = CALIBRATED["cell.INITIAL_CELL_ENERGY"]
    assert e.conversion_fn(e.legacy_default) == pytest.approx(1e9)   # 10^9 ATP

    d = CALIBRATED["population.SIGNAL_DIFFUSION_COEFFICIENT"]
    assert d.conversion_fn(d.legacy_default) == pytest.approx(244.9, rel=1e-2)

    decay = CALIBRATED["grn.GRN.DECAY"]
    assert decay.conversion_fn(decay.legacy_default) == pytest.approx(1.943, rel=1e-2)
    assert decay.physical_value == pytest.approx(
        decay_from_half_life_ticks(PROTEIN_HALF_LIFE_MEDIAN_TICKS))


def test_calibrated_diffusion_physical_value_uses_declared_lattice():
    d = CALIBRATED["population.SIGNAL_DIFFUSION_COEFFICIENT"]
    assert d.physical_value == pytest.approx(
        diffusion_to_lattice(AI2_DIFFUSION_UM2_S, DIFFUSION_DT_S,
                             LATTICE_SPACING_UM))


def test_calibrated_audit_coverage():
    """The registry covers every gameplay-unit catalog row from §2."""
    expected = {
        "cell.INITIAL_CELL_ENERGY",
        "cell.CELL_PROTEIN_SLOT_COUNT",
        "cell.MOVE_ENERGY_COST",
        "cell.FEED_ENERGY_AMOUNT",
        "cell.MIN_DIVISION_ENERGY",
        "cell.MAX_MEMBRANE_PERMEABILITY",
        "cell.DEFAULT_MEMBRANE_PERMEABILITY",
        "population.DEFAULT_MAX_POPULATION_SIZE",
        "population.DEFAULT_GRID_WIDTH",
        "population.DEFAULT_GRID_HEIGHT",
        "population.DIVISION_ENERGY_THRESHOLD",
        "population.DEATH_ENERGY_THRESHOLD",
        "population.SIGNAL_DIFFUSION_COEFFICIENT",
        "population.QUORUM_SIGNAL_THRESHOLD",
        "population.SIGNAL_EMISSION_PER_STEP",
        "population.METABOLIC_COST_PER_STEP",
        "population.ENERGY_INTAKE_PER_STEP",
        "population.POPULATION_CELL_INITIAL_ENERGY",
        "vm.REGULATE_EDGE_WEIGHT",
        "vm.BIND_LEVEL_BOOST",
        "vm.EMIT_MORPHOGEN_SCALE",
        "vm.SIGNAL_EMISSION_AMOUNT",
        "vm.RIBO_SOME_DENSITY_PER_100NT",
        "vm.PROTEIN_YIELD_PER_MRNA_AA",
        "vm.PROTEIN_TO_GRN_GAIN",
        "vm.MORPHOGEN_TO_GRN_GAIN",
        "vm.CONSTITUTIVE_PROMOTER_STRENGTH",
        "grn.GRN.DECAY",
        "central_dogma.PROTEINS_PER_MRNA_LIFETIME",
    }
    assert expected <= set(CALIBRATED)


def test_module_exports():
    assert units.ENERGY_UNIT_ATP == 1e7
    assert units.SIGNAL_UNIT_UM == 2.0
    assert units.TIME_TICK_MIN == 1.0
    assert units.LATTICE_SPACING_UM == 10.0
    assert units.ATP_PER_GLUCOSE == 38
