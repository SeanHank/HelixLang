"""Physical calibration registry for gameplay units.

This module turns HelixLang's runtime simulation constants (cell energy,
signals, diffusion, GRN decay) into a single, cited, unit-carrying
registry.  It is the Tier 1 deliverable of
``doc/gameplay-units-upgrade.md``: base-axis constants (§3.1), the
derived conversions (§3.2) and the ``CALIBRATED`` table mapping every
audit entry in §2 to ``(physical_value, unit, citation, conversion_fn,
legacy_default)``.

The module is **stdlib-only** and imports nothing from the rest of
HelixLang, so any runtime module (cell, population, vm, grn,
central_dogma) may import it without circularity.  ``CALIBRATED`` is
*data, not execution*: reading it never changes behavior.  Defaults
never change — ``legacy_default`` records today's value so drift is
caught by the registry invariant tests, not silently accepted.

Base axes (see ``doc/gameplay-units-upgrade.md`` §3.1):

- Time: 1 tick = 1 minute (Neidhardt 1996; E. coli doubling ~20 min)
- Space: lattice site edge, default 10 µm (100 x 100 grid = 1 mm patch)
- Energy: 10^7 ATP molecules = ~1 min of maintenance ATP
  (Orth 2010 + E. coli dry mass ~0.3 pg, Alberts)
- Concentration: 2 µM per lattice unit (quorum threshold 5.0 lattice
  units = 10 µM AI-2, Xavier & Bassler 2003)
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

# ============================================================================
# Base axes (gameplay -> physical anchors)
# ============================================================================

#: 1 simulation tick = 1 minute (Neidhardt 1996, rich medium)
TIME_TICK_MIN = 1.0

#: default lattice site edge, µm (a 100x100 grid = 1 mm biofilm patch)
LATTICE_SPACING_UM = 10.0

#: 1 energy unit = 10^7 ATP molecules (~1 min of maintenance ATP,
#: Orth 2010 8.39 mmol/gDW/h x ~0.3 pg DW x 6e23 / 60 ~= 2.5e7 ATP/min)
ENERGY_UNIT_ATP = 1.0e7

#: 1 signal lattice unit = 2 µM (quorum threshold 5.0 = 10 µM AI-2,
#: Xavier & Bassler 2003)
SIGNAL_UNIT_UM = 2.0

#: textbook aerobic glucose yield (Alberts, Molecular Biology of the Cell)
ATP_PER_GLUCOSE = 38

#: E. coli protein half-life median (Mosteller 1980, Helbig 2011)
PROTEIN_HALF_LIFE_MEDIAN_TICKS = 110.0

#: AI-2 intercellular diffusion coefficient ~1e-6 cm^2/s = 100 µm^2/s
#: (Miller & Bassler 2001; order-of-magnitude anchor)
AI2_DIFFUSION_UM2_S = 100.0

#: tick duration (s) used for the on-lattice diffusion conversion
DIFFUSION_DT_S = 60.0


# ============================================================================
# Derived conversions (§3.2)
# ============================================================================
def energy_to_atp(energy: float) -> float:
    """Convert an energy-unit count to ATP molecules."""
    return energy * ENERGY_UNIT_ATP


def signal_to_um(signal: float) -> float:
    """Convert signal lattice units to µM."""
    return signal * SIGNAL_UNIT_UM


def ticks_to_min(ticks: float) -> float:
    """Convert ticks to minutes (identity while TIME_TICK_MIN == 1.0)."""
    return ticks * TIME_TICK_MIN


def diffusion_to_lattice(D_um2_s: float, dt_s: float, dx_um: float) -> float:
    """Dimensionless on-lattice diffusion coefficient.

    ``D_lattice = D_phys * dt / dx^2`` (the factor used by the explicit
    5-point Laplacian in :func:`helixlang.population.signal_diffusion_step`).

    Worked example: D = 100 µm^2/s, dt = 60 s, dx = 10 µm gives
    ``100 * 60 / 100 = 60`` — the calibrated replacement for the legacy
    on-lattice 0.1 (which instead implies a coarse dx ~= 245 µm).
    """
    if dx_um <= 0:
        raise ValueError("dx_um must be > 0")
    if dt_s < 0:
        raise ValueError("dt_s must be >= 0")
    return D_um2_s * dt_s / (dx_um * dx_um)


def diffusion_lattice_to_dx(D_um2_s: float, dt_s: float,
                            D_lattice: float) -> float:
    """Implied lattice spacing (µm) for a dimensionless on-lattice D.

    ``dx = sqrt(D_phys * dt / D_lattice)``.  The legacy ``0.1`` therefore
    corresponds to a *colony-scale* patch: sqrt(100 * 60 / 0.1) ~= 245 µm.
    """
    if D_lattice <= 0:
        raise ValueError("D_lattice must be > 0")
    return math.sqrt(D_um2_s * dt_s / D_lattice)


def decay_from_half_life_ticks(half_life_ticks: float) -> float:
    """Per-tick decay coefficient from a protein half-life (ticks).

    ``decay = 0.5 ** (1 / half_life_ticks)``: the fraction of protein
    remaining after one tick given first-order degradation.  With one
    tick per minute, a 110 min median half-life gives decay ~= 0.994,
    far slower than the legacy universal 0.7.
    """
    if half_life_ticks <= 0:
        raise ValueError("half_life_ticks must be > 0")
    return float(0.5 ** (1.0 / half_life_ticks))


def decay_to_half_life_ticks(decay: float) -> float:
    """Implied protein half-life (ticks) of a per-tick decay coefficient.

    Inverse of :func:`decay_from_half_life_ticks`: the legacy ``0.7``
    implies a half-life of ~1.9 ticks, i.e. levels halve every ~2 ticks.
    """
    if decay <= 0.0 or decay >= 1.0:
        raise ValueError("decay must be in (0, 1)")
    return math.log(0.5) / math.log(decay)


# ============================================================================
# Calibration registry
# ============================================================================
class Calibration(NamedTuple):
    """A single calibrated gameplay constant.

    Attributes:
        physical_value: the calibrated default (gameplay count carrying
            the physical meaning described by ``unit``)
        unit: physical meaning of the calibrated value
        citation: primary-literature anchor (see doc/gameplay-units-upgrade.md §4)
        conversion_fn: optional ``gameplay -> physical`` converter
            (None when no numerical conversion applies)
        legacy_default: today's default — calibration must never change it
    """
    physical_value: float
    unit: str
    citation: str
    conversion_fn: Callable[[float], float] | None
    legacy_default: float


#: single source of truth for calibrated defaults.  Keys follow the audit
#: ledger §2 (<module>.<CONSTANT>); every value is data, not execution.
CALIBRATED: dict[str, Calibration] = {
    # --- cell.py: single-cell energy budget (§2.1) ---
    "cell.INITIAL_CELL_ENERGY": Calibration(
        100.0, "energy units (= 10^9 ATP molecules)",
        "Orth 2010; Alberts (dry mass)", energy_to_atp, 100),
    "cell.CELL_PROTEIN_SLOT_COUNT": Calibration(
        256.0, "protein slots (symbolic; E. coli ~4300 genes)",
        "symbolic capacity", None, 256),
    "cell.MOVE_ENERGY_COST": Calibration(
        1.0, "energy units (= 10^7 ATP; flagellar motor ~10^3-10^4 ATP/rev)",
        "Orth 2010; Alberts", energy_to_atp, 1),
    "cell.FEED_ENERGY_AMOUNT": Calibration(
        10.0, "energy units per feed (= 10^8 ATP; glucose uptake -> ATP)",
        "Alberts (aerobic glucose yield, ATP_PER_GLUCOSE)", energy_to_atp, 10),
    "cell.MIN_DIVISION_ENERGY": Calibration(
        2.0, "energy units (minimum biomass to divide)",
        "growth threshold -> biomass accumulation", energy_to_atp, 2),
    "cell.MAX_MEMBRANE_PERMEABILITY": Calibration(
        255.0, "permeability scale (porin / transporter density)",
        "porin / transporter density", None, 255),
    "cell.DEFAULT_MEMBRANE_PERMEABILITY": Calibration(
        255.0, "permeability (fully permeable; high-porin rich-medium)",
        "high-porin rich-medium default", None, 255),

    # --- population.py: multicellular lattice budget (§2.2) ---
    "population.DEFAULT_MAX_POPULATION_SIZE": Calibration(
        10000.0, "cells (numerical lattice capacity)",
        "numerical", None, 10000),
    "population.DEFAULT_GRID_WIDTH": Calibration(
        100.0, "lattice sites (= 1 mm biofilm patch at 10 um)",
        "O'Toole 2000; LATTICE_SPACING_UM", None, 100),
    "population.DEFAULT_GRID_HEIGHT": Calibration(
        100.0, "lattice sites (= 1 mm biofilm patch at 10 um)",
        "O'Toole 2000; LATTICE_SPACING_UM", None, 100),
    "population.DIVISION_ENERGY_THRESHOLD": Calibration(
        180.0, "energy units (= 1.8 x 10^9 ATP; reachable in ~20 ticks "
               "at +4 energy/tick, rich medium)",
        "Neidhardt 1996", energy_to_atp, 200.0),
    "population.DEATH_ENERGY_THRESHOLD": Calibration(
        0.0, "energy units (starvation death)",
        "starvation death", energy_to_atp, 0.0),
    "population.SIGNAL_DIFFUSION_COEFFICIENT": Calibration(
        diffusion_to_lattice(AI2_DIFFUSION_UM2_S, DIFFUSION_DT_S,
                             LATTICE_SPACING_UM),
        "dimensionless on-lattice D (dx=10 um, dt=60 s); legacy 0.1 "
        "implies dx ~= 245 um (colony-scale)",
        "Miller & Bassler 2001; doc/gameplay-units-upgrade.md §3.2",
        lambda d: diffusion_lattice_to_dx(
            AI2_DIFFUSION_UM2_S, DIFFUSION_DT_S, d),
        0.1),
    "population.QUORUM_SIGNAL_THRESHOLD": Calibration(
        5.0, "lattice units (= 10 uM AI-2)",
        "Xavier & Bassler 2003", signal_to_um, 5.0),
    "population.SIGNAL_EMISSION_PER_STEP": Calibration(
        1.0, "lattice units per cell per step (= 2 uM AI-2)",
        "Xavier & Bassler 2003; Miller & Bassler 2001",
        signal_to_um, 1.0),
    "population.METABOLIC_COST_PER_STEP": Calibration(
        1.0, "energy units per step (= 10^7 ATP; maintenance flux)",
        "Orth 2010", energy_to_atp, 1.0),
    "population.ENERGY_INTAKE_PER_STEP": Calibration(
        5.0, "energy units per step (= 5 x 10^7 ATP/min ~= 0.1% of a "
             "division budget)",
        "Alberts (glucose uptake); doc/gameplay-units-upgrade.md §5.2",
        energy_to_atp, 5.0),
    "population.POPULATION_CELL_INITIAL_ENERGY": Calibration(
        100.0, "energy units (newborn biomass)",
        "newborn biomass", energy_to_atp, 100.0),

    # --- vm.py: runtime opcode semantics (§2.3) ---
    "vm.REGULATE_EDGE_WEIGHT": Calibration(
        1.0, "effector increment (nM, consistent with Hill kd=)",
        "Berg & von Hippel 1987", None, 1.0),
    "vm.BIND_LEVEL_BOOST": Calibration(
        0.5, "TF occupancy fold-change",
        "McClure 1985", None, 0.5),
    "vm.EMIT_MORPHOGEN_SCALE": Calibration(
        256.0, "morphogen dose divisor (1/256 emission)",
        "Pearson 1993", None, 256),
    "vm.SIGNAL_EMISSION_AMOUNT": Calibration(
        0.25, "lattice units per OP_SIGNAL (= 0.5 uM AI-2/event)",
        "Xavier & Bassler 2003; doc/gameplay-units-upgrade.md §5.3",
        signal_to_um, 0.25),
    "vm.RIBO_SOME_DENSITY_PER_100NT": Calibration(
        0.1, "ribosomes per 100 nt mRNA (ribosome loading)",
        "Ingolia 2009", None, 0.1),
    "vm.PROTEIN_YIELD_PER_MRNA_AA": Calibration(
        0.1, "proteins per mRNA per amino acid (coupling gain)",
        "Bernstein 2002; doc/gameplay-units-upgrade.md §5.3", None, 0.1),
    "vm.PROTEIN_TO_GRN_GAIN": Calibration(
        0.01, "gene-level boost per protein molecule",
        "Berg & von Hippel 1987; McClure 1985", None, 0.01),
    "vm.MORPHOGEN_TO_GRN_GAIN": Calibration(
        0.1, "gene-level boost per field V unit",
        "Berg & von Hippel 1987; McClure 1985", None, 0.1),
    "vm.CONSTITUTIVE_PROMOTER_STRENGTH": Calibration(
        0.5, "constitutive promoter reference level (0..1)",
        "Salgado 2013", None, 0.5),

    # --- grn.py: regulatory kinetics (§2.4) ---
    "grn.GRN.DECAY": Calibration(
        decay_from_half_life_ticks(PROTEIN_HALF_LIFE_MEDIAN_TICKS),
        "1/tick (calibrated: 110 min half-life ~= 0.994)",
        "Mosteller 1980; Helbig 2011",
        decay_to_half_life_ticks, 0.7),

    # --- central_dogma.py: protein yield mapping (§5.5) ---
    "central_dogma.PROTEINS_PER_MRNA_LIFETIME": Calibration(
        100.0, "proteins per mRNA lifetime (10^2-10^3)",
        "Bernstein 2002", None, 100.0),
}

__all__ = [
    # base axes
    "TIME_TICK_MIN", "LATTICE_SPACING_UM", "ENERGY_UNIT_ATP",
    "SIGNAL_UNIT_UM", "ATP_PER_GLUCOSE", "PROTEIN_HALF_LIFE_MEDIAN_TICKS",
    "AI2_DIFFUSION_UM2_S", "DIFFUSION_DT_S",
    # conversions
    "energy_to_atp", "signal_to_um", "ticks_to_min",
    "diffusion_to_lattice", "diffusion_lattice_to_dx",
    "decay_from_half_life_ticks", "decay_to_half_life_ticks",
    # registry
    "Calibration", "CALIBRATED",
]
