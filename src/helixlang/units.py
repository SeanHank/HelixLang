"""Physical unit system for the HelixLang simulation runtime.

The runtime simulates quantities in **real physical/chemical/biological
units** rather than a dimensionless budget.  The unit system
follows the conventions of quantitative systems biology (SBML Level 3,
Hucka et al.; whole-cell modeling, Karr et al. 2012; COBRA flux
analysis, COBRApy): substance in molecule counts, concentrations in
molar (µM), space in micrometres, time in seconds/minutes, fluxes in
mmol/gDW/h (see :mod:`helixlang.metabolism`).

Base axes (anchored to primary literature):

- Time: 1 tick = 1 minute = 60 s (Neidhardt 1996; E. coli doubling
  ~20 min in rich medium at 37 °C)
- Space: lattice site edge, default 10 µm (a 100 x 100 grid = 1 mm
  biofilm patch, O'Toole 2000)
- Energy/substance: ATP molecule counts (Orth 2010 + E. coli dry mass
  ~0.3 pg, Alberts).  A newborn cell holds ~10^9 ATP molecules; the
  maintenance flux is ~2.5e7 ATP/min.
- Concentration: µM (Xavier & Bassler 2003; AI-2 quorum threshold
  ~10 µM)

The module is **stdlib-only** and imports nothing from the rest of
HelixLang, so any runtime module (cell, population, vm, grn,
central_dogma) may import it without circularity.
"""
from __future__ import annotations

import math

# ============================================================================
# Base axes (SI-derived)
# ============================================================================

#: 1 simulation tick = 1 minute (Neidhardt 1996, rich medium)
TIME_TICK_MIN = 1.0

#: 1 simulation tick in seconds (SI)
TIME_TICK_S = 60.0

#: default lattice site edge, µm (a 100x100 grid = 1 mm biofilm patch)
LATTICE_SPACING_UM = 10.0

#: textbook aerobic glucose yield (Alberts, Molecular Biology of the Cell)
ATP_PER_GLUCOSE = 38

#: E. coli protein half-life median (Mosteller 1980, Helbig 2011);
#: expressed in ticks (= minutes while TIME_TICK_MIN == 1.0)
PROTEIN_HALF_LIFE_MEDIAN_TICKS = 110.0

#: AI-2 intercellular diffusion coefficient ~1e-6 cm^2/s = 100 µm^2/s
#: (Miller & Bassler 2001; order-of-magnitude anchor)
AI2_DIFFUSION_UM2_S = 100.0

#: tick duration (s) used for the on-lattice diffusion conversion
DIFFUSION_DT_S = 60.0

# ============================================================================
# Chromosome replication timing (Phase 1: cell cycle, Cooper & Helmstetter
# 1968 J Mol Biol 31:519-540)
# ============================================================================
#: C period: minutes for one round of bidirectional chromosome replication
#: (~40 min for E. coli B/r in rich medium at 37 °C).
UNITS_CELL_C_PERIOD_MIN = 40.0

#: D period: minutes between replication termination and cell division
#: (~20 min for E. coli B/r).
UNITS_CELL_D_PERIOD_MIN = 20.0

#: rich-medium doubling time tau (min).  Replication origins fire every tau
#: minutes; with tau < C + D this is the multifork regime (origin copies
#: double before the previous round terminates), which is the canonical
#: E. coli fast-growth pattern (Cooper & Helmstetter 1968; Karr et al.
#: 2012 integrate it as scheduled per-chromosome replication).
UNITS_CELL_DOUBLING_TIME_RICH_MIN = 20.0

# ============================================================================
# Cell size / density anchors (Phase 2: volume growth & adder)
# ============================================================================
# E. coli cell-density anchors used to convert biomass flux into a physical
# volume (µm^3).  ~0.28 pg of dry mass per µm^3 of wet cell and ~0.15 pg per
# µm^3 of dry cell (Milo & Phillips 2015, Cell Biology by the Numbers;
# Taheri-Araghi 2015 Curr Biol 25:385-391).
UNITS_CELL_DENSITY_WET_PG_UM3 = 0.28
UNITS_CELL_DENSITY_DRY_PG_UM3 = 0.15

#: newborn E. coli volume in rich medium (~1.6 µm^3; Taheri-Araghi 2015)
UNITS_CELL_VOLUME_NEWBORN_UM3 = 1.6

#: adder rule: a rod-shaped bacterium divides when it has added a constant
#: volume Δ since birth, independent of birth size (Taheri-Araghi 2015;
#: Jun 2018 Rep Prog Phys 81:056601).  In rich medium Δ ≈ V_birth.
UNITS_ADDER_VOLUME_UM3 = 1.6

#: surface-to-volume exponent for uptake scaling (V^(2/3) ~ surface of a
#: sphere/rod), used by the Phase-2 ``surface_scaling`` option.
UNITS_CELL_SURFACE_EXPONENT = 2.0 / 3.0

# ============================================================================
# Protein maturation / folding / QC anchors (Phase 3, Balchin 2016)
# ============================================================================
#: ATP cost to fold one protein through the chaperone machinery (order
#: 10^1-10^2 ATP per protein; GroEL-GroES uses ~7 ATP per folding cycle and
#: a substrate may require several cycles; Balchin 2016 Science
#: 353:aac4354).
PROTEIN_FOLDING_ATP_PER_PROTEIN = 50.0
#: first-order folding rate (per min) for unfolded, chaperone-bound protein
PROTEIN_FOLD_RATE_PER_MIN = 1.0
#: first-order misfolding rate (per min) competing with folding; the folded
#: fraction at equilibrium is k_fold/(k_fold+k_misfold)
PROTEIN_MISFOLD_RATE_PER_MIN = 0.05
#: misfolded -> aggregate rate (per min); aggregates are inert (not removed)
PROTEIN_AGGREGATION_RATE_PER_MIN = 0.02
#: misfolded -> degraded rate (per min; Lon/Clp remove misfolded protein
#: much faster than folding)
PROTEIN_DEGRADED_RATE_PER_MIN = 5.0


# ============================================================================
# Derived conversions
# ============================================================================
def ticks_to_min(ticks: float) -> float:
    """Convert ticks to minutes (identity while TIME_TICK_MIN == 1.0)."""
    return ticks * TIME_TICK_MIN


def diffusion_to_lattice(D_um2_s: float, dt_s: float, dx_um: float) -> float:
    """Dimensionless on-lattice diffusion coefficient from physical D.

    ``D_lattice = D_phys * dt / dx^2`` (the factor used by the explicit
    5-point Laplacian in :func:`helixlang.population.signal_diffusion_step`).

    Worked example: D = 100 µm^2/s (AI-2), dt = 60 s, dx = 10 µm gives
    ``100 * 60 / 100 = 60``.
    """
    if dx_um <= 0:
        raise ValueError("dx_um must be > 0")
    if dt_s < 0:
        raise ValueError("dt_s must be >= 0")
    return D_um2_s * dt_s / (dx_um * dx_um)


def diffusion_lattice_to_dx(D_um2_s: float, dt_s: float,
                            D_lattice: float) -> float:
    """Implied lattice spacing (µm) for a dimensionless on-lattice D.

    ``dx = sqrt(D_phys * dt / D_lattice)`` — the inverse of
    :func:`diffusion_to_lattice`, useful for checking that a given
    on-lattice coefficient is physically self-consistent.
    """
    if D_lattice <= 0:
        raise ValueError("D_lattice must be > 0")
    return math.sqrt(D_um2_s * dt_s / D_lattice)


def decay_from_half_life_ticks(half_life_ticks: float) -> float:
    """Per-tick decay coefficient from a protein half-life (ticks).

    ``decay = 0.5 ** (1 / half_life_ticks)``: the fraction of protein
    remaining after one tick given first-order degradation.  With one
    tick per minute, a 110 min median half-life gives decay ~= 0.994.
    """
    if half_life_ticks <= 0:
        raise ValueError("half_life_ticks must be > 0")
    return float(0.5 ** (1.0 / half_life_ticks))


def decay_to_half_life_ticks(decay: float) -> float:
    """Implied protein half-life (ticks) of a per-tick decay coefficient.

    Inverse of :func:`decay_from_half_life_ticks`.
    """
    if decay <= 0.0 or decay >= 1.0:
        raise ValueError("decay must be in (0, 1)")
    return math.log(0.5) / math.log(decay)


__all__ = [
    # base axes
    "TIME_TICK_MIN", "TIME_TICK_S", "LATTICE_SPACING_UM",
    "ATP_PER_GLUCOSE", "PROTEIN_HALF_LIFE_MEDIAN_TICKS",
    "AI2_DIFFUSION_UM2_S", "DIFFUSION_DT_S",
    # chromosome replication timing (Phase 1)
    "UNITS_CELL_C_PERIOD_MIN", "UNITS_CELL_D_PERIOD_MIN",
    "UNITS_CELL_DOUBLING_TIME_RICH_MIN",
    # cell volume / density / adder (Phase 2)
    "UNITS_CELL_DENSITY_WET_PG_UM3", "UNITS_CELL_DENSITY_DRY_PG_UM3",
    "UNITS_CELL_VOLUME_NEWBORN_UM3", "UNITS_ADDER_VOLUME_UM3",
    "UNITS_CELL_SURFACE_EXPONENT",
    # protein maturation / folding / QC (Phase 3)
    "PROTEIN_FOLDING_ATP_PER_PROTEIN", "PROTEIN_FOLD_RATE_PER_MIN",
    "PROTEIN_MISFOLD_RATE_PER_MIN", "PROTEIN_AGGREGATION_RATE_PER_MIN",
    "PROTEIN_DEGRADED_RATE_PER_MIN",
    # conversions
    "ticks_to_min", "diffusion_to_lattice", "diffusion_lattice_to_dx",
    "decay_from_half_life_ticks", "decay_to_half_life_ticks",
]
