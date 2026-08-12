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
    # conversions
    "ticks_to_min", "diffusion_to_lattice", "diffusion_lattice_to_dx",
    "decay_from_half_life_ticks", "decay_to_half_life_ticks",
]
