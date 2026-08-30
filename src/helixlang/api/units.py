"""Public physical / simulation constants (doc/38 §6.2 ``api.units``).

Re-exports the full public surface of ``core.units`` (stdlib-only constants
and conversions) so plugins stop importing ``helixlang.core.units`` directly.
The set below is frozen; it mirrors ``core.units.__all__``.
"""
from __future__ import annotations

from helixlang.core.units import (  # noqa: F401
    AI2_DIFFUSION_UM2_S,
    ATP_PER_GLUCOSE,
    DIFFUSION_DT_S,
    LATTICE_SPACING_UM,
    PROTEIN_AGGREGATION_RATE_PER_MIN,
    PROTEIN_DEGRADED_RATE_PER_MIN,
    PROTEIN_FOLD_RATE_PER_MIN,
    PROTEIN_FOLDING_ATP_PER_PROTEIN,
    PROTEIN_HALF_LIFE_MEDIAN_TICKS,
    PROTEIN_MISFOLD_RATE_PER_MIN,
    TIME_TICK_MIN,
    TIME_TICK_S,
    UNITS_ADDER_VOLUME_UM3,
    UNITS_CELL_C_PERIOD_MIN,
    UNITS_CELL_D_PERIOD_MIN,
    UNITS_CELL_DENSITY_DRY_PG_UM3,
    UNITS_CELL_DENSITY_WET_PG_UM3,
    UNITS_CELL_DOUBLING_TIME_RICH_MIN,
    UNITS_CELL_SURFACE_EXPONENT,
    UNITS_CELL_VOLUME_NEWBORN_UM3,
    decay_from_half_life_ticks,
    decay_to_half_life_ticks,
    diffusion_lattice_to_dx,
    diffusion_to_lattice,
    ticks_to_min,
)

__all__ = [
    "ATP_PER_GLUCOSE",
    "AI2_DIFFUSION_UM2_S",
    "DIFFUSION_DT_S",
    "LATTICE_SPACING_UM",
    "PROTEIN_AGGREGATION_RATE_PER_MIN",
    "PROTEIN_DEGRADED_RATE_PER_MIN",
    "PROTEIN_FOLD_RATE_PER_MIN",
    "PROTEIN_FOLDING_ATP_PER_PROTEIN",
    "PROTEIN_HALF_LIFE_MEDIAN_TICKS",
    "PROTEIN_MISFOLD_RATE_PER_MIN",
    "TIME_TICK_MIN",
    "TIME_TICK_S",
    "UNITS_ADDER_VOLUME_UM3",
    "UNITS_CELL_C_PERIOD_MIN",
    "UNITS_CELL_D_PERIOD_MIN",
    "UNITS_CELL_DENSITY_DRY_PG_UM3",
    "UNITS_CELL_DENSITY_WET_PG_UM3",
    "UNITS_CELL_DOUBLING_TIME_RICH_MIN",
    "UNITS_CELL_SURFACE_EXPONENT",
    "UNITS_CELL_VOLUME_NEWBORN_UM3",
    "decay_from_half_life_ticks",
    "decay_to_half_life_ticks",
    "diffusion_lattice_to_dx",
    "diffusion_to_lattice",
    "ticks_to_min",
]
