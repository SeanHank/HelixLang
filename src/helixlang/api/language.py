"""Language configuration + public constants (doc/38 §6.2 ``api.language``).

Re-exports the goal-#2 ``LanguageConfig`` plus the public constants of
``core.codon_table`` and ``core.units`` so plugins stop importing
``helixlang.core.{codon_table,units,...}`` directly.
"""
from __future__ import annotations

from helixlang.core.codon_table import (  # noqa: F401
    STANDARD_AMINO_ACIDS,
    STANDARD_TABLE,
    Op,
    get_table,
    start_codons_from_table,
    stop_codons_from_table,
    translation_table_from_ncbi,
    wobble,
)
from helixlang.core.language import LanguageConfig  # noqa: F401
from helixlang.core.units import (  # noqa: F401
    ATP_PER_GLUCOSE,
    DIFFUSION_DT_S,
    LATTICE_SPACING_UM,
    PROTEIN_HALF_LIFE_MEDIAN_TICKS,
    TIME_TICK_MIN,
    TIME_TICK_S,
    UNITS_CELL_C_PERIOD_MIN,
    UNITS_CELL_D_PERIOD_MIN,
    UNITS_CELL_DOUBLING_TIME_RICH_MIN,
    UNITS_CELL_SURFACE_EXPONENT,
)

__all__ = [
    "LanguageConfig",
    "Op",
    "STANDARD_AMINO_ACIDS",
    "STANDARD_TABLE",
    "get_table",
    "wobble",
    "stop_codons_from_table",
    "start_codons_from_table",
    "translation_table_from_ncbi",
    "TIME_TICK_MIN",
    "TIME_TICK_S",
    "LATTICE_SPACING_UM",
    "ATP_PER_GLUCOSE",
    "PROTEIN_HALF_LIFE_MEDIAN_TICKS",
    "DIFFUSION_DT_S",
    "UNITS_CELL_C_PERIOD_MIN",
    "UNITS_CELL_D_PERIOD_MIN",
    "UNITS_CELL_DOUBLING_TIME_RICH_MIN",
    "UNITS_CELL_SURFACE_EXPONENT",
]
