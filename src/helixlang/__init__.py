"""HelixLang: DNA codons → bytecode → biological simulation."""

__version__ = "0.1.0"

from helixlang.codon_table import (
    CILIATE_TABLE,
    MITO_VERTEBRATE_TABLE,
    OP_OPERAND_BYTES,
    STANDARD_TABLE,
    TABLES,
    Op,
    get_table,
    wobble,
)
from helixlang.errors import (
    CompileError,
    HelixError,
    LexError,
    ParseError,
    RegulationError,
    RuntimeHelixError,
    SemanticError,
)
from helixlang.units import (
    ATP_PER_GLUCOSE,
    CALIBRATED,
    ENERGY_UNIT_ATP,
    LATTICE_SPACING_UM,
    SIGNAL_UNIT_UM,
    TIME_TICK_MIN,
    Calibration,
    decay_from_half_life_ticks,
    decay_to_half_life_ticks,
    diffusion_lattice_to_dx,
    diffusion_to_lattice,
    energy_to_atp,
    signal_to_um,
    ticks_to_min,
)

__all__ = [
    "Op", "STANDARD_TABLE", "MITO_VERTEBRATE_TABLE", "CILIATE_TABLE",
    "TABLES", "get_table", "wobble", "OP_OPERAND_BYTES",
    "HelixError", "LexError", "ParseError", "SemanticError",
    "CompileError", "RegulationError", "RuntimeHelixError",
    "CALIBRATED", "Calibration",
    "TIME_TICK_MIN", "LATTICE_SPACING_UM", "ENERGY_UNIT_ATP",
    "SIGNAL_UNIT_UM", "ATP_PER_GLUCOSE",
    "energy_to_atp", "signal_to_um", "ticks_to_min",
    "diffusion_to_lattice", "diffusion_lattice_to_dx",
    "decay_from_half_life_ticks", "decay_to_half_life_ticks",
    "__version__",
]
