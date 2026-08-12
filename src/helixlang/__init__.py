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
    AI2_DIFFUSION_UM2_S,
    ATP_PER_GLUCOSE,
    DIFFUSION_DT_S,
    LATTICE_SPACING_UM,
    PROTEIN_HALF_LIFE_MEDIAN_TICKS,
    TIME_TICK_MIN,
    TIME_TICK_S,
    decay_from_half_life_ticks,
    decay_to_half_life_ticks,
    diffusion_lattice_to_dx,
    diffusion_to_lattice,
    ticks_to_min,
)

__all__ = [
    "Op", "STANDARD_TABLE", "MITO_VERTEBRATE_TABLE", "CILIATE_TABLE",
    "TABLES", "get_table", "wobble", "OP_OPERAND_BYTES",
    "HelixError", "LexError", "ParseError", "SemanticError",
    "CompileError", "RegulationError", "RuntimeHelixError",
    "TIME_TICK_MIN", "TIME_TICK_S", "LATTICE_SPACING_UM",
    "AI2_DIFFUSION_UM2_S", "DIFFUSION_DT_S", "ATP_PER_GLUCOSE",
    "PROTEIN_HALF_LIFE_MEDIAN_TICKS",
    "ticks_to_min", "diffusion_to_lattice", "diffusion_lattice_to_dx",
    "decay_from_half_life_ticks", "decay_to_half_life_ticks",
    "__version__",
]
