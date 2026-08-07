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

__all__ = [
    "Op", "STANDARD_TABLE", "MITO_VERTEBRATE_TABLE", "CILIATE_TABLE",
    "TABLES", "get_table", "wobble", "OP_OPERAND_BYTES",
    "HelixError", "LexError", "ParseError", "SemanticError",
    "CompileError", "RegulationError", "RuntimeHelixError",
    "__version__",
]
