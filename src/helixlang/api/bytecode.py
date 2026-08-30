"""Public bytecode / executed-program surface (doc/38 §6.2 ``api.bytecode``).

Re-exports the bytecode *interpreter* types and tuning constants plugins need
when driving cells that run compiled helix gene programs: ``Chunk`` (a compiled
co-routine body the runtime executes), ``Op``/``OP_OPERAND_BYTES`` (opcode
layout) and the opcode-semantics tuning constants.  All names are stdlib-only;
plugins never import ``core.bytecode`` / ``core.opcode_semantics`` directly.
"""
from __future__ import annotations

from helixlang.core.bytecode import Chunk  # noqa: F401
from helixlang.core.codon_table import (  # noqa: F401
    OP_OPERAND_BYTES,
    Op,
)
from helixlang.core.opcode_semantics import (  # noqa: F401
    BIND_LEVEL_BOOST,
    CONSTITUTIVE_PROMOTER_STRENGTH,
    EMIT_MORPHOGEN_SCALE,
    MORPHOGEN_TO_GRN_GAIN,
    PROTEIN_TO_GRN_GAIN,
    PROTEIN_YIELD_PER_MRNA_AA,
    REGULATE_EDGE_WEIGHT,
    RIBO_SOME_DENSITY_PER_100NT,
    SIGNAL_EMISSION_AMOUNT,
)

__all__ = [
    "Chunk",
    "Op",
    "OP_OPERAND_BYTES",
    "BIND_LEVEL_BOOST",
    "CONSTITUTIVE_PROMOTER_STRENGTH",
    "EMIT_MORPHOGEN_SCALE",
    "MORPHOGEN_TO_GRN_GAIN",
    "PROTEIN_TO_GRN_GAIN",
    "PROTEIN_YIELD_PER_MRNA_AA",
    "REGULATE_EDGE_WEIGHT",
    "RIBO_SOME_DENSITY_PER_100NT",
    "SIGNAL_EMISSION_AMOUNT",
]
