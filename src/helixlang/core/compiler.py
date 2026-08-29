"""Compiler: AST -> typed IR -> Bytecode Chunk.

Pipeline (doc/37 §4):

    Program (AST)
      -> IRBuilder (:mod:`ir_builder`)   typed biological IR
      -> IROpt      (:mod:`ir_opt`, optional)  optimization
      -> IRLowerer  (:mod:`ir_lower`)    bytecode Chunk (existing ABI)

The lowering stage reproduces the exact bytecode the former direct emitter
produced (same opcodes, operand bytes, constant pool, CALL_GENE back-patching
and inter-gene jump barriers), so the VM + C dispatch kernel work unchanged and
golden results are preserved.  Optimization is opt-in because folding removes
pure instructions and can shift the ``ops_per_tick`` quota boundary inside a
gene ORF; :func:`Compiler.compile_ir` gives both artefacts for runtimes that
consume the IR directly.
"""
from __future__ import annotations

from helixlang.core.ast_nodes import Program
from helixlang.core.bytecode import Chunk
from helixlang.core.codon_table import STANDARD_TABLE, Op
from helixlang.core.errors import CompileError  # noqa: F401  (re-exported)
from helixlang.core.ir import IRProgram
from helixlang.core.ir_builder import IRBuilder
from helixlang.core.ir_lower import IRLowerer
from helixlang.core.ir_opt import IROpt


class Compiler:
    """Program -> typed IR -> Chunk."""

    def __init__(self, table: dict[str, Op] = STANDARD_TABLE):
        self.table = table

    def compile(self, program: Program) -> Chunk:
        """Compile the AST straight to bytecode (IR path, no optimization).

        Byte-identical to the legacy direct emitter; safe default for the
        classic VM and for golden validation.
        """
        return self.compile_ir(program, optimize=False)[1]

    def compile_ir(self, program: Program, *,
                   optimize: bool = False,
                   passes: list[str] | None = None) -> tuple[IRProgram, Chunk]:
        """Full pipeline: return ``(ir, chunk)``.  With ``optimize=True`` the
        IR is optimised before lowering (bits covered by :mod:`ir_opt`)."""
        builder = IRBuilder(self.table)
        ir = builder.build(program)
        if optimize:
            IROpt().optimize(ir, passes=passes)
        chunk = IRLowerer().lower(ir)
        return ir, chunk

    # -------- convenience: IR construction / optimisation hooks --------
    def build_ir(self, program: Program) -> IRProgram:
        """Build the typed IR without lowering (e.g. for ``--dump-ir``)."""
        return IRBuilder(self.table).build(program)
