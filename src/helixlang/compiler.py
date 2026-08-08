"""Compiler: AST -> Bytecode Chunk.

First pass: emits each gene's ORF bytecode (codon -> opcode via the codon table).
Second pass: back-patches OP_CALL_GENE operands to gene offsets; back-patches
OP_JUMP barriers to the end of the chunk.

P0-1 fix: OP_PUSH_CONST's operand is a constant pool index (consistent with VM
        semantics); at compile time the wobble value (0..3) is added to the
        constant pool and its index is emitted.
P1-3 fix: OP_CALL_GENE supports the ``#gene call_target=<name>`` field to break
        the 4-gene limit (the wobble value is only 0..3, so it cannot address the
        5th gene and beyond); without call_target it falls back to
        ``wobble % len(names)`` modulo selection for backwards compatibility.
        If the target gene does not exist, raise a CompileError instead of
        silently back-patching offset 0.
        An OP_JUMP barrier is emitted between genes to prevent fall-through
        after the quota is exhausted.
"""
from __future__ import annotations

from helixlang.ast_nodes import Codon, Gene, Program
from helixlang.bytecode import Chunk
from helixlang.codon_table import (
    OP_OPERAND_BYTES,
    STANDARD_TABLE,
    Op,
    wobble,
)
from helixlang.errors import CompileError


class Compiler:
    """Program -> Chunk."""

    def __init__(self, table: dict[str, Op] = STANDARD_TABLE):
        self.table = table
        # Record each CALL_GENE's position in the chunk + wobble value + owning gene name
        self._call_sites: list[tuple[int, int, str]] = []
        # Back-patch positions for inter-gene OP_JUMP barriers (first operand byte ip)
        self._jump_to_end_sites: list[int] = []

    def compile(self, program: Program) -> Chunk:
        chunk = Chunk()
        # First pass: emit ORFs
        for gene in program.genes:
            chunk.gene_offsets[gene.name] = len(chunk.code)
            last_op_ip = self._compile_orf(chunk, gene.orf, gene.name)
            if not self._last_op_is_halt(chunk, last_op_ip):
                chunk.emit(Op.OP_HALT, line=0, codon_index=-1)
            # Inter-gene barrier: emit OP_JUMP to the end of the chunk to prevent fall-through after the quota is exhausted
            if gene is not program.genes[-1]:
                start = chunk.emit_u16(Op.OP_JUMP, 0, line=0, codon_index=-1)
                self._jump_to_end_sites.append(start + 1)  # first operand byte
        # Second pass: back-patch CALL_GENE and JUMP
        self._patch_calls(chunk, program.genes)
        self._patch_jumps_to_end(chunk)
        # Constant pool: holds L-system rules and strings (for the VM)
        self._emit_constants(chunk, program)
        return chunk

    # -------- ORF compilation --------
    def _compile_orf(self, chunk: Chunk, orf: list[Codon],
                     gene_name: str) -> int:
        """Emit one ORF, returning the ip of its last instruction (or -1 if empty)."""
        last_op_ip = -1
        for c in orf:
            if c.seq not in self.table:
                raise CompileError(
                    f"unknown codon {c.seq!r} (table has {len(self.table)} entries)",
                    line=c.line, codon_index=c.index)
            op = self.table[c.seq]
            arg = wobble(c.seq)  # third-base wobble value 0..3
            nbytes = OP_OPERAND_BYTES[op]
            if op == Op.OP_CALL_GENE:
                # Placeholder: offset back-patched in the second pass; record wobble and owning gene (for call_target)
                last_op_ip = chunk.emit(op, 0, 0, line=c.line,
                                        codon_index=c.index)
                self._call_sites.append((last_op_ip + 1, arg, gene_name))
            elif op == Op.OP_PUSH_CONST:
                # P0-1: wobble value added to the constant pool, emit the constant pool index (consistent with VM semantics)
                const_idx = chunk.add_constant(arg)
                last_op_ip = chunk.emit(op, const_idx, line=c.line,
                                        codon_index=c.index)
            elif nbytes == 1:
                last_op_ip = chunk.emit(op, arg, line=c.line,
                                        codon_index=c.index)
            elif nbytes == 0:
                last_op_ip = chunk.emit(op, line=c.line, codon_index=c.index)
            else:
                # Should not happen
                last_op_ip = chunk.emit(op, *([0] * nbytes),
                                        line=c.line, codon_index=c.index)
        return last_op_ip

    @staticmethod
    def _last_op_is_halt(chunk: Chunk, last_op_ip: int) -> bool:
        """O(1) check: did the gene's ORF end with an explicit OP_HALT?

        Previously this rescanned the entire emitted chunk for every gene
        (``_ends_with_halt``), making compile time O(genes × chunk size).
        ``_compile_orf`` now reports the ip of the last instruction it
        emitted, so this is a single byte comparison.
        """
        return last_op_ip >= 0 and chunk.code[last_op_ip] == int(Op.OP_HALT)

    # -------- Back-patching CALL_GENE --------
    def _patch_calls(self, chunk: Chunk, genes: list[Gene]) -> None:
        if not genes:
            return
        names = [g.name for g in genes]
        gene_by_name = {g.name: g for g in genes}
        for ip, wobble_arg, owner_name in self._call_sites:
            owner = gene_by_name.get(owner_name)
            # Prefer the call_target field to explicitly specify the call target (breaks the 4-gene
            # limit: the wobble value is only 0..3, so it cannot address the 5th gene and beyond)
            target_name = (
                owner.fields.get("call_target") if owner else None
            )
            if target_name is None:
                # Fall back to modulo selection via the wobble value (backwards compatible; supports the first 4 genes)
                target_name = names[wobble_arg % len(names)]
            offset = chunk.gene_offsets.get(target_name)
            if offset is None:
                # Target gene does not exist: raise an error instead of silently back-patching offset 0 (to avoid jumping to a wrong location)
                raise CompileError(
                    f"CALL_GENE target {target_name!r} not defined; "
                    f"use #gene ... call_target=<name> to specify a target")
            chunk.code[ip] = (offset >> 8) & 0xFF
            chunk.code[ip + 1] = offset & 0xFF

    # -------- Back-patching JUMP barriers --------
    def _patch_jumps_to_end(self, chunk: Chunk) -> None:
        """Back-patch all inter-gene JUMP barriers to the end of the chunk.

        OP_JUMP's operand is a relative offset (``vm.ip += off``).
        emit_u16 returns the ip of the opcode (noted as ``start``); the operand
        occupies ``start+1`` and ``start+2``. After the VM reads the u16,
        ip=start+3, hence ``off = end - start - 3``.
        """
        end = len(chunk.code)
        for start_plus_1 in self._jump_to_end_sites:
            start = start_plus_1 - 1  # opcode ip
            off = end - start - 3
            chunk.code[start_plus_1] = (off >> 8) & 0xFF
            chunk.code[start_plus_1 + 1] = off & 0xFF

    # -------- Constant pool --------
    def _emit_constants(self, chunk: Chunk, program: Program) -> None:
        # L-system axioms and rules as string constants (the VM looks them up by name)
        for name, decl in program.lsystems.items():
            chunk.add_constant(("lsystem_axiom", name, decl.axiom))
            chunk.add_constant(("lsystem_rules", name, decl.rules))
            chunk.add_constant(("lsystem_angle", name, decl.angle))
            chunk.add_constant(("lsystem_step", name, decl.step))
        # Gene name list (for disassembly and debugging)
        for g in program.genes:
            chunk.add_constant(("gene_name", g.name))
