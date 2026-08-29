"""IRLowerer: typed IR (IRProgram) -> bytecode (Chunk).

Reproduces the classic compiler's two-pass emission with identical bytes:

1. plugin opt-in directives (:token:`OP_USE_PLUGIN`) at the chunk head;
2. per-gene ORF emission with a HALT guard and inter-gene jump barrier;
3. back-patching of :token:`OP_CALL_GENE` operands (resolved names carried in
   the IR) and of the jump barriers to the chunk end;
4. constant pool tails for L-system declarations and gene names.

Because :mod:`ir_builder` already resolved ``OP_CALL_GENE`` to a concrete gene
name, folding the legacy compiler's *modulo selection* into the IR, lowering
needs no codon-table access.  When the IR was built without optimization the
emitted chunk is byte-identical to ``Compiler().compile(program)``.
"""
from __future__ import annotations

from helixlang.core.bytecode import Chunk
from helixlang.core.codon_table import OP_OPERAND_BYTES, Op
from helixlang.core.ir import IRFunction, IRProgram


class IRLowerer:
    """IRProgram -> Chunk."""

    def __init__(self) -> None:
        # (operand-field ip, target gene name) for OP_CALL_GENE back-patching
        self._call_sites: list[tuple[int, str]] = []
        # (first operand byte ip) for inter-gene jump barriers
        self._jump_to_end_sites: list[int] = []

    def lower(self, ir: IRProgram) -> Chunk:
        chunk = Chunk()
        self._call_sites = []
        self._jump_to_end_sites = []

        # Plugin opt-ins (doc/36 §3.2), in source order.
        for plugin, flags in ir.use_directives:
            const_idx = chunk.add_constant(("use_plugin", plugin, flags))
            chunk.emit(Op.OP_USE_PLUGIN, const_idx, line=0, codon_index=-1)

        # First pass: encode each gene ORF.
        for i, fn in enumerate(ir.functions):
            chunk.gene_offsets[fn.name] = len(chunk.code)
            last_op_ip = self._emit_orf(chunk, fn)
            if not self._last_op_is_halt(chunk, last_op_ip):
                chunk.emit(Op.OP_HALT, line=0, codon_index=-1)
            is_last = i == len(ir.functions) - 1
            if not is_last:
                start = chunk.emit_u16(Op.OP_JUMP, 0, line=0, codon_index=-1)
                self._jump_to_end_sites.append(start + 1)

        # Second pass: back-patch calls and end-of-chunk jump barriers.
        self._patch_calls(chunk, ir)
        self._patch_jumps_to_end(chunk)
        self._emit_constants(chunk, ir)
        return chunk

    # -------- ORF encoding --------
    def _emit_orf(self, chunk: Chunk, fn: IRFunction) -> int:
        """Emit one IR function, returning the ip of its last instruction."""
        last_op_ip = -1
        for inst in fn.instrs:
            op = inst.opcode
            if op is Op.OP_PUSH_CONST:
                # P0-1: the IR carries the literal; add it to the constant pool
                # and emit its (deduplicated) index — identical to the legacy
                # compiler which added the wobble value to the pool.
                const_idx = chunk.add_constant(inst.operand)
                last_op_ip = chunk.emit(op, const_idx, line=inst.line,
                                        codon_index=inst.codon_index)
            elif op is Op.OP_CALL_GENE:
                # Target name resolved at IR build time; emit a placeholder u16
                # address to be back-patched once all gene offsets are known.
                target = inst.operand
                ip = chunk.emit(op, 0, 0, line=inst.line,
                                codon_index=inst.codon_index)
                self._call_sites.append((ip + 1, str(target)))
                last_op_ip = ip
            elif op is Op.OP_JUMP or op is Op.OP_JUMP_IF_ZERO:
                # Relative u16 targets are back-patched by the caller when the
                # target label is inside the same ORF; for IR-embedded jumps the
                # builder never emitted them (only symbolically), so this branch
                # is defensive: emits a zero operand placeholder.
                last_op_ip = chunk.emit_u16(op, 0, line=inst.line,
                                            codon_index=inst.codon_index)
            else:
                nbytes = OP_OPERAND_BYTES.get(op, 0)
                operand = inst.operand if isinstance(inst.operand, int) else 0
                if nbytes == 1:
                    last_op_ip = chunk.emit(op, operand, line=inst.line,
                                            codon_index=inst.codon_index)
                elif nbytes == 0:
                    last_op_ip = chunk.emit(op, line=inst.line,
                                            codon_index=inst.codon_index)
                else:
                    last_op_ip = chunk.emit(op, *([0] * nbytes),
                                            line=inst.line,
                                            codon_index=inst.codon_index)
        return last_op_ip

    @staticmethod
    def _last_op_is_halt(chunk: Chunk, last_op_ip: int) -> bool:
        return last_op_ip >= 0 and chunk.code[last_op_ip] == int(Op.OP_HALT)

    # -------- back-patching --------
    def _patch_calls(self, chunk: Chunk, ir: IRProgram) -> None:
        offsets = chunk.gene_offsets
        for ip, target in self._call_sites:
            offset = offsets.get(target)
            if offset is None:
                raise LookupError(
                    f"CALL_GENE target {target!r} not present in IR offsets; "
                    f"the IR program is inconsistent")
            chunk.code[ip] = (offset >> 8) & 0xFF
            chunk.code[ip + 1] = offset & 0xFF

    def _patch_jumps_to_end(self, chunk: Chunk) -> None:
        """Back-patch inter-gene barriers to the chunk end (relative u16)."""
        end = len(chunk.code)
        for start_plus_1 in self._jump_to_end_sites:
            start = start_plus_1 - 1
            off = end - start - 3
            chunk.code[start_plus_1] = (off >> 8) & 0xFF
            chunk.code[start_plus_1 + 1] = off & 0xFF

    # -------- constant pool tails --------
    @staticmethod
    def _emit_constants(chunk: Chunk, ir: IRProgram) -> None:
        for name, (axiom, rules, angle, step) in ir.lsystems.items():
            chunk.add_constant(("lsystem_axiom", name, axiom))
            chunk.add_constant(("lsystem_rules", name, rules))
            chunk.add_constant(("lsystem_angle", name, angle))
            chunk.add_constant(("lsystem_step", name, step))
        for fn in ir.functions:
            chunk.add_constant(("gene_name", fn.name))


def lower(ir: IRProgram) -> Chunk:
    return IRLowerer().lower(ir)
