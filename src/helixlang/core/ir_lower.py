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

from collections.abc import Callable

from helixlang.core.bytecode import Chunk
from helixlang.core.codon_table import OP_OPERAND_BYTES, Op
from helixlang.core.ir import IRFunction, IRInst, IRProgram


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
    def _encode_inst(self, chunk: Chunk, inst: IRInst,
                     resolve_call: Callable[[str], int] | None) -> int:
        """Encode one IR instruction, returning its start ip.

        ``resolve_call`` is ``None`` on the full-lower path (a zero placeholder
        is emitted and the site recorded for back-patching) or a gene-name →
        absolute-offset function on the incremental splice path, where the
        target offset of every ``OP_CALL_GENE`` is already known.  Both modes
        emit identical bytes for the same IR, so the splice of
        :mod:`helixlang.core.incr` is byte-faithful to :meth:`lower`.
        """
        op = inst.opcode
        if op is Op.OP_PUSH_CONST:
            # P0-1: the IR carries the literal; add it to the constant pool
            # and emit its (deduplicated) index.
            const_idx = chunk.add_constant(inst.operand)
            return chunk.emit(op, const_idx, line=inst.line,
                              codon_index=inst.codon_index)
        if op is Op.OP_CALL_GENE:
            target = str(inst.operand)
            if resolve_call is None:
                ip = chunk.emit(op, 0, 0, line=inst.line,
                                codon_index=inst.codon_index)
                self._call_sites.append((ip + 1, target))
                return ip
            return chunk.emit_u16(op, resolve_call(target), line=inst.line,
                                  codon_index=inst.codon_index)
        if op is Op.OP_JUMP or op is Op.OP_JUMP_IF_ZERO:
            # IR-embedded jumps carry no operand (the builder never emits
            # them symbolically) -- defensive zero placeholder.
            return chunk.emit_u16(op, 0, line=inst.line,
                                  codon_index=inst.codon_index)
        nbytes = OP_OPERAND_BYTES.get(op, 0)
        operand = inst.operand if isinstance(inst.operand, int) else 0
        if nbytes == 1:
            return chunk.emit(op, operand, line=inst.line,
                              codon_index=inst.codon_index)
        if nbytes == 0:
            return chunk.emit(op, line=inst.line,
                              codon_index=inst.codon_index)
        return chunk.emit(op, *([0] * nbytes), line=inst.line,
                          codon_index=inst.codon_index)

    def _emit_orf(self, chunk: Chunk, fn: IRFunction) -> int:
        """Emit one IR function, returning the ip of its last instruction."""
        last_op_ip = -1
        for inst in fn.instrs:
            last_op_ip = self._encode_inst(chunk, inst, None)
        return last_op_ip

    @staticmethod
    def _last_op_is_halt(chunk: Chunk, last_op_ip: int) -> bool:
        return last_op_ip >= 0 and chunk.code[last_op_ip] == int(Op.OP_HALT)

    def emit_gene_region(self, chunk: Chunk, fn: IRFunction,
                         offset_of: Callable[[str], int],
                         is_last: bool, end: int) -> tuple[int, int]:
        """Emit one gene's complete code region with every reference resolved.

        Produces exactly the bytes :meth:`lower` would emit for ``fn``: the
        ORF, the trailing HALT guard, and (unless ``is_last``) the inter-gene
        jump barrier to ``end``.  ``offset_of`` supplies the absolute byte
        offset of every ``OP_CALL_GENE`` target and ``end`` the chunk end —
        both are fixed on the incremental splice path (``core/incr.py``) by
        the invariant that a splice-safe edit preserves every gene offset.

        Returns the ``(start, end)`` ips of the region appended to ``chunk``.
        """
        start = len(chunk.code)
        last_op_ip = -1
        for inst in fn.instrs:
            last_op_ip = self._encode_inst(chunk, inst, offset_of)
        if not self._last_op_is_halt(chunk, last_op_ip):
            chunk.emit(Op.OP_HALT, line=0, codon_index=-1)
        if not is_last:
            # Inter-gene barrier: relative to its own ip, exactly as lower's
            # ``_patch_jumps_to_end`` back-patches (`end - barrier_ip - 3`).
            barrier_ip = len(chunk.code)
            chunk.emit_u16(Op.OP_JUMP, end - barrier_ip - 3, line=0,
                           codon_index=-1)
        return start, len(chunk.code)

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
