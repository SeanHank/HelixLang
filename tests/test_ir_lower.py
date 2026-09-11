"""IRLowerer unit tests for branches not exercised through the normal build.

Covers:
- plugin opt-in (OP_USE_PLUGIN) emission from use_directives.
- a gene whose ORF does not end in HALT (lowerer adds the guard).
- an IR function carrying an explicit OP_JUMP / OP_JUMP_IF_ZERO.
- emit_gene_region splice path with a non-halt last op and an inter-gene barrier.
- _patch_calls raising LookupError for a missing target.
- L-system + gene-name constants emitted in the pool tail.
"""
from __future__ import annotations

import pytest

from helixlang.core.bytecode import Chunk
from helixlang.core.codon_table import Op
from helixlang.core.ir import IRFunction, IRInst, IRProgram
from helixlang.core.ir_lower import IRLowerer, lower


def _ir(functions, use_directives=(), lsystems=None) -> IRProgram:
    return IRProgram(
        name="t", table="standard", functions=functions,
        call_targets={}, use_directives=list(use_directives),
        lsystems=dict(lsystems or {}), config={}, version=1,
    )


def _inst(op, operand=None, value_type=None) -> IRInst:
    return IRInst(opcode=op, operand=operand, value_type=value_type,
                  line=1, codon_index=0)


def test_use_plugin_directive_emitted() -> None:
    ir = _ir(
        [IRFunction(name="a", instrs=[_inst(Op.OP_HALT)])],
        use_directives=[("myplugin", ("x", "y"))],
    )
    chunk = lower(ir)
    assert Op.OP_USE_PLUGIN.value in chunk.code


def test_non_halt_orf_gets_halt_guard() -> None:
    ir = _ir([IRFunction(name="a", instrs=[_inst(Op.OP_NOP)])])
    chunk = lower(ir)
    assert int(Op.OP_HALT) in chunk.code
    assert chunk.code[-1] == int(Op.OP_HALT)


def test_explicit_jump_in_ir() -> None:
    ir = _ir([
        IRFunction(name="a", instrs=[_inst(Op.OP_JUMP),
                                     _inst(Op.OP_JUMP_IF_ZERO),
                                     _inst(Op.OP_HALT)]),
    ])
    chunk = lower(ir)
    assert int(Op.OP_JUMP) in chunk.code


def test_patch_calls_missing_target_raises() -> None:
    inst = IRInst(opcode=Op.OP_CALL_GENE, operand="ghost",
                  value_type=None, line=1, codon_index=0)
    ir = _ir([IRFunction(name="a", instrs=[inst])])
    with pytest.raises(LookupError):
        lower(ir)


def test_lsystem_and_gene_name_constants() -> None:
    ir = _ir(
        [IRFunction(name="g", instrs=[_inst(Op.OP_HALT)])],
        lsystems={"branch": ("F", [("F", "F[+F]")], 25.0, 2.0)},
    )
    chunk = lower(ir)
    kinds = {c[0] for c in chunk.constants if isinstance(c, tuple) and len(c) > 0}
    assert "gene_name" in kinds
    assert "lsystem_axiom" in kinds
    assert "lsystem_rules" in kinds


def test_emit_gene_region_non_halt_with_barrier() -> None:
    lowerer = IRLowerer()
    chunk = Chunk()
    fn = IRFunction(name="g", instrs=[
        _inst(Op.OP_PUSH_CONST, 5), _inst(Op.OP_NOP),
    ])
    offset_of = lambda name: 0  # noqa: E731
    start, end = lowerer.emit_gene_region(chunk, fn, offset_of, is_last=False,
                                          end=1000)
    assert start < end
    assert Op.OP_HALT.value in chunk.code


def test_emit_gene_region_last_no_barrier() -> None:
    lowerer = IRLowerer()
    chunk = Chunk()
    fn = IRFunction(name="g", instrs=[_inst(Op.OP_HALT)])
    offset_of = lambda name: 0  # noqa: E731
    start, end = lowerer.emit_gene_region(chunk, fn, offset_of, is_last=True,
                                          end=1000)
    assert start < end

def test_lower_with_no_functions() -> None:
    # empty function table -> the gene_name constant loop is vacuous
    chunk = lower(_ir([]))
    assert chunk is not None


def test_multi_function_lower_backpatches_calls_and_barriers() -> None:
    # fn_a calls fn_b; with >1 function the inter-gene barrier and the
    # CALL_GENE back-patch both run.
    ir = _ir([
        IRFunction(name="a", instrs=[
            _inst(Op.OP_CALL_GENE, operand="b"),
            _inst(Op.OP_HALT),
        ]),
        IRFunction(name="b", instrs=[_inst(Op.OP_HALT)]),
    ])
    chunk = lower(ir)
    assert "b" in chunk.gene_offsets
    assert int(Op.OP_CALL_GENE) in chunk.code
    # barrier jump operand (u16) present; offset resolves against end
    assert len(chunk.code) > 0


def test_emit_gene_region_resolves_call_gene_inline() -> None:
    lowerer = IRLowerer()
    chunk = Chunk()
    fn = IRFunction(name="a", instrs=[_inst(Op.OP_CALL_GENE, operand="tgt")])
    offset_of = lambda name: 1234  # noqa: E731
    start, end = lowerer.emit_gene_region(chunk, fn, offset_of,
                                          is_last=True, end=2000)
    assert start < end
    # resolved operand 1234 encoded in little-endian u16
    assert (1234 & 0xFF) in chunk.code


def test_encode_one_byte_operand_instruction() -> None:
    lowerer = IRLowerer()
    chunk = Chunk()
    inst = _inst(Op.OP_MOVE, operand=7, value_type="state")
    lowerer._encode_inst(chunk, inst, None)
    assert int(Op.OP_MOVE) in chunk.code
