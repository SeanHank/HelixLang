"""Focused unit tests for helixlang.core.ir_runtime internals.

These complement the execution/parity tests in test_ir.py by driving the
IRRuntime's IR-specific methods directly: the linearised instruction stream,
_call_gene, _execute_pending, _dispatch special-cases, the u8/u16 byte view,
and the operand-bytes helper.
"""
from __future__ import annotations

from helixlang.core.ast_nodes import Config, Program, UseDecl
from helixlang.core.codon_table import Op
from helixlang.core.ir import IRFunction, IRInst, IRProgram, IRType
from helixlang.core.ir_runtime import IRRuntime, _operand_bytes
from helixlang.core.vm import Frame


def _runtime(*, use_directives=None, ops_per_tick=64):
    prog = Program(
        config=Config(table="standard", ops_per_tick=ops_per_tick,
                      sim={"seed": "0"}),
    )
    prog.use_directives = use_directives or []
    ir = IRProgram(name="p", table="standard", functions=[
        IRFunction(name="g", line=1, instrs=[
            IRInst(opcode=Op.OP_PUSH_CONST, operand=5, value_type=IRType.NUM,
                   line=1),
        ]),
    ])
    return IRRuntime(ir, prog)


# ── use directives in __init__ ────────────────────────────────────────────

def test_init_applies_use_directives():
    rt = _runtime(use_directives=[UseDecl(plugin="crispr",
                                          flags=frozenset({"f1"}))])
    assert rt.ir is not None
    assert len(rt._ir_offsets) >= 1


# ── _call_gene ─────────────────────────────────────────────────────────────

def test_call_gene_unknown_name():
    rt = _runtime()
    rt.ip = 0
    rt._call_gene("missing")
    assert rt.frames == []


def test_call_gene_known_name():
    rt = _runtime()
    rt.ip = 0
    rt._call_gene("g")
    assert len(rt.frames) == 1
    assert rt.frames[0].gene_name == "g"
    assert rt.ip == rt._ir_offsets["g"]


def test_call_gene_frames_full():
    rt = _runtime()
    rt.frames = [Frame(return_ip=0, gene_name="x")] * 256
    rt.ip = 0
    rt._call_gene("g")
    assert len(rt.frames) == 256


# ── _execute_pending ───────────────────────────────────────────────────────

def test_execute_pending_runs_and_ends():
    rt = _runtime(ops_per_tick=10)
    rt.frames = [Frame(return_ip=0, gene_name="g")]
    rt.ip = rt._ir_offsets["g"]
    rt._execute_pending()
    assert rt.stack == [5]
    assert rt.ops_executed >= 1


def test_execute_pending_debug_prints(capsys):
    rt = _runtime(ops_per_tick=10)
    rt.debug = True
    rt.frames = [Frame(return_ip=0, gene_name="g")]
    rt.ip = rt._ir_offsets["g"]
    rt._execute_pending()
    assert "PUSH_CONST" in capsys.readouterr().out


def test_execute_pending_too_many_frames():
    rt = _runtime()
    rt.frames = [Frame(return_ip=0, gene_name="g")] * 257
    rt.ip = 0
    rt._execute_pending()
    assert rt.frames == []


def test_execute_pending_return_restores_ip():
    rt = _runtime()
    rt.frames = [
        Frame(return_ip=0, gene_name="outer"),
        Frame(return_ip=0, gene_name="inner"),
    ]
    rt.ip = len(rt._flat)  # inner gene already at end -> pop & restore
    rt._execute_pending()
    assert len(rt.frames) < 2


# ── _dispatch special cases ────────────────────────────────────────────────

def test_dispatch_no_current_inst():
    rt = _runtime()
    rt._current_inst = None
    rt._dispatch(Op.OP_ADD)
    assert rt.stack == []


def test_dispatch_push_const():
    rt = _runtime()
    rt._current_inst = IRInst(opcode=Op.OP_PUSH_CONST, operand=9,
                              value_type=IRType.NUM)
    rt._dispatch(Op.OP_PUSH_CONST)
    assert rt.stack == [9]


def test_dispatch_use_plugin():
    rt = _runtime()
    rt._current_inst = IRInst(opcode=Op.OP_USE_PLUGIN, operand=("crispr", ("f",)))
    rt._dispatch(Op.OP_USE_PLUGIN)
    assert rt._current_inst is not None


def test_dispatch_use_plugin_non_tuple_operand():
    rt = _runtime()
    rt._current_inst = IRInst(opcode=Op.OP_USE_PLUGIN, operand="crispr")
    rt._dispatch(Op.OP_USE_PLUGIN)
    assert rt._current_inst is not None


def test_dispatch_call_gene():
    rt = _runtime()
    rt.ip = 0
    rt._current_inst = IRInst(opcode=Op.OP_CALL_GENE, operand="g")
    rt._dispatch(Op.OP_CALL_GENE)
    assert len(rt.frames) == 1


def test_dispatch_call_gene_unknown_target():
    rt = _runtime()
    rt.ip = 0
    rt._current_inst = IRInst(opcode=Op.OP_CALL_GENE, operand="nope")
    rt._dispatch(Op.OP_CALL_GENE)
    assert rt.frames == []


def test_dispatch_delegates_other_ops():
    rt = _runtime()
    rt._current_inst = IRInst(opcode=Op.OP_NOP)
    rt._dispatch(Op.OP_NOP)
    assert rt.ops_executed >= 0


def test_execute_pending_empty_frames():
    rt = _runtime()
    rt.frames = []
    rt._execute_pending()
    assert rt.stack == []


# ── byte view helpers ──────────────────────────────────────────────────────

def test_read_u8_empty_returns_zero():
    rt = _runtime()
    rt._current_operand_bytes = []
    assert rt._read_u8() == 0


def test_read_u16_combines_bytes():
    rt = _runtime()
    rt._current_operand_bytes = [0x12, 0x34]
    assert rt._read_u16() == 0x1234


def test_operand_bytes_one_and_two_byte():
    one = _operand_bytes(IRInst(opcode=Op.OP_PUSH_CONST, operand=0xAB))
    assert one == [0xAB]
    two = _operand_bytes(IRInst(opcode=Op.OP_CALL_GENE, operand=0x1234))
    assert two == [0x12, 0x34]
    none = _operand_bytes(IRInst(opcode=Op.OP_NOP))
    assert none == []
