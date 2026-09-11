"""IROpt edge-case unit tests.

Covers optimizer branches not exercised by the higher-level compiler tests:
- unknown pass name raising ValueError.
- constant folding refusing unterminated fold windows and non-numeric results.
- fold_fail guards: folders that raise, and folders yielding non-int/float.
"""
from __future__ import annotations

import pytest

import helixlang.core.ir_opt as ir_opt_mod
from helixlang.core.codon_table import Op
from helixlang.core.ir import IRInst, IRProgram, IRType
from helixlang.core.ir_opt import IROpt, optimize_program

_FOLDERS = ir_opt_mod._FOLDERS


def _program(*instrs: IRInst) -> IRProgram:
    from helixlang.core.ir import IRFunction

    return IRProgram(
        name="g",
        table="standard",
        functions=[IRFunction(name="g", instrs=list(instrs))],
        call_targets={},
        use_directives=[],
        lsystems={},
        config={},
    )


def _push(v: object) -> IRInst:
    return IRInst(opcode=Op.OP_PUSH_CONST, operand=v,
                  value_type=IRType.NUM, line=1, codon_index=0)


def test_unknown_pass_raises() -> None:
    opt = IROpt()
    with pytest.raises(ValueError):
        opt.optimize(_program(_push(1)), passes=["not_a_pass"])


def test_unknown_pass_name_in_message() -> None:
    opt = IROpt()
    with pytest.raises(ValueError, match="not_a_pass"):
        opt.optimize(_program(_push(1)), passes=["not_a_pass"])


def test_fold_tail_too_short() -> None:
    # A single PUSH has nothing to fold.
    insts = [_push(5)]
    assert IROpt._try_fold_tail(insts) is None


def test_fold_tail_operand_not_numeric_rejected() -> None:
    # A PUSH whose operand is a string blocks folding (guard at line 99).
    bad = IRInst(opcode=Op.OP_PUSH_CONST, operand="abc",
                 value_type=IRType.NUM, line=1, codon_index=0)
    tail = IRInst(opcode=Op.OP_ADD, operand=None, value_type=IRType.NUM,
                  line=1, codon_index=1)
    insts = [bad, tail]
    assert IROpt._try_fold_tail(insts) is None


def test_fold_tail_folder_raising_is_swallowed(monkeypatch) -> None:
    def boom(vs):
        raise OverflowError("boom")

    monkeypatch.setitem(_FOLDERS, Op.OP_ADD, (2, boom, IRType.NUM))
    a, b, tail = _push(1), _push(2), IRInst(
        opcode=Op.OP_ADD, operand=None, value_type=IRType.NUM,
        line=1, codon_index=2)
    assert IROpt._try_fold_tail([a, b, tail]) is None


def test_fold_tail_non_numeric_result_rejected(monkeypatch) -> None:
    def returns_str(vs):
        return "not-a-number"

    monkeypatch.setitem(_FOLDERS, Op.OP_ADD, (2, returns_str, IRType.NUM))
    a, b, tail = _push(1), _push(2), IRInst(
        opcode=Op.OP_ADD, operand=None, value_type=IRType.NUM,
        line=1, codon_index=2)
    assert IROpt._try_fold_tail([a, b, tail]) is None


def test_fold_tail_bool_result_rejected(monkeypatch) -> None:
    def returns_bool(vs):
        return True

    monkeypatch.setitem(_FOLDERS, Op.OP_ADD, (2, returns_bool, IRType.NUM))
    a, b, tail = _push(1), _push(2), IRInst(
        opcode=Op.OP_ADD, operand=None, value_type=IRType.NUM,
        line=1, codon_index=2)
    assert IROpt._try_fold_tail([a, b, tail]) is None


def test_fold_tail_float_result_truncated_to_int(monkeypatch) -> None:
    def returns_float(vs):
        return 4.9

    monkeypatch.setitem(_FOLDERS, Op.OP_ADD, (2, returns_float, IRType.NUM))
    a, b, tail = _push(1), _push(2), IRInst(
        opcode=Op.OP_ADD, operand=None, value_type=IRType.NUM,
        line=1, codon_index=2)
    folded, n = IROpt._try_fold_tail([a, b, tail])
    assert folded.operand == 4
    assert n == 3


def test_optimize_records_runs() -> None:
    opt = IROpt()
    ir = _program(_push(1), IRInst(opcode=Op.OP_HALT, operand=None,
                                   value_type=None, line=1, codon_index=1))
    out = opt.optimize(ir, passes=["fold", "dead", "unreachable"])
    assert out is ir
    assert opt.runs["fold"] == 1


def test_dead_eliminates_swap_swap() -> None:
    swap = IRInst(opcode=Op.OP_SWAP, operand=None, value_type=None,
                  line=1, codon_index=0)
    insts = [swap, swap]
    assert IROpt._pass_dead(insts) == []


def test_unreachable_truncates_at_halt() -> None:
    halt = IRInst(opcode=Op.OP_HALT, operand=None, value_type=None,
                  line=1, codon_index=0)
    dead = IRInst(opcode=Op.OP_ADD, operand=None, value_type=IRType.NUM,
                  line=1, codon_index=1)
    insts = [halt, dead]
    assert IROpt._pass_unreachable(insts) == [halt]


def test_unreachable_noop_without_terminal() -> None:
    adds = [IRInst(opcode=Op.OP_ADD, operand=None, value_type=IRType.NUM,
                   line=1, codon_index=i) for i in range(2)]
    assert IROpt._pass_unreachable(adds) == adds

def _iop(op, vt=IRType.NUM):
    return IRInst(opcode=op, operand=None, value_type=vt, line=1, codon_index=0)


def test_fold_optimize_folds_constant_expression():
    # PUSH a; PUSH b; ADD --> PUSH (a+b); exercises the end-of-run fold loop.
    opt = IROpt()
    prog = opt.optimize(_program(_push(5), _push(3), _iop(Op.OP_ADD)))
    insts = prog.functions[0].instrs
    assert len(insts) == 1
    assert insts[0].opcode is Op.OP_PUSH_CONST and insts[0].operand == 8


def test_fold_tail_arity_exceeds_window(monkeypatch):
    monkeypatch.setitem(_FOLDERS, Op.OP_ADD, (3, lambda vs: vs[0], IRType.NUM))
    a, b, tail = _push(1), _push(2), _iop(Op.OP_ADD)
    # arity 3 but only 2 pushes available -> None at the arity guard
    assert IROpt._try_fold_tail([a, b, tail]) is None


def test_fold_tail_non_push_operand_rejected():
    # operand slot is a DUP, not a PUSH_CONST -> blocked at line 100
    dup = IRInst(opcode=Op.OP_DUP, operand=None, value_type=IRType.NUM,
                 line=1, codon_index=0)
    tail = _iop(Op.OP_ADD)
    assert IROpt._try_fold_tail([dup, _push(1), tail]) is None


def test_dead_eliminates_push_pop():
    insts = [_push(1), _iop(Op.OP_POP)]
    assert IROpt._pass_dead(insts) == []


def test_dead_eliminates_dup_pop_and_read_mem_pop():
    dup = IRInst(opcode=Op.OP_DUP, operand=None, value_type=IRType.NUM,
                 line=1, codon_index=0)
    assert IROpt._pass_dead([dup, _iop(Op.OP_POP)]) == []
    rm = IRInst(opcode=Op.OP_READ_MEM, operand=None, value_type=IRType.NUM,
                line=1, codon_index=0)
    assert IROpt._pass_dead([rm, _iop(Op.OP_POP)]) == []


def test_optimize_program_convenience_function():
    prog = optimize_program(_program(_push(7), _push(2), _iop(Op.OP_SUB)))
    insts = prog.functions[0].instrs
    assert len(insts) == 1
    assert insts[0].operand == 5
