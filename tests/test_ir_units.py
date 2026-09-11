"""Focused unit tests for helixlang.core.ir helper/accessor methods.

These complement the execution-focused test_ir.py by covering the typed-IR
value model directly: IRType conversions, numeric promotion, IRInst stack
effects, and the disassembly / enumeration helpers on IRFunction / IRProgram.
"""
from __future__ import annotations

import pytest

from helixlang.core.codon_table import Op
from helixlang.core.dimensions import Dimension
from helixlang.core.ir import (
    IRFunction,
    IRInst,
    IRProgram,
    IRType,
    promote_numeric,
)

# ── IRType ──────────────────────────────────────────────────────────────

def test_irtype_from_string_known():
    assert IRType.from_string("gene") is IRType.GENE
    assert IRType.from_string("f64") is IRType.F64
    assert IRType.from_string("bool") is IRType.BOOL
    assert IRType.from_string("void") is IRType.VOID


def test_irtype_from_string_unknown():
    with pytest.raises(ValueError):
        IRType.from_string("bogus")


def test_is_numeric():
    assert IRType.NUM.is_numeric()
    assert IRType.I64.is_numeric()
    assert IRType.F64.is_numeric()
    assert not IRType.GENE.is_numeric()
    assert not IRType.BOOL.is_numeric()


def test_is_boolean():
    assert IRType.BOOL.is_boolean()
    assert not IRType.F64.is_boolean()
    assert not IRType.VOID.is_boolean()


# ── promote_numeric ─────────────────────────────────────────────────────

def test_promote_numeric_widest():
    assert promote_numeric(IRType.F64, IRType.NUM) is IRType.F64
    assert promote_numeric(IRType.NUM, IRType.F64) is IRType.F64
    assert promote_numeric(IRType.I64, IRType.NUM) is IRType.I64
    assert promote_numeric(IRType.NUM, IRType.I64) is IRType.I64
    assert promote_numeric(IRType.NUM, IRType.NUM) is IRType.NUM
    assert promote_numeric(IRType.I64, IRType.F64) is IRType.F64


# ── IRInst stack effects and repr ────────────────────────────────────────

def test_irinst_effects():
    add = IRInst(opcode=Op.OP_ADD, value_type=IRType.NUM)
    assert add.pop_effect() == 2
    assert add.push_effect() == 1
    assert add.net_effect() == -1
    assert not add.is_effect()

    const = IRInst(opcode=Op.OP_PUSH_CONST, operand=42, value_type=IRType.NUM)
    assert const.push_effect() == 1
    assert const.pop_effect() == 0
    assert const.net_effect() == 1


def test_irinst_default_effect_zero():
    inst = IRInst(opcode=Op.OP_HALT)
    assert inst.pop_effect() == 0
    assert inst.push_effect() == 0
    assert inst.net_effect() == 0
    assert inst.is_effect()


def test_irinst_repr_variants():
    i1 = IRInst(opcode=Op.OP_PUSH_CONST, operand=5, value_type=IRType.NUM)
    assert i1.__repr__() == "PUSH_CONST 5 :num"
    i2 = IRInst(opcode=Op.OP_ADD, value_type=IRType.F64)
    assert i2.__repr__() == "ADD :f64"
    i3 = IRInst(opcode=Op.OP_ADD)
    assert i3.__repr__() == "ADD"


def test_irinst_dim():
    inst = IRInst(opcode=Op.OP_PUSH_CONST, operand=1, value_type=IRType.NUM,
                  dim=Dimension(1, 2))
    assert inst.dim is not None


# ── IRFunction ──────────────────────────────────────────────────────────

def test_irfunction_len_and_disassemble():
    fn = IRFunction(name="g", line=3)
    fn.instrs = [
        IRInst(opcode=Op.OP_PUSH_CONST, operand=7, value_type=IRType.NUM),
        IRInst(opcode=Op.OP_ADD, value_type=IRType.NUM),
    ]
    assert len(fn) == 2
    lines = fn.disassemble().splitlines()
    assert len(lines) == 2


# ── IRProgram ───────────────────────────────────────────────────────────

def test_irprogram_gene_names_and_registers():
    p = IRProgram(name="p", table="standard")
    p.functions = [
        IRFunction(name="a"),
        IRFunction(name="b"),
    ]
    assert p.gene_names() == ["a", "b"]
    assert p.num_registers() == 0


def test_irprogram_version_set_in_post_init():
    p = IRProgram(version=999)
    assert p.version == 1


def test_irprogram_disassemble():
    fn = IRFunction(name="g")
    fn.instrs = [IRInst(opcode=Op.OP_PUSH_CONST, operand=1,
                        value_type=IRType.NUM)]
    p = IRProgram(name="prog", functions=[fn])
    out = p.disassemble()
    assert "program 'prog'" in out
    assert "gene g:" in out


def test_irprogram_patch_gene():
    p = IRProgram(functions=[
        IRFunction(name="a"),
        IRFunction(name="b"),
    ])
    new_b = IRFunction(name="b")
    new_b.instrs = [IRInst(opcode=Op.OP_NOP)]
    p.patch_gene("b", new_b)
    assert p.functions[1] is new_b


def test_irprogram_patch_gene_missing_raises():
    p = IRProgram(functions=[IRFunction(name="a")])
    with pytest.raises(KeyError):
        p.patch_gene("nope", IRFunction(name="nope"))
