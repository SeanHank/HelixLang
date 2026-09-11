"""IRBuilder direct unit tests (doc/38 §3).

Covers pure-builder branches not exercised by downstream compiler tests:
- use_table chaining and the _table is None default path.
- build_function with no preset table.
- _function raising CompileError on an unknown codon.
- _snapshot_config handling non-dataclass program config.
- _call_target_resolver raising on a missing target.
"""
from __future__ import annotations

import pytest

from helixlang.core.codon_table import STANDARD_TABLE, Op
from helixlang.core.errors import CompileError
from helixlang.core.ir_builder import IRBuilder
from helixlang.core.language import LanguageConfig
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.core.semantic import SemanticAnalyzer

SIMPLE = """\
#gene name=a
ATG TGG TAA
#end
#gene name=b call_target=a
ATG CGC TGG TAA
#end
#config ticks=2 output=stdout
"""


def parse(src: str):
    toks = list(Lexer(src).tokens())
    prog = Parser(toks, config=LanguageConfig.for_table("standard")).parse()
    SemanticAnalyzer(prog).check()
    return prog


def test_use_table_chains() -> None:
    builder = IRBuilder()
    ret = builder.use_table(STANDARD_TABLE)
    assert ret is builder
    assert builder._table is STANDARD_TABLE


def test_build_uses_default_table_when_none() -> None:
    program = parse(SIMPLE)
    ir = IRBuilder().build(program)
    assert ir.functions
    assert ir.name == ""


def test_build_function_without_preset_table() -> None:
    program = parse(SIMPLE)
    gene = program.genes[0]
    fn = IRBuilder().build_function(program, gene)
    assert fn.name == gene.name


def test_unknown_codon_raises_compile_error() -> None:
    program = parse(SIMPLE)
    program.genes[0].orf[1].seq = "ZZZ"
    with pytest.raises(CompileError):
        IRBuilder().build(program)


def test_call_resolver_missing_target_raises() -> None:
    # Point a's call at an undefined gene entirely: craft a program where
    # a calls a name not among the defined genes.
    src = "#gene name=a call_target=ghost\nATG CGC TGG TAA\n#end"
    prog = parse(src)
    with pytest.raises(CompileError):
        IRBuilder().build(prog)


def test_snapshot_config_non_dataclass() -> None:
    program = parse(SIMPLE)
    # program.config is parsed as an object; monkeypatch a plain dict.
    original = program.config
    program.config = {"a": 1, "_hidden": 2}  # type: ignore[assignment]
    try:
        ir = IRBuilder().build(program)
        assert ir.config.get("a") == 1
    finally:
        program.config = original


def test_lsystem_copied_into_ir() -> None:
    src = """\
#gene name=a
ATG TGG TAA
#end
#lsystem name=branch axiom=F rules=0:F->F[+F] angle=25 step=2.0
#config ticks=2 output=stdout
"""
    program = parse(src)
    ir = IRBuilder().build(program)
    assert "branch" in ir.lsystems


def test_read_mem_opcodes_typed_as_metab() -> None:
    src = "#gene name=a\nATG ATC TGG TAA\n#end"
    program = parse(src)
    ir = IRBuilder().build(program)
    types = {i.value_type for fn in ir.functions for i in fn.instrs}
    from helixlang.core.ir import IRType

    assert IRType.METAB in types

def test_type_of_push_const_returns_num() -> None:
    from helixlang.core.ir import IRType
    from helixlang.core.ir_builder import _type_of
    assert _type_of(Op.OP_PUSH_CONST) is IRType.NUM


def test_build_with_explicit_table_takes_none_false_branch() -> None:
    program = parse(SIMPLE)
    ir = IRBuilder(STANDARD_TABLE).build(program)
    assert ir.functions


def test_build_function_with_explicit_table_takes_none_false_branch() -> None:
    program = parse(SIMPLE)
    gene = program.genes[0]
    fn = IRBuilder(STANDARD_TABLE).build_function(program, gene)
    assert fn.name == gene.name
