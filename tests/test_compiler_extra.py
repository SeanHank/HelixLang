"""Compiler unit tests covering direct API branches not hit by CLI smoke tests.

- config property accessor.
- compile_ir with optimize=True (runs the IROpt pass).
- Compiler constructed with a LanguageConfig directly.
- Compiler constructed with table + config together raises.
- _table_name_of for a registered and an unregistered table.
"""
from __future__ import annotations

import pytest

from helixlang.core.codon_table import STANDARD_TABLE, TABLES
from helixlang.core.compiler import Compiler, _table_name_of
from helixlang.core.errors import CompileError
from helixlang.core.language import LanguageConfig
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.core.semantic import SemanticAnalyzer

SRC = "#gene name=a\nATG TGG TAA\n#end\n#config ticks=2 output=stdout\n"


def parse():
    toks = list(Lexer(SRC).tokens())
    prog = Parser(toks, config=LanguageConfig.for_table("standard")).parse()
    SemanticAnalyzer(prog).check()
    return prog


def test_config_property() -> None:
    cfg = LanguageConfig.for_table("standard")
    c = Compiler(config=cfg)
    assert c.config is cfg


def test_compile_ir_with_optimize() -> None:
    c = Compiler(STANDARD_TABLE)
    prog = parse()
    ir, chunk = c.compile_ir(prog, optimize=True)
    assert len(chunk.code) > 0
    assert ir.functions


def test_compile_with_config_only() -> None:
    cfg = LanguageConfig.for_table("standard")
    c = Compiler(config=cfg)
    chunk = c.compile(parse())
    assert len(chunk.code) > 0


def test_table_and_config_mutually_exclusive() -> None:
    cfg = LanguageConfig.for_table("standard")
    with pytest.raises(CompileError):
        Compiler(STANDARD_TABLE, config=cfg)


def test_table_name_of_registered() -> None:
    # STANDARD_TABLE should map back to its registered name.
    name = _table_name_of(STANDARD_TABLE)
    assert name in TABLES


def test_table_name_of_unregistered() -> None:
    assert _table_name_of(dict(STANDARD_TABLE)) == "standard"


def test_build_ir() -> None:
    c = Compiler(STANDARD_TABLE)
    prog = parse()
    ir = c.build_ir(prog)
    assert ir.functions
    assert ir.functions[0].name == "a"
