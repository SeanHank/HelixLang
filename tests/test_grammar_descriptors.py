"""doc/41 §4 acceptance: declarative GrammarDescriptor + #use activation.

Covers the four acceptance points from doc/41 §4.3:

1. A demo plugin (``helixlang.plugins.cardiology``) declares a brand-new
   ``#cardiac_cycle`` keyword with typed fields, a semantic validator and a
   ``.helixc`` round-trip — with zero edits to ``parser.py``/``lexer.py``.
2. ``#use not-builtin-plugin`` alone does not change parse behavior (the
   grammar stays inert).
3. Keyword collisions still raise ``PluginConflictError``.
4. The grammar↔backend bridge: ``grammar_registry.get(kw)`` and
   ``plugin_registry.provider_for_keyword(kw)`` are views of one table.
"""
from __future__ import annotations

import pytest

from helixlang.core.errors import (
    ParseError,
    PluginConflictError,
    SemanticError,
    UnknownKeywordError,
)
from helixlang.core.grammar_registry import (
    FieldSpec,
    GrammarDescriptor,
    compile_descriptor,
    grammar_registry,
)
from helixlang.core.hxbc import decompile, dumps_program, loads_program
from helixlang.core.lexer import Lexer
from helixlang.core.parser import parse_source
from helixlang.core.plugin_registry import Registry
from helixlang.core.semantic import SemanticAnalyzer


def _discover_cardiology() -> None:
    """Ensure the demo provider's grammar is registered (idempotent)."""
    Registry().discover("cardiology")


CARDIAC_SRC = """\
#use cardiology
#cardiac_cycle period=0.8 conduction=compact
#gene name=heart_1
ATG GCT TAA
#end
#config ticks=2 output=stdout
"""


# ---------------------------------------------------------------------------
# descriptor machinery (unit level)
# ---------------------------------------------------------------------------
class TestDescriptorCompile:
    def test_fields_body_compiles_to_generic_parse_hook(self):
        desc = GrammarDescriptor(
            keyword="demo_float",
            fields=(FieldSpec("value", "float", required=True),),
        )
        grammar = compile_descriptor(desc)
        assert grammar.keyword == "demo_float"
        assert grammar.parse is not None
        assert callable(grammar.parse)

    def test_coerces_and_validates_types(self):
        grammar_registry.register_descriptor(GrammarDescriptor(
            keyword="demo_types",
            fields=(
                FieldSpec("n", "int", required=True),
                FieldSpec("ratio", "float", required=True),
                FieldSpec("on", "bool", default="false"),
                FieldSpec("tags", "list"),
                FieldSpec("mapping", "dict"),
            ),
        ))
        src = "#demo_types n=3 ratio=0.5 tags=a,b mapping=x=1,y=2\n"
        prog = parse_source(src)
        entry = prog.sim_extensions["demo_types"][0]
        assert entry == {"n": "3", "ratio": "0.5", "on": "false",
                         "tags": "a,b", "mapping": "x=1,y=2"}
        out = decompile(prog)
        assert "#demo_types mapping=x=1,y=2 n=3 on=false ratio=0.5 tags=a,b" in out

    def test_float_rejects_garbage(self):
        grammar_registry.register_descriptor(GrammarDescriptor(
            keyword="demo_f2", fields=(FieldSpec("value", "float", required=True),)))
        with pytest.raises(ParseError, match="expects a float"):
            parse_source("#demo_f2 value=abc\n")

    def test_int_rejects_garbage(self):
        grammar_registry.register_descriptor(GrammarDescriptor(
            keyword="demo_i2", fields=(FieldSpec("value", "int", required=True),)))
        with pytest.raises(ParseError, match="expects an int"):
            parse_source("#demo_i2 value=1.5\n")

    def test_bool_normalises(self):
        grammar_registry.register_descriptor(GrammarDescriptor(
            keyword="demo_b2", fields=(FieldSpec("flag", "bool", required=True),)))
        prog = parse_source("#demo_b2 flag=on\n")
        assert prog.sim_extensions["demo_b2"][0] == {"flag": "true"}

    def test_required_field_enforced(self):
        grammar_registry.register_descriptor(GrammarDescriptor(
            keyword="demo_r2", fields=(FieldSpec("must", "str", required=True),)))
        with pytest.raises(ParseError, match="requires must= field"):
            parse_source("#demo_r2 optional=1\n")

    def test_default_applied_when_missing(self):
        grammar_registry.register_descriptor(GrammarDescriptor(
            keyword="demo_d2",
            fields=(FieldSpec("shown", "str", default="fallback"),
                    FieldSpec("value", "float"))))
        prog = parse_source("#demo_d2 value=1.0\n")
        assert prog.sim_extensions["demo_d2"][0] == {
            "shown": "fallback", "value": "1"}

    def test_allow_extra_rejects_unknown(self):
        grammar_registry.register_descriptor(GrammarDescriptor(
            keyword="demo_x2", fields=(), allow_extra=False))
        with pytest.raises(ParseError, match="unknown field"):
            parse_source("#demo_x2 nope=1\n")

    def test_allow_extra_keeps_unknown(self):
        grammar_registry.register_descriptor(GrammarDescriptor(
            keyword="demo_e2", fields=(FieldSpec("known", "str"),)))
        prog = parse_source("#demo_e2 known=a surprise=b\n")
        assert prog.sim_extensions["demo_e2"][0] == {"known": "a", "surprise": "b"}


# ---------------------------------------------------------------------------
# demo plugin acceptance (doc/41 §4.3)
# ---------------------------------------------------------------------------
class TestDemoPluginGrammar:
    @pytest.fixture(autouse=True)
    def _discover(self):
        _discover_cardiology()

    def test_keyword_lexes_as_annotation_once_registered(self):
        kinds = [t.kind for t in Lexer("#cardiac_cycle period=0.8").tokens()]
        assert kinds[0] == "ANNOT_START"

    def test_grammar_inert_without_use(self):
        with pytest.raises(UnknownKeywordError, match="use cardiology"):
            parse_source("#cardiac_cycle period=0.8\n")

    def test_use_not_builtin_does_not_activate_grammar(self):
        with pytest.raises(UnknownKeywordError, match="use cardiology"):
            parse_source("#use unknown\n#cardiac_cycle period=0.8\n")

    def test_parse_with_use(self):
        prog = parse_source(CARDIAC_SRC)
        expect = [{"period": "0.8", "conduction": "compact"}]
        assert prog.sim_extensions["cardiac_cycle"] == expect

    def test_required_period(self):
        with pytest.raises(ParseError, match="requires period= field"):
            parse_source("#use cardiology\n#cardiac_cycle conduction=compact\n")

    def test_period_type_enforced(self):
        with pytest.raises(ParseError, match="expects a float"):
            parse_source("#use cardiology\n#cardiac_cycle period=fast\n")

    def test_semantic_validator_runs(self):
        prog = parse_source(
            "#use cardiology\n#cardiac_cycle period=9.0\n#config ticks=1\n")
        with pytest.raises(SemanticError, match="physiological"):
            SemanticAnalyzer(prog).check()

    def test_hxbc_roundtrip(self):
        prog = parse_source(CARDIAC_SRC)
        art = loads_program(dumps_program(prog))
        # hxbc bytecode does not serialize `use` directives (parse-time only);
        # the extension payload it carries must round-trip unchanged.
        assert art.program.sim_extensions == prog.sim_extensions
        out = decompile(art.program)
        assert "#cardiac_cycle conduction=compact period=0.8" in out
        # hxbc's textual decompile keeps #use out of the emitted source (parity
        # with every existing plugin) — the caller re-declares it on reparse.
        reprog = parse_source("#use cardiology\n" + out)
        assert reprog.sim_extensions["cardiac_cycle"] == \
            prog.sim_extensions["cardiac_cycle"]


# ---------------------------------------------------------------------------
# grammar ↔ backend bridge (doc/41 §4.2)
# ---------------------------------------------------------------------------
class TestGrammarBackendBridge:
    def test_keyword_to_owner_via_grammar_registry(self):
        _discover_cardiology()
        grammar = grammar_registry.get("cardiac_cycle")
        assert grammar is not None
        assert grammar.owner == "cardiology"
        assert grammar.requires_use == "cardiology"

    def test_provider_for_keyword_matches(self):
        _discover_cardiology()
        r = Registry()
        r.discover("cardiology")
        prov = r.provider_for_keyword("cardiac_cycle")
        assert prov is not None and prov.name == "cardiology"

    def test_core_keyword_shadowing_raises_conflict(self):
        _discover_cardiology()
        clash = GrammarDescriptor(keyword="gene", fields=(FieldSpec("x", "str"),))
        with pytest.raises(PluginConflictError) as exc:
            grammar_registry.register_descriptor(clash)
        assert exc.value.key == "#gene"
        # the core grammar is untouched
        assert grammar_registry.get("gene").owner_name == "core"


# ---------------------------------------------------------------------------
# Parser.token_hooks (the documented raw-body surface)
# ---------------------------------------------------------------------------
class TestParserTokenHooks:
    def test_hooks_drive_field_collection(self):
        def parse_hook(parser, prog):
            parser.token_hooks.advance()  # consume `#demo_raw_body`
            prog.sim_extensions["hook_scaffold"] = \
                parser.token_hooks.collect_fields(allow_no_end=True)
        desc = GrammarDescriptor(
            keyword="demo_raw_body",
            body="raw",
            parse=parse_hook,
        )
        grammar_registry.register_descriptor(desc)
        prog = parse_source("#demo_raw_body scaffold=pSC101 marker=KAN\n")
        assert prog.sim_extensions["hook_scaffold"] == {
            "scaffold": "pSC101", "marker": "KAN"}

    def test_raw_hook_without_progress_is_a_hard_error(self):
        desc = GrammarDescriptor(
            keyword="demo_raw_stall",
            body="raw",
            parse=lambda parser, prog: None,
        )
        grammar_registry.register_descriptor(desc)
        with pytest.raises(ParseError, match="made no progress"):
            parse_source("#demo_raw_stall x=1\n")
