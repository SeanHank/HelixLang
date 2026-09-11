"""Grammar registry tests (doc/38 §5).

Covers:
- the registry holds the full core grammar table (parse/dispatch, and the
  decompile hooks that used to live as special cases in ``hxbc.decompile``)
- registry-driven dispatch preserves established language behavior
- a *test grammar* (``#carrier``) becomes parse/compile/decompile/load
  compatible with no edits to ``parser.py`` or ``hxbc.py``
- the shipped example plugin grammar (``#vector``) round-trips too
- keyword collision raises ``PluginConflictError``
- ``--info`` ("helixc info") lists registered grammars
- grammar ``validate`` hooks run in the semantic phase
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest

import helixlang.plugins.annotation.vector as vector_plugin  # noqa: F401
from helixlang.core import hxbc
from helixlang.core.codon_table import get_table
from helixlang.core.compiler import Compiler
from helixlang.core.errors import (
    ParseError,
    PluginConflictError,
    SemanticError,
    UnknownKeywordError,
)
from helixlang.core.grammar_registry import (
    AFTER_SIM,
    BEFORE_SIM,
    AnnotationGrammar,
    FieldSpec,
    GrammarDescriptor,
    GrammarRegistry,
    compile_descriptor,
    dict_entry_decompile,
    ensure_core_grammars,
    fmt_fields,
    fmt_float,
    fmt_str,
    gem_inline_decompile,
    grammar_registry,
    prefix_decompile,
    sim_entry_decompile,
)
from helixlang.core.lexer import Lexer
from helixlang.core.parser import BIO_INSTRUCTION_KINDS, Parser, parse_source
from helixlang.core.semantic import SemanticAnalyzer
from helixlang.plugins.runtime.seq_utils import stop_codons_from_table

STANDARD = get_table("standard")

# Expected core keyword set: the 27 structural annotation kinds + the 7
# biological instruction kinds (P0-1.1).
STRUCT_KEYWORDS = {
    "promoter", "gene", "regulate", "lsystem", "field", "morphogen",
    "config", "type", "media", "enzyme", "reaction", "metabolite", "sim",
    "genome", "species", "patch", "gem",
    "person", "trait", "disease", "disease_gene", "disease_metabolite",
    "drug", "pd_effect", "qsp_binding", "endocrine_config", "immune_config",
    "tumor_biopsy",
}


def _parse(src: str) -> tuple[Any, Any]:
    """Parse + semcheck + compile, mirroring tests/test_helixc.py helpers."""
    tokens = list(Lexer(src).tokens())
    prog = Parser(tokens, stop_codons=set(stop_codons_from_table(STANDARD))).parse()
    SemanticAnalyzer(prog).check()
    return prog, Compiler(STANDARD).compile(prog)


def cli(argv: list[str]) -> tuple[int, str]:
    """Run helixlang.cli.main with stdout captured (like test_helixc)."""
    from helixlang.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


# ---------------------------------------------------------------------------
# The doc/38 §5 acceptance test grammar: #carrier scaffold=... insert=...
# ---------------------------------------------------------------------------
def _parse_carrier(parser: Any, prog: Any) -> None:
    t = parser._advance()  # ANNOT_START
    fields = parser._collect_fields_until_block_end(allow_no_end=True)
    if not fields.get("scaffold"):
        raise ParseError("#carrier requires scaffold= field", line=t.line)
    parser._append_sim_list(prog, "carriers", fields)


CARRIER_GRAMMAR = AnnotationGrammar(
    keyword="carrier",
    parse=_parse_carrier,
    decompile=sim_entry_decompile("carriers", "carrier"),
    list_valued_keys=frozenset({"carriers"}),
    core=False,
    owner="test-carrier",
)

grammar_registry.register(CARRIER_GRAMMAR)

CARRIER_SRC = """\
#carrier scaffold=pUC19 insert=TP53
#gene name=TP53
ATG GCT TAA
#end
#config ticks=2 output=stdout
"""


# ---------------------------------------------------------------------------
# core table
# ---------------------------------------------------------------------------
class TestCoreRegistry:
    def test_all_core_grammars_registered(self):
        ensure_core_grammars()
        for keyword in STRUCT_KEYWORDS | set(BIO_INSTRUCTION_KINDS):
            g = grammar_registry.get(keyword)
            assert g is not None, f"missing grammar #{keyword}"
            assert g.parse is not None, f"#{keyword} has no parse hook"
            assert g.core is True, f"#{keyword} must be a core grammar"

    def test_bio_kinds_match_parser_constant(self):
        ensure_core_grammars()
        for kind in BIO_INSTRUCTION_KINDS:
            assert grammar_registry.contains(kind)

    def test_decompile_hooks_present_for_roundtrip_grammars(self):
        ensure_core_grammars()
        for keyword in ("person", "trait", "disease", "drug", "disease_gene",
                        "disease_metabolite", "pd_effect", "qsp_binding",
                        "endocrine_config", "immune_config", "tumor_biopsy",
                        "gene", "gem"):
            assert grammar_registry.get(keyword).decompile is not None, keyword

    def test_dispatch_preserved_kitchen_sink(self):
        src = """\
#promoter name=p1 strength=0.7
#gene name=g1 promoter=p1
ATG GCT TAA
#end
#regulate p1 -> g1 strength=0.8
#lsystem name=tree axiom=F rules=0:F->F[+F]F;1:F->FF angle=25 step=1.0
#field size=16 F=0.035 k=0.065 Du=0.16 Dv=0.08
#morphogen gene=g1 channel=V gain=0.2
#media nutrient=GLC concentration=10.0
#enzyme gene=g1 reaction=CS kcat=100
#reaction id=R1 substrate=A product=B lower_bound=-10 upper_bound=10
#metabolite name=glcD init=0.5
#type g1=Protein
#config ticks=5 output=stdout
#sim kind=demo
"""
        prog, chunk = _parse(src)
        assert [g.name for g in prog.genes] == ["g1"]
        assert chunk is not None

    def test_unknown_keyword_is_hard_error(self):
        with pytest.raises(UnknownKeywordError):
            parse_source("#export gene=foo\n")

    def test_gem_inline_dna_is_grammar_property(self):
        src = """\
#gem organism=e_coli_k12
#gene_a
ATG AAA
#gene_b
CTG TAA
#end
"""
        prog = parse_source(src)
        assert prog.sim_extensions["gem_inline_genes"] == [
            ["gene_a", "ATGAAA"], ["gene_b", "CTGTAA"]]
        assert prog.sim_extensions["gem_inline_genome"] == "ATGAAACTGTAA"
        # and it round-trips byte-for-byte through the gem decompile hook
        out = hxbc.decompile(prog)
        assert "#gem organism=e_coli_k12" in out
        assert "#gene_a" in out and "#gene_b" in out
        assert out.rstrip().endswith("#end")
        prog2 = parse_source(out)
        assert prog2.sim_extensions["gem_inline_genes"] == \
            prog.sim_extensions["gem_inline_genes"]

    def test_ensure_core_grammars_idempotent(self):
        ensure_core_grammars()
        ensure_core_grammars()  # must not raise / duplicate


# ---------------------------------------------------------------------------
# registration / conflict
# ---------------------------------------------------------------------------
class TestRegistration:
    def test_conflicting_keyword_raises(self):
        clash = AnnotationGrammar(keyword="carrier", core=False, owner="clash")
        with pytest.raises(PluginConflictError) as exc:
            grammar_registry.register(clash)
        assert exc.value.key == "#carrier"
        assert "clash" in str(exc.value)
        # the original grammar is untouched
        assert grammar_registry.get("carrier").owner_name == "test-carrier"

    def test_re_register_same_object_is_noop(self):
        grammar_registry.register(CARRIER_GRAMMAR)  # no error
        assert grammar_registry.get("carrier") is CARRIER_GRAMMAR


# ---------------------------------------------------------------------------
# acceptance: parse/compile/decompile/load with no parser.py / hxbc.py edits
# ---------------------------------------------------------------------------
class TestPluginGrammarAcceptance:
    def test_lexer_recognizes_registered_keyword(self):
        # unregistered idents do not lex as annotations
        assert [t.kind for t in Lexer("#plasmid").tokens()] == \
            ["GENE_ID", "EOF"]
        # a registered grammar keyword lexes as an annotation
        kinds = [t.kind for t in Lexer("#carrier scaffold=pUC19").tokens()]
        assert kinds[0] == "ANNOT_START"

    def test_parse(self):
        prog, _ = _parse(CARRIER_SRC)
        assert prog.sim_extensions["carriers"] == [
            {"scaffold": "pUC19", "insert": "TP53"}]

    def test_parse_requires_scaffold(self):
        with pytest.raises(ParseError, match="scaffold"):
            parse_source("#carrier insert=TP53\n")

    def test_compile(self):
        _, chunk = _parse(CARRIER_SRC)
        assert chunk is not None

    def test_decompile(self):
        prog, _ = _parse(CARRIER_SRC)
        out = hxbc.decompile(prog)
        assert "#carrier insert=TP53 scaffold=pUC19" in out

    def test_load_roundtrip(self):
        prog, chunk = _parse(CARRIER_SRC)
        data = hxbc.dumps_program(prog, chunk=chunk)
        art = hxbc.loads_program(data)
        assert art.program == prog
        assert "#carrier insert=TP53 scaffold=pUC19" in hxbc.decompile(art.program)

    def test_compile_load_via_file(self, tmp_path: Path):
        src = tmp_path / "carrier.helix"
        art_path = tmp_path / "carrier.helixc"
        src.write_text(CARRIER_SRC)
        hxbc.compile_file(src, art_path)
        art = hxbc.load_program(art_path)
        out = hxbc.decompile(art.program)
        assert "#carrier insert=TP53 scaffold=pUC19" in out
        # reparse of the decompiled text preserves the plugin extension
        reprog = parse_source(out)
        assert reprog.sim_extensions["carriers"] == [
            {"scaffold": "pUC19", "insert": "TP53"}]

    def test_shipped_vector_example_roundtrips(self):
        src = """\
#vector gene=TP53 plasmid=pUC19 payload_len=3821
#gene name=TP53
ATG GCT TAA
#end
#config ticks=2 output=stdout
"""
        prog, _ = _parse(src)
        assert prog.sim_extensions["vectors"] == [
            {"gene": "TP53", "plasmid": "pUC19", "payload_len": "3821"}]
        art = hxbc.loads_program(hxbc.dumps_program(prog))
        assert art.program == prog
        out = hxbc.decompile(art.program)
        assert "#vector gene=TP53 payload_len=3821 plasmid=pUC19" in out

    def test_scalar_sim_key_falls_through_to_fallback(self):
        """A scalar spelling of a list-valued key keeps its #sim round-trip."""
        src = """\
#gene name=g1
ATG GCT TAA
#end
#config ticks=2 output=stdout
#sim genes=near:12.0,mid:6.0
"""
        prog, _ = _parse(src)
        out = hxbc.decompile(prog)
        assert "#sim genes=near:12.0,mid:6.0" in out
        assert "#gene genes=near:12.0,mid:6.0" not in out
        assert parse_source(out).sim_extensions["genes"] == \
            prog.sim_extensions["genes"]


# ---------------------------------------------------------------------------
# decompile-migration regression: prefix / list / dict grammars
# ---------------------------------------------------------------------------
class TestDecompileMigration:
    def test_person_trait_disease_prefix_roundtrip(self):
        src = """\
#gene name=g1
ATG GCT TAA
#end
#config ticks=2 output=stdout
#person age=40 sex=M condition="type 2 diabetes"
#trait bmi=high activity=low
#disease name=cancer severity=0.7
"""
        prog, _ = _parse(src)
        out = hxbc.decompile(prog)
        assert "#person age=40 condition=\"type 2 diabetes\" sex=M" in out
        assert "#trait activity=low bmi=high" in out
        assert "#disease name=cancer severity=0.7" in out
        reprog = parse_source(out)
        assert reprog.sim_extensions == prog.sim_extensions

    def test_dict_valued_grammar_roundtrip(self):
        src = """\
#gene name=g1
ATG GCT TAA
#end
#config ticks=2 output=stdout
#tumor_biopsy lesion_type=LCIS er=positive her2=negative
"""
        prog, _ = _parse(src)
        out = hxbc.decompile(prog)
        assert "#tumor_biopsy er=positive her2=negative lesion_type=LCIS" in out
        assert parse_source(out).sim_extensions == prog.sim_extensions

    def test_list_valued_annotations_roundtrip(self):
        src = """\
#gene name=g1
ATG GCT TAA
#end
#config ticks=2 output=stdout
#drug name=paclitaxel dose=175 dosed_via=iv
#drug name=carboplatin dose=5 auc=4
#pd_effect drug=paclitaxel effect=neutropenia probability=0.3
#qsp_binding drug=paclitaxel kind=competitive target=beta_tubulin kd=0.5
#endocrine_config axis=diabetes severity=0.7
#immune_config checkpoint=pd1
#disease_gene gene=BRCA1 type=downregulate
#disease_metabolite id=Lactate type=upregulate
"""
        prog, _ = _parse(src)
        out = hxbc.decompile(prog)
        assert "#drug dose=175 dosed_via=iv name=paclitaxel" in out
        assert "#drug auc=4 dose=5 name=carboplatin" in out
        assert "#pd_effect drug=paclitaxel effect=neutropenia " \
               "probability=0.3" in out
        assert "#qsp_binding drug=paclitaxel kd=0.5 kind=competitive " \
               "target=beta_tubulin" in out
        assert "#endocrine_config axis=diabetes severity=0.7" in out
        assert "#immune_config checkpoint=pd1" in out
        assert "#disease_gene gene=BRCA1 type=downregulate" in out
        assert "#disease_metabolite id=Lactate type=upregulate" in out
        reprog = parse_source(out)
        assert reprog.sim_extensions == prog.sim_extensions

    def test_list_valued_grammar_order_after_sim_fallback(self):
        """List-valued grammars are registered in the AFTER_SIM section."""
        assert grammar_registry.get("gene").sim_section == AFTER_SIM
        assert grammar_registry.get("drug").sim_section == AFTER_SIM
        assert grammar_registry.get("tumor_biopsy").sim_section == AFTER_SIM


# ---------------------------------------------------------------------------
# semantic validation hooks
# ---------------------------------------------------------------------------
def _parse_demo(parser: Any, prog: Any) -> None:
    parser._advance()  # ANNOT_START
    flowers = parser._collect_fields_until_block_end(allow_no_end=True)
    if "flag" in flowers:
        prog.sim_extensions["demo_flags"] = flowers["flag"]


def _validate_demo(analyzer: Any, prog: Any) -> None:
    if prog.sim_extensions.get("demo_flags") == "boom":
        raise SemanticError("#demo rejects flag=boom")


DEMO_GRAMMAR_A = AnnotationGrammar(
    keyword="demo_flag", parse=_parse_demo, validate=_validate_demo,
    extension_keys=frozenset({"demo_flags"}), core=False, owner="test-demo")


class TestGrammarValidators:
    @pytest.fixture(autouse=True)
    def _register_demo(self):
        try:
            grammar_registry.register(DEMO_GRAMMAR_A)
            yield
        finally:  # pytest doesn't unregister; validate is inert unless present
            pass

    def test_validate_raises_in_semantic_phase(self):
        prog = parse_source("#demo_flag flag=boom\n")
        with pytest.raises(SemanticError, match="boom"):
            SemanticAnalyzer(prog).check()

    def test_validate_passes_when_not_violated(self):
        prog = parse_source("#demo_flag flag=calm\n")
        SemanticAnalyzer(prog).check()  # no raise
        assert prog.sim_extensions["demo_flags"] == "calm"


# ---------------------------------------------------------------------------
# helixc info
# ---------------------------------------------------------------------------
class TestInfo:
    def test_info_lists_registered_grammars(self):
        rc, out = cli(["--info"])
        assert rc == 0
        assert "#promoter" in out and "core" in out
        assert "#crispr" in out
        assert "#carrier" in out and "plugin" in out
        assert "#vector" in out and "vector" in out

    def test_info_requires_no_source(self):
        rc, out = cli(["--info"])
        assert rc == 0 and out.count("\n") >= 30


# ---------------------------------------------------------------------------
# Coverage edge cases (internal formatting + descriptor/registry internals)
# ---------------------------------------------------------------------------
class TestFormattingInternals:
    def test_fmt_float_integer_repr(self):
        assert fmt_float(2) == "2"  # repr "2" has no "."
        assert fmt_float(2.5) == "2.5"
        assert fmt_float(2.0) == "2"

    def test_fmt_str_quoting(self):
        assert fmt_str("plain") == "plain"
        assert fmt_str("has space") == '"has space"'
        assert fmt_str("has=eq") == '"has=eq"'

    def test_fmt_fields_quote_spaces(self):
        out = fmt_fields({"a": "x y", "b": "z"})
        assert out.count('"') == 2
        assert "a=\"x y\"" in out

    def test_fmt_fields_keeps_existing_quote(self):
        out = fmt_fields({"k": '"already"'})
        assert '"already"' in out


class TestDecompileClosures:
    def _prog(self, **kw):
        from types import SimpleNamespace
        return SimpleNamespace(sim_extensions=kw)

    def test_sim_entry_decompile_non_list_empty(self):
        fn = sim_entry_decompile("k", "kw")
        assert fn(self._prog(k="scalar")) == []
        assert fn(self._prog()) == []

    def test_sim_entry_decompile_skips_non_dict(self):
        fn = sim_entry_decompile("k", "kw")
        assert fn(self._prog(k=[{"a": "1"}, "not_a_dict"])) == ["#kw a=1"]

    def test_dict_entry_decompile_empty_dict_returns_empty(self):
        fn = dict_entry_decompile("k", "kw")
        assert fn(self._prog(k={})) == []
        assert fn(self._prog(k="str")) == []

    def test_prefix_decompile(self):
        fn = prefix_decompile("person_", "person")
        assert fn(self._prog(person_age="40", person_sex="M",
                             person_flags=["x"])) == ["#person age=40 sex=M"]
        assert fn(self._prog()) == []

    def test_gem_inline_decompile_empty(self):
        fn = gem_inline_decompile()
        assert fn(self._prog()) == []
        assert fn(self._prog(gem_inline_genes=[])) == []

    def test_gem_inline_decompile_populated(self):
        fn = gem_inline_decompile()
        lines = fn(self._prog(
            gem_inline_genes=[("g1", "atg aaa"), ("g2", "ctg taa")],
            gem_organism="e_coli_k12", gem_duration="60"))
        assert lines[0].startswith("#gem organism=e_coli_k12 duration=60")
        assert "#g1" in lines and "#g2" in lines
        assert lines[-1] == "#end"

    def test_gem_inline_decompile_long_gene_wraps_and_skips_bad_entry(self):
        fn = gem_inline_decompile()
        long_seq = "atg" * 40  # 120 nt -> codons exceed 78-col line width
        lines = fn(self._prog(
            gem_inline_genes=[(1, 2, 3), ("g3", long_seq)]))
        assert "#g3" in lines
        assert not any(len(line) > 78 for line in lines)
        assert lines[0] == "#gem organism=e_coli_k12"  # no params fallback

    def test_gem_inline_decompile_empty_gene_seq(self):
        fn = gem_inline_decompile()
        lines = fn(self._prog(gem_inline_genes=[("g0", "")]))
        assert "#g0" in lines
        assert "#gem organism=e_coli_k12" in lines
        assert lines[-1] == "#end"


class TestCoerceField:
    def _coerce(self, **kw):
        from helixlang.core.grammar_registry import FieldSpec
        spec = FieldSpec(**kw)
        return spec

    def test_float_coerce_and_error(self):
        from helixlang.core.grammar_registry import _coerce_field
        assert _coerce_field(self._coerce(key="x", type="float"), "2.50",
                             keyword="kw") == "2.5"
        with pytest.raises(ParseError):
            _coerce_field(self._coerce(key="x", type="float"), "abc", keyword="kw")

    def test_int_coerce_and_error(self):
        from helixlang.core.grammar_registry import _coerce_field
        assert _coerce_field(self._coerce(key="x", type="int"), "42",
                             keyword="kw") == "42"
        with pytest.raises(ParseError):
            _coerce_field(self._coerce(key="x", type="int"), "4.5", keyword="kw")

    def test_bool_coerce_true_false_and_error(self):
        from helixlang.core.grammar_registry import _coerce_field
        for raw in ("TRUE", "1", "yes", "on"):
            assert _coerce_field(
                self._coerce(key="x", type="bool"), raw, keyword="kw") == "true"
        for raw in ("FALSE", "0", "no", "off"):
            assert _coerce_field(
                self._coerce(key="x", type="bool"), raw, keyword="kw") == "false"
        with pytest.raises(ParseError):
            _coerce_field(self._coerce(key="x", type="bool"), "maybe", keyword="kw")

    def test_list_and_dict_coerce(self):
        from helixlang.core.grammar_registry import _coerce_field
        assert _coerce_field(self._coerce(key="x", type="list"), " a , b ,,c ",
                             keyword="kw") == "a,b,c"
        assert _coerce_field(self._coerce(key="x", type="dict"),
                             " k1 = v1 , k2=v2 ", keyword="kw") == "k1=v1,k2=v2"

    def test_dict_coerce_with_empty_chunk(self):
        from helixlang.core.grammar_registry import _coerce_field
        assert _coerce_field(self._coerce(key="x", type="dict"),
                             "k1=v1,,k2=v2", keyword="kw") == "k1=v1,k2=v2"

    def test_dict_coerce_without_equals_raises(self):
        from helixlang.core.grammar_registry import _coerce_field
        with pytest.raises(ParseError):
            _coerce_field(self._coerce(key="x", type="dict"), "k1v1", keyword="kw")

    def test_quoted_string_passthrough(self):
        from helixlang.core.grammar_registry import _coerce_field
        spec = self._coerce(key="x", type="str")
        assert _coerce_field(spec, '"hello world"', keyword="kw") == "hello world"

    def test_str_default(self):
        from helixlang.core.grammar_registry import _coerce_field
        assert _coerce_field(self._coerce(key="x", type="str"), "rock",
                             keyword="kw") == "rock"


class TestDescriptorInternals:
    def test_compile_descriptor_raw_and_owner(self):
        def raw_parse(parser, prog):
            prog.sim_extensions["rk"] = "raw"
        desc = GrammarDescriptor(
            keyword="rawkw", body="raw", parse=raw_parse,
            owner="pluginX", validate=None)
        grammar = compile_descriptor(desc)
        assert grammar.owner_name == "pluginX"
        assert grammar.keyword == "rawkw"

    def test_compile_descriptor_section_target_keys(self):
        desc = GrammarDescriptor(keyword="seckey", target="section")
        grammar = compile_descriptor(desc)
        # section target -> no sim_extensions ownership
        assert grammar.owns_key("seckey", "x") is False

    def test_annotation_grammar_owns_key_and_has_data(self):
        g = AnnotationGrammar(keyword="k", list_valued_keys=frozenset({"k"}))
        assert g.owns_key("k", ["x"]) is True
        assert g.owns_key("k", "scalar") is False
        assert g.has_data(self.prog(ext={"k": ["x"]})) is True
        assert g.has_data(self.prog(ext={"k": "scalar"})) is False

    def prog(self, ext):
        from types import SimpleNamespace
        return SimpleNamespace(sim_extensions=ext)

    def test_annotation_grammar_prefix_owning(self):
        g = AnnotationGrammar(keyword="person", extension_prefixes=frozenset({"person_"}))
        assert g.owns_key("person_age", 40) is True
        assert g.has_data(self.prog(ext={"person_age": 40})) is True


class TestRegistryEdgeCases:
    def test_register_descriptor_returns_grammar(self):
        reg = GrammarRegistry()
        desc = GrammarDescriptor(keyword="edgekw", owner="pluginEDGE")
        grammar = reg.register_descriptor(desc)
        assert grammar.keyword == "edgekw"
        assert reg.contains("edgekw")
        assert "edgekw" in reg.keywords()
        assert reg.get("edgekw") is grammar

    def test_register_same_object_noop(self):
        reg = GrammarRegistry()
        grammar = AnnotationGrammar(keyword="x", core=True)
        reg.register(grammar)
        assert reg.grammars() == [grammar]

    def test_register_conflict_different_owners(self):
        from helixlang.core.errors import PluginConflictError
        reg = GrammarRegistry()
        a = AnnotationGrammar(keyword="dupkw", owner="pluginA", core=False)
        b = AnnotationGrammar(keyword="dupkw", owner="pluginB", core=False)
        reg.register(a)
        with pytest.raises(PluginConflictError):
            reg.register(b)

    def test_sim_grammars_filters_section(self):
        reg = GrammarRegistry()
        reg.register(AnnotationGrammar(keyword="before", sim_section=BEFORE_SIM))
        reg.register(AnnotationGrammar(keyword="after", sim_section=AFTER_SIM))
        assert [g.keyword for g in reg.sim_grammars(AFTER_SIM)] == ["after"]


class TestModuleLevelAPIs:
    def test_register_grammar_module_function(self):
        from helixlang.core.grammar_registry import register_grammar
        grammar = AnnotationGrammar(keyword="modfnk", owner="pluginMOD")
        register_grammar(grammar)
        assert grammar_registry.get("modfnk") is grammar

    def test_ensure_core_grammars_idempotent(self):
        ensure_core_grammars()
        ensure_core_grammars()
        assert grammar_registry.contains("promoter")

    def test_register_descriptor_module_function(self):
        from helixlang.core.grammar_registry import register_descriptor
        grammar = register_descriptor(
            GrammarDescriptor(keyword="moddesck", owner="pluginMOD2"))
        assert grammar_registry.get("moddesck") is grammar


class _FakeToken:
    line = 1


class _FakeParser:
    def __init__(self, fields):
        self._fields = dict(fields)

    def _advance(self):
        return _FakeToken()

    def _collect_fields_until_block_end(self, allow_no_end):
        return dict(self._fields)


class TestParseSpecKeywordHook:
    def _hook(self, **desc_kw):
        from helixlang.core.grammar_registry import _parse_spec_keyword
        desc = GrammarDescriptor(keyword="spec", **desc_kw)
        return _parse_spec_keyword(desc)

    def _prog(self):
        from types import SimpleNamespace
        return SimpleNamespace(sim_extensions={}, genes=[])

    def test_basic_fields_require_default(self):
        hook = self._hook(fields=(
            FieldSpec(key="age", type="int", required=True),
            FieldSpec(key="note", default="n/a"),
            FieldSpec(key="opt"),  # optional, no default, absent from raw
        ))
        prog = self._prog()
        hook(_FakeParser({"age": "30"}), prog)
        assert prog.sim_extensions["spec"] == [{"age": "30", "note": "n/a"}]

    def test_missing_required_raises(self):
        hook = self._hook(fields=(FieldSpec(key="age", type="int", required=True),))
        with pytest.raises(ParseError):
            hook(_FakeParser({}), self._prog())

    def test_disallow_extra_raises(self):
        hook = self._hook(fields=(), allow_extra=False)
        with pytest.raises(ParseError):
            hook(_FakeParser({"bogus": "1"}), self._prog())

    def test_section_target_appends(self):
        hook = self._hook(
            target="section", extension_key="genes",
            fields=(FieldSpec(key="name", type="str"),))
        prog = self._prog()
        hook(_FakeParser({"name": "g1"}), prog)
        assert prog.genes == [{"name": "g1"}]

    def test_section_target_not_list_raises(self):
        prog = self._prog()
        prog.genes = None
        hook = self._hook(
            target="section", extension_key="genes",
            fields=(FieldSpec(key="name", type="str"),))
        with pytest.raises(ParseError):
            hook(_FakeParser({"name": "g1"}), prog)

    def test_extra_fields_passthrough(self):
        hook = self._hook(fields=(FieldSpec(key="a"),))
        prog = self._prog()
        hook(_FakeParser({"a": "1", "extra": "2"}), prog)
        assert prog.sim_extensions["spec"] == [{"a": "1", "extra": "2"}]

    def test_second_append_uses_existing_list(self):
        hook = self._hook(fields=(FieldSpec(key="a"),))
        prog = self._prog()
        hook(_FakeParser({"a": "1"}), prog)
        hook(_FakeParser({"a": "2"}), prog)
        assert prog.sim_extensions["spec"] == [{"a": "1"}, {"a": "2"}]

    def test_allow_extra_false_with_known_fields_passes(self):
        hook = self._hook(
            allow_extra=False, fields=(FieldSpec(key="deep"),))
        prog = self._prog()
        hook(_FakeParser({"deep": "x"}), prog)
        assert prog.sim_extensions["spec"] == [{"deep": "x"}]

    def test_no_sim_extensions_store_skips(self):
        from types import SimpleNamespace
        prog = SimpleNamespace()  # no sim_extensions attribute
        hook = self._hook(fields=(FieldSpec(key="a"),))
        hook(_FakeParser({"a": "1"}), prog)  # store is None -> no-op


class TestParseQuantityStmt:
    def _prog(self):
        from types import SimpleNamespace
        return SimpleNamespace(sim_extensions={})

    def test_compact_form(self):
        from helixlang.core.grammar_registry import _parse_quantity_stmt
        prog = self._prog()
        _parse_quantity_stmt(_FakeParser({"total": "g + v"}), prog)
        assert prog.sim_extensions["quantity"] == [
            {"name": "total", "expr": "g + v"}]

    def test_named_form(self):
        from helixlang.core.grammar_registry import _parse_quantity_stmt
        prog = self._prog()
        _parse_quantity_stmt(_FakeParser({"name": "t", "expr": "a-b"}), prog)
        assert prog.sim_extensions["quantity"] == [{"name": "t", "expr": "a-b"}]

    def test_ambiguous_bad_fields_raises(self):
        from helixlang.core.grammar_registry import _parse_quantity_stmt
        with pytest.raises(ParseError):
            _parse_quantity_stmt(_FakeParser({"a": "1", "b": "2"}), self._prog())

    def test_empty_name_raises(self):
        from helixlang.core.grammar_registry import _parse_quantity_stmt
        with pytest.raises(ParseError):
            _parse_quantity_stmt(_FakeParser({"": "g"}), self._prog())

    def test_malformed_expr_raises(self):
        from helixlang.core.grammar_registry import _parse_quantity_stmt
        with pytest.raises(ParseError):
            _parse_quantity_stmt(_FakeParser({"total": "!!!"}), self._prog())

    def test_quantity_decompile_filter(self):
        from helixlang.core.grammar_registry import _quantity_decompile
        prog = self._prog()
        prog.sim_extensions["quantity"] = [
            {"name": "t", "expr": "a+b"},
            "not-a-dict",
            {"name": "u"},  # no expr
        ]
        out = _quantity_decompile(prog)
        assert out == ["#quantity name=t expr=a+b"]

    def test_quantity_decompile_empty(self):
        from helixlang.core.grammar_registry import _quantity_decompile
        assert _quantity_decompile(self._prog()) == []
