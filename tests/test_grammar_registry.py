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
    AnnotationGrammar,
    ensure_core_grammars,
    grammar_registry,
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
