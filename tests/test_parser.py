"""Parser unit tests."""
import pytest

from helixlang.core.errors import ParseError, UnknownKeywordError
from helixlang.core.grammar_registry import AnnotationGrammar, register_grammar
from helixlang.core.language import LanguageConfig
from helixlang.core.lexer import Lexer, Token
from helixlang.core.parser import (
    Parser,
    _parse_float,
    _parse_int,
    parse_source,
    register_core_grammars,
)


def mk_tokens(rows):
    toks = [Token(k, v, l, c) for (k, v, l, c) in rows] + [
        Token("EOF", "", 0, 0)]
    p = Parser([])
    p.toks = toks
    p.i = 0
    return p


def parse(src, stop_codons=None):
    toks = list(Lexer(src).tokens())
    return Parser(toks, stop_codons=stop_codons).parse()


def test_simple_gene():
    prog = parse("#gene name=hello\nATG GCT TAA\n#end")
    assert len(prog.genes) == 1
    g = prog.genes[0]
    assert g.name == "hello"
    assert len(g.orf) == 3
    assert g.orf[0].seq == "ATG"
    assert g.orf[-1].seq == "TAA"


def test_promoter():
    prog = parse("#promoter name=p1 strength=0.8")
    assert len(prog.promoters) == 1
    assert prog.promoters[0].name == "p1"
    assert prog.promoters[0].strength == 0.8


def test_regulate():
    src = """#gene name=a
ATG TAA
#end
#gene name=b
ATG TAA
#end
#regulate a -> b strength=0.5
"""
    prog = parse(src)
    assert len(prog.regulations) == 1
    r = prog.regulations[0]
    assert r.source == "a"
    assert r.target == "b"
    assert r.strength == 0.5


def test_config():
    prog = parse("#config ticks=42 output=csv,png table=mito_vertebrate")
    assert prog.config.ticks == 42
    assert prog.config.output == ["csv", "png"]
    assert prog.config.table == "mito_vertebrate"


def test_config_units_key_ignored():
    """The legacy #config units= key is no longer parsed (physical units are
    always on); non-classic keys are collected into Config.sim for the sim
    backends instead of being dropped."""
    prog = parse("#config ticks=5 units=real")
    assert prog.config.ticks == 5
    assert not hasattr(prog.config, "units")
    assert prog.config.sim == {"units": "real"}


def test_lsystem():
    src = "#lsystem name=plant axiom=F rules=0:F->F[+F]F[-F]F angle=25 step=1.0"
    prog = parse(src)
    assert "plant" in prog.lsystems
    decl = prog.lsystems["plant"]
    assert decl.axiom == "F"
    assert decl.rules[0]["F"] == "F[+F]F[-F]F"
    assert decl.angle == 25.0


def test_orf_no_start():
    with pytest.raises(ParseError):
        parse("#gene name=no_start\nGCT GCT TAA\n#end")


def test_orf_no_stop():
    with pytest.raises(ParseError):
        parse("#gene name=no_stop\nATG GCT GCT\n#end")


def test_anon_gene():
    prog = parse("ATG GCT TAA")
    assert len(prog.genes) == 1
    assert prog.genes[0].name.startswith("__anon")


def test_stop_codons_override():
    """In the mito table, TGA is not a stop, so the ORF should cross TGA."""
    src = "#gene name=m\nATG TGA GCT TAA\n#end"
    # Standard table: TGA is a stop, ORF = ATG TGA
    prog_std = parse(src, stop_codons={"TAA", "TAG", "TGA"})
    assert len(prog_std.genes[0].orf) == 2
    # Mito table: TGA is not a stop, ORF = ATG TGA GCT TAA
    prog_mito = parse(src, stop_codons={"TAA", "TAG", "AGA", "AGG"})
    assert len(prog_mito.genes[0].orf) == 4


def test_field_with_negative_strength():
    prog = parse("#promoter name=p strength=-0.5")
    assert prog.promoters[0].strength == -0.5


# ---------------------------------------------------------------------------
# W-1: simulation wiring - Config.backend / Config.sim and the new
# structural annotations (#media / #enzyme / #metabolite / #sim)
# ---------------------------------------------------------------------------


def test_config_sim_collects_unknown_keys():
    """Extra #config keys land in Config.sim; the classic keys still parse."""
    prog = parse("#config ticks=42 backend=whole_cell division_rule=adder "
                 "adder_volume_um3=1.6 seed=0 output=energy,volume_um3")
    assert prog.config.ticks == 42
    assert prog.config.output == ["energy", "volume_um3"]
    assert prog.config.backend == "whole_cell"
    assert prog.config.sim == {
        "division_rule": "adder",
        "adder_volume_um3": "1.6",
        "seed": "0",
    }


def test_config_backend_default_classic():
    prog = parse("#config ticks=5")
    assert prog.config.backend == "classic"
    assert prog.config.sim == {}


def test_media_enzyme_metabolite_parsed():
    src = """
#gene name=glk
ATG GCT GCT GCT TAA
#end

#media nutrient=GLC concentration=10.0 diffusion_um2_s=300
#enzyme gene=glk reaction=HEX1 kcat=2800
#metabolite name=glc__D init=0.5
"""
    prog = parse(src)
    assert len(prog.media) == 1
    assert prog.media[0].nutrient == "GLC"
    assert prog.media[0].concentration == 10.0
    assert prog.media[0].diffusion_um2_s == 300.0
    assert len(prog.enzymes) == 1
    assert prog.enzymes[0].gene == "glk"
    assert prog.enzymes[0].reaction == "HEX1"
    assert prog.enzymes[0].kcat == 2800.0
    assert len(prog.pools) == 1
    assert prog.pools[0].name == "glc__D"
    assert prog.pools[0].init == 0.5


def test_media_repeatable_and_defaults():
    prog = parse("#media nutrient=GLC concentration=10.0\n"
                 "#media nutrient=O2 concentration=0.25")
    assert len(prog.media) == 2
    assert prog.media[1].diffusion_um2_s is None
    assert prog.enzymes == []
    assert prog.pools == []


def test_enzyme_kcat_optional():
    prog = parse("#enzyme gene=pgi reaction=PGI")
    assert prog.enzymes[0].kcat is None


def test_media_requires_nutrient_and_concentration():
    with pytest.raises(ParseError):
        parse("#media concentration=10.0")
    with pytest.raises(ParseError):
        parse("#media nutrient=GLC")


def test_enzyme_requires_gene_and_reaction():
    with pytest.raises(ParseError):
        parse("#enzyme reaction=PGI")
    with pytest.raises(ParseError):
        parse("#enzyme gene=pgi")


def test_sim_extension_point_collects_fields():
    prog = parse("#sim kind=spatial_dfba length=32\n"
                 "#sim inlet_glucose_mm=5.0 initial_biomass_gdw=0.05")
    assert prog.sim_extensions == {
        "kind": "spatial_dfba",
        "length": "32",
        "inlet_glucose_mm": "5.0",
        "initial_biomass_gdw": "0.05",
    }
    assert prog.config.sim == {}


def test_species_genome_field_and_dna_block():
    """#species accepts the genotype as genome= or as a DNA code block.

    The block form is concatenated (analogous to #gene) and lands on the
    same ``species.<name>.genome`` extension key.
    """
    field_form = parse(
        "#species name=consumer genome=ATGCTAATGCTA substrate=glucose\n")
    assert field_form.sim_extensions["species.consumer.genome"] \
        == "ATGCTAATGCTA"

    block_form = parse(
        "#species name=producer substrate=acetate vmax=0.012 ks=0.05\n"
        "ATGCTAATGCTAATGCTA\n"
        "ATGCTAATGCTAATGCTA\n"
        "#end\n"
        "#patch name=water kind=water\n")
    assert block_form.sim_extensions["species.producer.genome"] \
        == "ATGCTA" * 6
    assert block_form.sim_extensions["species.producer.substrate"] \
        == "acetate"
    # the block DNA was NOT wrapped as an anonymous gene
    assert block_form.genes == []


def test_species_genome_field_and_block_conflict():
    with pytest.raises(ParseError, match="not both"):
        parse(
            "#species name=consumer genome=ATGCTAATGCTA\n"
            "ATGCTAATGCTA\n"
            "#end\n")


def test_species_block_accepts_space_and_newline_separated_codons():
    """Space- and newline-separated codons (the #gene style) are joined
    into the species genome exactly like a contiguous run."""
    prog = parse(
        "#species name=consumer substrate=glucose vmax=0.02 ks=0.1\n"
        "ATG CTA ATG CTA ATG CTA\n"
        "ATGCTA ATGCTA ATG CTA\n"
        "#end\n")
    assert prog.sim_extensions["species.consumer.genome"] == "ATGCTA" * 6
    assert prog.genes == []


# ---------------------------------------------------------------------------
# Numeric coercions (_parse_float / _parse_int typed-error paths)
# ---------------------------------------------------------------------------


def test_parse_float_success_and_error():
    assert _parse_float("1.5", "x", 1) == 1.5
    with pytest.raises(ParseError):
        _parse_float("not-a-number", "x", 1)


def test_parse_int_error():
    with pytest.raises(ParseError):
        _parse_int("abc", "ticks", 1)


# ---------------------------------------------------------------------------
# Constructor guardrails (config/stop_codons exclusivity)
# ---------------------------------------------------------------------------


def test_parser_init_with_config_only():
    p = Parser([], config=LanguageConfig.for_table("standard"))
    assert p.config is not None


def test_parser_init_config_and_stop_codons_conflict():
    with pytest.raises(ParseError):
        Parser([], config=LanguageConfig.for_table("standard"),
               stop_codons={"TAA"})


# ---------------------------------------------------------------------------
# Main dispatch loop edge cases
# ---------------------------------------------------------------------------


def test_unknown_keyword_error():
    with pytest.raises(UnknownKeywordError, match="unknown keyword"):
        Parser(mk_tokens([("ANNOT_START", "foobar", 1, 1)]).toks).parse()


def test_user_directive_parsed():
    prog = parse("#use grn")
    assert [d.plugin for d in prog.use_directives] == ["grn"]


def test_use_error_wrapped():
    with pytest.raises(ParseError, match="#use:"):
        parse("#use")


def test_unexpected_token_main_loop():
    with pytest.raises(ParseError, match="unexpected token"):
        Parser(mk_tokens([("BOGUS", "?", 1, 1)]).toks).parse()


def test_newline_skipped_in_main_loop():
    p = mk_tokens([("NEWLINE", "\n", 1, 1)])
    calls = {"n": 0}

    def fake_peek(k=0):
        calls["n"] += 1
        if calls["n"] <= 3:
            return Token("NEWLINE", "\n", 1, 1)
        return Token("EOF", "", 0, 0)

    p._peek = fake_peek
    assert len(p.parse().genes) == 0


def test_type_check_enabled_runs():
    p = Parser(list(Lexer("#type foo=Protein").tokens()), enable_type_check=True)
    with pytest.raises(ParseError, match="type check failed"):
        p.parse()


# ---------------------------------------------------------------------------
# #config field coverage
# ---------------------------------------------------------------------------


def test_config_all_fields():
    prog = parse("#config ticks=1 ops_per_tick=2 react_steps=3 "
                 "use_central_dogma=true species=human backend=x "
                 "skip_validity=1")
    assert prog.config.ticks == 1
    assert prog.config.ops_per_tick == 2
    assert prog.config.react_steps == 3
    assert prog.config.use_central_dogma is True
    assert prog.config.species == "human"
    assert prog.config.backend == "x"
    assert prog.config.skip_validity is True


def test_config_without_ticks_reaches_output():
    prog = parse("#config output=csv")
    assert prog.config.output == ["csv"]


def test_config_empty_sim_value_rejected():
    with pytest.raises(ParseError, match="empty value"):
        parse("#config foo=")


def test_config_numeric_sim_parameter_with_unit():
    prog = parse("#config timeout=5min")
    assert "timeout" in prog.config.quantities


def test_config_sim_parameter_without_unit():
    prog = parse("#config timeout=abc")
    assert prog.config.sim == {"timeout": "abc"}


def test_config_ticks_invalid():
    with pytest.raises(ParseError):
        parse("#config ticks=abc")


# ---------------------------------------------------------------------------
# #type annotation parsing
# ---------------------------------------------------------------------------


def test_type_annotation_basic():
    prog = parse("#type a=Protein")
    assert prog.type_annotations == {"a": "Protein"}


def test_type_annotation_unit():
    prog = parse("#type a=Float<min>")
    assert prog.type_annotations == {"a": "Float<min>"}


def test_type_annotation_repeat_same():
    prog = parse("#type a=Protein\n#type a=Protein")
    assert prog.type_annotations == {"a": "Protein"}


def test_type_annotation_conflict():
    with pytest.raises(ParseError, match="conflicting #type"):
        parse("#type a=Protein\n#type a=Float")


def test_type_annotation_unknown_type():
    with pytest.raises(ParseError, match="unknown type annotation"):
        parse("#type a=BogusTypeXYZ")


def test_type_annotation_unknown_dimension():
    with pytest.raises(ParseError, match="unknown unit"):
        parse("#type a=Float<zzz>")


# ---------------------------------------------------------------------------
# Type checking (_run_type_check)
# ---------------------------------------------------------------------------


def _type_check(src):
    return Parser(list(Lexer(src).tokens()), enable_type_check=True).parse()


def test_type_check_pass():
    _type_check(
        "#gene name=a\nATG TAA\n#end\n"
        "#type a=Protein\n"
        "#crispr target=a position=50\n"
        "#regulate a -> a strength=0.5\n")


def test_type_check_pass_no_annotations():
    _type_check("#gene name=a\nATG TAA\n#end\n")


def test_type_check_undefined_type_symbol():
    with pytest.raises(ParseError, match="undefined symbol"):
        _type_check("#type foo=Protein")


def test_type_check_undefined_regulation_source_and_target():
    with pytest.raises(ParseError, match="regulation source"):
        _type_check("#regulate a -> b strength=0.5")


def test_type_check_defined_source_undefined_target():
    with pytest.raises(ParseError, match="regulation target"):
        _type_check("#gene name=a\nATG TAA\n#end\n#regulate a -> b strength=0.5")


def test_type_check_undefined_bio_target():
    with pytest.raises(ParseError, match="target"):
        _type_check("#crispr target=foo position=50")


# ---------------------------------------------------------------------------
# Token hook surface (_ParserTokenHooks)
# ---------------------------------------------------------------------------


def test_token_hooks_drive_token_stream():
    p = mk_tokens([("ANNOT_START", "config", 1, 1),
                   ("FIELD", "ticks=5", 1, 2)])
    hooks = p.token_hooks
    assert hooks.peek().kind == "ANNOT_START"
    assert hooks.advance().kind == "ANNOT_START"
    assert hooks.peek().kind == "FIELD"
    assert hooks.expect("FIELD", "ticks=5").kind == "FIELD"
    assert hooks.collect_fields(allow_no_end=True) == {}


def test_token_hooks_expect_error():
    p = mk_tokens([("CODON", "ATG", 1, 1)])
    with pytest.raises(ParseError, match="expected"):
        p.token_hooks.expect("ANNOT_START", "config")


# ---------------------------------------------------------------------------
# _peek / _expect out-of-range and errors
# ---------------------------------------------------------------------------


def test_peek_out_of_range_clamps_to_eof():
    p = mk_tokens([("CODON", "ATG", 1, 1)])
    assert p._peek(5).kind == "EOF"
    assert p._peek(2).kind == "EOF"


# ---------------------------------------------------------------------------
# Field collection branch coverage
# ---------------------------------------------------------------------------


def test_collect_fields_annot_end():
    assert mk_tokens([("FIELD", "a=1", 1, 1),
                      ("ANNOT_END", "", 1, 2)])._collect_fields_until_block_end() \
        == {"a": "1"}


def test_collect_fields_arrow():
    p = mk_tokens([("FIELD", "a=1", 1, 1), ("ARROW", "->", 1, 2),
                   ("EOF", "", 0, 0)])
    assert p._collect_fields_until_block_end() == {"a": "1"}


def test_collect_fields_eof_no_end_raises():
    p = mk_tokens([("FIELD", "a=1", 1, 1)])
    with pytest.raises(ParseError, match="unexpected EOF"):
        p._collect_fields_until_block_end(allow_no_end=False)


def test_collect_fields_codon():
    p = mk_tokens([("FIELD", "a=1", 1, 1), ("CODON", "ATG", 1, 2),
                   ("EOF", "", 0, 0)])
    assert p._collect_fields_until_block_end() == {"a": "1"}


def test_collect_fields_gene_id():
    p = mk_tokens([("FIELD", "a=1", 1, 1), ("GENE_ID", "g", 1, 2),
                   ("EOF", "", 0, 0)])
    assert p._collect_fields_until_block_end() == {"a": "1"}


def test_collect_fields_newline():
    p = mk_tokens([("FIELD", "a=1", 1, 1), ("NEWLINE", "\n", 1, 2),
                   ("FIELD", "b=2", 1, 3), ("ANNOT_END", "", 1, 4)])
    assert p._collect_fields_until_block_end() == {"a": "1", "b": "2"}


def test_collect_fields_unknown_token():
    p = mk_tokens([("FIELD", "a=1", 1, 1), ("BOGUS", "?", 1, 2),
                   ("EOF", "", 0, 0)])
    assert p._collect_fields_until_block_end() == {"a": "1"}


def test_collect_fields_loop_exit():
    p = mk_tokens([("ANNOT_END", "", 1, 1)])
    p._peek = lambda k=0: None
    assert p._collect_fields_until_block_end() == {}


# ---------------------------------------------------------------------------
# parse_source convenience wrapper
# ---------------------------------------------------------------------------


def test_parse_source_wrapper():
    prog = parse_source("#gene name=a\nATG TAA\n#end\n")
    assert prog.genes[0].name == "a"


# ---------------------------------------------------------------------------
# Biological instructions (P0-1.1)
# ---------------------------------------------------------------------------


def test_bio_instruction_parsed():
    prog = parse('#crispr target=a position=50 new_sequence="GGGG"')
    inst = prog.bio_instructions[0]
    assert inst.kind == "crispr"
    assert inst.target == "a"
    assert inst.params["new_sequence"] == "GGGG"


def test_bio_instruction_quoted_params():
    prog = parse("#evolve target=a gene_pool=\"x y\"")
    assert prog.bio_instructions[0].params["gene_pool"] == "x y"


def test_bio_instruction_requires_target():
    with pytest.raises(ParseError, match="requires target="):
        parse("#crispr position=50")


# ---------------------------------------------------------------------------
# Plugin grammar activation / progress checks
# ---------------------------------------------------------------------------


def _nop_grammar(parser, prog):
    pass


def _advance_grammar(parser, prog):
    parser._advance()


def test_requires_use_grammar_inert_until_use():
    register_grammar(AnnotationGrammar("zz_requse", parse=_advance_grammar,
                                       requires_use="grn"))
    with pytest.raises(UnknownKeywordError, match="enable it with"):
        Parser(mk_tokens([("ANNOT_START", "zz_requse", 1, 1)]).toks).parse()


def test_grammar_made_no_progress():
    register_grammar(AnnotationGrammar("zz_noprogress", parse=_nop_grammar))
    with pytest.raises(ParseError, match="made no progress"):
        Parser(mk_tokens([("ANNOT_START", "zz_noprogress", 1, 1)]).toks).parse()


# ---------------------------------------------------------------------------
# register_core_grammars idempotency
# ---------------------------------------------------------------------------


def test_register_core_grammars_idempotent():
    register_core_grammars()
