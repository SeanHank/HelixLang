"""Comprehensive tests for grammar_handlers.py – target 100 % line + branch."""
from __future__ import annotations

import pytest

from helixlang.core.errors import ParseError
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser


def parse(src: str):
    toks = list(Lexer(src).tokens())
    return Parser(toks).parse()


# ======================================================================
# _parse_float / _parse_int error paths (lines 50-51, 59-60)
# ======================================================================

class TestParseFloatInt:
    def test_parse_float_error_via_lsystem(self):
        with pytest.raises(ParseError, match="invalid angle"):
            parse("#lsystem name=p axiom=F angle=abc")

    def test_parse_int_error_via_field(self):
        with pytest.raises(ParseError, match="invalid size"):
            parse("#field size=abc")

    def test_parse_float_error_via_reaction(self):
        with pytest.raises(ParseError, match="invalid substrate_coeff"):
            parse("#reaction id=PGI substrate_coeff=xyz")


# ======================================================================
# _parse_promoter (lines 93-106)
# ======================================================================

class TestParsePromoter:
    def test_missing_name(self):
        with pytest.raises(ParseError, match="missing name"):
            parse("#promoter strength=0.8")

    def test_missing_strength(self):
        with pytest.raises(ParseError, match="missing strength"):
            parse("#promoter name=p1")

    def test_invalid_strength(self):
        with pytest.raises(ParseError, match="invalid strength"):
            parse("#promoter name=p1 strength=abc")


# ======================================================================
# _parse_gene (lines 108-138)
# ======================================================================

class TestParseGene:
    def test_anonymous_gene_no_name_field(self):
        src = "#gene\nATG GCT TAA\n#end"
        prog = parse(src)
        assert len(prog.genes) == 1
        assert prog.genes[0].name.startswith("__anon")

    def test_pharma_form_with_name(self):
        src = "#gene name=INSR allele=*1\n#promoter name=p1 strength=0.5"
        prog = parse(src)
        assert "genes" in prog.sim_extensions

    def test_pharma_form_no_name(self):
        with pytest.raises(ParseError, match="requires name="):
            parse("#gene allele=*1\n#end")

    def test_no_codons(self):
        with pytest.raises(ParseError, match="has no DNA codons"):
            parse("#gene name=g1\n#promoter name=p1 strength=0.5")

    def test_gene_block_without_end(self):
        src = "#gene name=g1\nATG GCT TAA\n#promoter name=p1 strength=0.5"
        prog = parse(src)
        assert len(prog.genes) == 1
        assert prog.genes[0].name == "g1"


# ======================================================================
# _parse_regulate (lines 140-154)
# ======================================================================

class TestParseRegulate:
    def test_invalid_strength(self):
        with pytest.raises(ParseError, match="invalid strength"):
            parse("#regulate a -> b strength=abc")

    def test_no_strength_default(self):
        prog = parse("#regulate a -> b")
        assert len(prog.regulations) == 1
        assert prog.regulations[0].strength == 0.5


# ======================================================================
# _parse_lsystem (lines 156-183)
# ======================================================================

class TestParseLSystem:
    def test_empty_rules_field(self):
        prog = parse("#lsystem name=plant axiom=F angle=25")
        assert "plant" in prog.lsystems
        assert prog.lsystems["plant"].rules == {}

    def test_empty_rule_entry(self):
        prog = parse('#lsystem name=p axiom=F rules=0:F->F;;1:F->FF angle=25')
        decl = prog.lsystems["p"]
        assert 0 in decl.rules
        assert 1 in decl.rules

    def test_non_int_rule_key(self):
        prog = parse('#lsystem name=p axiom=F rules=abc:F->F angle=25')
        assert prog.lsystems["p"].rules == {}

    def test_no_arrow_in_pair(self):
        prog = parse('#lsystem name=p axiom=F rules=0:F->F,XYZ angle=25')
        decl = prog.lsystems["p"]
        assert "F" in decl.rules[0]
        assert decl.rules[0]["F"] == "F"

    def test_multiple_rules_with_mixed_entries(self):
        prog = parse('#lsystem name=p axiom=F rules=0:F->F;;bad;1:F->FF,XYZ angle=25')
        decl = prog.lsystems["p"]
        assert 0 in decl.rules
        assert 1 in decl.rules
        assert "F" in decl.rules[0]
        assert decl.rules[1].get("XYZ") is None


# ======================================================================
# _parse_morphogen (lines 195-217)
# ======================================================================

class TestParseMorphogen:
    def test_missing_gene(self):
        with pytest.raises(ParseError, match="requires gene="):
            parse("#morphogen channel=V gain=0.1")

    def test_bad_channel(self):
        with pytest.raises(ParseError, match="must be 'U' or 'V'"):
            parse("#morphogen gene=g1 channel=X gain=0.1")

    def test_invalid_gain(self):
        with pytest.raises(ParseError, match="invalid gain"):
            parse("#morphogen gene=g1 channel=V gain=abc")


# ======================================================================
# _parse_media (lines 219-249)
# ======================================================================

class TestParseMedia:
    def test_invalid_concentration(self):
        with pytest.raises(ParseError, match="invalid concentration"):
            parse("#media nutrient=GLC concentration=abc")

    def test_invalid_diffusion(self):
        with pytest.raises(ParseError, match="invalid diffusion"):
            parse("#media nutrient=GLC concentration=10.0 diffusion_um2_s=abc")


# ======================================================================
# _parse_enzyme (lines 251-280)
# ======================================================================

class TestParseEnzyme:
    def test_invalid_kcat(self):
        with pytest.raises(ParseError, match="invalid kcat"):
            parse("#enzyme gene=g1 reaction=R1 kcat=abc")

    def test_invalid_km(self):
        with pytest.raises(ParseError, match="invalid km"):
            parse("#enzyme gene=g1 reaction=R1 km=abc")


# ======================================================================
# _parse_reaction (lines 282-325)
# ======================================================================

class TestParseReaction:
    def test_missing_id(self):
        with pytest.raises(ParseError, match="requires id="):
            parse("#reaction name=PGI substrate=g6p product=f6p")

    def test_no_substrate_no_product(self):
        prog = parse("#reaction id=PGI")
        assert len(prog.reactions) == 1
        assert prog.reactions[0].substrate == ""
        assert prog.reactions[0].product == ""


# ======================================================================
# _parse_metabolite (lines 327-344)
# ======================================================================

class TestParseMetabolite:
    def test_missing_name(self):
        with pytest.raises(ParseError, match="requires name="):
            parse("#metabolite init=0.5")

    def test_invalid_init(self):
        with pytest.raises(ParseError, match="invalid init"):
            parse("#metabolite name=glc init=abc")


# ======================================================================
# _parse_species (lines 370-416)
# ======================================================================

class TestParseSpecies:
    def test_missing_name(self):
        with pytest.raises(ParseError, match="requires name="):
            parse("#species substrate=glucose")

    def test_dna_block_no_end(self):
        src = ("#species name=prod substrate=acetate\n"
               "ATGCTAATGCTAATGCTA\n"
               "#gene name=g1\nATG GCT TAA\n#end")
        prog = parse(src)
        assert "species.prod.genome" in prog.sim_extensions


# ======================================================================
# _parse_gem (lines 418-508)
# ======================================================================

class TestParseGem:
    def test_missing_organism(self):
        with pytest.raises(ParseError, match="requires organism="):
            parse("#gem genome=genome.fasta")

    def test_genome_and_codon_conflict(self):
        with pytest.raises(ParseError, match="not both"):
            parse("#gem organism=e_coli_k12 genome=genome.fasta\n"
                  "ATG GCT TAA\n#end")

    def test_inline_dna_gene_ids_only(self):
        prog = parse("#gem organism=e_coli_k12\n#myGene\n#gene name=g1\n"
                     "ATG GCT TAA\n#end")
        assert "gem_organism" in prog.sim_extensions

    def test_inline_dna_gene_id_with_codons(self):
        src = ("#gem organism=e_coli_k12\n"
               "#gltA\nATG TCT CAG CAA ATT CGT GTG GCG CTG AAT GTA GAG CTT\n"
               "#end")
        prog = parse(src)
        assert "gem_inline_genes" in prog.sim_extensions

    def test_inline_dna_no_annot_end(self):
        src = ("#gem organism=e_coli_k12\n"
               "#myGene\n"
               "#gene name=g1\nATG GCT TAA\n#end")
        prog = parse(src)
        assert "gem_organism" in prog.sim_extensions


# ======================================================================
# _parse_patch (lines 510-545)
# ======================================================================

class TestParsePatch:
    def test_missing_name(self):
        with pytest.raises(ParseError, match="requires name="):
            parse("#patch kind=water")


# ======================================================================
# _parse_disease_gene (lines 622-638)
# ======================================================================

class TestParseDiseaseGene:
    def test_missing_gene(self):
        with pytest.raises(ParseError, match="requires gene="):
            parse("#disease_gene type=downregulate activity=0.3")


# ======================================================================
# _parse_disease_metabolite (lines 640-658)
# ======================================================================

class TestParseDiseaseMetabolite:
    def test_missing_id(self):
        with pytest.raises(ParseError, match="requires id="):
            parse("#disease_metabolite type=accumulate concentration=7.8")


# ======================================================================
# _parse_drug (lines 660-675)
# ======================================================================

class TestParseDrug:
    def test_missing_name(self):
        with pytest.raises(ParseError, match="requires name="):
            parse("#drug smiles=CN dose=500")


# ======================================================================
# _parse_pd_effect (lines 677-693)
# ======================================================================

class TestParsePdEffect:
    def test_missing_drug(self):
        with pytest.raises(ParseError, match="requires drug="):
            parse("#pd_effect target=BIOMASS ec50=5")


# ======================================================================
# _parse_qsp_binding (lines 695-712)
# ======================================================================

class TestParseQspBinding:
    def test_missing_drug(self):
        with pytest.raises(ParseError, match="requires drug="):
            parse("#qsp_binding kind=tmdd")

    def test_missing_kind(self):
        with pytest.raises(ParseError, match="requires kind="):
            parse("#qsp_binding drug=trastuzumab")


# ======================================================================
# _parse_endocrine_config (lines 714-730)
# ======================================================================

class TestParseEndocrineConfig:
    def test_missing_axis(self):
        with pytest.raises(ParseError, match="requires axis="):
            parse("#endocrine_config severity=0.5")


# ======================================================================
# _parse_immune_config / _parse_tumor_biopsy – basic coverage
# ======================================================================

class TestParseImmuneConfig:
    def test_basic(self):
        prog = parse("#immune_config infection_severity=0.8")
        assert "immune_configs" in prog.sim_extensions

class TestParseTumorBiopsy:
    def test_basic(self):
        prog = parse("#tumor_biopsy mutation=EGFR_L858R")
        assert "tumor_biopsy" in prog.sim_extensions


# ======================================================================
# _clean_value / _append_sim_list / human-sim annotations
# ======================================================================

class TestCleanValue:
    def test_strip_quotes(self):
        from helixlang.core.grammar_handlers import ParserGrammarMixin
        assert ParserGrammarMixin._clean_value('"hello"') == "hello"
        assert ParserGrammarMixin._clean_value("hello") == "hello"
        assert ParserGrammarMixin._clean_value('"') == '"'

class TestAppendSimList:
    def test_basic(self):
        from helixlang.core.ast_nodes import Program
        from helixlang.core.grammar_handlers import ParserGrammarMixin
        p = ParserGrammarMixin()
        prog = Program()
        p._append_sim_list(prog, "items", {"a": "1"})
        p._append_sim_list(prog, "items", {"a": "2"})
        ext = prog.extensions
        items = ext.extension_for("items").get("items")
        assert isinstance(items, list)
        assert len(items) == 2

class TestParsePerson:
    def test_basic(self):
        prog = parse("#person name=John age=55")
        assert "person_name" in prog.sim_extensions
        assert prog.sim_extensions["person_name"] == "John"

class TestParseTrait:
    def test_basic(self):
        prog = parse("#trait smoking=former")
        assert "trait_smoking" in prog.sim_extensions

class TestParseDisease:
    def test_basic(self):
        prog = parse("#disease name=diabetes category=metabolic")
        assert "disease_name" in prog.sim_extensions
