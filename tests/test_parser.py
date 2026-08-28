"""Parser unit tests."""
import pytest

from helixlang.core.errors import ParseError
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser


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
