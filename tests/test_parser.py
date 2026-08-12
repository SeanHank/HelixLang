"""Parser unit tests."""
import pytest

from helixlang.errors import ParseError
from helixlang.lexer import Lexer
from helixlang.parser import Parser


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
    always on); unknown keys are ignored."""
    prog = parse("#config ticks=5 units=real")
    assert prog.config.ticks == 5
    assert not hasattr(prog.config, "units")


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
