"""Lexer unit tests."""
import pytest

from helixlang.errors import LexError
from helixlang.lexer import Lexer


def test_simple_dna():
    toks = [t for t in Lexer("ATG GCT TAA").tokens() if t.kind != "NEWLINE"]
    assert len(toks) == 4  # 3 codons + EOF
    assert toks[0].kind == "CODON" and toks[0].value == "ATG"
    assert toks[1].kind == "CODON" and toks[1].value == "GCT"
    assert toks[2].kind == "CODON" and toks[2].value == "TAA"
    assert toks[3].kind == "EOF"


def test_lowercase_dna():
    toks = [t for t in Lexer("atg gct taa").tokens() if t.kind == "CODON"]
    assert [t.value for t in toks] == ["ATG", "GCT", "TAA"]


def test_codon_indexing():
    toks = [t for t in Lexer("ATG GCT GGT TAA").tokens() if t.kind == "CODON"]
    assert [t.codon_index for t in toks] == [0, 1, 2, 3]


def test_dna_length_not_multiple_of_3():
    with pytest.raises(LexError):
        list(Lexer("ATGG").tokens())


def test_annotation_block():
    src = "#gene name=hello\nATG TAA\n#end"
    toks = [t for t in Lexer(src).tokens() if t.kind != "NEWLINE"]
    kinds = [t.kind for t in toks]
    assert "ANNOT_START" in kinds
    assert "FIELD" in kinds
    assert "ANNOT_END" in kinds


def test_comment_lines():
    src = "# this is a comment\n#gene name=hello\nATG TAA\n#end"
    toks = list(Lexer(src).tokens())
    # The first non-NEWLINE token should be ANNOT_START (gene), not a comment
    non_newline = [t for t in toks if t.kind != "NEWLINE"]
    assert non_newline[0].kind == "ANNOT_START"
    assert non_newline[0].value == "gene"


def test_hash_only_line_is_comment():
    src = "#\n#gene name=hello\nATG TAA\n#end"
    toks = list(Lexer(src).tokens())
    non_newline = [t for t in toks if t.kind != "NEWLINE"]
    assert non_newline[0].kind == "ANNOT_START"


def test_field_with_special_chars():
    """L-system rules may contain characters like : -> [ ]."""
    src = "#lsystem axiom=F rules=0:F->F[+F]F[-F]F angle=25"
    toks = [t for t in Lexer(src).tokens() if t.kind == "FIELD"]
    fields = {t.value.split("=", 1)[0]: t.value.split("=", 1)[1] for t in toks}
    assert fields["rules"] == "0:F->F[+F]F[-F]F"
    assert fields["axiom"] == "F"
    assert fields["angle"] == "25"
