"""Lexer unit tests."""
import pytest

from helixlang.core.errors import LexError
from helixlang.core.lexer import Lexer


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


# ---------------------------------------------------------------------------
# backslash line continuation (Python-style, doc/02 §2.3)
# ---------------------------------------------------------------------------


def test_backslash_continuation_joins_fields():
    src = "#config rounds=5 \\\n    transport=on \\\n    population=50"
    toks = [t for t in Lexer(src).tokens() if t.kind != "NEWLINE"]
    fields = [t for t in toks if t.kind == "FIELD"]
    assert [t.value for t in fields] == [
        "rounds=5", "transport=on", "population=50",
    ]


def test_backslash_continuation_joins_value():
    """A trailing backslash inside an unquoted value joins it with the next line."""
    src = "#gene name=long_identifier_part1\\\n_part2\nATG TAA\n#end"
    fields = [t.value for t in Lexer(src).tokens() if t.kind == "FIELD"]
    assert "name=long_identifier_part1_part2" in fields


def test_backslash_continuation_arrow_target():
    src = "#regulate lacI -> \\\n     p_lac"
    arrows = [t.value for t in Lexer(src).tokens() if t.kind == "ARROW"]
    assert arrows == ["lacI->p_lac"]


def test_backslash_continuation_dna():
    src = "ATG T\\\nCT TAA"
    codons = [t.value for t in Lexer(src).tokens() if t.kind == "CODON"]
    assert codons == ["ATG", "TCT", "TAA"]


def test_backslash_continuation_use_directive():
    src = "#use numpy --array \\\n--fft"
    u = [t for t in Lexer(src).tokens() if t.kind == "USERDIRECTIVE"]
    assert u and u[0].value == "numpy --array --fft"


def test_backslash_continuation_crlf():
    src = "#gene name=hello \\\r\n    lifespan=100\nATG TAA\n#end"
    fields = [t.value for t in Lexer(src).tokens() if t.kind == "FIELD"]
    assert "lifespan=100" in fields


def test_backslash_in_comment_is_not_continuation():
    src = "# a comment with a trailing \\\n#gene name=x"
    non_newline = [t for t in Lexer(src).tokens() if t.kind != "NEWLINE"]
    assert non_newline[0].kind == "ANNOT_START"
    assert non_newline[0].value == "gene"


def test_trailing_whitespace_after_backslash_is_not_continuation():
    """Python requires the backslash to be the very last character: '\\ ' (with
    trailing space) must NOT join the next line into the annotation."""
    src = "#gene name=hello \\ \nATG TAA\n#end"
    codons = [t.value for t in Lexer(src).tokens() if t.kind == "CODON"]
    assert codons == ["ATG", "TAA"]


def test_backslash_continuation_absorbs_following_line():
    """The joined line becomes part of the annotation (logical line), so the
    DNA bases on it lex as bare fields, not as a separate DNA block."""
    src = "#gene name=hello \\\nATG TAA\n#end"
    fields = [t.value for t in Lexer(src).tokens() if t.kind == "FIELD"]
    assert fields == ["name=hello", "ATG=", "TAA="]
