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


def test_codon_token_repr_includes_index():
    """repr() on a CODON token exercises the codon_index branch."""
    toks = [t for t in Lexer("ATG TAA").tokens() if t.kind == "CODON"]
    assert "#0" in repr(toks[0])
    assert "#1" in repr(toks[1])


def test_use_continuation_with_leading_space():
    """A continuation with indented next line hits the whitespace-skip loop."""
    src = "#use numpy --array \\\n    --fft\n"
    u = [t for t in Lexer(src).tokens() if t.kind == "USERDIRECTIVE"]
    assert u and u[0].value == "numpy --array --fft"


def test_unterminated_quoted_value_at_eof():
    """A quote opened at end-of-input skips the closing-quote advance."""
    toks = list(Lexer('#gene name="hello').tokens())
    fields = [t for t in toks if t.kind == "FIELD"]
    assert fields and fields[0].value.startswith("name=")


def test_use_continuation_backslash_at_eof():
    """A trailing backslash with no newline still stops cleanly."""
    u = [t for t in Lexer("#use numpy --array \\").tokens()
         if t.kind == "USERDIRECTIVE"]
    assert u and (u[0].value == "numpy --array" or "numpy" in u[0].value)


def test_gene_id_marker():
    toks = [t for t in Lexer("#promoter_x\nATG TAA\n").tokens()
            if t.kind != "NEWLINE"]
    assert toks[0].kind == "GENE_ID"
    assert toks[0].value == "promoter_x"


def test_grammar_keyword_recognized_via_registry():
    # 'sim' is in ANNOTATION_KEYWORDS; use a registry-only corner via the
    # direct predicate.
    from helixlang.core import grammar_registry as _gr
    from helixlang.core.lexer import _is_annotation_keyword
    _gr.ensure_core_grammars()
    assert _is_annotation_keyword("quantity")  # not in ANNOTATION_KEYWORDS
    assert not _is_annotation_keyword("not_a_real_keyword")


def test_unexpected_char_raises():
    with pytest.raises(LexError):
        list(Lexer("ATG???\n").tokens())


def test_missing_annotation_name_raises():
    with pytest.raises(LexError):
        list(Lexer("#=oops\n").tokens())


def test_new_annotation_on_same_line_ends_fields():
    # '#' mid-field ends the current annotation's field list and the
    # following keyword begins a fresh annotation on the same line.
    toks = [t for t in Lexer("#gene name=a #sim\n").tokens()
            if t.kind != "NEWLINE"]
    kinds = [t.kind for t in toks]
    assert kinds.count("ANNOT_START") == 2


def test_terminated_quoted_value():
    toks = [t for t in Lexer('#gene name="hello world"\n').tokens()
            if t.kind == "FIELD"]
    assert toks and "hello world" in toks[0].value


def test_top_level_backslash_newline_and_leading_space():
    # a leading space (line 104) followed by a top-level backslash-newline
    # (line 108) before a DNA block.
    toks = [t for t in Lexer(" \\\nATG TAA\n").tokens()
            if t.kind != "NEWLINE"]
    assert [t.kind for t in toks] == ["CODON", "CODON", "EOF"]


def test_at_line_continuation_false_for_non_backslash():
    lx = Lexer("ATG\n")
    lx.pos = 0
    # 'A' is not a backslash -> False
    assert lx._at_line_continuation() is False


def test_plain_arrow_without_continuation():
    toks = [t for t in Lexer("#regulate a -> b\n").tokens()
            if t.kind != "NEWLINE"]
    kinds = [t.kind for t in toks]
    assert "ARROW" in kinds
    arrow = next(t for t in toks if t.kind == "ARROW")
    assert "->" in arrow.value
