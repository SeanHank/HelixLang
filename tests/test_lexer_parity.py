"""Native (impl_cext) vs reference (impl_python) lexer parity (doc/06 §19).

The compiled scanner is the C-only front-end stage; ``impl_python`` is the
non-selectable reference it must match token-for-token, including error type
and message.  The corpus mirrors every scanner behaviour covered by
``test_lexer.py`` plus continuation/EOF edge cases so the reference stays
branch-covered for the CI coverage gate.
"""
from __future__ import annotations

import pytest

from helixlang._accel.lexer import backend
from helixlang.core.lexer import _is_annotation_keyword

CORPUS: tuple[str, ...] = (
    "ATG GCT TAA",
    "atg gct taa",
    "ATG GCT GGT TAA",
    "ATGGCTTAA",
    "AAACCCGGGTTT",
    "ATGG",
    "",
    "#gene name=hello\nATG TAA\n#end",
    "#karl_x\nATG TAA\n",
    "# this is a comment\n#gene name=hello\nATG TAA\n#end",
    "#\n#gene name=hello\nATG TAA\n#end",
    "#lsystem axiom=F rules=0:F->F[+F]F[-F]F angle=25",
    "#config rounds=5 \\\n    transport=on \\\n    population=50",
    "#gene name=long_identifier_part1\\\n_part2\nATG TAA\n#end",
    "#regulate lacI -> \\\n     p_lac",
    "ATG T\\\nCT TAA",
    "#use numpy --array \\\n--fft",
    "#gene name=hello \\\r\n    lifespan=100\nATG TAA\n#end",
    "# a comment with a trailing \\\n#gene name=x",
    "#gene name=hello \\ \nATG TAA\n#end",
    "#gene name=hello \\\nATG TAA\n#end",
    "#use numpy --array \\\n    --fft\n",
    '#gene name="hello',
    "#use numpy --array \\",
    "#promoter_x\nATG TAA\n",
    "#gene name=a #sim\n",
    '#gene name="hello world"\n',
    " \\\nATG TAA\n",
    "#regulate a -> b\n",
    "#=oops\n",
    "ATG???\n",
    "#sim rounds=5 \\\n    transport=on",
    "#gene promoter=my_gene\nATG\n#end",
    "#use gem --core\n#sim max_ticks=100 dt=0.1",
    "#gene name=x",
    "#GENE name=x\nATG TAA\n#end",
    "#Use numpy --array\n",
    "#PromoterName field=1\nATG\n#end",
)


def _run(tokenize, src: str):
    try:
        toks = tokenize(src)
        return ("ok", [
            (t.kind, t.value, t.line, t.col, t.codon_index) for t in toks
        ])
    except Exception as e:  # parity compares the observable error contract
        return ("err", type(e).__name__, str(e))


@pytest.mark.parametrize("src", CORPUS)
def test_native_matches_reference(src: str) -> None:
    native = _run(
        lambda s: backend.tokenize(s, is_annotation_keyword=_is_annotation_keyword),
        src,
    )
    reference = _run(
        lambda s: backend.tokenize_reference(s, is_annotation_keyword=_is_annotation_keyword),
        src,
    )
    assert native == reference


@pytest.mark.parametrize("src", CORPUS)
def test_facade_matches_native(src: str) -> None:
    """core.lexer.Lexer.tokens() routes through the C kernel by default."""
    from helixlang.core.lexer import Lexer

    native = _run(
        lambda s: backend.tokenize(s, is_annotation_keyword=_is_annotation_keyword),
        src,
    )
    facade = _run(
        lambda s: [t for t in Lexer(s).tokens()],
        src,
    )
    assert facade == native


@pytest.mark.parametrize("src", ("#promoter_x\nATG TAA\n", "ATG GCT TAA"))
def test_default_predicate_matches_reference(src: str) -> None:
    """Without an injected predicate both kernels treat every ``#ident`` as a
    gene-ID marker (exercise the ``is_annotation_keyword=None`` default)."""
    native = _run(lambda s: backend.tokenize(s), src)
    reference = _run(lambda s: backend.tokenize_reference(s), src)
    assert native == reference


def test_default_predicate_errs_on_bare_annotation_fields():
    """With no keyword predicate a ``#gene name=x`` is a GENE_ID marker and the
    following unquoted text lexes as DNA (both kernels raise identically)."""
    src = "#gene name=x\nATG TAA\n#end"
    native = _run(lambda s: backend.tokenize(s), src)
    reference = _run(lambda s: backend.tokenize_reference(s), src)
    assert native == reference
    assert native[0] == "err"
