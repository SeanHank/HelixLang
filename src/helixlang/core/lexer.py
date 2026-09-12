"""Lexer facade: dual-mode DNA scanner, whole algorithm in the C kernel.

The scanning algorithm shipped to the compiler+VM C-only mandate (doc/06 §19):
the dual-mode (DNA / annotation) tokenizer executes on the hand-written C
kernel ``helixlang._accel.lexer.impl_cext``.  The pure-Python scanner lives on
as the **non-selectable reference** in ``helixlang._accel.lexer.impl_python``
(used only to verify native equivalence); ``choose_backend`` refuses a Python
backend for this stage.  This module keeps the public contract — ``Token``,
``ANNOTATION_KEYWORDS``, ``_is_annotation_keyword`` and ``Lexer.tokens()`` —
unchanged so the parser, CLI, server and every downstream caller keep working.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

#: #keywords always treated as annotations (never gene-ID markers), matched
#: case-exact against the lexed identifier (doc/38 §5).  Plugin grammars are
#: recognized dynamically through the shared grammar registry instead of this
#: tuple, so a new #keyword needs no lexer edit.
ANNOTATION_KEYWORDS = frozenset({
    "use", "gem", "config", "sim", "end", "species", "genome",
    "media", "patch", "type", "enzyme", "metabolite",
    "regulate", "export", "reaction",
    "gene", "promoter", "lsystem", "morphogen",
    "crispr", "evolve", "methylate", "histone",
    "transcribe", "translate", "quorum",
    "gff", "sequence", "table", "field",
    "person", "trait", "disease", "disease_gene",
    "disease_metabolite", "drug", "pd_effect",
    "qsp_binding", "endocrine_config", "immune_config",
    "tumor_biopsy",
})


def _is_annotation_keyword(name: str) -> bool:
    """Core keywords plus anything a (plugin) grammar registered.

    ``name`` is matched case-exact: a registered grammar keyword is an
    annotation (``ANNOT_START``); anything else that looks like ``#ident`` is a
    gene-ID marker (``GENE_ID``).
    """
    if name in ANNOTATION_KEYWORDS:
        return True
    from helixlang.core import grammar_registry as _grammar

    return _grammar.grammar_registry.contains(name)


@dataclass(slots=True)
class Token:
    kind: str   # CODON | ANNOT_START | ANNOT_END | FIELD | ARROW | NEWLINE | EOF | GENE_ID | USERDIRECTIVE
    value: str
    line: int
    col: int
    codon_index: int = -1

    def __repr__(self) -> str:
        ci = f" #{self.codon_index}" if self.codon_index >= 0 else ""
        return f"Token({self.kind},{self.value!r} L{self.line}:{self.col}{ci})"


class Lexer:
    """Dual-mode scanner facade.

    The token stream is produced by the compiled C kernel in a single eager
    call (the parser and every consumer materialise the stream with
    ``list(...)`` anyway, so error sites are identical to the reference).
    """

    def __init__(self, source: str):
        # Preserves the source (public contract); the scan itself runs in C.
        self.src = source

    def tokens(self) -> Iterator[Token]:
        from helixlang._accel.lexer.backend import tokenize

        yield from tokenize(self.src, is_annotation_keyword=_is_annotation_keyword)
