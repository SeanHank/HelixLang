"""Example plugin annotation grammar: ``#vector`` (doc/38 §5).

A minimal domain schema ("gene therapy vector") registered with the core
:class:`~helixlang.core.grammar_registry.GrammarRegistry`:

    #vector gene=TP53 plasmid=pUC19 payload_len=3821

Registering a grammar makes parse, compile, decompile and ``.helixc``
round-trips work with **no changes to ``parser.py`` or ``hxbc.py``**: the
Parser dispatches ``#vector`` through the registry and ``hxbc.decompile`` walks
the same registry to write the annotation back because ``list_valued_keys``
names the ``extensions["vectors"]`` key this grammar round-trips.

Importing this module registers the grammar (idempotent).  It is not imported
by :mod:`helixlang` at startup; opt in from a driver or test that needs it::

    import helixlang.plugins.annotation.vector  # noqa: F401

``program.extensions["vectors"]`` accumulates one cleaned dict entry per
annotation, consumed by any downstream driver.
"""
from __future__ import annotations

from typing import Any

from helixlang.api.ast import Program
from helixlang.api.grammar import AnnotationGrammar, sim_entry_decompile
from helixlang.api.registry import grammar_registry
from helixlang.core.errors import ParseError


def _parse_vector(parser: Any, prog: Program) -> None:
    """Collect ``#vector`` fields into ``program.extensions["vectors"]``."""
    t = parser._advance()  # ANNOT_START
    fields = parser._collect_fields_until_block_end(allow_no_end=True)
    if not fields.get("gene"):
        raise ParseError("#vector requires gene= field", line=t.line)
    parser._append_sim_list(prog, "vectors", fields)


VECTOR_GRAMMAR = AnnotationGrammar(
    keyword="vector",
    parse=_parse_vector,
    decompile=sim_entry_decompile("vectors", "vector"),
    list_valued_keys=frozenset({"vectors"}),
    core=False,
    owner="vector",
)


def register() -> None:
    """Register (or re-register) the vector grammar."""
    grammar_registry.register(VECTOR_GRAMMAR)


register()
