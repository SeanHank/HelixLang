"""Cardiology DSL plugin — the doc/41 §4.3 grammar-descriptor demo.

A brand-new ``#cardiac_cycle`` keyword with typed fields, a semantic
validator and a ``.helixc`` round-trip — declared entirely from this plugin
module via the frozen :class:`~helixlang.api.grammar.GrammarDescriptor`.
No edits to ``parser.py`` or ``lexer.py`` were required to add it.

The grammar is inert until a program says ``#use cardiology``
(``requires_use``), matching doc/41 §4.2: a plugin's grammar set is
activated by its ``#use``.
"""
from __future__ import annotations

from typing import Any

from helixlang.api.grammar import FieldSpec, GrammarDescriptor
from helixlang.api.registry import PluginProvider
from helixlang.core.errors import SemanticError


def _validate_cardiac_cycle(analyser: Any, program: Any) -> None:
    """Semantic hook: the declared period must lie in the physiological range."""
    entries = (program.sim_extensions or {}).get("cardiac_cycle")
    if not isinstance(entries, list):
        return
    for entry in entries:
        try:
            period = float(entry["period"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0.0 < period <= 5.0:
            raise SemanticError(
                f"#cardiac_cycle period={entry['period']!r} outside the "
                "physiological range (0, 5] s")


CARDIAC_DESCRIPTOR = GrammarDescriptor(
    keyword="cardiac_cycle",
    fields=(
        FieldSpec(key="period", type="float", required=True, unit="s"),
        FieldSpec(key="conduction", type="str", default="normal"),
    ),
    validate=_validate_cardiac_cycle,
)


PLUGIN = PluginProvider(
    name="cardiology",
    extra="cardiology",
    keywords=("cardiac_cycle",),
    grammars=(CARDIAC_DESCRIPTOR,),
)

__all__ = ["CARDIAC_DESCRIPTOR", "PLUGIN", "_validate_cardiac_cycle"]
