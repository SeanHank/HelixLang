"""Grammar Registry public entry types (doc/38 §6.2 ``api.grammar``).

``AnnotationGrammar`` is the §5 annotation-grammar entry type, and
``GrammarDescriptor``/``FieldSpec`` are the doc/41 §4.2 *declarative* grammar
form (a descriptor is compiled by :func:`register_descriptor`).  Plugins
receive all of these — never ``core.grammar_registry`` directly.  These are the
*same* classes the core parser uses, re-exported through the frozen surface.
"""
from __future__ import annotations

from helixlang.core.grammar_registry import (  # noqa: F401
    AFTER_SIM,
    BEFORE_SIM,
    AnnotationGrammar,
    Decompiler,
    FieldSpec,
    GrammarDescriptor,
    ParserMethod,
    Validator,
    fmt_float,
    fmt_str,
    register_descriptor,
    register_grammar,
    sim_entry_decompile,
)

__all__ = [
    "AnnotationGrammar",
    "GrammarDescriptor",
    "FieldSpec",
    "ParserMethod",
    "Validator",
    "Decompiler",
    "BEFORE_SIM",
    "AFTER_SIM",
    "fmt_float",
    "fmt_str",
    "register_grammar",
    "register_descriptor",
    "sim_entry_decompile",
]
