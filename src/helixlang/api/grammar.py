"""Grammar Registry public entry types (doc/38 §6.2 ``api.grammar``).

``AnnotationGrammar`` is the §5 annotation-grammar entry type.  Plugins receive
it (and the registry object) from :mod:`helixlang.api.registry`; they never
import ``core.grammar_registry`` directly.  These are the *same* classes the
core parser uses, re-exported through the frozen surface.
"""
from __future__ import annotations

from helixlang.core.grammar_registry import (  # noqa: F401
    AFTER_SIM,
    BEFORE_SIM,
    AnnotationGrammar,
    Decompiler,
    ParserMethod,
    Validator,
    fmt_float,
    fmt_str,
    register_grammar,
    sim_entry_decompile,
)

__all__ = [
    "AnnotationGrammar",
    "ParserMethod",
    "Validator",
    "Decompiler",
    "BEFORE_SIM",
    "AFTER_SIM",
    "fmt_float",
    "fmt_str",
    "register_grammar",
    "sim_entry_decompile",
]
