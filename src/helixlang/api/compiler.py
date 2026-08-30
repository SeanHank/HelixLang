"""Public compiler-front-end surface (doc/38 §6.2 ``api.compiler``).

Re-exports the four-stage front end (``Lexer`` → ``Parser`` →
``SemanticAnalyzer`` → ``Compiler``) so plugins can compile embedded helix
source at runtime — e.g. an app platform quoting a circuit — without importing
``core.lexer`` / ``core.parser`` / ``core.semantic`` / ``core.compiler``.
All four stages are stdlib-only.
"""
from __future__ import annotations

from helixlang.core.compiler import Compiler  # noqa: F401
from helixlang.core.lexer import Lexer  # noqa: F401
from helixlang.core.parser import Parser  # noqa: F401
from helixlang.core.semantic import SemanticAnalyzer  # noqa: F401

__all__ = ["Compiler", "Lexer", "Parser", "SemanticAnalyzer"]
