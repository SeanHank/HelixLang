"""Lexer hot-loop package (doc/06 §19 — compiler+VM C-only mandate).

Front-end stage of the compiler pipeline: the dual-mode DNA/annotation scanner
moved out of ``helixlang.core.lexer`` into a compiled kernel so the *language's
first pipeline stage* executes on hand-written C (``impl_cext``), never on an
interpreter loop.  ``impl_python`` is kept only as the non-selectable reference
the native kernel must match token-for-token; ``backend.tokenize`` resolves
through the uniform hot-loop loader, so a compiled kernel with a Python/numpy
request raises :class:`~helixlang.core.errors.NativeBackendError`.
"""
