"""Lexer alias: exposes ``tokenize`` via the uniform hot-loop loader.

The scanner algorithm lives in the compiled ``impl_cext`` (mandated, never the
Python reference).  Additionally exposes ``tokenize_reference`` for tests that
verify the C kernel against the non-selectable pure-Python implementation.
"""
from __future__ import annotations

from helixlang._accel._loaders import load_hot


def tokenize(source: str, is_annotation_keyword=None):
    """Tokenize ``source`` with the compiler+VM front-end stage's C kernel.

    Returns the full list of :class:`~helixlang.core.lexer.Token`` instances
    (the reference and the parser both consume the stream eagerly).  A missing
    compiled kernel raises ``NativeBackendError`` — there is no interpreter
    fallback (doc/03 §6.5).
    """
    mod = load_hot("helixlang._accel.lexer")
    return mod.tokenize(source, is_annotation_keyword=is_annotation_keyword)


def tokenize_reference(source: str, is_annotation_keyword=None):
    """Run the pure-Python reference scanner (tests/verification only)."""
    from helixlang._accel.lexer.impl_python import tokenize as _ref

    return _ref(source, is_annotation_keyword=is_annotation_keyword)
