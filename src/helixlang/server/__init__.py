"""HelixLang web visualization server package (Flask).

Package facade.  The Flask app factory and its route handlers live in
:mod:`helixlang.server.app`; this module re-exports the public API surface so
callers keep ``from helixlang.server import create_app, run_server``.
"""
from __future__ import annotations

from helixlang.server.app import (
    _DEBUG_SESSIONS,
    _get_debug_lock,
    app,
    create_app,
    run_server,
)

__all__ = [
    "create_app",
    "run_server",
    "app",
    "_DEBUG_SESSIONS",
    "_get_debug_lock",
]
