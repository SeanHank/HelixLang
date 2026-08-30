"""Core sim backends (doc/38 §6.5, §9): the engine dispatch registry.

Replaces ``_SIM_BACKENDS`` and ``_engine.run``'s ``elif backend == ...`` chain.
Every backend is a :class:`~helixlang.api.backend.Backend` subclass registered
by ``id``, with ``kinds`` aliases usable as ``#sim kind=...``.

Executors stay in ``sim_runtime._engine`` during the E3→E4 window; each
backend resolves its executor lazily on first ``run`` so importing this package
never pulls the scientific stack.
"""
from __future__ import annotations

from helixlang.api.backend import Backend, BackendRegistry

from .core import CORE_BACKENDS

__all__ = ["Backend", "BackendRegistry", "CORE_BACKENDS", "get_backend_registry"]


_default: BackendRegistry | None = None


def get_backend_registry() -> BackendRegistry:
    """Return the process-wide registry with every core backend registered."""
    global _default
    if _default is None:
        r = BackendRegistry()
        for backend in CORE_BACKENDS:
            r.register(backend)
        _default = r
    return _default
