"""GEM ↔ simulation integration adapter (doc/38 §6.2 ``api.gem``).

Thin, stdlib-only adapters for the FBA-medium injection helpers a GEM provider
needs at simulation-setup time.  The heavy ``sim_runtime`` machinery is
imported lazily inside each call, so importing this module (or ``helixlang.api``)
never pulls the scientific stack.
"""
from __future__ import annotations

from typing import Any


def add_gem_core_reactions(*args: Any, **kwargs: Any) -> Any:
    """Add the GEM core-metabolism reactions to a whole-cell reaction graph."""
    from helixlang.sim_runtime import _add_gem_core_reactions
    return _add_gem_core_reactions(*args, **kwargs)


def add_gem_transport_reactions(*args: Any, **kwargs: Any) -> Any:
    """Add the GEM membrane-transport reactions to a whole-cell reaction graph."""
    from helixlang.sim_runtime import _add_gem_transport_reactions
    return _add_gem_transport_reactions(*args, **kwargs)


def set_gem_medium(*args: Any, **kwargs: Any) -> Any:
    """Set FBA medium bounds for a GEM model (organism-specific defaults)."""
    from helixlang.sim_runtime import _set_gem_medium
    return _set_gem_medium(*args, **kwargs)


__all__ = ["add_gem_core_reactions", "add_gem_transport_reactions",
           "set_gem_medium"]
