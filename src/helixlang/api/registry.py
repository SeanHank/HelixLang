"""Plugin activation-layer re-export (doc/38 §6.2 ``api.registry``).

``PluginProvider``, ``NativeBackend`` and ``Registry`` are the core
activation layer (discovery, capability flags, dependency checks, conflict
detection) re-exported from ``core.plugin_registry`` through the frozen
surface.  ``grammar_registry`` is the shared §5 annotation-grammar singleton
the parser reads at parse time, so plugins register their grammars onto the
*same* instance.  Plugins never import the private module paths.
"""
from __future__ import annotations

from helixlang.core.grammar_registry import grammar_registry  # noqa: F401
from helixlang.core.plugin_registry import (  # noqa: F401
    NativeBackend,
    PluginProvider,
    Registry,
    get_registry,
)

__all__ = ["PluginProvider", "NativeBackend", "Registry", "get_registry",
           "grammar_registry"]
