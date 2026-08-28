"""HelixLang core (doc/36 §3).

The minimal, always-installed semantic core: language errors, the plugin
registry, the ``use`` statement model, and the silent-fallback linter.  All
scientific functionality lives in optional plugins loaded through
:mod:`helixlang.core.plugin_registry`.
"""
from helixlang.core.errors import (
    ABIVersionError,
    ModelMissingError,
    NativeBackendError,
    PluginConflictError,
    PluginDependencyError,
    PluginError,
    PluginMissingError,
    StackUnderflowError,
    UnknownKeywordError,
    UnknownNodeError,
)
from helixlang.core.plugin_registry import (
    NativeBackend,
    PluginProvider,
    Registry,
    get_registry,
)
from helixlang.core.use_stmt import (
    KNOWN_FLAGS,
    UseDirective,
    UseError,
    parse_use_line,
)

__all__ = [
    "ABIVersionError", "ModelMissingError", "NativeBackendError",
    "PluginConflictError", "PluginDependencyError", "PluginError",
    "PluginMissingError", "StackUnderflowError", "UnknownKeywordError",
    "UnknownNodeError",
    "NativeBackend", "PluginProvider", "Registry", "get_registry",
    "KNOWN_FLAGS", "UseDirective", "UseError", "parse_use_line",
]
