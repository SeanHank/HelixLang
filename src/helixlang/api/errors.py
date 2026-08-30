"""Typed error family re-export (doc/38 §6.2 ``api.errors``).

Plugins catch and raise the *same* exception classes the core does, imported
from one frozen surface instead of the private ``core.errors`` path.
"""
from __future__ import annotations

from helixlang.core.dimensions import UnitError  # noqa: F401
from helixlang.core.errors import (
    ABIVersionError,
    BioError,
    CompileError,
    HelixError,
    LexError,
    ModelMissingError,
    NativeBackendError,
    ParseError,
    PluginConflictError,
    PluginDependencyError,
    PluginError,
    PluginMissingError,
    RegulationError,
    RuntimeHelixError,
    SemanticError,
    SemanticVersionError,
    SimConfigError,
    StackUnderflowError,
    UnknownKeywordError,
    UnknownNodeError,
)

__all__ = [
    "HelixError",
    "LexError",
    "ParseError",
    "SemanticError",
    "CompileError",
    "RegulationError",
    "RuntimeHelixError",
    "BioError",
    "SimConfigError",
    "UnknownKeywordError",
    "StackUnderflowError",
    "UnknownNodeError",
    "UnitError",
    "PluginError",
    "PluginMissingError",
    "PluginDependencyError",
    "PluginConflictError",
    "ModelMissingError",
    "ABIVersionError",
    "SemanticVersionError",
    "NativeBackendError",
]
