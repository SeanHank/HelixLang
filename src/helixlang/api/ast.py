"""AST extension contract (doc/38 §6.3 ``api.ast``).

Defines the typed, namespaced replacement for the free-form
``Program.sim_extensions`` dict: each ``ASTExtension`` owns a read-only
section on ``ProgramView.extensions`` (``program.extensions.<id>.<field>``),
and writes happen only through ``ProgramBuilder`` with declared fields —
an unknown key is a hard error, never a silent ignore.

The runtime implements these protocols in the parser/AST layer; plugins
import only the contract types from here.  ``Program`` (the parsed root AST
node) is re-exported too so plugins can type-annotate grammar / decompile
hooks without importing ``core.ast_nodes``.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Protocol

from helixlang.core.ast_nodes import Program  # noqa: F401

__all__ = ["FieldType", "SectionField", "ASTExtension", "ProgramView",
           "ProgramBuilder", "Program"]


class FieldType(enum.IntEnum):
    """The typed slots an extension section may contain (coerced on write)."""

    INT = 1
    FLOAT = 2
    BOOL = 3
    STR = 4
    LIST = 5
    MAP = 6


@dataclass(frozen=True)
class SectionField:
    """One declared field of an AST-extension section.

    ``ProgramBuilder`` coerces writes to ``type``; a write of an undeclared
    key is a hard :class:`~helixlang.api.errors.UnknownKeywordError`.
    """

    key: str
    type: FieldType


@dataclass(frozen=True)
class ASTExtension:
    """A namespaced extension of the program AST (replaces ``sim_extensions``).

    ``grammars`` must be registered with the Grammar Registry (§5);
    ``fields`` is the *only* key set this extension may consume; ``abi_version``
    must match the plugin manifest's.  ``parse`` builds the section from a
    grammar directive; ``validate`` runs in the semantic phase.
    """

    id: str
    grammars: tuple[str, ...]
    parse: Any
    fields: tuple[SectionField, ...] = ()
    validate: Any = None
    decompile: Any = None
    abi_version: int = 1


class ProgramView(Protocol):
    """Read-only view of a program handed to backends (§6.3, §6.5).

    Backends can read typed extension sections and the effective config but
    can never reach the raw :class:`Program`/``Config``/``Chunk`` objects.
    """

    id: str
    config: Any
    extensions: Any

    def source(self) -> str | None: ...


class ProgramBuilder(Protocol):
    """Write-side program builder used by ``ASTExtension.parse`` hooks."""

    def extension(self, ext_id: str, /) -> Any: ...

    def set_field(self, ext_id: str, key: str, value: Any, /) -> None: ...
