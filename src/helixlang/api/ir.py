"""IR extension contract (doc/38 §6.4 ``api.ir``).

New IR instruction kinds, namespaced ``"<plugin>.<inst>"``.  Compiler-emitted
plugin insts are never generated without an explicit ``#use``; an IR kind that
is not registered at load is refused (never silently skipped).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = ["OperandMode", "OperandSlot", "OperandSchema", "IRExtension"]


class OperandMode(enum.IntEnum):
    """How an operand slot of a plugin IR inst serializes (ABI)."""

    INLINE = 1   # literal value embedded in the inst stream
    LABEL = 2    # relative label/offset
    CONST = 3    # index into the chunk constant pool


@dataclass(frozen=True)
class OperandSlot:
    """One typed operand slot of a plugin IR inst."""

    name: str
    mode: OperandMode


OperandSchema = tuple[OperandSlot, ...]


class IRRuntime(Protocol):
    """The IR interpreter surface an ``execute`` hook drives."""

    def read(self, slot: OperandSlot) -> Any: ...

    def write(self, slot: OperandSlot, value: Any) -> None: ...


@dataclass(frozen=True)
class IRExtension:
    """A namespaced IR instruction set owned by one plugin.

    ``build`` lowers plugin translation output into ``IRInst`` records;
    ``execute`` is the ``IRRuntime`` dispatch arm; ``operand_schema`` is the
    serialization ABI (how each operand writes into a ``.helixc`` PLUGIN_EXT
    payload).  ``abi_version`` must match the plugin manifest.
    """

    id: str
    kinds: tuple[str, ...]
    build: Any
    execute: Any
    operand_schema: OperandSchema = ()
    abi_version: int = 1
