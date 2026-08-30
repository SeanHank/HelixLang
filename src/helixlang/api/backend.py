"""Backend interface & registry (doc/38 §6.5 ``api.backend``).

Kills the two structural dispatch leaks (``_engine.run``'s ``elif`` chain and
the private ``_SIM_BACKENDS`` table): every backend is a :class:`Backend`
subclass registered by id, with ``kinds`` aliases usable as ``#sim kind=...``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from helixlang.api.ast import ProgramView
from helixlang.api.capabilities import Capability
from helixlang.api.errors import PluginConflictError, PluginMissingError

if TYPE_CHECKING:
    from helixlang.sim_runtime._types import SimResult

__all__ = ["Backend", "RunRequest", "EffectiveConfig", "SimResult",
           "BackendRegistry"]


def __getattr__(name: str) -> Any:
    """Lazily re-export the sim result types (doc/38 §6.2).

    ``import helixlang.api`` must stay lightweight — importing
    ``helixlang.sim_runtime`` pulls the scientific stack, which is never
    wanted at package-import time.  ``SimResult`` & co. resolve on first
    access (a backend importing them for its run path pays the cost anyway).
    """
    if name in {"SimResult", "HistoryResult", "FluxResult", "ColonyResult",
                "ScoreResult"}:
        import helixlang.sim_runtime._types as _types

        return getattr(_types, name)
    raise AttributeError(name)


@dataclass(frozen=True)
class EffectiveConfig:
    """Typed, merged ``#config`` + core section (§6.3 migration: core keys)."""

    kind: str = "classic"
    backend: str | None = None
    table: str = "standard"
    ticks: int | None = None
    ops_per_tick: int | None = None
    react_steps: int = 1
    use_central_dogma: bool = True
    species: str | None = None
    output: str = "stdout"


@dataclass
class RunRequest:
    """A backend invocation (doc/38 §6.5)."""

    program: ProgramView
    config: EffectiveConfig
    registry: Any = None
    seed: int | None = None
    source: str | None = None


class Backend(ABC):
    """A simulation backend (doc/38 §6.5).

    ``id`` is the canonical name (e.g. ``"human"``); ``kinds`` are aliases
    acceptable as ``#sim kind=...``.  ``capabilities`` declares the flags this
    backend honours so ``Registry.fidelity`` stays honest.
    """

    id: str
    kinds: tuple[str, ...] = ()

    @abstractmethod
    def run(self, req: RunRequest) -> SimResult: ...

    def capabilities(self) -> tuple[Capability, ...]:
        return ()


class BackendRegistry:
    """Process-scoped backend registry (doc/38 §6.5).

    Replaces ``_SIM_BACKENDS`` + the ``elif backend ==`` chain in
    ``sim_runtime/_engine.py``; ``run()`` shrinks to
    ``backend_registry.resolve(backend=..., kind=...).run(req)``.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, Backend] = {}
        self._by_kind: dict[str, str] = {}

    def register(self, backend: Backend) -> None:
        if backend.id in self._by_id:
            raise PluginConflictError("backend", self._by_id[backend.id].id,
                                      backend.id)
        self._by_id[backend.id] = backend
        for kind in backend.kinds:
            if kind in self._by_kind:
                raise PluginConflictError(
                    f"#sim kind={kind}", self._by_kind[kind], backend.id)
            self._by_kind[kind] = backend.id

    def resolve(self, *, backend: str | None = None,
                kind: str | None = None) -> Backend:
        """Resolve a ``#config backend=`` or ``#sim kind=`` to a backend.

        ``kind`` wins over ``backend`` when both map to ids (a kind alias is a
        more specific directive); an unknown name is a hard
        :class:`~helixlang.api.errors.PluginMissingError` — never a silent skip.
        """
        if backend is not None:
            found = self._by_id.get(backend)
            if found is not None:
                return found
            raise PluginMissingError(backend, "backend registry")
        if kind is not None:
            found_id = self._by_kind.get(kind)
            if found_id is None:
                found = self._by_id.get(kind)
                if found is not None:
                    return found
                raise PluginMissingError(f"kind={kind}", "backend registry")
            return self._by_id[found_id]
        raise PluginMissingError("<none>", "backend registry")

    def has(self, kind: str | None = None, backend: str | None = None) -> bool:
        try:
            self.resolve(backend=backend, kind=kind)
        except PluginMissingError:
            return False
        return True

    def ids(self) -> list[str]:
        return sorted(self._by_id)
