"""Typed program-extension sections (doc/38 §6.3 E2).

``Program.extensions`` replaces the free-form ``sim_extensions`` design: every
parser write is routed through a typed :class:`ExtensionSection` that (a) names
its owner (§6.3 migration map), (b) declares the only keys it may consume, and
(c) coerces values.  For the migration window the section *is a view over*
``Program.sim_extensions`` (the single store), so ``.helixc`` encoding, the
legacy decode path and every ``sim_runtime`` reader keep working byte-for-byte
until E4 converts them to ``ProgramView``.

A write to a governed section with an undeclared key is a hard
:class:`~helixlang.core.errors.UnknownKeywordError` — never a silent ignore
(doc/36 F7).  ``core`` is the open escape hatch that mirrors ``#sim``'s
long-tail semantics; keys it claims are inert until a backend registers them.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from helixlang.core.errors import UnknownKeywordError

if TYPE_CHECKING:
    from helixlang.core.ast_nodes import Program

#: The §6.3 migration-map ownership table, keyed by section id.  ``exact`` /
#: ``lists`` / ``maps`` are exact ``sim_extensions`` keys; ``prefixes`` claim any
#: key with that prefix (e.g. ``species.<name>.k``).  Matching order is exact
#: (any value) > declared list/map type > prefix.
_OWNERSHIP: dict[str, dict[str, set[str]]] = {
    "human": {
        "exact": {"genes", "drugs", "disease_genes", "disease_metabolites",
                  "pd_effects", "qsp_bindings", "endocrine_configs",
                  "immune_configs"},
        "maps": {"tumor_biopsy"},
        "prefixes": {"person_", "trait_", "disease_"},
    },
    "gem": {
        "exact": {"gem_inline_genes", "gem_inline_genome", "gem_dynamic",
                  "gem_duration", "gem_dt"},
        "prefixes": {"gem_"},
    },
    "population": {
        "exact": {"genome", "mechanics", "lbm"},
        "prefixes": {"genome_", "species.", "patch."},
    },
}

_CORE_OPEN = "core"

#: Ordered governed section ids serialized as ``PLUGIN_EXT`` records (doc/38
#: §6.7).  Order is the serialization order — stable ABI.
GOVERNED_IDS = ("human", "gem", "population")

#: Plugin-extension ABI for governed sections (doc/38 §6.8 ``abi_version``).
PLUGIN_EXT_ABI = 1


def split_sim_extensions(
        program: Program) -> tuple[dict[str, dict[str, Any]],
                                     dict[str, Any]]:
    """Split ``sim_extensions`` into governed section payloads + legacy keys.

    Returns ``(owned, unowned)`` where ``owned[sid]`` holds that section's
    declared keys (serialized as a ``PLUGIN_EXT`` payload) and ``unowned`` is
    the keys the legacy flat ``sim_extensions`` TAG keeps carrying (doc/38
    §6.7 read-only decode window).  Deterministic: section order follows
    :data:`GOVERNED_IDS`, keys are emitted sorted by the codec.
    """
    owned: dict[str, dict[str, Any]] = {}
    unowned: dict[str, Any] = {}
    store = program.sim_extensions
    for sid in GOVERNED_IDS:
        section = program.extensions.extension(sid)
        data = section.to_dict()
        if data:
            owned[sid] = data
    for key, value in store.items():
        if not any(key in owned_payload
                   for owned_payload in owned.values()) and not any(
                       program.extensions.extension(sid)._belongs(key, value)
                       for sid in GOVERNED_IDS):
            unowned[key] = value
    return owned, unowned


@dataclass
class ExtensionSection:
    """One typed extension section (an ``extensions.<id>`` namespace).

    Writes are validated against the declared segments and reflected into the
    shared store (``Program.sim_extensions``) so the legacy engine/hxbc paths
    see exactly what they see today.
    """

    id: str
    store: dict[str, Any]
    exact: set[str] = field(default_factory=set)
    maps: set[str] = field(default_factory=set)
    prefixes: set[str] = field(default_factory=set)
    open_: bool = False

    # ── membership ──────────────────────────────────────────────────────────
    def owns_key(self, key: str, value: Any) -> bool:
        if self.open_:
            return True
        if key in self.exact:
            return True
        if key in self.maps:
            return isinstance(value, dict)
        return any(key.startswith(p) for p in self.prefixes)

    # ── write (the only sanctioned way to extend program state) ─────────────
    def set(self, key: str, value: Any) -> None:
        """Write ``value`` to ``key`` with ownership + type enforcement."""
        if not self.open_:
            if key in self.maps:
                if not isinstance(value, dict):
                    raise UnknownKeywordError(
                        f"extension '{self.id}' field {key!r} expects a map")
            elif key not in self.exact and not any(
                    key.startswith(p) for p in self.prefixes):
                raise UnknownKeywordError(
                    f"extension '{self.id}' does not declare key {key!r} "
                    f"(owner of {sorted(self.exact)} + prefixes "
                    f"{sorted(self.prefixes)})")
        self.store[key] = value

    def append(self, key: str, item: Any) -> None:
        """Append one entry to a list-valued section field (may create it)."""
        if not self.open_ and key not in self.exact:
            raise UnknownKeywordError(
                f"extension '{self.id}' does not declare list key {key!r}")
        existing = self.store.get(key)
        if isinstance(existing, list):
            existing.append(item)
        else:
            self.store[key] = [item]

    # ── read (typed attribute access for declared fields) ───────────────────
    def get(self, key: str, default: Any = None) -> Any:
        value = self.store.get(key)
        return value if value is not None else default

    def __getattr__(self, name: str) -> Any:
        section = object.__getattribute__(self, "store")
        if name in section:
            return section[name]
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.store.items() if self._belongs(k, v)}

    def _belongs(self, key: str, value: Any) -> bool:
        if self.open_:
            return True
        if key in self.maps:
            return isinstance(value, dict)
        if key in self.exact:
            return True
        return any(key.startswith(p) for p in self.prefixes)


class ProgramExtensions(Mapping[str, Any]):
    """The typed ``program.extensions`` namespace (doc/38 §6.3).

    ``extension_for(key, value)`` resolves a write to its owning section
    (exact > map > prefix, across the §6.3 table); anything unclaimed falls to
    the open ``core`` section — the ``#sim`` long-tail escape hatch.

    The namespace is also a read-only ``Mapping`` over the single store, so
    engine/`_opt_*` readers (``ext.get(...)``, ``key in ext``, ``ext.items()``,
    ``{**ext}``) move from the raw ``sim_extensions`` dict to this typed view
    without changing a byte (E4).  A ``__setitem__`` write routes through
    ``extension_for`` for ownership + type enforcement.
    """

    def __init__(self, program: Program) -> None:
        self._program = program
        self._sections = self._build()

    def _build(self) -> dict[str, ExtensionSection]:
        store = self._program.sim_extensions
        return {
            sid: ExtensionSection(
                id=sid, store=store,
                exact=set(cfg.get("exact", ())),
                maps=set(cfg.get("maps", ())),
                prefixes=set(cfg.get("prefixes", ())),
                open_=sid == _CORE_OPEN,
            )
            for sid, cfg in _OWNERSHIP.items()
        } | {_CORE_OPEN: ExtensionSection(
            id=_CORE_OPEN, store=store, open_=True)}

    # ── Mapping read surface (the single store, owner-agnostic) ─────────────
    def __getitem__(self, key: str) -> Any:
        return self._program.sim_extensions[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._program.sim_extensions)

    def __len__(self) -> int:
        return len(self._program.sim_extensions)

    def __setitem__(self, key: str, value: Any) -> None:
        """Typed write routed to the owning section (E4 readers)."""
        self.extension_for(key, value).set(key, value)

    # ── section access ───────────────────────────────────────────────────────
    def extension(self, sid: str) -> ExtensionSection:
        if sid not in self._sections:
            raise KeyError(
                f"no extension section '{sid}' "
                f"(registered: {sorted(self._sections)})")
        return self._sections[sid]

    def extension_for(self, key: str, value: Any = None) -> ExtensionSection:
        for sid in (*_OWNERSHIP, _CORE_OPEN):
            section = self._sections[sid]
            if section.owns_key(key, value):
                return section
        return self._sections[_CORE_OPEN]

    def ids(self) -> list[str]:
        return sorted(self._sections)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {sid: section.to_dict()
                for sid, section in self._sections.items()}

    def __getattr__(self, name: str) -> ExtensionSection:
        if name.startswith("_"):
            raise AttributeError(name)
        section = object.__getattribute__(self, "_sections").get(name)
        if section is None:
            raise AttributeError(name)
        assert isinstance(section, ExtensionSection)
        return section
