"""Plugin manifest parsing (doc/38 §6.8 ``helix.plugin.toml``).

Stdlib-only (``tomllib``), so ``helixc plugin list``, dependency resolution and
conflict checks run cold — no plugin import, no heavy scientific deps.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from helixlang.api.grammar import AnnotationGrammar
from helixlang.core.errors import PluginError

MANIFEST_NAME = "helix.plugin.toml"


def _expect_str_list(value: Any, path: str, source: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PluginError(f"{source}: {path} must be a list of strings")
    return list(value)


@dataclass(frozen=True)
class ManifestProvides:
    """The ``[provides]`` table: what the plugin makes core-visible."""

    grammars: tuple[str, ...] = ()
    ast: tuple[str, ...] = ()
    ir: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()


@dataclass(frozen=True)
class PluginManifest:
    """Parsed ``helix.plugin.toml`` (doc/38 §6.8)."""

    name: str
    version: str
    entry_point: str
    abi_version: int = 1
    provides: ManifestProvides = field(default_factory=ManifestProvides)
    capability_flags: tuple[str, ...] = ()
    requires_pip: tuple[str, ...] = ()
    native_module: str | None = None
    native_rebuild: str | None = None
    source: str = "<toml>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "entry_point": self.entry_point,
            "abi_version": self.abi_version,
            "provides": {
                "grammars": list(self.provides.grammars),
                "ast": list(self.provides.ast),
                "ir": list(self.provides.ir),
                "backends": list(self.provides.backends),
            },
            "capabilities": {"flags": list(self.capability_flags)},
            "requires": {"pip": list(self.requires_pip)},
            "native": ({"module": self.native_module,
                        "rebuild": self.native_rebuild}
                       if self.native_module else None),
        }


def parse_manifest(text: str, source: str = "<toml>") -> PluginManifest:
    """Parse manifest TOML text with no plugin import (doc/38 §6.8)."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover - tomllib detail
        raise PluginError(f"{source}: malformed TOML: {exc}") from exc
    name = data.get("name")
    version = data.get("version")
    entry_point = data.get("entry_point")
    if not (isinstance(name, str) and name and isinstance(version, str)
            and version and isinstance(entry_point, str) and entry_point):
        raise PluginError(f"{source}: manifest needs name, version, entry_point")
    abi = data.get("abi_version", 1)
    if not isinstance(abi, int) or abi < 1:
        raise PluginError(f"{source}: abi_version must be an int >= 1")

    provides = data.get("provides", {}) or {}
    provides = provides if isinstance(provides, dict) else {}
    provide_section = ManifestProvides(
        grammars=tuple(_expect_str_list(provides.get("grammars", []),
                                        "provides.grammars", source)),
        ast=tuple(_expect_str_list(provides.get("ast", []),
                                   "provides.ast", source)),
        ir=tuple(_expect_str_list(provides.get("ir", []),
                                  "provides.ir", source)),
        backends=tuple(_expect_str_list(provides.get("backends", []),
                                        "provides.backends", source)),
    )
    caps = data.get("capabilities", {}) or {}
    caps = caps if isinstance(caps, dict) else {}
    flags = tuple(_expect_str_list(caps.get("flags", []),
                                   "capabilities.flags", source))
    requires = data.get("requires", {}) or {}
    requires = requires if isinstance(requires, dict) else {}
    pip = tuple(_expect_str_list(requires.get("pip", []), "requires.pip", source))

    native = data.get("native", {}) or {}
    native = native if isinstance(native, dict) else {}
    native_module = native.get("module")
    native_rebuild = native.get("rebuild", "python -m helixlang._accel.build")
    if native_module is not None and not isinstance(native_module, str):
        raise PluginError(f"{source}: native.module must be a string")

    return PluginManifest(
        name=name, version=version, entry_point=entry_point, abi_version=abi,
        provides=provide_section, capability_flags=flags, requires_pip=pip,
        native_module=native_module, native_rebuild=native_rebuild,
        source=source,
    )


def load_manifest(path: Path) -> PluginManifest:
    """Load a ``helix.plugin.toml`` from disk."""
    path = Path(path)
    if not path.is_file():
        raise PluginError(f"{path}: no such manifest file")
    return parse_manifest(path.read_text(encoding="utf-8"), source=str(path))


def discover_manifests(root: Path) -> list[PluginManifest]:
    """Discover every ``helix.plugin.toml`` under ``root`` (without imports)."""
    root = Path(root)
    if not root.is_dir():
        return []
    found: list[PluginManifest] = []
    for path in sorted(root.rglob(MANIFEST_NAME)):
        found.append(load_manifest(path))
    return found


def manifest_matches_grammars(manifest: PluginManifest,
                              grammars: dict[str, AnnotationGrammar]) -> None:
    """Enforce §6.8: each declared grammar is registered by this plugin.

    Drift between the manifest ``[provides].grammars`` and the grammars the
    plugin *actually* registered is a hard :class:`PluginError`.
    """
    registered_owner = {kw: g.owner for kw, g in grammars.items()}
    missing = [kw for kw in manifest.provides.grammars
               if registered_owner.get(kw) != manifest.name]
    if missing:
        raise PluginError(
            f"{manifest.source}: manifest declares grammars not owned by "
            f"'{manifest.name}': {sorted(missing)}")
