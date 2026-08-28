"""Plugin registry (doc/36 §3 / §4).

The registry tracks which plugins are installed, what backend each provides, and
which explicit capability flags the current program has declared.

Design rules enforced here and at runtime:

- A :class:`PluginProvider` registers a ``name``, optional ``keywords`` (the
  ``#keyword`` statements it handles), an optional ``native`` backend
  descriptor, and the pip ``extra`` needed to install it.
- Registering a keyword/backend name that another provider already claims raises
  :class:`~helixlang.core.errors.PluginConflictError` (doc/36 F7).
- Activation of a provider is *lazy*: :meth:`Registry.activate` imports its
  module and verifies its declared capability flags (``--pure-python``,
  ``--approx-euler``, ``--low-fidelity``) before the dependency check.  A
  missing dependency raises
  :class:`~helixlang.core.errors.PluginDependencyError` pointing at the pip
  extra.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from importlib import import_module
from typing import Any

from helixlang.core.errors import (
    PluginConflictError,
    PluginDependencyError,
    PluginMissingError,
)

# Bundled first-party plugins (doc/36 §8). ``discover`` imports each lightweight
# ``helixlang.plugins.<name>`` module lazily; heavy scientific deps are only
# pulled on activation.
_BUNDLED_PLUGINS: tuple[str, ...] = (
    "grn",
    "fba",
    "human",
    "apps",
    "annotation",
    "gem",
    "kinetics",
    "omics",
)


@dataclass(slots=True)
class NativeBackend:
    """Descriptor for an optional compiled accelerator backend."""

    module: str          # import path, e.g. helixlang._accel.sim
    fidelity: str = "high"
    rebuild_cmd: str = "python -m helixlang._accel.build"


@dataclass(slots=True)
class PluginProvider:
    """A loadable plugin.

    ``load`` is the entry point that returns the plugin object; ``checks`` are
    callables returning True when an optional dependency is satisfied.
    """

    name: str
    extra: str
    keywords: tuple[str, ...] = ()
    native: NativeBackend | None = None
    capability_flags: tuple[str, ...] = ()
    checks: dict[str, Callable[[], bool]] = field(default_factory=dict)
    load: Callable[[], Any] | None = None

    def check_dependencies(self) -> list[str]:
        """Return the list of unsatisfied declared optional deps."""
        return [dep for dep, ok in self.checks.items() if not ok()]


class Registry:
    """Process-wide registry of plugin providers plus active capability flags.

    This is a lightweight singleton; the CLI/VM opens a fresh :class:`Registry`
    per program so capability flags from one ``use`` statement do not leak into
    the next.
    """

    def __init__(self) -> None:
        self._providers: dict[str, PluginProvider] = {}
        self._by_keyword: dict[str, str] = {}
        self._by_native_module: dict[str, str] = {}
        self._active: set[str] = set()
        self._capabilities: set[str] = set()

    # ── discovery (doc/36 §3.5) ───────────────────────────────────────────
    def discover(self, *names: str) -> list[str]:
        """Lazily register bundled plugins by importing their lightweight module.

        Each ``helixlang.plugins.<name>.PLUGIN`` is a :class:`PluginProvider`
        describing the plugin's metadata, keywords and dependencies — importing
        that module carries **no heavy scientific dependencies** (numpy/scipy/
        rdkit/... are pulled only when :meth:`activate` runs the provider's
        ``load``).  Returns the names newly registered.
        """
        names = names or _BUNDLED_PLUGINS
        registered: list[str] = []
        for name in names:
            if name in self._providers:
                continue
            try:
                mod = import_module(f"helixlang.plugins.{name}")
            except ImportError:
                continue
            plugin = getattr(mod, "PLUGIN", None)
            if plugin is None or not isinstance(plugin, PluginProvider):
                continue
            # Register a per-registry *copy* (with a copied checks dict) so that
            # mutating one registry's provider state never leaks to the shared
            # module-level PLUGIN or to other per-program registries.
            plugin = replace(plugin, checks=dict(plugin.checks))
            self.register(plugin)
            registered.append(name)
        return registered

    # ── registration ────────────────────────────────────────────────────────
    def register(self, provider: PluginProvider) -> None:
        if provider.name in self._providers:
            raise PluginConflictError("name", self._providers[provider.name].name,
                                      provider.name)
        self._providers[provider.name] = provider
        for kw in provider.keywords:
            if kw in self._by_keyword:
                raise PluginConflictError(
                    f"#keyword:{kw}", self._by_keyword[kw], provider.name)
            self._by_keyword[kw] = provider.name
        if provider.native is not None:
            mod = provider.native.module
            if mod in self._by_native_module:
                raise PluginConflictError(
                    f"native:{mod}", self._by_native_module[mod], provider.name)
            self._by_native_module[mod] = provider.name

    # ── lookup ──────────────────────────────────────────────────────────────
    def provider(self, name: str) -> PluginProvider:
        p = self._providers.get(name)
        if p is None:
            raise PluginMissingError(name, "core")
        return p

    def provider_for_keyword(self, keyword: str) -> PluginProvider | None:
        name = self._by_keyword.get(keyword)
        return self._providers.get(name or "")

    def is_registered(self, name: str) -> bool:
        return name in self._providers

    def providers(self) -> list[PluginProvider]:
        return list(self._providers.values())

    # ── capability flags (declared via `use NAME --flag`) ───────────────────
    def declare_capability(self, flag: str) -> None:
        self._capabilities.add(flag)

    def has_capability(self, flag: str) -> bool:
        return flag in self._capabilities

    def active(self) -> set[str]:
        """Return the set of currently-activated plugin names."""
        return set(self._active)

    def fidelity(self) -> dict[str, Any]:
        """Return the backend + fidelity record for provenance (doc/36 §3ξ.6).

        Summarises every activated backend plus the explicit capability flags
        declared for the current program.  The declared flags are the fidelity
        class (``--pure-python``/``--approx-euler``/``--low-fidelity``); the
        absence of any reduced-fidelity flag means full fidelity.
        """
        plugins: list[dict[str, Any]] = []
        for name in sorted(self._active):
            prov = self._providers.get(name)
            plugins.append({
                "name": name,
                "backend": prov.native.module if prov and prov.native else "python",
            })
        return {
            "plugins": plugins,
            "capability_flags": sorted(self._capabilities),
            "fidelity": "reduced" if self._capabilities else "full",
        }

    # ── activation ──────────────────────────────────────────────────────────
    def activate(self, name: str) -> Any:
        """Lazily import and validate a named provider.

        Raises:
            PluginMissingError: provider not registered.
            PluginDependencyError: an optional dependency is unsatisfied and no
                reduced-fidelity capability flag has been declared.
        """
        prov = self._providers.get(name)
        if prov is None:
            raise PluginMissingError(name, prov_extra_hint(name))

        unmet = prov.check_dependencies()
        # Any unmet check is a hard error unless the program explicitly opted
        # into a capability flag this plugin honours (doc/36 §3ξ.3: reduced
        # fidelity must be an explicit opt-in, never an implicit fallback).
        honoured = self._capabilities & set(prov.capability_flags)
        for dep in unmet:
            if honoured:
                continue
            raise PluginDependencyError(name, dep, prov.extra)

        self._active.add(name)
        if prov.load is not None:
            return prov.load()
        return import_module(name)


def prov_extra_hint(name: str) -> str:
    """Best-effort pip-extra hint for an unknown plugin name."""
    # Every bundled plugin's extra matches its module name (doc/36 §3.4).
    if name in _BUNDLED_PLUGINS:
        return name
    return "core"


# Standalone registry importable from experiments/scripts.
_default: Registry | None = None


def get_registry() -> Registry:
    """Return the process-wide default :class:`Registry`.

    The default registry auto-discovers the bundled plugins (lazily, name-only)
    so ``use grn`` / ``#use grn`` resolve out of the box.
    """
    global _default
    if _default is None:
        r = Registry()
        r.discover()
        _default = r
    return _default
