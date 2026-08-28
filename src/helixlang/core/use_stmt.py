"""The ``use`` program statement (doc/36 §4).

``use`` is the explicit opt-in mechanism that replaces silent fallbacks.  Syntax
(embedded in a helix source file):

.. code-block:: helix

    #use grn --pure-python --approx-euler
    #use metabolism

Each statement names a plugin and zero or more capability flags that relax
fidelity *explicitly*.  A program that never says ``use`` simply auto-detects
installed plugins; the semantics are identical for the common case, which keeps
existing sources valid.

This module is the pure model + parser used by the CLI and the VM; it has no
dependency on the compiled language pipeline.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from helixlang.core.plugin_registry import Registry

# Capability flags that may appear after a plugin name.  Each lowers fidelity
# (or switches implementation) and therefore must be declared explicitly — this
# is the core anti-silent-fallback guarantee (doc/36 §3ξ).
KNOWN_FLAGS = (
    "--pure-python",    # use a pure-Python implementation even if a native one exists
    "--approx-euler",   # use the approximate fixed-step Euler integrator (vs adapative)
    "--low-fidelity",   # allow reduced-fidelity scientific models when deps are missing
)

# Flags that may NOT be combined because they are contradictory.
INCOMPATIBLE: tuple[tuple[str, str], ...] = (
    ("--pure-python", "native"),
)


@dataclass(slots=True)
class UseDirective:
    """A single parsed ``#use`` line."""

    plugin: str
    flags: set[str] = field(default_factory=set)
    model: str | None = None    # optional quoted model/alias, e.g. "ecoli_core"
    line: int = 0
    col: int = 0

    def provides(self, flag: str) -> bool:
        return flag in self.flags


class UseError(ValueError):
    """Malformed ``#use`` directive (unknown flag, duplicate plugin, ...)."""


def parse_use_line(text: str, *, line: int = 0, col: int = 0) -> UseDirective:
    """Parse the content after ``#use`` into a :class:`UseDirective`.

    Expects ``text`` to be the whitespace-separated remainder of the line, e.g.
    ``"grn --pure-python"`` or ``'fba "ecoli_core"'``.  An optional quoted
    model/alias argument may follow the plugin name (doc/36 §3.2); the rest must
    be recognised capability flags.
    """
    parts = text.split()
    if not parts:
        raise UseError("`use` requires a plugin name", )
    plugin = parts[0]
    if not plugin.isidentifier():
        raise UseError(f"invalid plugin name {plugin!r}")
    model: str | None = None
    flags: set[str] = set()
    i = 1
    # Optional quoted model/alias immediately after the plugin name.
    if i < len(parts) and parts[i].startswith('"') and parts[i].endswith('"'):
        model = parts[i][1:-1]
        i += 1
    for tok in parts[i:]:
        if tok not in KNOWN_FLAGS:
            raise UseError(
                f"unknown capability flag {tok!r} for plugin {plugin!r} "
                f"(known: {', '.join(KNOWN_FLAGS)})")
        flags.add(tok)
    if "--pure-python" in flags and any(
        a in flags and b in flags for (a, b) in INCOMPATIBLE
    ):
        raise UseError(
            f"flag set for {plugin!r} is internally incompatible")
    d = UseDirective(plugin=plugin, flags=flags, model=model, line=line, col=col)
    return d


def emit_use_statements(plugin: str, flags: tuple[str, ...] = ()) -> str:
    """Render a ``#use`` line (inverse of :func:`parse_use_line`)."""
    suffix = (" " + " ".join(flags)) if flags else ""
    return f"#use {plugin}{suffix}"


def apply_use_directives(
    directives: Iterable[UseDirective],
    registry: Registry,
) -> list[str]:
    """Activate plugins declared by a program's ``#use`` statements.

    ``directives`` is an iterable of :class:`UseDirective` (or any object with
    ``plugin`` and ``flags`` attributes).  Each plugin is activated through the
    ``registry`` (declared capability flags first), returning the list of
    activated plugin names.  Missing/unavailable plugins raise the registry's
    explicit errors (``SemanticError`` must already have validated names; this
    step only triggers dependency checks / native selection).
    """
    from helixlang.core.plugin_registry import PluginMissingError

    activated: list[str] = []
    for d in directives:
        for flag in d.flags:
            registry.declare_capability(flag)
        if not registry.is_registered(d.plugin):
            raise PluginMissingError(d.plugin, "core")
        registry.activate(d.plugin)
        activated.append(d.plugin)
    return activated
