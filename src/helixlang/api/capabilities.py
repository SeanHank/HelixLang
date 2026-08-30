"""Declared, typed capabilities (doc/38 §6.6 ``api.capabilities``).

Extends the string capability flags of ``PluginProvider.capability_flags``
with declared capabilities: a plugin *claims* one in its manifest
(``[capabilities] flags``) and ``#use <plugin> --flag`` remains the only way to
activate it.  ``reduces_fidelity=True`` drives the existing fidelity record so
provenance stays honest.  ``opt_in`` re-exports ``core.fidelity.opt_in`` — the
strict reduced-fidelity gate plugins use before degrading an optional model.
"""
from __future__ import annotations

from dataclasses import dataclass

from helixlang.core.fidelity import opt_in as opt_in  # noqa: F401

__all__ = ["Capability", "PURE_PYTHON", "LOWER_FIDELITY", "opt_in"]

#: Canonical full-fidelity capability (no reduced-fidelity claim).
PURE_PYTHON = "--pure-python"

#: Canonical reduced-fidelity capability (opt-in, never implicit).
LOWER_FIDELITY = "--lower-fidelity"


@dataclass(frozen=True)
class Capability:
    """A declarable capability a plugin may claim in its manifest.

    ``id`` is the ``#use <plugin> --<id>`` flag; ``summary`` feeds
    ``helixc plugin info``; ``reduces_fidelity`` feeds ``Registry.fidelity``
    so a reduced-fidelity run is always provenance-recorded as such.
    """

    id: str
    summary: str
    reduces_fidelity: bool
