"""FBA plugin (doc/36 §7: metabolism.py -> plugins/fba/).

Provides flux balance analysis of metabolic models.  The heavy implementation
remains in :mod:`helixlang.plugins.runtime.metabolism` (import surface preserved for the
validation suite and downstream code); this package is the *plugin-facing* entry
that the registry discovers, owns the ``#media``/``#sim`` keywords, and returns
the FBA backend factory on ``use``/``#use fba``.

Optional-dependency contract (doc/36 §3ξ):
- ``numpy`` is required for the solver (hard check; no fallback).
- ``cobra`` is only needed for *SBML model import*.  When absent and the
  program wants SBML, this is an explicit error unless the user opts into
  ``--low-fidelity`` (the curated core model), matching the long-standing
  ``BioError`` behavior in ``metabolism.py``.
"""
from __future__ import annotations

from collections.abc import Callable

from helixlang.api.registry import PluginProvider
from helixlang.core.errors import PluginDependencyError


def _check(pkg: str) -> bool:
    def _probe() -> bool:
        try:
            __import__(pkg)
            return True
        except ImportError:
            return False
    return _probe()


def _check_numpy() -> bool:
    return _check("numpy")


def _check_cobra() -> bool:
    # cobra is only required for the SBML-import path; treat any import error
    # (eg. missing biopython/optlang dep) as absent.
    return _check("cobra")


def _make_backend(cfg: dict | None = None) -> type:
    """Construct the FBA backend for the current program configuration."""
    from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis

    return FluxBalanceAnalysis


def _load() -> Callable[[dict | None], type]:
    """Activate the plugin: validate core deps (numpy) then return the backend.

    ``cobra`` absence is *not* fatal here — the solver itself never needs it;
    it is gated at SBML-import time and by the registry's ``--low-fidelity``
    opt-in (doc/36 §3ξ.3).
    """
    if not _check_numpy():
        raise PluginDependencyError("fba", "numpy", "fba")
    return _make_backend


PLUGIN = PluginProvider(
    name="fba",
    extra="fba",
    keywords=("media", "sim"),
    native=None,
    capability_flags=("--low-fidelity",),
    checks={"numpy": _check_numpy, "cobra": _check_cobra},
    load=_load,
)
