"""GRN plugin (doc/36 §7: grn.py -> plugins/grn/).

Provides the gene regulatory network capability.  The heavy implementation
remains in :mod:`helixlang.plugins.runtime.grn` (import surface preserved for the validation
suite and downstream code); this package is the *plugin-facing* entry that the
registry discovers, owns the ``#gene``/``#regulate`` keywords, and returns the
GRN backend factory on ``use``/``#use grn``.
"""
from __future__ import annotations

from collections.abc import Callable

from helixlang.api.registry import PluginProvider
from helixlang.core.errors import PluginDependencyError


def _check_numpy() -> bool:
    """numpy is an optional dependency for the vectorized GRN path.

    Return False only when we are sure it is absent (any import error).  The
    pure-Python reference always works; ``step_accel``'s numpy batch kernel is
    the optional accelerated path (doc/36 §4.1 P1).
    """
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def _make_backend(cfg: dict | None = None) -> type:
    """Construct the GRN backend for the current program configuration."""
    from helixlang.plugins.runtime.grn import GRN

    return GRN


def _load() -> Callable[[dict | None], type]:
    """Activate the plugin: validate optional deps then return the backend."""
    if not _check_numpy():
        raise PluginDependencyError("grn", "numpy", "grn")
    return _make_backend


PLUGIN = PluginProvider(
    name="grn",
    extra="grn",
    keywords=("gene", "regulate"),
    native=None,
    capability_flags=("--pure-python",),
    checks={"numpy": _check_numpy},
    load=_load,
)
