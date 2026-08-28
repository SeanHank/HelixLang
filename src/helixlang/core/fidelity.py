"""Reduced-fidelity opt-in gate (doc/36 §3ξ.3).

Helper for converting silent fallbacks into explicit opt-ins.  A function that
would otherwise silently degrade (missing optional model/dep/backend → lower
fidelity result) must raise a typed error UNLESS the caller explicitly opted in
by:

1. passing ``allow_low_fidelity=True`` directly, or
2. declaring the corresponding capability flag on the process registry
   (``use <name> --low-fidelity`` / ``--approx-euler`` / ``--pure-python``), or
3. setting the documented environment variable (e.g. ``HELIX_ALLOW_LOW_FIDELITY=1``).

The default is the strict mode: a missing high-fidelity path errors rather than
silently computing at lower fidelity.
"""
from __future__ import annotations

import os

_STANDARD_FLAGS = ("--low-fidelity", "--approx-euler", "--pure-python")


def _declared(flag: str) -> bool:
    try:
        from helixlang.core.plugin_registry import get_registry
        return get_registry().has_capability(flag)
    except Exception:  # noqa: BLE001 - registry absent in a bare import
        return False


def opt_in(flag: str = "--low-fidelity", *, allow: bool = False,
           env: str = "HELIX_ALLOW_LOW_FIDELITY") -> bool:
    """True if the reduced-fidelity path is explicitly permitted.

    ``allow`` is an explicit per-call opt-in; ``flag`` is a standard capability
    flag the program may have declared; ``env`` is a documented env override.
    """
    if allow:
        return True
    if flag and _declared(flag):
        return True
    if env and os.environ.get(env) in ("1", "true", "True", "yes"):
        return True
    return False


def require(flag: str = "--low-fidelity", *, allow: bool = False,
            env: str = "HELIX_ALLOW_LOW_FIDELITY",
            name: str, dep: str, extra: str) -> None:
    """Raise :class:`~helixlang.core.errors.PluginDependencyError` unless the
    reduced-fidelity path is explicitly opted in."""
    if opt_in(flag, allow=allow, env=env):
        return
    from helixlang.core.errors import PluginDependencyError
    raise PluginDependencyError(name, dep, extra)
