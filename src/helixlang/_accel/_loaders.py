"""Shared backend selector for hot-loop packages (doc/36 §4.2).

Selection is a **speed-only switch among implementations of the SAME algorithm
with IDENTICAL numerics**.  It NEVER silently crosses a fidelity boundary
(e.g. numpy-GRN -> pure-python-GRN-that-changes-rounding, scipy-RK45 -> Euler,
cobra -> built-in model, rdkit -> MW-only, esm -> Chou-Fasman).  Crossing such a
boundary must be declared explicitly by the caller (e.g. a ``use`` capability
flag, doc/36 §3ξ.3) — otherwise the caller raises
:class:`~helixlang.core.errors.NativeBackendError` /
``PluginMissingError`` and the run FAILS rather than degrades.

If the requested backend is unavailable, :func:`choose_backend` raises — it does
NOT reselect.  Program authors choose fidelity via explicit ``use`` flags; this
selector only maps a chosen fidelity level to its fastest available impl.
"""
from __future__ import annotations

import importlib
import os
from typing import Any

from helixlang.core.errors import NativeBackendError

# Technology tags in default priority order.  `native` expands to whichever
# compiled impl (cext/cython/rust) is present on disk.
_NATIVE_IMPLS = ("impl_cext", "impl_cython", "impl_rust")
_SUFFIX_IMPLS = ("impl_numpy", "impl_numba", "impl_python")


def _importable(pkg: str, impl: str) -> bool:
    try:
        mod = importlib.import_module(f"{pkg}.{impl}")
    except (ImportError, ModuleNotFoundError):
        return False
    return bool(getattr(mod, "_NATIVE_PRESENT", True))


def choose_backend(pkg: str, prefer: str | None = None) -> str:
    """Return the implementation module name for a hot-loop ``pkg``.

    Priority is configurable via env ``HELIX_ACCEL`` (comma-separated tags,
    e.g. ``native,numpy,python``) or the ``prefer`` argument.  ``native`` means
    any compiled impl (cext/cython) present on disk.

    Raises:
        NativeBackendError: when the *chosen* implementation is absent.  This is
            deliberate — there is no silent fallback to another fidelity class.

    Default order ``native,numpy,python`` auto-selects among **equivalent
    fidelity** implementations (a pure speed switch, doc/36 §3ξ.5); an operator
    may override with ``HELIX_ACCEL`` to require a specific stack (e.g.
    ``HELIX_ACCEL=native``), in which case an absent backend raises
    `NativeBackendError` rather than silently degrading.
    """
    order = (prefer or os.environ.get("HELIX_ACCEL") or "native,numpy,python").split(",")
    order = [o.strip() for o in order if o.strip()]
    tried: list[str] = []
    for tag in order:
        if tag == "native":
            for impl in _NATIVE_IMPLS:
                tried.append(f"{pkg}.{impl}")
                if _importable(pkg, impl):
                    return impl
        elif tag in _SUFFIX_IMPLS or tag == "python":
            impl = "impl_python" if tag == "python" else tag
            tried.append(f"{pkg}.{impl}")
            if _importable(pkg, impl):
                return impl
        else:
            tried.append(f"{pkg}.impl_{tag}")
            if _importable(pkg, f"impl_{tag}"):
                return f"impl_{tag}"
    raise NativeBackendError(
        f"No implementation of {pkg} for requested backend(s) "
        f"{','.join(order)!r}. Tried: {', '.join(tried)}. "
        f"Rebuild with `pip install helixlang[native]`, or declare the "
        f"explicit `--pure-python` capability flag to use the pure-Python impl.",
        rebuild="pip install helixlang[native]",
    )


def load_hot(pkg: str, prefer: str | None = None) -> Any:
    """Import and return a hot-loop package's chosen backend module."""
    name = choose_backend(pkg, prefer=prefer)
    return importlib.import_module(f"{pkg}.{name}")


def backend_for(pkg: str, prefer: str | None = None) -> Any:
    """Return the ``backend`` module of a hot-loop package (doc/36 layout)."""
    return importlib.import_module(f"{pkg}.backend")
