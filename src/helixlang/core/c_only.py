"""C-only mandate enforcement for the compiler + VM core (doc/03 §6.5, doc/06 §19).

The compiler and virtual-machine hot path are mandated to run on a compiled C
kernel, not a Python/numpy implementation (plugins are exempt).  This gate
verifies two properties that are structurally enforced by the loader:

1. ``helixlang._accel.dispatch`` resolves to the native C backend
   (``impl_cext``) — never to a Python fallback;
2. requesting a Python implementation for the dispatch package raises
   ``NativeBackendError`` (i.e. no silent Python hot loop).

Deployments that do not carry the compiled kernel fail this gate loudly with a
rebuild hint; they never silently degrade the VM hot loop to an interpreter
loop (doc/36 §3ξ.4 — no silent fallback).
"""

from __future__ import annotations

from helixlang._accel._loaders import choose_backend, load_hot
from helixlang.core.errors import NativeBackendError

_DISPATCH_PKG = "helixlang._accel.dispatch"


def dispatch_backend() -> str | None:
    """Resolver's backend module name for the dispatch kernel, or ``None`` if
    no compiled kernel is importable."""
    try:
        mod = load_hot(_DISPATCH_PKG)
    except NativeBackendError:
        return None
    return str(mod.__name__)


def python_not_selectable() -> bool:
    """True when forcing ``python`` for the dispatch package raises (the C-only
    guarantee that a Python hot loop can never be selected)."""
    try:
        choose_backend(_DISPATCH_PKG, prefer="python")
    except NativeBackendError:
        return True
    return False


def check() -> tuple[bool, str]:
    """Return ``(ok, message)`` for the C-only mandate on the dispatch kernel."""
    backend = dispatch_backend()
    if backend is None:
        return (
            False,
            "dispatch C kernel not built; run `python -m helixlang._accel.build` "
            "or `pip install helixlang[native]`",
        )
    if not backend.endswith("impl_cext"):
        return (
            False,
            f"dispatch resolved to non-C backend {backend!r} "
            f"(C-only mandate, doc/03 §6.5)",
        )
    if not python_not_selectable():
        return (
            False,
            "dispatch still permits a Python hot loop "
            "(C-only mandate, doc/03 §6.5)",
        )
    return True, f"dispatch hot loop is C-only ({backend})"


def main(argv: list[str] | None = None) -> int:
    """CLI entry: print the mandate verdict and exit 0/1 (quality gate)."""
    ok, msg = check()
    print(f"[c-only] {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

