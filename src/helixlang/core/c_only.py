"""C-only mandate enforcement for the compiler + VM core (doc/03 §6.5, doc/06 §19).

The compiler and virtual-machine hot path are mandated to run on a compiled C
kernel, not a Python/numpy implementation (plugins are exempt).  This gate
verifies, for every native stage registered in
:mod:`helixlang.core.native_manifest`, two properties that are structurally
enforced by the loader:

1. the stage's ``_accel`` package resolves to the native C backend
   (``impl_cext``) — never to a Python fallback;
2. requesting a Python implementation for that package raises
   ``NativeBackendError`` (i.e. no silent Python hot loop).

Deployments that do not carry the compiled kernel fail this gate loudly with a
rebuild hint; they never silently degrade the VM hot loop to an interpreter
loop (doc/36 §3ξ.4 — no silent fallback).
"""

from __future__ import annotations

from helixlang._accel._loaders import choose_backend, load_hot
from helixlang.core.errors import NativeBackendError
from helixlang.core.native_manifest import native_stages


def stage_backend(pkg: str) -> str | None:
    """Resolver's backend module name for ``pkg``, or ``None`` if no compiled
    kernel is importable."""
    try:
        mod = load_hot(pkg)
    except NativeBackendError:
        return None
    return str(mod.__name__)


def python_not_selectable(pkg: str) -> bool:
    """True when forcing ``python`` for ``pkg`` raises (the C-only guarantee
    that a Python hot loop can never be selected)."""
    try:
        choose_backend(pkg, prefer="python")
    except NativeBackendError:
        return True
    return False


def _stage_check(pkg: str, name: str) -> tuple[bool, str]:
    """Check one native stage against the C-only mandate."""
    backend = stage_backend(pkg)
    if backend is None:
        return False, (
            f"{name} C kernel not built; run `python -m helixlang._accel.build` "
            "or `pip install helixlang[native]`"
        )
    if not backend.endswith("impl_cext"):
        return False, (
            f"{name} resolved to non-C backend {backend!r} "
            "(C-only mandate, doc/03 §6.5)"
        )
    if not python_not_selectable(pkg):
        return False, (
            f"{name} still permits a Python hot loop "
            "(C-only mandate, doc/03 §6.5)"
        )
    return True, f"{name} is C-only ({backend})"


def check() -> tuple[bool, str]:
    """Return ``(ok, message)`` for the C-only mandate across all native stages."""
    verdicts = [_stage_check(pkg, name) for pkg, name in native_stages()]
    if not verdicts:
        return False, "no native stages registered in the manifest (doc/03 §6.5)"
    if all(ok for ok, _ in verdicts):
        return True, "; ".join(msg for _, msg in verdicts)
    return False, "; ".join(msg for ok, msg in verdicts if not ok)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: print the mandate verdict and exit 0/1 (quality gate)."""
    ok, msg = check()
    print(f"[c-only] {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
