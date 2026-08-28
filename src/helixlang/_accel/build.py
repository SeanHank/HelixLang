"""Native acceleration build (doc/36 §4.2.1 / §5).

Compiles the optional compiled hot-loop backends (``impl_cython`` from .pyx and
``impl_cext`` from .c) directly into the source tree so the ``_accel`` loader
picks them up ahead of the numpy/python stacks::

    python -m helixlang._accel.build

This is the ``rebuild_cmd`` referenced by :class:`~helixlang.core.errors.
NativeBackendError` and is run only when a *declared* native backend is missing.
Normal ``pip install`` remains pure-Python (a native wheel is built from CI via
the separate ``helixlang._accel.build`` entry, not from the shared pyproject
metadata), so installs never require a compiler unless the operator opts in.

Building is best-effort: if Cython or setuptools is absent, the CLI prints a
hint and exits nonzero — the loader then raises ``NativeBackendError`` unless
the program declared ``--pure-python``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

_BASE = Path(__file__).resolve().parent          # helixlang/_accel
_PKG = "helixlang"


def _cython_sources() -> list[Path]:
    return sorted(_BASE.glob("*/impl_cython.pyx"))


def _c_sources() -> list[Path]:
    return sorted(_BASE.glob("*/impl_cext.c"))


def _module_name(src: Path) -> str:
    rel = src.relative_to(_BASE)
    return f"{_PKG}._accel.{rel.parent.name}.{src.stem}"


def _bootstrap_imports():
    """Return (cythonize_or_None, Extension, build_ext, Distribution)."""
    try:
        from Cython.Build import cythonize
    except ImportError:
        cythonize = cast(Any, None)
    from setuptools import Extension
    from setuptools.command.build_ext import build_ext
    from setuptools.dist import Distribution
    return cythonize, Extension, build_ext, Distribution


def build_extensions() -> int:
    cythonize, Extension, build_ext, Distribution = _bootstrap_imports()
    pyx = _cython_sources()
    csrc = _c_sources()
    if not pyx and not csrc:
        print("No native sources found under helixlang/_accel/: nothing to build.")
        return 0
    if pyx and cythonize is None:
        print("Cython is required to build impl_cython (pip install cython).")
        return 1

    ext_modules = [
        Extension(_module_name(p), [str(p)]) for p in pyx
    ] + [
        Extension(_module_name(c), [str(c)]) for c in csrc
    ]
    if cythonize is not None:
        ext_modules = cythonize(ext_modules, language_level="3")

    dist = Distribution({
        "ext_modules": ext_modules,
        "package_dir": {"": _root_src()},
        "zip_safe": False,
    })
    cmd = build_ext(dist)
    cmd.inplace = 1
    cmd.ensure_finalized()
    try:
        cmd.run()
    except Exception as exc:  # noqa: BLE001 - surface toolchain errors clearly
        print(f"Native build failed: {exc}")
        print("Declare `--pure-python` (or use HELIX_ACCEL=python) to run "
              "without a compiled backend.")
        return 1

    built = sorted(_BASE.glob("*/impl_cext*.so")) + sorted(
        _BASE.glob("*/impl_cython*.so"))
    for so in built:
        print(f"built {so.name}")
    if not built:
        print("No .so produced; check the compiler toolchain.")
        return 1
    return 0


def _root_src() -> str:
    """src/ dir for a src-layout install, else the install root."""

    # The `helixlang` package dir; its parent holds `src` only in editable
    # src-layout installs.  Compiling in place needs the *source* root.
    import helixlang
    return str(Path(helixlang.__file__).resolve().parent.parent)


def main() -> None:
    raise SystemExit(build_extensions())


if __name__ == "__main__":
    main()
