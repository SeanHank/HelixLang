"""setuptools shim for optional native acceleration builds (doc/36 §4.2.1/§5).

The default build is pure-Python: ``pip install helixlang`` / ``python -m build``
produces a ``py3-none-any`` wheel with **no** compiled hot-loop backends (no
compiler required, ~2 MB install).  Set ``HELIX_BUILD_NATIVE=1`` to compile the
Cython/C accelerators into the wheel (platform-tagged ``cp311-cp311-*``):

    HELIX_BUILD_NATIVE=1 python -m build --wheel

This implements the CI dual-wheel (py + native) shipping requirement (doc/36
§10 Phase 4 item 2).  ``cython>=3.0`` is declared in ``[build-system].requires``
so isolated builds provide it; the hot-loop sources discovered here mirror
``helixlang/_accel/build.py``.
"""
from __future__ import annotations

import os

from setuptools import Extension, setup
from setuptools.command.build_py import build_py as _build_py

_PKG = "helixlang"
_ACCEL = "src/helixlang/_accel"

# Native hot-loop sources that a compiled native wheel must never ship —
# they are build inputs only, never runtime artifacts.  ``build_py`` copies
# *every* file in a package directory, including any ``.c``/``.pyx``/``.h``
# already present in ``src/`` (e.g. left behind by an in-place
# ``helixlang._accel.build``), so we prune them here.  Compiled ``.so``
# backends (installed by ``build_ext``) are untouched.
_NATIVE_SOURCE_SUFFIXES = (".c", ".pyx", ".h")


class _build_py_prune(_build_py):
    """Like build_py but drops native hot-loop sources from the wheel."""

    def run(self):
        super().run()
        # ``build_lib`` holds the assembled package; delete the files there.
        for root, _dirs, files in os.walk(self.build_lib):
            for name in files:
                if name.endswith(_NATIVE_SOURCE_SUFFIXES):
                    os.remove(os.path.join(root, name))


def _ext_modules() -> list[Extension]:
    if os.environ.get("HELIX_BUILD_NATIVE", "0") != "1":
        return []
    import pathlib

    base = pathlib.Path(_ACCEL)
    ext = []
    for pyx in sorted(base.glob("*/impl_cython.pyx")):
        rel = pyx.relative_to(base)
        ext.append(Extension(
            f"{_PKG}._accel.{rel.parent.name}.{rel.stem}", [str(pyx)]))
    for c in sorted(base.glob("*/impl_cext.c")):
        rel = c.relative_to(base)
        ext.append(Extension(
            f"{_PKG}._accel.{rel.parent.name}.{rel.stem}", [str(c)]))
    # Cython sources must be cythonized; do so lazily only when natives are on.
    if any(str(p).endswith(".pyx") for e in ext for p in e.sources):
        try:
            from Cython.Build import cythonize
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "HELIX_BUILD_NATIVE=1 requires Cython (pip install cython)"
            ) from exc
        return list(cythonize(ext, language_level="3"))
    return ext


setup(ext_modules=_ext_modules(), cmdclass={"build_py": _build_py_prune})
