"""Native acceleration build (doc/36 §4.2.1 / §5; doc/39 O12 rust).

Compiles the optional compiled hot-loop backends — ``impl_cython`` from .pyx,
``impl_cext`` from .c, and ``impl_rust`` (a Rust/PyO3 ``abi3`` extension) from
its embedded ``cargo`` crate — directly into the source tree so the ``_accel``
loader picks them up ahead of the numpy/python stacks::

    python -m helixlang._accel.build

This is the ``rebuild_cmd`` referenced by :class:`~helixlang.core.errors.
NativeBackendError` and is run only when a *declared* native backend is missing.
Normal ``pip install`` remains pure-Python (a native wheel is built from CI via
the separate ``helixlang._accel.build`` entry, not from the shared pyproject
metadata), so installs never require a compiler unless the operator opts in.

Building is best-effort: if Cython/setuptools is absent the CLI prints a hint
and exits nonzero — the loader then raises ``NativeBackendError`` unless the
program declared ``--pure-python``.  The Rust backend requires only ``cargo``
(no maturin/setuptools-rust); the resulting ``impl_rust.abi3.so`` is version-
independent across supported CPython ABIs.
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


def _rust_crates() -> list[Path]:
    """Rust/PyO3 backend crates: any ``*/rust/Cargo.toml`` under ``_accel``."""
    return sorted(_BASE.glob("*/rust/Cargo.toml"))


def _build_rust_backends() -> int:
    """Compile Rust/PyO3 ``impl_rust`` backends via cargo (best-effort).

    For each ``<pkg>/rust/Cargo.toml`` crate, runs ``cargo build --release``
    (with ``PYO3_PYTHON`` from the running interpreter) and copies the resulting
    extension (``lib_*.dylib``/``.so``) into the owning ``_accel`` package as
    ``impl_rust.abi3.so`` so the loader picks it up as a native backend (doc/39
    O12).  Because the artifact is ``abi3`` (stable Python ABI) it imports on any
    supported interpreter without rebuild.
    """
    crates = _rust_crates()
    if not crates:
        return 0
    import os
    import shutil
    import subprocess

    env = dict(os.environ)
    env.setdefault("PYO3_PYTHON", os.environ.get("PYTHON", "python"))
    built = 0
    for manifest in crates:
        crate_dir = manifest.parent
        try:
            subprocess.run(
                ["cargo", "build", "--release"],
                cwd=str(crate_dir), env=env, check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"Rust build of {crate_dir.name} skipped: {exc}")
            continue
        pkg_dir = crate_dir.parent
        ext_dir = crate_dir / "target" / "release"
        for so in sorted(ext_dir.glob("lib*.dylib")) + sorted(
                ext_dir.glob("*.so")):
            dest = pkg_dir / "impl_rust.abi3.so"
            shutil.copy(so, dest)
            print(f"built {dest.relative_to(_BASE)}")
            built += 1
    return 1 if built else 0


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
    rust_rc = _build_rust_backends()
    if not built and rust_rc == 0:
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
