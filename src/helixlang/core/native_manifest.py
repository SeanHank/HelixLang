"""Compiler + VM native stage manifest (doc/06 §19, doc/03 §6.5).

Single source of truth for the *C-only mandate*: which stages of the compiler
and virtual-machine core must execute on a hand-written C kernel
(``impl_cext.c``) and which are still pending their port.

Stage lifecycle:
- ``native``   — a raw-C ``impl_cext`` kernel ships in ``_accel/<stage>/``; the
  stage's package is in ``_NATIVE_ONLY_PACKAGES`` so a Python/numpy backend can
  never be selected (``NativeBackendError`` instead of a silent interpreter
  loop);
- ``pending``  — the stage still ships under ``helixlang/core/<module>.py``;
  the manifest entry records the *planned* ``_accel<stage>`` package so the
  gate and docs can enumerate the full roadmap without inventing names.

Plugins are excluded by the mandate (doc/06 §19): their hot kernels
(``grn_step``, ``diffusion``, ``simplex``, …) keep explicit capability flags.
"""
from __future__ import annotations

from typing import Final

#: (``_accel`` package, stage name, status).  Order = front-end front to back:
#: lexer -> parser -> semantic -> compiler -> hxbc -> ir -> vm (dispatch).
STAGES: Final = (
    ("helixlang._accel.lexer", "Lexer / dual-mode scanner", "native"),
    ("helixlang._accel.parser", "Parser (token stream -> AST)", "pending"),
    ("helixlang._accel.semantic", "Semantic analyser", "pending"),
    ("helixlang._accel.compiler", "Codon decode -> bytecode emitter", "pending"),
    ("helixlang._accel.hxbc", "hxbc codec (chunk <-> bytes)", "pending"),
    ("helixlang._accel.ir", "IR pipeline (builder/lower/opt)", "pending"),
    ("helixlang._accel.dispatch", "VM opcode dispatch hot loop", "native"),
    ("helixlang._accel.vm", "Full-opcode VM execution engine", "pending"),
    ("helixlang._accel.disassembler", "Bytecode disassembler", "pending"),
)


def native_stages() -> tuple[tuple[str, str], ...]:
    """``(package, name)`` for the stages already shipping a C kernel."""
    return tuple((pkg, name) for pkg, name, status in STAGES
                 if status == "native")


def pending_stages() -> tuple[tuple[str, str], ...]:
    """``(package, name)`` for the stages still shipping pure-Python kernels."""
    return tuple((pkg, name) for pkg, name, status in STAGES
                 if status == "pending")


def native_packages() -> frozenset[str]:
    """Packages refused a Python/numpy backend by :mod:`helixlang._accel._loaders`."""
    return frozenset(pkg for pkg, _, status in STAGES if status == "native")


def stage_names() -> tuple[str, ...]:
    """Stable ordering of stage names for gate output and docs."""
    return tuple(name for _, name, _ in STAGES)


if __name__ == "__main__":
    for pkg, name, status in STAGES:
        print(f"{status:8s} {name} ({pkg})")
