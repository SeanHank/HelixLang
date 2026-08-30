"""Phase E5: plugin boundary purity — zero exemptions (doc/38 §6.9).

Plugins may import **only** ``helixlang.api.*`` and ``helixlang.core.errors``.
These tests make that a live-tree, zero-exemption contract: any new core /
runtime / interop / accel import in a plugin is an immediate failure, and the
``KNOWN_COMPLIANT_EXCEPTIONS`` allowlist must stay empty.
"""
from __future__ import annotations

import re
from pathlib import Path

from helixlang.api import __all__ as api_all
from helixlang.core import find_core_imports as fci

_KNOWN_ALLOWED = ("helixlang.api", "helixlang.core.errors")

_FORBIDDEN_IMPORT_RE = re.compile(
    r"^(?:from|import)\s+helixlang\.(?:core(?!\.errors)|sim_runtime|interop|_accel)\b",
)

PLUGINS_ROOT = Path(__file__).resolve().parents[1] / "src" / "helixlang" / "plugins"


def test_known_exceptions_are_empty():
    assert fci.KNOWN_COMPLIANT_EXCEPTIONS == frozenset()


def test_live_tree_is_pure():
    v = fci.scan([PLUGINS_ROOT])
    assert v == []
    assert not any(x.known for x in v)


def test_strict_gate_is_clean():
    assert fci.main(["--strict"]) == 0


def test_no_forbidden_import_text_in_plugins():
    hits: list[tuple[str, str]] = []
    for path in sorted(PLUGINS_ROOT.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if _FORBIDDEN_IMPORT_RE.match(stripped):
                hits.append((f"{path.relative_to(PLUGINS_ROOT)}:{lineno}", stripped))
    assert hits == [], "\n".join(f"{path} → {line}" for path, line in hits)


def test_api_surface_is_stdlib_only():
    import subprocess
    import sys

    # Checked in a fresh interpreter so sibling tests cannot pre-load the sim
    # stack: importing ``helixlang.api`` must never pull numpy/scipy/pandas/rdkit
    # or any ``helixlang.sim_runtime`` / ``helixlang.plugins`` module into
    # ``sys.modules``.
    code = (
        "import sys; import helixlang.api; "
        "bad={m for m in sys.modules if m.startswith(("
        "'numpy','pandas','rdkit','scipy',"
        "'helixlang.sim_runtime','helixlang.plugins'))}; "
        "raise SystemExit(bool(bad))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert set(api_all) >= {"accel", "ast", "backend", "bytecode", "capabilities",
                            "compiler", "errors", "gem", "grammar", "language",
                            "registry", "sbol", "units"}
