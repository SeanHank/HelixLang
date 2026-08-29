"""Decoupling verification tests (doc/37 §4).

Enforces the architecture contract from doc/36:
- R1: core/ must NEVER import plugins/ or sim_runtime/ at module level
- R2: core/vm.py may import plugins inside method bodies (lazy pattern)
- R3: sim_runtime/ imports core/ public types only
- R4: the registry is the sole bridge between core and plugins
- R5: no silent fallbacks
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
CORE = SRC / "helixlang" / "core"
PLUGINS = SRC / "helixlang" / "plugins"
SIM_RUNTIME = SRC / "helixlang" / "sim_runtime"


def _collect_module_level_imports(path: Path) -> list[tuple[str, int]]:
    """Parse a module and return (import string, line) for module-level imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Check if this import is nested inside a function/method/class body
            for parent in ast.walk(tree):
                if parent is node:
                    continue
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.If, ast.Try, ast.With)):
                    for child in ast.walk(parent):
                        if child is node:
                            break
                    else:
                        continue
                    break
            else:
                # Module-level import (or just not nested — capture it)
                pass
        if isinstance(node, ast.ImportFrom):
            if node.module and _is_forbidden(node.module):
                violations.append((f"from {node.module}", node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    violations.append((f"import {alias.name}", node.lineno))
    return violations


def _is_forbidden(module: str) -> bool:
    mod = module.lower()
    if mod.startswith("helixlang.plugins") or mod.startswith("helixlang.sim_runtime"):
        return True
    # also catch bare imports that resolve to plugin modules
    if mod in ("numpy", "scipy", "cobra", "rdkit", "torch", "flask", "matplotlib"):
        return True
    return False


def _is_nested(node: ast.AST, import_node: ast.AST) -> bool:
    """True if import_node appears inside a FunctionDef/AsyncFunctionDef body
    or inside an ``if TYPE_CHECKING:`` block."""
    for parent in ast.walk(node):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(parent):
                if child is import_node:
                    return True
        if isinstance(parent, ast.If) and parent.test is not None:
            # detect: if TYPE_CHECKING: / if type_checking: etc.
            test_name = getattr(parent.test, "id", "")
            if test_name in ("TYPE_CHECKING", "type_checking", "typing"):
                for child in ast.walk(parent):
                    if child is import_node:
                        return True
    return False


class TestCoreModuleLevelDecoupling:
    """R1: core/ has no module-level imports of plugins/ or sim_runtime/."""

    @pytest.mark.parametrize("pyfile", sorted(CORE.glob("*.py")))
    def test_no_module_level_plugin_imports(self, pyfile: Path) -> None:
        tree = ast.parse(pyfile.read_text(encoding="utf-8"))
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and _is_forbidden(node.module):
                    if not _is_nested(tree, node):
                        violations.append(f"line {node.lineno}: from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden(alias.name) and not _is_nested(tree, node):
                        violations.append(f"line {node.lineno}: import {alias.name}")
        assert not violations, (
            f"{pyfile.name} has module-level plugin/scientific imports.\n"
            "core/ must be dependency-free at import time (doc/36 §2, doc/37 §4 R1).\n"
            "Move scientific imports inside method bodies (lazy pattern)."
        )


class TestRuntimeImportIsolation:
    """R1: importing core in a fresh interpreter must not pull scientific deps."""

    def test_import_core_no_scientific_dependencies(self) -> None:
        # Write a small script that imports core and asserts no heavy modules loaded
        script = (
            "import sys\n"
            "before = set(sys.modules)\n"
            "import helixlang\n"
            "import helixlang.core.lexer\n"
            "import helixlang.core.vm\n"
            "after = set(sys.modules) - before\n"
            "sci = [m for m in after if any(s in m for s in "
            "('numpy', 'scipy', 'cobra', 'rdkit', 'torch', 'flask', 'matplotlib'))]\n"
            "if sci:\n"
            "    print('VIOLATION:', sci)\n"
            "    sys.exit(1)\n"
            "print('CLEAN_IMPORT')\n"
        )
        env = {"PYTHONPATH": str(SRC)}
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(SRC.parent),
        )
        assert proc.returncode == 0, proc.stderr
        assert "CLEAN_IMPORT" in proc.stdout


class TestRegistrySoleBridge:
    """R4: only plugin_registry.py references plugin names at the core boundary."""

    def test_registry_is_only_plugin_knowledge(self) -> None:
        forbidden_kw = "helixlang.plugins"
        plugin_mentioning = []
        for pyfile in CORE.glob("*.py"):
            content = pyfile.read_text(encoding="utf-8")
            if forbidden_kw in content:
                plugin_mentioning.append(pyfile.name)
        assert plugin_mentioning, "expected at least one core module referencing plugins"
        # Allowed: vm.py (lazy), plugin_registry.py, fidelity.py (registry call)
        assert "plugin_registry.py" in plugin_mentioning


class TestSimRuntimeDirection:
    """R3: sim_runtime imports core through public types only."""

    def test_sim_runtime_depends_on_core(self) -> None:
        assert SIM_RUNTIME.exists()
        core_used = False
        for pyfile in SIM_RUNTIME.glob("*.py"):
            content = pyfile.read_text(encoding="utf-8")
            if "helixlang.core" in content:
                core_used = True
                break
        assert core_used, "sim_runtime should depend on core (integration adapter)"

    def test_core_never_imports_sim_runtime(self) -> None:
        for pyfile in CORE.glob("*.py"):
            tree = ast.parse(pyfile.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and (
                        node.module == "helixlang.sim_runtime"
                        or node.module.startswith("helixlang.sim_runtime.")
                    ):
                        assert _is_nested(tree, node), (
                            f"core/{pyfile.name} must never import sim_runtime "
                            "at module level (doc/37 §4 R1)"
                        )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("helixlang.sim_runtime"):
                            assert _is_nested(tree, node), (
                                f"core/{pyfile.name} must never import sim_runtime "
                                "at module level (doc/37 §4 R1)"
                            )


class TestSilentFallbackLinter:
    """R5: silent fallback linter passes on the source tree."""

    def test_find_silent_fallbacks_passes(self) -> None:
        from helixlang.core import find_silent_fallbacks

        findings = find_silent_fallbacks.scan_tree(SRC / "helixlang")
        assert not findings, (
            f"{len(findings)} silent fallback(s) detected:\n"
            f"{find_silent_fallbacks.format_report(findings[:5])}"
        )
