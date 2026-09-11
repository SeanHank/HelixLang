"""Tests for helixlang.core.find_core_imports.

Covers the import-boundary scanner: Violation dataclass, _module_names,
scan() over directories and files, syntax-error resilience, and the CLI
main() entry point (text, JSON, and strict modes).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from helixlang.core import find_core_imports as fci


def _write(dirpath: Path, name: str, content: str) -> Path:
    p = dirpath / name
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def clean_root(tmp_path: Path) -> Path:
    root = tmp_path / "clean"
    root.mkdir()
    _write(
        root, "ok.py",
        "from helixlang.api import build\n"
        "from helixlang.core.errors import HelixError\n"
        "def f():\n    return build\n",
    )
    _write(
        root, "rel.py",
        "from helixlang.plugins.gem import bridge\n"
        "from . import sibling\n"
        "import os\n"
    )
    return root


@pytest.fixture
def dirty_root(tmp_path: Path) -> Path:
    root = tmp_path / "dirty"
    root.mkdir()
    _write(
        root, "bad.py",
        "from helixlang.core.compiler import compile\n"
        "import helixlang.interop\n"
        "from helixlang.sim_runtime.population import run\n",
    )
    return root


# ── Violation / _module_names ──────────────────────────────────────────────

def test_violation_dataclass(tmp_path):
    v = fci.Violation("helixlang.interop", tmp_path / "x.py", 5, known=False)
    assert v.module == "helixlang.interop"
    assert v.lineno == 5
    assert v.known is False


def test_module_names_plain_import(tmp_path):
    tree = __import__("ast").parse("import os\nimport helixlang.interop\n")
    names = fci._module_names(tree)
    assert names == [("os", 1), ("helixlang.interop", 2)]


def test_module_names_from_import_and_relative(tmp_path):
    tree = ast.parse(
        "from helixlang.api import a\n"
        "from . import rel\n"
        "from .sub import x\n"
    )
    names = fci._module_names(tree)
    assert ("helixlang.api", 1) in names
    assert all("rel" not in m and not m.startswith(".") for m, _ in names)
    assert len(names) == 1


def test_module_names_importfrom_without_module():
    node = ast.ImportFrom(
        module=None, level=0, lineno=1, col_offset=0,
        names=[ast.alias(name="bare", asname=None)],
    )
    tree = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(tree)
    assert fci._module_names(tree) == []


# ── scan() ─────────────────────────────────────────────────────────────────

def test_scan_clean_directory(clean_root):
    assert fci.scan([clean_root]) == []


def test_scan_reports_violations(dirty_root):
    v = fci.scan([dirty_root])
    modules = {x.module for x in v}
    assert {"helixlang.core.compiler", "helixlang.interop"} <= modules
    assert all(not x.known for x in v)


def test_scan_accepts_file_path(dirty_root):
    bad = dirty_root / "bad.py"
    v = fci.scan([bad])
    assert len(v) == 3
    assert all(x.path == bad for x in v)


def test_scan_nonexistent_path(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert fci.scan([missing]) == []


def test_scan_skips_syntax_error(tmp_path):
    root = tmp_path / "syn"
    root.mkdir()
    _write(root, "broken.py", "def f(:\n")
    _write(root, "good.py", "import helixlang.interop\n")
    v = fci.scan([root])
    assert len(v) == 1
    assert v[0].path.name == "good.py"


def test_scan_allowed_and_sibling(clean_root):
    v = fci.scan([clean_root])
    assert v == []


# ── main() ─────────────────────────────────────────────────────────────────

def test_main_clean_text(capsys, clean_root):
    assert fci.main([str(clean_root)]) == 0
    out = capsys.readouterr().out
    assert "clean" in out


def test_main_clean_json(capsys, clean_root):
    assert fci.main(["--json", str(clean_root)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["violations"] == []
    assert data["strict"] is False


def test_main_violations_text(capsys, dirty_root):
    assert fci.main([str(dirty_root)]) == 1
    out = capsys.readouterr().out
    assert "import-boundary violation" in out


def test_main_violations_json(capsys, dirty_root):
    assert fci.main(["--json", str(dirty_root)]) == 1
    data = json.loads(capsys.readouterr().out)
    assert len(data["violations"]) == 3
    assert data["strict"] is False


def test_main_strict_json(capsys, dirty_root):
    assert fci.main(["--strict", "--json", str(dirty_root)]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["strict"] is True
