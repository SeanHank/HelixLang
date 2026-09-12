"""Unit tests for the stub/deferred-code linter (find_stubs)."""
from __future__ import annotations

from helixlang.core.find_stubs import (
    Finding,
    format_report,
    main,
    scan_file,
    scan_tree,
)


def _write(tmp_path, code, name="mod.py"):
    p = tmp_path / name
    p.write_text(code)
    return p


def test_ellipsis_body_is_s1(tmp_path):
    p = _write(tmp_path, "def density(a, b):\n    ...\n")
    findings = scan_file(p)
    assert findings and findings[0].category == "S1"


def test_protocol_ellipsis_exempt(tmp_path):
    p = _write(tmp_path,
               "from typing import Protocol\n"
               "class Model(Protocol):\n"
               "    def predict(self, s: str) -> float:\n"
               "        ...\n")
    assert scan_file(p) == []


def test_pass_only_body_is_s2(tmp_path):
    p = _write(tmp_path, "def f():\n    pass\n")
    findings = scan_file(p)
    assert findings and findings[0].category == "S2"


def test_pass_with_docstring_is_s2(tmp_path):
    p = _write(tmp_path, "def f():\n    \"\"\"doc\"\"\"\n    pass\n")
    findings = scan_file(p)
    assert findings and findings[0].category == "S2"


def test_benign_marker_suppresses_s2(tmp_path):
    p = _write(tmp_path,
               "def f():\n"
               "    # STUBBENIGN no-op hook\n"
               "    pass\n")
    assert scan_file(p) == []


def test_benign_marker_example_from_docstring(tmp_path):
    p = _write(tmp_path,
               "def f():\n"
               "    # STUBBENIGN handled inline\n"
               "    pass\n")
    assert scan_file(p) == []


def test_raise_not_implemented_is_s3(tmp_path):
    p = _write(tmp_path, "def f():\n    raise NotImplementedError(\"nope\")\n")
    findings = scan_file(p)
    assert findings and findings[0].category == "S3"


def test_raise_benign_suppressed(tmp_path):
    p = _write(tmp_path,
               "def f():\n"
               "    # STUBBENIGN misconfigured-backend guard\n"
               "    raise NotImplementedError(\"nope\")\n")
    assert scan_file(p) == []


def test_marker_above_multiline_raise(tmp_path):
    p = _write(tmp_path,
               "def f():\n"
               "    # first reason line\n"
               "    # STUBBENIGN second reason line\n"
               "    raise NotImplementedError(\n"
               "        \"text\")\n")
    assert scan_file(p) == []


def test_todo_comment_is_s4(tmp_path):
    p = _write(tmp_path, "# TODO: wire the solver\nx = 1\n")
    findings = scan_file(p)
    assert findings and findings[0].category == "S4"


def test_not_implemented_comment_is_s5(tmp_path):
    p = _write(tmp_path, "# not yet implemented: GPU path\nx = 1\n")
    findings = scan_file(p)
    assert findings and findings[0].category == "S5"


def test_placeholder_comment_is_s6(tmp_path):
    p = _write(tmp_path, "x = 0.0  # placeholder default\n")
    findings = scan_file(p)
    assert findings and findings[0].category == "S6"


def test_negation_directly_before_marker_suppresses_s6(tmp_path):
    p = _write(tmp_path, "# never a placeholder: real value\nx = 1.0\n")
    assert scan_file(p) == []


def test_negation_after_marker_does_not_suppress(tmp_path):
    p = _write(tmp_path, "# dummy route, not used yet\nx = 1\n")
    findings = scan_file(p)
    assert findings and findings[0].category == "S6"


def test_docstring_markers_not_flagged(tmp_path):
    p = _write(tmp_path,
               "def docs():\n"
               "    \"\"\"A placeholder example; TODO is only prose here.\"\"\"\n"
               "    return 1\n")
    assert scan_file(p) == []


def test_ellipsis_in_docstring_not_flagged(tmp_path):
    p = _write(tmp_path,
               "M = \"\"\"ATG...ACG...TGA\"\"\"\n")
    assert scan_file(p) == []


def test_benign_token_not_matched_by_stub_regex(tmp_path):
    p = _write(tmp_path, "# STUBBENIGN seen, not a finding\nx = 1\n")
    assert scan_file(p) == []


def test_missing_file_returns_empty(tmp_path):
    assert scan_file(tmp_path / "missing.py") == []


def test_empty_file_returns_empty(tmp_path):
    p = _write(tmp_path, "")
    assert scan_file(p) == []


def test_syntax_error_returns_empty(tmp_path):
    p = _write(tmp_path, "def f(:\n    pass\n")
    assert scan_file(p) == []


def test_scan_tree_skips_parts(tmp_path):
    (tmp_path / "core").mkdir(parents=True)
    (tmp_path / "core" / "keep.py").write_text("x = 1\n")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "bad.py").write_text("def f():\n    ...\n")
    assert scan_tree(tmp_path, skip=("vendor",)) == []


def test_scan_tree_finds_without_skip(tmp_path):
    (tmp_path / "bad.py").write_text("def f():\n    ...\n")
    findings = scan_tree(tmp_path, skip=())
    assert findings and findings[0].path == str(tmp_path / "bad.py")


def test_format_report_empty():
    assert format_report([]) == "No stub/deferred-code patterns found."


def test_format_report_lists_findings():
    f = [Finding("a.py", 3, "S2", "function body is only `pass`")]
    out = format_report(f)
    assert "a.py:3: [S2]" in out


def test_main_with_file_root(tmp_path):
    p = _write(tmp_path, "def f():\n    ...\n")
    assert main([str(p)]) == 0
    assert main([str(p), "--fail"]) == 1
    assert main([str(p), "--skip", "x"]) == 0


def test_main_with_directory_root(tmp_path):
    (tmp_path / "bad.py").write_text("def f():\n    ...\n")
    assert main([str(tmp_path), "--fail"]) == 1
