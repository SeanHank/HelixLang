"""Unit tests for the silent-fallback linter (doc/36 §3ξ)."""
from __future__ import annotations

from helixlang.core.find_silent_fallbacks import (
    Finding,
    format_report,
    scan_file,
    scan_tree,
)


def _write(tmp_path, code, name="mod.py"):
    p = tmp_path / name
    p.write_text(code)
    return p


def test_bare_except_category_is_none(tmp_path):
    p = _write(tmp_path, "try:\n    x = 1\nexcept:\n    pass\n")
    findings = scan_file(p)
    assert len(findings) == 1
    assert findings[0].category is None
    assert "[?]" in str(findings[0])


def test_keyerror_except_is_f12(tmp_path):
    p = _write(tmp_path, "try:\n    x = d['k']\nexcept KeyError:\n    pass\n")
    findings = scan_file(p)
    assert findings[0].category == "F12"


def test_other_except_predicate_false(tmp_path):
    p = _write(tmp_path, "try:\n    x = 1\nexcept ValueError:\n    pass\n")
    findings = scan_file(p)
    assert len(findings) == 1
    assert findings[0].category is None


def test_tuple_except_unparse(tmp_path):
    p = _write(tmp_path,
               "try:\n    x = 1\nexcept (ValueError, TypeError):\n    pass\n")
    findings = scan_file(p)
    assert findings and findings[0].detail.startswith("empty except ValueError, TypeError")


def test_benign_marker_suppressed(tmp_path):
    p = _write(tmp_path, "import x\nexcept ImportError:  # SILENTBENIGN marker\n    pass\n")
    # Not a valid try/except without try; use a real one:
    p.write_text(
        "try:\n    import optional_dep\nexcept ImportError:  # SILENTBENIGN probe\n    pass\n"
    )
    assert scan_file(p) == []


def test_benign_marker_multiline_handler(tmp_path):
    # Multi-line except body -> the marker-scan loop iterates more than once.
    p = _write(tmp_path,
               "try:\n    x = 1\nexcept ImportError:\n"
               "    # SILENTBENIGN probe\n    pass\n")
    assert scan_file(p) == []


def test_import_error_except_is_f1(tmp_path):
    p = _write(tmp_path, "try:\n    import rdkit\nexcept ImportError:\n    pass\n")
    findings = scan_file(p)
    assert findings[0].category == "F1"


def test_fallback_string_assignment_is_f4(tmp_path):
    p = _write(tmp_path, "backend = 'scipy_fallback'\n")
    findings = scan_file(p)
    assert findings and findings[0].category == "F4"


def test_missing_file_returns_empty(tmp_path):
    assert scan_file(tmp_path / "does_not_exist.py") == []


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
    (tmp_path / "vendor" / "skip.py").write_text(
        "try:\n    import z\nexcept ImportError:\n    pass\n")
    findings = scan_tree(tmp_path, skip=("vendor",))
    assert findings == []


def test_scan_tree_finds_without_skip(tmp_path):
    (tmp_path / "bad.py").write_text(
        "try:\n    import z\nexcept ImportError:\n    pass\n")
    findings = scan_tree(tmp_path, skip=())
    assert findings and findings[0].path == str(tmp_path / "bad.py")


def test_format_report_empty():
    assert format_report([]) == "No silent-fallback patterns found."


def test_format_report_lists_findings():
    f = [Finding("a.py", 3, "F1", "empty except ImportError")]
    out = format_report(f)
    assert "a.py:3: [F1]" in out


def test_finding_str_no_category():
    f = Finding("b.py", 1, None, "detail")
    assert "[?]" in str(f)


def test_main_with_file_root(tmp_path):
    from helixlang.core.find_silent_fallbacks import main
    p = _write(tmp_path, "try:\n    import z\nexcept ImportError:\n    pass\n")
    assert main([str(p)]) == 0
    assert main([str(p), "--fail"]) == 1
    assert main([str(p), "--skip", "x"]) == 0


def test_main_with_directory_root(tmp_path):
    from helixlang.core.find_silent_fallbacks import main
    (tmp_path / "bad.py").write_text(
        "try:\n    import z\nexcept ImportError:\n    pass\n")
    assert main([str(tmp_path)]) == 0
    assert main([str(tmp_path), "--fail"]) == 1


def test_benign_marker_line_beyond_file_length(tmp_path):
    import ast

    from helixlang.core.find_silent_fallbacks import _Visitor
    # 2-line file, but mock a handler with end_lineno=5 so
    # lineno > len(self._lines) triggers the False branch
    code = "x = 1\ny = 2\n"
    lines = code.splitlines()
    v = _Visitor(lines)
    handler = ast.ExceptHandler()
    handler.lineno = 1
    handler.end_lineno = 5
    handler.type = ast.Name(id="ImportError", ctx=ast.Load())
    handler.body = [ast.Pass()]
    assert v._is_benign(handler) is False
