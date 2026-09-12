"""Tests for the docs-truth gate (src/helixlang/core/docs_truth.py).

Fully synthetic filesystem fixtures patch the module's ROOT/SRC/validation
globals so every resolution, bounds, and CLI branch runs deterministically.
"""
from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from helixlang.core import docs_truth as dt
from helixlang.core.docs_truth import (
    Anchor,
    Finding,
    _match_text,
)


@pytest.fixture()
def tree(tmp_path: Path) -> tuple[object, object, object, object]:
    """Build a tiny repo with distinct files and return (ROOT, SRC, VAL, BENCH)."""
    root = tmp_path / "repo"
    src = root / "src" / "helixlang"
    core = src / "core"
    api = src / "api"
    core.mkdir(parents=True)
    api.mkdir(parents=True)
    (core / "__init__.py").write_text("x = 1\n")
    (api / "__init__.py").write_text("y = 2\n")
    (core / "units.py").write_text("\n".join(f"u{i}" for i in range(80)))
    (api / "units.py").write_text("\n".join(f"a{i}" for i in range(40)))
    val = root / "validation"
    val_bench = val / "benchmarks"
    bench = val_bench / "03_ecoli_fba"
    bench.mkdir(parents=True)
    (val / "schema.py").write_text("\n".join(f"s{i}" for i in range(300)))
    (bench / "run.py").write_text("\n".join(f"r{i}" for i in range(200)))
    pytest_mod = root / "pytest.rb"  # not .py on purpose
    pytest_mod.write_text("irrelevant")
    return root, src, val, val_bench


@pytest.fixture()
def t(tree: tuple[object, object, object, object], monkeypatch: pytest.MonkeyPatch) -> None:
    root, src, val, val_bench = tree
    monkeypatch.setattr(dt, "ROOT", root)
    monkeypatch.setattr(dt, "SRC", src)
    monkeypatch.setattr(dt, "VALIDATION", val)
    monkeypatch.setattr(dt, "VALIDATION_BENCHMARKS", val_bench)
    monkeypatch.setattr(dt, "_SEARCH_BASES", (src, root, val_bench, val))
    monkeypatch.setattr(dt, "_BASENAME_ROOTS", (src, val_bench, val))


class TestMatchText:
    def test_start_and_range_anchors(self) -> None:
        text = "See parser.py:60 and parser.py:60-99 ok\n"
        anchors = _match_text(text, "doc/x.md")
        assert [a.display() for a in anchors] == ["parser.py:60", "parser.py:60-99"]
        assert anchors[0].lineno == 1
        assert anchors[0].source == "doc/x.md"

    def test_path_with_directory(self) -> None:
        anchors = _match_text("a plugins/human/__init__.py:34 b\n", "doc/y.md")
        assert len(anchors) == 1
        assert anchors[0].file == "plugins/human/__init__.py"

    def test_no_anchors(self) -> None:
        assert _match_text("no anchors here\n", "doc/z.md") == []

    def test_two_on_one_line(self) -> None:
        anchors = _match_text("p1.py:1 p2.py:2-3\n", "doc/x.md")
        assert len(anchors) == 2
        assert anchors[0].raw == anchors[1].raw


class FakePath:
    """Stand-in for Path used by the anchor fixture below."""

    def __init__(self, s: str) -> None:
        self.s = s

    def __str__(self) -> str:
        return self.s

    def __fspath__(self) -> str:
        return self.s


class TestResolveFile:
    def test_slash_path_found_in_src(self, t: None) -> None:
        resolved = dt.resolve_file("core/units.py")
        assert resolved is not None and resolved.endswith("src/helixlang/core/units.py")

    def test_slash_path_found_in_validation(self, t: None) -> None:
        resolved = dt.resolve_file("schema.py")
        assert resolved is not None and resolved.endswith("validation/schema.py")

    def test_slash_path_found_in_benchmarks(self, t: None) -> None:
        resolved = dt.resolve_file("03_ecoli_fba/run.py")
        assert resolved is not None and resolved.endswith("03_ecoli_fba/run.py")

    def test_slash_path_missing(self, t: None) -> None:
        assert dt.resolve_file("nope/thing.py") is None

    def test_basename_unique(self, t: None) -> None:
        resolved = dt.resolve_file("schema.py")
        assert resolved is not None
        assert resolved == str((dt.VALIDATION / "schema.py").resolve())

    def test_basename_ambiguous(self, t: None) -> None:
        assert dt.resolve_file("units.py") == "AMBIGUOUS"

    def test_basename_missing(self, t: None) -> None:
        assert dt.resolve_file("dne.py") is None

    def test_basename_from_root_shallow(self, t: None, tmp_path: object,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
        (dt.ROOT / "tools.py").write_text("t = 1\n")
        monkeypatch.setattr(dt, "_BASENAME_ROOTS", (dt.SRC,))
        assert dt.resolve_file("tools.py") is not None

    def test_src_not_dir_skipped(self, t: None, monkeypatch: pytest.MonkeyPatch) -> None:
        (dt.ROOT / "file.txt").write_text("x")
        monkeypatch.setattr(dt, "_BASENAME_ROOTS", (dt.ROOT / "file.txt",))
        assert dt.resolve_file("anything.py") is None


class TestValidateAnchor:
    def _anch(self, file: str, start: int, end: int | None = None) -> Anchor:
        return Anchor(source="doc/x.md", lineno=1, raw=file, file=file, start=start, end=end)

    def test_valid_single(self, t: None) -> None:
        assert dt.validate_anchor(self._anch("schema.py", 5)) is None

    def test_valid_range(self, t: None) -> None:
        assert dt.validate_anchor(self._anch("schema.py", 5, 12)) is None

    def test_missing_file(self, t: None) -> None:
        detail = dt.validate_anchor(self._anch("gone.py", 3))
        assert detail is not None and "no such file" in detail

    def test_unambiguous_missing_file_detail(self, t: None) -> None:
        detail = dt.validate_anchor(self._anch("gone.py", 3))
        assert detail and "gone.py:3" in detail

    def test_ambiguous(self, t: None) -> None:
        detail = dt.validate_anchor(self._anch("units.py", 3))
        assert detail is not None and "ambiguous" in detail

    def test_start_below_one(self, t: None) -> None:
        detail = dt.validate_anchor(self._anch("schema.py", 0))
        assert detail and "below 1" in detail

    def test_start_beyond_file(self, t: None) -> None:
        detail = dt.validate_anchor(self._anch("schema.py", 99999))
        assert detail and "beyond file length" in detail

    def test_end_before_start(self, t: None) -> None:
        detail = dt.validate_anchor(self._anch("schema.py", 10, 5))
        assert detail and "end before start" in detail

    def test_end_beyond_file(self, t: None) -> None:
        detail = dt.validate_anchor(self._anch("schema.py", 5, 99999))
        assert detail and "beyond file length" in detail

    def test_unreadable(self, t: None) -> None:
        (dt.SRC / "locked.py").write_text("x = 1\n")

        def boom(*args: object, **kwargs: object) -> object:
            raise IsADirectoryError(args[0] if args else "?")

        real_open = builtins.open
        builtins.open = boom  # type: ignore[assignment]
        try:
            detail = dt.validate_anchor(self._anch("locked.py", 1))
        finally:
            builtins.open = real_open
        assert detail is not None and "unreadable" in detail


class TestScanning:
    def test_scan_file_ok(self, t: None, tmp_path: object) -> None:
        doc = dt.ROOT / "doc" / "g.md"
        doc.parent.mkdir(parents=True)
        doc.write_text("see schema.py:5-12 here\n")
        assert dt.scan_file(doc) == []

    def test_scan_file_finding(self, t: None, tmp_path: object) -> None:
        doc = dt.ROOT / "doc" / "b.md"
        doc.parent.mkdir(parents=True)
        doc.write_text("see units.py:5 here\n")
        findings = dt.scan_file(doc)
        assert len(findings) == 1
        assert "ambiguous" in findings[0].detail
        assert str(findings[0]).startswith(str(doc))

    def test_scan_file_unreadable(self, t: None) -> None:
        missing = dt.ROOT / "missing.md"
        assert dt.scan_file(missing) == []

    def test_scan_tree_skips(self, t: None, tmp_path: object) -> None:
        doc = dt.ROOT / "doc"
        doc.mkdir()
        (doc / "a.md").write_text("ok schema.py:5\n")
        skip = dt.ROOT / "archive"
        skip.mkdir()
        (skip / "z.md").write_text("bad units.py:5\n")
        findings = dt.scan_tree(dt.ROOT, skip=("archive",))
        assert all("z.md" not in f.path for f in findings)


class TestReport:
    def test_empty_report(self) -> None:
        assert dt.format_report([]) == "All doc code anchors resolve to the current tree."

    def test_findings_report(self) -> None:
        out = dt.format_report([Finding("doc/a.md", 3, "broken")])
        assert "doc/a.md:3: broken" in out

    def test_finding_string_always_has_location(self) -> None:
        assert str(Finding("doc/m.md", 7, "x")) == "doc/m.md:7: x"


class TestMain:
    def test_ok_no_fail(self, t: None, tmp_path: object) -> None:
        good = dt.ROOT / "good.md"
        good.write_text("see schema.py:5\n")
        assert dt.main([str(good)]) == 0

    def test_fail_with_findings(self, t: None, tmp_path: object) -> None:
        bad = dt.ROOT / "bad.md"
        bad.write_text("see units.py:5\n")
        assert dt.main([str(bad), "--fail"]) == 1

    def test_no_fail_returns_zero_success(self, t: None, tmp_path: object) -> None:
        bad = dt.ROOT / "bad.md"
        bad.write_text("see units.py:5\n")
        assert dt.main([str(bad)]) == 0

    def test_dir_root(self, t: None, tmp_path: object) -> None:
        doc = dt.ROOT / "doc"
        doc.mkdir()
        (doc / "a.md").write_text("see schema.py:5\n")
        assert dt.main([str(doc), "--fail"]) == 0

    def test_default_roots_doc(self, t: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """No-arg invocation scans the (patched) repo's doc/ directory."""
        doc = dt.ROOT / "doc"
        doc.mkdir(parents=True)
        (doc / "a.md").write_text("see schema.py:5\n")
        (doc / "b.md").write_text("see units.py:5\n")
        monkeypatch.setattr("sys.argv", ["docs_truth"])
        assert dt.main() == 0

    def test_skip_respected(self, t: None) -> None:
        root = dt.ROOT
        good = root / "g.md"
        bad = root / "h.md"
        good.write_text("schema.py:5\n")
        bad.write_text("units.py:5\n")
        assert dt.main([str(root), "--fail", "--skip", "h.md"]) == 0
        assert dt.main([str(root), "--fail"]) == 1
