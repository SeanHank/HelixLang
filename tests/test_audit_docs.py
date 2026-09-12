"""Unit tests for the doc-implementation-claim audit (audit_docs)."""
from __future__ import annotations

from helixlang.core.audit_docs import (
    Finding,
    format_report,
    main,
    scan_file,
    scan_tree,
)


def _write(tmp_path, text, name="d.md"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_clean_doc_is_empty(tmp_path):
    p = _write(tmp_path, "# Title\n\nImplemented and wired in, see §2.\n")
    assert scan_file(p) == []


def test_d1_planned(tmp_path):
    p = _write(tmp_path, "Phase 2 is planned for next quarter.\n")
    f = scan_file(p)
    assert len(f) == 1 and f[0].category == "D1"


def test_d1_will_be(tmp_path):
    p = _write(tmp_path, "The solver will be implemented in a later release.\n")
    f = scan_file(p)
    assert f and f[0].category == "D1"


def test_d2_deferred(tmp_path):
    p = _write(tmp_path, "The feature was deferred to a later release.\n")
    f = scan_file(p)
    assert f and f[0].category == "D2"


def test_d3_blocked_status(tmp_path):
    p = _write(tmp_path, "The migration is still blocked by the split.\n")
    f = scan_file(p)
    assert f and f[0].category == "D3"


def test_d3_blocked_flux_not_claim(tmp_path):
    p = _write(tmp_path, "transport_restriction 0.0 = fully blocked flux.\n")
    assert scan_file(p) == []


def test_d4_out_of_scope(tmp_path):
    p = _write(tmp_path, "BCF parsing is out of scope (see §21).\n")
    f = scan_file(p)
    assert f and f[0].category == "D4"


def test_d5_partial_implementation(tmp_path):
    p = _write(tmp_path, "The ODE path is partially vectorized.\n")
    f = scan_file(p)
    assert f and f[0].category == "D5"


def test_d5_status_column(tmp_path):
    p = _write(tmp_path, "| `random_seed` | ⚠️ partial | capture all seeds |\n")
    f = scan_file(p)
    assert f and f[0].category == "D5"


def test_d5_chinese(tmp_path):
    p = _write(tmp_path, "该模块当前为部分实现。\n")
    f = scan_file(p)
    assert f and f[0].category == "D5"


def test_d6_unchecked_checkbox(tmp_path):
    p = _write(tmp_path, "- [ ] Implement `genotype.py`\n")
    f = scan_file(p)
    assert f and f[0].category == "D6"


def test_d6_todo(tmp_path):
    p = _write(tmp_path, "TODO: wire the ramp math.\n")
    f = scan_file(p)
    assert f and f[0].category == "D6"


def test_d7_roadmap(tmp_path):
    p = _write(tmp_path, "## 14. Implementation roadmap\n")
    f = scan_file(p)
    assert f and f[0].category == "D7"


def test_d7_chinese(tmp_path):
    p = _write(tmp_path, "该能力计划内，暂缓接入。\n")
    f = scan_file(p)
    assert f and f[0].category == "D7"


def test_d8_status(tmp_path):
    p = _write(tmp_path, "> **Status:** PROPOSED (research synthesis)\n")
    f = scan_file(p)
    assert f and f[0].category == "D8"


def test_first_rule_wins(tmp_path):
    p = _write(tmp_path, "Deferred; partially shipped.\n")
    f = scan_file(p)
    assert f and f[0].category == "D2"


def test_fenced_code_ignored(tmp_path):
    p = _write(tmp_path, "before\n```\n- [ ] TODO keep this\n```\nafter\n")
    assert scan_file(p) == []


def test_inline_code_ignored(tmp_path):
    p = _write(tmp_path, "Use the `--planned` and `- [ ]` flags.\n")
    assert scan_file(p) == []


def test_html_comment_ignored(tmp_path):
    p = _write(tmp_path, "<!-- TODO: drop later -->\n")
    assert scan_file(p) == []


def test_html_comment_multiline(tmp_path):
    p = _write(tmp_path, "<!--\nTODO plan\n-->\n")
    assert scan_file(p) == []


def test_benign_token_skips_line(tmp_path):
    p = _write(tmp_path, "Quoted road map.  # DOCBENIGN verbatim external quote\n")
    assert scan_file(p) == []


def test_scan_file_missing_is_empty(tmp_path):
    assert scan_file(tmp_path / "nope.md") == []


def test_scan_file_blank_is_empty(tmp_path):
    p = _write(tmp_path, "   \n\n")
    assert scan_file(p) == []


def test_scan_tree_finds_and_skips(tmp_path):
    (tmp_path / "a.md").write_text("planned\n")
    (tmp_path / "b.md").write_text("ok\n")
    sub = tmp_path / "tests"
    sub.mkdir()
    (sub / "c.md").write_text("deferred\n")
    found = scan_tree(tmp_path)
    cats = {f.path.rsplit("/", 1)[-1]: f.category for f in found}
    assert cats["a.md"] == "D1"
    found_skip = scan_tree(tmp_path, skip=("tests",))
    assert "b.md" not in {f.path for f in found}  # b.md is clean anyway
    assert all("tests" not in f.path for f in found_skip)


def test_format_report_empty():
    assert format_report([]) == "No non-implementation claims found."


def test_format_report_lines():
    out = format_report([Finding("d.md", 3, "D1", "# x (planned)")])
    assert "d.md:3: [D1] # x (planned)" in out


def test_finding_str_no_category():
    out = str(Finding("d.md", 1, None, "detail"))
    assert out == "d.md:1: [?] detail"


def test_main_no_fail_clean(tmp_path):
    _write(tmp_path, "implemented\n", "ok.md")
    assert main([str(tmp_path)]) == 0


def test_main_fail_with_finding(tmp_path):
    _write(tmp_path, "planned\n")
    assert main([str(tmp_path), "--fail"]) == 1


def test_main_fail_clean(tmp_path):
    _write(tmp_path, "implemented\n")
    assert main([str(tmp_path), "--fail"]) == 0


def test_main_file_argument(tmp_path):
    p = _write(tmp_path, "implemented\n")
    assert main([str(p), "--fail"]) == 0
