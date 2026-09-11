"""Tests for the DIAMOND/BLAST+ wrapper (doc/20 §5.1, plugins/annotation/blast.py).

Covers result parsing, filtering, error paths (missing files / missing binary /
non-zero rc) and database building — with ``subprocess.run`` monkeypatched so no
external diamond binary is required.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from helixlang.plugins.annotation.blast import (
    Hit,
    SearchResult,
    build_database,
    run_diamond,
)


def _completed(stdout: str):
    return SimpleNamespace(stdout=stdout, stderr="", returncode=0)


def _mock_run(monkeypatch, completed):
    captured = {}
    def fake_run(cmd, **_kw):
        captured["cmd"] = cmd
        return completed
    monkeypatch.setattr("helixlang.plugins.annotation.blast.subprocess.run", fake_run)
    return captured


# ── Hit / SearchResult ─────────────────────────────────────────────────────────

def test_search_result_hits_for_filters_by_query():
    sr = SearchResult(hits=[
        Hit("q1", "s1", 95.0, 100, 1e-10, 200),
        Hit("q1", "s2", 80.0, 100, 1e-8, 150),
        Hit("q2", "s3", 99.0, 120, 1e-20, 250),
    ])
    assert [h.subject_id for h in sr.hits_for("q1")] == ["s1", "s2"]
    assert sr.hits_for("nope") == []


# ── run_diamond ────────────────────────────────────────────────────────────────

def test_run_diamond_parses_outfmt(monkeypatch, tmp_path):
    q = tmp_path / "q.fa"
    q.write_text(">a\nM\n")
    db = tmp_path / "db"
    db.write_text("x")
    stdout = (
        "q1\ts1\t95.0\t100\t1e-10\t200.5\tSome protein\n"
        "q1\ts2\t80.0\t90\t1e-8\t150\t\n"
        "\n"  # blank line ignored
        "q1\tbad_line\n"  # <6 fields ignored
    )
    cap = _mock_run(monkeypatch, _completed(stdout))
    sr = run_diamond(q, db)
    assert sr.query_count == 1  # only query q1 (blank/bad filtered in set)
    assert len(sr.hits) == 2
    h = sr.hits[0]
    assert h.query_id == "q1" and h.subject_id == "s1"
    assert h.identity == 95.0 and h.alignment_length == 100
    assert h.e_value == 1e-10 and h.bit_score == 200.5
    assert h.stitle == "Some protein"
    assert sr.hits[1].stitle == ""
    assert "blastp" in cap["cmd"]
    assert "--outfmt" in cap["cmd"]


def test_run_diamond_raises_when_query_missing(monkeypatch, tmp_path):
    db = tmp_path / "db"
    db.write_text("x")
    with pytest.raises(FileNotFoundError):
        run_diamond(tmp_path / "nope.fa", db)


def test_run_diamond_raises_when_db_missing(monkeypatch, tmp_path):
    q = tmp_path / "q.fa"
    q.write_text(">a\nM\n")
    with pytest.raises(FileNotFoundError):
        run_diamond(q, tmp_path / "nodefault")


def test_run_diamond_missing_binary(monkeypatch, tmp_path):
    q = tmp_path / "q.fa"
    q.write_text(">a\nM\n")
    db = tmp_path / "db"
    db.write_text("x")
    def fake_run(*_a, **_k):
        raise FileNotFoundError("diamond not on PATH")
    monkeypatch.setattr("helixlang.plugins.annotation.blast.subprocess.run", fake_run)
    with pytest.raises(FileNotFoundError):
        run_diamond(q, db)


def test_run_diamond_nonzero_exit(monkeypatch, tmp_path):
    q = tmp_path / "q.fa"
    q.write_text(">a\nM\n")
    db = tmp_path / "db"
    db.write_text("x")
    def fake_run(*_a, **_k):
        raise subprocess.CalledProcessError(1, "diamond", stderr="boom")
    monkeypatch.setattr("helixlang.plugins.annotation.blast.subprocess.run", fake_run)
    with pytest.raises(RuntimeError):
        run_diamond(q, db)


# ── build_database ─────────────────────────────────────────────────────────────

def test_build_database_success(monkeypatch, tmp_path):
    inp = tmp_path / "in.fa"
    inp.write_text(">a\nM\n")
    db = tmp_path / "mydb"
    cap = _mock_run(monkeypatch, _completed(""))
    out = build_database(inp, db)
    assert str(out).endswith(".dmnd")
    assert "makedb" in cap["cmd"]


def test_build_database_missing_binary(monkeypatch, tmp_path):
    inp = tmp_path / "in.fa"
    inp.write_text(">a\nM\n")
    def fake_run(*_a, **_k):
        raise FileNotFoundError("no diamond")
    monkeypatch.setattr("helixlang.plugins.annotation.blast.subprocess.run", fake_run)
    with pytest.raises(FileNotFoundError):
        build_database(inp, tmp_path / "db")


def test_build_database_nonzero_exit(monkeypatch, tmp_path):
    inp = tmp_path / "in.fa"
    inp.write_text(">a\nM\n")
    def fake_run(*_a, **_k):
        raise subprocess.CalledProcessError(2, "diamond", stderr="fail")
    monkeypatch.setattr("helixlang.plugins.annotation.blast.subprocess.run", fake_run)
    with pytest.raises(RuntimeError):
        build_database(inp, tmp_path / "db")
