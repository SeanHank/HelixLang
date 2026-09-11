"""Tests for transporter classification (doc/20 §5.4, plugins/annotation/transporter.py).

Covers the result helpers, the heuristic (no-Pfam) scan, substrate guessing, and
the HMMER domtbl scan. ``subprocess.run`` is monkeypatched so no HMMER binary is
required.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from helixlang.plugins.annotation.transporter import (
    TransporterInfo,
    TransporterScanResult,
    _guess_substrate,
    classify_transporters,
)


def _info(fam, substrate="", direction="import"):
    return TransporterInfo(gene_id="g", transporter_family=fam,
                           transport_type="active", direction=direction,
                           predicted_substrate=substrate)


# ── TransporterScanResult ──────────────────────────────────────────────────────

def test_transporter_count():
    r = TransporterScanResult(transporters=[_info("ABC_efflux"), _info("MFS")])
    assert r.transporter_count == 2


def test_by_substrate_groups_and_unknown_fallback():
    r = TransporterScanResult(transporters=[
        _info("ABC_efflux", "glucose"),
        _info("MFS", "glucose"),
        _info("MATE", ""),  # no predicted substrate -> "unknown"
    ])
    res = r.by_substrate()
    assert res["glucose"] == [r.transporters[0], r.transporters[1]]
    assert len(res["unknown"]) == 1


# ── heuristic scan ─────────────────────────────────────────────────────────────

def test_heuristic_scan_matches_family_and_guesses_substrate(tmp_path):
    fa = tmp_path / "proteins.fa"
    fa.write_text(
        ">gene1 ABC_efflux glucose-specific\nMAE\n"
        ">gene2 MFS translocator\nMAE\n"
        ">gene3 nothing known\nMAE\n"
    )
    r = classify_transporters(str(fa))
    assert r.total_genes == 3
    by_gene = {t.gene_id: t for t in r.transporters}
    assert "gene1" in by_gene and "gene2" in by_gene and "gene3" not in by_gene
    assert by_gene["gene1"].transporter_family == "ABC_efflux"
    assert by_gene["gene1"].direction == "export"
    assert by_gene["gene1"].predicted_substrate == "glucose"
    assert by_gene["gene2"].transporter_family == "MFS"


def test_heuristic_scan_missing_file_is_silent(tmp_path):
    r = classify_transporters(str(tmp_path / "nope.fa"))
    assert r.total_genes == 0
    assert r.transporters == []


# ── substrate guessing ─────────────────────────────────────────────────────────

def test_guess_substrate_match_and_miss():
    assert _guess_substrate("glucose ptsG permease") == "glucose"
    assert _guess_substrate("acetate acs") == "acetate"
    assert _guess_substrate("totally unrelated protein") == ""


# ── HMMER scan ─────────────────────────────────────────────────────────────────

def _fake_hmmer(monkeypatch, stdout):
    def fake_run(cmd, **_kw):
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)
    monkeypatch.setattr("subprocess.run", fake_run)


def _domtbl(overrides):
    # column 0=gene_id, 3=pfam family accession
    tokens = ["0"] * 22
    for idx, val in overrides.items():
        tokens[idx] = val
    return " ".join(tokens)


def test_hmmer_scan_parses_and_dedupes(tmp_path, monkeypatch):
    fa = tmp_path / "proteins.fa"
    fa.write_text(">g1\nMK\n")
    stdout = (
        _domtbl({0: "g1", 3: "PF00005"}) + "\n"   # ABC_ATPase
        + _domtbl({0: "g1", 3: "PF00355"}) + "\n"  # same gene, MFS
        + "# comment\n"
        + "too short\n"
        + _domtbl({0: "g2", 3: "PF99999"}) + "\n"  # unknown family -> skipped
    )
    _fake_hmmer(monkeypatch, stdout)
    r = classify_transporters(str(fa), pfam_database="Pfam-A.hmm")
    # gene g1 counted once, two transporter entries; g2 skipped
    assert r.total_genes == 1
    assert len(r.transporters) == 2
    fams = {t.transporter_family for t in r.transporters}
    assert fams == {"ABC_ATPase", "MFS"}
    assert r.transporters[0].confidence == 0.7


def test_hmmer_scan_missing_binary(tmp_path, monkeypatch):
    fa = tmp_path / "p.fa"
    fa.write_text(">g1\nMK\n")
    def fake_run(*_a, **_k):
        raise FileNotFoundError("no hmmsearch")
    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(FileNotFoundError):
        classify_transporters(str(fa), pfam_database="Pfam-A.hmm")


def test_hmmer_scan_nonzero_exit(tmp_path, monkeypatch):
    import subprocess
    fa = tmp_path / "p.fa"
    fa.write_text(">g1\nMK\n")
    def fake_run(*_a, **_k):
        raise subprocess.CalledProcessError(1, "hmmsearch", stderr="boom")
    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(RuntimeError):
        classify_transporters(str(fa), pfam_database="Pfam-A.hmm")
