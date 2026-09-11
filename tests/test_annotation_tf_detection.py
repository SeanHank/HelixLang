"""Tests for TF detection (doc/20 §7.2, plugins/annotation/tf_detection.py).

Covers the TFScanResult helpers, the heuristic (no-Pfam) scan, and the HMMER
domtbl scan. ``subprocess.run`` is monkeypatched and the temp domtbl output is
written by the fake runner, so no HMMER binary is required.
"""
from __future__ import annotations

import pytest

from helixlang.plugins.annotation.tf_detection import (
    TFCandidate,
    TFScanResult,
    _count_fasta_sequences,
    detect_transcription_factors,
)


def _cand(gene, family):
    return TFCandidate(gene_id=gene, tf_family=family, domain_accession="PF00072",
                       score=15.0, confidence=0.9)


# ── TFScanResult ───────────────────────────────────────────────────────────────

def test_tf_count_and_ids():
    r = TFScanResult(total_genes=10, tf_candidates=[_cand("g1", "HTH_LacI"),
                                                    _cand("g2", "HTH_TetR")])
    assert r.tf_count == 2
    assert r.tf_ids() == ["g1", "g2"]


def test_tf_fraction():
    r = TFScanResult(total_genes=4, tf_candidates=[_cand("g1", "X"), _cand("g2", "Y")])
    assert r.tf_fraction == 0.5
    assert TFScanResult(total_genes=0).tf_fraction == 0.0


def test_tfs_by_family_groups():
    r = TFScanResult(tf_candidates=[
        _cand("g1", "HTH_LacI"), _cand("g2", "HTH_TetR"), _cand("g3", "HTH_LacI"),
    ])
    assert r.tfs_by_family() == {"HTH_LacI": ["g1", "g3"], "HTH_TetR": ["g2"]}


# ── heuristic (no pfam database) ───────────────────────────────────────────────

def test_heuristic_scan_detects_family_in_header(tmp_path):
    fa = tmp_path / "proteins.fa"
    fa.write_text(
        ">gene_1 HTH_LacI repressor\nMAE\n"
        ">gene_2 unknown thing\nMAE\n"
        ">gene_3 response_reg two-component\nMAE\n"
    )
    r = detect_transcription_factors(str(fa))
    assert r.total_genes == 3
    ids = {t.gene_id for t in r.tf_candidates}
    assert "gene_1" in ids   # hth_laci family matched
    assert "gene_3" in ids   # response_reg family matched
    assert "gene_2" not in ids
    families = {t.gene_id: t.tf_family for t in r.tf_candidates}
    accs = {t.gene_id: t.domain_accession for t in r.tf_candidates}
    assert families["gene_1"] == "HTH_LacI" and accs["gene_1"] == "PF00356"
    assert families["gene_3"] == "Response_reg" and accs["gene_3"] == "PF00072"


def test_heuristic_scan_missing_file_is_silent(tmp_path):
    r = detect_transcription_factors(str(tmp_path / "nope.fa"))
    assert r.total_genes == 0
    assert r.tf_candidates == []


def test_count_fasta_sequences_missing_file_zero(tmp_path):
    assert _count_fasta_sequences(str(tmp_path / "missing.fa")) == 0


# ── HMMER scan (via detect_transcription_factors) ─────────────────────────────

def _fake_hmmer(monkeypatch, domtbl_lines):
    # _hmmer_tf_scan does `import subprocess` inside the function, so we must
    # patch the real subprocess.run (restored automatically by monkeypatch).
    def fake_run(cmd, **_kw):
        out_idx = cmd.index("--domtblout") + 1
        out = cmd[out_idx]
        with open(out, "w") as fh:
            fh.write(domtbl_lines)
    monkeypatch.setattr("subprocess.run", fake_run)


def _domtbl(overrides):
    # build a 22-column domtbl line; overrides maps {column_index: value}.
    # column 0=gene_id, 4=accession.version, 12=evalue, 13=bit score.
    tokens = ["0"] * 22
    for idx, val in overrides.items():
        tokens[idx] = val
    return " ".join(tokens)


def test_hmmer_scan_parses_hits(tmp_path, monkeypatch):
    fa = tmp_path / "proteins.fa"
    fa.write_text(">g1\nMK\n>g2\nMK\n")
    lines = (
        _domtbl({0: "g1", 4: "PF00072.24", 12: "1e-05", 13: "15.0"}) + "\n"
        + "# comment line ignored\n"  # starts with '#' -> skipped
        + "short only field\n"  # non-comment, <22 fields -> skipped
    )
    _fake_hmmer(monkeypatch, lines)
    r = detect_transcription_factors(str(fa), pfam_database=str(tmp_path / "Pfam-A.hmm"))
    assert r.total_genes == 2
    assert len(r.tf_candidates) == 1
    t = r.tf_candidates[0]
    assert t.gene_id == "g1" and t.tf_family == "Response_reg"
    assert t.domain_accession == "PF00072"
    assert t.e_value == 1e-05 and t.score == 15.0
    assert t.confidence == 1.0  # 0.5 + 15/20 = 1.25 -> clamped to 1.0


def test_hmmer_scan_drops_negative_score_and_unknown_accession(tmp_path, monkeypatch):
    fa = tmp_path / "proteins.fa"
    fa.write_text(">g1\nMK\n>g2\nMK\n>g3\nMK\n")
    lines = (
        _domtbl({0: "g1", 4: "PF99999.1", 12: "1e-05", 13: "5.0"}) + "\n"  # unknown
        + _domtbl({0: "g2", 4: "PF00072.24", 12: "1e-05", 13: "-3.0"}) + "\n"  # neg score
        + _domtbl({0: "g3", 4: "PF00072.24", 12: "1e-05", 13: "4.0"}) + "\n"
    )
    _fake_hmmer(monkeypatch, lines)
    r = detect_transcription_factors(str(fa), pfam_database=str(tmp_path / "Pfam-A.hmm"))
    assert [t.gene_id for t in r.tf_candidates] == ["g3"]
    assert r.tf_candidates[0].confidence == 0.7  # 0.5 + 4/20 = 0.7


def test_hmmer_scan_keeps_best_hit_per_gene(tmp_path, monkeypatch):
    fa = tmp_path / "proteins.fa"
    fa.write_text(">g1\nMK\n")
    lines = (
        _domtbl({0: "g1", 4: "PF00072.24", 12: "1e-05", 13: "8.0"}) + "\n"
        + _domtbl({0: "g1", 4: "PF00356.2", 12: "1e-05", 13: "12.0"}) + "\n"
        + _domtbl({0: "g1", 4: "PF00072.24", 12: "1e-05", 13: "3.0"}) + "\n"
    )
    _fake_hmmer(monkeypatch, lines)
    r = detect_transcription_factors(str(fa), pfam_database=str(tmp_path / "Pfam-A.hmm"))
    assert len(r.tf_candidates) == 1
    assert r.tf_candidates[0].tf_family == "HTH_LacI"  # PF00356
    assert r.tf_candidates[0].score == 12.0


def test_hmmer_scan_missing_binary(tmp_path, monkeypatch):
    fa = tmp_path / "proteins.fa"
    fa.write_text(">g1\nMK\n")
    def fake_run(*_a, **_k):
        raise FileNotFoundError("no hmmsearch")
    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(FileNotFoundError):
        detect_transcription_factors(str(fa), pfam_database="Pfam-A.hmm")


def test_hmmer_scan_nonzero_exit(tmp_path, monkeypatch):
    fa = tmp_path / "proteins.fa"
    fa.write_text(">g1\nMK\n")
    import subprocess
    def fake_run(*_a, **_k):
        raise subprocess.CalledProcessError(1, "hmmsearch", stderr="fail")
    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(RuntimeError):
        detect_transcription_factors(str(fa), pfam_database="Pfam-A.hmm")
