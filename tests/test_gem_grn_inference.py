"""Tests for helixlang.plugins.gem.grn_inference.

Covers the RegulatoryEdge/GRNInferenceResult dataclasses, the public
infer_grn() pipeline (database matching + genome validation + motif path),
and the private motif/upstream/PWM helpers.
"""
from __future__ import annotations

import pytest

from helixlang.plugins.annotation.tf_detection import TFCandidate, TFScanResult
from helixlang.plugins.gem.grn_inference import (
    EvidenceLevel,
    GRNInferenceResult,
    RegulatoryEdge,
    _extract_upstream_sequences,
    _predict_motifs,
    _score_pwm,
    infer_grn,
)


def _cand(gene: str, family: str) -> TFCandidate:
    return TFCandidate(
        gene_id=gene,
        tf_family=family,
        domain_accession="PF00072",
        domain_start=1,
        domain_end=60,
    )


def _tf_result(*genes: str) -> TFScanResult:
    return TFScanResult(
        total_genes=len(genes),
        tf_candidates=[_cand(g, g) for g in genes],
    )


# ============================================================================
# RegulatoryEdge
# ============================================================================

def test_edge_rejects_invalid_regulation_type():
    with pytest.raises(ValueError):
        RegulatoryEdge(
            tf_id="crp", target_gene="lacZ", regulation_type="bogus",
            evidence_level=EvidenceLevel.DATABASE,
        )


def test_edge_high_confidence_requires_db_and_conf():
    high = RegulatoryEdge(
        tf_id="crp", target_gene="lacZ", regulation_type="activation",
        evidence_level=EvidenceLevel.DATABASE, confidence=0.9,
    )
    assert high.is_high_confidence
    low_conf = RegulatoryEdge(
        tf_id="crp", target_gene="lacZ", regulation_type="activation",
        evidence_level=EvidenceLevel.DATABASE, confidence=0.5,
    )
    assert not low_conf.is_high_confidence
    low_evidence = RegulatoryEdge(
        tf_id="crp", target_gene="lacZ", regulation_type="activation",
        evidence_level=EvidenceLevel.PREDICTED, confidence=0.9,
    )
    assert not low_evidence.is_high_confidence


# ============================================================================
# GRNInferenceResult helpers
# ============================================================================

def _edge(tf: str, target: str, ev: EvidenceLevel, conf: float) -> RegulatoryEdge:
    return RegulatoryEdge(
        tf_id=tf, target_gene=target, regulation_type="activation",
        evidence_level=ev, confidence=conf,
    )


def test_result_grouping_helpers():
    r = GRNInferenceResult()
    r.regulatory_edges = [
        _edge("a", "t1", EvidenceLevel.DATABASE, 0.9),
        _edge("a", "t2", EvidenceLevel.DATABASE, 0.9),
        _edge("b", "t1", EvidenceLevel.PREDICTED, 0.5),
    ]
    by_tf = r.by_tf()
    assert set(by_tf) == {"a", "b"}
    assert len(by_tf["a"]) == 2
    by_target = r.by_target()
    assert set(by_target) == {"t1", "t2"}
    assert len(by_target["t1"]) == 2
    hc = r.high_confidence_edges()
    assert len(hc) == 2


# ============================================================================
# infer_grn: database matching
# ============================================================================

def test_infer_grn_database_match_and_unvalidated():
    res = infer_grn(
        _tf_result("crp"),
        database_interactions=[
            ("crp", "lacZ", "activation", 1, 0.95),
            ("fnr", "sdhCDAB", "repression", 1, 0.90),
        ],
        use_motif_prediction=False,
    )
    assert res.total_tfs == 1
    assert res.database_matches == 1
    assert res.total_edges == 2
    assert res.total_targets == 2
    crp = [e for e in res.regulatory_edges if e.tf_id == "crp"][0]
    assert crp.confidence == 0.95
    assert crp.source == "database"
    assert crp.regulation_type == "activation"
    fnr = [e for e in res.regulatory_edges if e.tf_id == "fnr"][0]
    assert fnr.confidence == pytest.approx(0.90 * 0.7)
    assert fnr.evidence_level == EvidenceLevel.DATABASE


def test_infer_grn_default_database():
    res = infer_grn(_tf_result("crp"), use_motif_prediction=False)
    assert res.database_matches >= 1


def test_infer_grn_genome_skip_unknown_tf():
    res = infer_grn(
        _tf_result("crp"),
        database_interactions=[("fnr", "sdhCDAB", "repression", 1, 0.90)],
        use_motif_prediction=False,
        genome_gene_ids={"crp", "lacZ"},
    )
    assert res.skipped_edges == 1
    assert res.regulatory_edges == []


def test_infer_grn_genome_skip_unknown_target():
    res = infer_grn(
        _tf_result("fnr"),
        database_interactions=[
            ("fnr", "sdhCDAB", "repression", 1, 0.90),
            ("fnr", "cydAB", "activation", 1, 0.95),
        ],
        use_motif_prediction=False,
        genome_gene_ids={"fnr", "cydAB"},
    )
    assert res.skipped_edges == 1
    assert res.database_matches == 1
    kept = [e for e in res.regulatory_edges if e.target_gene == "cydAB"]
    assert len(kept) == 1


# ============================================================================
# infer_grn: motif path
# ============================================================================

_PROTEIN_FASTA = (
    ">crp\nMKVTLS\n>lacZ\nMTMITDS\n>araBAD\nMKTQCY\n>random_gene\nMAAAA\n"
)


def test_infer_grn_motif_enabled(tmp_path):
    genome = tmp_path / "g.fa"
    genome.write_text(_PROTEIN_FASTA)
    res = infer_grn(
        _tf_result("crp", "fnr"),
        genome_fasta=str(genome),
        use_motif_prediction=True,
        database_interactions=[],
    )
    assert res.motif_predictions > 0


# ============================================================================
# _predict_motifs
# ============================================================================

def test_predict_motifs_protein_fallback(tmp_path):
    genome = tmp_path / "p.fa"
    genome.write_text(_PROTEIN_FASTA)
    edges = _predict_motifs({"crp": _cand("crp", "crp")}, str(genome), 300)
    assert edges
    assert all(e.source == "pwm_prediction" for e in edges)
    targets = {e.target_gene for e in edges}
    assert "lacZ" in targets
    assert "araBAD" in targets


def test_predict_motifs_pwm_nucleotide(tmp_path):
    genome = tmp_path / "n.fa"
    crp_seq = "TGTGATCACA" + "A" * 400
    body = f">crp\n{crp_seq}\n>lacZ\n{'A'*400}\n>araBAD\n{'A'*400}\n"
    genome.write_text(body)
    edges = _predict_motifs({"crp": _cand("crp", "crp")}, str(genome), 300)
    assert edges
    assert all(e.source == "pwm_prediction" for e in edges)
    assert all(e.motif_score >= 7.5 for e in edges)


def test_predict_motifs_positional_heuristic(tmp_path):
    genome = tmp_path / "p2.fa"
    genome.write_text(_PROTEIN_FASTA)
    edges = _predict_motifs(
        {"strangeTF": _cand("strangeTF", "unknown_family")}, str(genome), 300
    )
    assert edges


def test_predict_motifs_missing_genome(tmp_path):
    missing = tmp_path / "nope.fa"
    edges = _predict_motifs({"crp": _cand("crp", "crp")}, str(missing), 300)
    assert edges == []


def test_predict_motifs_empty_genome(tmp_path):
    genome = tmp_path / "empty.fa"
    genome.write_text("no fasta headers here\n")
    edges = _predict_motifs({"crp": _cand("crp", "crp")}, str(genome), 300)
    assert edges == []


# ============================================================================
# _extract_upstream_sequences
# ============================================================================

def test_extract_upstream_whole_genome(tmp_path):
    fa = tmp_path / "genome.fa"
    fa.write_text(">lcl|contig1.1 desc\n" + "ACGTACGT" * 200 + "\n")
    up = _extract_upstream_sequences(str(fa), upstream_bp=300)
    assert "contig1" in up
    assert len(up["contig1"]) == 300


def test_extract_upstream_short_record_kept(tmp_path):
    fa = tmp_path / "genome.fa"
    fa.write_text(">ref|NC_000913.1\n" + "ACGT" * 100 + "\n")
    up = _extract_upstream_sequences(str(fa), upstream_bp=300)
    assert "NC_000913" in up
    assert len(up["NC_000913"]) == 300


def test_extract_upstream_per_gene_empty(tmp_path):
    fa = tmp_path / "genes.fa"
    fa.write_text(">g1\nATGAAAGGG\n>g2\nATGCGTACG\n")
    assert _extract_upstream_sequences(str(fa), upstream_bp=300) == {}


def test_extract_upstream_no_records(tmp_path):
    fa = tmp_path / "empty.fa"
    fa.write_text("some non-fasta text\n")
    assert _extract_upstream_sequences(str(fa), upstream_bp=300) == {}


def test_extract_upstream_unreadable(tmp_path):
    missing = tmp_path / "missing.fa"
    assert _extract_upstream_sequences(str(missing), upstream_bp=300) == {}


# ============================================================================
# _score_pwm
# ============================================================================

def test_score_pwm_perfect_match():
    assert _score_pwm("TGTGATCACA" + "A" * 30, "TGTGA[ATCG]{6}TCACA") >= 7.5


def test_score_pwm_no_match_negative():
    score = _score_pwm("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "TTTTTTTT")
    assert score < 0.0


def test_score_pwm_unknown_bracket_unclosed():
    score = _score_pwm("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "A[ATCG")
    assert isinstance(score, float)


def test_score_pwm_non_iupac_char():
    score = _score_pwm("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "A_X")
    assert score >= 0.0


def test_score_pwm_empty_positions():
    assert _score_pwm("AAAAAAAAAAAAAAAAAAAAAAAA", "[]") == 0.0


# ============================================================================
# Additional branch coverage
# ============================================================================

def test_infer_grn_validated_but_not_predicted():
    res = infer_grn(
        _tf_result("crp"),
        database_interactions=[("fnr", "sdhCDAB", "repression", 1, 0.90)],
        use_motif_prediction=False,
        genome_gene_ids={"crp", "fnr", "sdhCDAB"},
    )
    assert res.regulatory_edges == []
    assert res.skipped_edges == 0


def test_predict_motifs_tf_upstream_missing(tmp_path):
    genome = tmp_path / "n.fa"
    body = f">lacZ\n{'A' * 400}\n>zzz\n{'C' * 400}\n"
    genome.write_text(body)
    edges = _predict_motifs({"crp": _cand("crp", "crp")}, str(genome), 300)
    assert edges


def test_predict_motifs_pwm_below_threshold(tmp_path):
    genome = tmp_path / "n.fa"
    genome.write_text(f">crp\n{'C' * 400}\n>lacZ\n{'A' * 400}\n")
    edges = _predict_motifs({"crp": _cand("crp", "crp")}, str(genome), 300)
    assert edges


def test_predict_motifs_tf_self_excluded(tmp_path):
    genome = tmp_path / "n.fa"
    fur_motif = "TAAATAATAGATAACGAT" + "T" * 400
    genome.write_text(
        f">fur\n{fur_motif}\n>feoABC\n{'A' * 400}\n>fhuAB\n{'A' * 400}\n"
    )
    edges = _predict_motifs({"fur": _cand("fur", "fur")}, str(genome), 300)
    assert edges
    assert all(e.target_gene != "fur" for e in edges)


def test_predict_motifs_positional_break(tmp_path):
    genome = tmp_path / "p.fa"
    genes = [f"gene{i}" for i in range(21)]
    body = "".join(f">{g}\nMAAAA\n" for g in genes)
    genome.write_text(body)
    edges = _predict_motifs(
        {"tfX": _cand("tfX", "unknown_family")}, str(genome), 300
    )
    assert edges


def test_predict_motifs_header_parsing(tmp_path):
    genome = tmp_path / "p.fa"
    genome.write_text(">lcl|geneA.1\nMAAA\n>gnl|geneB\nMCCC\n>lcl|\nMGGG\n")
    edges = _predict_motifs({"tfY": _cand("tfY", "unknown_family")}, str(genome), 300)
    assert edges


def test_predict_motifs_level2_self_excluded(tmp_path):
    genome = tmp_path / "p.fa"
    genome.write_text(">fur\nMAAA\n>feoABC\nMCCC\n>fhuAB\nMGGG\n")
    edges = _predict_motifs({"fur": _cand("fur", "fur")}, str(genome), 300)
    assert edges
    assert all(e.target_gene != "fur" for e in edges)


def test_extract_upstream_mixed_lengths(tmp_path):
    fa = tmp_path / "genome.fa"
    fa.write_text(f">long\n{'A' * 1000}\n>short\n{'A' * 200}\n")
    up = _extract_upstream_sequences(str(fa), upstream_bp=300)
    assert len(up["long"]) == 300
    assert len(up["short"]) == 200
