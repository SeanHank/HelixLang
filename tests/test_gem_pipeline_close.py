"""Coverage-closure tests for :mod:`helixlang.plugins.apps.gem_pipeline`.

Exercises the annotation fallback chain (FASTA header heuristic, DIAMOND,
UniProt ID-mapping / NCBI BLAST / sequence-search online paths with a mocked
``urllib``), the EC helpers, and every ``run_gem_pipeline`` stage-failure and
skipped-branch fallback.
"""
import json
import textwrap
import time
import urllib.parse
import urllib.request

import pytest

import helixlang.plugins.apps.gem_pipeline as gp
from helixlang.plugins.annotation import GeneAnnotation


@pytest.fixture
def mini_fasta(tmp_path):
    fasta = tmp_path / "mini.fasta"
    fasta.write_text(textwrap.dedent("""\
        >gene_001 hypothetical protein
        MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV
        >gene_002 hypothetical protein
        MSIQHFRVALIPFFAAFCLPKDIKNPEKFEKIDMRSLHALATSRYQGLLAAIER
        >gene_003 hypothetical protein
        MVSKLPEPVKNDDIELAKRTLTypeII_topoisomerase
    """))
    return str(fasta)


def _ann(gene_id, ec_numbers=(), seq="MKTAYIAKQR"):
    return GeneAnnotation(
        gene_id=gene_id,
        protein_seq=seq,
        ec_numbers=list(ec_numbers),
        kegg_ko=[],
        go_terms=[],
        confidence=0.8 if ec_numbers else 0.1,
    )


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload


class TestSummaryWithIssues:
    def test_summary_reports_warnings_and_errors(self):
        result = gp.GemPipelineResult(warnings=["w1"], errors=["e1"])
        text = result.summary()
        assert "Warnings: 1" in text
        assert "Errors: 1" in text

    def test_summary_without_issues(self):
        text = gp.GemPipelineResult().summary()
        assert "Warnings:" not in text
        assert "Errors:" not in text


class TestAnnotateFromFasta:
    def test_header_fallback_when_extraction_empty(self, tmp_path, monkeypatch):
        fasta = tmp_path / "headers.fasta"
        fasta.write_text(">\nACGT\n>gene_alpha alpha protein\n>gene_beta beta protein\n")
        monkeypatch.setattr(gp, "extract_protein_sequences", lambda *a, **k: {})
        annotations = gp._annotate_from_fasta(str(fasta))
        assert set(annotations) == {"gene_alpha", "gene_beta"}
        assert all(a.protein_seq == "" for a in annotations.values())
        assert all(a.confidence == 0.1 for a in annotations.values())

    def test_header_fallback_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gp, "extract_protein_sequences", lambda *a, **k: {})
        annotations = gp._annotate_from_fasta(str(tmp_path / "missing.fasta"))
        assert annotations == {}

    def test_diamond_success(self, mini_fasta, monkeypatch):
        class _Hit:
            def __init__(self, title, subject_id):
                self.stitle = title
                self.subject_id = subject_id

        class _DiamondResult:
            def __init__(self):
                self._hits = {
                    "gene_001": [
                        _Hit("tr|Q8|A0A0 EC:1.1.1.1 alcohol dehydrogenase", "K00001"),
                        _Hit("sp|P0|AAAA phosphate acetyltransferase 2.3.1.8", "X123"),
                    ],
                    "gene_002": [_Hit("just a name, no EC code", "K99999")],
                    "gene_003": [],
                }

            def hits_for(self, gene_id):
                return self._hits[gene_id]

        monkeypatch.setattr(
            "helixlang.plugins.annotation.blast.run_diamond",
            lambda *a, **k: _DiamondResult(),
        )
        annotations = gp._annotate_from_fasta(mini_fasta, diamond_db="x.dmnd")
        assert annotations["gene_001"].ec_numbers == ["1.1.1.1", "2.3.1.8"]
        assert annotations["gene_001"].kegg_ko == ["K00001"]
        assert annotations["gene_001"].confidence == 0.8
        assert annotations["gene_002"].ec_numbers == []
        assert annotations["gene_002"].kegg_ko == ["K99999"]
        assert annotations["gene_003"].ec_numbers == []
        assert annotations["gene_003"].confidence == 0.3

    def test_diamond_file_not_found_falls_through(self, mini_fasta, monkeypatch):
        def _boom(*a, **k):
            raise FileNotFoundError("diamond not installed")

        monkeypatch.setattr("helixlang.plugins.annotation.blast.run_diamond", _boom)
        annotations = gp._annotate_from_fasta(mini_fasta, diamond_db="x.dmnd")
        assert annotations["gene_001"].confidence == 0.1

    def test_diamond_generic_error_falls_through(self, mini_fasta, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("diamond crashed")

        monkeypatch.setattr("helixlang.plugins.annotation.blast.run_diamond", _boom)
        annotations = gp._annotate_from_fasta(mini_fasta, diamond_db="x.dmnd")
        assert annotations["gene_001"].confidence == 0.1

    def test_uniprot_api_results_returned_directly(self, mini_fasta, monkeypatch):
        monkeypatch.setattr(
            gp, "_annotate_via_uniprot_api",
            lambda proteins: {"gene_001": _ann("gene_001", ["1.1.1.1"])},
        )
        annotations = gp._annotate_from_fasta(mini_fasta)
        assert annotations["gene_001"].ec_numbers == ["1.1.1.1"]


class TestExtractEcHelpers:
    def test_extract_ec_from_hit_patterns(self):
        assert gp._extract_ec_from_hit("tr|Q8|A0A0 EC:1.2.3.4 note") == "1.2.3.4"
        assert gp._extract_ec_from_hit("anything ends 9.8.7.6") == "9.8.7.6"
        assert gp._extract_ec_from_hit("no ec here") is None

    def test_extract_ec_from_blast_title(self):
        assert gp._extract_ec_from_blast_title(
            "recName: Full=alcohol dehydrogenase EC:1.1.1.1") == "1.1.1.1"
        assert gp._extract_ec_from_blast_title(
            "some hit description 4.1.2.13 context") == "4.1.2.13"
        assert gp._extract_ec_from_blast_title("nothing relevant") is None


class TestAnnotateViaUniprotApi:
    def test_idmapping_covers_all_genes(self, monkeypatch):
        proteins = {"A": "seq", "B": "seq"}
        monkeypatch.setattr(
            gp, "_annotate_via_uniprot_idmapping",
            lambda p: {g: _ann(g) for g in p},
        )
        annotations = gp._annotate_via_uniprot_api(proteins)
        assert set(annotations) == {"A", "B"}

    def test_empty_proteins_skips_strategies(self, monkeypatch):
        monkeypatch.setattr(
            gp, "_annotate_via_uniprot_idmapping", lambda p: {},
        )
        assert gp._annotate_via_uniprot_api({}) == {}

    def test_empty_sequences_skip_network_strategies(self, monkeypatch):
        proteins = {"A": "", "B": ""}
        monkeypatch.setattr(
            gp, "_annotate_via_uniprot_idmapping", lambda p: {},
        )
        assert gp._annotate_via_uniprot_api(proteins) == {}


class TestUniprotIdMappingOnline:
    def _patch(self, monkeypatch, payloads):
        monkeypatch.setattr(gp, "_network_offline", lambda: False)
        monkeypatch.setattr(time, "sleep", lambda s: None)

        def _as_bytes(value):
            return value if isinstance(value, bytes) else value.encode()

        uri = _as_bytes(payloads["uri"])
        uri2 = _as_bytes(payloads.get("uri2", payloads["uri"]))
        stats = _as_bytes(payloads.get("status", b'{"jobStatus":"FINISHED"}'))
        results = json.dumps({"results": payloads["results"]}).encode()

        def _fake_urlopen(req, *args, **kwargs):
            url = req.full_url
            if "idmapping/run" in url:
                return _FakeResp(_as_bytes(payloads.get("job", b'{"jobId":"J1"}')))
            if "idmapping/status" in url:
                return _FakeResp(stats)
            if "idmapping/uniprotkb/results" in url:
                return _FakeResp(results)
            if "uniprotkb/" in url:
                if "P00000" in url:
                    return _FakeResp(RuntimeError("entry fetch failed"))
                if "Q54321" in url:
                    return _FakeResp(uri2)
                return _FakeResp(uri)
            raise AssertionError(f"unexpected URL {url}")

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    def test_online_happy_path(self, monkeypatch):
        proteins = {"G1": "MKTAY", "G2": "MKTAY", "G3": "MKTAY",
                    "G4": "MKTAY", "G5": "MKTAY"}
        self._patch(monkeypatch, {
            "uri": json.dumps({
                "proteinDescription": {"recommendedName": {
                    "ecNumbers": [{"value": "1.1.1.1"},
                                  {"value": "1.2.3.4"}, {"value": ""}]}},
                "uniProtKBCrossReferences": [
                    {"database": "GO", "id": "GO:0001"},
                    {"database": "PFAM", "id": "PF000"}, ]}).encode(),
            "uri2": json.dumps({
                "proteinDescription": {"recommendedName": {}}}).encode(),
            "results": [
                {"from": "G1", "to": {"primaryAccession": "P12345"}},
                {"from": "G2",
                 "to": {"primaryAccession": "", "accession": ["Q54321"]}},
                {"from": "G3", "to": {"primaryAccession": ""}},
                {"from": "G4", "to": "notadict"},
                {"from": "G5", "to": {"primaryAccession": "P00000"}}],
        })
        annotations = gp._annotate_via_uniprot_idmapping(proteins)
        assert annotations["G1"].ec_numbers == ["1.1.1.1", "1.2.3.4"]
        assert annotations["G1"].go_terms == ["GO:0001"]
        assert annotations["G1"].confidence == 0.7
        assert annotations["G2"].confidence == 0.3
        assert "G3" not in annotations and "G4" not in annotations
        assert "G5" not in annotations

    def test_online_poll_never_finishes(self, monkeypatch):
        proteins = {"A": "MKTAY"}
        self._patch(monkeypatch, {
            "status": b'{"jobStatus":"RUNNING"}',
            "uri": b'{"proteinDescription":{}}',
            "results": [],
        })
        assert gp._annotate_via_uniprot_idmapping(proteins) == {}

    def test_online_no_job_id(self, monkeypatch):
        self._patch(monkeypatch, {
            "job": b'{}', "uri": b'{}', "results": []})
        assert gp._annotate_via_uniprot_idmapping({"A": "x"}) == {}

    def test_online_network_error_warns(self, monkeypatch):
        monkeypatch.setattr(gp, "_network_offline", lambda: False)
        monkeypatch.setattr(time, "sleep", lambda s: None)
        monkeypatch.setattr(
            urllib.request, "urlopen",
            lambda *a, **k: _FakeResp(ConnectionError("offline")),
        )
        with pytest.warns(UserWarning):
            assert gp._annotate_via_uniprot_idmapping({"A": "x"}) == {}


class TestNcbiBlastOnline:
    def _patch(self, monkeypatch, put_html, poll_json):
        monkeypatch.setattr(gp, "_network_offline", lambda: False)
        monkeypatch.setattr(time, "sleep", lambda s: None)
        put_html = put_html.encode()

        def _fake_urlopen(req, *args, **kwargs):
            url = req.full_url
            if "CMD=Get" in url:
                return _FakeResp(poll_json)
            if "blast.ncbi.nlm.nih.gov/blast/Blast.cgi" in url:
                return _FakeResp(put_html)
            raise AssertionError(f"unexpected URL {url}")

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    def test_online_happy_path(self, monkeypatch):
        poll = json.dumps({
            "Status": "ready",
            "blastoutput": {"results": {"blasts": [
                {"query_title": "g1",
                 "hits": [{"title": "recName: Full=citrate synthase EC:2.3.3.8"},
                          {"title": "secondary hit"}]},
                {"query_title": "g2",
                 "hits": [{"title": "no ec description a"},
                          {"title": "no ec description b"}]},
                {"query_title": "unknown_zzz", "hits": []}]}},
        }).encode()
        self._patch(monkeypatch, "job text\nRID = DE9NK2HP014 reply\n", poll)
        annotations = gp._annotate_via_ncbi_blast({"g1": "MKTAY", "g2": "MKTAY"})
        assert annotations["g1"].ec_numbers == ["2.3.3.8"]
        assert annotations["g1"].confidence == 0.6
        assert annotations["g2"].ec_numbers == []
        assert annotations["g2"].confidence == 0.2
        assert "unknown_zzz" not in annotations

    def test_online_poll_never_ready(self, monkeypatch):
        self._patch(monkeypatch, "text\nRID = ABC1 x\n",
                    b"Status=WAITING still running")
        assert gp._annotate_via_ncbi_blast({"g1": "MKTAY"}) == {}

    def test_online_missing_rid(self, monkeypatch):
        self._patch(monkeypatch, "no job identifier here", b"{}")
        assert gp._annotate_via_ncbi_blast({"g1": "MKTAY"}) == {}

    def test_online_malformed_json(self, monkeypatch):
        self._patch(monkeypatch, "text\nRID = ABC123 x\n", b"Status=ready not json")
        assert gp._annotate_via_ncbi_blast({"g1": "MKTAY"}) == {}

    def test_online_network_error_warns(self, monkeypatch):
        monkeypatch.setattr(gp, "_network_offline", lambda: False)
        monkeypatch.setattr(time, "sleep", lambda s: None)
        monkeypatch.setattr(
            urllib.request, "urlopen",
            lambda *a, **k: _FakeResp(ConnectionError("offline")),
        )
        with pytest.warns(UserWarning):
            assert gp._annotate_via_ncbi_blast({"g1": "MKTAY"}) == {}


class TestUniprotSequenceOnline:
    def _patch(self, monkeypatch, success_seqs, fail_seqs=()):
        monkeypatch.setattr(gp, "_network_offline", lambda: False)
        monkeypatch.setattr(time, "sleep", lambda s: None)

        def _fake_urlopen(req, *args, **kwargs):
            url = urllib.parse.unquote(req.full_url)
            if any(s in url for s in success_seqs):
                return _FakeResp(json.dumps({
                    "results": [{"proteinDescription": {"recommendedName": {
                        "ecNumbers": [{"value": "6.2.1.1"}, {"value": ""}]}}}]}).encode())
            return _FakeResp(b'{"results":[]}')

        def _boom(req, *args, **kwargs):
            raise ConnectionError("offline")

        monkeypatch.setattr(
            urllib.request, "urlopen", _boom if fail_seqs else _fake_urlopen)

    def test_online_skips_short_and_annotates(self, monkeypatch):
        proteins = {
            "g1": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVG",
            "g2": "MSIQHFRVALIPFFAAFCLPKDIKNPEKFEKIDMRSLHALA",
            "short": "MKT",
            "empty": "",
        }
        self._patch(monkeypatch, success_seqs=("MKTAYIAKQR",))
        annotations = gp._annotate_via_uniprot_sequence(proteins)
        assert annotations["g1"].ec_numbers == ["6.2.1.1"]
        assert annotations["g1"].confidence == 0.5
        assert "g2" not in annotations

    def test_online_failure_warns_once(self, monkeypatch):
        proteins = {
            "a": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVG",
            "b": "MSIQHFRVALIPFFAAFCLPKDIKNPEKFEKIDMRSLHALA",
        }
        self._patch(monkeypatch, success_seqs=(), fail_seqs=("a",))
        with pytest.warns(UserWarning):
            annotations = gp._annotate_via_uniprot_sequence(proteins)
        assert annotations == {}


class TestInferSubstrate:
    def test_ec_substrate_map(self):
        cases = {"1.1.1.1": "NAD", "2.7.1.1": "ATP", "3.1.3.1": "H2O",
                 "6.2.1.1": "ATP", "7.2.1.1": "ADP", "2.3.3.1": "CoA",
                 "9.9.9.9": ""}
        for ec, expected in cases.items():
            assert gp._infer_substrate_from_ec(ec) == expected


class TestRunGemPipelineEdges:
    def test_invalid_fasta_first_line(self, tmp_path):
        fasta = tmp_path / "bad.fasta"
        fasta.write_text("not a fasta header\nACGT\n")
        result = gp.run_gem_pipeline(str(fasta))
        assert any("Invalid FASTA" in e for e in result.errors)
        assert result.stages_completed == 0

    def test_missing_genome_file(self, tmp_path):
        result = gp.run_gem_pipeline(str(tmp_path / "nope.fasta"))
        assert any("not found" in e for e in result.errors)
        assert result.stages_completed == 0

    def test_stage2_failure(self, mini_fasta, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("annotation crashed")

        monkeypatch.setattr(gp, "_annotate_from_fasta", _boom)
        result = gp.run_gem_pipeline(mini_fasta)
        assert any("Stage 2 failed" in e for e in result.errors)
        assert result.stages_completed == 0

    def test_stage3_failure(self, mini_fasta, monkeypatch):
        monkeypatch.setattr(
            gp, "_annotate_from_fasta", lambda *a, **k: {"g1": _ann("g1")})

        def _boom(*a, **k):
            raise ValueError("reconstruction failed")

        monkeypatch.setattr(gp, "bottom_up_reconstruct", _boom)
        result = gp.run_gem_pipeline(mini_fasta)
        assert any("Stage 3 failed" in e for e in result.errors)
        assert result.stages_completed == 2

    def test_run_gapfill_disabled(self, mini_fasta, monkeypatch):
        monkeypatch.setattr(gp, "infer_grn", lambda *a, **k: None)
        result = gp.run_gem_pipeline(mini_fasta, run_gapfill=False)
        assert result.stages_completed == 6
        assert result.gapfill is None
        assert not result.errors

    def test_stage4_failure_and_no_annotations(self, mini_fasta, monkeypatch):
        monkeypatch.setattr(gp, "_annotate_from_fasta", lambda *a, **k: {})

        def _boom(*a, **k):
            raise RuntimeError("grn inference crashed")

        monkeypatch.setattr(gp, "infer_grn", _boom)
        result = gp.run_gem_pipeline(mini_fasta)
        assert any("Stage 4 failed" in w for w in result.warnings)
        assert result.stages_completed == 6

    def test_stage5_failure_disables_enzyme_capacity(
            self, mini_fasta, monkeypatch):
        monkeypatch.setattr(gp, "_annotate_from_fasta", lambda *a, **k: {})

        def _boom(*a, **k):
            raise RuntimeError("kcat prediction crashed")

        monkeypatch.setattr(gp, "predict_kcat", _boom)
        result = gp.run_gem_pipeline(mini_fasta)
        assert any("Stage 5 failed" in w for w in result.warnings)
        assert result.kcat_predictions == []
        assert result.stages_completed == 6

    def test_stage6_failure_warns(self, mini_fasta, monkeypatch):
        monkeypatch.setattr(gp, "_annotate_from_fasta", lambda *a, **k: {})

        def _boom(*a, **k):
            raise RuntimeError("model build crashed")

        monkeypatch.setattr(
            "helixlang.plugins.gem.bridge.build_functional_model", _boom)
        result = gp.run_gem_pipeline(mini_fasta)
        assert any("Stage 6 failed" in w for w in result.warnings)
        assert result.stages_completed == 5

    def test_falsey_consensus_skips_stages(self, mini_fasta, monkeypatch):
        class _FalseyConsensus:
            reaction_count = 0
            from_bottom_up_only = 0
            from_both = 0
            reactions = []

            def __bool__(self):
                return False

            def reaction_ids(self):
                return []

        monkeypatch.setattr(gp, "_annotate_from_fasta", lambda *a, **k: {})
        monkeypatch.setattr(
            gp, "consensus_merge", lambda bu, td: _FalseyConsensus())
        monkeypatch.setattr(gp, "infer_grn", lambda *a, **k: None)
        result = gp.run_gem_pipeline(mini_fasta, run_gapfill=False)
        assert result.stages_completed == 6
        assert result.kcat_predictions == []

    def test_stage5_ec_mapping_and_substrate(self, mini_fasta, monkeypatch):
        annotations = {
            "g1": _ann("g1", ["2.3.3.1"]),
            "g2": _ann("g2", ["2.3.3.1", "6.2.1.1"]),
        }
        monkeypatch.setattr(
            gp, "_annotate_from_fasta", lambda *a, **k: annotations)
        result = gp.run_gem_pipeline(mini_fasta)
        assert result.stages_completed in (5, 6)
        assert "CS" in result.km_estimates
        assert len(result.kcat_predictions) > 0

    def _run_with_mock_fba(self, mini_fasta, monkeypatch, solve_raises):
        class _Model:
            biomass_reaction = "BIOMASS"
            _growth_rate = 0.0
            _fba_fluxes = {}

        class _FBA:
            def __init__(self, *a, **k):
                pass

            def set_enzyme_capacity(self, *a, **k):
                pass

            def solve(self, *a, **k):
                if solve_raises:
                    raise RuntimeError("solve failed")
                return {"BIOMASS": 0.3}

        monkeypatch.setattr(
            "helixlang.plugins.gem.bridge.build_functional_model",
            lambda **k: _Model())
        monkeypatch.setattr(
            "helixlang.plugins.runtime.metabolism.FluxBalanceAnalysis", _FBA)
        return gp.run_gem_pipeline(mini_fasta)

    def test_stage6_enzyme_solve_success_updates_fluxes(
            self, mini_fasta, monkeypatch):
        result = self._run_with_mock_fba(mini_fasta, monkeypatch,
                                          solve_raises=False)
        assert result.stages_completed == 6
        assert result.fba_fluxes == {"BIOMASS": 0.3}
        assert result.growth_rate == pytest.approx(0.3)
        assert not result.errors

    def test_stage6_enzyme_solve_exception_swallowed(
            self, mini_fasta, monkeypatch):
        result = self._run_with_mock_fba(mini_fasta, monkeypatch,
                                          solve_raises=True)
        assert result.stages_completed == 6
        assert not result.errors
        assert not any("Stage 6 failed" in w for w in result.warnings)
