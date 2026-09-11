"""Coverage-closure tests for the mid-size runtime modules (branch + line 100%)."""
from __future__ import annotations

import builtins
import importlib
import random

import pytest

from helixlang.plugins.runtime import bio_validity as bv
from helixlang.plugins.runtime import biocodec as bc
from helixlang.plugins.runtime import cell_body as cb
from helixlang.plugins.runtime import dna_codec as dc
from helixlang.plugins.runtime import epigenetics as ep

# ---------------------------------------------------------------------------
# bio_validity.py
# ---------------------------------------------------------------------------

class TestBioValidityClose:
    def test_contains_zero_span(self) -> None:
        r = bv.ParameterRange("x", 5.0, 5.0)
        assert r.contains(5.0) is bv.ScopeLevel.SAFE
        assert r.contains(6.0) is bv.ScopeLevel.OUT_OF_SCOPE

    def test_contains_warning(self) -> None:
        r = bv.ParameterRange("x", 0.0, 10.0)  # span = 5
        assert r.contains(-3.0) is bv.ScopeLevel.WARNING
        assert r.contains(13.0) is bv.ScopeLevel.WARNING
        assert r.contains(-9.0) is bv.ScopeLevel.OUT_OF_SCOPE
        assert r.contains(5.0) is bv.ScopeLevel.SAFE

    def test_report_empty_worst_level(self) -> None:
        report = bv.OutOfScopeReport()
        assert report.worst_level is bv.ScopeLevel.SAFE
        assert report.all_safe is True
        assert report.any_out_of_scope is False
        assert report.to_dict()["worst_level"] == "safe"

    def test_report_warning_worst_level(self) -> None:
        check = bv.ParameterCheck(name="x", value=7.0,
                                  range=bv.ParameterRange("x", 0.0, 10.0),
                                  level=bv.ScopeLevel.WARNING)
        report = bv.OutOfScopeReport(checks=[check])
        assert report.worst_level is bv.ScopeLevel.WARNING

    def test_check_skips_unknown_params(self) -> None:
        detector = bv.OutOfScopeDetector(
            ranges=[bv.ParameterRange("growth_rate", 0.1, 2.0)])
        report = detector.check({"not_a_param": 1.0})
        assert report.checks == []

    def test_fit_result_to_dict(self) -> None:
        result = bv.FitResult(
            fitted_params={}, initial_params={},
            residual_before=0.5, residual_after=0.1,
            improvement_pct=80.0, converged=True, n_iterations=3,
            message="ok")
        d = result.to_dict()
        assert d["converged"] is True
        assert d["message"] == "ok"

    def test_fit_scipy_with_objective_and_zero_target(self) -> None:
        fitter = bv.ParameterFitter(
            bounds={"growth_rate": (0.1, 2.0)},
            objective_fn=lambda p: 1e-6)
        result = fitter.fit(
            initial_params={"growth_rate": 0.5},
            target_values={"growth_rate": 1.0, "h": 0.0},
            maxiter=50)
        assert result.converged in (True, False)
        assert "growth_rate" in result.fitted_params

    def test_fit_coordinate_descent_fallback(self, monkeypatch) -> None:
        real_import = builtins.__import__

        def no_scipy(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("no scipy")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_scipy)
        fitter = bv.ParameterFitter(
            bounds={"growth_rate": (0.1, 2.0)},
            objective_fn=lambda p: 1e-6)
        result = fitter.fit(
            initial_params={"growth_rate": 0.5},
            target_values={"growth_rate": 1.0, "zz": 2.0},
            maxiter=30)
        assert result.initial_params["growth_rate"] == 0.5
        assert result.fitted_params["growth_rate"] != 0.5

    def test_fit_coordinate_descent_no_objective(self, monkeypatch) -> None:
        real_import = builtins.__import__

        def no_scipy(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("no scipy")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_scipy)
        fitter = bv.ParameterFitter(
            bounds={"growth_rate": (0.1, 2.0)})
        result = fitter.fit(
            initial_params={"growth_rate": 0.5},
            target_values={"growth_rate": 1.0},
            maxiter=20)
        assert "growth_rate" in result.fitted_params

    def test_uncertainty_forward_functions(self) -> None:
        q = bv.UncertaintyQuantifier(
            forward_fn=lambda p: p["x"] + 1.0, n_bootstrap=20, seed=7)
        boot = q.bootstrap(base_params={"x": 1.0}, reference_values=[])
        assert boot.method == "bootstrap"
        assert boot.n_samples == 20
        assert boot.to_dict()["n_samples"] == 20
        mc = q.monte_carlo(base_params={"x": 1.0}, param_stds={"x": 0.1},
                           n_samples=20)
        assert mc.method == "monte_carlo"

    def test_replication_run_fn_numeric(self) -> None:
        rv = bv.ReplicationVerifier(
            run_fn=lambda seed, run_index: float(seed) + run_index,
            n_runs=3)
        result = rv.verify()
        assert result.n_runs == 3
        assert result.max_deviation >= 0.0

    def test_compute_overall_partial(self) -> None:
        check = bv.ParameterCheck(name="x", value=7.0,
                                  range=bv.ParameterRange("x", 0.0, 10.0),
                                  level=bv.ScopeLevel.SAFE)
        report = bv.BioAccuracyReport(
            benchmark_id="b1",
            scope=bv.OutOfScopeReport(checks=[check]))
        assert report.compute_overall() == pytest.approx(1.0)
        assert report.status == "PASS"
        assert report.to_dict()["id"] == "b1"

    def test_compute_overall_empty_scope(self) -> None:
        report = bv.BioAccuracyReport(benchmark_id="b2")
        assert report.compute_overall() == 0.0
        assert report.status == "FAIL"


# ---------------------------------------------------------------------------
# cell_body.py
# ---------------------------------------------------------------------------

class TestCellBodyClose:
    def test_closest_point_degenerate_segment(self) -> None:
        assert cb._closest_point_on_segment(1.0, 2.0, 5.0, 5.0, 5.0, 5.0) \
            == (5.0, 5.0)

    def test_segment_distance_both_degenerate(self) -> None:
        sq, p1x, p1y, p2x, p2y = cb._segment_distance_sq(
            0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 3.0, 4.0)
        assert sq == pytest.approx(25.0)
        assert (p1x, p1y, p2x, p2y) == (0.0, 0.0, 3.0, 4.0)

    def test_segment_distance_point_vs_segment(self) -> None:
        sq, p1x, p1y, p2x, p2y = cb._segment_distance_sq(
            0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 10.0, 5.0)
        assert sq == pytest.approx(50.0)
        assert (p2x, p2y) == (5.0, 5.0)

    def test_sites_skips_outside_radius(self) -> None:
        body = cb.CellBody(x=5.0, y=5.0, length_um=0.0, diameter_um=2.0)
        sites = body.sites(10, 10, 1.0)
        assert set(sites) == set(
            [(4, 4), (4, 5), (5, 4), (5, 5)]
        )

    def test_resolve_rods_empty(self) -> None:
        cb.resolve_rod_contacts([], 0.0, 0.0, 10.0, 10.0)

    def test_resolve_rods_exhaust_loop(self) -> None:
        a = cb.CellBody(x=5.0, y=5.0, length_um=4.0, diameter_um=2.0)
        b = cb.CellBody(x=6.2, y=5.0, length_um=4.0, diameter_um=2.0)
        cb.resolve_rod_contacts([a, b], drag=0.0, stiffness=10.0,
                                x_max=20.0, y_max=20.0, iterations=3,
                                tolerance_um=-1.0)

    def test_resolve_rods_zero_force(self) -> None:
        a = cb.CellBody(x=5.0, y=5.0, length_um=4.0, diameter_um=2.0)
        b = cb.CellBody(x=6.2, y=5.0, length_um=4.0, diameter_um=2.0)
        cb.resolve_rod_contacts([a, b], drag=2.0, stiffness=0.0,
                                x_max=20.0, y_max=20.0)

    def _wall_rod(self) -> cb.CellBody:
        return cb.CellBody(x=0.3, y=5.0, length_um=0.0, diameter_um=2.0)

    def test_resolve_rod_wall_contact_drag(self) -> None:
        cb.resolve_rod_contacts([self._wall_rod()], drag=2.0, stiffness=50.0,
                                x_max=20.0, y_max=20.0)
        cb.resolve_rod_contacts([self._wall_rod()], drag=0.0, stiffness=50.0,
                                x_max=20.0, y_max=20.0)
        cb.resolve_rod_contacts([self._wall_rod()], drag=2.0, stiffness=0.0,
                                x_max=20.0, y_max=20.0)

    def test_advect_rods_no_flow(self) -> None:
        body = cb.CellBody(x=5.0, y=5.0, length_um=2.0)
        cb.advect_rods([body], None, 1.0)
        assert (body.x, body.y) == (5.0, 5.0)


# ---------------------------------------------------------------------------
# epigenetics.py
# ---------------------------------------------------------------------------

class TestEpigeneticsClose:
    def test_cpg_islands_disjoint(self) -> None:
        islands = ep.find_cpg_islands(
            "CGCGCG" + "T" * 50 + "CGCGCG",
            min_length=4, min_gc=0.5, min_oe=0.6)
        assert len(islands) == 2

    def test_gc_fraction_empty(self) -> None:
        assert ep._gc_fraction("") == 0.0

    def test_cpg_oe_edge_cases(self) -> None:
        assert ep._cpg_oe("") == 0.0
        assert ep._cpg_oe("AAAA") == 0.0
        assert ep._cpg_oe("CGCG") > 0.0

    def test_methylate_unknown_eukaryotic_methylase(self) -> None:
        with pytest.raises(ValueError):
            ep.methylate_dna("CGAT", "human", "dam")

    def test_histone_default_promoter_short_dna(self) -> None:
        marks = ep.add_histone_marks(
            "A" * 10, [{"name": "g1", "start": 2, "end": 4}])
        assert marks  # short dna skips heterochromatin (n <= 100)

    def test_histone_degenerate_gene(self) -> None:
        marks = ep.add_histone_marks(
            "A" * 120, [{"name": "x", "start": 50, "end": 30}])
        assert marks

    def test_accessibility_unknown_mark(self) -> None:
        chrom = ep.ChromatinState(
            methylation=ep.MethylationState(),
            histone_marks=[
                ep.HistoneMark(position=0, mark="BOGUS", level=1.0),
                ep.HistoneMark(position=0, mark="H3K4me3", level=1.0),
            ])
        acc = ep.calculate_accessibility(chrom)
        assert 0.0 <= acc[0] <= 1.0

    def test_expression_modifier_edges(self) -> None:
        chrom = ep.ChromatinState(
            methylation=ep.MethylationState({100: 0.9}),
            histone_marks=[ep.HistoneMark(position=3, mark="BOGUS", level=1.0)])
        mods = ep.calculate_expression_modifier(
            chrom, [{"name": "g", "start": 5, "end": 10}])
        assert mods["g"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# biocodec.py
# ---------------------------------------------------------------------------

class TestBioCodecClose:
    def test_translate_fallback_without_biopython(self, monkeypatch) -> None:
        real_import = builtins.__import__

        def no_bio(name, *args, **kwargs):
            if name == "Bio" or name.startswith("Bio."):
                raise ImportError("no Bio")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_bio)
        assert bc._translate("ATGGCT") == "MA"

    def test_find_restriction_sites_enzyme_list(self) -> None:
        sites = bc.find_restriction_sites("TTGAATTCTT", ["EcoRI"])
        assert sites == {"EcoRI": [2]}
        assert bc.find_restriction_sites("TTGAATTCTT", ["Bogus"]) == {}

    def test_has_restriction_sites_bool(self) -> None:
        assert bc.has_restriction_sites("TTGAATTCTT", ["EcoRI"]) is True
        assert bc.has_restriction_sites("ACGTACGT", ["EcoRI"]) is False

    def test_avoid_restriction_no_synonym(self) -> None:
        out = bc.avoid_restriction_sites("ATGAATTC", ["EcoRI"],
                                         rng=random.Random(1))
        assert "GAATTC" not in out

    def test_avoid_restriction_multi_site_final_check(self) -> None:
        out = bc.avoid_restriction_sites("TTGAATTCTTGAATTC", ["EcoRI"],
                                         max_attempts=2,
                                         rng=random.Random(3))
        assert "GAATTC" not in out

    def test_back_translate_edges(self) -> None:
        assert bc.back_translate("M*") == "ATGTAA"
        with pytest.raises(ValueError):
            bc.back_translate("MZ")
        assert len(bc.back_translate("MK", optimize="balanced")) == 6
        with pytest.raises(ValueError):
            bc.back_translate("MK", optimize="bogus")

    def test_dna_to_helix_with_restriction_sites(self) -> None:
        prog = bc.dna_to_helix("CCGAATTC" + "ATGGCTGCATA" + "A")
        assert prog.genes
        assert "restriction sites found" in " ".join(prog.notes)

    def test_helix_to_dna_no_gene_blocks(self) -> None:
        with pytest.raises(ValueError):
            bc.helix_to_dna("hello world")

    def test_helix_to_dna_all_flags(self) -> None:
        source = "#gene name=mygene\nATG AAA TAA\n#end"
        full = bc.helix_to_dna(source)
        assert full.startswith("TTTAC")  # lac promoter
        no_promoter = bc.helix_to_dna(source, add_promoter=False)
        assert no_promoter.startswith("ATGAAA")
        bare = bc.helix_to_dna(source, add_promoter=False,
                               add_terminator=False,
                               optimize_codons=False,
                               avoid_restriction=False)
        assert bare == "ATGAAATAA"

    def test_helix_to_dna_protein_without_stop(self) -> None:
        bare = bc.helix_to_dna("#gene name=x\nATG GAA\n#end",
                               optimize_codons=True,
                               avoid_restriction=False,
                               add_promoter=False, add_terminator=False)
        assert bare == "ATGGAATAA"  # ME -> back-translated + TAA

    def test_helix_to_dna_catches_unremovable_site(self, monkeypatch) -> None:
        def boom(dna, enzymes=None, max_attempts=100, rng=None):
            raise ValueError("cannot remove")

        monkeypatch.setattr(bc, "avoid_restriction_sites", boom)
        out = bc.helix_to_dna("#gene name=x\nATG GAA TAA\n#end")
        assert out  # ValueError swallowed -> original sequence kept

    def test_avoid_restriction_sites_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(
            bc, "find_restriction_sites",
            lambda dna, enzymes=None: {"EcoRI": [0]})
        with pytest.raises(ValueError):
            bc.avoid_restriction_sites("ATGAATTC", ["EcoRI"],
                                       max_attempts=5, rng=random.Random(0))

    def test_parse_helix_genes_variants(self) -> None:
        source = "\n#gene\n# comment\nXX\nATG TAA\n#end\n#gene name=only\n# end\n#end"
        genes = bc._parse_helix_genes(source)
        assert genes == [{"name": "unnamed", "codons": ["ATG", "TAA"],
                          "orf_dna": "ATGTAA"}]

    def test_unknown_promoter_and_terminator(self) -> None:
        src = "#gene name=x\nATG TAA\n#end"
        with pytest.raises(ValueError):
            bc.helix_to_dna(src, promoter="XXX")
        with pytest.raises(ValueError):
            bc.helix_to_dna(src, terminator="XXX")

    def test_codon_adaptation_index_full(self) -> None:
        assert 0.0 < bc.codon_adaptation_index_full("ATGGCT") <= 1.0


class _SeqRng:
    """Deterministic random.Random stand-in with a fixed value queue."""

    def __init__(self, values):
        self._values = list(values)

    def random(self) -> float:
        return self._values.pop(0)

    def choice(self, seq):
        return seq[0]


class _DupRNG(random.Random):
    def __init__(self, seed: int = 42, first=()):
        super().__init__(seed)
        self._first = list(first)

    def randint(self, a, b):
        if b - a > 1000 and self._first:
            return self._first.pop(0)
        return super().randint(a, b)


class _DupRandFactory:
    def __init__(self, first):
        self._first = list(first)

    def __call__(self, seed=42):
        return _DupRNG(seed, self._first)


# ---------------------------------------------------------------------------
# dna_codec.py
# ---------------------------------------------------------------------------

class TestDNACodecClose:
    def test_byte_to_trits_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            dc._byte_to_trits(256)

    def test_trits_to_byte_invalid_trit(self) -> None:
        with pytest.raises(ValueError):
            dc._trits_to_byte("7")

    def test_decode_resync_pop(self) -> None:
        # homopolymer introduces a trit 3 that breaks blocking; the
        # window grows >6 and must re-sync by dropping the head trit
        out = dc._dna_to_bytes_goldman("ACGTACGTAAAA" + "ACGT" * 6)
        assert isinstance(out, bytes)

    def test_encode_index_pad(self) -> None:
        index = dc._encode_index(12345, length=30)
        assert len(index) == 30
        assert all(b in "ACGT" for b in index)

    def test_align_payload_empty_window(self) -> None:
        assert dc._align_payload("ACGT", "", 0) == [0, 1, 2, 3]

    def test_align_payload_empty_payload(self) -> None:
        assert dc._align_payload("", "ACGT", 0) == []

    def test_align_payload_sw_and_tail(self) -> None:
        # fast-path has >1 mismatch -> real SW alignment; the
        # payload outruns the window -> tail appending kicks in
        mapping = dc._align_payload("TT" + "A" * 68, "A" * 60, 0)
        assert len(mapping) == 70

    def test_align_payload_gap_branches(self) -> None:
        # one extra base in the middle -> SW picks up/left moves
        mapping = dc._align_payload("CG" * 30 + "C" + "CG" * 30,
                                    "CG" * 60, 0)
        assert len(mapping) == 121

    def test_align_payload_window_consumed(self) -> None:
        # SW ends exactly on the last window column (i == 0 tail);
        # the trailing gap walk exercises the tb==3 (left move) branch
        consensus = ("AAGCCCAATAAACCACTCTGACTGGCCGAATAGGGATATAGG"
                     "CAACGACATGTGCGGCGACCCTTGCGACAGTGACGCTTTCGCC"
                     "GTTGCCTAAACCTAT")
        payload = ("AAGCCCAATAAACCACTCTGACTGGCCGAATAGGGATATAGGC"
                   "ACGACATGTGCGGCGACCCTTGCGACAGTGACGCTTTCGCCGTT"
                   "GCCTAAACCTAT")
        assert len(dc._align_payload(payload, consensus, 26)) == 99

    def test_align_payload_exhausts_payload(self) -> None:
        # all payload rows consumed by the SW traceback -> i == 0
        consensus = "CGCACCCGGGAACGTTCATTTTTAGCATCGGGGTGGCAAACATACTCTCGGCAGTTCTTTCCAA"
        payload = "CGTACCCGGGATCGTTTATTTTTCGCATCGGGTTTGCAAACATACTCTCG"
        assert len(dc._align_payload(payload, consensus, 13)) == 50

    def test_align_payload_nominal_fallthrough(self) -> None:
        # no confident alignment -> nominal mapping
        assert dc._align_payload("ACGT" * 4, "N" * 40, 0) == list(range(16))

    def test_merge_maps_skips_unmapped(self) -> None:
        cols = dc._merge_maps(
            {0: "ACGT"}, {0: [None, -1, 0, 1]})
        assert cols == [{"G": 1}, {"T": 1}]

    def test_assemble_dna_empty(self) -> None:
        assert dc._assemble_dna({}) == ""

    def test_assemble_dna_converges_then_truncates(self) -> None:
        out = dc._assemble_dna({0: "A" * 100}, total_len=2)
        assert out  # refinement converges on pass 1, then slices

    def test_assemble_dna_refines_consensus(self) -> None:
        # votes keep shifting across both refinement passes (consensus
        # is reassigned each pass, loop exhausts naturally)
        segs = {
            0: "CCACCACACAAAACCACAACCACCAACCCACCCACCAACAACCCACCAACCCCAAACACCAAAACCCACCCCAAAAACCCCCCCACAACAAACACCCCAAACCCCACC",
            1: "CACAAAACAACAAACACACCACCAAACAAAACAACACAAAACCCCCAAACCCAACACAACCAACAACAAAAAAAACACCAAACCCCCAAAAACACAACAACACCAAAA",
            2: "CAACACACAACCAAAAACACCAACACACACAAACCCAAACACAAACAACCAACCCAAAACACACCACAACACAAACACCCACACACAACCCAAACAAC",
        }
        out = dc._assemble_dna(segs)
        assert out

    def test_goldman_decode_fallbacks(self) -> None:
        enc = dc.goldman_encode(b"hello world")
        rc_oligo = dc.GoldmanOligo(
            index=0, payload=enc[0].payload, overhang=enc[0].overhang,
            full=dc._reverse_complement(enc[0].full))
        garbage = dc.GoldmanOligo(
            index=7, payload="ACGT" * 25, overhang="A" * 17, full="A" * 117)
        empty_payload = dc.GoldmanOligo(
            index=7, payload="", overhang="A" * 17, full="A" * 117)
        out = dc.goldman_decode([rc_oligo, garbage, empty_payload])
        assert isinstance(out, bytes) and out

    def test_goldman_decode_no_oligos(self) -> None:
        with pytest.raises(ValueError):
            dc.goldman_decode([])

    def test_robust_soliton_zero(self) -> None:
        assert dc.robust_soliton_distribution(0) == []

    def test_sample_degree_exhausts(self) -> None:
        assert dc._sample_degree([0.2, 0.2], random.Random(0)) == 2

    def test_reload_without_optional_deps(self, monkeypatch) -> None:
        real_import = builtins.__import__

        def no_optional(name, globals=None, locals=None, fromlist=(),
                        level=0):
            if name == "reedsolo" or name == "Bio" \
                    or name.startswith("Bio."):
                raise ImportError("unavailable for test")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", no_optional)
        try:
            importlib.reload(dc)
            assert dc._HAS_BIOPYTHON is False
            assert dc._HAS_REEDSOLO is False
            assert dc._RealReedSolomonError is dc.ReedSolomonError
            assert dc.validate_iupac_dna("ACGT")
            assert not dc.validate_iupac_dna("ACG R".replace(" ", ""))
            with pytest.raises(RuntimeError):
                dc.translate_dna("ATG")
            with pytest.raises(RuntimeError):
                dc.erlich_encode(b"x")
            with pytest.raises(RuntimeError):
                dc.erlich_decode([], K=1)
        finally:
            monkeypatch.setattr(builtins, "__import__", real_import)
            importlib.reload(dc)
            assert dc._HAS_BIOPYTHON is True

    def test_erlich_encode_empty_data(self) -> None:
        oligos = dc.erlich_encode(b"")
        assert oligos and all(o.payload for o in oligos)

    def test_erlich_duplicate_seed(self, monkeypatch) -> None:
        data = random.Random(42).randbytes(64)
        baseline = dc.erlich_encode(data)
        dup = [_DupRandFactory([baseline[0].seed, baseline[0].seed])]
        monkeypatch.setattr(dc.random, "Random", dup[0])
        out = dc.erlich_encode(data)
        assert out[0].seed in (baseline[0].seed,)
        assert len(out) == len(baseline)

    def test_erlich_rs_encode_error(self, monkeypatch) -> None:
        class FakeRS:
            def __init__(self, *a, **k):
                pass

            def encode(self, m):
                raise ValueError("out of field")

        monkeypatch.setattr(dc, "RSCodec", FakeRS)
        with pytest.raises(RuntimeError):
            dc.erlich_encode(random.Random(1).randbytes(64))

    def test_erlich_whiten_false_roundtrip(self) -> None:
        data = b"\x1b" * 32  # 0x1B -> ACGT, GC-balanced DNA
        oligos = dc.erlich_encode(data, whiten=False)
        assert oligos
        out = dc.erlich_decode(oligos, K=1, total_len=7, whiten=False)
        assert out == b"\x1b" * 7

    def test_erlich_decode_redundant_guard(self) -> None:
        # duplicated droplets make a degree-0 twin sit in the BP queue
        data = random.Random(5).randbytes(64)
        oligos = dc.erlich_encode(data, redundancy=2.5)
        out = dc.erlich_decode(oligos + oligos, K=2)
        assert out == data

    def test_erlich_decode_rs_error(self, monkeypatch) -> None:
        class FakeRS:
            def __init__(self, *a, **k):
                pass

            def decode(self, d):
                raise IndexError("bad input")

        monkeypatch.setattr(dc, "RSCodec", FakeRS)
        with pytest.raises(ValueError):
            dc.erlich_decode(
                [dc.ErlichOligo(index=0, seed=1, payload="A" * 152,
                                rs_oligo=b"x")],
                K=2)

    def test_erlich_decode_short_droplet(self) -> None:
        with pytest.raises(ValueError):
            dc.erlich_decode(
                [dc.ErlichOligo(index=0, seed=1, payload="ACGT",
                                rs_oligo=b"")],
                K=1)

    def test_erlich_decode_lt_failure(self, monkeypatch) -> None:
        data = random.Random(7).randbytes(64)
        enc_oligos = dc.erlich_encode(data)
        original = dc._sample_degree
        monkeypatch.setattr(dc, "_sample_degree",
                            lambda rsd, prng: 1)
        block0: list[dc.ErlichOligo] = []
        for oligo in enc_oligos:
            prng = random.Random(oligo.seed)
            neighbors = prng.sample(range(2), 1)
            if neighbors == [0]:
                block0.append(oligo)
        assert block0  # at least one droplet covers only block 0
        with pytest.raises(ValueError):
            dc.erlich_decode(block0, K=2)
        monkeypatch.undo()
        assert dc._sample_degree is original

    def test_synthesize_dna_branches(self) -> None:
        # default rates: del=1e-2, ins=1e-4, sub=1.4e-3; thresholds:
        #   0.01 deletion, 0.0101 insertion, 0.0115 substitution
        rng = _SeqRng([1e-3, 1.005e-2, 1.02e-2, 0.99])
        assert dc.synthesize_dna("ACGT", rng=rng) == "ACAT"
        assert dc.synthesize_dna("A", quality="low",
                                 rng=_SeqRng([1e-3])) == ""
        assert dc.synthesize_dna("A", quality="high",
                                 rng=_SeqRng([0.99])) == "A"
        assert isinstance(dc.synthesize_dna("ACGT"), str)

    def test_synthesis_yield(self) -> None:
        assert dc.synthesis_yield(0) == 1.0
        assert dc.synthesis_yield(1) == 1.0
        assert 0.0 < dc.synthesis_yield(140, 0.99) < 1.0

    def test_sequence_dna_branches(self) -> None:
        # illumina_hiseq: indel=1e-4, sub=1e-3 -> per base:
        #   1e-5 insertion (A), 7e-5 deletion, 2e-4 substitution, keep
        out = dc.sequence_dna("ACGT", rng=_SeqRng(
            [1e-5, 7e-5, 2e-4, 0.99]))
        assert out == "AAAT"
        with pytest.raises(ValueError):
            dc.sequence_dna("ACGT", platform="mars")

    def test_decay_dna_branches(self) -> None:
        out = dc.decay_dna("AT", years=10000,
                           rng=_SeqRng([0.0, 0.999]))
        assert out[0] == "A"
        assert out[1] == "N"
        assert dc.decay_dna("A", years=10).endswith("A")

    def test_validate_iupac_import_failure(self, monkeypatch) -> None:
        real_import = builtins.__import__

        def no_iupac(name, globals=None, locals=None, fromlist=(),
                     level=0):
            if "IUPACData" in name:
                raise ImportError("missing")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", no_iupac)
        assert dc.validate_iupac_dna("ATG") is False

    def test_helix_to_dna_schemes(self) -> None:
        out = dc.helix_to_dna("print(1)", scheme="erlich")
        assert out["scheme"] == "erlich"
        assert out["stats"]["K"] == 1
        with pytest.raises(ValueError):
            dc.helix_to_dna("x", scheme="nope")

    def test_dna_to_helix_schemes(self) -> None:
        enc = dc.helix_to_dna("print(1)", scheme="erlich")
        assert "print" in dc.dna_to_helix(enc, scheme="erlich")
        no_stats = {"oligos": enc["oligos"]}
        assert "print" in dc.dna_to_helix(no_stats, scheme="erlich")
        with pytest.raises(ValueError):
            dc.dna_to_helix({"oligos": []}, scheme="nope")
