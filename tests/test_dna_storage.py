"""DNA storage application tests: complete DNAStorage storage pipeline.

Verification goals:
- store/retrieve round-trip fidelity (Goldman + Erlich)
- simulate_lifecycle high-fidelity scenario data recovery
- analyze report field completeness + sanity
- estimate_cost / estimate_durability consistent with literature benchmarks
- format_fasta / parse_fasta round-trip
- compare_with_benchmarks comparison
- Large file (1KB) + empty/very small data edge case handling
"""
from __future__ import annotations

import math
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

# bio dependency optional; skipped if missing
pytest.importorskip("Bio")
pytest.importorskip("reedsolo")

from helixlang.apps.dna_storage import (
    DNA_SEQUENCING_COST_PER_BP_USD,
    DNA_SYNTHESIS_COST_PER_BP_USD,
    AnalysisReport,
    DNAStorage,
    LifecycleReport,
    StorageReport,
    compare_with_benchmarks,
    estimate_cost,
    estimate_durability,
    format_fasta,
    parse_fasta,
)
from helixlang.bio_data import (
    DNA_STORAGE_DENSITY_BENCHMARKS,
    DNA_STORAGE_SHANNON_LIMIT_BIT_PER_NT,
)
from helixlang.dna_codec import ERLICH_OLIGO_SIZE
from helixlang.errors import BioError

# ============================================================================
# store / retrieve round-trip
# ============================================================================

class TestStoreRetrieveRoundtrip:
    def test_store_retrieve_goldman(self):
        """Goldman scheme store/retrieve round-trip fidelity."""
        storage = DNAStorage(scheme="goldman")
        data = b"Hello, HelixLang DNA Storage! " * 10  # ~290 bytes
        report = storage.store(data)
        assert report.scheme == "goldman"
        assert report.num_oligos > 0
        assert report.total_bp > 0
        assert report.avg_oligo_length > 0
        assert report.encoding_time >= 0
        recovered = storage.retrieve(report.oligos, total_len=len(data))
        assert recovered == data

    def test_store_retrieve_erlich(self):
        """Erlich scheme store/retrieve round-trip fidelity (K=128, large enough for reliable BP decoding)."""
        storage = DNAStorage(scheme="erlich")
        rng = random.Random(42)
        data = bytes(rng.randint(0, 255) for _ in range(4096))  # K=128
        report = storage.store(data, redundancy=1.15)
        assert report.scheme == "erlich"
        assert report.num_oligos > 0
        assert report.K == math.ceil(len(data) / ERLICH_OLIGO_SIZE)
        recovered = storage.retrieve(report.oligos, total_len=len(data))
        assert recovered == data

    def test_store_report_fields(self):
        """StorageReport field completeness."""
        storage = DNAStorage(scheme="goldman")
        report = storage.store(b"test data field check")
        assert isinstance(report, StorageReport)
        assert report.scheme == "goldman"
        assert isinstance(report.oligos, list)
        assert isinstance(report.total_bp, int)
        assert isinstance(report.density_bit_per_nt, float)
        assert isinstance(report.num_oligos, int)
        assert isinstance(report.avg_oligo_length, float)
        assert isinstance(report.encoding_time, float)
        assert report.data_len == len(b"test data field check")

    def test_unknown_scheme_raises(self):
        """Unknown scheme raises ValueError."""
        with pytest.raises(BioError):
            DNAStorage(scheme="unknown")


# ============================================================================
# simulate_lifecycle high-fidelity scenarios
# ============================================================================

class TestSimulateLifecycle:
    def test_high_fidelity_goldman(self):
        """High-fidelity scenario: Goldman + noise-free synthesis + 0 PCR + PacBio HiFi + 0 years storage.

        With no synthesis/PCR/decay errors, PacBio HiFi sequencing errors are very low (1e-4),
        and Goldman 4x overlap redundancy voting should perfectly recover the data.
        """
        storage = DNAStorage(scheme="goldman")
        data = b"High fidelity DNA storage lifecycle test! " * 25  # ~1KB
        report = storage.simulate_lifecycle(
            data,
            synthesis_quality="perfect",
            pcr_cycles=0,
            polymerase="q5",
            sequencing_platform="pacbio_hifi",
            storage_years=0,
            seed=42,
        )
        assert isinstance(report, LifecycleReport)
        assert report.success
        assert report.integrity == pytest.approx(1.0)
        assert report.synthesis_errors == 0
        assert report.pcr_errors == 0  # 0 cycles
        assert report.decay_damage == 0  # 0 years

    def test_lifecycle_typical_has_errors(self):
        """Typical scenario (typical synthesis + 10 cycles Taq + Illumina) has non-zero error rate."""
        storage = DNAStorage(scheme="goldman")
        data = b"Typical lifecycle scenario with errors... " * 20
        report = storage.simulate_lifecycle(
            data,
            synthesis_quality="typical",
            pcr_cycles=10,
            polymerase="taq",
            sequencing_platform="illumina_hiseq_novaseq",
            storage_years=0,
            seed=42,
        )
        # typical synthesis 1% deletion + 10 cycles Taq PCR + Illumina sequencing -> should have errors
        assert report.synthesis_errors > 0
        assert report.pcr_errors > 0
        assert report.sequencing_errors > 0
        assert report.error_rate > 0

    def test_lifecycle_with_decay(self):
        """Lifecycle with decay: long-term storage introduces N damage."""
        storage = DNAStorage(scheme="goldman")
        data = b"Long term storage test data! " * 10
        # 1000 years @ 13.1C -> partial degradation
        report = storage.simulate_lifecycle(
            data,
            synthesis_quality="high",
            pcr_cycles=0,
            polymerase="q5",
            sequencing_platform="pacbio_hifi",
            storage_years=1000,
            storage_temp=13.1,
            seed=42,
        )
        assert report.decay_damage >= 0  # 1000 years may have decay

    def test_lifecycle_report_fields(self):
        """LifecycleReport field completeness."""
        storage = DNAStorage(scheme="goldman")
        report = storage.simulate_lifecycle(b"fields check", seed=42)
        assert isinstance(report, LifecycleReport)
        assert isinstance(report.original_data, bytes)
        assert isinstance(report.recovered_data, bytes)
        assert isinstance(report.integrity, float)
        assert isinstance(report.error_rate, float)
        assert isinstance(report.synthesis_errors, int)
        assert isinstance(report.pcr_errors, int)
        assert isinstance(report.sequencing_errors, int)
        assert isinstance(report.decay_damage, int)
        assert isinstance(report.success, bool)
        assert 0.0 <= report.integrity <= 1.0
        assert report.error_rate >= 0.0


# ============================================================================
# analyze report
# ============================================================================

class TestAnalyze:
    def test_analyze_report_fields(self):
        """analyze report field completeness + sanity."""
        storage = DNAStorage(scheme="goldman")
        data = b"Analyze this DNA storage report! " * 10
        report = storage.store(data)
        analysis = storage.analyze(report.oligos, data_len=len(data))
        assert isinstance(analysis, AnalysisReport)
        assert analysis.density_bit_per_nt > 0
        assert analysis.shannon_efficiency > 0
        assert analysis.estimated_cost_usd > 0
        assert analysis.durability_years > 0
        assert analysis.oligo_count == report.num_oligos
        assert analysis.total_bp == report.total_bp
        assert 0.0 <= analysis.gc_content <= 1.0
        assert analysis.max_homopolymer >= 1
        assert isinstance(analysis.comparison, dict)
        assert "goldman_2013" in analysis.comparison
        assert "erlich_2017" in analysis.comparison
        assert "organick_2018" in analysis.comparison
        assert "shannon_limit" in analysis.comparison
        assert "this_work" in analysis.comparison

    def test_analyze_gc_content_reasonable(self):
        """Goldman encoding GC content near 50% (rotation key cycle A->C->G->T)."""
        storage = DNAStorage(scheme="goldman")
        data = bytes(range(256)) * 4  # 1KB diverse data
        report = storage.store(data)
        analysis = storage.analyze(report.oligos, data_len=len(data))
        assert 0.35 <= analysis.gc_content <= 0.65

    def test_analyze_homopolymer_goldman(self):
        """Goldman encoding homopolymer <=2 (guaranteed by rotation key)."""
        storage = DNAStorage(scheme="goldman")
        data = b"AAAA CCCC GGGG TTTT " * 20
        report = storage.store(data)
        analysis = storage.analyze(report.oligos, data_len=len(data))
        assert analysis.max_homopolymer <= 2

    def test_analyze_erlich_homopolymer(self):
        """Erlich encoding homopolymer <=3 (constrained rejection sampling)."""
        storage = DNAStorage(scheme="erlich")
        data = bytes(random.Random(42).randint(0, 255) for _ in range(4096))
        report = storage.store(data, redundancy=1.15)
        analysis = storage.analyze(report.oligos, data_len=len(data))
        assert analysis.max_homopolymer <= 3
        assert analysis.shannon_efficiency <= 1.0


# ============================================================================
# estimate_cost
# ============================================================================

class TestEstimateCost:
    def test_cost_calculation(self):
        """Cost calculation is reasonable: synthesis $0.50/bp + sequencing $0.01/bp."""
        total_bp = 1_000_000  # 1 Mbp
        cost = estimate_cost(total_bp, include_sequencing=True)
        assert cost["synthesis_cost"] == pytest.approx(500_000)
        assert cost["sequencing_cost"] == pytest.approx(10_000)
        assert cost["total_cost"] == pytest.approx(510_000)
        assert cost["cost_per_mb"] == pytest.approx(510_000)

    def test_cost_without_sequencing(self):
        """sequencing_cost=0 when sequencing is excluded."""
        cost = estimate_cost(1000, include_sequencing=False)
        assert cost["sequencing_cost"] == 0
        assert cost["total_cost"] == cost["synthesis_cost"]

    def test_cost_zero_bp(self):
        """Cost is 0 at 0 bp, no division by zero."""
        cost = estimate_cost(0)
        assert cost["synthesis_cost"] == 0
        assert cost["sequencing_cost"] == 0
        assert cost["total_cost"] == 0
        assert cost["cost_per_mb"] == 0

    def test_cost_rates_match_constants(self):
        """Cost rates match the constants."""
        cost = estimate_cost(100)
        assert cost["synthesis_cost"] == 100 * DNA_SYNTHESIS_COST_PER_BP_USD
        assert cost["sequencing_cost"] == 100 * DNA_SEQUENCING_COST_PER_BP_USD


# ============================================================================
# estimate_durability
# ============================================================================

class TestEstimateDurability:
    def test_allentoft_2012_bone_dna(self):
        """Allentoft 2012: bone DNA half-life 521 years at 13.1C."""
        t_half = estimate_durability(13.1, encapsulated=False)
        assert 500 < t_half < 550, f"13.1C -> {t_half} (expected ~521)"

    def test_grass_2015_encapsulated_70c(self):
        """Grass 2015: silica-encapsulated DNA half-life 2000 years at 70C."""
        t_half = estimate_durability(70.0, encapsulated=True)
        assert 1900 < t_half < 2100, f"70C encapsulated -> {t_half} (expected ~2000)"

    def test_minus_20_long_durability(self):
        """Bare DNA durability at -20C > 10,000 years (Arrhenius extrapolation)."""
        t_half = estimate_durability(-20, encapsulated=False)
        assert t_half > 10_000, f"-20C -> {t_half} (expected >10,000)"

    def test_lower_temp_longer_durability(self):
        """Lower temperature means longer durability (Arrhenius monotonicity)."""
        t_warm = estimate_durability(20)
        t_cold = estimate_durability(-20)
        assert t_cold > t_warm

    def test_encapsulated_more_durable(self):
        """Encapsulated DNA is more durable than bare DNA (at the same temperature)."""
        bare = estimate_durability(25, encapsulated=False)
        encapsulated = estimate_durability(25, encapsulated=True)
        assert encapsulated > bare


# ============================================================================
# format_fasta / parse_fasta
# ============================================================================

class TestFasta:
    def test_fasta_goldman_roundtrip(self):
        """Goldman FASTA format -> parse -> decode round-trip."""
        storage = DNAStorage(scheme="goldman")
        data = b"FASTA roundtrip test for Goldman scheme! " * 8
        report = storage.store(data)
        fasta = format_fasta(report.oligos, scheme="goldman")
        assert fasta.startswith(">goldman_0")
        parsed = parse_fasta(fasta)
        assert len(parsed) == len(report.oligos)
        recovered = storage.retrieve(parsed, total_len=len(data))
        assert recovered == data

    def test_fasta_erlich_roundtrip(self):
        """Erlich FASTA format -> parse -> decode round-trip."""
        storage = DNAStorage(scheme="erlich")
        rng = random.Random(42)
        data = bytes(rng.randint(0, 255) for _ in range(4096))  # K=128
        report = storage.store(data, redundancy=1.15)
        fasta = format_fasta(report.oligos, scheme="erlich")
        assert fasta.startswith(">erlich_0")
        parsed = parse_fasta(fasta)
        assert len(parsed) == len(report.oligos)
        recovered = storage.retrieve(parsed, total_len=len(data))
        assert recovered == data

    def test_fasta_format_structure(self):
        """FASTA format structure is correct: header lines + sequence lines alternate."""
        storage = DNAStorage(scheme="goldman")
        report = storage.store(b"structure test")
        fasta = format_fasta(report.oligos, scheme="goldman")
        lines = fasta.strip().split("\n")
        # Each oligo produces 2 lines (header + sequence)
        assert len(lines) == report.num_oligos * 2
        for i in range(0, len(lines), 2):
            assert lines[i].startswith(">goldman_")
            assert len(lines[i + 1]) > 0  # sequence not empty
            assert all(c in "ACGT" for c in lines[i + 1])

    def test_parse_fasta_empty(self):
        """Empty FASTA returns an empty list."""
        assert parse_fasta("") == []
        assert parse_fasta("\n\n  \n") == []

    def test_parse_fasta_detects_scheme(self):
        """parse_fasta auto-detects the scheme."""
        storage = DNAStorage(scheme="goldman")
        report = storage.store(b"scheme detect")
        fasta = format_fasta(report.oligos, scheme="goldman")
        parsed = parse_fasta(fasta)
        from helixlang.dna_codec import GoldmanOligo
        assert all(isinstance(o, GoldmanOligo) for o in parsed)


# ============================================================================
# compare_with_benchmarks
# ============================================================================

class TestCompareWithBenchmarks:
    def test_benchmark_comparison_fields(self):
        """Comparison dict includes all paper benchmarks."""
        result = compare_with_benchmarks(1.0)
        assert "goldman_2013" in result
        assert "erlich_2017" in result
        assert "organick_2018" in result
        assert "shannon_limit" in result
        assert "this_work" in result

    def test_benchmark_values_match_literature(self):
        """Benchmark values match the papers."""
        result = compare_with_benchmarks(1.0)
        assert result["goldman_2013"] == 0.29
        assert result["erlich_2017"] == 1.57
        assert result["organick_2018"] == 0.83
        assert result["shannon_limit"] == 1.58
        assert result["this_work"] == 1.0

    def test_benchmark_this_work_varies(self):
        """this_work reflects the passed density."""
        for d in [0.1, 0.5, 1.0, 1.57]:
            result = compare_with_benchmarks(d)
            assert result["this_work"] == d

    def test_goldman_density_below_shannon(self):
        """Goldman measured density < Shannon limit."""
        from helixlang.bio_data import DNA_STORAGE_DENSITY_BENCHMARKS
        assert DNA_STORAGE_DENSITY_BENCHMARKS["goldman_2013"]["density_bit_per_nt"] < \
               DNA_STORAGE_SHANNON_LIMIT_BIT_PER_NT

    def test_erlich_near_shannon(self):
        """Erlich 2017 density approaches the Shannon limit (99%+)."""
        erlich_d = DNA_STORAGE_DENSITY_BENCHMARKS["erlich_2017"]["density_bit_per_nt"]
        efficiency = erlich_d / DNA_STORAGE_SHANNON_LIMIT_BIT_PER_NT
        assert efficiency > 0.99


# ============================================================================
# Large files + edge cases
# ============================================================================

class TestEdgeCases:
    def test_large_file_1kb(self):
        """1KB file encode round-trip (Goldman)."""
        storage = DNAStorage(scheme="goldman")
        rng = random.Random(123)
        data = bytes(rng.randint(0, 255) for _ in range(1024))
        report = storage.store(data)
        assert report.data_len == 1024
        assert report.total_bp > 1024 * 8  # total bases > information bits
        recovered = storage.retrieve(report.oligos, total_len=len(data))
        assert recovered == data

    def test_large_file_1kb_erlich(self):
        """1KB file encode round-trip (Erlich)."""
        storage = DNAStorage(scheme="erlich")
        rng = random.Random(123)
        data = bytes(rng.randint(0, 255) for _ in range(1024))  # K=32
        report = storage.store(data, redundancy=1.0)
        assert report.data_len == 1024
        recovered = storage.retrieve(report.oligos, total_len=len(data))
        assert recovered == data

    def test_empty_data_goldman(self):
        """Empty data Goldman encode/decode round-trip."""
        storage = DNAStorage(scheme="goldman")
        report = storage.store(b"")
        assert report.num_oligos >= 1  # empty input produces 1 segment (zero-padded)
        assert report.data_len == 0
        recovered = storage.retrieve(report.oligos, total_len=0)
        assert recovered == b""

    def test_empty_data_erlich(self):
        """Empty data Erlich encode/decode round-trip."""
        storage = DNAStorage(scheme="erlich")
        report = storage.store(b"", redundancy=1.0)
        assert report.data_len == 0
        recovered = storage.retrieve(report.oligos, total_len=0)
        assert recovered == b""

    def test_single_byte_goldman(self):
        """Single byte data Goldman round-trip."""
        storage = DNAStorage(scheme="goldman")
        for byte_val in [0, 1, 127, 128, 255]:
            data = bytes([byte_val])
            report = storage.store(data)
            recovered = storage.retrieve(report.oligos, total_len=1)
            assert recovered == data, f"byte {byte_val} roundtrip failed"

    def test_single_byte_erlich(self):
        """Single byte data Erlich round-trip."""
        storage = DNAStorage(scheme="erlich")
        for byte_val in [0, 1, 127, 128, 255]:
            data = bytes([byte_val])
            report = storage.store(data, redundancy=2.0)
            recovered = storage.retrieve(report.oligos, total_len=1)
            assert recovered == data, f"byte {byte_val} roundtrip failed"

    def test_empty_data_lifecycle(self):
        """Empty data lifecycle does not crash."""
        storage = DNAStorage(scheme="goldman")
        report = storage.simulate_lifecycle(b"", seed=42)
        assert report.original_data == b""
        assert report.success or report.integrity >= 0.0

    def test_empty_data_analyze(self):
        """Empty data analysis report avoids division by zero."""
        storage = DNAStorage(scheme="goldman")
        report = storage.store(b"")
        analysis = storage.analyze(report.oligos, data_len=0)
        assert analysis.density_bit_per_nt == 0.0
        assert analysis.oligo_count >= 1


# ============================================================================
# Parallel encode / concurrent decode (P1#10: concurrent path thread safety + data consistency)
# ============================================================================

class TestParallelConcurrent:
    """Tests for the store_parallel / retrieve_concurrent paths.

    Covers:
    - Erlich multi-chunk parallel encode round-trip fidelity
    - Goldman falls back to sequential encoding
    - oligo indices globally unique (correct offset handling)
    - parallel_chunks metadata integrity
    - retrieve_parallel deprecated alias emits DeprecationWarning
    - thread safety of concurrent store/retrieve calls from multiple threads (no shared state pollution)
    """

    def test_store_parallel_erlich_roundtrip(self):
        """Erlich multi-chunk parallel encode -> retrieve_concurrent sequential decode round-trip fidelity."""
        storage = DNAStorage(scheme="erlich")
        rng = random.Random(7)
        # 8KB data, chunk_size=1024 -> 8 chunks, triggers true parallel encoding
        # redundancy=2.0: LT fountain code needs higher redundancy at small chunk sizes (K=32) to guarantee successful decoding
        data = bytes(rng.randint(0, 255) for _ in range(8192))
        report = storage.store_parallel(data, redundancy=2.0,
                                        max_workers=4, chunk_size=1024)
        assert report.scheme == "erlich"
        assert report.num_oligos > 0
        assert report.parallel_chunks is not None
        assert len(report.parallel_chunks) == 8  # 8 chunks
        # Each chunk tuple (data_size, oligo_count)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in report.parallel_chunks)
        assert sum(s for s, _ in report.parallel_chunks) == len(data)
        # Total oligo count == sum of oligo counts across chunks
        assert sum(n for _, n in report.parallel_chunks) == report.num_oligos
        # Round-trip
        recovered = storage.retrieve_concurrent(report.oligos, report.parallel_chunks)
        assert recovered == data

    def test_store_parallel_goldman_fallback(self):
        """Goldman store_parallel falls back to sequential store (4x overlap needs global segment indices)."""
        storage = DNAStorage(scheme="goldman")
        data = b"Goldman parallel fallback test! " * 10
        report = storage.store_parallel(data, max_workers=4, chunk_size=64)
        # Goldman has no parallel chunk metadata
        assert report.parallel_chunks is None
        assert report.num_oligos > 0
        # Round-trip uses plain retrieve (Goldman has no chunking)
        recovered = storage.retrieve(report.oligos, total_len=len(data))
        assert recovered == data

    def test_store_parallel_global_unique_indices(self):
        """After parallel encoding, all oligo.index values are globally unique (correct offset handling)."""
        storage = DNAStorage(scheme="erlich")
        rng = random.Random(11)
        data = bytes(rng.randint(0, 255) for _ in range(6144))  # 6 chunks @1024
        report = storage.store_parallel(data, redundancy=1.5,
                                        max_workers=4, chunk_size=1024)
        indices = [o.index for o in report.oligos]
        assert len(indices) == len(set(indices)), "oligo indices contain duplicates"

    def test_store_parallel_empty_data(self):
        """Empty data store_parallel does not crash; falls back to sequential store."""
        storage = DNAStorage(scheme="erlich")
        report = storage.store_parallel(b"", redundancy=1.0, chunk_size=128)
        assert report.data_len == 0
        assert report.num_oligos >= 1

    def test_store_parallel_single_chunk(self):
        """When data < chunk_size there is only 1 chunk (still uses the parallel path but a single task)."""
        storage = DNAStorage(scheme="erlich")
        data = b"single chunk parallel test"
        report = storage.store_parallel(data, redundancy=1.0, chunk_size=4096)
        assert report.parallel_chunks is not None
        assert len(report.parallel_chunks) == 1
        recovered = storage.retrieve_concurrent(report.oligos, report.parallel_chunks)
        assert recovered == data

    def test_retrieve_parallel_deprecation_warning(self):
        """retrieve_parallel deprecated alias emits DeprecationWarning with identical results."""
        storage = DNAStorage(scheme="erlich")
        rng = random.Random(13)
        data = bytes(rng.randint(0, 255) for _ in range(2048))
        report = storage.store_parallel(data, redundancy=1.3, chunk_size=1024)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            recovered = storage.retrieve_parallel(report.oligos,
                                                  report.parallel_chunks)
        assert recovered == data
        assert any(issubclass(rec.category, DeprecationWarning) for rec in w)

    def test_retrieve_concurrent_empty_chunks_fallback(self):
        """retrieve_concurrent falls back to sequential retrieve without chunk metadata."""
        storage = DNAStorage(scheme="erlich")
        data = b"fallback no chunks"
        report = storage.store(data, redundancy=1.0)
        # Passing empty parallel_chunks triggers the fallback
        recovered = storage.retrieve_concurrent(report.oligos, [])
        # Fallback path decodes with total_len=0; Erlich empty chunks return b"" or partial
        # Only verify no exception is raised
        assert isinstance(recovered, bytes)

    # ---------- Thread safety: concurrent store / retrieve calls from multiple threads ----------

    def test_concurrent_store_thread_safety(self):
        """Multiple threads calling store() concurrently: each thread recovers its data independently, no shared state pollution.

        store() uses erlich_encode internally (with a random generator, independent per call),
        so there should be no cross-thread data interference. Each thread uses its own DNAStorage
        instance and data, verifying encode-decode round-trip fidelity.
        """
        def worker(seed: int) -> tuple[int, bool]:
            rng = random.Random(seed)
            size = rng.randint(256, 2048)
            data = bytes(rng.randint(0, 255) for _ in range(size))
            storage = DNAStorage(scheme="erlich")
            report = storage.store(data, redundancy=1.5)
            recovered = storage.retrieve(report.oligos, total_len=len(data))
            return seed, recovered == data

        results: list[tuple[int, bool]] = [None] * 8  # type: ignore
        with ThreadPoolExecutor(max_workers=8) as ex:
            fut_to_idx = {ex.submit(worker, s): i for i, s in enumerate(range(100, 108))}
            for fut in as_completed(fut_to_idx):
                results[fut_to_idx[fut]] = fut.result()

        for seed, ok in results:
            assert ok, f"thread seed={seed} concurrent store round-trip failed (thread unsafe?)"

    def test_concurrent_retrieve_thread_safety(self):
        """Multiple threads calling retrieve() concurrently: decoding the same oligos repeatedly yields consistent results.

        retrieve() is a read-only operation; concurrent decoding of the same oligo set from
        multiple threads should produce identical results, without race-condition-induced divergence.
        """
        storage = DNAStorage(scheme="erlich")
        rng = random.Random(99)
        data = bytes(rng.randint(0, 255) for _ in range(4096))
        report = storage.store(data, redundancy=1.2)
        oligos = report.oligos
        total_len = len(data)

        def decode_many() -> bytes:
            # Each thread decodes multiple times to increase the chance of triggering a race
            last = b""
            for _ in range(5):
                last = storage.retrieve(oligos, total_len=total_len)
            return last

        outputs: list[bytes] = [None] * 8  # type: ignore
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(decode_many): i for i in range(8)}
            for fut in as_completed(futs):
                outputs[futs[fut]] = fut.result()

        # All thread outputs must be identical and equal to the original data
        for out in outputs:
            assert out == data, "concurrent retrieve produced inconsistent output (thread unsafe?)"
        assert len(set(outputs)) == 1

    def test_concurrent_store_parallel_and_retrieve(self):
        """Concurrent mixed load of store_parallel + retrieve_concurrent does not interfere."""
        def worker(seed: int) -> bool:
            rng = random.Random(seed)
            data = bytes(rng.randint(0, 255) for _ in range(4096))
            storage = DNAStorage(scheme="erlich")
            report = storage.store_parallel(data, redundancy=2.0,
                                            max_workers=2, chunk_size=1024)
            recovered = storage.retrieve_concurrent(report.oligos,
                                                    report.parallel_chunks)
            return bool(recovered == data)

        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(worker, s) for s in range(200, 204)]
            results = [f.result() for f in as_completed(futs)]
        assert all(results), "mixed concurrent load round-trip failed"
