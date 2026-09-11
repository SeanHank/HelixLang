"""Coverage-closure tests for :mod:`helixlang.plugins.apps.dna_storage`.

Exercises the codec benchmark pipeline (fountain/Goldman/block-RS tuning and
erasure/error tolerance binary searches), the benchmark table renderer, the
Erlich ``simulate_lifecycle`` branch, and the retrieve-error fallback in
``simulate_lifecycle``.
"""
import pytest

pytest.importorskip("Bio")
pytest.importorskip("reedsolo")

from helixlang.core.errors import BioError  # noqa: E402
from helixlang.plugins.apps.dna_storage import (  # noqa: E402
    DNAStorage,
    _benchmark_fountain,
    _benchmark_goldman,
    _benchmark_rs,
    _fountain_error_max,
    _fountain_loss_max,
    _goldman_error_max,
    _goldman_loss_max,
    _rs_blocks,
    _rs_decode_ok,
    _rs_error_max,
    _rs_loss_max,
    benchmark_codecs,
    format_benchmark_table,
    parse_fasta,
)


class TestCodecBenchmarks:
    def test_goldman_row(self):
        data = bytes(range(200)) * 2
        row = _benchmark_goldman(data, seed=1)
        assert row.scheme == "goldman"
        assert row.target_density is None
        assert row.max_loss_fraction >= 0.0
        assert row.max_error_rate >= 0.0

    def test_fountain_rows(self):
        data = bytes(range(256)) * 2  # 512 bytes -> K=16, redundancy tunable
        row = _benchmark_fountain(data, target_density=1.0, seed=2)
        assert row.scheme == "fountain"
        assert row.redundancy is not None

    def test_rs_rows(self):
        data = bytes(range(256)) * 2
        encoded, codec, payload = _rs_blocks(data, block_len=32, nsym=4)
        row = _benchmark_rs(data, target_density=1.0, seed=4)
        assert row.scheme == "rs"
        loss = _rs_loss_max(encoded, codec, payload, 32, seed=5)
        err = _rs_error_max(encoded, codec, payload, 32, seed=5)
        assert loss >= 0.0
        assert err >= 0.0

    def test_loss_error_max_helpers(self):
        spec_erlich = DNAStorage("erlich")
        data = bytes(range(256)) * 2
        rep = spec_erlich.store(data, redundancy=2.0)
        assert _fountain_loss_max(rep.oligos, data, seed=1) >= 0.0
        assert _fountain_error_max(rep.oligos, data, seed=2) >= 0.0

        spec = DNAStorage("goldman")
        repg = spec.store(data)
        assert _goldman_loss_max(repg.oligos, data, seed=3) >= 0.0
        assert _goldman_error_max(repg.oligos, data, seed=4) >= 0.0

    def test_benchmark_codecs_custom_schemes(self):
        data = bytes(range(256)) * 2
        rows = benchmark_codecs(data=data, densities=(0.5,),
                                schemes=("goldman", "fountain", "rs"),
                                seed=6)
        assert {r.scheme for r in rows} == {"goldman", "fountain", "rs"}
        # goldman is native (1 row) + fountain/rs one per density
        assert len(rows) == 1 + 1 + 1

    def test_benchmark_codecs_default_data(self):
        rows = benchmark_codecs(data_size=512, densities=(0.5, 1.0),
                                schemes=("goldman",), seed=8)
        assert rows and rows[0].scheme == "goldman"

    def test_format_table(self):
        rows = benchmark_codecs(data=bytes(range(128)), densities=(0.5,),
                                schemes=("goldman", "fountain"), seed=9)
        text = format_benchmark_table(rows)
        assert text.startswith("scheme")
        assert "goldman" in text and "fountain" in text


class TestLifecycleErlichBranch:
    def test_erlich_lifecycle_reports_errors(self):
        storage = DNAStorage("erlich")
        data = bytes(range(256)) * 2
        report = storage.simulate_lifecycle(
            data, synthesis_quality="typical", pcr_cycles=8,
            polymerase="taq", sequencing_platform="illumina_hiseq_novaseq",
            storage_years=100, storage_temp=37.0, seed=1)
        assert report.decay_damage >= 0
        assert isinstance(report.recovered_data, bytes)
        assert report.original_data == data

    def test_lifecycle_unexpected_decode_error_falls_back(self, monkeypatch):
        storage = DNAStorage("erlich")
        data = bytes(range(64)) * 2

        class _Unexpected(Exception):
            pass

        def _boom(*args, **kwargs):
            raise _Unexpected("third-party decode failure")

        monkeypatch.setattr(storage, "retrieve", _boom)
        report = storage.simulate_lifecycle(data, synthesis_quality="perfect", seed=1)
        assert report.recovered_data == b""
        assert report.success is False
        assert report.integrity == 0.0


class TestParseFastaUnknownScheme:
    def test_unknown_scheme_oligos_discarded(self):
        oligos = parse_fasta(">foo_0\nACGTACGT\n>bar_1\nTGCA\n")
        assert oligos == []

    def test_blank_lines_and_whitespace_ignored(self):
        oligos = parse_fasta("\n  \n>goldman_0\n\nACGT\n")
        assert len(oligos) == 1
        assert oligos[0].full == "ACGT"


class TestReedSoloAvailability:
    def test_reload_without_reedsolo_falls_back(self, monkeypatch):
        import builtins
        import importlib

        import helixlang.plugins.apps.dna_storage as ds

        real_import = builtins.__import__

        def _block_reedsolo(name, *args, **kwargs):
            if name.split(".")[0] == "reedsolo":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_reedsolo)
        try:
            importlib.reload(ds)
            assert ds._HAS_REEDSOLO_BENCH is False
            assert ds._RSCodec is None
            assert ds._RealRSEncoderError is ValueError
        finally:
            builtins.__import__ = real_import
            importlib.reload(ds)
            assert ds._HAS_REEDSOLO_BENCH is True

    def test_benchmark_rs_requires_reedsolo(self, monkeypatch):
        import helixlang.plugins.apps.dna_storage as ds

        monkeypatch.setattr(ds, "_HAS_REEDSOLO_BENCH", False)
        with pytest.raises(BioError, match="reedsolo"):
            benchmark_codecs(data=b"x" * 16, schemes=("rs",))


class TestRsDecodeZeroFraction:
    def test_drop_false_zero_fraction_skips_mutation(self):
        data = bytes(range(256)) * 2
        encoded, codec, payload = _rs_blocks(data, block_len=32, nsym=4)
        ok = _rs_decode_ok(encoded, codec, payload, fraction=0.0,
                           drop=False, block_len=32, seed=1)
        assert ok is True
