"""Drive the last uncovered lines/branches in ``helixlang.cli``,
``helixlang.server.app`` and ``helixlang.__main__`` to 100% line+branch
coverage.

The regular end-to-end suite already covers the happy paths; these tests
target the parser/error/summary/reporting branches and server request
handlers that it does not reach, using surgical fakes/mocks only.  No
production code is modified, and no `# pragma: no cover` or pytest-cov
exclusions are used.
"""
import importlib
import runpy
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import flask
import pytest

import helixlang.cli as cli
from helixlang.core import hxbc
from helixlang.core.errors import (
    CompileError,
    LexError,
    RuntimeHelixError,
    SemanticError,
    SimConfigError,
)
from helixlang.sim_runtime._types import SimResult

srv = importlib.import_module("helixlang.server.app")

SOURCE = Path(__file__).resolve().parent.parent / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

HELLO = "#gene name=hello\nATG GCT TAA\n#end\n#config ticks=2\n"
HELLO_SIM = (
    "#gene name=hello\nATG GCT TAA\n#end\n"
    "#config ticks=2 backend=whole_cell\n"
)

_TRACE = [
    {"tick": 0, "x": 0, "y": 0, "energy": 1000000000.0, "alive": True,
     "proteins": {"p": 1.0}, "morphology_points_count": 1, "field_total_v": 0.0},
    {"tick": 1, "x": 0, "y": 0, "energy": 1000000000.0, "alive": True,
     "proteins": {"p": 2.0}, "morphology_points_count": 1, "field_total_v": 0.0},
]


class _FakeField:
    n = 2
    v = [[0.5, 0.25], [1.0, 0.0]]

    def total_v(self):
        return 9.5


# ===========================================================================
# helixlang/__main__.py
# ===========================================================================

def test_main_py_runpy_help(monkeypatch):
    monkeypatch.setenv("HELIX_BENCHMARK_OFFLINE", "1")
    monkeypatch.setattr(sys, "argv", ["helixlang", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_module("helixlang", run_name="__main__")


def test_main_py_import_guard():
    import importlib
    with mock.patch.object(cli, "main") as m:
        mod = importlib.import_module("helixlang.__main__")
    assert mod.__name__ == "helixlang.__main__"
    m.assert_not_called()


def test_main_py_subprocess_help():
    env = dict(__import__("os").environ)
    env["HELIX_BENCHMARK_OFFLINE"] = "1"
    out = subprocess.run(
        [sys.executable, "-m", "helixlang", "--help"],
        capture_output=True, text=True, env=env,
    )
    assert out.returncode == 0


# ===========================================================================
# helixlang.cli  --  argument parsing / DNA codec
# ===========================================================================

def test_missing_source_returns_2():
    with mock.patch.object(
        cli.argparse.ArgumentParser, "error",
        side_effect=lambda *a, **k: None,
    ) as m:
        rc = cli.main([])
    assert rc == 2
    assert m.call_count == 1


def test_decode_dna_fasta_success(tmp_path):
    dna = importlib.import_module("helixlang.plugins.runtime.dna_codec")
    src = tmp_path / "placeholder.helix"
    src.write_text(HELLO)
    fasta = tmp_path / "dna.fasta"
    fasta.write_text(">oligo_0\nAAA\n>oligo_1\nCCC\n")
    with mock.patch.object(dna, "dna_to_helix", return_value="// decoded //") as m:
        rc = cli.main([str(src), "--decode-dna", str(fasta)])
    assert rc == 0
    m.assert_called_once()


def test_decode_dna_fasta_last_empty(tmp_path, capsys):
    dna = importlib.import_module("helixlang.plugins.runtime.dna_codec")
    src = tmp_path / "placeholder.helix"
    src.write_text(HELLO)
    fasta = tmp_path / "dna2.fasta"
    fasta.write_text(">only\n")
    with mock.patch.object(
        dna, "dna_to_helix", side_effect=ValueError("bad dna"),
    ):
        rc = cli.main([str(src), "--decode-dna", str(fasta)])
    assert rc == 1
    assert "decode error" in capsys.readouterr().err


def test_encode_dna_fallthrough_with_fake_flag():
    """Cover the falsy ``args.decode_dna`` branch after the encode block."""
    real_parse = cli.argparse.ArgumentParser.parse_args

    class _Flip:
        def __init__(self):
            self.calls = 0

        def __bool__(self):
            self.calls += 1
            return self.calls == 1

    flip = _Flip()

    def fake_parse(parser, args=None, namespace=None):
        ns = real_parse(parser, args, namespace)
        ns.encode_dna = flip
        ns.decode_dna = None
        ns.source = SimpleNamespace(exists=lambda: False)
        return ns

    with mock.patch.object(
        cli.argparse.ArgumentParser, "parse_args",
        autospec=True, side_effect=fake_parse,
    ):
        rc = cli.main([])
    assert rc == 2
    assert flip.calls == 2


# ===========================================================================
# helixlang.cli  --  flag-driven paths with the real parser
# ===========================================================================

def test_disassemble_plain_source(tmp_path, capsys):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)
    rc = cli.main([str(src), "--disassemble"])
    assert rc == 0
    assert "hello" in capsys.readouterr().out


def test_skip_validity(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)
    assert cli.main([str(src), "--skip-validity"]) == 0


def test_gem_dynamic_flag(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)
    result = SimResult("gem", ["tick"], [{"tick": 0}])
    with mock.patch.object(cli, "run", return_value=result):
        rc = cli.main([str(src), "--gem", "--dynamic", "--duration", "5",
                       "--dt", "0.2", "--json"])
    assert rc == 0


def test_sim_json_csv_none_error(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)
    result = SimResult("whole_cell", ["tick"], [{"tick": 0}])
    with mock.patch.object(cli, "run", return_value=result):
        assert cli.main([str(src), "--backend", "whole_cell", "--json"]) == 0
    with mock.patch.object(cli, "run", return_value=result):
        assert cli.main([str(src), "--backend", "whole_cell", "--csv"]) == 0
    with mock.patch.object(cli, "run", return_value=None):
        assert cli.main([str(src), "--backend", "whole_cell"]) == 0
    with mock.patch.object(cli, "run", side_effect=SimConfigError("no sim")):
        assert cli.main([str(src), "--backend", "whole_cell"]) == 1


# ===========================================================================
# helixlang.cli  --  compile / runtime error branches + IR summaries
# ===========================================================================

def test_compile_ir_error(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)
    with mock.patch.object(cli.Compiler, "compile_ir",
                           side_effect=CompileError("ir fail")):
        rc = cli.main([str(src)])
    assert rc == 1


def test_dump_ir_error(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)
    with mock.patch.object(cli, "ir_dumps", side_effect=TypeError("dump fail")):
        rc = cli.main([str(src), "--dump-ir", str(tmp_path / "ir.json")])
    assert rc == 1


def test_runtime_error_classic(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)

    class _BadVm:
        def __init__(self, *a, **k):
            self.debug = False

        def run(self, ticks):
            raise RuntimeHelixError("vm boom")

    with mock.patch.object(cli, "CellVM", _BadVm):
        rc = cli.main([str(src)])
    assert rc == 1


def test_runtime_error_ir(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)

    class _BadIr:
        def __init__(self, *a, **k):
            self.debug = False

        def run(self, ticks):
            raise RuntimeHelixError("ir boom")

    with mock.patch.object(cli, "IRRuntime", _BadIr):
        rc = cli.main([str(src), "--runtime", "ir"])
    assert rc == 1


def test_runtime_error_batch(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)

    class _BadBatch:
        def __init__(self, *a, **k):
            pass

        def run(self, ticks):
            raise RuntimeHelixError("batch boom")

    with mock.patch.object(cli, "BatchRuntime", _BadBatch):
        rc = cli.main([str(src), "--runtime", "batch"])
    assert rc == 1


def test_batch_many_cells(tmp_path, capsys):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)
    rc = cli.main([str(src), "--runtime", "batch", "--batch-n", "12"])
    assert rc == 0
    assert "8 more cells" in capsys.readouterr().out


def test_classic_summary_empty_morph_field(tmp_path, capsys):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)

    class _FakeVm:
        def __init__(self, *a, **k):
            self.debug = False
            self.cell = SimpleNamespace(morphology_points=[])
            self.field = _FakeField()

        def run(self, ticks):
            return _TRACE

    with mock.patch.object(cli, "CellVM", _FakeVm):
        rc = cli.main([str(src)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "field total V: 9.500" in out
    assert "morphology points" not in out


def test_ir_summary_csv_png(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)

    class _FakeIr:
        def __init__(self, *a, **k):
            self.debug = False
            self.cell = SimpleNamespace(morphology_points=[(1.0, 1.0)])
            self.field = None

        def run(self, ticks):
            return _TRACE

    with mock.patch.object(cli, "IRRuntime", _FakeIr):
        rc = cli.main([str(src), "--runtime", "ir", "--csv",
                       "--png", str(tmp_path / "ir")])
    assert rc == 0
    assert (tmp_path / "ir.ppm").exists()


def test_ir_summary_empty_morph_field(tmp_path, capsys):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)

    class _FakeIr:
        def __init__(self, *a, **k):
            self.debug = False
            self.cell = SimpleNamespace(morphology_points=[])
            self.field = _FakeField()

        def run(self, ticks):
            return _TRACE

    with mock.patch.object(cli, "IRRuntime", _FakeIr):
        rc = cli.main([str(src), "--runtime", "ir"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "field total V: 9.500" in out
    assert "morphology points" not in out


# ===========================================================================
# helixlang.cli  --  full pipeline (mocked plugin module)
# ===========================================================================

def test_full_pipeline_full_and_minimal(tmp_path):
    fp = importlib.import_module("helixlang.plugins.apps.full_pipeline")
    fasta = tmp_path / "genome.fasta"
    fasta.write_text(">chr1\nACGTACGT\n")

    class _Ecgem:
        growth_rate = 0.31
        growth_rate_unconstrained = 0.7

    class _Community:
        total_biomass = 3.5
        converged = True

    class _Result:
        def __init__(self, full):
            self.stages_completed = ["genome", "annotation"] if full else []
            self.proteins = [1] if full else []
            self.structures = [1] if full else []
            self.kcat_predictions = [1] if full else []
            self.ecgem = _Ecgem() if full else None
            self.community = _Community() if full else None
            self.pipeline_time = 1.5
            self.warnings = ["w1", "w2"] if full else []

    with mock.patch.object(fp, "PipelineConfig", SimpleNamespace), \
            mock.patch.object(fp, "run_full_pipeline",
                              return_value=_Result(True)):
        assert cli.main([str(fasta), "--full-pipeline"]) == 0
    with mock.patch.object(fp, "PipelineConfig", SimpleNamespace), \
            mock.patch.object(fp, "run_full_pipeline",
                              return_value=_Result(False)):
        assert cli.main([str(fasta), "--full-pipeline"]) == 0


# ===========================================================================
# helixlang.cli  --  watch loop error branches
# ===========================================================================

def test_watch_compile_error(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)
    real_lexer = cli.Lexer
    state = {"n": 0}

    class _FlipLexer:
        def __init__(self, src_text):
            state["n"] += 1
            if state["n"] >= 3:
                raise LexError("flip lex failure")
            self._inner = real_lexer(src_text)

        def tokens(self):
            return self._inner.tokens()

    with mock.patch.object(cli, "Lexer", _FlipLexer):
        rc = cli.main([str(src), "--watch", "--watch-iterations", "2",
                       "--watch-interval", "0.005"])
    assert rc == 0
    assert state["n"] >= 3


def test_watch_read_error(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)
    real_read = Path.read_text
    state = {"n": 0}

    def flaky_read(self, *a, **k):
        if self == src:
            state["n"] += 1
            if state["n"] == 3:
                raise OSError("gone")
        return real_read(self, *a, **k)

    with mock.patch.object(Path, "read_text", flaky_read):
        rc = cli.main([str(src), "--watch", "--watch-iterations", "2",
                       "--watch-interval", "0.005"])
    assert rc == 0
    assert state["n"] >= 3


def test_watch_runtime_error(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)

    class _BadVm:
        def __init__(self, *a, **k):
            self.debug = False

        def run(self, ticks):
            raise RuntimeHelixError("watch boom")

    with mock.patch.object(cli, "CellVM", _BadVm):
        rc = cli.main([str(src), "--watch", "--watch-iterations", "1",
                       "--watch-interval", "0.005"])
    assert rc == 0


def test_watch_keyboard_interrupt(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)
    real_sleep = time.sleep

    def raiser(*a, **k):
        raise KeyboardInterrupt()

    time.sleep = raiser
    try:
        rc = cli.main([str(src), "--watch", "--watch-interval", "0.005"])
    finally:
        time.sleep = real_sleep
    assert rc == 0


# ===========================================================================
# helixlang.cli  --  standalone PPM emission branches
# ===========================================================================

def test_emit_ppm_empty_points(tmp_path):
    vm = SimpleNamespace(cell=SimpleNamespace(morphology_points=[]),
                         field=None)
    cli._emit_ppm(vm, str(tmp_path / "empty"))
    assert not (tmp_path / "empty.ppm").exists()


def test_emit_ppm_field(tmp_path):
    vm = SimpleNamespace(cell=SimpleNamespace(morphology_points=[]),
                         field=_FakeField())
    cli._emit_ppm(vm, str(tmp_path / "field"))
    assert (tmp_path / "field.ppm").exists()


def test_emit_ppm_out_of_bounds(tmp_path):
    vm = SimpleNamespace(cell=SimpleNamespace(morphology_points=[(7.0, 7.0)]),
                         field=None)
    with mock.patch("builtins.min", return_value=0), \
            mock.patch("builtins.max", return_value=1):
        cli._emit_ppm(vm, str(tmp_path / "off"))
    assert (tmp_path / "off.ppm").exists()


# ===========================================================================
# helixlang.cli  --  artifact (.helixc) modes
# ===========================================================================

def _compile_artifact(tmp_path, name="hello.helix", out="out.helixc",
                      src_text=None, flags=()):
    src = tmp_path / name
    src.write_text(src_text or HELLO)
    art = tmp_path / out
    rc = cli.main([str(src), "--compile", "-o", str(art), *flags])
    assert rc == 0
    return src, art


def test_artifact_disassemble(tmp_path, capsys):
    _, art = _compile_artifact(tmp_path)
    assert cli.main([str(art), "--disassemble"]) == 0
    assert capsys.readouterr().out


def test_artifact_table_override_success(tmp_path):
    _, art = _compile_artifact(tmp_path)
    assert cli.main([str(art), "--table", "mito_vertebrate"]) == 0


def test_artifact_no_chunk_success(tmp_path):
    _, art = _compile_artifact(tmp_path, flags=["--no-chunk"])
    assert cli.main([str(art)]) == 0


def test_sim_table_default(tmp_path, capsys):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)
    result = SimResult("whole_cell", ["tick"], [{"tick": 0}])
    with mock.patch.object(cli, "run", return_value=result):
        assert cli.main([str(src), "--backend", "whole_cell"]) == 0
    assert "backend=whole_cell rows=1" in capsys.readouterr().out


def test_artifact_run_binary_error(tmp_path):
    bad = tmp_path / "bad.helixc"
    bad.write_bytes(b"garbage")
    assert cli.main([str(bad)]) == 1


def test_artifact_run_semantic_error(tmp_path):
    _, art = _compile_artifact(tmp_path)

    class _BadSem:
        def __init__(self, *a, **k):
            pass

        def check(self):
            raise SemanticError("semantic fail")

    with mock.patch.object(cli, "SemanticAnalyzer", _BadSem):
        assert cli.main([str(art)]) == 1


def test_artifact_stale_ticks_fake_vm(tmp_path, capsys):
    _, art = _compile_artifact(tmp_path)
    real = hxbc.load_program(str(art))
    fake_art = SimpleNamespace(
        program=real.program, chunk_stale=True, chunk=real.chunk,
        source=real.source, table=real.table,
    )

    class _FakeVm:
        def __init__(self, *a, **k):
            self.debug = False
            self.cell = SimpleNamespace(morphology_points=[])
            self.field = _FakeField()

        def run(self, ticks):
            return _TRACE

    with mock.patch.object(cli.hxbc, "load_program", return_value=fake_art), \
            mock.patch.object(cli, "CellVM", _FakeVm):
        rc = cli.main([str(art), "--ticks", "3"])
    out = capsys.readouterr()
    assert rc == 0
    assert "stale compiled chunk" in out.err
    assert "field total V: 9.500" in out.out


def test_artifact_runtime_error(tmp_path):
    _, art = _compile_artifact(tmp_path)

    class _BadVm:
        def __init__(self, *a, **k):
            self.debug = False

        def run(self, ticks):
            raise RuntimeHelixError("artifact boom")

    with mock.patch.object(cli, "CellVM", _BadVm):
        assert cli.main([str(art)]) == 1


def test_artifact_png(tmp_path):
    _, art = _compile_artifact(tmp_path)
    assert cli.main([str(art), "--png", str(tmp_path / "morph")]) == 0
    assert (tmp_path / "morph.ppm").exists()


def test_artifact_sim_backend(tmp_path):
    _, art = _compile_artifact(tmp_path, src_text=HELLO_SIM)
    result = SimResult("whole_cell", ["tick"], [{"tick": 0}])
    with mock.patch.object(cli, "run", return_value=result):
        assert cli.main([str(art), "--json"]) == 0


def test_artifact_table_override_compile_error(tmp_path):
    _, art = _compile_artifact(tmp_path)

    class _BadCompile:
        def __init__(self, *a, **k):
            pass

        def compile(self, program):
            raise CompileError("table recompile fail")

    with mock.patch.object(cli, "Compiler", _BadCompile):
        assert cli.main([str(art), "--table", "mito_vertebrate"]) == 1


def test_artifact_no_chunk_compile_error(tmp_path):
    _, art = _compile_artifact(tmp_path, flags=["--no-chunk"])

    class _BadCompile:
        def __init__(self, *a, **k):
            pass

        def compile(self, program):
            raise CompileError("no-chunk recompile fail")

    with mock.patch.object(cli, "Compiler", _BadCompile):
        assert cli.main([str(art)]) == 1


def test_artifact_mode_compile_error(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)
    with mock.patch.object(cli.hxbc, "compile_file",
                           side_effect=hxbc.BinaryError("compile boom")):
        assert cli.main([str(src), "--compile", "-o",
                         str(tmp_path / "x.helixc")]) == 1


def test_artifact_decompile_embedded_and_no_source(tmp_path):
    _, art = _compile_artifact(tmp_path)
    out = tmp_path / "out.helix"
    assert cli.main([str(art), "--decompile", "-o", str(out)]) == 0
    assert out.read_text() == HELLO

    _, art2 = _compile_artifact(tmp_path, name="hello2.helix",
                                out="out2.helixc", flags=["--no-source"])
    out2 = tmp_path / "out2.helix"
    assert cli.main([str(art2), "--decompile", "-o", str(out2)]) == 0
    assert out2.read_text().startswith("#gene")


def test_artifact_decompile_binary_error(tmp_path):
    bad = tmp_path / "bad.helixc"
    bad.write_bytes(b"garbage")
    assert cli.main([str(bad), "--decompile", "-o",
                     str(tmp_path / "o.helix")]) == 1


def test_compare_ok(tmp_path):
    src, art = _compile_artifact(tmp_path)
    assert cli.main([str(src), "--compare", str(art)]) == 0


def test_compare_artifact_failed(tmp_path):
    src, art = _compile_artifact(tmp_path)
    with mock.patch.object(cli, "_capture_main",
                           side_effect=[(0, "a"), (2, "b")]):
        assert cli.main([str(src), "--compare", str(art)]) == 2


def test_compare_binary_error(tmp_path):
    src = tmp_path / "hello.helix"
    src.write_text(HELLO)
    bad = tmp_path / "bad.helixc"
    bad.write_bytes(b"garbage")
    assert cli.main([str(src), "--compare", str(bad)]) == 1


# ===========================================================================
# helixlang.server.app  --  helpers / caching / examples
# ===========================================================================

def test_estimate_bytes_errors():
    assert srv._estimate_bytes(SimpleNamespace(),
                               SimpleNamespace(code=None)) == 16


def test_pipeline_cache_eviction():
    srv._PIPELINE_CACHE.clear()
    src_a = "#gene name=cache_a\nATG GCT TAA\n#end\n"
    src_b = "#gene name=cache_b\nATG GCT TAA\n#end\n"
    with mock.patch.object(srv, "_CACHE_MEM_SOFT_LIMIT", 0):
        srv._pipeline_cached(src_a, "standard")
        srv._pipeline_cached(src_b, "standard")
    assert len(srv._PIPELINE_CACHE) == 1
    srv._PIPELINE_CACHE.clear()


def test_get_debug_lock(monkeypatch):
    monkeypatch.setattr(srv, "_DEBUG_LOCK", None)
    lock1 = srv._get_debug_lock()
    lock2 = srv._get_debug_lock()
    assert lock1 is lock2


def test_examples_dir_missing(client, tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_EXAMPLES_DIR", tmp_path / "no_such")
    r = client.get("/api/examples")
    assert r.status_code == 200
    assert r.get_json() == {"examples": []}


def test_get_example_invalid(client):
    r = client.get("/api/examples/foo")
    assert r.status_code == 400
    assert r.get_json()["error"] == "invalid example name"


def test_compile_unknown_table(client):
    r = client.post("/api/compile", json={"source": HELLO, "table": "bogus"})
    assert r.status_code == 400
    assert "unknown table" in r.get_json()["error"]


def test_api_run_unknown_table(client):
    r = client.post("/api/run", json={"source": HELLO, "table": "bogus"})
    assert r.status_code == 400
    assert "unknown table" in r.get_json()["error"]


def test_sim_run_unknown_table(client):
    r = client.post("/api/sim/run", json={"source": HELLO, "table": "bogus"})
    assert r.status_code == 400


def test_sim_run_helix_error(client):
    r = client.post("/api/sim/run", json={"source": "ATG GC"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_sim_run_none_with_ticks(client):
    with mock.patch.object(srv, "run", return_value=None):
        r = client.post("/api/sim/run", json={
            "source": HELLO, "ticks": "7"})
    assert r.status_code == 200
    assert r.get_json() == {"ok": True, "backend": "classic", "sim": None}


def test_epigenetics_histone_dict_ecoli(client):
    r = client.post("/api/epigenetics/histone", json={
        "dna": "ATGC" * 40,
        "gene_positions": {"g1": {"hp1": 1, "hp2": 2}},
        "cell_type": "ecoli",
    })
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_dna_storage_goldman_analyze_retrieve(client):
    r = client.post("/api/dna-storage/store", json={
        "text": "hello world", "scheme": "goldman"})
    assert r.status_code == 200
    oligos = r.get_json()["oligos"]
    assert oligos
    body = {"oligos": [
        {"index": o["index"], "payload": o["payload"],
         "overhang": o["overhang"], "full": o["full"]} for o in oligos],
        "scheme": "goldman", "data_len": 11}
    assert client.post("/api/dna-storage/analyze", json=body).status_code == 200
    assert client.post("/api/dna-storage/retrieve", json=body).status_code == 200


def test_fba_zero_uptake(client):
    r = client.post("/api/metabolism/fba", json={"glc_uptake": 0})
    assert r.status_code == 200


def test_unexpected_error_500(client):
    r = client.post("/api/evolution/run", json={"generations": "abc"})
    assert r.status_code == 500
    assert r.get_json()["ok"] is False


# ===========================================================================
# helixlang.server.app  --  debugger API
# ===========================================================================

def test_debug_bad_session_ids(client):
    sid = "does-not-exist"
    for path in ("/api/debug/step", "/api/debug/step-over",
                 "/api/debug/step-out", "/api/debug/continue"):
        r = client.post(path, json={"session_id": sid})
        assert r.status_code == 400


def test_debug_halted_step(client):
    r = client.post("/api/debug/session", json={"source": HELLO})
    assert r.status_code == 200
    sid = r.get_json()["session_id"]
    r = client.post("/api/debug/continue", json={"session_id": sid})
    assert r.get_json()["halted"] is True
    r = client.post("/api/debug/step", json={"session_id": sid})
    assert r.status_code == 200
    assert r.get_json() == {"ok": True, "state": None, "halted": True}


def test_debug_breakpoint_remove(client):
    r = client.post("/api/debug/session", json={"source": HELLO})
    sid = r.get_json()["session_id"]
    r = client.post("/api/debug/breakpoint", json={
        "session_id": sid, "action": "remove", "offset": 0})
    assert r.status_code == 200
    for off in (0, 1):
        r = client.post("/api/debug/breakpoint", json={
            "session_id": sid, "action": "set", "offset": off})
        assert r.status_code == 200
    r = client.post("/api/debug/breakpoint", json={
        "session_id": sid, "action": "remove", "offset": 0})
    assert r.status_code == 200
    assert [b["offset"] for b in r.get_json()["breakpoints"]] == [1]


# ===========================================================================
# helixlang.server.app  --  GEM / full-pipeline endpoints
# ===========================================================================

class _FakeGemResult:
    def __init__(self):
        self.stages_completed = 6
        self.annotated_genes = 10
        self.final_reaction_count = 57
        self.grn = SimpleNamespace(total_edges=3)
        self.kcat_predictions = [1, 2]
        self.km_estimates = [1]
        self.warnings = []
        self.errors = []

    def summary(self):
        return "gem summary"


def test_gem_reconstruct(client):
    gp = importlib.import_module("helixlang.plugins.apps.gem_pipeline")
    r = client.post("/api/gem/reconstruct", json={"organism": "x"})
    assert r.status_code == 400
    with mock.patch.object(gp, "run_gem_pipeline",
                           return_value=_FakeGemResult()):
        r = client.post("/api/gem/reconstruct", json={"fasta": "ACGT"})
        assert r.status_code == 200
        assert r.get_json()["stages_completed"] == 6
    with mock.patch.object(gp, "run_gem_pipeline",
                           side_effect=KeyError("kaput")):
        r = client.post("/api/gem/reconstruct", json={"fasta": "ACGT"})
        assert r.status_code == 400


def test_gem_simulate(client):
    gp = importlib.import_module("helixlang.plugins.apps.gem_pipeline")
    r = client.post("/api/gem/simulate", json={})
    assert r.status_code == 400
    with mock.patch.object(gp, "run_gem_pipeline",
                           return_value=_FakeGemResult()):
        r = client.post("/api/gem/simulate", json={"fasta": "ACGT"})
        assert r.status_code == 200
        assert r.get_json()["backend"] == "gem"
        assert len(r.get_json()["rows"]) == 3
    with mock.patch.object(gp, "run_gem_pipeline",
                           side_effect=ValueError("bad")):
        r = client.post("/api/gem/simulate", json={"fasta": "ACGT"})
        assert r.status_code == 400


class _FakePipelineResult:
    def __init__(self, full=True):
        self.stages_completed = "genome|annotation" if full else ""
        self.proteins = [1] if full else []
        self.structures = [1] if full else []
        self.kcat_predictions = [1] if full else []
        self.ecgem = SimpleNamespace(growth_rate=0.4) if full else None
        self.community = SimpleNamespace(total_biomass=2.1) if full else None
        self.pipeline_time = 1.2
        self.warnings = []


def test_full_pipeline_endpoint(client):
    fp = importlib.import_module("helixlang.plugins.apps.full_pipeline")
    r = client.post("/api/full-pipeline", json={})
    assert r.status_code == 400
    with mock.patch.object(fp, "PipelineConfig", SimpleNamespace), \
            mock.patch.object(fp, "run_full_pipeline",
                              return_value=_FakePipelineResult()):
        r = client.post("/api/full-pipeline", json={"fasta": "ACGT"})
        assert r.status_code == 200
        assert r.get_json()["stages_completed"] == "genome|annotation"
    with mock.patch.object(fp, "PipelineConfig", SimpleNamespace), \
            mock.patch.object(fp, "run_full_pipeline",
                              side_effect=RuntimeError("boom")):
        r = client.post("/api/full-pipeline", json={"fasta": "ACGT"})
        assert r.status_code == 400


def test_run_server(tmp_path):
    with mock.patch.object(flask.Flask, "run", return_value=None):
        assert srv.run_server(port=0) == 0
