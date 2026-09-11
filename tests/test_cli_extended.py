"""Extended CLI tests — covers --check-bytecode-version, --info, --runtime ir/batch,
--ir-text, --dump-ir, --compile/--decompile, --compare, --watch, --encode-dna
success path, _cell, _emit_sim_csv/_emit_sim_table.
"""
from __future__ import annotations

import json

import pytest

from helixlang.cli import (
    _cell,
    _emit_csv,
    _emit_sim_csv,
    _emit_sim_table,
    main,
)

HELLO_SRC = "#gene name=hello\nATG GCT TAA\n#end\n#config ticks=2\n"
LSYS_SRC = """\
#promoter name=p strength=-0.5
#gene name=grow promoter=p
ATG CTC TAA
#end
#regulate p -> grow strength=+0.6
#lsystem name=plant axiom=F rules=0:F->F[+F]F[-F]F angle=25
#config ticks=5
"""


@pytest.fixture
def hello_file(tmp_path):
    p = tmp_path / "hello.helix"
    p.write_text(HELLO_SRC)
    return p


@pytest.fixture
def lsys_file(tmp_path):
    p = tmp_path / "lsys.helix"
    p.write_text(LSYS_SRC)
    return p


# ── _cell helper ────────────────────────────────────────────────────────


def test_cell_float():
    assert _cell(3.14159) == "3.142"


def test_cell_dict():
    out = _cell({"a": 1})
    assert json.loads(out) == {"a": 1}


def test_cell_list():
    out = _cell([1, 2])
    assert json.loads(out) == [1, 2]


def test_cell_str():
    assert _cell("hello") == "hello"


def test_cell_int():
    assert _cell(42) == "42"


# ── _emit_sim_csv / _emit_sim_table ─────────────────────────────────────


class _FakeSimResult:
    def __init__(self):
        self.backend = "classic"
        self.columns = ["tick", "x", "y"]
        self.rows = [{"tick": 0, "x": 0, "y": 0},
                     {"tick": 1, "x": 1, "y": 0}]

    def to_dict(self):
        return {"backend": self.backend, "columns": self.columns,
                "rows": self.rows}


def test_emit_sim_csv(capsys):
    _emit_sim_csv(_FakeSimResult())
    out = capsys.readouterr().out
    assert "backend,tick,x,y" in out
    assert "classic,0,0,0" in out


def test_emit_sim_table(capsys):
    _emit_sim_table(_FakeSimResult(), "test.helix")
    out = capsys.readouterr().out
    assert "test.helix" in out
    assert "classic" in out


def test_emit_sim_table_many_rows(capsys):
    r = _FakeSimResult()
    r.rows = [{"tick": i, "x": i, "y": 0} for i in range(50)]
    _emit_sim_table(r, "big.helix")
    out = capsys.readouterr().out
    assert "20 more rows" in out


# ── --check-bytecode-version ────────────────────────────────────────────


def test_check_bytecode_version(capsys):
    rc = main(["--check-bytecode-version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OPCODE_VERSION" in out


# ── --info ──────────────────────────────────────────────────────────────


def test_info(capsys):
    rc = main(["--info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert len(out) > 0


# ── --ir-text ───────────────────────────────────────────────────────────


def test_ir_text(capsys, hello_file):
    rc = main([str(hello_file), "--ir-text"])
    assert rc == 0
    out = capsys.readouterr().out
    assert len(out) > 0


# ── --dump-ir ───────────────────────────────────────────────────────────


def test_dump_ir(capsys, hello_file, tmp_path):
    out_path = tmp_path / "ir.json"
    rc = main([str(hello_file), "--dump-ir", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert isinstance(data, (dict, list))


def test_dump_ir_no_output_returns_2(capsys, hello_file):
    with pytest.raises(SystemExit) as ei:
        main([str(hello_file), "--dump-ir"])
    assert ei.value.code == 2


# ── --runtime ir ────────────────────────────────────────────────────────


def test_runtime_ir(capsys, hello_file):
    rc = main([str(hello_file), "--runtime", "ir"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "tick=" in out or "hello" in out.lower()


def test_runtime_ir_error(capsys, tmp_path):
    bad = tmp_path / "bad.helix"
    bad.write_text("#gene name=g\nATG GCT\n#end\n")
    rc = main([str(bad), "--runtime", "ir"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err.lower()


# ── --runtime batch ─────────────────────────────────────────────────────


def test_runtime_batch(capsys, hello_file):
    rc = main([str(hello_file), "--runtime", "batch", "--batch-n", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "hello" in out.lower() or "tick" in out.lower()


def test_runtime_batch_csv(capsys, hello_file):
    rc = main([str(hello_file), "--runtime", "batch", "--csv"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "tick" in out.lower() or "cell" in out.lower()


def test_runtime_batch_error(capsys, tmp_path):
    bad = tmp_path / "bad.helix"
    bad.write_text("#gene name=g\nATG GCT\n#end\n")
    rc = main([str(bad), "--runtime", "batch"])
    assert rc == 1


# ── --compile / --decompile ────────────────────────────────────────────


def test_compile_and_decompile(capsys, hello_file, tmp_path):
    helixc = tmp_path / "out.helixc"
    rc = main([str(hello_file), "--compile", "-o", str(helixc)])
    assert rc == 0
    assert helixc.exists()

    decomp = tmp_path / "out.helix"
    rc = main([str(helixc), "--decompile", "-o", str(decomp)])
    assert rc == 0
    assert decomp.exists()
    assert len(decomp.read_text()) > 0


def test_compile_no_output_returns_2(capsys, hello_file):
    rc = main([str(hello_file), "--compile"])
    assert rc == 2


def test_decompile_no_output_returns_2(capsys, hello_file, tmp_path):
    helixc = tmp_path / "out.helixc"
    main([str(hello_file), "--compile", "-o", str(helixc)])
    rc = main([str(helixc), "--decompile"])
    assert rc == 2


# ── --compare ───────────────────────────────────────────────────────────


def test_compare_mismatch(capsys, hello_file, tmp_path):
    helixc = tmp_path / "out.helixc"
    main([str(hello_file), "--compile", "-o", str(helixc)])
    different_src = tmp_path / "diff.helix"
    different_src.write_text("#gene name=hello\nATG GCT TAA\n#end\n#config ticks=3\n")
    rc = main([str(different_src), "--compare", str(helixc)])
    assert rc in (0, 1)


def test_compare_source_run_fails(capsys, tmp_path):
    helixc = tmp_path / "out.helixc"
    good = tmp_path / "good.helix"
    good.write_text(HELLO_SRC)
    main([str(good), "--compile", "-o", str(helixc)])
    bad = tmp_path / "bad.helix"
    bad.write_text("#gene name=g\nATG GCT\n#end\n")
    rc = main([str(bad), "--compare", str(helixc)])
    assert rc != 0


# ── --watch ─────────────────────────────────────────────────────────────


def test_watch_iterations(capsys, hello_file):
    rc = main([str(hello_file), "--watch", "--watch-iterations", "2",
               "--watch-interval", "0.01"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[jit]" in out


def test_watch_with_ticks(capsys, hello_file):
    rc = main([str(hello_file), "--watch", "--watch-iterations", "1",
               "--watch-interval", "0.01", "--ticks", "3"])
    assert rc == 0


def test_watch_compile_error_returns_1(tmp_path):
    bad = tmp_path / "bad.helix"
    bad.write_text("#gene name=g\nATG GCT\n#end\n")
    rc = main([str(bad), "--watch", "--watch-iterations", "1",
               "--watch-interval", "0.01"])
    assert rc == 1


# ── --encode-dna success path ──────────────────────────────────────────


def test_encode_dna_success(capsys, hello_file):
    try:
        import helixlang.plugins.runtime.dna_codec  # noqa: F401
    except ImportError:
        pytest.skip("biopython not installed")
    rc = main([str(hello_file), "--encode-dna", "goldman"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "oligos" in out.lower() or "bp" in out.lower()


def test_encode_dna_with_pcr(capsys, hello_file):
    try:
        import helixlang.plugins.runtime.dna_codec  # noqa: F401
    except ImportError:
        pytest.skip("biopython not installed")
    rc = main([str(hello_file), "--encode-dna", "goldman", "--pcr-cycles", "5"])
    assert rc == 0


# ── --full-pipeline ────────────────────────────────────────────────────


def test_full_pipeline_import_error(capsys, hello_file, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "helixlang.plugins.apps.full_pipeline":
            raise ImportError("missing dep")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    rc = main([str(hello_file), "--full-pipeline"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err.lower()


# ── _emit_csv ──────────────────────────────────────────────────────────


def test_emit_csv(capsys):
    trace = [
        {"tick": 0, "x": 0, "y": 0, "energy": 1.0, "alive": True,
         "proteins": {"p": 0.5}, "morphology_points_count": 0,
         "field_total_v": 0.0},
    ]
    _emit_csv(trace)
    out = capsys.readouterr().out
    assert "tick,x,y" in out
    assert "0,0,0" in out


# ── --runtime batch StackDepthError path ───────────────────────────────


def test_runtime_batch_with_error(tmp_path):
    bad = tmp_path / "err.helix"
    bad.write_text("#gene name=g\nATG GCT\n#end\n")
    rc = main([str(bad), "--runtime", "batch"])
    assert rc == 1


# ── --runtime ir with compile error ────────────────────────────────────


def test_runtime_ir_compile_error(capsys, tmp_path):
    bad = tmp_path / "bad_ir.helix"
    bad.write_text("#gene name=g\nATG GCT\n#end\n")
    rc = main([str(bad), "--runtime", "ir"])
    assert rc == 1


# ── --compare with reversed arg order ──────────────────────────────────


def test_compare_reversed_order(capsys, hello_file, tmp_path):
    helixc = tmp_path / "out.helixc"
    main([str(hello_file), "--compile", "-o", str(helixc)])
    rc = main([str(helixc), "--compare", str(hello_file)])
    assert rc in (0, 1)
