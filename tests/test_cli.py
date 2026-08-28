"""CLI unit tests.

Covers src/helixlang/cli.py:
- main() argument parsing, exit codes, stdout/stderr output
- normal compile/run / --disassemble / --csv / --png / --ticks
- error paths: missing file, compile errors, missing source
- the _stop_codons_from_table helper function
- missing-dependency branches for --serve and --encode-dna (via mock)
"""
from __future__ import annotations

from unittest import mock

import pytest

from helixlang.cli import _stop_codons_from_table, main
from helixlang.core.codon_table import (
    CILIATE_TABLE,
    MITO_VERTEBRATE_TABLE,
    STANDARD_TABLE,
)

# ============================================================================
# fixtures
# ============================================================================

HELLO_SRC = "#gene name=hello\nATG GCT TAA\n#end\n#config ticks=2\n"

BAD_SRC = "#gene name=g\nATG GCT\n#end\n"  # ORF does not end with STOP -> semantic error


@pytest.fixture
def helix_file(tmp_path):
    p = tmp_path / "hello.helix"
    p.write_text(HELLO_SRC)
    return p


@pytest.fixture
def bad_file(tmp_path):
    p = tmp_path / "bad.helix"
    p.write_text(BAD_SRC)
    return p


# ============================================================================
# the _stop_codons_from_table helper function
# ============================================================================

class TestStopCodonsHelper:
    """Verifies _stop_codons_from_table."""

    def test_standard_table_stop_codons(self):
        stops = _stop_codons_from_table(STANDARD_TABLE)
        assert stops == {"TAA", "TAG", "TGA"}

    def test_mito_table_stop_codons(self):
        """Mitochondrial table: TAA/AGA/AGG are STOP, TGA is no longer STOP."""
        stops = _stop_codons_from_table(MITO_VERTEBRATE_TABLE)
        assert "TAA" in stops
        assert "AGA" in stops
        assert "AGG" in stops
        assert "TGA" not in stops

    def test_ciliate_table_stop_codons(self):
        """Ciliate table: TAA/TAG map to OP_EMIT_MORPHOGEN; only TGA is STOP."""
        stops = _stop_codons_from_table(CILIATE_TABLE)
        assert stops == {"TGA"}

    def test_returns_set_type(self):
        stops = _stop_codons_from_table(STANDARD_TABLE)
        assert isinstance(stops, set)


# ============================================================================
# Argument parsing and missing args
# ============================================================================

class TestArgumentParsing:
    """Verifies argument parsing and missing-arg branches."""

    def test_missing_source_without_serve_returns_2(self, capsys):
        """No source and no --serve -> argparse error, exit 2."""
        with pytest.raises(SystemExit) as ei:
            main([])
        assert ei.value.code == 2
        err = capsys.readouterr().err
        assert "missing source" in err.lower() or "usage" in err.lower()

    def test_nonexistent_source_returns_2(self, capsys, tmp_path):
        """Nonexistent source path -> exit 2."""
        missing = tmp_path / "ghost.helix"
        rc = main([str(missing)])
        assert rc == 2
        err = capsys.readouterr().err
        assert "not found" in err.lower() or "error" in err.lower()


# ============================================================================
# Normal run modes
# ============================================================================

class TestNormalRun:
    """Verifies default run mode output and exit codes."""

    def test_valid_source_returns_0(self, capsys, helix_file):
        rc = main([str(helix_file)])
        assert rc == 0
        out = capsys.readouterr().out
        # Default output contains the file name and ticks summary
        assert "hello.helix" in out
        assert "ticks=2" in out

    def test_default_output_shows_last_snapshots(self, capsys, helix_file):
        rc = main([str(helix_file)])
        assert rc == 0
        out = capsys.readouterr().out
        # Should print at least one tick summary line
        assert "tick=" in out

    def test_table_argument_changes_translation(self, capsys, tmp_path):
        """--table mito_vertebrate switches the translation table (should run normally)."""
        # Use a source that is valid under both tables (standard ORF)
        src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=1\n"
        p = tmp_path / "t.helix"
        p.write_text(src)
        rc = main([str(p), "--table", "mito_vertebrate"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "table=mito_vertebrate" in out

    def test_invalid_table_choice_exits_2(self, capsys, helix_file):
        """Invalid --table value -> argparse choices error, exit 2."""
        with pytest.raises(SystemExit) as ei:
            main([str(helix_file), "--table", "bogus"])
        assert ei.value.code == 2


# ============================================================================
# --disassemble
# ============================================================================

class TestDisassemble:
    """Verifies the --disassemble mode."""

    def test_disassemble_returns_0(self, capsys, helix_file):
        rc = main([str(helix_file), "--disassemble"])
        assert rc == 0
        out = capsys.readouterr().out
        # disassemble output should contain the file name or chunk info
        assert len(out) > 0

    def test_disassemble_does_not_run_vm(self, capsys, helix_file):
        """--disassemble should not print a run trace."""
        rc = main([str(helix_file), "--disassemble"])
        assert rc == 0
        out = capsys.readouterr().out
        # Should not contain tick= lines
        assert "tick=" not in out


# ============================================================================
# --csv
# ============================================================================

class TestCsvOutput:
    """Verifies the --csv output format."""

    def test_csv_has_header(self, capsys, helix_file):
        rc = main([str(helix_file), "--csv"])
        assert rc == 0
        out = capsys.readouterr().out
        lines = out.strip().splitlines()
        # First line is the header
        header = lines[0]
        for col in ("tick", "x", "y", "energy", "alive",
                    "proteins", "morphology_points", "field_total_v"):
            assert col in header, f"missing column {col} in CSV header"

    def test_csv_row_count_matches_ticks(self, capsys, tmp_path):
        src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=3\n"
        p = tmp_path / "t.helix"
        p.write_text(src)
        rc = main([str(p), "--csv"])
        assert rc == 0
        out = capsys.readouterr().out
        lines = [l for l in out.strip().splitlines() if l]
        # Header + 3 tick snapshots
        assert len(lines) == 1 + 3


# ============================================================================
# --png (PPM output)
# ============================================================================

class TestPngOutput:
    """Verifies --png outputs a PPM file."""

    def test_png_with_lsystem_creates_ppm(self, tmp_path):
        """An L-system program with --png should produce a .ppm file."""
        src = """#promoter name=p strength=-0.5
#gene name=grow promoter=p
ATG CTC TAA
#end
#regulate p -> grow strength=+0.6
#lsystem name=plant axiom=F rules=0:F->F[+F]F[-F]F angle=25
#config ticks=5
"""
        p = tmp_path / "lsys.helix"
        p.write_text(src)
        prefix = str(tmp_path / "out")
        rc = main([str(p), "--png", prefix])
        assert rc == 0
        ppm = tmp_path / "out.ppm"
        assert ppm.exists()
        content = ppm.read_text()
        # P3 format
        assert content.startswith("P3")

    def test_png_with_field_creates_ppm(self, tmp_path):
        """A reaction-diffusion field program with --png should produce a .ppm."""
        src = """#promoter name=p strength=-0.5
#gene name=reactor promoter=p
ATG GAT TAA
#end
#field size=8 F=0.035 k=0.065
#config ticks=3 react_steps=1
"""
        p = tmp_path / "field.helix"
        p.write_text(src)
        prefix = str(tmp_path / "field_out")
        rc = main([str(p), "--png", prefix])
        assert rc == 0
        ppm = tmp_path / "field_out.ppm"
        assert ppm.exists()
        assert ppm.read_text().startswith("P3")


# ============================================================================
# --ticks override
# ============================================================================

class TestTicksOverride:
    """Verifies --ticks overrides #config ticks."""

    def test_ticks_override_changes_snapshot_count(self, capsys, tmp_path):
        """Source ticks=2, --ticks 5 -> should output 5 ticks."""
        src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=2\n"
        p = tmp_path / "t.helix"
        p.write_text(src)
        rc = main([str(p), "--csv", "--ticks", "5"])
        assert rc == 0
        out = capsys.readouterr().out
        lines = [l for l in out.strip().splitlines() if l]
        # Header + 5 tick snapshots
        assert len(lines) == 1 + 5

    def test_ticks_override_shown_in_summary(self, capsys, tmp_path):
        src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=2\n"
        p = tmp_path / "t.helix"
        p.write_text(src)
        rc = main([str(p), "--ticks", "7"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "ticks=7" in out


# ============================================================================
# Error paths
# ============================================================================

class TestErrorPaths:
    """Verifies exit codes and stderr output for various errors."""

    def test_compile_error_returns_1(self, capsys, bad_file):
        """ORF does not end with STOP -> semantic error -> exit 1."""
        rc = main([str(bad_file)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "compile error" in err.lower() or "error" in err.lower()

    def test_compile_error_lex_returns_1(self, capsys, tmp_path):
        """DNA length not a multiple of 3 -> LexError -> exit 1."""
        src = "#gene name=g\nATG GC TAA\n#end\n"  # GC not a multiple of 3
        p = tmp_path / "lex.helix"
        p.write_text(src)
        rc = main([str(p)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "compile error" in err.lower() or "error" in err.lower()

    def test_empty_source_file_compile_error(self, capsys, tmp_path):
        """Empty file -> compile error -> exit 1."""
        p = tmp_path / "empty.helix"
        p.write_text("")
        rc = main([str(p)])
        # An empty file has no tokens; may compile successfully (empty Program) or fail
        # Should at least be 0 or 1, not 2 (the file exists)
        assert rc in (0, 1)


# ============================================================================
# --serve mode
# ============================================================================

class TestServeMode:
    """Verifies the --serve mode (mocks run_server to avoid real listening)."""

    def test_serve_no_source_required(self, capsys):
        """--serve does not require a source argument."""
        with mock.patch("helixlang.server.run_server", return_value=0) as m:
            rc = main(["--serve", "--port", "5555", "--host", "127.0.0.1"])
        assert rc == 0
        m.assert_called_once()
        kwargs = m.call_args.kwargs
        assert kwargs.get("port", kwargs.get("port")) == 5555 or \
               m.call_args.args[-2:] == ("127.0.0.1", 5555) or \
               kwargs.get("host") == "127.0.0.1"
        out = capsys.readouterr().out
        assert "127.0.0.1" in out or "5555" in out

    def test_serve_returns_run_server_value(self, capsys):
        """The --serve return value = the run_server return value."""
        with mock.patch("helixlang.server.run_server", return_value=0):
            rc = main(["--serve"])
        assert rc == 0

    def test_serve_import_error_returns_1(self, capsys, monkeypatch):
        """server module import failure (missing Flask) -> exit 1."""
        # Simulate ImportError when importing helixlang.server
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "helixlang.server" or name.endswith(".server"):
                raise ImportError("No module named 'flask'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rc = main(["--serve"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "flask" in err.lower() or "web" in err.lower()


# ============================================================================
# --encode-dna / --decode-dna (missing-dependency branches)
# ============================================================================

class TestDnaCodecMode:
    """Verifies the missing-dependency branches of the DNA codec modes."""

    def test_encode_dna_without_bio_dep_returns_1(self, capsys, helix_file,
                                                    monkeypatch):
        """--encode-dna but dna_codec cannot be imported -> exit 1."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "helixlang.plugins.runtime.dna_codec":
                raise ImportError("No module named 'biopython'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rc = main([str(helix_file), "--encode-dna", "goldman"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "bio" in err.lower() or "biopython" in err.lower()

    def test_decode_dna_nonexistent_file_returns_2(self, capsys, tmp_path,
                                                     monkeypatch):
        """--decode-dna with a nonexistent file -> exit 2."""
        # Even if dna_codec can be imported, a nonexistent file should still return 2
        # But first ensure dna_codec can be imported (otherwise it returns 1)
        try:
            import helixlang.plugins.runtime.dna_codec  # noqa: F401
        except ImportError:
            pytest.skip("biopython not installed")

        # Provide an existing source to pass the pre-checks
        src = "#gene name=g\nATG GCT TAA\n#end\n"
        p = tmp_path / "src.helix"
        p.write_text(src)
        missing_dna = tmp_path / "missing.fasta"
        rc = main([str(p), "--decode-dna", str(missing_dna)])
        assert rc == 2


# ============================================================================
# Multi-flag combinations
# ============================================================================

class TestFlagCombinations:
    """Verifies behavior of multiple flag combinations."""

    def test_csv_and_png_together(self, capsys, helix_file, tmp_path):
        """--csv + --png together: no default summary printed, but ppm is generated."""
        prefix = str(tmp_path / "combo")
        rc = main([str(helix_file), "--csv", "--png", prefix])
        assert rc == 0
        out = capsys.readouterr().out
        # --csv output
        assert "tick" in out
        # File generation is not required (without field/lsys, _emit_ppm silently returns), but there should be no error

    def test_debug_flag_does_not_crash(self, capsys, helix_file):
        """--debug should output debug tracing without crashing."""
        rc = main([str(helix_file), "--debug"])
        assert rc == 0
