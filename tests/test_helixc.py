"""Binary artifact (.helixc) codec tests.

Covers doc/helixc-binary-format.md:
- encode/decode round-trips (Program, Chunk, SRC, table)
- decompiler invariants R1 (reparse -> same chunk), R2 (canonical
  byte-for-byte), R3 (embedded source byte-for-byte)
- deterministic dumps, integrity/version errors, stale-chunk rebuild
- CLI: --compile / --decompile / --compare, run classic + sim from binary,
  --disassemble and --debug equivalence with the source path
- compile + load-verify for every examples/*.helix
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from helixlang import hxbc
from helixlang.codon_table import get_table
from helixlang.compiler import Compiler
from helixlang.lexer import Lexer
from helixlang.parser import Parser
from helixlang.semantic import SemanticAnalyzer
from helixlang.seq_utils import stop_codons_from_table

STANDARD = get_table("standard")


# ============================================================================
# helpers
# ============================================================================
def parse(src: str):
    stop = stop_codons_from_table(STANDARD)
    prog = Parser(list(Lexer(src).tokens()), stop_codons=stop).parse()
    SemanticAnalyzer(prog).check()
    return prog, Compiler(STANDARD).compile(prog)


def cli(argv: list[str]) -> tuple[int, str]:
    """Run helixlang.cli.main with stdout captured."""
    from helixlang.cli import main
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def normalize_lines(prog):
    """Zero out source-line metadata so semantic equality ignores layout."""
    for g in prog.genes:
        for c in (*g.codons, *g.orf):
            c.line = 0
    for inst in prog.bio_instructions:
        inst.line = 0
    return prog


def strip_headers(out: str) -> str:
    """Drop filename-bearing '===' summary lines (differ by extension)."""
    return "\n".join(l for l in out.splitlines() if not l.startswith("==="))


RICH_SRC = """\
#promoter name=p1 strength=0.7 extra="a b"
#gene name=g1 promoter=p1 call_target=g2
ATG GCT GGT GGT TAA
#end
#gene name=g2 promoter=p1
ATG GCT GGT TAA
#end
#regulate p1 -> g1 strength=0.8
#lsystem name=tree axiom=F rules=0:F->F[+F]F;1:F->FF angle=25 step=1.0
#field size=16 F=0.035 k=0.065 Du=0.16 Dv=0.08
#morphogen gene=g1 channel=V gain=0.2
#config ticks=5 output=stdout table=standard ops_per_tick=64 react_steps=2 use_central_dogma=true species=ecoli backend=classic
#sim kind=protein_structure
#type g1=Protein
#crispr target=g2 position=3 new_sequence="GGG"
#media nutrient=GLC concentration=10.0 diffusion_um2_s=300
#enzyme gene=g2 reaction=CS kcat=100
#metabolite name=glc__D init=0.5
ATG GCT GGT GAA TAA
"""

CANONICAL_SRC = """\
#gene name=g1
ATG GCT GGT TAA
#end
#config ticks=5 output=stdout table=standard ops_per_tick=64 react_steps=1 use_central_dogma=false species=ecoli backend=classic alpha=1 beta=two
#sim kind=protein_structure
"""


# ============================================================================
# 1. encode/decode round-trip
# ============================================================================
class TestCodecRoundtrip:
    def test_program_chunk_source_roundtrip(self):
        prog, chunk = parse(RICH_SRC)
        data = hxbc.dumps_program(prog, chunk=chunk, source=RICH_SRC)
        art = hxbc.loads_program(data)
        assert art.program == prog
        assert art.chunk is not None
        assert art.chunk.code == chunk.code
        assert art.chunk.constants == chunk.constants
        assert art.chunk.gene_offsets == chunk.gene_offsets
        assert art.source == RICH_SRC
        assert art.table == "standard"
        assert not art.chunk_stale

    def test_no_chunk_no_source_roundtrip(self):
        prog, chunk = parse(RICH_SRC)
        art = hxbc.loads_program(hxbc.dumps_program(prog))
        assert art.program == prog
        assert art.chunk is None
        assert art.source is None
        assert hxbc.decompile(art.program).strip() == \
            hxbc.decompile(prog).strip()

    def test_save_load_verify(self, tmp_path):
        prog, chunk = parse(CANONICAL_SRC)
        path = tmp_path / "prog.helixc"
        hxbc.save_program(prog, path, chunk=chunk, source=CANONICAL_SRC)
        art = hxbc.load_program(path)
        assert art.program == prog
        hxbc.verify(path)

    def test_table_id_roundtrip(self, tmp_path):
        prog, chunk = parse(CANONICAL_SRC)
        prog.config.table = "mito_vertebrate"
        data = hxbc.dumps_program(prog, chunk=chunk)
        assert hxbc.loads_program(data).table == "mito_vertebrate"


# ============================================================================
# 2-4. decompiler invariants
# ============================================================================
class TestDecompiler:
    def test_r1_reparse_same_chunk(self):
        prog, chunk = parse(RICH_SRC)
        text = hxbc.decompile(prog)
        prog2, chunk2 = parse(text)
        assert chunk2.code == chunk.code
        assert chunk2.constants == chunk.constants
        assert chunk2.gene_offsets == chunk.gene_offsets
        assert normalize_lines(prog2) == normalize_lines(prog)

    def test_r1_anonymous_genes_roundtrip(self):
        prog, chunk = parse(RICH_SRC)
        assert any(g.name.startswith("__anon") for g in prog.genes)
        prog2, chunk2 = parse(hxbc.decompile(prog))
        assert chunk2.code == chunk.code
        assert [g.name for g in prog2.genes] == [g.name for g in prog.genes]

    def test_r2_canonical_byte_identical(self):
        prog, chunk = parse(CANONICAL_SRC)
        assert hxbc.decompile(prog) == CANONICAL_SRC

    def test_r3_embedded_source_byte_for_byte(self, tmp_path):
        prog, chunk = parse(RICH_SRC)
        path = tmp_path / "rich.helixc"
        hxbc.save_program(prog, path, chunk=chunk, source=RICH_SRC)
        rc, out = cli(["--decompile", str(path), "-o", str(tmp_path / "out.helix")])
        assert rc == 0
        assert (tmp_path / "out.helix").read_text() == RICH_SRC

    def test_decompile_all_examples(self, examples_dir):
        stop = stop_codons_from_table(STANDARD)
        for src_path in sorted(examples_dir.glob("*.helix")):
            src = src_path.read_text()
            prog = Parser(list(Lexer(src).tokens()), stop_codons=stop).parse()
            SemanticAnalyzer(prog).check()
            chunk = Compiler(STANDARD).compile(prog)
            prog2, chunk2 = parse(hxbc.decompile(prog))
            assert chunk2.code == chunk.code, src_path.name
            assert chunk2.constants == chunk.constants, src_path.name
            assert chunk2.gene_offsets == chunk.gene_offsets, src_path.name
            assert normalize_lines(prog2) == normalize_lines(prog), src_path.name


# ============================================================================
# 5. determinism
# ============================================================================
class TestDeterminism:
    def test_dumps_deterministic(self):
        prog, chunk = parse(RICH_SRC)
        a = hxbc.dumps_program(prog, chunk=chunk, source=RICH_SRC)
        b = hxbc.dumps_program(prog, chunk=chunk, source=RICH_SRC)
        assert a == b

    def test_compile_file_deterministic(self, tmp_path):
        src = tmp_path / "p.helix"
        src.write_text(CANONICAL_SRC)
        out_a = tmp_path / "a.helixc"
        out_b = tmp_path / "b.helixc"
        hxbc.compile_file(src, out_a)
        hxbc.compile_file(src, out_b)
        assert out_a.read_bytes() == out_b.read_bytes()


# ============================================================================
# 11-12, 14. integrity / version / stale chunk
# ============================================================================
class TestSafety:
    def test_integrity_corrupt_payload_byte(self):
        prog, chunk = parse(CANONICAL_SRC)
        data = bytearray(hxbc.dumps_program(prog, chunk=chunk, source=CANONICAL_SRC))
        data[20] ^= 0x01  # corrupt a payload byte inside PROG
        with pytest.raises(hxbc.BinaryFormatError):
            hxbc.loads_program(bytes(data))

    def test_integrity_corrupt_checksum(self):
        prog, chunk = parse(CANONICAL_SRC)
        data = bytearray(hxbc.dumps_program(prog, chunk=chunk))
        data[-1] ^= 0xFF  # flip a digest byte
        with pytest.raises(hxbc.BinaryFormatError, match="checksum"):
            hxbc.loads_program(bytes(data))

    def test_version_bump_rejected(self):
        prog, chunk = parse(CANONICAL_SRC)
        data = bytearray(hxbc.dumps_program(prog, chunk=chunk))
        data[4] = 99
        with pytest.raises(hxbc.BinaryVersionError):
            hxbc.loads_program(bytes(data))

    def test_bad_magic_rejected(self):
        prog, chunk = parse(CANONICAL_SRC)
        data = bytearray(hxbc.dumps_program(prog, chunk=chunk))
        data[0:4] = b"BOGUS"
        with pytest.raises(hxbc.BinaryFormatError, match="magic"):
            hxbc.loads_program(bytes(data))

    def test_truncated_rejected(self):
        prog, chunk = parse(CANONICAL_SRC)
        data = hxbc.dumps_program(prog, chunk=chunk)
        with pytest.raises(hxbc.BinaryFormatError):
            hxbc.loads_program(data[: len(data) // 2])

    def test_stale_chunk_dropped_and_rebuilt(self):
        prog01, chunk01 = parse(CANONICAL_SRC)
        prog02, chunk02 = parse(RICH_SRC)  # different gene set
        art = hxbc.loads_program(hxbc.dumps_program(prog01, chunk=chunk02))
        assert art.chunk is None
        assert art.chunk_stale is True
        rebuilt = Compiler(STANDARD).compile(art.program)
        assert rebuilt.code == chunk01.code

    def test_verify_detects_corruption(self, tmp_path):
        prog, chunk = parse(CANONICAL_SRC)
        path = tmp_path / "v.helixc"
        hxbc.save_program(prog, path, chunk=chunk)
        hxbc.verify(path)
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 0x01
        path.write_bytes(bytes(raw))
        with pytest.raises(hxbc.BinaryFormatError):
            hxbc.verify(path)


# ============================================================================
# CLI: compile / decompile / compare / run
# ============================================================================
class TestCliArtifact:
    def test_compile_and_run_classic(self, examples_dir, tmp_path):
        src = examples_dir / "02_lac_operon.helix"
        art = tmp_path / "02.helixc"
        rc, out = cli(["--compile", str(src), "-o", str(art)])
        assert rc == 0 and "bytes" in out
        rc, out_src = cli([str(src), "--csv"])
        rc2, out_art = cli([str(art), "--csv"])
        assert rc == rc2 == 0
        assert out_art == out_src

    def test_compile_no_chunk_runs(self, examples_dir, tmp_path):
        src = examples_dir / "01_hello_dna.helix"
        art = tmp_path / "01.helixc"
        rc, _ = cli(["--compile", str(src), "-o", str(art), "--no-chunk"])
        assert rc == 0
        rc, out = cli([str(art), "--csv"])
        assert rc == 0
        assert out.startswith("tick,x,y,energy")

    def test_sim_backend_from_binary(self, examples_dir, tmp_path):
        src = examples_dir / "11_protein_structure.helix"
        art = tmp_path / "11.helixc"
        rc, _ = cli(["--compile", str(src), "-o", str(art)])
        assert rc == 0
        rc, out = cli([str(art), "--json"])
        assert rc == 0
        payload = json.loads(out)
        assert payload["backend"] == "protein_structure"

    def test_population_backend_from_binary(self, examples_dir, tmp_path):
        src = examples_dir / "16_population_dynamics.helix"
        art = tmp_path / "16.helixc"
        rc, _ = cli(["--compile", str(src), "-o", str(art)])
        assert rc == 0
        rc, out = cli([str(art), "--json"])
        assert rc == 0
        assert json.loads(out)["backend"] == "population"

    def test_compare_matches(self, examples_dir, tmp_path):
        src = examples_dir / "02_lac_operon.helix"
        art = tmp_path / "02.helixc"
        cli(["--compile", str(src), "-o", str(art)])
        rc, out = cli(["--compare", str(src), str(art)])
        assert rc == 0 and "compare OK" in out
        rc, out = cli([str(src), "--compare", str(art)])  # other flag order
        assert rc == 0 and "compare OK" in out

    def test_compare_mismatch_detected(self, examples_dir, tmp_path):
        src = examples_dir / "02_lac_operon.helix"
        other = examples_dir / "01_hello_dna.helix"
        art = tmp_path / "wrong.helixc"
        cli(["--compile", str(other), "-o", str(art)])
        rc, out = cli(["--compare", str(src), str(art)])
        assert rc == 1 and "MISMATCH" in out

    def test_disassemble_binary_matches_source(self, examples_dir, tmp_path):
        src = examples_dir / "02_lac_operon.helix"
        art = tmp_path / "02.helixc"
        cli(["--compile", str(src), "-o", str(art)])
        rc, out_src = cli([str(src), "--disassemble"])
        rc2, out_art = cli([str(art), "--disassemble"])
        assert rc == rc2 == 0
        assert strip_headers(out_art) == strip_headers(out_src)

    def test_debug_binary_matches_source(self, examples_dir, tmp_path):
        src = examples_dir / "01_hello_dna.helix"
        art = tmp_path / "01.helixc"
        cli(["--compile", str(src), "-o", str(art)])
        rc, out_src = cli([str(src), "--debug"])
        rc2, out_art = cli([str(art), "--debug"])
        assert rc == rc2 == 0
        assert "OP_START" in out_art
        assert strip_headers(out_art) == strip_headers(out_src)

    def test_decompile_no_source_regenerates(self, examples_dir, tmp_path):
        src = examples_dir / "11_protein_structure.helix"
        art = tmp_path / "11.helixc"
        out = tmp_path / "11.helix"
        cli(["--compile", str(src), "-o", str(art), "--no-source"])
        rc, _ = cli(["--decompile", str(art), "-o", str(out)])
        assert rc == 0
        text = out.read_text()
        prog2, chunk2 = parse(text)
        prog, chunk = parse(src.read_text())
        assert chunk2.code == chunk.code
        assert normalize_lines(prog2) == normalize_lines(prog)

    def test_table_override_recompiles(self, examples_dir, tmp_path):
        src = examples_dir / "05_table_switch.helix"
        art = tmp_path / "05.helixc"
        rc, _ = cli(["--compile", str(src), "-o", str(art)])
        assert rc == 0
        rc, out = cli([str(art), "--table", "ciliate", "--csv"])
        assert rc == 0
        assert out.startswith("tick,x,y,energy")

    def test_compile_requires_output(self, examples_dir):
        rc, _ = cli(["--compile", str(examples_dir / "01_hello_dna.helix")])
        assert rc == 2


# ============================================================================
# 13. compile every example and load-verify
# ============================================================================
class TestAllExamples:
    def test_compile_all_examples_load_verify(self, examples_dir, tmp_path):
        stop = stop_codons_from_table(STANDARD)
        for src_path in sorted(examples_dir.glob("*.helix")):
            src = src_path.read_text()
            prog = Parser(list(Lexer(src).tokens()), stop_codons=stop).parse()
            SemanticAnalyzer(prog).check()
            chunk = Compiler(STANDARD).compile(prog)
            art_path = tmp_path / f"{src_path.stem}.helixc"
            info = hxbc.compile_file(src_path, art_path)
            art = hxbc.load_program(art_path)
            assert art.program == prog, src_path.name
            assert art.chunk is not None
            assert art.chunk.code == chunk.code, src_path.name
            assert info.source is not None
