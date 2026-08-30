"""``LanguageConfig`` (doc/38 §4) — one object, four consumers.

Kills the grammatical-truth fragmentation: stop codons were a hardcoded
literal in ``Parser``, start codons never existed as a set, and the
amino-acid translation tables were duplicated in two plugins.
"""
import pytest

from helixlang.core.bytecode import Chunk
from helixlang.core.codon_table import (
    CILIATE_TABLE,
    MITO_VERTEBRATE_TABLE,
    STANDARD_AMINO_ACIDS,
    STANDARD_TABLE,
    Op,
    get_table,
    start_codons_from_table,
    stop_codons_from_table,
    translation_table_from_ncbi,
)
from helixlang.core.compiler import Compiler
from helixlang.core.errors import CompileError, ParseError
from helixlang.core.language import LanguageConfig
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser, parse_source
from helixlang.core.vm import CellVM

TABLES = ("standard", "mito_vertebrate", "ciliate")


# ---------------------------------------------------------------------------
# Derived-property correctness
# ---------------------------------------------------------------------------
def test_for_table_derives_everything():
    """The config derives codon->op, stops, starts, translation from one map."""
    for name in TABLES:
        cfg = LanguageConfig.for_table(name)
        table = get_table(name)
        assert cfg.table_name == name
        assert cfg.codon_to_op is table
        assert cfg.stop_codons == frozenset(stop_codons_from_table(table))
        assert cfg.start_codons == frozenset(start_codons_from_table(table))
        assert cfg.translation == translation_table_from_ncbi(table)
        # translation never leaks codons the table does not define
        assert set(cfg.translation) <= set(table)


def test_config_known_boundary_sets():
    std = LanguageConfig.for_table("standard")
    mito = LanguageConfig.for_table("mito_vertebrate")
    ciliate = LanguageConfig.for_table("ciliate")
    assert std.stop_codons == frozenset({"TAA", "TAG", "TGA"})
    assert std.start_codons == frozenset({"ATG"})
    assert mito.stop_codons == frozenset({"TAA", "TAG", "AGA", "AGG"})
    assert mito.start_codons == frozenset({"ATG", "ATA"})
    assert ciliate.stop_codons == frozenset({"TGA"})
    assert ciliate.start_codons == frozenset({"ATG"})
    # the target behaviour the doc calls out: TAA is NOT a stop for ciliate
    assert CILIATE_TABLE["TAA"] == Op.OP_EMIT_MORPHOGEN


def test_translation_from_ncbi_standard():
    tr = LanguageConfig.for_table("standard").translation
    assert tr["ATG"] == "M"
    assert tr["TAA"] == "*"  # stop
    # the plugin-facing canonical map is the same single source
    assert tr["GCT"] == STANDARD_AMINO_ACIDS["GCT"] == "A"


def test_standard_classmethod_matches_for_table():
    assert LanguageConfig.standard() == LanguageConfig.for_table("standard")


# ---------------------------------------------------------------------------
# Acceptance: parse_source works with NO stop_codons argument
# ---------------------------------------------------------------------------
ANNOT_ONLY = """\
#promoter name=p1 strength=0.8
#promoter name=p2 strength=-0.4
#regulate p1 -> g1 strength=0.9
#regulate p2 -> g1 strength=-0.6
#lsystem name=plant axiom=F rules=0:F->F[+F]F[-F]F angle=25 step=1.0
#config ticks=5
"""


def test_property_identical_programs_across_all_tables():
    """The acceptance property test: one source -> identical Programs for
    all three tables via parse_source with no stop_codons argument."""
    programs = [parse_source(ANNOT_ONLY, config=LanguageConfig.for_table(n))
                for n in TABLES]
    for p in programs[1:]:
        assert p == programs[0]


def test_parse_source_without_stop_codons_matches_standard_derivation():
    """:func:`parse_source` with no config defaults to the standard table
    (the previous hardcoded literal) — byte-identical parse result."""
    src = "#gene name=m\nATG TGA GCT TAA\n#end"
    via_config = parse_source(src)
    legacy = Parser(
        list(Lexer(src).tokens()),
        stop_codons={"TAA", "TAG", "TGA"},
    ).parse()
    assert via_config == legacy


# ---------------------------------------------------------------------------
# Config drives stop/start boundaries in the parser
# ---------------------------------------------------------------------------
def test_stop_boundaries_drive_orf_extraction():
    src = "#gene name=g\nATG TAA GCT TGA\n#end"
    std = parse_source(src, config=LanguageConfig.for_table("standard"))
    # standard: TAA is a stop -> ORF = ATG TAA
    assert [c.seq for c in std.genes[0].orf] == ["ATG", "TAA"]
    ciliate = parse_source(src, config=LanguageConfig.for_table("ciliate"))
    # ciliate: TAA -> OP_EMIT_MORPHOGEN (not a stop) -> ORF = ATG TAA GCT TGA
    assert [c.seq for c in ciliate.genes[0].orf] == ["ATG", "TAA", "GCT", "TGA"]


def test_mito_tga_is_not_a_stop():
    src = "#gene name=m\nATG TGA GCT TAA\n#end"
    mito = parse_source(src, config=LanguageConfig.for_table("mito_vertebrate"))
    assert len(mito.genes[0].orf) == 4


def test_start_boundaries_drive_orf_extraction():
    # mito start codons are ATG/ATA; standard is ATG only
    src = "#gene name=g\nATA GCT GCT TAA\n#end"
    with pytest.raises(ParseError, match="START codon"):
        parse_source(src, config=LanguageConfig.for_table("standard"))
    mito = parse_source(src, config=LanguageConfig.for_table("mito_vertebrate"))
    assert [c.seq for c in mito.genes[0].orf] == ["ATA", "GCT", "GCT", "TAA"]


def test_lexer_is_table_agnostic():
    """The lexer emits CODON tokens; the config never touches lexing."""
    src = "#gene name=g\nATG TAA TGA\n#end"
    seqs = [t.value for t in Lexer(src).tokens() if t.kind == "CODON"]
    assert seqs == ["ATG", "TAA", "TGA"]


# ---------------------------------------------------------------------------
# TAA -> OP_EMIT_MORPHOGEN consistently in IR, chunk, and VM (ciliate)
# ---------------------------------------------------------------------------
CILIATE_SRC = "#gene name=g\nATG TAA GCT TGA\n#end"
CILIATE = LanguageConfig.for_table("ciliate")


def test_ciliate_taa_morphogen_in_ir():
    ciliate_ir = Compiler(config=CILIATE).build_ir(
        parse_source(CILIATE_SRC, config=CILIATE))
    ciliate_fn = next(f for f in ciliate_ir.functions if f.name == "g")
    assert [i.opcode for i in ciliate_fn.instrs] == [
        Op.OP_START, Op.OP_EMIT_MORPHOGEN, Op.OP_BUILD_PROTEIN, Op.OP_HALT,
    ]
    std_ir = Compiler().build_ir(parse_source(CILIATE_SRC))
    std_fn = next(f for f in std_ir.functions if f.name == "g")
    # standard ORF stops at TAA -> no morphogen instruction at all
    assert [i.opcode for i in std_fn.instrs] == [Op.OP_START, Op.OP_HALT]


def test_ciliate_taa_morphogen_in_chunk():
    ciliate = Compiler(config=CILIATE).compile(
        parse_source(CILIATE_SRC, config=CILIATE))
    std = Compiler().compile(parse_source(CILIATE_SRC))
    assert int(Op.OP_EMIT_MORPHOGEN) in ciliate.code
    assert int(Op.OP_EMIT_MORPHOGEN) not in std.code


def test_ciliate_taa_emits_morphogen_in_vm():
    prog = parse_source(CILIATE_SRC, config=CILIATE)
    chunk = Compiler(config=CILIATE).compile(prog)
    vm = CellVM(chunk, prog)
    from helixlang.plugins.runtime.reaction_diffusion import GrayScott
    vm.field = GrayScott(n=8)
    from helixlang.core.vm import Frame
    vm.frames.append(Frame(return_ip=len(chunk.code), gene_name="main"))
    vm.ip = 0
    before = vm.field.v[0][0]
    vm._execute_pending()
    after = vm.field.v[0][0]
    assert after > before


# ---------------------------------------------------------------------------
# config vs legacy argument mutual exclusion
# ---------------------------------------------------------------------------
def test_parser_rejects_config_plus_stop_codons():
    tokens = list(Lexer("#gene name=g\nATG TAA\n#end").tokens())
    with pytest.raises(ParseError, match="exactly one of 'config' or 'stop_codons'"):
        Parser(tokens, stop_codons={"TAA", "TAG", "TGA"}, config=CILIATE)
    with pytest.raises(ParseError, match="exactly one of 'config' or 'stop_codons'"):
        parse_source("#promoter name=p strength=1.0\n",
                     stop_codons={"TAA", "TAG", "TGA"}, config=CILIATE)


def test_parser_legacy_stop_codons_still_works():
    tokens = list(Lexer("#gene name=m\nATG TGA GCT TAA\n#end").tokens())
    prog = Parser(tokens, stop_codons=stop_codons_from_table(MITO_VERTEBRATE_TABLE)).parse()
    assert len(prog.genes[0].orf) == 4


def test_compiler_rejects_config_plus_table():
    with pytest.raises(CompileError, match="exactly one of 'config' or 'table'"):
        Compiler(STANDARD_TABLE, config=CILIATE)


def test_compiler_equivalent_config_paths():
    prog = parse_source("#gene name=m\nATG TGA GCT TAA\n#end")
    a = Compiler().compile(prog)
    b = Compiler(config=LanguageConfig.standard()).compile(prog)
    c = Compiler(STANDARD_TABLE).compile(prog)
    for other in (b, c):
        assert isinstance(other, Chunk)
        assert other.code == a.code
        assert other.constants == a.constants


def test_vm_and_runtime_thread_config():
    prog = parse_source(CILIATE_SRC, config=CILIATE)
    chunk = Compiler(config=CILIATE).compile(prog)
    # without an explicit config the VM resolves program.config.table
    assert CellVM(chunk, prog).config.table_name == "standard"
    # pass-through config is honoured for provenance
    vm = CellVM(chunk, prog, config=CILIATE)
    assert vm.config == CILIATE
    assert vm.config.codon_to_op["TAA"] == Op.OP_EMIT_MORPHOGEN


# ---------------------------------------------------------------------------
# The two plugin AA tables are deleted in favour of codon_table
# ---------------------------------------------------------------------------
def test_plugin_aa_tables_deleted_in_favour_of_codon_table():
    import helixlang.plugins.annotation.sequences as sequences
    import helixlang.plugins.apps.full_pipeline as full_pipeline
    # no private duplicate maps remain
    assert not hasattr(sequences, "_CODON_TABLE")
    assert not hasattr(full_pipeline, "_CODON_TABLE")
    assert sequences.STANDARD_AMINO_ACIDS is STANDARD_AMINO_ACIDS
    # behaviour preserved through the shared map
    assert sequences.translate("ATGGCTTAA") == "MA"
    assert full_pipeline._translate_dna_to_protein("ATGGCTTAA") == "MA"  # noqa: SLF001
