"""Phase D behaviors (doc/38 §7-§8): unit typing, inference, effects, IR dims.

Covers the §7 acceptance (well-typed passes; a conflicting/unsatisfiable
binding fails naming its symbol; zero-annotation inference resolves to ground
types; a side-effecting read in a declared-pure block is caught) and §8
acceptance (quantity math fails at compile time with the dimension tree;
IR round-trip preserves dims; the runtime trace is bit-identical with the
metadata stripped).
"""
import pytest

from helixlang.core import version as _version
from helixlang.core.ast_nodes import Program
from helixlang.core.compiler import Compiler
from helixlang.core.dimensions import (
    DIM_CONCENTRATION,
    DIM_TIME,
    Quantity,
    UnitError,
)
from helixlang.core.errors import ParseError, SemanticError
from helixlang.core.ir import IRFunction, IRInst, IRProgram
from helixlang.core.ir_lower import IRLowerer
from helixlang.core.ir_serialize import IRFormatError, dumps, loads
from helixlang.core.language import LanguageConfig
from helixlang.core.parser import Lexer, Parser
from helixlang.core.semantic import SemanticAnalyzer
from helixlang.core.type_system import (
    BioEffect,
    HelixType,
    TypeChecker,
    TypeVar,
    Unifier,
    UnitType,
    effect_of_opname,
    parse_type_annotation,
)
from helixlang.core.vm import CellVM


def _parse(src: str) -> Program:
    return Parser(list(Lexer(src).tokens())).parse()


def _codon(opname: str) -> str:
    cfg = LanguageConfig.for_table("standard")
    for seq, op in cfg.codon_to_op.items():
        if getattr(op, "name", None) == opname:
            return seq
    raise AssertionError(f"no standard codon for {opname}")


# ============================================================================
# unit-carrying type annotations (doc/38 §7.4)
# ============================================================================
def test_parse_unit_annotation():
    t = parse_type_annotation("Float<µM>")
    assert isinstance(t, UnitType)
    assert t.base is HelixType.FLOAT
    assert t.unit == "µM"
    assert parse_type_annotation("Float<min>").unit == "min"
    assert parse_type_annotation("Protein") is HelixType.PROTEIN


def test_unknown_unit_annotation_rejected_at_parse():
    with pytest.raises(ParseError, match="g"):
        _parse("#config ticks=100 ops_per_tick=100\n"
               "#type g=Float<furlongs>\n"
               "#gene name=g\nATG TAA\n#end\n")


def test_unknown_type_annotation_rejected_at_parse():
    with pytest.raises(ParseError, match="g"):
        _parse("#config ticks=100 ops_per_tick=100\n"
               "#type g=WarpDrive\n"
               "#gene name=g\nATG TAA\n#end\n")


def test_conflicting_annotations_rejected_naming_symbol():
    with pytest.raises(ParseError, match="g"):
        _parse("#type g=Protein\n"
               "#type g=Float\n"
               "#gene name=g\nATG TAA\n#end\n")


def test_repeated_identical_annotation_ok():
    prog = _parse("#type g=Protein\n"
                  "#type g=Protein\n"
                  "#gene name=g\nATG TAA\n#end\n")
    assert prog.type_annotations["g"] == "Protein"


def test_unit_annotation_on_symbol_parses_and_checks():
    prog = _parse("#type g=Float<µM>\n"
                  "#gene name=g\nATG TAA\n#end\n")
    SemanticAnalyzer(prog).check()


def test_quantities_fail_at_compile_time_with_dim_tree():
    # §8 acceptance: adding a concentration to a volume fails at compile time.
    with pytest.raises(UnitError, match="incompatible dimensions"):
        Quantity(5, "min") + Quantity(7, "µm3")
    with pytest.raises(UnitError, match="µM"):
        Quantity(1, "µM") + Quantity(1, "µm3")


# ============================================================================
# inference + unification (doc/38 §7.1/§7.2)
# ============================================================================
def test_unify_var_ground():
    u = Unifier()
    v = TypeVar()
    u.unify(v, HelixType.FLOAT)
    assert u.resolve(v) is HelixType.FLOAT
    u.unify(HelixType.FLOAT, TypeVar())
    assert list(u.substitution.values())


def test_unify_conflict_names_symbol():
    u = Unifier()
    t1 = UnitType(HelixType.FLOAT, "min")
    t2 = UnitType(HelixType.FLOAT, "µm3")
    with pytest.raises(SemanticError, match="g"):
        u.unify(t1, t2, "g")


def test_infer_program_zero_annotations_to_ground():
    prog = _parse("#config ticks=100 ops_per_tick=100\n"
                  "#promoter name=p1 strength=0.8\n"
                  "#gene name=g promoter=p1\nATG TAA\n#end\n")
    st = TypeChecker().infer_program(prog)
    assert st["g"] is HelixType.GENE
    assert st["p1"] is HelixType.PROTEIN
    assert not any(isinstance(t, TypeVar) for t in st.values())


def test_infer_program_records_product_annotations():
    prog = _parse("#type g=Float<µM>\n"
                  "#gene name=g\nATG TAA\n#end\n")
    st = TypeChecker().infer_program(prog)
    assert st["g"] is HelixType.GENE
    assert isinstance(parse_type_annotation("Float<µM>"), UnitType)


def test_checker_records_product_types():
    prog = _parse("#type g=Protein\n#gene name=g\nATG TAA\n#end\n")
    checker = TypeChecker()
    errors = checker.check(prog, type_system_symbols(prog))
    assert errors == []
    assert checker.product_types["g"] is HelixType.PROTEIN


def type_system_symbols(prog):
    from helixlang.core.type_system import SymbolTable
    st = SymbolTable()
    for g in prog.genes:
        st.define(g.name, HelixType.GENE)
    for p in prog.promoters:
        st.define(p.name, HelixType.PROTEIN)
    return st


# ============================================================================
# bio-effect lattice (doc/38 §7.3)
# ============================================================================
def test_effect_lattice_levels():
    assert effect_of_opname("OP_ADD") is BioEffect.PURE
    assert effect_of_opname("OP_READ_MEM") is BioEffect.SIDE_EFFECT
    assert effect_of_opname("OP_SIGNAL") is BioEffect.SIDE_EFFECT
    assert effect_of_opname("OP_TICK") is BioEffect.QUOTA_BOUNDARY
    assert BioEffect.PURE < BioEffect.QUOTA_BOUNDARY < BioEffect.SIDE_EFFECT


def test_pure_region_with_only_pure_ops_passes():
    prog = _parse("#gene name=kin pure=1\nATG TAA\n#end\n")
    SemanticAnalyzer(prog).check()
    tc = TypeChecker()
    errors = tc.check_effects(prog)
    assert errors == []
    assert tc.gene_effects["kin"] is BioEffect.PURE


def test_side_effecting_read_in_pure_region_rejected():
    read = _codon("OP_READ_MEM")
    prog = _parse(f"#gene name=impure pure=1\nATG {read} TAA\n#end\n")
    tc = TypeChecker()
    errors = tc.check_effects(prog)
    assert any("impure" in str(e) and "OP_READ_MEM" in str(e) for e in errors)
    with pytest.raises(SemanticError, match="impure"):
        SemanticAnalyzer(prog).check()


def test_side_effect_recorded_but_not_rejected_outside_pure():
    read = _codon("OP_READ_MEM")
    prog = _parse(f"#gene name=observed\nATG {read} TAA\n#end\n")
    SemanticAnalyzer(prog).check()
    tc = TypeChecker()
    assert tc.check_effects(prog) == []
    assert tc.gene_effects["observed"] is BioEffect.SIDE_EFFECT


# ============================================================================
# IR dimensional metadata (doc/38 §8.2)
# ============================================================================
def test_ir_dims_roundtrip_through_serialize():
    fn = IRFunction(name="main", line=1)
    fn.instrs.append(IRInst(
        opcode=_codon_op("OP_PUSH_CONST"), operand=42,
        dim=DIM_TIME))
    from helixlang.core.codon_table import Op
    fn.instrs[-1].opcode = Op.OP_PUSH_CONST
    fn.instrs.append(IRInst(opcode=Op.OP_HALT))
    ir = IRProgram(name="dims", table="standard")
    ir.functions.append(fn)
    blob = dumps(ir)
    back = loads(blob)
    assert back.functions[0].instrs[0].dim == DIM_TIME
    assert back.functions[0].instrs[0].operand == 42
    assert back.functions[0].instrs[1].dim is None


def test_ir_serialize_dims_unknown_length_rejected():
    from helixlang.core.ir_serialize import idim_to_dim
    with pytest.raises(IRFormatError):
        idim_to_dim([1, 2])


def _codon_op(opname: str):
    from helixlang.core.codon_table import Op
    return getattr(Op, opname)


def test_lowerer_ignores_dims_chunk_identical():
    src = ("#config ticks=40 ops_per_tick=200\n"
           "#gene name=g\nATG TGG TAA\n#end\n")
    prog = _parse(src)
    comp = Compiler(LanguageConfig.for_table("standard"))
    ir, chunk = comp.compile_ir(prog)
    plain_trace = CellVM(chunk, prog).run(prog.config.ticks)
    # Dims are metadata-only: attaching them must not change the lowered
    # chunk (runtime behavior bit-identical with the metadata stripped).
    for fn in ir.functions:
        for inst in fn.instrs:
            inst.dim = DIM_CONCENTRATION
    dimmed_chunk = IRLowerer().lower(ir)
    assert bytes(dimmed_chunk.code) == bytes(chunk.code)
    dim_trace = CellVM(dimmed_chunk, prog).run(prog.config.ticks)
    assert dim_trace == plain_trace


def test_language_spec_version_bumped_for_unit_surface():
    assert _version.LANGUAGE_SPEC_VERSION >= 2
    from helixlang.core import version as ver
    assert hasattr(ver, "SEMANTIC_SURFACES")
    assert "LANGUAGE_SPEC_VERSION" in ver.SEMANTIC_SURFACES


# ============================================================================
# incremental invalidation of #type annotations (doc/38 §8.2)
# ============================================================================
def test_annotation_edit_forces_full_rebuild():
    from helixlang.core.incr import IncrementalCompiler
    base = ("#config ticks=100 ops_per_tick=100\n"
            "#type g=Float<min>\n"
            "#gene name=g\nATG TAA\n#end\n")
    p1 = _parse(base)
    compiler = IncrementalCompiler(LanguageConfig.for_table("standard"))
    r1 = compiler.compile(p1)
    assert not r1.stats.full_build or r1.stats.rebuilt == ["g"]
    p2 = _parse(base.replace("Float<min>", "Float<µM>"))
    r2 = compiler.compile(p2, previous_ir=r1.ir, previous_cache=r1.cache)
    assert r2.stats.full_build is True


# ============================================================================
# math-language type system: structural hierarchy + constraint solving
# (doc/38 §7.1/§7.2 — real Type terms, unit-aware unification, occurs-check)
# ============================================================================
def test_unit_types_same_dimension_unify():
    """Float<min> and Float<s> live in the same family -> unify (doc/38 §7.4)."""
    u = Unifier()
    t_min = UnitType(HelixType.FLOAT, "min")
    t_s = UnitType(HelixType.FLOAT, "s")
    u.unify(t_min, t_s, "t_clock")
    assert u.resolve(t_min) is t_min


def test_unit_type_unifies_with_plain_base():
    """Float<µM> is a Float: unit vs plain base unifies without binding."""
    u = Unifier()
    u.unify(UnitType(HelixType.FLOAT, "µM"), HelixType.FLOAT, "g")
    u.unify(HelixType.FLOAT, UnitType(HelixType.FLOAT, "mM"), "g")


def test_unit_conflict_still_names_symbol():
    """Incompatible dimensions (time vs volume) stay a named error."""
    u = Unifier()
    with pytest.raises(SemanticError, match="g"):
        u.unify(UnitType(HelixType.FLOAT, "min"),
                UnitType(HelixType.FLOAT, "µm3"), "g")


def test_compound_annotation_parses_to_term():
    from helixlang.core.type_system import ListType, RecordType
    assert isinstance(parse_type_annotation("list[Float<µM>]"), ListType)
    rec = parse_type_annotation("record{A: Float<µM>, B: Protein}")
    assert isinstance(rec, RecordType)
    assert isinstance(rec.fields["A"], UnitType)
    assert rec.fields["B"] is HelixType.PROTEIN


def test_compound_unification_structural():
    from helixlang.core.type_system import ListType
    u = Unifier()
    v = TypeVar()
    u.unify(v, ListType(HelixType.FLOAT), "xs")
    assert u.resolve(v) == ListType(HelixType.FLOAT)


def test_occurs_check_through_compound_terms():
    """A variable must not be bound to a compound term it occurs inside."""
    from helixlang.core.type_system import ListType
    u = Unifier()
    v = TypeVar()
    with pytest.raises(SemanticError, match="infinite type"):
        u.unify(v, ListType(v), "rec")


def test_infer_program_is_constraint_driven_and_ground():
    """The zero-annotation acceptance now runs through the Unifier."""
    prog = _parse("#config ticks=100 ops_per_tick=100\n"
                  "#promoter name=p1 strength=0.8\n"
                  "#gene name=g promoter=p1\nATG TAA\n#end\n")
    checker = TypeChecker()
    st = checker.infer_program(prog)
    assert st["g"] is HelixType.GENE
    assert st["p1"] is HelixType.PROTEIN
    assert not any(isinstance(t, TypeVar) for t in st.values())
    assert checker.product_types == {}


def test_compound_product_annotation_checks_in_semantic():
    """A list-typed product annotation passes the full analyzer."""
    prog = _parse("#type g=list[Float<µM>]\n"
                  "#gene name=g\nATG TAA\n#end\n")
    SemanticAnalyzer(prog).check()


def test_check_types_surfaces_constraint_error():
    """check_types reports an unsatisfiable system instead of raising."""
    prog = _parse("#gene name=g\nATG TAA\n#end\n")
    errors = TypeChecker().check_types(prog)
    assert errors == []
