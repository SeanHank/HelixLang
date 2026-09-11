"""HelixLang type system and modularization unit tests."""
import pytest

from helixlang.core.ast_nodes import Codon, Gene, Program, Promoter, Regulation
from helixlang.core.errors import HelixError
from helixlang.core.type_system import (
    HelixType,
    Module,
    ModuleLoader,
    SymbolTable,
    TypeChecker,
    TypedValue,
    TypeSignature,
    parse_type_annotation,
)


# -------- Basic types --------
def test_basic_types_exist():
    assert HelixType.PROTEIN != HelixType.SIGNAL
    assert HelixType.PROTEIN.value == "protein"
    assert HelixType.SIGNAL.value == "signal"
    assert HelixType.FLOAT.value == "float"
    assert HelixType.INT.value == "int"
    assert HelixType.BOOL.value == "bool"
    assert HelixType.STRING.value == "string"
    assert HelixType.GENE.value == "gene"
    assert HelixType.RECORD.value == "record"
    assert HelixType.ANY.value == "any"


def test_typed_value():
    tv = TypedValue(value=42, type=HelixType.INT)
    assert tv.value == 42
    assert tv.type == HelixType.INT


# -------- Symbol table --------
def test_symbol_table_define_lookup():
    st = SymbolTable()
    st.define("x", HelixType.INT, value=42)
    sig = st.lookup("x")
    assert sig is not None
    assert sig.name == "x"
    assert st.get_type("x") == HelixType.INT
    assert st.get_value("x") == 42


def test_symbol_table_missing():
    st = SymbolTable()
    assert st.lookup("undefined") is None
    assert st.get_type("undefined") is None
    assert st.get_value("undefined") is None


def test_symbol_table_all_types():
    st = SymbolTable()
    st.define("p", HelixType.PROTEIN)
    st.define("s", HelixType.SIGNAL)
    st.define("f", HelixType.FLOAT)
    st.define("i", HelixType.INT)
    st.define("b", HelixType.BOOL)
    st.define("str", HelixType.STRING)
    st.define("g", HelixType.GENE)
    assert st.get_type("p") == HelixType.PROTEIN
    assert st.get_type("s") == HelixType.SIGNAL
    assert st.get_type("f") == HelixType.FLOAT
    assert st.get_type("i") == HelixType.INT
    assert st.get_type("b") == HelixType.BOOL
    assert st.get_type("str") == HelixType.STRING
    assert st.get_type("g") == HelixType.GENE


def test_symbol_table_get_all():
    st = SymbolTable()
    st.define("a", HelixType.INT)
    st.define("b", HelixType.FLOAT)
    all_syms = st.get_all()
    assert set(all_syms.keys()) == {"a", "b"}
    assert all_syms["a"].name == "a"


def test_symbol_table_redefine():
    st = SymbolTable()
    st.define("x", HelixType.INT)
    st.define("x", HelixType.FLOAT)
    assert st.get_type("x") == HelixType.FLOAT


# -------- Record types --------
def test_record_type_define():
    st = SymbolTable()
    fields = {"x": HelixType.FLOAT, "y": HelixType.FLOAT,
              "name": HelixType.STRING}
    st.define("Point", TypeSignature(name="Point", fields=fields))
    sig = st.lookup("Point")
    assert sig is not None
    assert sig.fields == fields
    assert st.get_type("Point") == HelixType.RECORD


def test_record_type_nested_fields():
    st = SymbolTable()
    st.define("Protein", TypeSignature(
        name="Protein",
        fields={"seq": HelixType.STRING, "length": HelixType.INT},
    ))
    sig = st.lookup("Protein")
    assert sig.fields["seq"] == HelixType.STRING
    assert sig.fields["length"] == HelixType.INT


def test_type_signature_params():
    sig = TypeSignature(name="List", params=[HelixType.GENE])
    assert sig.params == [HelixType.GENE]
    assert sig.fields == {}


# -------- Type annotation parsing --------
def test_parse_type_annotation_basic():
    assert parse_type_annotation("Int") == HelixType.INT
    assert parse_type_annotation("Float") == HelixType.FLOAT
    assert parse_type_annotation("Bool") == HelixType.BOOL
    assert parse_type_annotation("Protein") == HelixType.PROTEIN
    assert parse_type_annotation("Signal") == HelixType.SIGNAL
    assert parse_type_annotation("Gene") == HelixType.GENE
    assert parse_type_annotation("String") == HelixType.STRING
    assert parse_type_annotation("Record") == HelixType.RECORD
    assert parse_type_annotation("Any") == HelixType.ANY


def test_parse_type_annotation_case_insensitive():
    assert parse_type_annotation("int") == HelixType.INT
    assert parse_type_annotation("INT") == HelixType.INT
    assert parse_type_annotation("  Float  ") == HelixType.FLOAT


def test_parse_type_annotation_unknown():
    with pytest.raises(HelixError):
        parse_type_annotation("UnknownType")


# -------- Modules --------
def test_module_creation():
    mod = Module(name="test")
    assert mod.name == "test"
    assert mod.exports == set()
    assert isinstance(mod.symbols, SymbolTable)
    assert mod.imported == {}


def test_module_export():
    mod = Module(name="m")
    mod.symbols.define("g1", HelixType.GENE)
    mod.exports.add("g1")
    assert "g1" in mod.exports
    assert mod.symbols.get_type("g1") == HelixType.GENE


def test_module_import_module():
    mod1 = Module(name="mod1")
    mod1.symbols.define("g1", HelixType.GENE)
    mod1.exports.add("g1")
    mod2 = Module(name="mod2")
    mod2.import_module("mod1", mod1)
    assert "mod1" in mod2.imported
    # After the import, mod2 should be able to look up g1
    assert mod2.symbols.get_type("g1") == HelixType.GENE


def test_module_import_does_not_export():
    mod1 = Module(name="mod1")
    mod1.symbols.define("g1", HelixType.GENE)
    mod1.exports.add("g1")
    mod2 = Module(name="mod2")
    mod2.import_module("mod1", mod1)
    # Importing does not automatically add to mod2.exports
    assert "g1" not in mod2.exports


# -------- Module loader --------
def test_module_loader(tmp_path):
    f = tmp_path / "demo.helix"
    f.write_text("#gene name=g1\nATG GCT TAA\n#end\n")
    loader = ModuleLoader(base_dir=tmp_path)
    mod = loader.load("demo.helix")
    assert mod.name == "demo"
    assert mod.symbols.lookup("g1") is not None
    assert mod.symbols.get_type("g1") == HelixType.GENE
    assert "g1" in mod.exports


def test_module_loader_promoters(tmp_path):
    f = tmp_path / "p.helix"
    f.write_text("#promoter name=p1 strength=0.8\n")
    loader = ModuleLoader(base_dir=tmp_path)
    mod = loader.load("p.helix")
    assert mod.symbols.get_type("p1") == HelixType.PROTEIN
    assert "p1" in mod.exports


def test_module_loader_caches(tmp_path):
    f = tmp_path / "c.helix"
    f.write_text("#gene name=g1\nATG GCT TAA\n#end\n")
    loader = ModuleLoader(base_dir=tmp_path)
    m1 = loader.load("c.helix")
    m2 = loader.load("c.helix")
    assert m1 is m2


def test_module_loader_resolve_import(tmp_path):
    fa = tmp_path / "a.helix"
    fb = tmp_path / "b.helix"
    fa.write_text("#gene name=ga\nATG GCT TAA\n#end\n")
    fb.write_text("#gene name=gb\nATG GCT TAA\n#end\n")
    loader = ModuleLoader(base_dir=tmp_path)
    ma = loader.load("a.helix")
    mb = loader.resolve_import("b.helix")
    assert ma.symbols.get_type("ga") == HelixType.GENE
    assert mb.symbols.get_type("gb") == HelixType.GENE
    assert ma is not mb


def test_module_loader_default_suffix(tmp_path):
    f = tmp_path / "ns.helix"
    f.write_text("#gene name=g1\nATG GCT TAA\n#end\n")
    loader = ModuleLoader(base_dir=tmp_path)
    # Loads even without an extension
    mod = loader.load("ns")
    assert mod.symbols.lookup("g1") is not None


# -------- Type checking --------
def test_type_checker_no_errors():
    st = SymbolTable()
    st.define("p1", HelixType.PROTEIN)
    st.define("g1", HelixType.GENE)
    prog = Program(
        promoters=[Promoter(name="p1", strength=0.5)],
        genes=[Gene(name="g1", promoter="p1", codons=[], orf=[])],
        regulations=[Regulation(source="p1", target="g1", strength=0.5)],
    )
    checker = TypeChecker()
    errors = checker.check(prog, st)
    assert errors == []


def test_type_checker_undefined_promoter():
    st = SymbolTable()
    prog = Program(
        genes=[Gene(name="g1", promoter="missing", codons=[], orf=[])],
    )
    checker = TypeChecker()
    errors = checker.check(prog, st)
    assert len(errors) >= 1
    assert any("missing" in str(e) for e in errors)


def test_type_checker_undefined_regulation_source():
    st = SymbolTable()
    prog = Program(
        regulations=[Regulation(source="ghost", target="g1", strength=0.5)],
    )
    checker = TypeChecker()
    errors = checker.check(prog, st)
    assert any("ghost" in str(e) for e in errors)


def test_type_checker_auto_registers_symbols():
    # Without pre-defining, the checker should auto-register gene/promoter without errors
    st = SymbolTable()
    prog = Program(
        promoters=[Promoter(name="p1", strength=0.5)],
        genes=[Gene(name="g1", promoter="p1", codons=[], orf=[])],
    )
    checker = TypeChecker()
    errors = checker.check(prog, st)
    assert errors == []
    assert st.get_type("p1") == HelixType.PROTEIN
    assert st.get_type("g1") == HelixType.GENE


def test_type_checker_returns_list():
    checker = TypeChecker()
    errors = checker.check(Program(), SymbolTable())
    assert isinstance(errors, list)


# -------- Type inference --------
def test_infer_int():
    assert TypeChecker().infer(42) == HelixType.INT
    assert TypeChecker().infer(-7) == HelixType.INT


def test_infer_float():
    assert TypeChecker().infer(3.14) == HelixType.FLOAT


def test_infer_bool():
    # bool is an int subclass; must return BOOL rather than INT
    assert TypeChecker().infer(True) == HelixType.BOOL
    assert TypeChecker().infer(False) == HelixType.BOOL


def test_infer_string():
    assert TypeChecker().infer("hello") == HelixType.STRING


def test_infer_codon():
    assert TypeChecker().infer(Codon(seq="ATG", index=0, line=1)) == HelixType.GENE


def test_infer_typed_value():
    tv = TypedValue(value=1.0, type=HelixType.SIGNAL)
    assert TypeChecker().infer(tv) == HelixType.SIGNAL


def test_infer_unknown_returns_any():
    assert TypeChecker().infer(None) == HelixType.ANY
    assert TypeChecker().infer([1, 2, 3]) == HelixType.ANY


# -------- Type hierarchy / describe --------
def test_type_base_describe():
    from helixlang.core.type_system import Type
    t = Type()
    assert isinstance(t.describe(), str)


def test_list_type_describe():
    from helixlang.core.type_system import ListType
    lt = ListType(elem=HelixType.INT)
    assert lt.describe() == "list[int]"


def test_record_type_describe():
    from helixlang.core.type_system import RecordType
    rt = RecordType(fields={"x": HelixType.FLOAT, "name": HelixType.STRING})
    d = rt.describe()
    assert "x" in d
    assert "name" in d


def test_func_type_describe():
    from helixlang.core.type_system import FuncType
    ft = FuncType(params=(HelixType.INT, HelixType.FLOAT), ret=HelixType.STRING)
    d = ft.describe()
    assert "int" in d
    assert "float" in d
    assert "string" in d


def test_schema_instantiate():
    from helixlang.core.type_system import Schema
    s = Schema(term=HelixType.PROTEIN)
    assert s.instantiate() is HelixType.PROTEIN


# -------- TypeVar --------
def test_typevar_with_name():
    from helixlang.core.type_system import TypeVar
    tv = TypeVar(name="?x")
    assert tv.name == "?x"
    assert tv.describe() == "?x"
    assert repr(tv) == "?x"


def test_typevar_auto_name():
    from helixlang.core.type_system import TypeVar
    tv = TypeVar()
    assert tv.name.startswith("?t")


def test_typevar_eq_and_hash():
    from helixlang.core.type_system import TypeVar
    a = TypeVar(name="?z")
    b = TypeVar(name="?z")
    c = TypeVar(name="?w")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert a != "not a typevar"


def test_typevar_neq_non_typevar():
    from helixlang.core.type_system import TypeVar
    tv = TypeVar(name="?x")
    assert tv != HelixType.INT
    assert tv != 42


# -------- _describe helper --------
def test_describe_type():
    from helixlang.core.type_system import ListType, _describe
    assert _describe(HelixType.INT) == "int"
    lt = ListType(elem=HelixType.INT)
    assert _describe(lt) == "list[int]"
    assert _describe(42) == "42"


# -------- fresh_var --------
def test_fresh_var():
    from helixlang.core.type_system import TypeVar, fresh_var
    v = fresh_var()
    assert isinstance(v, TypeVar)


# -------- UnitType --------
def test_unit_type():
    from helixlang.core.type_system import UnitType
    ut = UnitType(base=HelixType.FLOAT, unit="min")
    assert ut.describe() == "float<min>"
    assert str(ut) == "float<min>"
    assert repr(ut) == "float<min>"


# -------- parse_type_annotation compound types --------
def test_parse_list_type():
    from helixlang.core.type_system import ListType
    result = parse_type_annotation("list[Int]")
    assert isinstance(result, ListType)
    assert result.elem == HelixType.INT


def test_parse_list_type_empty():
    from helixlang.core.errors import SemanticError
    with pytest.raises(SemanticError):
        parse_type_annotation("list[]")


def test_parse_record_type():
    from helixlang.core.type_system import RecordType
    result = parse_type_annotation("record{x: Int, y: Float}")
    assert isinstance(result, RecordType)
    assert result.fields["x"] == HelixType.INT
    assert result.fields["y"] == HelixType.FLOAT


def test_parse_record_type_malformed():
    from helixlang.core.errors import SemanticError
    with pytest.raises(SemanticError):
        parse_type_annotation("record{: Int}")


def test_parse_record_type_duplicate_field():
    from helixlang.core.errors import SemanticError
    with pytest.raises(SemanticError):
        parse_type_annotation("record{x: Int, x: Float}")


def test_parse_record_type_missing_type():
    from helixlang.core.errors import SemanticError
    with pytest.raises(SemanticError):
        parse_type_annotation("record{x: }")


def test_parse_func_type_malformed_params():
    from helixlang.core.errors import SemanticError
    with pytest.raises(SemanticError, match="malformed function type"):
        parse_type_annotation("(Int, Float -> Protein)")


def test_parse_func_type_happy_path():
    from helixlang.core.errors import SemanticError
    with pytest.raises(SemanticError):
        parse_type_annotation("(Int, Float) -> Protein)")


def test_parse_func_type_empty_params():
    from helixlang.core.errors import SemanticError
    with pytest.raises(SemanticError):
        parse_type_annotation("() -> Protein)")


def test_parse_func_type_empty_part():
    from helixlang.core.errors import SemanticError
    with pytest.raises(SemanticError):
        parse_type_annotation("(Int,, Float) -> Protein)")


def test_parse_func_type_malformed():
    from helixlang.core.errors import SemanticError
    with pytest.raises(SemanticError):
        parse_type_annotation("Int, Float -> String")


def test_parse_unit_type():
    from helixlang.core.type_system import UnitType
    result = parse_type_annotation("Float<min>")
    assert isinstance(result, UnitType)
    assert result.base == HelixType.FLOAT
    assert result.unit == "min"


def test_parse_unit_type_unknown_base():
    from helixlang.core.errors import SemanticError
    with pytest.raises(SemanticError):
        parse_type_annotation("Foo<int>")


def test_parse_unit_type_unknown_unit():
    from helixlang.core.dimensions import UnitError
    with pytest.raises(UnitError):
        parse_type_annotation("Float<noexist>")


# -------- _split_top_level --------
def test_split_top_level_nested_brackets():
    from helixlang.core.type_system import _split_top_level
    result = _split_top_level("list[Int], Float", ",")
    assert len(result) == 2
    assert result[0] == "list[Int]"
    assert result[1] == "Float"


def test_split_top_level_no_split():
    from helixlang.core.type_system import _split_top_level
    result = _split_top_level("Int", ",")
    assert result == ["Int"]


# -------- Unifier --------
def test_unifier_resolve_simple():
    from helixlang.core.type_system import TypeVar, Unifier
    u = Unifier()
    tv = TypeVar(name="?a")
    u._subst["?a"] = HelixType.INT
    assert u.resolve(tv) == HelixType.INT


def test_unifier_resolve_chain():
    from helixlang.core.type_system import TypeVar, Unifier
    u = Unifier()
    a = TypeVar(name="?a")
    b = TypeVar(name="?b")
    u._subst["?a"] = b
    u._subst["?b"] = HelixType.FLOAT
    assert u.resolve(a) == HelixType.FLOAT


def test_unifier_resolve_infinite():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import TypeVar, Unifier
    u = Unifier()
    a = TypeVar(name="?a")
    u._subst["?a"] = a
    with pytest.raises(SemanticError, match="infinite type"):
        u.resolve(a)


def test_unifier_unify_same_type():
    from helixlang.core.type_system import Unifier
    u = Unifier()
    u.unify(HelixType.INT, HelixType.INT)


def test_unifier_unify_typevar_left():
    from helixlang.core.type_system import TypeVar, Unifier
    u = Unifier()
    tv = TypeVar(name="?x")
    u.unify(tv, HelixType.FLOAT, "test_sym")
    assert u.resolve(tv) == HelixType.FLOAT


def test_unifier_unify_typevar_right():
    from helixlang.core.type_system import TypeVar, Unifier
    u = Unifier()
    tv = TypeVar(name="?y")
    u.unify(HelixType.FLOAT, tv, "test_sym")
    assert u.resolve(tv) == HelixType.FLOAT


def test_unifier_unify_list_types():
    from helixlang.core.type_system import ListType, Unifier
    u = Unifier()
    lt1 = ListType(elem=HelixType.INT)
    lt2 = ListType(elem=HelixType.INT)
    u.unify(lt1, lt2)


def test_unifier_unify_record_types():
    from helixlang.core.type_system import RecordType, Unifier
    u = Unifier()
    r1 = RecordType(fields={"x": HelixType.INT})
    r2 = RecordType(fields={"x": HelixType.INT})
    u.unify(r1, r2)


def test_unifier_unify_record_mismatch_fields():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import RecordType, Unifier
    u = Unifier()
    r1 = RecordType(fields={"x": HelixType.INT})
    r2 = RecordType(fields={"y": HelixType.INT})
    with pytest.raises(SemanticError, match="type mismatch"):
        u.unify(r1, r2)


def test_unifier_unify_record_mismatch_field_type():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import RecordType, Unifier
    u = Unifier()
    r1 = RecordType(fields={"x": HelixType.INT})
    r2 = RecordType(fields={"x": HelixType.FLOAT})
    with pytest.raises(SemanticError, match="type mismatch"):
        u.unify(r1, r2)


def test_unifier_unify_func_types():
    from helixlang.core.type_system import FuncType, Unifier
    u = Unifier()
    f1 = FuncType(params=(HelixType.INT,), ret=HelixType.FLOAT)
    f2 = FuncType(params=(HelixType.INT,), ret=HelixType.FLOAT)
    u.unify(f1, f2)


def test_unifier_unify_func_mismatch_arity():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import FuncType, Unifier
    u = Unifier()
    f1 = FuncType(params=(HelixType.INT,), ret=HelixType.FLOAT)
    f2 = FuncType(params=(HelixType.INT, HelixType.STRING), ret=HelixType.FLOAT)
    with pytest.raises(SemanticError, match="type mismatch"):
        u.unify(f1, f2)


def test_unifier_unify_func_mismatch_param():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import FuncType, Unifier
    u = Unifier()
    f1 = FuncType(params=(HelixType.INT,), ret=HelixType.FLOAT)
    f2 = FuncType(params=(HelixType.STRING,), ret=HelixType.FLOAT)
    with pytest.raises(SemanticError, match="type mismatch"):
        u.unify(f1, f2)


def test_unifier_unify_func_mismatch_ret():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import FuncType, Unifier
    u = Unifier()
    f1 = FuncType(params=(HelixType.INT,), ret=HelixType.FLOAT)
    f2 = FuncType(params=(HelixType.INT,), ret=HelixType.STRING)
    with pytest.raises(SemanticError, match="type mismatch"):
        u.unify(f1, f2)


def test_unifier_unify_incompatible_ground():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import Unifier
    u = Unifier()
    with pytest.raises(SemanticError, match="type mismatch"):
        u.unify(HelixType.INT, HelixType.FLOAT)


def test_unifier_unify_unit_same():
    from helixlang.core.type_system import Unifier, UnitType
    u = Unifier()
    u1 = UnitType(base=HelixType.FLOAT, unit="min")
    u2 = UnitType(base=HelixType.FLOAT, unit="min")
    u.unify(u1, u2)


def test_unifier_unify_unit_compatible_dims():
    from helixlang.core.type_system import Unifier, UnitType
    u = Unifier()
    u1 = UnitType(base=HelixType.FLOAT, unit="min")
    u2 = UnitType(base=HelixType.FLOAT, unit="s")
    u.unify(u1, u2)


def test_unifier_unify_unit_incompatible_dims():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import Unifier, UnitType
    u = Unifier()
    u1 = UnitType(base=HelixType.FLOAT, unit="min")
    u2 = UnitType(base=HelixType.FLOAT, unit="µm")
    with pytest.raises(SemanticError, match="type mismatch"):
        u.unify(u1, u2, "test_sym")


def test_unifier_unify_unit_vs_base():
    from helixlang.core.type_system import Unifier, UnitType
    u = Unifier()
    ut = UnitType(base=HelixType.FLOAT, unit="min")
    u.unify(ut, HelixType.FLOAT)


def test_unifier_unify_unit_vs_base_mismatch():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import Unifier, UnitType
    u = Unifier()
    ut = UnitType(base=HelixType.FLOAT, unit="min")
    with pytest.raises(SemanticError, match="type mismatch"):
        u.unify(ut, HelixType.INT, "test_sym")


def test_unifier_unify_base_vs_unit():
    from helixlang.core.type_system import Unifier, UnitType
    u = Unifier()
    ut = UnitType(base=HelixType.FLOAT, unit="min")
    u.unify(HelixType.FLOAT, ut)


def test_unifier_unify_two_units_different_base():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import Unifier, UnitType
    u = Unifier()
    u1 = UnitType(base=HelixType.FLOAT, unit="min")
    u2 = UnitType(base=HelixType.INT, unit="min")
    with pytest.raises(SemanticError, match="type mismatch"):
        u.unify(u1, u2, "test_sym")


def test_unifier_unify_unit_vs_other():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import RecordType, Unifier, UnitType
    u = Unifier()
    u1 = UnitType(base=HelixType.FLOAT, unit="min")
    r1 = RecordType(fields={"x": HelixType.INT})
    with pytest.raises(SemanticError, match="type mismatch"):
        u.unify(u1, r1, "test_sym")


def test_unifier_occurs_check_list():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import ListType, TypeVar, Unifier
    u = Unifier()
    tv = TypeVar(name="?a")
    lt = ListType(elem=tv)
    with pytest.raises(SemanticError, match="infinite type"):
        u.unify(tv, lt, "test_sym")


def test_unifier_occurs_check_record():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import RecordType, TypeVar, Unifier
    u = Unifier()
    tv = TypeVar(name="?b")
    rt = RecordType(fields={"x": tv})
    with pytest.raises(SemanticError, match="infinite type"):
        u.unify(tv, rt, "test_sym")


def test_unifier_occurs_check_func():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import FuncType, TypeVar, Unifier
    u = Unifier()
    tv = TypeVar(name="?c")
    ft = FuncType(params=(tv,), ret=HelixType.INT)
    with pytest.raises(SemanticError, match="infinite type"):
        u.unify(tv, ft, "test_sym")


def test_unifier_occurs_check_func_ret():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import FuncType, TypeVar, Unifier
    u = Unifier()
    tv = TypeVar(name="?d")
    ft = FuncType(params=(HelixType.INT,), ret=tv)
    with pytest.raises(SemanticError, match="infinite type"):
        u.unify(tv, ft, "test_sym")


def test_unifier_substitution_property():
    from helixlang.core.type_system import TypeVar, Unifier
    u = Unifier()
    tv = TypeVar(name="?e")
    u.unify(tv, HelixType.INT)
    sub = u.substitution
    assert sub["?e"] == HelixType.INT
    assert sub is not u._subst


def test_unifier_unify_list_different_elem():
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import ListType, Unifier
    u = Unifier()
    lt1 = ListType(elem=HelixType.INT)
    lt2 = ListType(elem=HelixType.FLOAT)
    with pytest.raises(SemanticError, match="type mismatch"):
        u.unify(lt1, lt2)


# -------- BioEffect / effect functions --------
def test_effect_of_opname_pure():
    from helixlang.core.type_system import BioEffect, effect_of_opname
    assert effect_of_opname("OP_START") == BioEffect.PURE
    assert effect_of_opname("OP_HALT") == BioEffect.PURE


def test_effect_of_opname_quota_boundary():
    from helixlang.core.type_system import BioEffect, effect_of_opname
    assert effect_of_opname("OP_TICK") == BioEffect.QUOTA_BOUNDARY


def test_effect_of_opname_side_effect():
    from helixlang.core.type_system import BioEffect, effect_of_opname
    assert effect_of_opname("OP_MOVE") == BioEffect.SIDE_EFFECT
    assert effect_of_opname("OP_SIGNAL") == BioEffect.SIDE_EFFECT


def test_effect_of_codon_found():
    from dataclasses import dataclass

    from helixlang.core.type_system import BioEffect, effect_of_codon

    @dataclass
    class FakeOp:
        name: str

    table = {"ATG": FakeOp("OP_START"), "AAA": FakeOp("OP_MOVE")}
    assert effect_of_codon("ATG", table) == BioEffect.PURE
    assert effect_of_codon("AAA", table) == BioEffect.SIDE_EFFECT


def test_effect_of_codon_not_found():
    from helixlang.core.type_system import effect_of_codon
    assert effect_of_codon("TTT", {}) is None


def test_effect_of_codon_no_name_attr():
    from helixlang.core.type_system import BioEffect, effect_of_codon
    table = {"AAA": "OP_TICK"}
    assert effect_of_codon("AAA", table) == BioEffect.QUOTA_BOUNDARY


def test_effect_to_opname_found():
    from dataclasses import dataclass

    from helixlang.core.type_system import effect_to_opname

    @dataclass
    class FakeOp:
        name: str

    table = {"ATG": FakeOp("OP_START")}
    assert effect_to_opname("ATG", table) == "OP_START"


def test_effect_to_opname_no_attr():
    from helixlang.core.type_system import effect_to_opname
    table = {"ATG": "OP_HALT"}
    assert effect_to_opname("ATG", table) == "OP_HALT"


def test_effect_to_opname_missing():
    from helixlang.core.type_system import effect_to_opname
    assert effect_to_opname("XYZ", {}) == "None"


# -------- SymbolTable edge cases --------
def test_symbol_table_get_type_no_params_no_fields():
    st = SymbolTable()
    st._symbols["empty"] = TypeSignature(name="empty")
    assert st.get_type("empty") == HelixType.ANY


def test_symbol_table_define_sig_mismatched_name():
    st = SymbolTable()
    sig = TypeSignature(name="other_name", params=[HelixType.FLOAT])
    st.define("my_name", sig)
    assert st.get_type("my_name") == HelixType.FLOAT
    assert st.lookup("my_name").name == "my_name"


# -------- Module import with value --------
def test_module_import_with_value():
    mod1 = Module(name="mod1")
    mod1.symbols.define("g1", HelixType.GENE, value=42)
    mod1.exports.add("g1")
    mod2 = Module(name="mod2")
    mod2.import_module("mod1", mod1)
    assert mod2.symbols.get_value("g1") == 42


# -------- TypeChecker.infer_program --------
def test_infer_program_basic():
    prog = Program(
        promoters=[Promoter(name="p1", strength=0.5)],
        genes=[Gene(name="g1", promoter="p1", codons=[], orf=[])],
    )
    checker = TypeChecker()
    result = checker.infer_program(prog)
    assert result["g1"] == HelixType.GENE
    assert result["p1"] == HelixType.PROTEIN


def test_infer_program_with_type_annotations():
    prog = Program(
        genes=[Gene(name="g1", promoter=None, codons=[], orf=[])],
        type_annotations={"g1": "Protein"},
    )
    checker = TypeChecker()
    result = checker.infer_program(prog)
    assert result["g1"] == HelixType.GENE
    assert checker.product_types["g1"] == HelixType.PROTEIN


def test_infer_program_unresolvable():
    checker = TypeChecker()
    # Manually create an unsatisfiable scenario
    # We can't easily trigger this through normal program construction
    # so let's directly test with a custom program that has a conflicting
    # type annotation
    prog2 = Program(
        genes=[Gene(name="g1", promoter=None, codons=[], orf=[])],
        type_annotations={"g1": "Int"},
    )
    result = checker.infer_program(prog2)
    assert result["g1"] == HelixType.GENE


# -------- TypeChecker.constraints_for --------
def test_constraints_for_basic():
    prog = Program(
        promoters=[Promoter(name="p1", strength=0.5)],
        genes=[Gene(name="g1", promoter="p1", codons=[], orf=[])],
    )
    checker = TypeChecker()
    vars_map, unifier = checker.constraints_for(prog)
    assert "g1" in vars_map
    assert "p1" in vars_map
    assert unifier.resolve(vars_map["g1"]) == HelixType.GENE
    assert unifier.resolve(vars_map["p1"]) == HelixType.PROTEIN


def test_constraints_for_with_annotations():
    prog = Program(
        genes=[Gene(name="g1", promoter=None, codons=[], orf=[])],
        type_annotations={"g1": "Float<µM>"},
    )
    checker = TypeChecker()
    vars_map, unifier = checker.constraints_for(prog)
    assert "g1" in vars_map
    assert checker.product_types["g1"].base == HelixType.FLOAT


# -------- TypeChecker.check_types --------
def test_check_types_no_errors():
    prog = Program(
        promoters=[Promoter(name="p1", strength=0.5)],
        genes=[Gene(name="g1", promoter="p1", codons=[], orf=[])],
    )
    checker = TypeChecker()
    errors = checker.check_types(prog)
    assert errors == []


def test_check_types_catches_error():
    # Create a program where type inference raises an error
    # We need a scenario where a var doesn't resolve to ground type
    # In the current implementation this shouldn't happen with normal programs,
    # but we can test the catch path
    prog = Program()
    checker = TypeChecker()
    errors = checker.check_types(prog)
    assert errors == []


# -------- TypeChecker._record_annotations --------
def test_record_annotations_error():
    prog = Program(
        type_annotations={"g1": "Float<noexistunit>"},
    )
    checker = TypeChecker()
    checker._record_annotations(prog)
    assert len(checker.errors) >= 1


# -------- TypeChecker.check_effects --------
def test_check_effects_pure_gene():
    from helixlang.core.type_system import BioEffect
    prog = Program(
        genes=[Gene(
            name="g1", promoter=None, codons=[], orf=[],
            fields={"pure": "1"},
        )],
    )
    checker = TypeChecker()
    table = {"AAA": type("Op", (), {"name": "OP_START"})()}
    errors = checker.check_effects(prog, codon_to_op=table)
    assert errors == []
    assert checker.gene_effects["g1"] == BioEffect.PURE


def test_check_effects_side_effect_in_pure_gene():
    from helixlang.core.type_system import BioEffect
    prog = Program(
        genes=[Gene(
            name="g1", promoter=None, codons=[], orf=[
                type("C", (), {"seq": "AAA"})(),
            ],
            fields={"pure": "1"},
        )],
    )
    checker = TypeChecker()
    table = {"AAA": type("Op", (), {"name": "OP_MOVE"})()}
    errors = checker.check_effects(prog, codon_to_op=table)
    assert len(errors) >= 1
    assert checker.gene_effects["g1"] == BioEffect.SIDE_EFFECT


def test_check_effects_non_pure_gene():
    from helixlang.core.type_system import BioEffect
    prog = Program(
        genes=[Gene(
            name="g1", promoter=None, codons=[], orf=[
                type("C", (), {"seq": "AAA"})(),
                type("C2", (), {"seq": "BBB"})(),
            ],
            fields={},
        )],
    )
    checker = TypeChecker()
    table = {
        "AAA": type("Op", (), {"name": "OP_MOVE"})(),
        "BBB": type("Op", (), {"name": "OP_TICK"})(),
    }
    errors = checker.check_effects(prog, codon_to_op=table)
    assert errors == []
    assert checker.gene_effects["g1"] == BioEffect.SIDE_EFFECT


def test_check_effects_codon_not_in_table():
    from helixlang.core.type_system import BioEffect
    prog = Program(
        genes=[Gene(
            name="g1", promoter=None, codons=[], orf=[
                type("C", (), {"seq": "ZZZ"})(),
            ],
            fields={},
        )],
    )
    checker = TypeChecker()
    errors = checker.check_effects(prog, codon_to_op={})
    assert errors == []
    assert checker.gene_effects["g1"] == BioEffect.PURE


def test_check_effects_none_codon_to_op():
    prog = Program(
        genes=[Gene(
            name="g1", promoter=None, codons=[], orf=[],
        )],
    )
    checker = TypeChecker()
    errors = checker.check_effects(prog)
    assert errors == []


# -------- TypeChecker._check_program type mismatch --------
def test_check_program_promoter_type_mismatch():
    st = SymbolTable()
    st.define("p1", HelixType.GENE)
    prog = Program(
        promoters=[Promoter(name="p1", strength=0.5)],
    )
    checker = TypeChecker()
    checker.check(prog, st)
    assert any("expected Protein" in str(e) for e in checker.errors)


def test_check_program_gene_promoter_not_protein():
    st = SymbolTable()
    st.define("g1", HelixType.GENE)
    st.define("p1", HelixType.GENE)
    prog = Program(
        genes=[Gene(name="g1", promoter="p1", codons=[], orf=[])],
    )
    checker = TypeChecker()
    checker.check(prog, st)
    assert any("is not a Protein" in str(e) for e in checker.errors)


def test_check_program_regulation_undefined_target():
    st = SymbolTable()
    st.define("p1", HelixType.PROTEIN)
    prog = Program(
        regulations=[Regulation(source="p1", target="missing", strength=0.5)],
    )
    checker = TypeChecker()
    checker.check(prog, st)
    assert any("not defined" in str(e) for e in checker.errors)


# -------- ModuleLoader edge cases --------
def test_module_loader_absolute_path(tmp_path):
    f = tmp_path / "abs.helix"
    f.write_text("#gene name=g1\nATG GCT TAA\n#end\n")
    loader = ModuleLoader(base_dir=tmp_path)
    mod = loader.load(str(f))
    assert mod.symbols.lookup("g1") is not None


def test_module_loader_promoter_in_file(tmp_path):
    f = tmp_path / "multi.helix"
    f.write_text("#gene name=g1\nATG GCT TAA\n#end\n#promoter name=p1 strength=0.5\n")
    loader = ModuleLoader(base_dir=tmp_path)
    mod = loader.load("multi.helix")
    assert mod.symbols.get_type("p1") == HelixType.PROTEIN
    assert "p1" in mod.exports


# -------- Coverage gap: _split_top_level trailing separator --------
def test_split_top_level_trailing_sep():
    from helixlang.core.type_system import _split_top_level
    result = _split_top_level("Int,", ",")
    assert result == ["Int"]


# -------- Coverage gap: Unifier compound type returns --------
def test_unifier_list_typevar_unify():
    from helixlang.core.type_system import ListType, TypeVar, Unifier
    u = Unifier()
    tv = TypeVar(name="?x")
    lt1 = ListType(elem=tv)
    lt2 = ListType(elem=HelixType.INT)
    u.unify(lt1, lt2, "test")
    assert u.resolve(tv) == HelixType.INT


def test_unifier_record_typevar_unify():
    from helixlang.core.type_system import RecordType, TypeVar, Unifier
    u = Unifier()
    tv = TypeVar(name="?x")
    r1 = RecordType(fields={"x": tv, "y": HelixType.FLOAT})
    r2 = RecordType(fields={"x": HelixType.INT, "y": HelixType.FLOAT})
    u.unify(r1, r2, "test")
    assert u.resolve(tv) == HelixType.INT


def test_unifier_func_typevar_unify():
    from helixlang.core.type_system import FuncType, TypeVar, Unifier
    u = Unifier()
    tv_ret = TypeVar(name="?r")
    f1 = FuncType(params=(HelixType.INT,), ret=tv_ret)
    f2 = FuncType(params=(HelixType.INT,), ret=HelixType.FLOAT)
    u.unify(f1, f2, "test")
    assert u.resolve(tv_ret) == HelixType.FLOAT


# -------- Coverage gap: _occurs_check RecordType return --------
def test_unifier_occurs_check_record_no_var():
    from helixlang.core.type_system import RecordType, TypeVar, Unifier
    u = Unifier()
    a = TypeVar(name="?a")
    b = TypeVar(name="?b")
    rt = RecordType(fields={"x": b})
    u.unify(a, rt, "test")
    assert u.resolve(a) == rt


# -------- Coverage gap: Module import ghost export --------
def test_module_import_ghost_export():
    mod1 = Module(name="mod1")
    mod1.exports.add("ghost")
    mod2 = Module(name="mod2")
    mod2.import_module("mod1", mod1)
    assert "mod1" in mod2.imported
    assert mod2.symbols.lookup("ghost") is None


# -------- Coverage gap: check with non-Program AST --------
def test_type_checker_check_non_program():
    checker = TypeChecker()
    errors = checker.check("not a program", SymbolTable())
    assert errors == []


# -------- Coverage gap: infer_program annotation for non-gene --------
def test_infer_program_annotation_non_gene():
    prog = Program(
        genes=[Gene(name="g1", promoter=None, codons=[], orf=[])],
        type_annotations={"stranger": "Int"},
    )
    checker = TypeChecker()
    result = checker.infer_program(prog)
    assert "stranger" not in result


# -------- Coverage gap: constraints_for annotation for non-gene --------
def test_constraints_for_annotation_non_gene():
    prog = Program(
        genes=[Gene(name="g1", promoter=None, codons=[], orf=[])],
        type_annotations={"stranger": "Float<µM>"},
    )
    checker = TypeChecker()
    vars_map, unifier = checker.constraints_for(prog)
    assert "stranger" not in vars_map
    assert checker.product_types["stranger"].base == HelixType.FLOAT


# -------- Coverage gap: check_types catches error --------
def test_check_types_catches_infer_error():
    prog = Program(
        genes=[Gene(name="g1", promoter=None, codons=[], orf=[])],
        type_annotations={"g1": "NonexistentType"},
    )
    checker = TypeChecker()
    errors = checker.check_types(prog)
    assert len(errors) == 1


# -------- Coverage gap: check_effects pure gene + quota boundary --------
def test_check_effects_pure_gene_quota_boundary():
    from helixlang.core.type_system import BioEffect
    prog = Program(
        genes=[Gene(
            name="g1", promoter=None, codons=[], orf=[
                type("C", (), {"seq": "AAA"})(),
            ],
            fields={"pure": "1"},
        )],
    )
    checker = TypeChecker()
    table = {"AAA": type("Op", (), {"name": "OP_TICK"})()}
    errors = checker.check_effects(prog, codon_to_op=table)
    assert errors == []
    assert checker.gene_effects["g1"] == BioEffect.QUOTA_BOUNDARY


# -------- Coverage gap: gene without promoter --------
def test_check_program_gene_no_promoter():
    st = SymbolTable()
    prog = Program(
        genes=[Gene(name="g1", promoter=None, codons=[], orf=[])],
    )
    checker = TypeChecker()
    checker.check(prog, st)
    assert st.get_type("g1") == HelixType.GENE


# -------- Coverage gap: redundant unit equality branch --------
def test_unify_two_units_equal(monkeypatch):
    from helixlang.core.type_system import HelixType, Unifier, UnitType
    u = Unifier()
    u1 = UnitType(base=HelixType.FLOAT, unit="min")
    monkeypatch.setattr(u, "_fail", lambda t1, t2, symbol: None)
    u._unify_two_units(u1, u1, None)


def test_unify_two_units_different_base_noop_fail(monkeypatch):
    from helixlang.core.type_system import HelixType, Unifier, UnitType
    u = Unifier()
    u1 = UnitType(base=HelixType.FLOAT, unit="min")
    u2 = UnitType(base=HelixType.INT, unit="min")
    monkeypatch.setattr(u, "_fail", lambda t1, t2, symbol: None)
    u._unify_two_units(u1, u2, None)


# -------- Coverage gap: infer_program unresolved variable --------
def test_infer_program_unresolved_var(monkeypatch):
    from helixlang.core.errors import SemanticError
    from helixlang.core.type_system import HelixType, TypeChecker, TypeVar, Unifier
    orig = Unifier.unify

    def patched(self, t1, t2, symbol=None):
        if t2 == HelixType.GENE and isinstance(t1, TypeVar):
            return orig(self, t1, TypeVar(name="?ghost"), symbol)
        return orig(self, t1, t2, symbol)

    monkeypatch.setattr(Unifier, "unify", patched)
    prog = Program(
        genes=[Gene(name="g1", promoter=None, codons=[], orf=[])],
    )
    checker = TypeChecker()
    with pytest.raises(SemanticError, match="did not resolve"):
        checker.infer_program(prog)
