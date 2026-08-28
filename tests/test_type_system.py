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
