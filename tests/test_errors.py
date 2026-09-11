"""Core error hierarchy + ast_nodes.quantity coverage."""
from __future__ import annotations

import pytest

from helixlang.core import ast_nodes
from helixlang.core.dimensions import Quantity
from helixlang.core.errors import (
    ABIVersionError,
    BioError,
    CompileError,
    DimensionError,
    HelixError,
    LexError,
    ModelMissingError,
    NativeBackendError,
    ParseError,
    PluginConflictError,
    PluginDependencyError,
    PluginError,
    PluginMissingError,
    RegulationError,
    RuntimeHelixError,
    SemanticError,
    SemanticVersionError,
    SimConfigError,
    StackUnderflowError,
    UnknownKeywordError,
    UnknownNodeError,
    UnknownOpcodeError,
)


def test_error_str_with_line_and_codon():
    e = LexError("boom", line=3, col=44, codon_index=7)
    assert "[LexError @" in str(e)
    assert "line 3" in str(e)
    assert "codon #7" in str(e)
    assert e.msg == "boom"


def test_error_str_unknown_location_no_codon():
    e = ParseError("x")
    assert "<unknown>" in str(e)
    assert "codon #" not in str(e)


def test_exception_hierarchy():
    assert issubclass(LexError, HelixError)
    assert issubclass(ParseError, HelixError)
    assert issubclass(SemanticError, HelixError)
    assert issubclass(DimensionError, SemanticError)
    assert issubclass(CompileError, HelixError)
    assert issubclass(RegulationError, HelixError)
    assert issubclass(RuntimeHelixError, HelixError)
    assert issubclass(BioError, HelixError)
    assert issubclass(SimConfigError, HelixError)
    assert issubclass(UnknownKeywordError, SemanticError)
    assert issubclass(StackUnderflowError, RuntimeHelixError)
    assert issubclass(UnknownNodeError, RuntimeHelixError)
    assert issubclass(UnknownOpcodeError, RuntimeHelixError)
    assert issubclass(PluginError, HelixError)
    assert issubclass(PluginMissingError, PluginError)
    assert issubclass(PluginDependencyError, PluginError)
    assert issubclass(PluginConflictError, PluginError)
    assert issubclass(ModelMissingError, HelixError)
    assert issubclass(ABIVersionError, HelixError)
    assert issubclass(SemanticVersionError, HelixError)
    assert issubclass(NativeBackendError, HelixError)


def test_unknown_opcode_error_defers_to_default_message():
    e = UnknownOpcodeError(0xAB, 4)
    assert "0xab" in str(e) or "0xAB" in str(e)
    assert "ip=4" in str(e)
    assert e.opcode == 0xAB
    assert e.ip == 4


def test_unknown_opcode_error_custom_message():
    e = UnknownOpcodeError(0x01, 9, msg="custom")
    assert e.msg == "custom"
    assert e.ip == 9


def test_plugin_missing_error_message_and_fields():
    e = PluginMissingError("grn", "grn")
    assert "grn" in str(e)
    assert "pip install helixlang[grn]" in str(e)
    assert e.name == "grn" and e.extra == "grn"


def test_plugin_dependency_error_message_and_fields():
    e = PluginDependencyError("foo", "numpy", "extras")
    assert "numpy" in str(e)
    assert "pip install helixlang[extras]" in str(e)
    assert e.name == "foo" and e.dep == "numpy" and e.extra == "extras"


def test_plugin_conflict_error_message_and_fields():
    e = PluginConflictError("#gene", "a", "b")
    assert "'#gene'" in str(e)
    assert e.key == "#gene" and e.first == "a" and e.second == "b"


def test_model_missing_error_message_and_fields():
    e = ModelMissingError("iJO1366", "fba")
    assert "iJO1366" in str(e)
    assert "pip install helixlang[fba]" in str(e)
    assert e.model == "iJO1366" and e.extra == "fba"


def test_abi_version_error_message_and_fields():
    e = ABIVersionError(2, 3)
    assert "OPCODE_VERSION" in str(e)
    assert e.expected == 2 and e.got == 3


def test_semantic_version_error_message_and_fields():
    e = SemanticVersionError("LANGUAGE_SPEC", 1, 2)
    assert "LANGUAGE_SPEC mismatch" in str(e)
    assert e.surface == "LANGUAGE_SPEC" and e.expected == 1 and e.got == 2


def test_native_backend_error_with_and_without_rebuild():
    b = NativeBackendError("bad so")
    assert "Refit" not in str(b) or "pure-python" in str(b)
    assert b.rebuild == ""
    bak = NativeBackendError("bad so", rebuild="make")
    assert "pure-python" in str(bak)
    assert bak.rebuild == "make"


def test_config_quantity_convert():
    cfg = ast_nodes.Config()
    cfg.quantities = {"dur": Quantity(5, "min")}
    q = cfg.quantity("dur", "s")
    assert q.value == pytest.approx(300.0)


def test_config_quantity_missing_key_raises_key_error():
    cfg = ast_nodes.Config()
    with pytest.raises(KeyError):
        cfg.quantity("nope", "s")


def test_gene_block_digest_hashes_fields_and_orf():
    from helixlang.core.ast_nodes import Codon, Gene, gene_block_digest
    codon = Codon(seq="ATG", index=0, line=1)
    gene = Gene(name="g", promoter=None, codons=[codon], orf=[codon],
                fields={"a": "1", "b": "2"})
    d1 = gene_block_digest(gene)
    d2 = gene_block_digest(gene)
    assert d1.startswith("sha256:")
    assert d1 == d2
    # changing a field changes the digest
    codon2 = Codon(seq="ATG", index=0, line=1)
    gene2 = Gene(name="g", promoter=None, codons=[codon2], orf=[codon2],
                 fields={"a": "1", "b": "3"})
    assert gene_block_digest(gene2) != d1


def test_program_extensions_lazy_view():
    prog = ast_nodes.Program()
    ext1 = prog.extensions
    ext2 = prog.extensions
    assert ext1 is ext2
