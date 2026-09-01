"""Phase E1: public plugin surface, manifest, backend registry (§6.2–6.8)."""
from __future__ import annotations

from pathlib import Path

import pytest

import helixlang.api as api
from helixlang.api import ast as api_ast
from helixlang.api import backend as api_backend
from helixlang.api import capabilities as api_caps
from helixlang.api import errors as api_errors
from helixlang.api import grammar as api_grammar
from helixlang.api import language as api_language
from helixlang.api import registry as api_registry
from helixlang.core import find_core_imports
from helixlang.core.errors import PluginError
from helixlang.core.manifest import (
    ManifestProvides,
    PluginManifest,
    discover_manifests,
    load_manifest,
    manifest_matches_grammars,
    parse_manifest,
)


def test_import_surface_modules():
    assert set(api.__all__) == {"accel", "ast", "backend", "bytecode",
                                "capabilities", "compiler", "dimensions",
                                "errors", "gem", "grammar", "language",
                                "registry", "sbol", "units"}
    assert api_ast.ProgramView is not None
    assert api_backend.SimResult is not None
    assert api_backend.Backend is not None


def test_surface_reexports_are_identity():
    from helixlang.core.ast_nodes import Program
    from helixlang.core.bytecode import Chunk
    from helixlang.core.compiler import Compiler
    from helixlang.core.fidelity import opt_in
    from helixlang.core.lexer import Lexer
    from helixlang.core.opcode_semantics import BIND_LEVEL_BOOST
    from helixlang.core.parser import Parser
    from helixlang.core.semantic import SemanticAnalyzer
    assert api.ast.Program is Program
    assert api.bytecode.Chunk is Chunk
    assert api.bytecode.BIND_LEVEL_BOOST is BIND_LEVEL_BOOST
    assert api.compiler.Compiler is Compiler
    assert api.compiler.Lexer is Lexer
    assert api.compiler.Parser is Parser
    assert api.compiler.SemanticAnalyzer is SemanticAnalyzer
    assert api.capabilities.opt_in is opt_in
    assert api.sbol.SBOL_ROLE_GENE
    assert api.registry.grammar_registry is not None
    assert callable(api.gem.set_gem_medium)
    assert callable(api.accel.grn_step)


def test_language_is_the_same_class():
    from helixlang.core.language import LanguageConfig as CoreLanguageConfig
    assert api_language.LanguageConfig is CoreLanguageConfig
    assert api_grammar.AnnotationGrammar
    assert api_registry.Registry is not None
    assert api_registry.PluginProvider is not None
    assert api_errors.UnitError.__name__ == "UnitError"


def test_manifest_parses_human_style():
    m = parse_manifest("""
name = "human"
version = "1.0.0"
entry_point = "helixlang.plugins.human"
abi_version = 1

[provides]
grammars = ["person", "trait", "disease"]
ast    = ["human_profile"]
ir     = []
backends = ["human", "human_virtual_patient"]

[capabilities]
flags = ["--low-fidelity"]

[requires]
pip = ["numpy>=1.24", "pandas", "rdkit"]
""")
    assert m.name == "human" and m.abi_version == 1
    assert m.provides.grammars == ("person", "trait", "disease")
    assert (m.provides.ast, m.provides.ir) == (("human_profile",), ())
    assert m.provides.backends == ("human", "human_virtual_patient")
    assert m.capability_flags == ("--low-fidelity",)
    assert m.requires_pip == ("numpy>=1.24", "pandas", "rdkit")
    assert m.native_module is None
    assert m.entry_point == "helixlang.plugins.human"


def test_manifest_roundtrips():
    m = parse_manifest("""
name = "ecosystem"
version = "2.1.0"
entry_point = "helixlang.plugins.ecosystem"
abi_version = 2
[native]
module = "helixlang._accel.ecosystem_step"
rebuild = "python -m helixlang._accel.build"
""")
    d = m.to_dict()
    assert d["name"] == "ecosystem"
    assert d["capabilities"] == {"flags": []}
    assert d["native"] == {"module": "helixlang._accel.ecosystem_step",
                           "rebuild": "python -m helixlang._accel.build"}


def test_manifest_rejects_malformed():
    with pytest.raises(PluginError):
        parse_manifest("name = 5")
    with pytest.raises(PluginError):
        parse_manifest("name = 'x'\nversion = '1'\nentry_point = 'y'\n"
                       "abi_version = 0")
    with pytest.raises(PluginError):
        parse_manifest("name = 'x'\nversion = '1'\nentry_point = 'y'\n"
                       "[provides]\nbackends = 3")


def test_manifest_file_and_discovery(tmp_path):
    d = tmp_path / "pkgs" / "human"
    d.mkdir(parents=True)
    (d / "helix.plugin.toml").write_text(
        "name = 'human'\nversion = '1'\nentry_point = 'h'\n")
    found = discover_manifests(tmp_path / "pkgs")
    assert len(found) == 1 and found[0].name == "human"
    loaded = load_manifest(d / "helix.plugin.toml")
    assert loaded.source == str(d / "helix.plugin.toml")
    assert discover_manifests(tmp_path / "missing") == []


def test_manifest_grammar_drift():
    from helixlang.core.grammar_registry import AnnotationGrammar
    grammars = {"person": AnnotationGrammar(keyword="person", owner="human")}
    drift = PluginManifest(name="drift", version="1", entry_point="x",
                           provides=ManifestProvides(grammars=("person",)))
    with pytest.raises(PluginError):
        manifest_matches_grammars(drift, grammars)
    ok = PluginManifest(name="human", version="1", entry_point="x",
                        provides=ManifestProvides(grammars=("person",)))
    manifest_matches_grammars(ok, grammars)


def test_bundled_cardiology_manifest_contract():
    """Phase E1: the shipped cardiology manifest parses + matches its grammars."""
    from helixlang.core.grammar_registry import grammar_registry
    root = Path(__file__).resolve().parents[1] / "src" / "helixlang" / "plugins"
    manifests = discover_manifests(root)
    cardio = {m.name: m for m in manifests}.get("cardiology")
    assert cardio is not None, [m.name for m in manifests]
    assert cardio.entry_point == "helixlang.plugins.cardiology"
    assert cardio.provides.grammars == ("cardiac_cycle",)
    assert "cardiology" in cardio.provides.backends

    api_registry.Registry().discover("cardiology")
    registered = {g.keyword: g for g in grammar_registry.grammars()
                  if g.keyword in cardio.provides.grammars}
    assert set(registered) == set(cardio.provides.grammars)
    manifest_matches_grammars(cardio, registered)


class _DummyBackend(api_backend.Backend):
    id = "dummy"
    kinds = ("d", "test")

    def run(self, req):  # noqa: D102
        return api_backend.SimResult(backend="dummy", columns=[], rows=[])


def test_backend_registry_resolve():
    reg = api_backend.BackendRegistry()
    reg.register(_DummyBackend())
    assert reg.resolve(kind="d").id == "dummy"
    assert reg.resolve(backend="dummy").id == "dummy"
    assert reg.has(kind="d") and reg.has(backend="dummy")
    assert not reg.has(backend="nope")
    assert reg.ids() == ["dummy"]
    with pytest.raises(api_errors.PluginMissingError):
        reg.resolve(kind="absent")


def test_backend_registry_conflict():
    reg = api_backend.BackendRegistry()
    reg.register(_DummyBackend())

    class _Clash(api_backend.Backend):
        id = "dummy"
        kinds = ()
        def run(self, req):
            return api_backend.SimResult(backend="dummy", columns=[], rows=[])

    class _AliasClash(api_backend.Backend):
        id = "other"
        kinds = ("d",)
        def run(self, req):
            return api_backend.SimResult(backend="other", columns=[], rows=[])

    with pytest.raises(api_errors.PluginConflictError):
        reg.register(_Clash())
    with pytest.raises(api_errors.PluginConflictError):
        reg.register(_AliasClash())


def test_ast_contract_types():
    assert api_ast.FieldType.INT == 1
    field = api_ast.SectionField("drugs", api_ast.FieldType.LIST)
    assert field.key == "drugs"
    ext = api_ast.ASTExtension(
        id="human_profile", grammars=("person",),
        parse=lambda builder, ext: None, fields=(field,))
    assert ext.abi_version == 1
    assert api_ast.ProgramView is not None
    assert api_ast.ProgramBuilder is not None


def test_capability():
    c = api_caps.Capability(id="--low-fidelity", summary="coarse ODE",
                            reduces_fidelity=True)
    assert c.reduces_fidelity
    assert api_caps.PURE_PYTHON == "--pure-python"


def test_ir_contract_types():
    from helixlang.api.ir import IRExtension, OperandMode, OperandSlot
    schema = (OperandSlot("target", OperandMode.LABEL),)
    ext = IRExtension(id="ecosystem", kinds=("ecosystem.seed",),
                      build=None, execute=None, operand_schema=schema)
    assert ext.kinds == ("ecosystem.seed",)
    assert schema[0].mode == OperandMode.LABEL


def test_import_scanner_fixture(tmp_path):
    (tmp_path / "_sibling.py").write_text("x = 1\n")
    _write_plugin(tmp_path, "thirdparty.py", "import numpy as np\n")
    _write_plugin(tmp_path, "via_api.py", "from helixlang.api.backend import Backend\n")
    _write_plugin(tmp_path, "relative.py", "from ._sibling import x\n")
    _write_plugin(tmp_path, "via_errors.py", "from helixlang.core.errors import BioError\n")
    evil = _write_plugin(tmp_path, "knownexc.py",
                         "from helixlang.core.ast_nodes import Program\n")
    hard = _write_plugin(tmp_path, "hardexc.py",
                         "from helixlang.sim_runtime._engine import run\n")

    v = find_core_imports.scan([tmp_path])
    modules = {x.module for x in v}
    assert "helixlang.core.ast_nodes" in modules
    assert "helixlang.sim_runtime._engine" in modules
    assert "helixlang.api.backend" not in modules
    assert "helixlang.core.errors" not in modules
    assert "numpy" not in modules
    by_path = {x.path: x.known for x in v}
    assert by_path[evil] is False and by_path[hard] is False


def _write_plugin(dirpath: Path, name: str, body: str) -> Path:
    f = dirpath / name
    f.write_text(body)
    return f


def test_import_scanner_live_tree():
    from helixlang.core import find_core_imports as fci
    root = Path(__file__).resolve().parents[1] / "src" / "helixlang" / "plugins"
    v = find_core_imports.scan([root])
    known = {x.module for x in v if x.known}
    hard = {x.module for x in v if not x.known}
    assert known == set(fci.KNOWN_COMPLIANT_EXCEPTIONS)
    assert hard == set()


def test_api_import_has_no_heavy_deps():
    import subprocess
    import sys
    res = subprocess.run(
        [sys.executable, "-c",
         "import helixlang.api, sys\n"
         "assert not {'numpy','pandas','rdkit','scipy'} & set(sys.modules)"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
