"""Tests for the Phase 1 core restructure (doc/36 §3): errors, plugin
registry, use-statement model, and the silent-fallback linter."""
from __future__ import annotations

from pathlib import Path

import pytest

import helixlang
from helixlang.core import (
    KNOWN_FLAGS,
    ABIVersionError,
    ModelMissingError,
    NativeBackendError,
    PluginConflictError,
    PluginDependencyError,
    PluginMissingError,
    StackUnderflowError,
    UnknownKeywordError,
    UnknownNodeError,
    parse_use_line,
)
from helixlang.core.find_silent_fallbacks import scan_file, scan_tree
from helixlang.core.plugin_registry import NativeBackend, PluginProvider, Registry
from helixlang.core.use_stmt import UseError


def _provider(name: str, extra: str, keywords=()) -> PluginProvider:
    return PluginProvider(name=name, extra=extra, keywords=tuple(keywords))


# ── errors ──────────────────────────────────────────────────────────────────


def test_error_hierarchy_membership():
    from helixlang.core.errors import (
        PluginError,
        RuntimeHelixError,
        SemanticError,
    )
    for cls in (PluginMissingError, PluginDependencyError, PluginConflictError):
        assert issubclass(cls, PluginError)
    assert issubclass(UnknownKeywordError, SemanticError)
    for cls in (StackUnderflowError, UnknownNodeError):
        assert issubclass(cls, RuntimeHelixError)


def test_plugin_missing_message_names_extra():
    e = PluginMissingError("grn", "grn")
    assert "pip install helixlang[grn]" in str(e)


def test_dependency_error_suggests_flag():
    e = PluginDependencyError("metabolism", "scipy", "fast")
    assert "--low-fidelity" in str(e)


def test_abi_and_native_errors():
    assert "OPCODE_VERSION" in str(ABIVersionError(5, 4))
    assert "--pure-python" in str(NativeBackendError("failed", rebuild="make"))


def test_model_missing():
    assert "iML1515" in str(ModelMissingError("iML1515", "fast"))


# ── use-statement model ─────────────────────────────────────────────────────


def test_parse_use_simple():
    d = parse_use_line("grn")
    assert d.plugin == "grn" and d.flags == set()


def test_parse_use_with_flags():
    d = parse_use_line("grn --pure-python --approx-euler")
    assert d.plugin == "grn"
    assert d.flags == {"--pure-python", "--approx-euler"}


def test_parse_use_unknown_flag():
    with pytest.raises(UseError):
        parse_use_line("grn --magic")


def test_parse_use_no_plugin():
    with pytest.raises(UseError):
        parse_use_line("")


def test_known_flags_complete():
    assert {"--pure-python", "--approx-euler", "--low-fidelity"} == set(KNOWN_FLAGS)


def test_parse_use_rejects_invalid_plugin():
    with pytest.raises(UseError):
        parse_use_line("not-a-valid-ident")


# ── plugin registry ─────────────────────────────────────────────────────────


def test_register_and_lookup():
    r = Registry()
    r.register(_provider("grn", "grn", keywords=("regulate",)))
    assert r.provider_for_keyword("regulate").name == "grn"
    assert r.is_registered("grn")
    with pytest.raises(PluginMissingError):
        r.provider("nope")


def test_keyword_conflict():
    r = Registry()
    r.register(_provider("a", "a", keywords=("regulate",)))
    with pytest.raises(PluginConflictError):
        r.register(_provider("b", "b", keywords=("regulate",)))


def test_native_backend_conflict():
    r = Registry()
    r.register(PluginProvider(
        "a", "a", native=NativeBackend("helixlang._accel.sim")))
    with pytest.raises(PluginConflictError):
        r.register(PluginProvider(
            "b", "b", native=NativeBackend("helixlang._accel.sim")))


def test_activate_checks_dependencies():
    r = Registry()
    p = PluginProvider("metabolism", "fast")
    p.checks["scipy"] = lambda: False
    r.register(p)
    with pytest.raises(PluginDependencyError):
        r.activate("metabolism")


def test_activate_waives_dep_with_low_fidelity():
    r = Registry()
    p = PluginProvider("metabolism", "fast",
                       capability_flags=("--low-fidelity",),
                       load=lambda: object())
    p.checks["scipy"] = lambda: False
    r.register(p)
    r.declare_capability("--low-fidelity")
    # no dependency error: explicit reduced-fidelity opt-in declared
    obj = r.activate("metabolism")
    assert obj is not None


def test_capability_flags_isolated_per_registry():
    r1, r2 = Registry(), Registry()
    r1.declare_capability("--approx-euler")
    assert r1.has_capability("--approx-euler")
    assert not r2.has_capability("--approx-euler")


# ── silent-fallback linter ──────────────────────────────────────────────────


def _write(tmp_path, code: str):
    p = tmp_path / "mod.py"
    p.write_text(code)
    return p


def test_linter_flags_bare_except(tmp_path):
    p = _write(tmp_path, "try:\n    x = 1\nexcept Exception:\n    pass\n")
    findings = scan_file(p)
    assert findings and "swallow" in findings[0].detail


def test_linter_flags_fallback_string(tmp_path):
    p = _write(tmp_path, "backend = 'scipy_fallback'\n")
    findings = scan_file(p)
    assert findings and findings[0].category == "F4"


def test_linter_clean_file(tmp_path):
    p = _write(tmp_path, "def f(x):\n    return x + 1\n")
    assert scan_file(p) == []


def test_linter_core_and_accel_audit_is_clean():
    """doc/36 §3ξ.4 gate: the audited core + _accel foundation must be clean.
    Scanning _accel as an explicit root must not be silently skipped."""
    pkg = Path(helixlang.__file__).parent
    combined = scan_tree(pkg / "core", skip=("tests",))
    combined += scan_tree(pkg / "_accel", skip=("tests",))
    assert isinstance(combined, list)
    # The committed tree is clean by construction (CI would fail otherwise).
    assert not combined


def test_linter_fail_flag_exits_nonzero(tmp_path):
    from helixlang.core.find_silent_fallbacks import main
    p = _write(tmp_path, "try:\n    x=1\nexcept ImportError:\n    pass\n")
    assert main([str(p), "--fail"]) == 1
    assert main([str(p)]) == 0


# ── `use` DSL integration (doc/36 §4) ───────────────────────────────────────


def _parse(src: str):
    from helixlang.core.lexer import Lexer
    from helixlang.core.parser import Parser
    return Parser(list(Lexer(src).tokens())).parse()


def test_use_statement_parses_flags():
    prog = _parse("#use grn --pure-python\n")
    assert len(prog.use_directives) == 1
    d = prog.use_directives[0]
    assert d.plugin == "grn"
    assert d.flags == frozenset({"--pure-python"})


def test_use_statement_parses_model_alias():
    prog = _parse('#use fba "ecoli_core"\n')
    d = prog.use_directives[0]
    assert d.plugin == "fba"
    assert d.model == "ecoli_core"


def test_use_statement_multiple():
    prog = _parse("#use grn\n#use pk --approx-euler\n#use fba\n")
    assert [d.plugin for d in prog.use_directives] == ["grn", "pk", "fba"]


def test_use_statement_unknown_flag_is_parse_error():
    from helixlang.core.errors import ParseError
    with pytest.raises(ParseError):
        _parse("#use grn --bogus\n")


def test_unknown_keyword_is_explicit_error_f7():
    """doc/36 F7: an unknown #keyword is a hard error, never silently dropped."""
    from helixlang.core.errors import UnknownKeywordError
    # `#export` is lexed as an annotation keyword but has no parser handler, so
    # it must raise UnknownKeywordError rather than be silently dropped.
    with pytest.raises(UnknownKeywordError):
        _parse("#export gene=foo\n")


def test_semantic_unknown_plugin_raises():
    from helixlang.core.errors import SemanticError
    from helixlang.core.semantic import SemanticAnalyzer
    prog = _parse("#use nope_plugin\n")
    with pytest.raises(SemanticError):
        SemanticAnalyzer(prog).check()


def test_semantic_known_plugin_ok_and_declares_flag():
    from helixlang.core.plugin_registry import PluginProvider, Registry
    from helixlang.core.semantic import SemanticAnalyzer

    r = Registry()
    r.register(PluginProvider(name="grn", extra="grn"))
    prog = _parse("#use grn --pure-python\n")
    SemanticAnalyzer(prog, registry=r).check()
    assert r.has_capability("--pure-python")


def test_apply_use_directives_activates():
    from helixlang.core.plugin_registry import PluginProvider, Registry
    from helixlang.core.use_stmt import apply_use_directives

    r = Registry()
    p = PluginProvider(name="grn", extra="grn", load=lambda: object())
    r.register(p)
    prog = _parse("#use grn --pure-python\n")
    assert apply_use_directives(prog.use_directives, r) == ["grn"]


def test_apply_use_directives_missing_plugin():
    from helixlang.core.errors import PluginMissingError
    from helixlang.core.plugin_registry import Registry
    from helixlang.core.use_stmt import apply_use_directives

    r = Registry()
    prog = _parse("#use ghost\n")
    with pytest.raises(PluginMissingError):
        apply_use_directives(prog.use_directives, r)


# ── OP_USE_PLUGIN bytecode encoding (doc/36 §3.2 / Phase 1.3) ────────────────


def _compile(src: str):
    from helixlang.core.codon_table import STANDARD_TABLE
    from helixlang.core.compiler import Compiler
    return Compiler(STANDARD_TABLE).compile(_parse(src))


def test_compiler_emits_use_plugin_op():
    from helixlang.core.codon_table import Op
    chunk = _compile("#use grn --pure-python\n#use pk\n")
    # Each `use` directive becomes a leading OP_USE_PLUGIN with a 1-byte
    # constant-pool index operand (op @0, operand @1; op @2, operand @3).
    assert chunk.code[0] == int(Op.OP_USE_PLUGIN)
    assert chunk.code[2] == int(Op.OP_USE_PLUGIN)
    specs = [c for c in chunk.constants if c[0] == "use_plugin"]
    assert [s[1] for s in specs] == ["grn", "pk"]
    assert specs[0][2] == ("--pure-python",)


def test_compiler_no_use_no_use_plugin_op():
    from helixlang.core.codon_table import Op
    chunk = _compile("#gene name=g1\nATG GCT GGT TAA\n#end\n")
    assert int(Op.OP_USE_PLUGIN) not in chunk.code


def test_vm_activates_use_plugin_through_registry():
    from helixlang.core.codon_table import STANDARD_TABLE
    from helixlang.core.compiler import Compiler
    from helixlang.core.plugin_registry import PluginProvider, Registry
    from helixlang.core.semantic import SemanticAnalyzer
    from helixlang.core.vm import CellVM

    r = Registry()
    activated: list[str] = []
    r.register(PluginProvider(
        name="grn", extra="grn",
        load=lambda: (activated.append("grn"), object())[1]))
    prog = _parse("#use grn\n")
    SemanticAnalyzer(prog, registry=r).check()
    chunk = Compiler(STANDARD_TABLE).compile(prog)
    vm = CellVM(chunk, prog, registry=r)
    assert activated == ["grn"]
    assert vm.grn is not None

