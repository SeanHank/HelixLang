"""Plugin-system tests (doc/36 §3): discovery, lazy activation, and the
per-plugin missing-dependency contract (Phase 2.6).

doc/36 §3ξ.4 requires that removing a plugin's core optional dependency raises
``PluginDependencyError`` (an explicit error) — never a silent fallback to a
lower-fidelity computation.
"""
from __future__ import annotations

import pytest

from helixlang.core.errors import PluginDependencyError
from helixlang.core.plugin_registry import Registry


def test_bundled_grn_is_discoverable():
    r = Registry()
    assert r.discover("grn") == ["grn"]
    assert r.is_registered("grn")
    prov = r.provider("grn")
    assert prov.keywords == ("gene", "regulate")
    assert prov.extra == "grn"
    assert prov.capability_flags == ("--pure-python",)


def test_get_registry_autodiscovers_bundled():
    from helixlang.core.plugin_registry import get_registry
    assert get_registry().is_registered("grn")


def test_grn_keyword_routes_to_provider():
    from helixlang.core.plugin_registry import get_registry
    prov = get_registry().provider_for_keyword("gene")
    assert prov is not None and prov.name == "grn"


def test_use_grn_activates_backend():
    from helixlang.core.lexer import Lexer
    from helixlang.core.parser import Parser
    from helixlang.core.use_stmt import apply_use_directives
    r = Registry()
    r.discover("grn")
    prog = Parser(list(Lexer("#use grn\n").tokens())).parse()
    assert apply_use_directives(prog.use_directives, r) == ["grn"]


def test_missing_numpy_raises_plugin_dependency_error(monkeypatch):
    """doc/36 §3ξ.4: removing a core optional dep raises an explicit error."""
    import helixlang.plugins.grn as grn_mod

    r = Registry()
    r.discover("grn")
    monkeypatch.setattr(grn_mod, "_check_numpy", lambda: False)
    with pytest.raises(PluginDependencyError):
        r.activate("grn")


def test_missing_numpy_waived_by_optin_flag():
    """doc/36 §3ξ.3/§3ξ.5: with numpy absent, clMail the accelerated path must
    fail unless the program explicitly opts into --pure-python (an equivalent-
    fidelity pure-Python reference).  Opting in waives the error."""
    r = Registry()
    r.discover("grn")
    # Simulate an absent numpy.
    r._providers["grn"].checks["numpy"] = lambda: False
    # No opt-in declared -> hard error.
    with pytest.raises(PluginDependencyError):
        r.activate("grn")
    # Explicit --pure-python opt-in -> waives; backend still loads.
    r.declare_capability("--pure-python")
    assert callable(r.activate("grn"))


# ---------------------------------------------------------------------------
# FBA plugin (metabolism.py -> plugins/fba/)
# ---------------------------------------------------------------------------


def test_bundled_fba_is_discoverable():
    r = Registry()
    assert r.discover("fba") == ["fba"]
    prov = r.provider("fba")
    assert prov.keywords == ("media", "sim")
    assert prov.extra == "fba"
    assert prov.capability_flags == ("--low-fidelity",)


def test_get_registry_autodiscovers_fba():
    from helixlang.core.plugin_registry import get_registry
    assert get_registry().is_registered("fba")


def test_fba_keyword_routes_to_provider():
    from helixlang.core.plugin_registry import get_registry
    for kw in ("media", "sim"):
        prov = get_registry().provider_for_keyword(kw)
        assert prov is not None and prov.name == "fba"


def test_use_fba_activates_backend():
    from helixlang.core.lexer import Lexer
    from helixlang.core.parser import Parser
    from helixlang.core.use_stmt import apply_use_directives
    r = Registry()
    r.discover("fba")
    prog = Parser(list(Lexer("#use fba\n").tokens())).parse()
    assert apply_use_directives(prog.use_directives, r) == ["fba"]
    from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis
    assert r.activate("fba")(None) is FluxBalanceAnalysis


def test_fba_missing_numpy_raises_plugin_dependency_error(monkeypatch):
    """doc/36 §3ξ.4: numpy is a hard FBA solcery dep — no silent fallback."""
    import helixlang.plugins.fba as fba_mod
    r = Registry()
    r.discover("fba")
    monkeypatch.setattr(fba_mod, "_check_numpy", lambda: False)
    with pytest.raises(PluginDependencyError):
        r.activate("fba")


def test_missing_cobra_raises_unless_low_fidelity_optin():
    """cobra is SBML-import-only; absent => error unless --low-fidelity opt-in."""
    r = Registry()
    r.discover("fba")
    r._providers["fba"].checks["cobra"] = lambda: False
    with pytest.raises(PluginDependencyError):
        r.activate("fba")
    r.declare_capability("--low-fidelity")
    from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis
    assert r.activate("fba")(None) is FluxBalanceAnalysis


