"""Plugin registry unit tests (doc/36 §3)."""
from __future__ import annotations

import pytest

from helixlang.core.errors import (
    PluginConflictError,
    PluginDependencyError,
    PluginMissingError,
)
from helixlang.core.plugin_registry import (
    NativeBackend,
    PluginProvider,
    Registry,
    get_registry,
    prov_extra_hint,
)


def _prov(name, *, keywords=(), native=None, caps=(),
          checks=None, load=None, grammars=()):
    return PluginProvider(
        name=name, extra=name,
        keywords=tuple(keywords), native=native,
        capability_flags=tuple(caps),
        checks=dict(checks or {}), load=load, grammars=tuple(grammars),
    )


def test_check_dependencies_lists_unsatisfied():
    p = PluginProvider(name="x", extra="x",
                       checks={"a": lambda: True, "b": lambda: False})
    assert p.check_dependencies() == ["b"]


def test_prov_extra_hint():
    assert prov_extra_hint("grn") == "grn"
    assert prov_extra_hint("no_such_plugin") == "core"


def test_register_and_lookup():
    r = Registry()
    r.register(_prov("p1", keywords=("#k1",)))
    assert r.is_registered("p1")
    assert r.provider("p1").name == "p1"
    assert r.provider_for_keyword("#k1").name == "p1"
    assert r.provider_for_keyword("#missing") is None
    assert r.providers()[0].name == "p1"


def test_provider_missing_raises():
    r = Registry()
    with pytest.raises(PluginMissingError):
        r.provider("ghost")


def test_register_duplicate_name_raises():
    r = Registry()
    r.register(_prov("p1"))
    with pytest.raises(PluginConflictError):
        r.register(_prov("p1"))


def test_register_conflicting_keyword_raises():
    r = Registry()
    r.register(_prov("p1", keywords=("#k",)))
    with pytest.raises(PluginConflictError):
        r.register(_prov("p2", keywords=("#k",)))


def test_register_conflicting_native_module_raises():
    r = Registry()
    r.register(_prov("p1", native=NativeBackend(module="mod.x")))
    with pytest.raises(PluginConflictError):
        r.register(_prov("p2", native=NativeBackend(module="mod.x")))


def test_register_with_native_backend_ok():
    r = Registry()
    r.register(_prov("p1", native=NativeBackend(module="mod.x")))
    assert r.is_registered("p1")


def test_discover_imports_bundled_plugin():
    # 'grn' is a real bundled plugin module.
    r = Registry()
    got = r.discover("grn")
    assert "grn" in got
    assert r.is_registered("grn")
    # discover again -> already registered, no duplicates
    assert r.discover("grn") == []


def test_discover_unknown_plugin_skipped():
    r = Registry()
    assert r.discover("does_not_exist_xyz") == []
    assert not r.is_registered("does_not_exist_xyz")


def test_discover_skips_module_without_plugin(monkeypatch):
    import sys
    from types import ModuleType

    fake = ModuleType("helixlang.plugins.fake_plugin")
    fake.PLUGIN = None
    sys.modules["helixlang.plugins.fake_plugin"] = fake
    monkeypatch.setattr("helixlang.plugins", sys.modules["helixlang.plugins"])
    try:
        r = Registry()
        assert r.discover("fake_plugin") == []
        assert not r.is_registered("fake_plugin")
    finally:
        sys.modules.pop("helixlang.plugins.fake_plugin", None)


def test_capability_flags():
    r = Registry()
    r.declare_capability("--pure-python")
    assert r.has_capability("--pure-python")
    assert not r.has_capability("--approx-euler")
    r.declare_capability("--approx-euler")
    assert r.has_capability("--approx-euler")


def test_fidelity_full_and_reduced():
    r = Registry()
    assert r.fidelity()["fidelity"] == "full"
    r.declare_capability("--pure-python")
    assert r.fidelity()["fidelity"] == "reduced"


def test_fidelity_reports_active_plugin_backend():
    r = Registry()
    r.register(_prov("p1", native=NativeBackend(module="mod.x")))
    r._active.add("p1")
    rec = r.fidelity()
    assert rec["plugins"][0]["backend"] == "mod.x"
    # an active provider with no native -> python
    r.register(_prov("p2"))
    r._active.add("p2")
    backends = {p["name"]: p["backend"] for p in r.fidelity()["plugins"]}
    assert backends["p2"] == "python"


def test_activate_missing_raises():
    r = Registry()
    with pytest.raises(PluginMissingError):
        r.activate("ghost")


def test_activate_unmet_dependency_raises():
    r = Registry()
    r.register(_prov("p1", checks={"numpy": lambda: False}))
    with pytest.raises(PluginDependencyError):
        r.activate("p1")


def test_activate_unmet_dependency_honoured_by_capability():
    r = Registry()
    r.register(_prov("p1", caps=("--low-fidelity",),
                      checks={"numpy": lambda: False},
                      load=lambda: "loaded"))
    r._capabilities.add("--low-fidelity")
    assert r.activate("p1") == "loaded"
    assert "p1" in r.active()


def test_activate_load_entrypoint():
    r = Registry()
    calls = []
    r.register(_prov("p1", load=lambda: calls.append("loaded")))
    r.activate("p1")
    assert calls == ["loaded"]
    assert "p1" in r.active()


def test_activate_without_load_imports_module(monkeypatch):
    r = Registry()
    sentinel = object()
    monkeypatch.setattr(
        "helixlang.core.plugin_registry.import_module",
        lambda name: sentinel,
    )
    r.register(_prov("p1", checks={}))
    assert r.activate("p1") is sentinel
    assert "p1" in r.active()


def test_register_installs_grammars_with_requires_use():
    from helixlang.core.grammar_registry import GrammarDescriptor
    r = Registry()
    desc = GrammarDescriptor(keyword="zzz_custom")
    r.register(_prov("p1", grammars=(desc,)))
    from helixlang.core.grammar_registry import grammar_registry
    g = grammar_registry.get("zzz_custom")
    assert g is not None
    assert g.owner == "p1"
    # cleanup the test grammar so it doesn't leak into other tests
    grammar_registry._grammars.pop("zzz_custom", None)


def test_get_registry_singleton_autodiscovers():
    r1 = get_registry()
    r2 = get_registry()
    assert r1 is r2
    assert r1.is_registered("grn")
