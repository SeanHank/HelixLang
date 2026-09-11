"""Tests for the reduced-fidelity opt-in gate (doc/36 §3ξ.3, core/fidelity.py).

Covers every branch of ``opt_in`` and ``require``: the per-call ``allow``
override, the declared capability-flag path, the documented env override, and the
strict default that raises a typed :class:`PluginDependencyError`.
"""
from __future__ import annotations

import pytest

from helixlang.core.errors import PluginDependencyError
from helixlang.core.fidelity import _declared, opt_in, require


def test_opt_in_explicit_allow(monkeypatch):
    # allow=True short-circuits regardless of flag/env/registry state.
    monkeypatch.delenv("HELIX_ALLOW_LOW_FIDELITY", raising=False)
    assert opt_in(allow=True) is True


def test_opt_in_declared_flag_via_registry(monkeypatch):
    from helixlang.core.plugin_registry import get_registry
    registry = get_registry()
    monkeypatch.setattr(registry, "has_capability", lambda f: True)
    # No env, no allow, but the flag is declared on the registry -> True.
    monkeypatch.delenv("HELIX_ALLOW_LOW_FIDELITY", raising=False)
    assert opt_in("--pure-python") is True
    assert _declared("--pure-python") is True


def test_opt_in_env_override(monkeypatch):
    monkeypatch.setenv("HELIX_ALLOW_LOW_FIDELITY", "1")
    assert opt_in() is True
    monkeypatch.setenv("HELIX_ALLOW_LOW_FIDELITY", "true")
    assert opt_in() is True
    monkeypatch.setenv("HELIX_ALLOW_LOW_FIDELITY", "yes")
    assert opt_in() is True


def test_opt_in_false_when_nothing_declared(monkeypatch):
    from helixlang.core.plugin_registry import get_registry
    registry = get_registry()
    monkeypatch.setattr(registry, "has_capability", lambda f: False)
    monkeypatch.delenv("HELIX_ALLOW_LOW_FIDELITY", raising=False)
    assert opt_in() is False
    assert _declared("--pure-python") is False


def test_declared_returns_false_on_registry_error(monkeypatch):
    # A bare import with no registry must degrade to False, never raise.
    monkeypatch.setattr(
        "helixlang.core.plugin_registry.get_registry",
        lambda: (_ for _ in ()).throw(RuntimeError("no registry")),
    )
    assert _declared("--low-fidelity") is False


def test_require_passes_when_opted_in(monkeypatch):
    monkeypatch.setenv("HELIX_ALLOW_LOW_FIDELITY", "1")
    require(name="fba", dep="cobra", extra="need cobra")  # must not raise
    # explicit allow path
    require(name="fba", dep="cobra", extra="need cobra", allow=True)


def test_require_raises_when_strict(monkeypatch):
    from helixlang.core.plugin_registry import get_registry
    registry = get_registry()
    monkeypatch.setattr(registry, "has_capability", lambda f: False)
    monkeypatch.delenv("HELIX_ALLOW_LOW_FIDELITY", raising=False)
    with pytest.raises(PluginDependencyError) as ei:
        require(name="fba", dep="cobra", extra="need cobra")
    assert "fba" in str(ei.value)
    assert "cobra" in str(ei.value)


def test_require_uses_custom_env(monkeypatch):
    monkeypatch.setenv("MY_CUSTOM_ENV", "1")
    require(name="x", dep="y", extra="z", env="MY_CUSTOM_ENV")
