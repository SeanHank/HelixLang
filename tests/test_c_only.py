"""C-only mandate gate tests (doc/03 §6.5 / doc/06 §19).

Verification goals:
- every native stage registered in ``core.native_manifest`` resolves to the C
  backend (``impl_cext``) via the uniform loader;
- a Python/numpy request for those packages raises ``NativeBackendError``
  (a Python VM/lexer hot loop can never be selected);
- the ``c_only`` gate reports loudly (and exits nonzero) whenever a compiled
  kernel is missing or the resolver could fall back to Python;
- plugin-side acceleration packages (e.g. ``grn_step``) keep their explicit
  python capability (plugins are excluded from the mandate).

References:
- doc/03 §6.5 C-only execution mandate
- doc/06 §19 Compiler + VM C-only mandate
- doc/36 §3ξ.4 (no silent fallback)
"""
from __future__ import annotations

import runpy

import pytest

from helixlang._accel._loaders import choose_backend, load_hot
from helixlang.core import c_only
from helixlang.core.errors import NativeBackendError
from helixlang.core.native_manifest import native_stages

_STAGES = [pkg for pkg, _ in native_stages()]
_DISPATCH = "helixlang._accel.dispatch"
_LEXER = "helixlang._accel.lexer"


@pytest.mark.parametrize("pkg", _STAGES)
def test_stage_selects_cext_backend(pkg: str) -> None:
    mod = load_hot(pkg)
    assert mod.__name__ == f"{pkg}.impl_cext"


@pytest.mark.parametrize("pkg", _STAGES)
def test_stage_python_request_raises(pkg: str) -> None:
    with pytest.raises(NativeBackendError):
        choose_backend(pkg, prefer="python")


@pytest.mark.parametrize("pkg", _STAGES)
def test_stage_numpy_request_raises(pkg: str) -> None:
    with pytest.raises(NativeBackendError):
        choose_backend(pkg, prefer="numpy")


def test_lexer_backend_exposes_tokenize() -> None:
    mod = load_hot(_LEXER)
    assert hasattr(mod, "tokenize")


def test_plugin_package_keeps_explicit_python_capability() -> None:
    assert choose_backend("helixlang._accel.grn_step", prefer="python") == (
        "impl_python")


def test_stage_backend_none_when_kernel_missing(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise NativeBackendError("absent")

    monkeypatch.setattr(c_only, "load_hot", _raise)
    assert c_only.stage_backend(_DISPATCH) is None


def test_python_not_selectable_when_resolver_allows_it(monkeypatch) -> None:
    monkeypatch.setattr(
        c_only, "choose_backend", lambda pkg, prefer=None: "impl_python")
    assert c_only.python_not_selectable(_DISPATCH) is False


def test_stage_check_fails_when_kernel_missing(monkeypatch) -> None:
    monkeypatch.setattr(c_only, "stage_backend", lambda pkg: None)
    ok, msg = c_only._stage_check(_DISPATCH, "VM opcode dispatch hot loop")
    assert not ok
    assert "_accel.build" in msg


def test_stage_check_fails_when_non_c_backend(monkeypatch) -> None:
    monkeypatch.setattr(c_only, "stage_backend",
                        lambda pkg: f"{pkg}.impl_python")
    monkeypatch.setattr(c_only, "python_not_selectable", lambda pkg: True)
    ok, msg = c_only._stage_check(_DISPATCH, "VM opcode dispatch hot loop")
    assert not ok
    assert "non-C backend" in msg


def test_stage_check_fails_when_python_still_selectable(monkeypatch) -> None:
    monkeypatch.setattr(c_only, "stage_backend", lambda pkg: f"{pkg}.impl_cext")
    monkeypatch.setattr(c_only, "python_not_selectable", lambda pkg: False)
    ok, msg = c_only._stage_check(_DISPATCH, "VM opcode dispatch hot loop")
    assert not ok
    assert "Python hot loop" in msg


def test_check_passes_with_cext_backends() -> None:
    ok, msg = c_only.check()
    assert ok
    assert "impl_cext" in msg
    assert _LEXER in msg and _DISPATCH in msg


def test_check_fails_and_joins_messages(monkeypatch) -> None:
    monkeypatch.setattr(
        c_only, "_stage_check",
        lambda pkg, name: (False, f"{name} broken"))
    ok, msg = c_only.check()
    assert not ok
    assert "broken" in msg
    assert len(native_stages()) > 1 and "broken" in msg


def test_check_fails_when_manifest_empty(monkeypatch) -> None:
    monkeypatch.setattr(c_only, "native_stages", lambda: ())
    ok, msg = c_only.check()
    assert not ok
    assert "no native stages" in msg


def test_manifest_helpers_cover_all_stages() -> None:
    from helixlang._accel import _loaders
    from helixlang.core import native_manifest as nm

    assert isinstance(nm.native_packages(), frozenset)
    assert nm.native_packages() == _loaders._NATIVE_ONLY_PACKAGES
    pending = nm.pending_stages()
    assert ("helixlang._accel.parser", "Parser (token stream -> AST)") in pending
    assert all(pkg.startswith("helixlang._accel.") for pkg, _ in pending)
    names = nm.stage_names()
    assert len(names) == len(set(names))
    assert names[0] == "Lexer / dual-mode scanner"
    assert "VM opcode dispatch hot loop" in names


def test_manifest_entrypoint_prints_stages(capsys) -> None:
    import runpy

    runpy.run_module("helixlang.core.native_manifest", run_name="__main__")
    out = capsys.readouterr().out
    assert "native" in out and "pending" in out
    assert "VM opcode dispatch hot loop" in out


def test_main_exit_zero_on_pass(monkeypatch) -> None:
    monkeypatch.setattr(c_only, "check", lambda: (True, "ok"))
    assert c_only.main([]) == 0


def test_main_exit_one_on_fail(monkeypatch) -> None:
    monkeypatch.setattr(c_only, "check", lambda: (False, "broken"))
    assert c_only.main([]) == 1


def test_module_entrypoint_raises_systemexit() -> None:
    with pytest.raises(SystemExit) as ei:
        runpy.run_module("helixlang.core.c_only", run_name="__main__")
    assert ei.value.code == 0
