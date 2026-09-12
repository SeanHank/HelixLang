"""C-only mandate gate tests (doc/03 §6.5 / doc/06 §11).

Verification goals:
- the dispatch hot loop resolves to the C backend (``impl_cext``) via the
  uniform loader;
- a Python/numpy request for the dispatch package raises ``NativeBackendError``
  (a Python VM hot loop can never be selected);
- the ``c_only`` gate reports loudly (and exits nonzero) whenever the compiled
  kernel is missing or the resolver could fall back to Python;
- plugin-side acceleration packages (e.g. ``grn_step``) keep their explicit
  python capability (plugins are excluded from the mandate).

References:
- doc/03 §6.5 C-only execution mandate
- doc/06 §11 Compiler + VM C-only mandate
- doc/36 §3ξ.4 (no silent fallback)
"""
from __future__ import annotations

import pytest

from helixlang._accel._loaders import choose_backend, load_hot
from helixlang.core import c_only
from helixlang.core.errors import NativeBackendError

_DISPATCH = "helixlang._accel.dispatch"


def test_dispatch_selects_cext_backend() -> None:
    mod = load_hot(_DISPATCH)
    assert mod.__name__ == "helixlang._accel.dispatch.impl_cext"
    assert hasattr(mod, "run_quota")
    assert hasattr(mod, "run_many")


def test_dispatch_python_request_raises() -> None:
    with pytest.raises(NativeBackendError):
        choose_backend(_DISPATCH, prefer="python")


def test_dispatch_numpy_request_raises() -> None:
    with pytest.raises(NativeBackendError):
        choose_backend(_DISPATCH, prefer="numpy")


def test_plugin_package_keeps_explicit_python_capability() -> None:
    assert choose_backend("helixlang._accel.grn_step", prefer="python") == (
        "impl_python")


def test_dispatch_backend_none_when_kernel_missing(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise NativeBackendError("absent")

    monkeypatch.setattr(c_only, "load_hot", _raise)
    assert c_only.dispatch_backend() is None


def test_python_not_selectable_when_resolver_allows_it(monkeypatch) -> None:
    monkeypatch.setattr(
        c_only, "choose_backend", lambda pkg, prefer=None: "impl_python")
    assert c_only.python_not_selectable() is False


def test_check_fails_when_kernel_missing(monkeypatch) -> None:
    monkeypatch.setattr(c_only, "dispatch_backend", lambda: None)
    ok, msg = c_only.check()
    assert not ok
    assert "_accel.build" in msg


def test_check_fails_when_non_c_backend(monkeypatch) -> None:
    monkeypatch.setattr(c_only, "dispatch_backend",
                        lambda: f"{_DISPATCH}.impl_python")
    monkeypatch.setattr(c_only, "python_not_selectable", lambda: True)
    ok, msg = c_only.check()
    assert not ok
    assert "non-C backend" in msg


def test_check_fails_when_python_still_selectable(monkeypatch) -> None:
    monkeypatch.setattr(c_only, "dispatch_backend",
                        lambda: f"{_DISPATCH}.impl_cext")
    monkeypatch.setattr(c_only, "python_not_selectable", lambda: False)
    ok, msg = c_only.check()
    assert not ok
    assert "Python hot loop" in msg


def test_check_passes_with_cext_backend() -> None:
    ok, msg = c_only.check()
    assert ok
    assert "impl_cext" in msg


def test_main_exit_zero_on_pass(monkeypatch) -> None:
    monkeypatch.setattr(c_only, "check", lambda: (True, "ok"))
    assert c_only.main([]) == 0


def test_main_exit_one_on_fail(monkeypatch) -> None:
    monkeypatch.setattr(c_only, "check", lambda: (False, "broken"))
    assert c_only.main([]) == 1


def test_module_entrypoint_raises_systemexit() -> None:
    import runpy

    with pytest.raises(SystemExit) as ei:
        runpy.run_module("helixlang.core.c_only", run_name="__main__")
    assert ei.value.code == 0

