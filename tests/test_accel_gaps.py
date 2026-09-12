"""Targeted tests for _accel coverage gaps (dispatch, diffusion, grn_step,
simplex numba/numpy impls, and the build module).

Strategies:
  - NUMBA_DISABLE_JIT=1 makes ``@njit`` return the underlying Python function
    so coverage.py can trace the kernel bodies.
  - Reloading a module with ``numba`` import blocked exercises the
    ImportError / ``_HAS_NUMBA is False`` fallback branches.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from types import SimpleNamespace

import pytest

_NUMBA = importlib.util.find_spec("numba") is not None


def _clear(pkg: str) -> None:
    for name in list(sys.modules):
        if name.startswith(pkg):
            sys.modules.pop(name, None)


@pytest.fixture
def jit_off(monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")


@pytest.fixture
def block_numba(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)


@pytest.fixture
def reload_restore():
    """Relaod a module after a test mutates it back to the imported state."""
    yield
    _clear("helixlang._accel.diffusion")
    _clear("helixlang._accel.grn_step")
    _clear("helixlang._accel.simplex")


# ============================================================================
# dispatch/backend.py + dispatch/impl_python.py
# ============================================================================


def test_dispatch_backend_run_quota_and_run_many(monkeypatch):
    # C-only mandate (doc/03 §6.5): python is not a selectable dispatch backend.
    from helixlang._accel.dispatch import backend as db

    code = [0x20, 0, 0x20, 1, 0x92, 0x20, 2, 0x20, 3, 0x91,
            0x90, 0x21, 0x20, 4, 0x11]
    consts = [2.0, 3.0, 10.0, 4.0, 5.0]
    ops, stack, halted = db.run_quota(code, consts)
    assert (ops, stack, halted) == (9, [5.0], True)
    many = db.run_many(code, consts, n_cells=3)
    assert len(many) == 3
    assert all(row == (9, [5.0], True) for row in many)
    default_many = db.run_many(code, consts)
    assert len(default_many) == 1


def test_dispatch_run_quota_quota_exhausted():
    from helixlang._accel.dispatch.impl_python import run_quota

    code = [0x20, 0, 0x20, 1, 0x92]  # no HALT: only exits via quota/ip
    consts = [2.0, 3.0]
    ops, stack, halted = run_quota(code, consts, quota=2)
    assert ops == 2
    assert not halted
    assert stack == [2.0, 3.0]


def test_dispatch_run_quota_ip_exhausted_natural():
    from helixlang._accel.dispatch.impl_python import run_quota

    code = [0x20, 0, 0x20, 1, 0x92, 0x21]  # ends without HALT
    consts = [2.0, 3.0]
    ops, stack, halted = run_quota(code, consts, quota=100)
    assert ops == 4
    assert not halted
    assert stack == []


def test_dispatch_run_quota_unhandled_op():
    from helixlang._accel.dispatch.impl_python import run_quota

    with pytest.raises(NotImplementedError):
        run_quota([0x40], [])  # OP 0x40 not in _HANDLED
    with pytest.raises(IndexError):
        run_quota([0x21], [])  # POP on empty stack
    with pytest.raises(IndexError):
        run_quota([0x90], [])  # ADD with fewer than two values


# ============================================================================
# grn_step/backend.py step_mixed fallback
# ============================================================================


def test_grn_backend_step_mixed_fallback_without_attribute(monkeypatch):
    """A hot-loop module that predates ``step_mixed`` falls back to impl_python."""
    import helixlang._accel.grn_step.backend as gb

    levels = [0.9, 0.2, 0.5, 0.0]
    src = [0, 1, 1]
    dst = [2, 2, 3]
    weights = [0.5, 0.8, -0.3]
    decays = [0.5, None, 0.1, None]
    thresholds = [0.3, 0.3, 0.3, 0.3]
    hill_ns = [2.0, None, None, 3.0]
    kds = [1.0, None, None, 0.0]

    stub = SimpleNamespace(step=lambda *a, **k: None)  # no step_mixed
    monkeypatch.setattr(gb, "load_hot", lambda *a, **k: stub)
    new, trig = gb.step_mixed(levels, src, dst, weights, decays,
                              thresholds, 0.99, hill_ns, kds)
    assert len(new) == 4
    assert isinstance(trig, list)


def test_grn_backend_step_mixed_python_has_attribute(monkeypatch):
    import helixlang._accel.grn_step.backend as gb
    from helixlang._accel.grn_step import impl_python

    levels = [0.9, 0.2, 0.5, 0.0]
    src = [0, 1, 1]
    dst = [2, 2, 3]
    weights = [0.5, 0.8, -0.3]
    decays = [0.5, None, 0.1, None]
    thresholds = [0.3, 0.3, 0.3, 0.3]
    hill_ns = [2.0, None, None, 3.0]
    kds = [1.0, None, None, 0.0]

    monkeypatch.setattr(gb, "load_hot", lambda *a, **k: impl_python)
    new, trig = gb.step_mixed(levels, src, dst, weights, decays,
                              thresholds, 0.99, hill_ns, kds)
    assert len(new) == 4


# ============================================================================
# grn_step/impl_python.py — hill, clamps, step_mixed branches
# ============================================================================


def _case():
    levels = [1.0, 0.0, 0.5, 0.2]
    src = [0, 1, 1]
    dst = [2, 2, 3]
    weights = [1.0, 1.0, -0.3]
    return levels, src, dst, weights


def test_grn_python_step_clamps_above_one():
    from helixlang._accel.grn_step.impl_python import step

    levels = [1.0, 0.0, 0.5, 0.2]
    src = [0, 1, 1]
    dst = [2, 2, 3]
    weights = [1.0, 1.0, -0.3]
    # Negative decay makes the blend exceed 1.0 for high levels.
    decays = [-1.0, 0.5, None, 0.5]
    thresholds = [0.0, 0.0, 0.0, 0.0]
    new, trig = step(levels, src, dst, weights, decays, thresholds, 0.99)
    assert new[1] <= 1.0  # gene 1 has no inputs; level 1.0 + negative decay
    assert all(v <= 1.0 for v in new)


def test_grn_python_hill_full_branches():
    from helixlang._accel.grn_step.impl_python import _hill, step_mixed

    levels, src, dst, weights = _case()
    decays = [0.5, 0.5, 0.5, 0.5]
    thresholds = [0.3, 0.3, 1.0, 0.3]
    # gene2 has acc=1.0, hill n=2 -> hill term; gene3 gets -0.3*0.2 acc<0.
    hill_ns = [None, None, 2.0, None]
    kds = [None, None, 1.0, None]
    new, trig = step_mixed(levels, src, dst, weights, decays, thresholds,
                           0.99, hill_ns, kds)
    assert len(new) == 4

    # kd=0 -> unit step branch returns 1.0
    assert _hill(2.0, 2.0, 0.0) == 1.0
    # x <= 0 -> 0.0
    assert _hill(0.0, 2.0, 1.0) == 0.0
    assert _hill(-3.0, 2.0, 1.0) == 0.0
    # normal return
    assert 0.0 < _hill(1.0, 2.0, 1.0) < 1.0


def test_grn_python_step_mixed_full_branches():
    from helixlang._accel.grn_step.impl_python import step_mixed

    levels = [0.5, 0.0, 0.1, 0.9]
    src = [0, 1, 3]
    dst = [0, 0, 2]
    weights = [2.0, 1.0, 1.0]
    decays = [-1.0, 0.5, 0.5, 0.5]  # negative decay to force v > 1.0
    thresholds = [0.3, 0.9, 0.5, 0.5]
    hill_ns = [2.0, 2.0, None, None]
    kds = [1.0, None, None, None]
    new, trig = step_mixed(levels, src, dst, weights, decays, thresholds,
                           0.99, hill_ns, kds)
    assert len(new) == 4
    assert all(v <= 1.0 for v in new)
    assert isinstance(trig, list)


def test_grn_python_step_mixed_kd_none_falls_back_to_threshold():
    from helixlang._accel.grn_step.impl_python import step_mixed

    levels, src, dst, weights = _case()
    decays = [0.5, 0.5, 0.5, 0.5]
    thresholds = [0.3, 0.3, 0.5, 0.3]
    hill_ns = [None, None, 3.0, None]
    kds = [None, None, None, None]  # kd None -> threshold fallback
    new, trig = step_mixed(levels, src, dst, weights, decays, thresholds,
                           0.99, hill_ns, kds)
    assert len(new) == 4


# ============================================================================
# grn_step/impl_numpy.py step_mixed
# ============================================================================


def test_grn_numpy_step_mixed_matches_python():
    from helixlang._accel.grn_step.impl_numpy import step_mixed
    from helixlang._accel.grn_step.impl_python import step_mixed as py_mixed

    levels, src, dst, weights = _case()
    decays = [0.5, None, 0.1, None]
    thresholds = [0.3, 0.3, 0.5, 0.3]
    hill_ns = [2.0, None, 3.0, None]
    kds = [1.0, None, None, None]
    new_np, trig_np = step_mixed(list(levels), list(src), list(dst),
                                 list(weights), list(decays),
                                 list(thresholds), 0.99, hill_ns, kds)
    new_py, trig_py = py_mixed(list(levels), list(src), list(dst),
                               list(weights), list(decays),
                               list(thresholds), 0.99, hill_ns, kds)
    assert trig_np == trig_py
    for a, b in zip(new_np, new_py, strict=True):
        assert abs(a - b) < 1e-9


# ============================================================================
# grn_step/impl_numba.py
# ============================================================================


@pytest.mark.skipif(not _NUMBA, reason="numba not installed")
def test_grn_numba_kernels_traceable(jit_off, reload_restore):
    """With JIT disabled the njit kernels run as plain Python and are traced."""
    _clear("helixlang._accel.grn_step")
    mod = importlib.import_module("helixlang._accel.grn_step.impl_numba")
    assert mod._HAS_NUMBA

    levels = [0.9, 0.2, 0.5, 0.0]
    src = [0, 1, 1]
    dst = [2, 2, 3]
    weights = [0.5, 0.8, -0.3]
    decays = [0.5, None, 0.1, None]
    thresholds = [0.3, 0.3, 0.3, 0.3]
    hill_ns = [2.0, None, None, 3.0]
    kds = [1.0, None, None, 0.0]

    new, trig = mod.step(list(levels), list(src), list(dst), list(weights),
                         list(decays), list(thresholds), 0.99)
    assert len(new) == 4
    assert isinstance(trig, list)

    new_m, trig_m = mod.step_mixed(list(levels), list(src), list(dst),
                                   list(weights), list(decays),
                                   list(thresholds), 0.99, hill_ns, kds)
    assert len(new_m) == 4
    assert isinstance(trig_m, list)


@pytest.mark.skipif(not _NUMBA, reason="numba not installed")
def test_grn_numba_step_mixed_clamp_and_kd(monkeypatch, jit_off, reload_restore):

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NUMBA_DISABLE_JIT", "1")
        _clear("helixlang._accel.grn_step")
        mod = importlib.import_module("helixlang._accel.grn_step.impl_numba")

        levels = [1.0, 0.0, 0.1, 0.9]
        src = [0, 1, 3]
        dst = [0, 0, 2]
        weights = [2.0, 1.0, 1.0]
        decays = [-1.0, 0.5, 0.5, 0.5]
        thresholds = [0.3, 0.9, 0.5, 0.5]
        hill_ns = [2.0, 2.0, None, None]
        kds = [1.0, None, None, None]
        new, trig = mod.step_mixed(list(levels), list(src), list(dst),
                                   list(weights), list(decays),
                                   list(thresholds), 0.99, hill_ns, kds)
        assert all(v <= 1.0 for v in new)


@pytest.mark.skipif(not _NUMBA, reason="numba not installed")
def test_grn_numba_import_error_branch(block_numba, reload_restore):
    _clear("helixlang._accel.grn_step")
    mod = importlib.import_module("helixlang._accel.grn_step.impl_numba")
    assert not mod._HAS_NUMBA
    from helixlang.core.errors import NativeBackendError

    levels = [0.9, 0.2, 0.5, 0.0]
    with pytest.raises(NativeBackendError):
        mod.step(levels, [0], [2], [0.5], [0.5, None, 0.1, None],
                 [0.3, 0.3, 0.3, 0.3], 0.99)
    with pytest.raises(NativeBackendError):
        mod.step_mixed(levels, [0], [2], [0.5], [0.5, None, 0.1, None],
                       [0.3, 0.3, 0.3, 0.3], 0.99, [2.0], [1.0])


# ============================================================================
# diffusion/impl_python.py clamps
# ============================================================================


def test_diffusion_python_clamps_both_directions():
    from helixlang._accel.diffusion.impl_python import step

    # 3x3 with borders; interior cell (1,1) is active.
    base = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # Downward clamp for u and v: high D with negative Laplacian -> below 0.
    u_lo = [row[:] for row in base]
    u_lo[1][1] = 0.5
    v = [row[:] for row in base]
    v[1][1] = 0.25
    nu, nv = step(u_lo, v, 0.0, 0.0, 2.0, 2.0)
    assert nu[1][1] == 0.0
    assert nv[1][1] == 0.0

    # Upward clamp for u and v: neighbors high, interior low, big D.
    hi = [[1.0, 1.0, 1.0], [1.0, 0.5, 1.0], [1.0, 1.0, 1.0]]
    v_hi = [[1.0, 1.0, 1.0], [1.0, 0.25, 1.0], [1.0, 1.0, 1.0]]
    nu, nv = step(hi, v_hi, 0.0, 0.0, 2.0, 2.0)
    assert nu[1][1] == 1.0
    assert nv[1][1] == 1.0


# ============================================================================
# diffusion/impl_numba.py
# ============================================================================


@pytest.mark.skipif(not _NUMBA, reason="numba not installed")
def test_diffusion_numba_kernel_traceable(jit_off, reload_restore):
    _clear("helixlang._accel.diffusion")
    mod = importlib.import_module("helixlang._accel.diffusion.impl_numba")
    assert mod._HAS_NUMBA

    n = 6
    u = [[1.0] * n for _ in range(n)]
    v = [[0.0] * n for _ in range(n)]
    for i in range(2, 4):
        for j in range(2, 4):
            u[i][j] = 0.5
            v[i][j] = 0.3
    nu, nv = mod.step(u, v, 0.035, 0.065, 0.16, 0.08)
    assert len(nu) == n
    assert len(nv) == n


@pytest.mark.skipif(not _NUMBA, reason="numba not installed")
def test_diffusion_numba_import_error_branch(block_numba, reload_restore):
    _clear("helixlang._accel.diffusion")
    mod = importlib.import_module("helixlang._accel.diffusion.impl_numba")
    assert not mod._HAS_NUMBA
    from helixlang.core.errors import NativeBackendError

    with pytest.raises(NativeBackendError):
        mod.step([[]], [[]], 0.035, 0.065, 0.16, 0.08)


# ============================================================================
# simplex/impl_numpy.py edge branches
# ============================================================================


def test_simplex_numpy_empty_tableau():
    from helixlang._accel.simplex.impl_numpy import run

    assert run([], [], [], 2) == "optimal"


def test_simplex_numpy_forbidden_and_unbounded():
    from helixlang._accel.simplex.impl_numpy import run

    # degenerate LP: max x0 + x1  s.t. x0 + x1 + s0 = 4; forbid x0.
    T = [[1.0, 1.0, 1.0, 4.0]]
    b = [2]
    assert run(T, b, [1.0, 1.0, 0.0], 3, forbidden=[0]) == "optimal"
    assert b == [1]  # x1 pivoted in

    # unbounded: max x0  s.t. 0*x0 + s0 = 4 -> col of zeros.
    T2 = [[0.0, 1.0, 4.0]]
    assert run(T2, [1], [1.0, 0.0], 2) == "unbounded"


def test_simplex_numpy_max_iter():
    import copy

    from helixlang._accel.simplex.impl_numpy import run

    T = [[1.0, 1.0, 1.0, 0.0, 4.0], [1.0, 0.0, 0.0, 1.0, 2.0]]
    b = [2, 3]
    obj = [2.0, 3.0, 0.0, 0.0]
    assert run(copy.deepcopy(T), b[:], obj[:], 4, max_iter=0) == "max_iter"
    assert run(copy.deepcopy(T), b[:], obj[:], 4, max_iter=1) == "max_iter"
    assert run(copy.deepcopy(T), b[:], obj[:], 4, max_iter=100) == "optimal"


# ============================================================================
# simplex/impl_numba.py
# ============================================================================


@pytest.mark.skipif(not _NUMBA, reason="numba not installed")
def test_simplex_numba_traceable(jit_off, reload_restore):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NUMBA_DISABLE_JIT", "1")
        _clear("helixlang._accel.simplex")
        mod = importlib.import_module("helixlang._accel.simplex.impl_numba")

        T = [[1.0, 1.0, 1.0, 0.0, 4.0], [1.0, 0.0, 0.0, 1.0, 2.0]]
        b = [2, 3]
        status = mod.run(T, b, [2.0, 3.0, 0.0, 0.0], 4)
        assert status == "optimal"
        assert 1 in b  # at least one real variable pivoted in

        # unbounded
        T2 = [[0.0, 1.0, 4.0]]
        assert mod.run(T2, [1], [1.0, 0.0], 2) == "unbounded"

        # empty
        assert mod.run([], [], [], 2) == "optimal"

        # max_iter
        T3 = [[1.0, 1.0, 4.0], [1.0, 0.0, 2.0]]
        assert mod.run(T3, [2, 3], [2.0, 3.0, 0.0, 0.0], 2, max_iter=0) == "max_iter"

        # forbidden routes around an entering column
        T4 = [[1.0, 1.0, 1.0, 4.0]]
        b4 = [2]
        assert mod.run(T4, b4, [1.0, 1.0, 0.0], 3, max_iter=10,
                       forbidden=[0]) == "optimal"
        assert b4 == [1]


@pytest.mark.skipif(not _NUMBA, reason="numba not installed")
def test_simplex_numba_import_error_branch(block_numba, reload_restore):
    _clear("helixlang._accel.simplex")
    mod = importlib.import_module("helixlang._accel.simplex.impl_numba")
    assert mod.njit is None
    with pytest.raises(RuntimeError):
        mod.run([[1.0, 1.0, 4.0]], [2], [1.0, 0.0], 2)


# ============================================================================
# _accel/build.py
# ============================================================================


def test_build_no_sources(monkeypatch, capsys):
    import helixlang._accel.build as b

    monkeypatch.setattr(b, "_cython_sources", lambda: [])
    monkeypatch.setattr(b, "_c_sources", lambda: [])
    assert b.build_extensions() == 0
    out = capsys.readouterr().out
    assert "nothing to build" in out


def test_build_pyx_without_cython(monkeypatch, capsys):
    import helixlang._accel.build as b

    monkeypatch.setattr(b, "_cython_sources", lambda: [b._BASE / "x.pyx"])
    monkeypatch.setattr(b, "_c_sources", lambda: [])
    monkeypatch.setattr(b, "_bootstrap_imports",
                        lambda: (None, object, object, object))
    assert b.build_extensions() == 1
    out = capsys.readouterr().out
    assert "Cython is required" in out


def test_build_failure_returns_one(monkeypatch, capsys):
    import helixlang._accel.build as b

    class Boom(Exception):
        pass

    def fake_build_ext(dist):
        class Cmd:
            def ensure_finalized(self):
                pass

            def run(self):
                raise Boom("toolchain missing")

        return Cmd()

    def fake_ext(name, sources):
        return object()

    monkeypatch.setattr(b, "_cython_sources", lambda: [b._BASE / "x.pyx"])
    monkeypatch.setattr(b, "_c_sources", lambda: [])
    monkeypatch.setattr(b, "_bootstrap_imports",
                        lambda: (lambda mods, language_level=None: mods, fake_ext, fake_build_ext,
                                 dict))
    monkeypatch.setattr(b, "_build_rust_backends", lambda: 0)
    assert b.build_extensions() == 1
    out = capsys.readouterr().out
    assert "Native build failed" in out


def test_build_success(monkeypatch, capsys, tmp_path):
    import helixlang._accel.build as b

    def fake_cythonize(mods, language_level=None):
        return mods

    def fake_build_ext(dist):
        class Cmd:
            def ensure_finalized(self):
                pass

            def run(self):
                pass

        return Cmd()

    def fake_ext(name, sources):
        return object()

    (tmp_path / "grn_step").mkdir()
    (tmp_path / "grn_step" / "impl_cython.pyx").write_text("x = 1\n")
    (tmp_path / "grn_step" / "impl_cython.cpython-darwin.so").write_bytes(b"\x00")
    monkeypatch.setattr(b, "_BASE", tmp_path)
    monkeypatch.setattr(b, "_cython_sources",
                        lambda: [tmp_path / "grn_step/impl_cython.pyx"])
    monkeypatch.setattr(b, "_c_sources", lambda: [])
    monkeypatch.setattr(b, "_bootstrap_imports",
                        lambda: (fake_cythonize, fake_ext, fake_build_ext,
                                 dict))
    monkeypatch.setattr(b, "_build_rust_backends", lambda: 0)
    assert b.build_extensions() == 0
    out = capsys.readouterr().out
    assert "built impl_cython.cpython-darwin.so" in out


def test_build_no_so_produced(monkeypatch, capsys, tmp_path):
    import helixlang._accel.build as b

    def fake_build_ext(dist):
        class Cmd:
            def ensure_finalized(self):
                pass

            def run(self):
                pass

        return Cmd()

    def fake_ext(name, sources):
        return object()

    monkeypatch.setattr(b, "_BASE", tmp_path)
    monkeypatch.setattr(b, "_cython_sources", lambda: [tmp_path / "x.pyx"])
    monkeypatch.setattr(b, "_c_sources", lambda: [])
    monkeypatch.setattr(b, "_bootstrap_imports",
                        lambda: (lambda m, language_level=None: m, fake_ext,
                                 fake_build_ext, dict))
    monkeypatch.setattr(b, "_build_rust_backends", lambda: 0)
    assert b.build_extensions() == 1
    out = capsys.readouterr().out
    assert "No .so produced" in out


def test_build_rust_backends_crates_empty():
    from helixlang._accel import build as b

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(b, "_rust_crates", lambda: [])
    assert b._build_rust_backends() == 0
    monkeypatch.undo()


def test_build_rust_backend_builds_and_copies(monkeypatch, tmp_path, capsys):
    import shutil as _shutil_mod

    from helixlang._accel import build as b

    crate_dir = tmp_path / "crate"
    release = crate_dir / "target" / "release"
    release.mkdir(parents=True)
    so = release / "libsimplex.dylib"
    so.write_bytes(b"\x00")
    copied = {}

    monkeypatch.setattr(b, "_BASE", tmp_path)
    monkeypatch.setattr(b, "_rust_crates",
                        lambda: [SimpleNamespace(parent=crate_dir)])
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(_shutil_mod, "copy",
                        lambda src, dst: copied.__setitem__(dst, src))
    assert b._build_rust_backends() == 1
    out = capsys.readouterr().out
    assert "built" in out
    assert any("impl_rust.abi3.so" in str(k) for k in copied)


def test_build_module_name_and_root_src():
    from helixlang._accel.build import _BASE, _module_name, _root_src

    src = _BASE / "grn_step/impl_cython.pyx"
    assert _module_name(src) == "helixlang._accel.grn_step.impl_cython"
    root = _root_src()
    assert root.endswith("src")


def test_build_main_exits():
    import helixlang._accel.build as b

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(b, "build_extensions", lambda: 0)
        with pytest.raises(SystemExit) as exc:
            b.main()
        assert exc.value.code == 0
