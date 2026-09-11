"""Targeted coverage-close tests for remaining ``helixlang._accel`` gaps.

Pairs with ``test_accel_coverage_close.py`` and closes the last branch/line
gaps in ``build.py``, the dispatch backends, ``grn_step/backend`` fallback and
the numba/numPy impls that are only reachable under unusual conditions
(missing optional deps, empty build outputs, degenerate pivots).
"""
import builtins
import importlib.util

import pytest

from helixlang._accel import build as build_mod
from helixlang._accel.dispatch import backend as dispatch_backend
from helixlang._accel.dispatch import impl_python as dispatch_python
from helixlang._accel.grn_step import backend as grn_backend
from helixlang._accel.simplex import impl_numpy as simplex_numpy
from helixlang.core.errors import NativeBackendError


def _load_without_numba(module_path: str):
    """Exec a numba impl module source with ``import numba`` failing, under a
    throwaway module name so the canonical ``sys.modules`` entry is untouched."""
    spec = importlib.util.find_spec(module_path)
    throwaway = importlib.util.spec_from_file_location(
        "_no_numba_" + module_path.rsplit(".", 1)[-1], spec.origin)
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "numba" or name.startswith("numba."):
            raise ImportError("numba blocked")
        return real_import(name, *a, **k)

    mod = importlib.util.module_from_spec(throwaway)
    builtins.__import__ = fake_import
    try:
        throwaway.loader.exec_module(mod)  # type: ignore[union-attr]
    finally:
        builtins.__import__ = real_import
    return mod


class TestBootstrapImportsWithoutCython:
    def test_cython_missing_sets_cythonize_none(self, monkeypatch):
        from helixlang._accel.build import _bootstrap_imports
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "Cython" or name.startswith("Cython"):
                raise ImportError("Cython blocked")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        cythonize, Extension, build_ext, Distribution = _bootstrap_imports()
        assert cythonize is None
        assert callable(Extension)
        assert callable(build_ext)
        assert callable(Distribution)


class TestBuildExtensionsNoSo:
    def test_cythonize_none_with_c_sources_no_so(self, monkeypatch, tmp_path):
        fake_c = tmp_path / "dummy.c"
        fake_c.write_text("")
        monkeypatch.setattr(build_mod, "_BASE", tmp_path)
        monkeypatch.setattr(build_mod, "_cython_sources", lambda: [])
        monkeypatch.setattr(build_mod, "_c_sources", lambda: [fake_c])
        monkeypatch.setattr(build_mod, "_build_rust_backends", lambda: 0)

        class FakeExtension:
            def __init__(self, *a, **kw):
                pass

        class FakeBuildExt:
            def __init__(self, dist):
                pass

            def ensure_finalized(self):
                pass

            def run(self):
                pass

        class FakeDistribution:
            def __init__(self, *a, **kw):
                pass

        def fake_bootstrap():
            return None, FakeExtension, FakeBuildExt, FakeDistribution

        monkeypatch.setattr(build_mod, "_bootstrap_imports", fake_bootstrap)
        rc = build_mod.build_extensions()
        assert rc == 1


class TestGrnBackendFallback:
    def test_step_mixed_falls_back_when_module_lacks_hook(self, monkeypatch):
        class NoMixedStep:
            pass

        monkeypatch.setattr(
            grn_backend, "load_hot", lambda *a, **k: NoMixedStep())

        levels = [0.5, 0.4]
        src = [0, 1]
        dst = [1, 0]
        weights = [1.0, -1.0]
        decays = [0.5, 0.5]
        thresholds = [0.5, 0.5]
        default_decay = 0.5
        hill_ns = [None, 1.0]
        kds = [None, 0.3]

        new_levels, triggered = grn_backend.step_mixed(
            levels, src, dst, weights, decays, thresholds, default_decay,
            hill_ns, kds)

        from helixlang._accel.grn_step import impl_python
        exp_levels, exp_triggered = impl_python.step_mixed(
            levels, src, dst, weights, decays, thresholds, default_decay,
            hill_ns, kds)
        assert new_levels == exp_levels
        assert triggered == exp_triggered


class TestNumbaImplsWithoutNumba:
    def test_diffusion_impl_numba_raises_when_numba_absent(self):
        mod = _load_without_numba("helixlang._accel.diffusion.impl_numba")
        assert mod._HAS_NUMBA is False
        with pytest.raises(NativeBackendError):
            mod.step([[1.0]], [[1.0]], 0.035, 0.065, 0.16, 0.08)

    def test_simplex_impl_numba_raises_when_numba_absent(self):
        mod = _load_without_numba("helixlang._accel.simplex.impl_numba")
        assert mod.njit is None
        assert mod._RUN_NB is mod._run_nb
        with pytest.raises(RuntimeError):
            mod.run([[1.0, 0.0, 1.0]], [0], [0.0], 1)


class TestDispatchEdges:
    def test_run_quota_exits_via_condition(self):
        code = [0x20, 0, 0x20, 1, 0x90]
        ops, stack, halted = dispatch_python.run_quota(
            code, [1.0, 2.0], quota=4096)
        assert (ops, stack, halted) == (3, [3.0], False)

    def test_run_many_dispatches_per_cell(self):
        code = [0x20, 0, 0x20, 1, 0x90]
        out = dispatch_backend.run_many(code, [2.0, 5.0], n_cells=2)
        assert len(out) == 2
        for ops, stack, halted in out:
            assert (ops, stack, halted) == (3, [7.0], False)

    def test_run_many_single_cell_default(self):
        code = [0x20, 0]
        out = dispatch_backend.run_many(code, [3.0])
        assert out == [(1, [3.0], False)]


class TestSimplexNumpyDegeneratePivot:
    def test_infinite_rhs_avoids_tie_break(self):
        import numpy as np
        tbl = [[1.0, 1.0, 0.0, float("inf")]]
        basis = [2]
        with np.errstate(invalid="ignore"):
            res = simplex_numpy.run(tbl, basis, [1.0, 0.0, 0.0], 3)
        assert res in ("optimal", "unbounded", "max_iter")
        assert basis[0] == 0
