"""Targeted tests to close all remaining coverage gaps in _accel/.

Covers:
  - _accel/grn_step/impl_python.py: _hill edge cases, step_mixed, clipping
  - _accel/grn_step/impl_numpy.py: step_mixed
  - _accel/grn_step/impl_numba.py: njit kernels (via NUMBA_DISABLE_JIT), fallback
  - _accel/diffusion/impl_python.py: clamping branches
  - _accel/diffusion/impl_numba.py: njit kernel (via NUMBA_DISABLE_JIT)
  - _accel/simplex/impl_numpy.py: edge cases (empty, forbidden, tied, unbounded)
  - _accel/simplex/impl_numba.py: full kernel (via NUMBA_DISABLE_JIT)
  - _accel/build.py: all build helpers
"""
from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path

import pytest

from helixlang.core.errors import NativeBackendError

# ============================================================================
# _accel/grn_step/impl_python.py — _hill, step_mixed, clipping edge cases
# ============================================================================

class TestGrnImplPythonEdgeCases:
    def test_sigmoid_positive_branch(self):
        from helixlang._accel.grn_step.impl_python import _sigmoid
        assert _sigmoid(0.0) == pytest.approx(0.5)
        assert _sigmoid(100.0) > 0.99

    def test_sigmoid_negative_branch(self):
        from helixlang._accel.grn_step.impl_python import _sigmoid
        assert _sigmoid(-100.0) < 0.01

    def test_hill_x_zero(self):
        from helixlang._accel.grn_step.impl_python import _hill
        assert _hill(0.0, 2, 1.0) == 0.0

    def test_hill_x_negative(self):
        from helixlang._accel.grn_step.impl_python import _hill
        assert _hill(-1.0, 2, 1.0) == 0.0

    def test_hill_kdn_zero(self):
        from helixlang._accel.grn_step.impl_python import _hill
        assert _hill(1.0, 2, 0.0) == 1.0

    def test_hill_normal(self):
        from helixlang._accel.grn_step.impl_python import _hill
        assert _hill(1.0, 2, 1.0) == pytest.approx(0.5)

    def test_step_clipping_above_one(self):
        from helixlang._accel.grn_step.impl_python import step
        levels = [2.0]
        src, dst, weights = [], [], []
        decays = [1.0]
        thresholds = [0.0]
        new, trig = step(levels, src, dst, weights, decays, thresholds, 0.0)
        assert new[0] == 1.0

    def test_step_clipping_below_zero(self):
        from helixlang._accel.grn_step.impl_python import step
        levels = [-1.0]
        src, dst, weights = [0], [0], [0.0]
        decays = [0.5]
        thresholds = [10.0]
        new, trig = step(levels, src, dst, weights, decays, thresholds, 0.0)
        assert new[0] == 0.0

    def test_step_mixed_hill_path(self):
        from helixlang._accel.grn_step.impl_python import step_mixed
        levels = [0.9, 0.2]
        src = [0]
        dst = [1]
        weights = [0.8]
        decays = [0.5, 0.5]
        thresholds = [0.3, 0.3]
        hill_ns = [None, 2.0]
        kds = [None, 0.5]
        new, trig = step_mixed(levels, src, dst, weights, decays, thresholds,
                               0.99, hill_ns, kds)
        assert len(new) == 2

    def test_step_mixed_hill_kd_none_fallback(self):
        from helixlang._accel.grn_step.impl_python import step_mixed
        levels = [0.9, 0.2]
        src = [0]
        dst = [1]
        weights = [0.8]
        decays = [0.5, 0.5]
        thresholds = [0.3, 0.3]
        hill_ns = [None, 3.0]
        kds = [None, None]
        new, trig = step_mixed(levels, src, dst, weights, decays, thresholds,
                               0.99, hill_ns, kds)
        assert len(new) == 2

    def test_step_mixed_hill_x_zero(self):
        from helixlang._accel.grn_step.impl_python import step_mixed
        levels = [0.0, 0.0]
        src = []
        dst = []
        weights = []
        decays = [0.5, 0.5]
        thresholds = [0.0, 0.0]
        hill_ns = [None, 2.0]
        kds = [None, 1.0]
        new, trig = step_mixed(levels, src, dst, weights, decays, thresholds,
                               0.99, hill_ns, kds)
        assert new[1] == 0.0

    def test_step_mixed_clipping_above_one(self):
        from helixlang._accel.grn_step.impl_python import step_mixed
        levels = [2.0]
        src = []
        dst = []
        weights = []
        decays = [1.0]
        thresholds = [0.0]
        hill_ns = [None]
        kds = [None]
        new, trig = step_mixed(levels, src, dst, weights, decays, thresholds,
                               0.0, hill_ns, kds)
        assert new[0] == 1.0


# ============================================================================
# _accel/grn_step/impl_numpy.py — step_mixed
# ============================================================================

class TestGrnImplNumpyStepMixed:
    def test_step_mixed_basic(self):
        from helixlang._accel.grn_step.impl_numpy import step_mixed
        levels = [0.9, 0.2, 0.5, 0.0]
        src = [0, 1, 1]
        dst = [2, 2, 3]
        weights = [0.5, 0.8, -0.3]
        decays = [0.5, None, 0.1, None]
        thresholds = [0.3, 0.3, 0.3, 0.3]
        hill_ns = [None, None, None, None]
        kds = [None, None, None, None]
        new, trig = step_mixed(levels, src, dst, weights, decays, thresholds,
                               0.99, hill_ns, kds)
        assert len(new) == 4

    def test_step_mixed_with_hill(self):
        from helixlang._accel.grn_step.impl_numpy import step_mixed
        levels = [0.9, 0.2]
        src = [0]
        dst = [1]
        weights = [0.8]
        decays = [0.5, 0.5]
        thresholds = [0.3, 0.3]
        hill_ns = [None, 2.0]
        kds = [None, 0.5]
        new, trig = step_mixed(levels, src, dst, weights, decays, thresholds,
                               0.99, hill_ns, kds)
        assert len(new) == 2

    def test_step_mixed_hill_kd_none_fallback(self):
        from helixlang._accel.grn_step.impl_numpy import step_mixed
        levels = [0.9, 0.2]
        src = [0]
        dst = [1]
        weights = [0.8]
        decays = [0.5, 0.5]
        thresholds = [0.3, 0.3]
        hill_ns = [None, 3.0]
        kds = [None, None]
        new, trig = step_mixed(levels, src, dst, weights, decays, thresholds,
                               0.99, hill_ns, kds)
        assert len(new) == 2


# ============================================================================
# _accel/grn_step/impl_numba.py — njit kernels via NUMBA_DISABLE_JIT
# ============================================================================

def _reload_numba_mod(module_path: str):
    """Reload a numba impl module with JIT disabled so coverage can trace it."""
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    mod = importlib.import_module(module_path)
    importlib.reload(mod)
    return mod


def _load_without_numba(module_path: str):
    """Exec a numba impl module source with ``import numba`` failing, under a
    throwaway module name so the canonical ``sys.modules`` entry is untouched."""
    import builtins
    import importlib.util

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


class TestGrnImplNumba:
    def test_step_basic(self):
        mod = _reload_numba_mod("helixlang._accel.grn_step.impl_numba")
        levels = [0.9, 0.2, 0.5, 0.0]
        src = [0, 1, 1]
        dst = [2, 2, 3]
        weights = [0.5, 0.8, -0.3]
        decays = [0.5, None, 0.1, None]
        thresholds = [0.3, 0.3, 0.3, 0.3]
        new, trig = mod.step(levels, src, dst, weights, decays, thresholds, 0.99)
        assert len(new) == 4
        assert isinstance(trig, list)

    def test_step_negative_input(self):
        mod = _reload_numba_mod("helixlang._accel.grn_step.impl_numba")
        levels = [0.5, 0.5]
        src = [0, 1]
        dst = [1, 0]
        weights = [-1.0, -1.0]
        decays = [0.5, 0.5]
        thresholds = [0.0, 0.0]
        new, trig = mod.step(levels, src, dst, weights, decays, thresholds, 0.5)
        assert len(new) == 2

    def test_step_nan_decay_uses_default(self):
        mod = _reload_numba_mod("helixlang._accel.grn_step.impl_numba")
        levels = [0.5]
        src = []
        dst = []
        weights = []
        decays = [None]
        thresholds = [0.0]
        new, trig = mod.step(levels, src, dst, weights, decays, thresholds, 0.5)
        assert len(new) == 1

    def test_step_clipping(self):
        mod = _reload_numba_mod("helixlang._accel.grn_step.impl_numba")
        levels = [2.0]
        src = []
        dst = []
        weights = []
        decays = [1.0]
        thresholds = [0.0]
        new, trig = mod.step(levels, src, dst, weights, decays, thresholds, 0.0)
        assert new[0] == 1.0

    def test_step_mixed_basic(self):
        mod = _reload_numba_mod("helixlang._accel.grn_step.impl_numba")
        levels = [0.9, 0.2, 0.5, 0.0]
        src = [0, 1, 1]
        dst = [2, 2, 3]
        weights = [0.5, 0.8, -0.3]
        decays = [0.5, None, 0.1, None]
        thresholds = [0.3, 0.3, 0.3, 0.3]
        hill_ns = [None, None, None, None]
        kds = [None, None, None, None]
        new, trig = mod.step_mixed(levels, src, dst, weights, decays, thresholds,
                                   0.99, hill_ns, kds)
        assert len(new) == 4

    def test_step_mixed_with_hill(self):
        mod = _reload_numba_mod("helixlang._accel.grn_step.impl_numba")
        levels = [0.9, 0.2]
        src = [0]
        dst = [1]
        weights = [0.8]
        decays = [0.5, 0.5]
        thresholds = [0.3, 0.3]
        hill_ns = [None, 2.0]
        kds = [None, 0.5]
        new, trig = mod.step_mixed(levels, src, dst, weights, decays, thresholds,
                                   0.99, hill_ns, kds)
        assert len(new) == 2

    def test_step_mixed_hill_kd_nan_fallback(self):
        mod = _reload_numba_mod("helixlang._accel.grn_step.impl_numba")
        levels = [0.9, 0.2]
        src = [0]
        dst = [1]
        weights = [0.8]
        decays = [0.5, 0.5]
        thresholds = [0.3, 0.3]
        hill_ns = [None, 3.0]
        kds = [None, None]
        new, trig = mod.step_mixed(levels, src, dst, weights, decays, thresholds,
                                   0.99, hill_ns, kds)
        assert len(new) == 2

    def test_step_mixed_hill_kdn_zero(self):
        mod = _reload_numba_mod("helixlang._accel.grn_step.impl_numba")
        levels = [0.5, 0.5]
        src = [0]
        dst = [1]
        weights = [1.0]
        decays = [0.5, 0.5]
        thresholds = [0.0, 0.0]
        hill_ns = [None, 2.0]
        kds = [None, 0.0]
        new, trig = mod.step_mixed(levels, src, dst, weights, decays, thresholds,
                                   0.5, hill_ns, kds)
        assert len(new) == 2

    def test_step_mixed_hill_x_zero(self):
        mod = _reload_numba_mod("helixlang._accel.grn_step.impl_numba")
        levels = [0.0, 0.0]
        src = []
        dst = []
        weights = []
        decays = [0.5, 0.5]
        thresholds = [0.0, 0.0]
        hill_ns = [None, 2.0]
        kds = [None, 1.0]
        new, trig = mod.step_mixed(levels, src, dst, weights, decays, thresholds,
                                   0.99, hill_ns, kds)
        assert new[1] == 0.0

    def test_step_mixed_clipping(self):
        mod = _reload_numba_mod("helixlang._accel.grn_step.impl_numba")
        levels = [2.0, 2.0]
        src = []
        dst = []
        weights = []
        decays = [1.0, 1.0]
        thresholds = [-10.0, -10.0]
        hill_ns = [None, None]
        kds = [None, None]
        new, trig = mod.step_mixed(levels, src, dst, weights, decays, thresholds,
                                   0.0, hill_ns, kds)
        assert new[0] == 1.0

    def test_step_mixed_sigmoid_clipping(self):
        mod = _reload_numba_mod("helixlang._accel.grn_step.impl_numba")
        levels = [-1.0]
        src = [0]
        dst = [0]
        weights = [0.0]
        decays = [0.5]
        thresholds = [10.0]
        hill_ns = [None]
        kds = [None]
        new, trig = mod.step_mixed(levels, src, dst, weights, decays, thresholds,
                                   0.0, hill_ns, kds)
        assert new[0] == 0.0

    def test_step_fallback_raises(self):
        mod = _reload_numba_mod("helixlang._accel.grn_step.impl_numba")
        old = mod._HAS_NUMBA
        try:
            mod._HAS_NUMBA = False
            with pytest.raises(NativeBackendError):
                mod.step([0.5], [], [], [], [0.5], [0.0], 0.5)
        finally:
            mod._HAS_NUMBA = old

    def test_step_mixed_fallback_raises(self):
        mod = _reload_numba_mod("helixlang._accel.grn_step.impl_numba")
        old = mod._HAS_NUMBA
        try:
            mod._HAS_NUMBA = False
            with pytest.raises(NativeBackendError):
                mod.step_mixed([0.5], [], [], [], [0.5], [0.0], 0.5,
                               [None], [None])
        finally:
            mod._HAS_NUMBA = old

    def test_module_reload_without_numba_covers_false_branch(self):
        mod = _load_without_numba(
            "helixlang._accel.grn_step.impl_numba")
        assert mod._HAS_NUMBA is False
        with pytest.raises(NativeBackendError):
            mod.step([0.5], [], [], [], [0.5], [0.0], 0.5)
        with pytest.raises(NativeBackendError):
            mod.step_mixed([0.5], [], [], [], [0.5], [0.0], 0.5,
                           [None], [None])


# ============================================================================
# _accel/diffusion/impl_python.py — clamping branches
# ============================================================================

class TestDiffusionImplPythonClamping:
    def test_clamp_above_one(self):
        from helixlang._accel.diffusion.impl_python import step
        n = 4
        u = [[2.0] * n for _ in range(n)]
        v = [[0.1] * n for _ in range(n)]
        F, k, Du, Dv = 0.035, 0.065, 0.16, 0.08
        nu, nv = step(u, v, F, k, Du, Dv)
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                assert 0.0 <= nu[i][j] <= 1.0
                assert 0.0 <= nv[i][j] <= 1.0

    def test_clamp_below_zero(self):
        from helixlang._accel.diffusion.impl_python import step
        n = 4
        u = [[0.1] * n for _ in range(n)]
        v = [[2.0] * n for _ in range(n)]
        F, k, Du, Dv = 0.035, 0.065, 0.16, 0.08
        nu, nv = step(u, v, F, k, Du, Dv)
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                assert 0.0 <= nu[i][j] <= 1.0
                assert 0.0 <= nv[i][j] <= 1.0

    def test_clamp_v_below_zero(self):
        from helixlang._accel.diffusion.impl_python import step
        n = 4
        u = [[0.0] * n for _ in range(n)]
        v = [[1.0] * n for _ in range(n)]
        F, k, Du, Dv = 0.035, 1.0, 0.16, 0.08
        nu, nv = step(u, v, F, k, Du, Dv)
        for i in range(n):
            for j in range(n):
                assert 0.0 <= nu[i][j] <= 1.0
                assert 0.0 <= nv[i][j] <= 1.0

    def test_borders_preserved(self):
        from helixlang._accel.diffusion.impl_python import step
        n = 4
        u = [[0.5] * n for _ in range(n)]
        v = [[0.3] * n for _ in range(n)]
        u[0][0] = 0.1
        v[0][0] = 0.9
        F, k, Du, Dv = 0.035, 0.065, 0.16, 0.08
        nu, nv = step(u, v, F, k, Du, Dv)
        assert nu[0][0] == 0.1
        assert nv[0][0] == 0.9
        assert nu[-1][-1] == u[-1][-1]
        assert nv[-1][-1] == v[-1][-1]


# ============================================================================
# _accel/diffusion/impl_numba.py — njit kernel via NUMBA_DISABLE_JIT
# ============================================================================

class TestDiffusionImplNumba:
    def test_step_basic(self):
        mod = _reload_numba_mod("helixlang._accel.diffusion.impl_numba")
        n = 6
        u = [[1.0] * n for _ in range(n)]
        v = [[0.0] * n for _ in range(n)]
        mid = n // 2
        for i in range(mid - 1, mid + 1):
            for j in range(mid - 1, mid + 1):
                u[i][j] = 0.5
                v[i][j] = 0.25
        F, k, Du, Dv = 0.035, 0.065, 0.16, 0.08
        nu, nv = mod.step(u, v, F, k, Du, Dv)
        import numpy as np
        nu_arr = np.asarray(nu)
        nv_arr = np.asarray(nv)
        assert float(nu_arr[0, 0]) == 1.0
        assert float(nv_arr[0, 0]) == 0.0

    def test_step_clamp_above(self):
        mod = _reload_numba_mod("helixlang._accel.diffusion.impl_numba")
        n = 6
        u = [[1.0] * n for _ in range(n)]
        v = [[0.0] * n for _ in range(n)]
        u[2][2] = 0.99
        v[2][2] = 0.99
        F, k, Du, Dv = 0.035, 0.065, 0.16, 0.08
        nu, nv = mod.step(u, v, F, k, Du, Dv)
        import numpy as np
        assert float(np.asarray(nu).max()) <= 1.0
        assert float(np.asarray(nv).max()) <= 1.0

    def test_step_clamp_below(self):
        mod = _reload_numba_mod("helixlang._accel.diffusion.impl_numba")
        n = 6
        u = [[0.0] * n for _ in range(n)]
        v = [[0.0] * n for _ in range(n)]
        u[2][2] = 0.001
        v[2][2] = 0.5
        F, k, Du, Dv = 0.035, 0.065, 0.16, 0.08
        nu, nv = mod.step(u, v, F, k, Du, Dv)
        import numpy as np
        assert float(np.asarray(nu).min()) >= 0.0
        assert float(np.asarray(nv).min()) >= 0.0


# ============================================================================
# _accel/simplex/impl_numpy.py — edge cases
# ============================================================================

class TestSimplexImplNumpyEdgeCases:
    def test_empty_tableau(self):
        from helixlang._accel.simplex.impl_numpy import run
        status = run([], [], [1.0], 1)
        assert status == "optimal"

    def test_forbidden_variable(self):
        from helixlang._accel.simplex.impl_numpy import run
        tab = [[1.0, 1.0, 1.0, 4.0]]
        basis = [2]
        obj = [3.0, 2.0, 0.0]
        status = run(tab, basis, obj, 3, forbidden=[0])
        assert status == "optimal"

    def test_tied_ratio_takes_smaller_basis(self):
        from helixlang._accel.simplex.impl_numpy import run
        tab = [[1.0, 1.0, 1.0, 0.0, 4.0], [2.0, 0.0, 0.0, 1.0, 8.0]]
        basis = [2, 3]
        obj = [3.0, 1.0, 0.0, 0.0]
        status = run(tab, basis, obj, 4)
        assert status == "optimal"

    def test_max_iter_exhausted(self):
        from helixlang._accel.simplex.impl_numpy import run
        tab = [[1.0, 1.0, 1.0, 0.0, 4.0], [2.0, 1.0, 0.0, 1.0, 5.0]]
        basis = [2, 3]
        status = run(tab, basis, [3.0, 2.0, 0.0, 0.0], 4, max_iter=1)
        assert status == "max_iter"

    def test_normal_problem(self):
        from helixlang._accel.simplex.impl_numpy import run
        tab = [[1.0, 1.0, 1.0, 0.0, 4.0], [1.0, 0.0, 0.0, 1.0, 2.0]]
        basis = [2, 3]
        status = run(tab, basis, [2.0, 3.0, 0.0, 0.0], 4)
        assert status == "optimal"

    def test_no_eligible_column(self):
        from helixlang._accel.simplex.impl_numpy import run
        tab = [[0.0, 0.0, 1.0, 0.0, 2.0]]
        basis = [2]
        status = run(tab, basis, [0.0, 0.0, 0.0, 0.0], 4)
        assert status == "optimal"

    def test_no_valid_pivot_row(self):
        from helixlang._accel.simplex.impl_numpy import run
        tab = [[-1.0, 1.0, 2.0]]
        basis = [1]
        status = run(tab, basis, [1.0, 0.0], 2)
        assert status == "unbounded"

    def test_numpy_array_input(self):
        import numpy as np

        from helixlang._accel.simplex.impl_numpy import run
        tab = np.array([[1.0, 1.0, 1.0, 0.0, 4.0],
                        [1.0, 0.0, 0.0, 1.0, 2.0]], dtype=np.float64)
        basis = [2, 3]
        obj = np.array([2.0, 3.0, 0.0, 0.0], dtype=np.float64)
        status = run(tab, basis, obj, 4)
        assert status == "optimal"

    def test_no_tie_in_ratio(self):
        from helixlang._accel.simplex.impl_numpy import run
        tab = [[1.0, 2.0, 1.0, 0.0, 4.0], [3.0, 1.0, 0.0, 1.0, 6.0]]
        basis = [2, 3]
        obj = [5.0, 3.0, 0.0, 0.0]
        status = run(tab, basis, obj, 4)
        assert status == "optimal"


# ============================================================================
# _accel/simplex/impl_numba.py — full kernel via NUMBA_DISABLE_JIT
# ============================================================================

class TestSimplexImplNumba:
    def _reload(self):
        return _reload_numba_mod("helixlang._accel.simplex.impl_numba")

    def test_normal_problem(self):
        mod = self._reload()
        tab = [[1.0, 1.0, 1.0, 0.0, 4.0], [1.0, 0.0, 0.0, 1.0, 2.0]]
        basis = [2, 3]
        status = mod.run(tab, basis, [2.0, 3.0, 0.0, 0.0], 4)
        assert status == "optimal"

    def test_empty_tableau(self):
        mod = self._reload()
        status = mod.run([], [], [1.0], 1)
        assert status == "optimal"

    def test_unbounded(self):
        mod = self._reload()
        tab = [[-1.0, 1.0, 2.0]]
        basis = [1]
        status = mod.run(tab, basis, [1.0, 0.0], 2)
        assert status == "unbounded"

    def test_max_iter(self):
        mod = self._reload()
        tab = [[1.0, 1.0, 1.0, 0.0, 4.0], [2.0, 1.0, 0.0, 1.0, 5.0]]
        basis = [2, 3]
        status = mod.run(tab, basis, [3.0, 2.0, 0.0, 0.0], 4, max_iter=1)
        assert status == "max_iter"

    def test_forbidden_variable(self):
        mod = self._reload()
        tab = [[1.0, 1.0, 1.0, 4.0]]
        basis = [2]
        obj = [3.0, 2.0, 0.0]
        status = mod.run(tab, basis, obj, 3, forbidden=[0])
        assert status == "optimal"

    def test_ratio_tie_break(self):
        mod = self._reload()
        tab = [[1.0, 1.0, 1.0, 0.0, 4.0], [2.0, 0.0, 0.0, 1.0, 8.0]]
        basis = [2, 3]
        obj = [3.0, 1.0, 0.0, 0.0]
        status = mod.run(tab, basis, obj, 4)
        assert status == "optimal"

    def test_no_eligible_column(self):
        mod = self._reload()
        tab = [[0.0, 0.0, 1.0, 0.0, 2.0]]
        basis = [2]
        status = mod.run(tab, basis, [0.0, 0.0, 0.0], 2)
        assert status == "optimal"

    def test_no_valid_pivot_row(self):
        mod = self._reload()
        tab = [[-1.0, 1.0, 2.0]]
        basis = [1]
        status = mod.run(tab, basis, [1.0, 0.0], 2)
        assert status == "unbounded"

    def test_small_factor_skip(self):
        mod = self._reload()
        tab = [[1.0, 0.0, 1.0, 0.0, 2.0], [1e-12, 1.0, 0.0, 1.0, 3.0]]
        basis = [2, 3]
        status = mod.run(tab, basis, [1.0, 0.0, 0.0, 0.0], 2)
        assert status in ("optimal", "unbounded", "max_iter")


# ============================================================================
# _accel/build.py — build helpers
# ============================================================================

class TestBuildHelpers:
    def test_cython_sources(self):
        from helixlang._accel.build import _cython_sources
        result = _cython_sources()
        assert isinstance(result, list)

    def test_c_sources(self):
        from helixlang._accel.build import _c_sources
        result = _c_sources()
        assert isinstance(result, list)

    def test_module_name(self):
        from helixlang._accel.build import _BASE, _module_name
        p = _BASE / "grn_step" / "impl_cython.pyx"
        assert _module_name(p) == "helixlang._accel.grn_step.impl_cython"

    def test_module_name_c(self):
        from helixlang._accel.build import _BASE, _module_name
        p = _BASE / "simplex" / "impl_cext.c"
        assert _module_name(p) == "helixlang._accel.simplex.impl_cext"

    def test_rust_crates(self):
        from helixlang._accel.build import _rust_crates
        result = _rust_crates()
        assert isinstance(result, list)

    def test_root_src(self):
        from helixlang._accel.build import _root_src
        result = _root_src()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_bootstrap_imports(self):
        from helixlang._accel.build import _bootstrap_imports
        cythonize, Extension, build_ext, Distribution = _bootstrap_imports()
        assert Extension is not None
        assert build_ext is not None
        assert Distribution is not None

    def test_build_extensions_no_sources(self, monkeypatch):
        from helixlang._accel import build as build_mod
        monkeypatch.setattr(build_mod, "_cython_sources", lambda: [])
        monkeypatch.setattr(build_mod, "_c_sources", lambda: [])
        rc = build_mod.build_extensions()
        assert rc == 0

    def test_build_extensions_cython_missing(self, monkeypatch):
        from helixlang._accel import build as build_mod
        monkeypatch.setattr(build_mod, "_cython_sources",
                            lambda: [Path("test.pyx")])
        monkeypatch.setattr(build_mod, "_c_sources", lambda: [])

        def fake_bootstrap():
            return None, None, None, None

        monkeypatch.setattr(build_mod, "_bootstrap_imports", fake_bootstrap)
        rc = build_mod.build_extensions()
        assert rc == 1

    def test_build_extensions_build_fails(self, monkeypatch):
        from helixlang._accel import build as build_mod
        fake_path = build_mod._BASE / "grn_step" / "impl_cython.pyx"

        monkeypatch.setattr(build_mod, "_cython_sources",
                            lambda: [fake_path])
        monkeypatch.setattr(build_mod, "_c_sources", lambda: [])

        class FakeExtension:
            def __init__(self, *a, **kw):
                pass

        class FakeBuildExt:
            def __init__(self, dist):
                pass
            def ensure_finalized(self):
                pass
            def run(self):
                raise Exception("build failed")

        class FakeDistribution:
            def __init__(self, *a, **kw):
                pass

        def fake_cythonize(exts, **kw):
            return exts

        def fake_bootstrap():
            return fake_cythonize, FakeExtension, FakeBuildExt, FakeDistribution

        monkeypatch.setattr(build_mod, "_bootstrap_imports", fake_bootstrap)
        rc = build_mod.build_extensions()
        assert rc == 1

    def test_build_extensions_success(self, monkeypatch):
        from helixlang._accel import build as build_mod

        monkeypatch.setattr(build_mod, "_cython_sources", lambda: [])
        monkeypatch.setattr(build_mod, "_c_sources", lambda: [])

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
        assert rc == 0

    def test_build_extensions_ext_and_cython(self, monkeypatch):
        from helixlang._accel import build as build_mod

        fake_pyx = build_mod._BASE / "grn_step" / "impl_cython.pyx"
        fake_c = build_mod._BASE / "simplex" / "impl_cext.c"

        monkeypatch.setattr(build_mod, "_cython_sources",
                            lambda: [fake_pyx])
        monkeypatch.setattr(build_mod, "_c_sources",
                            lambda: [fake_c])

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

        def fake_cythonize(exts, **kw):
            return exts

        def fake_bootstrap():
            return fake_cythonize, FakeExtension, FakeBuildExt, FakeDistribution

        monkeypatch.setattr(build_mod, "_bootstrap_imports", fake_bootstrap)
        rc = build_mod.build_extensions()
        assert rc == 0

    def test_build_rust_backends_no_crates(self, monkeypatch):
        from helixlang._accel import build as build_mod
        monkeypatch.setattr(build_mod, "_rust_crates", lambda: [])
        rc = build_mod._build_rust_backends()
        assert rc == 0

    def test_build_rust_backends_with_crates(self, monkeypatch, tmp_path):
        from helixlang._accel import build as build_mod
        crate_dir = tmp_path / "grn_step" / "rust"
        crate_dir.mkdir(parents=True)
        cargo_toml = crate_dir / "Cargo.toml"
        cargo_toml.write_text("[package]\nname = 'test'\n")

        monkeypatch.setattr(build_mod, "_rust_crates", lambda: [cargo_toml])

        def fake_run(cmd, **kw):
            raise FileNotFoundError("no cargo")

        monkeypatch.setattr(subprocess, "run", fake_run)
        rc = build_mod._build_rust_backends()
        assert rc == 0

    def test_build_rust_backends_cargo_fails(self, monkeypatch, tmp_path):
        from helixlang._accel import build as build_mod

        crate_dir = tmp_path / "grn_step" / "rust"
        crate_dir.mkdir(parents=True)
        cargo_toml = crate_dir / "Cargo.toml"
        cargo_toml.write_text("[package]\nname = 'test'\n")

        monkeypatch.setattr(build_mod, "_rust_crates", lambda: [cargo_toml])

        def fake_run(cmd, **kw):
            raise subprocess.CalledProcessError(1, cmd, stderr="fail")

        monkeypatch.setattr(subprocess, "run", fake_run)
        rc = build_mod._build_rust_backends()
        assert rc == 0

    def test_main(self, monkeypatch):
        from helixlang._accel import build as build_mod
        monkeypatch.setattr(build_mod, "build_extensions", lambda: 0)
        with pytest.raises(SystemExit) as exc_info:
            build_mod.main()
        assert exc_info.value.code == 0
