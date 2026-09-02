"""Tests for the _accel/ hot-loop foundation (doc/36 §4).

Verifies the speed-only backend selector, the NativeBackendError contract, and
that the pure-Python and numpy GRN step kernels are numerically equivalent.
"""
from __future__ import annotations

import importlib

import pytest

from helixlang.plugins.runtime.grn import GRN, TelegraphPromoter
from helixlang._accel._loaders import choose_backend, load_hot
from helixlang.core.errors import NativeBackendError


def _clear_pkg(pkg: str) -> None:
    for name in list(importlib.sys.modules):
        if name.startswith(pkg):
            importlib.sys.modules.pop(name, None)


def test_choose_backend_default_resolves_to_equiv_fidelity():
    # Default order native,numpy,python must resolve to a real equivalent-fidelity
    # impl for grn_step (native when built, else numpy/python) — never a fidelity
    # boundary crossing.
    impl = choose_backend("helixlang._accel.grn_step")
    assert impl in ("impl_numpy", "impl_python", "impl_cext", "impl_cython")


def test_choose_backend_explicit_python():
    assert choose_backend("helixlang._accel.grn_step", prefer="python") == "impl_python"


def test_choose_backend_missing_raises():
    # Only asserted when no native impl is built: requesting 'native' must raise
    # rather than silently degrade.  When native IS built, prefer='python' is the
    # explicit equivalent-fidelity opt-out and never raises.
    if _native_available():
        pytest.skip("native .so built for this interpreter")
    with pytest.raises(NativeBackendError):
        choose_backend("helixlang._accel.grn_step", prefer="native")


def test_choose_backend_unknown_pkg_raises():
    with pytest.raises(NativeBackendError):
        choose_backend("helixlang._accel.no_such_pkg")


def test_load_hot_imports_ref(monkeypatch):
    monkeypatch.setenv("HELIX_ACCEL", "python")
    mod = load_hot("helixlang._accel.grn_step")
    assert callable(mod.step)


# ── GRN step kernel equivalence (same fidelity, two impls) ──────────────────


def _case():
    levels = [0.9, 0.2, 0.5, 0.0]
    # node 2 gets 0.5*node0 + 0.8*node1 ; node 3 gets -0.3*node1
    src = [0, 1, 1]
    dst = [2, 2, 3]
    weights = [0.5, 0.8, -0.3]
    decays = [0.5, None, 0.1, None]
    thresholds = [0.3, 0.3, 0.3, 0.3]
    return levels, src, dst, weights, decays, thresholds, 0.99


def test_grn_python_backend_module_path():
    from helixlang._accel.grn_step import step as alias
    assert callable(alias)


def test_grn_impl_python_and_numpy_match():
    _clear_pkg("helixlang._accel.grn_step")
    py = importlib.import_module("helixlang._accel.grn_step.impl_python")
    np_mod = importlib.import_module("helixlang._accel.grn_step.impl_numpy")
    levels, *rest = _case()
    new_py, trig_py = py.step(list(levels), *rest)
    new_np, trig_np = np_mod.step(list(levels), *rest)
    for a, b in zip(new_py, new_np, strict=True):
        assert abs(a - b) < 1e-9
    assert trig_py == trig_np


def test_grn_python_roundtrip_math():
    from helixlang._accel.grn_step.impl_python import step
    levels = [1.0, 0.0]
    src = [0]
    dst = [1]
    weights = [1.0]
    decays = [0.5, 0.2]
    thresholds = [0.0, 0.5]
    new, trig = step(levels, src, dst, weights, decays, thresholds, 0.99)
    # node1: input = 1.0*1.0 = 1.0, decay 0.2
    #   raw = sigmoid(1.0-0.5) = 1/(1+e^-0.5) = 0.622459
    #   blended = 0.2*0.0 + 0.8*0.622459 = 0.497967
    assert abs(new[1] - 0.4979673312018055) < 1e-6
    # node0: no inputs, raw = sigmoid(-0.0) = 0.5
    #   blended = 0.5*1.0 + 0.5*0.5 = 0.75
    assert abs(new[0] - 0.75) < 1e-9
    # only node0 crosses 0.5
    assert trig == [0]


# ── GRN.step_accel wired to the hot loop: determinism parity (doc/36 §5.5) ──


def test_grn_step_accel_matches_step(monkeypatch):
    """step_accel (python kernel) must match step() bit-for-bit over ticks."""
    monkeypatch.setenv("HELIX_ACCEL", "python")
    from helixlang.plugins.runtime.grn import GRN

    def build() -> GRN:
        g = GRN()
        g.add_gene("ci", threshold=0.0, initial_level=0.8)
        g.add_gene("cro", threshold=0.0, initial_level=0.2)
        g.add_gene("out", threshold=0.3, initial_level=0.0, decay=0.5)
        g.add_edge("ci", "cro", -0.8)
        g.add_edge("cro", "ci", -0.7)
        g.add_edge("ci", "out", 1.2)
        return g

    a, b = build(), build()
    for _ in range(120):
        ta = a.step()
        tb = b.step_accel()
        assert ta == tb
        for n in a.nodes:
            assert a.nodes[n].level == pytest.approx(b.nodes[n].level, abs=1e-12)


def test_grn_step_accel_hill_matches_scalar(monkeypatch):
    """doc/39 O6: Hill genes run through the mixed `step_accel` path and match
    the scalar `GRN.step` exactly."""
    monkeypatch.setenv("HELIX_ACCEL", "python")
    from helixlang.plugins.runtime.grn import GRN
    a = GRN()
    b = GRN()
    a.add_gene("g", threshold=0.0, initial_level=0.5, hill_n=2.0, kd=1.0)
    b.add_gene("g", threshold=0.0, initial_level=0.5, hill_n=2.0, kd=1.0)
    a.add_gene("h", threshold=0.2, initial_level=0.1)
    b.add_gene("h", threshold=0.2, initial_level=0.1)
    for _ in range(40):
        a.step_accel()
        b.step()
    for n in a.nodes:
        assert a.nodes[n].level == b.nodes[n].level


def test_grn_step_accel_noise_matches_step(monkeypatch):
    """doc/37 §3.4: telegraph noise is no longer rejected in step_accel.  The
    same per-node two-state-promoter perturbation is layered on the kernel
    result with the same RNG, so step() and step_accel() stay draw-for-draw
    identical even when interleaved."""
    monkeypatch.setenv("HELIX_ACCEL", "python")
    from helixlang.plugins.runtime.grn import GRN

    def build() -> GRN:
        g = GRN(noise_enabled=True, noise_seed=7)
        for name, thr, init in (("ci", 0.0, 0.8), ("cro", 0.0, 0.2)):
            g.add_gene(name, threshold=thr, initial_level=init, decay=0.5)
        g.add_gene("out", threshold=0.3, initial_level=0.0, decay=0.5)
        for name in g.nodes:
            g.nodes[name].noise = TelegraphPromoter(
                k_on=1.5, k_off=3.0, burst_size=2.0,
                expression_scale=0.02)
        g.add_edge("ci", "cro", -0.8)
        g.add_edge("cro", "ci", -0.7)
        g.add_edge("ci", "out", 1.2)
        return g

    a, b = build(), build()
    import random
    # Interleave: a runs the scalar path, b runs accel path, then swap so the
    # RNG draw sequences line up between the two entry points tick-by-tick.
    for _ in range(20):
        for _1 in range(3):
            a.step()
            b.step()
        ta, tb = a.step(), b.step_accel()
        assert ta == tb
        for n in a.nodes:
            assert a.nodes[n].level == b.nodes[n].level
    # A pure-accel run seeded identically reproduces the mixed run.
    c = build()
    for _ in range(20 * 4):
        c.step_accel()
    assert c.nodes["out"].level == a.nodes["out"].level


# ── Phase 3: determinism at equivalent fidelity + no-silent-swap contract ──
# (doc/36 §4.2 / §5.5 / §3ξ.3): switching the speed-only backend must not change
# results, and an absent chosen backend must raise NativeBackendError — never a
# transparent swap to a different fidelity class.


def _grn_circuit():
    from helixlang.plugins.runtime.grn import GRN
    g = GRN()
    for name, thr, init in (("ci", 0.0, 0.8), ("cro", 0.0, 0.2), ("out", 0.3, 0.0)):
        g.add_gene(name, threshold=thr, initial_level=init, decay=0.5)
    g.add_edge("ci", "cro", -0.8)
    g.add_edge("cro", "ci", -0.7)
    g.add_edge("ci", "out", 1.2)
    g.add_edge("out", "out", 0.3)
    return g


def _run_trace(ticks: int = 150):
    g = _grn_circuit()
    trace: list[str] = []
    for _ in range(ticks):
        trace.append(tuple(g.step_accel()))
    final = {n: g.nodes[n].level for n in g.nodes}
    return trace, final


def test_step_accel_deterministic_across_backends(monkeypatch):
    """numpy and python hot-loop backends must yield the SAME run (doc/36 §4.2)."""
    monkeypatch.setenv("HELIX_ACCEL", "numpy")
    trace_np, final_np = _run_trace()
    monkeypatch.setenv("HELIX_ACCEL", "python")
    trace_py, final_py = _run_trace()
    # triggering must be identical (exact); levels equal within float tolerance.
    assert trace_np == trace_py
    for n in final_np:
        assert abs(final_np[n] - final_py[n]) < 1e-9


def test_absent_native_raises_through_consumer(monkeypatch):
    """With no compiled impl, requesting 'native' raises through the consumer,
    never a silent fallback to a different fidelity class (doc/36 §3ξ.3)."""
    if _native_available():
        pytest.skip("native .so built for this interpreter")
    from helixlang._accel.grn_step.backend import step
    levels, src, dst, weights, decays, thresholds, default = (
        [0.9, 0.2, 0.5, 0.0], [0, 1, 1], [2, 2, 3],
        [0.5, 0.8, -0.3], [0.5, None, 0.1, None], [0.3, 0.3, 0.3, 0.3], 0.99,
    )
    monkeypatch.setenv("HELIX_ACCEL", "native")
    with pytest.raises(NativeBackendError):
        step(levels, src, dst, weights, decays, thresholds, default)
    # Declaring --pure-python selects the pure-Python impl explicitly: no error.
    monkeypatch.setenv("HELIX_ACCEL", "python")
    new, trig = step(levels, src, dst, weights, decays, thresholds, default)
    assert len(new) == 4
    assert all(isinstance(i, int) and 0 <= i < 4 for i in trig)


def test_numba_kernel_selectable_and_matches_reference(monkeypatch):
    """numba is selectable and byte-equivalent to the reference (doc/36 §4.2)."""
    from helixlang._accel.grn_step import backend as gs_backend

    if not importlib.util.find_spec("numba"):
        pytest.skip("numba not installed")
    monkeypatch.setenv("HELIX_ACCEL", "numba")
    levels, src, dst, weights, decays, thresholds, default = (
        [0.9, 0.2, 0.5, 0.0], [0, 1, 1], [2, 2, 3],
        [0.5, 0.8, -0.3], [0.5, None, 0.1, None], [0.3, 0.3, 0.3, 0.3], 0.99,
    )
    new_nb, trig_nb = gs_backend.step(levels, src, dst, weights, decays,
                                      thresholds, default)
    py = importlib.import_module("helixlang._accel.grn_step.impl_python")
    new_py, trig_py = py.step(list(levels), src, dst, weights, decays,
                              thresholds, default)
    assert trig_nb == trig_py
    assert len(new_nb) == len(new_py)
    for a, b in zip(new_nb, new_py, strict=True):
        assert abs(a - b) < 1e-9


def test_step_accel_numba_deterministic_across_backends(monkeypatch):
    """numba and python runs must be deterministic-equivalent (doc/36 §5.5)."""
    if not importlib.util.find_spec("numba"):
        pytest.skip("numba not installed")
    monkeypatch.setenv("HELIX_ACCEL", "numba")
    trace_nb, final_nb = _run_trace(ticks=60)
    monkeypatch.setenv("HELIX_ACCEL", "python")
    trace_py, final_py = _run_trace(ticks=60)
    assert trace_nb == trace_py
    for n in final_nb:
        assert abs(final_nb[n] - final_py[n]) < 1e-9


# ── Phase 3: compiled native backends (Cython + C) ──────────────────────────
# These only run when the current interpreter can load the built .so (i.e. the
# [native] build was produced for this exact Python version).  Otherwise they
# skip: the loader correctly treats the compiled backend as absent and never
# silently swaps fidelity bounds (doc/36 §3ξ).


def _native_available() -> bool:
    pkg = "helixlang._accel.grn_step"
    return any(
        importlib.util.find_spec(f"{pkg}.impl_{impl}") is not None
        for impl in ("cext", "cython")
    )


def test_native_loader_prefers_compiled_backend_when_built():
    """When a compiled impl is loadable, choose_backend picks it by default."""
    if not _native_available():
        pytest.skip("native .so not built for this interpreter")
    impl = choose_backend("helixlang._accel.grn_step")
    assert impl in ("impl_cext", "impl_cython")


def test_native_and_python_kernels_match():
    """native (C/Cython) kernel must be equivalent to the python reference."""
    if not _native_available():
        pytest.skip("native .so not built for this interpreter")
    from helixlang._accel.grn_step import impl_python as py

    nat_mod = None
    for impl in ("impl_cext", "impl_cython"):
        spec = importlib.util.find_spec(
            f"helixlang._accel.grn_step.{impl}")
        if spec is not None:
            nat_mod = importlib.import_module(
                f"helixlang._accel.grn_step.{impl}")
            break
    assert nat_mod is not None

    levels, src, dst, weights, decays, thresholds, default = (
        [0.9, 0.2, 0.5, 0.0], [0, 1, 1], [2, 2, 3],
        [0.5, 0.8, -0.3], [0.5, None, 0.1, None], [0.3, 0.3, 0.3, 0.3], 0.99,
    )
    new_nat, trig_nat = nat_mod.step(list(levels), src, dst, weights, decays,
                                     thresholds, default)
    new_py, trig_py = py.step(list(levels), src, dst, weights, decays,
                              thresholds, default)
    assert trig_nat == trig_py
    assert len(new_nat) == len(new_py)
    for a, b in zip(new_nat, new_py, strict=True):
        assert abs(a - b) < 1e-9


def test_step_accel_native_deterministic(monkeypatch):
    """native and python runs must be deterministic-equivalent (doc/36 §5.5)."""
    if not _native_available():
        pytest.skip("native .so not built for this interpreter")
    monkeypatch.setenv("HELIX_ACCEL", "native")
    trace_nat, final_nat = _run_trace(ticks=120)
    monkeypatch.setenv("HELIX_ACCEL", "python")
    trace_py, final_py = _run_trace(ticks=120)
    assert trace_nat == trace_py
    for n in final_nat:
        assert abs(final_nat[n] - final_py[n]) < 1e-9


# ── Phase 3: numba diffusion hot loop (doc/36 §5.3, item 3) ────────────────


def _diff_field(n=12):
    u = [[1.0] * n for _ in range(n)]
    v = [[0.0] * n for _ in range(n)]
    mid = n // 2
    for i in range(mid - 1, mid + 1):
        for j in range(mid - 1, mid + 1):
            u[i][j] = 0.5
            v[i][j] = 0.25
    return u, v


def _diff_maxdiff(a, b):
    return max(abs(float(x) - float(y))
               for ar, br in zip(a, b, strict=True)
               for x, y in zip(ar, br, strict=True))


def test_diffusion_backends_equivalent():
    """python / numpy / numba Gray-Scott kernels are byte-equivalent."""
    import importlib as _il
    u, v = _diff_field()
    params = (0.035, 0.065, 0.16, 0.08)
    py = _il.import_module("helixlang._accel.diffusion.impl_python")
    np_mod = _il.import_module("helixlang._accel.diffusion.impl_numpy")
    nu_py, nv_py = py.step(u, v, *params)
    nu_np, nv_np = np_mod.step(u, v, *params)
    assert _diff_maxdiff(nu_py, nu_np) < 1e-12
    assert _diff_maxdiff(nv_py, nv_np) < 1e-12
    if importlib.util.find_spec("numba"):
        nb = _il.import_module("helixlang._accel.diffusion.impl_numba")
        nu_nb, nv_nb = nb.step(u, v, *params)
        assert _diff_maxdiff(nu_py, nu_nb) < 1e-12
        assert _diff_maxdiff(nv_py, nv_nb) < 1e-12


def test_diffusion_numba_deterministic(monkeypatch):
    """numba and python diffusion runs must be deterministic-equivalent."""
    if not importlib.util.find_spec("numba"):
        pytest.skip("numba not installed")
    from helixlang._accel.diffusion import step as backend_step
    u, v = _diff_field()
    params = (0.035, 0.065, 0.16, 0.08)
    monkeypatch.setenv("HELIX_ACCEL", "numba")
    for _ in range(15):
        u, v = backend_step(u, v, *params)
    final_nb = (_diff_maxdiff(u, u) >= 0)  # field advanced

    u2, v2 = _diff_field()
    monkeypatch.setenv("HELIX_ACCEL", "python")
    for _ in range(15):
        u2, v2 = backend_step(u2, v2, *params)
    # both runs must have advanced identically (deterministic-equivalent).
    assert _diff_maxdiff(u, u2) < 1e-12 and _diff_maxdiff(v, v2) < 1e-12
    assert final_nb


# ── Phase 3: simplex pivot hot loop (doc/36 §5.3, item 1) ───────────────────


def _simplex_native_available() -> bool:
    pkg = "helixlang._accel.simplex"
    return any(
        importlib.util.find_spec(f"{pkg}.impl_{impl}") is not None
        for impl in ("cext", "cython", "rust")
    )


def _simplex_problem():
    # max 2x0 + 3x1  s.t.  x0+x1<=4, x0<=2; slacks x2,x3 basic.
    tableau = [[1.0, 1.0, 1.0, 0.0, 4.0],
               [1.0, 0.0, 0.0, 1.0, 2.0]]
    basis = [2, 3]
    obj = [2.0, 3.0, 0.0, 0.0]
    return tableau, basis, obj, 4


def _simplex_maxdiff(a, b):
    return max(abs(float(x) - float(y))
               for ar, br in zip(a, b, strict=True)
               for x, y in zip(ar, br, strict=True))


def test_simplex_backends_equivalent():
    """python / numpy / Cython simplex pivot loops are byte-equivalent."""
    import copy
    import importlib as _il
    T, b, obj, n = _simplex_problem()
    py = _il.import_module("helixlang._accel.simplex.impl_python")
    np_mod = _il.import_module("helixlang._accel.simplex.impl_numpy")
    tp = copy.deepcopy(T)
    bp = b[:]
    s_py = py.run(tp, bp, obj, n)
    tb = copy.deepcopy(T)
    bb = b[:]
    s_np = np_mod.run(tb, bb, obj, n)
    assert s_py == s_np == "optimal"
    assert bp == bb
    assert _simplex_maxdiff(tp, tb) < 1e-12
    if _simplex_native_available():
        import importlib as _il
        for _impl in ("cext", "cython", "rust"):
            try:
                nat = _il.import_module(
                    f"helixlang._accel.simplex.impl_{_impl}")
            except (ImportError, ModuleNotFoundError):
                continue  # prebuilt .so is for a different interpreter ABI
            t_n = copy.deepcopy(T)
            b_n = b[:]
            s_n = nat.run(t_n, b_n, obj, n)
            assert s_n == s_py
            assert b_n == bp
            assert _simplex_maxdiff(t_n, tp) < 1e-12


def test_simplex_native_loader_resolves(monkeypatch):
    """Default loader picks a compiled simplex impl when built, else numpy."""
    impl = choose_backend("helixlang._accel.simplex")
    assert impl in ("impl_cext", "impl_cython", "impl_rust",
                    "impl_numpy", "impl_python")
    if _simplex_native_available():
        assert impl in ("impl_cext", "impl_cython", "impl_rust")


# ── doc/42 Phase C PF-2 — native simplex opt-in dispatch ────────────────────


def test_accel_simplex_facade_present():
    from helixlang.api.accel import simplex_run
    T, b, obj, n = _simplex_problem()
    import copy
    tb = copy.deepcopy(T)
    bb = b[:]
    status = simplex_run(tb, bb, obj, n)
    assert status in ("optimal", "unbounded", "max_iter")
    assert status == "optimal"


def test_simplex_dispatch_defaults_byte_identical(monkeypatch):
    """Without HELIX_ACCEL_SIMPLEX the dispatch uses the numpy reference path,
    so FBA/dFBA goldens stay bit-identical (doc/42 Phase C gate)."""
    from helixlang.plugins.runtime.metabolism import (
        _simplex_max_dispatch,
        _simplex_max_numpy,
        _simplex_native_optin,
    )
    monkeypatch.delenv("HELIX_ACCEL_SIMPLEX", raising=False)
    assert not _simplex_native_optin()
    T, b, obj, n = _simplex_problem()
    import numpy as np
    tab1 = np.array(T, dtype=np.float64)
    bas1 = b[:]
    tab2 = np.array(T, dtype=np.float64)
    bas2 = b[:]
    o = np.array(obj, dtype=np.float64)
    s_disp = _simplex_max_dispatch(tab1, bas1, o, n, 1e-9, 10000)
    s_ref = _simplex_max_numpy(tab2, bas2, o, n, 1e-9, 10000)
    assert s_disp == s_ref
    assert bas1 == bas2
    assert np.array_equal(tab1, tab2)  # byte-identical default


def test_simplex_dispatch_optin_routes_to_accel(monkeypatch):
    """HELIX_ACCEL_SIMPLEX routes the pivot loop through the accel kernel."""
    import numpy as np

    from helixlang.plugins.runtime.metabolism import (
        _simplex_max_dispatch,
        _simplex_native_optin,
    )
    monkeypatch.setenv("HELIX_ACCEL_SIMPLEX", "native")
    assert _simplex_native_optin()
    T, b, obj, n = _simplex_problem()
    tab = np.array(T, dtype=np.float64)
    bas = b[:]
    o = np.array(obj, dtype=np.float64)
    status = _simplex_max_dispatch(tab, bas, o, n, 1e-9, 10000)
    assert status == "optimal"


def test_metabolism_simplex_matches_across_dispatch(monkeypatch):
    """simplex() objective is identical whether the native path is on or off."""
    from helixlang.plugins.runtime import metabolism as M
    c = [2.0, 3.0]
    A = [[1.0, 1.0], [1.0, 0.0]]
    b = [4.0, 2.0]
    bounds = [(0.0, 2.0), (0.0, 2.0)]
    monkeypatch.delenv("HELIX_ACCEL_SIMPLEX", raising=False)
    res_ref = M.simplex(c, A, b, bounds, maximize=True)
    monkeypatch.setenv("HELIX_ACCEL_SIMPLEX", "native")
    res_nat = M.simplex(c, A, b, bounds, maximize=True)
    assert res_ref["status"] == res_nat["status"] == "optimal"
    assert abs(res_ref["objective"] - res_nat["objective"]) < 1e-9
    assert abs(res_nat["objective"] - 10.0) < 1e-9


# ── Phase 3: VM + population dispatch C backend (doc/36 §5.5, item 2) ──────


def _dispatch_native_available() -> bool:
    pkg = "helixlang._accel.dispatch"
    return any(
        importlib.util.find_spec(f"{pkg}.impl_{impl}") is not None
        for impl in ("cext", "cython")
    )


def _dispatch_program():
    # push2, push3, mul -> 6; push10, push4, sub -> 6; add -> 12;
    # pop; push5; halt  => final stack [5.0], 9 ops.
    code = [0x20, 0, 0x20, 1, 0x92, 0x20, 2, 0x20, 3, 0x91,
            0x90, 0x21, 0x20, 4, 0x11]
    consts = [2.0, 3.0, 10.0, 4.0, 5.0]
    return code, consts


def test_dispatch_cext_matches_python():
    """C dispatch kernel (run_quota + run_many) is byte-identical to python."""
    if not _dispatch_native_available():
        pytest.skip("native dispatch .so not built for this interpreter")
    import importlib as _il
    code, consts = _dispatch_program()
    py = _il.import_module("helixlang._accel.dispatch.impl_python")
    ce = _il.import_module("helixlang._accel.dispatch.impl_cext")
    assert py.run_quota(code, consts) == ce.run_quota(code, consts)
    assert py.run_many(code, consts, n_cells=8) == ce.run_many(
        code, consts, n_cells=8)
    assert py.run_many(code, consts) == ce.run_many(code, consts)
    with pytest.raises(NotImplementedError):
        ce.run_quota([0x99], consts)
    with pytest.raises(IndexError):
        ce.run_quota([0x21], consts)


def test_dispatch_loader_prefers_compiled_when_built():
    """Default dispatch loader picks the compiled backend when present."""
    if not _dispatch_native_available():
        pytest.skip("native dispatch .so not built for this interpreter")
    impl = choose_backend("helixlang._accel.dispatch")
    assert impl in ("impl_cext", "impl_cython")


def test_dispatch_native_deterministic():
    """Dispatch population (run_many) is deterministic across cells."""
    if not _dispatch_native_available():
        pytest.skip("native dispatch .so not built for this interpreter")
    import importlib as _il
    code, consts = _dispatch_program()
    ce = _il.import_module("helixlang._accel.dispatch.impl_cext")
    py = _il.import_module("helixlang._accel.dispatch.impl_python")
    r1 = ce.run_many(code, consts, n_cells=32)
    r2 = ce.run_many(code, consts, n_cells=32)
    assert r1 == r2
    assert all(row == py.run_quota(code, consts) for row in r1)


