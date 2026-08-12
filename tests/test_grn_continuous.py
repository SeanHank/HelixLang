"""Continuous-time GRN ODE solvers (T2.2, gap G6).

Validates that the continuous ODE form of the GRN (Alon 2007;
GRN_modeler 2025) shares the fixed points of the discrete recurrence,
reproduces the classic bistable toggle switch (Gardner 2000) and the
repressilator oscillation (Elowitz 2000), and that the pure-Python
RK45 agrees with the analytic solution and with scipy's RK45 when
scipy is installed.
"""
import math

import pytest

from helixlang.grn import (
    GRN,
    ContinuousGRNResult,
    decay_from_half_life_ticks,
    grn_derivatives,
    integrate_grn,
    integrate_ode,
    rate_constant_from_decay,
    sigmoid,
)


def test_rate_constant_from_decay_matches_first_order() -> None:
    decay = 0.994
    k = rate_constant_from_decay(decay)
    assert k == pytest.approx(-math.log(decay))
    assert k == pytest.approx(1.0 - decay, rel=0.01)


def test_grn_ode_fixed_point_matches_discrete_recurrence() -> None:
    g = GRN()
    g.add_gene("a", threshold=0.0)
    g.add_edge("a", "a", 4.0)  # self-activation
    names, rhs, y0 = grn_derivatives(g, [0.5])
    assert names == ["a"]
    # ODE fixed point: dL/dt = 0  =>  L* = sigmoid(4*L - 0)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid - sigmoid(4.0 * mid) > 0:
            hi = mid
        else:
            lo = mid
    expected = (lo + hi) / 2
    # Fixed-point search: dL/dt = 0 means L = sigmoid(4L)
    fixed = expected
    # Compare against discrete recurrence steady state (exact same fixed point)
    g2 = GRN()
    g2.add_gene("a", threshold=0.0, initial_level=0.5)
    g2.add_edge("a", "a", 4.0)
    for _ in range(2000):
        g2.step()
    assert g2.nodes["a"].level == pytest.approx(fixed, abs=1e-3)
    # ODE at the fixed point has ~zero derivative
    deriv = rhs(0.0, [fixed])
    assert deriv[0] == pytest.approx(0.0, abs=1e-6)


def test_grn_ode_constant_input_matches_analytic() -> None:
    # A single gene with a fixed external input: dL/dt = k*(A - L),
    # A = sigmoid(w*const - threshold).  Analytic: L(t) = A - (A - L0)e^{-kt}.
    g = GRN()
    g.add_gene("x", threshold=0.5, initial_level=0.0)
    # drive x with a constant activator gene 'u' pinned at level 1.0
    g.add_gene("u", threshold=-10.0, initial_level=1.0)
    g.add_edge("u", "x", 2.0)
    k = rate_constant_from_decay(decay_from_half_life_ticks(110.0))
    a = sigmoid(2.0 * 1.0 - 0.5)
    res = integrate_grn(g, (0.0, 100.0), n_points=200, method="rk45")
    x = res.trajectory("x")
    for i, t in enumerate(res.times):
        analytic = a - (a - 0.0) * math.exp(-k * t)
        assert x[i] == pytest.approx(analytic, abs=1e-5)


def test_rk4_matches_analytic_linear_ode() -> None:
    k = 0.006
    rhs = lambda t, y: [k * (0.8 - y[0])]  # noqa: E731
    times, ys = integrate_ode(rhs, [0.0], (0.0, 500.0), n_points=50,
                              method="rk4")
    for t, y in zip(times, ys, strict=True):
        analytic = 0.8 - 0.8 * math.exp(-k * t)
        assert y[0] == pytest.approx(analytic, abs=1e-3)


def test_rk45_adaptive_matches_analytic() -> None:
    k = 0.006
    rhs = lambda t, y: [k * (0.8 - y[0])]  # noqa: E731
    times, ys = integrate_ode(rhs, [0.0], (0.0, 500.0), n_points=100,
                              method="rk45", atol=1e-9, rtol=1e-9)
    for t, y in zip(times, ys, strict=True):
        analytic = 0.8 - 0.8 * math.exp(-k * t)
        assert y[0] == pytest.approx(analytic, abs=1e-6)


def test_toggle_switch_bistability_two_stable_states() -> None:
    # Gardner 2000 mutual-repression toggle switch: two initial conditions
    # settle on two distinct stable fixed points.  With sigmoid kinetics
    # the loop gain must exceed 2 for bistability (|w·S'| > 2).
    def make_toggle(a0: float, b0: float) -> GRN:
        g = GRN()
        g.add_gene("a", threshold=0.0, initial_level=a0)
        g.add_gene("b", threshold=0.0, initial_level=b0)
        w = 10.0
        g.add_edge("a", "b", -w)
        g.add_edge("b", "a", -w)
        return g

    low = integrate_grn(make_toggle(0.9, 0.1), (0.0, 500.0), n_points=300)
    high = integrate_grn(make_toggle(0.1, 0.9), (0.0, 500.0), n_points=300)
    f1 = low.final()
    f2 = high.final()
    # bistable: (a high, b low) vs (a low, b high), and both different
    assert f1["a"] > 0.4 and f1["b"] < 0.1
    assert f2["b"] > 0.4 and f2["a"] < 0.1
    assert f1["a"] > f2["a"] + 0.4


def test_repressilator_oscillates() -> None:
    # Elowitz 2000 repressilator: a 3-gene negative-feedback ring
    # oscillates for loop gain > 2 (requires |w·S'| > 2 in the smooth
    # sigmoid model).  Symmetry-breaking initial conditions are needed.
    w = -30.0
    g = GRN()
    for name, lvl in zip(("lacI", "tetR", "cI"), (0.6, 0.4, 0.5), strict=True):
        g.add_gene(name, threshold=0.0, initial_level=lvl)
    for a, b in (("lacI", "tetR"), ("tetR", "cI"), ("cI", "lacI")):
        g.add_edge(a, b, w)
    res = integrate_grn(g, (0.0, 6000.0), n_points=6000, method="rk45",
                        atol=1e-9, rtol=1e-9)
    x = res.trajectory("lacI")
    # count local maxima in the second half = sustained oscillation
    n = len(x)
    k0 = n // 2
    peaks = 0
    for i in range(k0 + 1, n - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1]:
            peaks += 1
    assert peaks >= 4
    # and the signal is genuinely oscillating, not constant
    assert max(x[k0:]) - min(x[k0:]) > 0.05


def test_scipy_path_matches_pure_python_when_available() -> None:
    try:
        import scipy  # noqa: F401
    except ImportError:
        pytest.skip("scipy not installed")
    g = GRN()
    g.add_gene("a", threshold=0.5, initial_level=0.1)
    g.add_gene("b", threshold=0.5, initial_level=0.9)
    w = -3.0
    g.add_edge("a", "b", w)
    g.add_edge("b", "a", w)
    pure = integrate_grn(g, (0.0, 300.0), n_points=200, method="rk45",
                         atol=1e-9, rtol=1e-9)
    sc = integrate_grn(g, (0.0, 300.0), n_points=200, method="scipy",
                       atol=1e-9, rtol=1e-9)
    for name in ("a", "b"):
        pa = pure.trajectory(name)
        sa = sc.trajectory(name)
        for p, s in zip(pa, sa, strict=True):
            assert p == pytest.approx(s, abs=1e-6)


def test_continuous_result_api() -> None:
    g = GRN()
    g.add_gene("on", threshold=0.0, initial_level=0.0)
    g.add_gene("on", threshold=0.0)  # dup overwrite keeps one node
    g = GRN()
    g.add_gene("x", threshold=0.0, initial_level=0.0)
    g.add_gene("y", threshold=0.0, initial_level=1.0)
    res = integrate_grn(g, (0.0, 10.0), n_points=11)
    assert isinstance(res, ContinuousGRNResult)
    assert len(res.times) == 11
    assert len(res.levels) == 11
    assert res.final()["y"] > 0.9
    assert "y" in res.triggered()
    assert res.at(5.0)["y"] >= 0.0
    assert len(res.trajectory("x")) == 11


def test_integrate_ode_validation() -> None:
    with pytest.raises(ValueError):
        integrate_ode(lambda t, y: [0.0], [0.0], (5.0, 5.0))
    with pytest.raises(ValueError):
        integrate_ode(lambda t, y: [0.0], [0.0], (0.0, 1.0), n_points=1)
    with pytest.raises(ValueError):
        integrate_ode(lambda t, y: [0.0], [0.0], (0.0, 1.0), method="nope")
    with pytest.raises(ValueError):
        rate_constant_from_decay(1.0)
