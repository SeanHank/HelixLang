"""Tests for the cell-fate decision analysis (S7)."""
from __future__ import annotations

import pytest

from helixlang.apps.fate_analysis import (
    bistability_scan,
    critical_slowing_down,
    fate,
    make_toggle_grn,
    run_fate_analysis,
    switching_rate,
)
from helixlang.grn import integrate_grn


def test_bistability_scan_monostable_below_saddle_node() -> None:
    scan = bistability_scan((1.0, 3.0))
    for point in scan:
        assert point.n_stable_states == 1
        assert not point.is_bistable
        state = point.stable_states[0]
        # monostable fixed point is symmetric: a == b
        assert state.a == pytest.approx(state.b, abs=1e-6)


def test_bistability_scan_bistable_above_saddle_node() -> None:
    scan = bistability_scan((7.0,))
    (point,) = scan
    assert point.is_bistable
    assert point.n_stable_states == 2
    a_dominant = max(point.stable_states, key=lambda s: s.a)
    b_dominant = min(point.stable_states, key=lambda s: s.a)
    assert a_dominant.a > 0.35
    assert b_dominant.b > 0.35
    # the two fates are anti-correlated: A-high means B-low and vice versa
    assert a_dominant.b < 0.1
    assert b_dominant.a < 0.1
    # a non-trivial unstable boundary state separates them
    assert point.unstable_states


def test_bistability_scan_transition_between_w5_and_w6() -> None:
    scan = bistability_scan((5.0, 6.0))
    assert scan[0].n_stable_states == 1
    assert scan[1].n_stable_states == 2


def test_scan_stable_states_match_ode_steady_state() -> None:
    # root-finding scan must agree with ODE integration of the same toggle
    point = bistability_scan((7.0,))[0]
    a_dominant = max(point.stable_states, key=lambda s: s.a)
    grn = make_toggle_grn(7.0, a0=0.9, b0=0.1)
    result = integrate_grn(grn, (0.0, 3000.0), n_points=600)
    final = result.final()
    assert final["a"] == pytest.approx(a_dominant.a, abs=0.02)
    assert final["b"] == pytest.approx(a_dominant.b, abs=0.02)


def test_switching_rate_low_without_resource_competition() -> None:
    # a deep bistable toggle without resource coupling is locked in its fate
    assert switching_rate(7.0, resource_strength=0.0) < 0.05


def test_resource_competition_amplifies_switching() -> None:
    base = switching_rate(7.0, resource_strength=0.0)
    weak = switching_rate(7.0, resource_strength=0.5)
    strong = switching_rate(7.0, resource_strength=1.0)
    # shared-resource throttling collapses the barrier -> fate flips
    assert weak > 3.0 * base
    assert weak > 0.25
    assert strong >= weak


def test_critical_slowing_down_rises_toward_bifurcation() -> None:
    far = critical_slowing_down(3.0)
    near = critical_slowing_down(5.3)
    mid = critical_slowing_down(5.0)
    # lag-1 autocorrelation grows as w approaches the saddle-node (~5.5)
    assert near > far + 0.15
    assert mid > far
    assert 0.0 < far < 1.0


def test_fate_classifier() -> None:
    assert fate(0.40, 0.05) == "a"
    assert fate(0.05, 0.40) == "b"
    assert fate(0.20, 0.20) == "a"


def test_make_toggle_grn_mutual_repression_edges() -> None:
    grn = make_toggle_grn(8.0, a0=0.9, b0=0.1)
    assert set(grn.nodes) == {"a", "b"}
    weights = {edge.source: edge.weight for edge in grn.edges}
    assert weights["a"] == -8.0
    assert weights["b"] == -8.0


def test_run_fate_analysis_summary_structure() -> None:
    summary = run_fate_analysis(w_values=(3.0, 7.0), seed=0)
    assert len(summary["bifurcation"]) == 2
    assert summary["bifurcation"][0]["n_stable_states"] == 1
    assert summary["bifurcation"][1]["n_stable_states"] == 2
    assert set(summary["switching_rates"]) == {0.0, 0.5, 1.0}
    assert summary["switching_rates"][0.0] < summary["switching_rates"][1.0]
    assert len(summary["critical_slowing_down"]) == 4
