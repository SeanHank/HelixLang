"""Tests for helixlang.plugins.runtime.sparse_grn.

Covers the SparseGRN matrix construction helpers, the exact and budgeted
step paths, vectorized activation (sigmoid + Hill), the pure-numpy CSR
fallbacks (exercised by disabling scipy), validation errors, and the
to_grn / scalar_inputs oracles.
"""
from __future__ import annotations

import numpy as np
import pytest

from helixlang.plugins.runtime import sparse_grn as sg
from helixlang.plugins.runtime.grn import GRN, TelegraphPromoter
from helixlang.plugins.runtime.sparse_grn import (
    SparseGRN,
    _vectorized_activation,
    sparse_from_edges,
)


def _grn(noise_enabled: bool = False) -> GRN:
    grn = GRN(noise_enabled=noise_enabled)
    grn.add_gene("a", 0.5, decay=0.9)
    grn.add_gene("b", 0.4, decay=0.8)
    grn.add_gene("c", 0.3, decay=0.7,
                 hill_n=2.0, kd=0.6, noise=TelegraphPromoter(
                     k_on=5.0, k_off=5.0, burst_size=2.0))
    grn.add_edge("a", "b", 1.0)
    grn.add_edge("b", "c", 0.5)
    grn.add_edge("c", "a", -0.2)
    return grn


# ── vectorized activation ───────────────────────────────────────────────

def test_vectorized_activation_sigmoid():
    levels = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    thr = np.array([0.5, 0.5, 0.5])
    out = _vectorized_activation(levels, thr, np.zeros(3), np.ones(3),
                                 has_hill=False)
    assert out.shape == (2, 3)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_vectorized_activation_hill():
    levels = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    thr = np.array([0.5, 0.5, 0.5])
    hill_n = np.array([2.0, 0.0, 2.0])
    kd = np.array([0.6, 0.6, 0.0])
    out = _vectorized_activation(levels, thr, hill_n, kd, has_hill=True)
    assert out.shape == (2, 3)
    # the kd==0 column uses the (x > 0) fallback branch
    assert np.all(np.isfinite(out))


# ── validation errors in __post_init__ ────────────────────────────────────

def test_post_init_data_length_mismatch():
    g = 2
    with pytest.raises(ValueError):
        SparseGRN(
            names=["a", "b"],
            data=np.array([1.0, 2.0]),
            col_indices=np.array([0]),
            row_ptr=np.zeros(g + 1, dtype=np.int64),
            thresholds=np.zeros(g),
            decays=np.zeros(g),
            hill_n=np.zeros(g),
            kd=np.zeros(g),
            noise_fano=np.zeros(g),
            noise_expression_scale=np.ones(g),
        )


def test_post_init_rowptr_length_mismatch():
    with pytest.raises(ValueError):
        SparseGRN(
            names=["a", "b"],
            data=np.array([1.0, 2.0]),
            col_indices=np.array([0, 1]),
            row_ptr=np.zeros(2, dtype=np.int64),
            thresholds=np.zeros(2),
            decays=np.zeros(2),
            hill_n=np.zeros(2),
            kd=np.zeros(2),
            noise_fano=np.zeros(2),
            noise_expression_scale=np.ones(2),
        )


# ── construction helpers ─────────────────────────────────────────────────

def test_from_grn_roundtrip():
    grn = _grn()
    sg_obj = SparseGRN.from_grn(grn, noise_seed=42)
    assert sg_obj.n_genes == 3
    assert sg_obj.n_edges == 3
    assert sg_obj._noise_mask.any()
    assert sg_obj._noise_gen is not None


def test_properties():
    sg_obj = SparseGRN.from_grn(_grn())
    assert sg_obj.n_genes == 3
    assert sg_obj.n_edges == 3


# ── state ────────────────────────────────────────────────────────────────

def test_new_state_with_initial_genes():
    sg_obj = SparseGRN.from_grn(_grn())
    levels = sg_obj.new_state(4, initial_genes=("b",))
    assert levels.shape == (4, 3)
    assert np.all(levels[:, sg_obj._idx["b"]] == 1.0)
    assert np.all(levels[:, sg_obj._idx["a"]] == 0.0)


def test_new_state_unknown_initial_gene():
    sg_obj = SparseGRN.from_grn(_grn())
    levels = sg_obj.new_state(2, initial_genes=("missing",))
    assert np.all(levels == 0.0)


def test_triggered_and_n_active():
    sg_obj = SparseGRN.from_grn(_grn())
    levels = np.array([[0.0, 0.6, 0.1], [0.0, 0.3, 0.9]])
    trig = sg_obj.triggered(levels, threshold=0.5)
    assert trig.shape == (2, 3)
    assert trig[0, 1] and not trig[0, 0]
    counts = sg_obj.n_active(levels, threshold=0.5)
    assert counts.tolist() == [1, 1]


# ── exact step ────────────────────────────────────────────────────────────

def test_step_exact_matches_scalar():
    sg_obj = SparseGRN.from_grn(_grn(), noise_seed=7)
    levels = sg_obj.new_state(3, initial_genes=("a", "b"))
    out = sg_obj.step(levels)
    assert out.shape == (3, 3)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_step_hill_path():
    sg_obj = SparseGRN.from_grn(_grn())
    levels = sg_obj.new_state(1, initial_genes=("a", "b", "c"))
    out = sg_obj.step(levels)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_scalar_inputs_matrix():
    sg_obj = SparseGRN.from_grn(_grn())
    levels = sg_obj.new_state(2, initial_genes=("a", "b"))
    out = sg_obj.scalar_inputs(levels)
    assert out.shape == (2, 3)


# ── budgeted step ────────────────────────────────────────────────────────

def test_step_budgeted_no_budget():
    sg_obj = SparseGRN.from_grn(_grn())
    levels = sg_obj.new_state(2, initial_genes=("a",))
    out = sg_obj.step_budgeted(levels, budget=None)
    assert out.shape == (2, 3)


def test_step_budgeted_small_budget():
    sg_obj = SparseGRN.from_grn(_grn())
    levels = sg_obj.new_state(2, initial_genes=("a", "b", "c"))
    out = sg_obj.step_budgeted(levels, budget=1)
    assert out.shape == (2, 3)


def test_step_budgeted_silent_sources():
    sg_obj = SparseGRN.from_grn(_grn())
    levels = np.zeros((2, 3))
    out = sg_obj.step_budgeted(levels)
    assert out.shape == (2, 3)


# ── to_grn ───────────────────────────────────────────────────────────────

def test_to_grn_rebuilds_template():
    sg_obj = SparseGRN.from_grn(_grn())
    rebuilt = sg_obj.to_grn()
    assert sorted(rebuilt.nodes.keys()) == ["a", "b", "c"]
    pairs = {(e.source, e.target) for e in rebuilt.edges}
    assert pairs == {("a", "b"), ("b", "c"), ("c", "a")}


# ── sparse_from_edges ────────────────────────────────────────────────────

def test_sparse_from_edges_basic():
    s = sparse_from_edges(
        names=["x", "y"],
        edges=[("x", "y", 1.5)],
        thresholds=[0.5, 0.2],
        decays=[0.9, 0.8],
        noise_seed=3,
    )
    assert s.n_genes == 2
    assert s.n_edges == 1


def test_sparse_from_edges_defaults():
    s = sparse_from_edges(names=["x", "y"], edges=[("x", "y", 1.5)])
    assert s.n_genes == 2
    assert s.thresholds.tolist() == [0.5, 0.5]


def test_sparse_from_edges_unknown_gene():
    with pytest.raises(ValueError):
        sparse_from_edges(names=["x"], edges=[("x", "missing", 1.0)])


# ── pure-numpy fallbacks (scipy disabled) ────────────────────────────────

def test_pure_numpy_inputs(monkeypatch):
    monkeypatch.setattr(sg, "_HAS_SCIPY", False)
    sg_obj = SparseGRN.from_grn(_grn())
    assert sg_obj._csr is None
    levels = sg_obj.new_state(2, initial_genes=("a", "b"))
    out = sg_obj.step(levels)
    assert out.shape == (2, 3)


def test_pure_numpy_step_budgeted(monkeypatch):
    monkeypatch.setattr(sg, "_HAS_SCIPY", False)
    sg_obj = SparseGRN.from_grn(_grn())
    levels = sg_obj.new_state(2, initial_genes=("a", "b", "c"))
    out = sg_obj.step_budgeted(levels, budget=2)
    assert out.shape == (2, 3)


# ── scipy path with no surviving rows (line 321) + budget skip ─────────────

def test_step_budgeted_no_surviving_rows():
    s = sparse_from_edges(names=["a", "b"], edges=[("a", "b", 1.0)])
    levels = s.new_state(2, initial_genes=("b",))
    out = s.step_budgeted(levels, budget=1)
    assert out.shape == (2, 2)


# ── numpy-required guards ─────────────────────────────────────────────────

@pytest.mark.parametrize("fn", [
    "from_grn",
    "new_state",
    "_inputs",
    "step",
    "step_budgeted",
    "scalar_inputs",
    "sparse_from_edges",
])
def test_numpy_required_guards(monkeypatch, fn):
    s = SparseGRN.from_grn(_grn())
    monkeypatch.setattr(sg, "_HAS_NUMPY", False)
    if fn == "from_grn":
        with pytest.raises(ImportError):
            SparseGRN.from_grn(_grn())
    elif fn == "new_state":
        with pytest.raises(ImportError):
            s.new_state(2)
    elif fn == "_inputs":
        with pytest.raises(ImportError):
            s._inputs(np.zeros((1, s.n_genes)))
    elif fn == "step":
        with pytest.raises(ImportError):
            s.step(np.zeros((1, s.n_genes)))
    elif fn == "step_budgeted":
        with pytest.raises(ImportError):
            s.step_budgeted(np.zeros((1, s.n_genes)))
    elif fn == "scalar_inputs":
        with pytest.raises(ImportError):
            s.scalar_inputs(np.zeros((1, s.n_genes)))
    elif fn == "sparse_from_edges":
        with pytest.raises(ImportError):
            sparse_from_edges(names=["a"], edges=[])
