"""Tests for expression inference from GRN (doc/20 §14, plugins/omics/expression_inference.py).

Covers the Hill-activation function, expression-model building, steady-state
expression inference, and the time-point state snapshot.
"""
from __future__ import annotations

from types import SimpleNamespace

from helixlang.plugins.omics.expression_inference import (
    ExpressionModel,
    build_expression_model,
    hill_function,
    infer_expression,
    infer_expression_at_time,
)


def _edge(tf: str, target: str, rtype: str):
    return SimpleNamespace(tf_id=tf, target_gene=target, regulation_type=rtype)


def _cand(gene: str):
    return SimpleNamespace(gene_id=gene)


def _grn(tfs=(), edges=()):
    return SimpleNamespace(tf_candidates=[_cand(t) for t in tfs],
                           regulatory_edges=list(edges))


# ── hill_function ──────────────────────────────────────────────────────────────

def test_hill_activation_positive():
    # At tf==kd (ratio 1, n=1) the activation factor is 0.5.
    assert abs(hill_function(0.5, 0.5, 1.0, "activation") - 0.5) < 1e-9


def test_hill_repression_is_inverse():
    act = hill_function(1.0, 0.5, 2.0, "activation")
    rep = hill_function(1.0, 0.5, 2.0, "repression")
    assert abs((act + rep) - 1.0) < 1e-9
    assert rep < 0.21


def test_hill_zero_tf_gives_zero_activation():
    assert hill_function(0.0, 0.5, 2.0, "activation") == 0.0


def test_hill_kd_nonpositive_returns_one():
    # kd<=0 is a degenerate knob -> no modulation (factor 1).
    assert hill_function(0.5, 0.0, 2.0, "activation") == 1.0
    assert hill_function(0.5, -1.0, 2.0, "repression") == 1.0


def test_hill_clamps_to_unit_interval():
    v = hill_function(10.0, 0.1, 4.0, "activation")
    assert 0.0 <= v <= 1.0
    assert v > 0.99


def test_hill_default_effect_is_activation():
    assert abs(hill_function(0.5, 0.5, 1.0) - 0.5) < 1e-9


# ── build_expression_model ─────────────────────────────────────────────────────

def test_build_model_creates_defaults_for_annotated_genes():
    annot = SimpleNamespace
    annotations = {
        "g1": annot(), "g2": annot(),
    }
    model = build_expression_model(_grn(), annotations)
    assert set(model.promoter_strength) == {"g1", "g2"}
    assert set(model.rbs_strength) == {"g1", "g2"}
    assert set(model.mrna_half_life) == {"g1", "g2"}
    assert set(model.protein_half_life) == {"g1", "g2"}
    assert model.promoter_strength["g1"] == 1.0
    assert model.mrna_half_life["g2"] == 3.0
    assert model.protein_half_life["g1"] == 60.0


def test_build_model_populates_tf_effects_from_edges():
    grn = _grn(edges=[_edge("TF1", "gA", "activation"),
                      _edge("TF2", "gA", "repression")])
    model = build_expression_model(grn)
    effects = model.tf_effects["gA"]
    assert len(effects) == 2
    # each entry (tf_id, kd, n, effect)
    assert effects[0][0] == "TF1" and effects[0][3] == "activation"
    assert effects[1][0] == "TF2" and effects[1][3] == "repression"


def test_build_model_with_no_annotations_and_no_edges():
    model = build_expression_model(_grn())
    assert model.promoter_strength == {}
    assert model.tf_effects == {}


# ── infer_expression ───────────────────────────────────────────────────────────

def test_infer_expression_baseline_without_regulation():
    model = ExpressionModel(
        promoter_strength={"g1": 1.0, "g2": 0.5},
        rbs_strength={"g1": 1.0, "g2": 2.0},
    )
    # no TF candidates/edges, no environment
    out = infer_expression(_grn(), model=model)
    assert out["g1"] == 1.0
    assert out["g2"] == 0.5 * 2.0


def test_infer_expression_clamps_to_unit():
    model = ExpressionModel(promoter_strength={"g": 5.0}, rbs_strength={"g": 3.0})
    out = infer_expression(_grn(), model=model)
    assert out["g"] == 1.0


def test_infer_expression_applies_tf_environment():
    # gA activated by TF1; TF1 at env level 0.5, kd 0.5, n=1 -> factor 0.5
    grn = _grn(tfs=["TF1"], edges=[_edge("TF1", "gA", "activation")])
    model = build_expression_model(grn, {"gA": SimpleNamespace()})
    out = infer_expression(grn, model=model, environment={"TF1": 0.5})
    assert abs(out["gA"] - 0.5) < 1e-9


def test_infer_expression_ignores_env_for_non_candidates():
    grn = _grn(tfs=["TF1"], edges=[_edge("TF1", "gA", "activation")])
    model = build_expression_model(grn, {"gA": SimpleNamespace()})
    # env for an unknown gene is ignored (filtered by tf_levels)
    out = infer_expression(grn, model=model, environment={"NOPE": 0.9})
    assert abs(out["gA"] - 0.5) < 1e-9


def test_infer_expression_builds_model_when_none():
    grn = _grn(tfs=["TF1"], edges=[_edge("TF1", "gA", "activation")])
    out = infer_expression(grn, environment={"TF1": 0.0})
    assert "gA" in out
    assert out["gA"] == 0.0


def test_infer_expression_builds_with_annotations():
    grn = _grn(edges=[_edge("TF1", "g1", "activation")])
    out = infer_expression(grn, annotations={"g1": SimpleNamespace()})
    assert "g1" in out
    assert "TF1" in out  # tf added as a gene with target entries


# ── infer_expression_at_time ───────────────────────────────────────────────────

def test_infer_expression_at_time_returns_state():
    grn = _grn(edges=[_edge("TF1", "gA", "activation")])
    state = infer_expression_at_time(grn, time_hours=2.5)
    assert state.timestamp == 2.5
    assert set(state.gene_levels) == set(state.enzyme_levels) == set(state.mrna_levels)
    for g, v in state.enzyme_levels.items():
        assert state.mrna_levels[g] == v


def test_infer_expression_at_time_passes_model():
    model = ExpressionModel(promoter_strength={"gA": 1.0}, rbs_strength={"gA": 1.0})
    state = infer_expression_at_time(_grn(), model=model)
    assert state.enzyme_levels["gA"] == 1.0
