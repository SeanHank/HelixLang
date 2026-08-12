"""Spatial-omics-guided models (T3.1, gap G8) — MiMICS-style tests.

Verification goals:
- ExpressionMatrix importers (arrays + TSV/CSV with spatial columns),
  per-gene normalization, binarized barcodes, deterministic k-means
  states.
- expression_to_grn_states / build_state_grn: expression states map onto
  distinct GRN threshold + initial-level sets without mutating the
  template or losing edges/decays.
- expression_to_fba_bounds / apply_fba_bounds: states map onto distinct
  uptake / reaction bounds; the resulting FBA solutions differ
  (transcriptomics-guided metabolic switching).
- SpatialAtlas: nearest-spot state assignment for population cells.
- compare_heterogeneity / adjusted_rand_index: agreement between
  simulated and observed per-cell states.

References:
- Walsh et al. 2024 MiMICS (PLoS Comput Biol 20(4):e1012031)
- Dar et al. 2021 spatial transcriptomics (Nat Biotechnol 39:313-319)
- Hubert & Arabie 1985 (adjusted Rand index)
"""
from __future__ import annotations

import math

import pytest

from helixlang.grn import GRN
from helixlang.metabolism import ECOLI_CORE_MODEL, FluxBalanceAnalysis
from helixlang.omics import (
    ExpressionMatrix,
    SpatialAtlas,
    adjusted_rand_index,
    apply_fba_bounds,
    build_state_grn,
    compare_heterogeneity,
    expression_to_fba_bounds,
    expression_to_grn_states,
    from_arrays,
    read_expression_matrix,
)

GENES = ["A", "B", "C", "D", "E", "F"]


def _two_state_matrix() -> ExpressionMatrix:
    # 6 cells, 6 genes: state 0 = {A,B} high, state 1 = {C,D} high,
    # state 2 = {E,F} high; spatial x in {0..5}
    return from_arrays(
        genes=GENES,
        cells=[f"c{i}" for i in range(6)],
        values=[
            [5, 5, 0, 0, 0, 0],
            [4, 4, 1, 0, 0, 0],
            [0, 0, 5, 5, 0, 0],
            [0, 0, 4, 4, 1, 0],
            [1, 0, 1, 0, 5, 5],
            [0, 1, 0, 1, 4, 4],
        ],
        x=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        y=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


# ============================================================================
# ExpressionMatrix + importers
# ============================================================================

def test_from_arrays_shape_and_access() -> None:
    em = _two_state_matrix()
    assert em.shape == (6, 6)
    assert em.cell_vector("c0") == [5, 5, 0, 0, 0, 0]
    assert em.gene_profile("E") == [0, 0, 0, 1, 5, 4]
    assert em.x == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_from_arrays_validates_dimensions() -> None:
    with pytest.raises(ValueError):
        from_arrays(["A", "B"], ["c0"], [[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        from_arrays(["A", "B"], ["c0"], [[1]])
    with pytest.raises(ValueError):
        from_arrays(["A"], ["c0", "c1"], [[1], [2]], x=[0.0])


def test_normalized_max_and_sum() -> None:
    em = _two_state_matrix()
    nm = em.normalized(method="max")
    assert nm.values[0] == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    sm = em.normalized(method="sum")
    assert math.isclose(sum(sm.gene_profile("A")), 1.0)


def test_barcode() -> None:
    em = from_arrays(["A", "B"], ["c0", "c1"], [[0.9, 0.1], [0.2, 0.8]])
    bc = em.barcode(threshold=0.5)
    assert bc.values[0] == [1.0, 0.0]
    assert bc.values[1] == [0.0, 1.0]


def test_cluster_deterministic_and_separates_states() -> None:
    em = _two_state_matrix()
    ids, cents = em.cluster(3, seed=1)
    assert ids == [1, 1, 0, 0, 2, 2]
    assert sorted(set(ids)) == [0, 1, 2]
    ids2, _ = em.cluster(3, seed=1)
    assert ids2 == ids
    # state centroids are separated: A high in one state, absent in others
    gA = em.genes.index("A")
    assert max(c[gA] for c in cents) > 0 and min(c[gA] for c in cents) == 0


def test_state_centroids() -> None:
    em = _two_state_matrix()
    cents = em.state_centroids([0, 0, 1, 1, 1, 0])
    # state 0 = cells c0,c1,c5
    assert cents[0] == [3.0, 10.0 / 3, 1.0 / 3, 1.0 / 3, 4.0 / 3, 4.0 / 3]
    # state 1 = cells c2,c3,c4
    assert cents[1] == [1.0 / 3, 0.0, 10.0 / 3, 3.0, 2.0, 5.0 / 3]


def test_read_expression_matrix_tsv(tmp_path) -> None:
    f = tmp_path / "expr.tsv"
    f.write_text(
        "cell\tx\ty\tG1\tG2\tG3\n"
        "s0\t0\t0\t1\t2\t3\n"
        "s1\t1\t0\t4\t5\t6\n")
    em = read_expression_matrix(str(f), coords_columns=("x", "y"))
    assert em.genes == ["G1", "G2", "G3"]
    assert em.cells == ["s0", "s1"]
    assert em.values[1] == [4.0, 5.0, 6.0]
    assert em.x == [0.0, 1.0]
    assert em.y == [0.0, 0.0]


def test_read_expression_matrix_transpose(tmp_path) -> None:
    f = tmp_path / "expr.csv"
    f.write_text("gene,c1,c2\nG1,1,2\nG2,3,4\n")
    em = read_expression_matrix(str(f), delimiter=",", transpose=True)
    assert em.genes == ["G1", "G2"]
    assert em.cells == ["c1", "c2"]
    assert em.values == [[1.0, 3.0], [2.0, 4.0]]


def test_read_expression_matrix_skips_comments(tmp_path) -> None:
    f = tmp_path / "expr.tsv"
    f.write_text("# header comment\ncell\tG1\ns0\t1\n\ns1\t2\n")
    em = read_expression_matrix(str(f))
    assert em.cells == ["s0", "s1"]
    assert em.values == [[1.0], [2.0]]


# ============================================================================
# Expression -> GRN parameter sets
# ============================================================================

def test_expression_to_grn_states() -> None:
    em = _two_state_matrix()
    ids, cents, params = expression_to_grn_states(em, 3, seed=1,
                                                  base_threshold=0.5,
                                                  min_threshold=0.2)
    assert ids == [1, 1, 0, 0, 2, 2]
    # gene A is high in the {A,B} state -> low threshold, high level
    a_high = params[1]["A"]
    assert a_high["initial_level"] == 1.0
    assert a_high["threshold"] == 0.2
    a_low = params[0]["A"]
    assert a_low["initial_level"] == 0.0
    assert a_low["threshold"] == 0.5


def test_build_state_grn_preserves_structure() -> None:
    template = GRN()
    for name in ["A", "B", "C"]:
        template.add_gene(name, 0.5)
    template.add_edge("A", "B", 2.0)
    template.add_edge("B", "C", -1.5)
    params = {"A": {"initial_level": 0.8, "threshold": 0.3}}
    g = build_state_grn(template, params)
    assert g.nodes["A"].threshold == 0.3
    assert g.nodes["A"].level == 0.8
    assert g.nodes["B"].threshold == 0.5  # template value kept
    assert g.nodes["C"].level == 0.0
    assert [(e.source, e.target, e.weight) for e in g.edges] == \
        [("A", "B", 2.0), ("B", "C", -1.5)]
    # template untouched
    assert template.nodes["A"].threshold == 0.5
    assert template.nodes["A"].level == 0.0


def test_state_grns_run_and_diverge() -> None:
    # toggle-switch-like template with the state params applied
    template = GRN()
    for name in ["A", "B"]:
        template.add_gene(name, 0.5)
    template.add_edge("A", "B", -4.0)
    template.add_edge("B", "A", -4.0)
    em = _two_state_matrix()
    _, _, params = expression_to_grn_states(em, 3, seed=1)
    grn_a = build_state_grn(template, params[1])  # A high
    grn_b = build_state_grn(template, params[0])  # A off
    for _ in range(20):
        grn_a.step()
        grn_b.step()
    assert grn_a.nodes["A"].level > 0.5
    assert grn_b.nodes["A"].level < 0.5


# ============================================================================
# Expression -> FBA bound sets
# ============================================================================

def test_expression_to_fba_bounds() -> None:
    em = _two_state_matrix()
    ids, _, _ = expression_to_grn_states(em, 3, seed=1)
    bounds = expression_to_fba_bounds(em, ids, {"A": "EX_glc", "C": "EX_ac"},
                                      10.0)
    # state 1 = {A,B} high (c0,c1): glucose nearly saturating, little acetate
    assert math.isclose(bounds[1]["EX_glc"], 9.0)
    assert bounds[1]["EX_ac"] < 2.0
    # state 0 = {C,D} high: no glucose, acetate driven by C expression
    assert bounds[0]["EX_glc"] == 0.0
    assert math.isclose(bounds[0]["EX_ac"], 9.0)


def test_expression_to_fba_bounds_inverted_direction() -> None:
    em = _two_state_matrix()
    ids = [0, 0, 1, 1, 2, 2]
    bounds = expression_to_fba_bounds(
        em, ids, {"A": "EX_glc"}, 10.0, direction={"A": -1})
    # state 1 (cells c2,c3) has A = 0 -> inverted bound saturates
    assert bounds[1]["EX_glc"] == 10.0
    # state 0 (cells c0,c1) has A near max -> inverted bound collapsed
    assert math.isclose(bounds[0]["EX_glc"], 1.0, abs_tol=1e-6)


def test_apply_fba_bounds_switches_metabolism() -> None:
    em = _two_state_matrix()
    ids, _, _ = expression_to_grn_states(em, 3, seed=1)
    bounds = expression_to_fba_bounds(em, ids, {"A": "EX_glc", "C": "EX_ac"},
                                      10.0)
    fba_on = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
    apply_fba_bounds(fba_on, bounds[1])
    growth_on = fba_on.solve()["BIOMASS"]
    fba_off = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
    apply_fba_bounds(fba_off, bounds[0])
    growth_off = fba_off.solve()["BIOMASS"]
    assert growth_on > 0.1
    assert growth_off < growth_on


def test_apply_fba_bounds_ignores_unknown_reactions() -> None:
    fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
    apply_fba_bounds(fba, {"NOT_A_REACTION": 5.0})  # no raise
    assert "NOT_A_REACTION" not in fba.uptake_limits


def test_apply_fba_bounds_upper_bound_reaction() -> None:
    fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
    apply_fba_bounds(fba, {"PGI": 2.0})
    assert fba.model.reactions["PGI"].upper_bound == 2.0


# ============================================================================
# SpatialAtlas + heterogeneity
# ============================================================================

def test_spatial_atlas_nearest_state() -> None:
    atlas = SpatialAtlas([(0, 0, 0), (10, 0, 0)], [0, 1])
    assert atlas.state_at(1, 0) == 0
    assert atlas.state_at(9, 0) == 1
    assert atlas.state_at(5, 0) == 0
    assert atlas.state_at(5, 0, 3) == 0  # z included in distance


def test_spatial_atlas_assign_cells() -> None:
    class FakeCell:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    atlas = SpatialAtlas([(0, 0, 0), (10, 0, 0)], [0, 1])
    cells = [FakeCell(1, 0, 0), FakeCell(9, 0, 0), FakeCell(5, 0, 0)]
    assert atlas.assign_cell_states(cells) == [0, 1, 0]


def test_spatial_atlas_empty_raises() -> None:
    with pytest.raises(ValueError):
        SpatialAtlas([], [0])
    with pytest.raises(ValueError):
        SpatialAtlas([], []).state_at(0, 0)


def test_adjusted_rand_index() -> None:
    assert adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0
    assert adjusted_rand_index([0, 0, 0, 0], [0, 1, 2, 3]) == 0.0
    assert adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]) > 0.0
    with pytest.raises(ValueError):
        adjusted_rand_index([0, 0], [0])


def test_compare_heterogeneity_agreement() -> None:
    res = compare_heterogeneity([0, 0, 1, 1], [0, 0, 1, 1])
    assert res["ari"] == 1.0
    assert res["state_match"] == 1.0
    # label permutation does not hurt agreement
    res2 = compare_heterogeneity([1, 1, 0, 0], [0, 0, 1, 1])
    assert res2["ari"] == 1.0
    assert res2["state_match"] == 1.0
    # disagreement
    res3 = compare_heterogeneity([0, 0, 0, 0], [0, 1, 2, 3])
    assert res3["state_match"] < 0.5


def test_compare_heterogeneity_empty() -> None:
    assert compare_heterogeneity([], []) == {"ari": 1.0, "state_match": 1.0}
