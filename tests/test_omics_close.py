"""Coverage-closure tests for helixlang.plugins.omics.

Exercises the plugin contract, validation/edge paths in the expression
matrix, transpose+coords import, empty-cluster k-means, and the degenerate
adjusted-Rand-index branches.
"""
from __future__ import annotations

import math

import pytest

import helixlang.plugins.omics as omics_mod
from helixlang.core.errors import PluginDependencyError
from helixlang.plugins.omics import (
    ExpressionMatrix,
    _check,
    _load,
    _make_backend,
    adjusted_rand_index,
    compare_heterogeneity,
    from_arrays,
    read_expression_matrix,
)
from helixlang.plugins.omics._spatial_omics import _euclidean


class TestPluginContract:
    def test_check_importable_package(self):
        assert _check("helixlang") is True

    def test_check_missing_package(self):
        assert _check("helixlang_nonexistent_module_xyz") is False

    def test_load_returns_backend_factory(self):
        backend_factory = _load()
        assert backend_factory is _make_backend
        backend = backend_factory()
        from helixlang.plugins.omics.expression_inference import ExpressionModel
        assert backend is ExpressionModel

    def test_load_raises_when_dependency_missing(self, monkeypatch):
        monkeypatch.setattr(omics_mod, "_check", lambda pkg: False)
        with pytest.raises(PluginDependencyError):
            _load()

    def test_load_raises_message(self, monkeypatch):
        monkeypatch.setattr(omics_mod, "_check", lambda pkg: False)
        with pytest.raises(PluginDependencyError) as exc:
            _load()
        assert exc.value.name == "omics"
        assert exc.value.dep == "numpy"
        assert exc.value.extra == "ml"


class TestExpressionMatrixEdges:
    def test_euclidean(self):
        assert _euclidean([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(
            math.sqrt(27.0)
        )

    def test_coordinate_mismatch_y(self):
        with pytest.raises(ValueError):
            ExpressionMatrix(
                ["A"], ["c0", "c1"], [[1.0], [2.0]],
                x=[0.0, 1.0], y=[0.0],
            )

    def test_coordinate_mismatch_z(self):
        with pytest.raises(ValueError):
            ExpressionMatrix(
                ["A"], ["c0", "c1"], [[1.0], [2.0]],
                x=[0.0, 1.0], y=[0.0, 1.0], z=[0.0],
            )

    def test_normalized_zero_gene(self):
        em = from_arrays(["A", "B"], ["c0", "c1"], [[0.0, 1.0], [0.0, 2.0]])
        out = em.normalized(method="max")
        assert out.gene_profile("A") == [0.0, 0.0]
        assert out.gene_profile("B") == [0.5, 1.0]

    def test_cluster_nonpositive_k(self):
        em = from_arrays(["A"], ["c0"], [[1.0]])
        with pytest.raises(ValueError):
            em.cluster(0)
        with pytest.raises(ValueError):
            em.cluster(-1)

    def test_cluster_no_cells(self):
        em = from_arrays(["A"], [], [])
        with pytest.raises(ValueError):
            em.cluster(2)

    def test_cluster_empty_membership(self):
        em = from_arrays(
            ["A", "B"], ["c0", "c1", "c2"], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        )
        ids, cents = em.cluster(2, seed=0, iters=5)
        assert len(set(ids)) == 1
        assert len(cents) == 2

    def test_cluster_loop_exhausts(self):
        em = from_arrays(
            ["A", "B"],
            ["c0", "c1", "c2"],
            [[5.0, 0.0], [0.0, 5.0], [4.0, 1.0]],
        )
        ids, cents = em.cluster(3, seed=0, iters=1)
        assert len(ids) == 3
        assert all(len(c) == 2 for c in cents)


class TestImporterEdges:
    def test_empty_matrix_raises(self, tmp_path):
        f = tmp_path / "empty.tsv"
        f.write_text("# only comments\n\n")
        with pytest.raises(ValueError):
            read_expression_matrix(str(f))

    def test_transpose_with_coords(self, tmp_path):
        f = tmp_path / "t.csv"
        f.write_text("gene,c1,c2,x\nG1,1,2,10\nG2,3,4,20\n")
        with pytest.raises(ValueError):
            read_expression_matrix(
                str(f), delimiter=",", transpose=True, coords_columns=("x", "zzz")
            )

    def test_reserved_cell_label(self, tmp_path):
        f = tmp_path / "r.tsv"
        f.write_text("cell\tG1\n0\t1\n")
        with pytest.raises(ValueError):
            read_expression_matrix(str(f), coords_columns=("cell",))


class TestRandIndexDegenerate:
    def test_ari_single_cell_zero_total(self):
        assert adjusted_rand_index([0], [0]) == 0.0

    def test_ari_two_cells_perfect(self):
        assert adjusted_rand_index([0, 0], [0, 0]) == 1.0

    def test_ari_two_cells_single_clusters(self):
        # two single-cluster labelings are identical partitions -> ARI 1.0
        assert adjusted_rand_index([0, 0], [1, 1]) == 1.0


def test_compare_heterogeneity_more_simulated_states() -> None:
    res = compare_heterogeneity([0, 1, 2, 3], [0, 0, 1, 1])
    assert 0.0 <= res["state_match"] <= 1.0
