"""Coverage-closure tests for :mod:`helixlang.plugins.apps.genome_scale`.

Exercises the RegulonDB parsing edge cases (short lines, prefix header,
unknown/non-numeric effects), the ``_scale_free_edges`` collision arc, the
``build_genome`` validation guards + noise-repeat branch, ``powerlaw_fit``
degenerate inputs, and the ``GenomeColony`` grow-path / ``free_row(-1)`` /
unknown-knockout branches.
"""
import random

import pytest

from helixlang.plugins.apps.genome_scale import (
    GenomeColony,
    _core_genes,
    _parse_regulondb_effect,
    _scale_free_edges,
    build_genome,
    parse_regulondb,
    parse_regulondb_full,
    powerlaw_fit,
)


class TestParseRegulondb:
    def test_short_line_skipped(self):
        assert parse_regulondb("crp\tgltA\n") == []

    def test_bad_effect_value_skipped(self):
        assert parse_regulondb("crp\tgltA\t+x\n") == []

    def test_full_format_edges(self):
        dump = (
            "regulator\ttarget\teffect\n"
            "# comment line\n"
            "crp\tgltA\t+0.8\n"
            "arcA\tgltA\t+x\n"
            "fnr\tldhA\t-y\n"
            "fis\tzwf\tabc\n"
            "lrp\tppc\t\n"
            "a\tb\n"
            "ihf\tsucAB\tunknown\n"
        )
        assert parse_regulondb_full(dump) == [
            ("crp", "gltA", 0.8),
            ("arcA", "gltA", 1.0),
            ("fnr", "ldhA", -1.0),
        ]

    def test_tf_prefix_header(self):
        dump = "tf\ttarget\t-\nX\tY\t+\n"
        assert parse_regulondb_full(dump) == [("X", "Y", 1.0)]

    def test_effect_edge_cases(self):
        assert _parse_regulondb_effect("") == 0.0
        assert _parse_regulondb_effect("unknown") == 0.0
        assert _parse_regulondb_effect("+0.8") == 0.8
        assert _parse_regulondb_effect("activation") == 1.0
        assert _parse_regulondb_effect("repression") == -1.0
        assert _parse_regulondb_effect("+q") == 1.0
        assert _parse_regulondb_effect("-q") == -1.0
        assert _parse_regulondb_effect("abc") == 0.0

    def test_build_drops_self_loops_and_duplicates(self):
        spec = build_genome(
            n_genes=40, tf_map="regulondb", seed=3,
            regulondb="crp\tcrp\t+\ncrp\tgltA\t+\ncrp\tgltA\t-\n")
        assert spec.grn.n_edges == len(_core_genes()) + 1


class TestScaleFreeCollision:
    def test_multi_target_duplicate_draws(self):
        names = [f"g{i}" for i in range(12)]
        edges = _scale_free_edges(names, m=3, seed=4)
        lo = 2 + (len(names) - 2) * 3 - 1
        hi = 2 + (len(names) - 2) * 3
        assert lo <= len(edges) <= hi


class TestBuildGenomeValidation:
    def test_bad_grn_mode(self):
        with pytest.raises(ValueError, match="grn_mode"):
            build_genome(n_genes=40, grn_mode="bogus")

    def test_bad_tf_map(self):
        with pytest.raises(ValueError, match="tf_map"):
            build_genome(n_genes=40, tf_map="bogus")

    def test_noise_repeat_hits_same_gene(self, monkeypatch):
        calls: list[tuple] = []

        def _fixed_randrange(self, *a, **k):
            calls.append(a)
            return 7

        monkeypatch.setattr(random.Random, "randrange", _fixed_randrange)
        spec = build_genome(n_genes=40, tf_map="regulondb", seed=3,
                            noise_seed=1)
        assert spec.grn.n_genes > 0
        assert len(calls) >= 2


class TestPowerlaw:
    def test_empty_degrees(self):
        assert powerlaw_fit([]) == {"slope": 0.0, "r2": 0.0}

    def test_too_few_support(self):
        assert powerlaw_fit([1, 1, 1, 1]) == {"slope": 0.0, "r2": 0.0}


class TestGenomeColonyRows:
    def test_grow_beyond_initial_rows(self):
        spec = build_genome(n_genes=120, tf_map="off", seed=7)
        colony = GenomeColony(spec, n_cells=2)
        rows = [colony.alloc_row() for _ in range(3)]
        assert rows == [2, 3, 4]
        assert colony.levels.shape[0] >= 5

    def test_free_row_negative_is_ignored(self):
        spec = build_genome(n_genes=120, tf_map="off", seed=7)
        colony = GenomeColony(spec, n_cells=2)
        colony.free_row(-1)
        assert colony._free_rows == []

    def test_knockout_unknown_gene(self):
        spec = build_genome(n_genes=120, tf_map="off", seed=7)
        colony = GenomeColony(spec, n_cells=2)
        colony.knock_out(["no_such_gene"])
        assert colony.levels.sum() == 0.0
