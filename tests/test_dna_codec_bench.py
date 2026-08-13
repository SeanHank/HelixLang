"""DNA codec benchmark tests: cost-robustness trade-off (S5).

Verification goals (Nat Commun 2026 "DNA storage for which scenario"
methodology; Erlich & Zielinski 2017 Science 355:950):
- As the code rate falls (redundancy rises), the tolerated molecule
  loss rises: 0.5 bit/nt fountain tolerates well over half the molecules
  being lost, while the near-native 1.5 bit/nt code tolerates none.
- Block Reed-Solomon shows the complementary profile: it corrects base
  errors (more at low density) but cannot tolerate whole-block erasure
  (no redundancy across blocks).
- Goldman's 4x overlapping segments are reported at their native low
  density with no rate knob.
- The cost per stored GB reflects the rate: lower density -> more DNA
  per byte -> higher cost.
"""
from __future__ import annotations

import pytest

from helixlang.apps.dna_storage import (
    CodecBenchmarkRow,
    benchmark_codecs,
    format_benchmark_table,
)


@pytest.fixture(scope="session")
def rows() -> list[CodecBenchmarkRow]:
    return benchmark_codecs(data_size=256, seed=7)


def test_row_count_matches_schemes_times_densities(
        rows: list[CodecBenchmarkRow]) -> None:
    # goldman(1) + fountain(3 rates) + rs(3 rates)
    assert len(rows) == 7
    assert sum(1 for r in rows if r.scheme == "goldman") == 1
    assert sum(1 for r in rows if r.scheme == "fountain") == 3
    assert sum(1 for r in rows if r.scheme == "rs") == 3


def test_fountain_loss_tolerance_rises_as_rate_falls(
        rows: list[CodecBenchmarkRow]) -> None:
    fountain = [r for r in rows if r.scheme == "fountain"]
    fountain.sort(key=lambda r: r.achieved_density)
    densities = [r.achieved_density for r in fountain]
    tolerances = [r.max_loss_fraction for r in fountain]
    # achieved densities are ascending ...
    assert densities[0] < densities[1] < densities[2]
    # ... and loss tolerance strictly falls as density rises
    assert tolerances[0] > tolerances[1] > tolerances[2]


def test_fountain_low_density_tolerates_heavy_loss(
        rows: list[CodecBenchmarkRow]) -> None:
    # Literature anchor: ~0.5 bit/nt fountain survives ~60% molecule loss
    low = min((r for r in rows if r.scheme == "fountain"),
              key=lambda r: r.achieved_density)
    assert low.achieved_density < 0.6
    assert low.max_loss_fraction >= 0.5


def test_fountain_high_density_has_no_erasure_slack(
        rows: list[CodecBenchmarkRow]) -> None:
    high = max((r for r in rows if r.scheme == "fountain"),
               key=lambda r: r.achieved_density)
    assert high.achieved_density >= 1.3
    assert high.max_loss_fraction < 0.1


def test_rs_error_tolerance_rises_as_rate_falls(
        rows: list[CodecBenchmarkRow]) -> None:
    rs = [r for r in rows if r.scheme == "rs"]
    rs.sort(key=lambda r: r.achieved_density)
    assert (rs[0].max_error_rate > rs[1].max_error_rate
            > rs[2].max_error_rate)


def test_rs_block_code_cannot_tolerate_erasure(
        rows: list[CodecBenchmarkRow]) -> None:
    # Block RS corrects within codewords, not across them: losing any
    # whole block is irrecoverable, so loss tolerance is < 1/nblocks.
    for r in (r for r in rows if r.scheme == "rs"):
        assert r.max_loss_fraction < 0.25


def test_goldman_reported_at_native_density(
        rows: list[CodecBenchmarkRow]) -> None:
    goldman = next(r for r in rows if r.scheme == "goldman")
    assert goldman.target_density is None
    assert 0.0 < goldman.achieved_density < 0.5
    assert goldman.num_oligos > 0


def test_cost_per_gb_rises_as_density_falls(
        rows: list[CodecBenchmarkRow]) -> None:
    # Lower density = more DNA per stored byte = higher cost per GB
    fountain = [r for r in rows if r.scheme == "fountain"]
    fountain.sort(key=lambda r: r.achieved_density)
    assert fountain[0].cost_per_gb_usd > fountain[2].cost_per_gb_usd


def test_format_benchmark_table_renders_rows(
        rows: list[CodecBenchmarkRow]) -> None:
    table = format_benchmark_table(rows)
    lines = table.splitlines()
    assert "scheme" in lines[0]
    assert len(lines) == len(rows) + 1
    assert any("fountain" in line for line in lines)
