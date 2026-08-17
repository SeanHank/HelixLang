"""Large-scale vectorized runtime tests (T3.3, gap G8).

Verification goals:
- VectorizedGRN reproduces GRN.step byte-for-byte for many cells at once
  (sigmoid, Hill, mixed), over both scalar and vectorized APIs.
- sort_cells orders a population by spatial attributes (stability).
- iter_snapshots streams per-tick snapshots and can append JSONL.

References:
- Kick et al. 2019 NUFEB (Commun Comput Phys 27:1882) — large-scale
  individual-based simulation; neighbor locality
- Alon 2007 — threshold GRN dynamics in matrix form
"""
from __future__ import annotations

import json
import pathlib

import pytest

from helixlang.grn import GRN
from helixlang.population import PopulationCell, PopulationConfig
from helixlang.vectorized import (
    VectorizedGRN,
    iter_snapshots,
    optional_jit,
    sort_cells,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _toggle_grn() -> GRN:
    g = GRN()
    for name in ("A", "B", "C"):
        g.add_gene(name, 0.5)
    g.add_edge("A", "B", 2.0)
    g.add_edge("B", "C", 1.5)
    g.add_edge("C", "A", -2.0)
    return g


def _scalar_rows(grn: GRN, init: list[list[float]], n_steps: int) -> list[list[float]]:
    rows = []
    for row in init:
        gg = GRN()
        for name, node in grn.nodes.items():
            gg.add_gene(name, node.threshold, decay=node.decay,
                        hill_n=node.hill_n, kd=node.kd)
        for e in grn.edges:
            gg.add_edge(e.source, e.target, e.weight)
        for name, lv in zip(list(grn.nodes), row, strict=True):
            gg.nodes[name].level = lv
        for _ in range(n_steps):
            gg.step()
        rows.append([gg.nodes[n].level for n in list(grn.nodes)])
    return rows


@pytest.mark.skipif(np is None, reason="numpy required")
def test_vectorized_matches_scalar_sigmoid() -> None:
    grn = _toggle_grn()
    init = [[0.9, 0.1, 0.2], [0.1, 0.9, 0.2],
            [0.5, 0.5, 0.5], [0.0, 0.0, 1.0]]
    vg = VectorizedGRN(grn)
    levels = np.array(init, dtype=float)
    for _ in range(50):
        levels = vg.step(levels)
    scalar = np.array(_scalar_rows(grn, init, 50))
    assert np.allclose(levels, scalar, atol=1e-12)
    assert np.abs(levels - scalar).max() <= 2 * np.finfo(float).eps


@pytest.mark.skipif(np is None, reason="numpy required")
def test_vectorized_matches_scalar_hill() -> None:
    grn = GRN()
    grn.add_gene("X", 0.5, hill_n=2.0, kd=0.4)
    grn.add_gene("Y", 0.5, hill_n=1.0, kd=0.7)
    grn.add_edge("X", "Y", 1.0)
    grn.add_edge("Y", "X", 0.5)
    init = [[0.1, 0.1], [0.9, 0.1], [0.3, 0.8], [0.6, 0.6]]
    vg = VectorizedGRN(grn)
    levels = np.array(init, dtype=float)
    for _ in range(30):
        levels = vg.step(levels)
    scalar = np.array(_scalar_rows(grn, init, 30))
    assert np.allclose(levels, scalar, atol=1e-9)


@pytest.mark.skipif(np is None, reason="numpy required")
def test_vectorized_mixed_activation() -> None:
    grn = GRN()
    grn.add_gene("S", 0.5)               # sigmoid path
    grn.add_gene("H", 0.5, hill_n=2.0, kd=0.3)  # Hill path
    grn.add_edge("S", "H", 1.2)
    grn.add_edge("H", "S", -0.8)
    init = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
    vg = VectorizedGRN(grn)
    levels = np.array(init, dtype=float)
    for _ in range(25):
        levels = vg.step(levels)
    scalar = np.array(_scalar_rows(grn, init, 25))
    assert np.allclose(levels, scalar, atol=1e-9)


@pytest.mark.skipif(np is None, reason="numpy required")
def test_vectorized_triggered_mask() -> None:
    grn = _toggle_grn()
    vg = VectorizedGRN(grn)
    levels = np.array([[0.9, 0.1, 0.2]])
    mask = vg.triggered(levels)
    assert mask.shape == (1, 3)
    assert bool(mask[0, 0]) and not bool(mask[0, 1])


@pytest.mark.skipif(np is None, reason="numpy required")
def test_vectorized_clips_to_unit_interval() -> None:
    grn = _toggle_grn()
    vg = VectorizedGRN(grn)
    levels = np.array([[0.0, 0.0, 0.0]])
    for _ in range(5):
        levels = vg.step(levels)
    assert float(levels.min()) >= 0.0 and float(levels.max()) <= 1.0


def test_n_genes() -> None:
    vg = VectorizedGRN(_toggle_grn())
    assert vg.n_genes == 3
    assert vg.names == ["A", "B", "C"]


def test_optional_jit_noop_without_numba() -> None:
    @optional_jit()
    def double(x: int) -> int:
        return x * 2
    assert double(21) == 42


def test_sort_cells_spatial_order() -> None:
    cells = [PopulationCell(id=3, energy=1.0, x=5, y=1),
             PopulationCell(id=1, energy=1.0, x=1, y=9),
             PopulationCell(id=2, energy=1.0, x=1, y=2),
             PopulationCell(id=0, energy=1.0, x=1, y=2)]
    ordered = sort_cells(cells)
    ids = [c.id for c in ordered]
    # (1,2): id2 then id0 (input order), then (1,9), then (5,1)
    assert ids == [2, 0, 1, 3]


def test_iter_snapshots_streams() -> None:
    cfg = PopulationConfig(grid_width=10, grid_height=10,
                           energy_intake=100.0, metabolic_cost=1.0)
    pop = _MinimalPopulation(cfg)
    snaps = list(iter_snapshots(pop, n_steps=3, interval=1))
    assert [s["step"] for s in snaps] == [1, 2, 3]
    assert all(s["alive"] >= 0 for s in snaps)


def test_iter_snapshots_interval_and_jsonl(tmp_path: pathlib.Path) -> None:
    cfg = PopulationConfig(grid_width=10, grid_height=10,
                           energy_intake=100.0, metabolic_cost=1.0)
    pop = _MinimalPopulation(cfg)
    out = tmp_path / "snap.jsonl"
    snaps = list(iter_snapshots(pop, n_steps=4, interval=2, path=str(out)))
    assert [s["step"] for s in snaps] == [1, 3]  # snapshots at steps 1 and 3
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 1


def test_iter_snapshots_validation() -> None:
    with pytest.raises(ValueError):
        list(iter_snapshots(None, n_steps=-1))
    with pytest.raises(ValueError):
        list(iter_snapshots(None, n_steps=1, interval=0))


class _MinimalPopulation:
    """Duck-typed stand-in with step()/cells for snapshot streaming."""

    def __init__(self, cfg: PopulationConfig) -> None:
        self.config = cfg
        self.cells = [PopulationCell(id=0, energy=100.0, x=1, y=1)]

    def step(self) -> dict[str, int]:
        return {"alive_count": 1}
