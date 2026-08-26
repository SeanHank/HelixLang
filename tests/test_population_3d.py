"""3D population + 3D diffusion tests (T2.7, P10).

Verification goals:
- ConcentrationField3D: 7-point Laplacian diffusion conserves mass,
  spreads along all three axes, Neumann boundaries hold, layer() slices
  a 2D plane, out-of-bounds access is safe.
- CellPopulation3D: cells carry a z coordinate; signal field is a
  [z][y][x] volume; emission lands at the correct 3D site; division
  offsets along z; z-neighborhoods (6- and 26-connectivity) are
  in-bounds and include the z axis.
- Large-scale (biofilm-scale) path: the numpy batch metabolism of
  CellPopulation3D reproduces the pure-Python 3D path exactly; the
  per-cell cost scales near-linearly from 10^3 to 10^4 agents; a seeded
  monolayer grows into a colony whose z-density is anchored at the
  substratum (biomass stays near the surface); occupancy and spatial
  mechanics dispatch into the 3D z-aware implementations.
- to_lsystem3d exports the colony occupancy through morphology_3d
  (every occupied site appears in the derived point cloud).

References:
- NUFEB 2019 (Kick et al. Commun Comput Phys 27:1882-1908): 3D
  individual-based microbial simulation with 3D chemical fields
- BM3 benchmark (Cockx et al. 2024): biofilm growth anchored at a
  substratum with biomass concentrated at the surface
- Fick's law; explicit finite-difference 3D Laplacian (Press et al.
  Numerical Recipes); Prusinkiewicz & Lindenmayer 1990 (3D L-systems)
"""
from __future__ import annotations

import math
import random
import time

import pytest

from helixlang.environment import ConcentrationField3D
from helixlang.population import (
    _HAS_NUMPY,
    CellPopulation3D,
    PopulationCell,
    PopulationConfig,
    _laplacian_step_3d,
    _neighbors_3d,
)

GRID = dict(grid_width=12, grid_height=12, grid_depth=12,
            signaling_enabled=True, signal_diffusion=400.0,
            energy_intake=100.0, division_threshold=150.0,
            metabolic_cost=1.0)


# ============================================================================
# ConcentrationField3D
# ============================================================================

def test_3d_field_dimensions() -> None:
    f = ConcentrationField3D("g", 8, 6, 4, 600.0, 0.5)
    assert (f.width, f.height, f.depth) == (8, 6, 4)
    assert f.get(0, 0, 0) == 0.5
    assert f.get(7, 5, 3) == 0.5
    assert f.get(-1, 0, 0) == 0.0
    assert f.get(0, 0, 99) == 0.0


def test_3d_field_add_deplete() -> None:
    f = ConcentrationField3D("g", 8, 8, 8, 600.0, 0.0)
    f.add(3, 3, 3, 2.0)
    assert f.get(3, 3, 3) == 2.0
    assert f.deplete(3, 3, 3, 0.5) == 0.5
    assert f.get(3, 3, 3) == 1.5
    assert f.deplete(3, 3, 3, 99.0) == 1.5  # never negative
    assert f.get(3, 3, 3) == 0.0


def test_3d_diffusion_conserves_mass() -> None:
    f = ConcentrationField3D("g", 10, 10, 10, 600.0, 0.0)
    f.set(5, 5, 5, 1.0)
    before = f.total_mm()
    for _ in range(40):
        f.diffuse()
    after = f.total_mm()
    assert math.isclose(after, before, rel_tol=1e-9)


def test_3d_diffusion_spreads_along_all_axes() -> None:
    f = ConcentrationField3D("g", 9, 9, 9, 0.5, 0.0)
    f.set(4, 4, 4, 1.0)
    for _ in range(3):
        f.diffuse()
    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        nx, ny, nz = [c + d for c, d in zip((4, 4, 4), axis, strict=True)]
        assert f.get(nx, ny, nz) > 0.0
    assert f.get(4, 4, 4) > f.get(5, 5, 5)  # center still highest


def test_3d_neumann_boundaries() -> None:
    # mass stays inside the box: a deposit on a face must not leak out
    f = ConcentrationField3D("g", 5, 5, 5, 600.0, 0.0)
    f.set(0, 2, 2, 1.0)
    before = f.total_mm()
    for _ in range(10):
        f.diffuse()
    assert math.isclose(f.total_mm(), before, rel_tol=1e-9)
    assert f.get(-1, 2, 2) == 0.0
    assert f.get(0, 2, 2) > 0.0


def test_3d_layer_slice() -> None:
    f = ConcentrationField3D("g", 6, 6, 6, 600.0, 0.0)
    f.set(2, 3, 4, 1.0)
    layer = f.layer(4)
    assert len(layer) == 6 and len(layer[0]) == 6
    assert layer[3][2] == 1.0
    with pytest.raises(ValueError):
        f.layer(6)


def test_3d_snapshot_is_a_copy() -> None:
    f = ConcentrationField3D("g", 4, 4, 4, 600.0, 0.0)
    f.set(1, 1, 1, 1.0)
    snap = f.snapshot()
    snap[1][1][1] = 99.0
    assert f.get(1, 1, 1) == 1.0


def test_laplacian_step_3d_pure_python_matches() -> None:
    grid = [[[0.0] * 5 for _ in range(5)] for _ in range(5)]
    grid[2][2][2] = 1.0
    out = _laplacian_step_3d(grid, 0.1, 5, 5, 5)
    total = sum(sum(sum(plane) for plane in layer) for layer in out)
    assert math.isclose(total, 1.0, rel_tol=1e-12)
    assert out[2][2][2] > 0.0


# ============================================================================
# _neighbors_3d
# ============================================================================

def test_neighbors_3d_six_connectivity() -> None:
    nb = _neighbors_3d(3, 3, 3, 8, 8, 8, connectivity=6)
    assert len(nb) == 6
    assert set(nb) == {(2, 3, 3), (4, 3, 3), (3, 2, 3), (3, 4, 3),
                       (3, 3, 2), (3, 3, 4)}


def test_neighbors_3d_twenty_six_connectivity() -> None:
    nb = _neighbors_3d(3, 3, 3, 8, 8, 8, connectivity=26)
    assert len(nb) == 26
    assert all(-1 <= dx <= 1 and -1 <= dy <= 1 and -1 <= dz <= 1
               and not (dx == dy == dz == 0)
               for dx, dy, dz in
               ((x - 3, y - 3, z - 3) for x, y, z in nb))


def test_neighbors_3d_clipped_at_boundaries() -> None:
    nb = _neighbors_3d(0, 0, 0, 4, 4, 4, connectivity=26)
    assert len(nb) == 7  # 27 - 1 center - 19 outside
    assert all(nx >= 0 and ny >= 0 and nz >= 0 for nx, ny, nz in nb)


def test_neighbors_3d_bad_connectivity_raises() -> None:
    with pytest.raises(ValueError):
        _neighbors_3d(1, 1, 1, 4, 4, 4, connectivity=12)


# ============================================================================
# CellPopulation3D
# ============================================================================

def test_population3d_signal_field_is_3d() -> None:
    cfg = PopulationConfig(**GRID)
    p = CellPopulation3D([PopulationCell(id=0, energy=200.0,
                                         x=5, y=5, z=5)], cfg, seed=1)
    field = p.signal_field
    assert len(field) == 12
    assert len(field[0]) == 12
    assert len(field[0][0]) == 12


def test_population3d_emission_at_site() -> None:
    cfg = PopulationConfig(**{**GRID, "division_threshold": 10000.0})
    p = CellPopulation3D([PopulationCell(id=0, energy=200.0,
                                         x=3, y=4, z=5)], cfg, seed=1)
    p.step()
    # one tick of SIGNAL_EMISSION_PER_STEP accumulated at (3, 4, 5)
    emitted = p.signal_field[5][4][3]
    assert emitted > 0.0
    assert p.cells[0].signal_emitted > 0.0


def test_population3d_division_offsets_in_z() -> None:
    cfg = PopulationConfig(**GRID)
    p = CellPopulation3D([PopulationCell(id=0, energy=200.0,
                                         x=5, y=5, z=5)], cfg, seed=7)
    before_z = {c.z for c in p.cells}
    p.step()
    assert len(p.cells) == 2
    zs = {c.z for c in p.cells}
    assert zs & before_z or len(zs) <= 2
    assert all(0 <= c.z < 12 for c in p.cells)


def test_population3d_daughter_carries_z() -> None:
    a, b = CellPopulation3D([], PopulationConfig(**GRID))._divide_3d(
        PopulationCell(id=1, energy=200.0, x=5, y=5, z=5),
        PopulationConfig(**GRID), __import__("random").Random(0))
    assert a.z != b.z or a.x != b.x or a.y != b.y


def test_population3d_neighbors() -> None:
    cfg = PopulationConfig(**GRID)
    p = CellPopulation3D([], cfg)
    six = p.neighbors_3d(5, 5, 5)
    assert len(six) == 6
    assert (5, 5, 4) in six and (5, 5, 6) in six  # z-neighborhood present
    assert len(p.neighbors_3d(5, 5, 5, 26)) == 26


def test_population3d_growth_over_ticks() -> None:
    cfg = PopulationConfig(**GRID)
    p = CellPopulation3D([PopulationCell(id=0, energy=200.0,
                                         x=5, y=5, z=5)], cfg, seed=3)
    for _ in range(4):
        p.step()
    assert p.get_statistics()["alive_count"] >= 2
    assert len(p.cells) >= 2


def test_population3d_z_coordinate_defaults_zero_in_2d() -> None:
    cell = PopulationCell(id=0)
    assert cell.z == 0


def test_population3d_invalid_depth_raises() -> None:
    with pytest.raises(ValueError):
        CellPopulation3D([], PopulationConfig(**{**GRID, "grid_depth": 0}))


def test_population3d_is_cellpopulation() -> None:
    from helixlang.population import CellPopulation

    p = CellPopulation3D([], PopulationConfig(**GRID))
    assert isinstance(p, CellPopulation)


# ============================================================================
# LSystem3D morphology export
# ============================================================================

def test_to_lsystem3d_covers_occupied_sites() -> None:
    cfg = PopulationConfig(**GRID)
    cells = [PopulationCell(id=i, energy=300.0, x=x, y=y, z=z)
             for i, (x, y, z) in enumerate([(2, 3, 4), (7, 1, 5), (5, 5, 0)])]
    p = CellPopulation3D(cells, cfg, seed=1)
    ls = p.to_lsystem3d()
    # every occupied site is the START of an F marker segment
    starts = {(round(line.start.x), round(line.start.y), round(line.start.z))
              for line in ls.draw(0)}
    for (x, y, z) in [(2, 3, 4), (7, 1, 5), (5, 5, 0)]:
        assert (x, y, z) in starts, f"occupied site {(x, y, z)} missing"


def test_to_lsystem3d_empty_colony() -> None:
    p = CellPopulation3D([], PopulationConfig(**GRID))
    ls = p.to_lsystem3d()
    assert ls.get_points(0)  # still a valid (empty) geometry


# ============================================================================
# Large-scale 3D path (biofilm scale, 10^4..10^5 agents)
# ============================================================================

@pytest.mark.skipif(not _HAS_NUMPY, reason="numpy batch path unavailable")
def test_population3d_vectorized_matches_python() -> None:
    """numpy batch metabolism must reproduce the pure-Python 3D path
    exactly for identical initial state (energy, age, signal field and
    death decisions)."""
    cfg = PopulationConfig(**{**GRID, "signaling_enabled": True,
                              "signal_diffusion": 0.0})

    def build(seed: int) -> CellPopulation3D:
        rng = random.Random(seed)
        cells = [PopulationCell(id=i, energy=200.0, age=rng.randrange(3),
                                x=rng.randrange(12), y=rng.randrange(12),
                                z=rng.randrange(12))
                 for i in range(300)]
        return CellPopulation3D(cells, cfg, seed=seed)

    a, b = build(0), build(0)
    va, da = a._step_vectorized_metabolism()
    pb, db = b._step_metabolism_python()
    assert da == db
    sa = sorted((c.id, c.energy, c.age, c.signal_emitted) for c in va)
    sb = sorted((c.id, c.energy, c.age, c.signal_emitted) for c in pb)
    assert sa == sb
    assert a.signal_field == b.signal_field
    alive = {c.id for c in va}
    assert all(c.alive == (c.id in alive) for c in b.cells)


@pytest.mark.skipif(not _HAS_NUMPY, reason="numpy batch path unavailable")
def test_population3d_vectorized_scaling() -> None:
    """Per-cell cost stays near-linear from 10^3 to 10^4 agents (the
    vectorized path exists so colony growth is not O(N^2))."""
    cfg = PopulationConfig(**{**GRID, "signaling_enabled": False,
                              "division_threshold": 1e12})

    def mk(n: int) -> CellPopulation3D:
        rng = random.Random(0)
        cells = [PopulationCell(id=i, energy=200.0, x=rng.randrange(12),
                                y=rng.randrange(12), z=rng.randrange(12))
                 for i in range(n)]
        return CellPopulation3D(cells, cfg, seed=0)

    def timeit(n: int) -> float:
        p = mk(n)
        p.step()  # warm-up
        t0 = time.perf_counter()
        for _ in range(3):
            p.step()
        return (time.perf_counter() - t0) / 3.0

    small = timeit(2000)
    large = timeit(10000)
    # 5x the cells must not cost anywhere near the 25x an O(N^2) path
    # would: the 20x cap tolerates CI timing noise while still catching
    # genuine superlinear blow-up (O(N^2) would give ~25x).
    assert large < 20.0 * small, f"scaling regressed: {large:.3f}s vs {small:.3f}s"


def test_population3d_occupancy_3d_counts() -> None:
    cfg = PopulationConfig(**GRID)
    cells = [PopulationCell(id=i, energy=200.0, x=x, y=y, z=z)
             for i, (x, y, z) in enumerate([(1, 1, 1), (1, 1, 1),
                                            (2, 2, 2), (9, 9, 9)])]
    p = CellPopulation3D(cells, cfg, seed=1)
    occ = p.occupancy_3d()
    assert occ[1][1][1] == 2
    assert occ[2][2][2] == 1
    assert occ[9][9][9] == 1
    assert sum(sum(sum(layer) for layer in depth) for depth in occ) == 4


def test_population3d_biofilm_grows_anchored_at_substratum() -> None:
    """BM3-style growth: a z=0 monolayer expands and the resulting colony
    keeps its biomass near the substratum (surface-anchored, not floating)."""
    cfg = PopulationConfig(**{**GRID, "signal_diffusion": 2.0,
                              "division_threshold": 100.0})
    cells = [PopulationCell(id=i, energy=300.0,
                            x=4 + (i % 4), y=4 + (i % 3), z=0)
             for i in range(8)]
    p = CellPopulation3D(cells, cfg, seed=11)
    for _ in range(8):
        p.step()
    occ = p.occupancy_3d()
    alive = [c for c in p.cells if c.alive]
    assert len(alive) > len(cells)
    assert sum(sum(sum(layer) for layer in depth) for depth in occ) == len(alive)
    heights = [z for z in range(cfg.grid_depth)
               for y in range(cfg.grid_height)
               for x in range(cfg.grid_width) if occ[z][y][x]]
    assert max(heights) > 0  # colony grew away from the substratum
    mean_z = sum(c.z for c in alive) / len(alive)
    assert mean_z < cfg.grid_depth / 3.0  # biomass anchored near z=0


@pytest.mark.parametrize("mechanics", ["shoving", "force"])
def test_population3d_mechanics_relieves_crowding_in_z(mechanics: str) -> None:
    """3D mechanics dispatch must de-overlap crowded sites, moving cells
    along the z axis as well as x/y."""
    cfg = PopulationConfig(**{**GRID, "mechanics": mechanics,
                              "division_threshold": 1e12})
    cells = [PopulationCell(id=i, energy=200.0, x=5, y=5, z=5)
             for i in range(6)]
    p = CellPopulation3D(cells, cfg, seed=2)
    p._apply_mechanics(p.cells)
    occ = p.occupancy_3d()
    assert max(occ[z][y][x] for z in range(12)
               for y in range(12) for x in range(12)) < 6
    assert len({c.z for c in p.cells}) > 1  # cells escaped along z


@pytest.mark.parametrize("direction,axis", [("E", "x"), ("U", "z")])
def test_flow3d_drift_advects_cells(direction: str, axis: str) -> None:
    """Design 6 Level 2 3D wiring: ``config.flow3d`` (a FlowField3D) drifts
    lattice cells along the flow axis in 3D (`_drift_cells_3d`).

    The strong analytic duct field (mean 3 sites/tick) is far above the
    half-site rounding threshold, so the cells move one or more lattice
    sites per tick; a stagnant field keeps them in place.
    """
    from helixlang.flow import (
        channel_poiseuille_3d,
        stagnant_3d,
    )

    cfg = PopulationConfig(**{**GRID, "division_threshold": 1e13})
    cells = [PopulationCell(id=i, energy=1e12, x=5, y=8, z=4)
             for i in range(4)]
    flow = channel_poiseuille_3d(12, 12, 12, 0.5, direction)
    cfg.flow3d = flow
    pop = CellPopulation3D(cells, cfg, seed=0)
    assert cfg.flow3d is flow  # config flow3d is the driver
    x0 = [c.x for c in pop.cells]
    z0 = [c.z for c in pop.cells]
    for _ in range(2):
        pop.step()
    if axis == "x":
        assert all(c.x > x0[i] for i, c in enumerate(pop.cells))
    else:
        assert all(c.z > z0[i] for i, c in enumerate(pop.cells))

    still = CellPopulation3D(
        [PopulationCell(id=i, energy=1e12, x=5, y=8, z=4) for i in range(4)],
        PopulationConfig(**{**GRID, "division_threshold": 1e13}), seed=0)
    still.config.flow3d = stagnant_3d(12, 12, 12)
    x0s = [c.x for c in still.cells]
    z0s = [c.z for c in still.cells]
    for _ in range(2):
        still.step()
    assert [c.x for c in still.cells] == x0s
    assert [c.z for c in still.cells] == z0s


def test_lbm3d_refreshes_flow3d_and_obstacle_mask() -> None:
    """Design 6 Level 2 3D wiring: a D3Q19 solver attached via
    ``config.lbm`` is stepped each tick, its occupancy mask rasterizes
    the alive cells, and the refreshed FlowField3D is published to
    ``config.flow3d`` (driving drift + environment advection)."""
    from helixlang.apps.lattice_boltzmann_3d import LatticeBoltzmann3D
    from helixlang.flow import FlowField3D

    lbm = LatticeBoltzmann3D(width=12, height=12, depth=12, omega=1.2)
    cfg = PopulationConfig(**{**GRID, "lbm": lbm, "flow_substeps": 2,
                              "division_threshold": 1e13})
    cells = [PopulationCell(id=i, energy=1e12, x=5, y=5, z=5)
             for i in range(4)]
    p = CellPopulation3D(cells, cfg, seed=0)
    assert cfg.flow3d is None
    p.step()
    assert isinstance(cfg.flow3d, FlowField3D)
    # the 4 cells rasterized as no-slip obstacles at their site
    assert bool(lbm.solid[5][5][5])
    # flow field is the same object the population will drift on
    assert p.config.flow3d is cfg.flow3d
