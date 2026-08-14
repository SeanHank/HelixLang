"""Multicellular population simulation unit tests."""
import math
import random

import pytest

from helixlang.environment import (
    Environment,
    EnvironmentConfig,
)
from helixlang.population import (
    DIVISION_ENERGY_THRESHOLD,
    ENERGY_INTAKE_PER_STEP,
    METABOLIC_COST_PER_STEP,
    POPULATION_CELL_INITIAL_ENERGY,
    QUORUM_SIGNAL_THRESHOLD,
    SIGNAL_DIFFUSION_UM2_S,
    SIGNAL_EMISSION_PER_STEP,
    CellPopulation,
    PopulationCell,
    PopulationConfig,
    PopulationStatistics,
    cell_radius_um,
    divide_cell,
    quorum_sensing,
    signal_diffusion_step,
)


def _make_cell(id=0, energy=100.0, x=5, y=5, proteins=None,
               parent_id=None, alive=True, age=0):
    return PopulationCell(
        id=id, parent_id=parent_id, energy=energy, x=x, y=y,
        proteins=dict(proteins) if proteins else {},
        alive=alive, age=age,
    )


# -- population initialization --
def test_init_population():
    cells = [_make_cell(id=0), _make_cell(id=1, x=2, y=3)]
    pop = CellPopulation(cells, PopulationConfig(grid_width=10, grid_height=10))
    assert len(pop.cells) == 2
    assert pop.config.grid_width == 10
    assert pop._next_id == 2  # next available id
    # Signal field dimensions are correct
    field = pop.get_signal_field()
    assert len(field) == 10
    assert len(field[0]) == 10


def test_init_empty_population():
    pop = CellPopulation([], PopulationConfig(grid_width=4, grid_height=4))
    assert len(pop.cells) == 0
    assert pop._next_id == 0


# -- single-step advance: metabolism consumes energy --
def test_metabolism_consumes_energy():
    cfg = PopulationConfig(metabolic_cost=5.0, energy_intake=0.0,
                           division_threshold=1e9, signaling_enabled=False)
    cell = _make_cell(id=0, energy=100.0)
    pop = CellPopulation([cell], cfg)
    stats = pop.step()
    assert len(pop.cells) == 1
    assert pop.cells[0].energy == pytest.approx(95.0)
    assert pop.cells[0].age == 1
    assert stats["alive_count"] == 1


def test_metabolism_intake_adds_energy():
    cfg = PopulationConfig(metabolic_cost=1.0, energy_intake=10.0,
                           division_threshold=1e9, signaling_enabled=False)
    cell = _make_cell(id=0, energy=100.0)
    pop = CellPopulation([cell], cfg)
    pop.step()
    assert pop.cells[0].energy == pytest.approx(109.0)


# -- cell division --
def test_division_when_energy_above_threshold():
    cfg = PopulationConfig(division_threshold=200.0, metabolic_cost=0.0,
                           energy_intake=0.0, signaling_enabled=False)
    cell = _make_cell(id=0, energy=250.0)
    pop = CellPopulation([cell], cfg, seed=1)
    stats = pop.step()
    assert len(pop.cells) == 2
    for c in pop.cells:
        assert c.energy == pytest.approx(125.0)
    # Both daughter cells' parent_id point to the parent cell
    assert all(c.parent_id == 0 for c in pop.cells)
    assert stats["division_rate"] == pytest.approx(1.0)


def test_no_division_below_threshold():
    cfg = PopulationConfig(division_threshold=200.0, metabolic_cost=0.0,
                           energy_intake=0.0, signaling_enabled=False)
    cell = _make_cell(id=0, energy=150.0)
    pop = CellPopulation([cell], cfg)
    pop.step()
    assert len(pop.cells) == 1


# -- cell death --
def test_death_when_energy_depleted():
    cfg = PopulationConfig(metabolic_cost=50.0, energy_intake=0.0,
                           death_threshold=0.0, division_threshold=1e9,
                           signaling_enabled=False)
    cell = _make_cell(id=0, energy=10.0)
    pop = CellPopulation([cell], cfg)
    stats = pop.step()
    assert stats["alive_count"] == 0
    assert stats["dead_count"] == 1
    assert stats["death_rate"] == pytest.approx(1.0)
    assert len(pop.cells) == 0


# -- signal diffusion --
def test_signal_diffusion_spreads():
    field = [[0.0] * 5 for _ in range(5)]
    field[2][2] = 10.0
    new = signal_diffusion_step(field, 0.1)
    # Center drops, four neighbors rise
    assert new[2][2] < 10.0
    assert new[1][2] > 0.0
    assert new[3][2] > 0.0
    assert new[2][1] > 0.0
    assert new[2][3] > 0.0
    # Zero-flux boundaries -> total conserved
    assert sum(sum(row) for row in new) == pytest.approx(10.0, rel=1e-6)


def test_signal_diffusion_non_negative():
    field = [[10.0, 0.0], [0.0, 0.0]]
    new = signal_diffusion_step(field, 0.2)
    for row in new:
        for v in row:
            assert v >= 0.0


def test_signal_diffusion_empty_field():
    assert signal_diffusion_step([], 0.1) == []
    assert signal_diffusion_step([[], []], 0.1) == [[], []]


# -- quorum sensing --
def test_quorum_sensing_activated():
    cell = _make_cell(id=0)
    assert quorum_sensing(cell, 10.0, 5.0) is True
    assert cell.proteins["quorum"] == pytest.approx(1.0)


def test_quorum_sensing_not_activated():
    cell = _make_cell(id=0)
    assert quorum_sensing(cell, 1.0, 5.0) is False
    assert "quorum" not in cell.proteins


def test_quorum_sensing_in_population():
    """Cells activate their quorum gene once population signals accumulate to the threshold."""
    cfg = PopulationConfig(grid_width=5, grid_height=5,
                           signaling_enabled=True, signal_diffusion=0.0,
                           signal_threshold=3.0,
                           division_threshold=1e9,
                           metabolic_cost=0.0, energy_intake=0.0)
    # 5 cells share one grid cell, each emitting 1.0 signal per step -> accumulation exceeds threshold
    cells = [_make_cell(id=i, x=2, y=2, energy=100.0) for i in range(5)]
    pop = CellPopulation(cells, cfg)
    pop.step()
    activated = [c for c in pop.cells if c.proteins.get("quorum", 0.0) > 0.0]
    assert len(activated) > 0


# -- multi-generation evolution --
def test_evolve_population_grows():
    cfg = PopulationConfig(division_threshold=50.0, metabolic_cost=1.0,
                           energy_intake=20.0, signaling_enabled=False,
                           max_size=1000)
    pop = CellPopulation([_make_cell(id=0, energy=100.0)], cfg, seed=42)
    history = pop.evolve(30)
    assert len(history) == 30
    assert history[-1]["alive_count"] > 1
    assert history[-1]["generation"] == 30


def test_evolve_returns_stats_each_gen():
    cfg = PopulationConfig(signaling_enabled=False, division_threshold=1e9,
                           metabolic_cost=1.0, energy_intake=0.0)
    pop = CellPopulation([_make_cell(id=0, energy=100.0)], cfg)
    history = pop.evolve(3)
    assert len(history) == 3
    for stats in history:
        assert "alive_count" in stats
        assert "avg_energy" in stats


# -- population statistics --
def test_statistics_keys_and_values():
    cells = [_make_cell(id=0, energy=100.0, age=2),
             _make_cell(id=1, energy=200.0, age=4, x=1, y=1)]
    cfg = PopulationConfig(signaling_enabled=False, division_threshold=1e9,
                           metabolic_cost=0.0, energy_intake=0.0)
    pop = CellPopulation(cells, cfg)
    stats = pop.step()
    for key in ("population_size", "alive_count", "dead_count",
                "avg_energy", "max_energy", "min_energy", "avg_age",
                "division_rate", "death_rate", "diversity_index",
                "age_distribution"):
        assert key in stats
    assert stats["alive_count"] == 2
    assert stats["avg_energy"] == pytest.approx(150.0)
    assert stats["max_energy"] == pytest.approx(200.0)
    assert stats["min_energy"] == pytest.approx(100.0)
    # Each cell ages += 1 in step(): initial 2,4 -> 3,5 -> average 4.0
    assert stats["avg_age"] == pytest.approx(4.0)


def test_statistics_diversity_grows_after_division():
    cfg = PopulationConfig(division_threshold=50.0, metabolic_cost=0.0,
                           energy_intake=0.0, signaling_enabled=False,
                           max_size=1000)
    pop = CellPopulation([_make_cell(id=0, energy=100.0)], cfg, seed=3)
    stats0 = pop.get_statistics()
    # Single cell, single lineage -> diversity 0
    assert stats0["diversity_index"] == pytest.approx(0.0)
    pop.step()  # divide
    # After division both daughter cells have parent_id 0 -> still a single lineage
    # Lineage diversification after multiple generations
    pop.evolve(5)
    stats_n = pop.get_statistics()
    assert stats_n["alive_count"] > 1


def test_population_statistics_dataclass():
    stats = PopulationStatistics(
        population_size=10, alive_count=8, dead_count=2,
        avg_energy=100.0, max_energy=200.0, min_energy=50.0,
        avg_age=3.0, division_rate=0.1, death_rate=0.05,
        diversity_index=0.6,
    )
    assert stats.alive_count == 8
    assert stats.diversity_index == pytest.approx(0.6)


# -- spatial grid --
def test_grid_counts():
    cells = [_make_cell(id=0, x=2, y=3),
             _make_cell(id=1, x=2, y=3),
             _make_cell(id=2, x=4, y=5)]
    cfg = PopulationConfig(grid_width=10, grid_height=10,
                           signaling_enabled=False, division_threshold=1e9,
                           metabolic_cost=0.0, energy_intake=0.0)
    pop = CellPopulation(cells, cfg)
    pop.step()  # no division, no death
    grid = pop.get_grid()
    assert len(grid) == 10
    assert len(grid[0]) == 10
    assert grid[3][2] == 2  # 2 cells at (x=2, y=3)
    assert grid[5][4] == 1  # 1 cell at (x=4, y=5)


def test_grid_dimensions_match_config():
    cfg = PopulationConfig(grid_width=8, grid_height=6,
                           signaling_enabled=False)
    pop = CellPopulation([_make_cell(id=0)], cfg)
    grid = pop.get_grid()
    assert len(grid) == 6       # rows = grid_height
    assert len(grid[0]) == 8    # columns = grid_width


# -- maximum population limit --
def test_max_size_enforced():
    cfg = PopulationConfig(division_threshold=50.0, metabolic_cost=1.0,
                           energy_intake=20.0, signaling_enabled=False,
                           max_size=4)
    pop = CellPopulation([_make_cell(id=0, energy=100.0)], cfg, seed=99)
    pop.evolve(50)
    assert len(pop.cells) <= 4


def test_max_size_blocks_division_at_cap():
    cfg = PopulationConfig(division_threshold=1.0, metabolic_cost=0.0,
                           energy_intake=0.0, signaling_enabled=False,
                           max_size=2)
    # 2 cells already at the cap; with sufficient energy they still should not divide
    cells = [_make_cell(id=0, energy=100.0), _make_cell(id=1, energy=100.0)]
    pop = CellPopulation(cells, cfg)
    pop.step()
    assert len(pop.cells) == 2


# -- protein allocation after division --
def test_divide_cell_protein_allocation():
    cfg = PopulationConfig(grid_width=20, grid_height=20)
    parent = _make_cell(id=0, energy=200.0, proteins={"GFP": 100.0})
    rng = random.Random(123)
    a, b = divide_cell(parent, cfg, rng)
    # Energy split in half
    assert a.energy == pytest.approx(100.0)
    assert b.energy == pytest.approx(100.0)
    # Protein conservation
    total = a.proteins.get("GFP", 0.0) + b.proteins.get("GFP", 0.0)
    assert total == pytest.approx(100.0)
    # With a large amount, both daughter cells should get some GFP
    assert a.proteins.get("GFP", 0.0) > 0.0
    assert b.proteins.get("GFP", 0.0) > 0.0
    # parent_id points to the parent cell
    assert b.parent_id == 0
    # division_count incremented
    assert a.division_count == 1
    assert b.division_count == 1


def test_divide_cell_multiple_proteins():
    cfg = PopulationConfig(grid_width=20, grid_height=20)
    parent = _make_cell(id=0, energy=200.0,
                        proteins={"GFP": 50.0, "RFP": 30.0, "BFP": 20.0})
    rng = random.Random(7)
    a, b = divide_cell(parent, cfg, rng)
    for name in ("GFP", "RFP", "BFP"):
        total = a.proteins.get(name, 0.0) + b.proteins.get(name, 0.0)
        parent_amount = parent.proteins[name]
        assert total == pytest.approx(parent_amount)


def test_divide_cell_position_offset_in_bounds():
    cfg = PopulationConfig(grid_width=20, grid_height=20)
    parent = _make_cell(id=0, energy=200.0, x=10, y=10)
    rng = random.Random(7)
    a, b = divide_cell(parent, cfg, rng)
    assert 0 <= a.x < 20 and 0 <= a.y < 20
    assert 0 <= b.x < 20 and 0 <= b.y < 20


def test_divide_cell_position_clamped_at_edge():
    cfg = PopulationConfig(grid_width=10, grid_height=10)
    parent = _make_cell(id=0, energy=200.0, x=0, y=0)
    rng = random.Random(7)
    a, b = divide_cell(parent, cfg, rng)
    assert 0 <= a.x < 10 and 0 <= a.y < 10
    assert 0 <= b.x < 10 and 0 <= b.y < 10


def test_divide_cell_daughters_lineage():
    """Binary fission: the parent cell disappears; both new daughter cells' parent_id point to it."""
    cfg = PopulationConfig(grid_width=20, grid_height=20)
    parent = _make_cell(id=5, energy=200.0, parent_id=2)
    rng = random.Random(1)
    a, b = divide_cell(parent, cfg, rng)
    # Both daughter cells' parent_id point to the parent that divided
    assert a.parent_id == 5
    assert b.parent_id == 5
    # Both ids are pending assignment by Population
    assert a.id == -1
    assert b.id == -1
    # division_count incremented
    assert a.division_count == 1
    assert b.division_count == 1


def test_divide_cell_halves_volume():
    """Binary fission conserves volume: each daughter inherits half of
    the parent's physical volume (Phase 2, Taheri-Araghi 2015)."""
    cfg = PopulationConfig(grid_width=20, grid_height=20)
    parent = _make_cell(id=5, energy=200.0)
    parent.volume_um3 = 3.2
    rng = random.Random(1)
    a, b = divide_cell(parent, cfg, rng)
    assert a.volume_um3 == pytest.approx(1.6)
    assert b.volume_um3 == pytest.approx(1.6)
    assert a.volume_um3 + b.volume_um3 == pytest.approx(parent.volume_um3)
    assert a.volume_um3 == _make_cell().volume_um3  # default newborn 1.6


def test_cell_radius_matches_sphere_geometry():
    """Equivalent-sphere radius r = (3V/4π)^(1/3); a 1.6 µm^3 newborn is
    a ~0.73 µm sphere (Milo & Phillips 2015)."""
    r = cell_radius_um(1.6)
    assert r == pytest.approx((3.0 * 1.6 / (4.0 * 3.141592653589793))
                              ** (1.0 / 3.0))
    assert r == pytest.approx(0.725, abs=0.01)
    assert cell_radius_um(0.0) == 0.0


# ------------------------------------------------------------------ #
# Physical units (direct ATP + µM + µm²/s)
# ------------------------------------------------------------------ #
def test_config_physical_defaults():
    """PopulationConfig defaults are the physical values."""
    cfg = PopulationConfig()
    assert cfg.division_threshold == DIVISION_ENERGY_THRESHOLD == 1.8e9
    assert cfg.signal_diffusion == SIGNAL_DIFFUSION_UM2_S == 100.0
    assert cfg.signal_threshold == QUORUM_SIGNAL_THRESHOLD == 10.0
    assert cfg.metabolic_cost == METABOLIC_COST_PER_STEP == 1e7
    assert cfg.energy_intake == ENERGY_INTAKE_PER_STEP == 5e7


def test_default_cell_energy_is_atp_pool():
    """A newborn PopulationCell carries ~10^9 ATP molecules."""
    assert PopulationCell().energy == POPULATION_CELL_INITIAL_ENERGY == 1e9


def test_explicit_overrides_win():
    cfg = PopulationConfig(division_threshold=123.0, signal_diffusion=0.25)
    assert cfg.division_threshold == pytest.approx(123.0)
    assert cfg.signal_diffusion == pytest.approx(0.25)


def test_doubling_time_20_ticks():
    """Newborn cell (1e9) gains +4e7/tick net (5e7 intake - 1e7 cost);
    threshold 1.8e9 -> first division on tick 20."""
    cfg = PopulationConfig(grid_width=9, grid_height=9,
                           signaling_enabled=False)
    pop = CellPopulation([_make_cell(id=0, energy=1e9, x=2, y=2)], cfg, seed=3)
    for _ in range(19):
        pop.step()
    assert len(pop.cells) == 1          # 1e9 + 19*4e7 = 1.76e9 < 1.8e9
    pop.step()
    assert len(pop.cells) == 2          # 1e9 + 20*4e7 = 1.8e9 -> divides
    assert all(c.energy == pytest.approx(9e8) for c in pop.cells)


def test_cell_energy_is_atp_count_directly():
    """Cell energy counts ATP molecules, no calibration indirection."""
    cfg = PopulationConfig(grid_width=9, grid_height=9,
                           signaling_enabled=False, division_threshold=1e9,
                           metabolic_cost=0.0, energy_intake=0.0)
    pop = CellPopulation([_make_cell(id=0, energy=4.2e8, x=2, y=2)], cfg)
    pop.step()
    assert pop.cells[0].energy == pytest.approx(4.2e8)


def test_diffusion_gaussian_spread():
    """A point source diffuses to the analytical Gaussian: E[r^2] = 4Dt."""
    size = 101
    cfg = PopulationConfig(grid_width=size, grid_height=size)
    pop = CellPopulation([], cfg)
    field = [[0.0] * size for _ in range(size)]
    field[size // 2][size // 2] = 1000.0
    pop.signal_field = field
    new = pop._diffuse(cfg)
    mass = sum(sum(row) for row in new)
    assert mass == pytest.approx(1000.0, rel=1e-9)             # conserved
    cy = cx = size // 2
    var = sum(
        new[i][j] * ((i - cy) ** 2 + (j - cx) ** 2)
        for i in range(size) for j in range(size)
    ) / mass
    assert var == pytest.approx(4.0 * 60.0, rel=1e-3)          # 4Dt, D=60, t=1


def test_quorum_cluster_vs_isolate():
    """A 5-cell cluster emits 10 µM AI-2 and activates quorum; an isolated
    cell (2 µM) stays below the 10 µM threshold."""
    cfg = PopulationConfig(grid_width=9, grid_height=9,
                           signaling_enabled=True, signal_diffusion=0.0,
                           division_threshold=1e9,
                           metabolic_cost=0.0, energy_intake=0.0)
    cells = [_make_cell(id=i, x=2, y=2, energy=100.0) for i in range(5)]
    cells.append(_make_cell(id=9, x=8, y=8, energy=100.0))
    pop = CellPopulation(cells, cfg)
    pop.step()
    cluster = [c for c in pop.cells if (c.x, c.y) == (2, 2)]
    isolate = [c for c in pop.cells if (c.x, c.y) == (8, 8)]
    assert all(c.proteins.get("quorum", 0.0) > 0.0 for c in cluster)
    assert all(c.proteins.get("quorum", 0.0) == 0.0 for c in isolate)
    assert pop.signal_field[2][2] == pytest.approx(
        5 * SIGNAL_EMISSION_PER_STEP)   # 10 µM in physical units


# ============================================================================
# Phase 5: per-cell dynamic-FBA metabolism + colony observables
# ============================================================================


def _dfba_cells(n, x0=4, y0=4):
    return [PopulationCell(id=i, energy=1e5, x=x0 + (i % 3), y=y0 + (i // 3))
            for i in range(n)]


def _dfba_population(n, *, oxygen=0.25, glucose=10.0, diffusion=20.0,
                     ticks=6, width=12, height=12, x0=4, y0=4, **cfg_extra):
    env = Environment(EnvironmentConfig(
        width=width, height=height, glucose_initial_mm=glucose,
        oxygen_initial_mm=oxygen, glucose_diffusion_um2_s=diffusion,
        oxygen_diffusion_um2_s=diffusion))
    cfg = PopulationConfig(
        grid_width=width, grid_height=height, environment=env,
        dfba_enabled=True, division_threshold=1e9, **cfg_extra)
    pop = CellPopulation(_dfba_cells(n, x0, y0), cfg)
    for _ in range(ticks):
        pop.step()
    return pop


def test_dfba_enabled_requires_environment():
    """dfba_enabled without an environment is a config error."""
    cfg = PopulationConfig(grid_width=4, grid_height=4, dfba_enabled=True)
    with pytest.raises(ValueError):
        CellPopulation(_dfba_cells(1), cfg)


def test_dfba_cell_owns_batch_and_grows_energy():
    """An alive dFBA cell gains energy from its batch specific growth rate."""
    pop = _dfba_population(1)
    cell = pop.cells[0]
    assert cell.dfba is not None
    assert cell.dfba.history
    assert cell.dfba.growth_rate > 0.0
    assert cell.energy > 1e5
    # glucose was consumed from the local field
    assert pop.config.environment.glucose.get(cell.x, cell.y) < 10.0


def test_dfba_oxygen_capped_respiration():
    """Reducing-equivalent bounds are capped by the local O2: at low O2 the
    NADH_OX upper bound must drop well below the uncapped value."""
    from helixlang.population import monod_uptake
    env = Environment(EnvironmentConfig(width=12, height=12,
                                        oxygen_initial_mm=0.01))
    cfg = PopulationConfig(grid_width=12, grid_height=12, environment=env,
                           dfba_enabled=True, division_threshold=1e9)
    pop = CellPopulation(_dfba_cells(1), cfg)
    cell = pop.cells[0]
    pop.step()
    dfba = cell.dfba
    cap = 2.0 * monod_uptake(
        cfg.dfba_oxygen_max_uptake, 0.01, cfg.dfba_oxygen_half_saturation_mm)
    bound = dfba.fba.model.reactions["NADH_OX"].upper_bound
    assert bound == pytest.approx(min(bound, cap), rel=1e-6)


def test_dfba_anoxia_ferments_acetate():
    """O2-limited respiration redirects the LP to fermentative acetate
    overflow: anoxic runs deposit acetate at the colony."""
    pop = _dfba_population(9, oxygen=0.01, diffusion=5.0, ticks=10)
    env = pop.config.environment
    acetate = env.get_field("acetate")
    total = sum(sum(row) for row in acetate.concentration)
    assert total > 0.0
    strat = pop.dfba_stratification()
    assert strat["core_acetate_mm"] > 0.0
    # fermentation yields less ATP than respiration
    assert strat["core_growth_rate_h"] < 1.2


def test_dfba_stratification_core_edge_gradient():
    """With low diffusion the colony core runs low on glucose and O2 while
    the edge stays well supplied (Cockx 2024 stratification)."""
    pop = _dfba_population(25, diffusion=5.0, ticks=10,
                           width=20, height=20, x0=7, y0=7)
    strat = pop.dfba_stratification()
    assert strat["core_cell_count"] > 0
    assert strat["edge_cell_count"] > 0
    assert strat["core_glucose_mm"] < strat["edge_glucose_mm"]
    assert strat["core_oxygen_mm"] < strat["edge_oxygen_mm"]


def test_dfba_stratification_zeroed_without_dfba():
    """Without dFBA (or without an environment) the stratification summary
    is zeroed rather than crashing."""
    cfg = PopulationConfig(grid_width=8, grid_height=8,
                           division_threshold=1e9)
    pop = CellPopulation(_dfba_cells(6), cfg)
    for _ in range(3):
        pop.step()
    strat = pop.dfba_stratification()
    assert strat["core_cell_count"] == 0
    assert strat["core_glucose_mm"] == 0.0


def test_colony_observables_shape_and_growth():
    """colony_observables reports doubling times, a volume-weighted growth
    rate, a 10-band radial density profile and a colony radius."""
    pop = _dfba_population(9, ticks=8)
    obs = pop.colony_observables()
    assert len(obs["radial_density"]) == 10
    assert sum(obs["radial_density"]) == len([c for c in pop.cells if c.alive])
    assert obs["colony_radius_sites"] > 0.0
    assert obs["volume_weighted_growth_h"] > 0.0
    assert all(d > 0.0 for d in obs["doubling_times_h"])
    assert all(math.isfinite(v) for v in obs["birth_sizes_um3"])
