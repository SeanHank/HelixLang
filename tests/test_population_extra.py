"""Additional tests for population.py — free functions and CellPopulation methods.

Covers:
  - cell_radius_um, _clamp, _binomial, quorum_sensing
  - signal_diffusion_step, _crowded_laplacian_step
  - divide_cell
  - CellPopulation init / validation / step / evolve / get_grid / statistics
  - CellPopulation metabolism helpers
  - CellPopulation3D init / step / diffusion / mechanics
"""
from __future__ import annotations

import math
import random
from unittest.mock import MagicMock

import pytest

from helixlang.plugins.runtime.environment import (
    Environment,
    EnvironmentConfig,
)
from helixlang.plugins.runtime.population import (
    DIVISION_ENERGY_THRESHOLD,
    ENERGY_INTAKE_PER_STEP,
    METABOLIC_COST_PER_STEP,
    QUORUM_SIGNAL_THRESHOLD,
    CellPopulation,
    CellPopulation3D,
    PopulationCell,
    PopulationConfig,
    SpeciesParams,
    cell_radius_um,
    divide_cell,
    quorum_sensing,
    signal_diffusion_step,
)

# ─── free functions ─────────────────────────────────────────────────

class TestCellRadius:
    def test_positive_volume(self):
        r = cell_radius_um(1.6)
        expected = (3.0 * 1.6 / (4.0 * math.pi)) ** (1.0 / 3.0)
        assert r == pytest.approx(expected)

    def test_zero_volume(self):
        assert cell_radius_um(0.0) == 0.0

    def test_negative_volume(self):
        assert cell_radius_um(-5.0) == 0.0


class TestClamp:
    def test_below(self):
        from helixlang.plugins.runtime.population import _clamp
        assert _clamp(1, 5, 10) == 5

    def test_above(self):
        from helixlang.plugins.runtime.population import _clamp
        assert _clamp(15, 5, 10) == 10

    def test_in_range(self):
        from helixlang.plugins.runtime.population import _clamp
        assert _clamp(7, 5, 10) == 7


class TestBinomial:
    def test_zero(self):
        from helixlang.plugins.runtime.population import _binomial
        rng = random.Random(42)
        assert _binomial(rng, 0, 0.5) == 0

    def test_small_n(self):
        from helixlang.plugins.runtime.population import _binomial
        rng = random.Random(42)
        result = _binomial(rng, 10, 0.5)
        assert 0 <= result <= 10

    def test_large_n(self):
        from helixlang.plugins.runtime.population import _binomial
        rng = random.Random(42)
        result = _binomial(rng, 1000, 0.5)
        assert 0 <= result <= 1000
        assert abs(result - 500) < 200


class TestQuorumSensing:
    def test_active(self):
        cell = PopulationCell()
        result = quorum_sensing(cell, 15.0, QUORUM_SIGNAL_THRESHOLD)
        assert result is True
        assert cell.proteins.get("quorum") == 1.0

    def test_inactive(self):
        cell = PopulationCell()
        result = quorum_sensing(cell, 5.0, QUORUM_SIGNAL_THRESHOLD)
        assert result is False
        assert "quorum" not in cell.proteins

    def test_exact_threshold(self):
        cell = PopulationCell()
        result = quorum_sensing(cell, QUORUM_SIGNAL_THRESHOLD, QUORUM_SIGNAL_THRESHOLD)
        assert result is True


class TestSignalDiffusionStep:
    def test_empty(self):
        assert signal_diffusion_step([], 0.1) == []

    def test_single_cell(self):
        field = [[5.0]]
        result = signal_diffusion_step(field, 0.1)
        assert len(result) == 1
        assert result[0][0] == pytest.approx(5.0)

    def test_diffuses_inward(self):
        field = [
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        result = signal_diffusion_step(field, 0.25)
        assert result[1][1] < 10.0
        assert result[0][1] > 0.0

    def test_nonneg(self):
        field = [
            [0.0, 5.0, 0.0],
            [5.0, 10.0, 5.0],
            [0.0, 5.0, 0.0],
        ]
        result = signal_diffusion_step(field, 0.3)
        for row in result:
            for v in row:
                assert v >= 0.0


class TestCrowdedLaplacianStep:
    def test_uniform_factor(self):
        from helixlang.plugins.runtime.population import _crowded_laplacian_step
        grid = [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]]
        factors = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        result = _crowded_laplacian_step(grid, factors, 0.1, 3, 3)
        assert result[1][1] < 5.0
        assert result[0][1] > 0.0

    def test_zero_factor_blocks(self):
        from helixlang.plugins.runtime.population import _crowded_laplacian_step
        grid = [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]]
        factors = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        result = _crowded_laplacian_step(grid, factors, 0.1, 3, 3)
        assert result[1][1] == pytest.approx(5.0)

    def test_nonneg(self):
        from helixlang.plugins.runtime.population import _crowded_laplacian_step
        grid = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        factors = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        result = _crowded_laplacian_step(grid, factors, 0.1, 3, 3)
        for row in result:
            for v in row:
                assert v >= 0.0


# ─── divide_cell ────────────────────────────────────────────────────

class TestDivideCell:
    def test_basic(self):
        config = PopulationConfig(grid_width=10, grid_height=10)
        cell = PopulationCell(
            id=1, energy=DIVISION_ENERGY_THRESHOLD * 2,
            x=5, y=5, proteins={"lacZ": 10.0},
        )
        rng = random.Random(42)
        a, b = divide_cell(cell, config, rng)
        assert a.energy == cell.energy / 2.0
        assert b.energy == cell.energy / 2.0
        assert a.parent_id == 1
        assert b.parent_id == 1
        assert a.division_count == 1
        assert b.division_count == 1

    def test_protein_split(self):
        config = PopulationConfig(grid_width=10, grid_height=10)
        cell = PopulationCell(
            id=1, energy=2e9, x=5, y=5,
            proteins={"lacZ": 100.0, "GFP": 10.0},
        )
        a, b = divide_cell(cell, config, random.Random(7))
        total_a = a.proteins.get("lacZ", 0.0)
        total_b = b.proteins.get("lacZ", 0.0)
        assert total_a + total_b == pytest.approx(100.0, rel=0.01)

    def test_zero_protein_skipped(self):
        config = PopulationConfig(grid_width=10, grid_height=10)
        cell = PopulationCell(
            id=1, energy=2e9, x=5, y=5,
            proteins={"lacZ": 0.0},
        )
        a, b = divide_cell(cell, config, random.Random(42))
        assert "lacZ" not in a.proteins
        assert "lacZ" not in b.proteins

    def test_daughter_positions_in_bounds(self):
        config = PopulationConfig(grid_width=5, grid_height=5)
        cell = PopulationCell(id=1, energy=2e9, x=0, y=0)
        for seed in range(20):
            a, b = divide_cell(cell, config, random.Random(seed))
            assert 0 <= a.x < 5
            assert 0 <= a.y < 5
            assert 0 <= b.x < 5
            assert 0 <= b.y < 5

    def test_grn_deep_copy(self):
        from helixlang.plugins.runtime.grn import GRN
        grn = GRN()
        grn.add_gene("lacZ", threshold=-1.0, initial_level=1.0)
        config = PopulationConfig(grid_width=10, grid_height=10)
        cell = PopulationCell(id=1, energy=2e9, x=5, y=5, grn=grn)
        a, b = divide_cell(cell, config, random.Random(42))
        assert a.grn is not None
        assert b.grn is not None
        assert a.grn is not b.grn

    def test_volume_halved(self):
        config = PopulationConfig(grid_width=10, grid_height=10)
        cell = PopulationCell(id=1, energy=2e9, x=5, y=5, volume_um3=3.2)
        a, b = divide_cell(cell, config, random.Random(42))
        assert a.volume_um3 == pytest.approx(1.6)
        assert b.volume_um3 == pytest.approx(1.6)


# ─── CellPopulation init & validation ───────────────────────────────

class TestCellPopulationInit:
    def test_basic(self):
        cells = [PopulationCell(id=0, x=5, y=5)]
        pop = CellPopulation(cells)
        assert len(pop.cells) == 1
        assert pop._next_id == 1

    def test_empty(self):
        pop = CellPopulation([])
        assert len(pop.cells) == 0
        assert pop._next_id == 0

    def test_seed(self):
        pop = CellPopulation([PopulationCell(id=0)], seed=123)
        assert pop.rng.random() == random.Random(123).random()

    def test_signal_field_shape(self):
        pop = CellPopulation(
            [PopulationCell(id=0)],
            config=PopulationConfig(grid_width=5, grid_height=3),
        )
        assert len(pop.signal_field) == 3
        assert len(pop.signal_field[0]) == 5

    def test_mechanics_invalid(self):
        with pytest.raises(ValueError, match="mechanics"):
            CellPopulation(
                [PopulationCell(id=0)],
                config=PopulationConfig(mechanics="bogus"),
            )

    def test_cell_shape_invalid(self):
        with pytest.raises(ValueError, match="cell_shape"):
            CellPopulation(
                [PopulationCell(id=0)],
                config=PopulationConfig(cell_shape="sphere"),
            )

    def test_contact_requires_rod(self):
        with pytest.raises(ValueError, match="contact"):
            CellPopulation(
                [PopulationCell(id=0)],
                config=PopulationConfig(mechanics="contact"),
            )

    def test_dfba_requires_env(self):
        with pytest.raises(ValueError, match="dfba_enabled"):
            CellPopulation(
                [PopulationCell(id=0)],
                config=PopulationConfig(dfba_enabled=True),
            )

    def test_program_requires_chunk(self):
        from helixlang.api.ast import Program
        with pytest.raises(ValueError, match="chunk"):
            CellPopulation(
                [PopulationCell(id=0)],
                config=PopulationConfig(program=Program()),
            )

    def test_env_size_mismatch(self):
        env_cfg = EnvironmentConfig(width=20, height=20)
        env = Environment(env_cfg)
        with pytest.raises(ValueError, match="environment"):
            CellPopulation(
                [PopulationCell(id=0)],
                config=PopulationConfig(
                    grid_width=10, grid_height=10, environment=env),
            )


# ─── CellPopulation internal utilities ──────────────────────────────

class TestCellPopulationUtilities:
    def test_in_bounds(self):
        pop = CellPopulation(
            [PopulationCell(id=0)],
            config=PopulationConfig(grid_width=10, grid_height=10),
        )
        assert pop._in_bounds(0, 0) is True
        assert pop._in_bounds(9, 9) is True
        assert pop._in_bounds(10, 5) is False
        assert pop._in_bounds(-1, 5) is False

    def test_assign_ids(self):
        pop = CellPopulation(
            [PopulationCell(id=-1), PopulationCell(id=-1)],
        )
        pop._assign_ids()
        assert pop.cells[0].id == 0
        assert pop.cells[1].id == 1

    def test_build_program_grn(self):
        from helixlang.core.ast_nodes import Gene, Program, Promoter, Regulation
        prog = Program(
            promoters=[Promoter(name="p1", strength=-1.0)],
            genes=[Gene(name="g1", promoter="p1", codons=[], orf=[])],
            regulations=[Regulation(source="g1", target="g2", strength=1.0)],
        )
        grn = CellPopulation._build_program_grn(prog)
        assert "p1" in grn.nodes
        assert "g1" in grn.nodes

    def test_build_program_grn_no_promoter(self):
        from helixlang.core.ast_nodes import Gene, Program
        prog = Program(genes=[Gene(name="g1", promoter=None, codons=[], orf=[])])
        grn = CellPopulation._build_program_grn(prog)
        assert "g1" in grn.nodes


# ─── CellPopulation step & evolve ───────────────────────────────────

class TestCellPopulationStep:
    def test_step_basic(self):
        cells = [PopulationCell(id=0, x=5, y=5)]
        pop = CellPopulation(cells)
        stats = pop.step()
        assert isinstance(stats, dict)

    def test_step_ages_cells(self):
        cells = [PopulationCell(id=0, x=5, y=5)]
        pop = CellPopulation(cells)
        pop.step()
        assert pop.cells[0].age == 1

    def test_step_energy_intake(self):
        cells = [PopulationCell(id=0, x=5, y=5)]
        pop = CellPopulation(cells)
        old_energy = pop.cells[0].energy
        pop.step()
        assert pop.cells[0].energy > old_energy - METABOLIC_COST_PER_STEP

    def test_division(self):
        config = PopulationConfig(
            grid_width=10, grid_height=10,
            division_threshold=1e9, death_threshold=0.0,
            energy_intake=1e10, metabolic_cost=0.0,
        )
        cells = [PopulationCell(id=0, x=5, y=5, energy=2e9)]
        pop = CellPopulation(cells, config=config, seed=42)
        pop.step()
        assert len(pop.cells) >= 1

    def test_no_signaling(self):
        config = PopulationConfig(signaling_enabled=False)
        cells = [PopulationCell(id=0, x=5, y=5)]
        pop = CellPopulation(cells, config=config)
        pop.step()
        assert all(v == 0.0 for row in pop.signal_field for v in row)

    def test_evolve(self):
        cells = [PopulationCell(id=0, x=5, y=5)]
        pop = CellPopulation(cells)
        pop.evolve(generations=3)
        assert pop._generation == 3
        assert len(pop.cells) >= 1


# ─── CellPopulation queries ─────────────────────────────────────────

class TestCellPopulationQueries:
    def test_get_grid(self):
        cells = [PopulationCell(id=0, x=2, y=3)]
        pop = CellPopulation(
            cells,
            config=PopulationConfig(grid_width=10, grid_height=10),
        )
        grid = pop.get_grid()
        assert len(grid) == 10
        assert grid[3][2] == 1

    def test_get_signal_field(self):
        pop = CellPopulation([PopulationCell(id=0)])
        pop.signal_field[5][5] = 10.0
        sf = pop.get_signal_field()
        assert sf[5][5] == 10.0

    def test_species_counts(self):
        cells = [
            PopulationCell(id=0, species="A"),
            PopulationCell(id=1, species="A"),
            PopulationCell(id=2, species="B"),
        ]
        pop = CellPopulation(cells)
        counts = pop.species_counts()
        assert counts["A"] == 2
        assert counts["B"] == 1

    def test_get_statistics(self):
        cells = [PopulationCell(id=0, x=5, y=5)]
        pop = CellPopulation(cells)
        stats = pop.get_statistics()
        assert isinstance(stats, dict)
        assert stats["population_size"] == 1
        assert stats["alive_count"] == 1


# ─── CellPopulation with environment ────────────────────────────────

class TestCellPopulationEnvironment:
    def _make_pop_with_env(self):
        env_cfg = EnvironmentConfig(width=10, height=10)
        env = Environment(env_cfg)
        cells = [PopulationCell(id=0, x=5, y=5)]
        config = PopulationConfig(
            grid_width=10, grid_height=10, environment=env,
        )
        return CellPopulation(cells, config=config)

    def test_step_with_env(self):
        pop = self._make_pop_with_env()
        stats = pop.step()
        assert isinstance(stats, dict)

    def test_monod_uptake(self):
        pop = self._make_pop_with_env()
        pop.step()
        assert pop.cells[0].age == 1


# ─── CellPopulation trace ───────────────────────────────────────────

class TestCellPopulationTrace:
    def test_trace_streaming(self):
        config = PopulationConfig(trace_streaming=True)
        cells = [PopulationCell(id=0, x=5, y=5)]
        pop = CellPopulation(cells, config=config)
        pop.step()
        assert len(pop.trace) > 0
        assert "tick" in pop.trace[0]


# ─── SpeciesParams ──────────────────────────────────────────────────

class TestSpeciesParams:
    def test_defaults(self):
        sp = SpeciesParams()
        assert sp.energy_intake == ENERGY_INTAKE_PER_STEP
        assert sp.metabolic_cost == METABOLIC_COST_PER_STEP
        assert sp.division_threshold == DIVISION_ENERGY_THRESHOLD

    def test_custom(self):
        sp = SpeciesParams(energy_intake=1e8, metabolic_cost=1e6)
        assert sp.energy_intake == 1e8


# ─── CellPopulation3D ───────────────────────────────────────────────

class TestCellPopulation3DInit:
    def test_basic(self):
        pop = CellPopulation3D([PopulationCell(id=0, x=5, y=5, z=0)])
        assert len(pop.cells) == 1

    def test_grid_depth(self):
        pop = CellPopulation3D(
            [PopulationCell(id=0)],
            config=PopulationConfig(grid_depth=5),
        )
        assert pop.config.grid_depth == 5

    def test_step(self):
        pop = CellPopulation3D(
            [PopulationCell(id=0, x=5, y=5, z=0)],
            config=PopulationConfig(grid_width=10, grid_height=10),
        )
        stats = pop.step()
        assert isinstance(stats, dict)


# ─── dFBA integration (basic) ──────────────────────────────────────

class TestCellPopulationDFBA:
    def test_dfba_step(self):
        env_cfg = EnvironmentConfig(width=10, height=10)
        env = Environment(env_cfg)
        cells = [PopulationCell(id=0, x=5, y=5)]
        config = PopulationConfig(
            grid_width=10, grid_height=10,
            environment=env, dfba_enabled=True,
            acetate_switch=True,
        )
        pop = CellPopulation(cells, config=config)
        stats = pop.step()
        assert isinstance(stats, dict)

    def test_shared_batch(self):
        env_cfg = EnvironmentConfig(width=10, height=10)
        env = Environment(env_cfg)
        cells = [
            PopulationCell(id=0, x=5, y=5),
            PopulationCell(id=1, x=5, y=5),
        ]
        config = PopulationConfig(
            grid_width=10, grid_height=10,
            environment=env, dfba_enabled=True,
            dfba_shared_batch=True,
        )
        pop = CellPopulation(cells, config=config)
        stats = pop.step()
        assert isinstance(stats, dict)


# ─── Colony observables ─────────────────────────────────────────────

class TestColonyObservables:
    def test_basic(self):
        cells = [PopulationCell(id=0, x=5, y=5)]
        pop = CellPopulation(cells)
        obs = pop.colony_observables()
        assert isinstance(obs, dict)

    def test_empty(self):
        pop = CellPopulation([])
        obs = pop.colony_observables()
        assert isinstance(obs, dict)


# ─── DFBA stratification ────────────────────────────────────────────

class TestDFBAStratification:
    def test_basic(self):
        env_cfg = EnvironmentConfig(width=10, height=10)
        env = Environment(env_cfg)
        cell = PopulationCell(id=0, x=5, y=5)
        cell.dfba = MagicMock()
        cell.dfba.biomass_gdw = 0.1
        config = PopulationConfig(
            grid_width=10, grid_height=10,
            environment=env, dfba_enabled=True,
        )
        pop = CellPopulation([cell], config=config)
        strat = pop.dfba_stratification()
        assert isinstance(strat, dict)
