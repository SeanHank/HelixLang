"""Multicellular population frontier features: programmable cells
(per-cell GRN + bytecode), environment-coupled Monod metabolism,
CROMICS crowding diffusion, and spatial mechanics."""
import pytest

from helixlang.codon_table import STANDARD_TABLE
from helixlang.compiler import Compiler
from helixlang.environment import (
    Environment,
    EnvironmentConfig,
)
from helixlang.lexer import Lexer
from helixlang.parser import Parser
from helixlang.population import (
    CELL_VOLUME_FRACTION,
    SIGNAL_EMISSION_PER_STEP,
    CellPopulation,
    PopulationCell,
    PopulationConfig,
    divide_cell,
)
from helixlang.semantic import SemanticAnalyzer


def _compile(src: str):
    """Lex/parse/check/compile a .helix program -> (Program, Chunk)."""
    toks = list(Lexer(src).tokens())
    prog = Parser(toks).parse()
    SemanticAnalyzer(prog).check()
    return prog, Compiler(STANDARD_TABLE).compile(prog)


def _prog_config(src: str, **cfg) -> PopulationConfig:
    program, chunk = _compile(src)
    return PopulationConfig(program=program, chunk=chunk, **cfg)


def _cell(id=0, energy=1e9, x=2, y=2):
    return PopulationCell(id=id, energy=energy, x=x, y=y)


# ============================================================================
# Programmable cells (T1.1): per-cell GRN + bytecode
# ============================================================================
def test_program_cell_emits_signal_each_tick():
    """A constitutive signaler gene emits OP_SIGNAL every tick; the
    emission lands in the shared AI-2 field.  TCA (wobble A=0) emits
    SIGNAL_EMISSION_AMOUNT * (1 + 0) = 0.5 µM per execution."""
    cfg = _prog_config(
        "#gene name=sig\nATG TCA TAA\n#end\n#config ticks=1",
        signaling_enabled=False, signal_diffusion=0.0,
        division_threshold=1e9, metabolic_cost=0.0, energy_intake=0.0,
        grid_width=9, grid_height=9)
    pop = CellPopulation([_cell(energy=100.0)], cfg)
    pop.step()
    # metabolism emits nothing (signaling disabled); only the program does
    assert pop.signal_field[2][2] == pytest.approx(0.5)
    pop.step()
    assert pop.signal_field[2][2] == pytest.approx(1.0)


def test_program_cell_builds_protein():
    cfg = _prog_config(
        "#gene name=maker\nATG GCT TAA\n#end\n#config ticks=1",
        signaling_enabled=False, division_threshold=1e9,
        metabolic_cost=0.0, energy_intake=0.0)
    pop = CellPopulation([_cell()], cfg)
    pop.step()
    assert pop.cells[0].proteins.get(3, 0.0) == pytest.approx(1.0)


def test_program_cell_moves():
    cfg = _prog_config(
        "#gene name=mover\nATG GTG TAA\n#end\n#config ticks=1",
        signaling_enabled=False, division_threshold=1e9,
        metabolic_cost=0.0, energy_intake=0.0)
    pop = CellPopulation([_cell(x=2, y=2)], cfg)
    pop.step()
    # GTG = OP_MOVE arg=2 (South): y increases
    assert pop.cells[0].y == 3


def test_program_cell_feeds():
    cfg = _prog_config(
        "#gene name=feeder\nATG GAA TAA\n#end\n#config ticks=1",
        signaling_enabled=False, division_threshold=1e12,
        metabolic_cost=0.0, energy_intake=0.0)
    pop = CellPopulation([_cell(energy=1e9)], cfg)
    pop.step()
    assert pop.cells[0].energy == pytest.approx(1e9 + 1e8)


def test_program_cell_dies():
    cfg = _prog_config(
        "#gene name=killer\nATG AAA TAA\n#end\n#config ticks=1",
        signaling_enabled=False, division_threshold=1e9,
        metabolic_cost=0.0, energy_intake=0.0)
    pop = CellPopulation([_cell()], cfg)
    stats = pop.step()
    assert stats["dead_count"] == 1
    assert all(not c.alive for c in pop.cells)


def test_program_cell_build_pigment():
    cfg = _prog_config(
        "#gene name=pig\nATG TGG TAA\n#end\n#config ticks=1",
        signaling_enabled=False, division_threshold=1e9,
        metabolic_cost=0.0, energy_intake=0.0)
    pop = CellPopulation([_cell()], cfg)
    pop.step()
    assert pop.cells[0].color == (200, 50, 50)


def test_program_regulate_rewires_grn():
    """OP_REGULATE (CAT, wobble T -> mode 3) adds an edge from the
    executing gene to itself (only node)."""
    cfg = _prog_config(
        "#gene name=g\nATG CAT TAA\n#end\n#config ticks=1",
        signaling_enabled=False, division_threshold=1e9,
        metabolic_cost=0.0, energy_intake=0.0)
    pop = CellPopulation([_cell()], cfg)
    pop.step()
    grn = pop.cells[0].grn
    assert grn is not None
    assert any(e.source == "g" and e.target == "g" and e.weight == 1.0
               for e in grn.edges)


def test_each_cell_has_own_grn_copy():
    cfg = _prog_config(
        "#gene name=g\nATG TCT TAA\n#end\n#config ticks=1",
        signaling_enabled=False, division_threshold=1e9,
        metabolic_cost=0.0, energy_intake=0.0)
    pop = CellPopulation([_cell(id=0, x=1, y=1), _cell(id=1, x=4, y=4)], cfg)
    a, b = pop.cells
    assert a.grn is not None and b.grn is not None
    assert a.grn is not b.grn
    assert "g" in a.grn.nodes


def test_division_propagates_grn_to_daughters():
    cfg = _prog_config(
        "#gene name=g\nATG GCT TAA\n#end\n#config ticks=1",
        signaling_enabled=False, division_threshold=50.0,
        metabolic_cost=0.0, energy_intake=0.0, grid_width=9, grid_height=9)
    parent = PopulationCell(id=0, energy=200.0, x=4, y=4)
    pop = CellPopulation([parent], cfg)
    pop.step()  # divide
    daughters = [c for c in pop.cells if c.alive]
    assert len(daughters) == 2
    assert all(c.grn is not None for c in daughters)
    assert daughters[0].grn is not daughters[1].grn


def test_program_requires_chunk():
    program, _ = _compile("#gene name=g\nATG GCT TAA\n#end")
    with pytest.raises(ValueError, match="chunk"):
        CellPopulation([_cell()],
                       PopulationConfig(program=program, chunk=None))


def test_program_controlled_division_only_on_op_divide():
    """With program_controlled_division=True, an OP_DIVIDE gene divides
    while a control population (same energy) does not."""
    div_cfg = _prog_config(
        "#gene name=d\nATG AAT TAA\n#end\n#config ticks=1",
        signaling_enabled=False, division_threshold=100.0,
        metabolic_cost=0.0, energy_intake=0.0, grid_width=9, grid_height=9,
        program_controlled_division=True)
    pop_div = CellPopulation([_cell(id=0, energy=200.0)], div_cfg)
    pop_div.step()
    assert len(pop_div.cells) == 2

    no_cfg = _prog_config(
        "#gene name=g\nATG GCT TAA\n#end\n#config ticks=1",
        signaling_enabled=False, division_threshold=100.0,
        metabolic_cost=0.0, energy_intake=0.0, grid_width=9, grid_height=9,
        program_controlled_division=True)
    pop_no = CellPopulation([_cell(id=0, energy=200.0)], no_cfg)
    pop_no.step()
    assert len(pop_no.cells) == 1


def test_divide_cell_grn_isolation():
    import random
    cfg = PopulationConfig(grid_width=20, grid_height=20)
    program, chunk = _compile("#gene name=g\nATG GCT TAA\n#end")
    parent = _cell(id=0, energy=200.0)
    parent.grn = CellPopulation(
        [_cell()], PopulationConfig(program=program, chunk=chunk)).cells[0].grn
    a, b = divide_cell(parent, cfg, random.Random(1))
    assert a.grn is not None and b.grn is not None
    assert a.grn is not b.grn


# ============================================================================
# Environment-coupled Monod metabolism (T1.2)
# ============================================================================
def test_environment_starved_cell_gets_no_intake():
    """At zero local glucose the Monod factor is 0 -> no intake."""
    env = Environment(EnvironmentConfig(
        width=9, height=9, glucose_initial_mm=0.0, flow_rate=0.0))
    cfg = PopulationConfig(
        environment=env, grid_width=9, grid_height=9,
        signaling_enabled=False, division_threshold=1e9,
        metabolic_cost=1.0, energy_intake=100.0)
    pop = CellPopulation([_cell(id=0, energy=100.0)], cfg)
    pop.step()
    assert pop.cells[0].energy == pytest.approx(99.0)  # cost only


def test_environment_rich_cell_gets_monod_scaled_intake():
    """At S=1 mM, Ks=0.1 -> factor 1/(1+0.1) ~= 0.909."""
    env = Environment(EnvironmentConfig(
        width=9, height=9, glucose_initial_mm=1.0, flow_rate=0.0))
    cfg = PopulationConfig(
        environment=env, grid_width=9, grid_height=9,
        signaling_enabled=False, division_threshold=1e9,
        metabolic_cost=0.0, energy_intake=100.0,
        glucose_half_saturation_mm=0.1)
    pop = CellPopulation([_cell(id=0, energy=0.0)], cfg)
    pop.step()
    expected = 100.0 * 1.0 / (1.0 + 0.1)
    assert pop.cells[0].energy == pytest.approx(expected)


def test_environment_field_depleted_by_cells():
    # glucose diffusion disabled so the depletion is not washed out by
    # neighbor inflow; the field drops by exactly the Monod-scaled demand
    env = Environment(EnvironmentConfig(
        width=9, height=9, glucose_initial_mm=1.0, flow_rate=0.0,
        glucose_diffusion_um2_s=0.0))
    cfg = PopulationConfig(
        environment=env, grid_width=9, grid_height=9,
        signaling_enabled=False, division_threshold=1e9,
        metabolic_cost=0.0, energy_intake=0.0, max_glucose_uptake_mm=0.2)
    pop = CellPopulation([_cell(id=0, x=4, y=4)], cfg)
    before = env.glucose.get(4, 4)
    pop.step()
    after = env.glucose.get(4, 4)
    assert after < before
    assert after == pytest.approx(before - 0.2 * (1.0 / 1.1), rel=1e-6)


def test_environment_advances_each_tick():
    env = Environment(EnvironmentConfig(
        width=9, height=9, flow_rate=0.05))
    cfg = PopulationConfig(
        environment=env, grid_width=9, grid_height=9,
        signaling_enabled=False, division_threshold=1e9,
        metabolic_cost=0.0, energy_intake=0.0)
    pop = CellPopulation([_cell()], cfg)
    pop.step()
    pop.step()
    assert env.tick == 2


def test_environment_dimension_mismatch_raises():
    env = Environment(EnvironmentConfig(width=5, height=5))
    cfg = PopulationConfig(environment=env, grid_width=9, grid_height=9)
    with pytest.raises(ValueError, match="lattice"):
        CellPopulation([_cell()], cfg)


def test_environment_large_population_uses_python_path():
    """The numpy vectorized path is bypassed when an environment is set
    (all 150 cells keep their energy: no intake, no cost)."""
    env = Environment(EnvironmentConfig(
        width=20, height=20, glucose_initial_mm=0.0, flow_rate=0.0))
    cfg = PopulationConfig(
        environment=env, grid_width=20, grid_height=20,
        signaling_enabled=False, division_threshold=1e9,
        metabolic_cost=0.0, energy_intake=0.0)
    cells = [_cell(id=i, x=i % 20, y=i // 20, energy=100.0)
             for i in range(150)]
    pop = CellPopulation(cells, cfg)
    pop.step()
    assert len(pop.cells) == 150
    assert all(c.alive and c.energy == pytest.approx(100.0)
               for c in pop.cells)


# ============================================================================
# CROMICS crowding effective diffusion (T2.6)
# ============================================================================
def test_crowding_factor_per_cell_volume():
    assert CELL_VOLUME_FRACTION == pytest.approx(1.5e-3)


def test_crowding_volume_fractions_from_occupancy():
    cells = [_cell(id=i, x=2, y=2) for i in range(3)]
    cfg = PopulationConfig(grid_width=9, grid_height=9,
                           signaling_enabled=False, crowding_enabled=True)
    pop = CellPopulation(cells, cfg)
    fracs = pop.get_volume_fractions()
    assert fracs[2][2] == pytest.approx(3 * CELL_VOLUME_FRACTION)
    assert fracs[0][0] == pytest.approx(0.0)


def test_crowding_slows_diffusion_at_dense_sites():
    """The free-volume factor 1-phi cuts D near crowded sites: a point
    source at a dense site spreads less than the same source in empty
    medium."""
    src = "#gene name=g\nATG GCT TAA\n#end"
    program, chunk = _compile(src)
    n_cells = 600  # phi clamps to 0.999 -> factor ~ 0.001
    cfg = PopulationConfig(
        program=program, chunk=chunk, grid_width=9, grid_height=9,
        signaling_enabled=False, division_threshold=1e9,
        metabolic_cost=0.0, energy_intake=0.0, crowding_enabled=True)
    cells = [_cell(id=i, x=2, y=2) for i in range(n_cells)]
    pop = CellPopulation(cells, cfg)
    pop.signal_field[2][2] = 10.0
    crowded = pop._diffuse(pop.config)

    plain_cfg = PopulationConfig(grid_width=9, grid_height=9,
                                 signaling_enabled=False)
    pop_plain = CellPopulation([_cell(id=0, x=2, y=2)], plain_cfg)
    pop_plain.signal_field[2][2] = 10.0
    uncrowded = pop_plain._diffuse(plain_cfg)

    assert crowded[2][2] > uncrowded[2][2]
    # mass stays conserved in both
    assert sum(sum(r) for r in crowded) == pytest.approx(10.0, rel=1e-6)
    assert sum(sum(r) for r in uncrowded) == pytest.approx(10.0, rel=1e-6)


# ============================================================================
# Spatial mechanics (T1.3)
# ============================================================================
def test_shoving_separates_overlapping_cells():
    cells = [_cell(id=i, x=2, y=2) for i in range(3)]
    cfg = PopulationConfig(grid_width=9, grid_height=9,
                           signaling_enabled=False, mechanics="shoving")
    pop = CellPopulation(cells, cfg)
    pop._apply_mechanics(pop.cells)
    grid = pop.get_grid()
    assert all(v <= 1 for row in grid for v in row)
    assert sum(sum(row) for row in grid) == 3


def test_force_mechanics_reduces_peak_occupancy():
    cells = [_cell(id=i, x=2, y=2) for i in range(5)]
    cfg = PopulationConfig(grid_width=9, grid_height=9,
                           signaling_enabled=False, mechanics="force")
    pop = CellPopulation(cells, cfg)
    before = max(max(row) for row in pop.get_grid())
    pop._apply_mechanics(pop.cells)
    after = max(max(row) for row in pop.get_grid())
    assert after < before
    assert sum(sum(row) for row in pop.get_grid()) == 5


def test_mechanics_invalid_mode_raises():
    with pytest.raises(ValueError, match="mechanics"):
        CellPopulation([_cell()], PopulationConfig(mechanics="phasing"))


def test_mechanics_disabled_keeps_overlap():
    cells = [_cell(id=i, x=2, y=2) for i in range(3)]
    cfg = PopulationConfig(grid_width=9, grid_height=9, signaling_enabled=False)
    pop = CellPopulation(cells, cfg)
    pop._apply_mechanics(pop.cells)
    assert pop.get_grid()[2][2] == 3


# ============================================================================
# Streaming per-cell trace (T1.5)
# ============================================================================
def test_trace_streaming_appends_snapshots():
    cfg = _prog_config(
        "#gene name=g\nATG TCA TAA\n#end\n#config ticks=1",
        signaling_enabled=False, division_threshold=1e12,
        metabolic_cost=0.0, energy_intake=0.0,
        trace_streaming=True)
    pop = CellPopulation([_cell(id=0, energy=100.0)], cfg)
    pop.step()
    pop.step()
    assert len(pop.trace) == 2
    last = pop.trace[-1]
    assert last["tick"] == 2
    assert last["cells"][0]["id"] == 0
    assert last["cells"][0]["gene_levels"]["g"] > 0.5


def test_trace_disabled_by_default():
    cfg = PopulationConfig(signaling_enabled=False, division_threshold=1e9,
                           metabolic_cost=0.0, energy_intake=0.0)
    pop = CellPopulation([_cell()], cfg)
    pop.step()
    assert pop.trace == []


# ============================================================================
# Program + signaling combined: quorum-adaptive behavior
# ============================================================================
def test_program_plus_quorum_signal_field():
    """Program OP_SIGNAL adds to the same AI-2 field the metabolism
    emissions write to, and both are readable via get_signal_field."""
    cfg = _prog_config(
        "#gene name=sig\nATG TCA TAA\n#end\n#config ticks=1",
        signaling_enabled=True, signal_diffusion=0.0,
        division_threshold=1e9, metabolic_cost=0.0, energy_intake=0.0,
        signal_threshold=10.0)
    pop = CellPopulation([_cell(id=0, x=2, y=2)], cfg)
    pop.step()
    # metabolism (+2.0/tick) + one program emission (+0.5)
    assert pop.signal_field[2][2] == pytest.approx(
        SIGNAL_EMISSION_PER_STEP + 0.5)
