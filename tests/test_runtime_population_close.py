"""Closure tests for helixlang.plugins.runtime.population.

Targets every branch of ``population.py`` that the main population/evolution
test files do not reach:

- ``_HAS_NUMPY=False`` pure-Python fallbacks (via a controlled reload)
- ``_binomial`` large-n normal-approximation clamps
- rod ``divide_cell`` / contact mechanics paths
- LBM / flow-drift dispatch inside ``step()``
- the suspended per-cell bytecode VM (``_execute_cell``) with hand-built chunks
- genome-scale colony mode (matrix steps, genome-row lifecycle,
  expression -> dFBA reaction gating)
- per-cell and shared-batch dFBA edge arcs (dead/out-of-bounds cells,
  acetate deposition)
- numpy vectorized metabolism edge arcs (out-of-bounds scatter, empty)
- 3D population edge arcs (OOB emission, empty occupancy, depth-1 fission,
  contact dispatch, 2D-flow drift, LBM raster skip)
"""
from __future__ import annotations

import importlib
import math
import random
import sys
from types import SimpleNamespace

import pytest

import helixlang.plugins.runtime.population as popmod
from helixlang.api.bytecode import (
    REGULATE_EDGE_WEIGHT,
    SIGNAL_EMISSION_AMOUNT,
)
from helixlang.core.bytecode import Chunk
from helixlang.core.codon_table import STANDARD_TABLE, Op
from helixlang.core.compiler import Compiler
from helixlang.core.errors import UnknownOpcodeError
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.core.semantic import SemanticAnalyzer
from helixlang.plugins.apps.genome_scale import build_genome
from helixlang.plugins.apps.lattice_boltzmann import LatticeBoltzmann
from helixlang.plugins.apps.lattice_boltzmann_3d import LatticeBoltzmann3D
from helixlang.plugins.runtime.cell import (
    FEED_ENERGY_AMOUNT,
    MOVE_ENERGY_COST,
)
from helixlang.plugins.runtime.cell_body import CellBody
from helixlang.plugins.runtime.environment import (
    ACETATE_DIFFUSION_UM2_S,
    ConcentrationField,
    Environment,
    EnvironmentConfig,
)
from helixlang.plugins.runtime.flow import FlowField, stagnant
from helixlang.plugins.runtime.grn import GRN
from helixlang.plugins.runtime.population import (
    CellPopulation,
    CellPopulation3D,
    Population,
    PopulationCell,
    PopulationConfig,
    _binomial,
    divide_cell,
)

# ============================================================================
# helpers
# ============================================================================


def _compile(src: str):
    toks = list(Lexer(src).tokens())
    prog = Parser(toks).parse()
    SemanticAnalyzer(prog).check()
    return prog, Compiler(STANDARD_TABLE).compile(prog)


def _cell(id=0, energy=1e9, x=2, y=2, z=0, alive=True, **kw):
    return PopulationCell(id=id, energy=energy, x=x, y=y, z=z,
                          alive=alive, **kw)


def _chunk_pop(code, constants=(), gene_offsets=None, n_cells=0):
    chunk = Chunk(code=bytearray(code), constants=list(constants),
                  gene_offsets=dict(gene_offsets or {}))
    cfg = PopulationConfig(chunk=chunk, grid_width=8, grid_height=8,
                           signaling_enabled=False)
    return CellPopulation([_cell(id=i) for i in range(n_cells)], cfg)


def _run_vm(cell, code, constants=(), quota=64, grn=None):
    """Execute ``code`` once directly against a cell's suspended VM state."""
    pop = _chunk_pop(code, constants)
    cell.vm_frames = [0]
    cell.vm_ip = 0
    cell.grn = grn
    pop._execute_cell(cell, quota)
    return cell


# ============================================================================
# numpy-free pure-Python fallbacks (61-62, 513-527, 568-590, 656-658, 687,
# 2054-2075, 2234-2237, 2405-2408)
# ============================================================================
def test_numpy_absent_pure_python_fallbacks(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)
    importlib.reload(popmod)
    try:
        assert popmod._HAS_NUMPY is False

        field = [[1.0, 2.0], [3.0, 4.0]]
        new = popmod.signal_diffusion_step(field, 0.5)
        assert [len(r) for r in new] == [2, 2]
        assert all(v >= 0.0 for row in new for v in row)
        assert popmod.signal_diffusion_step([], 0.5) == []
        assert popmod.signal_diffusion_step([[]], 0.5) == [[]]
        # negative diffused value clamps to 0 (525)
        spike = popmod.signal_diffusion_step([[100.0, 0.0],
                                              [0.0, 0.0]], 1.0)
        assert spike[0][0] == 0.0

        grid = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        factors = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.5]]
        crowded = popmod._crowded_laplacian_step(grid, factors, 0.1, 3, 3)
        assert [len(r) for r in crowded] == [3, 3, 3]

        vol = [[[0.0, 1.0], [2.0, 0.0]], [[1.0, 0.0], [0.0, 3.0]]]
        three = popmod._laplacian_step_3d(vol, 0.1, 2, 2, 2)
        assert len(three) == 2 and len(three[0]) == 2

        pop = popmod.CellPopulation3D(
            [_cell(x=1, y=1, z=0)], popmod.PopulationConfig(
                grid_width=4, grid_height=4, grid_depth=3))
        occ = pop._occupancy_3d(pop.cells)
        assert occ[0][1][1] == 1
        pop.config.signal_diffusion = 0.0
        assert pop._diffuse(pop.config) is pop.signal_field  # D<=0 path
        pop.config.signal_diffusion = 5.0
        diff = pop._diffuse(pop.config)  # pure-Python 3D loop (2405-2408)
        assert len(diff) == 3

        with pytest.raises(ValueError, match="numpy"):
            popmod.CellPopulation([], popmod.PopulationConfig(lbm=object()))
        with pytest.raises(ValueError, match="numpy"):
            popmod.CellPopulation(
                [], popmod.PopulationConfig(genome=object()))
    finally:
        monkeypatch.undo()
        importlib.reload(popmod)
        assert popmod._HAS_NUMPY is True


# ============================================================================
# _binomial normal-approximation clamps (385, 387)
# ============================================================================
def test_binomial_normal_approx_clamps_edges():
    class _Lo:
        def gauss(self, mu, sigma):
            return -10 ** 6

    class _Hi:
        def gauss(self, mu, sigma):
            return 10 ** 6

    assert _binomial(_Lo(), 100, 0.5) == 0
    assert _binomial(_Hi(), 100, 0.5) == 100
    assert 0 <= _binomial(random.Random(4), 1000, 0.0001) <= 1000


# ============================================================================
# rod divide_cell (410-415) + contact mechanics (1348-1376)
# ============================================================================
def test_divide_cell_rod_branch():
    cfg = PopulationConfig(grid_width=40, grid_height=40,
                           cell_shape="rod", cell_length_um=2.0,
                           cell_diameter_um=1.0)
    parent = _cell(id=0, x=20, y=20, energy=1e9)
    parent.body = CellBody(x=205.0, y=205.0, length_um=8.0,
                           diameter_um=1.0, angle=0.0)
    a, b = divide_cell(parent, cfg, random.Random(3))
    assert a.body is not None and b.body is not None
    assert a.id == b.id == -1
    assert a.parent_id == b.parent_id == 0
    assert 0 <= a.x < 40 and 0 <= b.x < 40


def test_contact_mechanics_2d():
    cfg = PopulationConfig(
        grid_width=24, grid_height=24,
        cell_shape="rod", cell_length_um=2.0, cell_diameter_um=1.0,
        mechanics="contact", division_threshold=1e9,
        energy_intake=1.0, contact_stiffness=2.0e3,
        flow=stagnant(24, 24))
    cells = [_cell(id=0, x=5, y=12, energy=1e9)]
    cells[0].body = CellBody(x=55.0, y=125.0, length_um=4.0,
                             diameter_um=1.0, angle=0.0)
    pop = CellPopulation(cells, cfg)
    x0 = pop.cells[0].body.x
    pop._apply_contact_mechanics(pop.cells)
    pop._apply_mechanics(pop.cells)
    assert pop.cells[0].body.x >= x0
    assert pop.cells[0].x == 6  # body re-projected to the lattice

    # non-rod pop whose *config* is flipped to contact after construction
    plain = CellPopulation([_cell()], PopulationConfig(grid_width=8,
                                                       grid_height=8))
    plain.config.mechanics = "contact"
    plain._apply_contact_mechanics(plain.cells)
    plain._apply_mechanics(plain.cells)


def test_contact_mechanics_no_bodies_returns():
    cfg = PopulationConfig(
        grid_width=24, grid_height=24,
        cell_shape="rod", cell_length_um=2.0, cell_diameter_um=1.0,
        mechanics="contact", division_threshold=1e9)
    pop = CellPopulation([_cell(id=0, alive=False)], cfg)
    pop._apply_contact_mechanics(pop.cells)


# ============================================================================
# step() dispatch: LBM (852), flow drift (854), mechanics (856)
# ============================================================================
def test_step_lbm_2d_rasterizes_cells():
    lbm = LatticeBoltzmann(width=13, height=13, omega=1.0)
    cfg = PopulationConfig(grid_width=13, grid_height=13, lbm=lbm,
                           flow_substeps=2, division_threshold=1e13,
                           signaling_enabled=False)
    point = _cell(id=0, x=5, y=5, energy=1e12)
    rod = _cell(id=1, x=7, y=5, energy=1e12)
    rod.body = CellBody(x=75.0, y=55.0, length_um=4.0, diameter_um=1.0)
    pop = CellPopulation([point, rod], cfg)
    pop.step()
    assert isinstance(cfg.flow, FlowField)
    assert bool(lbm.solid[5][5])
    assert bool(lbm.solid[5][7])


def test_step_flow_drift_lattice_cells():
    u1 = [[1.0] * 8 for _ in range(8)]
    zeros = [[0.0] * 8 for _ in range(8)]
    flow = FlowField(8, 8, u1, zeros)
    cfg = PopulationConfig(grid_width=8, grid_height=8, flow=flow,
                           mechanics="shoving", division_threshold=1e13,
                           signaling_enabled=False)
    pop = CellPopulation([_cell(id=0, x=0, y=0, energy=1e12)], cfg)
    pop.step()
    assert pop.cells[0].x == 1
    assert pop.cells[0].y == 0


def test_step_flow_drift_rod_body_path():
    flow = stagnant(8, 8)
    flow.u[2][5] = 1.0
    cfg = PopulationConfig(grid_width=8, grid_height=8, flow=flow,
                           cell_shape="rod", cell_length_um=2.0,
                           cell_diameter_um=1.0,
                           division_threshold=1e13,
                           signaling_enabled=False)
    cell = _cell(id=0, x=5, y=2, energy=1e12)
    cell.body = CellBody(x=50.0, y=25.0, length_um=2.0, diameter_um=1.0)
    pop = CellPopulation([cell], cfg)
    pop.step()
    assert pop.cells[0].body.x == pytest.approx(60.0)
    assert pop.cells[0].x == 6


def test_mechanics_shoving_no_empty_neighbor():
    cfg = PopulationConfig(grid_width=1, grid_height=1,
                           mechanics="shoving", signaling_enabled=False)
    pop = CellPopulation([_cell(id=i, x=0, y=0) for i in range(3)], cfg)
    pop._apply_mechanics(pop.cells)
    assert pop.get_grid()[0][0] == 3  # no free neighbor to shove into


def test_mechanics_force_rejects_make_forced():
    cfg = PopulationConfig(grid_width=1, grid_height=1,
                           mechanics="force", signaling_enabled=False)
    pop = CellPopulation([_cell(id=i, x=0, y=0) for i in range(3)], cfg)
    pop._apply_mechanics(pop.cells)
    assert pop.get_grid()[0][0] == 3  # no better neighbor exists


def test_mechanics_shoves_overlapping_into_free_site():
    cfg = PopulationConfig(grid_width=3, grid_height=3,
                           mechanics="shoving", signaling_enabled=False)
    pop = CellPopulation([_cell(id=i, x=1, y=1) for i in range(2)], cfg)
    pop._apply_mechanics(pop.cells)
    grid = pop.get_grid()
    assert grid[1][1] <= 1
    assert sum(sum(r) for r in grid) == 2


def test_mechanics_skips_dead_and_oob_cells():
    cfg = PopulationConfig(grid_width=3, grid_height=3,
                           mechanics="shoving", signaling_enabled=False)
    cells = [_cell(id=0, x=1, y=1),
             _cell(id=1, x=1, y=1, alive=False),
             _cell(id=2, x=-1, y=1)]
    pop = CellPopulation(cells, cfg)
    pop._apply_mechanics(pop.cells)
    assert pop.get_grid()[1][1] == 1


# ============================================================================
# _read_cell_u8 / _read_cell_u16 operands (767, 774-782)
# ============================================================================
def test_read_u8_empty_or_past_end():
    pop = _chunk_pop([])
    assert pop._read_cell_u8(_cell()) == 0

    pop = _chunk_pop([Op.OP_NOP])
    cell = _cell(vm_ip=5)
    assert pop._read_cell_u8(cell) == 0


def test_read_u16_no_chunk_or_truncated():
    pop = CellPopulation([_cell()], PopulationConfig(grid_width=8,
                                                     grid_height=8))
    assert pop._read_cell_u16(_cell()) == 0

    pop = _chunk_pop([Op.OP_JUMP, 0x30])  # 1 trailing byte, needs 2
    cell = _cell(vm_ip=1)
    assert pop._read_cell_u16(cell) == 0
    assert cell.vm_ip == 2


def test_read_u8_in_bounds_advances_ip():
    pop = _chunk_pop([Op.OP_NOP, 0xAB, Op.OP_NOP])
    cell = _cell(vm_ip=1)
    assert pop._read_cell_u8(cell) == 0xAB
    assert cell.vm_ip == 2


# ============================================================================
# suspended bytecode VM: _execute_cell (1004-1199)
# ============================================================================
def test_execute_empty_frames_is_noop():
    pop = _chunk_pop([Op.OP_NOP])
    cell = _cell(vm_ip=0, vm_frames=[])
    pop._execute_cell(cell, 10)
    assert cell.vm_ip == 0


def test_execute_no_chunk_returns():
    pop = CellPopulation([_cell()], PopulationConfig(grid_width=8,
                                                     grid_height=8))
    cell = _cell(vm_ip=0, vm_frames=[0])
    pop._execute_cell(cell, 10)
    assert cell.vm_ip == 0


def test_execute_ip_past_end_pops_frame():
    pop = _chunk_pop([Op.OP_NOP])
    cell = _cell(vm_ip=9, vm_frames=[0])
    pop._execute_cell(cell, 10)
    assert cell.vm_frames == []

    cell2 = _cell(vm_ip=9, vm_frames=[4, 2])
    pop._execute_cell(cell2, 10)
    assert cell2.vm_frames == [4]
    assert cell2.vm_ip == 4  # resumes the caller's frame


def test_execute_frames_over_256_cleared():
    pop = _chunk_pop([Op.OP_NOP])
    cell = _cell(vm_ip=0, vm_frames=[1] * 300)
    pop._execute_cell(cell, 100)
    assert cell.vm_frames == []


def test_execute_unknown_opcode_raises():
    pop = _chunk_pop([0xEE])
    cell = _cell(vm_ip=0, vm_frames=[0])
    with pytest.raises(UnknownOpcodeError):
        pop._execute_cell(cell, 10)


def test_execute_start_nop_tick_pass():
    cell = _run_vm(_cell(), [Op.OP_START, Op.OP_NOP, Op.OP_TICK])
    assert cell.vm_ip > 0


def test_execute_halt_with_and_without_frames():
    cell = _cell()
    cell.vm_frames = [2]
    cell.vm_ip = 0
    pop = _chunk_pop([Op.OP_HALT, Op.OP_NOP])
    pop._execute_cell(cell, 2)
    # first HALT pops the return frame (back to ip 2, past end); the
    # second HALT (empty frames) falls through to the ip>=len exit
    assert cell.current_gene is None
    assert cell.vm_frames == []


def test_execute_push_const_and_out_of_range():
    cell = _run_vm(_cell(), [Op.OP_PUSH_CONST, 0x00], constants=[7.5])
    assert cell.vm_stack[-1] == 7.5

    cell2 = _run_vm(_cell(), [Op.OP_PUSH_CONST], constants=[])
    assert cell2.vm_stack[-1] == 0  # idx past the constant pool


def test_execute_stack_ops_guards():
    cell = _run_vm(_cell(), [Op.OP_POP, Op.OP_DUP, Op.OP_SWAP,
                             Op.OP_ADD, Op.OP_SUB, Op.OP_MUL, Op.OP_LT,
                             Op.OP_NOT])
    assert cell.vm_stack == []

    cell2 = _run_vm(_cell(), [Op.OP_PUSH_CONST, 0x00,
                              Op.OP_PUSH_CONST, 0x01,
                              Op.OP_DUP, Op.OP_SWAP,
                              Op.OP_ADD, Op.OP_SUB, Op.OP_MUL, Op.OP_LT,
                              Op.OP_NOT],
                    constants=[3.0, 2.0])
    assert cell2.vm_stack == [0]  # 3+2=5, 5-2=3, 3*2 needs 2 -> skipped,
                                  # LT skipped, NOT(-1) -> 0


def test_execute_build_protein_membrane_pigment():
    cell = _run_vm(_cell(), [Op.OP_BUILD_PROTEIN, 0x03])
    assert cell.proteins.get(3) == 1.0

    cell = _run_vm(_cell(), [Op.OP_BUILD_MEMBRANE, 0x64])
    assert cell.membrane_permeability == 100

    cell = _run_vm(_cell(), [Op.OP_BUILD_MEMBRANE])  # truncated operand
    assert cell.membrane_permeability == 0

    cell = _run_vm(_cell(), [Op.OP_BUILD_PIGMENT])
    assert cell.color == (200, 50, 50)


def test_execute_move_in_and_out_of_bounds():
    cell = _run_vm(_cell(id=0, x=0, y=0, energy=1e9),
                   [Op.OP_MOVE, 0x00])  # d=0 -> north (dy=-1)
    assert cell.y == 0  # 0-1=-1 clamped out: west/north moved failed
    assert cell.energy <= 1e9 + 1e-6

    cell = _run_vm(_cell(id=0, x=2, y=2, energy=1e9),
                   [Op.OP_MOVE, 0x02])  # d=2 -> south (dy=+1)
    assert cell.y == 3
    assert cell.energy == pytest.approx(1e9 - MOVE_ENERGY_COST)


def test_execute_signal_in_and_out_of_bounds():
    pop = _chunk_pop([Op.OP_SIGNAL, 0x01])
    inb = _cell(id=0, x=2, y=2)
    inb.vm_frames = [0]
    pop._execute_cell(inb, 10)
    assert pop.signal_field[2][2] == pytest.approx(
        min(1.0, SIGNAL_EMISSION_AMOUNT * 2))
    assert inb.signal_emitted == pytest.approx(SIGNAL_EMISSION_AMOUNT * 2)

    pop2 = _chunk_pop([Op.OP_SIGNAL, 0x00])
    oob = _cell(id=0, x=-1, y=2)
    oob.vm_frames = [0]
    pop2._execute_cell(oob, 10)
    assert pop2.signal_field[2][0] == 0.0
    assert oob.signal_emitted == pytest.approx(SIGNAL_EMISSION_AMOUNT)


def test_execute_divide_die_feed():
    cell = _run_vm(_cell(id=0, energy=1e9, flag_divide=False),
                   [Op.OP_DIVIDE, 0x00])
    assert cell.flag_divide is True

    cell = _run_vm(_cell(id=0, alive=True), [Op.OP_DIE, 0x00])
    assert cell.alive is False

    cell = _run_vm(_cell(id=0, energy=1e9), [Op.OP_FEED, 0x00])
    assert cell.energy == pytest.approx(1e9 + FEED_ENERGY_AMOUNT)

    cell = _run_vm(_cell(id=0, energy=1e9, membrane_permeability=0),
                   [Op.OP_FEED, 0x00])
    assert cell.energy == pytest.approx(1e9)


def test_execute_morphology_reads_operand():
    for op in (Op.OP_GROW_LSYSTEM, Op.OP_DIFFUSE, Op.OP_REACT,
               Op.OP_EMIT_MORPHOGEN):
        cell = _run_vm(_cell(), [op, 0x05])
        assert cell.vm_ip > 0


def test_execute_read_write_mem():
    code = [Op.OP_READ_MEM, 0x03,            # push slots[3] (0.0)
            Op.OP_PUSH_CONST, 0x00,
            Op.OP_WRITE_MEM, 0x03,           # slots[3] = 42.0
            Op.OP_WRITE_MEM, 0x04,           # empty stack -> skipped
            Op.OP_READ_MEM, 0x04]
    cell = _run_vm(_cell(), code, constants=[42.0])
    assert cell.slots[3] == 42.0
    assert cell.slots[4] == 0.0
    assert cell.vm_stack[-1] == 0.0


def test_execute_modify_state_flags():
    cell = _run_vm(_cell(), [Op.OP_MODIFY_STATE, 0x00])
    assert cell.color == (100, 200, 50)
    cell = _run_vm(_cell(age=0), [Op.OP_MODIFY_STATE, 0x01])
    assert cell.age == 1
    cell = _run_vm(_cell(), [Op.OP_MODIFY_STATE, 0x02])
    assert cell.color == (200, 200, 50)
    cell = _run_vm(_cell(), [Op.OP_MODIFY_STATE, 0x03])
    assert cell.color == (200, 50, 200)


def test_execute_regulate_no_grn_and_edges():
    grn = GRN()
    grn.add_gene("g0", 0.5)
    grn.add_gene("g1", 0.5)
    grn.add_edge("g0", "g1", 0.3)

    cell = _cell(current_gene="g0")
    cell.grn = grn
    # mode 0x03: +REGULATE_EDGE_WEIGHT, target names[3 % 2] = names[1]
    cell = _run_vm(cell, [Op.OP_REGULATE, 0x03], grn=grn)
    edge = next(e for e in grn.edges if e.source == "g0")
    assert edge.weight == REGULATE_EDGE_WEIGHT
    # existing edge updated in place (same object)
    cell = _run_vm(cell, [Op.OP_REGULATE, 0x03], grn=grn)
    assert edge.weight == REGULATE_EDGE_WEIGHT

    _run_vm(cell, [Op.OP_REGULATE, 0x83], grn=grn)  # negative
    e2 = next(e for e in grn.edges if e.source == "g0")
    assert e2.weight == -REGULATE_EDGE_WEIGHT

    plain = _run_vm(_cell(), [Op.OP_REGULATE, 0x00], grn=None)
    assert plain is not None


def test_execute_bind_branches():
    grn = GRN()
    grn.add_gene("g0", 0.5)
    grn.add_gene("g1", 0.5)

    # binder = current gene, a known protein
    cell = _cell(current_gene="tfA", proteins={"tfA": 0.5}, grn=grn)
    _run_vm(cell, [Op.OP_BIND, 0x00], grn=grn)
    assert cell.proteins["tfA"] == pytest.approx(0.0)

    # binder not a protein -> fall back to first protein
    grn2 = GRN()
    grn2.add_gene("g0", 0.5)
    cell2 = _cell(current_gene="x", proteins={"tfB": 2.0}, grn=grn2)
    _run_vm(cell2, [Op.OP_BIND, 0x00], grn=grn2)
    assert cell2.proteins["tfB"] == pytest.approx(1.0)

    # no proteins -> no consumption
    grn3 = GRN()
    grn3.add_gene("g0", 0.5)
    cell3 = _cell(current_gene="tfC", proteins={}, grn=grn3)
    _run_vm(cell3, [Op.OP_BIND, 0x00], grn=grn3)
    assert cell3.proteins == {}

    # zero-availability protein -> consumed==0 -> skipped (no set_level)
    grn4 = GRN()
    grn4.add_gene("g0", 0.5)
    cell4 = _cell(current_gene="tfD", proteins={"tfD": 0.0}, grn=grn4)
    lvl0 = grn4.nodes["g0"].level
    _run_vm(cell4, [Op.OP_BIND, 0x00], grn=grn4)
    assert grn4.nodes["g0"].level == lvl0


def test_execute_call_and_jumps():
    # OP_CALL_GENE: relative offset jump pushes a return ip onto the frame
    # stack; running off the end of the chunk resumes the caller frame.
    cell = _run_vm(_cell(), [Op.OP_CALL_GENE, 0x00, 0x06,
                             Op.OP_NOP, Op.OP_NOP, Op.OP_NOP,
                             Op.OP_NOP, Op.OP_NOP])
    assert cell.current_gene == "<call>"
    assert cell.vm_frames == [0]
    assert cell.vm_ip == 0

    # OP_JUMP: relative forward
    cell = _run_vm(_cell(), [Op.OP_JUMP, 0x00, 0x04, Op.OP_NOP,
                             Op.OP_NOP, Op.OP_HALT])
    assert cell.vm_ip > 0

    # OP_JUMP_IF_ZERO with zero on stack -> taken
    cell = _run_vm(_cell(), [Op.OP_PUSH_CONST, 0x00,
                             Op.OP_JUMP_IF_ZERO, 0x00, 0x04,
                             Op.OP_NOP, Op.OP_HALT, Op.OP_NOP],
                   constants=[0.0])
    assert cell.vm_ip > 0

    # OP_JUMP_IF_ZERO with nonzero / empty stack -> not taken
    cell = _run_vm(_cell(), [Op.OP_PUSH_CONST, 0x00,
                             Op.OP_JUMP_IF_ZERO, 0x00, 0x04,
                             Op.OP_HALT, Op.OP_NOP],
                   constants=[5.0])
    assert cell.vm_stack == []  # popped, jump not taken
    cell = _run_vm(_cell(), [Op.OP_JUMP_IF_ZERO, 0x00, 0x01, Op.OP_NOP])
    assert cell.vm_ip > 0


def test_execute_default_operand_bytes():
    cell = _run_vm(_cell(), [Op.OP_USE_PLUGIN, 0x00])
    assert cell.vm_ip == 2  # 1-byte operand consumed by the fallback


def test_execute_quota_exhaustion_preserves_state():
    cell = _cell()
    cell.vm_frames = [0]
    pop = _chunk_pop([Op.OP_NOP, Op.OP_NOP, Op.OP_NOP])
    pop._execute_cell(cell, 1)
    assert cell.vm_frames == [0]  # frame kept for the next tick


# ============================================================================
# _push_gene_frame guards (994-1002)
# ============================================================================
def test_push_gene_frame_guards():
    pop = CellPopulation([_cell()], PopulationConfig(grid_width=8,
                                                     grid_height=8))
    pop._push_gene_frame(_cell(), "g")  # chunk is None

    pop = _chunk_pop([Op.OP_NOP], gene_offsets={"other": 0})
    pop._push_gene_frame(_cell(), "g")  # not in gene_offsets

    pop = _chunk_pop([Op.OP_NOP], gene_offsets={"g": 3})
    full = _cell(vm_frames=[1] * 256, vm_ip=0)
    pop._push_gene_frame(full, "g")
    assert full.vm_ip == 0  # frames already at the ceiling

    cell = _cell(vm_frames=[1], vm_ip=4)
    pop._push_gene_frame(cell, "g")
    assert cell.current_gene == "g"
    assert cell.vm_frames == [1, 4]
    assert cell.vm_ip == 3


# ============================================================================
# genome-scale colony mode (655-676, 722-723, 730-733, 961-983, 1531-1551,
# 1578, 1898-1900)
# ============================================================================
def _genome_config(spec, program_src=None, **over):
    program, chunk = _compile(program_src or "#gene name=b0001\nATG TCA TAA\n#end\n#config ticks=1")
    cfg = dict(
        genome=spec, program=program, chunk=chunk,
        grid_width=10, grid_height=10, max_size=20,
        division_threshold=1e9, death_threshold=0.0,
        signaling_enabled=False,
    )
    cfg.update(over)
    return PopulationConfig(**cfg)


def test_genome_init_happy_and_validation():
    spec = build_genome(n_genes=8, seed=3)
    pop = CellPopulation([_cell(id=0), _cell(id=1)],
                         _genome_config(spec))
    assert pop._genome_colony is not None
    assert [c.genome_row for c in pop.cells] == [0, 1]
    assert pop._genome_overlap == {"b0001": 0}

    # chunk required
    cfg = PopulationConfig(genome=spec, grid_width=10, grid_height=10,
                           max_size=20)
    with pytest.raises(ValueError, match="compiled"):
        CellPopulation([_cell()], cfg)

    # max_size must fit the initial population
    program, chunk = _compile("#gene name=b0001\nATG TCA TAA\n#end\n#config ticks=1")
    cfg = PopulationConfig(genome=spec, program=program, chunk=chunk,
                           grid_width=10, grid_height=10, max_size=1)
    with pytest.raises(ValueError, match="max_size"):
        CellPopulation([_cell(id=0), _cell(id=1)], cfg)


def test_genome_step_steps_colony_and_writes_overlap():
    spec = build_genome(n_genes=8, seed=3)
    pop = CellPopulation([_cell(id=0), _cell(id=1)], _genome_config(spec))
    for c in pop.cells:
        c.grn.nodes["b0001"].level = 1.0
    stats = pop.step()
    assert "triggered_genes" in stats
    assert stats["genome_genes"] == spec.n_genes
    # the constitutive engineered gene wrote its overlap column
    col = pop._genome_overlap["b0001"]
    assert pop._genome_colony.levels[0, col] == 1.0
    assert pop._genome_colony.levels[1, col] == 1.0


def test_genome_assign_and_free_rows():
    spec = build_genome(n_genes=8, seed=3)
    cells = [_cell(id=0, energy=1e9, division_count=0),
             _cell(id=1, energy=0.0, alive=True)]
    pop = CellPopulation(
        cells, _genome_config(spec, division_threshold=100.0,
                              energy_intake=0.0, metabolic_cost=0.0))
    pop.step()
    # cell 1 died (its row pending free); cell 0 divided into fresh rows
    assert sorted(c.genome_row for c in pop.cells) == [0, 2]
    assert all(c.alive for c in pop.cells)


def test_genome_dfba_override_gates_silent_genes():
    spec = build_genome(n_genes=8, seed=5)
    pop = CellPopulation([_cell(id=0)], _genome_config(spec))
    pop._genome_colony.knock_out(["ptsG"])  # silent gating gene

    class _Rxn:
        lower_bound = -10.0

    class _Model:
        reactions = {"GLCpts": _Rxn()}

    class _FBA:
        model = _Model()

    batch = SimpleNamespace(fba=_FBA())
    ovr = pop._genome_dfba_override(0)
    off = ovr(0.0, batch)
    assert off["GLCpts"] == 0.0  # silent gene pinned to the lower bound

    # out-of-range row -> empty override
    assert pop._genome_dfba_override(999)(0.0, batch) == {}

    # per-cell dfba gets the override wired in genome mode
    dfba = pop._new_cell_dfba(0.05, genome_row=0)
    assert dfba.bound_override is not None
    dfba_no_gate = pop._new_cell_dfba(0.05, genome_row=-1)
    assert dfba_no_gate.bound_override is None


# ============================================================================
# per-cell dFBA edge arcs (1492, 1496->1509, 1513-1515)
# ============================================================================
def test_dfba_metabolism_skips_dead_and_oob_cells():
    env = Environment(EnvironmentConfig(width=10, height=10,
                                        glucose_initial_mm=5.0))
    cells = [
        _cell(id=0, x=5, y=5, alive=False),          # dead skipped
        _cell(id=1, x=-1, y=5, energy=1e9),          # OOB, no growth
        _cell(id=2, x=9, y=-1, energy=0.0),           # OOB starves -> dies
    ]
    cfg = PopulationConfig(grid_width=10, grid_height=10, environment=env,
                           dfba_enabled=True, dfba_shared_batch=False,
                           death_threshold=0.0, division_threshold=1e13)
    pop = CellPopulation(cells, cfg)
    stats = pop.step()
    assert stats["dead_count"] == 1  # id1 survives, id2 starves
    assert [c.id for c in pop.cells] == [1]


# ============================================================================
# shared-batch dFBA edge arcs (1601, 1603-1605, 1623-1666)
# ============================================================================
class _FakeDFBA:
    def __init__(self, acetate=0.0, growth=0.0):
        self.glucose_mm = 10.0
        self.byproducts_mm = {"acetate": acetate}
        self.growth_rate = growth
        self.last_fluxes = {"NADH_OX": 0.0, "FADH2_OX": 0.0}
        self.history = []
        self.biomass_gdw = 0.1
        from types import SimpleNamespace as _NS

        class _Rxn:
            upper_bound = 100.0

        self.fba = _NS(model=_NS(reactions={"NADH_OX": _Rxn(),
                                            "FADH2_OX": _Rxn()}))

    def update_from_environment(self, env, x, y):
        self.glucose_mm = env.glucose.get(x, y)

    def step(self, dt_h):
        self.glucose_mm = max(0.0, self.glucose_mm - 2.0)
        self.byproducts_mm = {"acetate": 3.0}

    def step_from_solution(self, sol, shared_S, dt_h):
        self.glucose_mm = max(0.0, shared_S - 4.0)
        self.byproducts_mm = {"acetate": 2.0}

    def apply_to_environment(self, env, x, y):
        pool = self.byproducts_mm.get("acetate", 0.0)
        if pool > 0.0:
            if "acetate" not in env.fields:
                env.add_field(
                    "acetate", ConcentrationField(
                        "acetate", env.config.width, env.config.height,
                        ACETATE_DIFFUSION_UM2_S, 0.0))
            env.get_field("acetate").set(
                x, y, env.get_field("acetate").get(x, y) + pool)
            self.byproducts_mm = {"acetate": 0.0}

    def set_state(self, acetate_mm=0.0):
        pass


def _fake_dfba_pop(acetate=0.0, growth=0.0, energies=(0.0, 100.0),
                   acetate_field=False, acetate_base=3.0):
    env = Environment(EnvironmentConfig(width=10, height=10))
    if acetate_field:
        env.add_field("acetate", ConcentrationField(
            "acetate", 10, 10, ACETATE_DIFFUSION_UM2_S, acetate_base))
    cells = [_cell(id=i, x=5, y=5, energy=e, alive=True)
             for i, e in enumerate(energies)]
    cells.append(_cell(id=99, x=5, y=5, alive=False))    # dead skipped
    cells.append(_cell(id=100, x=-1, y=5, energy=1e9))   # out of bounds
    cfg = PopulationConfig(
        grid_width=10, grid_height=10, environment=env,
        dfba_enabled=True, dfba_shared_batch=True,
        acetate_switch=True, division_threshold=1e13,
        death_threshold=0.0,
    )
    pop = CellPopulation(cells, cfg)
    pop._new_cell_dfba = lambda dt_h, genome_row=-1: _FakeDFBA(acetate,
                                                               growth)
    return pop, env


def test_shared_batch_deposits_acetate_field():
    pop, env = _fake_dfba_pop(acetate=3.0, growth=0.0)
    metabolized, deaths = pop._step_dfba_metabolism()
    assert "acetate" in env.fields
    assert env.fields["acetate"].get(5, 5) == pytest.approx(5.0)
    assert deaths == 1  # the energy-0 cell starved
    assert [c.id for c in metabolized] == [1]


def test_shared_batch_consumes_existing_acetate():
    pop, env = _fake_dfba_pop(acetate=0.0, growth=0.0,
                              acetate_field=True, acetate_base=12.0)
    pop._step_dfba_metabolism()
    # site_acetate (3+2) < base (12) -> net delta written by depletion
    assert env.fields["acetate"].get(5, 5) == pytest.approx(5.0)


# ============================================================================
# _sync_acetate / oxygen cap+deplete helpers (1684-1695, 1717-1719, 1728-1733)
# ============================================================================
def test_sync_acetate_branches():
    env = Environment(EnvironmentConfig(width=10, height=10))
    fake = _FakeDFBA()

    fake.byproducts_mm = {"acetate": 8.0}
    pop = CellPopulation([_cell()], PopulationConfig(grid_width=10,
                                                     grid_height=10,
                                                     environment=env))
    pop._sync_acetate(fake, env, 5, 5)
    assert env.get_field("acetate").get(5, 5) == pytest.approx(8.0)
    assert fake.byproducts_mm["acetate"] == 0.0

    # switch on, pool below site base -> deplete
    valv = Environment(EnvironmentConfig(width=10, height=10))
    valv.add_field("acetate", ConcentrationField(
        "acetate", 10, 10, ACETATE_DIFFUSION_UM2_S, 3.0))
    cfg = PopulationConfig(grid_width=10, grid_height=10, environment=valv,
                           acetate_switch=True)
    pop = CellPopulation([_cell()], cfg)
    fake2 = _FakeDFBA()
    fake2.byproducts_mm = {"acetate": 1.0}
    pop._sync_acetate(fake2, valv, 5, 5)
    assert valv.get_field("acetate").get(5, 5) == pytest.approx(1.0)

    # switch on, pool above site base -> set
    fake3 = _FakeDFBA()
    fake3.byproducts_mm = {"acetate": 5.0}
    pop._sync_acetate(fake3, valv, 5, 5)
    assert valv.get_field("acetate").get(5, 5) == pytest.approx(5.0)


def test_apply_dfba_oxygen_cap_and_deplete():
    env = Environment(EnvironmentConfig(width=10, height=10,
                                        oxygen_initial_mm=0.4))
    cfg = PopulationConfig(grid_width=10, grid_height=10, environment=env,
                           dfba_oxygen_max_uptake=40.0,
                           dfba_oxygen_half_saturation_mm=0.01)
    pop = CellPopulation([_cell()], cfg)

    class _Rxn:
        upper_bound = 100.0

    fake = SimpleNamespace(fba=SimpleNamespace(model=SimpleNamespace(
        reactions={"NADH_OX": _Rxn(), "FADH2_OX": _Rxn()})))
    pop._apply_dfba_oxygen_cap(fake, env, 5, 5)
    cap = 2.0 * 40.0 * (0.4 / (0.4 + 0.01))
    assert fake.fba.model.reactions["NADH_OX"].upper_bound == pytest.approx(
        min(100.0, cap))

    # a model without one of the respiratory reactions does nothing there
    fake2 = SimpleNamespace(fba=SimpleNamespace(model=SimpleNamespace(
        reactions={"FADH2_OX": _Rxn()})))
    pop._apply_dfba_oxygen_cap(fake2, env, 5, 5)
    assert fake2.fba.model.reactions["FADH2_OX"].upper_bound <= 100.0

    depleting = SimpleNamespace(last_fluxes={"NADH_OX": 2.0,
                                             "FADH2_OX": 2.0},
                                biomass_gdw=0.1)
    before = env.oxygen.get(5, 5)
    pop._deplete_dfba_oxygen(depleting, env, 5, 5, 0.05)
    assert env.oxygen.get(5, 5) < before


def test_deplete_dfba_oxygen_no_flux_noop():
    env = Environment(EnvironmentConfig(width=10, height=10))
    cfg = PopulationConfig(grid_width=10, grid_height=10, environment=env)
    pop = CellPopulation([_cell()], cfg)
    fake = SimpleNamespace(last_fluxes={"NADH_OX": 0.0, "FADH2_OX": 0.0},
                           biomass_gdw=0.1)
    before = env.oxygen.get(5, 5)
    pop._deplete_dfba_oxygen(fake, env, 5, 5, 0.05)
    assert env.oxygen.get(5, 5) == before


# ============================================================================
# numpy vectorized metabolism edge arcs (1760, 1783, 1787-1798, 1812)
# ============================================================================
def test_step_vectorized_empty_returns():
    pop = CellPopulation([_cell(id=0, alive=False)], PopulationConfig(
        grid_width=8, grid_height=8, signaling_enabled=False))
    metab, deaths = pop._step_vectorized_metabolism()
    assert metab == []
    assert deaths == 0


def test_step_vectorized_out_of_bounds_scatter_and_death():
    cells = [_cell(id=i, energy=100.0, x=i % 4, y=0) for i in range(102)]
    cells[0].x = -1  # one out-of-bounds emitter
    cells[1].energy = 0.0  # one starved cell
    cfg = PopulationConfig(grid_width=8, grid_height=1,
                           signaling_enabled=True, signal_diffusion=0.0,
                           energy_intake=1.0, metabolic_cost=0.0,
                           death_threshold=1.0, division_threshold=1e13)
    pop = CellPopulation(cells, cfg)
    metab, deaths = pop._step_vectorized_metabolism()
    assert deaths == 1
    assert len(metab) == 101
    assert pop.signal_field[0][1] > 0.0  # in-bounds scatter landed


def test_step_vectorized_match_python_fallback():
    cells = [_cell(id=i, energy=50.0, x=3, y=3) for i in range(102)]
    cfg = PopulationConfig(grid_width=8, grid_height=8,
                           signaling_enabled=False,
                           energy_intake=2.0, metabolic_cost=1.0,
                           death_threshold=0.0, division_threshold=1e13)
    pop = CellPopulation(cells, cfg)
    metab, deaths = pop._step_vectorized_metabolism()
    assert len(metab) == 102
    assert all(c.age == 1 and c.energy == 51.0 for c in pop.cells)


# ============================================================================
# statistics / trace / observables (1391, 1932-1934, 1898-1900)
# ============================================================================
def test_trace_gene_levels_none_without_grn():
    cfg = PopulationConfig(trace_streaming=True, signaling_enabled=False,
                           division_threshold=1e13)
    pop = CellPopulation([_cell()], cfg)
    pop.step()
    assert pop.trace[0]["cells"][0]["gene_levels"] is None


def test_colony_observables_with_dfba_history():
    cell = _cell(id=0, x=5, y=5, volume_um3=2.0)
    cell.dfba = SimpleNamespace(history=[1.0], growth_rate=0.4)
    pop = CellPopulation([cell], PopulationConfig(grid_width=10,
                                                  grid_height=10))
    obs = pop.colony_observables()
    assert obs["doubling_times_h"][0] == pytest.approx(
        math.log(2.0) / 0.4)
    assert obs["volume_weighted_growth_h"] == pytest.approx(0.4)
    assert len(obs["radial_density"]) == 10


def test_dfba_stratification_missing_fields_and_growth():
    env = Environment(EnvironmentConfig(width=10, height=10))
    del env.fields["oxygen"]
    cells = [_cell(id=i, x=5 + i % 2, y=5) for i in range(6)]
    cells[0].dfba = SimpleNamespace(history=[1.0], growth_rate=0.6)
    cfg = PopulationConfig(grid_width=10, grid_height=10, environment=env)
    pop = CellPopulation(cells, cfg)
    strat = pop.dfba_stratification(quantile=0.2, min_cells=4)
    assert strat["core_cell_count"] == 1
    assert strat["core_oxygen_mm"] == 0.0  # KeyError -> 0
    assert strat["core_acetate_mm"] == 0.0  # no acetate field -> 0
    assert strat["core_growth_rate_h"] == pytest.approx(0.6)


# ============================================================================
# Population alias (deprecated but importable)
# ============================================================================
def test_population_alias():
    assert Population is CellPopulation


# ============================================================================
# 3D population edge arcs
# ============================================================================
def test_pop3d_depth_property_and_get_signal_field_copy():
    pop = CellPopulation3D([_cell(x=1, y=1, z=0)],
                           PopulationConfig(grid_width=4, grid_height=4,
                                            grid_depth=3))
    assert pop.depth == 3
    pop.signal_field[2][1][1] = 9.0
    snap = pop.get_signal_field()
    snap[2][1][1] = 0.0
    assert pop.signal_field[2][1][1] == 9.0


def test_pop3d_emit_signal_out_of_bounds():
    pop = CellPopulation3D([_cell(x=1, y=1, z=9)],
                           PopulationConfig(grid_width=4, grid_height=4,
                                            grid_depth=3,
                                            signaling_enabled=False))
    oob = _cell(x=1, y=1, z=9)
    pop._emit_signal(oob)
    assert pop.signal_field[2][1][1] == 0.0
    inb = _cell(x=1, y=1, z=2)
    pop._emit_signal(inb)
    assert pop.signal_field[2][1][1] > 0.0


def test_pop3d_vectorized_empty_and_oob():
    pop = CellPopulation3D([_cell(id=0, alive=False)],
                           PopulationConfig(grid_width=4, grid_height=4,
                                            grid_depth=3))
    assert pop._step_vectorized_metabolism() == ([], 0)

    cells = [_cell(id=i, energy=100.0, x=1, y=1, z=9) for i in range(105)]
    pop = CellPopulation3D(cells, PopulationConfig(
        grid_width=4, grid_height=4, grid_depth=3,
        signaling_enabled=True, death_threshold=1.0,
        energy_intake=0.0, metabolic_cost=0.0, division_threshold=1e13))
    metab, deaths = pop._step_vectorized_metabolism()
    assert deaths == 0  # all OOB but alive; in_bounds.any() is False
    assert all(c.z == 9 for c in pop.cells)  # scatter skipped, z untouched


def test_pop3d_occupancy_empty_and_dead_oob():
    pop = CellPopulation3D([_cell(id=0, x=1, y=1, z=0, alive=False),
                            _cell(id=1, x=9, y=1, z=0)],
                           PopulationConfig(grid_width=4, grid_height=4,
                                            grid_depth=3))
    occ = pop.occupancy_3d()
    assert sum(sum(row) for plane in occ for row in plane) == 0
    assert len(occ) == 3 and len(occ[0]) == 4


def test_pop3d_mechanics_dispatch_contact_and_gaps():
    # 3D contact delegates to the base 2D rod solver
    cfg = PopulationConfig(grid_width=12, grid_height=12, grid_depth=4,
                           cell_shape="rod", cell_length_um=2.0,
                           cell_diameter_um=1.0, mechanics="contact",
                           division_threshold=1e13)
    cell = _cell(id=0, x=5, y=5, z=2, energy=1e12)
    cell.body = CellBody(x=55.0, y=55.0, length_um=2.0, diameter_um=1.0)
    pop = CellPopulation3D([cell], cfg)
    pop._apply_mechanics(pop.cells)
    assert pop.cells[0].x == 6

    # 3D shoving skips dead / OOB cells and no-op when nothing crowded
    pop2 = CellPopulation3D([_cell(id=0, x=1, y=1, z=0, alive=False),
                             _cell(id=1, x=9, y=1, z=0),
                             _cell(id=2, x=1, y=1, z=0)],
                            PopulationConfig(grid_width=4, grid_height=4,
                                             grid_depth=3,
                                             mechanics="shoving"))
    pop2._apply_mechanics(pop2.cells)
    assert pop2.occupancy_3d()[0][1][1] <= 2

    # 3D force with no reachable neighbor leaves everything in place
    pop3 = CellPopulation3D([_cell(id=i, x=0, y=0, z=0) for i in range(3)],
                            PopulationConfig(grid_width=1, grid_height=1,
                                             grid_depth=1,
                                             mechanics="force"))
    pop3._apply_mechanics(pop3.cells)
    assert pop3.occupancy_3d()[0][0][0] == 3


def test_pop3d_drift_contact_skip_and_rod():
    from helixlang.plugins.runtime.flow import FlowField3D

    u = [[[1.0 for _ in range(6)] for _ in range(6)] for _ in range(4)]
    flow3d = FlowField3D(6, 6, 4, u, [[[0.0] * 6] * 6] * 4,
                         [[[0.0] * 6] * 6] * 4)

    # no flow -> early return
    pop = CellPopulation3D([_cell(x=1, y=1, z=0)],
                           PopulationConfig(grid_width=6, grid_height=6,
                                            grid_depth=4))
    pop._drift_cells_3d(pop.cells)
    assert pop.cells[0].x == 1

    # flow + contact (rod) -> drift skipped, delegated to contact solver
    cfg = PopulationConfig(grid_width=6, grid_height=6, grid_depth=4,
                           cell_shape="rod", cell_length_um=2.0,
                           cell_diameter_um=1.0, mechanics="contact",
                           flow3d=flow3d)
    cell = _cell(id=0, x=3, y=3, z=1, energy=1e12)
    cell.body = CellBody(x=35.0, y=35.0, length_um=2.0, diameter_um=1.0)
    pop = CellPopulation3D([cell], cfg)
    x0 = pop.cells[0].body.x
    pop._drift_cells_3d(pop.cells)
    assert pop.cells[0].body.x == x0  # contact defers the drag

    # flow + rod body (non-contact) drifts the body and re-syncs the lattice
    cfg2 = PopulationConfig(grid_width=6, grid_height=6, grid_depth=4,
                            cell_shape="rod", cell_length_um=2.0,
                            cell_diameter_um=1.0, flow3d=flow3d)
    cell = _cell(id=0, x=3, y=3, z=1, energy=1e12)
    cell.body = CellBody(x=35.0, y=35.0, length_um=2.0, diameter_um=1.0)
    pop = CellPopulation3D([cell], cfg2)
    pop._drift_cells_3d(pop.cells)
    assert pop.cells[0].body.x == pytest.approx(45.0)
    assert pop.cells[0].x == 4  # round(45/10) back onto the lattice

    # flow + dead cell is skipped
    cfg3 = PopulationConfig(grid_width=6, grid_height=6, grid_depth=4,
                            flow3d=flow3d)
    pop = CellPopulation3D([_cell(id=0, x=3, y=3, z=1, alive=False)], cfg3)
    pop._drift_cells_3d(pop.cells)
    assert pop.cells[0].x == 3


def test_pop3d_lbm_2d_fallback_and_raster_skip():
    lbm = LatticeBoltzmann(width=6, height=6, omega=1.0)
    cfg = PopulationConfig(grid_width=6, grid_height=6, grid_depth=3,
                           lbm=lbm, flow_substeps=1,
                           division_threshold=1e13)
    cells = [_cell(id=0, x=2, y=2, z=1, alive=False),   # dead skipped
             _cell(id=1, x=9, y=2, z=1, energy=1e12),   # OOB skipped
             _cell(id=2, x=3, y=3, z=2, energy=1e12)]   # rasterized
    pop = CellPopulation3D(cells, cfg)
    pop.step()
    assert isinstance(cfg.flow, FlowField)
    assert bool(lbm.solid[3][3])


def test_pop3d_lbm3d_obstacle_mask_oob_dead():
    lbm = LatticeBoltzmann3D(width=6, height=6, depth=4, omega=1.2)
    cfg = PopulationConfig(grid_width=6, grid_height=6, grid_depth=4,
                           lbm=lbm, flow_substeps=1,
                           division_threshold=1e13)
    cells = [_cell(id=0, x=2, y=2, z=1, alive=False),
             _cell(id=1, x=9, y=2, z=1, energy=1e12),
             _cell(id=2, x=3, y=3, z=2, energy=1e12)]
    env_cfg = EnvironmentConfig(width=6, height=6)
    env = Environment(env_cfg)
    cfg.environment = env
    pop = CellPopulation3D(cells, cfg)
    pop.step()
    assert cfg.flow3d is not None
    assert bool(lbm.solid[2][3][3])
    assert env.flow3d is cfg.flow3d


def test_pop3d_divide_zero_depth_offset():
    pop = CellPopulation3D([_cell(x=2, y=2, z=0, energy=1e9)],
                           PopulationConfig(grid_width=8, grid_height=8,
                                            grid_depth=1,
                                            division_threshold=1e9))
    cfg = pop.config
    cell = pop.cells[0]
    a, b = pop._divide_3d(cell, cfg, pop.rng)
    assert a.z == b.z == 0  # grid_depth == 1 -> no z separation

    pop2 = CellPopulation3D([_cell(x=2, y=2, z=1, energy=1e9)],
                            PopulationConfig(grid_width=8, grid_height=8,
                                             grid_depth=5,
                                             division_threshold=1e9),
                            seed=7)
    a, b = pop2._divide_3d(pop2.cells[0], pop2.config, pop2.rng)
    assert a.z + b.z == 2  # offsets mirror around the parent's z
    assert 0 <= a.z < 5 and 0 <= b.z < 5


def test_pop3d_step_program_flow2d_env_trace():
    program, chunk = _compile("#gene name=sig\nATG TAA TAA TAA\n#end\n#config ticks=1")
    env = Environment(EnvironmentConfig(width=4, height=4))
    flow = stagnant(4, 4)
    cfg = PopulationConfig(grid_width=4, grid_height=4, grid_depth=3,
                           program=program, chunk=chunk,
                           environment=env, flow=flow,
                           trace_streaming=True,
                           signaling_enabled=False, division_threshold=1e13,
                           metabolic_cost=0.0, energy_intake=0.0)
    pop = CellPopulation3D([_cell(id=0, x=1, y=1, z=0, energy=100.0)], cfg)
    stats = pop.step()
    assert stats["alive_count"] == 1
    assert len(pop.trace) == 1
    assert env.tick == 1
    # drift went through the base 2D path (flow set, flow3d None)
    assert pop.cells[0].x == 1


def test_pop3d_to_lsystem3d_parts():
    pop = CellPopulation3D(
        [_cell(id=0, x=2, y=3, z=4, alive=True),
         _cell(id=1, x=0, y=0, z=0, alive=True),
         _cell(id=2, x=9, y=0, z=0, alive=True)],
        PopulationConfig(grid_width=10, grid_height=10, grid_depth=6))
    pop._in_bounds_3d(9, 0, 0)  # sanity: (9,0,0) is in bounds
    axiom = pop.to_lsystem3d().axiom
    # zero-offset site -> plain marker; x>0,+y,z>0 site -> full turtle
    # travel; (9,0,0) -> yaw + x-travel only
    assert "[F]" in axiom
    assert "-ff+fff^ffff&F" in axiom
    assert "-fffffffff+F" in axiom


# ============================================================================
# remaining branch arcs found by the xdist coverage pass
# ============================================================================
def test_vm_misc_guard_taken_arcs():
    # POP with a non-empty stack (1055)
    cell = _run_vm(_cell(), [Op.OP_PUSH_CONST, 0x00, Op.OP_POP],
                   constants=[9.0])
    assert cell.vm_stack == []

    # MUL / LT with two operands (1184-1186, 1189-1191)
    cell = _run_vm(_cell(),
                   [Op.OP_PUSH_CONST, 0x00, Op.OP_PUSH_CONST, 0x01,
                    Op.OP_MUL, Op.OP_PUSH_CONST, 0x00, Op.OP_LT],
                   constants=[6.0, 2.0])
    assert cell.vm_stack == [0]  # 6*2=12, push 6, 6<12=0

    # WRITE_MEM with an empty stack skips the write (1105->1199)
    cell = _run_vm(_cell(), [Op.OP_WRITE_MEM, 0x05])
    assert cell.slots[5] == 0.0 and cell.vm_stack == []

    # MODIFY_STATE default case skips (1115->1199)
    cell = _run_vm(_cell(), [Op.OP_MODIFY_STATE, 0x09])
    assert cell.color == (255, 255, 255)

    # BIND with no GRN does nothing (1141->1199)
    cell = _run_vm(_cell(), [Op.OP_BIND, 0x00], grn=None)
    assert cell.vm_stack == []


def test_genome_override_missing_reaction():
    spec = build_genome(n_genes=8, seed=5)
    pop = CellPopulation([_cell(id=0)], _genome_config(spec))
    pop._genome_colony.knock_out(["ptsG"])
    ovr = pop._genome_dfba_override(0)
    batch = SimpleNamespace(fba=SimpleNamespace(
        model=SimpleNamespace(reactions={})))
    assert ovr(0.0, batch) == {}  # gated rid missing from the model


def test_shared_batch_acetate_delta_zero():
    pop, env = _fake_dfba_pop(acetate=0.0, growth=0.0,
                              acetate_field=True, acetate_base=5.0)
    pop._step_dfba_metabolism()
    # site_acetate == base -> neither set nor deplete (1663->1665)
    assert env.fields["acetate"].get(5, 5) == pytest.approx(5.0)


def test_sync_acetate_delta_zero():
    env = Environment(EnvironmentConfig(width=10, height=10))
    env.add_field("acetate", ConcentrationField(
        "acetate", 10, 10, ACETATE_DIFFUSION_UM2_S, 5.0))
    cfg = PopulationConfig(grid_width=10, grid_height=10, environment=env,
                           acetate_switch=True)
    pop = CellPopulation([_cell()], cfg)
    fake = _FakeDFBA()
    fake.byproducts_mm = {"acetate": 5.0}
    pop._sync_acetate(fake, env, 5, 5)
    assert env.get_field("acetate").get(5, 5) == pytest.approx(5.0)


def test_vectorized_2d_all_out_of_bounds():
    cells = [_cell(id=i, energy=100.0, x=-1, y=0) for i in range(105)]
    pop = CellPopulation(cells, PopulationConfig(
        grid_width=8, grid_height=1, signaling_enabled=True,
        signal_diffusion=0.0, energy_intake=0.0, metabolic_cost=0.0,
        death_threshold=1.0, division_threshold=1e13))
    metab, deaths = pop._step_vectorized_metabolism()
    assert deaths == 0
    assert len(metab) == 105
    assert all(v == 0.0 for row in pop.signal_field for v in row)


def test_force_mechanics_skips_dead_and_oob():
    cfg = PopulationConfig(grid_width=8, grid_height=8, mechanics="force",
                           signaling_enabled=False)
    pop = CellPopulation(
        [_cell(id=0, x=0, y=0, alive=False),
         _cell(id=1, x=-1, y=0),
         _cell(id=2, x=0, y=0), _cell(id=3, x=0, y=0)], cfg)
    pop._apply_mechanics(pop.cells)
    assert pop.get_grid()[0][0] <= 2  # dead/OOB skipped; one displaced


def test_drift_and_sync_guards():
    flow = stagnant(8, 8)
    flow.u[0][1] = 1.0

    # no flow -> early return (1268)
    pop = CellPopulation([_cell(), _cell(id=1, alive=False)],
                         PopulationConfig(grid_width=8, grid_height=8))
    pop._drift_cells(pop.cells)

    # flow set, dead cell skipped in the drift loop (1271)
    pop = CellPopulation([_cell(id=0, x=0, y=0, alive=False), _cell(x=1, y=0)],
                         PopulationConfig(grid_width=8, grid_height=8,
                                          flow=flow))
    x0 = pop.cells[1].x
    pop._drift_cells(pop.cells)
    assert pop.cells[1].x == x0 + 1

    # non-rod pop short-circuits body re-projection (1287)
    pop = CellPopulation([_cell(x=1, y=1)],
                         PopulationConfig(grid_width=8, grid_height=8))
    pop._sync_bodies_to_lattice(pop.cells)

    # rod pop drifts and skips the body-less cell (1273/1274 body, 1292)
    rod = CellPopulation(
        [_cell(id=0, x=4, y=4, energy=1e12),
         _cell(id=1, x=2, y=2, energy=1e12)],
        PopulationConfig(grid_width=8, grid_height=8, flow=flow,
                         cell_shape="rod", cell_length_um=2.0,
                         cell_diameter_um=1.0))
    rod.cells[0].body = CellBody(x=40.0, y=25.0, length_um=2.0,
                                 diameter_um=1.0)
    rod._drift_cells(rod.cells)
    assert rod.cells[1].x == 2  # body-less cell untouched, not re-projected


def test_step_lbm_guards():
    pop = CellPopulation([_cell()], PopulationConfig(grid_width=8,
                                                     grid_height=8))
    pop._step_lbm()  # lbm is None (1311)

    pop = CellPopulation([_cell()],
                         PopulationConfig(grid_width=8, grid_height=8,
                                          lbm=object()))
    pop._step_lbm()  # not a LatticeBoltzmann (1315)

    lbm = LatticeBoltzmann(width=12, height=12, omega=1.0)
    env = Environment(EnvironmentConfig(width=12, height=12))
    pop = CellPopulation([_cell(id=0, energy=1e12)],
                         PopulationConfig(grid_width=12, grid_height=12,
                                          lbm=lbm, flow_substeps=2,
                                          signaling_enabled=False,
                                          environment=env))
    pop.step()
    assert env.flow is cfg_flow(pop.config)  # env.set_flow captured (1333)


def cfg_flow(cfg):
    return cfg.flow


def test_contact_mechanics_skips_bodyless_cell():
    cfg = PopulationConfig(grid_width=24, grid_height=24,
                           cell_shape="rod", cell_length_um=2.0,
                           cell_diameter_um=1.0, mechanics="contact",
                           division_threshold=1e9)
    cells = [_cell(id=0, x=5, y=5, energy=1e9),
             _cell(id=1, x=6, y=5, energy=1e9),
             _cell(id=2, x=7, y=5, energy=1e9, alive=False)]
    cells[0].body = CellBody(x=55.0, y=55.0, length_um=4.0,
                             diameter_um=1.0)
    cells[2].body = None
    pop = CellPopulation(cells, cfg)
    pop._apply_contact_mechanics(pop.cells)
    assert pop.cells[0].body.length_um > 2.0  # elongation ran


def test_genome_step_direct_call_edges():
    ply = ("#gene name=b0001\nATG TCA TAA\n#end\n"
           "#gene name=nope\nATG TAA TAA TAA\n#end\n#config ticks=1\n")
    spec = build_genome(n_genes=8, seed=3)
    pop = CellPopulation([_cell(id=0)], _genome_config(spec, ply))
    col = pop._genome_overlap["b0001"]
    pop._genome_colony.levels[:, col] = 2.0  # b0001 column triggers (969)
    pop.cells[0].grn.nodes["b0001"].level = 1.0
    pop.cells[0].grn.nodes["nope"].level = 1.0

    dead = _cell(id=1, alive=False)
    bare = _cell(id=2)
    bare.grn = None  # 975->984
    pop._step_programs([pop.cells[0], dead, bare])
    assert pop._genome_colony.levels[0, col] == 1.0  # background write


def test_genome_init_skips_existing_grn_cell():
    spec = build_genome(n_genes=8, seed=3)
    already = GRN()
    already.add_gene("b0001", 0.5)
    cells = [_cell(id=0), _cell(id=1)]
    cells[0].grn = already  # non-None -> template not applied (646->645)
    pop = CellPopulation(cells, _genome_config(spec))
    assert pop.cells[0].grn is already
    assert pop.cells[1].grn is not None
    # a program gene name missing from the genome index (675->673)
    assert "nope" not in pop._genome_overlap
    assert "b0001" in pop._genome_overlap


def test_vectorized_3d_death_writeback():
    cells = [_cell(id=i, energy=0.0, x=1, y=1, z=1) for i in range(105)]
    pop = CellPopulation3D(cells, PopulationConfig(
        grid_width=4, grid_height=4, grid_depth=3,
        signal_diffusion=0.0, energy_intake=0.0, metabolic_cost=0.0,
        death_threshold=0.0, division_threshold=1e13))
    metab, deaths = pop._step_vectorized_metabolism()
    assert deaths == 105
    assert metab == []


def test_neighbors_3d_connectivities():
    pop = CellPopulation3D([_cell(x=1, y=1, z=1)],
                           PopulationConfig(grid_width=4, grid_height=4,
                                            grid_depth=4))
    # a corner hit exercises the out-of-bounds 6-neighbor arcs (2092->2089)
    six = pop.neighbors_3d(0, 0, 0)  # default connectivity=6
    assert len(six) == 3
    twentysix = pop.neighbors_3d(1, 1, 1, connectivity=26)
    assert len(twentysix) == 26
    with pytest.raises(ValueError, match="6 or 26"):
        pop.neighbors_3d(1, 1, 1, connectivity=5)


def test_3d_force_skips_and_none_mechanics():
    pop = CellPopulation3D(
        [_cell(id=0, x=1, y=1, z=0, alive=False),
         _cell(id=1, x=9, y=1, z=0),
         _cell(id=2, x=1, y=1, z=0), _cell(id=3, x=1, y=1, z=0)],
        PopulationConfig(grid_width=4, grid_height=4, grid_depth=2,
                         mechanics="force"))
    pop._apply_mechanics(pop.cells)
    assert pop.occupancy_3d()[0][1][1] <= 2

    plain = CellPopulation3D([_cell(id=0, x=1, y=1, z=0)],
                             PopulationConfig(grid_width=4, grid_height=4,
                                              grid_depth=2))
    plain._apply_mechanics(plain.cells)  # mechanics None -> 2264->exit
