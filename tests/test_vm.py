"""VM unit tests."""
import pytest

from helixlang.core.bytecode import Chunk
from helixlang.core.codon_table import STANDARD_TABLE, Op
from helixlang.core.compiler import Compiler
from helixlang.core.errors import ModelMissingError, StackUnderflowError
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.core.semantic import SemanticAnalyzer
from helixlang.core.vm import CellVM, Frame
from helixlang.plugins.runtime.cell import (
    FEED_ENERGY_AMOUNT,
    INITIAL_CELL_ENERGY,
    MOVE_ENERGY_COST,
    Cell,
)


def run_src(src, ticks=10, table=STANDARD_TABLE):
    stop = {c for c, op in table.items() if op == Op.OP_HALT}
    toks = list(Lexer(src).tokens())
    prog = Parser(toks, stop_codons=stop).parse()
    SemanticAnalyzer(prog).check()
    chunk = Compiler(table).compile(prog)
    vm = CellVM(chunk, prog)
    return vm, vm.run(ticks)


def test_hello_dna_produces_protein():
    src = "#gene name=hello\nATG GCT TAA\n#end\n#config ticks=1"
    vm, trace = run_src(src, ticks=1)
    # GCT = OP_BUILD_PROTEIN, should synthesize protein
    assert trace[-1]["proteins"] != {}


def test_move_changes_position():
    src = "#gene name=mover\nATG GTA TAA\n#end\n#config ticks=1"
    vm, trace = run_src(src, ticks=1)
    # GTA = OP_MOVE arg=2 (South); y should change
    assert trace[-1]["y"] != 0


def test_die_kills_cell():
    src = "#gene name=killer\nATG AAA TAA\n#end\n#config ticks=5"
    vm, trace = run_src(src, ticks=5)
    # AAA = OP_DIE; the cell should die
    assert any(not t["alive"] for t in trace)


def test_feed_restores_energy():
    src = "#gene name=feeder\nATG GAA TAA\n#end\n#config ticks=1"
    vm, trace = run_src(src, ticks=1)
    # GAA = OP_FEED; energy should increase (initial 100 + 10)
    assert trace[-1]["energy"] >= 100


def test_constitutive_gene_runs_every_tick():
    src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=5"
    vm, trace = run_src(src, ticks=5)
    # No promoter -> constitutive, executes every tick -> protein concentration increases
    prots = [t["proteins"].get(3, 0.0) for t in trace]
    # Protein concentration in later ticks should be >= earlier ones
    assert prots[-1] >= prots[0]


def test_lsystem_grows_morphology():
    # Constitutive gene (no promoter): active from tick 0. Under the physical
    # GRN decay (~0.994/tick, 110-min half-life) a regulated gene takes
    # ~100+ ticks to trigger, so a constitutive growth gene is used here to
    # observe morphology growth on a short run.
    src = """#gene name=grow
ATG CTC TAA
#end
#lsystem name=plant axiom=F rules=0:F->F[+F]F[-F]F angle=25
#config ticks=10
"""
    vm, trace = run_src(src, ticks=10)
    # Morphology points should be produced
    assert trace[-1]["morphology_points_count"] > 1


def test_reaction_diffusion_field_created():
    src = """#promoter name=p strength=-0.5
#gene name=reactor promoter=p
ATG GAT TAA
#end
#field size=16 F=0.035 k=0.065
#config ticks=5 react_steps=1
"""
    vm, trace = run_src(src, ticks=5)
    # field_total_v should be > 0 (has a seed)
    assert trace[-1]["field_total_v"] > 0


def test_halt_terminates_orf():
    """OP_HALT should pop the frame and return."""
    src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=1"
    vm, trace = run_src(src, ticks=1)
    # After the VM runs, the frame stack should be empty
    assert len(vm.frames) == 0


def test_table_switch_different_behavior():
    """The same DNA produces different bytecode under different tables."""
    src = "#gene name=m\nATG TGA GCT TAA\n#end"
    from helixlang.core.codon_table import MITO_VERTEBRATE_TABLE
    # Standard: TGA=HALT -> ORF length 2 (ATG TGA)
    stop_std = {c for c, op in STANDARD_TABLE.items() if op == Op.OP_HALT}
    prog_std = Parser(list(Lexer(src).tokens()), stop_codons=stop_std).parse()
    assert len(prog_std.genes[0].orf) == 2
    # Mito: TGA=PIGMENT -> ORF length 4 (ATG TGA GCT TAA)
    stop_mito = {c for c, op in MITO_VERTEBRATE_TABLE.items() if op == Op.OP_HALT}
    prog_mito = Parser(list(Lexer(src).tokens()), stop_codons=stop_mito).parse()
    assert len(prog_mito.genes[0].orf) == 4


# ============================================================================
# Opcode coverage tests (construct Chunk directly + drive _dispatch)
# ============================================================================

from helixlang.core.ast_nodes import Config, Program  # noqa: E402
from helixlang.plugins.runtime.reaction_diffusion import GrayScott  # noqa: E402


def _make_vm(chunk: Chunk, program: Program | None = None,
             ops_per_tick: int = 256) -> CellVM:
    """Construct a CellVM and prepare a frame for executing chunk.code.

    Directly push a Frame whose return_ip points to the end of the chunk so
    that _execute_pending can start executing from ip=0.
    """
    if program is None:
        program = Program(config=Config(ops_per_tick=ops_per_tick))
    else:
        program.config.ops_per_tick = ops_per_tick
    vm = CellVM(chunk, program)
    # Manually push a frame: return_ip points to the end of the code so the frame is popped after execution
    vm.frames.append(Frame(return_ip=len(chunk.code), gene_name="test"))
    vm.ip = 0
    return vm


def _run_chunk(chunk: Chunk, program: Program | None = None,
               ops_per_tick: int = 256) -> CellVM:
    """Execute the chunk and return vm (stack/cell state can be inspected afterwards)."""
    vm = _make_vm(chunk, program, ops_per_tick)
    vm._execute_pending()
    return vm


class TestNopOpcodes:
    """OP_START / OP_NOP / OP_TICK are no-ops."""

    def test_op_start_noop(self):
        c = Chunk()
        c.emit(Op.OP_START)
        vm = _run_chunk(c)
        assert vm.stack == []

    def test_op_nop_noop(self):
        c = Chunk()
        c.emit(Op.OP_NOP)
        vm = _run_chunk(c)
        assert vm.stack == []

    def test_op_tick_noop(self):
        c = Chunk()
        c.emit(Op.OP_TICK)
        vm = _run_chunk(c)
        assert vm.stack == []

    def test_nop_does_not_change_cell(self):
        c = Chunk()
        c.emit(Op.OP_NOP)
        c.emit(Op.OP_NOP)
        vm = _run_chunk(c)
        assert vm.cell.energy == INITIAL_CELL_ENERGY
        assert vm.cell.alive is True
        assert vm.cell.x == 0 and vm.cell.y == 0


class TestHaltReturn:
    """OP_HALT / OP_RETURN pop the current frame."""

    def test_op_halt_pops_frame(self):
        c = Chunk()
        c.emit(Op.OP_HALT)
        vm = _run_chunk(c)
        # The frame is popped
        assert len(vm.frames) == 0

    def test_op_return_pops_frame(self):
        c = Chunk()
        c.emit(Op.OP_RETURN)
        vm = _run_chunk(c)
        assert len(vm.frames) == 0

    def test_halt_after_ops_stops_execution(self):
        """Bytecode after HALT should not execute (the frame is already popped)."""
        c = Chunk()
        c.emit(Op.OP_HALT)
        c.emit(Op.OP_PUSH_CONST, 0)  # this should not execute
        vm = _run_chunk(c)
        assert vm.stack == []


class TestStackOpcodes:
    """Stack operations: PUSH_CONST / POP / DUP / SWAP."""

    def test_push_const_in_range(self):
        c = Chunk()
        idx = c.add_constant(42)
        c.emit(Op.OP_PUSH_CONST, idx)
        vm = _run_chunk(c)
        assert vm.stack == [42]

    def test_push_const_out_of_range_falls_back_to_idx(self):
        """When idx >= len(constants), push idx itself."""
        c = Chunk()
        c.emit(Op.OP_PUSH_CONST, 99)  # constants is empty
        vm = _run_chunk(c)
        assert vm.stack == [99]

    def test_push_const_dedup(self):
        """add_constant deduplicates: the same value returns the same index."""
        c = Chunk()
        i1 = c.add_constant(7)
        i2 = c.add_constant(7)
        assert i1 == i2

    def test_pop_removes_top(self):
        c = Chunk()
        c.add_constant(1)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_POP)
        vm = _run_chunk(c)
        assert vm.stack == []

    def test_pop_empty_stack_safe(self):
        """POP on an empty stack is a stack underflow (doc/36 F11)."""
        c = Chunk()
        c.emit(Op.OP_POP)
        with pytest.raises(StackUnderflowError):
            _run_chunk(c)

    def test_dup_duplicates_top(self):
        c = Chunk()
        c.add_constant(5)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_DUP)
        vm = _run_chunk(c)
        assert vm.stack == [5, 5]

    def test_dup_empty_stack_safe(self):
        c = Chunk()
        c.emit(Op.OP_DUP)
        with pytest.raises(StackUnderflowError):
            _run_chunk(c)

    def test_swap_swaps_top_two(self):
        c = Chunk()
        c.add_constant(1)
        c.add_constant(2)
        c.emit(Op.OP_PUSH_CONST, 0)  # [1]
        c.emit(Op.OP_PUSH_CONST, 1)  # [1, 2]
        c.emit(Op.OP_SWAP)
        vm = _run_chunk(c)
        assert vm.stack == [2, 1]

    def test_swap_single_element_safe(self):
        """SWAP needs two operands; one element is a stack underflow (F11)."""
        c = Chunk()
        c.add_constant(1)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_SWAP)
        with pytest.raises(StackUnderflowError):
            _run_chunk(c)


class TestBuildOpcodes:
    """Synthesis opcodes: BUILD_PROTEIN / BUILD_MEMBRANE / BUILD_PIGMENT."""

    def test_build_protein_adds_to_cell(self):
        c = Chunk()
        c.emit(Op.OP_BUILD_PROTEIN, 3)
        vm = _run_chunk(c)
        assert vm.cell.proteins.get(3) == 1.0

    def test_build_protein_different_kinds(self):
        c = Chunk()
        c.emit(Op.OP_BUILD_PROTEIN, 1)
        c.emit(Op.OP_BUILD_PROTEIN, 2)
        c.emit(Op.OP_BUILD_PROTEIN, 1)  # accumulates
        vm = _run_chunk(c)
        assert vm.cell.proteins[1] == 2.0
        assert vm.cell.proteins[2] == 1.0

    def test_build_membrane_sets_permeability(self):
        """BUILD_MEMBRANE sets the cell's membrane permeability from its operand."""
        c = Chunk()
        c.emit(Op.OP_BUILD_MEMBRANE, 200)
        vm = _run_chunk(c)
        assert vm.cell.membrane_permeability == 200
        assert vm.cell.alive is True

    def test_build_membrane_scales_feed(self):
        """A lowered membrane permeability reduces the energy gained by FEED."""
        c = Chunk()
        c.emit(Op.OP_BUILD_MEMBRANE, 0)   # impermeable
        c.emit(Op.OP_FEED, 0)
        vm = _run_chunk(c)
        assert vm.cell.energy == INITIAL_CELL_ENERGY  # impermeable: no gain

    def test_build_membrane_in_snapshot(self):
        """The snapshot trace exposes the membrane permeability."""
        src = "#gene name=mem\nATG GGT TAA\n#end\n#config ticks=1"
        vm, trace = run_src(src, ticks=1)
        # GGT = OP_BUILD_MEMBRANE with wobble T=3 -> permeability 3
        assert trace[-1]["membrane_permeability"] == 3

    def test_build_pigment_sets_color(self):
        c = Chunk()
        c.emit(Op.OP_BUILD_PIGMENT)
        vm = _run_chunk(c)
        assert vm.cell.color == (200, 50, 50)


class TestBehaviorOpcodes:
    """Behavior opcodes: MOVE / SIGNAL / DIVIDE / DIE / FEED."""

    def test_move_changes_position(self):
        c = Chunk()
        c.emit(Op.OP_MOVE, 1)  # East
        vm = _run_chunk(c)
        assert vm.cell.x == 1
        assert vm.cell.y == 0
        assert vm.cell.energy == INITIAL_CELL_ENERGY - MOVE_ENERGY_COST

    def test_move_north(self):
        c = Chunk()
        c.emit(Op.OP_MOVE, 0)
        vm = _run_chunk(c)
        assert vm.cell.y == -1

    def test_signal_releases_into_field(self):
        """SIGNAL releases a quorum-sensing autoinducer into the field."""
        c = Chunk()
        c.emit(Op.OP_SIGNAL, 0)
        vm = _make_vm(c)
        vm.field = GrayScott(n=8)
        before = vm.field.v[0][0]
        vm._execute_pending()
        assert vm.field.v[0][0] > before
        assert vm._signal_emissions == 1

    def test_signal_channel_scales_amount(self):
        """A larger signal channel releases a larger amount (capped at 1.0)."""
        def v_after(ch):
            c = Chunk()
            c.emit(Op.OP_SIGNAL, ch)
            vm = _make_vm(c)
            vm.field = GrayScott(n=8)
            vm._execute_pending()
            return vm.field.v[0][0]
        assert v_after(3) > v_after(0)

    def test_signal_no_field_counts_emission(self):
        """SIGNAL without a field still records the emission and leaves no stack residue."""
        c = Chunk()
        c.emit(Op.OP_SIGNAL, 2)
        vm = _run_chunk(c)
        assert vm._signal_emissions == 1
        assert vm.stack == []
        assert vm.field is None

    def test_divide_halves_energy(self):
        c = Chunk()
        c.emit(Op.OP_DIVIDE, 0)
        vm = _run_chunk(c)
        assert vm.cell.energy == INITIAL_CELL_ENERGY // 2  # 1e9 // 2
        assert vm.cell.divisions == 1

    def test_die_sets_alive_false(self):
        c = Chunk()
        c.emit(Op.OP_DIE, 0)
        vm = _run_chunk(c)
        assert vm.cell.alive is False

    def test_feed_increases_energy(self):
        """FEED always adds the nutrient amount (1e8 ATP)."""
        c = Chunk()
        c.emit(Op.OP_FEED, 0)
        vm = _run_chunk(c)
        assert vm.cell.energy == INITIAL_CELL_ENERGY + FEED_ENERGY_AMOUNT

    def test_feed_from_low_energy(self):
        # First lower the energy to 5
        vm0 = Cell(energy=5)
        # Directly construct the vm and replace the cell
        chunk = Chunk()
        chunk.emit(Op.OP_FEED, 0)
        prog = Program()
        vm = CellVM(chunk, prog)
        vm.frames.append(Frame(return_ip=len(chunk), gene_name="t"))
        vm.ip = 0
        vm.cell = vm0
        vm._execute_pending()
        assert vm.cell.energy == 5 + FEED_ENERGY_AMOUNT


class TestArithmeticOpcodes:
    """Arithmetic: ADD / SUB / MUL / LT / NOT."""

    def test_add(self):
        c = Chunk()
        c.add_constant(3)
        c.add_constant(4)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_PUSH_CONST, 1)
        c.emit(Op.OP_ADD)
        vm = _run_chunk(c)
        assert vm.stack == [7]

    def test_sub(self):
        c = Chunk()
        c.add_constant(10)
        c.add_constant(3)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_PUSH_CONST, 1)
        c.emit(Op.OP_SUB)
        vm = _run_chunk(c)
        assert vm.stack == [7]

    def test_mul(self):
        c = Chunk()
        c.add_constant(6)
        c.add_constant(7)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_PUSH_CONST, 1)
        c.emit(Op.OP_MUL)
        vm = _run_chunk(c)
        assert vm.stack == [42]

    def test_lt_true(self):
        c = Chunk()
        c.add_constant(2)
        c.add_constant(5)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_PUSH_CONST, 1)
        c.emit(Op.OP_LT)
        vm = _run_chunk(c)
        assert vm.stack == [1]

    def test_lt_false(self):
        c = Chunk()
        c.add_constant(5)
        c.add_constant(2)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_PUSH_CONST, 1)
        c.emit(Op.OP_LT)
        vm = _run_chunk(c)
        assert vm.stack == [0]

    def test_lt_equal_returns_zero(self):
        c = Chunk()
        c.add_constant(5)
        c.add_constant(5)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_PUSH_CONST, 1)
        c.emit(Op.OP_LT)
        vm = _run_chunk(c)
        assert vm.stack == [0]

    def test_not_zero_returns_one(self):
        c = Chunk()
        c.add_constant(0)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_NOT)
        vm = _run_chunk(c)
        assert vm.stack == [1]

    def test_not_nonzero_returns_zero(self):
        c = Chunk()
        c.add_constant(5)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_NOT)
        vm = _run_chunk(c)
        assert vm.stack == [0]

    def test_add_insufficient_operands_safe(self):
        """ADD with <2 operands is a stack underflow (doc/36 F11)."""
        c = Chunk()
        c.add_constant(1)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_ADD)
        with pytest.raises(StackUnderflowError):
            _run_chunk(c)


class TestMemoryOpcodes:
    """Memory: READ_MEM / WRITE_MEM / MODIFY_STATE."""

    def test_read_mem_default_none(self):
        c = Chunk()
        c.emit(Op.OP_READ_MEM, 5)
        vm = _run_chunk(c)
        assert vm.stack == [None]

    def test_write_mem_pops_to_slot(self):
        c = Chunk()
        c.add_constant(99)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_WRITE_MEM, 10)
        vm = _run_chunk(c)
        assert vm.cell.slots[10] == 99
        assert vm.stack == []

    def test_write_mem_empty_stack_safe(self):
        """WRITE_MEM on an empty stack is a stack underflow (doc/36 F11)."""
        c = Chunk()
        c.emit(Op.OP_WRITE_MEM, 0)
        with pytest.raises(StackUnderflowError):
            _run_chunk(c)

    def test_read_write_roundtrip(self):
        c = Chunk()
        c.add_constant(42)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_WRITE_MEM, 100)
        c.emit(Op.OP_READ_MEM, 100)
        vm = _run_chunk(c)
        assert vm.stack == [42]

    def test_modify_state_zero_sets_color(self):
        c = Chunk()
        c.emit(Op.OP_MODIFY_STATE, 0)
        vm = _run_chunk(c)
        assert vm.cell.color == (100, 200, 50)

    def test_modify_state_one_increments_age(self):
        c = Chunk()
        c.emit(Op.OP_MODIFY_STATE, 1)
        c.emit(Op.OP_MODIFY_STATE, 1)
        vm = _run_chunk(c)
        assert vm.cell.age == 2

    def test_modify_state_two_sets_yellow(self):
        c = Chunk()
        c.emit(Op.OP_MODIFY_STATE, 2)
        vm = _run_chunk(c)
        assert vm.cell.color == (200, 200, 50)

    def test_modify_state_three_sets_magenta(self):
        c = Chunk()
        c.emit(Op.OP_MODIFY_STATE, 3)
        vm = _run_chunk(c)
        assert vm.cell.color == (200, 50, 200)


class TestRegulateBind:
    """OP_REGULATE / OP_BIND: runtime regulatory rewiring and protein-DNA binding."""

    def _grn_vm(self, chunk):
        vm = _make_vm(chunk)
        vm.grn.add_gene("a", threshold=0.5, initial_level=1.0)
        vm.grn.add_gene("b", threshold=0.5, initial_level=0.0)
        vm.frames[-1].gene_name = "a"
        return vm

    def test_regulate_adds_activating_edge(self):
        """REGULATE mode=1 adds a +1.0 edge from the current gene to gene b."""
        c = Chunk()
        c.emit(Op.OP_REGULATE, 1)
        vm = self._grn_vm(c)
        vm._execute_pending()
        assert len(vm.grn.edges) == 1
        e = vm.grn.edges[0]
        assert e.source == "a" and e.target == "b"
        assert e.weight == 1.0
        assert vm._regulation_events[0]["weight"] == 1.0

    def test_regulate_inhibit_negative_weight(self):
        """REGULATE with bit 7 set adds an inhibiting edge (-1.0)."""
        c = Chunk()
        c.emit(Op.OP_REGULATE, 0x81)   # low nibble 1 -> target b, sign bit -> inhibit
        vm = self._grn_vm(c)
        vm._execute_pending()
        assert len(vm.grn.edges) == 1
        assert vm.grn.edges[0].weight == -1.0

    def test_regulate_updates_existing_edge(self):
        """Repeated REGULATE on the same (source, target) updates the weight in place."""
        c = Chunk()
        c.emit(Op.OP_REGULATE, 1)
        c.emit(Op.OP_REGULATE, 0x81)
        vm = self._grn_vm(c)
        vm._execute_pending()
        assert len(vm.grn.edges) == 1
        assert vm.grn.edges[0].weight == -1.0

    def test_regulate_wraps_target_index(self):
        """The low nibble selects the target by index modulo the node count."""
        c = Chunk()
        c.emit(Op.OP_REGULATE, 0x0F)   # 15 % 2 == 1 -> gene b
        vm = self._grn_vm(c)
        vm._execute_pending()
        assert vm.grn.edges[0].target == "b"

    def test_regulate_empty_grn_safe(self):
        """REGULATE with no GRN nodes is a safe no-op."""
        c = Chunk()
        c.emit(Op.OP_REGULATE, 1)
        vm = _run_chunk(c)
        assert vm.grn.edges == []

    def test_bind_consumes_protein_and_boosts(self):
        """BIND consumes one TF unit and boosts the target gene's level."""
        c = Chunk()
        c.emit(Op.OP_BIND, 1)          # site 1 -> gene b
        vm = self._grn_vm(c)
        vm.cell.add_protein("a", 2.0)
        vm._execute_pending()
        assert vm.grn.nodes["b"].level == 0.5     # 0.0 + BIND_LEVEL_BOOST
        assert vm.cell.proteins.get("a") == 1.0   # consumed one unit
        assert len(vm._binding_events) == 1
        assert vm._binding_events[0]["target"] == "b"

    def test_bind_no_protein_no_binding(self):
        """BIND is protein-limited: no transcription factor, no binding."""
        c = Chunk()
        c.emit(Op.OP_BIND, 1)
        vm = self._grn_vm(c)
        vm._execute_pending()
        assert vm.grn.nodes["b"].level == 0.0
        assert vm._binding_events == []

    def test_bind_empty_grn_safe(self):
        """BIND with no GRN nodes is a safe no-op."""
        c = Chunk()
        c.emit(Op.OP_BIND, 1)
        vm = _run_chunk(c)
        assert vm._binding_events == []


class TestJumpOpcodes:
    """OP_JUMP / OP_JUMP_IF_ZERO / OP_CALL_GENE."""

    def test_jump_advances_ip(self):
        """OP_JUMP skips intermediate instructions."""
        c = Chunk()
        # offset 2 skips one PUSH_CONST (after reading the u16, the jmp has already advanced ip)
        c.emit_u16(Op.OP_JUMP, 2)
        # This PUSH_CONST should be skipped
        c.add_constant(999)
        c.emit(Op.OP_PUSH_CONST, 0)
        # Landing point: push a different value
        c.add_constant(1)
        c.emit(Op.OP_PUSH_CONST, 1)
        vm = _run_chunk(c)
        assert vm.stack == [1]  # skipped 999

    def test_jump_if_zero_when_zero(self):
        """Top of stack is 0 -> jump."""
        c = Chunk()
        c.add_constant(0)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit_u16(Op.OP_JUMP_IF_ZERO, 2)  # skip the next PUSH_CONST
        c.add_constant(999)
        c.emit(Op.OP_PUSH_CONST, 1)  # skipped
        c.add_constant(1)
        c.emit(Op.OP_PUSH_CONST, 2)  # landing point
        vm = _run_chunk(c)
        assert vm.stack == [1]

    def test_jump_if_zero_when_nonzero(self):
        """Top of stack is nonzero -> no jump."""
        c = Chunk()
        c.add_constant(5)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit_u16(Op.OP_JUMP_IF_ZERO, 2)  # no jump
        c.add_constant(999)
        c.emit(Op.OP_PUSH_CONST, 1)  # executes
        c.add_constant(1)
        c.emit(Op.OP_PUSH_CONST, 2)  # executes
        vm = _run_chunk(c)
        assert vm.stack == [999, 1]

    def test_jump_if_zero_empty_stack_pops_zero(self):
        """JUMP_IF_ZERO on an empty stack is a stack underflow (doc/36 F11):
        it must raise StackUnderflowError, not silently pop 0 / skip."""
        c = Chunk()
        c.emit_u16(Op.OP_JUMP_IF_ZERO, 2)
        c.add_constant(999)
        c.emit(Op.OP_PUSH_CONST, 0)  # skipped at runtime
        c.add_constant(1)
        c.emit(Op.OP_PUSH_CONST, 1)
        with pytest.raises(StackUnderflowError):
            _run_chunk(c)

    def test_call_gene_pushes_frame(self):
        """OP_CALL_GENE pushes a new frame and jumps to the u16 offset."""
        c = Chunk()
        # Place a PUSH_CONST at offset 5
        # CALL_GENE itself takes 3 bytes (op + u16)
        # We set the target offset to 3 and put PUSH_CONST right after
        c.emit_u16(Op.OP_CALL_GENE, 3)
        c.add_constant(77)
        c.emit(Op.OP_PUSH_CONST, 0)  # offset 3
        c.emit(Op.OP_HALT)  # offset 5: pops the call frame
        vm = _run_chunk(c)
        # After CALL_GENE, ip=3, executes PUSH_CONST(77), then HALT pops the call frame
        # The outer frame's return_ip=0, but _execute_pending stops once the frame stack is empty
        assert 77 in vm.stack


class TestFieldOpcodes:
    """Field-dependent opcodes: DIFFUSE / REACT / EMIT_MORPHOGEN."""

    def _make_vm_with_field(self, chunk: Chunk, n: int = 8) -> CellVM:
        prog = Program(config=Config(ops_per_tick=256, react_steps=2))
        vm = _make_vm(chunk, prog)
        vm.field = GrayScott(n=n, F=0.035, k=0.065)
        return vm

    def test_diffuse_steps_field(self):
        c = Chunk()
        c.emit(Op.OP_DIFFUSE, 0)
        vm = self._make_vm_with_field(c)
        vm._execute_pending()
        # step may change the v distribution; total_v may stay the same (mass conservation),
        # but at least it should not crash. Verify the field is still accessible
        assert vm.field is not None
        assert vm.field.n == 8

    def test_react_calls_step_react_steps_times(self):
        """REACT should call field.step() react_steps times."""
        c = Chunk()
        c.emit(Op.OP_REACT, 0)
        prog = Program(config=Config(ops_per_tick=256, react_steps=3))
        vm = _make_vm(c, prog)
        vm.field = GrayScott(n=8)
        # mock step counting
        call_count = [0]
        orig_step = vm.field.step
        def counting_step():
            call_count[0] += 1
            orig_step()
        vm.field.step = counting_step
        vm._execute_pending()
        assert call_count[0] == 3

    def test_react_default_react_steps(self):
        """Default react_steps is 1."""
        c = Chunk()
        c.emit(Op.OP_REACT, 0)
        prog = Program(config=Config(ops_per_tick=256, react_steps=1))
        vm = _make_vm(c, prog)
        vm.field = GrayScott(n=8)
        call_count = [0]
        orig_step = vm.field.step
        def counting_step():
            call_count[0] += 1
            orig_step()
        vm.field.step = counting_step
        vm._execute_pending()
        assert call_count[0] == 1

    def test_emit_morphogen_increases_field_v(self):
        """EMIT_MORPHOGEN injects V at the cell position."""
        c = Chunk()
        c.emit(Op.OP_EMIT_MORPHOGEN, 0)
        vm = self._make_vm_with_field(c, n=8)
        before = vm.field.v[0][0]
        vm._execute_pending()
        after = vm.field.v[0][0]
        assert after > before

    def test_emit_morphogen_id_scales_amount(self):
        """A higher morphogen ID injects a larger amount than ID 0."""
        def v_after(m_id):
            c = Chunk()
            c.emit(Op.OP_EMIT_MORPHOGEN, m_id)
            vm = self._make_vm_with_field(c, n=8)
            vm._execute_pending()
            return vm.field.v[0][0]
        assert v_after(255) > v_after(0)
        assert v_after(3) > v_after(0)

    def test_diffuse_no_field_safe(self):
        """DIFFUSE should not crash when there is no field."""
        c = Chunk()
        c.emit(Op.OP_DIFFUSE, 0)
        vm = _run_chunk(c)
        assert vm.field is None

    def test_react_no_field_safe(self):
        c = Chunk()
        c.emit(Op.OP_REACT, 0)
        vm = _run_chunk(c)
        assert vm.field is None

    def test_emit_morphogen_no_field_safe(self):
        c = Chunk()
        c.emit(Op.OP_EMIT_MORPHOGEN, 0)
        vm = _run_chunk(c)
        assert vm.field is None


class TestGrowLSystem:
    """OP_GROW_LSYSTEM."""

    def test_grow_lsystem_no_lsystems_safe(self):
        """GROW_LSYSTEM should not crash when there are no lsystems."""
        c = Chunk()
        c.emit(Op.OP_GROW_LSYSTEM, 0)
        vm = _run_chunk(c)
        # Default morphology point is kept
        assert len(vm.cell.morphology_points) >= 1

    def test_grow_lsystem_appends_points(self):
        """GROW_LSYSTEM appends morphology points when an lsystem exists."""
        from helixlang.plugins.runtime.lsystem import LSystem
        c = Chunk()
        c.emit(Op.OP_GROW_LSYSTEM, 0)
        prog = Program()
        vm = _make_vm(c, prog)
        vm.lsystems["plant"] = LSystem(
            axiom="F", rules={"F": "F[+F]F[-F]F"}, angle=25.0, step=1.0)
        before = len(vm.cell.morphology_points)
        vm._execute_pending()
        after = len(vm.cell.morphology_points)
        assert after > before


class TestDebugOpcode:
    """OP_DEBUG prints the cell dump."""

    def test_debug_prints_to_stdout(self, capsys):
        c = Chunk()
        c.emit(Op.OP_DEBUG)
        _run_chunk(c)
        out = capsys.readouterr().out
        assert "DEBUG" in out
        assert "Cell(" in out


class TestUnknownOpcode:
    """Unknown opcodes are strict errors (doc/38), never silently skipped."""

    def test_unknown_opcode_raises(self):
        """A byte that is not a valid Op raises UnknownOpcodeError (doc/38)."""
        from helixlang.core.errors import UnknownOpcodeError
        c = Chunk()
        # Directly append an unknown byte followed by a valid PUSH_CONST: the
        # VM must stop at the unknown byte instead of skipping it.
        c.code.append(0x00)
        c.lines.append(0)
        c.codon_indices.append(-1)
        c.add_constant(5)
        c.emit(Op.OP_PUSH_CONST, 0)
        vm = _make_vm(c)
        with pytest.raises(UnknownOpcodeError, match="0x00"):
            vm._execute_pending()

    def test_unknown_opcode_raises_in_accel(self):
        """The accelerated path agrees with the pure loop on unknown bytes."""
        from helixlang.core.errors import UnknownOpcodeError
        c = Chunk()
        c.code.append(0x01)  # invalid opcode
        c.lines.append(0)
        c.codon_indices.append(-1)
        c.add_constant(7)
        c.emit(Op.OP_PUSH_CONST, 0)
        vm = _make_vm(c)
        vm.use_accel = True
        with pytest.raises(UnknownOpcodeError, match="0x01"):
            vm._execute_pending()


class TestVmRunIntegration:
    """VM.run end-to-end integration (covers the main loop)."""

    def test_run_dead_cell_stops_loop(self):
        """After DIE, cell.alive=False; run should stop the tick loop."""
        src = "#gene name=killer\nATG AAA TAA\n#end\n#config ticks=10"
        vm, trace = run_src(src, ticks=10)
        # Should die on the first tick; trace length < 10
        assert any(not t["alive"] for t in trace)
        # No snapshots are produced after death
        alive_ticks = [t for t in trace if t["alive"]]
        assert len(alive_ticks) < 10

    def test_run_snapshot_structure(self):
        """Each snapshot contains the required fields."""
        src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=2"
        vm, trace = run_src(src, ticks=2)
        for snap in trace:
            for key in ("tick", "x", "y", "energy", "alive",
                        "proteins", "color", "gene_levels",
                        "morphology_points_count",
                        "membrane_permeability", "signal_emissions",
                        "regulation_edges", "binding_events",
                        "field_total_v"):
                assert key in snap, f"snapshot missing key {key}"

    def test_run_trace_tick_increments(self):
        src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=3"
        vm, trace = run_src(src, ticks=3)
        ticks = [t["tick"] for t in trace]
        assert ticks == [0, 1, 2]

    def test_run_zero_ticks_empty_trace(self):
        src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=0"
        # ticks=0 would be rejected by SemanticAnalyzer; use valid ticks=1 but run(0)
        src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=1"
        vm, trace = run_src(src, ticks=0)
        assert trace == []

    def test_init_subsystems_no_field(self):
        """vm.field is None when there is no field_decl."""
        src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=1"
        vm, _ = run_src(src, ticks=1)
        assert vm.field is None

    def test_init_subsystems_with_field(self):
        src = ("#gene name=g\nATG GCT TAA\n#end\n"
               "#field size=16 F=0.035 k=0.065\n#config ticks=1")
        vm, _ = run_src(src, ticks=1)
        assert vm.field is not None
        assert vm.field.n == 16

    def test_gene_dna_cached(self):
        """_init_subsystems caches the gene DNA sequences."""
        src = "#gene name=g\nATG GCT GGT TAA\n#end\n#config ticks=1"
        vm, _ = run_src(src, ticks=1)
        assert vm._gene_dna.get("g") == "ATGGCTGGTTAA"

    def test_promoter_strengths_cached(self):
        src = ("#promoter name=p strength=-0.7\n"
               "#gene name=g promoter=p\nATG GCT TAA\n#end\n#config ticks=1")
        vm, _ = run_src(src, ticks=1)
        # abs(-0.7) normalized
        assert vm._promoter_strengths["p"] == 0.7


# ------------------------------------------------------------------ #
# Calibrated mode (doc/16-gameplay-units-upgrade.md §7 Tier 2)
# ------------------------------------------------------------------ #
def test_central_dogma_named_constants_match_legacy_literals():
    """Hardcoded literals were lifted to named constants with identical defaults."""
    import helixlang.core.vm as vm
    assert vm.RIBO_SOME_DENSITY_PER_100NT == 0.1
    assert vm.PROTEIN_YIELD_PER_MRNA_AA == 0.1
    assert vm.PROTEIN_TO_GRN_GAIN == 0.01
    assert vm.MORPHOGEN_TO_GRN_GAIN == 0.1
    assert vm.CONSTITUTIVE_PROMOTER_STRENGTH == 0.5


def test_op_feed_uses_named_constant(monkeypatch):
    """OP_FEED must feed the module constant, not a hardcoded literal."""
    import helixlang.core.vm as vm
    monkeypatch.setattr(vm, "FEED_ENERGY_AMOUNT", 37.0)
    src = "#gene name=feeder\nATG GAA TAA\n#end\n#config ticks=1"
    vm_obj, trace = run_src(src, ticks=1)
    assert trace[-1]["energy"] == pytest.approx(INITIAL_CELL_ENERGY + 37.0)


def test_vm_routes_grn_through_step_accel_when_use_accel(monkeypatch):
    """doc/37 §3.4: with use_accel on the VM must advance the GRN through the
    accelerated hot-loop kernel (bit-identical to scalar step), not the scalar
    Python fallback."""
    src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=3"
    stop = {c for c, op in STANDARD_TABLE.items() if op == Op.OP_HALT}
    toks = list(Lexer(src).tokens())
    prog = Parser(toks, stop_codons=stop).parse()
    SemanticAnalyzer(prog).check()
    chunk = Compiler(STANDARD_TABLE).compile(prog)

    calls = {"accel": 0, "prefer": []}
    vm = CellVM(chunk, prog)
    vm.use_accel = True
    orig = vm.grn.__class__.step_accel
    monkeypatch.setattr(
        vm.grn.__class__, "step_accel",
        lambda self, prefer=None: (
            calls.__setitem__("accel", calls["accel"] + 1)
            or calls["prefer"].append(prefer)
            or orig(self, prefer=prefer)))
    vm.run(3)

    assert calls["accel"] > 0, "use_accel path must route the GRN through step_accel"
    # doc/37 §3.4: with realism fully enforced (skip_validity=False, the
    # default) the VM must pin the byte-identical python kernel so traces and
    # goldens never drift a ULP.
    assert calls["prefer"] and all(p == "python" for p in calls["prefer"])

    # With accel disabled, the accelerated kernel is never entered.
    calls2 = {"accel": 0}
    vm2 = CellVM(chunk, prog)
    vm2.use_accel = False
    monkeypatch.setattr(
        vm2.grn.__class__, "step_accel",
        lambda self, prefer=None: (calls2.__setitem__("accel", calls2["accel"] + 1) or orig(self, prefer=prefer)))
    vm2.run(3)
    assert calls2["accel"] == 0, "non-accel path must use scalar step()"

    # With skip_validity=True the VM may use the fast native kernel (prefer=None).
    calls3 = {"accel": 0, "prefer": []}
    vm3 = CellVM(chunk, prog)
    vm3.use_accel = True
    vm3.skip_validity = True
    monkeypatch.setattr(
        vm3.grn.__class__, "step_accel",
        lambda self, prefer=None: (
            calls3.__setitem__("accel", calls3["accel"] + 1)
            or calls3["prefer"].append(prefer)
            or orig(self, prefer=prefer)))
    vm3.run(3)
    assert calls3["accel"] > 0
    assert all(p is None for p in calls3["prefer"])


# ============================================================================
# Targeted coverage: dispatch body (ADD/SUB/MUL) via non-accel path
# ============================================================================

class TestDispatchBody:
    """Exercise dispatch() bodies for ADD/SUB/MUL that are bypassed by the C
    accelerator when use_accel=True."""

    def test_add_dispatch_body(self):
        c = Chunk()
        c.add_constant(3)
        c.add_constant(4)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_PUSH_CONST, 1)
        c.emit(Op.OP_ADD)
        vm = _make_vm(c)
        vm.use_accel = False
        vm._execute_pending()
        assert vm.stack == [7]

    def test_sub_dispatch_body(self):
        c = Chunk()
        c.add_constant(10)
        c.add_constant(3)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_PUSH_CONST, 1)
        c.emit(Op.OP_SUB)
        vm = _make_vm(c)
        vm.use_accel = False
        vm._execute_pending()
        assert vm.stack == [7]

    def test_mul_dispatch_body(self):
        c = Chunk()
        c.add_constant(6)
        c.add_constant(7)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_PUSH_CONST, 1)
        c.emit(Op.OP_MUL)
        vm = _make_vm(c)
        vm.use_accel = False
        vm._execute_pending()
        assert vm.stack == [42]


# ============================================================================
# OP_USE_PLUGIN dispatch
# ============================================================================

class TestUsePluginDispatch:
    """OP_USE_PLUGIN reads a constant index and activates the named plugin."""

    def test_use_plugin_activates_registered_plugin(self):
        from helixlang.core.plugin_registry import PluginProvider, Registry
        c = Chunk()
        idx = c.add_constant(("use_plugin", "grn", ()))
        c.emit(Op.OP_USE_PLUGIN, idx)
        prog = Program(config=Config(ops_per_tick=256))
        registry = Registry()
        provider = PluginProvider(name="grn", extra="grn", load=lambda: None)
        registry.register(provider)
        vm = CellVM(c, prog, registry=registry)
        vm.frames.append(Frame(return_ip=len(c.code), gene_name="t"))
        vm.ip = 0
        vm._execute_pending()
        assert "grn" in registry.active()

    def test_use_plugin_spec_none_noop(self):
        c = Chunk()
        c.emit(Op.OP_USE_PLUGIN, 0)
        prog = Program(config=Config(ops_per_tick=256))
        vm = CellVM(c, prog)
        vm.frames.append(Frame(return_ip=len(c.code), gene_name="t"))
        vm.ip = 0
        vm._execute_pending()

    def test_use_plugin_idx_out_of_range(self):
        c = Chunk()
        c.emit(Op.OP_USE_PLUGIN, 99)
        prog = Program(config=Config(ops_per_tick=256))
        vm = CellVM(c, prog)
        vm.frames.append(Frame(return_ip=len(c.code), gene_name="t"))
        vm.ip = 0
        vm._execute_pending()


# ============================================================================
# OP_REGULATE source fallback
# ============================================================================

class TestRegulateSourceFallback:
    """When the current gene name is not in GRN nodes, source falls back to names[0]."""

    def test_regulate_source_not_in_nodes(self):
        c = Chunk()
        c.emit(Op.OP_REGULATE, 0)
        vm = _make_vm(c)
        vm.grn.add_gene("x", threshold=0.5, initial_level=1.0)
        vm.grn.add_gene("y", threshold=0.5, initial_level=0.0)
        vm.frames[-1].gene_name = "nonexistent"
        vm._execute_pending()
        assert len(vm.grn.edges) == 1
        assert vm.grn.edges[0].source == "x"

    def test_regulate_no_frames_fallback(self):
        c = Chunk()
        c.emit(Op.OP_REGULATE, 0)
        vm = _make_vm(c)
        vm.grn.add_gene("x", threshold=0.5, initial_level=1.0)
        vm.frames.clear()
        vm._dispatcher.dispatch(Op.OP_REGULATE)
        assert len(vm.grn.edges) == 1
        assert vm.grn.edges[0].source == "x"


# ============================================================================
# Bio instruction handlers
# ============================================================================

class TestBioInstructionHandlers:
    """Test all bio instruction handlers via process_bio_instructions."""

    def _make_vm_with_bio(self, bio_instructions, **kwargs):
        from helixlang.core.ast_nodes import Config, Program
        chunk = Chunk()
        chunk.emit(Op.OP_NOP)
        prog = Program(config=Config(ops_per_tick=256), **kwargs)
        prog.bio_instructions = bio_instructions
        vm = CellVM(chunk, prog)
        return vm

    # -- crispr --
    def test_crispr_edit_gene(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="crispr", target="g1",
                           params={"position": "0", "new_sequence": "ATG",
                                   "cas": "SpCas9"})
        ])
        vm._gene_dna["g1"] = "ATGGCTGGTAAAGGCATCGAT" * 2
        vm._dispatcher.process_bio_instructions()
        assert len(vm._crispr_edits) == 1
        assert vm._crispr_edits[0]["target"] == "g1"

    def test_crispr_empty_dna_early_return(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="crispr", target="g1", params={})
        ])
        vm._dispatcher.process_bio_instructions()
        assert vm._crispr_edits == []

    def test_crispr_error_path(self, monkeypatch):
        from helixlang.core.ast_nodes import BioInstruction
        from helixlang.plugins.runtime import crispr as _crispr
        vm = self._make_vm_with_bio([
            BioInstruction(kind="crispr", target="g1",
                           params={"position": "0", "new_sequence": "CCC",
                                   "cas": "SpCas9"})
        ])
        vm._gene_dna["g1"] = "ATGGCTGGTAAAGGC"
        monkeypatch.setattr(_crispr, "edit_gene",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                ValueError("mock edit error")))
        vm._dispatcher.process_bio_instructions()
        assert len(vm._crispr_edits) == 1
        assert vm._crispr_edits[0]["success"] is False
        assert "error" in vm._crispr_edits[0]

    def test_crispr_gem_dirty(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="crispr", target="g1",
                           params={"position": "0", "new_sequence": "ATG",
                                   "cas": "SpCas9"})
        ])
        vm._gene_dna["g1"] = "ATGGCTGGTAAAGGCATCGAT" * 2
        vm._gem_gpr_map["g1"] = ["rxn1"]
        vm._dispatcher.process_bio_instructions()
        edit = vm._crispr_edits[0]
        if edit.get("success"):
            assert vm._gem_dirty is True

    def test_crispr_gem_dirty_no_success(self, monkeypatch):
        from helixlang.core.ast_nodes import BioInstruction
        from helixlang.plugins.runtime import crispr as _crispr
        vm = self._make_vm_with_bio([
            BioInstruction(kind="crispr", target="g1",
                           params={"position": "0", "new_sequence": "CCC",
                                   "cas": "SpCas9"})
        ])
        vm._gene_dna["g1"] = "ATGGCTGGTAAAGGC"
        vm._gem_gpr_map["g1"] = ["rxn1"]
        monkeypatch.setattr(_crispr, "edit_gene",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                ValueError("mock edit error")))
        vm._dispatcher.process_bio_instructions()
        assert vm._gem_dirty is False

    # -- evolve --
    def test_evolve_mutate_gene(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="evolve", target="g1",
                           params={"mutation_rate": "0.5", "indel_rate": "0.1"})
        ])
        vm._gene_dna["g1"] = "ATGGCTGGTAAAGGCATCGAT"
        vm._dispatcher.process_bio_instructions()
        assert len(vm._evolution_history) == 1
        assert vm._evolution_history[0]["target"] == "g1"
        assert vm._evolution_history[0]["mutations"] >= 0

    def test_evolve_empty_dna_early_return(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="evolve", target="g1", params={})
        ])
        vm._dispatcher.process_bio_instructions()
        assert vm._evolution_history == []

    def test_evolve_gem_dirty(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="evolve", target="g1",
                           params={"mutation_rate": "0.5", "indel_rate": "0.1"})
        ])
        vm._gene_dna["g1"] = "ATGGCTGGTAAAGGCATCGAT"
        vm._gem_gpr_map["g1"] = ["rxn1"]
        vm._dispatcher.process_bio_instructions()
        hist = vm._evolution_history[0]
        if hist["mutations"] > 0:
            assert vm._gem_dirty is True

    def test_evolve_gem_dirty_no_mutations(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="evolve", target="g1",
                           params={"mutation_rate": "0.0", "indel_rate": "0.0"})
        ])
        vm._gene_dna["g1"] = "ATGGCTGGTAAAGGCATCGAT"
        vm._gem_gpr_map["g1"] = ["rxn1"]
        vm._dispatcher.process_bio_instructions()
        assert vm._evolution_history[0]["mutations"] == 0
        assert vm._gem_dirty is False

    # -- _update_enzyme_levels_from_edits --
    def test_update_enzyme_levels_substitution(self):
        vm = self._make_vm_with_bio([])
        vm._crispr_edits = [
            {"success": True, "target": "g1", "edit_type": "substitution"}
        ]
        vm._gem_gpr_map["g1"] = ["rxn1"]
        vm._enzyme_kcat["rxn1"] = 1.0
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert vm._enzyme_kcat["rxn1"] == 0.7
        assert vm._gem_dirty is False

    def test_update_enzyme_levels_frameshift(self):
        vm = self._make_vm_with_bio([])
        vm._crispr_edits = [
            {"success": True, "target": "g1", "edit_type": "frameshift"}
        ]
        vm._gem_gpr_map["g1"] = ["rxn1"]
        vm._enzyme_kcat["rxn1"] = 1.0
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert vm._enzyme_kcat["rxn1"] == 0.0

    def test_update_enzyme_levels_deletion(self):
        vm = self._make_vm_with_bio([])
        vm._crispr_edits = [
            {"success": True, "target": "g1", "edit_type": "deletion"}
        ]
        vm._gem_gpr_map["g1"] = ["rxn1"]
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert vm._enzyme_kcat["rxn1"] == 0.0

    def test_update_enzyme_levels_nonsense(self):
        vm = self._make_vm_with_bio([])
        vm._crispr_edits = [
            {"success": True, "target": "g1", "edit_type": "nonsense"}
        ]
        vm._gem_gpr_map["g1"] = ["rxn1"]
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert vm._enzyme_kcat["rxn1"] == 0.0

    def test_update_enzyme_levels_failed_edit(self):
        vm = self._make_vm_with_bio([])
        vm._crispr_edits = [
            {"success": False, "target": "g1", "edit_type": "substitution"}
        ]
        vm._gem_gpr_map["g1"] = ["rxn1"]
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert "rxn1" not in vm._enzyme_kcat

    def test_update_enzyme_levels_no_reactions(self):
        vm = self._make_vm_with_bio([])
        vm._crispr_edits = [
            {"success": True, "target": "g1", "edit_type": "substitution"}
        ]
        vm._gem_gpr_map["g1"] = []
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert vm._gem_dirty is False

    def test_update_enzyme_levels_substitution_default_kcat(self):
        vm = self._make_vm_with_bio([])
        vm._crispr_edits = [
            {"success": True, "target": "g1", "edit_type": "substitution"}
        ]
        vm._gem_gpr_map["g1"] = ["rxn_new"]
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert vm._enzyme_kcat["rxn_new"] == 1.0 * 0.7

    # -- methylate --
    def test_methylate_gene(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="methylate", target="g1",
                           params={"methylase": "dam"})
        ])
        vm._gene_dna["g1"] = "ATGGCTGGTAAAGGCATCGAT"
        vm._dispatcher.process_bio_instructions()
        assert len(vm._epigenetic_marks) == 1
        assert vm._epigenetic_marks[0]["type"] == "methylation"
        assert vm._epigenetic_marks[0]["target"] == "g1"

    def test_methylate_empty_dna_early_return(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="methylate", target="g1", params={})
        ])
        vm._dispatcher.process_bio_instructions()
        assert vm._epigenetic_marks == []

    def test_methylate_repression_applied(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="methylate", target="g1",
                           params={"methylase": "dam"})
        ])
        vm._gene_dna["g1"] = "ATGGCTGGTAAAGGCATCGAT"
        vm._dispatcher.process_bio_instructions()
        assert "g1" in vm._chromatin_modifier

    # -- histone --
    def test_histone_modification(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="histone", target="g1",
                           params={"mark": "H3K4me3"})
        ])
        vm._dispatcher.process_bio_instructions()
        assert len(vm._epigenetic_marks) == 1
        assert vm._epigenetic_marks[0]["type"] == "histone"
        assert vm._epigenetic_marks[0]["mark"] == "H3K4me3"
        assert vm._epigenetic_marks[0]["score"] == 0.5
        assert "g1" in vm._chromatin_modifier

    def test_histone_repressive_mark(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="histone", target="g1",
                           params={"mark": "H3K27me3"})
        ])
        vm._dispatcher.process_bio_instructions()
        assert vm._chromatin_modifier["g1"] < 1.0

    def test_histone_unknown_mark(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="histone", target="g1",
                           params={"mark": "UNKNOWN_MARK"})
        ])
        vm._dispatcher.process_bio_instructions()
        assert vm._epigenetic_marks[0]["score"] == 0.0

    # -- quorum --
    def test_quorum_sensing_activates(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="quorum", target="g1",
                           params={"threshold": "0.5", "activate": "g1"})
        ])
        vm.grn.add_gene("g1", threshold=0.5, initial_level=0.0)
        vm.field = GrayScott(n=8)
        vm.field.emit(vm.cell.x % 8, vm.cell.y % 8, 1.0)
        vm._dispatcher.process_bio_instructions()
        assert vm.grn.nodes["g1"].level >= 1.0

    def test_quorum_signal_below_threshold(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="quorum", target="g1",
                           params={"threshold": "100.0"})
        ])
        vm.grn.add_gene("g1", threshold=0.5, initial_level=0.0)
        vm._dispatcher.process_bio_instructions()
        assert vm.grn.nodes["g1"].level == 0.0

    def test_quorum_no_field(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="quorum", target="g1",
                           params={"threshold": "1.0"})
        ])
        vm.grn.add_gene("g1", threshold=0.5, initial_level=0.0)
        vm._dispatcher.process_bio_instructions()
        assert vm.grn.nodes["g1"].level == 0.0

    def test_quorum_activate_not_in_grn(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="quorum", target="nonexistent",
                           params={"threshold": "0.0", "activate": "nonexistent"})
        ])
        vm._dispatcher.process_bio_instructions()

    # -- transcribe --
    def test_transcribe_sets_level(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="transcribe", target="g1", params={})
        ])
        vm.grn.add_gene("g1", threshold=0.5, initial_level=0.0)
        vm._dispatcher.process_bio_instructions()
        assert vm.grn.nodes["g1"].level == 1.0

    def test_transcribe_target_not_in_grn(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="transcribe", target="nonexistent", params={})
        ])
        vm._dispatcher.process_bio_instructions()

    # -- translate --
    def test_translate_increases_protein(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="translate", target="g1", params={})
        ])
        vm.cell.add_protein("g1", 1.0)
        vm._dispatcher.process_bio_instructions()
        assert vm.cell.proteins["g1"] == 2.0

    def test_translate_no_existing_protein(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="translate", target="nonexistent", params={})
        ])
        vm._dispatcher.process_bio_instructions()

    def test_unknown_bio_instruction_noop(self):
        from helixlang.core.ast_nodes import BioInstruction
        vm = self._make_vm_with_bio([
            BioInstruction(kind="unknown_kind", target="g1", params={})
        ])
        vm._dispatcher.process_bio_instructions()


# ============================================================================
# _use_plugin
# ============================================================================

class TestUsePluginMethod:
    """_use_plugin resolves/activates a plugin through the registry."""

    def test_use_plugin_with_registry(self):
        from helixlang.core.plugin_registry import PluginProvider, Registry
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256))
        registry = Registry()
        provider = PluginProvider(name="grn", extra="grn", load=lambda: None)
        registry.register(provider)
        vm = CellVM(c, prog, registry=registry)
        vm._use_plugin("grn", ())
        assert "grn" in registry.active()

    def test_use_plugin_with_flags(self):
        from helixlang.core.plugin_registry import PluginProvider, Registry
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256))
        registry = Registry()
        provider = PluginProvider(name="grn", extra="grn",
                                 capability_flags=("--low-fidelity",),
                                 load=lambda: None)
        registry.register(provider)
        vm = CellVM(c, prog, registry=registry)
        vm._use_plugin("grn", ("--low-fidelity",))
        assert registry.has_capability("--low-fidelity")
        assert "grn" in registry.active()

    def test_use_plugin_uses_default_registry(self):
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256))
        vm = CellVM(c, prog)
        vm._use_plugin("grn", ())
        from helixlang.core.plugin_registry import get_registry
        assert "grn" in get_registry().active()


# ============================================================================
# _get_promoter_strength
# ============================================================================

class TestGetPromoterStrength:
    def test_none_returns_constitutive(self):
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256))
        vm = CellVM(c, prog)
        from helixlang.core.opcode_semantics import CONSTITUTIVE_PROMOTER_STRENGTH
        assert vm._get_promoter_strength(None) == CONSTITUTIVE_PROMOTER_STRENGTH

    def test_known_promoter(self):
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256))
        vm = CellVM(c, prog)
        vm._promoter_strengths["p1"] = 0.8
        assert vm._get_promoter_strength("p1") == 0.8

    def test_unknown_promoter_returns_constitutive(self):
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256))
        vm = CellVM(c, prog)
        from helixlang.core.opcode_semantics import CONSTITUTIVE_PROMOTER_STRENGTH
        assert vm._get_promoter_strength("unknown") == CONSTITUTIVE_PROMOTER_STRENGTH


# ============================================================================
# _call_gene depth cap
# ============================================================================

class TestCallGeneDepthCap:
    def test_frame_depth_cap(self):
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=1024))
        vm = CellVM(c, prog)
        vm.chunk.gene_offsets["g"] = 0
        for _ in range(256):
            vm.frames.append(Frame(return_ip=0, gene_name="t"))
        vm._call_gene("g")
        assert len(vm.frames) == 256


# ============================================================================
# _execute_pending non-accel path
# ============================================================================

class TestExecutePendingNonAccel:
    """Test the Python dispatch loop in _execute_pending (use_accel=False)."""

    def test_ip_exceeds_code_pops_frame(self):
        c = Chunk()
        c.emit(Op.OP_NOP)
        vm = _make_vm(c)
        vm.use_accel = False
        vm.ip = len(c.code) + 1
        vm._execute_pending()
        assert vm.frames == [] or vm.ip <= len(c.code)

    def test_frame_depth_exceeds_256(self):
        c = Chunk()
        c.emit(Op.OP_NOP)
        vm = _make_vm(c)
        vm.use_accel = False
        for _ in range(300):
            vm.frames.append(Frame(return_ip=0, gene_name="t"))
        vm._execute_pending()
        assert len(vm.frames) == 0

    def test_quota_exhausted(self):
        c = Chunk()
        c.emit(Op.OP_NOP)
        c.emit(Op.OP_NOP)
        c.emit(Op.OP_NOP)
        vm = _make_vm(c, ops_per_tick=1)
        vm.use_accel = False
        vm._execute_pending()
        assert vm.ops_executed <= 2

    def test_debug_prints(self, capsys):
        c = Chunk()
        c.emit(Op.OP_DEBUG)
        vm = _make_vm(c)
        vm.use_accel = False
        vm.debug = True
        vm._execute_pending()
        out = capsys.readouterr().out
        assert "DEBUG" in out

    def test_unknown_opcode_in_non_accel(self):
        from helixlang.core.errors import UnknownOpcodeError
        c = Chunk()
        c.code.append(0xFF)
        c.lines.append(0)
        c.codon_indices.append(-1)
        vm = _make_vm(c)
        vm.use_accel = False
        with pytest.raises(UnknownOpcodeError):
            vm._execute_pending()

    def test_dispatch_body_executed(self):
        c = Chunk()
        c.add_constant(3)
        c.add_constant(4)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_PUSH_CONST, 1)
        c.emit(Op.OP_ADD)
        vm = _make_vm(c)
        vm.use_accel = False
        vm._execute_pending()
        assert vm.stack == [7]

    def test_dispatch_body_sub(self):
        c = Chunk()
        c.add_constant(10)
        c.add_constant(3)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_PUSH_CONST, 1)
        c.emit(Op.OP_SUB)
        vm = _make_vm(c)
        vm.use_accel = False
        vm._execute_pending()
        assert vm.stack == [7]

    def test_dispatch_body_mul(self):
        c = Chunk()
        c.add_constant(6)
        c.add_constant(7)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_PUSH_CONST, 1)
        c.emit(Op.OP_MUL)
        vm = _make_vm(c)
        vm.use_accel = False
        vm._execute_pending()
        assert vm.stack == [42]


# ============================================================================
# _read_u8 / _read_u16 edge cases
# ============================================================================

class TestReadOperands:
    def test_read_u8_at_end(self):
        c = Chunk()
        c.emit(Op.OP_NOP)
        vm = _make_vm(c)
        vm.ip = len(c.code)
        assert vm._read_u8() == 0

    def test_read_u16_at_end(self):
        c = Chunk()
        c.emit(Op.OP_NOP)
        vm = _make_vm(c)
        vm.ip = len(c.code)
        assert vm._read_u16() == 0

    def test_read_u16_one_byte_left(self):
        c = Chunk()
        c.emit(Op.OP_NOP)
        vm = _make_vm(c)
        vm.ip = len(c.code) - 1
        assert vm._read_u16() == 0

    def test_read_u16_normal(self):
        c = Chunk()
        vm = _make_vm(c)
        c.code.extend([0x01, 0x02])
        c.lines.extend([0, 0])
        c.codon_indices.extend([-1, -1])
        vm.ip = len(c.code) - 2
        val = vm._read_u16()
        assert val == (0x01 << 8) | 0x02

    def test_read_u8_normal(self):
        c = Chunk()
        vm = _make_vm(c)
        c.code.append(42)
        c.lines.append(0)
        c.codon_indices.append(-1)
        vm.ip = len(c.code) - 1
        assert vm._read_u8() == 42


# ============================================================================
# _divide when cell can't divide
# ============================================================================

class TestDivideEdgeCases:
    def test_divide_insufficient_energy(self):
        from helixlang.plugins.runtime.cell import MIN_DIVISION_ENERGY, Cell
        c = Chunk()
        c.emit(Op.OP_DIVIDE, 0)
        vm = _make_vm(c)
        vm.cell = Cell(energy=MIN_DIVISION_ENERGY - 1)
        vm._execute_pending()
        assert vm.cell.divisions == 0
        assert len(vm.daughters) == 0


# ============================================================================
# _feedback morphogen wiring
# ============================================================================

class TestFeedbackMorphogen:
    def test_feedback_with_morphogen_wiring(self):
        c = Chunk()
        c.emit(Op.OP_NOP)
        prog = Program(config=Config(ops_per_tick=256))
        from helixlang.core.ast_nodes import MorphogenFeedback
        prog.morphogen_feedback = [
            MorphogenFeedback(gene="g1", channel="U", gain=0.5)
        ]
        vm = CellVM(c, prog)
        vm.grn.add_gene("g1", threshold=0.5, initial_level=0.0)
        vm.field = GrayScott(n=8)
        vm.field.emit(0, 0, 1.0)
        vm._feedback()
        assert vm.grn.nodes["g1"].level > 0.0

    def test_feedback_with_v_channel(self):
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256))
        from helixlang.core.ast_nodes import MorphogenFeedback
        prog.morphogen_feedback = [
            MorphogenFeedback(gene="g1", channel="V", gain=0.5)
        ]
        vm = CellVM(c, prog)
        vm.grn.add_gene("g1", threshold=0.5, initial_level=0.0)
        vm.field = GrayScott(n=8)
        vm.field.emit(0, 0, 1.0)
        vm._feedback()
        assert vm.grn.nodes["g1"].level > 0.0

    def test_feedback_gene_not_in_grn(self):
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256))
        from helixlang.core.ast_nodes import MorphogenFeedback
        prog.morphogen_feedback = [
            MorphogenFeedback(gene="nonexistent", channel="U", gain=0.5)
        ]
        vm = CellVM(c, prog)
        vm.field = GrayScott(n=8)
        vm._feedback()

    def test_feedback_pigment_gene(self):
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256))
        vm = CellVM(c, prog)
        vm.grn.add_gene("pigment", threshold=0.5, initial_level=0.0)
        vm.field = GrayScott(n=8)
        vm.field.emit(0, 0, 1.0)
        vm._feedback()
        assert vm.grn.nodes["pigment"].level > 0.0

    def test_feedback_no_pigment_gene(self):
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256))
        vm = CellVM(c, prog)
        vm.field = GrayScott(n=8)
        vm._feedback()


# ============================================================================
# _snapshot downsampling skip
# ============================================================================

class TestSnapshotDownsampling:
    def test_snapshot_skipped_when_downsampled(self):
        c = Chunk()
        c.emit(Op.OP_NOP)
        prog = Program(config=Config(ops_per_tick=256, ticks=10000))
        vm = CellVM(c, prog)
        vm._snapshot_downsampler.configure(10000)
        vm.cell.alive = True
        vm.tick = 501
        before = len(vm.trace)
        vm._snapshot()
        after = len(vm.trace)
        assert before == after


# ============================================================================
# _transcribe_translate paths
# ============================================================================

class TestTranscribeTranslate:
    def test_central_dogma_with_protein_result(self):
        src = ("#gene name=g\nATG GCT TAA\n#end\n"
               "#config ticks=1 use_central_dogma=true species=ecoli")
        vm, trace = run_src(src, ticks=1)
        assert trace[-1]["alive"]

    def test_gene_with_no_dna_skipped(self):
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256, use_central_dogma=True,
                                     species="ecoli"))
        vm = CellVM(c, prog)
        from helixlang.core.ast_nodes import Gene
        prog.genes.append(Gene(name="empty_gene", promoter=None,
                               codons=[], orf=[]))
        vm._gene_dna["empty_gene"] = ""
        vm._transcribe_translate()

    def test_transcribe_translate_with_accel(self):
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256, use_central_dogma=True,
                                     species="ecoli"))
        vm = CellVM(c, prog)
        from helixlang.core.ast_nodes import Codon, Gene
        prog.genes.append(Gene(name="g", promoter=None,
                               codons=[Codon("ATG", 0, 1),
                                       Codon("GCT", 1, 1),
                                       Codon("TAA", 2, 1)],
                               orf=[Codon("ATG", 0, 1),
                                    Codon("GCT", 1, 1),
                                    Codon("TAA", 2, 1)]))
        vm._gene_dna["g"] = "ATGGCTTAA"
        vm.use_accel = True
        vm._transcribe_translate()

    def test_transcribe_translate_no_accel(self):
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256, use_central_dogma=True,
                                     species="ecoli"))
        vm = CellVM(c, prog)
        from helixlang.core.ast_nodes import Codon, Gene
        prog.genes.append(Gene(name="g", promoter=None,
                               codons=[Codon("ATG", 0, 1),
                                       Codon("GCT", 1, 1),
                                       Codon("TAA", 2, 1)],
                               orf=[Codon("ATG", 0, 1),
                                    Codon("GCT", 1, 1),
                                    Codon("TAA", 2, 1)]))
        vm._gene_dna["g"] = "ATGGCTTAA"
        vm.use_accel = False
        vm._transcribe_translate()

    def test_transcribe_translate_with_promoter(self):
        src = ("#promoter name=p strength=0.8\n"
               "#gene name=g promoter=p\nATG GCT TAA\n#end\n"
               "#config ticks=1 use_central_dogma=true species=ecoli")
        vm, trace = run_src(src, ticks=1)
        assert trace[-1]["alive"]


# ============================================================================
# GEM re-solve paths (run method)
# ============================================================================

class TestGemReSolve:
    def test_gem_re_solve_central_dogma(self):
        """Central dogma path with GEM dirty and metabolic model."""
        from helixlang.plugins.runtime.metabolism import (
            MetabolicModel,
            Reaction,
        )
        c = Chunk()
        c.emit(Op.OP_NOP)
        prog = Program(config=Config(ops_per_tick=256, ticks=1,
                                     use_central_dogma=True, species="ecoli"))
        vm = CellVM(c, prog)
        model = MetabolicModel()
        rxn = Reaction(id="biomass_r", name="biomass",
                       stoichiometry={"biomass_met": 1.0})
        model.add_reaction(rxn)
        model.set_biomass("biomass_r")
        vm._metabolic_model = model
        vm._gem_gpr_map["g1"] = ["biomass_r"]
        vm._enzyme_kcat["biomass_r"] = 1.0
        vm._gem_dirty = True
        vm.run(1)

    def test_gem_re_solve_central_dogma_no_enzyme_kcat(self):
        from helixlang.plugins.runtime.metabolism import MetabolicModel, Reaction
        c = Chunk()
        c.emit(Op.OP_NOP)
        prog = Program(config=Config(ops_per_tick=256, ticks=1,
                                     use_central_dogma=True, species="ecoli"))
        vm = CellVM(c, prog)
        model = MetabolicModel()
        rxn = Reaction(id="biomass_r", name="biomass",
                       stoichiometry={"biomass_met": 1.0})
        model.add_reaction(rxn)
        model.set_biomass("biomass_r")
        vm._metabolic_model = model
        vm._gem_dirty = True
        vm._enzyme_kcat = {}
        vm.run(1)

    def test_gem_error_central_dogma(self):
        """Central dogma GEM re-solve error path."""
        from helixlang.plugins.runtime.metabolism import MetabolicModel
        c = Chunk()
        c.emit(Op.OP_NOP)
        prog = Program(config=Config(ops_per_tick=256, ticks=1,
                                     use_central_dogma=True, species="ecoli"))
        vm = CellVM(c, prog)
        model = MetabolicModel()
        vm._metabolic_model = model
        vm._gem_dirty = True
        vm._enzyme_kcat["rxn1"] = 1.0
        with pytest.raises(ModelMissingError):
            vm.run(1)

    def test_gem_re_solve_non_central_dogma(self):
        """Non-central-dogma path with GEM dirty."""
        from helixlang.plugins.runtime.metabolism import MetabolicModel, Reaction
        c = Chunk()
        c.emit(Op.OP_NOP)
        prog = Program(config=Config(ops_per_tick=256, ticks=1,
                                     use_central_dogma=False))
        vm = CellVM(c, prog)
        model = MetabolicModel()
        rxn = Reaction(id="biomass_r", name="biomass",
                       stoichiometry={"biomass_met": 1.0})
        model.add_reaction(rxn)
        model.set_biomass("biomass_r")
        vm._metabolic_model = model
        vm._gem_gpr_map["g1"] = ["biomass_r"]
        vm._enzyme_kcat["biomass_r"] = 1.0
        vm._gem_dirty = True
        vm.run(1)

    def test_gem_error_non_central_dogma(self):
        """Non-central-dogma GEM re-solve error path."""
        from helixlang.plugins.runtime.metabolism import MetabolicModel
        c = Chunk()
        c.emit(Op.OP_NOP)
        prog = Program(config=Config(ops_per_tick=256, ticks=1,
                                     use_central_dogma=False))
        vm = CellVM(c, prog)
        model = MetabolicModel()
        vm._metabolic_model = model
        vm._gem_dirty = True
        vm._enzyme_kcat["rxn1"] = 1.0
        with pytest.raises(ModelMissingError):
            vm.run(1)


# ============================================================================
# _init_subsystems with use directives
# ============================================================================

class TestInitSubsystems:
    def test_use_directives_processed(self):
        from helixlang.core.ast_nodes import UseDecl
        from helixlang.core.plugin_registry import PluginProvider, Registry
        src = ("#gene name=g\nATG GCT TAA\n#end\n"
               "#config ticks=1")
        prog = Parser(list(Lexer(src).tokens()),
                      stop_codons={c for c, op in STANDARD_TABLE.items()
                                   if op == Op.OP_HALT}).parse()
        SemanticAnalyzer(prog).check()
        prog.use_directives = [UseDecl(plugin="test_plugin")]
        chunk = Compiler(STANDARD_TABLE).compile(prog)
        registry = Registry()
        provider = PluginProvider(name="test_plugin", extra="core",
                                  load=lambda: None)
        registry.register(provider)
        CellVM(chunk, prog, registry=registry)
        assert "test_plugin" in registry.active()

    def test_morphogen_feedback_init(self):
        src = ("#gene name=g\nATG GCT TAA\n#end\n"
               "#morphogen gene=g channel=V gain=0.2\n"
               "#config ticks=1")
        vm, trace = run_src(src, ticks=1)
        assert "g" in vm._morphogen_feedback

    def test_no_morphogen_feedback(self):
        src = ("#gene name=g\nATG GCT TAA\n#end\n"
               "#config ticks=1")
        vm, trace = run_src(src, ticks=1)
        assert vm._morphogen_feedback == {}

    def test_lsystems_init(self):
        src = ("#gene name=g\nATG GCT TAA\n#end\n"
               "#lsystem name=tree axiom=F rules=0:F->F[+F]F angle=25 step=1.0\n"
               "#config ticks=1")
        vm, trace = run_src(src, ticks=1)
        assert "tree" in vm.lsystems

    def test_promoter_negative_strength_constitutive(self):
        src = ("#promoter name=p strength=-0.5\n"
               "#gene name=g promoter=p\nATG GCT TAA\n#end\n"
               "#config ticks=1")
        vm, trace = run_src(src, ticks=1)
        assert vm.grn.nodes["p"].level > 0.9

    def test_gene_with_promoter_zero_initial(self):
        src = ("#promoter name=p strength=0.5\n"
               "#gene name=g promoter=p\nATG GCT TAA\n#end\n"
               "#config ticks=1")
        vm, trace = run_src(src, ticks=1)
        assert vm.grn.nodes["g"].level < 0.5

    def test_gene_no_promoter_constitutive(self):
        src = ("#gene name=g\nATG GCT TAA\n#end\n"
               "#config ticks=1")
        vm, trace = run_src(src, ticks=1)
        assert vm.grn.nodes["g"].level > 0.9

    def test_regulations_init(self):
        src = ("#promoter name=p strength=0.5\n"
               "#gene name=g1 promoter=p\nATG GCT TAA\n#end\n"
               "#gene name=g2\nATG GCT TAA\n#end\n"
               "#regulate g1 -> g2 strength=0.8\n"
               "#config ticks=1")
        vm, trace = run_src(src, ticks=1)
        assert len(vm.grn.edges) == 1


# ============================================================================
# _dispatch (thin wrapper) ops_executed counter
# ============================================================================

class TestDispatchWrapper:
    def test_ops_executed_increments(self):
        c = Chunk()
        c.emit(Op.OP_NOP)
        vm = _make_vm(c)
        vm.use_accel = False
        before = vm.ops_executed
        vm._execute_pending()
        assert vm.ops_executed > before

    def test_dispatch_wraps_to_dispatcher(self):
        c = Chunk()
        c.emit(Op.OP_NOP)
        vm = _make_vm(c)
        vm.use_accel = False
        vm._dispatch(Op.OP_NOP)


# ============================================================================
# _execute_pending with remaining frames after quota
# ============================================================================

class TestExecutePendingEdgeCases:
    def test_frames_empty_stops(self):
        c = Chunk()
        vm = _make_vm(c)
        vm.use_accel = False
        vm.frames.clear()
        vm._execute_pending()

    def test_non_accel_halts_resume(self):
        """HALT in non-accel path pops frame and resumes."""
        c = Chunk()
        c.emit(Op.OP_HALT)
        c.emit(Op.OP_NOP)
        vm = _make_vm(c)
        vm.use_accel = False
        vm.frames.append(Frame(return_ip=1, gene_name="outer"))
        vm._execute_pending()
        assert vm.ip == 1 or len(vm.frames) == 0


# ============================================================================
# GEM re-solve low-fidelity opt-in path
# ============================================================================

class TestGemLowFidelity:
    def test_gem_error_low_fidelity_optin_central_dogma(self, monkeypatch):
        from helixlang.core import fidelity
        monkeypatch.setattr(fidelity, "opt_in", lambda *a: True)
        from helixlang.plugins.runtime.metabolism import MetabolicModel
        c = Chunk()
        c.emit(Op.OP_NOP)
        prog = Program(config=Config(ops_per_tick=256, ticks=1,
                                     use_central_dogma=True, species="ecoli"))
        vm = CellVM(c, prog)
        model = MetabolicModel()
        vm._metabolic_model = model
        vm._gem_dirty = True
        vm._enzyme_kcat["rxn1"] = 1.0
        vm.run(1)

    def test_gem_error_low_fidelity_optin_non_central_dogma(self, monkeypatch):
        from helixlang.core import fidelity
        monkeypatch.setattr(fidelity, "opt_in", lambda *a: True)
        from helixlang.plugins.runtime.metabolism import MetabolicModel
        c = Chunk()
        c.emit(Op.OP_NOP)
        prog = Program(config=Config(ops_per_tick=256, ticks=1,
                                     use_central_dogma=False))
        vm = CellVM(c, prog)
        model = MetabolicModel()
        vm._metabolic_model = model
        vm._gem_dirty = True
        vm._enzyme_kcat["rxn1"] = 1.0
        vm.run(1)


# ============================================================================
# _get_transcription_factors
# ============================================================================

class TestGetTranscriptionFactors:
    def test_tf_with_regulations(self):
        src = ("#promoter name=p strength=0.5\n"
               "#gene name=g1 promoter=p\nATG GCT TAA\n#end\n"
               "#gene name=g2\nATG GCT TAA\n#end\n"
               "#regulate g1 -> g2 strength=0.8\n"
               "#config ticks=1")
        vm, trace = run_src(src, ticks=1)
        tfs = vm._get_transcription_factors("g2")
        assert "g1" in tfs

    def test_tf_no_regulations(self):
        src = ("#gene name=g\nATG GCT TAA\n#end\n"
               "#config ticks=1")
        vm, trace = run_src(src, ticks=1)
        tfs = vm._get_transcription_factors("g")
        assert tfs == {}

    def test_tf_source_level_zero(self):
        src = ("#promoter name=p strength=0.5\n"
               "#gene name=g1 promoter=p\nATG GCT TAA\n#end\n"
               "#gene name=g2\nATG GCT TAA\n#end\n"
               "#regulate g1 -> g2 strength=0.8\n"
               "#config ticks=1")
        vm, trace = run_src(src, ticks=1)
        vm.grn.nodes["g1"].level = 0.0
        tfs = vm._get_transcription_factors("g2")
        assert tfs["g1"] == 1.0


# ============================================================================
# _flush_morphology
# ============================================================================

class TestFlushMorphology:
    def test_flush_morphology_is_noop(self):
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256))
        vm = CellVM(c, prog)
        vm._flush_morphology()


# ============================================================================
# _process_bio_instructions delegation
# ============================================================================

class TestProcessBioInstructionsDelegation:
    def test_delegates_to_dispatcher(self):
        from helixlang.core.ast_nodes import BioInstruction
        c = Chunk()
        c.emit(Op.OP_NOP)
        prog = Program(config=Config(ops_per_tick=256))
        prog.bio_instructions = [
            BioInstruction(kind="transcribe", target="g1", params={})
        ]
        vm = CellVM(c, prog)
        vm.grn.add_gene("g1", threshold=0.5, initial_level=0.0)
        vm._process_bio_instructions()
        assert vm.grn.nodes["g1"].level == 1.0


# ============================================================================
# run() with accel + skip_validity
# ============================================================================

class TestRunWithAccel:
    def test_run_with_accel_and_skip_validity(self):
        src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=2"
        vm, trace = run_src(src, ticks=2)
        vm.use_accel = True
        vm.skip_validity = True
        trace2 = vm.run(2)
        assert len(trace2) > 0

    def test_run_non_central_dogma_grn_path(self):
        src = "#gene name=g\nATG GCT TAA\n#end\n#config ticks=2"
        vm, trace = run_src(src, ticks=2)
        assert trace[-1]["alive"]


# ============================================================================
# Additional branch coverage: BIND with protein as binder
# ============================================================================

class TestBindBranches:
    def _grn_vm(self, chunk):
        vm = _make_vm(chunk)
        vm.grn.add_gene("a", threshold=0.5, initial_level=1.0)
        vm.grn.add_gene("b", threshold=0.5, initial_level=0.0)
        vm.frames[-1].gene_name = "a"
        return vm

    def test_bind_binder_not_in_proteins_uses_fallback(self):
        c = Chunk()
        c.emit(Op.OP_BIND, 1)
        vm = self._grn_vm(c)
        vm.cell.add_protein("fallback", 2.0)
        vm._execute_pending()
        assert vm.grn.nodes["b"].level == 0.5
        assert vm._binding_events[0]["protein"] == "fallback"

    def test_bind_consumed_zero_no_event(self):
        c = Chunk()
        c.emit(Op.OP_BIND, 1)
        vm = self._grn_vm(c)
        vm.cell.add_protein("a", 0.001)
        vm.cell.consume_protein("a", 10.0)
        vm._execute_pending()
        assert vm._binding_events == []


# ============================================================================
# OP_SIGNAL with field at non-zero cell position
# ============================================================================

class TestSignalNonZeroPosition:
    def test_signal_at_offset_position(self):
        c = Chunk()
        c.emit(Op.OP_SIGNAL, 0)
        vm = _make_vm(c)
        vm.field = GrayScott(n=8)
        vm.cell.x = 3
        vm.cell.y = 5
        vm._execute_pending()
        assert vm._signal_emissions == 1
        assert vm.field.v[3][5] > 0


# ============================================================================
# OP_EMIT_MORPHOGEN with field at non-zero position
# ============================================================================

class TestEmitMorphogenNonZeroPosition:
    def test_emit_at_offset_position(self):
        c = Chunk()
        c.emit(Op.OP_EMIT_MORPHOGEN, 0)
        vm = _make_vm(c)
        vm.field = GrayScott(n=8)
        vm.cell.x = 3
        vm.cell.y = 5
        before = vm.field.v[3][5]
        vm._execute_pending()
        assert vm.field.v[3][5] > before


# ============================================================================
# _divide successful with daughter
# ============================================================================

class TestDivideSuccess:
    def test_divide_creates_daughter(self):
        c = Chunk()
        c.emit(Op.OP_DIVIDE, 0)
        vm = _run_chunk(c)
        assert len(vm.daughters) == 1
        assert vm.daughters[0].energy == vm.cell.energy


# ============================================================================
# Additional branch coverage: dispatch() fall-through paths
# ============================================================================

class TestDispatchFallthrough:
    """Cover remaining branch points in bioinstruction dispatcher."""

    def test_halt_with_no_frames(self):
        c = Chunk()
        c.emit(Op.OP_HALT)
        vm = _make_vm(c)
        vm.frames.clear()
        vm._dispatcher.dispatch(Op.OP_HALT)

    def test_return_with_no_frames(self):
        c = Chunk()
        c.emit(Op.OP_RETURN)
        vm = _make_vm(c)
        vm.frames.clear()
        vm._dispatcher.dispatch(Op.OP_RETURN)

    def test_modify_state_unmatched_value(self):
        c = Chunk()
        c.emit(Op.OP_MODIFY_STATE, 4)
        vm = _make_vm(c)
        vm._dispatcher.dispatch(Op.OP_MODIFY_STATE)
        assert vm.cell.color == (255, 255, 255)

    def test_bind_consumed_zero(self):
        c = Chunk()
        c.emit(Op.OP_BIND, 1)
        vm = _make_vm(c)
        vm.grn.add_gene("a", threshold=0.5, initial_level=1.0)
        vm.grn.add_gene("b", threshold=0.5, initial_level=0.0)
        vm.frames[-1].gene_name = "a"
        vm.cell.add_protein("a", 0.0)
        vm._dispatcher.dispatch(Op.OP_BIND)
        assert vm._binding_events == []

    def test_bind_no_protein_consumed(self):
        c = Chunk()
        c.emit(Op.OP_BIND, 1)
        vm = _make_vm(c)
        vm.grn.add_gene("a", threshold=0.5, initial_level=1.0)
        vm.grn.add_gene("b", threshold=0.5, initial_level=0.0)
        vm.frames[-1].gene_name = "a"
        vm.cell.proteins.pop("a", None)
        vm._dispatcher.dispatch(Op.OP_BIND)
        assert vm._binding_events == []

    def test_use_plugin_not_registered(self):
        c = Chunk()
        c.emit(Op.OP_USE_PLUGIN, 0)
        vm = _make_vm(c)
        vm._dispatcher.dispatch(Op.OP_USE_PLUGIN)

    def test_dispatch_unknown_value_noop(self):
        """dispatch() with a non-Op value falls through all case patterns
        silently (the match has no catch-all case; every Op member is covered
        explicitly above, so a non-Op is the only path to the end)."""
        c = Chunk()
        vm = _make_vm(c)
        vm._dispatcher.dispatch(object())


# ============================================================================
# _update_enzyme_levels edit_type not substitution/frameshift
# ============================================================================

class TestUpdateEnzymeOtherEditType:
    def test_edit_type_not_matched(self):
        vm = CellVM(Chunk(), Program(config=Config(ops_per_tick=256)))
        vm._crispr_edits = [
            {"success": True, "target": "g1", "edit_type": "no_edit"}
        ]
        vm._gem_gpr_map["g1"] = ["rxn1"]
        vm._enzyme_kcat["rxn1"] = 1.0
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert vm._enzyme_kcat["rxn1"] == 1.0


# ============================================================================
# _use_plugin with unregistered plugin
# ============================================================================

class TestUsePluginUnregistered:
    def test_plugin_not_registered_noop(self):
        from helixlang.core.plugin_registry import Registry
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256))
        registry = Registry()
        vm = CellVM(c, prog, registry=registry)
        vm._use_plugin("nonexistent", ())
        assert "nonexistent" not in registry.active()


# ============================================================================
# _transcribe_translate: gene with no protein result
# ============================================================================

class TestTranscribeNoProtein:
    def test_transcribe_translate_stop_only(self):
        from helixlang.core.ast_nodes import Gene
        c = Chunk()
        prog = Program(config=Config(ops_per_tick=256, use_central_dogma=True,
                                     species="ecoli"))
        vm = CellVM(c, prog)
        prog.genes.append(Gene(name="g", promoter=None, codons=[], orf=[]))
        vm._gene_dna["g"] = "TAA"
        vm.use_accel = False
        vm._transcribe_translate()


# ============================================================================
# _execute_pending: ip >= len(code) with remaining frame
# ============================================================================

class TestExecuteIpOutOfBounds:
    def test_ip_out_of_bounds_resumes_from_frame(self):
        c = Chunk()
        c.emit(Op.OP_NOP)
        vm = _make_vm(c)
        vm.use_accel = False
        vm.frames.append(Frame(return_ip=0, gene_name="outer"))
        vm.ip = len(c.code)
        vm._execute_pending()
        assert vm.ip == 1
        assert len(vm.frames) == 1


# ============================================================================
# Chunk.read_u8 / Chunk.read_u16 (bytecode.py coverage)
# ============================================================================

class TestChunkReadOperands:
    def test_read_u8_returns_value_and_new_ip(self):
        c = Chunk()
        c.code.append(0xAB)
        c.lines.append(0)
        c.codon_indices.append(-1)
        val, new_ip = c.read_u8(0)
        assert val == 0xAB
        assert new_ip == 1

    def test_read_u16_returns_big_endian_value_and_new_ip(self):
        c = Chunk()
        c.code.extend([0x01, 0x02])
        c.lines.extend([0, 0])
        c.codon_indices.extend([-1, -1])
        val, new_ip = c.read_u16(0)
        assert val == (0x01 << 8) | 0x02
        assert new_ip == 2


# ============================================================================
# LanguageConfig.standard() and __repr__ (language.py coverage)
# ============================================================================

class TestLanguageConfigExtras:
    def test_standard_classmethod(self):
        from helixlang.core.language import LanguageConfig
        cfg = LanguageConfig.standard()
        assert cfg.table_name == "standard"
        assert isinstance(cfg.stop_codons, frozenset)
        assert isinstance(cfg.start_codons, frozenset)

    def test_repr_contains_table_name(self):
        from helixlang.core.language import LanguageConfig
        cfg = LanguageConfig.standard()
        r = repr(cfg)
        assert "LanguageConfig" in r
        assert "standard" in r
        assert "stops=" in r
        assert "starts=" in r

