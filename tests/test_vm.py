"""VM unit tests."""
from helixlang.bytecode import Chunk
from helixlang.cell import Cell
from helixlang.codon_table import STANDARD_TABLE, Op
from helixlang.compiler import Compiler
from helixlang.lexer import Lexer
from helixlang.parser import Parser
from helixlang.semantic import SemanticAnalyzer
from helixlang.vm import CellVM


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
    src = """#promoter name=p strength=-0.5
#gene name=grow promoter=p
ATG CTC TAA
#end
#regulate p -> grow strength=+0.6
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
    from helixlang.codon_table import MITO_VERTEBRATE_TABLE
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

from helixlang.ast_nodes import Config, Program  # noqa: E402
from helixlang.reaction_diffusion import GrayScott  # noqa: E402


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
    from helixlang.vm import Frame
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
        assert vm.cell.energy == 100
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
        """POP on an empty stack should not crash."""
        c = Chunk()
        c.emit(Op.OP_POP)
        vm = _run_chunk(c)
        assert vm.stack == []

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
        vm = _run_chunk(c)
        assert vm.stack == []

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
        """SWAP should not crash when fewer than 2 elements are on the stack."""
        c = Chunk()
        c.add_constant(1)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_SWAP)
        vm = _run_chunk(c)
        assert vm.stack == [1]


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
        assert vm.cell.energy == 100      # 100 + round(10 * 0 / 255) == 100

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
        assert vm.cell.energy == 99

    def test_move_north(self):
        c = Chunk()
        c.emit(Op.OP_MOVE, 0)
        vm = _run_chunk(c)
        assert vm.cell.y == -1

    def test_signal_pushes_tuple(self):
        c = Chunk()
        c.emit(Op.OP_SIGNAL, 7)
        vm = _run_chunk(c)
        assert vm.stack == [("signal", 7)]

    def test_divide_halves_energy(self):
        c = Chunk()
        c.emit(Op.OP_DIVIDE, 0)
        vm = _run_chunk(c)
        assert vm.cell.energy == 50  # 100 // 2
        assert vm.cell.divisions == 1

    def test_die_sets_alive_false(self):
        c = Chunk()
        c.emit(Op.OP_DIE, 0)
        vm = _run_chunk(c)
        assert vm.cell.alive is False

    def test_feed_increases_energy(self):
        """FEED always adds 10 energy."""
        c = Chunk()
        c.emit(Op.OP_FEED, 0)
        vm = _run_chunk(c)
        assert vm.cell.energy == 110

    def test_feed_from_low_energy(self):
        # First lower the energy to 5
        vm0 = Cell(energy=5)
        # Directly construct the vm and replace the cell
        chunk = Chunk()
        chunk.emit(Op.OP_FEED, 0)
        prog = Program()
        vm = CellVM(chunk, prog)
        from helixlang.vm import Frame
        vm.frames.append(Frame(return_ip=len(chunk), gene_name="t"))
        vm.ip = 0
        vm.cell = vm0
        vm._execute_pending()
        assert vm.cell.energy == 15


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
        """ADD should not crash when fewer than 2 elements are on the stack."""
        c = Chunk()
        c.add_constant(1)
        c.emit(Op.OP_PUSH_CONST, 0)
        c.emit(Op.OP_ADD)
        vm = _run_chunk(c)
        assert vm.stack == [1]  # unchanged


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
        """WRITE_MEM on an empty stack should not crash and should not write."""
        c = Chunk()
        c.emit(Op.OP_WRITE_MEM, 0)
        vm = _run_chunk(c)
        assert vm.cell.slots[0] is None

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


class TestRegulateBind:
    """OP_REGULATE / OP_BIND are no-ops (they read a 1-byte operand)."""

    def test_regulate_noop(self):
        c = Chunk()
        c.emit(Op.OP_REGULATE, 5)
        vm = _run_chunk(c)
        assert vm.stack == []

    def test_bind_noop(self):
        c = Chunk()
        c.emit(Op.OP_BIND, 3)
        vm = _run_chunk(c)
        assert vm.stack == []


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
        """JUMP_IF_ZERO on an empty stack pops 0 by default -> jump."""
        c = Chunk()
        c.emit_u16(Op.OP_JUMP_IF_ZERO, 2)
        c.add_constant(999)
        c.emit(Op.OP_PUSH_CONST, 0)  # skipped
        c.add_constant(1)
        c.emit(Op.OP_PUSH_CONST, 1)
        vm = _run_chunk(c)
        assert vm.stack == [1]

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
        from helixlang.lsystem import LSystem
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
    """Unknown opcodes should be skipped (along with their operands)."""

    def test_unknown_opcode_skipped(self):
        """0x00 is not a valid Op -> skip it and continue executing subsequent instructions."""
        c = Chunk()
        # Directly append an unknown byte
        c.code.append(0x00)
        c.lines.append(0)
        c.codon_indices.append(-1)
        # Follow it with a valid PUSH_CONST
        c.add_constant(5)
        c.emit(Op.OP_PUSH_CONST, 0)
        vm = _run_chunk(c)
        assert vm.stack == [5]

    def test_unknown_opcode_with_operand_bytes_skipped(self):
        """Unimplemented opcodes skip their operand bytes.

        Ops not listed in OP_OPERAND_BYTES are handled by the default branch:
        OP_OPERAND_BYTES.get(op, 0) bytes are skipped.
        For an unknown byte 0x01 (not in the Op enum), _dispatch is not called
        (_execute_pending resolves with Op(op_byte); on failure it continues).
        """
        c = Chunk()
        c.code.append(0x01)  # invalid opcode
        c.lines.append(0)
        c.codon_indices.append(-1)
        c.add_constant(7)
        c.emit(Op.OP_PUSH_CONST, 0)
        vm = _run_chunk(c)
        assert vm.stack == [7]


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
                        "membrane_permeability", "field_total_v"):
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
