"""HelixDebugger unit tests."""
from helixlang.core.codon_table import STANDARD_TABLE, Op
from helixlang.core.compiler import Compiler
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.core.semantic import SemanticAnalyzer
from helixlang.core.vm import CellVM
from helixlang.debugger import (
    Breakpoint,
    DebugState,
    HelixDebugger,
    format_disasm_around,
    format_state,
)
from helixlang.plugins.runtime.cell import FEED_ENERGY_AMOUNT, INITIAL_CELL_ENERGY


def make_debugger(src, table=STANDARD_TABLE):
    """Build the debugger from source code (without running the VM)."""
    stop = {c for c, op in table.items() if op == Op.OP_HALT}
    toks = list(Lexer(src).tokens())
    prog = Parser(toks, stop_codons=stop).parse()
    SemanticAnalyzer(prog).check()
    chunk = Compiler(table).compile(prog)
    vm = CellVM(chunk, prog)
    return HelixDebugger(vm, prog)


# ------------------------------------------------------------------ #
# Breakpoint management
# ------------------------------------------------------------------ #
def test_set_and_list_breakpoint():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    bp = dbg.set_breakpoint(offset=3)
    assert isinstance(bp, Breakpoint)
    assert bp.offset == 3
    assert bp.enabled is True
    assert bp.hit_count == 0
    listed = dbg.list_breakpoints()
    assert len(listed) == 1
    assert listed[0] is bp


def test_remove_breakpoint():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    bp = dbg.set_breakpoint(offset=3)
    dbg.remove_breakpoint(bp)
    assert dbg.list_breakpoints() == []
    # Removing a nonexistent breakpoint should not raise
    dbg.remove_breakpoint(bp)


def test_enable_disable_breakpoint():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    bp = dbg.set_breakpoint(offset=3)
    assert bp.enabled is True
    dbg.disable_breakpoint(bp)
    assert bp.enabled is False
    dbg.enable_breakpoint(bp)
    assert bp.enabled is True


def test_breakpoint_by_line():
    # Line-number breakpoint: look up the line of the first instruction and set the breakpoint
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    target_line = dbg.vm.chunk.lines[0]
    dbg.set_breakpoint(line=target_line)
    # Continue should hit the line-number breakpoint
    state = dbg.continue_run()
    assert state is not None
    assert state.line == target_line


def test_breakpoint_by_codon_index():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    # codon #1 corresponds to GCT (OP_BUILD_PROTEIN)
    dbg.set_breakpoint(codon_index=1)
    state = dbg.continue_run()
    assert state is not None
    # When hit, codon_index should be 1
    assert state.codon_index == 1


# ------------------------------------------------------------------ #
# Step execution
# ------------------------------------------------------------------ #
def test_step_advances_ip():
    # ORF: ATG GCT TAA
    # Bytecode: 0 OP_START | 1 OP_BUILD_PROTEIN 2(operand) | 3 OP_HALT
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    assert dbg.vm.ip == 0
    state = dbg.step()  # OP_START
    assert state.ip == 1
    assert state.op == "OP_BUILD_PROTEIN"
    state = dbg.step()  # OP_BUILD_PROTEIN
    assert state.ip == 3
    assert state.op == "OP_HALT"


def test_step_executes_side_effect():
    # GCT = OP_BUILD_PROTEIN arg=3 (wobble T=3)
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    dbg.step()  # OP_START
    dbg.step()  # OP_BUILD_PROTEIN
    # Protein kind=3 should have been synthesized
    assert dbg.vm.cell.proteins.get(3, 0.0) == 1.0


def test_step_until_halt():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    dbg.step()  # OP_START
    dbg.step()  # OP_BUILD_PROTEIN
    dbg.step()  # OP_HALT -> frame popped
    assert len(dbg.vm.frames) == 0
    # Continuing to step should not crash
    state = dbg.step()
    assert state.gene is None


# ------------------------------------------------------------------ #
# Step over
# ------------------------------------------------------------------ #
def test_step_over_skips_called_gene():
    # caller: ATG CGT TAA → OP_START | OP_CALL_GENE(target) | OP_HALT
    # target: ATG GCT TAA → OP_START | OP_BUILD_PROTEIN(3) | OP_HALT
    src = """#gene name=caller
ATG CGT TAA
#end
#gene name=target
ATG GCT TAA
#end
"""
    dbg = make_debugger(src)
    dbg.start("caller")
    dbg.step()  # OP_START, ip -> 1 (OP_CALL_GENE)
    # step over should skip target's internals and return to caller's OP_HALT
    state = dbg.step_over()
    assert state.gene == "caller"
    assert state.ip == 4  # offset of the caller's OP_HALT
    # BUILD_PROTEIN inside target should have executed
    assert dbg.vm.cell.proteins.get(3, 0.0) == 1.0
    # The call stack should have only one frame (caller)
    assert len(dbg.vm.frames) == 1


def test_step_over_without_call():
    # Without CALL_GENE, step_over is equivalent to step
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    state = dbg.step_over()  # OP_START
    assert state.ip == 1
    state = dbg.step_over()  # OP_BUILD_PROTEIN
    assert state.ip == 3


# ------------------------------------------------------------------ #
# Continue execution
# ------------------------------------------------------------------ #
def test_continue_run_to_breakpoint():
    # ORF: ATG GCT GCT TAA
    # 0 OP_START | 1 OP_BUILD_PROTEIN(3) | 3 OP_BUILD_PROTEIN(3) | 5 OP_HALT
    dbg = make_debugger("#gene name=g\nATG GCT GCT TAA\n#end")
    dbg.start()
    bp = dbg.set_breakpoint(offset=3)
    state = dbg.continue_run()
    assert state is not None
    assert state.ip == 3
    assert bp.hit_count == 1


def test_continue_run_to_halt():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    state = dbg.continue_run()
    assert state is None
    assert len(dbg.vm.frames) == 0


def test_continue_run_disabled_breakpoint_ignored():
    dbg = make_debugger("#gene name=g\nATG GCT GCT TAA\n#end")
    dbg.start()
    bp = dbg.set_breakpoint(offset=3)
    dbg.disable_breakpoint(bp)
    state = dbg.continue_run()
    # A disabled breakpoint should not trigger; should run to HALT
    assert state is None
    assert bp.hit_count == 0


# ------------------------------------------------------------------ #
# Variable watches
# ------------------------------------------------------------------ #
def test_watch_energy():
    # GAA = OP_FEED arg=0 (wobble A=0); feed(1e8) brings energy 1e9 -> 1.1e9
    dbg = make_debugger("#gene name=g\nATG GAA TAA\n#end")
    dbg.start()
    dbg.add_watch("ene", "energy")
    dbg.step()  # OP_START
    watches = dbg.get_watches()
    assert watches[0].last_value == INITIAL_CELL_ENERGY
    dbg.step()  # OP_FEED
    watches = dbg.get_watches()
    assert watches[0].last_value == INITIAL_CELL_ENERGY + FEED_ENERGY_AMOUNT


def test_watch_protein():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    dbg.add_watch("p3", "protein.3")
    dbg.step()  # OP_START
    dbg.step()  # OP_BUILD_PROTEIN kind=3
    watches = dbg.get_watches()
    assert watches[0].last_value == 1.0


def test_watch_grn_level():
    # g is a constitutive gene with an initial level of 1.0
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    dbg.add_watch("g_lvl", "grn.g")
    watches = dbg.get_watches()
    assert watches[0].last_value is not None
    assert watches[0].last_value > 0.0


# ------------------------------------------------------------------ #
# Conditional breakpoints
# ------------------------------------------------------------------ #
def test_conditional_breakpoint_triggers_when_condition_true():
    dbg = make_debugger("#gene name=g\nATG GCT GCT TAA\n#end")
    dbg.start()
    dbg.vm.cell.energy = 30  # energy < 50
    bp = dbg.set_breakpoint(offset=3, condition="energy < 50")
    state = dbg.continue_run()
    assert state is not None
    assert state.ip == 3
    assert bp.hit_count == 1


def test_conditional_breakpoint_skipped_when_condition_false():
    dbg = make_debugger("#gene name=g\nATG GCT GCT TAA\n#end")
    dbg.start()
    # energy is initially 100, so the condition energy < 50 is false
    bp = dbg.set_breakpoint(offset=3, condition="energy < 50")
    state = dbg.continue_run()
    assert state is None  # ran to HALT
    assert bp.hit_count == 0


def test_conditional_breakpoint_operators():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    dbg.vm.cell.energy = 100
    # == 100 should trigger
    bp = dbg.set_breakpoint(offset=1, condition="energy == 100")
    state = dbg.continue_run()
    assert state is not None
    assert bp.hit_count == 1


# ------------------------------------------------------------------ #
# Call stack
# ------------------------------------------------------------------ #
def test_call_stack_single_frame():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    stack = dbg.get_call_stack()
    assert len(stack) == 1
    assert stack[0]["gene"] == "g"
    assert stack[0]["depth"] == 0


def test_call_stack_nested_call():
    src = """#gene name=caller
ATG CGT TAA
#end
#gene name=target
ATG GCT TAA
#end
"""
    dbg = make_debugger(src)
    dbg.start("caller")
    dbg.step()  # OP_START
    dbg.step()  # OP_CALL_GENE -> enter target
    stack = dbg.get_call_stack()
    assert len(stack) == 2
    # Innermost frame first
    assert stack[0]["depth"] == 0
    assert stack[0]["gene"] == "target"  # resolved via _gene_at_offset
    assert stack[1]["depth"] == 1
    assert stack[1]["gene"] == "caller"


# ------------------------------------------------------------------ #
# State snapshots
# ------------------------------------------------------------------ #
def test_state_snapshot_fields():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    state = dbg.get_state()
    assert isinstance(state, DebugState)
    assert state.ip == 0
    assert state.op == "OP_START"
    assert state.gene == "g"
    assert "energy" in state.cell_state
    assert state.cell_state["energy"] == INITIAL_CELL_ENERGY
    assert isinstance(state.grn_state, dict)


def test_state_reflects_execution():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    dbg.step()  # OP_START
    dbg.step()  # OP_BUILD_PROTEIN
    state = dbg.get_state()
    assert state.cell_state["proteins"].get(3) == 1.0


# ------------------------------------------------------------------ #
# inspect
# ------------------------------------------------------------------ #
def test_inspect_energy():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    assert dbg.inspect("energy") == INITIAL_CELL_ENERGY


def test_inspect_protein():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    dbg.step()
    dbg.step()  # BUILD_PROTEIN
    assert dbg.inspect("protein.3") == 1.0


def test_inspect_grn():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    val = dbg.inspect("grn.g")
    assert val is not None
    assert val > 0.0


def test_inspect_unknown_returns_none():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    assert dbg.inspect("nonexistent") is None


# ------------------------------------------------------------------ #
# Formatted output
# ------------------------------------------------------------------ #
def test_format_state():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    dbg.start()
    state = dbg.get_state()
    s = format_state(state)
    assert "ip=" in s
    assert "op=OP_START" in s
    assert "gene=g" in s
    assert f"energy={INITIAL_CELL_ENERGY}" in s
    assert "stack:" in s
    assert "grn:" in s


def test_format_disasm_around():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    chunk = dbg.vm.chunk
    # ip=1 corresponds to OP_BUILD_PROTEIN
    s = format_disasm_around(chunk, 1, context=2)
    assert "OP_BUILD_PROTEIN" in s
    assert ">>>" in s  # current instruction marker
    # Should include context instructions
    assert "OP_START" in s
    assert "OP_HALT" in s


def test_format_disasm_around_marker_at_correct_offset():
    dbg = make_debugger("#gene name=g\nATG GCT TAA\n#end")
    chunk = dbg.vm.chunk
    s = format_disasm_around(chunk, 0, context=5)
    lines = s.splitlines()
    # The first line should carry the >>> marker and be OP_START (offset 0)
    assert ">>>" in lines[0]
    assert "OP_START" in lines[0]
