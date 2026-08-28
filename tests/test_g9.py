"""G9 — declarative couplings and closed feedback loops.

Verification goals:
- ``#morphogen gene=<name> channel=U|V gain=<float>`` wires a morphogen
  channel to any gene's GRN level (replacing the hard-coded ``pigment``
  gene), clamped to [0, 1].
- The legacy no-declaration path still feeds the ``pigment`` gene on the
  V channel (backwards compatible).
- ``OP_DIVIDE`` performs real division: it halves the parent's energy
  *and* spawns a daughter cell (observable via ``vm.daughters``).
- ``OP_CALL_GENE`` addresses the full gene space via ``call_target=``,
  breaking the 4-gene wobble limit.

References: doc/10-frontier-biology-analysis.md G9.
"""
from __future__ import annotations

import pytest

from helixlang.core.ast_nodes import Program
from helixlang.core.bytecode import Chunk
from helixlang.core.codon_table import STANDARD_TABLE, Op
from helixlang.core.compiler import CompileError, Compiler
from helixlang.core.errors import ParseError
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.core.vm import CellVM, Frame
from helixlang.plugins.runtime.cell import INITIAL_CELL_ENERGY, Cell


def _compile(src: str) -> tuple[Chunk, Program]:
    stop = {c for c, op in STANDARD_TABLE.items() if op == Op.OP_HALT}
    prog = Parser(list(Lexer(src).tokens()), stop_codons=stop).parse()
    chunk = Compiler().compile(prog)
    return chunk, prog


def _vm_for(src: str) -> CellVM:
    chunk, prog = _compile(src)
    return CellVM(chunk, prog)


def _chunk_vm(code: list[int]) -> CellVM:
    """Build a VM with a frame so _execute_pending runs from ip=0."""
    c = Chunk()
    for op in code:
        c.emit(op, 0)
    prog = Program()
    vm = CellVM(c, prog)
    vm.frames.append(Frame(return_ip=len(c.code), gene_name="test"))
    vm.ip = 0
    return vm


# ============================================================================
# #morphogen declarative wiring
# ============================================================================

def test_morphogen_parses_declaration():
    prog = _compile("#morphogen gene=clocker channel=V gain=0.25\n"
                    "#field size=8\n#config ticks=1")[1]
    assert len(prog.morphogen_feedback) == 1
    mf = prog.morphogen_feedback[0]
    assert mf.gene == "clocker"
    assert mf.channel == "V"
    assert mf.gain == 0.25


def test_morphogen_defaults_channel_and_gain():
    prog = _compile("#morphogen gene=clocker\n#field size=8\n"
                    "#config ticks=1")[1]
    mf = prog.morphogen_feedback[0]
    assert mf.channel == "V"
    assert mf.gain == 0.1


def test_morphogen_channel_u():
    prog = _compile("#morphogen gene=subs channel=U gain=0.5\n"
                    "#field size=8\n#config ticks=1")[1]
    assert prog.morphogen_feedback[0].channel == "U"


def test_morphogen_requires_gene():
    with pytest.raises(ParseError):
        _compile("#morphogen channel=V\n#field size=8\n#config ticks=1")


def test_morphogen_invalid_channel():
    with pytest.raises(ParseError):
        _compile("#morphogen gene=g channel=X\n#field size=8\n"
                 "#config ticks=1")


def test_feedback_wires_declared_gene_v_channel():
    vm = _vm_for("#promoter name=p strength=0.4\n"
                 "#gene name=clocker promoter=p\nATG GCT TAA\n#end\n"
                 "#morphogen gene=clocker channel=V gain=0.5\n"
                 "#field size=8\n#config ticks=1")
    vm.field.v[0][0] = 0.8
    vm._feedback()
    assert vm.grn.nodes["clocker"].level == pytest.approx(0.4)  # 0.8*0.5


def test_feedback_wires_declared_gene_u_channel():
    vm = _vm_for("#promoter name=p strength=0.4\n"
                 "#gene name=subs promoter=p\nATG GCT TAA\n#end\n"
                 "#morphogen gene=subs channel=U gain=0.25\n"
                 "#field size=8\n#config ticks=1")
    vm.field.u[0][0] = 0.4
    vm._feedback()
    assert vm.grn.nodes["subs"].level == pytest.approx(0.1)  # 0.4*0.25


def test_feedback_clamps_to_one():
    vm = _vm_for("#promoter name=p strength=0.4\n"
                 "#gene name=g promoter=p\nATG GCT TAA\n#end\n"
                 "#morphogen gene=g channel=V gain=5.0\n"
                 "#field size=8\n#config ticks=1")
    vm.field.v[0][0] = 1.0
    vm._feedback()
    assert vm.grn.nodes["g"].level == pytest.approx(1.0)


def test_feedback_ignores_undeclared_gene():
    vm = _vm_for("#promoter name=p strength=0.4\n"
                 "#gene name=other promoter=p\nATG GCT TAA\n#end\n"
                 "#morphogen gene=clocker channel=V gain=0.5\n"
                 "#field size=8\n#config ticks=1")
    vm.field.v[0][0] = 0.9
    vm._feedback()
    assert vm.grn.nodes["other"].level == pytest.approx(0.0)


def test_feedback_legacy_pigment_fallback():
    vm = _vm_for("#promoter name=p_pigment strength=0.4\n"
                 "#gene name=pigment promoter=p_pigment\nATG GCT TAA\n#end\n"
                 "#field size=8\n#config ticks=1")
    vm.field.v[0][0] = 0.5
    vm._feedback()
    assert vm.grn.nodes["pigment"].level == pytest.approx(0.05)  # 0.5*0.1


# ============================================================================
# OP_DIVIDE real division
# ============================================================================

def test_divide_spawns_daughter():
    vm = _chunk_vm([Op.OP_DIVIDE])
    vm._execute_pending()
    assert len(vm.daughters) == 1
    assert vm.cell.energy == INITIAL_CELL_ENERGY // 2
    assert vm.cell.divisions == 1
    daughter = vm.daughters[0]
    assert daughter.energy == vm.cell.energy
    assert daughter.divisions == 0


def test_divide_daughter_inherits_state():
    vm = _chunk_vm([Op.OP_DIVIDE])
    vm.cell.proteins = {3: 42.0}
    vm.cell.slots[0] = "payload"
    vm._execute_pending()
    daughter = vm.daughters[0]
    assert daughter.proteins == {3: 42.0}
    assert daughter.slots[0] == "payload"


def test_divide_no_daughter_when_low_energy():
    vm = _chunk_vm([Op.OP_DIVIDE])
    vm.cell = Cell(energy=1.0)
    vm._execute_pending()
    assert vm.daughters == []
    assert vm.cell.divisions == 0


def test_divide_multiple_daughters_unique_names():
    vm = _chunk_vm([Op.OP_DIVIDE, Op.OP_DIVIDE])
    vm.cell.energy = 1.0e9
    vm._execute_pending()
    assert len(vm.daughters) == 2
    assert len({d.name for d in vm.daughters}) == 2


# ============================================================================
# OP_CALL_GENE full address space via call_target=
# ============================================================================

def test_call_gene_patches_fifth_gene_target():
    src = ("#gene name=g0 call_target=g4\nATG CGT TAA\n#end\n"
           "#gene name=g1\nATG GCT TAA\n#end\n"
           "#gene name=g2\nATG GCT TAA\n#end\n"
           "#gene name=g3\nATG GCT TAA\n#end\n"
           "#gene name=g4\nATG GAA TAA\n#end\n"
           "#config ticks=1")
    chunk, prog = _compile(src)
    # g0's ORF emits OP_CALL_GENE (CGT); the back-patched u16 offset must point
    # at g4's entry point (beyond the 4-gene wobble range).
    g0_ip = chunk.gene_offsets["g0"]
    call_ip = chunk.code.index(int(Op.OP_CALL_GENE), g0_ip)
    patched = (chunk.code[call_ip + 1] << 8) | chunk.code[call_ip + 2]
    assert patched == chunk.gene_offsets["g4"]
    assert patched != chunk.gene_offsets["g1"]
    vm = CellVM(chunk, prog)
    vm.run(1)
    # smoke: execution did not crash
    assert vm._signal_emissions == 0


def test_call_gene_target_must_exist():
    src = ("#gene name=g0 call_target=ghost\nATG CGT TAA\n#end\n"
           "#config ticks=1")
    with pytest.raises(CompileError):
        _compile(src)
