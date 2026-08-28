"""Tests for CALL GENE frame identity — verifying frame push/pop, gene name
resolution, return_ip correctness, and the debugger's frame introspection.

Covers:
  - Frame dataclass correctness
  - OP_CALL_GENE pushes frame with gene_name="<call>"
  - GRN _call_gene() pushes frame with actual gene name
  - Frame depth grows/shrinks with nested calls
  - Frame depth cap at 256
  - Debugger resolves <call> via _gene_at_offset()
  - Step over/out operate on frame depth
  - Return restores correct return_ip
  - call_target field routes to correct gene
"""
from __future__ import annotations

import pytest

from helixlang.core.bytecode import Chunk
from helixlang.core.codon_table import STANDARD_TABLE, Op
from helixlang.core.compiler import Compiler
from helixlang.core.errors import CompileError
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.core.semantic import SemanticAnalyzer
from helixlang.core.vm import CellVM, Frame
from helixlang.debugger import HelixDebugger
from helixlang.plugins.runtime.grn import GRN

# ── helpers ──────────────────────────────────────────────────────────


def _compile_src(src: str) -> tuple[Chunk, object]:
    """Compile source and return (chunk, program)."""
    stop = {c for c, op in STANDARD_TABLE.items() if op == Op.OP_HALT}
    toks = list(Lexer(src).tokens())
    prog = Parser(toks, stop_codons=stop).parse()
    SemanticAnalyzer(prog).check()
    chunk = Compiler(STANDARD_TABLE).compile(prog)
    return chunk, prog


def _make_vm(src: str) -> CellVM:
    """Compile source and create a VM (not yet run)."""
    chunk, prog = _compile_src(src)
    return CellVM(chunk, prog)


def _make_debugger(src: str) -> HelixDebugger:
    """Build a debugger from source."""
    chunk, prog = _compile_src(src)
    vm = CellVM(chunk, prog)
    return HelixDebugger(vm, prog)


# ── Frame dataclass ─────────────────────────────────────────────────


class TestFrameDataclass:
    def test_frame_stores_return_ip(self):
        f = Frame(return_ip=42, gene_name="lacI")
        assert f.return_ip == 42

    def test_frame_stores_gene_name(self):
        f = Frame(return_ip=0, gene_name="target")
        assert f.gene_name == "target"

    def test_frame_fields_are_settable(self):
        """Frame is a plain dataclass with slots (not frozen)."""
        f = Frame(return_ip=0, gene_name="g")
        f.return_ip = 99
        f.gene_name = "other"
        assert f.return_ip == 99
        assert f.gene_name == "other"


# ── OP_CALL_GENE pushes frame with "<call>" ─────────────────────────


class TestCallGeneFrameIdentity:
    def test_call_gene_pushes_frame_via_debugger(self):
        """OP_CALL_GENE creates a new frame on the frame stack."""
        # caller: ATG CGT TAA → OP_START(0) | OP_CALL_GENE(1) | OP_HALT(4)
        # target: ATG GCT TAA → OP_START(3) | OP_BUILD_PROTEIN(4) | OP_HALT(6)
        src = """#gene name=caller
ATG CGT TAA
#end
#gene name=target
ATG GCT TAA
#end
"""
        dbg = _make_debugger(src)
        dbg.start("caller")
        # Step 1: OP_START in caller (ip 0 → 1)
        state = dbg.step()
        assert state.ip == 1  # now at OP_CALL_GENE
        assert len(dbg.vm.frames) == 1  # root frame for caller
        # Step 2: OP_CALL_GENE → pushes frame, enters target
        state = dbg.step()
        # Now we're inside target; the pushed frame should have gene_name="<call>"
        assert len(dbg.vm.frames) == 2
        assert dbg.vm.frames[-1].gene_name == "<call>"

    def test_call_gene_return_ip_points_to_caller(self):
        """The pushed frame's return_ip points to the instruction after CALL_GENE."""
        # caller: 0 OP_START | 1 OP_CALL_GENE(→3) | 4 OP_HALT
        # target: 3 OP_START | 4 OP_BUILD_PROTEIN | 6 OP_HALT
        src = """#gene name=caller
ATG CGT TAA
#end
#gene name=target
ATG GCT TAA
#end
"""
        dbg = _make_debugger(src)
        dbg.start("caller")
        dbg.step()  # OP_START → ip=1
        dbg.step()  # OP_CALL_GENE → pushes frame, enters target
        # The pushed frame (index -1) should have return_ip pointing past CALL_GENE
        frame = dbg.vm.frames[-1]
        assert frame.return_ip == 4  # offset of OP_HALT in caller

    def test_call_gene_frame_sentinel_is_call(self):
        """OP_CALL_GENE pushes frame with gene_name='<call>' (sentinel)."""
        src = """#gene name=caller
ATG CGT TAA
#end
#gene name=target
ATG GCT TAA
#end
"""
        dbg = _make_debugger(src)
        dbg.start("caller")
        dbg.step()  # OP_START
        dbg.step()  # OP_CALL_GENE
        # The raw frame has gene_name="<call>"
        assert dbg.vm.frames[-1].gene_name == "<call>"


# ── GRN _call_gene() pushes frame with actual gene name ─────────────


class TestGRNCallGeneFrameIdentity:
    def test_grn_call_gene_pushes_named_frame(self):
        """GRN-driven _call_gene() sets gene_name to the actual gene name."""
        grn = GRN()
        grn.add_gene("g1", threshold=-1.0, initial_level=1.0)

        chunk, prog = _compile_src(
            "#gene name=g1\nATG GCT TAA\n#end\n"
        )
        vm = CellVM(chunk, prog)
        vm.grn = grn
        vm._call_gene("g1")
        assert len(vm.frames) == 1
        assert vm.frames[-1].gene_name == "g1"

    def test_grn_call_gene_multiple_named_frames(self):
        """Multiple GRN calls push frames with correct gene names."""
        grn = GRN()
        grn.add_gene("a", threshold=-1.0, initial_level=1.0)

        chunk, prog = _compile_src(
            "#gene name=a\nATG GCT TAA\n#end\n"
        )
        vm = CellVM(chunk, prog)
        vm.grn = grn
        vm._call_gene("a")
        vm._call_gene("a")
        assert len(vm.frames) == 2
        assert vm.frames[0].gene_name == "a"
        assert vm.frames[1].gene_name == "a"

    def test_grn_call_gene_uses_correct_return_ip(self):
        """GRN _call_gene() captures the current ip as return_ip."""
        chunk, prog = _compile_src(
            "#gene name=g\nATG GCT TAA\n#end\n"
        )
        vm = CellVM(chunk, prog)
        vm.ip = 42  # set to arbitrary ip
        vm._call_gene("g")
        assert vm.frames[-1].return_ip == 42
        assert vm.ip == chunk.gene_offsets["g"]  # jumped to gene


# ── Frame stack behavior ────────────────────────────────────────────


class TestFrameStackBehavior:
    def test_halt_pops_frame(self):
        """OP_HALT pops one frame from the stack."""
        src = "#gene name=g\nATG GCT TAA\n#end"
        dbg = _make_debugger(src)
        dbg.start()
        assert len(dbg.vm.frames) == 1  # root frame
        # Step through: OP_START → OP_BUILD_PROTEIN → OP_HALT
        dbg.step()  # OP_START
        dbg.step()  # OP_BUILD_PROTEIN
        dbg.step()  # OP_HALT → pops frame
        assert len(dbg.vm.frames) == 0

    def test_call_then_return_restores_frame(self):
        """CALL_GENE pushes frame, HALT in target pops it, returning to caller.

        Note: the debugger captures state BEFORE the frame pop on OP_HALT.
        So after the OP_HALT step, the frame is still on the stack (2 frames).
        The actual pop happens between the step and the next get_state() call.
        """
        src = """#gene name=caller
ATG CGT TAA
#end
#gene name=target
ATG GCT TAA
#end
"""
        dbg = _make_debugger(src)
        dbg.start("caller")
        dbg.step()  # OP_START in caller
        dbg.step()  # CALL_GENE → enters target, pushes frame
        depth_after_call = len(dbg.vm.frames)
        assert depth_after_call == 2
        dbg.step()  # OP_START in target
        dbg.step()  # OP_HALT in target → frame pop is captured AFTER this step
        # Step once more to confirm the caller is now active
        state = dbg.step()
        assert state.gene == "caller"
        depth_after_return = len(dbg.vm.frames)
        assert depth_after_return == 1  # only caller's root frame left

    def test_nested_calls_increase_frame_depth(self):
        """Nested CALL_GENE increases frame depth."""
        # a calls b, b calls c
        src = """#gene name=a call_target=b
ATG CGT TAA
#end
#gene name=b call_target=c
ATG CGT TAA
#end
#gene name=c
ATG GCT TAA
#end
"""
        dbg = _make_debugger(src)
        dbg.start("a")
        dbg.step()  # OP_START in a
        dbg.step()  # CALL_GENE → enters b
        depth_after_first = len(dbg.vm.frames)
        dbg.step()  # OP_START in b
        dbg.step()  # CALL_GENE → enters c
        depth_after_second = len(dbg.vm.frames)
        assert depth_after_second == depth_after_first + 1

    def test_frame_depth_cap_256(self):
        """Frame stack cannot exceed 256 frames (GRN path)."""
        grn = GRN()
        grn.add_gene("g", threshold=-1.0, initial_level=1.0)

        chunk, prog = _compile_src(
            "#gene name=g\nATG GCT TAA\n#end\n"
        )
        vm = CellVM(chunk, prog)
        vm.grn = grn
        for _ in range(256):
            vm._call_gene("g")
        assert len(vm.frames) == 256
        vm._call_gene("g")
        assert len(vm.frames) == 256

    def test_frame_depth_256_guard_clears_stack(self):
        """When frame depth exceeds 256 during execution, stack is cleared."""
        chunk, prog = _compile_src(
            "#gene name=g\nATG GCT TAA\n#end\n"
        )
        vm = CellVM(chunk, prog)
        # Manually push 257 frames
        for i in range(257):
            vm.frames.append(Frame(return_ip=i, gene_name=f"f{i}"))
        # _execute_pending should detect >256 and clear
        vm._execute_pending()
        assert len(vm.frames) == 0


# ── Debugger frame introspection ────────────────────────────────────


class TestDebuggerFrameIdentity:
    def test_call_stack_shows_resolved_gene_name(self):
        """Debugger call stack resolves <call> to the actual gene name."""
        src = """#gene name=caller
ATG CGT TAA
#end
#gene name=target
ATG GCT TAA
#end
"""
        dbg = _make_debugger(src)
        dbg.start("caller")
        dbg.step()  # OP_START
        dbg.step()  # CALL_GENE → enters target
        stack = dbg.get_call_stack()
        genes_on_stack = [frame["gene"] for frame in stack]
        assert "target" in genes_on_stack

    def test_call_stack_shows_caller_gene(self):
        """After entering a called gene, caller is also on the stack."""
        src = """#gene name=caller
ATG CGT TAA
#end
#gene name=target
ATG GCT TAA
#end
"""
        dbg = _make_debugger(src)
        dbg.start("caller")
        dbg.step()  # OP_START
        dbg.step()  # CALL_GENE → enters target
        stack = dbg.get_call_stack()
        genes_on_stack = [frame["gene"] for frame in stack]
        assert "caller" in genes_on_stack
        assert "target" in genes_on_stack

    def test_call_stack_depth_after_nested_calls(self):
        """Nested calls increase call stack depth."""
        src = """#gene name=a call_target=b
ATG CGT TAA
#end
#gene name=b call_target=c
ATG CGT TAA
#end
#gene name=c
ATG GCT TAA
#end
"""
        dbg = _make_debugger(src)
        dbg.start("a")
        dbg.step()  # OP_START in a
        dbg.step()  # CALL_GENE → enters b
        depth_first = len(dbg.get_call_stack())
        dbg.step()  # OP_START in b
        dbg.step()  # CALL_GENE → enters c
        depth_second = len(dbg.get_call_stack())
        assert depth_second > depth_first

    def test_gene_at_offset_resolves_correctly(self):
        """_gene_at_offset finds the gene containing the given IP."""
        src = """#gene name=alpha
ATG GCT TAA
#end
#gene name=beta
ATG GGT TAA
#end
"""
        dbg = _make_debugger(src)
        offsets = dbg.vm.chunk.gene_offsets
        assert "alpha" in offsets
        assert "beta" in offsets
        # beta's offset should resolve to beta
        assert dbg._gene_at_offset(offsets["beta"]) == "beta"
        # alpha's offset should resolve to alpha
        assert dbg._gene_at_offset(offsets["alpha"]) == "alpha"


# ── Debugger step over / step out with frames ──────────────────────


class TestDebuggerStepOverOut:
    def test_step_over_skips_called_gene(self):
        """step_over() executes the called gene but stops at caller's next instruction."""
        src = """#gene name=caller
ATG CGT TAA
#end
#gene name=target
ATG GCT TAA
#end
"""
        dbg = _make_debugger(src)
        dbg.start("caller")
        dbg.step()  # OP_START
        state = dbg.step_over()
        assert state.gene == "caller"
        # Protein from target should have been built
        assert dbg.vm.cell.proteins.get(3, 0.0) == 1.0

    def test_step_out_returns_to_caller(self):
        """step_out() executes until the current gene returns."""
        src = """#gene name=caller
ATG CGT TAA
#end
#gene name=target
ATG GCT TAA
#end
"""
        dbg = _make_debugger(src)
        dbg.start("caller")
        dbg.step()  # OP_START in caller
        dbg.step()  # CALL_GENE → enters target
        state = dbg.step_out()
        assert state.gene == "caller"

    def test_frame_depth_tracks_across_steps(self):
        """Frame depth increases on entry and decreases on return."""
        src = """#gene name=caller
ATG CGT TAA
#end
#gene name=target
ATG GCT TAA
#end
"""
        dbg = _make_debugger(src)
        dbg.start("caller")
        depth_before = len(dbg.vm.frames)
        dbg.step()  # OP_START
        dbg.step()  # CALL_GENE → pushes frame
        depth_after_call = len(dbg.vm.frames)
        assert depth_after_call == depth_before + 1
        dbg.step()  # OP_START in target
        dbg.step()  # OP_HALT in target → pop captured after step
        # Step once more to confirm the caller is active and frame was popped
        dbg.step()
        depth_after_return = len(dbg.vm.frames)
        assert depth_after_return == depth_before


# ── call_target field routing ────────────────────────────────────────


class TestCallTargetField:
    def test_call_target_routes_to_specified_gene(self):
        """call_target=g4 routes OP_CALL_GENE to gene g4, not wobble-selected."""
        src = """#gene name=g0 call_target=g4
ATG CGT TAA
#end
#gene name=g1
ATG GCT TAA
#end
#gene name=g2
ATG GGT TAA
#end
#gene name=g3
ATG GCT TAA
#end
#gene name=g4
ATG GGT TAA
#end
"""
        chunk, prog = _compile_src(src)
        bytecode = chunk.code
        g0_offset = chunk.gene_offsets["g0"]
        # Find OP_CALL_GENE within g0's bytecode range only
        call_ip = None
        for i in range(g0_offset, len(bytecode) - 2):
            if bytecode[i] == Op.OP_CALL_GENE:
                call_ip = i
                break
        assert call_ip is not None
        # u16 is big-endian: high byte first, low byte second
        patched_offset = (bytecode[call_ip + 1] << 8) | bytecode[call_ip + 2]
        g4_offset = chunk.gene_offsets["g4"]
        assert patched_offset == g4_offset

    def test_call_target_nonexistent_raises_compile_error(self):
        """call_target to a nonexistent gene raises CompileError."""
        src = """#gene name=g0 call_target=ghost
ATG CGT TAA
#end
"""
        stop = {c for c, op in STANDARD_TABLE.items() if op == Op.OP_HALT}
        toks = list(Lexer(src).tokens())
        prog = Parser(toks, stop_codons=stop).parse()
        SemanticAnalyzer(prog).check()
        with pytest.raises(CompileError):
            Compiler(STANDARD_TABLE).compile(prog)


# ── End-to-end: frame identity through full execution ──────────────


class TestEndToEndFrameIdentity:
    def test_caller_and_target_proteins_both_produced(self):
        """Both caller and target produce proteins when caller calls target."""
        src = """#gene name=caller
ATG CGT TAA
#end
#gene name=target
ATG GCT TAA
#end
"""
        dbg = _make_debugger(src)
        dbg.start("caller")
        # Step through entire execution
        for _ in range(20):
            state = dbg.step()
            if state is None:
                break
        # target produces protein (GCT = OP_BUILD_PROTEIN)
        assert dbg.vm.cell.proteins.get(3, 0.0) >= 1.0

    def test_call_gene_frame_return_ip_allows_continuation(self):
        """After CALL_GENE returns, execution continues at the caller's return_ip."""
        # caller: OP_START(0) | OP_CALL_GENE(1) | OP_BUILD_PROTEIN(4) | OP_HALT(6)
        # target: OP_START(10) | OP_HALT(11)
        src = """#gene name=caller
ATG CGT GCT TAA
#end
#gene name=target
ATG TAA
#end
"""
        dbg = _make_debugger(src)
        dbg.start("caller")
        dbg.step()  # OP_START → ip=1 (OP_CALL_GENE)
        dbg.step()  # OP_CALL_GENE → enters target
        dbg.step()  # OP_START in target
        dbg.step()  # OP_HALT in target → returns to caller, state shows OP_BUILD_PROTEIN
        # The state shows we're about to execute OP_BUILD_PROTEIN in caller
        state = dbg.step()
        # After executing OP_BUILD_PROTEIN, the state shows OP_HALT
        assert state.op == "OP_HALT"
        assert state.gene == "caller"
        # Protein should now be produced
        assert dbg.vm.cell.proteins.get(3, 0.0) >= 1.0

    def test_frame_identity_preserved_across_multiple_ticks(self):
        """Frame stack is correctly maintained across multiple VM ticks."""
        src = """#gene name=caller
ATG CGT TAA
#end
#gene name=target
ATG GCT TAA
#end
"""
        vm = _make_vm(src)
        vm.run(5)
        # After all ticks, frames should be empty (all calls returned)
        assert len(vm.frames) == 0
