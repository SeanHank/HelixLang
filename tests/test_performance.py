"""Tests for performance optimization (doc/37 §3).

Covers SnapshotDownsampler, accelerated_execute_pending, and VMProfiler.
"""
from __future__ import annotations

import pytest

from helixlang.core.parser import parse_source
from helixlang.core.performance import (
    SnapshotDownsampler,
    VMProfiler,
    accelerated_execute_pending,
)
from helixlang.core.semantic import SemanticAnalyzer


@pytest.fixture
def simple_program():
    src = (
        "#gene name=gfp\n"
        "ATG GCT GGT GCT TAA\n"
        "#end\n"
        "#config ticks=20\n"
    )
    prog = parse_source(src)
    SemanticAnalyzer(prog).check()
    return prog


class TestSnapshotDownsampler:
    def test_short_simulations_full_fidelity(self) -> None:
        ds = SnapshotDownsampler()
        ds.configure(500)
        assert ds.interval == 1

    def test_long_simulations_downsample(self) -> None:
        ds = SnapshotDownsampler()
        ds.configure(10000)
        assert ds.interval > 1
        assert 10000 / ds.interval <= 550  # at most ~500 + 1 snapshots

    def test_should_snapshot_first(self) -> None:
        ds = SnapshotDownsampler(interval=10)
        assert ds.should_snapshot(0, 1000) is True

    def test_should_snapshot_interval(self) -> None:
        ds = SnapshotDownsampler(interval=10)
        assert ds.should_snapshot(10, 1000) is True
        assert ds.should_snapshot(13, 1000) is False

    def test_is_final(self) -> None:
        ds = SnapshotDownsampler(interval=10)
        assert ds.is_final(99, 100, True) is True
        assert ds.is_final(50, 100, True) is False


class TestVMProfiler:
    def test_profiles_program(self, simple_program) -> None:
        profiler = VMProfiler(enable_tracemalloc=False)
        result = profiler.profile(simple_program, max_ticks=20)
        assert result.ticks_executed == 20
        assert result.trace_entries == 20
        assert result.vm_run_time_ms >= 0
        assert "compile" in result.component_times
        assert "vm_run" in result.component_times

    def test_downsamples_long_run(self) -> None:
        src = (
            "#gene name=gfp\n"
            "ATG GCT GGT GCT TAA\n"
            "#end\n"
            "#config ticks=1200\n"
        )
        prog = parse_source(src)
        SemanticAnalyzer(prog).check()
        profiler = VMProfiler(enable_tracemalloc=False)
        result = profiler.profile(prog, max_ticks=1200)
        assert result.snapshot_interval > 1
        assert result.trace_entries < result.ticks_executed
        assert result.trace_entries <= result.ticks_executed // result.snapshot_interval + 2
        assert result.trace_entries < result.ticks_executed

    def test_profile_json_serializable(self, simple_program) -> None:
        import json
        profiler = VMProfiler(enable_tracemalloc=False)
        result = profiler.profile(simple_program, max_ticks=10)
        json.dumps(result.to_dict())


class TestValidityDecoupling:
    """doc/37 §2 (P2): a first-class skip-validity knob decouples realism
    checks from the accelerated path."""

    def _prog(self, *, skip: bool):
        src = (
            "#gene name=gfp\n"
            "ATG GCT GGT GCT TAA\n"
            "#end\n"
            f"#config ticks=5 skip_validity={'true' if skip else 'false'}\n"
        )
        prog = parse_source(src)
        SemanticAnalyzer(prog).check()
        return prog

    def test_parser_defaults_and_toggle(self):
        assert self._prog(skip=True).config.skip_validity is True
        assert self._prog(skip=False).config.skip_validity is False
        # default False (validity enforced)
        prog = parse_source("#gene name=g\nATG GCT TAA\n#end\n#config ticks=2\n")
        assert prog.config.skip_validity is False

    def test_cellvm_and_profiler_observe_skip(self):
        import helixlang.core.language as lang
        from helixlang.core.compiler import Compiler
        from helixlang.core.performance import VMProfiler
        from helixlang.core.vm import CellVM
        config = lang.LanguageConfig.for_table("standard")
        prog = self._prog(skip=True)
        chunk = Compiler(config).compile(prog)
        vm = CellVM(chunk, prog)
        assert vm.skip_validity is True
        prof = VMProfiler(enable_tracemalloc=False).profile(prog, max_ticks=5)
        assert prof.validity_skipped is True

    def test_profiler_reports_validity_kept_by_default(self):
        from helixlang.core.performance import VMProfiler
        prof = VMProfiler(enable_tracemalloc=False)
        result = prof.profile(self._prog(skip=False), max_ticks=5)
        assert result.validity_skipped is False

    def test_step_accel_prefer_python_is_byte_identical_to_step(self):
        from helixlang.plugins.runtime.grn import GRN
        # a multi-edge sigmoid network where the native kernel drifts a ULP
        a, b = GRN(), GRN()
        for g in (a, b):
            g.add_gene("ci", threshold=0.0, initial_level=0.8)
            g.add_gene("cro", threshold=0.0, initial_level=0.2)
            g.add_gene("out", threshold=0.3, initial_level=0.0, decay=0.5)
            g.add_edge("ci", "cro", -0.8)
            g.add_edge("cro", "ci", -0.7)
            g.add_edge("ci", "out", 1.2)
        for _ in range(120):
            a.step()
            b.step_accel(prefer="python")
            for n in a.nodes:
                assert a.nodes[n].level == b.nodes[n].level


class TestAcceleratedExecution:
    def test_importable(self) -> None:
        assert callable(accelerated_execute_pending)

    @staticmethod
    def _simple_chunk(program) -> None:
        from helixlang.core.bytecode import Chunk
        from helixlang.core.codon_table import Op

        chunk = Chunk()
        chunk.gene_offsets[program.genes[0].name] = 0
        c0 = chunk.add_constant(1.0)
        # Note: no OP_START here — the native kernel covers only the
        # arithmetic/stack subset {HALT, PUSH_CONST, POP, ADD, SUB, MUL}
        # (doc/36 §5.5), so a fully-native chunk must not open with a bio op.
        chunk.emit(Op.OP_PUSH_CONST, c0)
        chunk.emit(Op.OP_PUSH_CONST, c0)
        chunk.emit(Op.OP_ADD)
        chunk.emit(Op.OP_HALT)
        return chunk

    def test_accel_matches_python_for_simple_segments(self, simple_program) -> None:
        """Verify accelerated execution is tick-equivalent to pure Python when
        the bytecode is entirely simple (arithmetic/stack ops + HALT guard)."""
        from helixlang.core.vm import CellVM

        chunk = self._simple_chunk(simple_program)

        # Pure Python path (doc/38 §2.2: accel is opt-out per VM)
        vm_pp = CellVM(chunk, simple_program, use_accel=False)
        trace_pp = vm_pp.run(20)

        # Accelerated segment path (the pre-wired default)
        vm_acc = CellVM(chunk, simple_program)
        trace_acc = vm_acc.run(20)

        # Identical ticks, identical traces, identical total work
        assert len(trace_pp) == len(trace_acc)
        assert vm_pp.tick == vm_acc.tick
        assert trace_pp == trace_acc
        assert vm_pp.ops_executed == vm_acc.ops_executed
        # doc/38 §2.2: counters are observations, not requests
        assert vm_acc.ops_executed > 0
        assert vm_acc.accel_native_ops == vm_acc.ops_executed
        assert vm_pp.accel_native_ops == 0

    def test_nested_call_halts_stay_in_tick(self, simple_program) -> None:
        """A gene that CALLs a simple helper must resume the caller in the SAME
        tick (pure-loop semantics).  The pre-fix accelerator returned on HALT,
        deferring the caller's tail to the next tick and changing the trace."""
        from helixlang.core.bytecode import Chunk
        from helixlang.core.codon_table import Op
        from helixlang.core.vm import CellVM

        prog = parse_source(
            "#gene name=gfp\nATG GCT GGT GCT TAA\n#end\n"
            "#config ticks=12\n"
        )
        SemanticAnalyzer(prog).check()
        chunk = Chunk()
        chunk.gene_offsets["gfp"] = 0
        c0 = chunk.add_constant(1.0)
        # main: START | PUSH PUSH ADD | CALL helper | PUSH PUSH ADD | HALT
        chunk.emit(Op.OP_START)
        chunk.emit(Op.OP_PUSH_CONST, c0)
        chunk.emit(Op.OP_PUSH_CONST, c0)
        chunk.emit(Op.OP_ADD)
        # helper lives after CALL (3 bytes) + tail (PUSH 2 + PUSH 2 + ADD 1
        # + HALT 1 = 6 bytes), i.e. 9 bytes after the current code length.
        helper_off = len(chunk.code) + 9
        chunk.emit_u16(Op.OP_CALL_GENE, helper_off)
        chunk.emit(Op.OP_PUSH_CONST, c0)
        chunk.emit(Op.OP_PUSH_CONST, c0)
        chunk.emit(Op.OP_ADD)
        chunk.emit(Op.OP_HALT)
        chunk.gene_offsets["__hlx_helper"] = len(chunk.code)
        # helper: START | PUSH PUSH ADD POP | HALT
        chunk.emit(Op.OP_START)
        chunk.emit(Op.OP_PUSH_CONST, c0)
        chunk.emit(Op.OP_PUSH_CONST, c0)
        chunk.emit(Op.OP_ADD)
        chunk.emit(Op.OP_POP)
        chunk.emit(Op.OP_HALT)
        assert helper_off == chunk.gene_offsets["__hlx_helper"]

        vm_pp = CellVM(chunk, prog, use_accel=False)
        trace_pp = vm_pp.run(12)
        vm_acc = CellVM(chunk, prog)
        trace_acc = vm_acc.run(12)

        assert trace_pp == trace_acc
        assert vm_pp.ops_executed == vm_acc.ops_executed
        assert vm_acc.accel_native_ops > 0

    def test_quota_gate_never_skips_segment_tail(self, simple_program) -> None:
        """A simple body longer than the per-tick quota must not skip ops when
        vm.ip jumps to the segment end (run_quota stops at its quota budget)."""
        from helixlang.core.bytecode import Chunk
        from helixlang.core.codon_table import Op
        from helixlang.core.vm import CellVM

        prog = parse_source(
            "#gene name=gfp\nATG GCT GGT GCT TAA\n#end\n"
            "#config ticks=20 ops_per_tick=4\n"
        )
        SemanticAnalyzer(prog).check()
        chunk = Chunk()
        chunk.gene_offsets["gfp"] = 0
        c0 = chunk.add_constant(2.0)
        chunk.emit(Op.OP_START)
        for _ in range(12):
            chunk.emit(Op.OP_PUSH_CONST, c0)
        chunk.emit(Op.OP_HALT)

        vm_pp = CellVM(chunk, prog, use_accel=False)
        trace_pp = vm_pp.run(20)
        vm_acc = CellVM(chunk, prog)
        trace_acc = vm_acc.run(20)

        assert trace_pp == trace_acc
        assert vm_pp.ops_executed == vm_acc.ops_executed

    def test_out_of_range_const_falls_back_to_python(self, simple_program) -> None:
        """PUSH_CONST with an operand that is not a constants-table index is
        dispatched in Python (run_quota would raise IndexError)."""
        from helixlang.core.bytecode import Chunk
        from helixlang.core.codon_table import Op
        from helixlang.core.vm import CellVM

        chunk = Chunk()
        chunk.gene_offsets["gfp"] = 0
        c0 = chunk.add_constant(1.0)
        chunk.emit(Op.OP_START)
        chunk.emit(Op.OP_PUSH_CONST, 0xF0)  # no such constant index
        chunk.emit(Op.OP_PUSH_CONST, c0)
        chunk.emit(Op.OP_PUSH_CONST, c0)
        chunk.emit(Op.OP_ADD)
        chunk.emit(Op.OP_HALT)

        vm = CellVM(chunk, simple_program)
        trace = vm.run(8)
        assert len(trace) == 8
        assert vm.ops_executed > 0
        assert vm.accel_native_ops < vm.ops_executed


class TestVMProfilerCounters:
    def test_profiler_reports_measured_counters(self, simple_program) -> None:
        profiler = VMProfiler(enable_tracemalloc=False)
        result = profiler.profile(simple_program, max_ticks=20)
        # doc/38 §2.2: accel_used is an OBSERVATION of real native ops, and with
        # the byte-identical segment executor even a bio-heavy body accelerates
        # its trailing per-tick HALT guard, so it legitimately reports on here.
        assert result.ops_executed > 0
        assert result.accel_used == (result.accel_ops > 0)
        assert result.accel_ops <= result.ops_executed
        assert result.ops_per_sec > 0

    def test_profiler_accel_off_reports_no_native_ops(self, simple_program) -> None:
        profiler = VMProfiler(enable_tracemalloc=False)
        result = profiler.profile(simple_program, max_ticks=20, use_accel=False)
        assert result.accel_used is False
        assert result.accel_ops == 0
        assert result.ops_executed > 0
        assert result.ops_per_sec > 0


# ============================================================================
# _segment_admissible edge cases
# ============================================================================
class TestSegmentAdmissibleEdgeCases:
    def test_pop_empty_stack_returns_none(self) -> None:
        """POP with kernel-stack depth=0 → inadmissible."""
        from helixlang.core.performance import _segment_admissible
        code = bytes([0x21])  # just POP
        assert _segment_admissible(code, 0, 1, n_consts=10) is None

    def test_pop_at_depth_one_succeeds(self) -> None:
        """PUSH_CONST + POP is admissible."""
        from helixlang.core.performance import _segment_admissible
        # PUSH_CONST 0x00, POP
        code = bytes([0x20, 0x00, 0x21])
        assert _segment_admissible(code, 0, 3, n_consts=10) == 2

    def test_add_needs_two_values(self) -> None:
        """ADD with depth < 2 → inadmissible."""
        from helixlang.core.performance import _segment_admissible
        # PUSH_CONST 0, ADD  (depth=1 when ADD reached)
        code = bytes([0x20, 0x00, 0x90])
        assert _segment_admissible(code, 0, 3, n_consts=10) is None

    def test_halt_only_segment(self) -> None:
        """HALT-only segment is admissible."""
        from helixlang.core.performance import _segment_admissible
        code = bytes([0x11])
        assert _segment_admissible(code, 0, 1, n_consts=10) == 1

    def test_push_const_out_of_range_returns_none(self) -> None:
        """PUSH_CONST with operand >= n_consts → inadmissible."""
        from helixlang.core.performance import _segment_admissible
        # PUSH_CONST 0x05 but n_consts=2
        code = bytes([0x20, 0x05])
        assert _segment_admissible(code, 0, 2, n_consts=2) is None

    def test_push_const_at_seg_end_returns_none(self) -> None:
        """PUSH_CONST whose operand byte is past seg_end → inadmissible."""
        from helixlang.core.performance import _segment_admissible
        # Segment covers only the opcode byte, not the operand
        code = bytes([0x20])
        assert _segment_admissible(code, 0, 1, n_consts=10) is None

    def test_empty_segment_returns_zero(self) -> None:
        """Empty segment [start, start) → 0 ops."""
        from helixlang.core.performance import _segment_admissible
        code = bytes([0x11])
        assert _segment_admissible(code, 0, 0, n_consts=10) == 0


# ============================================================================
# accelerated_execute_pending edge cases
# ============================================================================
class TestAcceleratedExecuteEdgeCases:
    def _make_vm(self, simple_program, *, code_ops=None):
        """Helper: build a minimal VM with a hand-crafted chunk."""
        from helixlang.core.bytecode import Chunk
        from helixlang.core.codon_table import Op
        from helixlang.core.vm import CellVM
        chunk = Chunk()
        chunk.gene_offsets["gfp"] = 0
        c0 = chunk.add_constant(1.0)
        if code_ops is None:
            chunk.emit(Op.OP_START)
            chunk.emit(Op.OP_PUSH_CONST, c0)
            chunk.emit(Op.OP_HALT)
        else:
            for op, *args in code_ops:
                chunk.emit(op, *args)
        return CellVM(chunk, simple_program)

    def test_quota_zero_returns_immediately(self, simple_program) -> None:
        """ops_per_tick=0 → function returns without execution."""
        vm = self._make_vm(simple_program)
        vm.program.config.ops_per_tick = 0
        accelerated_execute_pending(vm)
        assert vm.ops_executed == 0

    def test_import_failure_falls_back_to_python(self, monkeypatch,
                                                  simple_program) -> None:
        """run_quota import failure → can_accel=False → Python dispatch."""
        import sys
        import types
        fake = types.ModuleType('helixlang._accel.dispatch.backend')
        monkeypatch.setitem(sys.modules,
                            'helixlang._accel.dispatch.backend', fake)
        vm = self._make_vm(simple_program)
        from helixlang.core.vm import Frame
        vm.frames.append(Frame(return_ip=0, gene_name="gfp"))
        accelerated_execute_pending(vm)
        assert vm.ops_executed > 0

    def test_too_many_frames_clears_and_breaks(self, simple_program) -> None:
        """len(frames) > 256 → frames.clear() + break."""
        from helixlang.core.vm import Frame
        vm = self._make_vm(simple_program)
        vm.frames = [Frame(return_ip=0, gene_name="gfp") for _ in range(257)]
        accelerated_execute_pending(vm)
        assert vm.frames == []

    def test_ip_past_code_with_remaining_frames(self, simple_program) -> None:
        """ip >= len(code) with frames → pop frame, resume at return_ip."""
        from helixlang.core.vm import Frame
        vm = self._make_vm(simple_program)
        vm.ip = len(vm.chunk.code) + 1
        vm.frames = [
            Frame(return_ip=0, gene_name="gfp"),
            Frame(return_ip=0, gene_name="gfp"),
        ]
        accelerated_execute_pending(vm)
        assert len(vm.frames) == 1
        assert vm.ip == 0

    def test_ip_past_code_no_remaining_frames(self, simple_program) -> None:
        """ip >= len(code) with one frame → pop, break (no resume)."""
        from helixlang.core.vm import Frame
        vm = self._make_vm(simple_program)
        vm.ip = len(vm.chunk.code) + 1
        vm.frames = [Frame(return_ip=0, gene_name="gfp")]
        accelerated_execute_pending(vm)
        assert vm.frames == []

    def test_continue_after_accel_reaches_code_end(self, simple_program) -> None:
        """Accel segment consumes code to end → continue → frame pop."""
        from helixlang.core.bytecode import Chunk
        from helixlang.core.codon_table import Op
        from helixlang.core.vm import CellVM, Frame
        chunk = Chunk()
        chunk.gene_offsets["gfp"] = 0
        c0 = chunk.add_constant(1.0)
        # No OP_START — just simple ops so the whole code is one native segment
        chunk.emit(Op.OP_PUSH_CONST, c0)
        chunk.emit(Op.OP_POP)
        vm = CellVM(chunk, simple_program)
        vm.frames.append(Frame(return_ip=0, gene_name="gfp"))
        accelerated_execute_pending(vm)
        assert vm.accel_native_ops == 2
        assert vm.frames == []

    def test_unknown_opcode_raises(self, simple_program) -> None:
        """Unknown opcode byte → UnknownOpcodeError (strict runtime error)."""
        from helixlang.core.bytecode import Chunk
        from helixlang.core.codon_table import Op
        from helixlang.core.errors import UnknownOpcodeError
        from helixlang.core.vm import CellVM, Frame
        chunk = Chunk()
        chunk.gene_offsets["gfp"] = 0
        c0 = chunk.add_constant(1.0)
        chunk.emit(Op.OP_PUSH_CONST, c0)   # positions 0-1
        chunk.code.append(0xFF)              # position 2: unknown opcode
        chunk.emit(Op.OP_HALT)              # position 3
        vm = CellVM(chunk, simple_program)
        vm.frames.append(Frame(return_ip=0, gene_name="gfp"))
        with pytest.raises(UnknownOpcodeError):
            accelerated_execute_pending(vm)

    def test_debug_prints_to_stdout(self, simple_program, capsys) -> None:
        """vm.debug=True → debug info printed for each op."""
        from helixlang.core.vm import Frame
        vm = self._make_vm(simple_program)
        vm.frames.append(Frame(return_ip=0, gene_name="gfp"))
        vm.debug = True
        accelerated_execute_pending(vm)
        captured = capsys.readouterr()
        assert "[tick=" in captured.out
        assert "PUSH_CONST" in captured.out or "START" in captured.out

    def test_seg_ops_exceeds_quota_breaks(self, simple_program) -> None:
        """Segment with more ops than remaining quota → break (no accel)."""
        from helixlang.core.bytecode import Chunk
        from helixlang.core.codon_table import Op
        from helixlang.core.vm import CellVM, Frame
        chunk = Chunk()
        chunk.gene_offsets["gfp"] = 0
        c0 = chunk.add_constant(1.0)
        # 5 PUSH_CONST ops = 10 native ops (> small quota of 3)
        for _ in range(5):
            chunk.emit(Op.OP_PUSH_CONST, c0)
        chunk.emit(Op.OP_HALT)
        vm = CellVM(chunk, simple_program)
        vm.program.config.ops_per_tick = 3
        vm.frames.append(Frame(return_ip=0, gene_name="gfp"))
        accelerated_execute_pending(vm)
        # Segment had 11 ops (10 PUSH_CONST + 1 HALT) > quota=3, so
        # the accelerator breaks and falls back to Python dispatch for the
        # remaining quota.
        assert vm.ops_executed >= 3

    def test_non_simple_segment_gap_breaks(self, simple_program) -> None:
        """Non-simple op between simple segments → for-loop break (gap)."""
        from helixlang.core.bytecode import Chunk
        from helixlang.core.codon_table import Op
        from helixlang.core.vm import CellVM, Frame
        chunk = Chunk()
        chunk.gene_offsets["gfp"] = 0
        c0 = chunk.add_constant(1.0)
        chunk.emit(Op.OP_START)              # non-simple (not in _SIMPLE_OPS)
        chunk.emit(Op.OP_PUSH_CONST, c0)     # simple segment starts here
        chunk.emit(Op.OP_HALT)
        vm = CellVM(chunk, simple_program)
        vm.frames.append(Frame(return_ip=0, gene_name="gfp"))
        accelerated_execute_pending(vm)
        # OP_START dispatched in Python (1 op), then accel handles
        # PUSH_CONST+HALT if they form an admissible segment starting at ip=1.
        assert vm.ops_executed > 0


# ============================================================================
# VMProfiler edge cases
# ============================================================================
class TestVMProfilerEdgeCases:
    def test_tracemalloc_enabled(self, simple_program) -> None:
        """enable_tracemalloc=True → start/stop tracemalloc, record peak."""
        import tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        profiler = VMProfiler(enable_tracemalloc=True)
        result = profiler.profile(simple_program, max_ticks=5)
        assert result.peak_memory_bytes >= 0
        assert not tracemalloc.is_tracing()

    def test_tracemalloc_already_running(self, simple_program) -> None:
        """tracemalloc already running → don't restart, just read peak."""
        import tracemalloc
        tracemalloc.start()
        profiler = VMProfiler(enable_tracemalloc=True)
        result = profiler.profile(simple_program, max_ticks=5)
        assert result.peak_memory_bytes >= 0
        assert tracemalloc.is_tracing()
        tracemalloc.stop()

    def test_snapshot_interval_override(self, simple_program) -> None:
        """snapshot_interval kwarg → custom downsampler interval."""
        profiler = VMProfiler(enable_tracemalloc=False)
        result = profiler.profile(simple_program, max_ticks=10,
                                  snapshot_interval=5)
        assert result.ticks_executed == 10

    def test_zero_runtime_skips_rates(self, monkeypatch, simple_program) -> None:
        """vm_run_time_ms=0 → ticks_per_sec and ops_per_sec stay at 0."""
        import time
        monkeypatch.setattr(time, 'perf_counter', lambda: 0.0)
        profiler = VMProfiler(enable_tracemalloc=False)
        result = profiler.profile(simple_program, max_ticks=5)
        assert result.vm_run_time_ms == 0.0
        assert result.ticks_per_sec == 0.0
        assert result.ops_per_sec == 0.0

    def test_to_dict_roundtrip(self, simple_program) -> None:
        """VMProfileResult.to_dict returns correct structure."""
        import json
        profiler = VMProfiler(enable_tracemalloc=False)
        result = profiler.profile(simple_program, max_ticks=10)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "compile_time_ms" in d
        assert "vm_run_time_ms" in d
        assert "ticks_executed" in d
        assert "ops_executed" in d
        assert "accel_used" in d
        assert "component_times" in d
        # Verify JSON-serializable
        json.dumps(d)

    def test_component_times_recorded(self, simple_program) -> None:
        """profile() records compile and vm_run component times."""
        profiler = VMProfiler(enable_tracemalloc=False)
        result = profiler.profile(simple_program, max_ticks=5)
        assert "compile" in result.component_times
        assert "vm_run" in result.component_times
        assert result.component_times["compile"] >= 0
        assert result.component_times["vm_run"] >= 0
