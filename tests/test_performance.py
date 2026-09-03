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
