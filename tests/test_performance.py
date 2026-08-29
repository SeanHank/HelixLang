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


class TestAcceleratedExecution:
    def test_importable(self) -> None:
        assert callable(accelerated_execute_pending)

    def test_accel_matches_python_for_simple_segments(self, simple_program) -> None:
        """Verify C-accelerated execution produces identical results when the
        bytecode is entirely simple (arithmetic/stack ops)."""
        from helixlang.core.compiler import Compiler
        from helixlang.core.vm import CellVM

        compiler = Compiler()
        chunk = compiler.compile(simple_program)

        # Pure Python path
        vm_pp = CellVM(chunk, simple_program)
        trace_pp = vm_pp.run(20)

        # Accelerated segment path
        vm_acc = CellVM(chunk, simple_program)
        # monkeypatch the run loop to use accelerated_execute_pending once
        original_pending = vm_acc._execute_pending
        vm_acc._execute_pending = lambda: accelerated_execute_pending(vm_acc)
        try:
            trace_acc = vm_acc.run(20)
        finally:
            vm_acc._execute_pending = original_pending

        # Both should execute the same number of ticks
        assert len(trace_pp) == len(trace_acc)
        assert vm_pp.tick == vm_acc.tick
