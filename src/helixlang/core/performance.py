"""Performance optimization integration (doc/37 §3).

Provides:
- ``accelerated_execute_pending`` — C-dispatch-accelerated bytecode execution
- ``SnapshotDownsampler`` — bounded-memory trace accumulation
- ``VMProfiler`` — structured performance profiling

The C dispatch kernel handles the arithmetic/stack subset (HALT, PUSH_CONST,
POP, ADD, SUB, MUL) at ~50 ns/op.  For segments containing only these opcodes
we delegate to the native kernel; for segments with bio-opcodes we fall back to
the Python dispatcher.  ``_segment_admissible`` gates delegation so the
accelerator is **tick-for-tick equivalent** to the pure loop on the live operand
stack (doc/38 §2.2): the kernel only sees segments it can run on a fresh stack,
and HALT/quota accounting is restored to pure-loop semantics.
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from helixlang.core.vm import CellVM

_SIMPLE_OPS = frozenset({0x11, 0x20, 0x21, 0x90, 0x91, 0x92})
_OPERAND_BYTES: dict[int, int] = {
    0x11: 0, 0x20: 1, 0x21: 0, 0x90: 0, 0x91: 0, 0x92: 0,
    0x10: 0, 0x12: 0, 0x13: 0, 0x14: 1,
    0x30: 1, 0x31: 1, 0x32: 0,
    0x40: 1, 0x41: 1, 0x42: 1, 0x43: 1, 0x44: 1,
    0x50: 1, 0x51: 1, 0x52: 1, 0x53: 1,
    0x60: 1, 0x61: 1, 0x62: 1, 0x63: 1, 0x64: 1,
    0x70: 2, 0x80: 2, 0x81: 2,
    0x93: 0, 0x94: 0,
    0xF0: 0, 0xFE: 0,
}


def _find_simple_segments(code: bytes | bytearray, start: int, end: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    ip = start
    while ip < end:
        op = code[ip]
        if op in _SIMPLE_OPS:
            seg_start = ip
            while ip < end and code[ip] in _SIMPLE_OPS:
                op = code[ip]
                ip += 1 + _OPERAND_BYTES.get(op, 0)
            if ip > seg_start:
                segments.append((seg_start, ip))
        else:
            ip += 1 + _OPERAND_BYTES.get(op, 0)
    return segments


def _segment_admissible(code: bytes | bytearray, seg_start: int, seg_end: int,
                        n_consts: int) -> int | None:
    """Op count of ``[seg_start, seg_end)`` if the whole segment can be handed
    to ``run_quota`` transparently, else ``None``.

    Guards for exact parity with the Python dispatcher:

    - every PUSH_CONST operand must index an existing constant (the classic
      dispatcher falls back to pushing the raw byte, but ``run_quota`` would
      raise IndexError);
    - the segment must be executable on a *fresh* dispatch-kernel stack: the
      kernel starts empty and never sees values already on the VM operand
      stack, so every op is checked for its own precondition (POP needs one
      value, ADD/SUB/MUL need two).  Violations fall back to the Python
      dispatcher which reads the live stack.
    """
    depth = 0
    count = 0
    ip = seg_start
    while ip < seg_end:
        op = code[ip]
        if op == 0x11:  # HALT (per-gene return guard)
            ip += 1
            count += 1
        elif op == 0x20:  # PUSH_CONST  idx
            if ip + 1 >= seg_end or code[ip + 1] >= n_consts:
                return None
            depth += 1
            ip += 2
            count += 1
        elif op == 0x21:  # POP: needs one value on the kernel stack
            if depth < 1:
                return None
            depth -= 1
            ip += 1
            count += 1
        else:  # ADD / SUB / MUL: need two values on the kernel stack
            if depth < 2:
                return None
            depth -= 1
            ip += 1
            count += 1
    return count


def accelerated_execute_pending(vm: CellVM) -> None:
    """Execute bytecode with C-dispatch acceleration for arithmetic segments.

    Scans the current code region for contiguous blocks of simple opcodes
    (HALT, PUSH_CONST, POP, ADD, SUB, MUL) and runs them through the C
    dispatch kernel.  Bio-opcodes and control flow fall back to the Python
    dispatcher.

    The acceleration is transparent and **tick-for-tick equivalent** to the
    pure loop (doc/38 §2.2, asserted by ``tests/test_performance.py``):
    ``_segment_admissible`` only hands the kernel segments that are runnable on
    a fresh stack with in-range constants and that fit the remaining quota; a
    HALT resumes the caller within the same tick and is charged against the
    quota exactly like the pure dispatcher, so the (stack, ip, frames) state
    and per-tick snapshots are identical.
    """
    quota = vm.program.config.ops_per_tick
    if quota <= 0:
        return

    try:
        from helixlang._accel.dispatch.backend import run_quota
        can_accel = True
    except Exception:  # noqa: BLE001
        can_accel = False

    code = vm.chunk.code
    constants = vm.chunk.constants

    while vm.frames and quota > 0:
        if len(vm.frames) > 256:
            vm.frames.clear()
            break
        if vm.ip >= len(code):
            vm.frames.pop()
            if vm.frames:
                vm.ip = vm.frames[-1].return_ip
            break

        if can_accel and not vm.debug:
            seg_end = min(vm.ip + quota * 3, len(code))
            segments = _find_simple_segments(code, vm.ip, seg_end)
            for seg_start, seg_end_pos in segments:
                if seg_start != vm.ip:
                    break
                # run_quota returns only (ops_consumed, stack, halted), not
                # the final instruction index, so accelerate only segments
                # whose whole op stream fits inside the remaining quota;
                # otherwise the tail of the segment would be skipped when
                # vm.ip jumps to seg_end_pos.
                seg_ops = _segment_admissible(code, seg_start, seg_end_pos,
                                              len(constants))
                if seg_ops is None or seg_ops > quota:
                    break
                seg_code = bytes(code[seg_start:seg_end_pos])
                seg_consts = list(constants)
                ops_consumed, stack_result, halted = run_quota(
                    seg_code, seg_consts, quota=quota)
                for v in stack_result:
                    vm.stack.append(v)
                vm.ip = seg_end_pos
                quota -= ops_consumed
                # doc/38 §2.2: count native ops as executed, so accel_used is
                # an observation of what really ran rather than a request.
                vm.ops_executed += ops_consumed
                vm.accel_native_ops += ops_consumed
                if halted:
                    # run_quota reports ops_consumed excluding the HALT, but
                    # the pure loop charges quota for it like any op; restore
                    # identical tick-boundary accounting.
                    vm.ops_executed += 1
                    vm.accel_native_ops += 1
                    quota -= 1
                    returned = vm.frames.pop()
                    # HALT is the per-gene return guard: resume at the address
                    # recorded on the returned frame (pure dispatcher: pops one
                    # frame and uses ITS return address) and keep going within
                    # this tick exactly like the pure loop.
                    if vm.frames:
                        vm.ip = returned.return_ip
                    else:
                        return
            # A segment that consumed the whole code region leaves ip at
            # len(code); the pure loop re-checks against the frame bound at the
            # top of every iteration, so do the same here before falling into
            # the Python dispatch below (code[vm.ip] would be out of range).
            if vm.ip >= len(code):
                continue

        op_byte = code[vm.ip]
        vm.ip += 1
        try:
            from helixlang.core.codon_table import Op
            op = Op(op_byte)
        except ValueError:
            # Strict runtime error (doc/38): the accelerated path must agree
            # with the pure loop — an unknown opcode is never silently skipped.
            from helixlang.core.errors import UnknownOpcodeError
            raise UnknownOpcodeError(opcode=op_byte, ip=vm.ip - 1) from None
        if vm.debug:
            print(f"[tick={vm.tick} ip={vm.ip - 1}] "
                  f"{op.name} stack={vm.stack}")
        vm._dispatch(op)
        quota -= 1


@dataclass(slots=True)
class SnapshotDownsampler:
    """Bounded-memory trace accumulation via snapshot downsampling.

    For long simulations (10k+ ticks), storing every tick's snapshot causes
    O(ticks) memory growth.  This downsampler only stores every Nth tick,
    plus tick 0 and the final tick.
    """
    interval: int = 1
    max_ticks_threshold: int = 1000
    _tick_counter: int = field(default=0, init=False)
    _final_tick: int = field(default=-1, init=False)

    def should_snapshot(self, tick: int, max_ticks: int) -> bool:
        if tick == 0:
            return True
        if self.interval <= 1:
            return True
        return tick % self.interval == 0

    def is_final(self, tick: int, max_ticks: int, alive: bool) -> bool:
        return tick >= max_ticks - 1 or not alive

    def compute_interval(self, max_ticks: int) -> int:
        if max_ticks <= self.max_ticks_threshold:
            return 1
        return max(1, max_ticks // 500)

    def configure(self, max_ticks: int) -> None:
        self.interval = self.compute_interval(max_ticks)


@dataclass(slots=True)
class VMProfileResult:
    """Structured performance profile of a VM execution."""
    compile_time_ms: float = 0.0
    vm_run_time_ms: float = 0.0
    total_time_ms: float = 0.0
    ticks_executed: int = 0
    ops_executed: int = 0
    ops_per_sec: float = 0.0
    ticks_per_sec: float = 0.0
    peak_memory_bytes: int = 0
    trace_entries: int = 0
    snapshot_interval: int = 1
    accel_used: bool = False
    accel_ops: int = 0
    component_times: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compile_time_ms": round(self.compile_time_ms, 2),
            "vm_run_time_ms": round(self.vm_run_time_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "ticks_executed": self.ticks_executed,
            "ops_executed": self.ops_executed,
            "ops_per_sec": round(self.ops_per_sec, 0),
            "ticks_per_sec": round(self.ticks_per_sec, 1),
            "peak_memory_bytes": self.peak_memory_bytes,
            "trace_entries": self.trace_entries,
            "snapshot_interval": self.snapshot_interval,
            "accel_used": self.accel_used,
            "accel_ops": self.accel_ops,
            "component_times": {
                k: round(v, 2) for k, v in self.component_times.items()
            },
        }


class VMProfiler:
    """Comprehensive profiling harness for HelixLang VM execution.

    Measures compile time, VM run time, memory usage, and per-component
    breakdown.  Outputs a structured ``VMProfileResult``.
    """

    def __init__(self, enable_tracemalloc: bool = True) -> None:
        self._enable_tracemalloc = enable_tracemalloc

    def profile(
        self,
        program: Any,
        *,
        max_ticks: int = 100,
        use_accel: bool = True,
        snapshot_interval: int | None = None,
    ) -> VMProfileResult:
        from helixlang.core.compiler import Compiler
        from helixlang.core.vm import CellVM

        result = VMProfileResult()

        tracemalloc_was_running = tracemalloc.is_tracing()
        if self._enable_tracemalloc and not tracemalloc_was_running:
            tracemalloc.start()

        t0 = time.perf_counter()
        compiler = Compiler()
        chunk = compiler.compile(program)
        t_compile = time.perf_counter()

        result.compile_time_ms = (t_compile - t0) * 1000
        result.component_times["compile"] = result.compile_time_ms

        vm = CellVM(chunk, program, use_accel=use_accel)

        if snapshot_interval is not None:
            vm._snapshot_downsampler = SnapshotDownsampler(interval=snapshot_interval)
        else:
            ds = SnapshotDownsampler()
            ds.configure(max_ticks)
            vm._snapshot_downsampler = ds
            result.snapshot_interval = ds.interval

        t1 = time.perf_counter()
        trace = vm.run(max_ticks)
        t_run = time.perf_counter()

        result.vm_run_time_ms = (t_run - t1) * 1000
        result.total_time_ms = (t_run - t0) * 1000
        result.ticks_executed = vm.tick
        result.trace_entries = len(trace)
        result.component_times["vm_run"] = result.vm_run_time_ms

        if result.vm_run_time_ms > 0:
            result.ticks_per_sec = (
                result.ticks_executed / (result.vm_run_time_ms / 1000))
        # doc/38 §2.1: ops_per_sec is measured from the counters the VM really
        # accumulated, never a placeholder.
        result.ops_executed = vm.ops_executed
        result.accel_ops = vm.accel_native_ops
        result.accel_used = vm.accel_native_ops > 0
        if result.vm_run_time_ms > 0:
            result.ops_per_sec = (
                result.ops_executed / (result.vm_run_time_ms / 1000))

        if self._enable_tracemalloc and not tracemalloc_was_running:
            current, peak = tracemalloc.get_traced_memory()
            result.peak_memory_bytes = peak
            tracemalloc.stop()
        elif tracemalloc_was_running:
            current, peak = tracemalloc.get_traced_memory()
            result.peak_memory_bytes = peak

        return result
