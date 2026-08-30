"""doc/38 §10 goal 12: seeded fuzz of the interpreter runtimes (doc/38 §10.3).

Random *valid* programs (compiler-built, not the dispatch subset) run under
both :class:`~helixlang.core.vm.CellVM` and
:class:`~helixlang.core.ir_runtime.IRRuntime` with small tick budgets.

Invariants:

- determinism — an identical run yields an identical trace;
- typed runtime errors only — any exception is in the ``RuntimeHelixError``
  family (``StackUnderflowError`` ...); never a bare ``IndexError`` crash;
- parity — ``CellVM`` and ``IRRuntime`` produce byte-identical traces, and
  every run's ``skipped_unknown`` stays 0 (compiler-built chunks are closed
  under the opcode table);
- acceleration (doc/38 §2.2) — the ``use_accel=True`` path yields the same
  trace and never reports more accel ops than ops executed.
"""
from __future__ import annotations

import random

from helixlang.core import vm
from helixlang.core.codon_table import STANDARD_TABLE
from helixlang.core.compiler import Compiler
from helixlang.core.errors import RuntimeHelixError
from helixlang.core.ir_builder import IRBuilder
from helixlang.core.ir_runtime import IRRuntime
from helixlang.core.parser import parse_source

# Standard-table ops only: every codon below maps to a real opcode, so
# compiler-built chunks are closed under the opcode table (skipped_unknown=0)
# and IR/bytecode paths agree bit-for-bit.
_BODY = ["GCT", "GGT", "GTA", "GAT", "GAA", "TGT"]
_START = "ATG"
_STOP = "TAA"

_TICKS = 2


def _valid_program(rng: random.Random):
    """A guaranteed-well-formed program (valid ORF per gene, standard table)."""
    genes: list[str] = []
    for _ in range(rng.randint(1, 2)):
        body = " ".join(rng.choice(_BODY) for _ in range(rng.randint(1, 6)))
        genes.append(f"#gene name=g{rng.randint(1, 2)}")
        genes.append(f"{_START} {body} {_STOP}")
        genes.append("#end")
    genes.append("#config ticks=2 output=stdout")
    return "\n".join(genes)


def _run_vm(make):
    """-> ("ok", trace) | ("err", exc); exceptions must be RuntimeHelixError."""
    try:
        return ("ok", make().run(_TICKS))
    except RuntimeHelixError as exc:  # typed runtime errors only
        return ("err", type(exc).__name__)
    except Exception as exc:  # noqa: BLE001 - anything else is a fuzz catch
        return ("boom", exc)


def test_fuzz_interp_determinism_typed_and_parity():
    """1000 seeded valid-program trials: determinism + typed errors + parity."""
    checked = 0
    for trial in range(1000):
        rng = random.Random(0x1FA12 + trial)
        src = _valid_program(rng)
        try:
            prog = parse_source(src)
        except Exception as exc:  # frontend fuzz owns the typed-error gate
            assert type(exc).__name__ in ("ParseError", "SemanticError",
                                          "LexError", "SimConfigError"), \
                f"trial {trial} untyped {type(exc).__name__}: {exc}"
            continue
        compiler = Compiler(table=STANDARD_TABLE)
        chunk = compiler.compile(prog)
        ir = IRBuilder(STANDARD_TABLE).build(prog)

        vm_a = _run_vm(
            lambda chunk=chunk, prog=prog: vm.CellVM(chunk, prog))
        assert vm_a[0] != "boom", f"trial {trial}: untyped CellVM error {vm_a[1]}"
        vm_b = _run_vm(
            lambda chunk=chunk, prog=prog: vm.CellVM(chunk, prog))
        assert vm_b == vm_a, f"trial {trial}: CellVM non-deterministic"
        ir_run = _run_vm(
            lambda ir=ir, prog=prog: IRRuntime(ir, prog))
        assert ir_run[0] != "boom", f"trial {trial}: untyped IR error {ir_run[1]}"
        # Parity where supported: identical success traces or identical typed
        # error classes.
        assert ir_run == vm_a, f"trial {trial}: CellVM/IRRuntime parity"
        # Instrumentation: compiler-built chunks never skip unknown opcodes.
        cell = vm.CellVM(chunk, prog)
        cell.run(_TICKS)
        assert cell.skipped_unknown == 0, f"trial {trial}: unknown opcode ran"
        checked += 1

        # Acceleration: accel path is observationally identical and never
        # over-reports native ops (doc/38 §2.2).
        accel = vm.CellVM(chunk, prog, use_accel=True)
        accel_trace = accel.run(_TICKS)
        assert accel_trace == vm_a[1], f"trial {trial}: accel trace mismatch"
        assert accel.accel_native_ops <= accel.ops_executed, \
            f"trial {trial}: accel_native_ops > ops_executed"
    assert checked >= 500, f"only {checked} valid trials; fuzzer too broken"
