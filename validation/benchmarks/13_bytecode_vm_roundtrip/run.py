#!/usr/bin/env python3
"""Benchmark 13: Bytecode/VM roundtrip — compile → serialize → deserialize → execute."""
from __future__ import annotations

import json
import sys
import time

SOURCE = """\
#gene name=lacI
ATG AAA TAT ACC GCT TCA CCG GAT AAA ACG TAA
#end
#config ticks=5
"""


def _build_chunk_and_program():
    from helixlang.codon_table import STANDARD_TABLE
    from helixlang.compiler import Compiler
    from helixlang.lexer import Lexer
    from helixlang.parser import Parser

    tokens = list(Lexer(SOURCE).tokens())
    program = Parser(tokens).parse()
    chunk = Compiler(STANDARD_TABLE).compile(program)
    return program, chunk


def _run_vm(chunk, program, ticks):
    from helixlang.vm import CellVM
    vm = CellVM(chunk, program)
    return vm.run(ticks)


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "13_bytecode_vm_roundtrip"}
    try:
        from helixlang.hxbc import dumps_program, loads_program

        program, chunk = _build_chunk_and_program()

        serialized = dumps_program(program, chunk=chunk, source=SOURCE)
        assert len(serialized) > 0, "dumps_program returned empty bytes"
        loaded = loads_program(serialized)
        assert loaded.program is not None, "loads_program returned no program"
        assert loaded.chunk is not None, "loads_program returned no chunk"
        assert loaded.source == SOURCE, "source text did not roundtrip"
        assert loaded.chunk.code == chunk.code, "bytecode did not roundtrip"
        assert loaded.chunk.gene_offsets == chunk.gene_offsets, (
            "gene_offsets did not roundtrip"
        )

        trace1 = _run_vm(chunk, program, 5)
        trace2 = _run_vm(loaded.chunk, loaded.program, 5)
        assert len(trace1) == len(trace2), (
            f"traces differ in length: {len(trace1)} vs {len(trace2)}"
        )
        assert trace1[-1]["energy"] == trace2[-1]["energy"], (
            "VM execution not deterministic: energy differs"
        )

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "serialized_bytes": len(serialized),
            "trace_ticks": len(trace1),
            "deterministic": trace1[-1]["energy"] == trace2[-1]["energy"],
            "final_energy": trace1[-1]["energy"],
            "runtime_seconds": elapsed,
        })
    except Exception as e:
        results.update({
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        })
    return results


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
