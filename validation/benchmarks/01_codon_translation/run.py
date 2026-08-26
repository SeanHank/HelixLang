#!/usr/bin/env python3
"""Benchmark 01: Codon translation — 64-codon mapping + VM translation product."""
from __future__ import annotations

import json
import sys
import time

from helixlang.codon_table import STANDARD_TABLE, Op

ALL_CODONS = [
    "".join(b) for b in [
        (a, c, g)
        for a in "ACGT" for c in "ACGT" for g in "ACGT"
    ]
]

SOURCE = """\
#gene name=test
ATG GCT TAA
#end
"""


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "01_codon_translation"}
    try:
        # --- Part A: 64-codon mapping check ---
        errors = []
        for codon in ALL_CODONS:
            op = STANDARD_TABLE.get(codon)
            if op is None:
                errors.append(f"{codon}: not in table")
            elif not isinstance(op, Op):
                errors.append(f"{codon}: not an Op enum")

        mapped = sum(1 for c in ALL_CODONS if c in STANDARD_TABLE)
        stop_codons = [c for c in ALL_CODONS if STANDARD_TABLE.get(c) == Op.OP_HALT]

        # --- Part B: VM translation of known sequence ---
        from helixlang.compiler import Compiler
        from helixlang.lexer import Lexer
        from helixlang.parser import Parser
        from helixlang.vm import CellVM

        stop_set = {c for c, op in STANDARD_TABLE.items() if op == Op.OP_HALT}
        toks = list(Lexer(SOURCE).tokens())
        prog = Parser(toks, stop_codons=stop_set).parse()
        chunk = Compiler(STANDARD_TABLE).compile(prog)
        vm = CellVM(chunk, prog)
        vm.run(50)

        # OP_BUILD_PROTEIN stores by wobble integer key; GCT → wobble(T)=3
        has_protein = len(vm.cell.proteins) > 0
        protein_amount = sum(vm.cell.proteins.values())

        # Verify stop codons map to OP_HALT
        expected_stops = {"TAA", "TAG", "TGA"}
        actual_stops = {c for c in stop_codons}
        stop_codon_ok = expected_stops == actual_stops

        mapping_ok = not errors and mapped == 64
        vm_ok = has_protein and stop_codon_ok
        all_ok = mapping_ok and vm_ok

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS" if all_ok else "FAIL",
            "expected": {
                "codon_count": 64,
                "all_mapped": True,
                "stop_codons": sorted(expected_stops),
                "vm_produces_protein": True,
            },
            "actual": {
                "codon_count": len(ALL_CODONS),
                "mapped_count": mapped,
                "stop_codon_count": len(stop_codons),
                "stop_codons": sorted(actual_stops),
                "mapping_errors": errors[:10],
                "vm_protein_produced": has_protein,
                "vm_protein_total": round(protein_amount, 4),
                "vm_ticks_run": vm.tick,
            },
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
