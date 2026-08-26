#!/usr/bin/env python3
"""Benchmark 02: lac operon — GRN quantitative repression verification."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

EXAMPLES = Path(__file__).parent.parent.parent.parent / "examples"


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "02_lac_operon"}
    try:
        from helixlang.codon_table import STANDARD_TABLE, Op
        from helixlang.compiler import Compiler
        from helixlang.lexer import Lexer
        from helixlang.parser import Parser
        from helixlang.vm import CellVM

        source = (EXAMPLES / "02_lac_operon.helix").read_text()
        stop = {c for c, op in STANDARD_TABLE.items() if op == Op.OP_HALT}
        toks = list(Lexer(source).tokens())
        prog = Parser(toks, stop_codons=stop).parse()
        chunk = Compiler(STANDARD_TABLE).compile(prog)
        vm = CellVM(chunk, prog)
        trace = vm.run(200)

        # --- Quantitative checks (steady-state: last 50 ticks) ---
        tail = trace[-50:] if len(trace) >= 50 else trace

        def get_steady_levels(gene: str) -> list[float]:
            return [snap["gene_levels"].get(gene, 0.0) for snap in tail]

        lacI_levels = get_steady_levels("lacI")
        lacZ_levels = get_steady_levels("lacZ")
        lacY_levels = get_steady_levels("lacY")

        lacI_mean = sum(lacI_levels) / len(lacI_levels) if lacI_levels else 0.0
        lacZ_mean = sum(lacZ_levels) / len(lacZ_levels) if lacZ_levels else 0.0
        lacY_mean = sum(lacY_levels) / len(lacY_levels) if lacY_levels else 0.0

        # p_lacI is constitutive (strength=-0.5 < 0) → lacI should be elevated
        lacI_high = lacI_mean > 0.4

        # lacI represses p_lac → lacZ and lacY should be low
        lacZ_low = lacZ_mean < 0.3
        lacY_low = lacY_mean < 0.3

        # Repression: lacI should be higher than lacZ
        repression_ratio = (lacI_mean / lacZ_mean) if lacZ_mean > 0.001 else float("inf")
        strong_repression = repression_ratio > 1.2

        all_ok = lacI_high and lacZ_low and lacY_low and strong_repression

        # GRN structure check
        has_grn = len(vm.grn.nodes) > 0

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS" if all_ok and has_grn else "FAIL",
            "expected": {
                "grn_nodes": True,
                "lacI_steady_above": 0.4,
                "lacZ_steady_below": 0.3,
                "lacY_steady_below": 0.3,
                "repression_ratio_above": 1.2,
            },
            "actual": {
                "grn_node_count": len(vm.grn.nodes),
                "grn_edge_count": len(vm.grn.edges),
                "ticks_run": len(trace),
                "lacI_steady_mean": round(lacI_mean, 4),
                "lacZ_steady_mean": round(lacZ_mean, 4),
                "lacY_steady_mean": round(lacY_mean, 4),
                "repression_ratio": round(repression_ratio, 2),
                "lacI_high": lacI_high,
                "lacZ_low": lacZ_low,
                "lacY_low": lacY_low,
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
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
