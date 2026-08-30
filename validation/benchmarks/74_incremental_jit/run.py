#!/usr/bin/env python3
"""Benchmark 74: incremental JIT — closure-limited recompile (doc/38 §3).

Verifies the incremental-compiler invariants on a 16-gene call chain:
  - a single-gene edit re-derives exactly its dependency closure (leaf edit
    -> 2 genes, never the whole program)
  - each patch lowers to a Chunk byte-identical to a from-scratch compile
  - an unchanged source rebuilds nothing
and measures the cost gradient: summed per-edit compile time vs the naive
per-edit whole-program recompile.

Gate: full `release.py` + this diff bench (doc/38 §11 Phase C).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from helixlang.core.incr import IncrementalCompiler
from helixlang.core.language import LanguageConfig
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.core.semantic import SemanticAnalyzer

N_GENES = 16


def chain_source(n: int) -> str:
    """A gene call chain g0 <- g1 <- ... <- g_{n-1} (explicit call_targets)."""
    lines = []
    for i in range(n):
        target = "" if i == 0 else f" call_target=g_{i - 1}"
        body = "ATG TGG TAA" if i == 0 else "ATG CGG TAA"
        lines.append(f"#gene name=g_{i}{target}\n{body}\n#end")
    lines.append("#config ticks=2 output=stdout\n")
    return "\n".join(lines)


def parse(src: str):
    tokens = list(Lexer(src).tokens())
    program = Parser(tokens, config=LanguageConfig.for_table("standard")).parse()
    SemanticAnalyzer(program).check()
    return program


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        compiler = IncrementalCompiler(LanguageConfig.for_table("standard"))
        src = chain_source(N_GENES)
        program = parse(src)

        # ── base: naive full compile cost ─────────────────────────────────
        full_t = time.perf_counter()
        base = compiler.compile(program)
        full_seconds = time.perf_counter() - full_t
        checks["full_build_rebuilds_all"] = \
            base.stats.rebuilt == [f"g_{i}" for i in range(N_GENES)]

        # ── sequential single-codon edits, always byte-identical ─────────
        cur = src
        prev_ir, prev_cache = base.ir, base.cache
        incremental_seconds = 0.0
        rebuilt_sizes: list[int] = []
        byte_identical = True

        # pass 0: unchanged source must rebuild nothing
        res = compiler.compile(
            parse(src), previous_ir=prev_ir, previous_cache=prev_cache)
        checks["unchanged_source_rebuilds_nothing"] = res.stats.rebuilt == []
        prev_ir, prev_cache = res.ir, res.cache

        # pass 1: edit the leaf g_0 -> closure {g_0, g_1}, not the chain
        leaf_edit = cur.replace("#gene name=g_0\nATG TGG TAA",
                                "#gene name=g_0\nATG TCT TAA")
        t = time.perf_counter()
        res = compiler.compile(parse(leaf_edit),
                               previous_ir=prev_ir, previous_cache=prev_cache)
        incremental_seconds += time.perf_counter() - t
        prev_ir, prev_cache = res.ir, res.cache
        checks["leaf_edit_rebuilds_closure_only"] = \
            sorted(res.stats.rebuilt) == ["g_0", "g_1"]
        byte_identical = bytes(res.chunk.code) == \
            bytes(compiler.compile(parse(leaf_edit)).chunk.code)
        rebuilt_sizes.append(len(res.stats.rebuilt))
        cur = leaf_edit

        # passes 2..N-1: edit gene g_i (its closure is {g_i} U {g_{i+1}})
        for i in range(1, N_GENES):
            gene_i = f"g_{i}"
            edited = cur.replace(
                f"#gene name={gene_i} call_target=g_{i - 1}\nATG CGG TAA",
                f"#gene name={gene_i} call_target=g_{i - 1}\nATG CGC TAA")
            t = time.perf_counter()
            res = compiler.compile(
                parse(edited), previous_ir=prev_ir,
                previous_cache=prev_cache)
            incremental_seconds += time.perf_counter() - t
            prev_ir, prev_cache = res.ir, res.cache
            fresh = compiler.compile(parse(edited))
            byte_identical = byte_identical and \
                bytes(res.chunk.code) == bytes(fresh.chunk.code)
            rebuilt_sizes.append(len(res.stats.rebuilt))
            cur = edited

        checks["all_patches_byte_identical_to_full"] = byte_identical
        checks["edit_cost_proportional_to_closure"] = \
            max(rebuilt_sizes) <= 2 and sum(rebuilt_sizes) < N_GENES * 2

        details.update({
            "genes": N_GENES,
            "full_rebuild_seconds": full_seconds,
            "per_edit_closure_sizes": rebuilt_sizes,
            "incremental_total_seconds": incremental_seconds,
            "naive_recompile_total_seconds": full_seconds * (N_GENES - 1),
            "work_reduction_ratio":
                full_seconds * (N_GENES - 1) / (incremental_seconds or 1e-12),
        })

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "74_incremental_jit",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": "doc/38 §3 incremental JIT (core/incr.py)",
            "runtime_seconds": elapsed,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "id": "74_incremental_jit",
            "status": "FAIL",
            "error": str(e),
            "checks": checks,
            "details": details,
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] in ("PASS", "SKIP") else 1)