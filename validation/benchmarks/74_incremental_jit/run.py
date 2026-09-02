#!/usr/bin/env python3
"""Benchmark 74: incremental JIT — closure-limited recompile (doc/38 §3).

Verifies the incremental-compiler invariants on a 16-gene call chain:
  - a single-gene edit re-derives exactly its dependency closure (leaf edit
    -> 2 genes, never the whole program)
  - each patch lowers to a Chunk byte-identical to a from-scratch compile
  - an unchanged source rebuilds nothing
and measures the compile-cost gradient: summed per-edit *compile* time vs the
summed whole-program *compile* time a naive edit loop would spend.

Both arms isolate the compile phase (the naive arm never pays the parse cost,
so the incremental arm must not either) — the parse of the edited source is
shared and identical in a real watch loop.  Edits are same-width codon swaps
(SIGNAL wobble ``TCT <-> TCC``, CALL_GENE ``CGG <-> CGC``), so every edit is
splice-safe and the measured incremental compile is proportional to the
dependency closure (~2 genes), not the 16-gene program.

Gate: full `release.py` + this diff bench (doc/38 §11 Phase C): the
incremental compile must be faster than a whole-program compile
(``work_reduction_ratio >= 1``).
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
        body = "ATG TCT TAA" if i == 0 else "ATG CGG TAA"
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

        # ── base: full compile (compile-only, warm reference) ─────────────
        base = compiler.compile(parse(src))
        checks["full_build_rebuilds_all"] = \
            base.stats.rebuilt == [f"g_{i}" for i in range(N_GENES)]

        # ── sequential single-codon edits ────────────────────────────────
        prev_ir, prev_cache, prev_chunk = base.ir, base.cache, base.chunk
        incremental_seconds = 0.0
        naive_seconds = 0.0
        rebuilt_sizes: list[int] = []
        splices: list[bool] = []
        byte_identical = True

        # pass 0: unchanged source must rebuild nothing
        res = compiler.compile(
            parse(src), previous_ir=prev_ir, previous_cache=prev_cache,
            previous_chunk=prev_chunk)
        checks["unchanged_source_rebuilds_nothing"] = res.stats.rebuilt == []
        prev_ir, prev_cache, prev_chunk = res.ir, res.cache, res.chunk

        for i in range(N_GENES):
            gene = f"g_{i}"
            if i == 0:
                old_body, new_body = "ATG TCT TAA", "ATG TCC TAA"
            else:
                old_body = f"ATG CGG TAA"  # CALL_GENE wobble 3 (modulo n -> n-1)
                new_body = f"ATG CGC TAA"
            edited = src.replace(f"#gene name={gene}\n{old_body}",
                                 f"#gene name={gene}\n{new_body}") \
                if i == 0 else \
                src.replace(
                    f"#gene name={gene} call_target=g_{i - 1}\n{old_body}",
                    f"#gene name={gene} call_target=g_{i - 1}\n{new_body}")
            program_i = parse(edited)
            # incremental compile: proportional to the closure
            t = time.perf_counter()
            res = compiler.compile(
                program_i, previous_ir=prev_ir, previous_cache=prev_cache,
                previous_chunk=prev_chunk)
            incremental_seconds += time.perf_counter() - t
            prev_ir, prev_cache, prev_chunk = res.ir, res.cache, res.chunk
            # whole-program reference compile of the SAME source
            t = time.perf_counter()
            fresh = compiler.compile(program_i)
            naive_seconds += time.perf_counter() - t
            byte_identical = byte_identical and \
                bytes(res.chunk.code) == bytes(fresh.chunk.code)
            if i == 0:
                checks["leaf_edit_rebuilds_closure_only"] = \
                    sorted(res.stats.rebuilt) == ["g_0", "g_1"]
            rebuilt_sizes.append(len(res.stats.rebuilt))
            splices.append(res.stats.splice)
            src = edited

        checks["all_patches_byte_identical_to_full"] = byte_identical
        checks["edit_cost_proportional_to_closure"] = \
            max(rebuilt_sizes) <= 2 and sum(rebuilt_sizes) < N_GENES * 2
        checks["all_edits_splice_safe"] = all(splices)
        # the incremental COMPILE must beat the whole-program COMPILE
        ratio = naive_seconds / (incremental_seconds or 1e-12)
        checks["incremental_beats_naive_recompile"] = ratio >= 1.0

        details.update({
            "genes": N_GENES,
            "per_edit_closure_sizes": rebuilt_sizes,
            "edits_spliced": sum(splices),
            "incremental_compile_seconds": incremental_seconds,
            "naive_recompile_seconds": naive_seconds,
            "work_reduction_ratio": ratio,
        })

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "74_incremental_jit",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": "doc/38 §3 incremental JIT (core/incr.py, "
                         "splice fast-path)",
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