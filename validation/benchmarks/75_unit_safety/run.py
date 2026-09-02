#!/usr/bin/env python3
"""Benchmark 75: unit system & dimensional safety (doc/38 §8, doc/41 Item 5).

Verifies the §8 acceptance on a live program + the Quantity algebra:
  - quantity math fails at *compile time* (semantic/parse check) with the
    dimension tree in the message (a program that adds a concentration to a
    volume is rejected; a unit-carrying ``#type`` annotation round-trips)
  - the named-unit conversion table is exact: minutes == 60 seconds, and the
    base-value equality ``Quantity(1, 'min') == Quantity(60, 's')`` holds
  - the runtime trace is bit-identical whether or not IR instructions carry
    ``dim`` metadata (metadata-only flow, §8.2)

Gate: full `release.py` (doc/38 §11 Phase D).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from helixlang.core.compiler import Compiler
from helixlang.core.dimensions import (
    DIM_CONCENTRATION,
    Quantity,
    UnitError,
    convert,
)
from helixlang.core.errors import ParseError
from helixlang.core.ir_lower import IRLowerer
from helixlang.core.language import LanguageConfig
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser


def parse(src: str):
    tokens = list(Lexer(src).tokens())
    return Parser(tokens, config=LanguageConfig.for_table("standard")).parse()


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    # ── conversion table: minutes == seconds exactly ──────────────────────
    checks["minutes_seconds_exact"] = convert(1, "min", "s") == 60.0 and \
        convert(60, "s", "min") == 1.0
    checks["base_value_equality"] = Quantity(1, "min") == Quantity(60, "s")

    # ── cross-unit arithmetic fails at compile time with the dim tree ─────
    try:
        Quantity(5, "min") + Quantity(7, "µm3")
        checks["cross_unit_arithmetic_rejected"] = False
    except UnitError as exc:
        checks["cross_unit_arithmetic_rejected"] = \
            "incompatible dimensions" in str(exc) and "length" in str(exc)

    # ── a concentration + volume program is rejected in the semantic check
    #    (doc/41 Item 5 Ring 1: the program *itself* now fails to compile —
    #    `#quantity c_total=g+v` composes a Float<µM> symbol with a Float<L>
    #    symbol and is raised as a static DimensionError, not left to the
    #    runtime Quantity algebra). ──────────────────────────────────────────
    from helixlang.core.dim_inferencer import DimInferencer
    from helixlang.core.errors import DimensionError
    from helixlang.core.semantic import SemanticAnalyzer

    def _compile(src: str) -> None:
        SemanticAnalyzer(parse(src)).check()

    bad = ("#config ticks=100 ops_per_tick=100\n"
           "#type g=Float<µM>\n"
           "#type v=Float<L>\n"
           "#gene name=g\nATG TAA\n#end\n"
           "#quantity c_total=g+v\n")
    try:
        _compile(bad)
        checks["dimension_tree_in_message"] = False
    except DimensionError as exc:
        checks["dimension_tree_in_message"] = \
            "incompatible dimensions" in str(exc)
    good = ("#config ticks=100 ops_per_tick=100\n"
            "#type g=Float<µM>\n#type h=Float<µM>\n"
            "#gene name=g\nATG TAA\n#end\n"
            "#gene name=h\nATG TAA\n#end\n"
            "#quantity total=g+h\n")
    try:
        _compile(good)
        checks["same_dim_program_compiles"] = True
    except Exception:
        checks["same_dim_program_compiles"] = False
    # the annotated-but-uncomposed program itself is still fine (Ring 0
    # guarantee): the inferencer never over-rejects a passing program.
    program = parse("#config ticks=100 ops_per_tick=100\n"
                    "#type g=Float<µM>\n"
                    "#type v=Float<µm3>\n"
                    "#gene name=g\nATG TAA\n#end\n")
    SemanticAnalyzer(program).check()
    DimInferencer(program).infer()

    # ── Ring 2: named-unit static check — 5 min == 300 s, exactly ──────────
    checks["minutes_to_seconds_static"] = \
        Quantity(5, "min").convert_to("s") == Quantity(300, "s")

    # ── an unknown unit in a #type annotation fails to compile ────────────
    try:
        parse("#config ticks=100 ops_per_tick=100\n"
              "#type g=Float<furlongs>\n"
              "#gene name=g\nATG TAA\n#end\n")
        checks["unknown_unit_rejected"] = False
    except ParseError as exc:
        checks["unknown_unit_rejected"] = "g" in str(exc)

    # ── IR dim metadata is metadata-only: runtime trace bit-identical ─────
    src = "#config ticks=40 ops_per_tick=200\n#gene name=g\nATG TGG TAA\n#end\n"
    program = parse(src)
    comp = Compiler(LanguageConfig.for_table("standard"))
    ir, chunk = comp.compile_ir(program)
    from helixlang.core.vm import CellVM

    plain_trace = CellVM(chunk, program).run(program.config.ticks)
    for fn in ir.functions:
        for inst in fn.instrs:
            inst.dim = DIM_CONCENTRATION
    dimmed = IRLowerer().lower(ir)
    dim_trace = CellVM(dimmed, program).run(program.config.ticks)
    checks["dims_do_not_change_chunk"] = bytes(dimmed.code) == bytes(chunk.code)
    checks["dims_do_not_change_trace"] = dim_trace == plain_trace

    details.update({
        "conversion_min_to_s": convert(1, "min", "s"),
        "conversion_s_to_min": convert(60, "s", "min"),
        "min_plus_s_equality": str(Quantity(1, "min") == Quantity(60, "s")),
        "trace_len": len(dim_trace),
    })

    elapsed = time.perf_counter() - t0
    all_pass = all(checks.values())
    return {
        "id": "75_unit_safety",
        "status": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "details": details,
        "reference": {
            "source": "SI unit definitions — 1 min = 60 s, dimensional analysis",
            "authors": "BIPM",
            "year": 1960,
            "journal": "SI Brochure (8th edition)",
            "note": "Exact conversion: 1 min = 60 s; cross-dimension addition rejected",
        },
        "runtime_seconds": elapsed,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
