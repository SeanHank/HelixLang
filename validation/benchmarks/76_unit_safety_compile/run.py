#!/usr/bin/env python3
"""Benchmark 76: compile-time dimensional rejection (doc/41 Item 5 Ring 3).

Controlled rejection: a `.helix` program that composes a ``Float<µM>`` symbol
with a ``Float<L>`` symbol now fails during the semantic compile-time check
(:class:`SemanticAnalyzer.check`) with a :class:`DimensionError` naming both
dimension trees — previous behaviour left it to the runtime Quantity algebra.

Also re-verifies Ring 2's static ``#config`` quantity round-trip
(5 min == 300 s). This benchmark is the "new controlled currently-failing
program" called for in doc/41 §6.3 acceptance item 1.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from helixlang.core.dimensions import Quantity, UnitError
from helixlang.core.dim_inferencer import DimInferencer
from helixlang.core.errors import DimensionError
from helixlang.core.language import LanguageConfig
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.core.semantic import SemanticAnalyzer


def parse(src: str):
    tokens = list(Lexer(src).tokens())
    return Parser(tokens, config=LanguageConfig.for_table("standard")).parse()


def _compile(src: str) -> None:
    program = parse(src)
    SemanticAnalyzer(program).check()
    DimInferencer(program).infer()


_BASIC = "#config ticks=100 ops_per_tick=100\n"


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}

    # ── acceptance 1: Float<µM> + Float<L> fails at compile ───────────────
    bad = (_BASIC +
           "#type g=Float<µM>\n#type v=Float<L>\n"
           "#gene name=g\nATG TAA\n#end\n"
           "#quantity c_total=g+v\n")
    try:
        _compile(bad)
        checks["cross_unit_symbol_add_rejected"] = False
    except DimensionError as exc:
        checks["cross_unit_symbol_add_rejected"] = True
        msg = str(exc)
        checks["dimension_tree_in_message"] = \
            "incompatible dimensions" in msg and "length" in msg and "amount" in msg
        checks["millimolar_plus_litre_named"] = \
            "µM" in msg or "molar" in msg or "amount" in msg

    # same-dim composition must still compile (never over-reject)
    good = (_BASIC +
            "#type g=Float<µM>\n#type h=Float<µM>\n"
            "#gene name=g\nATG TAA\n#end\n"
            "#gene name=h\nATG TAA\n#end\n"
            "#quantity total=g+h\n")
    try:
        _compile(good)
        checks["same_dim_program_compiles"] = True
    except Exception:
        checks["same_dim_program_compiles"] = False

    # ── Ring 2 static `#config` quantity round-trip (5 min == 300 s) ──────
    try:
        q = Quantity(5, "min").convert_to("s")
        checks["minutes_to_seconds_static"] = q == Quantity(300, "s")
        checks["config_quantity_round_trip"] = \
            Quantity(300, "s").convert_to("min") == Quantity(5, "min")
    except UnitError:
        checks["minutes_to_seconds_static"] = False
        checks["config_quantity_round_trip"] = False

    all_pass = all(checks.values())
    return {
        "id": "76_unit_safety_compile",
        "status": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "details": {"5min_in_s": 300, "converted": "5min == 300s"},
        "reference": "doc/41 §6.3 acceptance — compile-time DimensionError "
                     "(core/dim_inferencer.py, core/semantic.py)",
        "runtime_seconds": time.perf_counter() - t0,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
