#!/usr/bin/env python3
"""Benchmark 73: HLIR serialization robustness (doc/37 §5.3, §6).

The typed IR must be a portable artifact: JSON writes/reads round-trip with
full fidelity (typed operands, metadata, config snapshot), a version-capable
reader refuses future payloads instead of mis-decoding, and corrupt input is
rejected loudly.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from helixlang.core.ast_nodes import LSystemDecl, UseDecl
from helixlang.core.codon_table import STANDARD_TABLE, Op
from helixlang.core.ir import IRType
from helixlang.core.ir_builder import IRBuilder
from helixlang.core.ir_serialize import IRFormatError, dumps, from_dict, loads, to_dict
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser

# Arithmetic mapping on top of the standard base (mirrors tests/test_ir.py).
TEST_TABLE: dict[str, Op] = {
    **STANDARD_TABLE,
    "AAA": Op.OP_CALL_GENE,
    "TCT": Op.OP_PUSH_CONST,   # wobble 3
    "TCC": Op.OP_PUSH_CONST,   # wobble 1
    "CCT": Op.OP_ADD,
    "TGG": Op.OP_BUILD_PROTEIN,
}

RICH = ("#gene name=caller call_target=target\nATG TCT AAA TAA\n#end\n"
        "#gene name=target\nATG TCC TGG TAA\n#end")


def _parse(src: str, table: dict[str, Op]):
    stop = {c for c, op in table.items() if op == Op.OP_HALT}
    return Parser(list(Lexer(src).tokens()), stop_codons=stop).parse()


def _rich_ir():
    prog = _parse(RICH, TEST_TABLE)
    prog.lsystems["tree"] = LSystemDecl(name="tree", axiom="F", angle=25.7,
                                        step=1.0, rules={})
    prog.use_directives = [UseDecl(plugin="grn", flags=frozenset(["hill"]))]
    prog.config.ticks = 42
    prog.config.ops_per_tick = 32
    return IRBuilder(TEST_TABLE).build(prog), prog


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        # --- 1. rich feature-set round trip is identical ---
        ir, prog = _rich_ir()
        rt = loads(dumps(ir))
        checks["rich_roundtrip"] = rt.disassemble() == ir.disassemble()
        details["genes"] = ir.gene_names()

        # --- 2. every metadata field survives the round trip ---
        checks["metadata_preserved"] = (
            rt.name == ir.name and rt.table == ir.table
            and rt.call_targets == ir.call_targets
            and rt.use_directives == ir.use_directives
            and rt.lsystems == ir.lsystems
            and rt.config.get("ticks") == 42
            and rt.config.get("ops_per_tick") == 32)
        details["call_targets"] = rt.call_targets
        details["lsystems"] = rt.lsystems
        details["config_ticks"] = rt.config.get("ticks")

        # --- 3. typed operands survive (PUSH_CONST literals, value_type) ---
        pushes_orig = [i for i in ir.functions[0].instrs if i.opcode is Op.OP_PUSH_CONST]
        pushes_rt = [i for i in rt.functions[0].instrs if i.opcode is Op.OP_PUSH_CONST]
        checks["typed_operands"] = (
            len(pushes_orig) == len(pushes_rt)
            and all(a.operand == b.operand and a.value_type == b.value_type
                    for a, b in zip(pushes_orig, pushes_rt, strict=True))
            and all(i.value_type is IRType.NUM for i in pushes_rt))

        # --- 4. to_dict/from_dict API mirrors the text round trip ---
        checks["dict_roundtrip"] = from_dict(to_dict(ir)).disassemble() == ir.disassemble()
        checks["dump_deterministic"] = dumps(ir) == dumps(from_dict(to_dict(ir)))

        # --- 5. version guard: a future payload is refused, not mis-decoded ---
        payload = to_dict(ir)
        payload["version"] = 999
        guard_hit = False
        try:
            loads(json.dumps(payload))
        except IRFormatError:
            guard_hit = True
        checks["version_guard"] = guard_hit

        # --- 6. corrupt payloads are rejected loudly ---
        bad = False
        try:
            loads('{"version": 1, "functions": [}')
        except (IRFormatError, ValueError, json.JSONDecodeError, KeyError):
            bad = True
        checks["corrupt_rejected"] = bad

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "73_ir_serialization",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": "doc/37 §5.3 HLIR serialization (ir_serialize.py)",
            "runtime_seconds": elapsed,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "id": "73_ir_serialization",
            "status": "FAIL",
            "error": str(e),
            "checks": checks,
            "details": details,
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
