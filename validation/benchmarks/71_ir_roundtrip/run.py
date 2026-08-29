#!/usr/bin/env python3
"""Benchmark 71: Helix IR pipeline round-trip & optimizer correctness (doc/37 §5-6).

Verifies the first-class IR invariants:
  - builder: one typed IR function per gene, faithful opcode stream
  - lowering: byte-identical to the bytecode ABI (golden chunk code)
  - optimization: pass 1p semantic parity — CellVM(plain) == CellVM(optimized),
    idempotent, and never rewrites effect boundaries
  - serialization: HLIR JSON dumps/loads is a stable round trip
  - metadata: call_target resolution + config snapshot are preserved
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from helixlang.core.codon_table import STANDARD_TABLE, Op
from helixlang.core.compiler import Compiler
from helixlang.core.ir_builder import IRBuilder
from helixlang.core.ir_lower import lower
from helixlang.core.ir_opt import optimize_program
from helixlang.core.ir_serialize import dumps, loads
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.core.vm import CellVM

# Arithmetic mapping on top of the standard base (mirrors tests/test_ir.py).
# The standard table maps no codon to pure arithmetic opcodes, so optimizer
# checks use this table exactly like the unit tests do.
TEST_TABLE: dict[str, Op] = {
    **STANDARD_TABLE,
    "AAA": Op.OP_CALL_GENE,
    "TCT": Op.OP_PUSH_CONST,   # wobble 3
    "TCC": Op.OP_PUSH_CONST,   # wobble 1
    "CCT": Op.OP_ADD,
    "CCC": Op.OP_SUB,
    "CCA": Op.OP_MUL,
    "CCG": Op.OP_LT,
    "GAA": Op.OP_NOT,
    "TGG": Op.OP_BUILD_PROTEIN,
}

ARITH = "#gene name=g\nATG TCT TCC CCC TCC CCG TCC CCA GAA TAA\n#end"
STOCK = "#gene name=g\nATG GCT TAA\n#end"


def _parse(src: str, table: dict[str, Op]):
    stop = {c for c, op in table.items() if op == Op.OP_HALT}
    return Parser(list(Lexer(src).tokens()), stop_codons=stop).parse()


def _build_ir(src: str, table: dict[str, Op]):
    return IRBuilder(table).build(_parse(src, table))


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        # --- 1. builder: one typed IR function per gene, faithful stream ---
        ir = _build_ir(STOCK, STANDARD_TABLE)
        checks["builder_gene_mapping"] = [f.name for f in ir.functions] == ["g"]
        checks["builder_opcode_faithful"] = [i.opcode for i in ir.functions[0].instrs] == [
            Op.OP_START, Op.OP_BUILD_PROTEIN, Op.OP_HALT]
        details["stock_ops"] = [i.opcode.name for i in ir.functions[0].instrs]

        # --- 2. lowering: byte-identical to the legacy bytecode ABI ---
        golden = [16, 48, 3, 17]  # START / BUILD_PROTEIN 3 / HALT
        prog = _parse(STOCK, STANDARD_TABLE)
        chunk = Compiler(STANDARD_TABLE).compile(prog)
        checks["lowering_byte_golden"] = list(chunk.code) == golden
        ir2, chunk2 = Compiler(STANDARD_TABLE).compile_ir(prog, optimize=False)
        checks["compile_vs_compile_ir"] = chunk == chunk2
        checks["ir_vs_lower_code"] = list(lower(ir2).code) == golden
        details["golden_code"] = golden

        # --- 3. optimizer: semantic parity + idempotence ---
        ir_plain = _build_ir(ARITH, TEST_TABLE)
        ir_opt = _build_ir(ARITH, TEST_TABLE)
        optimize_program(ir_opt)
        p1 = CellVM(lower(ir_plain), _parse(ARITH, TEST_TABLE)).run(40)
        p2 = CellVM(lower(ir_opt), _parse(ARITH, TEST_TABLE)).run(40)
        checks["optimizer_semantic_parity"] = (p1 == p2 and p1[-1]["alive"])
        details["folded_instrs"] = len(ir_opt.functions[0].instrs)
        details["plain_instrs"] = len(ir_plain.functions[0].instrs)
        ir_twice = _build_ir(ARITH, TEST_TABLE)
        optimize_program(ir_twice)
        optimize_program(ir_twice)
        checks["optimizer_idempotent"] = (
            [i.opcode for i in ir_twice.functions[0].instrs]
            == [i.opcode for i in ir_opt.functions[0].instrs])
        # pure-op streams survive intact; effect boundaries are never removed
        opt_effect = _build_ir("#gene name=g\nATG TCT TCC TGG TAA\n#end", TEST_TABLE)
        optimize_program(opt_effect)
        checks["effect_boundary_preserved"] = any(
            i.opcode is Op.OP_BUILD_PROTEIN for i in opt_effect.functions[0].instrs)

        # --- 4. serialization: stable round trip ---
        rt = loads(dumps(ir_opt))
        checks["serialize_roundtrip"] = rt.disassemble() == ir_opt.disassemble()
        checks["serialize_stable"] = dumps(rt) == dumps(ir_opt)

        # --- 5. call_target resolution + config snapshot metadata ---
        src = ("#gene name=caller call_target=target\nATG AAA TAA\n#end\n"
               "#gene name=target\nATG TGG TAA\n#end")
        call_ir = _build_ir(src, TEST_TABLE)
        call = [i for i in call_ir.functions[0].instrs if i.opcode is Op.OP_CALL_GENE]
        checks["call_target_resolved"] = bool(call) and call[0].operand == "target"
        cfg_ir = _build_ir(ARITH, TEST_TABLE)
        check_cfg = cfg_ir.config.get("ticks") == _parse(ARITH, TEST_TABLE).config.ticks
        checks["config_snapshot_preserved"] = check_cfg

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "71_ir_roundtrip",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": "doc/37 §5-6 Helix IR pipeline (ir_builder/ir_lower/ir_opt/ir_serialize)",
            "runtime_seconds": elapsed,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "id": "71_ir_roundtrip",
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
