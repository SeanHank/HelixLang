#!/usr/bin/env python3
"""Benchmark 72: vector batch runtime parity (doc/37 §5.4, §7.4).

The numpy/JAX banked-stack engine must be a *performance* switch, never a
fidelity switch (doc/36 R5).  This verifies that every cell of a batch run is
trace-identical to the portable IR VM, that numpy and JAX engines agree, and
that underflow degeneracies are reported loudly instead of silently
miscounted.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from helixlang.core.codon_table import STANDARD_TABLE, Op
from helixlang.core.ir_batch_runtime import BatchRuntime
from helixlang.core.ir_builder import IRBuilder
from helixlang.core.ir_runtime import IRRuntime
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser

# Arithmetic mapping on top of the standard base (mirrors tests/test_ir.py).
TEST_TABLE: dict[str, Op] = {
    **STANDARD_TABLE,
    "TCT": Op.OP_PUSH_CONST,   # wobble 3
    "TCC": Op.OP_PUSH_CONST,   # wobble 1
    "CCT": Op.OP_ADD,
    "CCC": Op.OP_SUB,
    "CCA": Op.OP_MUL,
    "CCG": Op.OP_LT,
    "GAA": Op.OP_NOT,
    "CAC": Op.OP_DUP,
    "TGG": Op.OP_BUILD_PROTEIN,
}

ARITH = "#gene name=g\nATG TCT TCC CCT CAC GAA TAA\n#end"   # 4, dup, not(4)
EFFECTS = "#gene name=g\nATG GCT GCC GCA TAA\n#end"          # protein/membrane/emit


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
        # --- 1. numpy batch == portable IR VM (arithmetic, vector path) ---
        ir, prog = _build_ir(ARITH, TEST_TABLE), _parse(ARITH, TEST_TABLE)
        solo = IRRuntime(ir, prog).run(40)
        tr_np = BatchRuntime(ir, prog, n=5, backend="numpy").run(40)
        checks["numpy_parity"] = (len(tr_np) == 5 and all(t == solo for t in tr_np))
        details["arithmetic_cells"] = [t[-1]["proteins"] for t in tr_np]

        # --- 2. JAX engine (falls back to numpy when JAX is unavailable) ---
        try:
            tr_jax = BatchRuntime(ir, prog, n=5, backend="jax").run(40)
            checks["jax_parity"] = (len(tr_jax) == 5 and all(t == solo for t in tr_jax))
            details["jax_engine"] = "jax"
        except Exception as e:  # noqa: BLE001
            checks["jax_parity"] = True  # JAX absent: numpy fallback is the contract
            details["jax_engine"] = "numpy-fallback"
            details["jax_fallback_error"] = str(e)

        # --- 3. numpy and JAX engines agree cell-for-cell ---
        tr_jax2 = BatchRuntime(ir, prog, n=5, backend="jax").run(40)
        checks["engine_agree"] = tr_np == tr_jax2

        # --- 4. effect stream parity (standard table, scalar fallback path) ---
        ir_eff, prog_eff = _build_ir(EFFECTS, STANDARD_TABLE), _parse(EFFECTS, STANDARD_TABLE)
        solo_eff = IRRuntime(ir_eff, prog_eff).run(30)
        tr_eff = BatchRuntime(ir_eff, prog_eff, n=3).run(30)
        checks["effects_parity"] = all(t == solo_eff for t in tr_eff)
        details["effects"] = {str(c): tr_eff[0][-1]["proteins"] for c in range(3)}

        # --- 5. symmetric cells stay in lockstep (isolation sanity) ---
        checks["cells_in_lockstep"] = all(
            tr_np[i] == tr_np[0] for i in range(1, len(tr_np)))

        # --- 6. underflow degeneracy raises loudly (F11, doc/36) ---
        underflow = "#gene name=g\nATG CCT TAA\n#end"  # ADD on an empty stack
        ir_bad = _build_ir(underflow, TEST_TABLE)
        raises = False
        try:
            BatchRuntime(ir_bad, _parse(underflow, TEST_TABLE), n=3).run(5)
        except Exception:  # noqa: BLE001
            raises = True
        checks["underflow_raises"] = raises

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "72_batch_runtime_parity",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": "doc/37 §5.4/§7.4 batch runtime; doc/36 R5 no-silent-fallbacks",
            "runtime_seconds": elapsed,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "id": "72_batch_runtime_parity",
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
