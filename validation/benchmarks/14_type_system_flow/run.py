#!/usr/bin/env python3
"""Benchmark 14: Type system & flow — type checking + module imports."""
from __future__ import annotations

import json
import sys
import time

VALID_SOURCE = """\
#gene lacI
ATG AAA TAT ACC GCT TCA CCG GAT AAA ACG TAA
#end
"""

INVALID_SOURCE = """\
#gene lacI
ATG AAA TAT ACC GCT TCA CCG GAT AAA ACG TAA
#end
#regulate lacI -> nonExistent strength=0.9
"""


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "14_type_system_flow"}
    try:
        from helixlang.flow import FlowField
        from helixlang.lexer import Lexer
        from helixlang.parser import Parser
        from helixlang.semantic import SemanticAnalyzer
        from helixlang.type_system import (
            HelixType,
            Module,
            SymbolTable,
            TypeChecker,
            parse_type_annotation,
        )

        st = SymbolTable()
        st.define("lacI", HelixType.GENE)
        assert st.get_type("lacI") == HelixType.GENE
        st.define("prom", HelixType.PROTEIN)
        assert st.get_type("prom") == HelixType.PROTEIN

        for name in ("protein", "signal", "float", "int", "bool", "string", "gene", "any"):
            ht = parse_type_annotation(name)
            assert ht.value == name, f"parse_type_annotation({name!r}) = {ht}"

        mod = Module(name="test")
        mod.symbols.define("gene_a", HelixType.GENE)
        mod.symbols.define("gene_b", HelixType.GENE)
        mod.exports.add("gene_a")
        mod2 = Module(name="consumer")
        mod2.import_module("test", mod)
        assert mod2.symbols.lookup("gene_a") is not None
        assert mod2.symbols.lookup("gene_b") is None

        valid_tokens = list(Lexer(VALID_SOURCE).tokens())
        valid_prog = Parser(valid_tokens).parse()
        SemanticAnalyzer(valid_prog).check()
        tc = TypeChecker()
        errors = tc.check(valid_prog, SymbolTable())
        assert len(errors) == 0, f"valid program has type errors: {errors}"

        invalid_tokens = list(Lexer(INVALID_SOURCE).tokens())
        try:
            Parser(invalid_tokens).parse()
        except Exception:
            pass
        zeros = [[0.0] * 10 for _ in range(10)]
        ff = FlowField(width=10, height=10, u=zeros, v=[row[:] for row in zeros])
        assert ff is not None, "FlowField construction failed"
        assert ff.max_magnitude() == 0.0, "FlowField max_magnitude should be 0"

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "helix_types": len(HelixType),
            "symbol_table_define_lookup": True,
            "type_annotation_parsing": True,
            "module_import_export": True,
            "valid_program_type_check": True,
            "flow_field_construction": True,
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
