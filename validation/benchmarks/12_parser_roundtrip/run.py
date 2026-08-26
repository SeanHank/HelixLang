#!/usr/bin/env python3
"""Benchmark 12: Parser roundtrip — source → AST → bytecode."""
from __future__ import annotations

import json
import sys
import time

SOURCE = """\
#gene name=lacI
ATG AAA TAT ACC GCT TCA CCG GAT AAA ACG TAA
#end
#config ticks=10
"""


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "12_parser_roundtrip"}
    try:
        from helixlang.codon_table import STANDARD_TABLE
        from helixlang.compiler import Compiler
        from helixlang.lexer import Lexer
        from helixlang.parser import Parser

        tokens = list(Lexer(SOURCE).tokens())
        assert len(tokens) > 0, "lexer produced no tokens"

        parser = Parser(tokens)
        program = parser.parse()

        assert len(program.genes) >= 1, "parser produced no genes"
        gene = program.genes[0]
        assert gene.name == "lacI", f"expected gene name 'lacI', got {gene.name!r}"
        assert len(gene.codons) == 11, f"expected 11 codons, got {len(gene.codons)}"
        assert gene.orf[0].seq == "ATG", f"ORF must start with ATG, got {gene.orf[0].seq}"
        assert gene.orf[-1].seq in ("TAA", "TAG", "TGA"), (
            f"ORF must end with stop codon, got {gene.orf[-1].seq}"
        )
        assert program.config.ticks == 10, f"expected ticks=10, got {program.config.ticks}"

        compiler = Compiler(STANDARD_TABLE)
        chunk = compiler.compile(program)

        assert len(chunk.code) > 0, "compiler produced empty bytecode"
        assert gene.name in chunk.gene_offsets, (
            f"gene {gene.name!r} missing from chunk gene_offsets"
        )

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "tokens": len(tokens),
            "genes": len(program.genes),
            "gene_name": gene.name,
            "codon_count": len(gene.codons),
            "orf_length": len(gene.orf),
            "bytecode_size": len(chunk.code),
            "constants_count": len(chunk.constants),
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
