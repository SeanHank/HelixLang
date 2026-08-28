#!/usr/bin/env python3
"""Benchmark 15: DNA encoding — dna_codec + biocodec + codon_table roundtrips."""
from __future__ import annotations

import json
import sys
import time

TEST_DATA = b"HelloHelixLang"


def _test_goldman_roundtrip() -> tuple[bool, dict]:
    from helixlang.plugins.runtime.dna_codec import goldman_decode, goldman_encode
    oligos = goldman_encode(TEST_DATA)
    decoded = goldman_decode(oligos, total_len=len(TEST_DATA))
    ok = decoded == TEST_DATA
    return ok, {
        "oligo_count": len(oligos),
        "decoded_length": len(decoded),
        "roundtrip_match": ok,
    }


def _test_2bit_dna_roundtrip() -> tuple[bool, dict]:
    from helixlang.plugins.runtime.dna_codec import _bytes_to_dna_2bit, _dna_to_bytes_2bit
    dna = _bytes_to_dna_2bit(TEST_DATA)
    decoded = _dna_to_bytes_2bit(dna)
    ok = decoded == TEST_DATA
    return ok, {
        "dna_length": len(dna),
        "decoded_length": len(decoded),
        "roundtrip_match": ok,
    }


def _test_codon_table() -> tuple[bool, dict]:
    from helixlang.core.codon_table import STANDARD_TABLE, Op
    all_codons = [f"{a}{c}{g}" for a in "ACGT" for c in "ACGT" for g in "ACGT"]
    assert len(all_codons) == 64
    mapped = sum(1 for c in all_codons if c in STANDARD_TABLE)
    assert mapped == 64, f"only {mapped}/64 codons mapped"
    expected_op = {
        "ATG": Op.OP_START,
        "TAA": Op.OP_HALT,
        "TAG": Op.OP_HALT,
        "TGA": Op.OP_HALT,
    }
    for codon, exp in expected_op.items():
        assert STANDARD_TABLE[codon] == exp, f"{codon}: expected {exp}, got {STANDARD_TABLE[codon]}"
    return True, {"codon_count": 64, "all_mapped": True}


def _test_gc_content() -> tuple[bool, dict]:
    from helixlang.plugins.runtime.seq_utils import gc_content
    gc_at = gc_content("ATATATATAT")
    assert abs(gc_at - 0.0) < 1e-9, f"GC of ATATAT should be 0.0, got {gc_at}"
    gc_gc = gc_content("GCGCGCGCGC")
    assert abs(gc_gc - 1.0) < 1e-9, f"GC of GCGCGC should be 1.0, got {gc_gc}"
    gc_mixed = gc_content("AACCGGTT")
    assert abs(gc_mixed - 0.5) < 1e-9, f"GC of AACCGGTT should be 0.5, got {gc_mixed}"
    return True, {
        "gc_at_only": round(gc_at, 4),
        "gc_gc_only": round(gc_gc, 4),
        "gc_mixed": round(gc_mixed, 4),
    }


def _test_biocodec() -> tuple[bool, dict]:
    from helixlang.plugins.runtime.biocodec import back_translate, find_orfs
    protein = "MKYATS"
    dna = back_translate(protein, optimize="cai")
    assert dna[:3] == "ATG", f"back_translate should start with ATG, got {dna[:3]}"
    assert len(dna) % 3 == 0, f"DNA length {len(dna)} not multiple of 3"
    dna_with_stop = dna + "TAA"
    orfs = find_orfs(dna_with_stop, min_length_aa=1, both_strands=False)
    assert len(orfs) >= 1, f"find_orfs found no ORFs in {dna_with_stop}"
    return True, {"back_translated_length": len(dna), "orfs_found": len(orfs)}


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "15_dna_encoding"}
    try:
        checks = {}

        ok_g, info_g = _test_goldman_roundtrip()
        checks["goldman"] = info_g
        assert ok_g, f"Goldman roundtrip failed: {info_g}"

        ok_2, info_2 = _test_2bit_dna_roundtrip()
        checks["2bit_dna"] = info_2
        assert ok_2, f"2-bit DNA roundtrip failed: {info_2}"

        ok_c, info_c = _test_codon_table()
        checks["codon_table"] = info_c
        assert ok_c, f"Codon table check failed: {info_c}"

        ok_gc, info_gc = _test_gc_content()
        checks["gc_content"] = info_gc
        assert ok_gc, f"GC content check failed: {info_gc}"

        ok_b, info_b = _test_biocodec()
        checks["biocodec"] = info_b
        assert ok_b, f"Biocodec check failed: {info_b}"

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "checks": checks,
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
