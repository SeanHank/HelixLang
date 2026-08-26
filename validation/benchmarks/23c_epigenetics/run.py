#!/usr/bin/env python3
"""Benchmark 23c: Epigenetics — CpG and Dam site detection."""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "23c_epigenetics"}
    try:
        from helixlang.epigenetics import (
            find_cpg_sites,
            find_dam_sites,
        )

        # ── CpG sites in a known sequence ────────────────────────────
        # ACGTACGT repeated 25 times = 200 nt
        # Each "ACGTACGT" has CG at positions 1 and 5 → 2 CGs per repeat
        seq_part1 = "ACGTACGT" * 25  # 200 nt
        cpg_part1 = find_cpg_sites(seq_part1)
        expected_part1 = seq_part1.count("CG")
        part1_ok = len(cpg_part1) == expected_part1

        # Alternating CG: "CG" * 18 = 36 nt, 18 CGs
        seq_part2 = "CG" * 18
        cpg_part2 = find_cpg_sites(seq_part2)
        expected_part2 = seq_part2.count("CG")
        part2_ok = len(cpg_part2) == expected_part2

        # No-CG sequence
        seq_no_cpg = "ATATATATAT" * 20
        cpg_none = find_cpg_sites(seq_no_cpg)
        no_cpg_ok = len(cpg_none) == 0

        # ── Dam sites (GATC) ─────────────────────────────────────────
        dam_seq = "AGATCGATCGATCGATC"
        dam_sites = find_dam_sites(dam_seq)
        dam_ok = len(dam_sites) > 0

        all_ok = part1_ok and part2_ok and no_cpg_ok and dam_ok

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS" if all_ok else "FAIL",
            "validation": {
                "cpg_acgt_repeat": part1_ok,
                "cpg_alternating": part2_ok,
                "no_cpg_in_at": no_cpg_ok,
                "dam_sites_found": dam_ok,
            },
            "cpg": {
                "acgt_repeat_count": len(cpg_part1),
                "acgt_repeat_expected": expected_part1,
                "alternating_count": len(cpg_part2),
                "alternating_expected": expected_part2,
                "at_only_count": len(cpg_none),
            },
            "dam": {
                "sequence": dam_seq,
                "sites_found": len(dam_sites),
                "positions": dam_sites,
            },
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
