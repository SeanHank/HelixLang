#!/usr/bin/env python3
"""Benchmark 40: DNA Storage — encode/decode roundtrip."""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.apps.dna_storage import DNAStorage

        checks["import_dna_storage_module"] = True

        storage = DNAStorage(scheme="goldman")
        data = b"HELLO"
        report = storage.store(data)
        details["scheme"] = report.scheme
        details["num_oligos"] = report.num_oligos
        details["total_bp"] = report.total_bp
        details["density"] = report.density_bit_per_nt
        assert report.num_oligos > 0, "Should produce at least one oligo"
        checks["encode_produces_oligos"] = True

        valid_chars = set("ACGT")
        all_valid = True
        for oligo in report.oligos:
            seq = oligo.full.upper()
            if not all(c in valid_chars for c in seq):
                all_valid = False
                break
        assert all_valid, "DNA output must contain only ATCG characters"
        checks["dna_output_valid_characters"] = True

        recovered = storage.retrieve(report.oligos, total_len=len(data))
        assert recovered == data, (
            f"Roundtrip failed: expected {data!r}, got {recovered!r}"
        )
        details["original"] = data.decode()
        details["recovered"] = recovered.decode()
        checks["roundtrip_identical_output"] = True

        storage_e = DNAStorage(scheme="erlich")
        report_e = storage_e.store(data, redundancy=0.15)
        recovered_e = storage_e.retrieve(report_e.oligos, total_len=len(data))
        assert recovered_e == data, (
            f"Erlich roundtrip failed: expected {data!r}, got {recovered_e!r}"
        )
        details["erlich_roundtrip_ok"] = True
        checks["decode_returns_original"] = True

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "40_dna_storage_codec",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "40_dna_storage_codec",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
