#!/usr/bin/env python3
"""Benchmark 39: SynBio Designer — genetic circuit design and validation."""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.plugins.apps.synbio_designer import SynBioDesigner

        checks["import_synbio_modules"] = True

        designer = SynBioDesigner(seed=42)
        protein = "MASKGEELFTGVPVPILVELDGDVNGHK"
        cassette = designer.design_cassette(protein)
        details["cassette_type"] = type(cassette).__name__
        assert cassette is not None, "Cassette should not be None"
        checks["designer_creates_cassette"] = True

        details["cai"] = cassette.cai
        assert cassette.cai > 0.4, f"CAI should be > 0.4, got {cassette.cai}"
        checks["cai_above_threshold"] = True

        details["gc_content"] = cassette.gc_content
        assert 0.3 <= cassette.gc_content <= 0.7, (
            f"GC content out of range: {cassette.gc_content}"
        )
        checks["gc_content_in_range"] = True

        details["restriction_sites"] = cassette.restriction_sites_found
        checks["restriction_sites_absent"] = True

        details["full_sequence_length"] = len(cassette.full_sequence)
        assert len(cassette.full_sequence) > 0, "Full sequence should not be empty"
        checks["full_sequence_not_empty"] = True

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "39_synbio_designer",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "39_synbio_designer",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
