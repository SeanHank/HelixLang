#!/usr/bin/env python3
"""Benchmark 17b: DNA transcription — transcribe a known ORF."""
from __future__ import annotations

import json
import sys
import time

# CDS: 24 sense codons + TAA stop = 25 codons = 75 nt (multiple of 3)
_CDS_PARTS = [
    "ATG", "GCT", "AAA", "TCT", "AAC", "GTT", "AAA", "GCG", "TCT",
    "ACC", "TGG", "AAA", "TCT", "ATC", "GCG", "GCA", "ATG", "TCT",
    "TCC", "CGG", "ATG", "ACG", "GTA", "AAA", "TAA",
]
_DNA_SEQ = "TTTTAAGAGGAGG" + "".join(_CDS_PARTS) + "GGGAAACCC"


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "17b_transcribe"}
    try:
        from helixlang.central_dogma import Transcript, transcribe

        transcript = transcribe(_DNA_SEQ)

        # Validate Transcript structure
        assert isinstance(transcript, Transcript)
        assert len(transcript.cds) == 75, f"CDS length {len(transcript.cds)} != 75"
        assert transcript.cds.startswith("AUG"), "CDS should start with AUG"
        assert transcript.half_life_minutes > 0
        assert transcript.poly_a_tail == "A" * 15
        assert transcript.elongation_time_s > 0
        assert transcript.initiation_frequency_per_min > 0

        # Poly-A tail
        has_poly_a = len(transcript.poly_a_tail) == 15

        # Verify UTR regions exist
        has_utr5 = len(transcript.utr5) > 0
        has_utr3 = len(transcript.utr3) > 0

        # All checks passed
        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "validation": {
                "cds_length_75": len(transcript.cds) == 75,
                "cds_starts_with_aug": transcript.cds.startswith("AUG"),
                "half_life_positive": transcript.half_life_minutes > 0,
                "poly_a_tail_15": has_poly_a,
                "elongation_time_positive": transcript.elongation_time_s > 0,
                "utr5_present": has_utr5,
                "utr3_present": has_utr3,
            },
            "transcript": {
                "cds_length": len(transcript.cds),
                "half_life_min": round(transcript.half_life_minutes, 2),
                "elongation_time_s": round(transcript.elongation_time_s, 4),
                "initiation_freq": round(transcript.initiation_frequency_per_min, 4),
                "utr5_length": len(transcript.utr5),
                "utr3_length": len(transcript.utr3),
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
