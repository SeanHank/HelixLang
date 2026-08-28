#!/usr/bin/env python3
"""Benchmark 17c: mRNA translation — translate a known transcript."""
from __future__ import annotations

import json
import sys
import time

_CDS_PARTS = [
    "ATG", "GCT", "AAA", "TCT", "AAC", "GTT", "AAA", "GCG", "TCT",
    "ACC", "TGG", "AAA", "TCT", "ATC", "GCG", "GCA", "ATG", "TCT",
    "TCC", "CGG", "ATG", "ACG", "GTA", "AAA", "TAA",
]
_DNA_SEQ = "TTTTAAGAGGAGG" + "".join(_CDS_PARTS) + "GGGAAACCC"
_EXPECTED_PROTEIN = "MAKSNVKASTWKSIAAMSSRMTVK"


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "17c_translate"}
    try:
        from helixlang.plugins.runtime.central_dogma import TranslationResult, transcribe, translate

        transcript = transcribe(_DNA_SEQ)
        tx_result = translate(transcript)

        assert isinstance(tx_result, TranslationResult)
        assert tx_result.protein == _EXPECTED_PROTEIN
        assert len(tx_result.protein) == 24
        assert tx_result.stop_codon == "TAA"
        assert tx_result.elongation_time > 0
        assert tx_result.rbs_found is True
        assert len(tx_result.codon_rates) == 24

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "validation": {
                "protein_length_24": len(tx_result.protein) == 24,
                "protein_sequence_match": tx_result.protein == _EXPECTED_PROTEIN,
                "stop_codon_taa": tx_result.stop_codon == "TAA",
                "elongation_time_positive": tx_result.elongation_time > 0,
                "rbs_found": tx_result.rbs_found,
                "codon_rates_length": len(tx_result.codon_rates) == 24,
            },
            "translation": {
                "protein": tx_result.protein,
                "protein_length": len(tx_result.protein),
                "elongation_time_s": round(tx_result.elongation_time, 4),
                "stop_codon": tx_result.stop_codon,
                "stop_efficiency": tx_result.stop_efficiency,
                "rbs_sequence": tx_result.rbs_sequence,
                "mean_codon_rate": round(
                    sum(tx_result.codon_rates) / len(tx_result.codon_rates), 2
                ),
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
