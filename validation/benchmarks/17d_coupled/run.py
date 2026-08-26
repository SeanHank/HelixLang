#!/usr/bin/env python3
"""Benchmark 17d: Coupled transcription-translation + protein pool."""
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


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "17d_coupled"}
    try:
        from helixlang.central_dogma import (
            ProteinPool,
            advance_protein_pool,
            coupled_transcription_translation,
        )

        # ── Coupled model ────────────────────────────────────────────
        coupled = coupled_transcription_translation(_DNA_SEQ)

        assert "transcript" in coupled
        assert "protein" in coupled
        assert "mrna_level" in coupled
        assert "time_course" in coupled
        assert coupled["protein"] == "MAKSNVKASTWKSIAAMSSRMTVK"
        assert coupled["transcription_time_s"] > 0
        assert coupled["translation_time_s"] > 0
        assert coupled["coupling_offset_s"] > 0
        assert coupled["mrna_steady_state"] > 0
        assert len(coupled["time_course"]) > 0

        # ── Protein pool ─────────────────────────────────────────────
        pool = ProteinPool(unfolded=100.0)
        deltas = advance_protein_pool(pool, dt=1.0)

        assert pool.folded > 0
        assert deltas["folded"] > 0
        assert deltas["atp_cost"] > 0
        assert deltas["unfolded_consumed"] > 0

        # Multiple steps advance the pool further
        for _ in range(5):
            advance_protein_pool(pool, dt=1.0)
        assert pool.folded > 50

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "validation": {
                "coupled_model_works": True,
                "protein_pool_works": True,
            },
            "coupled": {
                "protein_length": len(coupled["protein"]),
                "mrna_level": round(coupled["mrna_level"], 2),
                "mrna_steady_state": round(coupled["mrna_steady_state"], 2),
                "transcription_time_s": round(coupled["transcription_time_s"], 4),
                "translation_time_s": round(coupled["translation_time_s"], 4),
                "coupling_offset_s": round(coupled["coupling_offset_s"], 4),
                "time_course_points": len(coupled["time_course"]),
            },
            "protein_pool": {
                "folded_after_6_steps": round(pool.folded, 2),
                "atp_cost_per_step": round(deltas["atp_cost"], 2),
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
