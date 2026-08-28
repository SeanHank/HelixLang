#!/usr/bin/env python3
"""Benchmark 42: Remaining Modules — bio_data, morphology_3d, lsystem, protein_structure, protein_fitness, units, seq_utils."""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.plugins.runtime.bio_data import ECOLI_CODON_USAGE
        checks["import_bio_data"] = True
        assert len(ECOLI_CODON_USAGE) > 0, "Codon usage table should not be empty"
        details["codon_usage_entries"] = len(ECOLI_CODON_USAGE)
        checks["bio_data_codon_usage_loaded"] = True

        from helixlang.plugins.runtime.morphology_3d import LSystem3D
        checks["import_morphology_3d"] = True
        preset = LSystem3D(
            axiom="F",
            rules={"F": "F[\\F][/F]F"},
            angle=25.0,
            step=1.0,
        )
        lines = preset.draw(iterations=2)
        details["morphology_3d_lines"] = len(lines)
        assert len(lines) > 0, "3D L-system should produce lines"
        checks["morphology_3d_mesh_created"] = True

        from helixlang.plugins.runtime.lsystem import LSystem
        checks["import_lsystem"] = True
        ls = LSystem(
            axiom="F",
            rules={"F": "F+F-F-F+F"},
            angle=90.0,
            step=1.0,
        )
        pts = ls.iterate()
        details["lsystem_points_after_1_iter"] = len(pts)
        assert len(pts) > 0, "L-system should produce points after iteration"
        checks["lsystem_iteration_runs"] = True

        from helixlang.plugins.runtime.protein_structure import predict_secondary
        checks["import_protein_structure"] = True
        seq = "MASKGEELFTGVPVPILVELDGDVNGHK"
        ss_string, segments = predict_secondary(seq)
        details["ss_length"] = len(ss_string)
        details["ss_segments"] = len(segments)
        assert len(ss_string) == len(seq), (
            f"SS length {len(ss_string)} != seq length {len(seq)}"
        )
        checks["protein_structure_prediction_runs"] = True

        from helixlang.plugins.runtime.protein_fitness import BLOSUMOracle, blosum62_normalized
        checks["import_protein_fitness"] = True
        ref = "ACDEF"
        var = "ACDEF"
        identity_score = blosum62_normalized(ref, var)
        details["identity_score"] = identity_score
        assert abs(identity_score - 1.0) < 1e-9, (
            f"Identity score should be 1.0, got {identity_score}"
        )
        oracle = BLOSUMOracle()
        assert oracle.available, "BLOSUMOracle should be available"
        oracle_score = oracle.score("ACDEF", "ACDEG")
        details["oracle_score"] = oracle_score
        assert 0.0 <= oracle_score <= 1.0, (
            f"Oracle score out of range: {oracle_score}"
        )
        checks["protein_fitness_scoring_works"] = True

        from helixlang.core.units import TIME_TICK_MIN, ticks_to_min
        checks["import_units"] = True
        assert TIME_TICK_MIN == 1.0, f"TIME_TICK_MIN should be 1.0, got {TIME_TICK_MIN}"
        assert ticks_to_min(5.0) == 5.0, "ticks_to_min should be identity"
        checks["units_conversion_correct"] = True

        from helixlang.plugins.runtime.seq_utils import gc_content, reverse_complement
        checks["import_seq_utils"] = True
        assert gc_content("ATGC") == 0.5, "GC(ATGC) should be 0.5"
        assert gc_content("GGCC") == 1.0, "GC(GGCC) should be 1.0"
        assert gc_content("") == 0.0, "GC('') should be 0.0"
        rc = reverse_complement("ATGC")
        assert rc == "GCAT", f"RC(ATGC) should be GCAT, got {rc}"
        details["gc_atgc"] = gc_content("ATGC")
        details["rc_atgc"] = rc
        checks["seq_utils_gc_content_works"] = True

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "42_remaining_modules",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "42_remaining_modules",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
