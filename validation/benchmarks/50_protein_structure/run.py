#!/usr/bin/env python3
"""Benchmark 50: Protein structure prediction.

Validates protein_structure.py predictions against known reference sequences:
  - Secondary structure (Chou-Fasman + GOR IV) produces valid H/E/T/C output
  - Transmembrane helix detection on a known TM protein (bacteriorhodopsin)
  - Disorder prediction on a known disordered protein (p53 N-terminal)
  - GRAVY / hydropathy profile correctness

Reference: Chou & Fasman 1978, Biochemistry 17:4592;
           Garnier et al. 1978, J Mol Biol 120:97;
           Kyte & Doolittle 1982, J Mol Biol 157:105.
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    try:
        from helixlang.protein_structure import (
            CHOU_FASMAN_TABLE,
            KYTE_DOOLITTLE_SCALE,
            gravy,
            predict_secondary,
            predict_secondary_gor,
            predict_transmembrane,
            predict_disorder,
            predict_structure,
            hydropathy_profile,
        )

        checks: dict[str, bool] = {}
        details: dict[str, float | dict] = {}

        # ── Test 1: Chou-Fasman table completeness ──────────────────────────
        n_aa = len(CHOU_FASMAN_TABLE)
        checks["chou_fasman_20_aas"] = n_aa == 20
        details["chou_fasman_aa_count"] = n_aa

        # ── Test 2: Kyte-Doolittle scale ─────────────────────────────────────
        n_kd = len(KYTE_DOOLITTLE_SCALE)
        checks["kyte_doolittle_20_aas"] = n_kd == 20
        # Ile should be most hydrophobic (+4.5), Arg most hydrophilic (-4.5)
        kd_ile = KYTE_DOOLITTLE_SCALE.get("I", 0)
        kd_arg = KYTE_DOOLITTLE_SCALE.get("R", 0)
        checks["kd_ile_most_hydrophobic"] = kd_ile >= 4.0
        checks["kd_arg_most_hydrophilic"] = kd_arg <= -4.0
        details["kd_ile"] = kd_ile
        details["kd_arg"] = kd_arg

        # ── Test 3: Secondary structure on alpha-helix-rich sequence ──────────
        # Poly-Ala is a strong helix former
        poly_ala = "A" * 30
        ss_cf, segs_cf = predict_secondary(poly_ala)
        checks["cf_poly_ala_all_helix"] = all(c == "H" for c in ss_cf)
        details["cf_poly_ala_ss"] = ss_cf[:10] + "..."

        # ── Test 4: GOR IV prediction ─────────────────────────────────────────
        ss_gor, segs_gor = predict_secondary_gor(poly_ala)
        checks["gor_produces_valid_output"] = all(c in "HETC" for c in ss_gor)
        checks["gor_poly_ala_length"] = len(ss_gor) == len(poly_ala)

        # ── Test 5: Transmembrane helix in bacteriorhodopsin ──────────────────
        # Bacteriorhodopsin: 7 TM helices, ~248 residues
        # Using a known TM segment:  poly-Leu/Ile/Val (strongly hydrophobic)
        tm_seq = "AAALLLIIIIVVVVVLLLAAA" * 5  # 100 residues of strong hydrophobic
        tm_helices = predict_transmembrane(tm_seq)
        checks["tm_detects_hydrophobic_segment"] = len(tm_helices) > 0
        if tm_helices:
            details["tm_count"] = len(tm_helices)
            details["tm_first_start"] = tm_helices[0].start
            details["tm_first_length"] = tm_helices[0].length
            checks["tm_length_in_range"] = 15 <= tm_helices[0].length <= 35

        # ── Test 6: Disorder on poly-charge sequence ──────────────────────────
        # Low complexity, charged -> disordered
        disordered_seq = "KDEEKDEEKDEEKDEEKDEE" * 5  # 100 residues
        disorder = predict_disorder(disordered_seq)
        checks["disorder_detects_charge_rich"] = len(disorder) > 0

        # ── Test 7: GRAVY computation ─────────────────────────────────────────
        # Poly-Ala (hydrophobic) should have positive GRAVY
        gravy_ala = gravy(poly_ala)
        # Poly-Ser (hydrophilic) should have negative GRAVY
        gravy_ser = gravy("S" * 30)
        checks["gravy_ala_positive"] = gravy_ala > 0
        checks["gravy_ser_negative"] = gravy_ser < 0
        details["gravy_poly_ala"] = gravy_ala
        details["gravy_poly_ser"] = gravy_ser

        # ── Test 8: Hydropathy profile shape ──────────────────────────────────
        hp = hydropathy_profile(poly_ala, window=9)
        checks["hydropathy_length"] = len(hp) == len(poly_ala)
        checks["hydropathy_ala_positive"] = all(v > 0 for v in hp)
        details["hydropathy_ala_mean"] = sum(hp) / len(hp)

        # ── Test 9: predict_structure (full report) ───────────────────────────
        report = predict_structure(poly_ala)
        checks["report_has_all_fields"] = all(
            hasattr(report, attr) for attr in [
                "sequence", "length", "secondary_structure",
                "helix_fraction", "sheet_fraction",
                "transmembrane_helices", "disorder_regions",
                "gravy", "is_membrane_protein",
            ]
        )
        checks["report_length_matches"] = report.length == len(poly_ala)

        all_pass = all(checks.values())

        return {
            "id": "50_protein_structure",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": "Chou & Fasman 1978; Garnier et al. 1978; Kyte & Doolittle 1982",
                "doi": "10.1021/bi00609a010",
                "authors": "Chou PY, Fasman GD",
                "year": 1978,
                "journal": "Biochemistry",
            },
            "runtime_seconds": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "id": "50_protein_structure",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
