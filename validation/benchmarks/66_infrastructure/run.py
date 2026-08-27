#!/usr/bin/env python3
"""Benchmark 66: Infrastructure — units, seq_utils, bio_data, biocodec modules."""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        # ── helixlang.units ──────────────────────────────────────────
        from helixlang.units import (
            ATP_PER_GLUCOSE,
            LATTICE_SPACING_UM,
            TIME_TICK_MIN,
            UNITS_CELL_VOLUME_NEWBORN_UM3,
            ticks_to_min,
        )

        checks["units_import"] = True

        checks["units_TIME_TICK_MIN"] = TIME_TICK_MIN == 1.0
        details["TIME_TICK_MIN"] = TIME_TICK_MIN

        checks["units_LATTICE_SPACING_UM"] = LATTICE_SPACING_UM == 10.0
        details["LATTICE_SPACING_UM"] = LATTICE_SPACING_UM

        checks["units_ATP_PER_GLUCOSE"] = ATP_PER_GLUCOSE == 38
        details["ATP_PER_GLUCOSE"] = ATP_PER_GLUCOSE

        checks["units_ticks_to_min"] = ticks_to_min(5.0) == 5.0
        details["ticks_to_min(5.0)"] = ticks_to_min(5.0)

        checks["units_NEWBORN_VOLUME"] = UNITS_CELL_VOLUME_NEWBORN_UM3 > 0
        details["UNITS_CELL_VOLUME_NEWBORN_UM3"] = UNITS_CELL_VOLUME_NEWBORN_UM3

        # ── helixlang.seq_utils ──────────────────────────────────────
        from helixlang.seq_utils import (
            gc_content,
            max_homopolymer,
            reverse_complement,
        )

        checks["seq_utils_import"] = True

        checks["gc_ATGC"] = gc_content("ATGC") == 0.5
        details["gc_ATGC"] = gc_content("ATGC")

        checks["gc_GGCC"] = gc_content("GGCC") == 1.0
        details["gc_GGCC"] = gc_content("GGCC")

        checks["gc_empty"] = gc_content("") == 0.0
        details["gc_empty"] = gc_content("")

        checks["rc_ATGC"] = reverse_complement("ATGC") == "GCAT"
        details["rc_ATGC"] = reverse_complement("ATGC")

        checks["rc_AAAA"] = reverse_complement("AAAA") == "TTTT"
        details["rc_AAAA"] = reverse_complement("AAAA")

        checks["homopolymer_AAATTT"] = max_homopolymer("AAATTT") == 3
        details["max_homopolymer_AAATTT"] = max_homopolymer("AAATTT")

        checks["homopolymer_AAAA"] = max_homopolymer("AAAA") == 4
        details["max_homopolymer_AAAA"] = max_homopolymer("AAAA")

        # ── helixlang.bio_data ───────────────────────────────────────
        from helixlang.bio_data import ECOLI_CODON_USAGE

        checks["bio_data_import"] = True

        codon_count = len(ECOLI_CODON_USAGE)
        checks["codon_table_64"] = codon_count == 64
        details["codon_table_count"] = codon_count

        atg_entry = ECOLI_CODON_USAGE.get("ATG")
        checks["ATG_present"] = atg_entry is not None
        if atg_entry is not None:
            aa, per_thou, frac = atg_entry
            checks["ATG_maps_to_M"] = aa == "M"
            checks["ATG_entry_is_tuple3"] = (
                isinstance(atg_entry, tuple) and len(atg_entry) == 3
            )
            details["ATG_entry"] = list(atg_entry)
        else:
            checks["ATG_maps_to_M"] = False
            checks["ATG_entry_is_tuple3"] = False

        all_entries_valid = True
        bad_entries = []
        for codon, entry in ECOLI_CODON_USAGE.items():
            if not isinstance(entry, tuple) or len(entry) != 3:
                all_entries_valid = False
                bad_entries.append(codon)
        checks["all_entries_tuple3"] = all_entries_valid
        details["bad_entry_codons"] = bad_entries[:5]

        # ── helixlang.biocodec ───────────────────────────────────────
        import helixlang.biocodec as biocodec

        checks["biocodec_import"] = True

        lac_prom = biocodec.LAC_PROMOTER
        checks["LAC_PROMOTER_string"] = isinstance(lac_prom, str) and len(lac_prom) > 50
        details["LAC_PROMOTER_len"] = len(lac_prom)

        t7_prom = biocodec.T7_PROMOTER
        checks["T7_PROMOTER_string"] = isinstance(t7_prom, str) and len(t7_prom) > 10
        details["T7_PROMOTER_len"] = len(t7_prom)

        sites = biocodec.RESTRICTION_SITES
        checks["RESTRICTION_SITES_dict"] = isinstance(sites, dict) and len(sites) >= 5
        details["RESTRICTION_SITES_count"] = len(sites)

        # round-trip: back_translate (protein -> DNA) then _translate_fallback (DNA -> protein)
        test_protein = "MKFLIV"
        test_dna = biocodec.back_translate(test_protein, optimize="cai")
        rt_protein = biocodec._translate_fallback(test_dna)
        checks["biocodec_roundtrip"] = rt_protein == test_protein
        details["roundtrip_input"] = test_protein
        details["roundtrip_dna"] = test_dna
        details["roundtrip_output"] = rt_protein

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "66_infrastructure",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": (
                "Kanehisa M 2000, Nucleic Acids Res 28:27; "
                "NCBI Standard Codon Table; "
                "Neidhardt 1996 (units); "
                "Sharp & Li 1987 Nucleic Acids Res 15:1281-1295 (CAI)"
            ),
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "66_infrastructure",
            "status": "FAIL",
            "checks": checks,
            "details": details,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
