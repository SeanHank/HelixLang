#!/usr/bin/env python3
"""Benchmark 55: Annotation tools — KEGG mapping, sequences, TF detection, transporters."""
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
        from helixlang.plugins.annotation.kegg_mapping import KOReactionDB, build_ko_db
        from helixlang.plugins.annotation.sequences import reverse_complement, translate
        from helixlang.plugins.annotation.tf_detection import TF_PFAM_DOMAINS
        from helixlang.plugins.annotation.transporter import TRANSPORT_FAMILIES
        checks["import_all_4_modules"] = True

        # --- KEGG database ---
        db = build_ko_db()
        assert isinstance(db, KOReactionDB), "build_ko_db should return KOReactionDB"
        db_size = db.size
        assert db_size > 10, f"KO database should have > 10 entries, got {db_size}"
        checks["ko_db_size"] = True
        details["ko_db_size"] = db_size

        assert db.has_ko("K00844"), "KO database should contain K00844"
        checks["ko_db_has_k00844"] = True

        mapping = db.lookup("K00844")
        assert mapping is not None, "lookup('K00844') should not be None"
        assert hasattr(mapping, "reaction_ids"), "lookup result should have reaction_ids"
        assert len(mapping.reaction_ids) > 0, "K00844 should map to at least one reaction"
        checks["ko_db_lookup"] = True
        details["k00844_reactions"] = mapping.reaction_ids

        # --- Sequence translation ---
        protein = translate("ATGAAATTT")
        assert protein == "MKF", f"translate('ATGAAATTT') should be 'MKF', got '{protein}'"
        checks["translate_mkf"] = True

        protein_stop = translate("ATGTAA")
        assert protein_stop.startswith("M"), f"translate('ATGTAA') should start with 'M', got '{protein_stop}'"
        checks["translate_stop_codon"] = True
        details["translate_atgtaa"] = protein_stop

        # --- Reverse complement ---
        rc = reverse_complement("ATGC")
        assert rc == "GCAT", f"reverse_complement('ATGC') should be 'GCAT', got '{rc}'"
        checks["reverse_complement"] = True

        # --- TF PFAM domains ---
        assert isinstance(TF_PFAM_DOMAINS, dict), "TF_PFAM_DOMAINS should be a dict"
        assert len(TF_PFAM_DOMAINS) >= 10, f"TF_PFAM_DOMAINS should have >= 10 entries, got {len(TF_PFAM_DOMAINS)}"
        checks["tf_pfam_domains_count"] = True
        details["tf_pfam_domains_count"] = len(TF_PFAM_DOMAINS)

        # --- Transporter families ---
        assert isinstance(TRANSPORT_FAMILIES, dict), "TRANSPORT_FAMILIES should be a dict"
        assert len(TRANSPORT_FAMILIES) >= 5, f"TRANSPORT_FAMILIES should have >= 5 entries, got {len(TRANSPORT_FAMILIES)}"
        checks["transport_families_count"] = True
        details["transport_families_count"] = len(TRANSPORT_FAMILIES)

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "55_annotation_tools",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "kegg": "Kanehisa M, Goto S 2000, Nucleic Acids Res 28:27-30",
                "pfam": "El-Gebali S et al. 2019, Nucleic Acids Res 47:D427",
            },
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "55_annotation_tools",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
