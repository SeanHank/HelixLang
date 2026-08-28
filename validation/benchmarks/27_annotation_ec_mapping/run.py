#!/usr/bin/env python3
"""Benchmark 27: Annotation EC Mapping — EC number to reaction mapping."""
from __future__ import annotations

import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.plugins.annotation.ec_mapping import (
            ECReactionDB,
            ReactionMapping,
            build_ec_db,
        )
        checks["import_ec_mapping_module"] = True

        db = build_ec_db()
        assert isinstance(db, ECReactionDB), "build_ec_db should return ECReactionDB"
        assert db.size > 0, "EC database should have entries"
        checks["build_ec_db"] = True
        details["ec_db_size"] = db.size

        result = db.lookup("1.1.1.1")
        assert result is not None, "EC 1.1.1.1 should be in the database"
        assert isinstance(result, ReactionMapping), "Lookup should return ReactionMapping"
        assert result.ec_number == "1.1.1.1", "EC number should match"
        assert len(result.reaction_ids) > 0, "EC 1.1.1.1 should map to at least one reaction"
        checks["lookup_known_ec_number"] = True
        details["ec_1_1_1_1_reactions"] = result.reaction_ids

        assert db.size >= 10, f"EC database should have >= 10 entries, got {db.size}"
        checks["ec_database_has足够的_entries"] = True

        additional_lookups = {}
        for ec in ["2.7.1.1", "2.3.3.1", "1.2.4.1", "4.1.2.13", "5.3.1.1"]:
            r = db.lookup(ec)
            if r is not None:
                additional_lookups[ec] = r.reaction_ids
        details["additional_lookups"] = additional_lookups
        assert len(additional_lookups) >= 4, "Should find >= 4 additional EC mappings"

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "27_annotation_ec_mapping",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "27_annotation_ec_mapping",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
