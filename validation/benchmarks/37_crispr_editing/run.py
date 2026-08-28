#!/usr/bin/env python3
"""Benchmark 37: CRISPR Editing — PAM finding, guide design, off-target scoring."""
from __future__ import annotations

import json
import sys
import time

TEST_SEQUENCE = (
    "ATCGATCGATCGATCGATCGAAGGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
    "ATCGATCGATCGATCGATCGATCGATCGATCG"
)


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        from helixlang.plugins.runtime.crispr import (
            GuideRNA,
            OffTargetSite,
            design_guide,
            find_pam_sites,
            off_target_score,
            on_target_score,
        )

        checks["import_crispr_modules"] = True

        sites = find_pam_sites(TEST_SEQUENCE, "SpCas9", both_strands=False)
        details["pam_sites_found"] = len(sites)
        assert len(sites) > 0, "Should find PAM sites (NGG) in test sequence"
        checks["find_pam_sites_found"] = True

        has_expected_pam = any(s["pam"].upper().endswith("GG") for s in sites)
        assert has_expected_pam, "At least one PAM should be NGG"
        for s in sites[:5]:
            assert len(s["spacer"]) == 20, "SpCas9 spacer should be 20nt"
        checks["pam_at_expected_positions"] = True

        guide = design_guide(TEST_SEQUENCE, "SpCas9", position=30)
        assert isinstance(guide, GuideRNA), "Should return a GuideRNA"
        assert len(guide.spacer) == 20, "Spacer should be 20nt"
        assert guide.cas_variant == "SpCas9"
        details["guide_spacer"] = guide.spacer
        details["guide_pam"] = guide.pam
        checks["design_guide_works"] = True

        score = on_target_score(guide)
        details["on_target_score"] = score
        assert score > 0, f"On-target score should be > 0, got {score}"
        assert score <= 1.0, f"On-target score should be <= 1.0, got {score}"
        checks["on_target_score_positive"] = True

        off_targets = off_target_score(guide, TEST_SEQUENCE, max_mismatches=3)
        details["off_target_count"] = len(off_targets)
        assert isinstance(off_targets, list), "Off-targets should be a list"
        for ot in off_targets:
            assert isinstance(ot, OffTargetSite)
            assert ot.score >= 0, f"Off-target score should be >= 0, got {ot.score}"
        checks["off_target_score_non_negative"] = True

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "37_crispr_editing",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "37_crispr_editing",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
