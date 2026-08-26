#!/usr/bin/env python3
"""Benchmark 28: Genotype CYP2D6 star-allele mapping."""
from __future__ import annotations

import json
import sys
import time

from helixlang.human.genotype import (
    CYP_ALLELE_ACTIVITIES,
    CYPStatus,
    GenotypeProfile,
    create_default_genotype,
    create_genotype_from_vcf,
)


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "28_genotype_cyp2d6"}
    try:
        # 1. create_default_genotype() returns a valid GenotypeProfile
        profile = create_default_genotype()
        assert isinstance(profile, GenotypeProfile), "Not a GenotypeProfile"
        assert len(profile.cyp_status) > 0, "No CYP status entries"

        # 2. CYP2D6*4/*4 -> "poor_metabolizer" status
        # Build a VCF with two CYP2D6*4 hom_alt calls
        vcf_cyp2d6_star4 = """\
chr22 42128000 CYP2D6*4 A G . . GENE=CYP2D6;CSQ=hom_alt
chr22 42128000 CYP2D6*4 A G . . GENE=CYP2D6;CSQ=hom_alt
"""
        pm_profile = create_genotype_from_vcf(vcf_cyp2d6_star4)
        cyp2d6_status = pm_profile.cyp_status.get("CYP2D6")
        assert cyp2d6_status is not None, "CYP2D6 status missing"
        assert cyp2d6_status.phenotype == "PM", (
            f"Expected PM for CYP2D6*4/*4, got {cyp2d6_status.phenotype}"
        )

        # 3. CYP2D6*1/*1 -> "normal_metabolizer" (EM) status
        # create_default_genotype is wild-type (*1/*1)
        default_cyp2d6 = profile.cyp_status.get("CYP2D6")
        assert default_cyp2d6 is not None
        assert default_cyp2d6.phenotype == "EM", (
            f"Expected EM for CYP2D6*1/*1, got {default_cyp2d6.phenotype}"
        )
        assert default_cyp2d6.activity_score == 2.0, (
            f"Expected activity_score=2.0, got {default_cyp2d6.activity_score}"
        )

        # 4. At least 3 star alleles are recognized
        cyp2d6_alleles = CYP_ALLELE_ACTIVITIES.get("CYP2D6", {})
        assert len(cyp2d6_alleles) >= 3, (
            f"Expected >=3 CYP2D6 star alleles, got {len(cyp2d6_alleles)}"
        )

        # Verify specific known activities
        assert cyp2d6_alleles["*1"] == 1.0, "CYP2D6*1 should have activity 1.0"
        assert cyp2d6_alleles["*4"] == 0.0, "CYP2D6*4 should have activity 0.0"
        assert cyp2d6_alleles["*10"] == 0.5, "CYP2D6*10 should have activity 0.5"

        # Verify CYPStatus dataclass behavior
        bad_status = CYPStatus(enzyme="CYP2D6", phenotype="EM", activity_score=2.0)
        assert bad_status.enzyme == "CYP2D6"
        assert bad_status.phenotype == "EM"

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "checks": {
                "create_default_genotype_returns_valid_profile": True,
                "cyp2d6_star4_star4_is_poor_metabolizer": True,
                "cyp2d6_star1_star1_is_normal_metabolizer": True,
                "at_least_3_star_alleles_recognized": True,
            },
            "details": {
                "default_genotype_type": type(profile).__name__,
                "cyp_status_count": len(profile.cyp_status),
                "cyp2d6_pm_phenotype": cyp2d6_status.phenotype,
                "cyp2d6_pm_activity_score": cyp2d6_status.activity_score,
                "cyp2d6_em_phenotype": default_cyp2d6.phenotype,
                "cyp2d6_em_activity_score": default_cyp2d6.activity_score,
                "cyp2d6_star_allele_count": len(cyp2d6_alleles),
                "transporter_count": len(profile.transporter_status),
                "non_cyp_enzyme_count": len(profile.non_cyp_enzyme_status),
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
