#!/usr/bin/env python3
"""Benchmark 29: Drug ADME predefined library lookup."""
from __future__ import annotations

import json
import sys
import time

from helixlang.plugins.human.drug import (
    Drug,
    get_predefined_drug,
    list_predefined_drugs,
)


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "29_drug_adme"}
    try:
        # 1. list_predefined_drugs() returns >= 10 drugs
        drug_names = list_predefined_drugs()
        assert len(drug_names) >= 10, (
            f"Expected >=10 predefined drugs, got {len(drug_names)}"
        )

        # 2. get_predefined_drug("warfarin") returns a Drug object
        warfarin = get_predefined_drug("warfarin")
        assert warfarin is not None, "warfarin not found"
        assert isinstance(warfarin, Drug), f"Expected Drug, got {type(warfarin).__name__}"
        assert warfarin.molecule.name.lower() == "warfarin", (
            f"Expected name 'warfarin', got {warfarin.molecule.name!r}"
        )

        # 3. Warfarin has oral bioavailability > 0
        assert warfarin.bioavailability > 0.0, (
            f"Warfarin bioavailability should be >0, got {warfarin.bioavailability}"
        )
        assert warfarin.route == "oral", (
            f"Warfarin route should be oral, got {warfarin.route!r}"
        )

        # 4. At least 3 drugs have valid SMILES strings
        valid_smiles_count = 0
        drugs_with_smiles: list[str] = []
        for name in drug_names:
            drug = get_predefined_drug(name)
            if drug is not None and drug.molecule.smiles:
                valid_smiles_count += 1
                drugs_with_smiles.append(name)
        assert valid_smiles_count >= 3, (
            f"Expected >=3 drugs with SMILES, got {valid_smiles_count}"
        )

        # Additional: verify warfarin has expected ADME properties
        assert warfarin.half_life_h > 0, "Warfarin half-life should be positive"
        assert warfarin.molecule.molecular_weight_da > 0, "Warfarin MW should be positive"
        assert warfarin.molecule.formula, "Warfarin should have a formula"

        # Verify case-insensitive lookup
        upper_drug = get_predefined_drug("WARFARIN")
        assert upper_drug is not None, "Case-insensitive lookup failed"

        # Verify non-existent drug returns None
        missing = get_predefined_drug("nonexistent_drug_xyz")
        assert missing is None, "Non-existent drug should return None"

        # Verify Drug dataclass has expected methods
        assert hasattr(warfarin, "validate"), "Drug should have validate()"
        assert hasattr(warfarin, "elimination_rate_constant"), "Drug should have elimination_rate_constant()"
        problems = warfarin.validate()
        assert isinstance(problems, list), "validate() should return a list"

        elapsed = time.perf_counter() - t0
        results.update({
            "status": "PASS",
            "reference": {
                "source": "DrugBank, CPIC guidelines, Rowland & Tozer Clinical PK/PD",
                "doi": "10.1093/nar/gkx1004",
                "note": "Wishart DS et al. 2018, Nucleic Acids Res 46:D618-D625 (DrugBank 5.0); CPIC ADME gene-drug tables.",
            },
            "checks": {
                "list_predefined_drugs_returns_at_least_10": True,
                "get_predefined_drug_warfarin_returns_drug": True,
                "warfarin_oral_bioavailability_positive": True,
                "at_least_3_drugs_have_valid_smiles": True,
            },
            "details": {
                "drug_count": len(drug_names),
                "drug_names": sorted(drug_names),
                "warfarin_bioavailability": warfarin.bioavailability,
                "warfarin_half_life_h": warfarin.half_life_h,
                "warfarin_mw_da": warfarin.molecule.molecular_weight_da,
                "warfarin_formula": warfarin.molecule.formula,
                "drugs_with_smiles_count": valid_smiles_count,
                "drugs_with_smiles": drugs_with_smiles[:10],
                "warfarin_validation_problems": problems,
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
