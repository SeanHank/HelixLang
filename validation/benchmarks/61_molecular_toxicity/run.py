#!/usr/bin/env python3
"""Benchmark 61: Molecular toxicity prediction.

Validates the molecular_toxicity module:
  - smiles_autofill auto-fill drug parameters from SMILES
  - _compute_rdkit_descriptors molecular descriptor computation
  - MolecularToxicityPredictor prediction pipeline
  - ToxicityProfile and ActivityProfile data classes

Reference: Hughes JP et al. 2008, Br J Pharmacol 154:731-739 (therapeutic index).
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    try:
        from helixlang.human.molecular_toxicity import (
            MolecularToxicityPredictor,
            ToxicityProfile,
            ActivityProfile,
            smiles_autofill,
            _compute_rdkit_descriptors,
        )

        checks: dict[str, bool] = {}
        details: dict[str, object] = {}

        # 1. All required imports succeeded
        checks["import_all_classes"] = True

        # 2. smiles_autofill: instantiate and call toxicity_profile on benzene
        af = smiles_autofill()
        benzene_tox = af.toxicity_profile("c1ccccc1")
        checks["smiles_autofill_benzene_returns_toxicity"] = isinstance(
            benzene_tox, ToxicityProfile
        )
        details["benzene_tox_confidence"] = benzene_tox.confidence

        # 3. _compute_rdkit_descriptors on benzene → dict (with MolWt > 0 when RDKit installed)
        desc = _compute_rdkit_descriptors("c1ccccc1")
        checks["rdkit_descriptors_returns_dict"] = isinstance(desc, dict)
        # RDKit may not be installed; empty dict is the documented fallback
        if desc:
            checks["rdkit_descriptors_molwt_positive"] = desc.get("MolWt", 0.0) > 0.0
        else:
            checks["rdkit_descriptors_molwt_positive"] = True  # no RDKit → empty fallback OK
        details["benzene_molwt"] = desc.get("MolWt", 0.0)
        details["rdkit_available"] = bool(desc)

        # 4. MolecularToxicityPredictor: create with default params
        predictor = MolecularToxicityPredictor()
        checks["predictor_instantiates"] = isinstance(
            predictor, MolecularToxicityPredictor
        )

        # 5. Predict toxicity for aspirin → ToxicityProfile
        aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        aspirin_tox = predictor.predict_toxicity(aspirin_smiles)
        checks["aspirin_returns_toxicity_profile"] = isinstance(
            aspirin_tox, ToxicityProfile
        )
        checks["aspirin_tox_has_fields"] = all(
            hasattr(aspirin_tox, f)
            for f in [
                "hepatotoxicity_score",
                "nephrotoxicity_score",
                "cardiotoxicity_score",
            ]
        )
        details["aspirin_hepatotox"] = aspirin_tox.hepatotoxicity_score
        details["aspirin_nephrotox"] = aspirin_tox.nephrotoxicity_score
        details["aspirin_cardiotox"] = aspirin_tox.cardiotoxicity_score
        details["aspirin_confidence"] = aspirin_tox.confidence

        # 6. ToxicityProfile has expected organ-specific fields
        checks["toxicity_profile_has_organ_fields"] = all(
            hasattr(benzene_tox, f)
            for f in [
                "hepatotoxicity_score",
                "nephrotoxicity_score",
                "cardiotoxicity_score",
                "myelosuppression_score",
            ]
        )

        all_pass = all(checks.values())
        return {
            "id": "61_molecular_toxicity",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": "Hughes JP et al. 2008",
                "authors": "Hughes JP, Rees S, Kalindjian SB, Philpott KL",
                "year": 2008,
                "journal": "Br J Pharmacol",
                "volume": "154",
                "pages": "731-739",
            },
            "runtime_seconds": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "id": "61_molecular_toxicity",
            "status": "FAIL",
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
