#!/usr/bin/env python3
"""Benchmark 54: SBML import + GRN inference — round-trip and regulatory data."""
from __future__ import annotations

import json
import sys
import tempfile
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

MINIMAL_SBML = """\
<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core"
      level="3" version="1">
  <model id="test_minimal_model" name="Minimal Test Model">
    <listOfCompartments>
      <compartment id="c" name="cytosol" spatialDimensions="3"
                   size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A_c" name="A" compartment="c"
               initialConcentration="1" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="B_c" name="B" compartment="c"
               initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="C_c" name="C" compartment="c"
               initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="R1" name="Reaction 1" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="A_c" stoichiometry="1"
                            constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B_c" stoichiometry="1"
                            constant="true"/>
        </listOfProducts>
      </reaction>
      <reaction id="R2" name="Reaction 2" reversible="true" fast="false">
        <listOfReactants>
          <speciesReference species="B_c" stoichiometry="1"
                            constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="C_c" stoichiometry="1"
                            constant="true"/>
        </listOfProducts>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    reference = "Keating SM et al. 2020, Nat Biotechnol 38:534-543"
    try:
        from helixlang.plugins.gem.sbml_import import load_sbml_model
        from helixlang.plugins.gem.grn_inference import (
            GRNInferenceResult,
            RegulatoryEdge,
            EvidenceLevel,
            KNOWN_REGULATORY_INTERACTIONS,
        )
        checks["import_modules"] = True

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False
        ) as f:
            f.write(MINIMAL_SBML)
            tmp_path = f.name

        model = load_sbml_model(tmp_path)
        from helixlang.plugins.runtime.metabolism import MetabolicModel
        assert isinstance(model, MetabolicModel), (
            f"Expected MetabolicModel, got {type(model)}"
        )
        assert len(model.reactions) == 2, (
            f"Expected 2 reactions, got {len(model.reactions)}"
        )
        assert "R1" in model.reactions, "R1 should be in model"
        assert "R2" in model.reactions, "R2 should be in model"
        checks["sbml_import"] = True
        details["model_reaction_count"] = len(model.reactions)
        details["model_metabolite_count"] = len(model.metabolites)
        details["reaction_ids"] = list(model.reactions.keys())

        grn = GRNInferenceResult()
        assert isinstance(grn, GRNInferenceResult)
        assert grn.total_edges == 0
        assert grn.regulatory_edges == []
        checks["grn_result_instantiation"] = True

        assert isinstance(KNOWN_REGULATORY_INTERACTIONS, list), (
            "KNOWN_REGULATORY_INTERACTIONS should be a list"
        )
        assert len(KNOWN_REGULATORY_INTERACTIONS) > 0, (
            "KNOWN_REGULATORY_INTERACTIONS should be non-empty"
        )
        sample = KNOWN_REGULATORY_INTERACTIONS[0]
        assert isinstance(sample, tuple) and len(sample) == 5, (
            "Each entry should be a 5-tuple (tf, target, reg_type, evidence, confidence)"
        )
        checks["known_interactions_nonempty"] = True
        details["known_interaction_count"] = len(KNOWN_REGULATORY_INTERACTIONS)

        assert hasattr(EvidenceLevel, "DATABASE"), (
            "EvidenceLevel should have DATABASE member"
        )
        assert hasattr(EvidenceLevel, "LITERATURE"), (
            "EvidenceLevel should have LITERATURE member"
        )
        assert hasattr(EvidenceLevel, "SEQUENCE_MOTIF"), (
            "EvidenceLevel should have SEQUENCE_MOTIF member"
        )
        assert EvidenceLevel.DATABASE.value == 1
        assert EvidenceLevel.LITERATURE.value == 2
        assert EvidenceLevel.SEQUENCE_MOTIF.value == 3
        checks["evidence_level_enum"] = True

        edge = RegulatoryEdge(
            tf_id="crp",
            target_gene="lacZ",
            regulation_type="activation",
            evidence_level=EvidenceLevel.DATABASE,
            confidence=0.95,
        )
        assert edge.is_high_confidence
        grn.regulatory_edges.append(edge)
        grn.total_edges = 1
        by_tf = grn.by_tf()
        assert "crp" in by_tf
        assert len(by_tf["crp"]) == 1
        checks["regulatory_edge_lifecycle"] = True

        import os
        os.unlink(tmp_path)

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "54_sbml_grn_inference",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": reference,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "54_sbml_grn_inference",
            "status": "FAIL",
            "checks": checks,
            "details": details,
            "reference": reference,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
