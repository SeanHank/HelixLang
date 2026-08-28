#!/usr/bin/env python3
"""Benchmark 53: GEM reconstruction pipeline — consensus merge, bridge, SBML export."""
from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    reference = "Thiele I, Palsson BO 2010, Nat Protoc 5:93-110"
    try:
        from helixlang.plugins.gem.consensus import (
            ConsensusReaction,
            ConsensusResult,
            consensus_merge,
        )
        from helixlang.plugins.gem.bottom_up import BottomUpResult, GPRRule, ReactionEntry
        from helixlang.plugins.gem.top_down import TopDownResult
        from helixlang.plugins.gem.bridge import consensus_to_metabolic_model
        from helixlang.plugins.gem.sbml_export import model_to_sbml_string
        checks["import_modules"] = True

        bu_reactions = [
            ReactionEntry(
                reaction_id="PFK",
                equation="f6p + atp -> fdp + adp",
                gene_ids=["b3916", "b1721"],
                gpr=GPRRule(reaction_id="PFK", gene_ids=["b3916", "b1721"]),
            ),
            ReactionEntry(
                reaction_id="GAPD",
                equation="g3p + nad + pi <=> 13dpg + nadh",
                gene_ids=["b1779"],
                gpr=GPRRule(reaction_id="GAPD", gene_ids=["b1779"]),
            ),
            ReactionEntry(
                reaction_id="BU_ONLY",
                equation="a -> b",
                gene_ids=["b9999"],
                gpr=GPRRule(reaction_id="BU_ONLY", gene_ids=["b9999"]),
            ),
        ]
        td_reactions = [
            ReactionEntry(
                reaction_id="PFK",
                equation="f6p + atp -> fdp + adp",
                confidence=0.8,
                source="top_down",
            ),
            ReactionEntry(
                reaction_id="GAPD",
                equation="g3p + nad + pi <=> 13dpg + nadh",
                confidence=0.8,
                source="top_down",
            ),
            ReactionEntry(
                reaction_id="TD_ONLY",
                equation="x -> y",
                confidence=0.6,
                source="top_down",
            ),
        ]
        bottom_up = BottomUpResult(reactions=bu_reactions, genes_annotated=3, ec_matched=3)
        top_down = TopDownResult(reactions=td_reactions, kept_reactions=3)
        checks["create_reaction_dicts"] = True
        details["bu_count"] = len(bu_reactions)
        details["td_count"] = len(td_reactions)

        consensus = consensus_merge(bottom_up, top_down)
        assert isinstance(consensus, ConsensusResult)
        assert consensus.reaction_count >= 3, (
            f"Expected at least 3 merged reactions, got {consensus.reaction_count}"
        )
        high_ids = consensus.high_confidence_ids()
        assert len(high_ids) > 0, "Should have at least one HIGH confidence reaction"
        all_confidences = [r.confidence for r in consensus.reactions]
        assert any(c >= 0.8 for c in all_confidences), (
            "Expected at least one HIGH confidence (>=0.8)"
        )
        assert consensus.high_confidence >= 1, "high_confidence count should be >= 1"
        assert consensus.medium_confidence >= 1, "medium_confidence count should be >= 1"
        checks["consensus_merge"] = True
        details["consensus_reaction_count"] = consensus.reaction_count
        details["high_confidence"] = consensus.high_confidence
        details["medium_confidence"] = consensus.medium_confidence
        details["reaction_ids"] = consensus.reaction_ids()
        details["high_confidence_ids"] = high_ids

        model = consensus_to_metabolic_model(consensus, biomass_rxn_id="PFK")
        from helixlang.plugins.runtime.metabolism import MetabolicModel
        assert isinstance(model, MetabolicModel), (
            f"Expected MetabolicModel, got {type(model)}"
        )
        assert len(model.reactions) > 0, "Model should have reactions"
        assert model.biomass_reaction == "PFK"
        checks["consensus_to_metabolic_model"] = True
        details["model_reaction_count"] = len(model.reactions)
        details["model_metabolite_count"] = len(model.metabolites)
        details["biomass_reaction"] = model.biomass_reaction

        test_model_id = "benchmark53_test_model"
        test_organism = "Escherichia coli K-12"
        sbml_str = model_to_sbml_string(model, model_id=test_model_id, organism_name=test_organism)
        assert isinstance(sbml_str, str), f"Expected string, got {type(sbml_str)}"
        assert len(sbml_str) > 0, "SBML string should not be empty"
        assert "<sbml" in sbml_str, "SBML string should contain '<sbml'"

        root = ET.fromstring(sbml_str)
        assert root.tag.endswith("sbml") or "sbml" in root.tag, (
            f"Root tag should be sbml, got {root.tag}"
        )
        checks["sbml_valid_xml"] = True
        details["sbml_length"] = len(sbml_str)

        sbml_reaction_ids = [r.get("id") for r in root.iter() if r.tag.endswith("reaction")]
        assert len(sbml_reaction_ids) > 0, "SBML should contain reaction elements"
        model_rxn_ids = list(model.reactions.keys())
        for rid in model_rxn_ids:
            assert rid in sbml_reaction_ids, (
                f"Reaction {rid} not found in SBML output"
            )
        checks["sbml_contains_reactions"] = True
        details["sbml_reaction_ids"] = sbml_reaction_ids

        assert test_model_id in sbml_str, (
            f"SBML string should contain model_id '{test_model_id}'"
        )
        assert test_organism in sbml_str, (
            f"SBML string should contain organism_name '{test_organism}'"
        )
        checks["sbml_model_id_and_organism"] = True

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "53_gem_reconstruction",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": reference,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "53_gem_reconstruction",
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
