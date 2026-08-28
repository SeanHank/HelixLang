#!/usr/bin/env python3
"""Benchmark 25: GEM Reconstruction — import gem modules and build minimal model."""
from __future__ import annotations

import importlib
import json
import sys
import time


def run() -> dict:
    t0 = time.perf_counter()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        bridge_mod = importlib.import_module("helixlang.plugins.gem.bridge")
        full_model_mod = importlib.import_module("helixlang.plugins.gem.full_model")
        consensus_mod = importlib.import_module("helixlang.plugins.gem.consensus")
        gapfill_mod = importlib.import_module("helixlang.plugins.gem.gapfill")
        grn_mod = importlib.import_module("helixlang.plugins.gem.grn_inference")
        biomass_mod = importlib.import_module("helixlang.plugins.gem.biomass")
        org_mod = importlib.import_module("helixlang.plugins.gem.organism_registry")
        sbml_mod = importlib.import_module("helixlang.plugins.gem.sbml_import")

        checks["import_gem_submodules"] = True

        assert hasattr(bridge_mod, "consensus_to_metabolic_model")
        assert hasattr(bridge_mod, "build_enzyme_capacity")
        assert hasattr(bridge_mod, "regulatory_edges_to_grn")
        assert hasattr(full_model_mod, "FullModelAdapter")
        assert hasattr(consensus_mod, "ConsensusResult")
        assert hasattr(gapfill_mod, "GapfillResult")
        assert hasattr(grn_mod, "RegulatoryEdge")
        assert hasattr(biomass_mod, "build_biomass_reaction")
        assert hasattr(org_mod, "get_organism_config")
        assert hasattr(org_mod, "list_supported_organisms")
        assert hasattr(sbml_mod, "load_bigg_model")
        assert hasattr(sbml_mod, "load_sbml_model")
        checks["key_classes_exist"] = True
        details["modules_verified"] = [
            "bridge", "full_model", "consensus", "gapfill",
            "grn_inference", "biomass", "organism_registry", "sbml_import",
        ]

        organisms = org_mod.list_supported_organisms()
        details["supported_organisms"] = organisms
        assert len(organisms) > 0, "Should have at least one supported organism"

        cfg = org_mod.get_organism_config("e_coli_k12")
        details["ecoli_config_bigg_id"] = cfg.bigg_id if cfg else None

        adapter_instantiated = False
        try:
            adapter = full_model_mod.FullModelAdapter.from_bigg("e_coli_k12")
            adapter.apply_medium("glucose_minimal")
            adapter.solve()
            adapter_instantiated = True
            details["n_reactions"] = len(adapter.model.reactions)
            details["n_metabolites"] = len(adapter.model.metabolites)
            details["growth_rate"] = adapter.growth_rate
            assert adapter.growth_rate > 0, "Growth rate should be positive"
        except Exception as e:
            details["adapter_error"] = str(e)
        checks["full_model_adapter_instantiable"] = adapter_instantiated

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "25_gem_reconstruction",
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "runtime_seconds": elapsed,
        }
    except Exception as e:
        return {
            "id": "25_gem_reconstruction",
            "status": "FAIL",
            "checks": checks,
            "error": str(e),
            "runtime_seconds": time.perf_counter() - t0,
        }


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
