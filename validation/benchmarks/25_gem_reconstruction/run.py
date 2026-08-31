#!/usr/bin/env python3
"""Benchmark 25: GEM Reconstruction — import gem modules and build minimal model.

Requests the downloaded iML1515 (e_coli_k12) model first; if the download is
unavailable (offline CI / no vendored copy) it warns and falls back to the
always-vendored E. coli core model, so the reconstruction adapter path is still
validated instead of being skipped.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "references" / "models"


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

        # Load the model download-first, falling back to the vendored core.
        from helixlang.plugins.gem.full_model import FullModelAdapter

        try:
            adapter = FullModelAdapter.from_bigg(
                "e_coli_k12", model_dir=MODEL_DIR,
            )
            source = "exact"
        except Exception as e:
            import warnings

            from helixlang.plugins.gem.sbml_import import load_bigg_cobra_model
            from helixlang.plugins.runtime.metabolism import _from_cobra_model

            warnings.warn(
                f"BiGG iML1515 (e_coli_k12) unavailable: {e}; "
                "falling back to vendored E. coli core",
                RuntimeWarning,
                stacklevel=2,
            )
            cobra_model = load_bigg_cobra_model(
                "e_coli_core", model_dir=MODEL_DIR, offline=False)
            model = _from_cobra_model(cobra_model)
            adapter = FullModelAdapter(
                model, "e_coli_k12",
                biomass_rxn=model.biomass_reaction,
            )
            source = "fallback"
        adapter.apply_medium("glucose_minimal")
        adapter.solve()
        details["n_reactions"] = len(adapter.model.reactions)
        details["n_metabolites"] = len(adapter.model.metabolites)
        details["growth_rate"] = adapter.growth_rate
        details["source"] = source
        assert adapter.growth_rate > 0, "Growth rate should be positive"
        checks["full_model_adapter_instantiable"] = True

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        return {
            "id": "25_gem_reconstruction",
            "status": "PASS" if all_pass else "FAIL",
            "source": source,
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
    sys.exit(0 if r["status"] in ("PASS", "SKIP") else 1)
