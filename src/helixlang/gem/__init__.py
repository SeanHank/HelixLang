"""Genome-scale metabolic model reconstruction (doc/20 §6, doc/24 full GEM import)."""
from __future__ import annotations

from helixlang.gem.biomass import (
    build_biomass_reaction,
    get_biomass_composition,
    list_available_templates,
)
from helixlang.gem.bottom_up import GPRRule, bottom_up_reconstruct
from helixlang.gem.bridge import (
    build_enzyme_capacity,
    consensus_to_metabolic_model,
    regulatory_edges_to_grn,
)
from helixlang.gem.consensus import consensus_merge
from helixlang.gem.full_model import FullModelAdapter
from helixlang.gem.gapfill import GapfillPool, GapfillResult, gapfill, lp_gapfill
from helixlang.gem.grn_inference import RegulatoryEdge, infer_grn
from helixlang.gem.organism_registry import (
    OrganismConfig,
    get_organism_config,
    has_full_model,
    list_supported_organisms,
)
from helixlang.gem.sbml_export import export_sbml, model_to_sbml_string
from helixlang.gem.sbml_import import (
    detect_compartments,
    detect_exchange_reactions,
    get_model_info,
    load_bigg_model,
    load_sbml_model,
)
from helixlang.gem.top_down import top_down_reconstruct
from helixlang.gem.validation import (
    GemValidationResult,
    check_mass_balance,
    gene_essentiality_test,
    validate_model,
)

__all__ = [
    "GPRRule",
    "GapfillPool",
    "GapfillResult",
    "GemValidationResult",
    "OrganismConfig",
    "RegulatoryEdge",
    "FullModelAdapter",
    "bottom_up_reconstruct",
    "top_down_reconstruct",
    "consensus_merge",
    "gapfill",
    "lp_gapfill",
    "check_mass_balance",
    "gene_essentiality_test",
    "validate_model",
    "build_biomass_reaction",
    "get_biomass_composition",
    "list_available_templates",
    "export_sbml",
    "model_to_sbml_string",
    "infer_grn",
    "consensus_to_metabolic_model",
    "build_enzyme_capacity",
    "regulatory_edges_to_grn",
    "load_bigg_model",
    "load_sbml_model",
    "detect_exchange_reactions",
    "detect_compartments",
    "get_model_info",
    "get_organism_config",
    "has_full_model",
    "list_supported_organisms",
]
