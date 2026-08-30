"""Genome-scale metabolic model reconstruction (doc/20 §6, doc/24 full GEM import, doc/26 Phases D-E)."""
from __future__ import annotations

from helixlang.plugins.gem.biomass import (
    build_biomass_reaction,
    get_biomass_composition,
    list_available_templates,
)
from helixlang.plugins.gem.bottom_up import GPRRule, bottom_up_reconstruct
from helixlang.plugins.gem.bridge import (
    build_enzyme_capacity,
    consensus_to_metabolic_model,
    regulatory_edges_to_grn,
)
from helixlang.plugins.gem.community import (
    CommunityFBAExtended,
    CommunityResult,
    ExchangeNetwork,
    OrganismModel,
)
from helixlang.plugins.gem.consensus import consensus_merge
from helixlang.plugins.gem.ecgem import (
    ECGEMBuilder,
    ECGEMResult,
    EnzymeConstraint,
    EnzymePoolConstraint,
)
from helixlang.plugins.gem.full_model import FullModelAdapter
from helixlang.plugins.gem.gapfill import GapfillPool, GapfillResult, gapfill, lp_gapfill
from helixlang.plugins.gem.grn_inference import RegulatoryEdge, infer_grn
from helixlang.plugins.gem.organism_registry import (
    OrganismConfig,
    get_organism_config,
    has_full_model,
    list_supported_organisms,
)
from helixlang.plugins.gem.sbml_export import export_sbml, model_to_sbml_string
from helixlang.plugins.gem.sbml_import import (
    detect_compartments,
    detect_exchange_reactions,
    get_model_info,
    load_bigg_model,
    load_sbml_model,
)
from helixlang.plugins.gem.top_down import top_down_reconstruct
from helixlang.plugins.gem.validation import (
    GemValidationResult,
    check_mass_balance,
    gene_essentiality_test,
    validate_model,
)

__all__ = [
    "CommunityFBAExtended",
    "CommunityResult",
    "ECGEMBuilder",
    "ECGEMResult",
    "ExchangeNetwork",
    "GPRRule",
    "GapfillPool",
    "GapfillResult",
    "GemValidationResult",
    "OrganismConfig",
    "OrganismModel",
    "RegulatoryEdge",
    "EnzymeConstraint",
    "EnzymePoolConstraint",
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


# ---------------------------------------------------------------------------
# Plugin contract (doc/36 §7: gem/* -> plugins/gem/)
# ---------------------------------------------------------------------------
from collections.abc import Callable

from helixlang.api.registry import PluginProvider


def _check(pkg: str) -> bool:
    def _probe() -> bool:
        try:
            __import__(pkg)
            return True
        except ImportError:
            return False
    return _probe()


def _make_backend(cfg: dict | None = None) -> type:
    from helixlang.plugins.gem.community import CommunityFBAExtended
    return CommunityFBAExtended


def _load() -> Callable[[dict | None], type]:
    if not _check("numpy"):
        from helixlang.core.errors import PluginDependencyError
        raise PluginDependencyError("gem", "numpy", "gem")
    return _make_backend


PLUGIN = PluginProvider(
    name="gem",
    extra="gem",
    keywords=("gem", "genome"),
    native=None,
    capability_flags=("--low-fidelity",),
    checks={"numpy": lambda: _check("numpy")},
    load=_load,
)
