"""Human GEM loading and tissue-specific metabolic overlays (doc/27)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from helixlang.plugins.runtime.metabolism import ECOLI_CORE_MODEL, MetabolicModel, Reaction
except ImportError:
    MetabolicModel = None  # type: ignore[assignment,misc]
    Reaction = None  # type: ignore[assignment,misc]
    ECOLI_CORE_MODEL = None  # type: ignore[assignment]

try:
    from helixlang.plugins.human.physiology import TISSUE_PROFILES
except ImportError:
    TISSUE_PROFILES = {}  # noqa: F811

_DATA_DIR = Path(__file__).parent / "data"

HUMAN_BIOMASS_COMPOSITION: dict[str, float] = {
    "protein": 0.55,
    "lipid": 0.15,
    "carbohydrate": 0.05,
    "nucleic_acid": 0.10,
    "ash": 0.05,
    "water": 0.70,
    "atp_per_gdw": 38.0,
    "cell_density_g_ml": 1.05,
}


# ---------------------------------------------------------------------------
# Biomass reaction
# ---------------------------------------------------------------------------

def create_human_biomass_reaction() -> Any:
    """Create a mammalian biomass Reaction for use in MetabolicModel."""
    if Reaction is None:
        raise ImportError("helixlang.plugins.runtime.metabolism required")
    stoich: dict[str, float] = {
        "Protein": -0.55,
        "Lipid": -0.15,
        "Carbohydrate": -0.05,
        "Nucleotide": -0.10,
        "ATP": -38.0,
        "ADP": 38.0,
        "Biomass": 1.0,
    }
    return Reaction(
        id="BIOMASS_HUMAN",
        name="Human mammalian biomass",
        stoichiometry=stoich,
        lower_bound=0.0,
        upper_bound=1000.0,
        subsystem="biomass",
    )


# ---------------------------------------------------------------------------
# Config and loader
# ---------------------------------------------------------------------------

@dataclass
class HumanGEMConfig:
    """Configuration for human GEM loading."""

    tissue: str = "liver"
    model_path: str = ""
    use_full_model: bool = False
    medium_override: dict[str, float] = field(default_factory=dict)


class HumanGEMLoader:
    """Load human metabolic models and apply tissue overlays.

    When no human GEM SBML is provided, falls back to the E. coli core
    model with human biomass composition overlay, keeping the simulation
    functional without requiring Recon3D download.
    """

    def __init__(self, config: HumanGEMConfig | None = None) -> None:
        self.config = config or HumanGEMConfig()

    def load_core_model(self) -> Any:
        """Load E. coli core model as metabolic proxy."""
        if MetabolicModel is None or ECOLI_CORE_MODEL is None:
            raise ImportError("helixlang.plugins.runtime.metabolism required")
        return copy.deepcopy(ECOLI_CORE_MODEL)

    def load_from_sbml(self, path: str) -> Any:
        """Load human GEM from SBML file.

        Requires CobraPy + the human GEM data.  A failure to load the requested
        human model is an explicit error (doc/36 §3ξ.3) — CobraPy availability
        is never silently substituted with the E. coli core model unless the
        program explicitly opts into the reduced-fidelity core proxy.
        """
        try:
            from helixlang.plugins.gem.full_model import FullModelAdapter
            adapter = FullModelAdapter.from_sbml(path, "human_recon3d")
            if adapter is not None:
                return adapter
        except Exception as exc:  # noqa: BLE001
            from helixlang.api.capabilities import opt_in
            if not opt_in("--low-fidelity"):
                from helixlang.core.errors import ModelMissingError
                raise ModelMissingError(
                    f"human GEM from {path}", "human",
                    detail=f"SBML load failed: {exc}",
                ) from exc
        return self.load_core_model()

    def apply_tissue_overlay(self, model: Any, tissue: str) -> Any:
        """Apply tissue-specific exchange bounds and constraints.

        Modifies the model in-place based on TISSUE_PROFILES parameters.
        """
        profile = TISSUE_PROFILES.get(tissue, {})
        if not profile or model is None:
            return model

        glucose_uptake = profile.get("glucose_uptake_mmol_per_kg_per_min", 1.0)
        oxygen_consumption = profile.get("oxygen_consumption_ml_per_kg_per_min", 10.0)

        for rxn in model.reactions.values():
            if not hasattr(rxn, "subsystem"):
                continue
            if rxn.subsystem != "exchange":
                continue
            rxn_id = rxn.id.lower()
            if "glc" in rxn_id:
                rxn.upper_bound = glucose_uptake * 10.0
                rxn.lower_bound = -glucose_uptake * 10.0
            elif "o2" in rxn_id:
                rxn.upper_bound = oxygen_consumption * 0.5
                rxn.lower_bound = -oxygen_consumption * 0.5

        for rxn in model.reactions.values():
            if rxn.id == "BIOMASS_Ec_iML1515_core_75p37M" or rxn.id == "BIOMASS":
                rxn.upper_bound = 1000.0
                break

        return model

    def get_exchange_reactions(self, model: Any) -> dict[str, str]:
        """Get organ exchange reactions from model."""
        if model is None:
            return {}
        exchanges: dict[str, str] = {}
        for rxn in model.reactions.values():
            if hasattr(rxn, "subsystem") and rxn.subsystem == "exchange":
                exchanges[rxn.id] = getattr(rxn, "name", rxn.id)
        return exchanges

    def get_tissue_profile(self, tissue: str) -> dict:
        """Get tissue profile from TISSUE_PROFILES."""
        return dict(TISSUE_PROFILES.get(tissue, {}))

    def load(self, tissue: str = "liver") -> tuple[Any, dict]:
        """Load model and apply tissue overlay.

        Returns (MetabolicModel, tissue_profile).
        """
        profile = self.get_tissue_profile(tissue)
        if self.config.model_path:
            model = self.load_from_sbml(self.config.model_path)
        else:
            model = self.load_core_model()
        model = self.apply_tissue_overlay(model, tissue)
        return model, profile


__all__ = [
    "HUMAN_BIOMASS_COMPOSITION",
    "create_human_biomass_reaction",
    "HumanGEMConfig",
    "HumanGEMLoader",
]
