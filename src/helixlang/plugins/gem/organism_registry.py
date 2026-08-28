"""Organism registry for genome-scale metabolic models (doc/24 Phase C).

Maps organism identifiers to BiGG model IDs, biomass reaction names,
exchange patterns, and medium configurations for genome-scale models.
"""
from __future__ import annotations


class OrganismConfig:
    """Configuration for a supported organism's genome-scale model."""

    __slots__ = (
        "organism_id", "bigg_id", "name", "model_type",
        "biomass_rxn", "exchange_prefix", "default_medium",
        "glucose_exchange", "oxygen_exchange", "co2_exchange",
        "light_reactions", "calvin_reactions",
    )

    def __init__(
        self,
        organism_id: str,
        bigg_id: str,
        name: str,
        model_type: str,
        biomass_rxn: str,
        exchange_prefix: str = "EX_",
        default_medium: str = "glucose_minimal",
        glucose_exchange: str | None = None,
        oxygen_exchange: str | None = None,
        co2_exchange: str | None = None,
        light_reactions: list[str] | None = None,
        calvin_reactions: list[str] | None = None,
    ) -> None:
        self.organism_id = organism_id
        self.bigg_id = bigg_id
        self.name = name
        self.model_type = model_type
        self.biomass_rxn = biomass_rxn
        self.exchange_prefix = exchange_prefix
        self.default_medium = default_medium
        self.glucose_exchange = glucose_exchange
        self.oxygen_exchange = oxygen_exchange
        self.co2_exchange = co2_exchange
        self.light_reactions = light_reactions or []
        self.calvin_reactions = calvin_reactions or []

    def to_dict(self) -> dict:
        return {
            "organism_id": self.organism_id,
            "bigg_id": self.bigg_id,
            "name": self.name,
            "model_type": self.model_type,
            "biomass_rxn": self.biomass_rxn,
            "default_medium": self.default_medium,
        }


# ---------------------------------------------------------------------------
# Registry of supported organisms with genome-scale models
# ---------------------------------------------------------------------------

ORGANISM_REGISTRY: dict[str, OrganismConfig] = {
    "e_coli_k12": OrganismConfig(
        organism_id="e_coli_k12",
        bigg_id="iML1515",
        name="Escherichia coli K-12 MG1655",
        model_type="gram_negative",
        biomass_rxn="BIOMASS_Ec_iML1515_core_75p37M",
        default_medium="glucose_minimal",
        glucose_exchange="EX_glc_e",
        oxygen_exchange="EX_o2_e",
    ),
    "e_coli": OrganismConfig(
        organism_id="e_coli",
        bigg_id="iML1515",
        name="Escherichia coli K-12 MG1655",
        model_type="gram_negative",
        biomass_rxn="BIOMASS_Ec_iML1515_core_75p37M",
        default_medium="glucose_minimal",
        glucose_exchange="EX_glc_e",
        oxygen_exchange="EX_o2_e",
    ),
    "e_coli_mg1655": OrganismConfig(
        organism_id="e_coli_mg1655",
        bigg_id="iML1515",
        name="Escherichia coli K-12 MG1655",
        model_type="gram_negative",
        biomass_rxn="BIOMASS_Ec_iML1515_core_75p37M",
        default_medium="glucose_minimal",
        glucose_exchange="EX_glc_e",
        oxygen_exchange="EX_o2_e",
    ),
    "synechocystis_pcc6803": OrganismConfig(
        organism_id="synechocystis_pcc6803",
        bigg_id="iJN678",
        name="Synechocystis sp. PCC 6803",
        model_type="cyanobacteria",
        biomass_rxn="BIOMASS_Ec_SynAuto",
        default_medium="bg11",
        co2_exchange="EX_co2_e",
        oxygen_exchange="EX_o2_e",
        light_reactions=["PSII", "PSI", "Cytb6f", "PET"],
        calvin_reactions=["RBPC", "PRUK", "GAPD", "FBPASE", "SBPase", "TKT2"],
    ),
    "synechocystis": OrganismConfig(
        organism_id="synechocystis",
        bigg_id="iJN678",
        name="Synechocystis sp. PCC 6803",
        model_type="cyanobacteria",
        biomass_rxn="BIOMASS_Ec_SynAuto",
        default_medium="bg11",
        co2_exchange="EX_co2_e",
        oxygen_exchange="EX_o2_e",
        light_reactions=["PSII", "PSI", "Cytb6f", "PET"],
        calvin_reactions=["RBPC", "PRUK", "GAPD", "FBPASE", "SBPase", "TKT2"],
    ),
    "s_cerevisiae": OrganismConfig(
        organism_id="s_cerevisiae",
        bigg_id="iMM904",
        name="Saccharomyces cerevisiae S288C",
        model_type="yeast",
        biomass_rxn="BIOMASS",
        default_medium="glucose_minimal",
        glucose_exchange="EX_glc_e",
        oxygen_exchange="EX_o2_e",
    ),
    "b_subtilis": OrganismConfig(
        organism_id="b_subtilis",
        bigg_id="iBsu1103",
        name="Bacillus subtilis 168",
        model_type="gram_positive",
        biomass_rxn="BIOMASS",
        default_medium="glucose_minimal",
        glucose_exchange="EX_glc_e",
        oxygen_exchange="EX_o2_e",
    ),
    "human_recon3d": OrganismConfig(
        organism_id="human_recon3d",
        bigg_id="Recon3D",
        name="Homo sapiens Recon3D",
        model_type="mammalian",
        biomass_rxn="BIOMASS_reaction",
        default_medium="human_plasma",
    ),
}


def get_organism_config(organism: str) -> OrganismConfig | None:
    """Look up an organism configuration by ID (case-insensitive)."""
    return ORGANISM_REGISTRY.get(organism.lower().strip())


def list_supported_organisms() -> list[str]:
    """Return sorted list of organism IDs with genome-scale model support."""
    return sorted(set(ORGANISM_REGISTRY.keys()))


def has_full_model(organism: str) -> bool:
    """Check if an organism has a full GEM in the registry."""
    return organism.lower().strip() in ORGANISM_REGISTRY
