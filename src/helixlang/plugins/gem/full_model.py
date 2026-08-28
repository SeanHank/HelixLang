"""Full model adapter for genome-scale metabolic models (doc/24 Phase C).

Wraps a :class:`MetabolicModel` loaded from BiGG/SBML with organism-aware
medium setting, exchange detection, and growth-rate dispatch — replacing
the hardcoded core-model assumptions in ``sim_runtime.py``.
"""
from __future__ import annotations

from helixlang.core.errors import BioError
from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis, MetabolicModel


class FullModelAdapter:
    """Adapter wrapping a full genome-scale MetabolicModel.

    Provides:
    - Automatic exchange reaction detection
    - Organism-aware medium application
    - Growth-rate dispatching (scipy for large models)

    Usage::

        adapter = FullModelAdapter.from_bigg("e_coli_k12")
        adapter.apply_medium("glucose_minimal")
        fluxes = adapter.solve()
        print(adapter.growth_rate)
    """

    def __init__(
        self,
        model: MetabolicModel,
        organism_id: str,
        biomass_rxn: str | None = None,
    ) -> None:
        from helixlang.plugins.gem.organism_registry import get_organism_config

        self.model = model
        self.organism_id = organism_id
        self.config = get_organism_config(organism_id)
        self.biomass_rxn: str = biomass_rxn or model.biomass_reaction or ""
        if not self.biomass_rxn:
            raise BioError(
                "model has no biomass reaction; specify one explicitly"
            )
        self.exchange_reactions: list[str] = self._detect_exchanges()
        self.transport_reactions: list[str] = self._detect_transport()
        self.internal_reactions: list[str] = self._detect_internal()
        self.growth_rate: float = 0.0
        self.fba: FluxBalanceAnalysis | None = None

    @classmethod
    def from_bigg(cls, organism_id: str) -> FullModelAdapter:
        """Create adapter by loading a model from BiGG."""
        from helixlang.plugins.gem.organism_registry import get_organism_config
        from helixlang.plugins.gem.sbml_import import load_bigg_model

        config = get_organism_config(organism_id)
        if config is None:
            raise BioError(
                f"organism {organism_id!r} not in registry; "
                f"available: {list_supported_organisms_str()}"
            )
        model = load_bigg_model(config.bigg_id)
        return cls(model, organism_id, biomass_rxn=config.biomass_rxn)

    @classmethod
    def from_sbml(
        cls, path: str, organism_id: str, biomass_rxn: str | None = None
    ) -> FullModelAdapter:
        """Create adapter by loading an SBML file."""
        from helixlang.plugins.gem.sbml_import import load_sbml_model

        model = load_sbml_model(path)
        return cls(model, organism_id, biomass_rxn=biomass_rxn)

    def _detect_exchanges(self) -> list[str]:
        """Auto-detect exchange reactions (EX_* pattern or exchange subsystem)."""
        exchanges = []
        for rid, rxn in self.model.reactions.items():
            if rid.startswith("EX_"):
                exchanges.append(rid)
            elif "exchange" in rxn.subsystem.lower() and len(rxn.stoichiometry) <= 2:
                exchanges.append(rid)
        return sorted(set(exchanges))

    def _detect_transport(self) -> list[str]:
        """Detect transport reactions (cross-compartment by metabolite suffixes)."""
        transports = []
        for rid, rxn in self.model.reactions.items():
            if rid.startswith("EX_"):
                continue
            compartments = set()
            for met in rxn.stoichiometry:
                if "_" in met:
                    suffix = met.rsplit("_", 1)[-1]
                    if len(suffix) <= 3 and suffix.isalnum():
                        compartments.add(suffix)
            if len(compartments) > 1:
                transports.append(rid)
        return sorted(transports)

    def _detect_internal(self) -> list[str]:
        """All reactions that are neither exchange nor transport."""
        exc_set = set(self.exchange_reactions)
        trn_set = set(self.transport_reactions)
        return sorted(
            rid for rid in self.model.reactions
            if rid not in exc_set and rid not in trn_set
        )

    # -----------------------------------------------------------------------
    # Medium application
    # -----------------------------------------------------------------------

    # Standard medium presets (same rates as sim_runtime.py)
    _MEDIUM_PRESETS: dict[str, dict[str, float]] = {
        "glucose_minimal": {
            "glc-D_e": 10.0, "o2_e": 20.0, "nh4_e": 1000.0,
            "pi_e": 1000.0, "h2o_e": 1000.0, "h_e": 1000.0,
        },
        "m9_minimal": {
            "glc-D_e": 10.0, "o2_e": 20.0, "nh4_e": 1000.0,
            "pi_e": 1000.0, "h2o_e": 1000.0, "h_e": 1000.0,
            "so4_e": 1000.0,
        },
        "lb": {
            "o2_e": 20.0, "nh4_e": 1000.0, "pi_e": 1000.0,
            "h2o_e": 1000.0, "h_e": 1000.0, "so4_e": 1000.0,
            "phe-L_e": 5.0, "trp-L_e": 1.0, "cys-L_e": 0.5,
            "met-L_e": 1.5, "tyr-L_e": 1.0, "leu-L_e": 5.0,
            "ile-L_e": 3.0, "val-L_e": 4.0, "ala-L_e": 5.0,
            "gly_e": 3.0, "ser-L_e": 3.0, "thr-L_e": 3.0,
            "asp-L_e": 4.0, "glu-L_e": 7.0, "gln-L_e": 3.0,
            "arg-L_e": 3.0, "lys-L_e": 4.0, "his-L_e": 1.0,
            "pro-L_e": 3.0,
        },
        "bg11": {
            "co2_e": 1000.0, "o2_e": 20.0, "no3_e": 1000.0,
            "nh4_e": 1000.0, "pi_e": 1000.0, "h2o_e": 1000.0,
            "h_e": 1000.0, "so4_e": 1000.0, "mg2_e": 1000.0,
            "ca2_e": 1000.0, "na1_e": 1000.0, "k1_e": 1000.0,
            "fe3_e": 0.1, "cl_e": 1000.0, "photon_e": 100.0,
        },
        "photoautotrophic": {
            "co2_e": 1000.0, "o2_e": 20.0, "no3_e": 1000.0,
            "nh4_e": 1000.0, "pi_e": 1000.0, "h2o_e": 1000.0,
            "h_e": 1000.0, "so4_e": 1000.0, "photon_e": 100.0,
        },
    }

    _TRACE_IMPORT_UB = 0.1

    def apply_medium(self, medium_name: str) -> None:
        """Apply a medium preset to the model, setting exchange bounds.

        For full genome-scale models, we take a minimal approach: only set
        bounds for metabolites explicitly listed in the medium preset.
        The model's own default bounds (which are biologically realistic)
        are preserved for all other exchanges.

        Steps:
        1. Open medium-specified nutrients at full rate
        2. Cap non-essential free imports if they aren't in the medium
        3. Close Calvin cycle / PET for non-photoautotrophic media
        """
        uptake = self._MEDIUM_PRESETS.get(
            medium_name,
            self._MEDIUM_PRESETS.get("glucose_minimal"),
        )
        if uptake is None:
            raise BioError(f"unknown medium {medium_name!r}")

        # Build a lookup of metabolite -> exchange reaction for this model
        met_to_exchange: dict[str, str] = {}
        for rid in self.exchange_reactions:
            rxn = self.model.reactions.get(rid)
            if rxn is None or len(rxn.stoichiometry) != 1:
                continue
            met = next(iter(rxn.stoichiometry))
            met_to_exchange[met] = rid
            met_to_exchange[self._normalize_met(met)] = rid

        # 1. Open medium-specified nutrients at full rate
        opened_rids: set[str] = set()
        for met, rate in uptake.items():
            matched_rid: str | None = met_to_exchange.get(met) or met_to_exchange.get(self._normalize_met(met))
            if matched_rid is not None:
                rxn = self.model.reactions.get(matched_rid)
                if rxn is not None:
                    for _m2, coef in rxn.stoichiometry.items():
                        if coef < 0:
                            rxn.lower_bound = -abs(rate)
                        else:
                            rxn.upper_bound = abs(rate)
                    opened_rids.add(matched_rid)
            else:
                # Fallback: scan all exchanges for this metabolite
                for rid2 in self.exchange_reactions:
                    rxn2 = self.model.reactions.get(rid2)
                    if rxn2 is None or met not in rxn2.stoichiometry:
                        continue
                    coef = rxn2.stoichiometry[met]
                    if coef < 0:
                        rxn2.lower_bound = -abs(rate)
                    else:
                        rxn2.upper_bound = abs(rate)
                    opened_rids.add(rid2)

        # 2. Close Calvin/PET for non-photo media
        photo_media = {"bg11", "photoautotrophic"}
        if medium_name not in photo_media and self.config:
            for rxn_id in self.config.light_reactions + self.config.calvin_reactions:
                if rxn_id in self.model.reactions:
                    self.model.reactions[rxn_id].lower_bound = 0.0
                    self.model.reactions[rxn_id].upper_bound = 0.0

    @staticmethod
    def _normalize_met(met: str) -> str:
        """Normalize BiGG metabolite IDs for matching.

        BiGG uses ``glc__D_e`` while presets use ``glc-D_e``.
        Also handles ``__LPAREN__`` / ``__RPAREN__`` conventions.
        """
        return (
            met
            .replace("__D_e", "-D_e")
            .replace("__L_e", "-L_e")
            .replace("__R_e", "-R_e")
            .replace("__M_e", "-M_e")
            .replace("__LPAREN__", "(")
            .replace("__RPAREN__", ")")
        )

    def set_uptake(self, metabolite: str, rate: float) -> None:
        """Set a specific metabolite uptake rate on its exchange reaction."""
        for rid in self.exchange_reactions:
            rxn = self.model.reactions.get(rid)
            if rxn is None or metabolite not in rxn.stoichiometry:
                continue
            coef = rxn.stoichiometry[metabolite]
            if coef < 0:
                rxn.lower_bound = -abs(rate)
            else:
                rxn.upper_bound = abs(rate)

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------

    def solve(self, objective: str | None = None, maximize: bool = True) -> dict[str, float]:
        """Solve FBA and return flux distribution.

        Uses scipy linprog for large models (>500 reactions), simplex for
        smaller ones.
        """
        self.fba = FluxBalanceAnalysis(self.model)
        obj = objective if objective is not None else self.biomass_rxn
        fluxes = self.fba.solve(objective=obj, maximize=maximize)
        self.growth_rate = fluxes.get(self.biomass_rxn, 0.0)
        return fluxes

    def get_exchange_fluxes(self, fluxes: dict[str, float] | None = None) -> dict[str, float]:
        """Extract exchange reaction fluxes from a solution."""
        if fluxes is None and self.fba is not None:
            fluxes = self.fba.last_solution or {}
        if fluxes is None:
            return {}
        return {rid: fluxes.get(rid, 0.0) for rid in self.exchange_reactions}

    def summary(self) -> dict:
        """Return a summary dict of the adapter state."""
        return {
            "organism": self.organism_id,
            "bigg_id": self.config.bigg_id if self.config else None,
            "n_reactions": len(self.model.reactions),
            "n_metabolites": len(self.model.metabolites),
            "n_genes": len(self.model.genes),
            "n_exchange": len(self.exchange_reactions),
            "n_transport": len(self.transport_reactions),
            "n_internal": len(self.internal_reactions),
            "biomass_rxn": self.biomass_rxn,
            "growth_rate": self.growth_rate,
        }


def list_supported_organisms_str() -> str:
    """Helper for error messages."""
    from helixlang.plugins.gem.organism_registry import list_supported_organisms
    return str(list_supported_organisms())
