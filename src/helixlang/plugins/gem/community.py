"""Community FBA extension with ecGEM and cross-feeding (doc/26 Phase E).

Extends the OptCom framework (Zomorrodi & Maranas 2012, PLoS One) to support
per-organism enzyme-constrained models, dynamic metabolite exchange, and
cross-feeding network detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrganismModel:
    organism_id: str
    model: Any
    ecgem: Any | None = None
    exchange_reactions: list[str] = field(default_factory=list)
    production: dict[str, float] = field(default_factory=dict)
    consumption: dict[str, float] = field(default_factory=dict)
    growth_rate: float = 0.0


@dataclass
class ExchangeNetwork:
    metabolites: list[str] = field(default_factory=list)
    producers: dict[str, dict[str, float]] = field(default_factory=dict)
    consumers: dict[str, dict[str, float]] = field(default_factory=dict)
    balance: dict[str, float] = field(default_factory=dict)


@dataclass
class CommunityResult:
    organisms: list[OrganismModel] = field(default_factory=list)
    exchange_network: ExchangeNetwork | None = None
    total_biomass: float = 0.0
    iterations: int = 0
    converged: bool = False
    objective_value: float = 0.0


class CommunityFBAExtended:
    """Multi-level community FBA with ecGEM and cross-feeding.

    Protocol:
    1. Solve each organism's FBA independently
    2. Identify exchange metabolites
    3. Set exchange bounds from production/consumption
    4. Re-solve with exchange constraints
    5. Iterate until convergence
    """

    def __init__(
        self,
        organisms: list[OrganismModel],
        medium: dict[str, float] | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> None:
        self.organisms = organisms
        self.medium = medium or {}
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(self) -> CommunityResult:
        prev_biomass = 0.0
        for iteration in range(self.max_iterations):
            for org in self.organisms:
                gr = self._solve_single(org)
                org.growth_rate = gr

            total_biomass = sum(org.growth_rate for org in self.organisms)

            if iteration > 0 and abs(total_biomass - prev_biomass) < self.tolerance:
                exchange = self._identify_exchanges()
                return CommunityResult(
                    organisms=self.organisms,
                    exchange_network=exchange,
                    total_biomass=total_biomass,
                    iterations=iteration + 1,
                    converged=True,
                    objective_value=total_biomass,
                )

            exchange = self._identify_exchanges()
            self._apply_exchange_bounds(exchange)
            prev_biomass = total_biomass

        exchange = self._identify_exchanges()
        total_biomass = sum(org.growth_rate for org in self.organisms)
        return CommunityResult(
            organisms=self.organisms,
            exchange_network=exchange,
            total_biomass=total_biomass,
            iterations=self.max_iterations,
            converged=False,
            objective_value=total_biomass,
        )

    def _solve_single(self, org: OrganismModel) -> float:
        try:
            from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis
            fba = FluxBalanceAnalysis(org.model)
            if org.model.biomass_reaction:
                fluxes = fba.solve(
                    objective=org.model.biomass_reaction, maximize=True
                )
                return fluxes.get(org.model.biomass_reaction, 0.0)
        except Exception as exc:  # noqa: BLE001
            # FBA solver crash silently zeroing an organism's biomass would
            # understate the community objective (doc/36 §3ξ.3, F2/F3).  Surface
            # it unless the program explicitly opted into reduced fidelity.
            from helixlang.core import fidelity
            if not fidelity.opt_in("--low-fidelity"):
                from helixlang.core.errors import ModelMissingError
                raise ModelMissingError(
                    f"{org.organism_id} FBA", "fba",
                    detail=f"solver failed: {exc}",
                ) from exc
        return 0.0

    def _identify_exchanges(self) -> ExchangeNetwork:
        network = ExchangeNetwork()
        all_exchange: dict[str, set[str]] = {}
        for org in self.organisms:
            for rxn_id in org.exchange_reactions:
                if rxn_id.startswith("EX_"):
                    met = rxn_id[3:]
                    all_exchange.setdefault(met, set()).add(org.organism_id)

        for met, orgs in all_exchange.items():
            if len(orgs) > 1:
                network.metabolites.append(met)
                for org in self.organisms:
                    rxn_id = f"EX_{met}"
                    if rxn_id in org.exchange_reactions:
                        if met in org.production:
                            network.producers.setdefault(org.organism_id, {})[met] = (
                                org.production[met]
                            )
                        if met in org.consumption:
                            network.consumers.setdefault(org.organism_id, {})[met] = (
                                org.consumption[met]
                            )
                net = sum(
                    network.producers.get(o, {}).get(met, 0.0)
                    for o in network.producers
                ) - sum(
                    network.consumers.get(o, {}).get(met, 0.0)
                    for o in network.consumers
                )
                network.balance[met] = net

        return network

    def _apply_exchange_bounds(self, network: ExchangeNetwork) -> None:
        for met in network.metabolites:
            rxn_id = f"EX_{met}"
            for org in self.organisms:
                if rxn_id in org.exchange_reactions:
                    rxn = org.model.reactions.get(rxn_id)
                    if rxn is not None:
                        if met in network.producers.get(org.organism_id, {}):
                            rate = network.producers[org.organism_id][met]
                            rxn.upper_bound = min(rxn.upper_bound, rate)
                        if met in network.consumers.get(org.organism_id, {}):
                            rate = network.consumers[org.organism_id][met]
                            rxn.lower_bound = max(rxn.lower_bound, -rate)


__all__ = [
    "CommunityFBAExtended",
    "CommunityResult",
    "ExchangeNetwork",
    "OrganismModel",
]
