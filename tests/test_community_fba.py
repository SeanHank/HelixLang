"""Tests for community FBA extension (doc/26 Phase E)."""
from __future__ import annotations

from helixlang.plugins.gem.community import (
    CommunityFBAExtended,
    CommunityResult,
    ExchangeNetwork,
    OrganismModel,
)
from helixlang.plugins.runtime.metabolism import MetabolicModel, Reaction


def _make_model(biomass_id: str = "BIOMASS") -> MetabolicModel:
    model = MetabolicModel()
    model.add_reaction(Reaction(
        id="EX_glc", name="EX_glc", stoichiometry={"glc": 1.0},
        lower_bound=-10.0, upper_bound=0.0, subsystem="exchange",
    ))
    model.add_reaction(Reaction(
        id="BIOMASS", name="BIOMASS", stoichiometry={"glc": -1.0},
        lower_bound=0.0, upper_bound=1000.0, subsystem="biomass",
    ))
    model.set_biomass(biomass_id)
    return model


class TestOrganismModel:
    def test_creation(self):
        m = OrganismModel(organism_id="ecoli", model=_make_model())
        assert m.organism_id == "ecoli"
        assert m.growth_rate == 0.0


class TestExchangeNetwork:
    def test_creation(self):
        net = ExchangeNetwork()
        assert len(net.metabolites) == 0


class TestCommunityFBAExtended:
    def test_single_organism(self):
        org = OrganismModel(organism_id="ecoli", model=_make_model())
        community = CommunityFBAExtended(organisms=[org])
        result = community.solve()
        assert isinstance(result, CommunityResult)
        assert result.total_biomass >= 0.0

    def test_two_organisms(self):
        org1 = OrganismModel(organism_id="ecoli", model=_make_model())
        org2 = OrganismModel(organism_id="b_sub", model=_make_model())
        community = CommunityFBAExtended(organisms=[org1, org2])
        result = community.solve()
        assert isinstance(result, CommunityResult)
        assert result.total_biomass >= 0.0
        assert result.iterations >= 1

    def test_convergence(self):
        org = OrganismModel(organism_id="ecoli", model=_make_model())
        community = CommunityFBAExtended(
            organisms=[org], max_iterations=50, tolerance=1e-6
        )
        result = community.solve()
        assert result.converged or result.iterations == 50

    def test_empty_organisms(self):
        community = CommunityFBAExtended(organisms=[])
        result = community.solve()
        assert result.total_biomass == 0.0
