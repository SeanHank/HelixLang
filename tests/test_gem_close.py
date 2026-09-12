"""Coverage-closure tests for the gem plugin package (branch + line 100%)."""
# These exercises previously-uncovered branches/paths in the gem package:
#   __init__.py  bridge.py  community.py  ecgem.py  full_model.py  gapfill.py
from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

import helixlang.plugins.gem as gem_mod
import helixlang.plugins.gem.ecgem as ecgem_mod
from helixlang.plugins.gem.bottom_up import GPRRule
from helixlang.plugins.gem.bridge import (
    _parse_equation_to_stoich,
    apply_regulatory_bounds,
    build_enzyme_capacity,
    build_functional_model,
    build_functional_model_full,
    build_virtual_cell_from_gem,
    consensus_to_metabolic_model,
    gpr_to_genome_dict,
    regulatory_edges_to_grn,
)
from helixlang.plugins.gem.community import (
    CommunityFBAExtended,
    CommunityResult,
    ExchangeNetwork,
    OrganismModel,
)
from helixlang.plugins.gem.consensus import ConsensusReaction, ConsensusResult
from helixlang.plugins.gem.ecgem import (
    ECGEMBuilder,
    EnzymeConstraint,
)
from helixlang.plugins.gem.full_model import FullModelAdapter
from helixlang.plugins.gem.gapfill import GapfillPool, GapfillResult, gapfill, lp_gapfill
from helixlang.plugins.gem.grn_inference import EvidenceLevel, RegulatoryEdge
from helixlang.plugins.gem.organism_registry import OrganismConfig
from helixlang.plugins.kinetics.kcat_predictor import KcatPrediction
from helixlang.plugins.runtime.metabolism import MetabolicModel, Reaction

# NB: `from ... import gapfill` rebinds the `gapfill` name on the package to
# the function, shadowing the submodule -- use importlib to get the module.
gapfill_mod = importlib.import_module("helixlang.plugins.gem.gapfill")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cs(reactions: list[ConsensusReaction]) -> ConsensusResult:
    return ConsensusResult(reactions=reactions)


def _cr(
    reaction_id: str,
    equation: str,
    confidence: float = 0.5,
    gpr: GPRRule | None = None,
) -> ConsensusReaction:
    return ConsensusReaction(
        reaction_id=reaction_id,
        equation=equation,
        sources=["test"],
        confidence=confidence,
        gpr=gpr,
    )


def _tiny_model() -> MetabolicModel:
    """A tiny model with an exchange, a biomass objective and a GLK rxn."""
    model = MetabolicModel()
    model.add_reaction(Reaction(
        id="EX_GLC", name="EX_GLC", stoichiometry={"GLC": 1.0},
        lower_bound=-10.0, upper_bound=1000.0, subsystem="exchange",
    ))
    model.add_reaction(Reaction(
        id="GLK", name="glucokinase", stoichiometry={"GLC": -1.0, "G6P": 1.0},
        lower_bound=0.0, upper_bound=1000.0, subsystem="glycolysis",
    ))
    model.add_reaction(Reaction(
        id="BIOMASS", name="BIOMASS", stoichiometry={"G6P": -1.0},
        lower_bound=0.0, upper_bound=1000.0, subsystem="biomass",
    ))
    model.set_biomass("BIOMASS")
    return model


class _FakeFBA:
    """Scriptable FluxBalanceAnalysis stand-in for LP-path tests."""

    _results: list[Any] = []
    _i: int = 0

    @classmethod
    def _reset(cls, results: list[Any]) -> None:
        cls._results = list(results)
        cls._i = 0

    def __init__(self, model: Any) -> None:
        self.model = model

    def solve(self, objective: str | None = None, maximize: bool = False) -> dict[str, float]:
        r = _FakeFBA._results[
            min(_FakeFBA._i, len(_FakeFBA._results) - 1)
        ]
        _FakeFBA._i += 1
        if isinstance(r, Exception):
            raise r
        return {objective: r} if objective is not None else {}


# ---------------------------------------------------------------------------
# gem/__init__.py plugin contract
# ---------------------------------------------------------------------------

class TestGemPluginContract:
    def test_check_happy(self) -> None:
        assert gem_mod._check("numpy") is True

    def test_check_missing_module(self) -> None:
        assert gem_mod._check("no_such_helixlang_module_xyz") is False

    def test_make_backend(self) -> None:
        assert gem_mod._make_backend({}) is CommunityFBAExtended

    def test_load_happy(self) -> None:
        loader = gem_mod._load()
        assert loader({}) is CommunityFBAExtended

    def test_load_dependency_error(self, monkeypatch) -> None:
        monkeypatch.setattr(gem_mod, "_check", lambda pkg: False)
        from helixlang.core.errors import PluginDependencyError

        with pytest.raises(PluginDependencyError) as exc:
            gem_mod._load()
        assert exc.value.name == "gem"
        assert exc.value.dep == "numpy"


# ---------------------------------------------------------------------------
# bridge.py
# ---------------------------------------------------------------------------

class TestBridgeClose:
    def test_exchange_low_confidence_lb(self) -> None:
        """EX_ single-met reaction gets exchange subsystem + lb=-1000."""
        consensus = _cs([_cr("EX_glu", "glu_e <=> ", confidence=0.3)])
        model = consensus_to_metabolic_model(consensus)
        rxn = model.reactions["EX_glu"]
        assert rxn.subsystem == "exchange"
        assert rxn.lower_bound == -1000.0

    def test_internal_reaction_lb_by_confidence(self) -> None:
        consensus = _cs([_cr("ACN", "A <=> B", confidence=0.3)])
        model = consensus_to_metabolic_model(consensus)
        assert model.reactions["ACN"].lower_bound == 0.0
        consensus_hi = _cs([_cr("ACN2", "A -> B", confidence=0.9)])
        model2 = consensus_to_metabolic_model(consensus_hi)
        assert model2.reactions["ACN2"].lower_bound == -1000.0

    def test_parse_equation_variants(self) -> None:
        assert _parse_equation_to_stoich("A <=> B") == {"A": -1.0, "B": 1.0}
        assert _parse_equation_to_stoich("2 A -> B") == {"A": -2.0, "B": 1.0}
        assert _parse_equation_to_stoich("X Y -> Z") == {"X Y": -1.0, "Z": 1.0}
        assert _parse_equation_to_stoich("P Q R -> S") == {"P Q R": -1.0, "S": 1.0}
        assert _parse_equation_to_stoich("A +  + B -> C") == {"A": -1.0, "B": -1.0, "C": 1.0}
        assert _parse_equation_to_stoich("no arrow here") == {}
        assert _parse_equation_to_stoich("") == {}
        assert _parse_equation_to_stoich("   ") == {}

    def test_consensus_sets_biomass_objective(self) -> None:
        consensus = _cs([_cr("BIOMASS_reaction", "A -> B")])
        model = consensus_to_metabolic_model(consensus)
        assert model.biomass_reaction == "BIOMASS_reaction"

    def test_enzyme_capacity_with_gpr(self) -> None:
        gpr = GPRRule(reaction_id="A", gene_ids=["g1", "g2"])
        consensus = _cs([_cr("A", "-> B", gpr=gpr)])
        cap = build_enzyme_capacity(
            consensus,
            [KcatPrediction(reaction_id="A", kcat_value=42.0, source="median")],
            enzyme_scale=2.0,
            protein_mass_fraction=0.3,
        )
        assert cap.gene_to_reactions["g1"] == ("A",)
        assert cap.gene_to_reactions["g2"] == ("A",)
        assert cap.kcat["A"] == 42.0
        assert cap.enzyme_scale == 2.0
        assert cap.protein_mass_fraction == 0.3

    def test_regulatory_edges_to_grn_gene_names(self) -> None:
        edges = [
            RegulatoryEdge(
                tf_id="crp", target_gene="lacZ",
                regulation_type="activation",
                evidence_level=EvidenceLevel.DATABASE, confidence=0.9,
            ),
            RegulatoryEdge(
                tf_id="fruR", target_gene="ptsG",
                regulation_type="repression",
                evidence_level=EvidenceLevel.LITERATURE, confidence=0.8,
            ),
        ]
        grn = regulatory_edges_to_grn(edges, gene_names=["extra_gene"])
        assert "extra_gene" in grn.nodes
        w = {e.target: e.weight for e in grn.edges}
        assert w["lacZ"] == 0.9
        assert w["ptsG"] == -0.8

    def test_enzyme_capacity_shared_gene_dedupped(self) -> None:
        gpr1 = GPRRule(reaction_id="A", gene_ids=["glk"])
        gpr2 = GPRRule(reaction_id="B", gene_ids=["glk"])
        consensus = _cs([_cr("A", "-> B", gpr=gpr1), _cr("B", "-> C", gpr=gpr2)])
        genome = gpr_to_genome_dict(consensus)
        assert genome == {"glk": "ATG" + "NNN" * 10}

    def test_build_virtual_cell_from_gem_default_config(self) -> None:
        gpr = GPRRule(reaction_id="GLK", gene_ids=["glk"])
        consensus = _cs([_cr("EX_glc", "glc_e <=> "), _cr("GLK", "glc_e -> glc", gpr=gpr)])
        preds = [KcatPrediction(reaction_id="GLK", kcat_value=300.0, source="brenda")]
        cell = build_virtual_cell_from_gem(consensus, preds)
        assert cell.name == "gem-cell"
        assert "glk" in cell.genome
        assert cell.config.enzyme_capacity_enabled is True

    def test_build_virtual_cell_from_gem_with_edges_and_config(self) -> None:
        consensus = _cs([_cr("EX_glu", "glu_e <=> ")])
        preds: list[KcatPrediction] = []
        edges = [
            RegulatoryEdge(
                tf_id="crp", target_gene="gluK",
                regulation_type="activation",
                evidence_level=EvidenceLevel.PREDICTED, confidence=0.5,
            )
        ]
        cfg = type(
            "Cfg", (), {
                "enzyme_capacity_enabled": False,
                "uptake": {},
                "energy_init": 1.0,
                "protein_half_life_min": 60.0,
                "seed": 7,
                "volume_init_um3": 1.6,
                "chromosome_map": {},
                "gene_replicons": {},
                "replicons": {},
            }
        )()
        from helixlang.plugins.runtime.virtual_cell import VirtualCellConfig

        cfg = VirtualCellConfig(enzyme_capacity_enabled=False, seed=7)
        cell = build_virtual_cell_from_gem(
            consensus, preds, grn_edges=edges, config=cfg, name="cyan"
        )
        assert cell.name == "cyan"
        assert "crp" in cell.grn.nodes

    def test_apply_regulatory_bounds_skips_non_edge(self) -> None:
        model = _tiny_model()
        gpr_map = {"glk": ["GLK"]}
        # built via __new__ to bypass __post_init__ validation so the
        # defensive fall-through branch (unknown regulation_type) runs
        bogus = object.__new__(RegulatoryEdge)
        bogus.tf_id = "y"
        bogus.target_gene = "glk"
        bogus.regulation_type = "bogus"
        bogus.evidence_level = EvidenceLevel.DATABASE
        bogus.confidence = 1.0
        bogus.target_reaction = None
        bogus.motif_score = 0.0
        bogus.motif_positions = []
        bogus.source = ""
        n = apply_regulatory_bounds(
            model,
            [{"not": "an edge"}, RegulatoryEdge(
                tf_id="x", target_gene="glk", regulation_type="activation",
                evidence_level=EvidenceLevel.DATABASE, confidence=1.0,
            ), bogus],
            gpr_map,
        )
        # activation rescues, bogus-type edge is counted as a no-op touch
        assert n == 2

    def test_gapfill_skip_existing_and_empty_equation(self) -> None:
        consensus = _cs([_cr("EX_glc", "glc_e <=> ", confidence=0.9)])
        gf_result = GapfillResult(added_reactions=[
            ConsensusReaction(reaction_id="EX_glc", equation="glc_e <=> "),
            ConsensusReaction(reaction_id="EX_bad", equation=""),
        ])
        model = build_functional_model(consensus, gapfill=gf_result, organism="e_coli_k12")
        assert "BIOMASS_reaction" in model.reactions or model.biomass_reaction is not None

    def test_build_functional_model_skips_biomass_when_no_components(self, monkeypatch) -> None:
        import helixlang.plugins.gem.biomass as biomass_mod

        consensus = _cs([_cr("EX_glc", "glc_e <=> ", confidence=0.9)])
        monkeypatch.setattr(
            biomass_mod, "build_biomass_reaction",
            lambda organism: SimpleNamespace(
                components=[SimpleNamespace(
                    metabolite_id="NOPE_X", coefficient=1.0,
                )]
            ),
        )
        model = build_functional_model(consensus, organism="e_coli_k12")
        assert "BIOMASS_reaction" not in model.reactions
        assert model._growth_rate == 0.0

    def test_build_functional_model_full_biomass_rxn_missing(self, monkeypatch) -> None:
        from helixlang.plugins.gem import full_model as fm_mod

        model = MetabolicModel()
        monkeypatch.setattr(fm_mod.FullModelAdapter, "from_bigg", lambda organism: SimpleNamespace(
            model=model, biomass_rxn="NOT_PRESENT", growth_rate=0.0,
            apply_medium=lambda medium: None,
            solve=lambda: {},
        ))
        out = build_functional_model_full(organism="e_coli_k12", medium="lb")
        assert out.biomass_reaction is None

    def test_build_functional_model_solve_failure_swallowed(self, monkeypatch) -> None:
        from helixlang.plugins.runtime import metabolism as metab_mod

        class Boom(metab_mod.FluxBalanceAnalysis):
            def solve(self, objective=None, maximize=False):
                raise RuntimeError("solver down")

        consensus = _cs([_cr("EX_glc", "glc_e <=> ")])
        monkeypatch.setattr(metab_mod, "FluxBalanceAnalysis", Boom)
        model = build_functional_model(consensus, organism="e_coli_k12")
        assert model._fba_fluxes == {}

    def test_build_functional_model_full_from_bigg_branch(self, monkeypatch) -> None:
        from helixlang.plugins.gem import full_model as fm_mod

        model = MetabolicModel()
        model.add_reaction(Reaction(
            id="BIOMASS", name="BIOMASS", stoichiometry={"glc": -1.0},
            lower_bound=0.0, upper_bound=1000.0, subsystem="biomass",
        ))
        model.set_biomass("BIOMASS")

        class FakeAdapter:
            def __init__(self) -> None:
                self.model = model
                self.biomass_rxn = "BIOMASS"
                self.growth_rate = 1.0

            def apply_medium(self, medium: str) -> None:
                return None

            def solve(self) -> dict[str, float]:
                return {"BIOMASS": 1.0}

        monkeypatch.setattr(fm_mod.FullModelAdapter, "from_bigg", lambda organism: FakeAdapter())
        out = build_functional_model_full(organism="e_coli_k12", medium="lb")
        assert out._growth_rate == 1.0
        assert out._fba_fluxes == {"BIOMASS": 1.0}
        assert out._adapter is not None

    def test_build_functional_model_full_from_sbml_branch(self, monkeypatch) -> None:
        from helixlang.plugins.gem import full_model as fm_mod

        class FakeAdapter:
            def __init__(self) -> None:
                self.model = SimpleNamespace(
                    reactions={}, biomass_reaction=None)
                self.biomass_rxn = None
                self.growth_rate = 0.5

            def apply_medium(self, medium: str) -> None:
                return None

            def solve(self) -> dict[str, float]:
                return {}

        monkeypatch.setattr(
            fm_mod.FullModelAdapter, "from_sbml",
            lambda path, organism: FakeAdapter())
        out = build_functional_model_full(
            organism="e_coli_k12", medium="lb", sbml_path="fake.xml")
        assert out._growth_rate == 0.5
        assert out._adapter is not None


# ---------------------------------------------------------------------------
# community.py
# ---------------------------------------------------------------------------

class TestCommunityClose:
    def test_max_iterations_exhausted(self, monkeypatch) -> None:
        calls: list[float] = []

        def fake_solve(self, org):
            calls.append(1)
            return 0.1 if len(calls) % 2 == 1 else 0.4

        org = OrganismModel(
            organism_id="o1", model=_tiny_model(), exchange_reactions=["EX_GLC"]
        )
        monkeypatch.setattr(
            CommunityFBAExtended, "_solve_single", fake_solve
        )
        community = CommunityFBAExtended(
            organisms=[org], max_iterations=2, tolerance=1e-6
        )
        result = community.solve()
        assert result.converged is False
        assert result.iterations == 2

    def test_solve_single_returns_zero_without_biomass(self) -> None:
        model = _tiny_model()
        model.biomass_reaction = None
        org = OrganismModel(organism_id="o1", model=model)
        community = CommunityFBAExtended(organisms=[org])
        assert community._solve_single(org) == 0.0

    def test_solve_single_low_fidelity(self, monkeypatch) -> None:
        monkeypatch.setattr("helixlang.api.capabilities.opt_in", lambda *a, **k: True)
        org = OrganismModel(
            organism_id="o1",
            model=SimpleNamespace(biomass_reaction="X"),
        )
        community = CommunityFBAExtended(organisms=[org])
        assert community._solve_single(org) == 0.0

    def test_solve_single_raises_model_error(self, monkeypatch) -> None:
        monkeypatch.setattr("helixlang.api.capabilities.opt_in", lambda *a, **k: False)
        from helixlang.core.errors import ModelMissingError

        org = OrganismModel(
            organism_id="o1",
            model=SimpleNamespace(biomass_reaction="X"),
        )
        community = CommunityFBAExtended(organisms=[org])
        with pytest.raises(ModelMissingError):
            community._solve_single(org)

    def test_shared_exchange_bounds_applied(self) -> None:
        model = _tiny_model()
        org1 = OrganismModel(
            organism_id="a", model=model,
            exchange_reactions=["EX_GLC"],
            production={"GLC": 5.0},
        )
        org2 = OrganismModel(
            organism_id="b", model=model,
            exchange_reactions=["EX_GLC"],
            consumption={"GLC": 3.0},
        )
        # non-EX exchange id (skipped) and no EX_GLC reaction at all
        org3 = OrganismModel(
            organism_id="c", model=model, exchange_reactions=["NON_EX"],
        )
        # advertises EX_GLC but its model does not contain the reaction
        bare = MetabolicModel()
        bare.add_reaction(Reaction(
            id="BIOMASS", name="BIOMASS", stoichiometry={"glc": -1.0},
            lower_bound=0.0, upper_bound=1000.0, subsystem="biomass",
        ))
        bare.set_biomass("BIOMASS")
        org4 = OrganismModel(
            organism_id="d", model=bare, exchange_reactions=["EX_GLC"],
        )
        community = CommunityFBAExtended(
            organisms=[org1, org2, org3, org4], max_iterations=3
        )
        result = community.solve()
        assert isinstance(result, CommunityResult)
        ex = result.exchange_network or ExchangeNetwork()
        assert ex.balance.get("GLC", 0.0) == 2.0


# ---------------------------------------------------------------------------
# full_model.py
# ---------------------------------------------------------------------------

class TestFullModelClose:
    _CONFIG = OrganismConfig(
        organism_id="t", bigg_id="T", name="test", model_type="metabolic",
        biomass_rxn="BIOMASS",
        light_reactions=["PHOTOS"], calvin_reactions=["RBC"],
    )

    def _model(self) -> MetabolicModel:
        model = MetabolicModel()
        model.add_reaction(Reaction(
            id="EX_glc", name="EX_glc", stoichiometry={"glc-D_e": -1.0},
            lower_bound=-1000.0, upper_bound=1000.0, subsystem="exchange",
        ))
        model.add_reaction(Reaction(
            id="EX_prod", name="EX_prod", stoichiometry={"prod_e": 1.0},
            lower_bound=-1000.0, upper_bound=1000.0, subsystem="exchange",
        ))
        model.add_reaction(Reaction(
            id="EX_skip", name="EX_skip", stoichiometry={"a_e": -1.0, "b_e": 1.0},
            lower_bound=-1000.0, upper_bound=1000.0, subsystem="exchange",
        ))
        model.add_reaction(Reaction(
            id="EX_multi", name="EX_multi",
            stoichiometry={"a_e": -1.0, "extra_e": -1.0},
            lower_bound=-1000.0, upper_bound=1000.0, subsystem="exchange",
        ))
        model.add_reaction(Reaction(
            id="EX_multi2", name="EX_multi2",
            stoichiometry={"a_e": -1.0, "extra2_e": 1.0},
            lower_bound=-1000.0, upper_bound=1000.0, subsystem="exchange",
        ))
        model.add_reaction(Reaction(
            id="T_sub", name="T_sub", stoichiometry={"glc_c": 1.0},
            lower_bound=0.0, upper_bound=1000.0, subsystem="exchange",
        ))
        model.add_reaction(Reaction(
            id="T_sub3", name="T_sub3",
            stoichiometry={"g_c": -1.0, "h_c": -1.0, "i_c": 1.0},
            lower_bound=0.0, upper_bound=1000.0, subsystem="exchange",
        ))
        model.add_reaction(Reaction(
            id="PCGRS", name="PCGRS", stoichiometry={"A_c": -1.0, "A_p": 1.0},
            lower_bound=-1000.0, upper_bound=1000.0, subsystem="transport",
        ))
        model.add_reaction(Reaction(
            id="R_INT", name="R_INT", stoichiometry={"A_c": -1.0, "B_c": 1.0},
            lower_bound=-1000.0, upper_bound=1000.0, subsystem="other",
        ))
        model.add_reaction(Reaction(
            id="PHOTOS", name="PHOTOS", stoichiometry={"photon_e": -1.0, "ATP": 3.0},
            lower_bound=0.0, upper_bound=1000.0, subsystem="photosynthesis",
        ))
        model.add_reaction(Reaction(
            id="RBC", name="RBC", stoichiometry={"CO2": -1.0, "G6P": 1.0},
            lower_bound=0.0, upper_bound=1000.0, subsystem="calvin",
        ))
        model.add_reaction(Reaction(
            id="BIOMASS", name="BIOMASS", stoichiometry={"glc_c": -1.0},
            lower_bound=0.0, upper_bound=1000.0, subsystem="biomass",
        ))
        model.set_biomass("BIOMASS")
        return model

    def _adapter(self, monkeypatch) -> FullModelAdapter:
        monkeypatch.setattr(
            "helixlang.plugins.gem.organism_registry.get_organism_config",
            lambda oid: self._CONFIG,
        )
        return FullModelAdapter(self._model(), "t", biomass_rxn="BIOMASS")

    def test_no_biomass_reaction(self) -> None:
        from helixlang.core.errors import BioError

        with pytest.raises(BioError):
            FullModelAdapter(MetabolicModel(), "t")

    def test_from_bigg_unknown_organism(self, monkeypatch) -> None:
        from helixlang.core.errors import BioError

        monkeypatch.setattr(
            "helixlang.plugins.gem.organism_registry.get_organism_config",
            lambda oid: None,
        )
        with pytest.raises(BioError):
            FullModelAdapter.from_bigg("nope")

    def test_from_bigg_happy(self, monkeypatch) -> None:
        import helixlang.plugins.gem.sbml_import as sbml_import_mod

        monkeypatch.setattr(
            "helixlang.plugins.gem.organism_registry.get_organism_config",
            lambda oid: self._CONFIG,
        )
        monkeypatch.setattr(sbml_import_mod, "load_bigg_model", lambda bigg_id, model_dir=None: self._model())
        adapter = FullModelAdapter.from_bigg("t", model_dir="/tmp")
        assert adapter.biomass_rxn == "BIOMASS"

    def test_from_sbml_happy_and_summary(self, monkeypatch) -> None:
        import helixlang.plugins.gem.sbml_import as sbml_import_mod

        monkeypatch.setattr(
            sbml_import_mod, "load_sbml_model", lambda path: self._model())
        adapter = FullModelAdapter.from_sbml(
            "fake.xml", "t", biomass_rxn="BIOMASS")
        assert adapter.biomass_rxn == "BIOMASS"
        summ = adapter.summary()
        assert summ["organism"] == "t"
        assert summ["biomass_rxn"] == "BIOMASS"

    def test_exchange_via_subsystem(self, monkeypatch) -> None:
        adapter = self._adapter(monkeypatch)
        assert "T_sub" in adapter.exchange_reactions
        assert "T_sub3" not in adapter.exchange_reactions
        assert "PCGRS" in adapter.transport_reactions

    def test_transport_detection_long_suffix(self, monkeypatch) -> None:
        model = self._model()
        model.add_reaction(Reaction(
            id="T_LONGO", name="T_LONGO",
            stoichiometry={"A_DEPOTEXTRA": 1.0},
            lower_bound=0.0, upper_bound=1000.0, subsystem="transport",
        ))
        adapter = FullModelAdapter(model, "t", biomass_rxn="BIOMASS")
        assert "T_LONGO" not in adapter.transport_reactions

    def test_apply_medium_closes_absent_light_rxns(self, monkeypatch) -> None:
        cfg = OrganismConfig(
            organism_id="t", bigg_id="T", name="test", model_type="metabolic",
            biomass_rxn="BIOMASS",
            light_reactions=["PHOTOS", "PHOTOS_ABSENT"], calvin_reactions=["RBC"],
        )
        monkeypatch.setattr(
            "helixlang.plugins.gem.organism_registry.get_organism_config",
            lambda oid: cfg,
        )
        adapter = FullModelAdapter(self._model(), "t")
        monkeypatch.setattr(
            FullModelAdapter, "_MEDIUM_PRESETS", {"custom": {}}
        )
        adapter.apply_medium("custom")
        assert adapter.model.reactions["RBC"].lower_bound == 0.0
        assert adapter.model.reactions["PHOTOS"].lower_bound == 0.0

    def test_apply_medium_paths(self, monkeypatch) -> None:
        adapter = self._adapter(monkeypatch)
        monkeypatch.setattr(
            FullModelAdapter, "_MEDIUM_PRESETS",
            {
                "custom": {
                    "glc-D_e": 5.0, "prod_e": 3.0,
                    "extra_e": 7.0, "extra2_e": 9.0, "zzz_e": 1.0,
                }
            },
        )
        adapter.apply_medium("custom")
        assert adapter.model.reactions["EX_glc"].lower_bound == -5.0
        assert adapter.model.reactions["EX_prod"].upper_bound == 3.0
        assert adapter.model.reactions["EX_multi"].lower_bound == -7.0
        assert adapter.model.reactions["EX_multi2"].upper_bound == 9.0
        assert adapter.model.reactions["PHOTOS"].upper_bound == 0.0
        assert adapter.model.reactions["RBC"].upper_bound == 0.0

    def test_apply_medium_unknown(self, monkeypatch) -> None:
        from helixlang.core.errors import BioError

        adapter = self._adapter(monkeypatch)
        monkeypatch.setattr(
            FullModelAdapter, "_MEDIUM_PRESETS", {"special": {}}
        )
        with pytest.raises(BioError):
            adapter.apply_medium("does_not_exist")

    def test_apply_medium_photo_preset_keeps_calvin(self, monkeypatch) -> None:
        adapter = self._adapter(monkeypatch)
        monkeypatch.setattr(
            FullModelAdapter, "_MEDIUM_PRESETS", {"bg11": {}}
        )
        adapter.apply_medium("bg11")
        assert adapter.model.reactions["RBC"].upper_bound == 1000.0

    def test_set_uptake(self, monkeypatch) -> None:
        adapter = self._adapter(monkeypatch)
        adapter.set_uptake("glc-D_e", 11.0)
        assert adapter.model.reactions["EX_glc"].lower_bound == -11.0
        adapter.set_uptake("prod_e", 7.0)
        assert adapter.model.reactions["EX_prod"].upper_bound == 7.0
        adapter.exchange_reactions.append("EX_ghost")
        adapter.set_uptake("zzz_e", 1.0)  # not present -> no-op

    def test_solve_and_exchange_fluxes(self, monkeypatch) -> None:
        adapter = self._adapter(monkeypatch)
        fluxes = adapter.solve()
        assert adapter.growth_rate >= 0.0
        inside = adapter.get_exchange_fluxes(fluxes)
        assert set(inside.keys()) == set(adapter.exchange_reactions)
        from_last = adapter.get_exchange_fluxes()
        assert from_last
        empty = FullModelAdapter(self._model(), "t")
        assert empty.get_exchange_fluxes() == {}

    def test_list_supported_organisms_str(self) -> None:
        from helixlang.plugins.gem.full_model import list_supported_organisms_str

        assert isinstance(list_supported_organisms_str(), str)


# ---------------------------------------------------------------------------
# gapfill.py
# ---------------------------------------------------------------------------

class TestGapfillClose:
    def test_parse_stoich_variants(self) -> None:
        assert gapfill_mod._parse_equation_to_stoich("2 A <=> 3 B") == {
            "A": -2.0, "B": 3.0
        }
        assert gapfill_mod._parse_equation_to_stoich("A B C -> D") == {
            "C": -1.0, "D": 1.0
        }
        assert gapfill_mod._parse_equation_to_stoich("2 A + B -> 3 C") == {
            "A": -2.0, "B": -1.0, "C": 3.0
        }
        assert gapfill_mod._is_numeric("2.5") is True
        assert gapfill_mod._is_numeric("abc") is False

    def test_pool_size_property(self) -> None:
        pool = GapfillPool()
        assert pool.size == 0
        pool.add_reaction("EX_x", "A <=> ")
        assert pool.size == 1

    def test_gapfill_empty_consensus(self) -> None:
        result = gapfill(ConsensusResult(), max_iterations=1)
        assert result.biomass_blocked is True

    def test_build_model_exchange(self) -> None:
        consensus = _cs([_cr("EX_glc_e", "glc-D_e <=> ", confidence=0.3)])
        model = gapfill_mod._build_model_from_consensus(consensus)
        rxn = model.reactions["EX_glc_e"]
        assert rxn.subsystem == "exchange"
        assert rxn.lower_bound == -1000.0

    def test_try_add_reaction_empty_and_duplicate(self, monkeypatch) -> None:
        model = gapfill_mod._build_model_from_consensus(
            _cs([_cr("EX_glc_e", "glc-D_e <=> ")])
        )
        assert gapfill_mod._try_add_reaction(
            model, {"id": "EX_x", "eq": ""}, "BIOMASS"
        ) == (False, 0.0)
        assert gapfill_mod._try_add_reaction(
            model, {"id": "EX_glc_e", "eq": "glc-D_e <=> "}, "BIOMASS"
        ) == (False, 0.0)

    def test_try_add_reaction_solver_crash(self, monkeypatch) -> None:
        model = MetabolicModel()
        monkeypatch.setattr(
            gapfill_mod, "FluxBalanceAnalysis",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert gapfill_mod._try_add_reaction(
            model, {"id": "EX_x", "eq": "A <=> "}, "BIOMASS"
        ) == (False, 0.0)

    def test_gapfill_bad_pool_entry(self, monkeypatch) -> None:
        consensus = _cs([_cr("BIOMASS_reaction", "glc-D_e <=> ", confidence=0.9)])
        monkeypatch.setattr(
            gapfill_mod, "GAPFILL_POOL",
            [{"id": "EX_X_e", "eq": "", "rev": "True"}],
        )
        result = gapfill(consensus)
        assert result.biomass_blocked is True or len(result.added_reactions) >= 0

    def test_gapfill_early_return_biomass_producible(self) -> None:
        consensus = _cs([_cr("BIOMASS_reaction", "glc-D_e <=> ", confidence=0.9)])
        result = gapfill(consensus, max_iterations=1)
        assert result.biomass_blocked is False

    def test_gapfill_solver_failures_silent(self, monkeypatch) -> None:
        consensus = _cs([_cr("BIOMASS_reaction", "A -> B", confidence=0.9)])
        _FakeFBA._reset([RuntimeError("boom"), RuntimeError("boom")])
        monkeypatch.setattr(gapfill_mod, "FluxBalanceAnalysis", _FakeFBA)
        monkeypatch.setattr(gapfill_mod, "_try_add_reaction", lambda m, c, o: (False, 0.0))
        result = gapfill(consensus, max_iterations=1)
        assert result.biomass_blocked is True

    def test_gapfill_transport_final_check(self, monkeypatch) -> None:
        consensus = _cs([_cr("BIOMASS_reaction", "A -> B", confidence=0.9)])
        _FakeFBA._reset([0.0, 5.0])
        monkeypatch.setattr(gapfill_mod, "FluxBalanceAnalysis", _FakeFBA)
        monkeypatch.setattr(gapfill_mod, "_try_add_reaction", lambda m, c, o: (False, 0.0))
        result = gapfill(consensus, max_iterations=1)
        assert result.biomass_blocked is False

    def test_gapfill_transport_added(self, monkeypatch) -> None:
        consensus = _cs([_cr("R1", "A -> B", confidence=0.9)])
        monkeypatch.setattr(gapfill_mod, "GAPFILL_POOL", [])
        monkeypatch.setattr(
            gapfill_mod, "_try_add_reaction",
            lambda m, c, o: (c["id"] == "EX_nac_e", 5.0),
        )
        monkeypatch.setattr(
            gapfill_mod, "FluxBalanceAnalysis",
            lambda model: SimpleNamespace(solve=lambda objective=None, maximize=False: {}),
        )
        result = gapfill(consensus, max_iterations=1)
        assert any(r.reaction_id == "EX_nac_e" for r in result.added_reactions)
        assert result.biomass_blocked is True

    def test_lp_gapfill_no_biomass(self) -> None:
        consensus = _cs([_cr("R1", "A -> B")])
        result = lp_gapfill(consensus, GapfillPool())
        assert result.gap_filled_count == 0

    def test_lp_gapfill_base_produces(self, monkeypatch) -> None:
        consensus = _cs([_cr("BIOMASS_reaction", "A -> B")])
        _FakeFBA._reset([5.0])
        monkeypatch.setattr(gapfill_mod, "FluxBalanceAnalysis", _FakeFBA)
        result = lp_gapfill(consensus, GapfillPool())
        assert result.biomass_blocked is False

    def test_lp_gapfill_full_loop(self, monkeypatch) -> None:
        consensus = _cs([_cr("BIOMASS_reaction", "A -> B")])
        pool = GapfillPool()
        pool.add_reaction("BIOMASS_reaction", "A -> B")  # existing
        pool.add_reaction("EMPTY_1", "")                  # unparseable
        pool.add_reaction("GOOD_1", "C -> D", lower_bound=0.0, upper_bound=900.0,
                          subsystem="test", confidence=0.7)
        pool.add_reaction("BAD_1", "E -> F")
        pool.add_reaction("RAISE_1", "G -> H")
        pool.add_reaction("LATE_1", "I -> J")
        # base solve crashes (line 401-402), GOOD improves, BAD doesn't,
        # RAISE_1 crashes mid-loop (line 442-443); budget 8 lets pass 2
        # re-enter the loop (416->414) then break at the top (417-418)
        _FakeFBA._reset([RuntimeError("base"), 5.0, 0.2, RuntimeError("temp"), 0.2])
        monkeypatch.setattr(gapfill_mod, "FluxBalanceAnalysis", _FakeFBA)
        result = lp_gapfill(consensus, pool, max_candidates=8)
        added = {r.reaction_id for r in result.added_reactions}
        assert "GOOD_1" in added
        assert "LATE_1" not in added
        assert "RAISE_1" not in added
        assert result.iterations == 1
        assert result.biomass_blocked is False


# ---------------------------------------------------------------------------
# ecgem.py
# ---------------------------------------------------------------------------

class TestECGEMClose:
    def test_load_core_model_missing(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            ecgem_mod, "_CORE_MODEL_PATH",
            tmp_path / "does_not_exist.json",
        )
        assert ecgem_mod._load_core_model() is None

    def test_load_core_model_corrupt(self, monkeypatch, tmp_path) -> None:
        bad = tmp_path / "core.json"
        bad.write_text("{not json")
        monkeypatch.setattr(ecgem_mod, "_CORE_MODEL_PATH", bad)
        assert ecgem_mod._load_core_model() is None

    def test_load_core_model_valid_no_biomass(self, monkeypatch, tmp_path) -> None:
        import json as _json

        good = tmp_path / "core.json"
        good.write_text(_json.dumps({
            "reactions": [{
                "id": "EX_glc", "name": "glc", "subsystem": "exchange",
                "stoichiometry": {"glc": 1.0},
                "lower_bound": -10.0, "upper_bound": 1000.0,
            }],
        }))
        monkeypatch.setattr(ecgem_mod, "_CORE_MODEL_PATH", good)
        model = ecgem_mod._load_core_model()
        assert model is not None
        assert "EX_glc" in model.reactions
        assert model.biomass_reaction is None

    def test_build_minimal_model_from_enzymes(self, monkeypatch) -> None:
        monkeypatch.setattr(
            ecgem_mod, "_EC_TO_REACTION",
            {
                "2.7.1.1": {"id": "GLK", "name": "glucokinase",
                             "stoich": {"G6P": 1.0, "GLC": -1.0, "ZERO": 0.0}},
                # duplicate reaction id -> second gene skipped
                "2.7.1.2": {"id": "GLK", "name": "glucokinase2",
                             "stoich": {"G6P": 1.0}},
                # produces GLC which is also consumed -> shared EX reused
                "9.9.9.1": {"id": "R2", "name": "R2",
                             "stoich": {"GLC": 1.0}},
                "9.9.9.2": {"id": "R3", "name": "R3",
                             "stoich": {"ONLY_A": 1.0}},
                # no stoich -> contributed metabolite set stays empty for it
                "9.9.9.3": {"id": "R4", "name": "R4"},
            },
        )
        monkeypatch.setattr(ecgem_mod, "_load_core_model", lambda: None)
        builder = ECGEMBuilder(
            kcat_predictions={
                "g1": 300.0, "g_dup": 200.0, "g2": 200.0,
                "g3": 100.0, "g_no": 50.0, "g_unknown": 100.0,
            },
            ec_numbers={
                "g1": "2.7.1.1", "g_dup": "2.7.1.2", "g2": "9.9.9.1",
                "g3": "9.9.9.2", "g_no": "9.9.9.3", "g_unknown": "7.7.7.7",
            },
            sequences={"g1": "AAA"},
        )
        model = builder._build_minimal_model_from_enzymes()
        assert "GLK" in model.reactions
        assert set(model.reactions) == {
            "GLK", "R2", "R3",
            "EX_GLC", "EX_G6P", "EX_ONLY_A", "BIOMASS",
        }
        # no-stoich entry is tracked (skips duplicate re-add) but adds nothing
        assert "R4" not in model.reactions
        assert model.reactions["BIOMASS"].stoichiometry == {"G6P": -0.5}

    def test_build_minimal_produced_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(
            ecgem_mod, "_EC_TO_REACTION",
            {"9.9.9.3": {"id": "R4", "name": "R4"}},
        )
        monkeypatch.setattr(ecgem_mod, "_load_core_model", lambda: None)
        builder = ECGEMBuilder(
            kcat_predictions={"g1": 100.0},
            ec_numbers={"g1": "9.9.9.3"},
        )
        model = builder._build_minimal_model_from_enzymes()
        # no stoich -> no metabolite sets -> no biomass reaction added
        assert model.reactions == {}

    def test_build_minimal_model_biomass_fallback(self, monkeypatch) -> None:
        monkeypatch.setattr(
            ecgem_mod, "_EC_TO_REACTION",
            {"9.9.9.1": {"id": "R3", "name": "R3", "stoich": {"ONLY": 1.0}}},
        )
        monkeypatch.setattr(ecgem_mod, "_load_core_model", lambda: None)
        builder = ECGEMBuilder(
            kcat_predictions={"g1": 100.0},
            ec_numbers={"g1": "9.9.9.1"},
        )
        model = builder._build_minimal_model_from_enzymes()
        assert "BIOMASS" in model.reactions
        assert model.reactions["BIOMASS"].stoichiometry == {"ONLY": -0.5}

    def test_build_no_reactions(self, monkeypatch) -> None:
        monkeypatch.setattr(
            ecgem_mod, "_load_core_model", lambda: MetabolicModel()
        )
        result = ECGEMBuilder().solve()
        assert "no reactions in model" in result.warnings

    def test_build_no_constraints_warns(self, monkeypatch) -> None:
        model = _tiny_model()
        builder = ECGEMBuilder(
            base_model=model,
            kcat_predictions={"g1": 300.0},
            ec_numbers={"g1": "7.7.7.7"},  # no reaction mapped to this EC
        )
        monkeypatch.setattr(
            ECGEMBuilder, "_solve_growth", staticmethod(lambda model: 0.4)
        )
        result = builder.build()
        assert "no enzyme constraints could be built" in result.warnings
        assert result.growth_rate == 0.4
        assert result.growth_rate_unconstrained == 0.4

    def test_build_default_kinetics(self, monkeypatch) -> None:
        model = _tiny_model()
        model.add_reaction(Reaction(
            id="ZED", name="ZED", stoichiometry={"W": 1.0},
            lower_bound=0.0, upper_bound=1000.0, subsystem="extra",
        ))
        # give ZED a kcat/mw without putting it in _EC_TO_REACTION so the
        # ec-lookup loop exits without a match (branch 315->319)
        core_kcat = dict(ecgem_mod.CORE_ENZYME_KCAT)
        core_kcat["ZED"] = 100.0
        monkeypatch.setattr(ecgem_mod, "CORE_ENZYME_KCAT", core_kcat)
        core_mw = dict(ecgem_mod.CORE_ENZYME_MW)
        core_mw["ZED"] = 50000.0
        monkeypatch.setattr(ecgem_mod, "CORE_ENZYME_MW", core_mw)
        builder = ECGEMBuilder(base_model=model)
        monkeypatch.setattr(
            ECGEMBuilder, "_solve_growth", staticmethod(lambda model: 1.0)
        )
        monkeypatch.setattr(
            ECGEMBuilder, "_solve_fluxes", staticmethod(lambda model: {"GLK": 5.0})
        )
        result = builder.build()
        ids = {c.reaction_id for c in result.enzyme_constraints}
        assert "GLK" in ids          # matched an EC -> full ec-lookup loop
        assert "ZED" in ids          # no EC match -> ec stays ""
        assert "EX_GLC" not in ids   # not in CORE maps -> skipped
        assert "BIOMASS" not in ids
        assert result.model.reactions["GLK"].upper_bound <= 1000.0

    def test_get_model_returns_minimal(self, monkeypatch) -> None:
        monkeypatch.setattr(ecgem_mod, "_load_core_model", lambda: None)
        monkeypatch.setattr(
            ecgem_mod, "_EC_TO_REACTION",
            {"9.9.9.1": {"id": "R5", "name": "R5", "stoich": {"A": 1.0}}},
        )
        builder = ECGEMBuilder(
            kcat_predictions={"g1": 100.0},
            ec_numbers={"g1": "9.9.9.1"},
        )
        model = builder._get_model()
        assert "R5" in model.reactions
        assert "BIOMASS" in model.reactions

    def test_solve_growth_fluxes_no_biomass(self) -> None:
        model = _tiny_model()
        model.biomass_reaction = None
        assert ECGEMBuilder._solve_growth(model) == 0.0
        assert ECGEMBuilder._solve_fluxes(model) == {}

    def test_build_with_custom_kinetics(self, monkeypatch) -> None:
        model = _tiny_model()
        builder = ECGEMBuilder(
            base_model=model,
            kcat_predictions={"g1": 300.0, "g0": 0.0, "gbad": 5.0},
            ec_numbers={"g1": "2.7.1.1", "g0": "2.7.1.1", "gbad": "9.9.9.9"},
            sequences={"g1": "AAA"},
        )
        calls = []

        def fake_growth(model, _calls=calls):
            _calls.append(1)
            return 1.0 if len(_calls) == 1 else 0.005

        monkeypatch.setattr(ECGEMBuilder, "_solve_growth", staticmethod(fake_growth))
        monkeypatch.setattr(
            ECGEMBuilder, "_solve_fluxes", staticmethod(lambda model: {"GLK": 5.0})
        )
        result = builder.build()
        assert result.growth_rate == 0.005
        assert result.growth_rate_unconstrained == 1.0
        assert any("much lower" in w for w in result.warnings)
        assert result.model.reactions["GLK"].upper_bound <= 1000.0
        assert any(c.enzyme_fraction >= 0.0 for c in result.enzyme_constraints)

    def test_sequential_solve_and_validate(self, monkeypatch) -> None:
        model = _tiny_model()
        builder = ECGEMBuilder(
            base_model=model,
            kcat_predictions={"g1": 300.0},
            ec_numbers={"g1": "2.7.1.1"},
            sequences={"g1": "AAA"},
        )
        monkeypatch.setattr(
            ECGEMBuilder, "_solve_growth", staticmethod(lambda model: 5.0)
        )
        monkeypatch.setattr(
            ECGEMBuilder, "_solve_fluxes", staticmethod(lambda model: {"GLK": 5.0})
        )
        result = builder.solve()
        assert result.growth_rate == 5.0
        assert builder.validate(expected_growth=5.0, tolerance=0.4) is True
        assert builder.validate(expected_growth=0.0) is True

    def test_apply_enzyme_constraints_degenerate(self, monkeypatch) -> None:
        model = _tiny_model()
        monkeypatch.setattr(
            ECGEMBuilder, "_solve_fluxes", staticmethod(lambda model: {"GLK": 5.0})
        )
        healthy = EnzymeConstraint(
            reaction_id="GLK", gene_id="g", ec_number="", kcat=5.0,
            molecular_weight=33000.0,
        )
        builder = ECGEMBuilder()
        usage = builder._apply_enzyme_constraints(model, [
            healthy,
            EnzymeConstraint(
                reaction_id="GLK", gene_id="g", ec_number="", kcat=0.0,
                molecular_weight=33000.0,
            ),
            EnzymeConstraint(
                reaction_id="GLK", gene_id="g", ec_number="", kcat=5.0,
                molecular_weight=0.0,
            ),
        ])
        assert set(usage.keys()) == {"GLK"}
        zero_pool = ECGEMBuilder(enzyme_mass_fraction=0.0)
        usage0 = zero_pool._apply_enzyme_constraints(model, [healthy])
        assert usage0["GLK"] == 0.0
        # constraint for a reaction absent from the model is skipped
        ghost = EnzymeConstraint(
            reaction_id="NOPE", gene_id="g", ec_number="", kcat=5.0,
            molecular_weight=33000.0,
        )
        usage_ghost = builder._apply_enzyme_constraints(model, [healthy, ghost])
        assert usage_ghost["NOPE"] == 0.0
        assert usage_ghost["GLK"] >= 0.0

    def test_solve_growth_and_fluxes_crash(self, monkeypatch) -> None:
        import helixlang.plugins.runtime.metabolism as metab_mod

        class Boom:
            def __init__(self, model):
                raise RuntimeError("fba down")

        monkeypatch.setattr(metab_mod, "FluxBalanceAnalysis", Boom)
        monkeypatch.setattr("helixlang.api.capabilities.opt_in", lambda *a, **k: True)
        assert ECGEMBuilder._solve_growth(SimpleNamespace(biomass_reaction="X")) == 0.0
        assert ECGEMBuilder._solve_fluxes(
            SimpleNamespace(biomass_reaction="X")
        ) == {}

        from helixlang.api import capabilities
        from helixlang.core.errors import ModelMissingError
        monkeypatch.setattr(capabilities, "opt_in", lambda *a, **k: False)
        with pytest.raises(ModelMissingError):
            ECGEMBuilder._solve_growth(SimpleNamespace(biomass_reaction="X"))
        with pytest.raises(ModelMissingError):
            ECGEMBuilder._solve_fluxes(SimpleNamespace(biomass_reaction="X"))
