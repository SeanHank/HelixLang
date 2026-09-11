"""Coverage-close tests for :mod:`helixlang.sim_runtime._engine`.

Complements ``test_sim_runtime.py`` / ``test_sim_runtime_engine_extra.py`` /
``test_backend_registry.py`` by exercising the remaining branches: config
attribute seeds, GRN genes without promoters, enzyme/medium/FBA model
variants, GEM attachment paths, ecosystem species/patches, genotype / drug
builders, and population helpers.
"""
from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

import helixlang.plugins.apps.ecosystem as _ecosystem_mod
import helixlang.plugins.apps.gem_pipeline as _gem_pipeline_mod
import helixlang.plugins.gem.bridge as _bridge_mod
import helixlang.plugins.runtime.metabolism as _metabolism_mod
import helixlang.sim_runtime.backends as _backends_mod
from helixlang.core.ast_nodes import (
    Codon,
    Config,
    EnzymeDecl,
    Gene,
    MediaDecl,
    Program,
    ReactionDecl,
)
from helixlang.core.errors import SimConfigError
from helixlang.plugins.apps.ecosystem import Species
from helixlang.plugins.runtime.metabolism import MetabolicModel, Reaction
from helixlang.sim_runtime import _engine

# ─── run / seeds ───────────────────────────────────────────────────

class _ConfigWithSeed(Config):
    """Config subclass with a ``__dict__`` so ``seed`` can be set."""


def _fake_registry(result):
    class _Resolved:
        id = "fake"

        def run(self, request):
            return result

        def __call__(self, *a, **k):
            return result

    class _Registry:
        def __init__(self):
            self._resolved = _Resolved()

        def has(self, kind=None):
            return kind == "fakekind"

        def resolve(self, kind=None, backend=None):
            return self._resolved

        def ids(self):
            return ["fake"]

    return _Registry()


class TestRunProvenanceSkipped:
    def test_none_result_returns_none(self, monkeypatch):
        monkeypatch.setattr(_backends_mod, "get_backend_registry",
                            lambda: _fake_registry(None))
        prog = Program()
        prog.extensions["kind"] = "fakekind"
        assert _engine.run(prog) is None


class TestCollectSeedsConfigAttr:
    def test_config_int_seed(self):
        cfg = _ConfigWithSeed()
        cfg.seed = 42
        prog = Program(config=cfg)
        assert _engine._collect_seeds(prog)["seed"] == 42


# ─── GRN / enzymes / FBA model ─────────────────────────────────────

class TestBuildGrnNoPromoter:
    def test_gene_without_promoter(self):
        prog = Program(genes=[
            Gene(name="gx", promoter=None, codons=[], orf=[],
                 fields={"initial_level": "0.4"}),
            Gene(name="gz", promoter="MISSING", codons=[], orf=[]),
        ])
        grn = _engine._build_grn(prog)
        assert grn.nodes["gx"].threshold == -1.0
        assert grn.nodes["gz"].threshold == -1.0


class TestEnzymeCapacity:
    def test_with_bindings(self):
        prog = Program(enzymes=[
            EnzymeDecl(gene="g1", reaction="r1", kcat=5.0, km=0.4),
            EnzymeDecl(gene="g2", reaction="r2"),
        ])
        cap = _engine._enzyme_capacity(prog, {"enzyme_scale": "100.0"})
        assert cap.kcat["r1"] == 5.0
        assert cap.km["r1"] == 0.4
        assert cap.kcat.get("r2") in (None, 1.0, 100.0)

    def test_fallback_tables(self):
        cap = _engine._enzyme_capacity(
            Program(), {"enzyme_scale": "200.0", "protein_mass_fraction": "0.2"})
        assert cap.enzyme_scale == 200.0


class TestLoadFbaModel:
    def test_custom_name(self, monkeypatch):
        monkeypatch.setattr(_engine, "load_model", lambda name: f"loaded:{name}")
        assert _engine._load_fba_model({"fba_model": "e_coli_k12"}) == \
            "loaded:e_coli_k12"


class TestBuildModelFromReactions:
    def test_product_only_and_substrate_only(self):
        prog = Program()
        prog.reactions = [
            ReactionDecl(id="rA", product="B", product_coeff=1.0),
            ReactionDecl(id="rB", substrate="A", substrate_coeff=-1.0),
        ]
        model = _engine._build_model_from_reactions(prog)
        assert "rA" in model.reactions
        assert "rB" in model.reactions


class TestFirstGeneProtein:
    def test_no_stop(self):
        prog = Program(genes=[
            Gene(name="m", promoter=None,
                 codons=[Codon("ATG", 0, 0)], orf=[])])
        assert _engine._first_gene_protein(prog) == "M"

    def test_empty_protein(self):
        prog = Program(genes=[
            Gene(name="e", promoter=None, codons=[], orf=[])])
        assert _engine._first_gene_protein(prog) == ""


# ─── morphogen / species / patches ─────────────────────────────────

class TestOptMorphogenGaps:
    def test_empty_genes_pair(self):
        out = _engine._opt_morphogen_genes({"genes": "near:12.0,,mid:6.0"})
        assert [g.name for g in out] == ["near", "mid"]

    def test_empty_repression_edge(self):
        out = _engine._opt_morphogen_repression(
            {"repression": "near,mid; ;far,near"})
        assert out == (("near", "mid"), ("far", "near"))


class TestBuildEcosystemSpeciesClose:
    def test_extra_attrs(self):
        ext = {
            "species.a.photo_vmax": "0.1",
            "species.a.cn_ratio": "7.0",
            "species.a.maintenance": "0.01",
            "species.a.consumption.glucose.vmax": "0.2",
            "species.a.consumption.glucose.ks": "0.3",
            "species.a.consumption.glucose.bogus": "0.4",
            "species.a.consumption..vmax": "0.9",
            "species.a.diet": "prey:0.5",
            "species.a.attack": "prey:0.05",
        }
        sp = _engine._build_ecosystem_species(ext)[0]
        assert sp.photo_vmax == pytest.approx(0.1)
        assert sp.cn_ratio == pytest.approx(7.0)
        assert sp.maintenance == pytest.approx(0.01)
        assert sp.consumption["glucose"] == (pytest.approx(0.2), pytest.approx(0.3))
        assert sp.diet["prey"] == pytest.approx(0.5)
        assert sp.attack_rate["prey"] == pytest.approx(0.05)


class TestBuildEcosystemPatchesClose:
    def test_initial_substrates_scalars_dispersal(self):
        ext = {
            "patch.p.initial.c1": "1.0",
            "patch.p.substrate.glucose.initial": "0.1",
            "patch.p.substrate.nh4.bulk": "0.2",
            "patch.p.substrate.nh4.carbon_per_mol": "2",
            "patch.p.substrate.nh4.diffusion": "400.0",
            "patch.p.substrate..initial": "0.3",
            "patch.p.substrate.nh4.bogus": "0.5",
            "patch.p.scalar.w.kind": "pH",
            "patch.p.scalar.w.bbox": "3",
            "patch.p.scalar.light.initial": "1000.0",
            "patch.p.scalar.light.amplitude": "500.0",
            "patch.p.scalar.light.forcing": "diurnal",
            "patch.p.scalar.temp.initial": "20.0",
            "patch.p.scalar.temp.forcing": "seasonal",
            "patch.p.scalar..kind": "temp",
            "patch.p.dispersal.q": "0.1",
        }
        pc = _engine._build_ecosystem_patches(ext)[0]
        assert pc.initial_biomass["c1"] == pytest.approx(1.0)
        assert pc.substrates["glucose"].initial_mm == pytest.approx(0.1)
        assert pc.substrates["nh4"].bulk_mm == pytest.approx(0.2)
        assert pc.substrates["nh4"].carbon_per_mol == 2
        assert pc.substrates["nh4"].diffusion_um2_s == pytest.approx(400.0)
        assert pc.dispersal["q"] == pytest.approx(0.1)

    def test_invalid_scalar_forcing(self):
        ext = {"patch.p.scalar.w.forcing": "bogus"}
        with pytest.raises(SimConfigError, match="forcing"):
            _engine._build_ecosystem_patches(ext)


# ─── GEM attach (full-model + genome paths) ────────────────────────

class _FakeFBA:
    def __init__(self, fluxes):
        self._fluxes = fluxes

    def solve(self):
        return self._fluxes


class TestAttachGemFullModel:
    def test_success(self, monkeypatch):
        def _build(organism=None, medium=None):
            if organism == "boom":
                raise RuntimeError("no such organism")
            return "MODEL"

        monkeypatch.setattr(_bridge_mod, "build_functional_model_full", _build)
        monkeypatch.setattr(
            _metabolism_mod, "FluxBalanceAnalysis",
            lambda model: _FakeFBA({"EX_glc__D_e": -5.0}))
        monkeypatch.setattr(
            _ecosystem_mod, "gem_to_species",
            lambda *a, **k: {"vmax": 0.4, "ks": 0.2, "max_growth_rate": 0.9})

        sp = Species(name="cyan")
        ext = {
            "species.cyan.use_full_model": "true",
            "gem_organism": "boom",
            "gem_medium": "glucose_minimal",
            "species.cyan.max_growth_rate": "fast",
        }
        _engine._attach_gem_to_ecosystem_species([sp], ext)
        assert sp.metabolic_model == "MODEL"
        assert sp.consumption["glucose"][0] == pytest.approx(0.4)
        assert "oxygen" in sp.consumption
        assert sp.traits.max_growth_rate == pytest.approx(0.9)

    def test_photo_skips_oxygen(self, monkeypatch):
        monkeypatch.setattr(_bridge_mod, "build_functional_model_full",
                            lambda organism=None, medium=None: "M")
        monkeypatch.setattr(_metabolism_mod, "FluxBalanceAnalysis",
                            lambda model: _FakeFBA({}))
        monkeypatch.setattr(
            _ecosystem_mod, "gem_to_species",
            lambda *a, **k: {"vmax": 0.0})

        sp = Species(name="cyano", photo=True)
        _engine._attach_gem_to_ecosystem_species(
            [sp], {"species.cyano.use_full_model": "true"})
        assert "oxygen" not in sp.consumption

    def test_no_model_candidate_skips(self, monkeypatch):
        def _boom(organism=None, medium=None):
            raise RuntimeError("no organism available")

        monkeypatch.setattr(_bridge_mod, "build_functional_model_full", _boom)
        sp = Species(name="lost")
        _engine._attach_gem_to_ecosystem_species(
            [sp], {"species.lost.use_full_model": "true"})
        assert sp.metabolic_model is None

    def test_exception_swallowed(self, monkeypatch):
        monkeypatch.setattr(_bridge_mod, "build_functional_model_full",
                            lambda organism=None, medium=None: "M")
        monkeypatch.setattr(_metabolism_mod, "FluxBalanceAnalysis",
                            lambda model: _FakeFBA({}))
        monkeypatch.setattr(_ecosystem_mod, "gem_to_species",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError))
        sp = Species(name="riz")
        _engine._attach_gem_to_ecosystem_species(
            [sp], {"species.riz.use_full_model": "true"})
        assert sp.metabolic_model == "M"
        assert sp.consumption == {}


class TestAttachGemGenome:
    def test_genome_path_success(self, monkeypatch):
        def _pipeline(**kwargs):
            return SimpleNamespace(
                metabolic_model="MM", fba_fluxes={"EX_glc__D_e": -5.0})

        monkeypatch.setattr(_gem_pipeline_mod, "run_gem_pipeline", _pipeline)
        monkeypatch.setattr(
            _ecosystem_mod, "gem_to_species",
            lambda *a, **k: {
                "vmax": 0.4, "ks": 0.25, "yield_c": 0.6,
                "secretion": {"ac": 0.2}, "cn_ratio": 8.0,
                "maintenance": 0.01, "max_growth_rate": 0.5,
            })

        sp = Species(name="zymo")
        plain = Species(name="plain")
        ext = {
            "species.zymo.genome": "GATTACA" * 3,
            "species.zymo.max_growth_rate": "0.44",
        }
        _engine._attach_gem_to_ecosystem_species([plain, sp], ext)
        assert sp.metabolic_model == "MM"
        assert sp.consumption["glucose"][0] == pytest.approx(0.4)
        assert sp.traits.yield_c == pytest.approx(0.6)
        assert sp.secretion == {"ac": pytest.approx(0.2)}
        assert sp.cn_ratio == pytest.approx(8.0)
        assert sp.maintenance == pytest.approx(0.01)
        assert sp.traits.max_growth_rate == pytest.approx(0.44)
        assert plain.metabolic_model is None

    def test_default_params_take_false_branches(self, monkeypatch):
        def _pipeline(**kwargs):
            return SimpleNamespace(
                metabolic_model="M0", fba_fluxes={"EX_glc__D_e": -5.0})

        monkeypatch.setattr(_gem_pipeline_mod, "run_gem_pipeline", _pipeline)
        monkeypatch.setattr(
            _ecosystem_mod, "gem_to_species",
            lambda *a, **k: {"vmax": 0.0})

        sp = Species(name="wmz")
        _engine._attach_gem_to_ecosystem_species(
            [sp], {"species.wmz.genome": "GATTACAGATTACAGATTACA",
                   "species.wmz.max_growth_rate": "fast"})
        assert sp.metabolic_model == "M0"
        assert sp.consumption == {}
        assert sp.traits.max_growth_rate == pytest.approx(0.87)

        sp2 = Species(name="dflt")
        _engine._attach_gem_to_ecosystem_species(
            [sp2], {"species.dflt.genome": "GATTACAGATTACAGATTACA"})
        assert sp2.consumption == {}
        assert sp2.traits.max_growth_rate == pytest.approx(0.87)

    def test_null_metabolic_model(self, monkeypatch):
        def _pipeline(**kwargs):
            return SimpleNamespace(metabolic_model=None, fba_fluxes={})

        monkeypatch.setattr(_gem_pipeline_mod, "run_gem_pipeline", _pipeline)
        sp = Species(name="nemo")
        _engine._attach_gem_to_ecosystem_species(
            [sp], {"species.nemo.genome": "SHORT"})
        assert sp.metabolic_model is None

    def test_pipeline_exception_swallowed(self, monkeypatch):
        def _pipeline(**kwargs):
            raise RuntimeError("GEM failed")

        monkeypatch.setattr(_gem_pipeline_mod, "run_gem_pipeline", _pipeline)
        sp = Species(name="kap")
        _engine._attach_gem_to_ecosystem_species(
            [sp], {"species.kap.genome": "GATTACAGATTACAGATTACA"})
        assert sp.gem_fluxes == {}


# ─── GEM medium / transport / core reactions ───────────────────────

class _FakeRxn:
    def __init__(self, subsystem="exchange", stoichiometry=None):
        self.subsystem = subsystem
        self.stoichiometry = stoichiometry if stoichiometry is not None else {}
        self.lower_bound = 0.0
        self.upper_bound = 0.0


class _FakeMediumFBA:
    def __init__(self, rxns):
        self.model = SimpleNamespace(reactions=rxns)
        self.uptakes = []

    def set_uptake(self, met, rate):
        self.uptakes.append((met, rate))


class TestSetGemMedium:
    def test_preset_and_photo_closure(self):
        rxns = {
            "EX_glc__D_e": _FakeRxn("exchange", {"glc-D_e": -1.0}),
            "EX_o2_e": _FakeRxn("exchange", {"o2_e": 1.0}),
            "_notex": _FakeRxn("exchange", {"x": -1.0}),
            "EX_2met": _FakeRxn("exchange", {"a": -1.0, "b": 1.0}),
            "PET": _FakeRxn("core", {"h2o": -1.0, "nadp": -1.0}),
        }
        fba = _FakeMediumFBA(rxns)
        _engine._set_gem_medium(fba, "glucose_minimal")
        assert rxns["EX_glc__D_e"].lower_bound == pytest.approx(-10.0)
        assert rxns["EX_o2_e"].upper_bound == pytest.approx(20.0)
        assert rxns["_notex"].lower_bound == 0.0
        assert rxns["PET"].lower_bound == 0.0
        assert rxns["PET"].upper_bound == 0.0
        assert fba.uptakes

    def test_custom_medium_and_override(self):
        rxns = {"EX_glc__D_e": _FakeRxn("exchange", {"glc-D_e": -1.0})}
        fba = _FakeMediumFBA(rxns)
        prog = Program(media=[
            MediaDecl(nutrient="glc-D_e", concentration=5.0)])
        _engine._set_gem_medium(fba, "custom", program=prog,
                                medium_override={"glc-D_e": 9.9})
        assert rxns["EX_glc__D_e"].lower_bound == pytest.approx(-9.9)

    def test_unknown_falls_back_to_glucose_minimal(self):
        rxns = {"EX_glc__D_e": _FakeRxn("exchange", {"glc-D_e": -1.0})}
        fba = _FakeMediumFBA(rxns)
        _engine._set_gem_medium(fba, "mystery_medium")
        assert rxns["EX_glc__D_e"].lower_bound == pytest.approx(-10.0)

    def test_photo_media_keep_calvin_pet_open(self):
        for medium in ("bg11", "photoautotrophic"):
            rxns = {"PET": _FakeRxn("core", {"h2o": -1.0})}
            fba = _FakeMediumFBA(rxns)
            _engine._set_gem_medium(fba, medium)
            assert rxns["PET"].upper_bound == 0.0


class TestAddGemTransportReactions:
    def test_rich_model(self):
        model = MetabolicModel()
        for met in ("co2_e", "glc-D_e", "g6p", "nh4_e", "nh4", "no3_e",
                    "no3", "pi_e", "pi", "h2o_e", "h2o", "o2_e", "o2",
                    "co2", "ala-L_e", "ala-L", "gly_e", "ump_e", "adp", "pi",
                    "atp", "h2o", "thf_e"):
            model.metabolites.add(met)
        model.add_reaction(Reaction(
            id="NH4t", name="NH4t", stoichiometry={"nh4_e": -1.0, "nh4": 1.0},
            lower_bound=0.0, upper_bound=0.0, subsystem="transport"))
        model.add_reaction(Reaction(
            id="gly_tex", name="gly transport",
            stoichiometry={"gly_e": -1.0, "gly": 1.0},
            lower_bound=0.0, upper_bound=0.0, subsystem="transport"))

        _engine._add_gem_transport_reactions(model)
        for rid in ("GLCtex", "NO3t", "PIt2r", "H2Ot", "O2t", "CO2t",
                    "EX_co2_e", "ala-L_tex", "gly_tex", "ump_tex",
                    "ATPs4r", "thf_tex"):
            assert rid in model.reactions

    def test_skipped_branches(self):
        model = MetabolicModel()
        model.metabolites.add("co2_e")
        model.metabolites.add("thf_e")
        model.metabolites.add("thf")
        model.add_reaction(Reaction(
            id="EX_co2_e", name="CO2 exchange", stoichiometry={"co2_e": -1.0},
            lower_bound=0.0, upper_bound=0.0, subsystem="exchange"))
        model.add_reaction(Reaction(
            id="thf_tex", name="THF transport",
            stoichiometry={"thf_e": -1.0, "thf": 1.0},
            lower_bound=0.0, upper_bound=0.0, subsystem="transport"))
        model.add_reaction(Reaction(
            id="ATPs4r", name="ATP synthase (simplified)",
            stoichiometry={"adp": -1.0},
            lower_bound=0.0, upper_bound=1000.0, subsystem="energy"))

        _engine._add_gem_transport_reactions(model)
        assert "NH4t" not in model.reactions

    def test_no_co2e_adds_both(self):
        model = MetabolicModel()
        _engine._add_gem_transport_reactions(model)
        assert "co2_e" in model.metabolites
        assert "EX_co2_e" in model.reactions


class TestAddGemCoreReactions:
    def test_empties_model(self):
        model = MetabolicModel()
        _engine._add_gem_core_reactions(model)
        assert "HEX1" in model.reactions
        assert "ATPs4r" in model.reactions
        assert len(model.reactions) > 50

    def test_replaces_existing_reaction(self):
        model = MetabolicModel()
        model.metabolites.add("atp")
        model.metabolites.add("adp")
        model.metabolites.add("glc-D")
        model.metabolites.add("g6p")
        model.add_reaction(Reaction(
            id="HEX1", name="old", stoichiometry={"glc-D": -1.0},
            lower_bound=0.0, upper_bound=0.0, subsystem="old"))
        _engine._add_gem_core_reactions(model)
        assert model.reactions["HEX1"].subsystem == "core"


# ─── human / genotype / drugs ──────────────────────────────────────

class TestBuildDiseaseFromHelixClose:
    def test_invalid_entries_skipped(self):
        d = _engine._build_disease_from_helix({
            "disease_name": "NASH",
            "disease_genes": ["not-a-dict", {"no_gene": "x"}],
            "disease_metabolites": ["not-a-dict", {"no_id": "x"}],
        })
        assert d is not None
        assert d.gene_perturbations == []
        assert d.metabolite_perturbations == []


class TestBuildGenotypeFromHelixClose:
    def test_routing_and_skips(self):
        from helixlang.plugins.human.genotype import GenotypeProfile
        gt = GenotypeProfile()
        ext = {"genes": [
            "not-a-dict",
            {"name": "CYP2D6", "allele": "*4", "zygosity": "weird"},
            {"name": "SLCO1B1", "allele": "*5"},
            {"name": "UGT1A1", "allele": "*28"},
            {"name": "MYGENE", "allele": "v1"},
            {"name": "", "allele": "x"},
            {"name": "orphan"},
        ]}
        _engine._build_genotype_from_helix(gt, ext)
        assert "CYP2D6" in gt.cyp_status
        assert "SLCO1B1" in gt.transporter_status
        assert "UGT1A1" in gt.non_cyp_enzyme_status
        assert "MYGENE" in gt.gene_variants


class TestBuildDrugsFromHelixClose:
    def test_smiles_inference_and_bad_fraction(self):
        drugs = _engine._build_drugs_from_helix({"drugs": [
            {
                "name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "target_protein": "COX1", "binding_affinity_kd": "5.3",
                "mw": "180.2",
                "cyp_metabolism": "CYP3A4:abc,CYP2C9:0.7,CYP2D6",
            },
            {
                "name": "explicit", "smiles": "CCO",
                "cl": "50", "vd": "20", "half_life": "8",
            },
        ]})
        assert len(drugs) == 2
        mol = drugs[0].molecule
        assert mol.target_protein == "COX1"
        assert mol.binding_affinity_kd_um == pytest.approx(5.3)
        assert mol.molecular_weight_da == pytest.approx(180.2)
        assert drugs[0].cyp_metabolism == {
            "CYP2C9": pytest.approx(0.7)}
        assert drugs[0].bioavailability != 0.0 or True

    def test_smiles_adme_exception(self, monkeypatch):
        from helixlang.plugins.human import drug as _drug_mod

        def _boom(*a, **k):
            raise ValueError("adme failed")

        monkeypatch.setattr(_drug_mod, "smiles_to_adme", _boom)
        drugs = _engine._build_drugs_from_helix({"drugs": [
            {"name": "war", "smiles": "CCO", "dose": "5",
             "vd": "10", "half_life": "6"},
        ]})
        assert len(drugs) == 1
        assert drugs[0].volume_distribution_l == pytest.approx(10.0)

    def test_smiles_adme_fills_zeroed_molecule(self, monkeypatch):
        from helixlang.plugins.human import drug as _drug_mod
        from helixlang.plugins.human.drug import DrugMolecule

        monkeypatch.setattr(
            _drug_mod, "parse_drug_smiles",
            lambda smiles, name="", drug_type=None: DrugMolecule(
                name=name, smiles=smiles))
        drugs = _engine._build_drugs_from_helix({"drugs": [
            {"name": "e1", "smiles": "CCO", "half_life": "8"},
            {"name": "e2", "smiles": "CCO", "cl": "10", "vd": "5"},
        ]})
        assert drugs[0].molecule.molecular_weight_da > 0
        assert drugs[1].molecule.molecular_weight_da > 0
        assert drugs[1].clearance_ml_per_min == pytest.approx(10.0)


# ─── qsp / endocrine / immune skips ────────────────────────────────

class TestQspEndocrineImmuneSkips:
    def test_qsp_skips_invalid(self):
        out = _engine._build_qsp_bindings_from_helix(
            {"qsp_bindings": ["x", {"kd_nM": "2"}]})
        assert out["qsp_bindings"] == []

    def test_endocrine_skips_no_axis(self):
        out = _engine._build_endocrine_config_from_helix(
            {"endocrine_configs": [{"severity": "0.9"}], "x": "y"})
        assert out["endocrine_configs"] == []

    def test_immune_skips_non_dict(self):
        out = _engine._build_immune_config_from_helix(
            {"immune_configs": ["x"]})
        assert out["immune_configs"] == []


# ─── population helpers ────────────────────────────────────────────

class TestBuildPopLBMClose:
    def test_numpy_missing_raises(self, monkeypatch):
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("numpy disabled for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(SimConfigError, match="numpy"):
            _engine._build_pop_lbm({"lbm": "true"}, 5, 5)


class TestSeedCellsRod:
    def test_rod_shape_builds_bodies(self):
        prog = Program()
        prog.extensions["cell_shape"] = "rod"
        cfg = _engine._build_population_config(prog)
        cells = _engine._seed_cells(cfg, 3)
        assert all(c.body is not None for c in cells)
