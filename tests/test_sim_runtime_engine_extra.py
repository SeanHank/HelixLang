"""Additional tests for sim_runtime/_engine.py — helper functions.

Covers:
  - _config_seed / _collect_seeds / _solver_probe
  - _effective_config / _effective_config_dict
  - _parse_bit_sequence
  - _opt_morphogen_genes / _opt_morphogen_repression
  - _parse_lsystem_rules
  - _cello_truth_table
  - _group_prefixed / _split_pair
  - _build_ecosystem_species / _build_ecosystem_patches
  - _gene_orfs / _first_gene_protein
  - _build_model_from_reactions
  - _build_pop_flow / _build_pop_lbm
  - _seed_cells / _build_population_config validation
"""
from __future__ import annotations

import pytest

from helixlang.core.ast_nodes import Program
from helixlang.core.errors import SimConfigError
from helixlang.sim_runtime._engine import (
    _build_disease_from_helix,
    _build_drugs_from_helix,
    _build_ecosystem_patches,
    _build_ecosystem_species,
    _build_endocrine_config_from_helix,
    _build_genotype_from_helix,
    _build_immune_config_from_helix,
    _build_model_from_reactions,
    _build_pd_from_helix,
    _build_pop_flow,
    _build_pop_lbm,
    _build_population_config,
    _build_qsp_bindings_from_helix,
    _build_traits_from_helix,
    _build_tumor_biopsy_from_helix,
    _cello_truth_table,
    _collect_seeds,
    _config_seed,
    _effective_config,
    _effective_config_dict,
    _environment,
    _first_gene_protein,
    _gene_orfs,
    _group_prefixed,
    _opt_morphogen_genes,
    _opt_morphogen_repression,
    _parse_bit_sequence,
    _parse_lsystem_rules,
    _seed_cells,
    _solver_probe,
    _split_pair,
)


def _make_program(**kw):
    cfg_kw = kw.pop("config", {})
    cfg = Program.Config(**cfg_kw)
    return Program(config=cfg, **kw)


# ─── _config_seed / _collect_seeds ─────────────────────────────────

class TestConfigSeed:
    def test_config_seed(self):
        prog = Program()
        assert _config_seed(prog) is None

    def test_config_seed_from_sim(self):
        from helixlang.core.ast_nodes import Config
        prog = Program(config=Config(sim={"seed": "42"}))
        assert _config_seed(prog) == 42

    def test_config_seed_error(self):
        from helixlang.core.ast_nodes import Config
        prog = Program(config=Config(sim={"seed": "abc"}))
        assert _config_seed(prog) is None


class TestCollectSeeds:
    def test_empty(self):
        prog = Program()
        assert _collect_seeds(prog) == {}

    def test_noise_seed_from_config(self):
        from helixlang.core.ast_nodes import Config
        prog = Program(config=Config(sim={"noise_seed": "7"}))
        seeds = _collect_seeds(prog)
        assert seeds.get("noise_seed") == 7

    def test_extension_seed(self):
        from helixlang.core.ast_nodes import Config
        prog = Program(config=Config())
        prog.extensions["sde_seed"] = 99
        seeds = _collect_seeds(prog)
        assert seeds.get("sim.sde_seed") == 99

    def test_seed_alias_dict(self):
        from helixlang.core.ast_nodes import Config
        prog = Program(config=Config(sim={"seed": "12"}))
        seeds = _collect_seeds(prog)
        assert seeds.get("seed") == 12

    def test_bad_seed_skipped(self):
        from helixlang.core.ast_nodes import Config
        prog = Program(config=Config(sim={"seed": "NaN"}))
        seeds = _collect_seeds(prog)
        assert "seed" not in seeds


class TestSolverProbe:
    def test_no_meta(self):
        class R:
            pass
        result = R()
        assert _solver_probe(result) == {"id": "unknown:see backend"}

    def test_meta_dict_solver(self):
        class R:
            meta = {"solver": {"id": "scipy", "status": "optimal"}}
        assert _solver_probe(R()) == {"id": "scipy", "status": "optimal"}

    def test_meta_method(self):
        class R:
            meta = {"solver_method": "simplex", "solver_status": "optimal"}
        assert _solver_probe(R()) == {"id": "simplex", "status": "optimal"}

    def test_meta_method_only(self):
        class R:
            meta = {"solver_method": "simplex"}
        assert _solver_probe(R()) == {"id": "simplex", "status": "None"}


class TestEffectiveConfig:
    def test_basic(self):
        from helixlang.core.ast_nodes import Config
        prog = Program(config=Config(ticks=100, backend="classic"))
        cfg = _effective_config(prog)
        assert cfg.kind == "classic"
        assert cfg.backend is None
        assert cfg.ticks == 100

    def test_nonclassic_backend(self):
        from helixlang.core.ast_nodes import Config
        prog = Program(config=Config(backend="fba"))
        cfg = _effective_config(prog)
        assert cfg.backend == "fba"

    def test_extensions_kind(self):
        prog = Program()
        prog.extensions["kind"] = "ecosystem"
        cfg = _effective_config(prog)
        assert cfg.kind == "ecosystem"

    def test_output(self):
        from helixlang.core.ast_nodes import Config
        prog = Program(config=Config(output=["json"]))
        cfg = _effective_config(prog)
        assert cfg.output == "json"

    def test_dict(self):
        prog = Program()
        d = _effective_config_dict(prog)
        assert isinstance(d, dict)


# ─── _parse_bit_sequence ───────────────────────────────────────────

class TestParseBitSequence:
    def test_commas(self):
        assert _parse_bit_sequence("1,0,1,0") == (1, 0, 1, 0)

    def test_compact(self):
        assert _parse_bit_sequence("1010") == (1, 0, 1, 0)

    def test_bad_char_raises(self):
        with pytest.raises(SimConfigError):
            _parse_bit_sequence("1,2,3")

    def test_empty_raises(self):
        with pytest.raises(SimConfigError):
            _parse_bit_sequence("  ")


# ─── morphogen helpers ─────────────────────────────────────────────

class TestOptMorphogenGenes:
    def test_empty(self):
        from helixlang.plugins.apps.morphogen_gradient import MorphogenGradientConfig
        assert _opt_morphogen_genes({}) == MorphogenGradientConfig().genes

    def test_parse(self):
        genes = _opt_morphogen_genes({"genes": "near:12.0, mid:6.0"})
        assert len(genes) == 2
        assert genes[0].name == "near"
        assert genes[0].threshold_um == pytest.approx(12.0)

    def test_missing_threshold(self):
        with pytest.raises(SimConfigError, match="genes"):
            _opt_morphogen_genes({"genes": "near"})


class TestOptMorphogenRepression:
    def test_empty(self):
        from helixlang.plugins.apps.morphogen_gradient import MorphogenGradientConfig
        assert _opt_morphogen_repression({}) == MorphogenGradientConfig().repression

    def test_parse(self):
        edges = _opt_morphogen_repression({"repression": "near,mid;near,far"})
        assert len(edges) == 2
        assert ("near", "mid") in edges

    def test_missing_target(self):
        with pytest.raises(SimConfigError, match="repression"):
            _opt_morphogen_repression({"repression": "near"})


# ─── _parse_lsystem_rules ──────────────────────────────────────────

class TestParseLsystemRules:
    def test_parse(self):
        rules = _parse_lsystem_rules("X=F+[[X]-X];F=FF")
        assert rules["X"] == "F+[[X]-X]"
        assert rules["F"] == "FF"

    def test_empty_raises(self):
        with pytest.raises(SimConfigError, match="rules"):
            _parse_lsystem_rules("  ")

    def test_no_production_raises(self):
        with pytest.raises(SimConfigError, match="rules"):
            _parse_lsystem_rules("X")


# ─── _cello_truth_table ────────────────────────────────────────────

class TestCelloTruthTable:
    def test_not(self):
        tt = _cello_truth_table("not")
        assert tt is not None

    def test_nand(self):
        tt = _cello_truth_table("nand")
        assert tt is not None

    def test_default_xor(self):
        tt = _cello_truth_table("xor")
        assert tt is not None


# ─── _group_prefixed / _split_pair ─────────────────────────────────

class TestGroupPrefixed:
    def test_basic(self):
        ext = {"species.A.genome": "g1", "species.B.substrate": "glc"}
        groups = _group_prefixed(ext, "species.")
        assert groups["A"]["genome"] == "g1"
        assert groups["B"]["substrate"] == "glc"

    def test_ignore_non_matching(self):
        assert _group_prefixed({"foo": "1"}, "species.") == {}

    def test_ignore_partial(self):
        assert _group_prefixed({"species.A": "1"}, "species.") == {}


class TestSplitPair:
    def test_valid(self):
        name, val = _split_pair("glucose:0.02", "k")
        assert name == "glucose"
        assert val == pytest.approx(0.02)

    def test_missing_colon(self):
        with pytest.raises(SimConfigError):
            _split_pair("glucose", "k")

    def test_empty_name(self):
        with pytest.raises(SimConfigError):
            _split_pair(":0.02", "k")


# ─── ecosystem helpers ─────────────────────────────────────────────

class TestBuildEcosystemSpecies:
    def test_basic(self):
        ext = {
            "species.cyano.genome": "synth",
            "species.cyano.photo": "true",
            "species.cyano.substrate": "glucose",
            "species.cyano.vmax": "0.05",
            "species.cyano.secretion": "oxygen:0.01",
        }
        species = _build_ecosystem_species(ext)
        assert len(species) == 1
        sp = species[0]
        assert sp.name == "cyano"
        assert sp.photo is True
        assert sp.genome == "synth"

    def test_consumption(self):
        ext = {
            "species.E.substrate": "glucose",
            "species.E.vmax": "0.1",
            "species.E.ks": "0.2",
            "species.E.secretion.glucose": "0.01",
            "species.E.diet.prey": "0.5",
            "species.E.attack.prey": "0.05",
        }
        species = _build_ecosystem_species(ext)
        assert len(species) == 1
        sp = species[0]
        assert sp.consumption["glucose"][0] == pytest.approx(0.1)
        assert sp.secretion["glucose"] == pytest.approx(0.01)
        assert sp.diet["prey"] == pytest.approx(0.5)
        assert sp.attack_rate["prey"] == pytest.approx(0.05)

    def test_no_species(self):
        assert _build_ecosystem_species({}) == []


class TestBuildEcosystemPatches:
    def test_basic(self):
        ext = {
            "patch.p1.kind": "ocean",
            "patch.p1.width": "10",
            "patch.p1.height": "10",
            "patch.p1.anoxic": "true",
        }
        patches = _build_ecosystem_patches(ext)
        assert len(patches) == 1
        assert patches[0].name == "p1"
        assert patches[0].width == 10
        assert patches[0].height == 10
        assert patches[0].anoxic is True

    def test_no_patches(self):
        assert _build_ecosystem_patches({}) == []


# ─── genes ─────────────────────────────────────────────────────────

class TestGeneOrfs:
    def test_empty(self):
        from helixlang.core.ast_nodes import Program
        assert _gene_orfs(Program()) == []

    def test_with_gene(self):
        from helixlang.core.ast_nodes import Gene, Program
        prog = Program(genes=[Gene(name="g1", promoter=None, codons=[], orf=[])])
        assert _gene_orfs(prog) == [("g1", "")]

    def test_first_gene_protein_empty(self):
        from helixlang.core.ast_nodes import Program
        assert _first_gene_protein(Program()) == ""


# ─── model from reactions ──────────────────────────────────────────

class TestBuildModelFromReactions:
    def test_no_reactions(self):
        from helixlang.core.ast_nodes import Program
        assert _build_model_from_reactions(Program()) is None

    def test_with_reactions(self):
        from helixlang.core.ast_nodes import ReactionDecl
        prog = Program()
        prog.reactions = [ReactionDecl(
            id="r1", name="r1", substrate="A", substrate_coeff=-1.0,
            product="B", product_coeff=1.0, lower_bound=0.0,
            upper_bound=10.0, subsystem="tca")]
        model = _build_model_from_reactions(prog)
        assert model is not None
        assert "r1" in model.reactions


# ─── population config build ───────────────────────────────────────

class TestBuildPopFlow:
    def test_none(self):
        assert _build_pop_flow({}, 10, 10) is None

    def test_channel_poiseuille(self):
        flow = _build_pop_flow({"flow": "channel_poiseuille"}, 10, 10)
        assert flow is not None

    def test_stagnant(self):
        flow = _build_pop_flow({"flow": "stagnant"}, 10, 10)
        assert flow is not None

    def test_invalid(self):
        with pytest.raises(SimConfigError):
            _build_pop_flow({"flow": "bogus"}, 10, 10)


class TestBuildPopLBM:
    def test_none(self):
        assert _build_pop_lbm({}, 10, 10) is None

    def test_lbm_true(self):
        lbm = _build_pop_lbm({"lbm": "true"}, 10, 10)
        assert lbm is not None

    def test_lbm_3d(self):
        lbm = _build_pop_lbm({"lbm_3d": "true"}, 10, 10, depth=3)
        assert lbm is not None


class TestSeedCells:
    def test_basic(self):
        from helixlang.sim_runtime._engine import _build_population_config
        prog = Program()
        cfg = _build_population_config(prog)
        cells = _seed_cells(cfg, 5)
        assert len(cells) == 5
        for cell in cells:
            assert cell.alive is True


class TestBuildPopulationConfig:
    def test_default(self):
        prog = Program()
        cfg = _build_population_config(prog)
        assert cfg.grid_width == 32
        assert cfg.grid_height == 32

    def test_mechanics_invalid(self):
        prog = Program()
        prog.extensions["mechanics"] = "bogus"
        with pytest.raises(SimConfigError, match="mechanics"):
            _build_population_config(prog)

    def test_cell_shape_invalid(self):
        prog = Program()
        prog.extensions["cell_shape"] = "cube"
        with pytest.raises(SimConfigError, match="cell_shape"):
            _build_population_config(prog)

    def test_lbm_3d_needs_depth(self):
        prog = Program()
        prog.extensions["lbm_3d"] = "true"
        with pytest.raises(SimConfigError, match="grid_depth"):
            _build_population_config(prog)

    def test_flow_lbm_exclusive(self):
        prog = Program()
        prog.extensions["flow"] = "channel_poiseuille"
        prog.extensions["lbm"] = "true"
        with pytest.raises(SimConfigError):
            _build_population_config(prog)

    def test_dfba(self):
        prog = Program()
        prog.extensions["dfba"] = "true"
        prog.extensions["acetate_switch"] = "true"
        cfg = _build_population_config(prog)
        assert cfg.dfba_enabled is True
        assert cfg.acetate_switch is True

    def test_dfba_params(self):
        prog = Program()
        prog.extensions["dfba"] = "true"
        prog.extensions["dfba_dt_h"] = "0.1"
        cfg = _build_population_config(prog)
        assert cfg.dfba_dt_h == pytest.approx(0.1)


# ─── human builder helpers ─────────────────────────────────────────

class TestBuildTraits:
    def test_defaults(self):
        traits = _build_traits_from_helix({})
        assert traits.age_years == pytest.approx(30.0)
        assert traits.sex == "male"
        assert traits.body_weight_kg == pytest.approx(70.0)

    def test_custom(self):
        traits = _build_traits_from_helix({
            "person_age": "45", "person_sex": "female",
            "trait_smoking": "former",
        })
        assert traits.age_years == pytest.approx(45.0)
        assert traits.sex == "female"


class TestBuildDisease:
    def test_none(self):
        assert _build_disease_from_helix({}) is None

    def test_basic(self):
        d = _build_disease_from_helix({
            "disease_name": "NASH",
            "disease_genes": [{"gene": "PNPLA3", "type": "downregulate"}],
            "disease_metabolites": [{"id": "TG", "concentration": "1.0"}],
        })
        assert d is not None
        assert d.name == "NASH"
        assert len(d.gene_perturbations) == 1
        assert len(d.metabolite_perturbations) == 1


class TestBuildTumorBiopsy:
    def test_none(self):
        assert _build_tumor_biopsy_from_helix({}) is None

    def test_basic(self):
        bio = _build_tumor_biopsy_from_helix({
            "tumor_biopsy": {
                "mutation": "EGFR, KRAS",
                "amplification": "ERBB2",
                "msi_status": "MSI-H",
            },
        })
        assert bio is not None
        assert bio["mutations"] == ["EGFR", "KRAS"]
        assert bio["amplifications"] == ["ERBB2"]
        assert bio["msi_status"] == "MSI-H"

    def test_no_biopsy(self):
        assert _build_tumor_biopsy_from_helix({"tumor_biopsy": "notadict"}) is None


class TestBuildDrugs:
    def test_empty(self):
        assert _build_drugs_from_helix({}) == []

    def test_no_smiles(self):
        drugs = _build_drugs_from_helix({
            "drugs": [{
                "name": "warfarin",
                "target_protein": "VKORC1",
                "dose": "5", "cyp_metabolism": "CYP2C9:0.7",
            }],
        })
        assert len(drugs) == 1
        assert drugs[0].molecule.name == "warfarin"
        assert drugs[0].cyp_metabolism.get("CYP2C9") == pytest.approx(0.7)

    def test_invalid_entries_skipped(self):
        drugs = _build_drugs_from_helix({
            "drugs": [{"target": "X"}, "not-a-dict", {"name": "Aspirin"}],
        })
        assert len(drugs) == 1
        assert drugs[0].molecule.name == "Aspirin"


class TestBuildPD:
    def test_empty(self):
        assert _build_pd_from_helix({}) == {}

    def test_basic(self):
        pd = _build_pd_from_helix({
            "pd_effects": [{"drug": "Warfarin", "target": "VGOR", "ec50": "1.0"}],
        })
        assert "warfarin" in pd
        assert len(pd["warfarin"].effects) == 1

    def test_invalid_skipped(self):
        pd = _build_pd_from_helix({"pd_effects": [{"target": "X"}]})
        assert pd == {}


class TestBuildQSPEndocrineImmune:
    def test_qsp(self):
        out = _build_qsp_bindings_from_helix({})
        assert out["qsp_bindings"] == []

    def test_qsp_basic(self):
        out = _build_qsp_bindings_from_helix({
            "qsp_bindings": [{"drug": "D1", "kd_nM": "1.0"}],
        })
        assert out["qsp_bindings"][0]["drug"] == "D1"
        assert out["qsp_bindings"][0]["kd_nM"] == pytest.approx(1.0)

    def test_endocrine(self):
        out = _build_endocrine_config_from_helix({
            "endocrine_configs": [{"axis": "HPA", "severity": "0.5"}],
        })
        assert out["endocrine_configs"][0]["axis"] == "HPA"

    def test_immune(self):
        out = _build_immune_config_from_helix({
            "immune_configs": [{"infection_severity": "0.2"}],
        })
        assert out["immune_configs"][0]["infection_severity"] == pytest.approx(0.2)
        assert out["immune_configs"][0] == out["immune_configs"][0]


class TestBuildGenotype:
    def test_empty(self):
        from helixlang.plugins.human.genotype import GenotypeProfile
        gt = GenotypeProfile()
        _build_genotype_from_helix(gt, {})
        assert not gt.cyp_status

    def test_non_list_genes_returns(self):
        from helixlang.plugins.human.genotype import GenotypeProfile
        gt = GenotypeProfile()
        _build_genotype_from_helix(gt, {"genes": "not-a-list"})
        assert not gt.cyp_status


# ─── _environment ──────────────────────────────────────────────────

class TestEnvironment:
    def test_empty(self):
        from helixlang.core.ast_nodes import Program
        env = _environment(Program(), 10, 10)
        assert env is not None

    def test_glucose_medium(self):
        from helixlang.core.ast_nodes import MediaDecl, Program
        prog = Program(media=[MediaDecl(
            nutrient="GLC", concentration=5.0, diffusion_um2_s=None)])
        env = _environment(prog, 10, 10)
        assert env.glucose is not None

    def test_oxygen_medium(self):
        from helixlang.core.ast_nodes import MediaDecl, Program
        prog = Program(media=[MediaDecl(
            nutrient="O2", concentration=8.0, diffusion_um2_s=None)])
        env = _environment(prog, 10, 10)
        assert env.oxygen is not None

    def test_custom_field(self):
        from helixlang.core.ast_nodes import MediaDecl, Program
        prog = Program(media=[MediaDecl(
            nutrient="lactate", concentration=1.0, diffusion_um2_s=200.0)])
        env = _environment(prog, 10, 10)
        assert "lactate" in env.fields
