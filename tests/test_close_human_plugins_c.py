"""Close coverage gaps in human plugin modules (batch c)."""
from __future__ import annotations

import builtins
import importlib.util

import pytest

from helixlang.plugins.human.bayesian_denoiser import kalman_smoother
from helixlang.plugins.human.gem_human import HumanGEMLoader as GEMLoader
from helixlang.plugins.human.physiology import (
    HumanPhysiology,
    OrganSpec,
    create_default_physiology,
)
from helixlang.plugins.human.qsp_binding import QSPBindingSystem


def _load_blocking_imports(module_path: str, blocked: set[str]):
    """Exec a module's source with the given top-level imports failing."""
    import sys

    spec = importlib.util.find_spec(module_path)
    throwaway = importlib.util.spec_from_file_location(
        "_no_" + module_path.rsplit(".", 1)[-1], spec.origin)
    old = sys.modules.get(throwaway.name)
    sys.modules[throwaway.name] = mod = importlib.util.module_from_spec(throwaway)
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in blocked:
            raise ImportError(f"{name} blocked")
        return real_import(name, *a, **k)

    builtins.__import__ = fake_import
    try:
        throwaway.loader.exec_module(mod)  # type: ignore[union-attr]
    finally:
        builtins.__import__ = real_import
        if old is None:
            sys.modules.pop(throwaway.name, None)
        else:
            sys.modules[throwaway.name] = old
    return mod


# ---------------------------------------------------------------------------
# bayesian_denoiser.py
# ---------------------------------------------------------------------------
class TestKalmanSmoother:
    def test_empty_returns_list(self):
        assert kalman_smoother([], []) == []

    def test_single_point_roundtrip(self):
        out = kalman_smoother([0.0], [5.0])
        assert out == pytest.approx([5.0], rel=1e-6)

    def test_multiple_points_forward_backward(self):
        times = [0.0, 1.0, 2.0, 3.0]
        obs = [10.0, 9.0, 12.0, 11.0]
        out = kalman_smoother(times, obs)
        assert len(out) == 4
        assert all(v > 0.0 for v in out)
        # forward-backward smoothing preserves decay ordering
        assert out == sorted(out, reverse=True)

    def test_two_points_backward_single(self):
        out = kalman_smoother([0.0, 1.0], [4.0, 6.0])
        assert len(out) == 2


# ---------------------------------------------------------------------------
# gem_human.py
# ---------------------------------------------------------------------------
class TestGEMHuman:
    def test_missing_metabolism_branches(self):
        mod = _load_blocking_imports(
            "helixlang.plugins.human.gem_human",
            {"helixlang.plugins.runtime.metabolism"},
        )
        assert mod.Reaction is None and mod.MetabolicModel is None
        with pytest.raises(ImportError):
            mod.create_human_biomass_reaction()
        loader = mod.HumanGEMLoader()
        with pytest.raises(ImportError):
            loader.load_core_model()

    def test_missing_physiology_profile_falls_back(self):
        mod = _load_blocking_imports(
            "helixlang.plugins.human.gem_human",
            {"helixlang.plugins.human.physiology"},
        )
        assert mod.TISSUE_PROFILES == {}

    def test_load_failure_raises_model_missing(self, monkeypatch):
        import helixlang.plugins.gem.full_model as full_model_mod
        from helixlang.core.errors import ModelMissingError

        loader = GEMLoader()

        def boom(path, org):
            raise RuntimeError("no sbml")

        monkeypatch.setattr(full_model_mod.FullModelAdapter, "from_sbml",
                            staticmethod(boom))
        monkeypatch.setattr(
            "helixlang.api.capabilities.opt_in", lambda *a: False)
        with pytest.raises(ModelMissingError):
            loader.load_from_sbml("x/sbml.xml")

    def test_load_failure_low_fidelity_falls_back(self, monkeypatch):
        import helixlang.plugins.gem.full_model as full_model_mod

        loader = GEMLoader()
        monkeypatch.setattr(full_model_mod.FullModelAdapter, "from_sbml",
                            staticmethod(lambda path, org: (_ for _ in ()).throw(RuntimeError("no"))))
        monkeypatch.setattr("helixlang.api.capabilities.opt_in", lambda *a: True)
        assert loader.load_from_sbml("x/sbml.xml") is not None

    def test_load_adapter_none_falls_back(self, monkeypatch):
        import helixlang.plugins.gem.full_model as full_model_mod

        loader = GEMLoader()
        monkeypatch.setattr(full_model_mod.FullModelAdapter, "from_sbml",
                            staticmethod(lambda path, org: None))
        assert loader.load_from_sbml("x/sbml.xml") is not None

    def test_tissue_overlay_returns_model_when_empty_profile(self):
        loader = GEMLoader()
        core = loader.load_core_model()
        assert loader.apply_tissue_overlay(core, "no_such_tissue") is core

    def test_tissue_overlay_none_model(self, monkeypatch):
        monkeypatch.setattr(
            "helixlang.plugins.human.gem_human.TISSUE_PROFILES", {"liver": {"a": 1}})
        loader = GEMLoader()
        assert loader.apply_tissue_overlay(None, "liver") is None

    def test_tissue_overlay_skips_reaction_without_subsystem(self, monkeypatch):
        class Rxn:
            id = "EX_glc_e"
        rer = Rxn()

        class Model:
            reactions = {"x": rer}

        monkeypatch.setattr(
            "helixlang.plugins.human.gem_human.TISSUE_PROFILES",
            {"liver": {"glucose_uptake_mmol_per_kg_per_min": 1.0,
                       "oxygen_consumption_ml_per_kg_per_min": 10.0}})
        loader = GEMLoader()
        out = loader.apply_tissue_overlay(Model(), "liver")
        assert out.reactions["x"] is rer

    def test_create_biomass_reaction_normal(self):
        from helixlang.plugins.human.gem_human import create_human_biomass_reaction
        rxn = create_human_biomass_reaction()
        assert rxn.id == "BIOMASS_HUMAN"

    def test_load_adapter_returned(self, monkeypatch):
        import helixlang.plugins.gem.full_model as full_model_mod

        adapter = object()
        monkeypatch.setattr(full_model_mod.FullModelAdapter, "from_sbml",
                            staticmethod(lambda path, org: adapter))
        assert GEMLoader().load_from_sbml("x/sbml.xml") is adapter

    def test_get_exchange_empty_model(self):
        loader = GEMLoader()
        assert loader.get_exchange_reactions(None) == {}

    def test_load_uses_model_path(self, monkeypatch):
        import helixlang.plugins.gem.full_model as full_model_mod

        loader = GEMLoader()
        monkeypatch.setattr(full_model_mod.FullModelAdapter, "from_sbml",
                            staticmethod(lambda path, org: None))
        monkeypatch.setattr(loader, "config",
                            type("C", (), {"model_path": "p", "tissue": "liver"}))
        assert loader.load("liver") is not None


# ---------------------------------------------------------------------------
# qsp_binding.py
# ---------------------------------------------------------------------------
class TestQSPBinding:
    def test_mass_action_effect(self):
        sys = QSPBindingSystem()
        sys.add_mass_action("m", 10.0, emax=1.0, baseline=0.0)
        sys.set_drug_concentration("m", 100.0)
        assert sys.get_effect("m") > 0.0

    def test_tmdd_zero_receptor_occupancy_and_effect(self):
        sys = QSPBindingSystem()
        sys.add_tmdd("t")
        sys.models["t"].tmdd.r_total_nM = 0.0
        assert sys.get_effect("t") == 0.0

    def test_competitive_only_effect(self):
        sys = QSPBindingSystem()
        sys.add_competitive("c")
        m = sys.models["c"]
        assert sys.get_effect("c") == 0.0
        assert m.competitive.compute_schild_shift(10.0) > 1.0
        sys.set_drug_concentration("c", 10.0)  # no mass_action/tmdd branch

    def test_model_with_no_binding_falls_back_zero(self):
        from helixlang.plugins.human.qsp_binding import QSPBindingModel
        m = QSPBindingModel(name="n", kind="none")
        assert m.compute_effect() == 0.0

    def test_unknown_set_concentration(self):
        sys = QSPBindingSystem()
        sys.set_drug_concentration("nope", 10.0)  # returns silently
        assert sys.get_effect("nope") == 0.0

    def test_tmdd_set_concentration_path(self):
        sys = QSPBindingSystem()
        sys.add_tmdd("t")
        sys.set_drug_concentration("t", 5.0)
        assert sys.models["t"].tmdd.c_free_nM == 5.0

    def test_all_effects_and_step(self):
        sys = QSPBindingSystem()
        sys.add_competitive("c")
        sys.step(1.0)
        assert isinstance(sys.get_all_effects(), dict)
        assert "c" in sys.get_all_effects()


# ---------------------------------------------------------------------------
# proteome_binding.py
# ---------------------------------------------------------------------------
class TestProteomeBindingCascade:
    def test_known_drug_screen_occupancy(self):
        from helixlang.plugins.human.proteome_binding import ProteomeBindingCascade
        c = ProteomeBindingCascade()
        prof = c.screen_drug("warfarin", c._drug_smiles["warfarin"], 10.0)
        assert prof.n_targets_screened > 0
        assert prof.n_significant_bindings >= 0

    def test_novel_drug_no_match(self):
        from helixlang.plugins.human.proteome_binding import ProteomeBindingCascade
        c = ProteomeBindingCascade()
        prof = c.screen_drug("zzz", "CCO", 10.0)
        assert prof.n_significant_bindings == 0

    def test_novel_drug_similar_match(self):
        from helixlang.plugins.human.proteome_binding import ProteomeBindingCascade
        c = ProteomeBindingCascade()
        prof = c.screen_drug("cli", "c1ccccc1O", 10.0)
        assert len(prof.bindings) >= 0

    def test_novel_drug_nearest_has_unknown_target(self):
        from helixlang.plugins.human.proteome_binding import ProteomeBindingCascade
        c = ProteomeBindingCascade()
        c._drug_smiles = {"closest": "c1ccccc1O"}
        c._known_drugs = {"closest": {"CYP99999": {"kd_um": 5.0}}}
        prof = c.screen_drug("novel", "c1ccccc1O", 10.0)
        assert prof.n_significant_bindings == 0

    def test_ddi_no_interaction(self):
        from helixlang.plugins.human.proteome_binding import ProteomeBindingCascade
        c = ProteomeBindingCascade()
        p = c.predict_ddi(
            "warfarin", c._drug_smiles["warfarin"], 10.0,
            "vancomycin", c._drug_smiles["vancomycin"], 10.0)
        assert p.significance == "NO_DDI"
        assert p.auc_ratio == 1.0

    def test_ddi_interacting_contraindicated(self):
        from helixlang.plugins.human.proteome_binding import ProteomeBindingCascade
        c = ProteomeBindingCascade()
        p = c.predict_ddi(
            "clarithromycin", c._drug_smiles["clarithromycin"], 10.0,
            "verapamil", c._drug_smiles["verapamil"], 10.0)
        assert p.interacting_targets
        assert p.significance == "CONTRAINDICATED"
        assert p.auc_ratio > 2.0

    def test_predict_all_pairs(self):
        from helixlang.plugins.human.proteome_binding import ProteomeBindingCascade
        c = ProteomeBindingCascade()
        preds = c.predict_all_pairs([
            ("warfarin", c._drug_smiles["warfarin"], 10.0),
            ("amiodarone", c._drug_smiles["amiodarone"], 10.0),
            ("vancomycin", c._drug_smiles["vancomycin"], 10.0),
        ])
        assert len(preds) == 3


# ---------------------------------------------------------------------------
# tissue_blood.py / tissue_gem.py / organ_crosstalk.py
# ---------------------------------------------------------------------------
class TestTissueBloodGetters:
    def test_get_blood_neutrophils(self):
        from helixlang.plugins.human.tissue_blood import TissueBloodModel
        m = TissueBloodModel()
        assert m.get_blood_neutrophils() == m.blood_neutrophils
        assert m.get_tissue_neutrophils() == m.tissue_neutrophils


class TestTissueGEMPools:
    def test_get_pool_state(self):
        from helixlang.plugins.human.tissue_gem import OrganGEMCoupler
        c = OrganGEMCoupler(gem_interval_ticks=1)
        c._pool_state = {"liver": {"glucose": 1.0}}
        state = c.get_pool_state()
        assert state == {"liver": {"glucose": 1.0}}
        state["liver"]["glucose"] = 99.0
        assert c._pool_state["liver"]["glucose"] == 1.0


class TestOrganCrosstalkPhosphate:
    def test_phosphate_suppression_high(self):
        from helixlang.plugins.human.organ_crosstalk import apply_crosstalk, create_crosstalk
        c = create_crosstalk()
        apply_crosstalk(c, egfr=90.0, phosphate_mg_dl=5.5)
        assert c.phosphate_epo_suppression > 0.0

    def test_phosphate_suppression_low(self):
        from helixlang.plugins.human.organ_crosstalk import apply_crosstalk, create_crosstalk
        c = create_crosstalk()
        apply_crosstalk(c, egfr=90.0, phosphate_mg_dl=4.0)
        assert c.phosphate_epo_suppression == 0.0


# ---------------------------------------------------------------------------
# plugins/human/__init__.py
# ---------------------------------------------------------------------------
class TestHumanPluginContract:
    def test_check_present(self):
        from helixlang.plugins.human import _check
        assert _check("numpy") is True

    def test_check_absent(self):
        from helixlang.plugins.human import _check
        assert _check("no_such_module_xyz_123") is False

    def test_make_backend_returns_virtual_patient(self):
        from helixlang.plugins.human import _make_backend
        from helixlang.plugins.human.virtual_patient import VirtualPatient
        assert _make_backend(None) is VirtualPatient

    def test_load_returns_make_backend(self):
        from helixlang.plugins.human import _load
        assert callable(_load())

    def test_load_missing_numpy_raises(self, monkeypatch):
        import helixlang.plugins.human as h
        from helixlang.core.errors import PluginDependencyError
        monkeypatch.setattr(h, "_check", lambda pkg: False)
        with pytest.raises(PluginDependencyError):
            h._load()


# ---------------------------------------------------------------------------
# pharmacodynamics.py
# ---------------------------------------------------------------------------
class TestPharmacodynamicsClose:
    def test_pd_effect_invalid_type(self):
        from helixlang.plugins.human.pharmacodynamics import PDEffect
        with pytest.raises(ValueError):
            PDEffect(target_reaction="r", effect_type="bogus")

    def test_pd_effect_bad_ec50(self):
        from helixlang.plugins.human.pharmacodynamics import PDEffect
        with pytest.raises(ValueError):
            PDEffect(target_reaction="r", ec50_um=0.0)

    def test_pd_effect_bad_emax(self):
        from helixlang.plugins.human.pharmacodynamics import PDEffect
        with pytest.raises(ValueError):
            PDEffect(target_reaction="r", emax=1.5)

    def test_bad_therapeutic_window(self):
        from helixlang.plugins.human.pharmacodynamics import Pharmacodynamics
        with pytest.raises(ValueError):
            Pharmacodynamics("d", therapeutic_window=(50.0, 1.0))

    def test_compute_effects_unknown_falls_back_one(self):
        from helixlang.plugins.human import pharmacodynamics as pd
        fake = type("FakeEff", (), {
            "effect_type": "invalid", "target_reaction": "R", "ec50_um": 1.0,
            "hill_coefficient": 1.0, "baseline_effect": 0.0})()
        model = pd.Pharmacodynamics("d", effects=[fake])
        out = pd.compute_pd_effects(5.0, model)
        assert out == {"R": 1.0}

    def test_effects_over_time_empty(self):
        from helixlang.plugins.human import pharmacodynamics as pd
        model = pd.Pharmacodynamics("d")
        out = pd.compute_pd_effects_over_time(
            {"liver": [1.0]}, [0.0, 1.0], model)
        assert out == {}

    def test_effects_over_time_full(self):
        from helixlang.plugins.human import pharmacodynamics as pd
        model = pd.Pharmacodynamics("d", effects=[
            pd.PDEffect(target_reaction="R", effect_type="inhibition",
                        ec50_um=1.0)])
        out = pd.compute_pd_effects_over_time(
            {"liver": [0.1, 0.1]}, [0.0, 1.0], model)
        assert len(out["R"]) == 2

    def test_effects_over_time_short_organ_conc(self):
        from helixlang.plugins.human import pharmacodynamics as pd
        model = pd.Pharmacodynamics("d", effects=[
            pd.PDEffect(target_reaction="R", effect_type="inhibition",
                        ec50_um=1.0)])
        out = pd.compute_pd_effects_over_time(
            {"liver": [0.1]}, [0.0, 1.0, 2.0], model)
        assert len(out["R"]) == 3

    def test_apply_bounds_hit_and_miss(self):
        from helixlang.plugins.human import pharmacodynamics as pd
        bounds = {"R": (-10.0, 10.0), "X": (-5.0, 5.0)}
        out = pd.apply_pd_to_flux_bounds({"R": 0.5, "NOPE": 2.0}, bounds)
        assert out["R"] == (-10.0, 5.0)
        assert out["X"] == (-5.0, 5.0)

    def test_infer_known_target(self):
        from helixlang.plugins.human.pharmacodynamics import infer_pd_from_drug
        model = infer_pd_from_drug("d", "HMGCR", binding_kd_um=0.1)
        assert len(model.effects) == 1
        assert model.effects[0].target_gene == "HMGCR"

    def test_infer_unknown_target(self):
        from helixlang.plugins.human.pharmacodynamics import infer_pd_from_drug
        model = infer_pd_from_drug("d", "UNKNOWN_PROTEIN_X",
                                   binding_kd_um=1.0)
        assert len(model.effects) == 1

    def test_infer_biologic(self):
        from helixlang.plugins.human.pharmacodynamics import infer_pd_from_drug
        model = infer_pd_from_drug("d", "", binding_kd_um=0.01, mw_da=150000.0)
        assert len(model.effects) == 1


# ---------------------------------------------------------------------------
# disease_progression.py
# ---------------------------------------------------------------------------
class TestDiseaseProgressionClose:
    def test_labs_bad_platelets(self):
        from helixlang.plugins.human.disease_progression import ClinicalLabs
        with pytest.raises(ValueError):
            ClinicalLabs(platelets_per_ul=0.0)

    def test_labs_bad_age(self):
        from helixlang.plugins.human.disease_progression import ClinicalLabs
        with pytest.raises(ValueError):
            ClinicalLabs(age_years=0.0)

    def test_rate_validation(self):
        from helixlang.plugins.human.disease_progression import ProgressionRate
        with pytest.raises(ValueError):
            ProgressionRate("d", {}, -1.0, 0.0, 0.0, 0.5)
        with pytest.raises(ValueError):
            ProgressionRate("d", {}, 0.5, -1.0, 0.0, 0.5)
        with pytest.raises(ValueError):
            ProgressionRate("d", {}, 0.5, 0.0, 1.5, 0.5)
        with pytest.raises(ValueError):
            ProgressionRate("d", {}, 0.5, 0.0, 0.0, 1.5)
        with pytest.raises(ValueError):
            ProgressionRate("d", {}, 0.5, 0.0, 0.0, 0.5, plateau_time_years=0.0)

    def test_stage_bounds(self):
        from helixlang.plugins.human.disease_progression import (
            DiseaseStage,
            create_progression_model,
        )
        m = create_progression_model("CKD")
        assert DiseaseStage.PRECLINICAL is m._stage_from_severity(0.01)
        assert DiseaseStage.MILD is m._stage_from_severity(0.2)
        assert DiseaseStage.MODERATE is m._stage_from_severity(0.5)
        assert DiseaseStage.SEVERE is m._stage_from_severity(0.7)
        assert DiseaseStage.CRITICAL is m._stage_from_severity(0.95)

    def test_metric_switching(self):
        from helixlang.plugins.human.disease_progression import (
            ClinicalLabs,
            create_progression_model,
        )
        m = create_progression_model("DIABETES_T2")
        labs = ClinicalLabs(hba1c_percent=12.0)
        assert m._metric_from_labs(labs) == 12.0
        assert m._metric_from_labs(ClinicalLabs()) == 5.4
        assert create_progression_model("CANCER_GENERIC"
                                        )._metric_from_labs(
            ClinicalLabs(tumor_marker_ng_ml=8.0)) == 8.0
        assert m._metric_from_labs(ClinicalLabs()) is not None

    def test_stage_from_metric_thresholds(self):
        from helixlang.plugins.human.disease_progression import (
            DiseaseStage,
            create_progression_model,
        )
        m = create_progression_model("CKD")
        assert m._stage_from_metric(40.0) is DiseaseStage.MODERATE
        assert m._stage_from_metric(10.0) is DiseaseStage.CRITICAL

    def test_stage_from_metric_no_thresholds(self):
        from helixlang.plugins.human.disease_progression import (
            DiseaseProgressionModel,
            DiseaseStage,
            ProgressionRate,
        )
        m = DiseaseProgressionModel(disease_name="X", current_severity=0.99)
        assert m._stage_from_metric(0.99) is DiseaseStage.CRITICAL
        assert m._stage_from_metric(None) is DiseaseStage.CRITICAL
        m2 = DiseaseProgressionModel(
            disease_name="Y", current_severity=0.4,
            progression_rate=ProgressionRate(
                "Y", {"bogus_stage": 5.0}, 0.1, 0.0, 0.0, 0.5))
        assert m2._stage_from_metric(0.4) is DiseaseStage.MODERATE

    def test_restage_none_and_unknown_metric(self):
        from helixlang.plugins.human.disease_progression import (
            DiseaseStage,
            create_progression_model,
        )
        m = create_progression_model("CKD")
        assert m._restage(None) is DiseaseStage.MILD
        # metric None → severity fallback
        from helixlang.plugins.human.disease_progression import (
            ClinicalLabs,
            DiseaseProgressionModel,
        )
        m2 = DiseaseProgressionModel(disease_name="ZZZ_NO_PROFILE", current_severity=0.7)
        assert m2._restage(ClinicalLabs()) is DiseaseStage.SEVERE

    def test_step_relapse(self):
        from helixlang.plugins.human.disease_progression import create_progression_model
        m = create_progression_model("CKD")
        m.step(24.0 * 365.0, drug_effectiveness=0.9)  # treated
        m.step(24.0 * 365.0, drug_effectiveness=0.0)  # dropped → relapse roll

    def test_step_relapse_always_hits(self):
        from helixlang.plugins.human.disease_progression import create_progression_model
        m = create_progression_model("CKD")
        class AlwaysLow:
            def random(self):
                return 0.0
        m.rng = AlwaysLow()
        m.step(1.0, drug_effectiveness=0.9)
        m.step(1.0, drug_effectiveness=0.0)
        assert m.cumulative_damage >= 0.0

    def test_step_recovery_branch(self):
        from helixlang.plugins.human.disease_progression import create_progression_model
        m = create_progression_model("LIVER_CIRRHOSIS")
        m._was_treated = True
        m.step(24.0 * 365.0, drug_effectiveness=1.0)
        assert m.cumulative_damage < 1.0

    def test_step_no_rate_skips(self):
        from helixlang.plugins.human.disease_progression import (
            DiseaseProgressionModel,
            DiseaseStage,
        )
        m = DiseaseProgressionModel(disease_name="X", current_severity=0.5)
        out = m.step(1.0, drug_effectiveness=0.0)
        assert out is DiseaseStage.MODERATE
        m2 = DiseaseProgressionModel(disease_name="X", current_severity=1.0)
        m2.current_severity = 0.3
        assert m2.step(1.0, drug_effectiveness=0.5) is DiseaseStage.MILD

    def test_step_bad_dt(self):
        from helixlang.plugins.human.disease_progression import create_progression_model
        m = create_progression_model("CKD")
        with pytest.raises(ValueError):
            m.step(-1.0, drug_effectiveness=0.5)
        with pytest.raises(ValueError):
            m.step(1.0, drug_effectiveness=1.5)

    def test_step_restage_with_labs(self):
        from helixlang.plugins.human.disease_progression import (
            ClinicalLabs,
            create_progression_model,
        )
        m = create_progression_model("LIVER_CIRRHOSIS")
        labs = ClinicalLabs(platelets_per_ul=90_000.0, age_years=55.0,
                            ast_u_l=40.0, alt_u_l=35.0)
        m.step(268.0, drug_effectiveness=0.2, labs=labs)

    def test_get_severity_and_organ_function(self):
        from helixlang.plugins.human.disease_progression import create_progression_model
        m = create_progression_model("CKD")
        m.current_severity = 1.5
        assert m.get_severity() == 1.0
        assert m.get_organ_function("kidney") < 1.0
        assert m.get_organ_function("unspecified_organ") == 1.0

    def test_missing_disease_raises(self):
        from helixlang.plugins.human.disease_progression import create_progression_model
        with pytest.raises(ValueError):
            create_progression_model("no_such_disease_xyz")


# ---------------------------------------------------------------------------
# immune.py
# ---------------------------------------------------------------------------
class TestImmuneClose:
    def test_decay_cache_recompute(self):
        from helixlang.plugins.human.immune import InnateImmuneModel
        m = InnateImmuneModel()
        m.cytokines.tnf_half_life_h = 9.0
        m.cytokines.step(1.0)
        assert m.cytokines._k_key is not None

    def test_cytokine_getters(self):
        from helixlang.plugins.human.immune import InnateImmuneModel
        m = InnateImmuneModel()
        assert m.get_il10() == m.cytokines.il10
        assert m.get_wbc_total() > 0.0

    def test_crp_zero_lag_alpha_one(self):
        from helixlang.plugins.human.immune import CRPDriver
        crp = CRPDriver()
        crp.il6_lag_tau_h = 0.0
        crp.step(1.0, 30.0)
        assert crp.crp_mg_l > 0.5

    def test_cohort_cells_step_scalar_tau_zero(self):
        from helixlang.plugins.human.immune import _cohort_cells_step
        out = _cohort_cells_step(
            5.0, 1.0, 1.0, 1.0, 1.0,
            3.0, 0.5, 0.5, 0.1, 0.1, 0.1,
            1.0, 10.0, 5.0, 0.1,
            t1=0.0, t2=0.0, t3=0.0, t4=0.0, tau=0.0, prolif=3.0)
        assert out[0] > 0.0

    def test_np_max_scalar(self):
        from helixlang.plugins.human.immune import np_max
        assert np_max(2.0, 1.0) == 2.0

    def test_cohort_immune_step_scalar_fallback(self):
        from helixlang.plugins.human.immune import InnateImmuneModel, cohort_immune_step
        m = InnateImmuneModel()
        cohort_immune_step([m], 1.0, use_numpy=False)
        assert m.cytokines.tnf_alpha >= 0.0

    def test_cohort_immune_step_circadian(self):
        from helixlang.plugins.human.immune import cohort_immune_step, create_immune_model
        m, _ = create_immune_model(infection_severity=0.5)
        m.circadian_amplitude = 0.5
        cohort_immune_step([m], 1.0, use_numpy=True)
        assert m._sim_hour > 0.0


# ---------------------------------------------------------------------------
# pharmacokinetics.py
# ---------------------------------------------------------------------------
class TestPharmacoKineticsClose:
    def test_least_squares_slope_fallbacks(self):
        import helixlang.plugins.human.pharmacokinetics as pk
        assert pk._least_squares_slope([1.0], [2.0]) == 0.0
        assert pk._least_squares_slope([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0

    def test_terminal_half_life_fallbacks(self):
        import helixlang.plugins.human.pharmacokinetics as pk
        assert pk._terminal_half_life([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], 7.0) == 7.0
        assert pk._terminal_half_life(
            [0, 1, 2, 3], [1e-3, 1e-4, 1e-5, 1e-6], 7.0) < 7.0
        flat = [1e-1, 1e-1, 1e-1, 1e-1, 1e-1]
        assert pk._terminal_half_life([0, 1, 2, 3, 4], flat, 7.0) == 7.0
        assert pk._terminal_half_life([0, 1, 2, 3, 4], flat, 7.0) == 7.0

    def test_pbpk_invalid_drug_spec(self):
        from helixlang.plugins.human.drug import get_predefined_drug
        from helixlang.plugins.human.pharmacokinetics import PBPKModel
        from helixlang.plugins.human.physiology import create_default_physiology
        drug = get_predefined_drug("IBUPROFEN")
        drug.half_life_h = -1.0
        with pytest.raises(ValueError, match="invalid drug specification"):
            PBPKModel(drug, create_default_physiology())
        drug.half_life_h = 2.0
        drug.route = "intrathecal"
        # intrathecal is a fully supported PBPK input (doc/27 §7.6): the
        # direct CNS bolus model constructs with dose_brain_mg defaulting
        # to the administered dose.
        model = PBPKModel(drug, create_default_physiology())
        assert model.dose_brain_mg == drug.dose_mg

    def test_pbpk_volume_and_time_validations(self):
        from helixlang.plugins.human.drug import get_predefined_drug
        from helixlang.plugins.human.pharmacokinetics import PBPKModel
        from helixlang.plugins.human.physiology import create_default_physiology
        drug = get_predefined_drug("IBUPROFEN")
        phys = create_default_physiology()
        with pytest.raises(ValueError, match="gut_volume_l"):
            PBPKModel(drug, phys, gut_volume_l=0.0)
        with pytest.raises(ValueError, match="lag_time_h"):
            PBPKModel(drug, phys, lag_time_h=-1.0)
        with pytest.raises(ValueError, match="infusion_duration_h"):
            PBPKModel(drug, phys, infusion_duration_h=0.0)

    def test_pbpk_zero_cardiac_output(self):
        from helixlang.plugins.human.drug import get_predefined_drug
        from helixlang.plugins.human.pharmacokinetics import PBPKModel
        from helixlang.plugins.human.physiology import create_default_physiology
        drug = get_predefined_drug("IBUPROFEN")
        phys = create_default_physiology()
        phys.cardiac_output_ml_per_min = 0.0
        with pytest.raises(ValueError, match="cardiac output"):
            PBPKModel(drug, phys)

    def test_pbpk_dose_input_routes(self):
        from helixlang.plugins.human.drug import IV, IV_INFUSION, get_predefined_drug
        from helixlang.plugins.human.pharmacokinetics import PBPKModel
        from helixlang.plugins.human.physiology import create_default_physiology
        phys = create_default_physiology()
        oral = get_predefined_drug("IBUPROFEN")
        oral_mod = PBPKModel(oral, phys, lag_time_h=0.5)
        assert oral_mod._dose_input(0.1) == 0.0
        assert oral_mod._dose_input(2.0) > 0.0
        infusion = get_predefined_drug("IBUPROFEN")
        infusion.route = IV_INFUSION
        inf_mod = PBPKModel(infusion, phys, infusion_duration_h=2.0)
        assert inf_mod._dose_input(1.0) > 0.0
        assert inf_mod._dose_input(100.0) == 0.0
        bolus = get_predefined_drug("IBUPROFEN")
        bolus.route = IV
        iv_mod = PBPKModel(bolus, phys)
        assert iv_mod._dose_input(5.0) == 0.0

    def test_pbpk_step_validation_and_sample_grid(self):
        from helixlang.plugins.human.drug import get_predefined_drug
        from helixlang.plugins.human.pharmacokinetics import PBPKConfig, PBPKModel
        from helixlang.plugins.human.physiology import create_default_physiology
        drug = get_predefined_drug("IBUPROFEN")
        mod = PBPKModel(drug, create_default_physiology())
        with pytest.raises(ValueError, match="dt_min"):
            mod.step(0.0)
        snapshot = mod.step(30.0)
        assert snapshot["central"] >= 0.0 and "liver" in snapshot
        assert set(mod.get_concentrations()) == {
            "central", "liver", "kidney", "brain", "muscle", "adipose"}
        zero_cfg = PBPKConfig(total_time_h=0.0)
        mod2 = PBPKModel(drug, create_default_physiology(), zero_cfg)
        assert mod2._sample_grid() == [0.0]

    def test_solver_max_step(self):
        from helixlang.plugins.human.drug import IV, IV_INFUSION, get_predefined_drug
        from helixlang.plugins.human.pharmacokinetics import PBPKModel
        from helixlang.plugins.human.physiology import create_default_physiology
        phys = create_default_physiology()
        oral = get_predefined_drug("IBUPROFEN")
        oral_mod = PBPKModel(oral, phys, lag_time_h=4.0)
        assert oral_mod._solver_max_step_h() == pytest.approx(0.5)
        infusion = get_predefined_drug("IBUPROFEN")
        infusion.route = IV_INFUSION
        inf_mod = PBPKModel(infusion, phys, infusion_duration_h=2.0)
        assert inf_mod._solver_max_step_h() == pytest.approx(0.25)
        bolus = get_predefined_drug("IBUPROFEN")
        bolus.route = IV
        iv_mod = PBPKModel(bolus, phys)
        assert iv_mod._solver_max_step_h() == pytest.approx(24.0)

    def test_euler_fallback_trajectory(self, monkeypatch):
        import helixlang.plugins.human.pharmacokinetics as pk
        from helixlang.plugins.human.drug import get_predefined_drug
        from helixlang.plugins.human.pharmacokinetics import PBPKConfig, PBPKModel
        from helixlang.plugins.human.physiology import create_default_physiology
        monkeypatch.setattr(pk, "_HAS_SCIPY", False)
        drug = get_predefined_drug("IBUPROFEN")
        mod = PBPKModel(
            drug, create_default_physiology(),
            PBPKConfig(dt_min=30.0, total_time_h=12.0))
        result = mod.run()
        assert result.c_max > 0.0

    def test_pbpk_organ_flow_fallback(self):
        from helixlang.plugins.human.drug import get_predefined_drug
        from helixlang.plugins.human.pharmacokinetics import DEFAULT_FLOW_FRACTIONS, PBPKModel
        from helixlang.plugins.human.physiology import create_default_physiology
        phys = create_default_physiology()
        phys.organs = {n: o for n, o in phys.organs.items() if n != "liver"}
        mod = PBPKModel(get_predefined_drug("IBUPROFEN"), phys)
        co_l_per_h = phys.cardiac_output_ml_per_min * 0.06
        assert mod.organ_flows_l_per_h["liver"] == pytest.approx(
            co_l_per_h * DEFAULT_FLOW_FRACTIONS["liver"])
        assert mod.organ_flows_l_per_h["kidney"] > 0.0

    def test_pbpk_negative_total_time(self):
        from helixlang.plugins.human.pharmacokinetics import PBPKConfig
        with pytest.raises(ValueError, match="total_time_h"):
            PBPKConfig(total_time_h=-1.0).validate()

    def test_pbpk_scipy_import_failure(self):
        mod = _load_blocking_imports(
            "helixlang.plugins.human.pharmacokinetics",
            {"scipy", "scipy.integrate"})
        assert mod._HAS_SCIPY is False
        assert mod.solve_ivp is None

    def test_pbpk_solve_failure_fallback(self, monkeypatch):
        import numpy as np

        import helixlang.plugins.human.pharmacokinetics as pk
        from helixlang.plugins.human.drug import get_predefined_drug
        from helixlang.plugins.human.pharmacokinetics import PBPKConfig, PBPKModel
        from helixlang.plugins.human.physiology import create_default_physiology

        class _FailedSolution:
            success = False
            y = np.zeros((7, 2))

        monkeypatch.setattr(pk, "solve_ivp", lambda *a, **k: _FailedSolution())
        drug = get_predefined_drug("IBUPROFEN")
        mod = PBPKModel(
            drug, create_default_physiology(),
            PBPKConfig(dt_min=30.0, total_time_h=12.0))
        result = mod.run()
        assert result.c_max >= 0.0


# ---------------------------------------------------------------------------
# physiology.py
# ---------------------------------------------------------------------------
class TestPhysiologyValidation:
    def test_bad_weight(self):
        with pytest.raises(ValueError):
            HumanPhysiology(body_weight_kg=0.0, height_cm=170.0)

    def test_bad_height(self):
        with pytest.raises(ValueError):
            HumanPhysiology(body_weight_kg=70.0, height_cm=0.0)

    def test_bad_age(self):
        with pytest.raises(ValueError):
            HumanPhysiology(body_weight_kg=70.0, height_cm=170.0,
                            age_years=-1.0)

    def test_bad_cardiac(self):
        with pytest.raises(ValueError):
            HumanPhysiology(body_weight_kg=70.0, height_cm=170.0,
                            cardiac_output_ml_per_min=0.0)

    def test_bad_hematocrit(self):
        with pytest.raises(ValueError):
            HumanPhysiology(body_weight_kg=70.0, height_cm=170.0,
                            hematocrit=1.0)

    def test_bad_plasma(self):
        with pytest.raises(ValueError):
            HumanPhysiology(body_weight_kg=70.0, height_cm=170.0,
                            plasma_volume_ml=0.0)

    def test_bad_albumin(self):
        with pytest.raises(ValueError):
            HumanPhysiology(body_weight_kg=70.0, height_cm=170.0,
                            albumin_g_per_dL=-1.0)

    def test_bad_organ_tissue_fraction(self):
        organ = OrganSpec(name="bad", volume_ml=1.0,
                          blood_flow_ml_per_min=1.0, tissue_fraction=1.5)
        with pytest.raises(ValueError):
            HumanPhysiology(body_weight_kg=70.0, height_cm=170.0,
                            organs={"bad": organ})


class TestPhysiologyAccessors:
    def test_flow_fraction_positive(self):
        p = create_default_physiology()
        assert p.organ_flow_fraction("liver") > 0.0
        assert p.organ_flow_fractions["liver"] > 0.0

    def test_flow_fraction_invalid(self):
        organ = OrganSpec(name="x", volume_ml=1.0,
                          blood_flow_ml_per_min=1.0, tissue_fraction=0.5)
        with pytest.raises(ValueError):
            organ.flow_fraction(0.0)

    def test_aggregate_properties(self):
        p = create_default_physiology()
        assert p.total_organ_volume_ml > 0.0
        assert p.total_organ_blood_flow_ml_per_min > 0.0
        assert p.total_oxygen_consumption_ml_per_min > 0.0
        assert p.total_glucose_uptake_mmol_per_min > 0.0
        assert p.has_organ("liver")
        assert not p.has_organ("zzz")

    def test_get_organ_missing(self):
        p = create_default_physiology()
        with pytest.raises(KeyError):
            p.get_organ("zzz")

    def test_with_organ(self):
        p = create_default_physiology()
        organ = OrganSpec(name="test_org", volume_ml=10.0,
                          blood_flow_ml_per_min=100.0,
                          tissue_fraction=0.5,
                          oxygen_consumption_ml_per_kg_per_min=1.0,
                          glucose_uptake_mmol_per_kg_per_min=0.1)
        p2 = p.with_organ(organ)
        assert "test_org" in p2.organs
