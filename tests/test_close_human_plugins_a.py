"""Coverage-closing tests for 12 human plugin modules (batch A).

Raises branch+line coverage to 100% for:
adaptive, immune, spatial_abm, tissue_blood, tissue_gem, renal_model,
patient_params, phenotype, pharmacogenomic_ae, ddi,
physiology_constraints and proteome_binding.

Each test targets a specific uncovered statement/branch via direct
function/class invocation; the handful of otherwise-unreachable branches
(e.g. the ``break`` in ``run_cohort``'s merge loop and the empty-grams
guard in ``_compute_similarity_fallback``) are hit with a minimal, fully
restored stand-in for the surrounding machinery.
"""
from __future__ import annotations

import json
import math
import multiprocessing

import numpy as np
import pytest

import helixlang.plugins.human.patient_params as patient_params
import helixlang.plugins.human.proteome_binding as proteome_binding
import helixlang.plugins.human.spatial_abm as spatial_abm
from helixlang.plugins.human.adaptive import (
    AdaptiveImmuneModel,
    PD1Checkpoint,
    cohort_adaptive_step,
)
from helixlang.plugins.human.ddi import DDIModel, DDIRule, assess_additive_toxicity
from helixlang.plugins.human.genotype import GenotypeProfile
from helixlang.plugins.human.immune import (
    CRPDriver,
    CytokinePool,
    InnateImmuneModel,
    _cohort_cells_step,
    cohort_immune_step,
    np_max,
    run_cohort,
)
from helixlang.plugins.human.patient_params import (
    N_PARAMS,
    ParamSlice,
    PatientParameterSet,
    PatientParameterTable,
    nominal_params,
    to_array,
)
from helixlang.plugins.human.pharmacogenomic_ae import AERisk, GenotypeAEPredictor
from helixlang.plugins.human.phenotype import (
    ExternalTraits,
    PhenotypeCalculator,
    _alcohol_cyp_factor,
)
from helixlang.plugins.human.physiology_constraints import (
    PhysiologyConstraints,
    ThermodynamicChecker,
)
from helixlang.plugins.human.renal_model import DiseaseStage, RenalFunctionModel
from helixlang.plugins.human.spatial_abm import (
    AgentState,
    CellType,
    SpatialABMConfig,
    SpatialAgentGrid,
    TissueAgent,
    run_spatial_abm,
)
from helixlang.plugins.human.tissue_blood import (
    TissueBloodModel,
    cohort_tissue_blood_step,
)
from helixlang.plugins.human.tissue_gem import (
    TISSUE_REACTION_SETS,
    GEMDecomposer,
    OrganGEMCoupler,
)


class TestAdaptiveCloseGaps:
    def test_effective_blockade_relieves_brake(self):
        checkpoint = PD1Checkpoint()
        assert checkpoint.effective_blockade() == pytest.approx(
            1.0 - checkpoint.immune_brake()
        )

    def test_default_numpy_path(self):
        model = AdaptiveImmuneModel()
        titers = cohort_adaptive_step([model], 1.0, [0.0])
        assert titers == [pytest.approx(model.get_total_antibody())]

    def test_scalar_fallback(self):
        models = [AdaptiveImmuneModel(), AdaptiveImmuneModel()]
        cohort_adaptive_step(models, 1.0, [0.0, 0.0], use_numpy=False)
        assert models[0].effector_cd4 == 0.0

    def test_vectorized_with_doses(self):
        model = AdaptiveImmuneModel()
        titers = cohort_adaptive_step([model], 1.0, [0.0], doses=[1.0])
        assert len(titers) == 1


class TestImmuneCloseGaps:
    def test_decay_cache_recomputed_after_half_life_change(self):
        pool = CytokinePool()
        pool.tnf_half_life_h = 100.0
        pool.step(1.0)
        assert pool._k_key[0] == 100.0

    def test_accessors(self):
        model = InnateImmuneModel()
        assert model.get_il10() == model.cytokines.il10
        assert model.get_wbc_total() == model.cells.get_wbc_total()

    def test_crp_no_lag(self):
        driver = CRPDriver(il6_lag_tau_h=0.0)
        driver.step(1.0, 10.0)
        assert driver.crp_mg_l > 0.5

    def test_cytokine_step_no_friberg(self):
        out = _cohort_cells_step(
            neut=1.0, macro=0.5, mono=0.25, dc=0.1, tcells=0.8,
            prod_neut=0.1, prod_mono=0.05, prod_tcell=0.02,
            clr_neut=0.05, clr_mono=0.04, clr_macro=0.03,
            gcsf=0.5, il6=5.0, tnf=3.0, dt_h=1.0, tau=0.0,
        )
        assert out[0] > 0.0

    def test_np_max_scalar(self):
        assert np_max(0.0, 5.0) == 5.0

    def test_empty_cohort_returns(self):
        cohort_immune_step([], 1.0, use_numpy=True)

    def test_circadian_cortisol_vectorized(self):
        model = InnateImmuneModel(circadian_amplitude=0.6)
        cohort_immune_step([model], 1.0, use_numpy=True)

    def test_run_cohort_no_steps_returns(self):
        run_cohort([InnateImmuneModel()], 0, 1.0, workers=1)

    def test_run_cohort_single_worker(self):
        model = InnateImmuneModel()
        run_cohort([model], 1, 1.0, workers=1)

    def test_run_cohort_single_slab_fallback(self):
        model = InnateImmuneModel()
        run_cohort([model], 1, 1.0, workers=2)

    @staticmethod
    def _fake_multiprocessing(monkeypatch, models):
        model = models[0]

        class _FakePool:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            @staticmethod
            def starmap(func, args):
                return [[model], []]

        class _FakeContext:
            @staticmethod
            def Pool(processes=1):
                return _FakePool()

        monkeypatch.setattr(
            multiprocessing, "get_context", lambda method: _FakeContext()
        )

    def test_run_cohort_merge_break(self, monkeypatch):
        models = [InnateImmuneModel(), InnateImmuneModel()]
        self._fake_multiprocessing(monkeypatch, models)
        run_cohort(models, 1, 1.0, workers=2)


class TestSpatialAbmCloseGaps:
    def test_pos(self):
        agent = TissueAgent(uid=0, cell_type=CellType.TCELL, x=3, y=4)
        assert agent.pos() == (3, 4)

    def test_numpy_missing_raises(self, monkeypatch):
        monkeypatch.setattr(spatial_abm, "_HAS_NUMPY", False)
        with pytest.raises(RuntimeError):
            SpatialAgentGrid(SpatialABMConfig())

    def test_homogeneous_tissue(self):
        grid = SpatialAgentGrid(SpatialABMConfig(tissue_heterogeneity=False))
        assert grid._wound is None
        assert grid.tissue.sum() == 0.0

    def test_epithelial_agent_does_not_migrate(self):
        agent = TissueAgent(
            uid=0, cell_type=CellType.EPITHELIAL, x=2, y=2
        )
        grid = SpatialAgentGrid(SpatialABMConfig(), agents=[agent])
        grid.step()
        assert grid.agents[0].lifetime_h > 0.0

    @staticmethod
    def _grid_with_agent(state, activation=0.0):
        return SpatialAgentGrid(
            SpatialABMConfig(),
            agents=[
                TissueAgent(
                    uid=0,
                    cell_type=CellType.TCELL,
                    x=2,
                    y=2,
                    state=state,
                    activation=activation,
                )
            ],
        )

    def test_exhausted_cell_clears_to_apoptotic(self):
        grid = self._grid_with_agent(AgentState.EXHAUSTED)
        grid._age_states(100.0)
        assert grid.agents[0].state == AgentState.APOPTOTIC

    def test_exhausted_cell_persists(self):
        grid = self._grid_with_agent(AgentState.EXHAUSTED)
        grid._age_states(0.0)
        assert grid.agents[0].state == AgentState.EXHAUSTED

    def test_activated_cell_decays_to_resting(self):
        grid = self._grid_with_agent(AgentState.ACTIVATED, activation=0.2)
        grid._age_states(10.0)
        assert grid.agents[0].state == AgentState.RESTING

    def test_steps_negative_uses_config_max_steps(self):
        grid = run_spatial_abm(SpatialABMConfig(max_steps=3), steps=-1)
        assert grid.step_index == 3

    def test_activated_tcell_count(self):
        grid = self._grid_with_agent(AgentState.ACTIVATING)
        assert grid.activated_tcells() == 1


class TestTissueBloodCloseGaps:
    def test_scalar_fallback(self):
        models = [TissueBloodModel(), TissueBloodModel()]
        out = cohort_tissue_blood_step(
            models, 1.0, [0.0, 0.0], [5.0, 5.0], [4.0, 4.0], [0.4, 0.4],
            use_numpy=False,
        )
        assert len(out) == 2

    def test_default_numpy_path(self):
        models = [TissueBloodModel(), TissueBloodModel()]
        out = cohort_tissue_blood_step(
            models, 1.0, [0.0, 0.0], [5.0, 5.0], [4.0, 4.0], [0.4, 0.4],
        )
        assert len(out) == 2

    def test_blood_il6_accessor(self):
        assert TissueBloodModel().get_blood_il6() == 1.0


class TestTissueGemCloseGaps:
    def test_global_model_json_load(self, tmp_path):
        model_json = tmp_path / "model.json"
        model_json.write_text(
            json.dumps({"reactions": [{"id": "PGI"}, {"id": "PFK"}]})
        )
        decomposer = GEMDecomposer(str(model_json))
        assert decomposer._global_reactions == {"PGI", "PFK"}

    def test_global_model_non_json_ignored(self, tmp_path):
        model_xml = tmp_path / "model.xml"
        model_xml.write_text("<model/>")
        decomposer = GEMDecomposer(str(model_xml))
        assert decomposer._global_reactions == set()

    def test_prune_high_expression_reactions(self):
        liver = TISSUE_REACTION_SETS["liver"]["reactions"]
        decomposer = GEMDecomposer()
        gem = decomposer.decompose("liver", {r: 100.0 for r in liver})
        assert gem.reactions == set(liver) | {
            "PGI", "PFK", "FBA", "CS", "ACONTa", "MDH", "PK"
        } & set(liver)

    def test_incremental_exchange_rebuilds_missing_organ(self):
        coupler = OrganGEMCoupler(gem_interval_ticks=2)
        coupler.step(1.0, {"liver": {"glucose": 5.0}})
        coupler._pool_state.pop("brain")
        state = coupler.step(1.0, {"liver": {"glucose": 5.0}})
        assert "brain" in state


class TestRenalModelCloseGaps:
    def test_zero_weight_rejected(self):
        with pytest.raises(ValueError):
            RenalFunctionModel(weight_kg=0.0)

    def test_drug_channel_toggles(self):
        model = RenalFunctionModel()
        model.start_sglt2i()
        model.stop_sglt2i()
        assert model.sglt2i_active is False
        model.start_raas_blockade()
        model.stop_raas_blockade()
        assert model.raas_blockade_active is False
        model.set_nsaid(True)
        assert model.nsaid_active is True
        model.set_loop_diuretic(True)
        assert model.loop_diuretic_active is True

    def test_nsaid_penalties(self):
        model = RenalFunctionModel()
        model.start_raas_blockade()
        model.set_nsaid(True)
        model.set_loop_diuretic(True)
        assert model._nsaid_penalty_fraction() == pytest.approx(0.15 + 0.08 + 0.12)

    def test_g5_critical_and_krt_inf(self):
        model = RenalFunctionModel()
        model.serum_creatinine = 8.0
        assert model.kdigo_g_stage() == "G5"
        assert model.to_disease_stage() == DiseaseStage.CRITICAL
        assert model.time_to_krt_years() == math.inf

    def test_a2_category(self):
        model = RenalFunctionModel()
        model.uacr_mg_g = 100.0
        assert model.kdigo_a_category() == "A2"

    def test_moderate_stage(self):
        model = RenalFunctionModel()
        model.serum_creatinine = 2.2
        assert model.kdigo_g_stage() == "G3b"
        assert model.to_disease_stage() == DiseaseStage.MODERATE

    def test_acid_base_summary(self):
        assert set(RenalFunctionModel().acid_base_summary()) == {
            "ph", "bicarbonate_meq_per_l", "paco2_mmhg"
        }


class TestPatientParamsCloseGaps:
    def test_slice_extract(self):
        assert ParamSlice("proliferation", 0, 48).extract(
            nominal_params()
        )[:2] == [0.06, 0.07]

    def test_raw_array_construction(self):
        raw = np.zeros(N_PARAMS)
        assert len(list(PatientParameterSet(raw=raw)._params)) == N_PARAMS

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError):
            PatientParameterSet(params=[1.0, 2.0])

    def test_domain_and_slice(self):
        params = PatientParameterSet()
        assert len(params.domain("cytokine")) == 96
        assert params.slice("pd1").domain == "pd1"

    def test_copy(self):
        params = PatientParameterSet()
        assert len(list(params.copy().to_list())) == N_PARAMS

    def test_prior(self):
        assert PatientParameterSet.prior(0.1)[:2] == [0.1, 0.1]

    def test_table_names(self):
        assert PatientParameterTable().names() == []

    def test_to_array(self):
        assert to_array(nominal_params()).shape == (N_PARAMS,)

    def test_to_array_without_numpy(self, monkeypatch):
        monkeypatch.setattr(patient_params, "_np", None)
        with pytest.raises(RuntimeError):
            to_array(nominal_params())


class TestPhenotypeCloseGaps:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"body_weight_kg": 0.0},
            {"height_cm": 0.0},
            {"age_years": -1.0},
            {"sex": "other"},
            {"ethnicity": "unknown"},
            {"smoking_status": "heavy"},
            {"exercise_level": "extreme"},
            {"pack_years": -1.0},
            {"gestational_weeks": -1.0},
            {"pregnant": True, "gestational_weeks": 0.0},
        ],
    )
    def test_external_traits_validation(self, kwargs):
        with pytest.raises(ValueError):
            ExternalTraits(**kwargs)

    def test_vigorous_exercise_muscle_scaling(self):
        calculator = PhenotypeCalculator(
            genotype=GenotypeProfile(),
            traits=ExternalTraits(exercise_level="vigorous"),
        )
        assert calculator.compute_physiology().cardiac_output_ml_per_min == pytest.approx(
            5400.0
        )

    def test_alcohol_cyp_induction_heavy(self):
        assert _alcohol_cyp_factor("CYP2E1", 20.0) == pytest.approx(1.5)

    def test_alcohol_cyp_induction_moderate(self):
        assert _alcohol_cyp_factor("CYP2E1", 10.0) == pytest.approx(1.2)

    def test_alcohol_cyp_other_enzyme_unchanged(self):
        assert _alcohol_cyp_factor("CYP1A2", 20.0) == 1.0


class TestPharmacogenomicAECloseGaps:
    @staticmethod
    def _predict(concentration, activities):
        return GenotypeAEPredictor().predict_ae(
            "acetaminophen", concentration, activities
        )[0]

    def test_low_risk_and_pm_production(self):
        prediction = self._predict(20.0, {"CYP2E1": 0.2})
        assert prediction.risk_level is AERisk.LOW
        assert "PM phenotype" in prediction.mechanism

    def test_minimal_risk(self):
        prediction = self._predict(-400.0, {"CYP2E1": 1.0})
        assert prediction.ae_probability < 0.05

    def test_um_production_mechanism(self):
        prediction = self._predict(20.0, {"CYP2E1": 2.0})
        assert "UM phenotype" in prediction.mechanism

    def test_pm_detoxification_mechanism(self):
        prediction = self._predict(20.0, {"CYP2E1": 1.0, "GST": 0.1})
        assert "PM phenotype for GST" in prediction.mechanism


class TestDDICloseGaps:
    def test_invalid_interaction_type_rejected(self):
        with pytest.raises(ValueError):
            DDIRule("a", "b", "CYP3A4", "unknown", 0.5)

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValueError):
            DDIRule("a", "b", "CYP3A4", "inhibition", 0.5, severity="fatal")

    def test_invalid_fold_change_rejected(self):
        with pytest.raises(ValueError):
            DDIRule("a", "b", "CYP3A4", "inhibition", 0.0)

    def test_enzyme_state_fallback_effect(self):
        model = DDIModel(
            rules=[DDIRule("a", "CYP2D6", "CYP2D6", "induction", 2.0)]
        )
        alerts = model.get_clinical_alerts(["a"], {"CYP2D6": 2.0})
        assert len(alerts) == 1
        assert "CYP2D6 activity" in alerts[0]["effect"]

    def test_drug_rule_fallback_effect(self):
        model = DDIModel(
            rules=[
                DDIRule(
                    "b", "c", "CYP3A4", "inhibition", 0.5, severity="mild"
                )
            ]
        )
        alerts = model.get_clinical_alerts(["b", "c"], {})
        assert len(alerts) == 1
        assert "inhibits" in alerts[0]["effect"]

    def test_duplicate_rule_skipped(self):
        rule = DDIRule("a", "b", "CYP3A4", "inhibition", 0.5)
        model = DDIModel(rules=[rule, rule])
        assert len(model.get_clinical_alerts(["a", "b"], {})) == 1

    def test_additive_nephrotoxicity(self):
        alerts = assess_additive_toxicity(["vancomycin", "gentamicin"])
        assert any(a["toxicity_type"] == "nephrotoxicity" for a in alerts)

    def test_metformin_renal_risk(self):
        alerts = assess_additive_toxicity(["metformin", "vancomycin"])
        assert any(a["toxicity_type"] == "nephrotoxicity" for a in alerts)

    def test_additive_hepatotoxicity(self):
        alerts = assess_additive_toxicity(["imatinib", "tamoxifen"])
        assert any(a["toxicity_type"] == "hepatotoxicity" for a in alerts)


class TestPhysiologyConstraintsCloseGaps:
    def test_homeostatic_penalty_missing_vars(self):
        constraints = PhysiologyConstraints()
        assert constraints.homeostatic_penalty({"ph": 7.4}) == 0.0

    def test_unknown_reaction_returns_none(self):
        assert ThermodynamicChecker().check_reaction("not_a_reaction") is None

    def test_irreversible_non_spontaneous_flag_no_concentrations(self):
        checker = ThermodynamicChecker()
        checker.REACTION_DG0 = {"hexokinase": 5.0}
        checker.IRREVERSIBLE = {"hexokinase"}
        violation = checker.check_reaction("hexokinase")
        assert violation is not None

    def test_irreversible_saturated_no_spontaneous(self):
        checker = ThermodynamicChecker()
        violation = checker.check_reaction(
            "hexokinase",
            {"g6p": 1.0, "glucose": 1e-6},
            [1.0, -1.0],
            ["g6p", "glucose"],
        )
        assert violation is not None

    def test_check_all_else_branch_appends(self):
        checker = ThermodynamicChecker()
        checker.REACTION_DG0 = {"hexokinase": 5.0}
        checker.IRREVERSIBLE = {"hexokinase"}
        result = checker.check_all({"hexokinase": 1.0})
        assert len(result.violations) == 1


class TestProteomeBindingCloseGaps:
    def test_similarity_fallback_too_short(self):
        assert proteome_binding._compute_similarity_fallback("ab", "abcdef") == 0.0

    @staticmethod
    def _shrinking_sequence():
        class _ShrinkingSequence:
            def __init__(self):
                self.calls = 0

            def __len__(self):
                self.calls += 1
                return 3 if self.calls == 1 else 2

            def __getitem__(self, key):
                return ""

        return _ShrinkingSequence()

    def test_similarity_fallback_empty_grams(self):
        assert proteome_binding._compute_similarity_fallback(
            self._shrinking_sequence(), "abcdef"
        ) == 0.0

    def test_rdkit_exception_falls_back(self, monkeypatch):
        from rdkit.Chem import AllChem

        def _boom(*args, **kwargs):
            raise RuntimeError("forced rafale")

        monkeypatch.setattr(AllChem, "GetMorganFingerprintAsBitVect", _boom)
        assert proteome_binding._compute_similarity_rdkit("CCO", "CCO") > 0.0

    def test_contraindicated_ddi(self):
        cascade = proteome_binding.ProteomeBindingCascade()
        prediction = cascade.predict_ddi(
            "fluconazole",
            "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",
            100.0,
            "warfarin",
            "CC(=O)Cc1ccccc1C(=O)O",
            100.0,
        )
        assert prediction.significance == "CONTRAINDICATED"

    def test_all_pairs(self):
        cascade = proteome_binding.ProteomeBindingCascade()
        predictions = cascade.predict_all_pairs(
            [
                ("fluconazole", "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F", 10.0),
                ("warfarin", "CC(=O)Cc1ccccc1C(=O)O", 10.0),
            ]
        )
        assert len(predictions) == 1
