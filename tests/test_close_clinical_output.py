"""Tests closing remaining branch/line gaps in clinical output (doc/28)."""
from __future__ import annotations

from types import SimpleNamespace

from helixlang.plugins.human.clinical_output import (
    _RECOVERY_HALF_LIFE_H,
    ClinicalLabModel,
    ClinicalLabs,
    VitalSigns,
    VitalsModel,
    _potency_scale,
)
from helixlang.plugins.human.physiology import create_default_physiology


class TestClinicalLabsEdges:
    def test_is_abnormal_unknown_field(self):
        labs = ClinicalLabs()
        assert labs.is_abnormal("not_a_real_field") is False

    def test_is_abnormal_none_value(self):
        labs = ClinicalLabs(alt_u_per_l=None)
        assert labs.is_abnormal("alt_u_per_l") is False

    def test_is_abnormal_within_range(self):
        labs = ClinicalLabs()
        assert labs.is_abnormal("alt_u_per_l") is False

    def test_is_abnormal_out_of_range(self):
        labs = ClinicalLabs(alt_u_per_l=500.0)
        assert labs.is_abnormal("alt_u_per_l") is True

    def test_abnormal_count(self):
        labs = ClinicalLabs(alt_u_per_l=500.0, wbc_per_ul=500.0)
        assert labs.abnormal_count() >= 2

    def test_to_progression_labs_keys(self):
        labs = ClinicalLabs()
        d = labs.to_progression_labs()
        assert d["age_years"] == 30.0
        assert d["alt_u_l"] == 25.0
        assert d["tumor_marker_ng_ml"] == 0.5


class TestClinicalLabModelEdges:
    def test_baseline_female_lowers_muscle_factor(self):
        phys = create_default_physiology(sex="female", body_weight_kg=84.0)
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        assert baseline.creatinine_mg_per_dl < 1.2

    def test_baseline_low_cyp1a2_raises_alt(self):
        phys = create_default_physiology()
        phys.cytochrome_p450_activity["CYP1A2"] = 0.05
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        assert baseline.alt_u_per_l == 25.0 * 1.3

    def test_get_current_returns_snapshot(self):
        phys = create_default_physiology()
        model = ClinicalLabModel(ClinicalLabs(), phys)
        result = model.get_current()
        assert isinstance(result, ClinicalLabs)

    def test_potency_scale_nonpositive_returns_zero(self):
        assert _potency_scale(0.0, 50.0, 3.0) == 0.0
        assert _potency_scale(-5.0, 50.0, 3.0) == 0.0
        assert _potency_scale(30.0, 0.0, 3.0) == 0.0

    def test_potency_scale_scaling(self):
        assert abs(_potency_scale(30.0, 50.0, 3.0) - 0.6) < 1e-9
        assert _potency_scale(300.0, 50.0, 3.0) == 3.0

    def test_hepatotoxicity_triggers_hy_law_bilirubin(self):
        phys = create_default_physiology()
        model = ClinicalLabModel(ClinicalLabs(), phys)
        model.current.alt_u_per_l = 200.0
        model.current.bilirubin_total_mg_per_dl = 3.0
        model.update(dt_h=1.0, drug_concentrations={"cisplatin": 50.0})
        assert model.current.bilirubin_total_mg_per_dl > 3.0

    def test_hepatotoxicity_below_hy_threshold_no_bilirubin(self):
        phys = create_default_physiology()
        model = ClinicalLabModel(ClinicalLabs(), phys)
        model.update(dt_h=1.0, drug_concentrations={"cisplatin": 50.0})
        assert model.current.alt_u_per_l < 168.0
        assert model.current.bilirubin_total_mg_per_dl == 0.7

    def test_recovery_skips_missing_current_value(self):
        phys = create_default_physiology()
        model = ClinicalLabModel(ClinicalLabs(), phys)
        model.current.ast_u_per_l = None
        result = model.update(dt_h=1.0)
        assert isinstance(result, ClinicalLabs)

    def test_recovery_skips_pathological_half_life(self, monkeypatch):
        phys = create_default_physiology()
        model = ClinicalLabModel(ClinicalLabs(), phys)
        halflives = dict(_RECOVERY_HALF_LIFE_H)
        halflives["sodium_meq_per_l"] = 1e15
        monkeypatch.setattr(
            "helixlang.plugins.human.clinical_output._RECOVERY_HALF_LIFE_H", halflives,
        )
        result = model.update(dt_h=1.0)
        assert result.sodium_meq_per_l == 140.0

    def test_structure_toxicity_no_smiles_returns_none(self):
        phys = create_default_physiology()
        model = ClinicalLabModel(ClinicalLabs(), phys)
        assert model._get_structure_toxicity("unknown_drug") is None

    def test_structure_toxicity_low_confidence_returns_none(self, monkeypatch):
        phys = create_default_physiology()
        model = ClinicalLabModel(ClinicalLabs(), phys)
        model.register_drug_smiles("low_conf", "CCO")

        def _low_confidence(self, smiles):
            return SimpleNamespace(confidence=0.0)

        monkeypatch.setattr(
            "helixlang.plugins.human.molecular_toxicity."
            "MolecularToxicityPredictor.predict_toxicity",
            _low_confidence,
        )
        assert model._get_structure_toxicity("low_conf") is None

    def test_structure_toxicity_caches_result(self, monkeypatch):
        phys = create_default_physiology()
        model = ClinicalLabModel(ClinicalLabs(), phys)
        model.register_drug_smiles("cache_me", "CCO")

        def _predict(self, smiles):
            return SimpleNamespace(
                confidence=0.9,
                hepatotoxicity_score=0.5,
                nephrotoxicity_score=0.4,
                myelosuppression_score=0.3,
                cardiotoxicity_score=0.2,
            )

        monkeypatch.setattr(
            "helixlang.plugins.human.molecular_toxicity."
            "MolecularToxicityPredictor.predict_toxicity",
            _predict,
        )
        first = model._get_structure_toxicity("cache_me")
        second = model._get_structure_toxicity("cache_me")
        assert first["hepatotoxicity"] == 0.5
        assert first == second

    def test_unknown_drug_without_smiles_skips_toxicity(self):
        phys = create_default_physiology()
        model = ClinicalLabModel(ClinicalLabs(), phys)
        alt_before = model.current.alt_u_per_l
        model.update(dt_h=24.0, drug_concentrations={"orphan_drug": 30.0})
        assert model.current.alt_u_per_l == alt_before

    def test_unknown_drug_smiles_toxicity_fallbacks(self, monkeypatch):
        phys = create_default_physiology()
        model = ClinicalLabModel(ClinicalLabs(), phys)
        model.register_drug_smiles("novel_drug", "CCO")

        def _predict(self, smiles):
            return SimpleNamespace(
                confidence=0.9,
                hepatotoxicity_score=0.5,
                nephrotoxicity_score=0.4,
                myelosuppression_score=0.3,
                cardiotoxicity_score=0.2,
            )

        monkeypatch.setattr(
            "helixlang.plugins.human.molecular_toxicity."
            "MolecularToxicityPredictor.predict_toxicity",
            _predict,
        )
        alt_before = model.current.alt_u_per_l
        cr_before = model.current.creatinine_mg_per_dl
        wbc_before = model.current.wbc_per_ul
        model.update(dt_h=24.0, drug_concentrations={"novel_drug": 60.0})
        assert model.current.alt_u_per_l > alt_before
        assert model.current.creatinine_mg_per_dl > cr_before
        assert model.current.wbc_per_ul < wbc_before

    def test_structure_toxicity_exception_returns_none(self, monkeypatch):
        phys = create_default_physiology()
        model = ClinicalLabModel(ClinicalLabs(), phys)
        model.register_drug_smiles("unknown_drug", "C(C)C")

        def _boom(self, smiles):
            raise RuntimeError("predictor failure")

        monkeypatch.setattr(
            "helixlang.plugins.human.molecular_toxicity."
            "MolecularToxicityPredictor.predict_toxicity",
            _boom,
        )
        assert model._get_structure_toxicity("unknown_drug") is None

    def test_electrolytes_with_low_egfr(self):
        phys = create_default_physiology()
        model = ClinicalLabModel(ClinicalLabs(), phys)
        model.current.egfr_ml_per_min = 25.0
        result = model.update(dt_h=48.0)
        assert result.potassium_meq_per_l > 4.0
        assert result.phosphate_mg_per_dl > 3.5
        assert result.bicarbonate_meq_per_l < 24.0

    def test_inr_rises_with_disease_and_ibuprofen(self):
        phys = create_default_physiology()
        model = ClinicalLabModel(ClinicalLabs(), phys)
        result = model.update(
            dt_h=48.0,
            disease_severity=0.8,
            drug_concentrations={"ibuprofen": 40.0},
        )
        assert result.inr > 1.5


class TestDiseaseBaselineEdges:
    def test_pku_sets_phenylalanine(self):
        phys = create_default_physiology()
        disease = SimpleNamespace(name="PKU", severity=0.5)
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys, disease)
        assert abs(baseline.phenylalanine_mmol_per_l - (0.09 + 0.5 * 2.31)) < 1e-9

    def test_msud_raises_alt_and_lactate(self):
        phys = create_default_physiology()
        disease = SimpleNamespace(name="MSUD", severity=0.7)
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys, disease)
        assert baseline.alt_u_per_l == 25.0 + 0.7 * 80.0
        assert baseline.lactate_mmol_per_l == 1.2 + 0.7 * 4.0

    def test_gaucher_raises_alp(self):
        phys = create_default_physiology()
        disease = SimpleNamespace(name="Gaucher", severity=0.5)
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys, disease)
        assert baseline.alp_u_per_l == 70.0 + 0.5 * 300.0
        assert baseline.hemoglobin_g_per_dl == 12.5

    def test_diabetes_raises_glucose(self):
        phys = create_default_physiology()
        disease = SimpleNamespace(name="Type2 Diabetes", severity=0.5)
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys, disease)
        assert baseline.glucose_mg_per_dl == 90.0 + 0.5 * 210.0
        assert baseline.hba1c_pct == 5.5 + 0.5 * 4.5

    def test_cancer_warburg_raises_lactate(self):
        phys = create_default_physiology()
        disease = SimpleNamespace(name="warburg", severity=0.5)
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys, disease)
        assert baseline.crp_mg_per_l == 1.0 + 0.5 * 40.0
        assert baseline.albumin_g_per_dl == 3.0
        assert baseline.lactate_mmol_per_l == 1.2 + 0.5 * 8.0

    def test_fabry_raises_creatinine(self):
        phys = create_default_physiology()
        disease = SimpleNamespace(name="Fabry", severity=0.5)
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys, disease)
        assert baseline.creatinine_mg_per_dl == 1.0 + 0.5 * 1.5
        assert baseline.egfr_ml_per_min < 120.0

    def test_unknown_disease_leaves_alt_unchanged(self):
        phys = create_default_physiology()
        disease = SimpleNamespace(name="influenza", severity=0.5)
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys, disease)
        assert baseline.alt_u_per_l == 25.0


class TestVitalsModelEdges:
    def test_physio_driver_default_when_no_physiology(self):
        model = VitalsModel(base_vitals=VitalSigns(), physiological_core=True)
        assert model._physio_driver is not None
        result = model.update(dt_h=1.0)
        assert result.systolic_bp_mmhg > 0.0

    def test_physio_driver_bicarbonate_missing_uses_default(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys, physiological_core=True)
        labs = ClinicalLabs(bicarbonate_meq_per_l=None)
        result = model.update(dt_h=1.0, labs=labs)
        assert 6.0 < result.respiratory_rate_per_min < 45.0

    def test_physio_driver_uses_lab_bicarbonate(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys, physiological_core=True)
        labs = ClinicalLabs(bicarbonate_meq_per_l=18.0)
        result = model.update(dt_h=1.0, labs=labs)
        assert result.spo2_pct > 0.0

    def test_physio_driver_weight_drift_from_drug(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys, physiological_core=True)
        result = model.update(dt_h=24.0, drug_concentrations={"ibuprofen": 60.0})
        assert result.weight_kg > model.baseline.weight_kg

    def test_obese_physiology_raises_blood_pressure(self):
        phys = create_default_physiology(body_weight_kg=100.0)
        model = VitalsModel.create_from_physiology(phys)
        vs = model.current
        assert vs.systolic_bp_mmhg > 120.0
        assert vs.diastolic_bp_mmhg > 80.0
        assert vs.heart_rate_bpm > 72.0

    def test_update_without_labs_skips_lab_blocks(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        result = model.update(dt_h=1.0)
        assert isinstance(result, VitalSigns)
        assert result.respiratory_rate_per_min == 16.0

    def test_drugs_affect_vitals_and_qt(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        result = model.update(dt_h=24.0, drug_concentrations={"ibuprofen": 60.0})
        assert result.systolic_bp_mmhg > model.baseline.systolic_bp_mmhg
        assert result.weight_kg > model.baseline.weight_kg
        assert result.qt_interval_ms > 380.0

    def test_disease_fever_and_renal_hypetension(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        labs = ClinicalLabs(crp_mg_per_l=50.0, egfr_ml_per_min=30.0)
        result = model.update(dt_h=1.0, labs=labs, disease_severity=0.8)
        assert result.temperature_c > model.baseline.temperature_c
        assert result.systolic_bp_mmhg > model.baseline.systolic_bp_mmhg
        assert result.diastolic_bp_mmhg > model.baseline.diastolic_bp_mmhg

    def test_disease_with_normal_hemoglobin_no_anemia_tachycardia(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        labs = ClinicalLabs(hemoglobin_g_per_dl=12.0)
        result = model.update(dt_h=1.0, labs=labs, disease_severity=0.8)
        assert result.heart_rate_bpm == model.baseline.heart_rate_bpm
        assert result.spo2_pct == model.baseline.spo2_pct

    def test_severe_anemia_tachycardia_and_low_spo2(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        labs = ClinicalLabs(hemoglobin_g_per_dl=7.5)
        result = model.update(dt_h=1.0, labs=labs, disease_severity=0.8)
        assert result.heart_rate_bpm > model.baseline.heart_rate_bpm
        assert result.spo2_pct < model.baseline.spo2_pct

    def test_hypokalemia_prolongs_qt(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        labs = ClinicalLabs(potassium_meq_per_l=2.5)
        result = model.update(dt_h=1.0, labs=labs)
        assert result.qtc_ms > model.baseline.qtc_ms

    def test_hypocalcemia_prolongs_qt(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        labs = ClinicalLabs(calcium_mg_per_dl=6.5)
        result = model.update(dt_h=1.0, labs=labs)
        assert result.qtc_ms > model.baseline.qtc_ms

    def test_hyperkalemia_bradycardia_path(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        labs = ClinicalLabs(potassium_meq_per_l=7.0)
        result = model.update(dt_h=1.0, labs=labs)
        assert result.heart_rate_bpm == model.baseline.heart_rate_bpm
        assert result.qtc_ms > model.baseline.qtc_ms

    def test_hyponatremia_hypotension_path(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        labs = ClinicalLabs(sodium_meq_per_l=120.0)
        result = model.update(dt_h=1.0, labs=labs)
        assert result.systolic_bp_mmhg == model.baseline.systolic_bp_mmhg

    def test_metabolic_acidosis_kussmaul(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        labs = ClinicalLabs(bicarbonate_meq_per_l=15.0)
        result = model.update(dt_h=1.0, labs=labs)
        assert result.respiratory_rate_per_min == 19.0

    def test_metabolic_alkalosis_hypoventilation(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        labs = ClinicalLabs(bicarbonate_meq_per_l=34.0)
        result = model.update(dt_h=1.0, labs=labs)
        assert result.respiratory_rate_per_min == 14.0

    def test_map_mmhg_and_pulse_pressure(self):
        vs = VitalSigns(systolic_bp_mmhg=140.0, diastolic_bp_mmhg=90.0)
        assert abs(vs.map_mmhg - 106.67) < 0.01
        assert vs.pulse_pressure == 50.0

    def test_get_current_vitals(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        result = model.get_current()
        assert isinstance(result, VitalSigns)
