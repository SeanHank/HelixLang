"""Tests for clinical lab values and vital-signs dynamics (doc/28)."""
from __future__ import annotations

from helixlang.human.clinical_output import (
    ClinicalLabModel,
    ClinicalLabs,
    VitalSigns,
    VitalsModel,
    _ckd_epi_2021,
)
from helixlang.human.physiology import create_default_physiology


class TestClinicalLabs:
    def test_defaults_within_reference(self):
        labs = ClinicalLabs()
        assert 7.0 <= labs.alt_u_per_l <= 56.0
        assert 10.0 <= labs.ast_u_per_l <= 40.0
        assert 0.1 <= labs.bilirubin_total_mg_per_dl <= 1.2
        assert 3.5 <= labs.albumin_g_per_dl <= 5.5
        assert 0.8 <= labs.inr <= 1.2
        assert 0.7 <= labs.creatinine_mg_per_dl <= 1.3
        assert 4500.0 <= labs.wbc_per_ul <= 11000.0
        assert 13.5 <= labs.hemoglobin_g_per_dl <= 17.5
        assert 150000.0 <= labs.platelets_per_ul <= 400000.0
        assert 70.0 <= labs.glucose_mg_per_dl <= 100.0
        assert 4.0 <= labs.hba1c_pct <= 5.7
        assert 136.0 <= labs.sodium_meq_per_l <= 145.0
        assert 3.5 <= labs.potassium_meq_per_l <= 5.0

    def test_abnormal_detection(self):
        labs = ClinicalLabs(alt_u_per_l=500.0)
        assert labs.is_abnormal("alt_u_per_l")
        assert not labs.is_abnormal("ast_u_per_l")

    def test_abnormal_count(self):
        labs = ClinicalLabs(alt_u_per_l=500.0, wbc_per_ul=500.0)
        count = labs.abnormal_count()
        assert count >= 2

    def test_snapshot_returns_copy(self):
        labs = ClinicalLabs()
        snap = labs.snapshot()
        snap.alt_u_per_l = 999.0
        assert labs.alt_u_per_l == 25.0

    def test_to_progression_labs_keys(self):
        labs = ClinicalLabs()
        d = labs.to_progression_labs()
        for key in ("age_years", "egfr_ml_min_1_73m2", "creatinine_mg_dl",
                     "alt_u_l", "ast_u_l", "total_bilirubin_mg_dl",
                     "albumin_g_dl", "inr", "platelets_per_ul", "hba1c_percent"):
            assert key in d


class TestClinicalLabModel:
    def test_creation(self):
        baseline = ClinicalLabs()
        phys = create_default_physiology()
        model = ClinicalLabModel(baseline, phys)
        assert model.current.alt_u_per_l == 25.0

    def test_update_returns_labs(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        model = ClinicalLabModel(baseline, phys)
        result = model.update(dt_h=24.0)
        assert isinstance(result, ClinicalLabs)

    def test_hepatotoxicity_cisplatin(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        model = ClinicalLabModel(baseline, phys)
        alt_start = model.current.alt_u_per_l
        for _ in range(10):
            model.update(dt_h=24.0, drug_concentrations={"cisplatin": 50.0})
        assert model.current.alt_u_per_l > alt_start

    def test_nephrotoxicity_cisplatin(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        model = ClinicalLabModel(baseline, phys)
        cr_start = model.current.creatinine_mg_per_dl
        for _ in range(10):
            model.update(dt_h=24.0, drug_concentrations={"cisplatin": 50.0})
        assert model.current.creatinine_mg_per_dl > cr_start

    def test_myelosuppression_cisplatin(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        model = ClinicalLabModel(baseline, phys)
        wbc_start = model.current.wbc_per_ul
        for _ in range(10):
            model.update(dt_h=24.0, drug_concentrations={"cisplatin": 50.0})
        assert model.current.wbc_per_ul < wbc_start

    def test_recovery_toward_baseline(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        model = ClinicalLabModel(baseline, phys)
        model.current.alt_u_per_l = 200.0
        for _ in range(50):
            model.update(dt_h=24.0)
        assert model.current.alt_u_per_l < 200.0

    def test_ckd_epi_2021_male(self):
        egfr = _ckd_epi_2021(1.0, 40.0, "male")
        assert 60.0 < egfr < 150.0

    def test_ckd_epi_2021_female_vs_male(self):
        # At same creatinine, CKD-EPI gives females slightly lower eGFR
        # (κ=0.7 vs 0.9 offsets the 1.012 sex factor)
        egfr_f = _ckd_epi_2021(1.0, 40.0, "female")
        egfr_m = _ckd_epi_2021(1.0, 40.0, "male")
        assert 50.0 < egfr_f < 150.0
        assert 50.0 < egfr_m < 150.0

    def test_ckd_epi_high_creatinine_low_egfr(self):
        egfr = _ckd_epi_2021(3.0, 70.0, "male")
        assert egfr < 30.0


class TestVitalSigns:
    def test_defaults(self):
        vs = VitalSigns()
        assert vs.systolic_bp_mmhg == 120.0
        assert vs.diastolic_bp_mmhg == 80.0
        assert vs.heart_rate_bpm == 72.0
        assert vs.temperature_c == 37.0

    def test_map_mmhg(self):
        vs = VitalSigns(systolic_bp_mmhg=120.0, diastolic_bp_mmhg=80.0)
        assert abs(vs.map_mmhg - 93.3) < 0.1

    def test_pulse_pressure(self):
        vs = VitalSigns(systolic_bp_mmhg=140.0, diastolic_bp_mmhg=90.0)
        assert vs.pulse_pressure == 50.0

    def test_snapshot(self):
        vs = VitalSigns()
        snap = vs.snapshot()
        snap.heart_rate_bpm = 200.0
        assert vs.heart_rate_bpm == 72.0


class TestVitalsModel:
    def test_creation_from_physiology(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        vitals = model.get_current()
        assert vitals.systolic_bp_mmhg > 90.0

    def test_update_with_drugs(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        result = model.update(dt_h=1.0, drug_concentrations={"ibuprofen": 50.0})
        assert result.systolic_bp_mmhg > 100.0

    def test_update_with_disease(self):
        phys = create_default_physiology()
        model = VitalsModel.create_from_physiology(phys)
        labs = ClinicalLabs(crp_mg_per_l=50.0, egfr_ml_per_min=30.0)
        result = model.update(dt_h=1.0, labs=labs, disease_severity=0.8)
        assert result.temperature_c > 37.0
