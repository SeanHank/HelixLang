"""Comprehensive tests for doc/29: critical defect fixes and full-parameter coverage.

Validates:
- PBPK state accumulation across steps
- Unit conversion (µM-normalized potency scalers)
- DDI clearance modifier behavior
- Recovery biomarker relaxation
- Dynamic electrolytes (Na+, K+, Ca2+, PO4, Cl-, HCO3-)
- Dynamic lipids (LDL, HDL, TG)
- Dynamic coagulation (INR)
- QTc prolongation (drug + electrolyte driven)
- Dynamic SpO2 and respiratory rate
- Full VirtualPatientResult channel coverage
- Disease name propagation
"""
from __future__ import annotations

import pytest

from helixlang.plugins.human.clinical_output import (
    _POTENCY_REFERENCE_UM,
    ClinicalLabModel,
    ClinicalLabs,
    VitalSigns,
    VitalsModel,
    _potency_scale,
)
from helixlang.plugins.human.drug import get_predefined_drug
from helixlang.plugins.human.pharmacokinetics import PBPKConfig, PBPKModel
from helixlang.plugins.human.physiology import create_default_physiology

# ---------------------------------------------------------------------------
# Fix 1: PBPK state accumulation across steps
# ---------------------------------------------------------------------------

class TestPBPKStatefulEngine:
    """PBPK must maintain concentration state across step() calls."""

    def _make_pbpk(self, drug_name: str = "IBUPROFEN") -> PBPKModel:
        phys = create_default_physiology()
        drug = get_predefined_drug(drug_name)
        return PBPKModel(drug, phys, PBPKConfig(dt_min=60.0, total_time_h=24.0))

    def test_conc_accumulates_over_multiple_steps(self):
        pbpk = self._make_pbpk()
        concs = []
        for _ in range(24):
            snap = pbpk.step(dt_min=60.0)
            concs.append(snap.get("central", 0.0))
        # Plasma concentration should be non-zero after dosing
        assert concs[-1] > 0.0

    def test_run_gives_full_trajectory(self):
        pbpk = self._make_pbpk()
        result = pbpk.run()
        assert len(result.time_h) > 1
        assert result.c_max > 0.0
        assert result.auc > 0.0


# ---------------------------------------------------------------------------
# Fix 2: Unit conversion (µM-normalized scalers)
# ---------------------------------------------------------------------------

class TestUnitConversion:
    """Drug-effect scalers must use µM-normalized concentrations."""

    def test_potency_scale_at_reference(self):
        for role, ref_um in _POTENCY_REFERENCE_UM.items():
            scale = _potency_scale(ref_um, ref_um, max_scale=1.0)
            assert scale == pytest.approx(1.0), f"{role}: {scale}"

    def test_potency_scale_below_reference(self):
        scale = _potency_scale(25.0, 50.0, max_scale=1.0)
        assert scale == pytest.approx(0.5)

    def test_potency_scale_above_reference_clamps(self):
        scale = _potency_scale(200.0, 50.0, max_scale=1.0)
        assert scale == pytest.approx(1.0)

    def test_potency_scale_zero_conc_returns_zero(self):
        scale = _potency_scale(0.0, 50.0, max_scale=1.0)
        assert scale == 0.0

    def test_potency_scale_negative_conc_returns_zero(self):
        scale = _potency_scale(-5.0, 50.0, max_scale=1.0)
        assert scale == 0.0

    def test_potency_scale_custom_max(self):
        scale = _potency_scale(100.0, 50.0, max_scale=0.5)
        assert scale == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Fix 3: DDI clearance modifier
# ---------------------------------------------------------------------------

class TestDDIClearanceModifier:
    """DDI rules must apply clearance modifiers correctly."""

    def test_create_default_has_rules(self):
        from helixlang.plugins.human.ddi import create_default_ddi_model

        model = create_default_ddi_model()
        assert len(model.rules) > 0

    def test_single_drug_no_interaction(self):
        from helixlang.plugins.human.ddi import create_default_ddi_model

        model = create_default_ddi_model()
        cyp_profile = {"CYP2D6": "extensive"}
        modifiers = model.compute_clearance_modifiers(["metformin"], cyp_profile)
        assert modifiers.get("metformin", 1.0) == pytest.approx(1.0, abs=0.2)


# ---------------------------------------------------------------------------
# Fix 4: Recovery biomarker relaxation
# ---------------------------------------------------------------------------

class TestRecoveryBiomarkers:
    """Recovery model should relax biomarkers toward baseline."""

    def test_biomarker_relaxes_toward_baseline(self):
        from helixlang.plugins.human.recovery import RecoveryModel, Sequela

        baseline = {"alt": 30.0, "creatinine": 1.0}
        current = {"alt": 150.0, "creatinine": 2.5}
        sequela = [
            Sequela(
                name="liver_injury",
                organ="liver",
                severity=0.6,
                onset_delay_h=0.0,
                reversible=True,
                recovery_half_life_h=48.0,
            ),
        ]
        model = RecoveryModel(
            baseline_biomarkers=baseline,
            current_biomarkers=current.copy(),
            organ_recovery_rates={"liver": 0.02},
            sequela_list=sequela,
        )
        model.set_treatment_inactive()

        vals = current.copy()
        for i in range(1, 100):
            vals = model.step(dt_h=24.0, current_time_h=float(i * 24))

        # ALT should have moved toward baseline
        assert vals["alt"] < 150.0
        assert vals["alt"] > baseline["alt"]


# ---------------------------------------------------------------------------
# Fix 5: Dynamic electrolytes
# ---------------------------------------------------------------------------

class TestDynamicElectrolytes:
    """Electrolytes must respond to disease severity and renal function."""

    def test_severe_disease_hyponatremia(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        labs = ClinicalLabModel(baseline, phys)
        for _ in range(100):
            labs.update(24.0, {}, disease_severity=0.9)
        current = labs.get_current()
        assert current.sodium_meq_per_l < 138.0

    def test_ckd_hyperkalemia(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        baseline.creatinine_mg_per_dl = 4.0
        labs = ClinicalLabModel(baseline, phys)
        for _ in range(50):
            labs.update(24.0, {}, disease_severity=0.0)
        current = labs.get_current()
        assert current.potassium_meq_per_l > 4.2

    def test_calcium_tracks_albumin(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        baseline.albumin_g_per_dl = 2.0
        labs = ClinicalLabModel(baseline, phys)
        for _ in range(100):
            labs.update(24.0, {}, disease_severity=0.5)
        current = labs.get_current()
        assert current.calcium_mg_per_dl > 7.0

    def test_ckd_hyperphosphatemia(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        baseline.creatinine_mg_per_dl = 5.0
        labs = ClinicalLabModel(baseline, phys)
        for _ in range(100):
            labs.update(24.0, {}, disease_severity=0.0)
        current = labs.get_current()
        assert current.phosphate_mg_per_dl > 4.0

    def test_chloride_tracks_sodium(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        labs = ClinicalLabModel(baseline, phys)
        for _ in range(50):
            labs.update(24.0, {}, disease_severity=0.8)
        current = labs.get_current()
        assert current.chloride_meq_per_l < 104.0

    def test_bicarbonate_drops_with_severe_disease(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        labs = ClinicalLabModel(baseline, phys)
        for _ in range(200):
            labs.update(24.0, {}, disease_severity=0.9)
        current = labs.get_current()
        # High severity drives lactate up, which drives HCO3 down
        assert current.bicarbonate_meq_per_l < 24.0


# ---------------------------------------------------------------------------
# Fix 6: Dynamic lipids
# ---------------------------------------------------------------------------

class TestDynamicLipids:
    """Lipids must shift with disease severity (T2DM dyslipidemia)."""

    def test_high_severity_worsens_ldl(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        labs = ClinicalLabModel(baseline, phys)
        for _ in range(200):
            labs.update(24.0, {}, disease_severity=0.8)
        current = labs.get_current()
        assert current.ldl_mg_per_dl > 130.0

    def test_high_severity_worsens_hdl(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        labs = ClinicalLabModel(baseline, phys)
        for _ in range(200):
            labs.update(24.0, {}, disease_severity=0.8)
        current = labs.get_current()
        assert current.hdl_mg_per_dl < 45.0

    def test_high_severity_worsens_triglycerides(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        labs = ClinicalLabModel(baseline, phys)
        for _ in range(200):
            labs.update(24.0, {}, disease_severity=0.8)
        current = labs.get_current()
        assert current.triglycerides_mg_per_dl > 160.0


# ---------------------------------------------------------------------------
# Fix 7: Dynamic coagulation (INR)
# ---------------------------------------------------------------------------

class TestDynamicCoagulation:
    """INR must reflect liver synthetic function and drug effects."""

    def test_severe_disease_elevates_inr(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        labs = ClinicalLabModel(baseline, phys)
        for _ in range(100):
            labs.update(24.0, {}, disease_severity=0.9)
        current = labs.get_current()
        assert current.inr > 1.1

    def test_ibuprofen_elevates_inr(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        labs = ClinicalLabModel(baseline, phys)
        for _ in range(48):
            labs.update(6.0, {"ibuprofen": 30.0}, disease_severity=0.0)
        current = labs.get_current()
        assert current.inr > 1.0


# ---------------------------------------------------------------------------
# Fix 8: QTc prolongation
# ---------------------------------------------------------------------------

class TestQTcProlongation:
    """QTc must respond to drug effects and electrolyte derangements."""

    def _make_vitals(self) -> VitalsModel:
        phys = create_default_physiology()
        return VitalsModel.create_from_physiology(phys)

    def test_baseline_qtc_normal(self):
        vitals = self._make_vitals()
        result = vitals.update(1.0, {}, ClinicalLabs(), 0.0)
        assert 370.0 < result.qtc_ms < 450.0

    def test_tamoxifen_prolongs_qtc(self):
        vitals = self._make_vitals()
        result = vitals.update(6.0, {"tamoxifen": 10.0}, ClinicalLabs(), 0.0)
        assert result.qtc_ms > 400.0

    def test_hypokalemia_prolongs_qtc(self):
        vitals = self._make_vitals()
        labs_hypo = ClinicalLabs(potassium_meq_per_l=2.5)
        result = vitals.update(6.0, {}, labs_hypo, 0.0)
        assert result.qtc_ms > 420.0

    def test_hypocalcemia_prolongs_qtc(self):
        vitals = self._make_vitals()
        labs_hypo = ClinicalLabs(calcium_mg_per_dl=6.0)
        result = vitals.update(6.0, {}, labs_hypo, 0.0)
        assert result.qtc_ms > 420.0

    def test_drug_concentration_scale_clamps(self):
        vitals = self._make_vitals()
        result = vitals.update(6.0, {"tamoxifen": 100.0}, ClinicalLabs(), 0.0)
        assert result.qtc_ms < 600.0

    def test_cisplatin_prolongs_qtc(self):
        vitals = self._make_vitals()
        result = vitals.update(6.0, {"cisplatin": 20.0}, ClinicalLabs(), 0.0)
        assert result.qtc_ms > 400.0


# ---------------------------------------------------------------------------
# Fix 9: Dynamic SpO2 and respiratory rate
# ---------------------------------------------------------------------------

class TestDynamicSpO2:
    """SpO2 must drop with severe anemia."""

    def _make_vitals(self) -> VitalsModel:
        phys = create_default_physiology()
        return VitalsModel.create_from_physiology(phys)

    def test_severe_anemia_lowers_spo2(self):
        vitals = self._make_vitals()
        labs_severe = ClinicalLabs(hemoglobin_g_per_dl=6.0)
        result = vitals.update(12.0, {}, labs_severe, 0.0)
        assert result.spo2_pct < 95.0

    def test_normal_hemoglobin_keeps_spo2(self):
        vitals = self._make_vitals()
        labs_norm = ClinicalLabs(hemoglobin_g_per_dl=14.0)
        result = vitals.update(12.0, {}, labs_norm, 0.0)
        assert result.spo2_pct >= 97.0


class TestDynamicRespiratoryRate:
    """Respiratory rate must respond to acid-base status."""

    def _make_vitals(self) -> VitalsModel:
        phys = create_default_physiology()
        return VitalsModel.create_from_physiology(phys)

    def test_metabolic_acidosis_increases_rr(self):
        vitals = self._make_vitals()
        labs_acid = ClinicalLabs(bicarbonate_meq_per_l=12.0)
        result = vitals.update(6.0, {}, labs_acid, 0.0)
        assert result.respiratory_rate_per_min > 16.0

    def test_metabolic_alkalosis_decreases_rr(self):
        vitals = self._make_vitals()
        labs_alk = ClinicalLabs(bicarbonate_meq_per_l=35.0)
        result = vitals.update(6.0, {}, labs_alk, 0.0)
        assert result.respiratory_rate_per_min < 16.0

    def test_normal_hco3_keeps_rr_normal(self):
        vitals = self._make_vitals()
        labs_norm = ClinicalLabs(bicarbonate_meq_per_l=24.0)
        result = vitals.update(6.0, {}, labs_norm, 0.0)
        assert result.respiratory_rate_per_min == pytest.approx(16.0, abs=0.1)


# ---------------------------------------------------------------------------
# Fix 10: VitalSigns snapshot includes QTc
# ---------------------------------------------------------------------------

class TestVitalSignsQTc:
    """VitalSigns must include qt_interval_ms and qtc_ms fields."""

    def test_defaults(self):
        vs = VitalSigns()
        assert vs.qt_interval_ms == 380.0
        assert vs.qtc_ms == 400.0

    def test_snapshot_copies_qtc(self):
        vs = VitalSigns(qt_interval_ms=420.0, qtc_ms=450.0)
        snap = vs.snapshot()
        assert snap.qt_interval_ms == 420.0
        assert snap.qtc_ms == 450.0

    def test_electrolyte_driven_qtc_hypokalemia(self):
        phys = create_default_physiology()
        vitals = VitalsModel.create_from_physiology(phys)
        labs = ClinicalLabs(potassium_meq_per_l=2.5)
        result = vitals.update(1.0, {}, labs, 0.0)
        assert result.qtc_ms > 400.0
        assert result.qtc_ms > result.qt_interval_ms


# ---------------------------------------------------------------------------
# Fix 11: VirtualPatientResult full channel coverage
# ---------------------------------------------------------------------------

class TestResultChannels:
    """VirtualPatientResult must include all new channels."""

    def test_result_has_electrolyte_channels(self):
        from helixlang.plugins.human.virtual_patient import VirtualPatientResult

        result = VirtualPatientResult()
        assert hasattr(result, "calcium")
        assert hasattr(result, "phosphate")
        assert hasattr(result, "chloride")
        assert hasattr(result, "bicarbonate")
        assert hasattr(result, "ldl")
        assert hasattr(result, "hdl")
        assert hasattr(result, "triglycerides")
        assert hasattr(result, "qtc_ms")

    def test_result_to_dict_includes_all_channels(self):
        from helixlang.plugins.human.virtual_patient import VirtualPatientResult

        result = VirtualPatientResult()
        result.time_h = [0.0, 1.0]
        result.calcium = [9.5, 9.2]
        result.phosphate = [3.5, 3.8]
        result.chloride = [102.0, 101.0]
        result.bicarbonate = [24.0, 22.0]
        result.ldl = [120.0, 125.0]
        result.hdl = [50.0, 48.0]
        result.triglycerides = [150.0, 160.0]
        result.qtc_ms = [400.0, 410.0]

        d = result.to_dict()
        labs = d["labs"]
        assert "calcium_mg_per_dl" in labs
        assert "phosphate_mg_per_dl" in labs
        assert "chloride_meq_per_l" in labs
        assert "bicarbonate_meq_per_l" in labs
        assert "ldl_mg_per_dl" in labs
        assert "hdl_mg_per_dl" in labs
        assert "triglycerides_mg_per_dl" in labs
        vitals = d["vitals"]
        assert "qtc_ms" in vitals

    def test_result_summary_format(self):
        from helixlang.plugins.human.virtual_patient import VirtualPatientResult

        result = VirtualPatientResult()
        result.time_h = [0.0]
        result.calcium = [9.5]
        result.phosphate = [3.5]
        result.chloride = [102.0]
        result.bicarbonate = [24.0]
        result.ldl = [120.0]
        result.hdl = [50.0]
        result.triglycerides = [150.0]
        result.qtc_ms = [400.0]
        s = result.summary()
        assert "--- Vitals ---" in s
        assert "--- Labs ---" in s


# ---------------------------------------------------------------------------
# Fix 12: Disease name propagation through ClinicalLabModel.update()
# ---------------------------------------------------------------------------

class TestDiseaseNamePropagation:
    """ClinicalLabModel.update() should accept disease_name parameter."""

    def test_update_accepts_disease_name(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        labs = ClinicalLabModel(baseline, phys)
        result = labs.update(1.0, {}, disease_severity=0.5, disease_name="type2_diabetes")
        assert isinstance(result, ClinicalLabs)

    def test_update_backward_compatible(self):
        phys = create_default_physiology()
        baseline = ClinicalLabModel.compute_baseline_from_physiology(phys)
        labs = ClinicalLabModel(baseline, phys)
        result = labs.update(1.0, {}, disease_severity=0.5)
        assert isinstance(result, ClinicalLabs)


# ---------------------------------------------------------------------------
# Fix 13: VitalSigns MAP and pulse pressure
# ---------------------------------------------------------------------------

class TestVitalSignsProperties:
    """VitalSigns computed properties should still work with new fields."""

    def test_map_calculation(self):
        vs = VitalSigns(systolic_bp_mmhg=120.0, diastolic_bp_mmhg=80.0)
        expected = 120.0 / 3.0 + 2.0 * 80.0 / 3.0
        assert vs.map_mmhg == pytest.approx(expected)

    def test_pulse_pressure(self):
        vs = VitalSigns(systolic_bp_mmhg=140.0, diastolic_bp_mmhg=90.0)
        assert vs.pulse_pressure == pytest.approx(50.0)

    def test_snapshot_preserves_all_fields(self):
        vs = VitalSigns(
            systolic_bp_mmhg=130.0,
            diastolic_bp_mmhg=85.0,
            heart_rate_bpm=68.0,
            respiratory_rate_per_min=14.0,
            temperature_c=36.8,
            spo2_pct=99.0,
            weight_kg=75.0,
            qt_interval_ms=400.0,
            qtc_ms=420.0,
        )
        snap = vs.snapshot()
        assert snap.systolic_bp_mmhg == 130.0
        assert snap.diastolic_bp_mmhg == 85.0
        assert snap.heart_rate_bpm == 68.0
        assert snap.respiratory_rate_per_min == 14.0
        assert snap.temperature_c == 36.8
        assert snap.spo2_pct == 99.0
        assert snap.weight_kg == 75.0
        assert snap.qt_interval_ms == 400.0
        assert snap.qtc_ms == 420.0
