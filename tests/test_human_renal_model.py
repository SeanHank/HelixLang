"""Tests for the renal function / CKD progression model (doc/30 §6)."""
from __future__ import annotations

import math

import pytest

from helixlang.human.disease_progression import DiseaseStage
from helixlang.human.renal_model import (
    RenalFunctionModel,
    ckd_epi_2021,
    create_renal_model,
    inverse_ckd_epi,
)


class TestCkdEpi2021:
    def test_known_reference_value(self):
        # 45-year-old male, creatinine 1.0 mg/dL -> ~94.6 mL/min/1.73m2
        assert ckd_epi_2021(1.0, 45, is_female=False) == pytest.approx(94.6, abs=0.5)

    def test_female_lower_at_same_creatinine(self):
        male = ckd_epi_2021(1.0, 45, is_female=False)
        female = ckd_epi_2021(1.0, 45, is_female=True)
        assert female < male * 0.85

    def test_monotone_decreasing_in_creatinine(self):
        values = [ckd_epi_2021(scr, 60, False) for scr in (0.7, 1.0, 1.5, 2.5, 4.0)]
        assert values == sorted(values, reverse=True)

    def test_inverse_round_trip(self):
        for target in (110.0, 80.0, 45.0, 20.0, 10.0):
            scr = inverse_ckd_epi(target, 55, is_female=False)
            recovered = ckd_epi_2021(scr, 55, is_female=False)
            assert recovered == pytest.approx(target, rel=0.01)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            ckd_epi_2021(0.0, 50, False)
        with pytest.raises(ValueError):
            inverse_ckd_epi(-5.0, 50, False)


class TestHealthyBaseline:
    def test_reported_matches_functional_at_rest(self):
        model = create_renal_model(age_years=45, initial_egfr=95.0)
        reported = model.step(24.0)
        assert reported == pytest.approx(model.functional_egfr, rel=0.02)

    def test_slow_age_related_decline_only(self):
        model = create_renal_model(age_years=45, initial_egfr=95.0)
        for _ in range(365):
            model.step(24.0)
        assert 93.0 < model.functional_egfr < 95.0

    def test_stage_mapping_healthy(self):
        model = create_renal_model(age_years=45, initial_egfr=95.0)
        assert model.to_disease_stage() == DiseaseStage.PRECLINICAL
        assert model.kdigo_g_stage() == "G1"
        assert model.kdigo_a_category() == "A1"


class TestDiabeticNephropathy:
    def _patient(self) -> RenalFunctionModel:
        return create_renal_model(
            age_years=65,
            diabetes=True,
            hypertension=True,
            initial_egfr=45.0,
            initial_uacr_mg_g=800.0,
        )

    def test_albuminuria_drives_steep_slope(self):
        model = self._patient()
        slope = model.chronic_slope_ml_per_year()
        # placebo arms of CREDENCE/FIDELIO report -4 to -5 mL/min/yr;
        # heavy albuminuria + comorbidities should land in that region
        assert -8.0 < slope <= -4.5

    def test_one_year_progression(self):
        model = self._patient()
        for _ in range(365):
            reported = model.step(24.0)
        assert 33.0 < reported < model.functional_egfr + 2.0
        assert model.kdigo_g_stage() in ("G3a", "G3b")

    def test_sglt2i_and_raas_improve_chronic_slope(self):
        untreated = self._patient().chronic_slope_ml_per_year()
        treated = self._patient()
        treated.start_sglt2i()
        treated.start_raas_blockade()
        assert treated.chronic_slope_ml_per_year() > untreated
        assert treated.chronic_slope_ml_per_year() > -4.0

    def test_treatment_wins_by_year_three(self):
        untreated = self._patient()
        treated = self._patient()
        treated.start_sglt2i()
        treated.start_raas_blockade()
        for _ in range(3 * 365):
            u = untreated.step(24.0)
            t = treated.step(24.0)
        assert t > u

    def test_acute_dip_after_raas_initiation(self):
        model = self._patient()
        model.start_raas_blockade()
        dip_min = None
        for _day in range(21):
            reported = model.step(24.0)
            dip_min = reported if dip_min is None else min(dip_min, reported)
        assert dip_min < 45.0 - 2.0

    def test_uacr_reduces_under_drugs(self):
        model = self._patient()
        model.start_sglt2i()
        model.start_raas_blockade()
        for _ in range(365):
            model.step(24.0)
        assert model.uacr_mg_g < 800.0


class TestAki:
    def test_creatinine_rises_and_partially_recovers(self):
        model = create_renal_model(age_years=55, initial_egfr=80.0)
        baseline_scr = model.serum_creatinine
        model.induce_aki(0.6, recovery_fraction=0.7, injury_duration_h=72)
        peak_scr = baseline_scr
        for _day in range(90):
            model.step(24.0)
            peak_scr = max(peak_scr, model.serum_creatinine)
        assert peak_scr > baseline_scr * 1.5
        # 30% of the 60% loss is permanent: 80 -> ~65.6
        assert model.functional_egfr == pytest.approx(65.6, abs=3.0)

    def test_staging_worsens_during_injury(self):
        model = create_renal_model(age_years=55, initial_egfr=70.0)
        before = model.to_disease_stage()
        model.induce_aki(0.75, injury_duration_h=48)
        for _ in range(14):
            model.step(24.0)
        after = model.to_disease_stage()
        order = list(DiseaseStage)
        assert order.index(after) > order.index(before)

    def test_full_recovery_when_recovery_fraction_is_one(self):
        model = create_renal_model(age_years=55, initial_egfr=80.0)
        model.induce_aki(0.5, recovery_fraction=1.0, injury_duration_h=24)
        for _ in range(180):
            model.step(24.0)
        assert model.functional_egfr == pytest.approx(80.0, abs=2.0)


class TestRiskOutputs:
    def test_kfre_bounds_and_monotonicity(self):
        sick = create_renal_model(age_years=70, diabetes=True, initial_egfr=35.0,
                                  initial_uacr_mg_g=500.0)
        mild = create_renal_model(age_years=60, initial_egfr=50.0,
                                  initial_uacr_mg_g=100.0)
        for m in (sick, mild):
            for years in (2, 5):
                risk = m.kfre_risk(years=years)
                assert 0.0 <= risk <= 1.0
        assert sick.kfre_risk(years=2) > mild.kfre_risk(years=2)
        assert sick.kfre_risk(years=5) > sick.kfre_risk(years=2)

    def test_time_to_krt(self):
        dn = create_renal_model(age_years=65, diabetes=True, hypertension=True,
                                initial_egfr=45.0, initial_uacr_mg_g=800.0)
        projected = dn.time_to_krt_years()
        assert math.isfinite(projected)
        assert 0.0 < projected < 15.0
        # healthy aging declines far slower, so projection is far longer
        healthy = create_renal_model(age_years=45, initial_egfr=95.0)
        assert healthy.time_to_krt_years() > projected * 5

    def test_heatmap_cell_and_stage_mapping(self):
        model = create_renal_model(age_years=65, diabetes=True,
                                   initial_egfr=25.0, initial_uacr_mg_g=500.0)
        cell = model.ckd_heatmap_cell()
        assert cell.startswith("G4") and cell.endswith("A3")
        assert model.to_disease_stage() == DiseaseStage.SEVERE

    def test_lab_values_keys(self):
        model = create_renal_model(initial_egfr=90.0)
        labs = model.lab_values()
        for key in (
            "egfr_ml_min_1_73m2", "functional_egfr_ml_min_1_73m2",
            "creatinine_mg_dl", "uacr_mg_g",
            "chronic_slope_ml_min_per_year", "kfre_5y_risk",
        ):
            assert key in labs


class TestValidation:
    def test_constructor_rejects_bad_age(self):
        with pytest.raises(ValueError):
            RenalFunctionModel(age_years=0)

    def test_induce_aki_range_checks(self):
        model = create_renal_model()
        with pytest.raises(ValueError):
            model.induce_aki(0.99)
        with pytest.raises(ValueError):
            model.induce_aki(0.3, recovery_fraction=1.5)

    def test_kfre_horizon_check(self):
        model = create_renal_model()
        with pytest.raises(ValueError):
            model.kfre_risk(years=3)

    def test_step_rejects_negative_dt(self):
        model = create_renal_model()
        with pytest.raises(ValueError):
            model.step(-24.0)
