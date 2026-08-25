"""Tests for disease progression modeling (doc/28)."""
from __future__ import annotations

from helixlang.human.disease_progression import (
    ClinicalLabs,
    DiseaseProgressionModel,
    DiseaseStage,
    create_progression_model,
)


class TestDiseaseStage:
    def test_all_values(self):
        stages = [s.value for s in DiseaseStage]
        assert "preclinical" in stages
        assert "mild" in stages
        assert "moderate" in stages
        assert "severe" in stages
        assert "critical" in stages


class TestClinicalLabs:
    def test_fib4_index(self):
        labs = ClinicalLabs(age_years=60, ast_u_l=80, platelets_per_ul=100_000)
        fib4 = labs.fib4_index()
        assert fib4 > 0

    def test_child_pugh_score(self):
        labs = ClinicalLabs(
            total_bilirubin_mg_dl=3.0,
            albumin_g_dl=2.5,
            inr=2.5,
            ascites_grade=1,
            encephalopathy_grade=1,
        )
        score = labs.child_pugh_score()
        assert 5 <= score <= 15


class TestProgressionModel:
    def test_create_ckd(self):
        model = create_progression_model("CKD")
        assert isinstance(model, DiseaseProgressionModel)
        assert model.current_stage in DiseaseStage

    def test_create_diabetes(self):
        model = create_progression_model("DIABETES_T2")
        assert isinstance(model, DiseaseProgressionModel)

    def test_create_cancer(self):
        model = create_progression_model("cancer")
        assert isinstance(model, DiseaseProgressionModel)

    def test_create_cirrhosis(self):
        model = create_progression_model("cirrhosis")
        assert isinstance(model, DiseaseProgressionModel)

    def test_severity_increases_without_treatment(self):
        model = create_progression_model("CKD")
        sev_start = model.get_severity()
        for _ in range(10):
            model.step(dt_h=24 * 30, drug_effectiveness=0.0)
        assert model.get_severity() >= sev_start

    def test_severity_decreases_with_treatment(self):
        model = create_progression_model("DIABETES_T2")
        for _ in range(20):
            model.step(dt_h=24 * 30, drug_effectiveness=0.8)
        sev_end = model.get_severity()
        for _ in range(20):
            model.step(dt_h=24 * 30, drug_effectiveness=0.95)
        assert model.get_severity() <= sev_end

    def test_organ_function_range(self):
        model = create_progression_model("CKD")
        frac = model.get_organ_function("kidney")
        assert 0.0 <= frac <= 1.0

    def test_step_returns_stage(self):
        model = create_progression_model("CKD")
        labs = ClinicalLabs(egfr_ml_min_1_73m2=60.0)
        stage = model.step(dt_h=24.0, drug_effectiveness=0.5, labs=labs)
        assert isinstance(stage, DiseaseStage)
