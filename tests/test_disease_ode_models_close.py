"""Closure tests for disease_ode_models.py (per-category mechanistic ODEs)."""
from __future__ import annotations

import pytest

import helixlang.plugins.human.disease_ode_models as dome
from helixlang.plugins.human.disease_ode_models import (
    AutoimmuneRAODE,
    CancerODE,
    CardiovascularODE,
    EndocrineODE,
    GastrointestinalODE,
    HematologicalODE,
    HepaticODE,
    InfectiousDiseaseODE,
    MetabolicT2DODE,
    NeurologicalODE,
    RenalODE,
    RespiratoryODE,
    TumorBiopsy,
    TumorClone,
    TumorHeterogeneity,
    _GenericDiseaseModel,
    create_disease_model,
    select_targeted_therapy,
)


class TestCardiovascularODE:
    def test_step_defaults(self):
        m = CardiovascularODE()
        m.step(1.0)
        assert m.co_l_min > 0.0

    def test_step_drug_and_failure(self):
        m = CardiovascularODE(
            heart_failure_severity=0.8,
            hypertension_severity=0.9,
            atherosclerosis_severity=0.7,
            blood_volume_l=2.0,
        )
        m.step(2.0, drug_svr_mod=0.7, drug_volume_mod=0.5)
        assert m.blood_volume_l >= 3.0
        assert m.map_mmhg <= 180.0

    def test_step_high_volume(self):
        m = CardiovascularODE(blood_volume_l=20.0)
        m.step(1.0)
        assert m.blood_volume_l <= 7.0


class TestMetabolicT2D:
    def test_step_glucotoxicity(self):
        m = MetabolicT2DODE()
        m.t2d_severity = 0.8
        m.step(1.0, glucose_mg_dl=250.0, insulin_uuml=30.0,
               drug_effectiveness=0.8)
        assert m.beta_cell_function < 1.0

    def test_step_normal(self):
        m = MetabolicT2DODE()
        m.step(1.0, glucose_mg_dl=100.0)
        assert m.hepatic_glr > 0.0


class TestCancerODE:
    def test_step_growth_below_capacity(self):
        m = CancerODE(tumor_volume=0.3)
        m.chemo_kill_rate = 0.001
        m.targeted_inhibition = 0.002
        m.immunotherapy_boost = 0.5
        m.pathway_effects = {"egfr": 0.5, "pd_l1": 0.8, "ctla4": 0.4,
                             "vegfr": 0.6}
        m.step(1.0)
        assert m.tumor_volume >= 0.0

    def test_step_at_capacity(self):
        m = CancerODE(tumor_volume=1.0, carrying_capacity=1.0)
        m.step(1.0)
        assert m.tumor_volume == 0.0 or m.tumor_volume >= 0.0

    def test_step_heterogeneity_positive(self):
        het = TumorHeterogeneity(clones=[
            TumorClone(name="parent", fraction=1.0, growth_rate=0.01,
                       drug_sensitivities={"egfr": 0.9}),
        ])
        m = CancerODE(tumor_volume=0.3, heterogeneity=het)
        m.pathway_effects = {"egfr": 0.5}
        m.step(1.0)
        assert len(het.clones) >= 1

    def test_step_heterogeneity_zero_volume(self):
        class ZeroHet:
            def step(self, dt_h, pathway_effects, drug_kill_capacity=0.05):
                return {"total_volume": 0.0}

        m = CancerODE(tumor_volume=0.3, heterogeneity=ZeroHet())
        m.pathway_effects = {"egfr": 0.5}
        m.step(1.0)
        assert m.tumor_volume == 0.0


class TestTumorClone:
    def test_effective_growth(self):
        assert TumorClone().effective_growth() == pytest.approx(0.01)
        c = TumorClone(growth_rate=0.2, resistance_mutations=["EGFR_T790M", "MET"]
                       , fitness_cost=0.3)
        assert c.effective_growth() == pytest.approx(0.08)

    def test_drug_kill_rate(self):
        c = TumorClone(drug_sensitivities={"egfr": 0.5})
        assert c.drug_kill_rate({"egfr": 0.8, "unknown": 0.5}) == \
            pytest.approx(0.9 * 0.05)


class TestTumorHeterogeneity:
    def test_empty_clones(self):
        h = TumorHeterogeneity(clones=[])
        out = h.step(1.0, {})
        assert out["total_volume"] == 0.0

    def test_step_zero_fraction_clone(self):
        h = TumorHeterogeneity(clones=[
            TumorClone(name="dead", fraction=0.0, growth_rate=0.01),
        ])
        out = h.step(1.0, {"egfr": 0.4})
        assert out["total_volume"] == 0.0

    def test_step_main_loop(self):
        h = TumorHeterogeneity(clones=[
            TumorClone(name="parent", fraction=1.0, growth_rate=0.02,
                       drug_sensitivities={"egfr": 0.9}),
            TumorClone(name="res", fraction=0.5, growth_rate=0.01,
                       resistance_mutations=["KRAS"])],
        )
        out = h.step(1.0, {"egfr": 0.5})
        assert out["total_volume"] > 0.0
        assert out["resistant_fraction"] > 0.0

    def test_step_effect_below_threshold(self):
        h = TumorHeterogeneity(clones=[
            TumorClone(name="parent", fraction=1.0, growth_rate=0.01)])
        out = h.step(1.0, {"egfr": 0.005})
        assert out["total_volume"] > 0.0

    def test_resistance_emergence(self):
        h = TumorHeterogeneity(
            clones=[TumorClone(name="parent", fraction=1.0, growth_rate=0.01)],
            resistance_rate=1.0,
        )
        h.step(1.0, {"egfr": 0.5})
        assert any("_R_" in c.name for c in h.clones)

    def test_get_clone_summary_filters(self):
        h = TumorHeterogeneity(clones=[
            TumorClone(name="a", fraction=1.0),
            TumorClone(name="b", fraction=0.0001),
        ])
        names = [s["name"] for s in h.get_clone_summary()]
        assert names == ["a"]


class TestTumorBiopsy:
    def test_matches(self):
        b = TumorBiopsy(
            mutations=["EGFR_L858R", "BRAF_V600E", "KRAS_G12C", "FLT3_ITD",
                       "IDH1_R132H"],
            amplifications=["HER2"],
            pd_l1_expression=0.8,
            msi_status="MSI-H",
            tmb_per_mb=20.0,
            hr_status="HRD",
            fusion_genes=["EML4-ALK", "BCR-ABL"],
        )
        assert b.has_mutation("egfr")
        assert not b.has_mutation("TP53")
        assert b.has_amplification("her2")
        assert not b.has_amplification("MET")
        assert b.has_fusion("ALK")
        assert not b.has_fusion("ROS1")


class TestSelectTargetedTherapy:
    def test_all_conditions(self):
        bio = TumorBiopsy(
            mutations=["EGFR_L858R", "BRAF_V600E", "KRAS_G12C", "FLT3_ITD",
                       "IDH1_R132H"],
            amplifications=["HER2"],
            pd_l1_expression=0.7,
            msi_status="MSI-H",
            tmb_per_mb=12.0,
            hr_status="HRD",
            fusion_genes=["EML4-ALK", "BCR-ABL"],
        )
        recs = select_targeted_therapy(bio)
        drugs = {r["drug"] for r in recs}
        assert "osimertinib" in drugs
        assert "pembrolizumab" in drugs
        assert "imatinib" in drugs
        assert "olaparib" in drugs
        assert any(r["pathway"] == "braf" for r in recs)
        assert any(r["priority"] == "high" for r in recs)

    def test_no_matches(self):
        bio = TumorBiopsy()
        assert select_targeted_therapy(bio) == []

    def test_partial_conditions(self):
        bio = TumorBiopsy(pd_l1_expression=0.4, tmb_per_mb=3.0,
                          msi_status="MSS", hr_status="HRC")
        assert select_targeted_therapy(bio) == []

    def test_unknown_condition_falls_through(self, monkeypatch):
        custom = [("CUSTOM", "custom_condition", ["placebo"], "r"),
                  ("BRCA_HRD", "hrd", ["olaparib"],
                   "BRCA/HRD: PARP inhibitor")]
        monkeypatch.setattr(dome, "_THERAPY_RULES", custom)
        bio = TumorBiopsy(hr_status="HRD")
        recs = select_targeted_therapy(bio)
        assert "olaparib" in [r["drug"] for r in recs]


class TestDiseasesODE:
    def test_autoimmune_ra(self):
        m = AutoimmuneRAODE(
            joint_inflammation=0.6, baseline_severity=0.2,
            dmard_effect=0.5, nsaid_effect=0.3, biologic_effect=0.8,
        )
        m.step(1.0)
        assert 0.0 <= m.joint_inflammation <= 1.0
        assert m.synovial_tnf >= 0.0
        assert m.erosive_damage <= 1.0

    def test_autoimmune_ra_no_ceiling(self):
        m = AutoimmuneRAODE(joint_inflammation=0.05, baseline_severity=0.9)
        m.step(1.0)
        assert m.joint_inflammation <= 1.0

    def test_neurological(self):
        m = NeurologicalODE(
            synaptic_density=0.9, neuroinflammation=0.5,
            disease_modifying_effect=0.6, cholinesterase_inhibition=0.4,
        )
        m.step(1.0)
        assert 0.0 <= m.synaptic_density <= 1.0
        assert m.cognitive_score <= 1.0

    def test_renal_compensates(self):
        m = RenalODE(nephron_mass=0.5, proteinuria=2.0, acei_effect=0.6,
                     sglt2_effect=0.5)
        m.step(1.0)
        assert m.nephron_mass <= 1.0
        assert m.proteinuria >= 0.0

    def test_renal_full_mass(self):
        m = RenalODE(nephron_mass=1.0)
        m.step(1.0)
        assert m.nephron_mass <= 1.0

    def test_hepatic(self):
        m = HepaticODE(
            fibrosis_stage=2.0, hepatic_inflammation=0.8,
            antiviral_effect=0.9, anti_fibrotic_effect=0.5,
        )
        m.step(1.0)
        assert 0.0 <= m.fibrosis_stage <= 4.0
        assert m.portal_pressure > 5.0

    def test_hematological(self):
        m = HematologicalODE(
            stem_cell_pool=0.4, hypomethylating_effect=0.7,
            growth_factor_effect=0.5,
        )
        m.step(1.0)
        assert m.stem_cell_pool >= 0.05
        assert m.blast_percentage <= 90.0

    def test_respiratory(self):
        m = RespiratoryODE(asthma_severity=0.6, copd_severity=0.4,
                           smoking_effect=0.3, inflammation_score=0.5)
        m.step(1.0, drug_bronchodilator=0.5, drug_anti_inflammatory=0.4)
        assert 0.0 < m.airway_resistance < 2.5
        assert m.fev1_percent >= 15.0

    def test_respiratory_healthy(self):
        m = RespiratoryODE()
        m.step(1.0)
        assert m.airway_resistance == pytest.approx(1.0)

    def test_gastrointestinal(self):
        m = GastrointestinalODE(gerd_severity=0.7, ibd_severity=0.6)
        m.step(1.0, drug_acid_suppression=0.6, drug_anti_inflammatory=0.5)
        assert m.acid_secretion >= 0.1
        assert m.pain_score >= 0.0

    def test_gastrointestinal_healthy(self):
        m = GastrointestinalODE()
        m.step(1.0)
        assert m.acid_secretion <= 1.0

    def test_endocrine_hypothyroid(self):
        m = EndocrineODE(hypothyroid_severity=0.8)
        m.step(1.0, drug_t4_supplement=0.5)
        assert m.t4_level >= 20.0
        assert m.tsh_level >= 0.1

    def test_endocrine_hyperthyroid(self):
        m = EndocrineODE(hyperthyroid_severity=0.8)
        m.step(1.0, drug_antithyroid=0.5)
        assert m.t4_level <= 400.0
        assert m.metabolic_rate >= 0.3

    def test_endocrine_healthy(self):
        m = EndocrineODE()
        m.step(1.0)
        assert m.t4_level == pytest.approx(120.0)

    def test_infectious_clearance(self):
        m = InfectiousDiseaseODE(
            viral_bacterial_load=2.0, hiv_severity=1.0,
        )
        m.step(1.0, drug_effectiveness=0.9)
        assert m.viral_bacterial_load <= 2.0
        assert m.cd4_count >= 20.0
        assert m.immune_function >= 0.05

    def test_infectious_no_drug(self):
        m = InfectiousDiseaseODE(bacterial_severity=0.8)
        m.step(1.0, drug_effectiveness=0.0)
        assert m.inflammation <= 1.0

    def test_generic(self):
        m = _GenericDiseaseModel(severity=0.3)
        m.step(1.0, drug_effectiveness=0.8)
        assert m.severity <= 1.0
        assert m.liver_function >= 0.3


class TestCreateDiseaseModel:
    @pytest.mark.parametrize("name,expected", [
        ("essential hypertension", CardiovascularODE),
        ("type 2 diabetes", MetabolicT2DODE),
        ("nsclc", CancerODE),
        ("rheumatoid arthritis", AutoimmuneRAODE),
        ("alzheimer's disease", NeurologicalODE),
        ("ckd stage 3", RenalODE),
        ("cirrhosis", HepaticODE),
        ("mds", HematologicalODE),
        ("asthma", RespiratoryODE),
        ("copd", RespiratoryODE),
        ("hiv", InfectiousDiseaseODE),
        ("tuberculosis", InfectiousDiseaseODE),
        ("bacterial pneumonia", InfectiousDiseaseODE),
        ("gerd", GastrointestinalODE),
        ("crohn's disease", GastrointestinalODE),
        ("colitis", GastrointestinalODE),
        ("hypothyroidism", EndocrineODE),
        ("hyperthyroidism", EndocrineODE),
        ("unknown illness", _GenericDiseaseModel),
    ])
    def test_name_dispatch(self, name, expected):
        assert isinstance(create_disease_model(name, severity=0.6), expected)

    @pytest.mark.parametrize("cat,expected", [
        ("cardiovascular", CardiovascularODE),
        ("respiratory", RespiratoryODE),
        ("neurological", NeurologicalODE),
        ("metabolic", MetabolicT2DODE),
        ("infectious", InfectiousDiseaseODE),
        ("hematological", HematologicalODE),
        ("autoimmune", AutoimmuneRAODE),
        ("endocrine", EndocrineODE),
        ("gastrointestinal", GastrointestinalODE),
        ("hepatic", HepaticODE),
        ("liver", HepaticODE),
        ("renal", RenalODE),
        ("kidney", RenalODE),
        ("oncology", CancerODE),
        ("cancer", CancerODE),
        ("unknown", _GenericDiseaseModel),
    ])
    def test_category_dispatch(self, cat, expected):
        assert isinstance(create_disease_model("zz", severity=0.5,
                                               category=cat), expected)

    def test_no_category_generic(self):
        assert isinstance(
            create_disease_model("something odd", severity=0.5),
            _GenericDiseaseModel)
