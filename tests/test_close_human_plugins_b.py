"""Close coverage gaps in additional human plugin modules (batch b)."""
from __future__ import annotations

import pytest

from helixlang.plugins.human.adaptive import (
    AdaptiveImmuneModel,
    PD1Checkpoint,
    cohort_adaptive_step,
)
from helixlang.plugins.human.ddi import (
    DDIModel,
    DDIRule,
    assess_additive_toxicity,
    create_default_ddi_model,
)
from helixlang.plugins.human.pharmacogenomic_ae import (
    AERisk,
    GenotypeAEPredictor,
)
from helixlang.plugins.human.phenotype import (
    ExternalTraits,
    _alcohol_cyp_factor,
)
from helixlang.plugins.human.tissue_blood import (
    TissueBloodModel,
    cohort_tissue_blood_step,
)


# ---------------------------------------------------------------------------
# adaptive.py
# ---------------------------------------------------------------------------
class TestAdaptiveCoverage:
    def test_pd1_immune_brake_bounded(self):
        mm = AdaptiveImmuneModel()
        mm.set_checkpoint_therapy(anti_pd1=0.5)
        assert 0.0 <= mm.pd1.immune_brake() <= 1.0

    def test_effective_blockade_network_preferred(self):
        mm = AdaptiveImmuneModel()
        mm.set_checkpoint_therapy(anti_pd1=1.0)
        bd = mm.effective_checkpoint_blockade()
        assert bd == 1.0 - mm.pd1.immune_brake()

    def test_cohort_scalar_fallback_with_doses(self):
        models = [AdaptiveImmuneModel(), AdaptiveImmuneModel()]
        out = cohort_adaptive_step(models, 1.0, [0.1, 0.1],
                                   doses=[0.2, None], use_numpy=False)
        assert len(out) == 2 and all(v >= 0.0 for v in out)

    def test_cohort_scalar_fallback_single(self):
        models = [AdaptiveImmuneModel()]
        out = cohort_adaptive_step(models, 1.0, [0.1], use_numpy=False)
        assert len(out) == 1

    def test_cohort_scalar_fallback_doses_positional(self):
        models = [AdaptiveImmuneModel()]
        out = cohort_adaptive_step(models, 1.0, [0.1], [None],
                                   use_numpy=False)
        assert len(out) == 1

    def test_effective_blockade_direct(self):
        cp = PD1Checkpoint()
        assert cp.effective_blockade() == 1.0 - cp.immune_brake()


# ---------------------------------------------------------------------------
# ddi.py
# ---------------------------------------------------------------------------
class TestDDIRuleValidation:
    def test_bad_interaction_type(self):
        with pytest.raises(ValueError):
            DDIRule(substrate="s", interacting_drug="d", enzyme="CYP3A4",
                    interaction_type="nope", fold_change=1.0)

    def test_bad_severity(self):
        with pytest.raises(ValueError):
            DDIRule(substrate="s", interacting_drug="d", enzyme="CYP3A4",
                    interaction_type="inhibition", fold_change=1.0,
                    severity="fatal")

    def test_nonpositive_fold_change(self):
        with pytest.raises(ValueError):
            DDIRule(substrate="s", interacting_drug="d", enzyme="CYP3A4",
                    interaction_type="inhibition", fold_change=0.0)


class TestDDIFallbackEffect:
    @staticmethod
    def _rule(**kw):
        defaults = dict(substrate="sub", interacting_drug="perpetrator",
                        enzyme="CYP3A4", interaction_type="inhibition",
                        fold_change=2.0)
        defaults.update(kw)
        return DDIRule(**defaults)

    def test_enzyme_state_limits(self):
        rule = self._rule(interacting_drug="CYP3A4", fold_change=0.5,
                          clinical_effect="")
        assert "limits" in DDIModel()._fallback_effect(rule)

    def test_enzyme_state_accelerates(self):
        rule = self._rule(interacting_drug="CYP3A4", fold_change=2.0,
                          clinical_effect="")
        assert "accelerates" in DDIModel()._fallback_effect(rule)

    def test_drug_inhibits(self):
        rule = self._rule(clinical_effect="")
        assert "inhibits" in DDIModel()._fallback_effect(rule)

    def test_drug_induces(self):
        rule = self._rule(interaction_type="induction", fold_change=2.0,
                          clinical_effect="")
        assert "induces" in DDIModel()._fallback_effect(rule)


class TestDDIRuleTriggered:
    def test_enzyme_state_induction_ultrarapid(self):
        m = DDIModel()
        r = DDIRule(substrate="sub", interacting_drug="CYP3A4",
                    enzyme="CYP3A4", interaction_type="induction",
                    fold_change=2.0)
        assert m._rule_triggered(r, {"sub"}, {"CYP3A4": 2.0}) is True
        assert m._rule_triggered(r, {"sub"}, {"CYP3A4": 1.0}) is False

    def test_enzyme_state_deficient(self):
        m = DDIModel()
        r = DDIRule(substrate="sub", interacting_drug="CYP3A4",
                    enzyme="CYP3A4", interaction_type="inhibition",
                    fold_change=0.5)
        assert m._rule_triggered(r, {"sub"}, {"CYP3A4": 0.1}) is True
        assert m._rule_triggered(r, {"sub"}, {"CYP3A4": 1.0}) is False

    def test_drug_in_name_set(self):
        m = DDIModel()
        r = DDIRule(substrate="sub", interacting_drug="perpetrator",
                    enzyme="CYP3A4", interaction_type="inhibition",
                    fold_change=0.5)
        assert m._rule_triggered(r, {"sub", "perpetrator"}, {}) is True
        assert m._rule_triggered(r, {"sub"}, {}) is False


class TestDDIComputeAndAlerts:
    def test_skips_additive_rules(self):
        m = DDIModel(rules=[
            DDIRule(substrate="s", interacting_drug="d", enzyme="E",
                    interaction_type="additive_toxicity", fold_change=5.0),
        ])
        assert m.compute_clearance_modifiers(["s"], {}) == {"s": 1.0}

    def test_clearance_floor(self):
        m = DDIModel(rules=[
            DDIRule(substrate="s", interacting_drug="d", enzyme="E",
                    interaction_type="inhibition", fold_change=0.0001),
        ])
        mods = m.compute_clearance_modifiers(["s", "d"], {})
        assert mods["s"] >= 0.01
        assert mods["d"] == 1.0

    def test_clinical_alerts_fallback_effect(self):
        m = DDIModel(rules=[
            DDIRule(substrate="s", interacting_drug="d", enzyme="E",
                    interaction_type="inhibition", fold_change=0.5,
                    clinical_effect=""),
        ])
        alerts = m.get_clinical_alerts(["s", "d"], {})
        assert len(alerts) == 1 and "inhibits" in alerts[0]["effect"]

    def test_clinical_alerts_dedupe(self):
        m = DDIModel(rules=[
            DDIRule(substrate="s", interacting_drug="d1", enzyme="E",
                    interaction_type="inhibition", fold_change=0.5),
            DDIRule(substrate="s", interacting_drug="d1", enzyme="E",
                    interaction_type="inhibition", fold_change=0.5),
        ])
        assert len(m.get_clinical_alerts(["s", "d1"], {})) == 1

    def test_clinical_alerts_no_match(self):
        m = DDIModel(rules=[
            DDIRule(substrate="other", interacting_drug="d", enzyme="E",
                    interaction_type="inhibition", fold_change=0.5),
        ])
        assert m.get_clinical_alerts(["s", "d"], {}) == []


class TestDDIAdditiveToxicity:
    def test_two_qt_drugs_severe(self):
        qt = [a for a in assess_additive_toxicity(["tamoxifen", "ondansetron"])
              if a["toxicity_type"] == "qt_prolongation"]
        assert qt and qt[0]["severity"] == "severe"

    def test_single_qt_drug_moderate(self):
        qt = [a for a in assess_additive_toxicity(["tamoxifen"])
              if a["toxicity_type"] == "qt_prolongation"]
        assert qt and qt[0]["severity"] == "moderate"

    def test_cisplatin_nsaid_severe(self):
        neph = [a for a in assess_additive_toxicity(["cisplatin", "ibuprofen"])
                if a["toxicity_type"] == "nephrotoxicity"]
        assert neph and neph[0]["severity"] == "severe"

    def test_two_renal_hits_moderate(self):
        neph = [a for a in assess_additive_toxicity(["gentamicin", "vancomycin"])
                if a["toxicity_type"] == "nephrotoxicity"]
        assert neph and neph[0]["severity"] == "moderate"

    def test_metformin_plus_renal(self):
        neph = [a for a in assess_additive_toxicity(["metformin", "gentamicin"])
                if a["toxicity_type"] == "nephrotoxicity"]
        assert neph and neph[0]["drugs"] == ["gentamicin", "metformin"]

    def test_two_hepatic_hits_moderate(self):
        hep = [a for a in assess_additive_toxicity(["imatinib", "tamoxifen"])
               if a["toxicity_type"] == "hepatotoxicity"]
        assert hep and hep[0]["severity"] == "moderate"

    def test_three_hepatic_hits_severe(self):
        hep = [a for a in
               assess_additive_toxicity(["imatinib", "tamoxifen", "atorvastatin"])
               if a["toxicity_type"] == "hepatotoxicity"]
        assert hep and hep[0]["severity"] == "severe"

    def test_no_alerts(self):
        assert assess_additive_toxicity(["aspirin"]) == []


class TestDDIDefaultModel:
    def test_create_default(self):
        m = create_default_ddi_model()
        assert len(m.rules) > 0
        assert isinstance(
            m.get_clinical_alerts([r.substrate for r in m.rules], {}), list)


# ---------------------------------------------------------------------------
# tissue_blood.py
# ---------------------------------------------------------------------------
class TestTissueBloodCoverage:
    def test_getters_return_indices(self):
        m = TissueBloodModel()
        m.blood_neutrophils = 5.0
        m.tissue_neutrophils = 9.0
        m.tissue_il6 = 25.0
        m.tissue_macrophages = 3.0
        assert m.get_tissue_blood_divergence() >= 0.0

    def test_cohort_scalar_fallback(self):
        models = [TissueBloodModel()]
        out = cohort_tissue_blood_step(models, 0.5, [1.0],
                                       blood_il6=[10.0],
                                       blood_neutrophils=[5.0],
                                       blood_monocytes=[0.5],
                                       use_numpy=False)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# pharmacogenomic_ae.py
# ---------------------------------------------------------------------------
class TestAEPredictorCoverage:
    def _predictor(self):
        return GenotypeAEPredictor()

    def test_high_prod_activity_mechanism(self):
        preds = self._predictor().predict_ae(
            "acetaminophen", 1000.0, {"CYP2E1": 2.0, "GST": 1.0})
        assert preds and preds[0].risk_level in list(AERisk)

    def test_low_ae_prob_minimal(self):
        preds = self._predictor().predict_ae(
            "acetaminophen", 1e-6, {"CYP2E1": 0.1, "GST": 0.1})
        assert preds and preds[0].ae_probability < 0.05

    def test_predict_all(self):
        out = self._predictor().predict_all(
            {"acetaminophen": 100.0}, {"CYP2E1": 1.0, "GST": 1.0})
        assert "acetaminophen" in out


# ---------------------------------------------------------------------------
# phenotype.py
# ---------------------------------------------------------------------------
class TestPhenotypeValidation:
    def test_bad_weight(self):
        with pytest.raises(ValueError):
            ExternalTraits(body_weight_kg=0.0)

    def test_bad_height(self):
        with pytest.raises(ValueError):
            ExternalTraits(height_cm=0.0)

    def test_bad_age(self):
        with pytest.raises(ValueError):
            ExternalTraits(age_years=-1.0)

    def test_bad_sex(self):
        with pytest.raises(ValueError):
            ExternalTraits(sex="robot")

    def test_bad_ethnicity(self):
        with pytest.raises(ValueError):
            ExternalTraits(ethnicity="martian")

    def test_bad_smoking(self):
        with pytest.raises(ValueError):
            ExternalTraits(smoking_status="chainsmoker")

    def test_bad_exercise(self):
        with pytest.raises(ValueError):
            ExternalTraits(exercise_level="all_day")

    def test_negative_pack_years(self):
        with pytest.raises(ValueError):
            ExternalTraits(pack_years=-1.0)

    def test_negative_gestational(self):
        with pytest.raises(ValueError):
            ExternalTraits(gestational_weeks=-1.0)

    def test_pregnant_zero_weeks(self):
        with pytest.raises(ValueError):
            ExternalTraits(pregnant=True, gestational_weeks=0.0)


class TestPhenotypeCyPForce:
    def test_alcohol_heavy(self):
        assert _alcohol_cyp_factor("CYP2E1", 999.0) > 1.0

    def test_alcohol_moderate(self):
        assert _alcohol_cyp_factor("CYP2E1", 8.0) > 1.0

    def test_alcohol_other_enzyme(self):
        assert _alcohol_cyp_factor("CYP1A2", 999.0) == 1.0

    def test_alcohol_low(self):
        assert _alcohol_cyp_factor("CYP2E1", 0.0) == 1.0
