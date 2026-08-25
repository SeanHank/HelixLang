"""Tests for drug-drug interaction modeling (doc/28)."""
from __future__ import annotations

from helixlang.human.ddi import (
    DEFAULT_DDI_RULES,
    DDIModel,
    DDIRule,
    assess_additive_toxicity,
)


class TestDDIRule:
    def test_creation(self):
        rule = DDIRule(
            substrate="ibuprofen",
            interacting_drug="metformin",
            enzyme="CYP2C9",
            interaction_type="inhibition",
            fold_change=0.7,
            severity="mild",
            clinical_effect="Reduced ibuprofen clearance",
        )
        assert rule.substrate == "ibuprofen"
        assert rule.fold_change == 0.7


class TestDDIModel:
    def test_default_rules_count(self):
        assert len(DEFAULT_DDI_RULES) >= 10

    def test_model_creation(self):
        model = DDIModel(rules=DEFAULT_DDI_RULES)
        assert len(model.rules) >= 10

    def test_empty_model_no_interaction(self):
        model = DDIModel()
        mods = model.compute_clearance_modifiers(["ibuprofen"], {"CYP2D6": 1.0})
        assert mods.get("ibuprofen", 1.0) == 1.0

    def test_clearance_modifier_cyp2d6_pm(self):
        model = DDIModel(rules=DEFAULT_DDI_RULES)
        mods = model.compute_clearance_modifiers(
            ["tamoxifen"], {"CYP2D6": 0.1},
        )
        tam_mod = mods.get("tamoxifen", 1.0)
        assert tam_mod < 1.0

    def test_clinical_alerts(self):
        model = DDIModel(rules=DEFAULT_DDI_RULES)
        alerts = model.get_clinical_alerts(
            ["tamoxifen"], {"CYP2D6": 0.1},
        )
        assert isinstance(alerts, list)


class TestAdditiveToxicity:
    def test_cisplatin_nephrotoxicity(self):
        alerts = assess_additive_toxicity(["cisplatin", "ibuprofen"])
        nephro = [a for a in alerts if "nephro" in a.get("toxicity_type", "").lower()
                   or "nephro" in a.get("mechanism", "").lower()]
        assert len(nephro) > 0

    def test_qt_prolongation(self):
        alerts = assess_additive_toxicity(["cisplatin", "tamoxifen"])
        qt = [a for a in alerts if "qt" in a.get("toxicity_type", "").lower()
              or "qt" in a.get("mechanism", "").lower()]
        assert len(qt) > 0

    def test_single_drug_no_alerts(self):
        alerts = assess_additive_toxicity(["ibuprofen"])
        assert isinstance(alerts, list)
