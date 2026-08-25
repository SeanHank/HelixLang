"""Tests for post-treatment recovery modeling (doc/28)."""
from __future__ import annotations

from helixlang.human.recovery import RecoveryModel, create_recovery_model


class TestRecoveryModel:
    def test_creation(self):
        baseline = {"ALT": 25.0, "creatinine": 1.0, "WBC": 7000.0, "hemoglobin": 15.0}
        model = create_recovery_model(["cisplatin", "metformin"], baseline)
        assert isinstance(model, RecoveryModel)
        assert model.is_treatment_active is True

    def test_set_treatment_inactive(self):
        baseline = {"ALT": 25.0}
        model = create_recovery_model(["metformin"], baseline)
        model.set_treatment_inactive()
        assert model.is_treatment_active is False

    def test_step_returns_biomarkers(self):
        baseline = {"ALT": 25.0, "creatinine": 1.0}
        model = create_recovery_model(["cisplatin"], baseline)
        result = model.step(dt_h=24.0, current_time_h=24.0)
        assert isinstance(result, dict)
        assert "ALT" in result

    def test_recovery_fraction_range(self):
        baseline = {"ALT": 25.0}
        model = create_recovery_model(["metformin"], baseline)
        frac = model.get_organ_recovery_fraction("liver", current_time_h=48.0)
        assert 0.0 <= frac <= 1.0

    def test_sequela_for_cisplatin(self):
        baseline = {"ALT": 25.0}
        model = create_recovery_model(["cisplatin"], baseline)
        assert len(model.sequela_list) > 0
        names = [s.name for s in model.sequela_list]
        assert any("ototox" in n.lower() for n in names)

    def test_biomarker_relaxes_toward_baseline(self):
        baseline = {"ALT": 25.0, "creatinine": 1.0}
        model = create_recovery_model(["metformin"], baseline)
        model.current_biomarkers["ALT"] = 200.0
        model.set_treatment_inactive()
        for _ in range(30):
            model.step(dt_h=24.0, current_time_h=24.0 * _)
        assert model.current_biomarkers["ALT"] < 200.0
