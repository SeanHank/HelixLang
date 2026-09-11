"""Tests for helixlang.plugins.human.recovery.

Covers: RecoveryEvent validation, ReboundSpec.envelope branches,
RecoveryModel.step with active/inactive treatment, check_rebound,
get_organ_recovery_fraction, and create_recovery_model factory.
"""
from __future__ import annotations

import math

import pytest

from helixlang.plugins.human.recovery import (
    ReboundSpec,
    RecoveryEvent,
    RecoveryModel,
    Sequela,
    _rate_from_half_life,
    create_recovery_model,
)

# ── _rate_from_half_life ─────────────────────────────────────────────────


def test_rate_from_half_life():
    assert _rate_from_half_life(24.0) == pytest.approx(math.log(2) / 24.0)


def test_rate_from_half_life_zero_raises():
    with pytest.raises(ValueError, match="half_life_h must be > 0"):
        _rate_from_half_life(0.0)


def test_rate_from_half_life_negative_raises():
    with pytest.raises(ValueError):
        _rate_from_half_life(-1.0)


# ── RecoveryEvent ────────────────────────────────────────────────────────


def test_recovery_event_valid():
    e = RecoveryEvent(time_h=1.0, event_type="rebound", organ="liver", description="test")
    assert e.time_h == 1.0


def test_recovery_event_invalid_type():
    with pytest.raises(ValueError, match="event_type must be one of"):
        RecoveryEvent(time_h=0.0, event_type="invalid", organ="liver", description="x")


# ── Sequela ──────────────────────────────────────────────────────────────


def test_sequela_valid():
    s = Sequela(name="ototox", organ="ear", severity=0.5, onset_delay_h=100.0)
    assert s.reversible is False


def test_sequela_severity_out_of_range():
    with pytest.raises(ValueError, match="severity"):
        Sequela(name="x", organ="ear", severity=1.5, onset_delay_h=0.0)


def test_sequela_negative_onset():
    with pytest.raises(ValueError, match="onset_delay_h"):
        Sequela(name="x", organ="ear", severity=0.5, onset_delay_h=-1.0)


def test_sequela_zero_half_life():
    with pytest.raises(ValueError, match="recovery_half_life_h"):
        Sequela(name="x", organ="ear", severity=0.5, onset_delay_h=0.0,
                recovery_half_life_h=0.0)


# ── ReboundSpec ──────────────────────────────────────────────────────────


def test_rebound_spec_valid():
    rs = ReboundSpec(name="wd", biomarker="pain_score", excursion_fraction=0.5,
                     direction=1, onset_delay_h=24.0, duration_h=48.0)
    assert rs.name == "wd"


def test_rebound_spec_bad_excursion():
    with pytest.raises(ValueError, match="excursion_fraction"):
        ReboundSpec(name="x", biomarker="y", excursion_fraction=0.0)


def test_rebound_spec_bad_direction():
    with pytest.raises(ValueError, match="direction"):
        ReboundSpec(name="x", biomarker="y", excursion_fraction=0.5, direction=2)


def test_rebound_spec_bad_onset():
    with pytest.raises(ValueError, match="onset_delay_h"):
        ReboundSpec(name="x", biomarker="y", excursion_fraction=0.5,
                    onset_delay_h=0.0, duration_h=10.0)


def test_rebound_spec_bad_duration():
    with pytest.raises(ValueError, match="onset_delay_h"):
        ReboundSpec(name="x", biomarker="y", excursion_fraction=0.5,
                    onset_delay_h=10.0, duration_h=0.0)


# ── ReboundSpec.envelope branches ───────────────────────────────────────


def test_envelope_zero():
    rs = ReboundSpec(name="x", biomarker="y", excursion_fraction=0.5)
    assert rs.envelope(0.0) == pytest.approx(0.0)


def test_envelope_negative():
    rs = ReboundSpec(name="x", biomarker="y", excursion_fraction=0.5)
    assert rs.envelope(-1.0) == pytest.approx(0.0)


def test_envelope_ramp_phase():
    rs = ReboundSpec(name="x", biomarker="y", excursion_fraction=0.5,
                     onset_delay_h=24.0, duration_h=48.0)
    val = rs.envelope(12.0)
    assert val == pytest.approx(0.5)


def test_envelope_peak():
    rs = ReboundSpec(name="x", biomarker="y", excursion_fraction=0.5,
                     onset_delay_h=24.0, duration_h=48.0)
    assert rs.envelope(24.0) == pytest.approx(1.0)


def test_envelope_decay():
    rs = ReboundSpec(name="x", biomarker="y", excursion_fraction=0.5,
                     onset_delay_h=24.0, duration_h=48.0)
    val = rs.envelope(48.0)
    assert val == pytest.approx(math.exp(-0.5))


# ── RecoveryModel basic ──────────────────────────────────────────────────


def _make_model(baselines=None, current=None, rates=None):
    baselines = baselines or {"wbc": 8.0}
    current = current or {"wbc": 4.0}
    return RecoveryModel(
        baseline_biomarkers=dict(baselines),
        current_biomarkers=dict(current),
        organ_recovery_rates=rates or {"bone_marrow": _rate_from_half_life(24.0)},
    )


def test_recovery_model_missing_baseline():
    with pytest.raises(ValueError, match="current biomarkers lack baselines"):
        RecoveryModel(baseline_biomarkers={}, current_biomarkers={"x": 1.0})


def test_step_active_treatment_no_change():
    m = _make_model()
    m.step(dt_h=1.0, current_time_h=0.0)
    assert m.current_biomarkers["wbc"] == pytest.approx(4.0)


def test_step_negative_dt_raises():
    m = _make_model()
    with pytest.raises(ValueError, match="dt_h must be >= 0"):
        m.step(dt_h=-1.0, current_time_h=0.0)


def test_step_recovery_start():
    m = _make_model()
    m.set_treatment_inactive()
    m.step(dt_h=1.0, current_time_h=100.0)
    assert m.is_treatment_active is False
    assert len(m.recovery_events) >= 1
    assert m.recovery_events[0].event_type == "recovery_start"


def test_step_recovery_convergence():
    m = _make_model()
    m.set_treatment_inactive()
    for i in range(500):
        m.step(dt_h=10.0, current_time_h=100.0 + i * 10.0)
    assert m.current_biomarkers["wbc"] == pytest.approx(8.0, abs=0.5)


# ── Sequelae expression ─────────────────────────────────────────────────


def test_sequela_expressed():
    seq = Sequela(name="ototox", organ="ear", severity=0.4, onset_delay_h=1.0)
    m = RecoveryModel(
        baseline_biomarkers={"marker": 10.0},
        current_biomarkers={"marker": 10.0},
        sequela_list=[seq],
    )
    m.set_treatment_inactive()
    m.step(dt_h=1.0, current_time_h=2.0)
    assert "ototox" in m._sequela_expressed
    assert any(e.event_type == "sequelae_onset" for e in m.recovery_events)


def test_sequela_already_expressed_skips():
    """A sequela already in the expressed set is skipped (continue)."""
    seq = Sequela(name="ototox", organ="ear", severity=0.4, onset_delay_h=1.0)
    m = RecoveryModel(
        baseline_biomarkers={"marker": 10.0},
        current_biomarkers={"marker": 10.0},
        sequela_list=[seq],
    )
    m._sequela_expressed.add("ototox")
    m.step(dt_h=1.0, current_time_h=2.0)
    assert not any(e.event_type == "sequelae_onset" for e in m.recovery_events)


# ── check_rebound ───────────────────────────────────────────────────────


def test_check_rebound_active_returns_empty():
    m = _make_model()
    assert m.check_rebound(10.0) == []


def test_check_rebound_fires():
    rs = ReboundSpec(name="op_wd", biomarker="pain_score",
                     excursion_fraction=0.5, direction=1,
                     onset_delay_h=12.0, duration_h=48.0)
    m = RecoveryModel(
        baseline_biomarkers={"pain_score": 5.0},
        current_biomarkers={"pain_score": 5.0},
        rebound_specs=[rs],
    )
    m.set_treatment_inactive()
    m.step(dt_h=1.0, current_time_h=0.0)
    events = m.check_rebound(current_time_h=10.0)
    assert len(events) == 1
    assert events[0].event_type == "rebound"
    assert events[0].organ == "cns"


def test_check_rebound_not_yet_mature():
    """A rebound whose onset time has not elapsed stays dormant."""
    rs = ReboundSpec(name="op_wd", biomarker="pain_score",
                     excursion_fraction=0.5, direction=1,
                     onset_delay_h=48.0, duration_h=48.0)
    m = RecoveryModel(
        baseline_biomarkers={"pain_score": 5.0},
        current_biomarkers={"pain_score": 5.0},
        rebound_specs=[rs],
    )
    m.set_treatment_inactive()
    m.step(dt_h=1.0, current_time_h=0.0)
    events = m.check_rebound(current_time_h=1.0)
    assert events == []
    assert "op_wd" not in m._rebound_emitted


def test_rebound_term_skips_other_biomarker():
    """The rebound loop skips specs targeting a different biomarker."""
    rs = ReboundSpec(name="op_wd", biomarker="pain_score",
                     excursion_fraction=0.5, direction=1,
                     onset_delay_h=12.0, duration_h=48.0)
    m = RecoveryModel(
        baseline_biomarkers={"wbc": 8.0, "pain_score": 5.0},
        current_biomarkers={"wbc": 4.0, "pain_score": 3.0},
        rebound_specs=[rs],
    )
    m.set_treatment_inactive()
    m.step(dt_h=1.0, current_time_h=100.0)
    assert 4.0 <= m.current_biomarkers["wbc"] <= 8.0


def test_sequela_penalty_skips_other_organ():
    seq = Sequela(name="ototox", organ="ear", severity=0.4, onset_delay_h=1.0)
    m = RecoveryModel(
        baseline_biomarkers={"marker": 10.0},
        current_biomarkers={"marker": 10.0},
        sequela_list=[seq],
    )
    m.set_treatment_inactive()
    m.step(dt_h=1.0, current_time_h=2.0)
    penalty = m._sequela_penalty("kidney", 1.0)
    assert penalty == 0.0


def test_sequela_non_reversible_no_decay():
    """A non-reversible sequela keeps its full penalty (no exp decay)."""
    seq = Sequela(name="nephrotox", organ="kidney", severity=0.6,
                  onset_delay_h=1.0, reversible=False)
    m = RecoveryModel(
        baseline_biomarkers={"creatinine": 1.0},
        current_biomarkers={"creatinine": 1.0},
        sequela_list=[seq],
    )
    m.set_treatment_inactive()
    m.step(dt_h=1.0, current_time_h=2.0)
    assert "nephrotox" in m._sequela_expressed
    penalty = m._sequela_penalty("kidney", 5.0)
    assert penalty == pytest.approx(0.6)


def test_check_rebound_does_not_fire_twice():
    rs = ReboundSpec(name="op_wd", biomarker="pain_score",
                     excursion_fraction=0.5, direction=1,
                     onset_delay_h=12.0, duration_h=48.0)
    m = RecoveryModel(
        baseline_biomarkers={"pain_score": 5.0},
        current_biomarkers={"pain_score": 5.0},
        rebound_specs=[rs],
    )
    m.set_treatment_inactive()
    m.step(dt_h=1.0, current_time_h=0.0)
    m.check_rebound(current_time_h=10.0)
    second = m.check_rebound(current_time_h=12.0)
    assert len(second) == 0


def test_check_rebound_negative_direction():
    rs = ReboundSpec(name="op_wd", biomarker="pain_score",
                     excursion_fraction=0.5, direction=-1,
                     onset_delay_h=12.0, duration_h=48.0)
    m = RecoveryModel(
        baseline_biomarkers={"pain_score": 5.0},
        current_biomarkers={"pain_score": 5.0},
        rebound_specs=[rs],
    )
    m.set_treatment_inactive()
    m.step(dt_h=1.0, current_time_h=0.0)
    events = m.check_rebound(current_time_h=10.0)
    assert len(events) == 1


# ── get_organ_recovery_fraction ─────────────────────────────────────────


def test_recovery_fraction_active():
    m = _make_model()
    assert m.get_organ_recovery_fraction("bone_marrow", 0.0) == pytest.approx(0.0)


def test_recovery_fraction_no_stop():
    m = _make_model()
    m.is_treatment_active = False
    m._stop_time_h = math.inf
    assert m.get_organ_recovery_fraction("bone_marrow", 10.0) == pytest.approx(0.0)


def test_recovery_fraction_fully_recovered():
    m = _make_model()
    m.set_treatment_inactive()
    m._initial_deviations = {"wbc": -4.0}
    m._residual_deviations = {"wbc": 0.0}
    m._stop_time_h = 0.0
    fr = m.get_organ_recovery_fraction("bone_marrow", 10.0)
    assert fr == pytest.approx(1.0)


def test_recovery_fraction_zero_initial():
    m = _make_model()
    m.set_treatment_inactive()
    m._initial_deviations = {"wbc": 0.0}
    m._residual_deviations = {"wbc": 0.0}
    m._stop_time_h = 0.0
    fr = m.get_organ_recovery_fraction("bone_marrow", 10.0)
    assert fr == pytest.approx(1.0)


def test_recovery_fraction_with_sequela():
    seq = Sequela(name="x", organ="bone_marrow", severity=0.5, onset_delay_h=0.0,
                  reversible=True, recovery_half_life_h=1000.0)
    m = RecoveryModel(
        baseline_biomarkers={"wbc": 8.0},
        current_biomarkers={"wbc": 8.0},
        organ_recovery_rates={"bone_marrow": _rate_from_half_life(24.0)},
        sequela_list=[seq],
    )
    m.set_treatment_inactive()
    m.step(dt_h=1.0, current_time_h=0.0)
    m._initial_deviations = {"wbc": -4.0}
    m._residual_deviations = {"wbc": 0.0}
    m._stop_time_h = 0.0
    m._sequela_expressed.add("x")
    fr = m.get_organ_recovery_fraction("bone_marrow", 10.0)
    assert 0.0 <= fr <= 1.0


def test_recovery_fraction_systemic_fallback():
    m = RecoveryModel(
        baseline_biomarkers={"custom_marker": 10.0},
        current_biomarkers={"custom_marker": 5.0},
        organ_recovery_rates={},
    )
    m.set_treatment_inactive()
    m.step(dt_h=1.0, current_time_h=100.0)
    fr = m.get_organ_recovery_fraction("systemic", 101.0)
    assert 0.0 <= fr <= 1.0


# ── create_recovery_model ───────────────────────────────────────────────


def test_factory_cisplatin():
    m = create_recovery_model(
        drug_names=["cisplatin"],
        baseline_biomarkers={"wbc": 8.0, "egfr_ml_min_1_73m2": 90.0},
    )
    assert m.is_treatment_active is True
    assert len(m.sequela_list) >= 1
    assert m.organ_recovery_rates["kidney"] > 0


def test_factory_anthracycline():
    m = create_recovery_model(
        drug_names=["doxorubicin"],
        baseline_biomarkers={"ejection_fraction": 60.0},
    )
    assert any(s.name == "anthracycline_cardiomyopathy" for s in m.sequela_list)
    assert m.organ_recovery_rates["heart"] > 0


def test_factory_opioid():
    m = create_recovery_model(
        drug_names=["morphine"],
        baseline_biomarkers={"pain_score": 3.0},
    )
    assert len(m.rebound_specs) == 1
    assert m.rebound_specs[0].name == "opioid_withdrawal"


def test_factory_corticosteroid():
    m = create_recovery_model(
        drug_names=["dexamethasone"],
        baseline_biomarkers={"cortisol_ug_dl": 12.0},
    )
    assert len(m.rebound_specs) == 1
    assert m.rebound_specs[0].name == "corticosteroid_withdrawal"


def test_factory_chemo_brain():
    m = create_recovery_model(
        drug_names=["methotrexate"],
        baseline_biomarkers={"wbc": 8.0},
    )
    assert any(s.name == "chemo_brain" for s in m.sequela_list)


def test_factory_no_risk_drug():
    m = create_recovery_model(
        drug_names=["aspirin"],
        baseline_biomarkers={"wbc": 8.0},
    )
    assert len(m.sequela_list) == 0
    assert len(m.rebound_specs) == 0


def test_factory_combined_drugs():
    m = create_recovery_model(
        drug_names=["cisplatin", "doxorubicin", "morphine"],
        baseline_biomarkers={"wbc": 8.0, "ejection_fraction": 60.0, "pain_score": 3.0},
    )
    assert len(m.sequela_list) >= 2
    assert len(m.rebound_specs) >= 1
