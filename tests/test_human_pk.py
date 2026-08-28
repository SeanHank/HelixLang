"""Tests for helixlang.plugins.human.pharmacokinetics (doc/27 PBPK layer).

Covers the six-compartment well-stirred PBPK model: configuration
defaults, route-specific dosing inputs (first-order oral absorption,
iv bolus initial condition), and non-compartmental summary endpoints
(AUC, Cmax, terminal half-life).
"""
from __future__ import annotations

import pytest

from helixlang.plugins.human.drug import IV, get_predefined_drug
from helixlang.plugins.human.pharmacokinetics import (
    ORGAN_NAMES,
    PBPKConfig,
    PBPKModel,
    PBPKResult,
)
from helixlang.plugins.human.physiology import create_default_physiology


def _physiology():
    return create_default_physiology()


def _oral_ibuprofen_model(total_time_h: float = 24.0) -> PBPKModel:
    """Oral ibuprofen on the reference physiology with coarse sampling."""
    drug = get_predefined_drug("IBUPROFEN")
    return PBPKModel(
        drug, _physiology(),
        PBPKConfig(dt_min=10.0, total_time_h=total_time_h),
    )


def test_pbpk_config_defaults():
    """PBPKConfig anchors to the doc/27 reference geometry."""
    config = PBPKConfig()
    assert config.dt_min == pytest.approx(1.0)
    assert config.total_time_h == pytest.approx(24.0)
    assert config.plasma_volume_l == pytest.approx(3.0)
    with pytest.raises(ValueError):
        PBPKConfig(dt_min=0.0).validate()


def test_pbpk_model_creation():
    """PBPKModel resolves flows, volumes, and clearances from inputs."""
    drug = get_predefined_drug("IBUPROFEN")
    phys = _physiology()
    model = PBPKModel(drug, phys)
    assert model.drug is drug
    assert model.physiology is phys
    assert set(model.organ_flows_l_per_h) == set(ORGAN_NAMES)
    assert all(flow > 0.0 for flow in model.organ_flows_l_per_h.values())
    assert model.cl_total_l_per_h > 0.0
    # renal fraction 0.9 -> kidney carries most clearance
    kidney_cl = model.organ_clearances_l_per_h["kidney"]
    liver_cl = model.organ_clearances_l_per_h["liver"]
    assert kidney_cl > liver_cl


def test_pbpk_run_returns_result():
    """run() returns a fully populated PBPKResult."""
    result = _oral_ibuprofen_model().run()
    assert isinstance(result, PBPKResult)
    assert len(result.time_h) > 1
    assert set(result.concentrations) == set(ORGAN_NAMES)
    assert len(result.central_concentration) == len(result.time_h)


def test_pbpk_cmax_positive():
    """Plasma exposure produces a positive peak concentration."""
    result = _oral_ibuprofen_model().run()
    assert result.c_max > 0.0
    assert result.t_max > 0.0


def test_pbpk_auc_positive():
    """The trapezoidal AUC over the horizon is positive."""
    result = _oral_ibuprofen_model().run()
    assert result.auc > 0.0


def test_pbpk_half_life_positive():
    """Terminal regression (or fallback) yields a positive half-life."""
    result = _oral_ibuprofen_model(total_time_h=8.0).run()
    assert result.half_life_h > 0.0


def test_pbpk_oral_dose():
    """Oral dosing starts at zero and rises through an absorption phase."""
    result = _oral_ibuprofen_model().run()
    assert result.central_concentration[0] == pytest.approx(0.0)
    assert max(result.central_concentration) > 0.0
    assert result.t_max >= result.time_h[0]


def test_pbpk_iv_bolus():
    """IV bolus applies the full dose as the initial central condition."""
    drug = get_predefined_drug("METFORMIN")
    drug.route = IV  # override the oral regimen for a bolus input
    model = PBPKModel(
        drug, _physiology(), PBPKConfig(dt_min=30.0, total_time_h=12.0),
    )
    expected_c0 = drug.dose_mg * drug.bioavailability / model.config.plasma_volume_l
    result = model.run()
    assert result.central_concentration[0] > 0.0
    assert result.central_concentration[0] == pytest.approx(expected_c0)
    assert result.c_max == pytest.approx(result.central_concentration[0])
