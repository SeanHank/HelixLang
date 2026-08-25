"""Tests for helixlang.human.physiology (doc/27 Stage A domain layer).

Covers the reference 70 kg adult male, TISSUE_PROFILES organ tables
(literature-anchored volumes and perfusion from Guyton & Hall 2016),
and derived anthropometrics used by the PBPK stack.
"""
from __future__ import annotations

import pytest

from helixlang.human.physiology import (
    DEFAULT_HUMAN,
    TISSUE_PROFILES,
    OrganSpec,
    create_default_physiology,
)

EXPECTED_ORGANS = {"liver", "kidney", "brain", "heart", "muscle", "adipose"}

EXPECTED_VOLUMES_ML = {
    "liver": 1500.0,
    "kidney": 300.0,
    "brain": 1400.0,
    "heart": 300.0,
    "muscle": 24000.0,
    "adipose": 15000.0,
}

EXPECTED_FLOWS_ML_PER_MIN = {
    "liver": 1500.0,
    "kidney": 1200.0,
    "brain": 750.0,
    "heart": 250.0,
    "muscle": 750.0,
    "adipose": 200.0,
}


def test_default_human_70kg():
    """DEFAULT_HUMAN anchors to the doc/27 reference 70 kg adult."""
    assert DEFAULT_HUMAN.body_weight_kg == pytest.approx(70.0)


def test_create_default_physiology():
    """create_default_physiology() yields a fresh independent instance."""
    phys = create_default_physiology()
    assert isinstance(phys, type(DEFAULT_HUMAN))
    assert phys is not DEFAULT_HUMAN
    assert phys.organs.keys() == DEFAULT_HUMAN.organs.keys()
    assert phys.get_organ("liver") is not DEFAULT_HUMAN.get_organ("liver")


def test_tissue_profiles_all_organs():
    """All six doc/27 organs have tissue profiles."""
    assert set(TISSUE_PROFILES) == EXPECTED_ORGANS
    for name in EXPECTED_ORGANS:
        profile = TISSUE_PROFILES[name]
        assert profile["volume_ml"] > 0
        assert profile["blood_flow_ml_per_min"] > 0
        assert profile["key_reactions"]


def test_organ_volumes():
    """Organ volumes match the literature values of doc/27 section 4."""
    phys = create_default_physiology()
    for name, volume_ml in EXPECTED_VOLUMES_ML.items():
        assert phys.get_organ(name).volume_ml == pytest.approx(volume_ml)


def test_blood_flows():
    """Resting perfusion matches Guyton & Hall fractions of cardiac output."""
    phys = create_default_physiology()
    for name, flow in EXPECTED_FLOWS_ML_PER_MIN.items():
        assert phys.get_organ(name).blood_flow_ml_per_min == pytest.approx(flow)
    liver_fraction = phys.organ_flow_fraction("liver")
    assert liver_fraction == pytest.approx(1500.0 / 5000.0)


def test_bsa_calculation():
    """Du Bois body surface area is positive and physiologically plausible."""
    bsa = DEFAULT_HUMAN.body_surface_area_m2
    assert bsa > 0.0
    assert 1.4 < bsa < 2.2


def test_blood_volume():
    """Blood volume exceeds plasma volume whenever hematocrit is positive."""
    phys = create_default_physiology()
    assert phys.hematocrit == pytest.approx(0.45)
    assert phys.blood_volume_ml > phys.plasma_volume_ml
    assert phys.red_cell_volume_ml > 0.0


def test_organ_spec_from_profile_properties():
    """Boolean flags outside core keys land in OrganSpec.properties."""
    heart = OrganSpec.from_profile("heart")
    assert heart.has_property("fatty_acid_oxidation")
    assert not heart.has_property("nonexistent_flag")
    assert heart.parenchymal_mass_kg > 0.0


def test_invalid_physiology_rejected():
    """validate() rejects impossible anthropometric parameters."""
    with pytest.raises(ValueError):
        create_default_physiology(body_weight_kg=-1.0)
    with pytest.raises(ValueError):
        create_default_physiology(sex="unknown")
