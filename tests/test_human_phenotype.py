"""Tests for helixlang.plugins.human.phenotype (doc/28 trait-scaling layer).

Covers :class:`ExternalTraits` anthropometrics and the genotype +
traits folding in :class:`PhenotypeCalculator`: demographic (age,
sex), obesity, smoking, and pregnancy scalings feeding
:class:`~helixlang.plugins.human.physiology.HumanPhysiology`.
"""
from __future__ import annotations

import pytest

from helixlang.plugins.human.genotype import create_default_genotype
from helixlang.plugins.human.phenotype import (
    ExternalTraits,
    PhenotypeCalculator,
    create_default_traits,
)
from helixlang.plugins.human.physiology import DEFAULT_CYP450_ACTIVITY, HumanPhysiology


def _calc(**overrides: object) -> PhenotypeCalculator:
    """Calculator over the reference genotype with overridden traits."""
    return PhenotypeCalculator(
        create_default_genotype(), ExternalTraits(**overrides)
    )


def test_external_traits_defaults():
    """ExternalTraits() defaults to the reference adult male."""
    traits = ExternalTraits()
    assert traits.age_years == pytest.approx(30.0)
    assert traits.sex == "male"
    assert traits.body_weight_kg == pytest.approx(70.0)
    assert traits.height_cm == pytest.approx(170.0)
    assert traits.ethnicity == "european"
    assert traits.smoking_status == "never"
    assert traits.pack_years == pytest.approx(0.0)
    assert traits.alcohol_drinks_per_week == pytest.approx(0.0)
    assert traits.exercise_level == "moderate"
    assert traits.pregnant is False
    assert traits.comorbidities == []


def test_bmi_calculation():
    """bmi equals weight divided by squared height in meters."""
    traits = create_default_traits()
    assert traits.bmi == pytest.approx(70.0 / 1.7**2)

    heavy = ExternalTraits(body_weight_kg=90.0, height_cm=150.0)
    assert heavy.bmi == pytest.approx(90.0 / 1.5**2)


def test_is_obese():
    """is_obese flags BMI at or above the WHO threshold of 30 kg/m^2."""
    assert not ExternalTraits(body_weight_kg=70.0).is_obese
    assert ExternalTraits(body_weight_kg=105.0).is_obese

    at_threshold = ExternalTraits(body_weight_kg=30.0 * 1.7**2)
    assert at_threshold.is_obese


def test_phenotype_calculator_creation():
    """PhenotypeCalculator stores its genotype and traits inputs."""
    geno = create_default_genotype()
    traits = create_default_traits()
    calc = PhenotypeCalculator(genotype=geno, traits=traits)
    assert calc.genotype is geno
    assert calc.traits is traits


def test_compute_physiology():
    """compute_physiology builds a validated six-organ HumanPhysiology."""
    phys = _calc().compute_physiology()
    assert isinstance(phys, HumanPhysiology)
    assert set(phys.organs) == {
        "liver",
        "kidney",
        "brain",
        "heart",
        "muscle",
        "adipose",
    }
    for organ in phys.organs.values():
        assert organ.volume_ml > 0.0
        assert organ.blood_flow_ml_per_min > 0.0
    assert phys.cardiac_output_ml_per_min > 0.0
    assert phys.plasma_volume_ml > 0.0
    phys.validate()


def test_age_scaling():
    """Cardiac output declines ~1%/yr past age 40 (Guyton & Hall)."""
    young = _calc(age_years=30).compute_physiology()
    old = _calc(age_years=80).compute_physiology()
    ratio = (
        old.cardiac_output_ml_per_min / young.cardiac_output_ml_per_min
    )
    assert ratio == pytest.approx((1.0 - 0.01) ** (80 - 40), rel=1e-3)
    assert old.cardiac_output_ml_per_min < young.cardiac_output_ml_per_min


def test_obesity_scaling():
    """Excess body weight enlarges liver and adipose compartments."""
    lean = _calc(body_weight_kg=70.0).compute_physiology()
    obese = _calc(body_weight_kg=110.0).compute_physiology()
    assert obese.get_organ("liver").volume_ml > lean.get_organ("liver").volume_ml
    assert obese.get_organ("adipose").volume_ml > lean.get_organ("adipose").volume_ml


def test_female_scaling():
    """Female sex scales cardiac output to ~85% of the male value."""
    male = _calc(sex="male").compute_physiology()
    female = _calc(sex="female").compute_physiology()
    ratio = female.cardiac_output_ml_per_min / male.cardiac_output_ml_per_min
    assert ratio == pytest.approx(0.85, rel=1e-3)


def test_smoking_effect():
    """Current smoking induces CYP1A2 and mildly induces CYP2E1."""
    never = _calc().compute_cyp_activity()
    smoker = _calc(smoking_status="current", pack_years=20.0).compute_cyp_activity()
    assert smoker["CYP1A2"] == pytest.approx(2.0 + 20.0 / 40.0)
    assert smoker["CYP1A2"] > never["CYP1A2"]
    assert smoker["CYP2E1"] == pytest.approx(1.5)
    # non-inducible isoforms stay untouched by tobacco
    assert smoker["CYP2D6"] == pytest.approx(never["CYP2D6"])


def test_pregnancy_effect():
    """Term pregnancy raises cardiac output and GFR by ~50%."""
    baseline = _calc(sex="female", gestational_weeks=40.0)
    pregnant = _calc(sex="female", pregnant=True, gestational_weeks=40.0)

    base_phys = baseline.compute_physiology()
    preg_phys = pregnant.compute_physiology()
    co_ratio = (
        preg_phys.cardiac_output_ml_per_min / base_phys.cardiac_output_ml_per_min
    )
    gfr_ratio = (
        pregnant.estimate_gfr_ml_per_min() / baseline.estimate_gfr_ml_per_min()
    )
    assert co_ratio == pytest.approx(1.50, rel=1e-3)
    assert gfr_ratio == pytest.approx(1.50, rel=1e-3)
    assert preg_phys.cardiac_output_ml_per_min > base_phys.cardiac_output_ml_per_min


def test_compute_cyp_activity():
    """The CYP panel covers every DEFAULT_CYP450_ACTIVITY isoform."""
    activity = _calc().compute_cyp_activity()
    assert set(activity) == set(DEFAULT_CYP450_ACTIVITY)
    for enzyme in DEFAULT_CYP450_ACTIVITY:
        assert activity[enzyme] >= 0.05
    assert activity["CYP3A4"] == pytest.approx(1.0)
    assert activity["CYP2D6"] == pytest.approx(1.0)


def test_default_traits():
    """create_default_traits() yields fresh reference instances."""
    first = create_default_traits()
    second = create_default_traits()
    assert isinstance(first, ExternalTraits)
    assert first is not second
    assert first.body_weight_kg == pytest.approx(70.0)
    assert first.height_cm == pytest.approx(170.0)
    assert first.age_years == pytest.approx(30.0)
    assert first.sex == "male"
