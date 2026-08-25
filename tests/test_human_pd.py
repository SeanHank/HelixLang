"""Tests for helixlang.human.pharmacodynamics (doc/27 PD layer).

Covers the Hill-equation core, inhibition/activation fractions,
multi-target effect computation, and the six PREDEFINED_PD profiles.
"""
from __future__ import annotations

import pytest

from helixlang.human.pharmacodynamics import (
    PREDEFINED_PD,
    PDEffect,
    Pharmacodynamics,
    activation_fraction,
    compute_pd_effects,
    get_predefined_pd,
    hill_equation,
    inhibition_fraction,
)


def test_hill_equation_zero():
    """Zero concentration returns the baseline effect E0."""
    assert hill_equation(0.0, ec50=2.0, emax=1.0, e0=0.2) == pytest.approx(0.2)
    assert hill_equation(-1.0, ec50=2.0, e0=0.3) == pytest.approx(0.3)


def test_hill_equation_ec50():
    """At C = EC50 the response is exactly halfway to Emax."""
    value = hill_equation(4.0, ec50=4.0, emax=1.0, n=1.0, e0=0.0)
    assert value == pytest.approx(0.5)


def test_hill_equation_saturation():
    """High concentrations saturate at Emax."""
    saturated = hill_equation(1e9, ec50=1.0, emax=1.0, e0=0.1)
    assert saturated == pytest.approx(1.0, abs=1e-6)
    # steeper Hill coefficient reaches saturation sooner
    mid = hill_equation(2.0, ec50=1.0, emax=1.0, n=8.0)
    assert mid > 0.99


def test_inhibition_fraction_zero():
    """No drug means no inhibition: full activity remains."""
    assert inhibition_fraction(0.0, ic50=5.0) == pytest.approx(1.0)


def test_inhibition_fraction_full():
    """Very high concentrations drive residual activity toward zero."""
    assert inhibition_fraction(1e9, ic50=5.0) == pytest.approx(0.0, abs=1e-6)
    assert inhibition_fraction(5.0, ic50=5.0) == pytest.approx(0.5)


def test_activation_fraction():
    """Activator fraction rises monotonically toward 1.0."""
    low = activation_fraction(0.5, ec50=2.0)
    high = activation_fraction(20.0, ec50=2.0)
    assert 0.0 <= low < high <= 1.0
    baseline = activation_fraction(1e6, ec50=2.0, baseline=0.05)
    assert baseline == pytest.approx(1.0, abs=1e-3)


def test_compute_pd_effects():
    """compute_pd_effects maps each target reaction to a multiplier."""
    pd_model = Pharmacodynamics(
        drug_name="combo",
        effects=[
            PDEffect(target_reaction="COX1", effect_type="inhibition",
                     ec50_um=5.0),
            PDEffect(target_reaction="GBA", effect_type="activation",
                     ec50_um=0.5, baseline_effect=0.05),
        ],
    )
    effects = compute_pd_effects(concentration_um=5.0, pd=pd_model)
    assert set(effects) == {"COX1", "GBA"}
    assert effects["COX1"] == pytest.approx(0.5)
    assert 0.05 < effects["GBA"] < 1.0


def test_predefined_pd_all():
    """All six doc/27 section 8 profiles are registered with effects."""
    expected = {
        "imiglucerase_gaucher", "ibuprofen_cox", "metformin_complex1",
        "cisplatin_dna", "tamoxifen_esr1", "imatinib_bcr_abl",
    }
    assert set(PREDEFINED_PD) == expected
    for key, pd in PREDEFINED_PD.items():
        assert pd.effects, key
    lookup = get_predefined_pd("Ibuprofen_COX")
    assert lookup is PREDEFINED_PD["ibuprofen_cox"]
    assert get_predefined_pd("unknown-drug") is None
