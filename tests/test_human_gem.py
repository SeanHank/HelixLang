"""Tests for helixlang.plugins.human.gem_human (doc/27 human GEM loading).

Covers the E. coli-core fallback loader, tissue-specific exchange
overlays derived from TISSUE_PROFILES, and the load() tuple contract
used by the doc/27 simulation engine.
"""
from __future__ import annotations

from helixlang.plugins.human.gem_human import HumanGEMConfig, HumanGEMLoader
from helixlang.plugins.runtime.metabolism import MetabolicModel


def _loader() -> HumanGEMLoader:
    return HumanGEMLoader(HumanGEMConfig(tissue="liver"))


def test_load_core_model():
    """load_core_model() returns a MetabolicModel proxy."""
    model = _loader().load_core_model()
    assert isinstance(model, MetabolicModel)
    assert model.reactions


def test_apply_tissue_overlay():
    """apply_tissue_overlay rescales exchange bounds per TISSUE_PROFILES."""
    loader = _loader()
    model = loader.load_core_model()
    before = model.reactions["EX_glc"].upper_bound
    overlaid = loader.apply_tissue_overlay(model, "liver")
    assert overlaid is model  # modified in place
    glucose_ub = overlaid.reactions["EX_glc"].upper_bound
    assert glucose_ub != before
    # liver profile: 1.5 mmol/kg/min * 10 scaling factor -> +/-15
    assert glucose_ub == 15.0
    assert overlaid.reactions["EX_glc"].lower_bound == -15.0


def test_get_exchange_reactions():
    """get_exchange_reactions() maps reaction ids to names."""
    loader = _loader()
    exchanges = loader.get_exchange_reactions(loader.load_core_model())
    assert isinstance(exchanges, dict)
    assert "EX_glc" in exchanges
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in exchanges.items())
    assert loader.get_exchange_reactions(None) == {}


def test_get_tissue_profile():
    """get_tissue_profile() returns the TISSUE_PROFILES entry as a copy."""
    profile = _loader().get_tissue_profile("kidney")
    assert isinstance(profile, dict)
    for key in ("volume_ml", "blood_flow_ml_per_min", "key_reactions"):
        assert key in profile
    assert profile["volume_ml"] == 300.0
    assert _loader().get_tissue_profile("nonexistent") == {}


def test_load_returns_tuple():
    """load() bundles (model, tissue_profile) with the overlay applied."""
    model, profile = _loader().load("liver")
    assert isinstance(model, MetabolicModel)
    assert isinstance(profile, dict)
    assert profile["glucose_uptake_mmol_per_kg_per_min"] == 1.5
    assert model.reactions["EX_glc"].upper_bound == 15.0
