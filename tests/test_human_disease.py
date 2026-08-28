"""Tests for helixlang.plugins.human.disease (doc/27 Stage B pathology layer).

Covers gene/metabolite perturbation data structures, the non-mutating
apply_disease_state() constraint propagation, and the literature-anchored
DISEASE_PROFILES registry (Gaucher, PKU, Warburg, ...).
"""
from __future__ import annotations

import pytest

from helixlang.plugins.human.disease import (
    DISEASE_PROFILES,
    DiseaseState,
    GenePerturbation,
    MetabolitePerturbation,
    apply_disease_state,
)
from helixlang.plugins.runtime.metabolism import MetabolicModel, Reaction

VALID_CATEGORIES = {
    "enzyme_deficiency",
    "transporter_defect",
    "metabolic_overload",
    "receptor_dysfunction",
    "cancer_metabolism",
    "cardiovascular",
    "respiratory",
    "neurological",
    "metabolic",
    "infectious",
    "hematological",
    "autoimmune",
    "endocrine",
    "gastrointestinal",
    "immune",
}


def _toy_model() -> MetabolicModel:
    """Minimal model: one GPR-gated and one ungated PAH reaction."""
    model = MetabolicModel()
    model.add_reaction(Reaction(
        id="PAHG", name="Phe hydroxylase (GPR-gated)",
        stoichiometry={"phe": -1.0, "tyr": 1.0},
        lower_bound=0.0, upper_bound=100.0,
        subsystem="amino_acids", gene_reaction_rule="PAH",
    ))
    model.add_reaction(Reaction(
        id="PAHU", name="Phe hydroxylase (ungated)",
        stoichiometry={"phe_x": -1.0, "tyr_x": 1.0},
        lower_bound=0.0, upper_bound=100.0,
        subsystem="amino_acids",
    ))
    return model


def test_gene_perturbation_knockout():
    """A knockout carries zero residual activity and closes its reactions."""
    gp = GenePerturbation("PAH", "knockout", 0.0)
    assert gp.activity_fraction == pytest.approx(0.0)
    disease = DiseaseState(
        name="test", category="enzyme_deficiency", gene_perturbations=[gp],
    )
    out = apply_disease_state(_toy_model(), disease)
    assert out.reactions["PAHG"].upper_bound == pytest.approx(0.0)


def test_gene_perturbation_downregulate():
    """Downregulation scales bounds by residual activity on ungated targets.

    The GPR-compensation rule skips reactions whose intact gene rule
    implies an active isozyme, so the effect must target the ungated
    twin explicitly via ``affected_reactions``.
    """
    gp = GenePerturbation(
        "PAH", "downregulate", 0.5, affected_reactions=["PAHU"],
    )
    assert gp.activity_fraction == pytest.approx(0.5)
    disease = DiseaseState(
        name="partial LOF", gene_perturbations=[gp], severity=1.0,
    )
    out = apply_disease_state(_toy_model(), disease)
    assert out.reactions["PAHU"].upper_bound == pytest.approx(50.0)


def test_metabolite_perturbation_accumulate():
    """Metabolite perturbations record accumulation type and pools."""
    mp = MetabolitePerturbation(
        "phenylalanine", "accumulate", initial_concentration_mm=2.4,
        normal_concentration_mm=0.09,
    )
    assert mp.perturbation_type == "accumulate"
    disease = DiseaseState(
        name="PKU-lite", metabolite_perturbations=[mp], severity=1.0,
    )
    out = apply_disease_state(_toy_model(), disease)
    assert out.metabolite_pool_initials["phenylalanine"] == pytest.approx(2.4)


def test_disease_state_creation():
    """DiseaseState stores identity fields and validates severity."""
    disease = DiseaseState(
        name="Fabry disease", category="enzyme_deficiency", severity=0.8,
        onset_age_years=10.0,
    )
    assert disease.name == "Fabry disease"
    assert disease.category == "enzyme_deficiency"
    with pytest.raises(ValueError):
        DiseaseState(name="bad", severity=1.5)


def test_apply_disease_state():
    """apply_disease_state never mutates its input model."""
    original = _toy_model()
    disease = DISEASE_PROFILES["GAUCHER"]
    diseased = apply_disease_state(original, disease)
    assert diseased is not original
    assert original.reactions["PAHG"].upper_bound == pytest.approx(100.0)
    assert isinstance(diseased, MetabolicModel)


def test_gaucher_profile():
    """The Gaucher type 1 profile encodes GBA1 knockout."""
    gaucher = DISEASE_PROFILES["GAUCHER"]
    assert gaucher.name == "Gaucher disease type 1"
    assert gaucher.category == "enzyme_deficiency"
    genes = [gp.gene_id for gp in gaucher.gene_perturbations]
    assert "GBA1" in genes


def test_pku_profile():
    """The PKU profile encodes PAH downregulation with Phe accumulation."""
    pku = next(d for d in DISEASE_PROFILES.values() if "Phenylketonuria" in d.name)
    genes = [gp.gene_id for gp in pku.gene_perturbations]
    mets = [mp.metabolite_id for mp in pku.metabolite_perturbations]
    assert "PAH" in genes
    assert "phenylalanine" in mets


def test_warburg_profile():
    """The Warburg cancer profile overexpresses HK2/PDK1."""
    warburg = next(d for d in DISEASE_PROFILES.values() if "Warburg" in d.name)
    types = {gp.perturbation_type for gp in warburg.gene_perturbations}
    assert types == {"overexpress"}
    assert warburg.category == "cancer_metabolism"


def test_all_disease_profiles_valid():
    """Every registered profile carries a name and a known category."""
    assert DISEASE_PROFILES
    for key, disease in DISEASE_PROFILES.items():
        assert disease.name, f"{key} missing name"
        assert disease.category in VALID_CATEGORIES, f"{key}: bad category"
