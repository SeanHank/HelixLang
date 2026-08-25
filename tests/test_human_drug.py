"""Tests for helixlang.human.drug (doc/27 drug specification layer).

Covers DrugMolecule/Drug data structures, SMILES parsing with RDKit
graceful degradation, regimen validation, and the six literature-anchored
PREDEFINED_DRUGS profiles.
"""
from __future__ import annotations

import pytest

from helixlang.human.drug import (
    BIOLOGIC,
    ORAL,
    PREDEFINED_DRUGS,
    VALID_ROUTES,
    Drug,
    DrugMolecule,
    get_predefined_drug,
    list_predefined_drugs,
    parse_drug_smiles,
)

EXPECTED_DRUGS = {
    "IMIGLUCERASE", "IBUPROFEN", "METFORMIN",
    "CISPLATIN", "TAMOXIFEN", "IMATINIB",
}

IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def test_drug_creation():
    """Drug pairs a molecule with a complete dosing regimen."""
    molecule = DrugMolecule(name="test drug")
    drug = Drug(
        molecule=molecule, dose_mg=400.0, route=ORAL,
        half_life_h=2.0, dosing_interval_h=8.0,
    )
    assert drug.molecule is molecule
    assert drug.dose_mg == pytest.approx(400.0)
    assert drug.route == ORAL
    assert drug.elimination_rate_constant() > 0.0


def test_drug_molecule_creation():
    """DrugMolecule records chemical identity and target metadata."""
    molecule = DrugMolecule(
        name="imiglucerase", drug_type=BIOLOGIC,
        molecular_weight_da=60000.0, target_protein="GBA1",
    )
    assert molecule.name == "imiglucerase"
    assert molecule.drug_type == BIOLOGIC
    assert molecule.molecular_weight_da == pytest.approx(60000.0)
    assert molecule.target_protein == "GBA1"


def test_parse_drug_smiles_ibuprofen():
    """parse_drug_smiles returns a molecule carrying the input structure."""
    molecule = parse_drug_smiles(IBUPROFEN_SMILES, name="ibuprofen")
    assert isinstance(molecule, DrugMolecule)
    assert molecule.smiles == IBUPROFEN_SMILES
    assert molecule.name == "ibuprofen"
    assert molecule.drug_type == "small_molecule"


def test_predefined_drugs_all():
    """All six doc/27 section 6.5 profiles are registered."""
    assert set(PREDEFINED_DRUGS) == EXPECTED_DRUGS
    assert set(list_predefined_drugs()) == EXPECTED_DRUGS


def test_predefined_drug_properties():
    """Every profile has a positive dose, positive half-life, valid route."""
    for key, drug in PREDEFINED_DRUGS.items():
        assert drug.dose_mg > 0.0, key
        assert drug.half_life_h > 0.0, key
        assert drug.route in VALID_ROUTES, key
        assert drug.molecule.name, key
        assert not drug.validate(), f"{key}: {drug.validate()}"


def test_drug_validation():
    """validate() flags bad routes and out-of-range fractions."""
    good = get_predefined_drug("ibuprofen")
    assert good.validate() == []

    bad_route = Drug(molecule=DrugMolecule(name="x"), route="rectal")
    problems = bad_route.validate()
    assert any("route" in p for p in problems)

    bad_bioav = Drug(
        molecule=DrugMolecule(name="x"), bioavailability=1.5,
    )
    assert any("bioavailability" in p for p in bad_bioav.validate())

    bad_half_life = Drug(molecule=DrugMolecule(name="x"), half_life_h=0.0)
    assert any("half_life_h" in p for p in bad_half_life.validate())


def test_get_predefined_drug():
    """get_predefined_drug resolves case-insensitively and copies deeply."""
    drug = get_predefined_drug("Ibuprofen")
    assert drug is not None
    assert drug is not PREDEFINED_DRUGS["IBUPROFEN"]
    assert drug.dose_mg == PREDEFINED_DRUGS["IBUPROFEN"].dose_mg
    assert get_predefined_drug("does-not-exist") is None
