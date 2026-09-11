"""Branch-completion tests for helixlang.plugins.human.drug."""
from __future__ import annotations

import builtins
import importlib.util
import sys

import pytest

from helixlang.plugins.human.drug import (
    BIOLOGIC,
    IV,
    Drug,
    DrugMolecule,
    _formula_counts_from_mol,
    _smiles_to_adme_heuristic,
    biologics_adme,
    get_predefined_drug,
    parse_drug_smiles,
    smiles_to_adme,
)


def _load_blocking_imports(module_path: str, blocked: set[str]):
    spec = importlib.util.find_spec(module_path)
    throwaway = importlib.util.spec_from_file_location(
        "_no_" + module_path.rsplit(".", 1)[-1], spec.origin)
    old = sys.modules.get(throwaway.name)
    sys.modules[throwaway.name] = mod = importlib.util.module_from_spec(throwaway)
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in blocked:
            raise ImportError(f"{name} blocked")
        return real_import(name, *a, **k)

    builtins.__import__ = fake_import
    try:
        throwaway.loader.exec_module(mod)  # type: ignore[union-attr]
    finally:
        builtins.__import__ = real_import
        if old is None:
            sys.modules.pop(throwaway.name, None)
        else:
            sys.modules[throwaway.name] = old
    return mod


def _drug(**overrides) -> Drug:
    base = dict(
        molecule=DrugMolecule(name="test"),
        dose_mg=100.0,
        dosing_interval_h=24.0,
        route="oral",
    )
    base.update(overrides)
    return Drug(**base)


class TestDrugImportGuard:
    def test_rdkit_missing(self):
        mod = _load_blocking_imports(
            "helixlang.plugins.human.drug", {"rdkit", "rdkit.Chem"})
        assert mod._HAS_RDKIT is False
        mol = mod.parse_drug_smiles("C", name="x")
        assert mol.smiles == "C" and mol.molecular_weight_da == 0.0
        adme = mod.smiles_to_adme("CCC")
        assert adme["half_life_h"] > 0.0


class TestDrugValidate:
    def test_bad_renal_fraction(self):
        drug = get_predefined_drug("IBUPROFEN")
        drug.renal_fraction = 1.5
        assert any("renal_fraction" in p for p in drug.validate())

    def test_bad_hepatic_extraction(self):
        drug = get_predefined_drug("IBUPROFEN")
        drug.hepatic_extraction_ratio = -0.2
        assert any("hepatic_extraction_ratio" in p for p in drug.validate())

    def test_cyp_fractions_off(self):
        drug = get_predefined_drug("IBUPROFEN")
        drug.cyp_metabolism = {"2C9": 0.3, "3A4": 0.3}
        assert any("cyp_metabolism" in p for p in drug.validate())


class TestDrugKineticsMethods:
    def test_elimination_rate_zero_half_life(self):
        assert _drug(half_life_h=0.0).elimination_rate_constant() == 0.0

    def test_elimination_rate_positive(self):
        assert _drug(half_life_h=6.0).elimination_rate_constant() > 0.0

    def test_total_administered_zero_interval(self):
        assert _drug(dosing_interval_h=0.0).total_administered_mg() == 0.0

    def test_total_administered_positive(self):
        assert _drug(
            dose_mg=50.0, dosing_interval_h=12.0,
            duration_days=2.0).total_administered_mg() == 200.0

    def test_accumulation_zero_interval(self):
        assert _drug(dosing_interval_h=0.0).accumulation_ratio() == 1.0

    def test_accumulation_zero_k(self):
        assert _drug(
            half_life_h=0.0, dosing_interval_h=12.0).accumulation_ratio() == 1.0

    def test_accumulation_positive(self):
        assert _drug(
            half_life_h=6.0, dosing_interval_h=24.0).accumulation_ratio() > 1.0

    def test_steady_state_average_zero_cl(self):
        assert _drug(clearance_ml_per_min=0.0).steady_state_average_mg_per_l() == 0.0

    def test_steady_state_average_zero_tau(self):
        assert _drug(dosing_interval_h=0.0).steady_state_average_mg_per_l() == 0.0

    def test_steady_state_average_positive(self):
        value = _drug(
            bioavailability=1.0, dose_mg=100.0, clearance_ml_per_min=100.0,
            dosing_interval_h=24.0).steady_state_average_mg_per_l()
        assert value > 0.0

    def test_peak_oral(self):
        value = _drug(route="oral").steady_state_peak_mg_per_l()
        assert value > 0.0

    def test_peak_iv(self):
        value = _drug(route=IV).steady_state_peak_mg_per_l()
        assert value > 0.0

    def test_peak_ka_equals_k(self):
        import math
        drug = _drug(route=IV, half_life_h=6.0)
        k = math.log(2.0) / 6.0
        drug.absorption_rate_h = k
        assert drug.steady_state_peak_mg_per_l() > 0.0


class TestFormulaCounts:
    def test_exception_fallback(self, monkeypatch):
        class _Atom:
            def __init__(self, symbol) -> None:
                self.symbol = symbol

            def GetSymbol(self):
                return self.symbol

        class _Mol:
            def GetAtoms(self):
                return [_Atom("C"), _Atom("O")]

        def _boom(*a, **k):
            raise RuntimeError("no formula")

        monkeypatch.setattr(
            "rdkit.Chem.rdMolDescriptors.CalcMolFormula", _boom)
        counts = _formula_counts_from_mol(_Mol())
        assert counts == {"C": 1, "O": 1}


class TestParseAndAdme:
    def test_parse_empty(self):
        mol = parse_drug_smiles("", name="empty")
        assert mol.name == "empty" and mol.smiles == ""

    def test_parse_invalid_smiles(self):
        mol = parse_drug_smiles("zzzz", name="bad")
        assert mol.smiles == "zzzz" and mol.molecular_weight_da == 0.0

    def test_biologics_adme_small(self):
        adme = biologics_adme(3000.0)
        assert adme["half_life_h"] > 0.0

    def test_biologics_adme_medium(self):
        adme = biologics_adme(80000.0)
        assert adme["half_life_h"] > 0.0

    def test_biologics_adme_large(self):
        adme = biologics_adme(200000.0)
        assert adme["half_life_h"] == pytest.approx(504.0, rel=1e-6)

    def test_smiles_adme_biologics(self):
        adme = smiles_to_adme("", drug_type=BIOLOGIC, mw_da=30000.0)
        assert adme["half_life_h"] > 0.0

    def test_smiles_adme_biologics_zero_mw(self):
        adme = smiles_to_adme("", drug_type=BIOLOGIC)
        assert adme["bioavailability"] >= 0.05

    def test_smiles_adme_invalid(self):
        adme = smiles_to_adme("zzzz")
        assert adme["bioavailability"] >= 0.05

    def test_smiles_adme_lipinski_violations(self):
        adme = smiles_to_adme("C" * 50)
        assert adme["bioavailability"] > 0.0

    def test_smiles_adme_polyol(self):
        adme = smiles_to_adme(
            "OCC(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)CO")
        assert adme["half_life_h"] > 0.0

    def test_heuristic_violations(self):
        adme = _smiles_to_adme_heuristic("F" * 120 + "Cl" * 120)
        assert adme["bioavailability"] > 0.0

    def test_heuristic_clean(self):
        adme = _smiles_to_adme_heuristic("CCC")
        assert adme["half_life_h"] > 0.0

    def test_hill_formula_without_hydrogen(self):
        from helixlang.plugins.human.drug import _hill_formula
        assert _hill_formula({"C": 2, "O": 3}) == "C2O3"

    def test_hill_formula_with_hydrogen(self):
        from helixlang.plugins.human.drug import _hill_formula
        assert _hill_formula({"C": 1, "H": 4}) == "CH4"
