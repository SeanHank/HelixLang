"""Branch-completion tests for helixlang.plugins.human.molecular_toxicity."""
from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from helixlang.plugins.human import molecular_toxicity as mt


@pytest.fixture
def no_rdkit(monkeypatch):
    real = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "rdkit" or name == "rdkit.Chem":
            raise ImportError(f"{name} blocked")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ANILINE = "Nc1ccccc1"


class TestRdkitFallbacks:
    def test_descriptors_import_blocked(self, no_rdkit):
        assert mt._compute_rdkit_descriptors("CCO") == {}

    def test_descriptors_invalid_smiles(self):
        assert mt._compute_rdkit_descriptors("zzzzz") == {}

    def test_structural_alerts_import_blocked(self, no_rdkit):
        alerts, weights = mt._match_structural_alerts(
            SimpleNamespace(smiles="ClC(Cl)(Cl)Cl"))
        assert "polyhalogenated" in alerts
        assert weights == {}

    def test_morgan_fingerprint_import_blocked(self, no_rdkit):
        assert mt._compute_morgan_fingerprint("CCO") == []

    def test_morgan_fingerprint_invalid_smiles(self):
        assert mt._compute_morgan_fingerprint("zzzzz") == []

    def test_morgan_fingerprint_real(self):
        fp = mt._compute_morgan_fingerprint("CCO", radius=2, n_bits=64)
        assert isinstance(fp, list)
        assert len(fp) == 64


class TestMatchAlertsFallback:
    def test_none_mol(self):
        assert mt._match_alerts_fallback(None) == []

    def test_aromatic_amine(self):
        alerts = mt._match_alerts_fallback(SimpleNamespace(smiles="xxnc1yy"))
        assert "aromatic_amine" in alerts

    def test_michael_acceptor(self):
        alerts = mt._match_alerts_fallback(SimpleNamespace(smiles="c=cc=o"))
        assert "michael_acceptor" in alerts

    def test_epoxide(self):
        alerts = mt._match_alerts_fallback(SimpleNamespace(smiles="c1oc1"))
        assert "epoxide" in alerts

    def test_nitrosamine(self):
        alerts = mt._match_alerts_fallback(SimpleNamespace(smiles="n(=o)n"))
        assert "nitrosamine" in alerts

    def test_thiophene(self):
        alerts = mt._match_alerts_fallback(SimpleNamespace(smiles="c1ccsc1"))
        assert "thiophene" in alerts

    def test_aniline_derivative(self):
        alerts = mt._match_alerts_fallback(SimpleNamespace(smiles="nc1ccccc1"))
        assert "aniline_derivative" in alerts

    def test_acyl_halide(self):
        alerts = mt._match_alerts_fallback(SimpleNamespace(smiles="c(=o)br"))
        assert "acyl_halide" in alerts

    def test_polyhalogenated(self):
        alerts = mt._match_alerts_fallback(SimpleNamespace(smiles="cc(=o)clcl"))
        assert "polyhalogenated" in alerts

    def test_plain_string_mol(self):
        alerts = mt._match_alerts_fallback("CCO")
        assert alerts == []


class TestStructuralAlertsReal:
    def test_aniline_match_accumulates_weights(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(ANILINE)
        alerts, weights = mt._match_structural_alerts(mol)
        assert "Aniline" in alerts
        assert weights["hepatotoxicity"] == pytest.approx(0.25)

    def test_fake_mol_raises_and_skips(self):
        mol = SimpleNamespace()
        alerts, weights = mt._match_structural_alerts(mol)
        assert alerts == []
        assert weights["hepatotoxicity"] == 0.0


class TestDescriptorScore:
    def test_empty_weights(self):
        assert mt._descriptor_score({}, {}) == 0.0

    def test_positive_weight_over_threshold(self):
        desc = {"LogP": 5.0}
        assert mt._descriptor_score(desc, {"LogP": (3.0, 0.1)}) == pytest.approx(0.1 * 2 / 3)

    def test_positive_weight_below_threshold(self):
        desc = {"LogP": 1.0}
        assert mt._descriptor_score(desc, {"LogP": (3.0, 0.1)}) == 0.0

    def test_negative_weight_below_threshold(self):
        desc = {"LogP": 1.0}
        assert mt._descriptor_score(
            desc, {"LogP": (3.0, -0.1)}) == pytest.approx(0.1 * 2 / 3)

    def test_negative_weight_above_threshold(self):
        desc = {"LogP": 3.0}
        assert mt._descriptor_score(desc, {"LogP": (3.0, -0.1)}) == 0.0

    def test_missing_key_uses_zero(self):
        assert mt._descriptor_score({}, {"LogP": (3.0, 0.1)}) == 0.0

    def test_clamped_to_one(self):
        desc = {"LogP": 6.0, "TPSA": 200.0}
        weights = {"LogP": (3.0, 0.5), "TPSA": (100.0, 0.5)}
        assert mt._descriptor_score(desc, weights) == 1.0


class TestPredictToxicity:
    def test_known_drug(self):
        profile = mt.MolecularToxicityPredictor().predict_toxicity(ASPIRIN)
        assert profile.confidence == 1.0
        assert profile.alerts_triggered == ()

    def test_rdkit_blocked(self, no_rdkit):
        profile = mt.MolecularToxicityPredictor().predict_toxicity("CCO")
        assert profile.confidence == 0.0

    def test_invalid_smiles(self):
        profile = mt.MolecularToxicityPredictor().predict_toxicity("zzzzz")
        assert profile.confidence == 0.0

    def test_empty_descriptors(self, monkeypatch):
        monkeypatch.setattr(mt, "_compute_rdkit_descriptors", lambda s: {})
        profile = mt.MolecularToxicityPredictor().predict_toxicity(ANILINE)
        assert profile.confidence == 0.0
        assert profile.alerts_triggered

    def test_full_ensemble(self):
        profile = mt.MolecularToxicityPredictor().predict_toxicity(ANILINE)
        assert 0.0 < profile.confidence <= 1.0
        assert profile.hepatotoxicity_score > 0.0
        assert profile.structural_alert_score == pytest.approx(0.25)


class TestPredictActivity:
    def test_empty_descriptors(self, monkeypatch):
        monkeypatch.setattr(mt, "_compute_rdkit_descriptors", lambda s: {})
        profile = mt.MolecularToxicityPredictor().predict_activity("CCO")
        assert profile.bioavailability == 0.5

    def test_full_profile(self):
        profile = mt.MolecularToxicityPredictor().predict_activity("CCO")
        assert 0.05 <= profile.bioavailability <= 0.95
        assert 0.0 < profile.half_life_hours <= 48.0
        assert 5.0 <= profile.volume_of_distribution <= 200.0


class TestActivityHelpers:
    def test_bioavailability_defaults(self):
        assert mt._bioavailability_from_descriptors({}) == pytest.approx(0.95)

    def test_bioavailability_penalties(self):
        desc = {"LogP": 6.0, "MolWt": 600.0, "NumHDonors": 6.0,
                "NumHAcceptors": 12.0, "TPSA": 200.0}
        assert mt._bioavailability_from_descriptors(desc) == pytest.approx(0.15)

    def test_half_life_defaults(self):
        assert mt._half_life_from_descriptors({}) == pytest.approx(4.0)

    def test_half_life_high_values(self):
        desc = {"LogP": 8.0, "MolWt": 600.0}
        assert mt._half_life_from_descriptors(desc) == pytest.approx(12.5)

    def test_half_life_clamped(self):
        desc = {"LogP": 8.0, "MolWt": 20000.0}
        assert mt._half_life_from_descriptors(desc) == pytest.approx(48.0)

    def test_vd_defaults(self):
        assert mt._vd_from_descriptors({}) == pytest.approx(50.0)

    def test_vd_high_mw(self):
        desc = {"LogP": 0.0, "MolWt": 600.0}
        assert mt._vd_from_descriptors(desc) == pytest.approx(15.0)

    def test_vd_clamped(self):
        desc = {"LogP": 20.0}
        assert mt._vd_from_descriptors(desc) == pytest.approx(200.0)

    def test_vd_low(self):
        desc = {"LogP": 0.0}
        assert mt._vd_from_descriptors(desc) == pytest.approx(30.0)


class TestSmilesAutofill:
    def test_toxicity_profile(self):
        assert mt.smiles_autofill().toxicity_profile("CCO").confidence > 0.0

    def test_activity_profile(self):
        profile = mt.smiles_autofill().activity_profile("CCO")
        assert profile.bioavailability > 0.0

    def test_auto_fill_drug_params(self):
        params = mt.smiles_autofill().auto_fill_drug_params("CCO")
        assert set(params) == {
            "bioavailability", "protein_binding", "half_life_hours",
            "volume_of_distribution", "hepatotoxicity_score",
            "nephrotoxicity_score", "cardiotoxicity_score",
            "myelosuppression_score",
        }
