"""Coverage-closure tests for helixlang.plugins.kinetics.

Covers the plugin-contract paths, BRENDA/ML/heuristic strategy branches,
protocol ellipsis bodies, and ESM-2 fallback/error paths not exercised by
the behavioural test suite.
"""
from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

import helixlang.api.capabilities as api_cap
import helixlang.plugins.kinetics as kinetics_mod
from helixlang.core.errors import PluginDependencyError
from helixlang.plugins.kinetics import (
    KcatPredictor,
    KmEstimator,
    _check,
    _load,
    _make_backend,
    estimate_km,
    predict_kcat,
)
from helixlang.plugins.kinetics import sequence_predictor as sp
from helixlang.plugins.kinetics.kcat_predictor import BRENDAEntry, KcatModel
from helixlang.plugins.kinetics.km_estimator import KmModel
from helixlang.plugins.kinetics.sequence_predictor import (
    SequenceKcatPredictor,
    SequenceKmEstimator,
    _aa_composition_features,
    _binding_site_score,
    _estimate_kcat_from_sequence,
    get_esm2_embedding,
    is_esm2_available,
)

# A sequence long enough to reach the sequence-heuristic branch in kcat/km.
_SEQ_X = "MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE" * 5


class TestPluginContract:
    def test_check_importable_package(self):
        assert _check("helixlang") is True

    def test_check_missing_package(self):
        assert _check("helixlang_nonexistent_module_xyz") is False

    def test_load_returns_backend_factory(self):
        backend_factory = _load()
        assert backend_factory is _make_backend
        backend = backend_factory()
        assert backend is KcatPredictor

    def test_load_raises_when_dependency_missing(self, monkeypatch):
        monkeypatch.setattr(kinetics_mod, "_check", lambda pkg: False)
        with pytest.raises(PluginDependencyError):
            _load()

    def test_load_raises_message(self, monkeypatch):
        monkeypatch.setattr(kinetics_mod, "_check", lambda pkg: False)
        with pytest.raises(PluginDependencyError) as exc:
            _load()
        assert exc.value.name == "kinetics"
        assert exc.value.dep == "numpy"
        assert exc.value.extra == "ml"


class TestKcatPredictorClose:
    def test_protocol_body(self):
        assert KcatModel.predict(None, "AA", "glucose") is None

    def test_brenda_organism_match(self):
        pred = KcatPredictor(
            target_organism="Escherichia coli",
            brenda_entries=[
                BRENDAEntry(
                    ec_number="1.1.1.1",
                    substrate="ethanol",
                    organism="Escherichia coli K12",
                    kcat_value=10.0,
                    km_value=2.0,
                ),
                BRENDAEntry(
                    ec_number="1.1.1.1",
                    substrate="ethanol",
                    organism="Bacillus subtilis",
                    kcat_value=50.0,
                    km_value=0.5,
                ),
            ],
        )
        result = pred.predict("ADH", ec_number="1.1.1.1")
        assert result.source == "brenda"
        assert result.kcat_value == 10.0
        assert result.organism == "Escherichia coli"

    def test_brenda_organism_no_match(self):
        pred = KcatPredictor(
            target_organism="Saccharomyces cerevisiae",
            brenda_entries=[
                BRENDAEntry(
                    ec_number="1.1.1.1",
                    substrate="ethanol",
                    organism="Escherichia coli K12",
                    kcat_value=10.0,
                    km_value=2.0,
                ),
            ],
        )
        result = pred.predict("ADH", ec_number="1.1.1.1")
        assert result.source == "brenda"
        assert result.kcat_value == 10.0

    def test_brenda_no_entries_none(self):
        pred = KcatPredictor(target_organism="Escherichia coli")
        assert pred._lookup_brenda("1.1.1.1") is None

    def test_median_with_complexity_scaling(self):
        pred = KcatPredictor(target_organism="Escherichia coli")
        result = pred.predict("PFK", ec_number="2.7.1.11")
        assert result.source == "organism_scaled"
        assert result.kcat_value == 214.0 * 1.1

    def test_ml_prediction_path(self):
        ml = SimpleNamespace(predict=lambda seq, sub: 100.0)
        pred = KcatPredictor(
            target_organism="Escherichia coli",
            ml_model=ml,
        )
        result = pred.predict("REAC-1", ec_number="9.9.9.9", sequence=_SEQ_X)
        assert result.source == "ml"
        assert result.kcat_value == 100.0

    def test_partial_organism_key_match(self):
        pred = KcatPredictor(target_organism="e_coli_hypothetical")
        scaled = pred._apply_organism_scale(2.0)
        assert scaled == 2.0

    def test_unknown_organism_default_scale(self):
        pred = KcatPredictor(target_organism="Fake org Xyz")
        scaled = pred._apply_organism_scale(4.0)
        assert scaled == 4.0

    def test_known_organism_exact_match(self):
        pred = KcatPredictor(target_organism="b_subtilis")
        scaled = pred._apply_organism_scale(10.0)
        assert scaled == 11.0

    def test_fallback(self):
        pred = KcatPredictor(target_organism="Escherichia coli", ml_model=None)
        result = pred.predict("R", ec_number="9.99.99.99")
        assert result.source == "fallback"
        assert result.kcat_value == 22.0

    def test_predict_without_ec_number(self):
        pred = KcatPredictor(target_organism="Escherichia coli", ml_model=None)
        result = pred.predict("R", ec_number="")
        assert result.source == "fallback"
        assert result.kcat_value == 22.0

    def test_brenda_blank_organism(self):
        pred = KcatPredictor(
            target_organism="",
            brenda_entries=[
                BRENDAEntry(
                    ec_number="1.1.1.1",
                    substrate="ethanol",
                    organism="Escherichia coli K12",
                    kcat_value=10.0,
                    km_value=2.0,
                ),
            ],
        )
        result = pred.predict("ADH", ec_number="1.1.1.1")
        assert result.source == "brenda"
        assert result.kcat_value == 10.0

    def test_predict_kcat_convenience(self):
        result = predict_kcat("PFK", ec_number="2.7.1.11", target_organism="b_subtilis")
        assert result.kcat_value == 214.0 * 1.1 * 1.1


class TestKmEstimatorClose:
    def test_protocol_body(self):
        assert KmModel.predict(None, "AA", "glucose") is None

    def test_known_substrate_known_organism(self):
        assert KmEstimator(target_organism="Escherichia coli").estimate(
            "GLK", substrate="glucose"
        ) == 0.1

    def test_known_substrate_uncharacterized_organism(self):
        km = KmEstimator(target_organism="Thermus thermophilus").estimate(
            "GLK", substrate="glucose"
        )
        assert km == 0.1 * 1.5

    def test_sequence_heuristic(self):
        km = KmEstimator(target_organism="Escherichia coli").estimate(
            "R1", substrate="", sequence="ACDEFGRKHILM"
        )
        assert 0.01 <= km <= 10.0

    def test_sequence_heuristic_molecular_weight(self):
        km = KmEstimator(target_organism="Escherichia coli").estimate(
            "R1",
            substrate="",
            sequence="ACDEFGRKHILM" * 10,
            molecular_weight=40000.0,
        )
        assert 0.01 <= km <= 10.0

    def test_ml_prediction(self):
        ml = SimpleNamespace(predict=lambda seq, sub: 0.25)
        km = KmEstimator(target_organism="Escherichia coli", ml_model=ml).estimate(
            "R1", substrate="mystery", sequence="ACDEFG"
        )
        assert km == 0.25

    def test_global_median_fallback(self):
        km = KmEstimator(target_organism="Escherichia coli").estimate(
            "R1", substrate="mystery", sequence=""
        )
        assert km == 0.5

    def test_heuristic_empty_sequence(self):
        est = KmEstimator()
        assert est._heuristic_km("", 0) == 0.5

    def test_estimate_km_convenience(self):
        assert estimate_km("GLK", substrate="glucose") == 0.1


class TestSequencePredictorClose:
    def test_ensure_esm2_import_error(self, monkeypatch):
        monkeypatch.setattr(sp, "_esm2_model", None)
        monkeypatch.setattr(sp, "_ESM2_AVAILABLE", False)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "transformers" or name.startswith("transformers."):
                raise ImportError("transformers unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert is_esm2_available() is False

    def test_embedding_raises_without_opt_in(self, monkeypatch):
        monkeypatch.setattr(sp, "_ensure_esm2", lambda: None)
        monkeypatch.setattr(sp, "_ESM2_AVAILABLE", False)
        with pytest.raises(PluginDependencyError):
            get_esm2_embedding("MKWVTFISLLFLFSSAYS")

    def test_embedding_returns_features_when_opted_in(self, monkeypatch):
        monkeypatch.setattr(sp, "_ensure_esm2", lambda: None)
        monkeypatch.setattr(sp, "_ESM2_AVAILABLE", False)
        monkeypatch.setattr(api_cap, "opt_in", lambda *a, **k: True)
        emb = get_esm2_embedding("MKWVTFISLLFLFSSAYS")
        assert len(emb) == 20
        assert any(x > 0.0 for x in emb)

    def test_aa_features_empty(self):
        assert _aa_composition_features("") == [0.0] * 20

    def test_binding_site_positive_substrate(self):
        score = _binding_site_score("RKHHDE", "ammonium")
        assert 0.0 <= score <= 1.0

    def test_heuristic_without_esm2(self):
        kcat, source, conf = _estimate_kcat_from_sequence(
            "A" * 300, "99.99.99", use_esm2=False
        )
        assert source == "sequence_heuristic"
        assert kcat > 0.0
        assert 0.0 <= conf <= 0.6

    def test_predictor_and_estimator_roundtrip(self):
        kp = SequenceKcatPredictor().predict(
            reaction_id="R9", sequence="ACDEFGHIKLMN" * 20, ec_number="7.7.7.7"
        )
        assert kp.kcat_value > 0.0
        km = SequenceKmEstimator().predict(sequence="ACDEFGHIKLMN" * 20, substrate="glucose")
        assert km.km_value > 0.0
