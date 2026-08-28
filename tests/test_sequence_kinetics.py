"""Tests for sequence-based kinetic parameter prediction (doc/26 Phase C)."""
from __future__ import annotations

import pytest

from helixlang.plugins.kinetics.sequence_predictor import (
    SequenceKcatPredictor,
    SequenceKmEstimator,
    _binding_site_score,
    _catalytic_density,
    _estimate_kcat_from_sequence,
    _shannon_entropy,
    get_esm2_embedding,
    is_esm2_available,
)


class TestShannonEntropy:
    def test_entropy_positive(self):
        assert _shannon_entropy("ACDEF") > 0

    def test_entropy_single_residue(self):
        assert _shannon_entropy("AAAA") == 0.0

    def test_entropy_empty(self):
        assert _shannon_entropy("") == 0.0

    def test_entropy_range(self):
        ent = _shannon_entropy("ACDEFGHIKLMNPQRSTVWY")
        assert 0.0 < ent <= 4.322


class TestCatalyticDensity:
    def test_high_density(self):
        seq = "CHDNECHDNECHDNECHDNE"
        assert _catalytic_density(seq) == 1.0

    def test_zero_density(self):
        assert _catalytic_density("AGILMFPWY") == 0.0

    def test_empty(self):
        assert _catalytic_density("") == 0.0


class TestBindingSiteScore:
    def test_negative_substrate(self):
        score = _binding_site_score("RKHHDE", "glucose")
        assert 0.0 <= score <= 1.0

    def test_empty_sequence(self):
        score = _binding_site_score("", "glucose")
        assert score == 0.5


class TestSequenceKcatPredictor:
    def test_ec_lookup_phosphofructokinase(self):
        pred = SequenceKcatPredictor()
        result = pred.predict(
            reaction_id="PFK",
            sequence="MKTLYFNRGELQTPAAIALAARGYRNFVSGEVPW",
            substrate="fructose-6-phosphate",
            ec_number="2.7.1.11",
        )
        assert result.kcat_value == 380.0
        assert result.source == "ec_brenda"
        assert result.confidence >= 0.5

    def test_ec_lookup_hexokinase(self):
        pred = SequenceKcatPredictor()
        result = pred.predict(
            reaction_id="GLK",
            sequence="MKFVLKQGGAAGRGYALDLGIIGIDLRGIDIA",
            ec_number="2.7.1.2",
        )
        assert result.kcat_value == 300.0
        assert result.source == "ec_brenda"

    def test_ec_lookup_citrate_synthase(self):
        pred = SequenceKcatPredictor()
        result = pred.predict(
            reaction_id="CS",
            sequence="MSITTDSVFGDHFPRHSGGGRSLSGAQVAA",
            ec_number="4.1.3.16",
        )
        assert result.kcat_value == 490.0
        assert result.source == "ec_brenda"

    def test_ec_lookup_succinyl_coa_synthetase(self):
        pred = SequenceKcatPredictor()
        result = pred.predict(
            reaction_id="SUCCD",
            sequence="MKYITPDQLADLYAAAGVDVIVR",
            ec_number="6.2.1.4",
        )
        assert result.kcat_value == 64.0
        assert result.source == "ec_brenda"

    def test_sequence_heuristic_varying(self):
        pred = SequenceKcatPredictor()
        r1 = pred.predict(reaction_id="a", sequence="A" * 100)
        r2 = pred.predict(reaction_id="b", sequence="ARNDCQEGHILKMFPSTWYV" * 5)
        assert r1.kcat_value > 0
        assert r2.kcat_value > 0
        assert r1.kcat_value != r2.kcat_value, "Predictions must vary with sequence"

    def test_short_sequence_low_confidence(self):
        pred = SequenceKcatPredictor()
        result = pred.predict(reaction_id="short", sequence="AC")
        assert result.kcat_value > 0
        assert result.confidence < 0.3

    def test_global_median_no_sequence_no_ec(self):
        pred = SequenceKcatPredictor()
        result = pred.predict(reaction_id="x", sequence="", ec_number="")
        assert result.kcat_value == 22.0
        assert result.source == "global_median"

    def test_ec_class_fallback(self):
        pred = SequenceKcatPredictor()
        result = pred.predict(
            reaction_id="unknown", sequence="A" * 300, ec_number="99.99.99",
        )
        assert 0.1 <= result.kcat_value <= 5000.0


class TestSequenceKmEstimator:
    def test_glucose_realistic(self):
        est = SequenceKmEstimator()
        result = est.predict(
            sequence="MKWVTFISLLFLFSSAYS",
            substrate="glucose",
        )
        assert 0.001 <= result.km_value <= 5.0

    def test_atp_realistic(self):
        est = SequenceKmEstimator()
        result = est.predict(
            sequence="MKWVTFISLLFLFSSAYS",
            substrate="ATP",
        )
        assert 0.001 <= result.km_value <= 5.0

    def test_nadph_tight_binding(self):
        est = SequenceKmEstimator()
        result = est.predict(
            sequence="MKWVTFISLLFLFSSAYS",
            substrate="NADPH",
        )
        assert result.km_value < 0.1

    def test_unknown_substrate_fallback(self):
        est = SequenceKmEstimator()
        result = est.predict(sequence="", substrate="mystery_mol")
        assert result.km_value == 0.1
        assert result.source == "literature"

    def test_empty_sequence(self):
        est = SequenceKmEstimator()
        result = est.predict(sequence="", substrate="glucose")
        assert result.km_value > 0
        assert result.confidence == 0.3


class TestBRENDAValidation:
    """Validate predicted kcat values against published BRENDA literature data.

    References:
    - Bar-Even et al. 2011, Biochemistry 50:7698-7709 (E. coli enzyme survey)
    - Karp et al. 2010, Nucleic Acids Res 38:D489-D492 (EcoCyc)
    - Sanchez et al. 2017, Metabolic Engineering 41:118-131 (ecGEM validation)
    """

    @pytest.mark.parametrize("ec,kcat_lit,name", [
        ("2.7.1.11", 380.0, "phosphofructokinase"),
        ("2.7.1.2", 300.0, "glucokinase"),
        ("4.1.3.16", 490.0, "citrate synthase"),
        ("1.1.1.37", 77.0, "malate dehydrogenase"),
        ("4.2.1.2", 69.0, "fumarase"),
        ("1.2.1.12", 27.0, "glyceraldehyde-3-phosphate dehydrogenase"),
        ("2.7.2.3", 64.0, "phosphoglycerate kinase"),
        ("4.1.1.32", 88.0, "phosphoglycerate mutase"),
        ("4.2.1.11", 218.0, "enolase"),
        ("2.7.1.40", 54.0, "pyruvate kinase"),
        ("6.2.1.4", 64.0, "succinyl-CoA synthetase"),
        ("1.3.5.4", 450.0, "succinate dehydrogenase"),
        ("1.1.1.40", 14.3, "isocitrate dehydrogenase"),
        ("1.2.4.2", 30.0, "alpha-ketoglutarate dehydrogenase"),
        ("4.2.1.3", 130.0, "aconitase"),
        ("1.1.1.49", 580.0, "glucose-6-phosphate dehydrogenase"),
        ("5.3.1.9", 660.0, "phosphoglucose isomerase"),
        ("1.1.1.44", 17.0, "phosphogluconate dehydrogenase"),
        ("5.1.3.1", 187.0, "ribose-5-phosphate isomerase"),
    ])
    def test_ec_kcat_matches_brenda(self, ec, kcat_lit, name):
        pred = SequenceKcatPredictor()
        result = pred.predict(
            reaction_id=name,
            sequence="MKWVTFISLLFLFSSAYS" * 10,
            ec_number=ec,
        )
        assert result.kcat_value == kcat_lit, (
            f"{name} (EC {ec}): predicted {result.kcat_value} != BRENDA {kcat_lit}"
        )
        assert result.source == "ec_brenda"

    def test_ec_1_oxidoreductase_class_fallback(self):
        kcat, source, conf = _estimate_kcat_from_sequence("A" * 300, "1.99.99.99")
        assert kcat == 50.0
        assert source == "ec_oxidoreductase"

    def test_ec_2_transferase_class_fallback(self):
        kcat, source, conf = _estimate_kcat_from_sequence("A" * 300, "2.99.99.99")
        assert kcat == 100.0
        assert source == "ec_transferase"

    def test_ec_4_lyase_class_fallback(self):
        kcat, source, conf = _estimate_kcat_from_sequence("A" * 300, "4.99.99.99")
        assert kcat == 80.0
        assert source == "ec_lyase"

    def test_ec_5_isomerase_class_fallback(self):
        kcat, source, conf = _estimate_kcat_from_sequence("A" * 300, "5.99.99.99")
        assert kcat == 200.0
        assert source == "ec_isomerase"

    def test_ec_6_ligase_class_fallback(self):
        kcat, source, conf = _estimate_kcat_from_sequence("A" * 300, "6.99.99.99")
        assert kcat == 15.0
        assert source == "ec_ligase"

    def test_ec_3_hydrolase_class_fallback(self):
        kcat, source, conf = _estimate_kcat_from_sequence("A" * 300, "3.99.99.99")
        assert kcat == 30.0
        assert source == "ec_hydrolase"

    def test_km_glucose_range(self):
        est = SequenceKmEstimator()
        result = est.predict(
            sequence="MKTTHTGIIIAIGA" * 10,
            substrate="glucose",
        )
        assert 0.01 <= result.km_value <= 2.0

    def test_km_atp_range(self):
        est = SequenceKmEstimator()
        result = est.predict(
            sequence="MKTTHTGIIIAIGA" * 10,
            substrate="ATP",
        )
        assert 0.01 <= result.km_value <= 2.0

    def test_km_nadph_range(self):
        est = SequenceKmEstimator()
        result = est.predict(
            sequence="MKTTHTGIIIAIGA" * 10,
            substrate="NADPH",
        )
        assert result.km_value < 0.2

    def test_km_nad_range(self):
        est = SequenceKmEstimator()
        result = est.predict(
            sequence="MKTTHTGIIIAIGA" * 10,
            substrate="NAD",
        )
        assert 0.005 <= result.km_value <= 1.0


class TestESM2Embedding:
    """Tests for ESM-2 embedding extraction (requires fair-esm)."""

    def test_esm2_available_returns_bool(self):
        assert isinstance(is_esm2_available(), bool)

    def test_esm2_embedding_dimension(self):
        if not is_esm2_available():
            pytest.skip("ESM-2 not installed")
        emb = get_esm2_embedding("MKWVTFISLLFLFSSAYS")
        assert len(emb) == 320
        assert all(isinstance(x, float) for x in emb)

    def test_esm2_embedding_varies_with_sequence(self):
        if not is_esm2_available():
            pytest.skip("ESM-2 not installed")
        emb1 = get_esm2_embedding("ACDEFGHIKLMNPQRSTVWY" * 10)
        emb2 = get_esm2_embedding("MKTLYFNRGELQTPAAIALAARGYRNFVSGEVPW" * 3)
        assert emb1 != emb2

    def test_esm2_empty_sequence_fallback(self):
        emb = get_esm2_embedding("")
        assert len(emb) == 20
        assert all(x == 0.0 for x in emb)
