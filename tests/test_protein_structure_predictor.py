"""Tests for ESM3-based protein structure prediction (doc/26 Phase B)."""
from __future__ import annotations

import pytest

from helixlang.protein_structure_predictor import (
    ProteinStructure3D,
    _derive_secondary_from_coords,
    _detect_disorder,
    _detect_tm_helices,
    _validate_sequence,
    is_available,
    predict_structure_esm,
)

HAS_ESM = is_available()

slow_esm = pytest.mark.skipif(not HAS_ESM, reason="esm not installed")


def _dummy_coords(n: int):
    import numpy as np
    return np.random.default_rng(42).random((n, 3)) * 10.0


def _dummy_plddt(n: int, high: float = 0.8):
    import numpy as np
    return np.full(n, high)


class TestAvailability:
    def test_is_available_returns_bool(self):
        assert isinstance(is_available(), bool)


class TestValidateSequence:
    def test_valid(self):
        assert _validate_sequence("ACDEFGHIKLMNPQRSTVWY") == "ACDEFGHIKLMNPQRSTVWY"

    def test_lowercase(self):
        assert _validate_sequence("mkwv") == "MKWV"

    def test_with_x(self):
        assert _validate_sequence("ACX") == "ACX"

    def test_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            _validate_sequence("")

    def test_invalid_aa(self):
        with pytest.raises(ValueError, match="invalid amino acid"):
            _validate_sequence("AC123")


class TestDeriveSecondary:
    def test_short(self):
        assert _derive_secondary_from_coords(_dummy_coords(3)) == "CCC"

    def test_all_coil_random(self):
        coords = _dummy_coords(20)
        ss = _derive_secondary_from_coords(coords)
        assert len(ss) == 20
        assert all(c in "HEC" for c in ss)

    def test_helix_like(self):
        import numpy as np
        n = 20
        coords = np.zeros((n, 3))
        for i in range(n):
            angle = i * 100 * np.pi / 180
            coords[i] = [5.4 * np.cos(angle), 5.4 * np.sin(angle), 1.5 * i]
        ss = _derive_secondary_from_coords(coords)
        assert len(ss) == n


class TestDetectDisorder:
    def test_all_ordered(self):
        plddt = _dummy_plddt(10, 0.9)
        regions = _detect_disorder(plddt)
        assert len(regions) == 0

    def test_all_disordered(self):
        plddt = _dummy_plddt(10, 0.2)
        regions = _detect_disorder(plddt)
        assert len(regions) == 1
        assert regions[0].start == 0
        assert regions[0].end == 10

    def test_mixed(self):
        import numpy as np
        plddt = np.array([0.2, 0.2, 0.8, 0.8, 0.2, 0.2])
        regions = _detect_disorder(plddt)
        assert len(regions) == 2
        assert regions[0].start == 0
        assert regions[0].end == 2
        assert regions[1].start == 4
        assert regions[1].end == 6


class TestDetectTMHelices:
    def test_no_tm(self):
        seq = "ACDEFGHIKLMNPQRSTVWY" * 3
        plddt = _dummy_plddt(len(seq), 0.9)
        helices = _detect_tm_helices(seq, plddt)
        assert len(helices) == 0

    def test_tm_detected(self):
        hydrophobic = "LIIIIIIIIIIIIIIIIIIIIL"
        seq = "ACDEFGHIKL" + hydrophobic + "ACDEFGHIKL"
        plddt = _dummy_plddt(len(seq), 0.9)
        helices = _detect_tm_helices(seq, plddt, min_len=10, max_len=30)
        assert len(helices) >= 1


class TestProteinStructure3D:
    def test_creation(self):
        import numpy as np
        s = ProteinStructure3D(
            sequence="ACDE",
            coords=np.zeros((4, 3)),
            plddt=np.ones(4) * 80.0,
            secondary_structure="CCEC",
            mean_plddt=80.0,
            ptm_score=0.5,
        )
        assert s.sequence == "ACDE"
        assert s.mean_plddt == 80.0
        assert len(s.tm_helices) == 0
        assert len(s.disorder) == 0


@slow_esm
class TestESMPrediction:
    def test_short_protein(self):
        result = predict_structure_esm("MKWVTFISLLFLFSSAYS", num_steps=3)
        assert isinstance(result, ProteinStructure3D)
        assert len(result.sequence) == 18
        assert result.coords.shape[0] == 18
        assert result.coords.shape[1] == 3
        assert len(result.plddt) == 18
        assert result.mean_plddt > 0

    def test_truncation(self):
        long_seq = "ACDEFGHIKLMNPQRSTVWY" * 50
        result = predict_structure_esm(long_seq, max_residues=20, num_steps=3)
        assert len(result.sequence) == 20

    def test_plddt_range(self):
        result = predict_structure_esm("MKWVTFISLLFLFSSAYS", num_steps=3)
        assert all(0 <= p <= 100 for p in result.plddt)

    def test_invalid_sequence(self):
        with pytest.raises(ValueError, match="invalid amino acid"):
            predict_structure_esm("AC123")
