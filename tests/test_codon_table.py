"""Codon table unit tests."""
import pytest

from helixlang.codon_table import (
    CILIATE_TABLE,
    MITO_VERTEBRATE_TABLE,
    STANDARD_TABLE,
    WOBBLE_BITS,
    Op,
    get_table,
    wobble,
)
from helixlang.errors import HelixError


def test_standard_table_has_64_codons():
    assert len(STANDARD_TABLE) == 64


def test_all_codons_present():
    bases = "ACGT"
    for a in bases:
        for b in bases:
            for c in bases:
                codon = a + b + c
                assert codon in STANDARD_TABLE, f"missing {codon}"


def test_start_and_stop():
    assert STANDARD_TABLE["ATG"] == Op.OP_START
    assert STANDARD_TABLE["TAA"] == Op.OP_HALT
    assert STANDARD_TABLE["TAG"] == Op.OP_HALT
    assert STANDARD_TABLE["TGA"] == Op.OP_HALT


def test_degeneracy_ala_family():
    """All 4 codons in the Ala family map to OP_BUILD_PROTEIN."""
    for codon in ("GCT", "GCC", "GCA", "GCG"):
        assert STANDARD_TABLE[codon] == Op.OP_BUILD_PROTEIN


def test_degeneracy_leu_family():
    """All 6 codons in the Leu family map to OP_GROW_LSYSTEM."""
    for codon in ("CTT", "CTC", "CTA", "CTG", "TTA", "TTG"):
        assert STANDARD_TABLE[codon] == Op.OP_GROW_LSYSTEM


def test_wobble_bits():
    assert WOBBLE_BITS == {"A": 0, "C": 1, "G": 2, "T": 3}
    assert wobble("GCA") == 0  # third base A
    assert wobble("GCC") == 1  # third base C
    assert wobble("GCG") == 2  # third base G
    assert wobble("GCT") == 3  # third base T


def test_mito_table_tga_is_pigment():
    assert MITO_VERTEBRATE_TABLE["TGA"] == Op.OP_BUILD_PIGMENT
    assert MITO_VERTEBRATE_TABLE["ATA"] == Op.OP_START
    assert MITO_VERTEBRATE_TABLE["AGA"] == Op.OP_HALT
    assert MITO_VERTEBRATE_TABLE["AGG"] == Op.OP_HALT


def test_ciliate_table_taa_tag_are_morphogen():
    assert CILIATE_TABLE["TAA"] == Op.OP_EMIT_MORPHOGEN
    assert CILIATE_TABLE["TAG"] == Op.OP_EMIT_MORPHOGEN
    # TGA is still a stop
    assert CILIATE_TABLE["TGA"] == Op.OP_HALT


def test_get_table_unknown():
    with pytest.raises(HelixError):
        get_table("nonexistent")


def test_get_table_returns_known():
    assert get_table("standard") is STANDARD_TABLE
    assert get_table("mito_vertebrate") is MITO_VERTEBRATE_TABLE
    assert get_table("ciliate") is CILIATE_TABLE


def test_stop_codons_per_table():
    """Each table's stop codon set should correctly reflect the OP_HALT mapping."""
    std_stops = {c for c, op in STANDARD_TABLE.items() if op == Op.OP_HALT}
    assert std_stops == {"TAA", "TAG", "TGA"}
    mito_stops = {c for c, op in MITO_VERTEBRATE_TABLE.items() if op == Op.OP_HALT}
    assert "TGA" not in mito_stops
    assert "AGA" in mito_stops
    assert "AGG" in mito_stops
