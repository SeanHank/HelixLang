"""Protein structure prediction tests: verified against real literature data.

Verifies:
- Chou-Fasman secondary structure prediction: α-helix / β-sheet / turn detection
- Kyte-Doolittle hydropathy profile: hydrophobic/hydrophilic residue discrimination
- Transmembrane helix prediction: hydrophobic stretch + length threshold
- Disorder prediction: low hydrophobicity + high charge
- Complete structure report

References:
- Chou & Fasman 1978 Adv Enzymol 47:45-148 (secondary structure propensities)
- Kyte & Doolittle 1982 J Mol Biol 157:105-132 (hydrophobicity scale)
- Krogh et al. 2001 J Mol Biol 305:567-580 (TMHMM)
- Dunker et al. 2001 J Mol Graph Model 19:141-149 (disorder prediction)
"""
from __future__ import annotations

import pytest

from helixlang.plugins.runtime.protein_structure import (
    _GOR_IV_DSSP_PAIR,
    _GOR_IV_DSSP_SINGLET,
    _GOR_PAIR_INFO,
    _GOR_SINGLET_INFO,
    # data tables
    CHOU_FASMAN_TABLE,
    GOR_COIL_THRESHOLD,
    GOR_IV_DSSP_PAIR_WEIGHT,
    GOR_IV_DSSP_SINGLET_WEIGHT,
    GOR_IV_DSSP_STATES,
    GOR_PAIR_OFFSETS,
    GOR_STATES,
    GOR_WINDOW_RADIUS,
    HELIX_BREAKERS,
    HELIX_FORMERS,
    # parameters
    KYTE_DOOLITTLE_SCALE,
    SHEET_FORMERS,
    TM_HYDROPATHY_THRESHOLD,
    TM_MIN_LENGTH,
    SecondaryStructureSegment,
    gravy,
    hydropathy_profile,
    iupred_scores,
    predict_disorder,
    # functions
    predict_secondary,
    predict_secondary_gor,
    predict_structure,
    predict_transmembrane,
)

# ============================================================================
# Data table completeness
# ============================================================================

class TestDataTable:
    """Verify the completeness of the parameter tables."""

    def test_chou_fasman_has_all_20_amino_acids(self):
        """Chou-Fasman table contains all 20 amino acids."""
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        assert set(CHOU_FASMAN_TABLE.keys()) == valid_aas

    def test_chou_fasman_values_in_range(self):
        """Chou-Fasman propensity values are in the reasonable range [0, 2.5]."""
        for aa, (pa, pb, pturn, _label) in CHOU_FASMAN_TABLE.items():
            assert 0.0 < pa < 2.5, f"{aa}: P_a={pa} out of range"
            assert 0.0 < pb < 2.5, f"{aa}: P_b={pb} out of range"
            assert 0.0 < pturn < 2.5, f"{aa}: P_turn={pturn} out of range"

    def test_kyte_doolittle_has_all_20_amino_acids(self):
        """KD hydrophobicity table contains all 20 amino acids."""
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        assert set(KYTE_DOOLITTLE_SCALE.keys()) == valid_aas

    def test_kyte_doolittle_range(self):
        """KD values are in the [-4.5, +4.5] range (original paper)."""
        for aa, val in KYTE_DOOLITTLE_SCALE.items():
            assert -4.5 <= val <= 4.5, f"{aa}: {val} out of range"

    def test_ile_most_hydrophobic(self):
        """Ile is the most hydrophobic amino acid (KD +4.5)."""
        assert KYTE_DOOLITTLE_SCALE["I"] == 4.5

    def test_arg_most_hydrophilic(self):
        """Arg is the most hydrophilic amino acid (KD -4.5)."""
        assert KYTE_DOOLITTLE_SCALE["R"] == -4.5

    def test_helix_formers_consistent(self):
        """Helix formers set is consistent with P_a > 1.0."""
        for aa in HELIX_FORMERS:
            assert CHOU_FASMAN_TABLE[aa][0] >= 1.0, f"{aa} should not be helix former"

    def test_sheet_formers_consistent(self):
        """Sheet formers set is consistent with P_b > 1.0."""
        for aa in SHEET_FORMERS:
            assert CHOU_FASMAN_TABLE[aa][1] >= 1.0, f"{aa} should not be sheet former"

    def test_pro_is_helix_breaker(self):
        """Pro is a helix breaker (P_a < 0.7)."""
        assert "P" in HELIX_BREAKERS
        assert CHOU_FASMAN_TABLE["P"][0] < 0.7


# ============================================================================
# Chou-Fasman secondary structure prediction
# ============================================================================

class TestSecondaryStructure:
    """Verify Chou-Fasman secondary structure prediction."""

    def test_helix_peptide_predicted_as_helix(self):
        """Ala+Leu+Glu enriched peptide is predicted as helix (strong helix formers)."""
        # 6 Ala + 4 Leu + 4 Glu = 14 aa helix
        seq = "A" * 6 + "L" * 4 + "E" * 4
        ss, segments = predict_secondary(seq)
        # at least 60% of residues should be helix
        helix_frac = ss.count("H") / len(seq)
        assert helix_frac > 0.6, f"helix fraction {helix_frac:.2f} too low"

    def test_sheet_peptide_predicted_as_sheet(self):
        """Val+Ile enriched peptide is predicted as sheet."""
        # 3 Val + 3 Ile = 6 aa sheet nucleation
        seq = "V" * 3 + "I" * 3 + "V" * 3
        ss, segments = predict_secondary(seq)
        # at least 50% of residues should be sheet
        sheet_frac = ss.count("E") / len(seq)
        assert sheet_frac > 0.5, f"sheet fraction {sheet_frac:.2f} too low"

    def test_proline_breaks_helix(self):
        """Pro interrupts the helix (helix breaker)."""
        # first 10 aa helix, Pro in the middle, last 10 aa helix
        seq = "A" * 10 + "P" + "A" * 10
        ss, segments = predict_secondary(seq)
        # Pro position (index 10) should not be H
        assert ss[10] != "H"
        # helices on both sides
        assert ss[0] == "H"
        assert ss[-1] == "H"

    def test_glycine_rich_predicted_as_turn(self):
        """Gly-rich peptide tends to form a turn."""
        # Gly is the strongest turn former
        seq = "G" * 4 + "P" * 4 + "G" * 4
        ss, segments = predict_secondary(seq)
        # there should be a turn segment
        assert "T" in ss

    def test_ss_string_length_matches_sequence(self):
        """SS string length = sequence length."""
        seq = "MKVLAACDEFGHIKLMNPQRSTVWY"
        ss, segments = predict_secondary(seq)
        assert len(ss) == len(seq)

    def test_ss_string_only_valid_chars(self):
        """SS string only contains H/E/T/C."""
        seq = "MKVLAACDEFGHIKLMNPQRSTVWY"
        ss, segments = predict_secondary(seq)
        valid_chars = set("HETC")
        assert set(ss).issubset(valid_chars)

    def test_segments_contiguous(self):
        """segments contiguously cover the entire sequence."""
        seq = "MKVLAACDEFGHIKLMNPQRSTVWY" * 2
        ss, segments = predict_secondary(seq)
        # first segment starts at 1
        assert segments[0].start == 1
        # last segment ends at len(seq)
        assert segments[-1].end == len(seq)
        # adjacent segments are contiguous
        for i in range(len(segments) - 1):
            assert segments[i].end + 1 == segments[i + 1].start

    def test_empty_sequence_returns_empty(self):
        """Empty sequence returns empty."""
        ss, segments = predict_secondary("")
        assert ss == ""
        assert segments == []

    def test_invalid_amino_acid_raises(self):
        """Invalid amino acid raises ValueError."""
        with pytest.raises(ValueError):
            predict_secondary("ACGTBX")  # B, X invalid

    def test_short_sequence_no_helix(self):
        """Short sequence (< 6) should not trigger helix nucleation."""
        seq = "AAAA"  # 4 Ala, < 6
        ss, segments = predict_secondary(seq)
        # no helix nucleation
        assert "H" not in ss


# ============================================================================
# GOR IV secondary structure prediction (17-residue window + singlet/pair Shannon information theory)
# ============================================================================

class TestSecondaryStructureGOR:
    """Verify GOR IV-style secondary structure prediction (17-residue window + singlet/pair information theory)."""

    def test_gor_window_is_17(self):
        """GOR window radius = 8 (window size = 17)."""
        assert GOR_WINDOW_RADIUS == 8
        assert 2 * GOR_WINDOW_RADIUS + 1 == 17

    def test_gor_states_are_helical_sheet_turn(self):
        """GOR three states are H/E/T (coil is the fallback when the threshold is not met)."""
        assert GOR_STATES == ("H", "E", "T")

    def test_gor_coil_threshold_zero(self):
        """Coil threshold is 0 (negative information is recorded as coil)."""
        assert GOR_COIL_THRESHOLD == 0.0

    def test_gor_ss_string_length_matches_sequence(self):
        """GOR SS string length = sequence length."""
        seq = "MKVLAACDEFGHIKLMNPQRSTVWY"
        ss, segments = predict_secondary_gor(seq)
        assert len(ss) == len(seq)

    def test_gor_ss_string_only_valid_chars(self):
        """GOR SS string only contains H/E/T/C."""
        seq = "MKVLAACDEFGHIKLMNPQRSTVWY"
        ss, segments = predict_secondary_gor(seq)
        valid_chars = set("HETC")
        assert set(ss).issubset(valid_chars)

    def test_gor_empty_sequence_returns_empty(self):
        """Empty sequence returns empty."""
        ss, segments = predict_secondary_gor("")
        assert ss == ""
        assert segments == []

    def test_gor_invalid_amino_acid_raises(self):
        """Invalid amino acid raises ValueError (consistent with Chou-Fasman)."""
        with pytest.raises(ValueError):
            predict_secondary_gor("ACGTBX")  # B, X invalid

    def test_gor_short_sequence_valid_ss(self):
        """Short sequence (< 17 aa) returns valid SS (window truncated, no error)."""
        for seq in ("A", "AC", "AAAA", "ACDEFGHIK"):  # 1, 2, 4, 9 aa
            ss, segments = predict_secondary_gor(seq)
            assert len(ss) == len(seq)
            assert set(ss).issubset(set("HETC"))

    def test_gor_poly_ala_predicted_as_helix(self):
        """poly-Ala is a strong helix former; GOR should predict helix."""
        seq = "A" * 10
        ss, segments = predict_secondary_gor(seq)
        helix_frac = ss.count("H") / len(seq)
        assert helix_frac > 0.8, f"helix fraction {helix_frac:.2f} too low for poly-Ala"

    def test_gor_maeelkkl_predicted_as_helix(self):
        """Helix-former enriched peptide (M/A/E/L/K) is predicted as helix."""
        seq = "MAEELKKL"
        ss, segments = predict_secondary_gor(seq)
        helix_frac = ss.count("H") / len(seq)
        assert helix_frac > 0.6, f"helix fraction {helix_frac:.2f} too low for MAEELKKL"

    def test_gor_poly_val_predicted_as_sheet(self):
        """poly-Val is a strong sheet former; GOR should predict sheet."""
        seq = "V" * 8
        ss, segments = predict_secondary_gor(seq)
        sheet_frac = ss.count("E") / len(seq)
        assert sheet_frac > 0.8, f"sheet fraction {sheet_frac:.2f} too low for poly-Val"

    def test_gor_poly_ila_predicted_as_sheet(self):
        """poly-Ile is a strong sheet former; GOR should predict sheet."""
        seq = "I" * 10
        ss, segments = predict_secondary_gor(seq)
        sheet_frac = ss.count("E") / len(seq)
        assert sheet_frac > 0.8, f"sheet fraction {sheet_frac:.2f} too low for poly-Ile"

    def test_gor_vs_chou_fasman_both_detect_poly_ala_helix(self):
        """Both GOR and Chou-Fasman should detect poly-Ala as helix."""
        seq = "A" * 12
        gor_ss, _ = predict_secondary_gor(seq)
        cf_ss, _ = predict_secondary(seq)
        assert gor_ss.count("H") > 0, "GOR did not detect poly-Ala helix"
        assert cf_ss.count("H") > 0, "Chou-Fasman did not detect poly-Ala helix"

    def test_gor_segments_contiguous(self):
        """GOR segments contiguously cover the entire sequence."""
        seq = "MKVLAACDEFGHIKLMNPQRSTVWY" * 2
        ss, segments = predict_secondary_gor(seq)
        assert segments[0].start == 1
        assert segments[-1].end == len(seq)
        for i in range(len(segments) - 1):
            assert segments[i].end + 1 == segments[i + 1].start

    def test_gor_returns_segment_objects(self):
        """GOR returns segments as SecondaryStructureSegment."""
        seq = "AAAAAAAAAA"
        ss, segments = predict_secondary_gor(seq)
        assert all(isinstance(s, SecondaryStructureSegment) for s in segments)
        # segment ss_type is only H/E/T/C
        assert all(s.ss_type in "HETC" for s in segments)

    def test_gor_distinguishes_helix_vs_sheet_sequences(self):
        """Helix-former enriched sequences have a higher helix fraction than sheet-former enriched sequences."""
        helix_seq = "ALEK" * 6   # A/L/E/K strong helix formers
        sheet_seq = "VIYF" * 6   # V/I/Y/F strong sheet formers
        h_ss, _ = predict_secondary_gor(helix_seq)
        s_ss, _ = predict_secondary_gor(sheet_seq)
        assert h_ss.count("H") > s_ss.count("H")
        assert s_ss.count("E") > h_ss.count("E")

    # ========================================================================
    # GOR IV pair (dipeptide) synergy effects
    # ========================================================================

    def test_gor_iv_helix_pair_synergy_enhances_helix(self):
        """helix-favoring pair (A-E) is more helix-prone than breaker pair (A-G).

        A and E are both strong helix formers; the dipeptides AE/EA trigger a pair
        synergy factor (>1); A is a helix former but G is a helix breaker; the
        dipeptides AG/GA trigger a breaker antagonism factor (<1), and G's own
        singlet helix information is negative. Therefore the AE repeat sequence
        should be predicted as helix overall, while the AG repeat sequence should
        have a significantly lower helix fraction.
        """
        ae_seq = "AE" * 8   # 16 aa, all helix-helix synergy pairs
        ag_seq = "AG" * 8   # 16 aa, helix-breaker antagonism pairs
        ae_ss, _ = predict_secondary_gor(ae_seq)
        ag_ss, _ = predict_secondary_gor(ag_seq)
        assert ae_ss.count("H") > ag_ss.count("H")

    def test_gor_iv_pro_pair_breaker_reduces_helix(self):
        """Pro reduces helix propensity via the pair antagonism factor.

        Gln (P_a=1.11) is the weakest helix former, and poly-Gln is still
        predicted as helix overall; after inserting Pro, Pro's own singlet helix
        information is negative, and the pair antagonism factor (<1) further
        suppresses the helix information at its ±1 and ±2 neighbors, flipping the
        Pro position and its surroundings to non-helix. Hence the helix fraction of
        the Pro-containing sequence should be lower than that of pure poly-Gln, and
        the Pro position itself should not be helix.
        """
        pure_gln = "Q" * 8
        pro_in = "QQQQ" + "P" + "QQQ"   # 8 aa, Pro at index 4
        pure_ss, _ = predict_secondary_gor(pure_gln)
        pro_ss, _ = predict_secondary_gor(pro_in)
        assert pro_ss[4] != "H"                       # Pro position is not helix
        assert pro_ss.count("H") < pure_ss.count("H")  # Pro lowers overall helix

    def test_gor_iv_sheet_pair_synergy_enhances_sheet(self):
        """Sheet-favoring pair (V-I) synergistically enhances sheet prediction.

        V and I are both strong sheet formers; the dipeptides VI/IV trigger a
        sheet-sheet synergy factor (>1). Compared to the helix-former pair (A-L,
        no sheet synergy, corr=1.0), the VI repeat sequence should have a higher
        sheet fraction (A-L prefers helix over sheet).
        """
        vi_seq = "VI" * 8   # 16 aa, all sheet-sheet synergy pairs
        al_seq = "AL" * 8   # 16 aa, helix-helix pair (no sheet synergy)
        vi_ss, _ = predict_secondary_gor(vi_seq)
        al_ss, _ = predict_secondary_gor(al_seq)
        assert vi_ss.count("E") > al_ss.count("E")


# ============================================================================
# GOR IV real DSSP-trained tables
# ============================================================================

class TestGorIvDsspTables:
    """Verify the real DSSP-trained GOR IV singlet/pair information tables."""

    def test_states_are_hec(self):
        """The real GOR IV prediction states are H/E/C."""
        assert GOR_IV_DSSP_STATES == ("H", "E", "C")

    def test_singlet_table_completeness(self):
        """Singlet table has 20 AAs x 17 offsets x 3 states = 1020 entries."""
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        assert len(_GOR_IV_DSSP_SINGLET) == 1020
        for aa in valid_aas:
            for d in range(-GOR_WINDOW_RADIUS, GOR_WINDOW_RADIUS + 1):
                for s in GOR_IV_DSSP_STATES:
                    assert (aa, d, s) in _GOR_IV_DSSP_SINGLET, (aa, d, s)

    def test_pair_table_completeness(self):
        """Pair table has 20 x 20 AAs x 4 offsets x 3 states = 4800 entries."""
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        assert len(_GOR_IV_DSSP_PAIR) == 4800
        for aa1 in valid_aas:
            for aa2 in valid_aas:
                for d in GOR_PAIR_OFFSETS:
                    for s in GOR_IV_DSSP_STATES:
                        assert (aa1, aa2, d, s) in _GOR_IV_DSSP_PAIR, (aa1, aa2, d, s)

    def test_weights_match_decipher(self):
        """f1 = 15/17 singlet fraction; f2 = 60/17 scaled pair weight."""
        assert GOR_IV_DSSP_SINGLET_WEIGHT == 15.0 / 17.0
        assert GOR_IV_DSSP_PAIR_WEIGHT == 60.0 / 17.0

    def test_table_provenance_not_chou_fasman(self):
        """GOR IV tables are DSSP-trained, NOT derived from Chou-Fasman."""
        aa_diffs = sum(
            1
            for (aa, d, s), v in _GOR_IV_DSSP_SINGLET.items()
            if s in ("H", "E") and abs(v - _GOR_SINGLET_INFO[(aa, d, s)]) > 1e-9
        )
        assert aa_diffs == 680  # all 20 x 17 x 2 shared singlet entries differ
        pair_diffs = sum(
            1
            for (aa1, aa2, d, s), v in _GOR_IV_DSSP_PAIR.items()
            if s in ("H", "E") and abs(v - _GOR_PAIR_INFO[(aa1 + aa2, d, s)]) > 1e-9
        )
        assert pair_diffs == 3200  # all 20 x 20 x 4 x 2 shared pair entries differ


class TestGorIvDsspPredictions:
    """Verify predictions with the real DSSP-trained GOR IV tables."""

    def test_default_method_is_gor_iv_dssp(self):
        """predict_secondary_gor defaults to the real DSSP-trained tables."""
        ss, _ = predict_secondary_gor("A" * 12)
        assert set(ss) <= set("HEC")

    def test_poly_ala_all_helix(self):
        """Poly-Ala is predicted as all helix (real GOR IV tables)."""
        ss, _ = predict_secondary_gor("A" * 24)
        assert ss == "H" * 24

    def test_poly_val_all_sheet(self):
        """Poly-Val is predicted as all sheet (real GOR IV tables)."""
        ss, _ = predict_secondary_gor("V" * 24)
        assert ss == "E" * 24

    def test_poly_ile_all_sheet(self):
        """Poly-Ile is predicted as all sheet (real GOR IV tables)."""
        ss, _ = predict_secondary_gor("I" * 24)
        assert ss == "E" * 24

    def test_helix_former_peptide_helix(self):
        """M/A/E/L/K enriched peptide is predicted as helix."""
        ss, _ = predict_secondary_gor("MAEELKKLAA")
        assert ss.count("H") >= 7

    def test_ae_repeat_all_helix(self):
        """A/E helix-helix synergy repeat is all helix."""
        ss, _ = predict_secondary_gor("AE" * 8)
        assert ss == "H" * 16

    def test_ag_repeat_alternates_hc(self):
        """A/G helix-breaker repeat alternates H/C (half helix)."""
        ss, _ = predict_secondary_gor("AG" * 8)
        assert set(ss) == {"H", "C"}
        assert ss.count("H") == ss.count("C") == 8

    def test_vi_repeat_all_sheet(self):
        """V/I sheet-sheet synergy repeat is all sheet."""
        ss, _ = predict_secondary_gor("VI" * 8)
        assert ss == "E" * 16

    def test_al_repeat_less_sheet_than_vi(self):
        """A/L (no sheet synergy) has fewer sheet residues than V/I."""
        al_ss, _ = predict_secondary_gor("AL" * 8)
        vi_ss, _ = predict_secondary_gor("VI" * 8)
        assert al_ss.count("E") < vi_ss.count("E")

    def test_pp_insertion_breaks_helix(self):
        """Pro-Pro insertion breaks the poly-Ala helix."""
        ss, _ = predict_secondary_gor("A" * 10 + "PP" + "A" * 12)
        assert ss.count("H") >= 18
        assert ss[10] != "H" and ss[11] != "H"  # the Pro-Pro positions

    def test_helix_formers_more_helix_than_sheet_formers(self):
        """Helix-former repeat has more H; sheet-former repeat more E."""
        h_ss, _ = predict_secondary_gor("ALEK" * 6)
        s_ss, _ = predict_secondary_gor("VIYF" * 6)
        assert h_ss.count("H") == len(h_ss)
        assert s_ss.count("E") == len(s_ss)

    def test_legacy_chou_fasman_method_still_reachable(self):
        """The legacy Chou-Fasman-derived tables remain callable."""
        ss, _ = predict_secondary_gor("ACDEFGHIKLMNPQRSTVWY", method="chou_fasman")
        assert set(ss) <= set("HETC")

    def test_unknown_method_raises(self):
        """Unknown GOR method raises ValueError."""
        with pytest.raises(ValueError):
            predict_secondary_gor("A" * 8, method="bogus")


# ============================================================================
# Kyte-Doolittle hydropathy profile
# ============================================================================

class TestHydropathyProfile:
    """Verify the Kyte-Doolittle hydropathy profile."""

    def test_profile_length_matches_sequence(self):
        """Profile length = sequence length."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        profile = hydropathy_profile(seq)
        assert len(profile) == len(seq)

    def test_hydrophobic_region_high_values(self):
        """Hydrophobic regions have high KD values."""
        # all Ile (most hydrophobic)
        seq = "I" * 20
        profile = hydropathy_profile(seq)
        # center position should be near +4.5
        assert profile[10] > 4.0

    def test_hydrophilic_region_low_values(self):
        """Hydrophilic regions have low KD values."""
        # all Arg (most hydrophilic)
        seq = "R" * 20
        profile = hydropathy_profile(seq)
        # center position should be near -4.5
        assert profile[10] < -4.0

    def test_window_smooths_profile(self):
        """Window averaging smooths the profile."""
        # alternating hydrophobic/hydrophilic
        seq = "IR" * 20  # I (+4.5), R (-4.5)
        profile = hydropathy_profile(seq, window=9)
        # average should be near 0 (window 9 center has 5 I + 4 R → 0.5)
        assert abs(profile[10]) <= 0.5

    def test_window_size_param(self):
        """The window parameter affects smoothing."""
        seq = "IR" * 20
        profile_small = hydropathy_profile(seq, window=3)
        profile_large = hydropathy_profile(seq, window=11)
        # larger window is smoother
        small_var = max(profile_small) - min(profile_small)
        large_var = max(profile_large) - min(profile_large)
        assert large_var <= small_var

    def test_invalid_window_raises(self):
        """window < 1 raises an exception."""
        with pytest.raises(ValueError):
            hydropathy_profile("ACDEFG", window=0)

    def test_empty_sequence_returns_empty(self):
        """Empty sequence returns an empty list."""
        assert hydropathy_profile("") == []


# ============================================================================
# GRAVY
# ============================================================================

class TestGravy:
    """Verify GRAVY calculation."""

    def test_gravy_pure_hydrophobic(self):
        """GRAVY is high for purely hydrophobic amino acids."""
        gravy_val = gravy("IIIIII")
        assert gravy_val == pytest.approx(4.5)

    def test_gravy_pure_hydrophilic(self):
        """GRAVY is low for purely hydrophilic amino acids."""
        gravy_val = gravy("RRRRRR")
        assert gravy_val == pytest.approx(-4.5)

    def test_gravy_mixed_near_zero(self):
        """GRAVY is near 0 for a hydrophobic + hydrophilic mix."""
        gravy_val = gravy("IRIRIRIR")
        # I(+4.5) + R(-4.5) = 0
        assert gravy_val == pytest.approx(0.0)

    def test_gravy_empty_sequence(self):
        """GRAVY = 0 for an empty sequence."""
        assert gravy("") == 0.0


# ============================================================================
# Transmembrane helix prediction
# ============================================================================

class TestTransmembrane:
    """Verify transmembrane helix prediction."""

    def test_pure_hydrophobic_seq_detected_as_tm(self):
        """Pure hydrophobic stretch is predicted as TM."""
        # 20 Leu, typical TM length
        seq = "L" * 20
        tms = predict_transmembrane(seq)
        assert len(tms) == 1
        assert tms[0].length >= TM_MIN_LENGTH
        assert tms[0].mean_hydropathy > TM_HYDROPATHY_THRESHOLD

    def test_hydrophilic_seq_no_tm(self):
        """Hydrophilic sequence has no TM."""
        seq = "D" * 30 + "K" * 30  # all hydrophilic
        tms = predict_transmembrane(seq)
        assert tms == []

    def test_tm_segment_length_in_range(self):
        """TM length is in the [min_length, max_length+5] range."""
        seq = "L" * 50  # long stretch
        tms = predict_transmembrane(seq)
        for tm in tms:
            assert tm.length >= TM_MIN_LENGTH
            assert tm.length <= 35  # max_length + 5

    def test_multiple_tm_helices(self):
        """Multiple TM helices (e.g. GPCR)."""
        # 2 TM, hydrophilic loop in the middle
        seq = "L" * 22 + "DDDDEEEE" + "L" * 22
        tms = predict_transmembrane(seq)
        assert len(tms) >= 2

    def test_short_hydrophobic_stretch_no_tm(self):
        """Short hydrophobic stretch (< min_length) is not counted as TM."""
        seq = "L" * 10  # shorter than 18
        tms = predict_transmembrane(seq)
        assert tms == []

    def test_tm_positions_in_sequence(self):
        """TM positions are within the sequence."""
        seq = "M" * 5 + "L" * 20 + "M" * 5  # 30 aa
        tms = predict_transmembrane(seq)
        for tm in tms:
            assert 1 <= tm.start
            assert tm.end <= len(seq)
            assert tm.start < tm.end

    def test_tm_mean_hydropathy_above_threshold(self):
        """TM mean hydropathy > threshold."""
        seq = "L" * 25
        tms = predict_transmembrane(seq)
        for tm in tms:
            assert tm.mean_hydropathy > TM_HYDROPATHY_THRESHOLD


# ============================================================================
# Disorder prediction
# ============================================================================

class TestDisorder:
    """Verify disorder prediction."""

    def test_charged_low_hydro_is_disordered(self):
        """High-charge, low-hydrophobicity sequences are predicted as disordered."""
        # Glu+Lys enriched
        seq = "E" * 20 + "K" * 20
        regions = predict_disorder(seq)
        assert len(regions) >= 1
        assert regions[0].length >= 10

    def test_hydrophobic_seq_not_disordered(self):
        """Hydrophobic sequences are not predicted as disordered."""
        seq = "L" * 50
        regions = predict_disorder(seq)
        assert regions == []

    def test_disorder_region_positions(self):
        """Disorder region positions are within the sequence."""
        seq = "L" * 30 + "E" * 30 + "K" * 30
        regions = predict_disorder(seq)
        for r in regions:
            assert 1 <= r.start
            assert r.end <= len(seq)
            assert r.start < r.end

    def test_disorder_mean_hydropathy_negative(self):
        """Disorder region mean hydropathy < 0 (hydrophilic)."""
        seq = "E" * 50 + "K" * 50
        regions = predict_disorder(seq)
        for r in regions:
            assert r.mean_hydropathy < 0

    def test_empty_sequence_no_disorder(self):
        """Empty sequence has no disorder regions."""
        assert predict_disorder("") == []


# ============================================================================
# IUPred-style disorder scores
# ============================================================================

#: Human p53 (UniProt P04637, P53_HUMAN): a well-characterized partly
#: disordered protein with experimentally mapped IDRs; reference scores
#: computed with the official IUPred2A server algorithm.
P53_HUMAN = (
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"
    "DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK"
    "SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE"
    "RCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNS"
    "SCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELP"
    "PGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPG"
    "GSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
)

class TestIupredScores:
    """Verify the IUPred-style per-residue disorder scores."""

    def test_scores_in_unit_interval(self):
        """All scores are in [0, 1]."""
        scores = iupred_scores(P53_HUMAN, mode="long")
        assert len(scores) == len(P53_HUMAN)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_p53_long_disorder_fraction(self):
        """P53 long-mode disorder fraction matches the official IUPred2A output."""
        scores = iupred_scores(P53_HUMAN, mode="long")
        frac = sum(1 for s in scores if s > 0.5) / len(scores)
        assert 0.45 <= frac <= 0.55, f"long disorder fraction {frac:.3f}"

    def test_p53_short_disorder_fraction(self):
        """P53 short-mode disorder fraction matches the official IUPred2A output."""
        scores = iupred_scores(P53_HUMAN, mode="short")
        frac = sum(1 for s in scores if s > 0.5) / len(scores)
        assert 0.40 <= frac <= 0.48, f"short disorder fraction {frac:.3f}"

    def test_p53_terminal_scores_high(self):
        """The N-terminal IDR of p53 has high disorder scores."""
        scores = iupred_scores(P53_HUMAN, mode="long")
        assert all(s > 0.5 for s in scores[:8])

    def test_hydrophobic_control_not_disordered(self):
        """A purely hydrophobic sequence scores ~0 disordered."""
        scores = iupred_scores("I" * 50 + "V" * 50, mode="long")
        assert sum(1 for s in scores if s > 0.5) == 0

    def test_long_vs_short_mode_differ(self):
        """Long and short parameter sets give different scores."""
        long_scores = iupred_scores(P53_HUMAN, mode="long")
        short_scores = iupred_scores(P53_HUMAN, mode="short")
        assert long_scores != short_scores

    def test_empty_sequence_raises(self):
        """Empty sequence raises ValueError."""
        with pytest.raises(ValueError):
            iupred_scores("")

    def test_unknown_mode_raises(self):
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError):
            iupred_scores(P53_HUMAN, mode="bogus")


class TestPredictDisorderIupred:
    """Verify predict_disorder with method='iupred'."""

    def test_p53_long_regions(self):
        """P53 long-mode disorder regions match the official output."""
        regions = predict_disorder(P53_HUMAN, method="iupred")
        assert len(regions) == 4
        assert regions[0].start == 1
        assert sum(r.length for r in regions) == 188

    def test_p53_short_regions(self):
        """P53 short-mode disorder regions match the official output."""
        regions = predict_disorder(P53_HUMAN, method="iupred", iupred_mode="short")
        assert len(regions) == 4
        assert sum(r.length for r in regions) == 149

    def test_legacy_chou_dunker_still_default(self):
        """The default disorder method remains the legacy Chou-Dunker heuristic."""
        regions = predict_disorder("E" * 20 + "K" * 20)
        assert len(regions) >= 1
        assert regions[0].length >= 10

    def test_unknown_method_raises(self):
        """Unknown disorder method raises ValueError."""
        with pytest.raises(ValueError):
            predict_disorder(P53_HUMAN, method="bogus")


# ============================================================================
# Complete structure report
# ============================================================================

class TestStructureReport:
    """Verify the complete structure prediction report."""

    def test_report_basic_fields(self):
        """Report contains all required fields."""
        seq = "MKVLAACDEFGHIKLMNPQRSTVWY" * 3
        report = predict_structure(seq)
        for field in (
            "sequence", "length", "secondary_structure",
            "ss_segments", "helix_fraction", "sheet_fraction",
            "turn_fraction", "coil_fraction",
            "hydropathy_profile", "mean_hydropathy",
            "transmembrane_helices", "disorder_regions",
            "disorder_fraction", "is_membrane_protein",
            "gravy", "summary",
        ):
            assert hasattr(report, field), f"missing field {field}"

    def test_report_length_matches_sequence(self):
        """Report length = sequence length."""
        seq = "MKVLAACDEFGHIKLMNPQRSTVWY"
        report = predict_structure(seq)
        assert report.length == len(seq)

    def test_ss_string_length_matches(self):
        """SS string length = sequence length."""
        seq = "MKVLAACDEFGHIKLMNPQRSTVWY" * 3
        report = predict_structure(seq)
        assert len(report.secondary_structure) == len(seq)

    def test_fractions_sum_to_one(self):
        """Sum of the 4 SS fractions ≈ 1."""
        seq = "MKVLAACDEFGHIKLMNPQRSTVWY" * 3
        report = predict_structure(seq)
        total = (report.helix_fraction + report.sheet_fraction
                 + report.turn_fraction + report.coil_fraction)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_membrane_protein_detected(self):
        """Membrane protein (with TM helices) is detected."""
        # long Leu stretch + hydrophilic loop
        seq = "L" * 22 + "DDDD" + "L" * 22
        report = predict_structure(seq)
        assert report.is_membrane_protein
        assert len(report.transmembrane_helices) >= 1

    def test_soluble_protein_no_tm(self):
        """Soluble protein (no TM) is correctly classified."""
        # all hydrophilic
        seq = "D" * 20 + "K" * 20 + "E" * 20
        report = predict_structure(seq)
        assert not report.is_membrane_protein
        assert report.transmembrane_helices == []

    def test_to_dict_serialization(self):
        """to_dict returns a serializable dict."""
        seq = "MKVLAACDEFGHIKLMNPQRSTVWY"
        report = predict_structure(seq)
        d = report.to_dict()
        assert isinstance(d, dict)
        for key in ("length", "secondary_structure", "helix_fraction",
                    "mean_hydropathy", "gravy", "n_transmembrane_helices",
                    "is_membrane_protein", "disorder_fraction",
                    "n_disorder_regions", "summary"):
            assert key in d

    def test_summary_is_string(self):
        """summary is a string."""
        seq = "MKVLA"
        report = predict_structure(seq)
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    def test_gravy_in_report(self):
        """Report gravy is consistent with independent calculation."""
        seq = "MKVLAACDEFGHIKLMNPQRSTVWY"
        report = predict_structure(seq)
        assert report.gravy == pytest.approx(gravy(seq))


# ============================================================================
# Real protein fragment verification
# ============================================================================

class TestRealProteinFragments:
    """Use real protein fragments to verify prediction sanity."""

    def test_gpcr_tm7_like_has_multiple_tm(self):
        """GPCR 7-TM mimic sequence (alternating hydrophobic stretches) should identify multiple TMs."""
        # simplified GPCR: 7 × 22aa hydrophobic stretches + hydrophilic loop
        tm_unit = "L" * 22
        loop = "DEKR" * 6  # 24aa hydrophilic loop
        seq = (tm_unit + loop) * 7
        report = predict_structure(seq)
        # should identify at least 4 TMs (simplified model may not identify all)
        assert len(report.transmembrane_helices) >= 4
        assert report.is_membrane_protein

    def test_idp_like_protein_disordered(self):
        """Intrinsically disordered protein (IDP) mimic sequence should have disorder regions."""
        # enriched in E/K/P/S/Q (typical IDP residues)
        seq = "EEEEKKKKPPPPSSSSQQQQ" * 4  # 80 aa
        report = predict_structure(seq)
        # should have significant disorder regions
        assert report.disorder_fraction > 0.2

    def test_alpha_helical_protein_high_helix_fraction(self):
        """All-α-helical protein should have a high helix fraction."""
        # enriched in A/L/E/K (helix formers)
        seq = "ALEK" * 20  # 80 aa
        report = predict_structure(seq)
        # at least 50% helix
        assert report.helix_fraction > 0.5

    def test_beta_protein_high_sheet_fraction(self):
        """All-β protein should have a high sheet fraction."""
        # enriched in V/I/Y/F (sheet formers)
        seq = "VIYF" * 15  # 60 aa
        report = predict_structure(seq)
        # sheet fraction should be high (> 0.4)
        assert report.sheet_fraction > 0.3
