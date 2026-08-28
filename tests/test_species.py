"""Multi-species support tests: verify the completeness and correctness of the yeast / human / ecoli codon usage tables.

Verifies:
- Completeness of the three-species codon usage tables (full 64-codon coverage)
- Codon-to-amino-acid mapping correctness (consistent with the standard genetic code)
- Synonymous codon fraction sums ≈ 1.0
- Species-specific preferences (yeast A/T preference, human C/G preference)
- Multi-species support for codon_adaptation_index / is_optimal_codon / is_rare_codon
- get_codon_usage / SPECIES_CODON_USAGE interface

References:
- Nakamura 2000 Nucleic Acids Res 28:292 (Kazusa CUTG)
- Sharp & Cowe 1991 Yeast 7:657-678 (S. cerevisiae codon preference)
- Plotkin 2004 Nature 428:926-930 (GC preference in human codon usage)
- Ikemura 1985 Mol Biol Evol 2:13-34 (codon usage and tRNA abundance)
"""
from __future__ import annotations

import math

import pytest

from helixlang.plugins.runtime.bio_data import (
    ECOLI_CODON_USAGE,
    HUMAN_CODON_USAGE,
    HUMAN_TRNA_ABUNDANCE,
    SPECIES_CODON_USAGE,
    SPECIES_DISPLAY_NAMES,
    SPECIES_TRNA_ABUNDANCE,
    YEAST_CODON_USAGE,
    YEAST_TRNA_ABUNDANCE,
    cai,
    codon_adaptation_index,
    get_codon_usage,
    get_species_display_name,
    get_species_trna,
    is_optimal_codon,
    is_rare_codon,
)

# ============================================================================
# Codon table completeness
# ============================================================================

class TestCodonTableCompleteness:
    """Verify the completeness of the three-species codon usage tables."""

    @pytest.mark.parametrize("table_name,table", [
        ("ecoli", ECOLI_CODON_USAGE),
        ("yeast", YEAST_CODON_USAGE),
        ("human", HUMAN_CODON_USAGE),
    ])
    def test_table_has_all_64_codons(self, table_name, table):
        """Each species codon usage table covers all 64 codons."""
        assert len(table) == 64, f"{table_name} has {len(table)} codons, expected 64"

    @pytest.mark.parametrize("table_name,table", [
        ("ecoli", ECOLI_CODON_USAGE),
        ("yeast", YEAST_CODON_USAGE),
        ("human", HUMAN_CODON_USAGE),
    ])
    def test_all_codons_are_valid(self, table_name, table):
        """All codons are valid 3-nt ACGT combinations."""
        valid_bases = {"A", "C", "G", "T"}
        for codon in table:
            assert len(codon) == 3, f"{table_name}: {codon} not 3 nt"
            assert set(codon) <= valid_bases, f"{table_name}: {codon} has invalid base"

    @pytest.mark.parametrize("table_name,table", [
        ("ecoli", ECOLI_CODON_USAGE),
        ("yeast", YEAST_CODON_USAGE),
        ("human", HUMAN_CODON_USAGE),
    ])
    def test_amino_acid_mapping_correct(self, table_name, table):
        """Codon → amino acid mapping matches the standard genetic code."""
        # standard genetic code
        standard_genetic_code = {
            "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
            "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
            "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
            "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
            "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
            "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
            "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
            "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
            "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
            "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
            "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
            "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
            "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
            "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
            "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
            "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
        }
        for codon, (aa, _, _) in table.items():
            assert aa == standard_genetic_code[codon], (
                f"{table_name}: {codon} → {aa} (expected {standard_genetic_code[codon]})"
            )

    @pytest.mark.parametrize("table_name,table", [
        ("ecoli", ECOLI_CODON_USAGE),
        ("yeast", YEAST_CODON_USAGE),
        ("human", HUMAN_CODON_USAGE),
    ])
    def test_fraction_sums_to_one(self, table_name, table):
        """The synonymous codon fractions for each amino acid sum to ≈ 1.0."""
        # group by amino acid
        aa_codons: dict[str, list[str]] = {}
        for codon, (aa, _, _) in table.items():
            aa_codons.setdefault(aa, []).append(codon)

        for aa, codons in aa_codons.items():
            total = sum(table[c][2] for c in codons)
            assert abs(total - 1.0) < 0.05, (
                f"{table_name}: AA {aa} fractions sum to {total}, expected ~1.0"
            )

    @pytest.mark.parametrize("table_name,table", [
        ("ecoli", ECOLI_CODON_USAGE),
        ("yeast", YEAST_CODON_USAGE),
        ("human", HUMAN_CODON_USAGE),
    ])
    def test_per_thousand_positive(self, table_name, table):
        """All per_thousand values are positive."""
        for codon, (_, per_thousand, _) in table.items():
            assert per_thousand > 0, f"{table_name}: {codon} per_thousand={per_thousand}"

    @pytest.mark.parametrize("table_name,table", [
        ("ecoli", ECOLI_CODON_USAGE),
        ("yeast", YEAST_CODON_USAGE),
        ("human", HUMAN_CODON_USAGE),
    ])
    def test_fraction_in_valid_range(self, table_name, table):
        """All fraction values are in (0, 1] range."""
        for codon, (_, _, frac) in table.items():
            assert 0.0 < frac <= 1.0, (
                f"{table_name}: {codon} fraction={frac} out of (0, 1]"
            )


# ============================================================================
# SPECIES_CODON_USAGE interface
# ============================================================================

class TestSpeciesInterface:
    """Verify the multi-species interface."""

    def test_species_dict_has_three_species(self):
        """SPECIES_CODON_USAGE contains ecoli / yeast / human."""
        assert "ecoli" in SPECIES_CODON_USAGE
        assert "yeast" in SPECIES_CODON_USAGE
        assert "human" in SPECIES_CODON_USAGE
        assert len(SPECIES_CODON_USAGE) == 3

    def test_get_codon_usage_ecoli(self):
        """get_codon_usage('ecoli') returns ECOLI_CODON_USAGE."""
        assert get_codon_usage("ecoli") is ECOLI_CODON_USAGE

    def test_get_codon_usage_yeast(self):
        """get_codon_usage('yeast') returns YEAST_CODON_USAGE."""
        assert get_codon_usage("yeast") is YEAST_CODON_USAGE

    def test_get_codon_usage_human(self):
        """get_codon_usage('human') returns HUMAN_CODON_USAGE."""
        assert get_codon_usage("human") is HUMAN_CODON_USAGE

    def test_get_codon_usage_default_ecoli(self):
        """get_codon_usage() defaults to ecoli."""
        assert get_codon_usage() is ECOLI_CODON_USAGE

    def test_get_codon_usage_unknown_raises(self):
        """Unknown species raises ValueError."""
        with pytest.raises(ValueError, match="unknown species"):
            get_codon_usage("mouse")

    def test_display_names_present(self):
        """All species have a display name."""
        for species in SPECIES_CODON_USAGE:
            assert species in SPECIES_DISPLAY_NAMES

    def test_get_species_display_name(self):
        """get_species_display_name returns the Latin name."""
        assert "Escherichia coli" in get_species_display_name("ecoli")
        assert "Saccharomyces cerevisiae" in get_species_display_name("yeast")
        assert "Homo sapiens" in get_species_display_name("human")

    def test_get_species_display_name_unknown(self):
        """Unknown species returns the species name itself."""
        assert get_species_display_name("mouse") == "mouse"


# ============================================================================
# Multi-species CAI / optimal codons / rare codons
# ============================================================================

class TestMultiSpeciesCAI:
    """Verify the multi-species codon adaptation index."""

    def test_cai_default_ecoli(self):
        """codon_adaptation_index defaults to ecoli."""
        # E. coli CTG fraction = 0.47
        assert codon_adaptation_index("CTG") == pytest.approx(0.47)

    def test_cai_ecoli_explicit(self):
        """Explicitly specify ecoli."""
        assert codon_adaptation_index("CTG", "ecoli") == pytest.approx(0.47)

    def test_cai_yeast(self):
        """Yeast CTG fraction = 0.14 (yeast prefers TTA)."""
        assert codon_adaptation_index("CTG", "yeast") == pytest.approx(0.14)

    def test_cai_human(self):
        """Human CTG fraction = 0.40 (human prefers CTG)."""
        assert codon_adaptation_index("CTG", "human") == pytest.approx(0.40)

    def test_cai_invalid_codon_returns_zero(self):
        """Invalid codons return 0."""
        assert codon_adaptation_index("XXX", "ecoli") == 0.0
        assert codon_adaptation_index("XXX", "yeast") == 0.0
        assert codon_adaptation_index("XXX", "human") == 0.0

    def test_cai_unknown_species_raises(self):
        """Unknown species raises an exception."""
        with pytest.raises(ValueError):
            codon_adaptation_index("ATG", "mouse")


class TestGeneLevelCAI:
    """Verify the true Sharp & Li 1987 geometric-mean gene-level CAI."""

    def test_cai_optimal_family_codon_is_one(self):
        """The most frequent codon of a family gives w = 1.0 (Sharp-Li)."""
        # E. coli: CTG is the most abundant Leu codon → CAI of pure CTG = 1.0
        assert cai("CTG" * 20) == pytest.approx(1.0)
        # ATG (single-codon family) also w = 1.0
        assert cai("ATG" * 5) == pytest.approx(1.0)

    def test_cai_geometric_vs_arithmetic(self):
        """True CAI is the geometric mean of relative adaptiveness
        (fraction / family max); the legacy simplified value is the
        arithmetic mean of the fractions."""
        # CTG (0.47, Leu family max) + CTA (0.04): relative adaptiveness
        # w = frac / family_max → geometric mean = sqrt(1 * 0.04/0.47)
        seq = "CTGCTA"
        assert cai(seq) == pytest.approx(
            math.sqrt(ECOLI_CODON_USAGE["CTA"][2] / ECOLI_CODON_USAGE["CTG"][2]))
        assert cai(seq, simplified=True) == pytest.approx(
            (ECOLI_CODON_USAGE["CTG"][2] + ECOLI_CODON_USAGE["CTA"][2]) / 2)

    def test_cai_species_specificity(self):
        """Same sequence scores differently across species tables."""
        seq = "TTACTG" * 5
        yeast = cai(seq, species="yeast")  # yeast prefers TTA
        ecoli = cai(seq, species="ecoli")  # ecoli prefers CTG
        assert yeast > ecoli

    def test_cai_stops_and_unknown_skipped(self):
        """Stop codons and unknown codons do not contribute."""
        assert cai("TAATGA") == 0.0
        assert cai("CTGCTG" + "TAA") == cai("CTGCTG")
        assert cai("CTGXXX") == cai("CTG")

    def test_cai_empty_and_noncoding(self):
        """Empty or all-stop sequences return 0.0."""
        assert cai("") == 0.0
        assert cai("TAG") == 0.0

    def test_cai_range(self):
        """CAI is always within [0, 1] for valid sequences."""
        for seq in ("CTG" * 10, "TTACTA" * 10, "ATG" + "CTG" * 30 + "TAA"):
            assert 0.0 <= cai(seq) <= 1.0

    def test_cai_unknown_species_raises(self):
        with pytest.raises(ValueError):
            cai("CTG" * 3, species="mouse")


class TestMultiSpeciesOptimalCodon:
    """Verify multi-species optimal codon determination."""

    def test_optimal_ecoli_ctg(self):
        """E. coli CTG (0.47) is an optimal codon."""
        assert is_optimal_codon("CTG", "ecoli") is True

    def test_optimal_yeast_tta(self):
        """Yeast TTA (0.27) is not an optimal codon (< 0.4); TTT (0.58) is."""
        assert is_optimal_codon("TTT", "yeast") is True
        assert is_optimal_codon("TTA", "yeast") is False

    def test_optimal_human_ctg(self):
        """Human CTG (0.40) is an optimal codon (>= 0.4)."""
        assert is_optimal_codon("CTG", "human") is True

    def test_optimal_default_ecoli(self):
        """Defaults to ecoli."""
        assert is_optimal_codon("CTG") is True

    def test_optimal_met_always_true(self):
        """ATG (Met) fraction = 1.0; optimal in all species."""
        for species in ("ecoli", "yeast", "human"):
            assert is_optimal_codon("ATG", species) is True

    def test_optimal_trp_always_true(self):
        """TGG (Trp) fraction = 1.0; optimal in all species."""
        for species in ("ecoli", "yeast", "human"):
            assert is_optimal_codon("TGG", species) is True


class TestMultiSpeciesRareCodon:
    """Verify multi-species rare codon determination."""

    def test_rare_ecoli_cta(self):
        """E. coli CTA (0.04) is a rare codon."""
        assert is_rare_codon("CTA", "ecoli") is True

    def test_rare_yeast_ctc(self):
        """Yeast CTC (0.06) is a rare codon."""
        assert is_rare_codon("CTC", "yeast") is True

    def test_rare_human_tta(self):
        """Human TTA (0.07) is a rare codon."""
        assert is_rare_codon("TTA", "human") is True

    def test_rare_default_ecoli(self):
        """Defaults to ecoli."""
        assert is_rare_codon("CTA") is True

    def test_not_rare_optimal_codon(self):
        """Optimal codons are not rare codons."""
        for species in ("ecoli", "yeast", "human"):
            assert is_rare_codon("ATG", species) is False

    def test_rare_codon_species_specific(self):
        """The same codon may differ between species: CTG is rare in yeast but not in human."""
        # Yeast CTG fraction = 0.11 < 0.15 → rare
        assert is_rare_codon("CTG", "yeast") is True
        # Human CTG fraction = 0.40 → not rare
        assert is_rare_codon("CTG", "human") is False


# ============================================================================
# Species-specific codon preferences
# ============================================================================

class TestSpeciesSpecificPreferences:
    """Verify species-specific codon preference patterns."""

    def test_yeast_prefers_at_ending(self):
        """Yeast prefers A/T-ending codons (high AT content).

        For each amino acid, compare total fraction of A/T-ending vs C/G-ending codons.
        """
        # pick a few representative amino acids
        at_ending = {"TTT", "TTA", "GTT", "GCT", "TCT", "CCT", "ACT", "CAT", "AAT", "GAT"}
        cg_ending = {"TTC", "CTG", "GTC", "GCC", "TCC", "CCC", "ACC", "CAC", "AAC", "GAC"}

        at_frac = sum(YEAST_CODON_USAGE[c][2] for c in at_ending if c in YEAST_CODON_USAGE)
        cg_frac = sum(YEAST_CODON_USAGE[c][2] for c in cg_ending if c in YEAST_CODON_USAGE)
        assert at_frac > cg_frac, (
            f"Yeast should prefer A/T-ending: AT={at_frac} vs CG={cg_frac}"
        )

    def test_human_prefers_cg_ending(self):
        """Human prefers C/G-ending codons (high GC content)."""
        at_ending = {"TTT", "TTA", "GTT", "GCT", "TCT", "CCT", "ACT", "CAT", "AAT", "GAT"}
        cg_ending = {"TTC", "CTG", "GTC", "GCC", "TCC", "CCC", "ACC", "CAC", "AAC", "GAC"}

        at_frac = sum(HUMAN_CODON_USAGE[c][2] for c in at_ending if c in HUMAN_CODON_USAGE)
        cg_frac = sum(HUMAN_CODON_USAGE[c][2] for c in cg_ending if c in HUMAN_CODON_USAGE)
        assert cg_frac > at_frac, (
            f"Human should prefer C/G-ending: CG={cg_frac} vs AT={at_frac}"
        )

    def test_ecoli_ctg_dominant_leu(self):
        """E. coli CTG is the dominant Leu codon (highest fraction)."""
        leu_codons = ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"]
        ecoli_leu = {c: ECOLI_CODON_USAGE[c][2] for c in leu_codons}
        assert max(ecoli_leu, key=ecoli_leu.get) == "CTG"

    def test_yeast_tta_dominant_leu(self):
        """Yeast TTA is the dominant Leu codon (highest fraction)."""
        leu_codons = ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"]
        yeast_leu = {c: YEAST_CODON_USAGE[c][2] for c in leu_codons}
        assert max(yeast_leu, key=yeast_leu.get) == "TTA"

    def test_human_ctg_dominant_leu(self):
        """Human CTG is the dominant Leu codon (highest fraction)."""
        leu_codons = ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"]
        human_leu = {c: HUMAN_CODON_USAGE[c][2] for c in leu_codons}
        assert max(human_leu, key=human_leu.get) == "CTG"

    def test_arg_preference_differs(self):
        """Arg codon preference differs among the three species.

        E. coli: CGT/CGC dominant (CGN preference)
        Yeast:   AGA dominant (AGR preference)
        Human:   AGA/AGG/CGG more even
        """
        arg_codons = ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"]

        ecoli_arg = {c: ECOLI_CODON_USAGE[c][2] for c in arg_codons}
        yeast_arg = {c: YEAST_CODON_USAGE[c][2] for c in arg_codons}

        # E. coli optimal Arg: CGT or CGC
        assert max(ecoli_arg, key=ecoli_arg.get) in ("CGT", "CGC")
        # Yeast optimal Arg: AGA
        assert max(yeast_arg, key=yeast_arg.get) == "AGA"

    def test_different_species_different_optimal_for_same_codon(self):
        """The same codon has different adaptation across species."""
        # AGA (Arg):
        #   E. coli: 0.07 (rare)
        #   Yeast:   0.48 (optimal)
        ecoli_aga = codon_adaptation_index("AGA", "ecoli")
        yeast_aga = codon_adaptation_index("AGA", "yeast")
        assert yeast_aga > ecoli_aga, (
            f"AGA should be better in yeast: yeast={yeast_aga} vs ecoli={ecoli_aga}"
        )

        # CGT (Arg):
        #   E. coli: 0.36 (moderate)
        #   Yeast:   0.14 (rare)
        ecoli_cgt = codon_adaptation_index("CGT", "ecoli")
        yeast_cgt = codon_adaptation_index("CGT", "yeast")
        assert ecoli_cgt > yeast_cgt, (
            f"CGT should be better in ecoli: ecoli={ecoli_cgt} vs yeast={yeast_cgt}"
        )


# ============================================================================
# Backward compatibility
# ============================================================================

class TestBackwardCompatibility:
    """Verify the multi-species changes do not break the original E. coli interface."""

    def test_ecoli_table_unchanged(self):
        """ECOLI_CODON_USAGE key values unchanged."""
        assert ECOLI_CODON_USAGE["CTG"] == ("L", 48.4, 0.47)
        assert ECOLI_CODON_USAGE["ATG"] == ("M", 26.4, 1.00)
        assert ECOLI_CODON_USAGE["TGG"] == ("W", 13.9, 1.00)

    def test_cai_without_species_arg(self):
        """codon_adaptation_index uses ecoli when no species argument is passed."""
        # consistent with explicit ecoli
        for codon in ("ATG", "CTG", "TTT", "AAA"):
            assert codon_adaptation_index(codon) == codon_adaptation_index(codon, "ecoli")

    def test_is_optimal_without_species_arg(self):
        """is_optimal_codon uses ecoli when no species argument is passed."""
        for codon in ("ATG", "CTG", "TTT", "AAA"):
            assert is_optimal_codon(codon) == is_optimal_codon(codon, "ecoli")

    def test_is_rare_without_species_arg(self):
        """is_rare_codon uses ecoli when no species argument is passed."""
        for codon in ("ATG", "CTG", "TTT", "AAA"):
            assert is_rare_codon(codon) == is_rare_codon(codon, "ecoli")


# ============================================================================
# tRNA abundance multi-species support (P0-1.2 + P2-3.2)
# ============================================================================

class TestTRNAAbundance:
    """Verify the completeness and correctness of the multi-species tRNA abundance tables.

    Data sources:
    - E. coli: Dong et al. J Mol Biol 1996 260:649-663
    - Yeast: Chan & Lowe GtRNAdb 2009 (tRNA gene copy numbers)
    - Human: Chan & Lowe GtRNAdb 2009; Dittmar et al. PLoS Genet 2006
    """

    def test_get_species_trna_three_species(self):
        """get_species_trna supports the three species."""
        for species in ("ecoli", "yeast", "human"):
            trna = get_species_trna(species)
            assert isinstance(trna, dict)
            assert len(trna) == 64  # 64-codon full coverage

    def test_get_species_trna_unknown_raises(self):
        """Unknown species should raise ValueError."""
        with pytest.raises(ValueError, match="unknown species"):
            get_species_trna("elephant")

    def test_get_species_trna_default_ecoli(self):
        """Defaults to ecoli when no argument is passed."""
        trna_default = get_species_trna()
        trna_ecoli = get_species_trna("ecoli")
        assert trna_default == trna_ecoli

    def test_trna_stop_codons_zero(self):
        """Stop codons have 0 tRNA abundance (no cognate tRNA)."""
        for species in ("ecoli", "yeast", "human"):
            trna = get_species_trna(species)
            for stop in ("TAA", "TAG", "TGA"):
                assert trna[stop] == 0, f"{species} {stop} should be 0"

    def test_trna_all_non_negative(self):
        """All tRNA abundances are non-negative."""
        for species in ("ecoli", "yeast", "human"):
            trna = get_species_trna(species)
            for codon, abundance in trna.items():
                assert abundance >= 0, f"{species} {codon}={abundance} < 0"

    def test_trna_ecoli_ctg_highest(self):
        """In E. coli, CTG corresponds to the highest-abundance tRNA (Leu-CAG, Dong 1996)."""
        trna = get_species_trna("ecoli")
        ctg = trna["CTG"]
        assert ctg == max(trna.values()), "CTG should have highest tRNA abundance"
        assert ctg >= 3000  # Dong 1996: ~3500

    def test_trna_ecoli_bridges_central_dogma(self):
        """E. coli tRNA abundance table matches central_dogma.TRNA_ABUNDANCE."""
        from helixlang.plugins.runtime.central_dogma import TRNA_ABUNDANCE
        trna = get_species_trna("ecoli")
        for codon in TRNA_ABUNDANCE:
            assert trna[codon] == TRNA_ABUNDANCE[codon]

    def test_yeast_trna_covers_64(self):
        """The yeast tRNA abundance table covers all 64 codons."""
        assert len(YEAST_TRNA_ABUNDANCE) == 64

    def test_human_trna_covers_64(self):
        """The human tRNA abundance table covers all 64 codons."""
        assert len(HUMAN_TRNA_ABUNDANCE) == 64

    def test_species_trna_dict_has_three(self):
        """SPECIES_TRNA_ABUNDANCE contains the three species."""
        assert set(SPECIES_TRNA_ABUNDANCE.keys()) == {"ecoli", "yeast", "human"}

    def test_trna_high_usage_codons_high_abundance(self):
        """High-usage codons correspond to high tRNA abundance (Ikemura 1985).

        In E. coli, CTG (fraction~0.55) has far higher abundance than CTA (fraction~0.04).
        """
        trna_ecoli = get_species_trna("ecoli")
        # CTG is high-usage, CTA is rare
        assert trna_ecoli["CTG"] > trna_ecoli["CTA"] * 5

    def test_trna_species_specific_diff(self):
        """tRNA abundance patterns differ between species.

        In human, GAG (Glu) is used more than in E. coli (due to GC preference),
        so the corresponding tRNA abundance ratios should differ.
        """
        trna_ecoli = get_species_trna("ecoli")
        trna_human = get_species_trna("human")
        # E. coli: GAA >> GAG (prefers A-ending)
        # Human: GAG close to GAA (GC preference makes GAG more common)
        ecoli_ratio = trna_ecoli["GAA"] / max(1, trna_ecoli["GAG"])
        human_ratio = trna_human["GAA"] / max(1, trna_human["GAG"])
        # E. coli's GAA/GAG ratio should be larger than human's
        assert ecoli_ratio > human_ratio, \
            f"ecoli GAA/GAG={ecoli_ratio} should > human={human_ratio}"

    def test_trna_start_codon_positive(self):
        """The start codon ATG has a cognate tRNA (Met-tRNA)."""
        for species in ("ecoli", "yeast", "human"):
            trna = get_species_trna(species)
            assert trna["ATG"] > 0, f"{species} ATG tRNA should be > 0"

    def test_trna_trp_positive(self):
        """Trp has only one codon TGG, with a cognate tRNA."""
        for species in ("ecoli", "yeast", "human"):
            trna = get_species_trna(species)
            assert trna["TGG"] > 0, f"{species} TGG tRNA should be > 0"
