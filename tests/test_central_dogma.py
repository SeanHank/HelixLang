"""Central dogma model tests: coupled transcription + translation + degradation.

Verifies (based on real paper parameters):
- Transcription: DNA → mRNA (U replaces T), elongation rate 50 nt/s (Proshkin 2010)
- Translation: mRNA → protein, base elongation rate 20 aa/s (Ingolia 2009)
- Codon-specific rates: determined by tRNA abundance (rare codons slow) (Dong 1996)
- mRNA degradation kinetics: steady state + exponential approach (Bernstein 2002)
- Coupled model: transcription and translation proceed simultaneously (Miller 1972)
- tRNA abundance data completeness: full coverage of 64 codons
- RBS detection: Shine-Dalgarno consensus sequence (Steitz & Jakes 1975)
- Stop codon efficiency: TAA > TGA > TAG (Poole 1995)

References:
- Proshkin 2010 Nature 458:507-511 (transcription elongation 50 nt/s)
- Ingolia 2009 Science 324:218-223 (translation elongation 20 aa/s)
- Bernstein 2002 J Bacteriol 184:6477-6486 (mRNA half-life 5 min)
- Dong 1996 J Mol Biol 260:649-663 (E. coli tRNA abundance)
- Miller 1972 Experiments in Molecular Genetics (transcription-translation coupling)
- Steitz & Jakes 1975 PNAS (RBS / Shine-Dalgarno)
- Poole 1995 RNA 1:1032-1043 (stop codon efficiency)
"""
from __future__ import annotations

import math

import pytest

from helixlang.central_dogma import (
    COUPLING_OFFSET_NT,
    E_COLI_POLY_A_TAIL_LENGTH,
    MAX_INITIATION_FREQUENCY_PER_MIN,
    MAX_TRNA_ABUNDANCE,
    MRNA_HALF_LIFE_MEDIAN_MIN,
    RBS_CONSENSUS,
    RBS_VARIANTS,
    STOP_CODON_EFFICIENCY,
    TRANSCRIPTION_ELONGATION_RATE_NT_PER_S,
    TRANSLATION_ELONGATION_RATE_AA_PER_S,
    # Data tables
    TRNA_ABUNDANCE,
    RibosomeState,
    calculate_mrna_level,
    coupled_transcription_translation,
    # Functions
    transcribe,
    translate,
)

# ============================================================================
# Transcription tests
# ============================================================================

class TestTranscription:
    """Verifies transcription matches Proshkin 2010 / Bernstein 2002 measured values."""

    def test_transcribe_produces_correct_mrna(self):
        """Produces correct mRNA from transcription (U replaces T).

        DNA: ATGGCTTAA → mRNA: AUGGCUUAA
        """
        dna = "ATGGCTTAA"
        transcript = transcribe(dna)
        # mRNA should contain no T, all replaced with U
        assert "T" not in transcript.sequence
        assert "U" in transcript.sequence
        # Full mRNA sequence should be DNA with T→U substitution
        assert transcript.sequence == "AUGGCUUAA"
        # CDS should contain start and stop
        assert transcript.cds == "AUGGCUUAA"
        assert transcript.utr5 == ""
        assert transcript.utr3 == ""

    def test_transcribe_separates_utr_and_cds(self):
        """Correctly splits 5'UTR / CDS / 3'UTR."""
        # 5'UTR = GGC, CDS = ATGCTGTAA, 3'UTR = CCG
        # ATG at position 3, scanned by codons: ATG(3-5) CTG(6-8) TAA(9-11)
        dna = "GGCATGCTGTAACCG"
        transcript = transcribe(dna)
        assert transcript.utr5 == "GGC"
        assert transcript.cds == "AUGCUGUAA"
        assert transcript.utr3 == "CCG"
        # Full sequence = utr5 + cds + utr3
        assert transcript.sequence == "GGCAUGCUGUAACCG"

    def test_transcribe_elongation_time(self):
        """Transcription elongation time: 1000 nt → ~20 s (50 nt/s, Proshkin 2010).

        Proshkin 2010 Nature 458:507-511: E. coli RNAP ~50 nt/s at 37°C.
        """
        # Build 1002 nt DNA (no terminator features, to avoid truncation)
        dna = "ATG" + "CTG" * 332 + "TAA"  # 3 + 996 + 3 = 1002 nt
        transcript = transcribe(dna)
        expected_time = len(dna) / TRANSCRIPTION_ELONGATION_RATE_NT_PER_S
        assert transcript.elongation_time_s == pytest.approx(expected_time)
        # 1002 nt / 50 nt/s = 20.04 s ≈ 20 s
        assert transcript.elongation_time_s == pytest.approx(20.0, abs=0.5)

    def test_transcribe_adds_poly_a_tail(self):
        """Adds a poly-A tail (E. coli ~15 nt, Mohanty & Kushner 2006)."""
        transcript = transcribe("ATGTAA")
        assert len(transcript.poly_a_tail) == E_COLI_POLY_A_TAIL_LENGTH
        assert transcript.poly_a_tail == "A" * E_COLI_POLY_A_TAIL_LENGTH
        assert E_COLI_POLY_A_TAIL_LENGTH == 15

    def test_transcribe_half_life(self):
        """mRNA half-life ~5 min (Bernstein 2002 J Bacteriol)."""
        transcript = transcribe("ATGTAA")
        assert transcript.half_life_minutes == MRNA_HALF_LIFE_MEDIAN_MIN
        assert MRNA_HALF_LIFE_MEDIAN_MIN == 5.0

    def test_transcribe_promoter_strength_affects_initiation(self):
        """Promoter strength determines transcription initiation frequency (Salgado 2013)."""
        strong = transcribe("ATGTAA", promoter_strength=1.0)
        weak = transcribe("ATGTAA", promoter_strength=0.1)
        assert strong.initiation_frequency_per_min > weak.initiation_frequency_per_min
        # Strong promoter (1.0) → maximum initiation frequency
        assert strong.initiation_frequency_per_min == pytest.approx(
            MAX_INITIATION_FREQUENCY_PER_MIN)
        # Weak promoter (0.1) → 1/10 initiation frequency
        assert weak.initiation_frequency_per_min == pytest.approx(
            MAX_INITIATION_FREQUENCY_PER_MIN * 0.1)

    def test_transcribe_transcription_factors(self):
        """Transcription factors affect effective promoter strength."""
        # Activator 2x
        t_activ = transcribe("ATGTAA", promoter_strength=0.5,
                             transcription_factors={"activator": 2.0})
        # 0.5 * 2.0 = 1.0 (capped)
        assert t_activ.promoter_strength == pytest.approx(1.0)
        # Repressor 0.1x
        t_repress = transcribe("ATGTAA", promoter_strength=1.0,
                               transcription_factors={"repressor": 0.1})
        assert t_repress.promoter_strength == pytest.approx(0.1)

    def test_transcribe_lowercase_input(self):
        """Lowercase DNA input is correctly normalized."""
        transcript = transcribe("atggcttaa")
        assert transcript.cds == "AUGGCUUAA"

    def test_transcribe_rho_independent_terminator(self):
        """Rho-independent terminator detection (GC stem-loop + poly-T tail)."""
        # Build terminator: stem(GGCC) + loop(AAA) + stem(GGCC, reverse complement) + poly-T(7)
        # GC content 100%, poly-T 7
        terminator = "GGCCAAA" + "GGCC" + "TTTTTTT"
        dna = "ATGCTGCTG" + terminator + "GGGCCC"  # sequence continues after the terminator
        transcript = transcribe(dna)
        assert transcript.has_terminator is True
        # Transcription should truncate after poly-T (no longer includes GGGCCC)
        assert "GGGCCC" not in transcript.sequence


# ============================================================================
# Translation tests
# ============================================================================

class TestTranslation:
    """Verifies translation matches Ingolia 2009 / Dong 1996 measured values."""

    def test_translate_produces_correct_protein(self):
        """Produces the correct protein sequence.

        DNA: ATGGCTGGTTAA → protein: MAG
        - ATG → M (Met)
        - GCT → A (Ala)
        - GGT → G (Gly)
        - TAA → stop
        """
        dna = "ATGGCTGGTTAA"
        transcript = transcribe(dna)
        result = translate(transcript)
        assert result.protein == "MAG"
        # Should record the stop codon
        assert result.stop_codon == "TAA"

    def test_translate_elongation_time(self):
        """Translation elongation time: 300 aa → ~15 s (20 aa/s, Ingolia 2009).

        Ingolia 2009 Science 324:218-223: E. coli ribosome ~20 aa/s.
        300 aa × (1/20 aa/s) = 15 s
        """
        # 1 ATG + 299 CTG + 1 TAA = 301 codons, protein 300 aa
        dna = "ATG" + "CTG" * 299 + "TAA"
        transcript = transcribe(dna)
        result = translate(transcript)
        # Protein length = 300
        assert len(result.protein) == 300
        # Elongation time ≈ 15 s (CTG is the most abundant codon, rate = 20 aa/s)
        # 1 ATG (rate 11.43) + 299 CTG (rate 20) ≈ 0.0875 + 14.95 = 15.04 s
        assert result.elongation_time == pytest.approx(15.0, abs=1.0)

    def test_rare_codon_reduces_translation_rate(self):
        """Rare codons reduce translation rate (Dong 1996).

        CTA (tRNA abundance 200) is much slower than CTG (tRNA abundance 3500).
        rate = base_rate × (abundance / max_abundance)
        - CTG: 20 × 3500/3500 = 20 aa/s
        - CTA: 20 × 200/3500 ≈ 1.14 aa/s
        """
        # All CTG (common Leu codon)
        dna_common = "ATG" + "CTG" * 10 + "TAA"
        # All CTA (rare Leu codon)
        dna_rare = "ATG" + "CTA" * 10 + "TAA"
        t_common = transcribe(dna_common)
        t_rare = transcribe(dna_rare)
        r_common = translate(t_common)
        r_rare = translate(t_rare)
        # The rare codon version should be much slower
        assert r_rare.elongation_time > r_common.elongation_time * 5
        # Single codon rate comparison (second codon, skipping ATG)
        # CTG rate = 20, CTA rate ≈ 1.14
        assert r_common.codon_rates[1] == pytest.approx(20.0, abs=0.1)
        assert r_rare.codon_rates[1] < 5.0  # CTA far below base rate
        # tRNA abundance verification
        assert TRNA_ABUNDANCE["CTG"] == 3500
        assert TRNA_ABUNDANCE["CTA"] == 200

    def test_translate_rbs_detection(self):
        """Translation initiation detects RBS (Shine-Dalgarno) sequences."""
        # AGGAGG (RBS) + TTTT (spacer 4) + ATG + GCT + TAA
        dna = "AGGAGGTTTTATGGCTTAA"
        transcript = transcribe(dna)
        result = translate(transcript)
        assert result.rbs_found is True
        assert result.rbs_sequence == "AGGAGG"
        # Control without RBS
        dna_no_rbs = "AAAATATGGCTTAA"  # no AGGAGG
        t_no_rbs = transcribe(dna_no_rbs)
        r_no_rbs = translate(t_no_rbs)
        assert r_no_rbs.rbs_found is False

    def test_stop_codon_efficiency_difference(self):
        """Stop codon efficiency difference: TAA > TGA > TAG (Poole 1995)."""
        assert STOP_CODON_EFFICIENCY["TAA"] > STOP_CODON_EFFICIENCY["TGA"]
        assert STOP_CODON_EFFICIENCY["TGA"] > STOP_CODON_EFFICIENCY["TAG"]
        # Translation result should record stop efficiency
        for stop, expected_eff in [("TAA", 0.99), ("TGA", 0.95), ("TAG", 0.90)]:
            dna = "ATGCTG" + stop  # M-L-stop
            transcript = transcribe(dna)
            result = translate(transcript)
            assert result.stop_codon == stop
            assert result.stop_efficiency == pytest.approx(expected_eff)

    def test_translate_empty_cds(self):
        """Translation returns an empty protein when there is no CDS."""
        # No ATG → CDS is empty
        transcript = transcribe("GGCGCCGGC")
        result = translate(transcript)
        assert result.protein == ""
        assert result.elongation_time == 0.0

    def test_translate_protein_correct_amino_acids(self):
        """Translation correctly maps amino acids using ECOLI_CODON_USAGE."""
        # Codons for various amino acids
        # ATG=M, GCT=A, GGT=G, TTT=F, TGG=W, TAA=stop
        dna = "ATGGCTGGTTTTTGGTAA"
        transcript = transcribe(dna)
        result = translate(transcript)
        assert result.protein == "MAGFW"
        # codon_rates length = protein length
        assert len(result.codon_rates) == len(result.protein)

    def test_ribosome_state_dataclass(self):
        """RibosomeState dataclass basic functionality."""
        ribo = RibosomeState(position=5, peptidyl="MAG", charged_trna="G")
        assert ribo.position == 5
        assert ribo.peptidyl == "MAG"
        assert ribo.charged_trna == "G"
        # Default values
        ribo_default = RibosomeState()
        assert ribo_default.position == 0
        assert ribo_default.peptidyl == ""
        assert ribo_default.charged_trna is None


# ============================================================================
# mRNA degradation kinetics tests
# ============================================================================

class TestMRNADegradation:
    """Verifies mRNA degradation kinetics match the Bernstein 2002 model."""

    def test_mrna_starts_at_zero(self):
        """mRNA concentration is 0 at t=0."""
        transcript = transcribe("ATGTAA", promoter_strength=1.0)
        assert calculate_mrna_level(transcript, 0.0) == 0.0

    def test_mrna_approaches_steady_state(self):
        """mRNA approaches the steady-state concentration after a long time.

        Steady-state [mRNA]_ss = synthesis_rate / degradation_rate
        """
        transcript = transcribe("ATGTAA", promoter_strength=1.0)
        degradation_rate = math.log(2) / transcript.half_life_minutes
        mrna_ss = (transcript.initiation_frequency_per_min
                   / degradation_rate)
        # After 1000 min should approach steady state (far > half-life 5 min)
        mrna_long = calculate_mrna_level(transcript, 1000.0, degradation_rate)
        assert mrna_long == pytest.approx(mrna_ss, rel=0.01)
        assert mrna_long < mrna_ss * 1.01  # no more than steady state

    def test_mrna_half_life_reaches_50_percent(self):
        """At t = half-life, mRNA reaches 50% of steady state.

        mrna(t) = mrna_ss × (1 - exp(-k × t))
        t = t_half → k × t = ln(2) → 1 - exp(-ln(2)) = 0.5
        """
        transcript = transcribe("ATGTAA", promoter_strength=1.0)
        degradation_rate = math.log(2) / transcript.half_life_minutes
        mrna_ss = (transcript.initiation_frequency_per_min
                   / degradation_rate)
        mrna_at_half = calculate_mrna_level(
            transcript, transcript.half_life_minutes, degradation_rate)
        # Should be 50% of steady state
        assert mrna_at_half == pytest.approx(mrna_ss * 0.5, rel=0.01)

    def test_mrna_monotonic_increase(self):
        """mRNA concentration increases monotonically (toward steady state)."""
        transcript = transcribe("ATGTAA", promoter_strength=1.0)
        times = [0, 1, 2, 5, 10, 20, 50]
        levels = [calculate_mrna_level(transcript, t) for t in times]
        for i in range(1, len(levels)):
            assert levels[i] >= levels[i - 1]

    def test_mrna_promoter_strength_scales_steady_state(self):
        """Promoter strength scales the steady-state concentration proportionally."""
        t_strong = transcribe("ATGTAA", promoter_strength=1.0)
        t_weak = transcribe("ATGTAA", promoter_strength=0.5)
        # After a long time the steady-state concentration should be proportional to promoter strength
        mrna_strong = calculate_mrna_level(t_strong, 1000.0)
        mrna_weak = calculate_mrna_level(t_weak, 1000.0)
        assert mrna_strong == pytest.approx(mrna_weak * 2.0, rel=0.01)


# ============================================================================
# Coupled model tests
# ============================================================================

class TestCoupledModel:
    """Verifies the coupled transcription-translation model matches Miller 1972 observations."""

    def test_coupled_model_returns_required_keys(self):
        """Coupled model returns a dict with required keys."""
        dna = "ATG" + "CTG" * 10 + "TAA"
        result = coupled_transcription_translation(dna, promoter_strength=1.0)
        # The 4 keys required by the task
        assert "transcript" in result
        assert "protein" in result
        assert "mrna_level" in result
        assert "time_course" in result
        # Additional useful keys
        assert "mrna_steady_state" in result
        assert "transcription_time_s" in result
        assert "translation_time_s" in result
        assert "coupling_offset_s" in result

    def test_coupled_model_protein_correct(self):
        """Coupled model produces the correct protein sequence."""
        dna = "ATGCTGCTGTAA"  # M-L-L-stop
        result = coupled_transcription_translation(dna)
        assert result["protein"] == "MLL"

    def test_coupled_model_time_course(self):
        """Coupled model time course sampling is correct."""
        dna = "ATG" + "CTG" * 10 + "TAA"
        result = coupled_transcription_translation(
            dna, time_course_min=30.0, time_step_min=5.0)
        tc = result["time_course"]
        # Should have 7 sample points (0, 5, 10, 15, 20, 25, 30)
        assert len(tc) == 7
        assert tc[0].time_min == 0.0
        assert tc[-1].time_min == pytest.approx(30.0)
        # First point mRNA = 0
        assert tc[0].mrna_level == 0.0
        # mRNA increases monotonically
        for i in range(1, len(tc)):
            assert tc[i].mrna_level >= tc[i - 1].mrna_level
        # Transcription progress from 0 to 1
        assert tc[0].transcription_progress == 0.0
        assert tc[-1].transcription_progress == pytest.approx(1.0)

    def test_coupled_model_translation_starts_after_transcription(self):
        """Translation starts with a delay after transcription begins (coupling offset ~0.6 s, Miller 1972)."""
        dna = "ATG" + "CTG" * 10 + "TAA"
        result = coupled_transcription_translation(dna)
        # Coupling offset = 30 nt / 50 nt/s = 0.6 s
        assert result["coupling_offset_s"] == pytest.approx(0.6)
        # Transcription time > translation delay
        assert result["transcription_time_s"] > 0
        # DNA total length = 3 (ATG) + 30 (CTG*10) + 3 (TAA) = 36 nt
        # Transcription time = 36 nt / 50 nt/s = 0.72 s
        expected_tx_time = 36 / TRANSCRIPTION_ELONGATION_RATE_NT_PER_S
        assert result["transcription_time_s"] == pytest.approx(expected_tx_time)

    def test_coupled_model_time_course_point_fields(self):
        """TimeCoursePoint contains all required fields."""
        dna = "ATGTAA"
        result = coupled_transcription_translation(dna)
        point = result["time_course"][0]
        assert hasattr(point, "time_min")
        assert hasattr(point, "mrna_level")
        assert hasattr(point, "transcription_progress")
        assert hasattr(point, "translation_progress")
        assert hasattr(point, "protein_accumulated")


# ============================================================================
# tRNA abundance data completeness tests
# ============================================================================

class TestTRNAAbundance:
    """Verifies E. coli tRNA abundance data completeness (Dong 1996)."""

    def test_trna_abundance_covers_64_codons(self):
        """The tRNA abundance dict covers all 64 codons."""
        bases = "ACGT"
        all_codons = {a + b + c for a in bases for b in bases for c in bases}
        assert set(TRNA_ABUNDANCE.keys()) == all_codons
        assert len(TRNA_ABUNDANCE) == 64

    def test_stop_codons_have_zero_abundance(self):
        """Stop codons have no cognate tRNA (abundance = 0)."""
        for stop in ("TAA", "TAG", "TGA"):
            assert TRNA_ABUNDANCE[stop] == 0, f"{stop} should have 0 tRNA"

    def test_rare_codons_have_low_abundance(self):
        """Rare codons have low tRNA abundance (Dong 1996 Table 2).

        CTA (Leu-UAG), AGA (Arg-UCU), AGG (Arg-CCU), ATA (Ile-UAU) are all rare.
        """
        rare_codons = {"CTA": 200, "AGA": 150, "AGG": 100, "ATA": 200}
        for codon, expected in rare_codons.items():
            assert TRNA_ABUNDANCE[codon] == expected
            assert TRNA_ABUNDANCE[codon] < 500  # rare threshold

    def test_common_codons_have_high_abundance(self):
        """High-frequency codons have high tRNA abundance."""
        # CTG (Leu-CAG) is one of the most common codons in E. coli
        assert TRNA_ABUNDANCE["CTG"] == 3500
        assert TRNA_ABUNDANCE["CTG"] == MAX_TRNA_ABUNDANCE
        # Other high-frequency
        assert TRNA_ABUNDANCE["GAA"] == 3000  # Glu-UUC
        assert TRNA_ABUNDANCE["AAA"] == 3000  # Lys-UUU
        assert TRNA_ABUNDANCE["GGC"] == 3200  # Gly-GCC
        assert TRNA_ABUNDANCE["GGT"] == 3200  # Gly-GCC (wobble)

    def test_max_trna_abundance_is_ctg(self):
        """Maximum tRNA abundance corresponds to CTG (E. coli optimal Leu codon)."""
        assert MAX_TRNA_ABUNDANCE == 3500
        assert MAX_TRNA_ABUNDANCE == max(TRNA_ABUNDANCE.values())

    def test_trna_abundance_all_non_negative(self):
        """All tRNA abundance values are non-negative integers."""
        for codon, abundance in TRNA_ABUNDANCE.items():
            assert isinstance(abundance, int)
            assert abundance >= 0, f"{codon} has negative abundance"

    def test_synonymous_codons_share_trna(self):
        """Synonymous codons (wobble pairing) share tRNA, with the same abundance.

        For example, Phe's UUU/UUC are both read by tRNA-Phe-GAA.
        """
        # Phe: TTT and TTC share tRNA-Phe-GAA
        assert TRNA_ABUNDANCE["TTT"] == TRNA_ABUNDANCE["TTC"]
        # Tyr: TAT and TAC share tRNA-Tyr-GUA
        assert TRNA_ABUNDANCE["TAT"] == TRNA_ABUNDANCE["TAC"]
        # Cys: TGT and TGC share tRNA-Cys-GCA
        assert TRNA_ABUNDANCE["TGT"] == TRNA_ABUNDANCE["TGC"]
        # His: CAT and CAC share tRNA-His-GUG
        assert TRNA_ABUNDANCE["CAT"] == TRNA_ABUNDANCE["CAC"]


# ============================================================================
# RBS detection tests
# ============================================================================

class TestRBSDetection:
    """Verifies Shine-Dalgarno (RBS) sequence detection (Steitz & Jakes 1975)."""

    def test_rbs_consensus_sequence(self):
        """RBS consensus sequence is AGGAGG."""
        assert RBS_CONSENSUS == "AGGAGG"

    def test_rbs_full_consensus_detected(self):
        """The full AGGAGG sequence is detected."""
        dna = "AGGAGGTTTTATGGCTTAA"
        transcript = transcribe(dna)
        result = translate(transcript)
        assert result.rbs_found is True
        assert result.rbs_sequence == "AGGAGG"

    def test_rbs_variant_detected(self):
        """RBS variants (short sequences) are also detected."""
        # GGAGG is a common variant
        dna = "GGAGGTTTTATGGCTTAA"
        transcript = transcribe(dna)
        result = translate(transcript)
        assert result.rbs_found is True
        # Should match the longest variant
        assert result.rbs_sequence in RBS_VARIANTS

    def test_rbs_no_match(self):
        """Returns not-found when there is no RBS sequence."""
        # AAAA does not contain AGGAGG or a variant
        dna = "AAAATTTTATGGCTTAA"
        transcript = transcribe(dna)
        result = translate(transcript)
        assert result.rbs_found is False
        assert result.rbs_sequence == ""

    def test_rbs_variants_ordered_by_length(self):
        """RBS variants are ordered by descending length (longer sequences matched first)."""
        lengths = [len(v) for v in RBS_VARIANTS]
        assert lengths == sorted(lengths, reverse=True)
        # AGGAGG should be first
        assert RBS_VARIANTS[0] == "AGGAGG"

    def test_rbs_spacing_realistic(self):
        """RBS-to-start-codon spacing is in the 5-13 nt range (Steitz & Jakes 1975)."""
        # AGGAGG + 5 nt spacer + ATG...
        dna = "AGGAGGAAAATATGGCTTAA"  # spacer = AAAAT (5 nt)
        transcript = transcribe(dna)
        result = translate(transcript)
        assert result.rbs_found is True


# ============================================================================
# Stop codon efficiency tests
# ============================================================================

class TestStopCodonEfficiency:
    """Verifies stop codon efficiency differences (Poole 1995, Major 1996)."""

    def test_efficiency_ordering(self):
        """Stop efficiency TAA > TGA > TAG (Poole 1995)."""
        assert STOP_CODON_EFFICIENCY["TAA"] == 0.99
        assert STOP_CODON_EFFICIENCY["TGA"] == 0.95
        assert STOP_CODON_EFFICIENCY["TAG"] == 0.90
        assert STOP_CODON_EFFICIENCY["TAA"] > STOP_CODON_EFFICIENCY["TGA"]
        assert STOP_CODON_EFFICIENCY["TGA"] > STOP_CODON_EFFICIENCY["TAG"]

    def test_all_stop_codons_in_efficiency_table(self):
        """All three stop codons are in the efficiency table."""
        for stop in ("TAA", "TAG", "TGA"):
            assert stop in STOP_CODON_EFFICIENCY
            eff = STOP_CODON_EFFICIENCY[stop]
            assert 0.0 < eff <= 1.0

    def test_translation_records_stop_efficiency(self):
        """Translation result records the stop codon efficiency."""
        for stop in ("TAA", "TGA", "TAG"):
            dna = "ATGCTG" + stop  # M-L-stop
            transcript = transcribe(dna)
            result = translate(transcript)
            assert result.stop_codon == stop
            assert result.stop_efficiency == STOP_CODON_EFFICIENCY[stop]

    def test_readthrough_flag(self):
        """Flags possible readthrough when efficiency < 1."""
        dna = "ATGCTGTAA"
        transcript = transcribe(dna)
        result = translate(transcript)
        # TAA efficiency 0.99 < 1, flags possible readthrough
        assert result.readthrough is True
        assert result.stop_efficiency < 1.0


# ============================================================================
# Constants verification
# ============================================================================

class TestConstants:
    """Verifies real biological parameter constants match literature values."""

    def test_transcription_rate(self):
        """Transcription elongation rate 50 nt/s (Proshkin 2010)."""
        assert TRANSCRIPTION_ELONGATION_RATE_NT_PER_S == 50.0

    def test_translation_rate(self):
        """Translation elongation rate 20 aa/s (Ingolia 2009)."""
        assert TRANSLATION_ELONGATION_RATE_AA_PER_S == 20.0

    def test_mrna_half_life(self):
        """mRNA half-life 5 min (Bernstein 2002)."""
        assert MRNA_HALF_LIFE_MEDIAN_MIN == 5.0

    def test_poly_a_tail_length(self):
        """poly-A tail length 15 nt (Mohanty & Kushner 2006)."""
        assert E_COLI_POLY_A_TAIL_LENGTH == 15

    def test_coupling_offset(self):
        """Coupling delay 30 nt (Miller 1972)."""
        assert COUPLING_OFFSET_NT == 30

    def test_max_initiation_frequency(self):
        """Maximum initiation frequency 10 mRNA/min."""
        assert MAX_INITIATION_FREQUENCY_PER_MIN == 10.0
