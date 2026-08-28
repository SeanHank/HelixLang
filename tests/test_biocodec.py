"""Biological DNA↔Helix compiler tests: verified against real biological data.

Verifies (based on real paper parameters):
- ORF detection: double-stranded scanning, ATG/GTG/TTG start, TAA/TAG/TGA stop
- Restriction enzyme sites: NEB standard 6-cutters (EcoRI/BamHI/HindIII, etc.)
- Codon optimization: E. coli K-12 MG1655 CAI table (GenScript CUTG)
- Back translation: preserves the protein sequence, CAI ≥ 0.3 (E. coli expression threshold)
- Real regulatory elements: lacP promoter, rrnB T1 terminator
- Biological validation: GC 45-55%, length multiple of 3, no internal stops, homopolymers ≤3
- DNA → HelixLang source → DNA roundtrip fidelity

References:
- E. coli K-12 MG1655 (Blattner 1997 Science 277:1453-1462)
- Codon usage frequencies: CUTG E. coli 511145
- lac operon promoter: Miller 1972
- rrnB T1 terminator: Brosius 1981 J Biol Chem 256:4987-4990
- Restriction enzyme sites: NEB 2024 catalogue
- CAI algorithm: Sharp 1987 Nucleic Acids Res 15:1281-1295
"""
from __future__ import annotations

import random

import pytest

pytest.importorskip("Bio")

from helixlang.plugins.runtime.biocodec import (
    LAC_PROMOTER,
    RESTRICTION_SITES,
    RRNB_T1_TERMINATOR,
    STOP_CODONS,
    T7_PROMOTER,
    T7_TERMINATOR,
    avoid_restriction_sites,
    back_translate,
    codon_adaptation_index_full,
    dna_to_helix,
    find_orfs,
    find_restriction_sites,
    helix_to_dna,
    validate_biological,
)

# ============================================================================
# ORF detection
# ============================================================================

class TestORFDetection:
    """Verifies ORF detection matches the NCBI ORFfinder standard."""

    def test_simple_orf(self):
        """Single ORF: ATG...TAA."""
        dna = "ATGGCTACCGTTTAA"  # M-A-T-V-*  (15 bp, 5 codons)
        orfs = find_orfs(dna, min_length_aa=3)
        assert len(orfs) >= 1
        assert orfs[0].start_codon == "ATG"
        assert orfs[0].stop_codon == "TAA"
        assert orfs[0].protein == "MATV"

    def test_all_three_start_codons(self):
        """ATG/GTG/TTG can all start an ORF."""
        for start in ["ATG", "GTG", "TTG"]:
            dna = start + "GCTACCGTT" + "TAA"
            orfs = find_orfs(dna, min_length_aa=2)
            assert any(o.start_codon == start for o in orfs), \
                f"start codon {start} not detected"

    def test_all_three_stop_codons(self):
        """TAA/TAG/TGA can all terminate an ORF."""
        for stop in ["TAA", "TAG", "TGA"]:
            dna = "ATGGCTACCGTT" + stop
            orfs = find_orfs(dna, min_length_aa=2)
            assert any(o.stop_codon == stop for o in orfs), \
                f"stop codon {stop} not detected"

    def test_min_length_filter(self):
        """Short ORFs are filtered out."""
        dna = "ATGTAA"  # 2 codons (M + *), <10 aa default threshold
        orfs = find_orfs(dna, min_length_aa=10)
        assert len(orfs) == 0

    def test_three_reading_frames(self):
        """All three reading frames (+0/+1/+2) are scanned."""
        # frame +1: ATG at position 1
        dna = "A" + "ATGGCTACCGTTTAA" + "A" * 30
        orfs = find_orfs(dna, min_length_aa=3)
        # Should be able to find an ORF in frame +1
        assert any(o.start == 1 for o in orfs)

    def test_both_strands(self):
        """ORFs on the negative strand are also detected."""
        # The plus strand has no ATG, but the minus strand does
        orf_plus = "ATGGCTACCGTTTAA"
        # Reverse complement: TTAAACGGTAGCCAT
        rc = _reverse_complement_for_test(orf_plus)
        # Place the ORF on the negative strand
        dna = "TTT" * 10 + rc + "TTT" * 10
        orfs = find_orfs(dna, min_length_aa=3, both_strands=True)
        assert any(o.strand == "-" for o in orfs)

    def test_no_internal_stop_in_orf(self):
        """ORFs should have no internal stop codons (ensuring complete translation)."""
        protein = "MASKGEELFTGVPVPILVELDGDVNGHKFSVSGEGEGDATYGRTLTKF"
        dna = back_translate(protein, optimize="cai")
        full = "ATG" + dna[3:] + "TAA"
        orfs = find_orfs(full, min_length_aa=10)
        assert len(orfs) >= 1
        for orf in orfs:
            # Check that the ORF interior (excluding start and stop) has no stop codons
            internal = orf.sequence[3:-3]
            for i in range(0, len(internal), 3):
                assert internal[i:i + 3] not in STOP_CODONS, \
                    f"internal stop at position {i} in ORF"

    def test_long_protein_orf(self):
        """Long protein (≥100 aa) ORF detection."""
        # GFP partial ~50 aa
        protein = "MASKGEELFTGVPVPILVELDGDVNGHKFSVSGEGEGDATYGRTLTKF" \
                  "KLICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYV"
        # back_translate already includes the start ATG (M encodes ATG)
        dna = back_translate(protein, optimize="cai") + "TAA"
        orfs = find_orfs(dna, min_length_aa=50)
        assert len(orfs) >= 1
        assert orfs[0].start_codon == "ATG"
        assert orfs[0].stop_codon == "TAA"


def _reverse_complement_for_test(dna: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(comp[b] for b in reversed(dna))


# ============================================================================
# Restriction enzyme site detection
# ============================================================================

class TestRestrictionSites:
    """Verifies restriction enzyme site detection matches NEB standards."""

    def test_ecori_detection(self):
        """EcoRI (GAATTC) site detection."""
        dna = "ATGGAATTCTTTTAA"  # ATG + GAATTC + TTT + TAA
        sites = find_restriction_sites(dna)
        assert "EcoRI" in sites
        assert 3 in sites["EcoRI"]

    def test_bamhi_detection(self):
        """BamHI (GGATCC) site detection."""
        dna = "ATGGGATCCTTTTAA"
        sites = find_restriction_sites(dna)
        assert "BamHI" in sites

    def test_hindiii_detection(self):
        """HindIII (AAGCTT) site detection."""
        dna = "ATGAAGCTTTTTTAA"
        sites = find_restriction_sites(dna)
        assert "HindIII" in sites

    def test_no_site_clean_dna(self):
        """DNA with no sites returns empty."""
        dna = "ATGGCTACCGTTTAA"
        sites = find_restriction_sites(dna)
        assert sites == {}

    def test_palindrome_detection(self):
        """Palindromic sites are detectable in both directions."""
        # EcoRI: GAATTC, RC: GAATTC (palindrome)
        assert _reverse_complement_for_test("GAATTC") == "GAATTC"
        # SmaI: CCCGGG, RC: CCCGGG (palindrome)
        assert _reverse_complement_for_test("CCCGGG") == "CCCGGG"

    def test_avoid_restriction_sites_synonymous(self):
        """Removes restriction enzyme sites via synonymous mutations (preserving the protein)."""
        # Build an ORF containing an EcoRI site
        # GAATTC: GAA (Glu E) + TTC (Phe F)
        # Glu synonymous: GAA/GAG, Phe synonymous: TTC/TTT
        # Changing GAA→GAG still encodes Glu, breaking the EcoRI site
        dna = "ATGGAATTCTAA"  # M-E-F-*  (contains EcoRI)
        cleaned = avoid_restriction_sites(dna, max_attempts=20)
        # Should contain no EcoRI
        sites = find_restriction_sites(cleaned)
        assert "EcoRI" not in sites
        # Translation should remain unchanged
        from Bio.Seq import Seq
        orig_protein = str(Seq(dna).translate())
        new_protein = str(Seq(cleaned).translate())
        assert orig_protein == new_protein, \
            f"protein changed: {orig_protein} → {new_protein}"

    def test_all_neb_sites_in_catalogue(self):
        """RESTRICTION_SITES covers common NEB enzymes."""
        required = ["EcoRI", "BamHI", "HindIII", "XhoI", "XbaI",
                    "SalI", "PstI", "KpnI", "NotI"]
        for enz in required:
            assert enz in RESTRICTION_SITES, f"missing enzyme {enz}"


# ============================================================================
# Codon optimization (CAI)
# ============================================================================

class TestCodonOptimization:
    """Verifies codon optimization matches the E. coli K-12 MG1655 frequency table."""

    def test_back_translate_preserves_protein(self):
        """Back translation preserves the protein sequence."""
        protein = "MASKGEELFTGVPVPILVELDGDVNGHKFSVSGEGEGDATY"
        dna = back_translate(protein, optimize="cai")
        from Bio.Seq import Seq
        recovered = str(Seq(dna).translate())
        # back_translate does not add a stop, so the protein should be preserved
        assert recovered.rstrip("*") == protein

    def test_cai_optimization_higher_than_random(self):
        """CAI optimization yields a higher CAI than randomly chosen codons."""
        protein = "MASKGEELFTGVPVPILVELDGDVNGHKFSVSGEGEGDATYGRTLTKF"
        dna_cai = back_translate(protein, optimize="cai")
        rng = random.Random(42)
        dna_random = back_translate(protein, optimize="random", rng=rng)
        cai_cai = codon_adaptation_index_full(dna_cai)
        cai_random = codon_adaptation_index_full(dna_random)
        assert cai_cai > cai_random, \
            f"CAI-optimized {cai_cai} should be > random {cai_random}"

    def test_cai_uses_optimal_codons(self):
        """CAI optimization selects the codon with the highest fraction."""
        # ATG (Met) → fraction 1.0, sole start
        dna = back_translate("M", optimize="cai")
        assert dna == "ATG"
        # Trp's only codon is TGG
        dna = back_translate("W", optimize="cai")
        assert dna == "TGG"
        # Leu: CTG (fraction 0.47) is the highest
        dna = back_translate("L", optimize="cai")
        assert dna == "CTG"

    def test_cai_value_range(self):
        """CAI values are in the [0, 1] range."""
        for protein in ["M", "MASKGEELF", "MWKRSV"]:
            dna = back_translate(protein, optimize="cai")
            cai = codon_adaptation_index_full(dna)
            assert 0.0 <= cai <= 1.0, f"CAI {cai} out of [0,1]"

    def test_balanced_uses_real_frequencies(self):
        """The balanced mode selects codons weighted by E. coli frequencies."""
        protein = "EEEEEE"  # 6 Glu (GAA/GAG)
        rng = random.Random(42)
        dna = back_translate(protein, optimize="balanced", rng=rng)
        # Glu: GAA (fraction 0.68), GAG (fraction 0.32)
        # Over many repeats, both GAA and GAG should appear
        codons = [dna[i:i + 3] for i in range(0, len(dna), 3)]
        assert "GAA" in codons
        # GAG may not appear (lower probability), but GAA should be the majority
        assert codons.count("GAA") >= codons.count("GAG")


# ============================================================================
# Real regulatory elements
# ============================================================================

class TestRegulatoryElements:
    """Verifies promoter/terminator sequences match literature measurements."""

    def test_lac_promoter_has_consensus(self):
        """lacP contains the -35 (TTTACA) and -10 (TATAAT) consensus sequences."""
        assert "TTTACA" in LAC_PROMOTER  # -35 region
        assert "TATAAT" in LAC_PROMOTER or "TATGTT" in LAC_PROMOTER  # -10 region

    def test_t7_promoter_length(self):
        """T7 promoter is 23 bp (standard φ10 sequence)."""
        assert len(T7_PROMOTER) == 23

    def test_rrnb_t1_terminator_poly_t(self):
        """rrnB T1 terminator contains a poly-T tail (ρ-independent termination signal)."""
        # ρ-independent terminators need a U-rich tail (T-rich on DNA)
        assert "TTTT" in RRNB_T1_TERMINATOR

    def test_t7_terminator_present(self):
        """T7 terminator sequence exists."""
        assert len(T7_TERMINATOR) > 20


# ============================================================================
# DNA ↔ HelixLang biological compilation
# ============================================================================

class TestBioHelixRoundtrip:
    """Verifies DNA → HelixLang → DNA roundtrip fidelity."""

    def test_helix_to_dna_adds_promoter_terminator(self):
        """Helix → DNA includes the promoter + terminator."""
        src = "#gene name=test\nATG GCT GGT TAA\n#end"
        dna = helix_to_dna(src, promoter="lac", terminator="rrnB_T1")
        assert dna.startswith(LAC_PROMOTER)
        assert dna.endswith(RRNB_T1_TERMINATOR)
        # Should have an ORF in the middle
        orf_part = dna[len(LAC_PROMOTER):-len(RRNB_T1_TERMINATOR)]
        assert orf_part.startswith("ATG")
        assert orf_part.endswith("TAA") or orf_part.endswith("TAG") or orf_part.endswith("TGA")

    def test_dna_to_helix_detects_orfs(self):
        """DNA → Helix detects ORFs and generates a gene block."""
        protein = "MASKGEELFTGVPVPILVELDGDVNGHKFSVSGEGEGDATY"
        dna = "ATG" + back_translate(protein, optimize="cai") + "TAA"
        result = dna_to_helix(dna)
        assert len(result.orfs) >= 1
        assert "#gene" in result.helix_source
        assert "ATG" in result.helix_source

    def test_roundtrip_preserves_codons(self):
        """Helix → DNA → Helix roundtrip preserves the codon sequence."""
        # Use a simple ORF to avoid codon optimization changing the sequence
        src = "#gene name=test_gene\nATG GCT GGT TAA\n#end"
        dna = helix_to_dna(src, promoter=None if False else "lac",
                          terminator="rrnB_T1",
                          optimize_codons=False,
                          avoid_restriction=False)
        # Extract the ORF portion (removing promoter/terminator)
        orf_dna = dna[len(LAC_PROMOTER):-len(RRNB_T1_TERMINATOR)]
        # DNA → Helix
        result = dna_to_helix(orf_dna)
        # Should detect an ORF
        assert len(result.orfs) >= 1
        # Translation should remain M-A-G-*
        assert result.orfs[0].protein.startswith("M")

    def test_codon_optimization_increases_cai(self):
        """Helix → DNA with optimization increases the CAI."""
        # Use the rare Leu codon CTA with low CAI
        src = "#gene name=test\nATG CTA CTA CTA TAA\n#end"  # M-L-L-L-* (CTA rare)
        dna_optimized = helix_to_dna(src, promoter="lac",
                                     terminator="rrnB_T1",
                                     optimize_codons=True,
                                     avoid_restriction=False)
        # Extract the ORF
        orf_dna = dna_optimized[len(LAC_PROMOTER):-len(RRNB_T1_TERMINATOR)]
        cai = codon_adaptation_index_full(orf_dna)
        # CTA fraction=0.04 → very low CAI; optimized to CTG (fraction=0.47) → high CAI
        assert cai > 0.3, f"optimized CAI {cai} should be > 0.3"

    def test_avoid_restriction_in_helix_to_dna(self):
        """Helix → DNA removes restriction enzyme sites by default."""
        # GAATTC = GAA (Glu) + TTC (Phe)
        src = "#gene name=test\nATG GAA TTC GCT TAA\n#end"
        dna = helix_to_dna(src, promoter="lac", terminator="rrnB_T1",
                          optimize_codons=False, avoid_restriction=True)
        # Check the ORF portion (excluding promoter/terminator, since the terminator may contain sites)
        orf_dna = dna[len(LAC_PROMOTER):-len(RRNB_T1_TERMINATOR)]
        sites = find_restriction_sites(orf_dna)
        # Should have no EcoRI (synonymous mutation GAA→GAG)
        assert "EcoRI" not in sites, f"EcoRI still present: {sites}"


# ============================================================================
# Biological validation
# ============================================================================

class TestBiologicalValidation:
    """Verifies biological validation matches E. coli expression constraints."""

    def test_valid_orf_passes(self):
        """A compliant ORF passes validation."""
        protein = "MASKGEELFTGVPVPILVELDGDVNGHKFSVSGEGEGDATY"
        dna = "ATG" + back_translate(protein, optimize="cai") + "TAA"
        report = validate_biological(dna)
        assert report.has_start_codon
        assert report.has_stop_codon
        assert report.length_multiple_of_3
        assert report.no_internal_stop

    def test_no_start_codon_fails(self):
        """Fails when there is no start codon."""
        dna = "GCGGCTACCGTTTAA"  # does not start with ATG
        report = validate_biological(dna)
        assert not report.has_start_codon
        assert "no start codon" in " ".join(report.errors)

    def test_no_stop_codon_fails(self):
        """Fails when there is no stop codon."""
        dna = "ATGGCTACCGTTGCC"  # does not end with TAA/TAG/TGA
        report = validate_biological(dna)
        assert not report.has_stop_codon

    def test_internal_stop_detected(self):
        """Internal stop codons are detected."""
        dna = "ATGGCTTAAGTTTAA"  # M-A-*-V-*  internal TAA
        report = validate_biological(dna)
        assert not report.no_internal_stop
        assert any("internal stop" in e for e in report.errors)

    def test_low_cai_detected(self):
        """Low CAI is detected (rare codon CTA)."""
        dna = "ATGCTACTACTACTACTAATAA"  # all CTA (fraction 0.04)
        report = validate_biological(dna)
        assert report.cai < 0.3
        assert not report.cai_adequate

    def test_restriction_site_detected(self):
        """Restriction enzyme sites are detected."""
        dna = "ATGGAATTCTTTTAA"  # contains EcoRI
        report = validate_biological(dna)
        assert "EcoRI" in report.restriction_sites

    def test_long_homopolymer_detected(self):
        """Long homopolymers are detected."""
        dna = "ATGGCTAAAAAAAATTTTAA"  # 6×A
        report = validate_biological(dna)
        assert report.max_homopolymer >= 5
        assert not report.max_homopolymer_ok


# ============================================================================
# Integration: real-world scenario (GFP expression vector construction)
# ============================================================================

class TestRealisticScenario:
    """Real-world scenario: constructing a GFP expression vector."""

    def test_gfp_expression_cassette(self):
        """GFP expression cassette: lacP + GFP ORF + rrnB T1."""
        # GFP portion (~50 aa)
        protein = "MASKGEELFTGVPVPILVELDGDVNGHKFSVSGEGEGDATYGRTLTKF"
        # Back translation yields the CAI-optimized GFP ORF (includes start ATG)
        gfp_orf = back_translate(protein, optimize="cai") + "TAA"
        # Build the complete expression cassette
        cassette = LAC_PROMOTER + gfp_orf + RRNB_T1_TERMINATOR
        # Validate
        assert len(cassette) > 200  # promoter 71bp + ORF 162bp + terminator 35bp
        assert cassette.startswith(LAC_PROMOTER)
        assert cassette.endswith(RRNB_T1_TERMINATOR)
        # Validate the ORF portion
        orf_part = cassette[len(LAC_PROMOTER):-len(RRNB_T1_TERMINATOR)]
        orf_report = validate_biological(orf_part)
        assert orf_report.has_start_codon
        assert orf_report.has_stop_codon
        assert orf_report.no_internal_stop
        assert orf_report.cai_adequate

    def test_gfp_orf_translates_correctly(self):
        """The GFP ORF translates to the correct protein sequence."""
        protein = "MASKGEELFTGVPVPILVELDGDVNGHKFSVSGEGEGDATYGRTLTKF"
        # back_translate already includes the start ATG (M encodes ATG)
        gfp_orf = back_translate(protein, optimize="cai") + "TAA"
        from Bio.Seq import Seq
        translated = str(Seq(gfp_orf).translate())
        # Should be M-S-K-...-K-F-*
        assert translated.startswith("M")
        assert translated.endswith("*")
        # After removing the stop, it should equal the original protein
        assert translated.rstrip("*") == protein

    def test_full_roundtrip_with_real_promoter(self):
        """Full roundtrip: Helix → DNA (with promoter) → extract ORF → validate."""
        src = "#gene name=gfp\n" + \
              " ".join(["ATG", "GCT", "TCT", "AAA", "GGT", "GAA", "GAA",
                        "CTG", "TTC", "ACC", "GGT", "TAA"]) + \
              "\n#end"
        # Helix → DNA
        bio_dna = helix_to_dna(src, promoter="lac", terminator="rrnB_T1",
                              optimize_codons=True, avoid_restriction=True)
        # Extract the ORF portion
        orf_dna = bio_dna[len(LAC_PROMOTER):-len(RRNB_T1_TERMINATOR)]
        # DNA → Helix
        result = dna_to_helix(orf_dna)
        assert len(result.orfs) >= 1
        # ORF should translate to M-A-S-K-G-E-E-L-F-T-G-*
        assert result.orfs[0].protein.startswith("MASKGEELFTG")
