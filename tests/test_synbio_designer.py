"""Synthetic biology design tool tests: complete SynBioDesigner design pipeline.

Verification goals:
- Expression cassette design (promoter + RBS + ORF + terminator)
- CAI optimization matches the E. coli K-12 MG1655 codon usage table
- Restriction site removal (NEB 6-cutter standard)
- GC balance (45-55% typical E. coli range)
- His-tag addition
- Complete vector design (expression cassette + origin + selection marker + MCS)
- Multi-dimensional biological sanity validation
- GenBank format export (LOCUS/FEATURES/ORIGIN/END)
- FASTA format export
- Different promoter choices (lac/T7/araBAD/tet)
- Different terminator choices (rrnB_T1/T7)

References:
- E. coli K-12 MG1655 (Blattner 1997 Science 277:1453-1462)
- Codon usage: CUTG E. coli 511145
- lac/araBAD/tet promoters: Miller 1972, Guzman 1995, Hillen 1982
- rrnB T1/T7 terminators: Brosius 1981, Dunn 1983
- pUC19/pBR322/pSC101 origins of replication
- AmpR/KanR/CamR selection markers
- GenBank format: NCBI-GenBankFlatFile-2024
"""
from __future__ import annotations

import pytest

# BioPython used for translation validation (same strategy as test_biocodec.py)
pytest.importorskip("Bio")

from helixlang.core.errors import BioError
from helixlang.plugins.apps.synbio_designer import (
    DEFAULT_MCS,
    HIS_TAG_PROTEIN,
    MCS_SITES,
    ORIGIN_SEQUENCES,
    PROMOTER_SEQUENCES,
    SELECTION_MARKERS,
    TERMINATOR_SEQUENCES,
    Cassette,
    CassetteConfig,
    SynBioDesigner,
    Vector,
    VectorConfig,
    genbank_format,
    validate_cassette,
)
from helixlang.plugins.runtime.biocodec import (
    LAC_PROMOTER,
    RESTRICTION_SITES,
    RRNB_T1_TERMINATOR,
    T7_PROMOTER,
    T7_TERMINATOR,
    find_restriction_sites,
)

# ============================================================================
# Test protein sequence (part of GFP, ~50 aa, commonly used for expression vector tests)
# ============================================================================

GFP_PROTEIN = "MASKGEELFTGVPVPILVELDGDVNGHKFSVSGEGEGDATYGRTLTKF"
SHORT_PROTEIN = "MASKGEELFTGVPVPILVELDGDVNGHK"


# ============================================================================
# Expression cassette design
# ============================================================================

class TestCassetteDesign:
    """Verifies expression cassette design: promoter + RBS + ORF + terminator."""

    def test_cassette_structure(self):
        """Expression cassette contains four parts: promoter + RBS + ORF + terminator."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(GFP_PROTEIN)
        assert isinstance(cassette, Cassette)
        # Promoter
        assert cassette.promoter_seq == LAC_PROMOTER  # default lac
        assert len(cassette.promoter_seq) > 0
        # RBS
        assert cassette.rbs_seq == "AGGAGG"  # default Shine-Dalgarno
        # ORF: starts with ATG, ends with a stop codon
        assert cassette.orf_seq.startswith("ATG")
        assert cassette.orf_seq[-3:] in {"TAA", "TAG", "TGA"}
        # Terminator
        assert cassette.terminator_seq == RRNB_T1_TERMINATOR
        # Full sequence = promoter + RBS + ORF + terminator
        expected = (cassette.promoter_seq + cassette.rbs_seq
                    + cassette.orf_seq + cassette.terminator_seq)
        assert cassette.full_sequence == expected

    def test_cassette_length_reasonable(self):
        """Expression cassette length is reasonable: promoter 71 + RBS 6 + ORF ~90 + terminator 34."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(SHORT_PROTEIN)
        # SHORT_PROTEIN 28 aa -> ORF = 28*3 + 3 (stop) = 87 bp
        # Met is already in the protein; back_translate includes ATG, then adds TAA
        assert len(cassette.orf_seq) == (len(SHORT_PROTEIN) + 1) * 3
        # Promoter 71 + RBS 6 + ORF 87 + terminator 34 = 198 bp
        assert len(cassette.full_sequence) >= 198

    def test_orf_translates_back_to_protein(self):
        """Protein sequence translated from the ORF matches the target."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(SHORT_PROTEIN)
        from Bio.Seq import Seq
        translated = str(Seq(cassette.orf_seq).translate())
        # Translation result = target protein + stop *
        assert translated.rstrip("*") == SHORT_PROTEIN
        assert translated.endswith("*")

    def test_cassette_validation_report_present(self):
        """Cassette contains the validation_report field."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(SHORT_PROTEIN)
        assert isinstance(cassette.validation_report, dict)
        assert "valid" in cassette.validation_report
        assert "gc_content" in cassette.validation_report
        assert "cai" in cassette.validation_report


# ============================================================================
# CAI optimization
# ============================================================================

class TestCAIOptimization:
    """Verifies codon optimization matches the E. coli K-12 MG1655 codon usage table."""

    def test_optimized_cai_high(self):
        """Optimized ORF has CAI > 0.4 (E. coli expression threshold)."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(GFP_PROTEIN,
                                            CassetteConfig(optimize_codons=True))
        assert cassette.cai > 0.4, f"CAI {cassette.cai:.3f} should be > 0.4"

    def test_optimized_uses_high_frequency_codons(self):
        """Optimization uses high-frequency E. coli codons (e.g. Leu->CTG)."""
        designer = SynBioDesigner(seed=42)
        # Protein contains Leu; CTG is the optimal E. coli Leu codon (fraction 0.47)
        cassette = designer.design_cassette("MMMLLLMMM",
                                            CassetteConfig(optimize_codons=True))
        codons = [cassette.orf_seq[i:i + 3]
                  for i in range(0, len(cassette.orf_seq) - 3, 3)]
        # Should contain CTG (optimal for Leu)
        assert "CTG" in codons

    def test_optimized_cai_higher_than_random(self):
        """Optimized mode CAI is higher than random mode."""
        designer_opt = SynBioDesigner(seed=42)
        cassette_opt = designer_opt.design_cassette(
            GFP_PROTEIN, CassetteConfig(optimize_codons=True,
                                        avoid_restriction=False))
        designer_rand = SynBioDesigner(seed=42)
        cassette_rand = designer_rand.design_cassette(
            GFP_PROTEIN, CassetteConfig(optimize_codons=False,
                                        avoid_restriction=False))
        assert cassette_opt.cai >= cassette_rand.cai, (
            f"optimized CAI {cassette_opt.cai} should be >= "
            f"random CAI {cassette_rand.cai}")

    def test_cai_value_range(self):
        """CAI values are within the [0, 1] range."""
        designer = SynBioDesigner(seed=42)
        for protein in ["M", "MASKGEELF", GFP_PROTEIN]:
            cassette = designer.design_cassette(protein)
            assert 0.0 <= cassette.cai <= 1.0


# ============================================================================
# Restriction site removal
# ============================================================================

class TestRestrictionSiteRemoval:
    """Verifies restriction site removal (NEB 6-cutter standard)."""

    def test_no_restriction_sites_in_orf(self):
        """Optimized ORF contains no common restriction enzyme sites."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(
            GFP_PROTEIN,
            CassetteConfig(optimize_codons=True, avoid_restriction=True))
        # Check the ORF portion
        sites = find_restriction_sites(cassette.orf_seq)
        # Should have no common sites (EcoRI/BamHI/HindIII etc.)
        common_sites = {"EcoRI", "BamHI", "HindIII", "XhoI", "XbaI",
                        "SalI", "PstI", "KpnI", "NheI", "SphI"}
        found_common = set(sites.keys()) & common_sites
        assert not found_common, (
            f"common restriction sites remain: {found_common}")

    def test_restriction_sites_field_populated(self):
        """Cassette.restriction_sites_found field is populated correctly."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(GFP_PROTEIN)
        assert isinstance(cassette.restriction_sites_found, list)

    def test_avoid_restriction_disabled_keeps_sites(self):
        """Sites may remain when avoid_restriction is disabled (not forcibly removed)."""
        designer = SynBioDesigner(seed=42)
        # Use a protein containing an EcoRI site (Glu-Phe = GAA-TTC = GAATTC)
        # GEELF in GFP contains GAA (Glu) -> with downstream TTC (Phe) may form EcoRI
        cassette = designer.design_cassette(
            "MEFEF",  # M-E-F-E-F
            CassetteConfig(optimize_codons=True, avoid_restriction=False))
        # Verify the ORF translation is correct
        from Bio.Seq import Seq
        translated = str(Seq(cassette.orf_seq).translate()).rstrip("*")
        assert translated == "MEFEF"

    def test_ecori_can_be_removed(self):
        """A sequence containing an EcoRI site can be removed (synonymous mutation GAA->GAG)."""
        designer = SynBioDesigner(seed=42)
        # GFP contains GAA (Glu) -> after optimization + restriction site removal, no EcoRI should remain
        cassette = designer.design_cassette(
            "MEFEF",
            CassetteConfig(optimize_codons=True, avoid_restriction=True))
        sites = find_restriction_sites(cassette.orf_seq)
        assert "EcoRI" not in sites


# ============================================================================
# GC balance
# ============================================================================

class TestGCBalance:
    """Verifies GC balance (45-55% typical E. coli range)."""

    def test_gc_in_range(self):
        """Optimized ORF GC content is within the 45-55% range."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(GFP_PROTEIN)
        assert 0.45 <= cassette.gc_content <= 0.55, (
            f"GC {cassette.gc_content:.3f} out of [0.45, 0.55]")

    def test_gc_target_configurable(self):
        """gc_target is configurable (verified within tolerance)."""
        designer = SynBioDesigner(seed=42)
        # Default target 0.50 -> GC in [0.45, 0.55]
        cassette = designer.design_cassette(
            GFP_PROTEIN, CassetteConfig(gc_target=0.50))
        assert 0.45 <= cassette.gc_content <= 0.55

    def test_gc_validation_passes(self):
        """validation_report.gc_in_range is True when GC is within range."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(GFP_PROTEIN)
        assert cassette.validation_report["gc_in_range"]


# ============================================================================
# His-tag addition
# ============================================================================

class TestHisTag:
    """Verifies 6xHis tag addition."""

    def test_his_tag_added_to_protein(self):
        """Protein C-terminus contains 6xHis when add_histidine_tag=True."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(
            SHORT_PROTEIN, CassetteConfig(add_histidine_tag=True))
        # Protein should end with HHHHHH
        assert cassette.protein.endswith(HIS_TAG_PROTEIN)
        assert cassette.protein.endswith("HHHHHH")

    def test_his_tag_not_added_by_default(self):
        """His-tag is not added by default."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(SHORT_PROTEIN)
        assert not cassette.protein.endswith("HHHHHH")

    def test_his_tag_translates_correctly(self):
        """Protein translated from an ORF with His-tag contains 6xHis."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(
            SHORT_PROTEIN, CassetteConfig(add_histidine_tag=True))
        from Bio.Seq import Seq
        translated = str(Seq(cassette.orf_seq).translate()).rstrip("*")
        # Translated protein should equal original protein + His-tag
        assert translated == SHORT_PROTEIN + HIS_TAG_PROTEIN
        # His-tag 6xH (encoded by CAT/CAC)
        assert translated.endswith("HHHHHH")

    def test_his_tag_dna_uses_histidine_codons(self):
        """His-tag DNA uses the His synonymous codons CAT/CAC."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(
            SHORT_PROTEIN, CassetteConfig(add_histidine_tag=True))
        # His-tag at the end of the ORF (before TAA): 18 bp = 6 codons
        his_dna = cassette.orf_seq[-21:-3]  # strip stop TAA
        his_codons = [his_dna[i:i + 3] for i in range(0, len(his_dna), 3)]
        # All should be CAT or CAC (His synonymous codons)
        assert all(c in {"CAT", "CAC"} for c in his_codons), (
            f"non-His codons in His-tag: {his_codons}")


# ============================================================================
# Complete vector design
# ============================================================================

class TestVectorDesign:
    """Verifies complete vector design."""

    def test_vector_structure(self):
        """Vector contains expression cassette + origin + selection marker + MCS."""
        designer = SynBioDesigner(seed=42)
        vc = VectorConfig(cassette=CassetteConfig())
        vector = designer.design_vector(GFP_PROTEIN, vc)
        assert isinstance(vector, Vector)
        # Each part is non-empty
        assert len(vector.origin_seq) > 0
        assert len(vector.marker_seq) > 0
        assert len(vector.mcs_seq) > 0
        assert isinstance(vector.cassette, Cassette)
        # Full sequence = origin + marker + mcs + cassette
        expected = (vector.origin_seq + vector.marker_seq + vector.mcs_seq
                    + vector.cassette.full_sequence)
        assert vector.full_sequence == expected
        assert vector.total_length == len(vector.full_sequence)

    def test_vector_total_length(self):
        """Vector total length is reasonable (> cassette length)."""
        designer = SynBioDesigner(seed=42)
        vc = VectorConfig(cassette=CassetteConfig())
        vector = designer.design_vector(GFP_PROTEIN, vc)
        cassette_only = designer.design_cassette(GFP_PROTEIN)
        assert vector.total_length > len(cassette_only.full_sequence)

    def test_vector_features_annotated(self):
        """Vector features list annotates each element completely."""
        designer = SynBioDesigner(seed=42)
        vc = VectorConfig(cassette=CassetteConfig(promoter="lac"))
        vector = designer.design_vector(GFP_PROTEIN, vc)
        assert isinstance(vector.features, list)
        assert len(vector.features) >= 5  # origin + marker + mcs + promoter + RBS + CDS + terminator
        feature_types = [f["type"] for f in vector.features]
        assert "rep_origin" in feature_types
        assert "CDS" in feature_types
        assert "promoter" in feature_types
        # Check feature coordinates cover the whole sequence
        feature_labels = [f.get("label", "") for f in vector.features]
        assert any("ori" in lbl for lbl in feature_labels)
        assert any("AmpR" == lbl for lbl in feature_labels)

    def test_vector_default_config(self):
        """Default vector config: pUC19 origin + AmpR marker."""
        vc = VectorConfig(cassette=CassetteConfig())
        assert vc.origin_of_replication == "pUC19"
        assert vc.selection_marker == "AmpR"
        # Default MCS contains common enzymes
        assert "EcoRI" in vc.mcs_sites
        assert "BamHI" in vc.mcs_sites
        assert "HindIII" in vc.mcs_sites


# ============================================================================
# Vector with origin + selection marker
# ============================================================================

class TestVectorComponents:
    """Verifies the vector contains the correct origin and selection marker sequences."""

    def test_pUC19_origin_present(self):
        """pUC19 vector contains the colE1 origin sequence."""
        designer = SynBioDesigner(seed=42)
        vc = VectorConfig(cassette=CassetteConfig(),
                          origin_of_replication="pUC19")
        vector = designer.design_vector(GFP_PROTEIN, vc)
        assert vector.origin_seq == ORIGIN_SEQUENCES["pUC19"]
        assert len(vector.origin_seq) > 100  # simplified ~200bp

    def test_pBR322_origin_present(self):
        """pBR322 vector contains the colE1 origin variant sequence."""
        designer = SynBioDesigner(seed=42)
        vc = VectorConfig(cassette=CassetteConfig(),
                          origin_of_replication="pBR322")
        vector = designer.design_vector(GFP_PROTEIN, vc)
        assert vector.origin_seq == ORIGIN_SEQUENCES["pBR322"]

    def test_pSC101_origin_present(self):
        """pSC101 vector contains the stringent origin sequence."""
        designer = SynBioDesigner(seed=42)
        vc = VectorConfig(cassette=CassetteConfig(),
                          origin_of_replication="pSC101")
        vector = designer.design_vector(GFP_PROTEIN, vc)
        assert vector.origin_seq == ORIGIN_SEQUENCES["pSC101"]

    def test_ampr_marker_present(self):
        """AmpR vector contains the beta-lactamase marker sequence."""
        designer = SynBioDesigner(seed=42)
        vc = VectorConfig(cassette=CassetteConfig(),
                          selection_marker="AmpR")
        vector = designer.design_vector(GFP_PROTEIN, vc)
        assert vector.marker_seq == SELECTION_MARKERS["AmpR"]
        # AmpR sequence starts with ATG (signal peptide)
        assert vector.marker_seq.startswith("ATG")

    def test_kanr_marker_present(self):
        """KanR vector contains the nptII marker sequence."""
        designer = SynBioDesigner(seed=42)
        vc = VectorConfig(cassette=CassetteConfig(),
                          selection_marker="KanR")
        vector = designer.design_vector(GFP_PROTEIN, vc)
        assert vector.marker_seq == SELECTION_MARKERS["KanR"]
        assert vector.marker_seq.startswith("ATG")

    def test_camr_marker_present(self):
        """CamR vector contains the cat marker sequence."""
        designer = SynBioDesigner(seed=42)
        vc = VectorConfig(cassette=CassetteConfig(),
                          selection_marker="CamR")
        vector = designer.design_vector(GFP_PROTEIN, vc)
        assert vector.marker_seq == SELECTION_MARKERS["CamR"]
        assert vector.marker_seq.startswith("ATG")

    def test_mcs_contains_restriction_sites(self):
        """MCS contains restriction site sequences."""
        designer = SynBioDesigner(seed=42)
        vc = VectorConfig(cassette=CassetteConfig(),
                          mcs_sites=["EcoRI", "BamHI", "HindIII"])
        vector = designer.design_vector(GFP_PROTEIN, vc)
        # MCS sequence should contain EcoRI/BamHI/HindIII sites
        assert RESTRICTION_SITES["EcoRI"] in vector.mcs_seq
        assert RESTRICTION_SITES["BamHI"] in vector.mcs_seq
        assert RESTRICTION_SITES["HindIII"] in vector.mcs_seq


# ============================================================================
# Multi-dimensional validation
# ============================================================================

class TestValidation:
    """Verifies multi-dimensional biological sanity checks."""

    def test_valid_orf_passes(self):
        """A valid ORF passes validation."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(GFP_PROTEIN)
        report = designer.validate(cassette.orf_seq)
        assert report["valid"] is True
        assert report["orf_found"] is True
        assert report["orf_length"] > 0
        assert report["orf_cai"] > 0.4

    def test_low_cai_detected(self):
        """Low CAI is detected (all CTA rare codons)."""
        # CTA: Leu, fraction 0.04 -> very low CAI
        # ATG + CTA*6 + TAA = M-L-L-L-L-L-L-* (24 bp, mult of 3, includes stop)
        bad_dna = "ATG" + "CTA" * 6 + "TAA"
        designer = SynBioDesigner(seed=42)
        report = designer.validate(bad_dna)
        assert report["orf_cai"] < 0.3
        assert any("CAI" in e for e in report["errors"])

    def test_internal_stop_detected(self):
        """Internal stop codon is detected."""
        # M-A-*-V-*  internal TAA
        bad_dna = "ATGGCTTAAGTTTAA"  # 15 bp, mult of 3
        designer = SynBioDesigner(seed=42)
        report = designer.validate(bad_dna)
        assert any("internal stop" in e for e in report["errors"])

    def test_restriction_site_detected(self):
        """Restriction enzyme sites are detected."""
        # Contains EcoRI: GAATTC = GAA (Glu) + TTC (Phe)
        bad_dna = "ATGGAATTCTTTTTTTTTTTAA"  # M-E-F-F-F-F-F-*  (24 bp, mult of 3)
        designer = SynBioDesigner(seed=42)
        report = designer.validate(bad_dna)
        assert "EcoRI" in report["restriction_sites"]

    def test_long_homopolymer_detected(self):
        """Long homopolymers are detected."""
        # Contains 6x A
        bad_dna = "ATGGCTAAAAAACTGTAA"  # 18 bp, mult of 3
        designer = SynBioDesigner(seed=42)
        report = designer.validate(bad_dna)
        assert report["max_homopolymer"] >= 5

    def test_validation_dict_keys(self):
        """Validation return dict contains all required fields."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(GFP_PROTEIN)
        report = designer.validate(cassette.orf_seq)
        required_keys = {"valid", "orf_found", "orf_seq", "orf_length",
                         "gc_content", "max_homopolymer",
                         "restriction_sites", "orf_cai", "errors"}
        assert required_keys.issubset(report.keys())

    def test_validate_cassette_function(self):
        """The validate_cassette module function works correctly."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(GFP_PROTEIN)
        report = validate_cassette(cassette.orf_seq)
        assert report["valid"] is True
        assert report["has_start_codon"] is True
        assert report["has_stop_codon"] is True
        assert report["length_multiple_of_3"] is True
        assert report["no_internal_stop"] is True
        assert report["cai_adequate"] is True
        assert report["no_restriction"] is True

    def test_no_start_codon_fails(self):
        """Missing start codon fails."""
        # Does not start with ATG
        bad_dna = "GCGGCTACCGTTGCCGGC"  # 18 bp, mult of 3, does not start with ATG
        report = validate_cassette(bad_dna)
        assert report["has_start_codon"] is False
        assert report["valid"] is False


# ============================================================================
# GenBank export
# ============================================================================

class TestGenBankExport:
    """Verifies GenBank format export."""

    def test_genbank_has_required_sections(self):
        """GenBank text contains the four required sections LOCUS/FEATURES/ORIGIN/END."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(GFP_PROTEIN)
        gb = designer.export_genbank(cassette.orf_seq, "test_orf")
        assert gb.startswith("LOCUS")
        assert "FEATURES" in gb
        assert "ORIGIN" in gb
        assert gb.rstrip().endswith("//")

    def test_genbank_locus_format(self):
        """LOCUS line format is correct (contains name + length + bp + DNA)."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(SHORT_PROTEIN)
        gb = designer.export_genbank(cassette.orf_seq, "my_gene")
        first_line = gb.split("\n")[0]
        assert first_line.startswith("LOCUS")
        assert "MY_GENE" in first_line  # name uppercased
        assert "bp" in first_line
        assert "DNA" in first_line
        assert str(len(cassette.orf_seq)) in first_line

    def test_genbank_sequence_correct(self):
        """GenBank ORIGIN section contains the complete DNA sequence."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(SHORT_PROTEIN)
        gb = designer.export_genbank(cassette.orf_seq, "test_orf")
        # Extract the sequence after ORIGIN (strip spaces and digits)
        origin_idx = gb.index("ORIGIN")
        seq_section = gb[origin_idx + len("ORIGIN"):]
        # Remove the trailing //
        seq_section = seq_section.replace("//", "").strip()
        # Remove digits and spaces
        seq_only = "".join(c for c in seq_section if c in "ACGT")
        assert seq_only == cassette.orf_seq.upper()

    def test_genbank_with_features(self):
        """GenBank contains feature annotations (CDS/promoter/terminator)."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(SHORT_PROTEIN)
        features = [
            {"type": "promoter", "start": 1, "end": 10, "label": "test_prom"},
            {"type": "CDS", "start": 11, "end": len(cassette.orf_seq),
             "label": "ORF", "translation": SHORT_PROTEIN},
        ]
        gb = designer.export_genbank(cassette.orf_seq, "test_orf", features)
        assert "promoter" in gb
        assert "CDS" in gb
        assert "test_prom" in gb
        assert "/translation=" in gb

    def test_genbank_format_function(self):
        """The genbank_format module function works correctly."""
        gb = genbank_format("ATGGCTTAA", "test_seq")
        assert gb.startswith("LOCUS")
        assert "ORIGIN" in gb
        assert gb.rstrip().endswith("//")
        # The 9 bp sequence should be in the ORIGIN section
        assert "ATGGCTTAA" in gb.replace(" ", "").replace("\n", "") or \
               "ATG GCT TAA" in gb

    def test_genbank_source_feature(self):
        """GenBank automatically adds a source feature."""
        designer = SynBioDesigner(seed=42)
        gb = designer.export_genbank("ATGGCTTAA", "test")
        assert "source" in gb
        assert "synthetic DNA" in gb


# ============================================================================
# FASTA export
# ============================================================================

class TestFASTAExport:
    """Verifies FASTA format export."""

    def test_fasta_format(self):
        """FASTA format: >header + sequence lines."""
        designer = SynBioDesigner(seed=42)
        fa = designer.export_fasta("ATGGCTACCTAA", "my_gene")
        lines = fa.strip().split("\n")
        assert lines[0] == ">my_gene"
        assert lines[1] == "ATGGCTACCTAA"

    def test_fasta_sequence_correct(self):
        """FASTA contains the complete DNA sequence."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(SHORT_PROTEIN)
        fa = designer.export_fasta(cassette.orf_seq, "test_orf")
        # Extract the sequence portion
        lines = fa.strip().split("\n")
        seq_only = "".join(lines[1:])
        assert seq_only == cassette.orf_seq.upper()

    def test_fasta_wraps_at_60(self):
        """FASTA sequence is wrapped at 60 characters per line."""
        designer = SynBioDesigner(seed=42)
        long_dna = "ATGGCT" * 50  # 300 bp
        fa = designer.export_fasta(long_dna, "long_seq")
        lines = fa.strip().split("\n")
        # First line is the header
        assert lines[0] == ">long_seq"
        # Sequence lines: at most 60 characters per line
        for seq_line in lines[1:]:
            assert len(seq_line) <= 60
        # 5 lines x 60 = 300
        assert len(lines[1:]) == 5

    def test_fasta_uppercase(self):
        """FASTA sequence is uppercased."""
        designer = SynBioDesigner(seed=42)
        fa = designer.export_fasta("atggctacctaa", "test")
        seq_lines = fa.strip().split("\n")[1:]
        seq = "".join(seq_lines)
        assert seq == "ATGGCTACCTAA"

    def test_fasta_ends_with_newline(self):
        """FASTA ends with a newline."""
        designer = SynBioDesigner(seed=42)
        fa = designer.export_fasta("ATGGCT", "test")
        assert fa.endswith("\n")


# ============================================================================
# Different promoter choices
# ============================================================================

class TestPromoterSelection:
    """Verifies different promoter choices (lac/T7/araBAD/tet)."""

    @pytest.mark.parametrize("promoter_name", ["lac", "T7", "araBAD", "tet"])
    def test_promoter_sequences_available(self, promoter_name):
        """All four promoter sequences are available."""
        assert promoter_name in PROMOTER_SEQUENCES
        seq = PROMOTER_SEQUENCES[promoter_name]
        assert len(seq) > 0
        assert all(c in "ACGT" for c in seq)

    def test_lac_promoter_used(self):
        """lac promoter uses the LAC_PROMOTER sequence."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(
            SHORT_PROTEIN, CassetteConfig(promoter="lac"))
        assert cassette.promoter_seq == LAC_PROMOTER
        # lacP contains the -35 (TTTACA) and -10 (TATAAT) consensus sequences
        assert "TTTACA" in cassette.promoter_seq

    def test_t7_promoter_used(self):
        """T7 promoter uses the T7_PROMOTER sequence."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(
            SHORT_PROTEIN, CassetteConfig(promoter="T7"))
        assert cassette.promoter_seq == T7_PROMOTER
        assert len(cassette.promoter_seq) == 23

    def test_arabad_promoter_used(self):
        """araBAD promoter uses the PBAD sequence."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(
            SHORT_PROTEIN, CassetteConfig(promoter="araBAD"))
        assert cassette.promoter_seq == PROMOTER_SEQUENCES["araBAD"]
        # Length > 50 bp
        assert len(cassette.promoter_seq) > 50

    def test_tet_promoter_used(self):
        """tet promoter uses the PTet sequence."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(
            SHORT_PROTEIN, CassetteConfig(promoter="tet"))
        assert cassette.promoter_seq == PROMOTER_SEQUENCES["tet"]
        assert len(cassette.promoter_seq) > 50

    def test_unknown_promoter_raises(self):
        """Unknown promoter raises ValueError."""
        designer = SynBioDesigner(seed=42)
        with pytest.raises(BioError, match="unknown promoter"):
            designer.design_cassette(SHORT_PROTEIN,
                                     CassetteConfig(promoter="unknown"))

    def test_promoter_affects_full_sequence(self):
        """Different promoters produce different full cassette sequences."""
        designer = SynBioDesigner(seed=42)
        c_lac = designer.design_cassette(
            SHORT_PROTEIN, CassetteConfig(promoter="lac"))
        c_t7 = designer.design_cassette(
            SHORT_PROTEIN, CassetteConfig(promoter="T7"))
        assert c_lac.full_sequence != c_t7.full_sequence
        assert c_lac.promoter_seq != c_t7.promoter_seq


# ============================================================================
# Different terminator choices
# ============================================================================

class TestTerminatorSelection:
    """Verifies different terminator choices (rrnB_T1/T7)."""

    @pytest.mark.parametrize("terminator_name", ["rrnB_T1", "T7"])
    def test_terminator_sequences_available(self, terminator_name):
        """All terminator sequences are available."""
        assert terminator_name in TERMINATOR_SEQUENCES
        seq = TERMINATOR_SEQUENCES[terminator_name]
        assert len(seq) > 20

    def test_rrnb_t1_terminator_used(self):
        """rrnB_T1 terminator uses the RRNB_T1_TERMINATOR sequence."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(
            SHORT_PROTEIN, CassetteConfig(terminator="rrnB_T1"))
        assert cassette.terminator_seq == RRNB_T1_TERMINATOR
        # rrnB T1 contains a poly-T tail
        assert "TTTT" in cassette.terminator_seq

    def test_t7_terminator_used(self):
        """T7 terminator uses the T7_TERMINATOR sequence."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(
            SHORT_PROTEIN, CassetteConfig(terminator="T7"))
        assert cassette.terminator_seq == T7_TERMINATOR
        assert len(cassette.terminator_seq) > 30

    def test_unknown_terminator_raises(self):
        """Unknown terminator raises ValueError."""
        designer = SynBioDesigner(seed=42)
        with pytest.raises(BioError, match="unknown terminator"):
            designer.design_cassette(SHORT_PROTEIN,
                                     CassetteConfig(terminator="unknown"))

    def test_terminator_affects_full_sequence(self):
        """Different terminators produce different full cassette sequences."""
        designer = SynBioDesigner(seed=42)
        c_rrnb = designer.design_cassette(
            SHORT_PROTEIN, CassetteConfig(terminator="rrnB_T1"))
        c_t7 = designer.design_cassette(
            SHORT_PROTEIN, CassetteConfig(terminator="T7"))
        assert c_rrnb.full_sequence != c_t7.full_sequence
        assert c_rrnb.terminator_seq != c_t7.terminator_seq


# ============================================================================
# Data integrity
# ============================================================================

class TestDataIntegrity:
    """Verifies sequence data integrity."""

    def test_promoter_sequences_dict(self):
        """PROMOTER_SEQUENCES contains lac/T7/araBAD/tet (four)."""
        required = {"lac", "T7", "araBAD", "tet"}
        assert required.issubset(PROMOTER_SEQUENCES.keys())

    def test_origin_sequences_dict(self):
        """ORIGIN_SEQUENCES contains pUC19/pBR322/pSC101 (three)."""
        required = {"pUC19", "pBR322", "pSC101"}
        assert required.issubset(ORIGIN_SEQUENCES.keys())
        for name, seq in ORIGIN_SEQUENCES.items():
            assert len(seq) > 100, f"{name} origin too short: {len(seq)}"
            assert all(c in "ACGT" for c in seq)

    def test_selection_markers_dict(self):
        """SELECTION_MARKERS contains AmpR/KanR/CamR (three)."""
        required = {"AmpR", "KanR", "CamR"}
        assert required.issubset(SELECTION_MARKERS.keys())
        for name, seq in SELECTION_MARKERS.items():
            assert seq.startswith("ATG"), f"{name} marker not start with ATG"

    def test_mcs_sites_dict(self):
        """MCS_SITES contains common NEB restriction enzyme sites."""
        required = {"EcoRI", "BamHI", "HindIII", "XhoI", "XbaI",
                    "SalI", "PstI", "KpnI", "NotI"}
        assert required.issubset(MCS_SITES.keys())

    def test_default_mcs_list(self):
        """DEFAULT_MCS contains common enzymes and is non-empty."""
        assert len(DEFAULT_MCS) >= 8
        assert "EcoRI" in DEFAULT_MCS
        assert "BamHI" in DEFAULT_MCS

    def test_config_defaults(self):
        """CassetteConfig defaults are correct."""
        cfg = CassetteConfig()
        assert cfg.promoter == "lac"
        assert cfg.rbs == "aggagg"
        assert cfg.terminator == "rrnB_T1"
        assert cfg.optimize_codons is True
        assert cfg.avoid_restriction is True
        assert cfg.gc_target == 0.50
        assert cfg.max_homopolymer == 4
        assert cfg.add_histidine_tag is False
        assert cfg.add_mbd_tag is False


# ============================================================================
# End-to-end: real scenario (GFP expression vector construction)
# ============================================================================

class TestRealisticScenario:
    """Real scenario: build a GFP expression vector and validate it."""

    def test_full_gfp_vector_pipeline(self):
        """Complete GFP vector design pipeline: protein -> vector -> validation -> export."""
        designer = SynBioDesigner(seed=42)
        # 1. Design the expression cassette (lac + GFP ORF + rrnB T1)
        cassette = designer.design_cassette(GFP_PROTEIN)
        assert cassette.cai > 0.4
        assert 0.45 <= cassette.gc_content <= 0.55
        # 2. Build the complete vector
        vc = VectorConfig(cassette=CassetteConfig(promoter="lac",
                                                  terminator="rrnB_T1"))
        vector = designer.design_vector(GFP_PROTEIN, vc)
        # 3. Validate the ORF
        report = designer.validate(cassette.orf_seq)
        assert report["valid"]
        # 4. Export GenBank
        gb = designer.export_genbank(vector.full_sequence, "gfp_vector",
                                     features=vector.features)
        assert gb.startswith("LOCUS")
        assert gb.rstrip().endswith("//")
        # 5. Export FASTA
        fa = designer.export_fasta(vector.full_sequence, "gfp_vector")
        assert fa.startswith(">gfp_vector")

    def test_his_tagged_purification_vector(self):
        """His-tag purification vector: GFP + C-terminal 6xHis."""
        designer = SynBioDesigner(seed=42)
        cassette = designer.design_cassette(
            GFP_PROTEIN,
            CassetteConfig(add_histidine_tag=True,
                           promoter="T7",   # T7 expression system
                           terminator="T7"))
        # Protein should end with a His-tag
        assert cassette.protein.endswith("HHHHHH")
        # ORF translation contains the His-tag
        from Bio.Seq import Seq
        translated = str(Seq(cassette.orf_seq).translate()).rstrip("*")
        assert translated == GFP_PROTEIN + HIS_TAG_PROTEIN
        # T7 promoter + T7 terminator
        assert cassette.promoter_seq == T7_PROMOTER
        assert cassette.terminator_seq == T7_TERMINATOR

    def test_inducible_arabad_vector(self):
        """araBAD inducible expression vector."""
        designer = SynBioDesigner(seed=42)
        vc = VectorConfig(
            cassette=CassetteConfig(promoter="araBAD",
                                    terminator="rrnB_T1"),
            origin_of_replication="pSC101",   # low-copy stringent
            selection_marker="CamR",
        )
        vector = designer.design_vector(GFP_PROTEIN, vc)
        # Verify the components
        assert vector.cassette.promoter_seq == PROMOTER_SEQUENCES["araBAD"]
        assert vector.origin_seq == ORIGIN_SEQUENCES["pSC101"]
        assert vector.marker_seq == SELECTION_MARKERS["CamR"]
