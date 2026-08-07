"""Epigenetic modification model tests: DNA methylation + histone modifications.

Verifies (based on real paper parameters):
- Dam site search: GATC sequence (Marinus 1973)
- Dcm site search: CCWGG sequence (W=A/T, Marinus 1984)
- CpG site search: CG dinucleotide (Bird 2002)
- CpG island detection: Takai 2002 criteria (GC>55%, length>200bp, O/E>0.65)
- Dam methylation: >95% of GATC sites methylated (Marinus 1973)
- Eukaryotic CpG methylation: islands low methylation (<20%), non-islands high methylation (>70%)
- Histone modification addition: H3K4me3/H3K27me3/H3K36me3/H3K9me3
- Chromatin accessibility: methylation down, H3K4me3 up, H3K27me3/H3K9me3 down
- Expression modifiers: methylation→repression, H3K4me3→activation, H3K27me3→repression

References:
- Marinus 1973 MGG 115:248-250 (Dam GATC methylation)
- Marinus 1984 (Dcm CCWGG methylation)
- Bird 2002 Cell 109:1-8 (CpG methylation and gene expression)
- Takai 2002 PNAS 99:3720-3725 (CpG island criteria)
"""
from __future__ import annotations

import random

import pytest

from helixlang.epigenetics import (
    # Constants
    HISTONE_MARK_TYPES,
    ChromatinState,
    HistoneMark,
    # Dataclasses
    MethylationState,
    add_histone_marks,
    # Computation
    calculate_accessibility,
    calculate_expression_modifier,
    find_cpg_islands,
    find_cpg_sites,
    # Site search
    find_dam_sites,
    find_dcm_sites,
    # Methylation/histone
    methylate_dna,
)

# ============================================================================
# Site search tests
# ============================================================================

class TestDamSiteSearch:
    """Verifies Dam methylation site (GATC) search."""

    def test_basic_gatc_search(self):
        """GATC site search."""
        dna = "GATCGATCGATC"  # GATC at 0, 4, 8
        sites = find_dam_sites(dna)
        assert sites == [0, 4, 8]

    def test_no_gatc_sites(self):
        """Returns an empty list when there are no GATC sites."""
        dna = "ATATATATATAT"
        sites = find_dam_sites(dna)
        assert sites == []

    def test_overlapping_gatc(self):
        """GATC does not overlap, but adjacent GATC sites should all be detected."""
        # GATCGATC: GATC at 0 and 4
        dna = "GATCGATC"
        sites = find_dam_sites(dna)
        assert sites == [0, 4]

    def test_case_insensitive(self):
        """Case insensitive."""
        dna = "gatcGATC"
        sites = find_dam_sites(dna)
        assert sites == [0, 4]

    def test_empty_dna(self):
        """Empty DNA returns an empty list."""
        assert find_dam_sites("") == []


class TestDcmSiteSearch:
    """Verifies Dcm methylation site (CCWGG, W=A/T) search."""

    def test_ccagg_search(self):
        """CCAGG site search (W=A)."""
        dna = "CCAGGCCAGG"  # CCAGG at 0 and 5
        sites = find_dcm_sites(dna)
        assert sites == [0, 5]

    def test_cctgg_search(self):
        """CCTGG site search (W=T)."""
        dna = "CCTGGCCTGG"  # CCTGG at 0 and 5
        sites = find_dcm_sites(dna)
        assert sites == [0, 5]

    def test_mixed_w_sites(self):
        """Mixed CCAGG and CCTGG sites."""
        dna = "CCAGGCCTGG"  # CCAGG at 0, CCTGG at 5
        sites = find_dcm_sites(dna)
        assert sites == [0, 5]

    def test_no_dcm_sites(self):
        """Returns an empty list when there are no CCWGG sites."""
        dna = "ATATATATATAT"
        sites = find_dcm_sites(dna)
        assert sites == []

    def test_sites_sorted(self):
        """The returned position list is sorted."""
        dna = "CCTGGCCAGG"  # CCTGG at 0, CCAGG at 5
        sites = find_dcm_sites(dna)
        assert sites == sorted(sites)
        assert sites == [0, 5]


class TestCpgSiteSearch:
    """Verifies CpG site (CG dinucleotide) search."""

    def test_basic_cpg_search(self):
        """CG dinucleotide site search."""
        dna = "ATCGATCGATCG"  # CG at 2, 6, 10
        sites = find_cpg_sites(dna)
        assert sites == [2, 6, 10]

    def test_no_cpg_sites(self):
        """Returns an empty list when there are no CG sites."""
        dna = "ATATATATATAT"
        sites = find_cpg_sites(dna)
        assert sites == []

    def test_dense_cpg(self):
        """Dense CpG sequence (CGCGCG...)."""
        dna = "CGCGCGCG"  # CG at 0, 2, 4, 6
        sites = find_cpg_sites(dna)
        assert sites == [0, 2, 4, 6]

    def test_case_insensitive(self):
        """Case insensitive."""
        dna = "cgCGcg"
        sites = find_cpg_sites(dna)
        assert sites == [0, 2, 4]


# ============================================================================
# CpG island detection tests
# ============================================================================

class TestCpgIslandDetection:
    """Verifies CpG island detection matches the Takai 2002 criteria."""

    def test_classic_cpg_island(self):
        """Classic CpG island: CG repeat ≥200 bp."""
        # CG repeat 300 bp: GC=1.0, O/E=2.0
        island = "CG" * 150  # 300 bp
        islands = find_cpg_islands(island)
        assert len(islands) >= 1
        first = islands[0]
        assert first["length"] >= 200
        assert first["gc_content"] >= 0.55
        assert first["cpg_oe"] >= 0.65
        assert first["start"] == 0

    def test_short_cg_not_island(self):
        """Short CG regions (<200 bp) are not CpG islands."""
        short_cg = "CG" * 50  # 100 bp, GC=1.0, OE=2.0, but too short
        islands = find_cpg_islands(short_cg)
        assert islands == []

    def test_at_rich_no_island(self):
        """AT-rich sequences have no CpG islands."""
        dna = "AT" * 200  # 400 bp, GC=0, no CpG
        islands = find_cpg_islands(dna)
        assert islands == []

    def test_island_separated_from_short_cg(self):
        """CpG island is separated from the short CG region."""
        # CpG island (300 bp) + AT spacer (400 bp) + short CG (100 bp)
        island = "CG" * 150   # 300 bp, position 0-299
        spacer = "AT" * 200   # 400 bp, position 300-699
        short_cg = "CG" * 50  # 100 bp, position 700-799
        dna = island + spacer + short_cg  # 800 bp

        islands = find_cpg_islands(dna)
        assert len(islands) >= 1
        # The first island is in the island region
        first = islands[0]
        assert first["start"] == 0
        assert first["length"] >= 200
        assert first["gc_content"] >= 0.55
        assert first["cpg_oe"] >= 0.65
        # The short CG region (>= 700) should not be detected as an island
        assert all(isl["start"] < 700 for isl in islands)

    def test_custom_parameters(self):
        """Custom parameters (stricter GC threshold)."""
        # Sequence with GC ~0.6
        dna = "CGCGCGATAT" * 40  # 400 bp, GC=0.6
        # GC threshold 0.7 should filter it out
        islands_strict = find_cpg_islands(dna, min_gc=0.7)
        assert len(islands_strict) == 0

    def test_island_return_fields(self):
        """The returned island dicts contain all required fields."""
        dna = "CG" * 150  # 300 bp
        islands = find_cpg_islands(dna)
        assert len(islands) >= 1
        for isl in islands:
            assert "start" in isl
            assert "end" in isl
            assert "length" in isl
            assert "gc_content" in isl
            assert "cpg_oe" in isl
            assert isl["end"] > isl["start"]
            assert isl["length"] == isl["end"] - isl["start"]

    def test_short_dna_returns_empty(self):
        """Short DNA (< min_length) returns an empty list."""
        dna = "CGCGCG"  # 6 bp
        islands = find_cpg_islands(dna)
        assert islands == []


# ============================================================================
# Dam methylation tests
# ============================================================================

class TestDamMethylation:
    """Verifies E. coli Dam methylation (>95% of GATC sites methylated)."""

    def test_dam_methylation_high(self):
        """Dam methylation: >95% of GATC sites methylated."""
        dna = "GATC" * 10 + "ATGCGATC" * 5  # multiple GATC sites
        rng = random.Random(42)
        state = methylate_dna(dna, cell_type="ecoli", methylase="dam", rng=rng)
        assert state.methylase == "dam"
        assert state.total_sites > 0
        # Each GATC site has methylation probability > 0.95
        for pos, prob in state.positions.items():
            assert prob > 0.95, f"position {pos} methylation {prob} <= 0.95"
        # All sites are methylated
        assert state.methylated_sites == state.total_sites

    def test_dam_only_methylates_gatc(self):
        """Dam only methylates GATC sites."""
        dna = "GATCATATGATCGGGG"  # GATC at 0 and 8
        rng = random.Random(42)
        state = methylate_dna(dna, cell_type="ecoli", methylase="dam", rng=rng)
        assert state.total_sites == 2
        # Only GATC start positions are methylated
        expected_positions = {0, 8}
        assert set(state.positions.keys()) == expected_positions

    def test_dam_no_sites(self):
        """Returns an empty state when there are no GATC sites."""
        dna = "ATATATATATAT"
        rng = random.Random(42)
        state = methylate_dna(dna, cell_type="ecoli", methylase="dam", rng=rng)
        assert state.total_sites == 0
        assert state.methylated_sites == 0
        assert state.positions == {}


class TestDcmMethylation:
    """Verifies E. coli Dcm methylation."""

    def test_dcm_methylation(self):
        """Dcm methylates CCWGG sites."""
        dna = "CCAGGCCTGGCCAGG"  # CCWGG at 0, 5, 10
        rng = random.Random(42)
        state = methylate_dna(dna, cell_type="ecoli", methylase="dcm", rng=rng)
        assert state.methylase == "dcm"
        assert state.total_sites == 3
        # Dcm is also highly methylated
        for pos, prob in state.positions.items():
            assert prob > 0.9, f"position {pos} methylation {prob} too low"
        assert state.methylated_sites == state.total_sites


# ============================================================================
# Eukaryotic CpG methylation tests
# ============================================================================

class TestEukaryoticCpgMethylation:
    """Verifies eukaryotic CpG methylation (islands low, non-islands high)."""

    def test_cpg_island_low_methylation(self):
        """Low methylation within CpG islands (<20%)."""
        # CpG island: CG repeat 300 bp
        island = "CG" * 150  # 300 bp
        # AT spacer: 400 bp
        spacer = "AT" * 200  # 400 bp
        # Short CG region: 100 bp (not a CpG island)
        short_cg = "CG" * 50  # 100 bp
        dna = island + spacer + short_cg  # 800 bp

        rng = random.Random(42)
        state = methylate_dna(dna, cell_type="mammal", methylase="cpg", rng=rng)
        assert state.methylase == "cpg"
        assert state.total_sites > 0

        # Sites inside the CpG island (position < 300) should be low-methylated <20%
        island_probs = [p for pos, p in state.positions.items() if pos < 300]
        assert len(island_probs) > 0, "no CpG sites in island region"
        for p in island_probs:
            assert p < 0.25, f"island methylation {p} should be < 0.25"

    def test_non_island_high_methylation(self):
        """High methylation in non-island regions (>70%)."""
        island = "CG" * 150   # 300 bp, CpG island
        spacer = "AT" * 200   # 400 bp, no CpG
        short_cg = "CG" * 50  # 100 bp, not a CpG island (too short)
        dna = island + spacer + short_cg  # 800 bp

        rng = random.Random(42)
        state = methylate_dna(dna, cell_type="mammal", methylase="cpg", rng=rng)

        # Non-island sites (position >= 700) should be high-methylated >70%
        nonisland_probs = [p for pos, p in state.positions.items() if pos >= 700]
        assert len(nonisland_probs) > 0, "no CpG sites in non-island region"
        for p in nonisland_probs:
            assert p > 0.65, f"non-island methylation {p} should be > 0.65"

    def test_cpg_methylation_contrast(self):
        """CpG island and non-island methylation rates differ significantly."""
        island = "CG" * 150
        spacer = "AT" * 200
        short_cg = "CG" * 50
        dna = island + spacer + short_cg

        rng = random.Random(42)
        state = methylate_dna(dna, cell_type="mammal", methylase="cpg", rng=rng)

        island_mean = sum(p for pos, p in state.positions.items() if pos < 300) / \
                      max(1, sum(1 for pos in state.positions if pos < 300))
        nonisland_mean = sum(p for pos, p in state.positions.items() if pos >= 700) / \
                         max(1, sum(1 for pos in state.positions if pos >= 700))
        # Island mean < 0.25, non-island mean > 0.65
        assert island_mean < 0.25
        assert nonisland_mean > 0.65
        assert nonisland_mean > island_mean

    def test_unknown_cell_type_raises(self):
        """Unknown cell type raises ValueError."""
        with pytest.raises(ValueError):
            methylate_dna("GATC", cell_type="plant")

    def test_unknown_methylase_raises(self):
        """Unknown methylase raises ValueError."""
        with pytest.raises(ValueError):
            methylate_dna("GATC", cell_type="ecoli", methylase="unknown")


# ============================================================================
# Histone modification tests
# ============================================================================

class TestHistoneMarks:
    """Verifies histone modification addition."""

    def test_add_histone_marks_basic(self):
        """Histone modification addition: promoter + gene body + heterochromatin."""
        dna = "A" * 1000
        gene_positions = [
            {"name": "geneA", "start": 100, "end": 500, "promoter": (50, 100)},
            {"name": "geneB", "start": 600, "end": 900, "promoter": (550, 600)},
        ]
        marks = add_histone_marks(dna, gene_positions, "eukaryote")
        assert len(marks) > 0
        # Should have promoter marks (H3K4me3 or H3K27me3)
        promoter_marks = [m for m in marks if m.mark in ("H3K4me3", "H3K27me3")]
        assert len(promoter_marks) >= 2  # one per gene
        # Should have a heterochromatin mark H3K9me3
        h3k9me3_marks = [m for m in marks if m.mark == "H3K9me3"]
        assert len(h3k9me3_marks) >= 1

    def test_active_gene_gets_h3k4me3(self):
        """Active gene (index 0) gets H3K4me3."""
        dna = "A" * 1000
        gene_positions = [
            {"name": "geneA", "start": 100, "end": 500, "promoter": (50, 100)},
        ]
        marks = add_histone_marks(dna, gene_positions, "eukaryote")
        h3k4me3 = [m for m in marks if m.mark == "H3K4me3"]
        assert len(h3k4me3) >= 1
        # H3K4me3 should be in the promoter region
        assert 50 <= h3k4me3[0].position < 100

    def test_active_gene_gets_h3k36me3(self):
        """Active gene gets H3K36me3 (gene body elongation mark)."""
        dna = "A" * 1000
        gene_positions = [
            {"name": "geneA", "start": 100, "end": 500, "promoter": (50, 100)},
        ]
        marks = add_histone_marks(dna, gene_positions, "eukaryote")
        h3k36me3 = [m for m in marks if m.mark == "H3K36me3"]
        assert len(h3k36me3) >= 1
        # H3K36me3 should be within the gene body
        assert 100 <= h3k36me3[0].position < 500

    def test_repressed_gene_gets_h3k27me3(self):
        """Repressed gene (index 1) gets H3K27me3."""
        dna = "A" * 1000
        gene_positions = [
            {"name": "geneA", "start": 100, "end": 500, "promoter": (50, 100)},
            {"name": "geneB", "start": 600, "end": 900, "promoter": (550, 600)},
        ]
        marks = add_histone_marks(dna, gene_positions, "eukaryote")
        h3k27me3 = [m for m in marks if m.mark == "H3K27me3"]
        assert len(h3k27me3) >= 1
        # H3K27me3 should be in the geneB promoter region
        assert 550 <= h3k27me3[0].position < 600

    def test_heterochromatin_at_ends(self):
        """Heterochromatin H3K9me3 is at both ends of the DNA."""
        dna = "A" * 1000
        gene_positions = []
        marks = add_histone_marks(dna, gene_positions, "eukaryote")
        h3k9me3 = [m for m in marks if m.mark == "H3K9me3"]
        assert len(h3k9me3) >= 2
        # One near the beginning
        assert any(m.position < 50 for m in h3k9me3)
        # One near the end
        assert any(m.position > 950 for m in h3k9me3)

    def test_ecoli_no_histone_marks(self):
        """E. coli has no histone modifications."""
        dna = "A" * 1000
        gene_positions = [
            {"name": "geneA", "start": 100, "end": 500},
        ]
        marks = add_histone_marks(dna, gene_positions, "ecoli")
        assert marks == []

    def test_mark_levels_in_range(self):
        """Histone modification levels are in the [0, 1] range."""
        dna = "A" * 1000
        gene_positions = [
            {"name": "geneA", "start": 100, "end": 500, "promoter": (50, 100)},
            {"name": "geneB", "start": 600, "end": 900, "promoter": (550, 600)},
        ]
        marks = add_histone_marks(dna, gene_positions, "eukaryote")
        for mark in marks:
            assert 0.0 <= mark.level <= 1.0


# ============================================================================
# Chromatin accessibility tests
# ============================================================================

class TestAccessibility:
    """Verifies chromatin accessibility computation."""

    def test_methylation_reduces_accessibility(self):
        """Methylation reduces accessibility."""
        chromatin = ChromatinState(
            methylation=MethylationState(positions={100: 1.0}),
            histone_marks=[],
        )
        acc = calculate_accessibility(chromatin)
        assert 100 in acc
        assert acc[100] < 0.5  # methylation → low accessibility

    def test_h3k4me3_increases_accessibility(self):
        """H3K4me3 increases accessibility."""
        chromatin = ChromatinState(
            methylation=MethylationState(),
            histone_marks=[
                HistoneMark(position=200, mark="H3K4me3", level=1.0),
            ],
        )
        acc = calculate_accessibility(chromatin)
        assert 200 in acc
        assert acc[200] > 0.5  # H3K4me3 → high accessibility

    def test_h3k27me3_reduces_accessibility(self):
        """H3K27me3 reduces accessibility."""
        chromatin = ChromatinState(
            methylation=MethylationState(),
            histone_marks=[
                HistoneMark(position=300, mark="H3K27me3", level=1.0),
            ],
        )
        acc = calculate_accessibility(chromatin)
        assert 300 in acc
        assert acc[300] < 0.5  # H3K27me3 → low accessibility

    def test_h3k9me3_reduces_accessibility(self):
        """H3K9me3 reduces accessibility."""
        chromatin = ChromatinState(
            methylation=MethylationState(),
            histone_marks=[
                HistoneMark(position=400, mark="H3K9me3", level=1.0),
            ],
        )
        acc = calculate_accessibility(chromatin)
        assert 400 in acc
        assert acc[400] < 0.5  # H3K9me3 → low accessibility

    def test_accessibility_in_range(self):
        """Accessibility values are in the [0, 1] range."""
        chromatin = ChromatinState(
            methylation=MethylationState(positions={100: 1.0, 200: 0.5}),
            histone_marks=[
                HistoneMark(position=100, mark="H3K4me3", level=1.0),
                HistoneMark(position=200, mark="H3K27me3", level=1.0),
                HistoneMark(position=300, mark="H3K9me3", level=1.0),
            ],
        )
        acc = calculate_accessibility(chromatin)
        for v in acc.values():
            assert 0.0 <= v <= 1.0

    def test_combined_marks(self):
        """Combined marks: methylation + H3K4me3 at the same position."""
        chromatin = ChromatinState(
            methylation=MethylationState(positions={100: 0.5}),
            histone_marks=[
                HistoneMark(position=100, mark="H3K4me3", level=1.0),
            ],
        )
        acc = calculate_accessibility(chromatin)
        # acc = 0.5 - 0.4*0.5 + 0.5*1.0 = 0.5 - 0.2 + 0.5 = 0.8
        assert abs(acc[100] - 0.8) < 0.01


# ============================================================================
# Expression modifier tests
# ============================================================================

class TestExpressionModifier:
    """Verifies gene expression modifier computation."""

    def test_no_marks_normal_expression(self):
        """Expression modifier = 1.0 (normal) when there are no marks."""
        chromatin = ChromatinState(
            methylation=MethylationState(),
            histone_marks=[],
        )
        gene_positions = [
            {"name": "geneA", "start": 100, "end": 500, "promoter": (50, 100)},
        ]
        modifiers = calculate_expression_modifier(chromatin, gene_positions)
        assert "geneA" in modifiers
        assert abs(modifiers["geneA"] - 1.0) < 0.01

    def test_methylation_reduces_expression(self):
        """DNA methylation reduces gene expression (~70%, Bird 2002)."""
        # Promoter 100% methylated
        chromatin = ChromatinState(
            methylation=MethylationState(
                positions={75: 1.0},  # methylation within the promoter
                methylase="cpg",
                total_sites=1,
                methylated_sites=1,
            ),
            histone_marks=[],
        )
        gene_positions = [
            {"name": "geneA", "start": 100, "end": 500, "promoter": (50, 100)},
        ]
        modifiers = calculate_expression_modifier(chromatin, gene_positions)
        # 70% reduction: modifier = 1.0 - 0.7*1.0 = 0.3
        assert modifiers["geneA"] < 0.5
        assert abs(modifiers["geneA"] - 0.3) < 0.01

    def test_h3k4me3_increases_expression(self):
        """H3K4me3 increases gene expression."""
        chromatin = ChromatinState(
            methylation=MethylationState(),
            histone_marks=[
                HistoneMark(position=75, mark="H3K4me3", level=1.0),
            ],
        )
        gene_positions = [
            {"name": "geneA", "start": 100, "end": 500, "promoter": (50, 100)},
        ]
        modifiers = calculate_expression_modifier(chromatin, gene_positions)
        # H3K4me3 score=+0.5: modifier = 1.0 + 0.5*1.0 = 1.5
        assert modifiers["geneA"] > 1.0
        assert abs(modifiers["geneA"] - 1.5) < 0.01

    def test_h3k27me3_represses_expression(self):
        """H3K27me3 represses gene expression."""
        chromatin = ChromatinState(
            methylation=MethylationState(),
            histone_marks=[
                HistoneMark(position=75, mark="H3K27me3", level=1.0),
            ],
        )
        gene_positions = [
            {"name": "geneA", "start": 100, "end": 500, "promoter": (50, 100)},
        ]
        modifiers = calculate_expression_modifier(chromatin, gene_positions)
        # H3K27me3 score=-0.7: modifier = 1.0 - 0.7*1.0 = 0.3
        assert modifiers["geneA"] < 1.0
        assert abs(modifiers["geneA"] - 0.3) < 0.01

    def test_h3k9me3_represses_expression(self):
        """H3K9me3 strongly represses gene expression (heterochromatin)."""
        chromatin = ChromatinState(
            methylation=MethylationState(),
            histone_marks=[
                HistoneMark(position=75, mark="H3K9me3", level=1.0),
            ],
        )
        gene_positions = [
            {"name": "geneA", "start": 100, "end": 500, "promoter": (50, 100)},
        ]
        modifiers = calculate_expression_modifier(chromatin, gene_positions)
        # H3K9me3 score=-0.9: modifier = 1.0 - 0.9*1.0 = 0.1
        assert modifiers["geneA"] < 0.3

    def test_h3k36me3_increases_expression(self):
        """H3K36me3 slightly increases gene expression (transcription elongation)."""
        chromatin = ChromatinState(
            methylation=MethylationState(),
            histone_marks=[
                HistoneMark(position=300, mark="H3K36me3", level=1.0),
            ],
        )
        gene_positions = [
            {"name": "geneA", "start": 100, "end": 500, "promoter": (50, 100)},
        ]
        modifiers = calculate_expression_modifier(chromatin, gene_positions)
        # H3K36me3 score=+0.3: modifier = 1.0 + 0.3*1.0 = 1.3
        assert modifiers["geneA"] > 1.0
        assert abs(modifiers["geneA"] - 1.3) < 0.01

    def test_modifier_in_range(self):
        """Expression modifiers are in the [0, 2] range."""
        chromatin = ChromatinState(
            methylation=MethylationState(positions={75: 1.0, 200: 1.0}),
            histone_marks=[
                HistoneMark(position=75, mark="H3K4me3", level=1.0),
                HistoneMark(position=300, mark="H3K9me3", level=1.0),
            ],
        )
        gene_positions = [
            {"name": "geneA", "start": 100, "end": 500, "promoter": (50, 100)},
        ]
        modifiers = calculate_expression_modifier(chromatin, gene_positions)
        for v in modifiers.values():
            assert 0.0 <= v <= 2.0

    def test_multiple_genes(self):
        """Multiple genes have independently computed expression modifiers."""
        chromatin = ChromatinState(
            methylation=MethylationState(),
            histone_marks=[
                HistoneMark(position=75, mark="H3K4me3", level=1.0),   # geneA promoter
                HistoneMark(position=550, mark="H3K27me3", level=1.0),  # geneB promoter
            ],
        )
        gene_positions = [
            {"name": "geneA", "start": 100, "end": 500, "promoter": (50, 100)},
            {"name": "geneB", "start": 600, "end": 900, "promoter": (500, 600)},
        ]
        modifiers = calculate_expression_modifier(chromatin, gene_positions)
        assert "geneA" in modifiers
        assert "geneB" in modifiers
        # geneA has H3K4me3 → expression increased
        assert modifiers["geneA"] > 1.0
        # geneB has H3K27me3 → expression repressed
        assert modifiers["geneB"] < 1.0


# ============================================================================
# Histone mark type data completeness tests
# ============================================================================

class TestHistoneMarkTypes:
    """Verifies histone modification type data completeness."""

    def test_all_required_marks_present(self):
        """All required histone modification types are present."""
        required = ["H3K4me3", "H3K27me3", "H3K36me3", "H3K9me3", "H3K27ac"]
        for mark in required:
            assert mark in HISTONE_MARK_TYPES, f"missing mark {mark}"

    def test_mark_info_fields(self):
        """Each modification type contains complete fields."""
        for mark, info in HISTONE_MARK_TYPES.items():
            assert "effect" in info, f"{mark} missing 'effect'"
            assert "location" in info, f"{mark} missing 'location'"
            assert "score" in info, f"{mark} missing 'score'"

    def test_score_range(self):
        """score is in the [-1, 1] range."""
        for mark, info in HISTONE_MARK_TYPES.items():
            assert -1.0 <= info["score"] <= 1.0, \
                f"{mark} score {info['score']} out of [-1, 1]"

    def test_activating_marks_positive_score(self):
        """Activating marks have score > 0."""
        assert HISTONE_MARK_TYPES["H3K4me3"]["score"] > 0
        assert HISTONE_MARK_TYPES["H3K27ac"]["score"] > 0
        assert HISTONE_MARK_TYPES["H3K36me3"]["score"] > 0

    def test_repressing_marks_negative_score(self):
        """Repressing marks have score < 0."""
        assert HISTONE_MARK_TYPES["H3K27me3"]["score"] < 0
        assert HISTONE_MARK_TYPES["H3K9me3"]["score"] < 0

    def test_mark_effects(self):
        """Modification effect fields are correct."""
        assert HISTONE_MARK_TYPES["H3K4me3"]["effect"] == "activating"
        assert HISTONE_MARK_TYPES["H3K27me3"]["effect"] == "repressing"
        assert HISTONE_MARK_TYPES["H3K36me3"]["effect"] == "elongation"
        assert HISTONE_MARK_TYPES["H3K9me3"]["effect"] == "heterochromatin"
        assert HISTONE_MARK_TYPES["H3K27ac"]["effect"] == "activating"

    def test_mark_locations(self):
        """Modification location fields are correct."""
        assert HISTONE_MARK_TYPES["H3K4me3"]["location"] == "promoter"
        assert HISTONE_MARK_TYPES["H3K27me3"]["location"] == "promoter"
        assert HISTONE_MARK_TYPES["H3K36me3"]["location"] == "gene_body"
        assert HISTONE_MARK_TYPES["H3K9me3"]["location"] == "any"
        assert HISTONE_MARK_TYPES["H3K27ac"]["location"] == "enhancer"


# ============================================================================
# Dataclass tests
# ============================================================================

class TestDataclasses:
    """Verifies dataclass basic behavior."""

    def test_methylation_state_defaults(self):
        """MethylationState defaults."""
        state = MethylationState()
        assert state.positions == {}
        assert state.methylase == "custom"
        assert state.total_sites == 0
        assert state.methylated_sites == 0

    def test_histone_mark_fields(self):
        """HistoneMark fields."""
        mark = HistoneMark(position=100, mark="H3K4me3", level=0.8)
        assert mark.position == 100
        assert mark.mark == "H3K4me3"
        assert mark.level == 0.8

    def test_chromatin_state_defaults(self):
        """ChromatinState defaults."""
        state = ChromatinState(methylation=MethylationState())
        assert state.histone_marks == []
        assert state.chromatin_accessibility == {}
        assert state.expression_modifier == {}

    def test_chromatin_state_with_data(self):
        """ChromatinState carries data."""
        meth = MethylationState(positions={10: 0.9}, methylase="dam", total_sites=1)
        marks = [HistoneMark(position=50, mark="H3K4me3", level=1.0)]
        state = ChromatinState(methylation=meth, histone_marks=marks)
        assert state.methylation.methylase == "dam"
        assert len(state.histone_marks) == 1
        assert state.histone_marks[0].mark == "H3K4me3"
