"""PAMIndex multi-bucket seed-and-extend index completeness tests.

Verifies the ``PAMIndex`` multi-bucket k-mer hash index in src/helixlang/crispr.py:

1. **Completeness** (core fix): when ``max_mismatches < num_buckets``, indexed results
   are **fully identical** to the :func:`off_target_score` full scan (no false negatives).
   The pigeonhole principle guarantees: if a candidate differs from the guide at
   ≤ M < B (num buckets) positions, then among the B non-overlapping buckets at least
   B − M > 0 buckets match completely, so the index must hit. These tests cover
   off-targets with mismatches in the first K nt (the old single-bucket implementation
   missed them; the multi-bucket implementation fixes this defect).

2. **Degenerate fallback**: when ``max_mismatches >= num_buckets``, the index may miss
   (mismatches spread evenly across all buckets); then :meth:`search_with_fallback` and
   :func:`off_target_score_indexed` automatically trigger a full-scan fallback, and the
   results are still fully identical to the full scan.

3. **Boundary values** and **multiple Cas variants** (SpCas9 / SaCas9 / Cas12a).

References:
- Hsu 2013 Nat Biotechnol 31:827-832 (off-target scoring)
- Jinek 2012 Science 337:816-821 (SpCas9 NGG PAM)
- Zetsche 2015 Cell 163:759-771 (Cas12a TTTV PAM)
"""
from __future__ import annotations

import pytest

from helixlang.crispr import (
    GuideRNA,
    OffTargetSite,
    PAMIndex,
    off_target_score,
    off_target_score_indexed,
)

# ============================================================================
# Helpers: build controlled genomes (containing only the expected PAM sites)
# ============================================================================

def _build_spcas9_genome() -> str:
    """Build an SpCas9 genome: 1 on-target + 2 off-targets.

    Structure (all-A background, only the forward strand has NGG PAMs):
    - site 1 (on-target): spacer = A×20, PAM = TGG  → 0 mismatches
    - site 2 (off-tail):  spacer = A×15+CG+A×3, PAM = AGG → 2 mm @ pos 15,16
    - site 3 (off-head):  spacer = A×5+C+A×14, PAM = CGG  → 1 mm @ pos 5

    The all-A/T/C background contains no GG (except the 3 expected PAMs), and the
    reverse complement strand has no GG either, so find_pam_sites(both_strands=True)
    returns only these 3 forward-strand sites.
    """
    on_target = "A" * 20 + "TGG"
    off_tail = "A" * 15 + "CG" + "A" * 3 + "AGG"
    off_head = "A" * 5 + "C" + "A" * 14 + "CGG"
    return on_target + off_tail + off_head


def _build_spcas9_guide() -> GuideRNA:
    """Build an SpCas9 guide: spacer = A×20, PAM = TGG, pos = 0."""
    return GuideRNA(
        spacer="A" * 20,
        pam="TGG",
        pam_position="3prime",
        cas_variant="SpCas9",
        target_position=0,
        strand="+",
    )


def _positions(off_targets: list[OffTargetSite]) -> set[int]:
    """Extract the set of off-target site positions."""
    return {ot.position for ot in off_targets}


def _keys(off_targets: list[OffTargetSite]) -> set[tuple[int, str]]:
    """Extract the set of off-target (position, strand) keys."""
    return {(ot.position, ot.strand) for ot in off_targets}


# ============================================================================
# Completeness: max_mismatches < num_buckets → index == full scan (no false negatives)
# ============================================================================

class TestPAMIndexCompleteness:
    """In the safe range, the multi-bucket index is fully identical to the full scan (fixes the old single-bucket false-negative defect)."""

    def test_num_buckets_for_spcas9(self):
        """SpCas9 (20nt, K=5) → 4 buckets."""
        index = PAMIndex(_build_spcas9_genome(), "SpCas9")
        assert index.K == 5
        assert index.spacer_len == 20
        assert index.num_buckets == 4

    def test_index_equals_full_scan_low_mismatches(self):
        """max_mismatches=3 < num_buckets=4 → index == full scan (including head mismatch)."""
        genome = _build_spcas9_genome()
        guide = _build_spcas9_guide()
        full = off_target_score(guide, genome, max_mismatches=3)
        index = PAMIndex(genome, "SpCas9")
        indexed = index.search(guide, max_mismatches=3)
        assert _keys(indexed) == _keys(full), (
            f"indexed {sorted(_keys(indexed))} != full {sorted(_keys(full))}"
        )

    def test_index_finds_head_mismatch(self):
        """An off-target with a mismatch in the first K nt is found by the multi-bucket index (fixes the old defect).

        off-head: 1 mm @ pos 5. K=5 → bucket 0=[0,5) matches completely, the index hits via bucket 0.
        """
        genome = _build_spcas9_genome()
        guide = _build_spcas9_guide()
        index = PAMIndex(genome, "SpCas9")
        indexed = index.search(guide, max_mismatches=3)
        # off-head PAM CGG position = 46+20 = 66
        assert 66 in _positions(indexed), (
            "multi-bucket index should find off-targets with mismatch in the first K nt (fixes old single-bucket defect)"
        )

    def test_index_finds_tail_mismatch(self):
        """An off-target with a tail mismatch is found by the index."""
        genome = _build_spcas9_genome()
        guide = _build_spcas9_guide()
        index = PAMIndex(genome, "SpCas9")
        indexed = index.search(guide, max_mismatches=3)
        # off-tail PAM AGG position = 23+20 = 43
        assert 43 in _positions(indexed)

    def test_index_zero_mismatches_both_empty(self):
        """max_mismatches=0 → both empty (only on-target is excluded)."""
        genome = _build_spcas9_genome()
        guide = _build_spcas9_guide()
        full = off_target_score(guide, genome, max_mismatches=0)
        index = PAMIndex(genome, "SpCas9")
        indexed = index.search(guide, max_mismatches=0)
        assert len(full) == 0
        assert len(indexed) == 0

    def test_index_scores_match_full_scan(self):
        """In the complete range, index and full-scan off-target scores match."""
        genome = _build_spcas9_genome()
        guide = _build_spcas9_guide()
        full = off_target_score(guide, genome, max_mismatches=3)
        index = PAMIndex(genome, "SpCas9")
        indexed = index.search(guide, max_mismatches=3)
        idx_by_key = {(ot.position, ot.strand): ot.score for ot in indexed}
        for ot in full:
            key = (ot.position, ot.strand)
            assert key in idx_by_key, f"missing {key} in indexed"
            assert ot.score == pytest.approx(idx_by_key[key]), (
                f"score mismatch at {key}: full={ot.score} vs indexed={idx_by_key[key]}"
            )


# ============================================================================
# Degenerate case: max_mismatches >= num_buckets → index may miss; fallback required
# ============================================================================

def _build_spread_mismatch_genome() -> str:
    """Construct an off-target with a mismatch in every bucket (index misses it; fallback required).

    on-target: A×20 + TGG
    off-spread: 1 mismatch per bucket 0/1/2/3 (pos 2, 7, 12, 17) → 4 mm, all buckets miss
    """
    on_target = "A" * 20 + "TGG"
    # 4 mm @ pos 2, 7, 12, 17 (one per K=5 bucket, all buckets miss)
    block = "AA" + "C" + "AA"  # 5 chars, C in the middle (pos+2)
    off_spread = block * 4 + "AGG"  # 20 + 3
    return on_target + off_spread


class TestPAMIndexFallback:
    """When max_mismatches >= num_buckets the index misses; fallback guarantees completeness."""

    def test_index_misses_spread_mismatch(self):
        """1 mismatch in each of the 4 buckets → all buckets miss, the index reports false negatives."""
        genome = _build_spread_mismatch_genome()
        guide = _build_spcas9_guide()
        index = PAMIndex(genome, "SpCas9")
        full = off_target_score(guide, genome, max_mismatches=4)
        # full scan can find off-spread (4 mm <= 4)
        assert len(full) >= 1
        indexed = index.search(guide, max_mismatches=4)
        # index false negatives: mismatches spread across all 4 buckets
        assert _keys(indexed).issubset(_keys(full))
        assert len(indexed) < len(full), "index should miss when all buckets mismatch"

    def test_search_with_fallback_equals_full_scan(self):
        """search_with_fallback always == full scan (complete)."""
        genome = _build_spread_mismatch_genome()
        guide = _build_spcas9_guide()
        full = off_target_score(guide, genome, max_mismatches=4)
        index = PAMIndex(genome, "SpCas9")
        recovered = index.search_with_fallback(guide, max_mismatches=4)
        assert _keys(recovered) == _keys(full), (
            f"fallback {sorted(_keys(recovered))} != full {sorted(_keys(full))}"
        )

    def test_off_target_score_indexed_always_complete(self):
        """off_target_score_indexed falls back automatically; result == full scan."""
        genome = _build_spread_mismatch_genome()
        guide = _build_spcas9_guide()
        for mm in (0, 1, 2, 3, 4, 5, 6):
            full = off_target_score(guide, genome, max_mismatches=mm)
            index = PAMIndex(genome, "SpCas9")
            via_func = off_target_score_indexed(guide, index, max_mismatches=mm)
            assert _keys(via_func) == _keys(full), (
                f"mm={mm}: indexed {sorted(_keys(via_func))} != "
                f"full {sorted(_keys(full))}"
            )

    def test_index_subset_of_full_scan_always(self):
        """For any max_mismatches, indexed results are always ⊆ full scan."""
        genome = _build_spcas9_genome()
        guide = _build_spcas9_guide()
        index = PAMIndex(genome, "SpCas9")
        for mm in (0, 1, 2, 3, 5, 9, 12, 15):
            full = off_target_score(guide, genome, max_mismatches=mm)
            indexed = index.search(guide, max_mismatches=mm)
            assert _keys(indexed).issubset(_keys(full)), (
                f"mm={mm}: indexed not subset of full"
            )


# ============================================================================
# Boundary values: max_mismatches near num_buckets
# ============================================================================

class TestPAMIndexBoundaryValues:
    """Boundary behavior of max_mismatches near num_buckets (=4)."""

    def test_max_mismatches_equals_num_buckets_minus_one(self):
        """max_mismatches = 3 = num_buckets-1 → index complete (== full scan)."""
        genome = _build_spcas9_genome()
        guide = _build_spcas9_guide()
        full = off_target_score(guide, genome, max_mismatches=3)
        index = PAMIndex(genome, "SpCas9")
        indexed = index.search(guide, max_mismatches=3)
        assert _keys(indexed) == _keys(full)

    def test_max_mismatches_equals_num_buckets(self):
        """max_mismatches = 4 = num_buckets → may miss; fallback is complete."""
        genome = _build_spread_mismatch_genome()
        guide = _build_spcas9_guide()
        full = off_target_score(guide, genome, max_mismatches=4)
        index = PAMIndex(genome, "SpCas9")
        recovered = index.search_with_fallback(guide, max_mismatches=4)
        assert _keys(recovered) == _keys(full)


# ============================================================================
# Different Cas variants
# ============================================================================

class TestPAMIndexCasVariants:
    """Multi-bucket index completeness for SaCas9 and Cas12a."""

    def test_sa_cas9_index_complete_low_mismatches(self):
        """SaCas9 (spacer=21, PAM=NNGRRT) max_mismatches=3 < 5 buckets → complete."""
        spacer_a = "A" * 21
        off = "A" * 5 + "C" + "A" * 15  # 1 mm @ pos 5
        genome = spacer_a + "AAGAAT" + off + "AAGAAT"
        guide = GuideRNA(
            spacer=spacer_a, pam="AAGAAT", pam_position="3prime",
            cas_variant="SaCas9", target_position=0, strand="+",
        )
        full = off_target_score(guide, genome, max_mismatches=3)
        index = PAMIndex(genome, "SaCas9")
        indexed = index.search(guide, max_mismatches=3)
        assert _keys(indexed) == _keys(full)
        assert index.num_buckets == 5  # ceil(21/5)

    def test_cas12a_index_complete_low_mismatches(self):
        """Cas12a (spacer=23, PAM=TTTV 5') max_mismatches=3 < 5 buckets → complete."""
        spacer_a = "A" * 23
        off_head = "A" * 5 + "C" + "A" * 17  # 1 mm @ pos 5
        off_tail = "A" * 16 + "CG" + "A" * 5  # 2 mm @ pos 16,17
        genome = ("TTTC" + spacer_a
                  + "TTTC" + off_tail
                  + "TTTC" + off_head)
        guide = GuideRNA(
            spacer=spacer_a, pam="TTTC", pam_position="5prime",
            cas_variant="Cas12a", target_position=4, strand="+",
        )
        full = off_target_score(guide, genome, max_mismatches=3)
        index = PAMIndex(genome, "Cas12a")
        indexed = index.search(guide, max_mismatches=3)
        # multi-bucket index finds head and tail mismatches (fixes the old defect)
        assert _keys(indexed) == _keys(full), (
            f"indexed {sorted(_keys(indexed))} != full {sorted(_keys(full))}"
        )
        assert index.num_buckets == 5  # ceil(23/5)

    def test_cas12a_fallback_complete_high_mismatches(self):
        """Cas12a max_mismatches=6 >= 5 buckets → fallback is complete."""
        spacer_a = "A" * 23
        off_head = "A" * 5 + "C" + "A" * 17
        genome = "TTTC" + spacer_a + "TTTC" + off_head
        guide = GuideRNA(
            spacer=spacer_a, pam="TTTC", pam_position="5prime",
            cas_variant="Cas12a", target_position=4, strand="+",
        )
        full = off_target_score(guide, genome, max_mismatches=6)
        index = PAMIndex(genome, "Cas12a")
        recovered = index.search_with_fallback(guide, max_mismatches=6)
        assert _keys(recovered) == _keys(full)

    def test_unknown_cas_variant_raises(self):
        """Unknown Cas variant raises BioError (integrates with the unified exception system)."""
        from helixlang.errors import BioError
        with pytest.raises(BioError):
            PAMIndex("ACGT" * 20, "NotACas")

    def test_k_too_small_raises(self):
        """K < 4 raises BioError."""
        from helixlang.errors import BioError
        with pytest.raises(BioError):
            PAMIndex("ACGT" * 20, "SpCas9", K=2)


# ============================================================================
# Index miscellaneous boundaries
# ============================================================================

class TestPAMIndexMisc:
    """Index miscellaneous boundaries."""

    def test_num_sites_property(self):
        """num_sites reflects the number of sites in the index."""
        genome = _build_spcas9_genome()
        index = PAMIndex(genome, "SpCas9")
        # 3 PAM sites (on-target + off-tail + off-head)
        assert index.num_sites == 3

    def test_off_target_score_indexed_equals_index_search_when_safe(self):
        """In the safe range, off_target_score_indexed == index.search."""
        genome = _build_spcas9_genome()
        guide = _build_spcas9_guide()
        index = PAMIndex(genome, "SpCas9")
        direct = index.search(guide, max_mismatches=3)
        via_func = off_target_score_indexed(guide, index, max_mismatches=3)
        assert _keys(direct) == _keys(via_func)
        assert [ot.score for ot in direct] == [ot.score for ot in via_func]

    def test_custom_k_overrides_default(self):
        """Explicit K overrides the default."""
        index = PAMIndex(_build_spcas9_genome(), "SpCas9", K=7)
        assert index.K == 7
        assert index.num_buckets == 3  # ceil(20/7)
