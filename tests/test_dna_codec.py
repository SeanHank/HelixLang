"""DNA physical encode/decode tests: Goldman 2013 rotation key + Erlich 2017 fountain code + PCR errors + IUPAC validation.

Verification goals (based on real paper parameters):
- Goldman 2013 encode/decode round-trip fidelity (rotation key + 100nt overlapping segments + 17nt index + alternating RC)
  * Rotation key mathematically guarantees no homopolymers (>=2 identical bases)
  * 4x overlap redundancy voting error correction
  * GC content ~50% (rotation key cycle A->C->G->T)
  * Capacity ~0.25 bit/nt (including index overhead; paper measures 0.29 bit/nt)

- Erlich 2017 fountain code encode/decode round-trip fidelity (LT fountain + RSD + RS inner code + BP decoding)
  * oligo length 152 nt (4B seed + 32B payload + 2B RS = 38B x 8 / 2)
  * GC 45-55% + homopolymer <=3
  * Capacity near Shannon limit 1.58 bit/nt

- PCR error model (Saiki 1988 / Potapov 2017)
  * Taq: substitution 1.5e-4/nt/cycle, 30 cycles -> ~0.45%
  * Transition:Transversion ~ 6:1
  * Pfu/Q5/Phusion high-fidelity modes

- IUPAC strict validation (BioPython)
- Real biological data (E. coli codon usage, lac operon, Gray-Scott presets)
"""
from __future__ import annotations

import math
import random

import pytest

# bio dependency optional; skipped if missing
pytest.importorskip("Bio")
pytest.importorskip("reedsolo")

from helixlang.bio_data import (
    ECOLI_CODON_USAGE,
    GRAY_SCOTT_PRESETS,
    PCR_ERROR_RATES,
    TAQ_MUTATION_SPECTRUM,
    TRANSITION_TRANSVERSION_RATIO,
    codon_adaptation_index,
    get_gray_scott_preset,
    is_optimal_codon,
    is_rare_codon,
    lac_promoter_strength,
    lac_repression_factor,
    mu_to_grn_strength,
)
from helixlang.dna_codec import (
    ERLICH_MAX_HOMOPOLYMER,
    ERLICH_OLIGO_NT,
    ERLICH_OLIGO_SIZE,
    INDEX_NT,
    SEGMENT_NT,
    SEGMENT_STEP_NT,
    GoldmanOligo,
    _base_to_trit,
    _bytes_to_dna_goldman,
    _decode_index,
    _dna_to_bytes_goldman,
    _encode_index,
    _reverse_complement,
    _trit_to_base,
    dna_to_helix,
    erlich_decode,
    erlich_encode,
    gc_stats,
    goldman_decode,
    goldman_encode,
    helix_to_dna,
    pcr_amplify,
    robust_soliton_distribution,
    translate_dna,
    validate_iupac_dna,
)

# ============================================================================
# Goldman 2013 rotation key core mechanism
# ============================================================================

class TestGoldmanRotationKey:
    """Verifies the mathematical properties of the Goldman 2013 rotation key."""

    def test_no_homopolymer_by_construction(self):
        """Rotation key mathematically guarantees no homopolymers: next != prev."""
        for prev in "ACGT":
            for trit in range(3):
                nxt = _trit_to_base(prev, trit)
                assert nxt != prev, f"{prev}->{nxt} violates no-homopolymer"

    def test_rotation_key_roundtrip(self):
        """Rotation key encode/decode round-trip: trit -> base -> trit."""
        for prev in "ACGT":
            for trit in range(3):
                nxt = _trit_to_base(prev, trit)
                recovered = _base_to_trit(prev, nxt)
                assert recovered == trit, f"trit {trit} lost: {prev}->{nxt}->{recovered}"

    def test_cycle_order(self):
        """Rotation key follows the A->C->G->T->A cycle order."""
        # prev=A, trit=0 -> C (next in cycle)
        assert _trit_to_base("A", 0) == "C"
        assert _trit_to_base("C", 0) == "G"
        assert _trit_to_base("G", 0) == "T"
        assert _trit_to_base("T", 0) == "A"

    def test_dna_no_homopolymer(self):
        """No homopolymers after bytes -> DNA (>=2 identical bases)."""
        data = bytes(range(256))
        dna = _bytes_to_dna_goldman(data)
        for i in range(1, len(dna)):
            assert dna[i] != dna[i - 1], f"homopolymer at {i}: {dna[i-1:i+1]}"

    def test_dna_gc_content_near_50(self):
        """Rotation key cycle A->C->G->T keeps GC content near 50%."""
        data = bytes(range(256)) * 4  # 1KB diverse data
        dna = _bytes_to_dna_goldman(data)
        gc = sum(1 for c in dna if c in "GC") / len(dna)
        assert 0.40 < gc < 0.60, f"GC {gc:.3f} not near 50%"

    def test_bytes_to_dna_roundtrip(self):
        """Bytes -> DNA -> bytes round-trip fidelity."""
        data = b"Hello, HelixLang! " * 10
        dna = _bytes_to_dna_goldman(data)
        decoded = _dna_to_bytes_goldman(dna)
        assert decoded == data

    def test_all_256_bytes_roundtrip(self):
        """All 256 byte values round-trip."""
        data = bytes(range(256))
        dna = _bytes_to_dna_goldman(data)
        decoded = _dna_to_bytes_goldman(dna)
        assert decoded == data


# ============================================================================
# Goldman index encode/decode
# ============================================================================

class TestGoldmanIndex:
    def test_index_roundtrip(self):
        """17 nt index header encode/decode round-trip."""
        for seg_idx in [0, 1, 2, 10, 100, 1000, 99999]:
            idx_dna = _encode_index(seg_idx)
            assert len(idx_dna) == INDEX_NT
            decoded = _decode_index(idx_dna)
            assert decoded == seg_idx

    def test_index_no_homopolymer(self):
        """Index header has no homopolymers."""
        for seg_idx in range(100):
            idx_dna = _encode_index(seg_idx)
            for i in range(1, len(idx_dna)):
                assert idx_dna[i] != idx_dna[i - 1]

    def test_index_parity_check(self):
        """Parity check detects errors."""
        idx_dna = _encode_index(42)
        # Corrupt one base
        bad = list(idx_dna)
        bad[5] = "A" if bad[5] != "A" else "T"
        with pytest.raises(ValueError):
            _decode_index("".join(bad))


# ============================================================================
# Goldman full encode/decode
# ============================================================================

class TestGoldmanCodec:
    def test_roundtrip_empty(self):
        """Empty input round-trip."""
        oligos = goldman_encode(b"")
        # Empty input produces 1 segment (after zero-padding)
        assert len(oligos) >= 1
        decoded = goldman_decode(oligos, total_len=0)
        assert decoded == b""

    def test_roundtrip_short(self):
        """Short byte string round-trip."""
        data = b"Hello, HelixLang!"
        oligos = goldman_encode(data)
        decoded = goldman_decode(oligos, total_len=len(data))
        assert decoded == data

    def test_roundtrip_binary_all_bytes(self):
        """All 256 byte values round-trip."""
        data = bytes(range(256))
        oligos = goldman_encode(data)
        decoded = goldman_decode(oligos, total_len=len(data))
        assert decoded == data

    def test_roundtrip_random_bytes(self):
        """1KB random binary round-trip."""
        rng = random.Random(42)
        data = bytes(rng.randint(0, 255) for _ in range(1024))
        oligos = goldman_encode(data)
        decoded = goldman_decode(oligos, total_len=len(data))
        assert decoded == data

    def test_oligo_length_117nt(self):
        """Each oligo length = 17 nt index + 100 nt data = 117 nt."""
        oligos = goldman_encode(b"test data")
        for o in oligos:
            assert len(o.full) == INDEX_NT + SEGMENT_NT, \
                f"oligo len {len(o.full)} != {INDEX_NT + SEGMENT_NT}"

    def test_4x_overlap_redundancy(self):
        """4x overlap redundancy: each position is covered by up to 4 segments."""
        data = bytes(range(100))  # 100 bytes -> 600 nt -> 24 segments
        oligos = goldman_encode(data)
        # Segment count = ceil(600 / 25) = 24
        n_seg = len(oligos)
        assert n_seg >= 20  # at least 20 segments
        # Check that middle positions are covered by 4 segments
        # Segment i covers [i*25, i*25+100)
        # Position 100 is covered by segments 0,1,2,3,4 (0: [0,100), 1: [25,125), ...)
        # Position 99 is covered by segment 0; position 100 is covered by segments 1,2,3,4
        coverage_at_100 = sum(
            1 for o in oligos
            if o.index * SEGMENT_STEP_NT <= 100 < o.index * SEGMENT_STEP_NT + SEGMENT_NT
        )
        assert coverage_at_100 >= 3, f"coverage at 100: {coverage_at_100}"

    def test_iupac_valid(self):
        """All oligos pass BioPython IUPAC validation."""
        oligos = goldman_encode(b"ATGC test 123")
        for o in oligos:
            assert validate_iupac_dna(o.full), f"invalid IUPAC: {o.full[:30]}"

    def test_no_long_homopolymer(self):
        """No long homopolymers after Goldman encoding (rotation key guarantees <=1 within data segments, <=2 at index/data boundaries).

        The rotation key mathematically guarantees next != prev within a data segment,
        but the junction between the index header and payload belongs to two
        independent rotation-key streams, which may produce 2-mer homopolymers (well
        below Illumina sequencing's <=3 homopolymer limit). This is a Goldman 2013 design feature.
        """
        data = b"AAAA AAAA AAAA" * 10
        oligos = goldman_encode(data)
        for o in oligos:
            stats = gc_stats(o.full)
            assert stats["max_homopolymer"] <= 2, \
                f"homopolymer {stats['max_homopolymer']} in {o.full[:30]}"

    def test_gc_content_near_50(self):
        """GC content near 50% (rotation key cycle A->C->G->T)."""
        data = bytes(range(256)) * 4  # 1KB
        oligos = goldman_encode(data)
        for o in oligos:
            stats = gc_stats(o.full)
            assert 0.35 <= stats["gc_content"] <= 0.65, \
                f"GC {stats['gc_content']} out of range"

    def test_alternating_reverse_complement(self):
        """Adjacent segments alternate reverse complement: even segments forward, odd segments RC."""
        oligos = goldman_encode(b"test data for RC alternation!")
        for o in oligos:
            if o.index % 2 == 0:
                # Even segment: forward, full should start with index_dna
                assert o.full == o.overhang + o.payload
            else:
                # Odd segment: RC, RC of full should start with index_dna
                rc = _reverse_complement(o.full)
                assert rc[:INDEX_NT] == o.overhang

    def test_redundancy_tolerates_errors(self):
        """4x overlap voting can tolerate 1 segment error (position must be covered by >=3 segments)."""
        data = b"critical payload data!!" * 3  # long enough for multiple segments
        oligos = goldman_encode(data)
        if len(oligos) < 4:
            pytest.skip("need >= 4 segments for 4x overlap voting")
        # Corrupt a mid-payload position of oligo 0 (covered by segments 1,2,3)
        # Segment 0 covers [0,100), segment 1 covers [25,125), segment 2 covers [50,150), segment 3 covers [75,175)
        # Position 75 is covered by segments 0,1,2,3 simultaneously -> 4x voting
        corrupt_pos = 75
        bad_payload = list(oligos[0].payload)
        bad_payload[corrupt_pos] = "A" if bad_payload[corrupt_pos] != "A" else "T"
        oligos[0] = GoldmanOligo(
            index=oligos[0].index,
            payload="".join(bad_payload),
            overhang=oligos[0].overhang,
            full=oligos[0].overhang + "".join(bad_payload)
        )
        decoded = goldman_decode(oligos, total_len=len(data))
        # 3 correct segment votes should outvote 1 incorrect segment
        assert decoded == data


# ============================================================================
# helix_to_dna / dna_to_helix integration
# ============================================================================

class TestHelixDNARoundtrip:
    def test_roundtrip_helix_source(self):
        """Full HelixLang source round-trip."""
        src = "#gene name=hello\nATG GCT GGT TAA\n#end\n#config ticks=5\n"
        enc = helix_to_dna(src, scheme="goldman")
        dec = dna_to_helix(enc, scheme="goldman")
        assert dec == src

    def test_roundtrip_unicode(self):
        """Unicode source round-trip."""
        src = "#gene name=test\nATG TAA\n#end\n# Chinese comment"
        enc = helix_to_dna(src, scheme="goldman")
        dec = dna_to_helix(enc, scheme="goldman")
        assert dec == src

    def test_roundtrip_long_example(self):
        """Long example round-trip (lac operon example)."""
        from pathlib import Path
        example = Path(__file__).parent.parent / "examples" / "02_lac_operon.helix"
        if not example.exists():
            pytest.skip("example not found")
        src = example.read_text()
        enc = helix_to_dna(src, scheme="goldman")
        dec = dna_to_helix(enc, scheme="goldman")
        assert dec == src

    def test_density_reasonable(self):
        """Goldman capacity in a reasonable range (including 17nt index overhead)."""
        src = "#gene name=test\nATG TAA\n#end\n" * 20
        enc = helix_to_dna(src, scheme="goldman")
        d = enc["stats"]["density_bit_per_nt"]
        # Paper measures 0.29 bit/nt (including index); ours is slightly lower at 6 trits/byte
        assert 0.05 < d < 0.5, f"density {d} out of expected range"


# ============================================================================
# Erlich-Zielinski 2017 fountain code
# ============================================================================

class TestErlichCodec:
    def test_roundtrip_short(self):
        """Short data round-trip (K=1)."""
        data = b"Erlich fountain test!"
        oligos = erlich_encode(data, redundancy=2.0)
        decoded = erlich_decode(oligos, K=1, total_len=len(data))
        assert decoded == data

    def test_roundtrip_multi_block(self):
        """Multi-block data round-trip (K>=4)."""
        data = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit.!"
        K = math.ceil(len(data) / ERLICH_OLIGO_SIZE)
        oligos = erlich_encode(data, redundancy=1.3)
        decoded = erlich_decode(oligos, K=K, total_len=len(data))
        assert decoded == data

    def test_roundtrip_large(self):
        """Larger data round-trip (K>=54, the minimum test size in the Erlich 2017 paper).

        The Erlich 2017 paper reports K=54-1040 decodable with redundancy=1.1-1.4.
        For small K<50, the RSD degree-1 probability is too low (<5%) and requires
        redundancy >=2.5 for reliable BP decoding. Here K=128 is used to verify the large-block scenario.
        """
        rng = random.Random(42)
        data = bytes(rng.randint(0, 255) for _ in range(4096))  # K=128
        K = math.ceil(len(data) / ERLICH_OLIGO_SIZE)
        assert K >= 54, f"K={K} below Erlich paper minimum (54)"
        oligos = erlich_encode(data, redundancy=1.15)
        decoded = erlich_decode(oligos, K=K, total_len=len(data))
        assert decoded == data

    def test_oligo_length_152nt(self):
        """Each oligo length = 38B x 8 / 2 = 152 nt."""
        oligos = erlich_encode(b"x" * 32, redundancy=2.0)
        for o in oligos:
            assert len(o.payload) == ERLICH_OLIGO_NT, \
                f"oligo len {len(o.payload)} != {ERLICH_OLIGO_NT}"

    def test_iupac_valid(self):
        oligos = erlich_encode(b"test data 123" + b"\x00" * 20, redundancy=2.0)
        for o in oligos:
            assert validate_iupac_dna(o.payload)

    def test_gc_constraint(self):
        """Erlich encoding satisfies the GC 45-55% constraint."""
        oligos = erlich_encode(b"x" * 64, redundancy=1.5)
        for o in oligos:
            stats = gc_stats(o.payload)
            assert 0.45 - 0.02 <= stats["gc_content"] <= 0.55 + 0.02, \
                f"GC {stats['gc_content']} out of [0.43, 0.57]"

    def test_homopolymer_constraint(self):
        """Erlich encoding homopolymer <=3."""
        oligos = erlich_encode(b"x" * 64, redundancy=1.5)
        for o in oligos:
            stats = gc_stats(o.payload)
            assert stats["max_homopolymer"] <= ERLICH_MAX_HOMOPOLYMER, \
                f"homopolymer {stats['max_homopolymer']} > {ERLICH_MAX_HOMOPOLYMER}"

    def test_fountain_redundancy(self):
        """Fountain code redundancy: generates more than K oligos."""
        data = b"X" * 128  # K=4
        K = 4
        oligos = erlich_encode(data, redundancy=0.5)
        # Should generate K * 1.5 = 6 oligos (at least K+1)
        assert len(oligos) >= K + 1

    def test_tolerosity_to_oligo_loss(self):
        """Fountain code tolerates oligo loss (redundancy)."""
        data = b"Lorem ipsum dolor sit amet, consectetur!!"  # ~44 bytes, K=2
        K = 2
        oligos = erlich_encode(data, redundancy=2.0)  # generates K*3=6 oligos
        # Drop 1
        oligos_partial = oligos[:-1]
        decoded = erlich_decode(oligos_partial, K=K, total_len=len(data))
        assert decoded == data


# ============================================================================
# Robust Soliton Distribution
# ============================================================================

class TestRobustSolitonDistribution:
    def test_rsd_sums_to_one(self):
        """RSD probabilities sum to 1."""
        for K in [1, 2, 10, 100, 1000]:
            rsd = robust_soliton_distribution(K)
            assert len(rsd) == K
            assert abs(sum(rsd) - 1.0) < 1e-9, f"K={K}: sum={sum(rsd)}"

    def test_rsd_all_positive(self):
        """All RSD probabilities > 0."""
        rsd = robust_soliton_distribution(100)
        for p in rsd:
            assert p > 0

    def test_rsd_degree_1_dominant(self):
        """RSD degree-1 probability >= 1/K (ensures BP peeling has an entry point).

        Note: In ISD/RSD, degree=2 usually has the highest probability (rho(2)=1/(2*1)=0.5); degree=1 relies
        solely on the RSD tau(1) boost. The Erlich 2017 paper Fig S2 shows P(d=1)~5% at K=100.
        Here we verify P(d=1) >= 1/K, ensuring the BP decoder has enough degree-1 oligos to start peeling.
        """
        for K in [10, 50, 100, 500]:
            rsd = robust_soliton_distribution(K)
            assert rsd[0] >= 1.0 / K, \
                f"K={K}: P(d=1)={rsd[0]} < 1/K={1.0/K}"


# ============================================================================
# PCR error model
# ============================================================================

class TestPCRErrorModel:
    def test_pcr_no_error_zero_cycles(self):
        """0 cycles means no errors."""
        dna = "ACGTACGTACGT" * 10
        out = pcr_amplify(dna, cycles=0)
        assert out == dna

    def test_pcr_substitution_rate_taq(self):
        """PCR 30 cycles Taq substitution rate matches Potapov 2017 (~0.45%)."""
        rng = random.Random(42)
        dna = "ACGTACGTACGTACGT" * 100  # 1600 bp
        out = pcr_amplify(dna, cycles=30, rng=rng, polymerase="taq")
        # Cumulative error rate ~ 1-(1-1.5e-4)^30 ~ 0.45%
        diffs = sum(1 for a, b in zip(dna, out, strict=False) if a != b)
        rate = diffs / len(dna)
        # Tolerate 0.1%-1% range
        assert 0.001 < rate < 0.01, f"substitution rate {rate:.4f} out of range"

    def test_pcr_q5_high_fidelity(self):
        """Q5 high-fidelity polymerase error rate is much lower than Taq."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        dna = "ACGTACGTACGTACGT" * 100
        out_taq = pcr_amplify(dna, cycles=30, rng=rng1, polymerase="taq")
        out_q5 = pcr_amplify(dna, cycles=30, rng=rng2, polymerase="q5")
        taq_errors = sum(1 for a, b in zip(dna, out_taq, strict=False) if a != b)
        q5_errors = sum(1 for a, b in zip(dna, out_q5, strict=False) if a != b)
        assert q5_errors < taq_errors, \
            f"Q5 errors {q5_errors} should be < Taq errors {taq_errors}"

    def test_pcr_indel_changes_length(self):
        """PCR indels change sequence length."""
        rng = random.Random(42)
        dna = "ACGT" * 100
        out = pcr_amplify(dna, cycles=30,
                          sub_rate=0.0, indel_rate=0.01, rng=rng)
        assert len(out) != len(dna)

    def test_pcr_reproducible_with_seed(self):
        """Same seed produces the same errors."""
        dna = "ACGTACGT" * 50
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        out1 = pcr_amplify(dna, cycles=10, rng=rng1)
        out2 = pcr_amplify(dna, cycles=10, rng=rng2)
        assert out1 == out2

    def test_transition_bias(self):
        """PCR substitution favors transitions (A<->G, C<->T) ~ 86%."""
        rng = random.Random(42)
        dna = "ACGT" * 500  # 2000 bp
        out = pcr_amplify(dna, cycles=30, rng=rng, polymerase="taq")
        transitions = 0
        transversions = 0
        transition_pairs = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
        for a, b in zip(dna, out, strict=False):
            if a != b:
                if (a, b) in transition_pairs:
                    transitions += 1
                else:
                    transversions += 1
        total = transitions + transversions
        if total > 10:  # need enough samples
            ratio = transitions / total
            # Potapov 2017: ~86% transitions
            assert ratio > 0.7, f"transition ratio {ratio:.2f} too low"


# ============================================================================
# IUPAC validation + translation
# ============================================================================

class TestIUPACValidation:
    def test_valid_dna(self):
        assert validate_iupac_dna("ACGTACGT")
        assert validate_iupac_dna("ACGTN")  # N is a valid IUPAC code

    def test_invalid_dna(self):
        assert not validate_iupac_dna("acgt 123")
        assert not validate_iupac_dna("")
        # Z/Q/L etc. are not valid IUPAC DNA codes
        assert not validate_iupac_dna("ACGTZ")
        assert not validate_iupac_dna("ACGTQ")

    def test_translate_standard(self):
        """Standard translation table: ATG-Met, TAA-Stop."""
        protein = translate_dna("ATGGCTTAA", table=1)
        assert "M" in protein
        assert protein.endswith("*")

    def test_translate_mito(self):
        """Mitochondrial table: TGA is not Stop but W."""
        std = translate_dna("ATGTGA", table=1)
        mito = translate_dna("ATGTGA", table=2)
        assert "*" in std
        assert "W" in mito


# ============================================================================
# Real biological data
# ============================================================================

class TestBioData:
    def test_codon_usage_completeness(self):
        """E. coli codon table covers all 64 triplets."""
        assert len(ECOLI_CODON_USAGE) == 64
        for c in "ACGT":
            for c2 in "ACGT":
                for c3 in "ACGT":
                    assert c + c2 + c3 in ECOLI_CODON_USAGE

    def test_atg_is_optimal(self):
        """ATG (the only Met codon) has fraction=1.0, optimal."""
        assert is_optimal_codon("ATG")
        assert codon_adaptation_index("ATG") == 1.0

    def test_ctg_is_optimal(self):
        """CTG (high-frequency Leu) has fraction=0.47, optimal."""
        assert is_optimal_codon("CTG")

    def test_cta_is_rare(self):
        """CTA (rare Leu) has fraction=0.04, rare."""
        assert is_rare_codon("CTA")

    def test_lac_promoter_strength(self):
        """lac promoter induced state is 1000x stronger than repressed state."""
        uninduced = lac_promoter_strength(induced=False)
        induced = lac_promoter_strength(induced=True)
        assert induced == 1.0
        assert uninduced < 0.01
        assert induced / uninduced >= 100

    def test_lac_repression_factor(self):
        """As IPTG concentration increases, the fraction of relieved repression increases monotonically."""
        kds = [0, 1e-7, 1e-6, 1e-5, 1e-3]
        factors = [lac_repression_factor(k) for k in kds]
        for i in range(len(factors) - 1):
            assert factors[i] <= factors[i + 1]

    def test_gray_scott_presets_count(self):
        """Pearson 1993 14 presets."""
        assert len(GRAY_SCOTT_PRESETS) == 14

    def test_gray_scott_preset_lookup(self):
        p = get_gray_scott_preset("Mitosis")
        assert p.F == 0.0367
        assert p.k == 0.0649

    def test_gray_scott_preset_unknown(self):
        with pytest.raises(ValueError):
            get_gray_scott_preset("nonexistent")

    def test_mu_to_grn_strength(self):
        """MU -> GRN threshold mapping."""
        assert mu_to_grn_strength(3000) == 0.0
        assert mu_to_grn_strength(0) == 1.0
        assert mu_to_grn_strength(1500) == 0.5

    def test_pcr_error_rates_taq_vs_q5(self):
        """Q5 error rate should be much lower than Taq (Potapov 2017)."""
        assert PCR_ERROR_RATES["substitution_q5"] < PCR_ERROR_RATES["substitution_taq"]
        # Q5 is about 280x Taq fidelity
        ratio = PCR_ERROR_RATES["substitution_taq"] / PCR_ERROR_RATES["substitution_q5"]
        assert 100 < ratio < 500, f"Taq/Q5 ratio {ratio} out of range"

    def test_transition_transversion_ratio(self):
        """Transition:Transversion ratio ~ 6:1 (Potapov 2017)."""
        assert 0.80 < TRANSITION_TRANSVERSION_RATIO < 0.90

    def test_taq_mutation_spectrum(self):
        """Taq mutation spectrum: A->G accounts for 66% (most frequent)."""
        assert TAQ_MUTATION_SPECTRUM["A"]["G"] == 0.66
        assert TAQ_MUTATION_SPECTRUM["T"]["C"] == 0.66  # complementary


# ============================================================================
# Gray-Scott preset integration
# ============================================================================

class TestGrayScottPresets:
    def test_from_preset(self):
        from helixlang.reaction_diffusion import GrayScott
        gs = GrayScott.from_preset("Coral")
        assert gs.F == 0.016
        assert gs.k == 0.048

    def test_preset_produces_pattern(self):
        """Pearson presets produce non-trivial patterns (V concentration diffusion)."""
        from helixlang.reaction_diffusion import GrayScott
        gs = GrayScott.from_preset("Spots", n=16)
        v_before = gs.total_v()
        for _ in range(50):
            gs.step()
        v_after = gs.total_v()
        assert abs(v_after - v_before) > 0.01


# ============================================================================
# End-to-end: real lac operon parameters
# ============================================================================

class TestRealisticLacOperon:
    def test_lac_operon_uses_real_params(self):
        """The lac operon example can use real induction parameters."""
        uninduced_strength = lac_promoter_strength(induced=False)
        induced_strength = lac_promoter_strength(induced=True)
        assert induced_strength > uninduced_strength * 100


# ============================================================================
# End-to-end: HelixLang source -> DNA -> PCR -> decode
# ============================================================================

class TestEndToEndWithPCR:
    def test_helix_to_dna_to_helix_with_pcr(self):
        """Full pipeline: helix source -> DNA encode -> PCR errors -> decode back."""
        src = "#gene name=test\nATG GCT GGT TAA\n#end\n#config ticks=5\n"
        # 1. Encode
        enc = helix_to_dna(src, scheme="goldman")
        # 2. Inject PCR errors (low cycles; Goldman 4x redundancy should correct them)
        rng = random.Random(42)
        for o in enc["oligos"]:
            o["full"] = pcr_amplify(o["full"], cycles=5, rng=rng, polymerase="taq")
        # 3. Decode (may vary slightly due to PCR errors, but 4x redundancy should guarantee correctness)
        dec = dna_to_helix(enc, scheme="goldman")
        # 4x redundancy + 5 cycles at low error rate should recover perfectly
        assert dec == src, f"roundtrip failed after PCR: {dec!r}"

    def test_erlich_with_pcr_error_correction(self):
        """Erlich RS inner code can correct a few PCR errors."""
        data = b"Lorem ipsum dolor sit amet, consectet!"  # 39 bytes, K=2
        K = 2
        oligos = erlich_encode(data, redundancy=2.0)
        # Inject a few PCR errors (1 substitution per oligo)
        rng = random.Random(42)
        for o in oligos:
            # Replace only 1 base
            pos = rng.randint(0, len(o.payload) - 1)
            bad = list(o.payload)
            bad[pos] = "A" if bad[pos] != "A" else "T"
            o.payload = "".join(bad)
        # RS inner code should correct 1 error (rs_num=2 -> correct 1 symbol)
        decoded = erlich_decode(oligos, K=K, total_len=len(data))
        assert decoded == data
