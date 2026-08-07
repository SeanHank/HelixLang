"""Central dogma: coupled transcription + translation + degradation
model.

Based on real biological data:
- Transcription elongation rate ~50 nt/s (Proshkin 2010 Nature
  458:507-511)
- Translation elongation rate ~20 aa/s (Ingolia 2009 Science
  324:218-223)
- E. coli mRNA median half-life ~5 min (Bernstein 2002 J Bacteriol
  184:6477-6486)
- Transcription initiation frequency is determined by promoter strength
  (Salgado 2013 Nucleic Acids Res)
- tRNA abundance affects codon-specific translation rates (Dong 1996
  J Mol Biol 260:649-663)
- Translation is coupled to transcription: in E. coli translation
  begins before transcription completes (Miller 1972)

Module composition:
1. Constants: transcription/translation rates, half-lives, poly-A tail
   length, RBS consensus sequence, stop-codon efficiencies
2. TRNA_ABUNDANCE: E. coli 64-codon tRNA abundance table (Dong 1996)
3. Dataclasses: Transcript / RibosomeState / TranslationResult /
   TimeCoursePoint
4. Core functions:
   - transcribe(): DNA -> mRNA (incl. terminator detection, poly-A tail
     addition)
   - translate(): mRNA -> protein (incl. RBS detection, codon-specific
     rates)
   - calculate_mrna_level(): mRNA concentration dynamics (production +
     degradation balance)
   - coupled_transcription_translation(): E. coli
     transcription-translation coupling model
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from helixlang.bio_data import ECOLI_CODON_USAGE, MAX_TRNA_ABUNDANCE, TRNA_ABUNDANCE

# ============================================================================
# Real biological parameters (based on literature measurements)
# ============================================================================

# transcription elongation rate (Proshkin 2010 Nature 458:507-511)
# E. coli RNA polymerase ~50 nt/s at 37°C, can slow to ~45 nt/s when
# coupled to ribosomes
TRANSCRIPTION_ELONGATION_RATE_NT_PER_S = 50.0

# base translation elongation rate (Ingolia 2009 Science 324:218-223)
# E. coli ribosome ~20 aa/s at 37°C (measured by ribosome profiling)
TRANSLATION_ELONGATION_RATE_AA_PER_S = 20.0

# mRNA median half-life (Bernstein 2002 J Bacteriol 184:6477-6486)
# E. coli mRNA median half-life ~5 min, ranging 1-20 min
MRNA_HALF_LIFE_MEDIAN_MIN = 5.0

# poly-A tail length (Mohanty & Kushner 2006 RNA 12:1398-1407)
# E. coli poly-A tail ~15 nt (shorter than eukaryotes; mainly promotes
# degradation rather than stability)
E_COLI_POLY_A_TAIL_LENGTH = 15

# RBS consensus sequence (Shine-Dalgarno, Steitz & Jakes 1975 PNAS)
# located 5-13 nt upstream of the start codon, complementary to the 3'
# end of 16S rRNA
RBS_CONSENSUS = "AGGAGG"
# RBS variants (sorted by match priority; longer sequences first to
# avoid short-sequence false matches)
RBS_VARIANTS = ["AGGAGG", "AGGAG", "GGAGG", "AGGA", "GGAG"]
# typical distance between RBS and the start codon (5-13 nt)
RBS_SPACING_MIN = 5
RBS_SPACING_MAX = 13

# stop-codon efficiencies (release factor 1 recognition efficiency,
# measured in E. coli)
# Data source: Poole et al. 1995 RNA 1:1032-1043, Major et al. 1996
# J Mol Biol 257:24-32
# In E. coli TAA is strongest (readthrough <1%), TAG weakest (readthrough
# ~3-10%)
STOP_CODON_EFFICIENCY: dict[str, float] = {
    "TAA": 0.99,   # strongest: ~99% termination efficiency (RF1 recognition)
    "TGA": 0.95,   # medium: ~95% termination efficiency (RF1 recognition)
    "TAG": 0.90,   # weakest: ~90% termination efficiency (higher readthrough)
}

# stop-codon set (standard NCBI table 1)
STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})

# maximum transcription initiation frequency (strong promoters, e.g.
# rrn P1 ~10 mRNA/min)
MAX_INITIATION_FREQUENCY_PER_MIN = 10.0

# translation-transcription coupling delay: ribosomes begin translation
# after ~30 nt of nascent mRNA appears
# 30 nt / 50 nt/s = 0.6 s (classic Miller 1972 observation)
COUPLING_OFFSET_NT = 30


# ============================================================================
# E. coli tRNA abundance table
# ============================================================================
# Note: TRNA_ABUNDANCE / MAX_TRNA_ABUNDANCE have been moved to
# helixlang.bio_data, and central_dogma references them via the import at
# the top. This eliminates the bio_data<->central_dogma circular
# dependency (bio_data no longer needs to lazily load
# central_dogma.TRNA_ABUNDANCE, avoiding the empty-table bug caused by
# module load order).


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(slots=True)
class Transcript:
    """mRNA transcript.

    Attributes:
        sequence: full mRNA sequence (5'UTR + CDS + 3'UTR, U-form,
                  without the poly-A tail)
        utr5: 5' untranslated region (U-form, contains the RBS)
        cds: coding sequence (U-form, includes the start AUG and stop
             codon)
        utr3: 3' untranslated region (U-form)
        poly_a_tail: poly-A tail (string of A's, E. coli ~15 nt)
        half_life_minutes: mRNA half-life (minutes, E. coli median
                           ~5 min)
        promoter_strength: effective promoter strength (0..1, after TF
                           adjustment)
        transcription_factors: transcription factor effects
                               ({tf_name: fold_change}, or None)
        elongation_time_s: transcription elongation time (seconds =
                           nt count / 50 nt/s)
        initiation_frequency_per_min: transcription initiation frequency
                                      (times/min)
        has_terminator: whether a rho-independent terminator was detected
    """
    sequence: str
    utr5: str
    cds: str
    utr3: str
    poly_a_tail: str
    half_life_minutes: float
    promoter_strength: float = 1.0
    transcription_factors: dict[str, float] | None = None
    elongation_time_s: float = 0.0
    initiation_frequency_per_min: float = 0.0
    has_terminator: bool = False


@dataclass(slots=True)
class RibosomeState:
    """Ribosome state (during translation elongation).

    Attributes:
        position: current position on the mRNA (codon index, 0-based)
        peptidyl: synthesized peptide chain (amino acid single-letter
                  string)
        charged_trna: aminoacyl-tRNA currently carried in the A site
                      (single-letter amino acid, or None)
    """
    position: int = 0
    peptidyl: str = ""
    charged_trna: str | None = None


@dataclass(slots=True)
class TranslationResult:
    """Translation result.

    Attributes:
        protein: translated protein sequence (without the stop *)
        elongation_time: total elongation time (seconds, accumulated
                         codon-specifically)
        ribosome_density: ribosome density (ribosomes/100 nt mRNA)
        codon_rates: elongation rate per codon (aa/s, length = protein
                     length)
        rbs_found: whether an RBS (Shine-Dalgarno) sequence was detected
        rbs_sequence: the detected RBS sequence (or empty string)
        stop_codon: the stop codon used (TAA/TAG/TGA, or empty)
        stop_efficiency: termination efficiency (0..1, TAA>TGA>TAG)
        readthrough: whether readthrough occurred (possible when
                     stop_efficiency < 1)
    """
    protein: str
    elongation_time: float
    ribosome_density: float
    codon_rates: list[float] = field(default_factory=list)
    rbs_found: bool = False
    rbs_sequence: str = ""
    stop_codon: str = ""
    stop_efficiency: float = 1.0
    readthrough: bool = False


@dataclass(slots=True)
class TimeCoursePoint:
    """A sample point in the coupling model's time trajectory.

    Attributes:
        time_min: time (minutes)
        mrna_level: relative mRNA concentration (normalized to steady
                    state)
        transcription_progress: transcription completion (0..1)
        translation_progress: translation completion (0..1)
        protein_accumulated: accumulated protein molecules (relative)
    """
    time_min: float
    mrna_level: float
    transcription_progress: float
    translation_progress: float
    protein_accumulated: float


# ============================================================================
# Transcription: DNA -> mRNA
# ============================================================================

def transcribe(dna: str,
               promoter_strength: float = 1.0,
               transcription_factors: dict[str, float] | None = None
               ) -> Transcript:
    """DNA -> mRNA transcription.

    Implementation based on real biological parameters:
    - Transcription initiation frequency is determined by promoter
      strength (Salgado 2013 Nucleic Acids Res)
      E. coli promoter strengths span 4 orders of magnitude (0.001-1.0
      normalized)
    - Transcription elongation rate ~50 nt/s (Proshkin 2010 Nature
      458:507-511)
    - Rho-independent terminator detection (GC stem-loop + poly-U tail)
    - mRNA half-life ~5 min (Bernstein 2002 J Bacteriol 184:6477-6486)
    - poly-A tail addition ~15 nt (Mohanty & Kushner 2006 RNA
      12:1398-1407)

    Args:
        dna: DNA template strand sequence (5'->3', coding-strand
             direction, A/T/C/G, lowercase allowed)
        promoter_strength: promoter strength (0..1, 1=strongest, e.g.
                           lacP induced state)
        transcription_factors: transcription factor effects,
            {tf_name: fold_change}, >1 = activation, <1 = repression,
            None or empty = no TF effects

    Returns:
        a Transcript object (with 5'UTR / CDS / 3'UTR / poly-A tail /
        half-life)
    """
    dna = dna.upper().replace("U", "T")  # normalize to DNA form

    # 1. apply transcription factor effects (multiplicative)
    tf_effect = 1.0
    if transcription_factors:
        for fold in transcription_factors.values():
            tf_effect *= float(fold)
    effective_strength = max(0.0, min(1.0, promoter_strength * tf_effect))

    # 2. initiation frequency: max about 10 mRNA/min (strong promoters
    #    such as rrn P1)
    initiation_freq = effective_strength * MAX_INITIATION_FREQUENCY_PER_MIN

    # 3. detect rho-independent terminator and truncate transcription
    terminator = _find_rho_independent_terminator(dna)
    has_terminator = terminator is not None
    if terminator is not None:
        # terminator = (stem_start, end_of_polyT); transcription ends
        # after the poly-U tail
        transcribed_dna = dna[:terminator[1]]
    else:
        transcribed_dna = dna

    # 4. split into 5'UTR / CDS / 3'UTR (find the first ATG through the
    #    first stop codon)
    utr5_dna, cds_dna, utr3_dna = _split_orf_regions(transcribed_dna)

    # 5. transcription elongation time = nt count / 50 nt/s
    #    (Proshkin 2010)
    elongation_time = len(transcribed_dna) / TRANSCRIPTION_ELONGATION_RATE_NT_PER_S

    # 6. mRNA half-life (E. coli median ~5 min)
    half_life = MRNA_HALF_LIFE_MEDIAN_MIN

    # 7. build the mRNA sequence (T -> U substitution)
    utr5 = utr5_dna.replace("T", "U")
    cds = cds_dna.replace("T", "U")
    utr3 = utr3_dna.replace("T", "U")
    full_mrna = utr5 + cds + utr3

    # 8. poly-A tail (E. coli ~15 nt)
    poly_a = "A" * E_COLI_POLY_A_TAIL_LENGTH

    return Transcript(
        sequence=full_mrna,
        utr5=utr5,
        cds=cds,
        utr3=utr3,
        poly_a_tail=poly_a,
        half_life_minutes=half_life,
        promoter_strength=effective_strength,
        transcription_factors=dict(transcription_factors) if transcription_factors else None,
        elongation_time_s=elongation_time,
        initiation_frequency_per_min=initiation_freq,
        has_terminator=has_terminator,
    )


def _find_rho_independent_terminator(dna: str) -> tuple[int, int] | None:
    """Detect a rho-independent terminator.

    Two features of rho-independent terminators (d'Aubenton Carafa 1990
    J Mol Biol 216:835-859):
    1. GC-rich stem-loop (inverted repeat, >=4 bp stem, GC content >=60%)
    2. A poly-U tail immediately after (poly-T in DNA, >=5 T, usually
       6-8 T)

    Returns the (stem_start, end_of_polyT) tuple (0-based, half-open
    interval), or None.
    """
    n = len(dna)
    if n < 15:
        return None
    # try stem lengths from longest to shortest (prefer matching longer
    # stem-loops)
    for stem_len in (8, 7, 6, 5, 4):
        # i + stem1 + loop(3-5) + stem2 + polyT(>=5) <= n
        max_i = n - stem_len * 2 - 8
        for i in range(max_i + 1):
            stem1 = dna[i:i + stem_len]
            for loop_len in (3, 4, 5):
                j = i + stem_len + loop_len
                if j + stem_len > n:
                    continue
                stem2 = dna[j:j + stem_len]
                if not _is_reverse_complement(stem1, stem2):
                    continue
                # check GC content >= 60%
                gc_count = sum(1 for c in stem1 + stem2 if c in "GC")
                if gc_count / (2 * stem_len) < 0.6:
                    continue
                # check the following poly-T tail (>=5 T, look at most 10)
                poly_t_start = j + stem_len
                poly_t_end = poly_t_start
                while (poly_t_end < n and dna[poly_t_end] == "T"
                       and poly_t_end - poly_t_start < 10):
                    poly_t_end += 1
                if poly_t_end - poly_t_start >= 5:
                    return (i, poly_t_end)
    return None


def _is_reverse_complement(s1: str, s2: str) -> bool:
    """Check whether s2 is the reverse complement of s1 (for stem-loop
    detection)."""
    if len(s1) != len(s2):
        return False
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return s2 == "".join(comp.get(b, "N") for b in reversed(s1))


def _split_orf_regions(dna: str) -> tuple[str, str, str]:
    """Split the DNA into (5'UTR, CDS, 3'UTR).

    Rules (standard NCBI ORF definition):
    - find the first ATG (start codon)
    - scanning codon-by-codon (3 nt steps) from that ATG, find the first
      stop codon
    - before ATG = 5'UTR (contains the RBS)
    - ATG...stop (inclusive) = CDS
    - after stop = 3'UTR
    - no ATG: the whole sequence is 5'UTR, CDS and 3'UTR are empty
    - no stop: CDS runs to the end, 3'UTR is empty
    """
    n = len(dna)
    # find the first ATG
    atg_pos = -1
    for i in range(n - 2):
        if dna[i:i + 3] == "ATG":
            atg_pos = i
            break
    if atg_pos < 0:
        return dna, "", ""
    # scanning codon-by-codon from the ATG, find the stop
    stop_pos = -1
    for i in range(atg_pos, n - 2, 3):
        if dna[i:i + 3] in STOP_CODONS:
            stop_pos = i
            break
    if stop_pos < 0:
        # no stop: CDS runs to the end
        return dna[:atg_pos], dna[atg_pos:], ""
    end = stop_pos + 3
    return dna[:atg_pos], dna[atg_pos:end], dna[end:]


# ============================================================================
# Translation: mRNA -> protein
# ============================================================================

def translate(transcript: Transcript,
              trna_abundance: dict[str, int] | None = None,
              ribosome_density: float = 1.0
              ) -> TranslationResult:
    """mRNA -> protein translation.

    Implementation based on real biological parameters:
    - Base translation elongation rate ~20 aa/s (Ingolia 2009 Science
      324:218-223)
    - Codon-specific rates: determined by tRNA abundance, rare codons
      are slow (Dong 1996)
      rate = base_rate x (tRNA_abundance / max_tRNA_abundance)
    - Translation initiation requires the RBS (Shine-Dalgarno) sequence
      (Steitz & Jakes 1975)
    - Translation termination efficiency: TAA > TGA > TAG (Poole 1995,
      Major 1996)

    Args:
        transcript: Transcript object (from transcribe())
        trna_abundance: codon -> tRNA abundance dict, None uses the
                        default TRNA_ABUNDANCE
        ribosome_density: ribosome density (ribosomes/100 nt mRNA,
                          affects yield)

    Returns:
        a TranslationResult object (with protein sequence, elongation
        time, codon rates, RBS information)
    """
    if trna_abundance is None:
        trna_abundance = TRNA_ABUNDANCE
    max_trna = max(trna_abundance.values()) if trna_abundance else 1

    # convert the mRNA CDS back to DNA form for tRNA table lookup
    cds_dna = transcript.cds.replace("U", "T").upper()
    if not cds_dna:
        return TranslationResult(
            protein="", elongation_time=0.0,
            ribosome_density=ribosome_density,
        )

    # 1. detect the RBS (find the Shine-Dalgarno sequence in the 5'UTR)
    utr5_dna = transcript.utr5.replace("U", "T").upper()
    rbs_found, rbs_seq, _ = _find_rbs(utr5_dna)

    # 2. translate codon by codon (3 nt steps)
    codons = [cds_dna[i:i + 3]
              for i in range(0, len(cds_dna) - len(cds_dna) % 3, 3)]
    protein_chars: list[str] = []
    codon_rates: list[float] = []
    total_time = 0.0
    stop_codon_used = ""
    stop_eff = 1.0

    for codon in codons:
        # stop codon: stop translation
        if codon in STOP_CODONS:
            stop_codon_used = codon
            stop_eff = STOP_CODON_EFFICIENCY.get(codon, 0.9)
            break
        # look up the amino acid (first item of ECOLI_CODON_USAGE)
        if codon in ECOLI_CODON_USAGE:
            aa = ECOLI_CODON_USAGE[codon][0]
        else:
            aa = "X"  # unknown codon
        protein_chars.append(aa)
        # this codon's tRNA abundance -> codon-specific rate
        abundance = trna_abundance.get(codon, 0)
        relative_rate = abundance / max_trna if max_trna > 0 else 0.0
        # actual rate = base rate x relative rate (minimum 0.5 aa/s to
        # avoid 0, simulating near-complete depletion)
        rate = max(0.5, TRANSLATION_ELONGATION_RATE_AA_PER_S * relative_rate)
        codon_rates.append(rate)
        # time per codon = 1 / rate (seconds)
        total_time += 1.0 / rate

    protein = "".join(protein_chars)

    return TranslationResult(
        protein=protein,
        elongation_time=total_time,
        ribosome_density=ribosome_density,
        codon_rates=codon_rates,
        rbs_found=rbs_found,
        rbs_sequence=rbs_seq,
        stop_codon=stop_codon_used,
        stop_efficiency=stop_eff,
        readthrough=stop_eff < 1.0,
    )


def _find_rbs(utr5_dna: str) -> tuple[bool, str, int]:
    """Detect a Shine-Dalgarno (RBS) sequence in the 5'UTR.

    RBS consensus sequence AGGAGG (Steitz & Jakes 1975), located 5-13 nt
    upstream of the start codon, base-pairs with the 16S rRNA 3' end
    (ACCUCCUUA).

    Returns (found, sequence, position). position = -1 if not found.
    """
    if not utr5_dna:
        return False, "", -1
    # search within the last 15 nt of the 5'UTR (near the start codon)
    region = utr5_dna[-15:]
    for variant in RBS_VARIANTS:  # longer sequences take priority
        idx = region.find(variant)
        if idx != -1:
            abs_pos = len(utr5_dna) - len(region) + idx
            return True, variant, abs_pos
    return False, "", -1


# ============================================================================
# mRNA concentration dynamics
# ============================================================================

def calculate_mrna_level(transcript: Transcript,
                        time: float,
                        degradation_rate: float | None = None
                        ) -> float:
    """mRNA concentration over time (production + degradation balance).

    First-order kinetic model (Bernstein 2002):
        d[mRNA]/dt = synthesis_rate - degradation_rate x [mRNA]
        steady-state concentration [mRNA]_ss = synthesis_rate /
        degradation_rate
        time evolution [mRNA](t) = [mRNA]_ss x (1 - exp(-k_deg x t))

    Args:
        transcript: Transcript object (with initiation_frequency and
                    half_life)
        time: time (minutes)
        degradation_rate: degradation rate (1/min), None computes it
                          from the half-life
            k_deg = ln(2) / t_half

    Returns:
        relative mRNA concentration (dimensionless, tending toward the
        steady-state value)
    """
    if degradation_rate is None:
        # derive from the half-life: t_1/2 = ln(2) / k_deg
        if transcript.half_life_minutes <= 0:
            # no degradation -> linear accumulation
            return transcript.initiation_frequency_per_min * time
        degradation_rate = math.log(2.0) / transcript.half_life_minutes
    if degradation_rate <= 0:
        return transcript.initiation_frequency_per_min * time
    # synthesis rate (mRNA/min)
    synthesis_rate = transcript.initiation_frequency_per_min
    # steady-state concentration
    mrna_ss = synthesis_rate / degradation_rate
    # time evolution (starting from 0, exponentially approaching steady
    # state)
    return mrna_ss * (1.0 - math.exp(-degradation_rate * time))


# ============================================================================
# Transcription-translation coupling model (E. coli-specific)
# ============================================================================

def coupled_transcription_translation(dna: str,
                                      promoter_strength: float = 1.0,
                                      ribosome_density: float = 1.0,
                                      time_course_min: float = 30.0,
                                      time_step_min: float = 5.0
                                      ) -> dict:
    """Transcription-translation coupling model (E. coli-specific).

    In E. coli transcription and translation occur simultaneously
    (Miller 1972 Experiments in Molecular Genetics):
    - Ribosomes attach to the nascent mRNA; translation begins before
      transcription completes
    - Translation can initiate ~30 nt (about 0.6 s) after transcription
      starts
    - Translation protects the mRNA from 5'->3' exonucleases (RNase J)
      degradation
    - Polysomes increase protein yield

    Args:
        dna: DNA sequence
        promoter_strength: promoter strength (0..1)
        ribosome_density: ribosome density (ribosomes/100 nt mRNA)
        time_course_min: total simulation time (minutes)
        time_step_min: sampling interval (minutes)

    Returns a dict with:
        - transcript: Transcript object
        - protein: translated protein sequence
        - mrna_level: final mRNA concentration
        - mrna_steady_state: steady-state mRNA concentration
        - time_course: list of TimeCoursePoint (time trajectory)
        - transcription_time_s: transcription duration (seconds)
        - translation_time_s: translation duration (seconds)
        - coupling_offset_s: translation lag relative to transcription
          (~0.6 s)
        - translation_result: full TranslationResult
    """
    # 1. transcription
    transcript = transcribe(dna, promoter_strength=promoter_strength)
    transcription_time_s = transcript.elongation_time_s

    # 2. translation
    translation_result = translate(transcript, ribosome_density=ribosome_density)
    translation_time_s = translation_result.elongation_time

    # 3. coupling offset: ribosomes begin translation after ~30 nt of
    #    nascent mRNA appears
    # 30 nt / 50 nt/s = 0.6 s
    coupling_offset_s = COUPLING_OFFSET_NT / TRANSCRIPTION_ELONGATION_RATE_NT_PER_S

    # 4. mRNA degradation dynamics
    degradation_rate = math.log(2.0) / transcript.half_life_minutes
    synthesis_rate = transcript.initiation_frequency_per_min
    mrna_ss = (synthesis_rate / degradation_rate
               if degradation_rate > 0 else 0.0)

    # 5. time-course sampling
    time_course: list[TimeCoursePoint] = []
    tx_time_min = transcription_time_s / 60.0
    tl_time_min = translation_time_s / 60.0
    tl_start_min = coupling_offset_s / 60.0
    t = 0.0
    while t <= time_course_min + 1e-9:
        # mRNA concentration
        mrna = calculate_mrna_level(transcript, t, degradation_rate)
        # transcription progress (time-based)
        if tx_time_min > 0:
            tx_progress = min(1.0, t / tx_time_min)
        else:
            tx_progress = 1.0
        # translation progress (relative to the transcription offset
        # coupling_offset_s)
        if t < tl_start_min:
            tl_progress = 0.0
        elif tl_time_min > 0:
            tl_progress = min(1.0, (t - tl_start_min) / tl_time_min)
        else:
            tl_progress = 1.0
        # accumulated protein: at steady state protein yield is
        # proportional to mRNA x ribosome density x effective
        # translation time
        protein_accum = mrna * ribosome_density * max(0.0, t - tl_start_min)
        time_course.append(TimeCoursePoint(
            time_min=t,
            mrna_level=mrna,
            transcription_progress=tx_progress,
            translation_progress=tl_progress,
            protein_accumulated=protein_accum,
        ))
        t += time_step_min

    return {
        "transcript": transcript,
        "protein": translation_result.protein,
        "mrna_level": calculate_mrna_level(transcript, time_course_min,
                                            degradation_rate),
        "mrna_steady_state": mrna_ss,
        "time_course": time_course,
        "transcription_time_s": transcription_time_s,
        "translation_time_s": translation_time_s,
        "coupling_offset_s": coupling_offset_s,
        "translation_result": translation_result,
    }


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # dataclasses
    "Transcript", "RibosomeState", "TranslationResult", "TimeCoursePoint",
    # data tables
    "TRNA_ABUNDANCE", "STOP_CODON_EFFICIENCY", "STOP_CODONS",
    "RBS_CONSENSUS", "RBS_VARIANTS",
    # constants
    "TRANSCRIPTION_ELONGATION_RATE_NT_PER_S",
    "TRANSLATION_ELONGATION_RATE_AA_PER_S",
    "MRNA_HALF_LIFE_MEDIAN_MIN",
    "E_COLI_POLY_A_TAIL_LENGTH",
    "MAX_TRNA_ABUNDANCE",
    "MAX_INITIATION_FREQUENCY_PER_MIN",
    "COUPLING_OFFSET_NT",
    # functions
    "transcribe", "translate", "calculate_mrna_level",
    "coupled_transcription_translation",
]
