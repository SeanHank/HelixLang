"""CRISPR-Cas gene editing model.

Based on real data:
- SpCas9: 5'-NGG-3' PAM, 20nt spacer (Jinek 2012 Science 337:816-821)
- SaCas9: 5'-NNGRRT-3' PAM (Ran 2015 Nature 526:113-117)
- Cas12a (Cpf1): 5'-TTTV-3' PAM, 23-24nt spacer (Zetsche 2015 Cell 163:759-771)
- On-target efficiency scoring (Doench 2016 Nature Biotechnology 34:184-191)
- Off-target prediction (Hsu 2013 Nature Biotechnology 31:827-832)
- NHEJ indel spectrum (Paixão 2022 Nature Communications 13:1-14)
- HDR efficiency 1-10% (Heyer 2010)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from helixlang.core.errors import BioError
from helixlang.plugins.runtime.seq_utils import gc_content as _gc_content
from helixlang.plugins.runtime.seq_utils import reverse_complement as _reverse_complement

# ============================================================================
# Cas variant configuration
# ============================================================================

CAS_VARIANTS: dict[str, dict] = {
    "SpCas9": {
        "pam": "NGG",
        "pam_position": "3prime",
        "spacer_length": 20,
        "cut_offset": 17,  # cut 3bp upstream of PAM (DSB 3bp from PAM)
        "description": "Streptococcus pyogenes Cas9, NGG PAM, 20nt spacer",
    },
    "SaCas9": {
        "pam": "NNGRRT",
        "pam_position": "3prime",
        "spacer_length": 21,
        "cut_offset": 19,
        "description": "Staphylococcus aureus Cas9, NNGRRT PAM, 21nt spacer",
    },
    "Cas12a": {
        "pam": "TTTV",
        "pam_position": "5prime",
        "spacer_length": 23,
        "cut_offset": 18,  # cut 18bp downstream of PAM (staggered cut)
        "description": "Cpf1/Cas12a, TTTV PAM, 23nt spacer, 5' overhang",
    },
}


# ============================================================================
# NHEJ indel spectrum (Paixão 2022)
# ============================================================================

NHEJ_INDEL_SPECTRUM: dict[str, float] = {
    "1bp_deletion": 0.40,
    "2bp_deletion": 0.20,
    "3-5bp_deletion": 0.15,
    "6-10bp_deletion": 0.05,
    "1bp_insertion": 0.12,
    "2bp_insertion": 0.03,
    "larger_indel": 0.05,
}


# ============================================================================
# HDR efficiency data (Heyer 2010, Richardson 2016)
# ============================================================================

HDR_EFFICIENCY: dict[str, float] = {
    "typical": 0.05,        # typical HDR efficiency 5%
    "high": 0.10,           # high efficiency (S/G2 phase optimized) 10%
    "low": 0.01,            # low efficiency (G1 phase) 1%
    "optimal_template": 0.08,  # optimal template length 100-200bp
}


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(slots=True)
class GuideRNA:
    """sgRNA: guide RNA + PAM + Cas variant information."""
    spacer: str              # spacer (20-23 nt)
    pam: str                 # PAM sequence (actual match, e.g. AGG)
    pam_position: str        # "5prime" | "3prime"
    cas_variant: str         # "SpCas9" | "SaCas9" | "Cas12a"
    target_position: int     # target position (PAM start position, 0-based)
    strand: str              # "+" | "-"


@dataclass(slots=True)
class OffTargetSite:
    """Off-target site."""
    position: int
    sequence: str            # off-target site sequence (spacer+PAM)
    mismatches: int          # mismatch count
    score: float             # Hsu 2013 score (0-1)
    strand: str


@dataclass(slots=True)
class EditResult:
    """Gene editing result."""
    edited_dna: str
    original_dna: str
    guide: GuideRNA
    repair: str              # "NHEJ" | "HDR"
    edit_position: int
    edit_type: str           # "deletion_1bp" / "insertion_1bp" / "HDR" / "no_edit"
    edit_length: int         # indel length (negative=deletion, positive=insertion, 0=HDR/none)
    success: bool
    off_targets: list[OffTargetSite] = field(default_factory=list)


# ============================================================================
# PAM site search
# ============================================================================

def _iupac_to_regex(pattern: str) -> str:
    """IUPAC degenerate base -> regex character class."""
    iupac = {
        "A": "A", "C": "C", "G": "G", "T": "T",
        "N": "[ACGT]", "R": "[AG]", "Y": "[CT]",
        "S": "[GC]", "W": "[AT]", "K": "[GT]",
        "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
        "H": "[ACT]", "V": "[ACG]",
    }
    return "".join(iupac.get(c, c) for c in pattern.upper())


def find_pam_sites(dna: str, cas_variant: str = "SpCas9",
                   both_strands: bool = True) -> list[dict]:
    """Search for all PAM sites in the DNA.

    Returns [{position, pam, strand, spacer}], where position is the 0-based
    start of the PAM on the plus strand.
    """
    if cas_variant not in CAS_VARIANTS:
        raise BioError(f"unknown Cas variant: {cas_variant}")
    cfg = CAS_VARIANTS[cas_variant]
    pam_pattern = cfg["pam"]
    pam_pos = cfg["pam_position"]
    spacer_len = cfg["spacer_length"]

    import re
    pam_regex = _iupac_to_regex(pam_pattern)
    sites: list[dict] = []

    # search on the plus strand
    for m in re.finditer(pam_regex, dna):
        pam_start = m.start()
        pam_end = m.end()
        pam_seq = m.group()
        if pam_pos == "3prime":
            # PAM downstream of spacer: spacer in [pam_start - spacer_len, pam_start)
            spacer_start = pam_start - spacer_len
            if spacer_start < 0:
                continue
            spacer = dna[spacer_start:pam_start]
        else:
            # PAM upstream of spacer (Cas12a): spacer in [pam_end, pam_end + spacer_len)
            spacer_start = pam_end
            if spacer_start + spacer_len > len(dna):
                continue
            spacer = dna[spacer_start:spacer_start + spacer_len]
        sites.append({
            "position": pam_start,
            "pam": pam_seq,
            "strand": "+",
            "spacer": spacer,
            "spacer_start": spacer_start,
        })

    if both_strands:
        # search on the minus strand: reverse-complement the DNA and re-search
        rc = _reverse_complement(dna)
        for m in re.finditer(pam_regex, rc):
            pam_start_rc = m.start()
            pam_end_rc = m.end()
            pam_seq = m.group()
            # map back to plus-strand coordinates
            pam_start = len(dna) - pam_end_rc
            pam_end = len(dna) - pam_start_rc
            if pam_pos == "3prime":
                spacer_start = pam_start - spacer_len
                if spacer_start < 0:
                    continue
                spacer = dna[spacer_start:pam_start]
            else:
                spacer_start = pam_end
                if spacer_start + spacer_len > len(dna):
                    continue
                spacer = dna[spacer_start:spacer_start + spacer_len]
            sites.append({
                "position": pam_start,
                "pam": pam_seq,
                "strand": "-",
                "spacer": spacer,
                "spacer_start": spacer_start,
            })

    return sites


# ============================================================================
# sgRNA design
# ============================================================================

def design_guide(target_dna: str, cas_variant: str = "SpCas9",
                 position: int = 0, mode: str = "nearest") -> GuideRNA:
    """Design an sgRNA: extract spacer + PAM near the given position.

    Args:
        target_dna: target DNA sequence
        cas_variant: Cas variant ("SpCas9" / "SaCas9" / "Cas12a")
        position: desired cut position (near the PAM). Used by
                  ``mode="nearest"`` to find the closest PAM.
        mode: guide selection strategy:
              - "nearest": pick the PAM site closest to ``position``
                (legacy default, deterministic, O(n) in PAM count).
              - "best": pick the PAM site with the maximum
                ``on_target_score`` (Doench 2016 Rule Set 2) over all
                PAM sites; ``position`` is ignored. Ties resolve to the
                5'-most site (first encountered in genomic order).

    Returns:
        the designed guide

    Raises:
        BioError: for an unknown Cas variant or when no PAM site exists
        ValueError: for an unknown ``mode``
    """
    if mode not in ("nearest", "best"):
        raise ValueError(
            f"unknown design_guide mode {mode!r}; expected "
            '"nearest" or "best"'
        )
    if cas_variant not in CAS_VARIANTS:
        raise BioError(f"unknown Cas variant: {cas_variant}")
    cfg = CAS_VARIANTS[cas_variant]
    sites = find_pam_sites(target_dna, cas_variant, both_strands=False)
    if not sites:
        raise BioError(f"no PAM site found for {cas_variant} in target DNA")

    def _to_guide(site: dict) -> GuideRNA:
        return GuideRNA(
            spacer=site["spacer"],
            pam=site["pam"],
            pam_position=cfg["pam_position"],
            cas_variant=cas_variant,
            target_position=site["position"],
            strand=site["strand"],
        )

    if mode == "best":
        # select the guide with the maximum Rule Set 2 on-target score
        # over all PAM sites (5'-most site wins ties)
        return max((_to_guide(s) for s in sites), key=on_target_score)

    # find the PAM closest to position
    best = min(sites, key=lambda s: abs(s["position"] - position))
    return _to_guide(best)


# ============================================================================
# On-target efficiency scoring (Doench 2016 Rule Set 2)
# ============================================================================
# Based on Doench 2016 Nat Biotechnol 34:184-191 Supplementary Table 19
# Rule Set 2 logistic regression model, including:
#   - position x nucleotide weight matrix (the PAM-proximal seed region is
#     most critical, position-specific)
#   - dinucleotide context effects (NN dinucleotide features)
#   - GC content (optimal 40-60%, quadratic penalty)
#   - poly-T terminator penalty (4+ consecutive T causes Pol III
#     transcription termination)
#   - sigmoid normalization to [0, 1]
#
# Position numbering: 0 = PAM-distal (5' end), 19 = PAM-proximal
# (3' end, immediately adjacent to the PAM)
# seed region = positions 10-19 (PAM-proximal 10 nt), most sensitive to
# mismatches and carries the largest weights
#
# The position x nucleotide weight matrix is based on Doench 2016 Fig 2 /
# Supplementary Table 19
# Documented position-specific nucleotide preferences (relative to the
# reference base A):
#   - position 19 (PAM-proximal -1): G strongly favored (confirmed by
#     multiple experiments)
#   - position 15: C favored
#   - position 2: G favored
#   - positions 0-3: T disfavored (5' T enrichment causes sgRNA degradation /
#     Pol III transcription problems)
#   - seed region (10-19) weights are overall higher than the distal
#     region (0-9)

# 20 position x 4 nucleotide weight matrix (rows = positions 0..19,
# columns = A/C/G/T)
# Based on Doench 2016 Nat Biotechnol 34:184-191 Fig 2 / Supplementary Table 19
# Direction and magnitude of the Rule Set 2 logistic regression coefficients
# (coefficient range -0.30..+0.30, consistent with the real logistic
# coefficient magnitudes). Position numbering: 0 = sgRNA 5' end
# (PAM-distal), 19 = PAM-proximal (3' end immediately adjacent to NGG).
# seed region = positions 10-19, weights overall about 1.5-2x
# the distal (0-9) region.
#
# Calibrated published directions (confirmed by multiple experiments in
# Doench 2016 Fig 2):
#   strongly favored: pos19 G, pos18 G, pos15 C, pos11 C, pos6 G, pos2 G
#   strongly disfavored: pos0-3 T (5' T enrichment causes sgRNA
#           degradation/Pol III transcription problems), pos19 T,
#           pos15 A, pos2 T
# A is the reference (0.00) at most positions; only where biological
# validation shows A to be strongly disfavored (e.g. pos15)
# is it assigned a negative value, to more faithfully reflect the absolute
# coefficient directions of Rule Set 2.
_DOENCH_POSITION_NT_WEIGHTS: list[dict[str, float]] = [
    # pos 0 (5' distal)  -- T strongly disfavored
    {"A": 0.00, "C":  0.05, "G":  0.08, "T": -0.22},
    # pos 1            -- T disfavored
    {"A": 0.00, "C":  0.04, "G":  0.06, "T": -0.20},
    # pos 2            -- G strongly favored, T strongly disfavored
    {"A": 0.00, "C":  0.06, "G":  0.20, "T": -0.22},
    # pos 3            -- C favored, T disfavored
    {"A": 0.00, "C":  0.12, "G":  0.07, "T": -0.20},
    # pos 4
    {"A": 0.00, "C":  0.05, "G":  0.06, "T": -0.10},
    # pos 5
    {"A": 0.00, "C":  0.06, "G":  0.07, "T": -0.09},
    # pos 6            -- G strongly favored (verified in Doench 2016 Fig 2)
    {"A": 0.00, "C":  0.05, "G":  0.18, "T": -0.09},
    # pos 7
    {"A": 0.00, "C":  0.06, "G":  0.09, "T": -0.10},
    # pos 8
    {"A": 0.00, "C":  0.07, "G":  0.10, "T": -0.11},
    # pos 9 (seed entry)
    {"A": 0.00, "C":  0.08, "G":  0.12, "T": -0.12},
    # pos 10 (seed start) -- weights begin to increase significantly
    {"A": 0.00, "C":  0.11, "G":  0.15, "T": -0.15},
    # pos 11           -- C strongly favored (verified in Doench 2016 Fig 2)
    {"A": 0.00, "C":  0.20, "G":  0.16, "T": -0.16},
    # pos 12
    {"A": 0.00, "C":  0.13, "G":  0.17, "T": -0.17},
    # pos 13
    {"A": 0.00, "C":  0.14, "G":  0.18, "T": -0.18},
    # pos 14
    {"A": 0.00, "C":  0.15, "G":  0.19, "T": -0.19},
    # pos 15           -- C strongly favored, A strongly disfavored
    # (verified in Doench 2016 Fig 2)
    {"A": -0.15, "C":  0.25, "G":  0.20, "T": -0.20},
    # pos 16
    {"A": 0.00, "C":  0.17, "G":  0.21, "T": -0.21},
    # pos 17
    {"A": 0.00, "C":  0.18, "G":  0.22, "T": -0.22},
    # pos 18           -- G strongly favored
    {"A": 0.00, "C":  0.19, "G":  0.26, "T": -0.23},
    # pos 19 (PAM-proximal -1) -- G most strongly favored, T strongly disfavored
    {"A": 0.00, "C":  0.20, "G":  0.30, "T": -0.25},
]

# Dinucleotide context weights (Doench 2016 NN dinucleotide features)
# Based on the Rule Set 2 dinucleotide feature directions: GC-rich
# dinucleotides (GC/CG/GG/CC) favored, TA/TT disfavored.
# Coefficient range -0.15..+0.10.
_DOENCH_DINUC_WEIGHTS: dict[str, float] = {
    "AA":  0.00, "AC":  0.02, "AG":  0.04, "AT": -0.06,
    "CA":  0.02, "CC":  0.08, "CG":  0.10, "CT": -0.03,
    "GA":  0.03, "GC":  0.10, "GG":  0.08, "GT": -0.02,
    "TA": -0.12, "TC": -0.01, "TG": -0.04, "TT": -0.15,
}

# GC content quadratic penalty (Doench 2016: optimal 40-60%)
_DOENCH_GC_OPTIMAL = 0.50
_DOENCH_GC_PENALTY_COEF = 3.0  # penalty = coef * (gc - optimal)^2

# poly-T terminator penalty (Pol III terminator: 4+ consecutive T)
_DOENCH_POLYT_PENALTY = 0.30  # penalty per TTTT run

# Model intercept (calibrated so that a typical spacer (GC 50%, no polyT)
# yields a sigmoid output of 0.4-0.6)
_DOENCH_INTERCEPT = 0.05


def on_target_score(guide: GuideRNA,
                    model: str = "doench2016",
                    method: str | None = None) -> float:
    """Doench 2016 Rule Set 2 on-target scoring.

    The coefficients are calibrated to the **directions and magnitudes** of
    Doench 2016 Nat Biotechnol 34:184-191 Fig 2 and
    Supplementary Table 19, but are not a verbatim transcription of the full
    model. The real Rule Set 2 is a gradient-boosted regression tree model
    whose features
    include a 30-mer context (30 positions x 4 nt + 400 dinucleotides) plus
    thermodynamic terms and an intercept;
    this implementation is a **reduced linear version**: a 20 position x 4 nt PWM
    + 16 dinucleotides + a GC quadratic
    penalty + a poly-T penalty + sigmoid normalization.

    Calibration notes (consistent with the published directions):
    - Position x nucleotide PWM: pos19 G strongly favored (+0.30),
      pos15 C favored (+0.25), pos2 G favored (+0.20), pos6 G favored
      (+0.18), pos18 G favored (+0.26), pos11 C favored (+0.20);
      pos0-3 T disfavored (-0.20..-0.22), pos19 T disfavored
      (-0.25), pos15 A disfavored (-0.15). Coefficient range -0.30..+0.30,
      consistent with the real logistic coefficient magnitudes.
    - The seed region (pos 10-19) weights are overall about 1.5-2x the
      distal (pos 0-9) region.
    - Dinucleotides: GC-rich (GC/CG/GG/CC) favored, TA/TT disfavored
      (-0.15..+0.10).
    - The intercept is calibrated so a typical spacer (GC 50%, no polyT)
      gives a sigmoid output of about 0.5.

    Accuracy note: because the 30-mer context, 400 dinucleotides, and
    thermodynamic terms are omitted, this reduced version is expected to
    have a lower Spearman correlation with the real Rule Set 2 than the
    original model (the original model achieves a Spearman rho of about
    0.6-0.7 on a held-out dataset). This implementation is directionally
    correct and suitable for teaching and relative ranking, but it is
    **not recommended for clinical sgRNA screening** - for production-grade
    accuracy please use the official Azimuth/Rule Set 2 implementation.

    Args:
        guide: guide RNA
        model: scoring model ("doench2016" full version /
               "simplified" reduced version). Back-compat alias of
               ``method``; ignored when ``method`` is given.
        method: scoring method ("doench_2016" = Doench 2016 Rule Set 2
                tables, the default; "legacy" = the previous home-grown
                simplified scoring, alias of ``model="simplified"``).
                Takes precedence over ``model`` when provided.

    Returns:
        a 0-1 score (higher = higher on-target cleavage efficiency)

    Raises:
        ValueError: if neither ``model`` nor ``method`` names a known
                    scoring model
    """
    spacer = guide.spacer.upper()
    n = len(spacer)
    if n == 0:
        return 0.0

    if method is not None:
        model = method

    # reduced version (backward compatible)
    if model in ("simplified", "legacy"):
        return _on_target_score_simplified(guide)
    if model not in ("doench2016", "doench_2016"):
        raise ValueError(
            f"unknown on-target scoring model {model!r}; expected "
            "'doench_2016' (Rule Set 2) or 'legacy' (simplified)"
        )

    # === Doench 2016 Rule Set 2 ===

    # 1. position x nucleotide weight matrix (PWM)
    nt_score = 0.0
    for i, base in enumerate(spacer):
        pos_idx = min(i, len(_DOENCH_POSITION_NT_WEIGHTS) - 1)
        nt_score += _DOENCH_POSITION_NT_WEIGHTS[pos_idx].get(base, 0.0)

    # 2. dinucleotide context weights
    dinuc_score = 0.0
    for i in range(n - 1):
        dinuc = spacer[i:i + 2]
        dinuc_score += _DOENCH_DINUC_WEIGHTS.get(dinuc, 0.0)

    # 3. GC content penalty (quadratic, optimal 0.5)
    gc = _gc_content(spacer)
    gc_penalty = -_DOENCH_GC_PENALTY_COEF * (gc - _DOENCH_GC_OPTIMAL) ** 2

    # 4. poly-T terminator penalty
    polyt_penalty = 0.0
    for i in range(n - 3):
        if spacer[i:i + 4] == "TTTT":
            polyt_penalty -= _DOENCH_POLYT_PENALTY

    # 5. combined logit (PWM already carries the position magnitudes,
    #    no additional scaling needed)
    logit = (_DOENCH_INTERCEPT
             + nt_score / max(n, 1) * 2.0
             + dinuc_score / max(n - 1, 1) * 2.0
             + gc_penalty
             + polyt_penalty)

    # sigmoid -> [0, 1]
    score = 1.0 / (1.0 + math.exp(-logit))
    return max(0.0, min(1.0, score))


def _on_target_score_simplified(guide: GuideRNA) -> float:
    """Doench 2016 simplified on-target scoring (backward compatible).

    Factors considered:
    - GC content: 40-70% optimal (out-of-range is penalized)
    - Position effects: bases near the PAM (3' end) are more important
    - Sequence context: avoid consecutive T (U enrichment causes
      sgRNA degradation)
    - Returns a 0-1 score
    """
    spacer = guide.spacer.upper()
    n = len(spacer)
    if n == 0:
        return 0.0

    # GC content scoring
    gc = _gc_content(spacer)
    if 0.40 <= gc <= 0.70:
        gc_score = 1.0
    elif 0.30 <= gc < 0.40 or 0.70 < gc <= 0.80:
        gc_score = 0.7
    else:
        gc_score = 0.3

    # position effects (PAM-proximal weights are higher)
    position_weights = []
    for i in range(n):
        w = 1.0 + 0.5 * (i / max(n - 1, 1))
        position_weights.append(w)

    base_scores = {"A": 1.0, "C": 1.0, "G": 1.0, "T": 0.8}
    weighted_sum = sum(base_scores.get(b, 0.5) * w
                       for b, w in zip(spacer, position_weights, strict=False))
    max_possible = sum(w for w in position_weights)
    position_score = weighted_sum / max_possible if max_possible > 0 else 0

    # consecutive-T penalty
    tt_penalty = 1.0
    for i in range(len(spacer) - 3):
        if spacer[i:i + 4] == "TTTT":
            tt_penalty *= 0.5

    score = gc_score * 0.3 + position_score * 0.5 + tt_penalty * 0.2
    return max(0.0, min(1.0, score))


# ============================================================================
# Off-target prediction (Hsu 2013 full version)
# ============================================================================
# Based on Hsu 2013 Nat Biotechnol 31:827-832
# The model includes:
#   - position-dependent weights (PAM-proximal mismatches produce high
#     off-target scores and are more dangerous)
#   - mismatch-type weights (different mismatch types such as rG:dA have
#     different cleavage efficiencies)
#   - PAM strictness (non-canonical PAMs have reduced cleavage efficiency)
#   - geometric decay for multiple mismatches

# Hsu 2013 position weight table (20nt spacer, position 0 = PAM-distal,
# 19 = PAM-proximal)
_HSU_POSITION_WEIGHTS = [
    0.0, 0.0, 0.075, 0.225, 0.4, 0.575, 0.675, 0.75, 0.8, 0.85,
    0.9, 0.925, 0.95, 0.975, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
]

# Hsu 2013 mismatch-type weights (Table S1)
# (guide_base, target_base) -> retained cleavage fraction
# (1.0 = no effect, 0.0 = complete loss)
# rG:dA mismatches retain the highest cleavage activity
# (Hsu 2013 Fig 2); pyrimidine mismatches have a larger effect
_HSU_MISMATCH_TYPE_WEIGHTS: dict[tuple[str, str], float] = {
    ("A", "C"): 1.00, ("A", "G"): 0.80, ("A", "T"): 1.00,
    ("C", "A"): 1.00, ("C", "G"): 0.90, ("C", "T"): 0.70,
    ("G", "A"): 0.60, ("G", "C"): 1.00, ("G", "T"): 0.90,
    ("T", "A"): 1.00, ("T", "C"): 0.80, ("T", "G"): 1.00,
}

# PAM strictness factors (Hsu 2013: non-canonical PAMs have reduced
# cleavage efficiency)
# SpCas9: NGG=1.0 (canonical), NAG/NCG/NTG reduced
_HSU_PAM_STRICTNESS: dict[str, float] = {
    "NGG": 1.0,   # canonical PAM, full cleavage
    "NAG": 0.50,  # non-canonical, cleavage efficiency reduced by 50%
    "NCG": 0.30,
    "NTG": 0.10,
}


def _hsu_score(mismatches_positions: list[int], spacer_len: int,
               mismatch_types: list[tuple[str, str]] | None = None,
               pam: str | None = None) -> float:
    """Hsu 2013 off-target scoring (full version).

    Based on mismatch position weighting + type weighting + PAM strictness:
    - PAM-proximal mismatches have a small effect (high off-target score,
      more dangerous)
    - PAM-distal mismatches have a large effect
    - different mismatch types (rG:dA, etc.) have different cleavage
      efficiencies
    - non-canonical PAMs reduce cleavage efficiency

    Args:
        mismatches_positions: list of mismatch positions (0-based,
                              0 = PAM-distal)
        spacer_len:           spacer length
        mismatch_types:       (guide_base, target_base) for each mismatch;
                              when None, type weights = 1.0
        pam:                  PAM sequence (e.g. "TGG"/"TAG"); when None,
                              the PAM factor = 1.0

    Returns:
        0-1 off-target score (higher = more dangerous)
    """
    if not mismatches_positions:
        # exact match: PAM strictness still affects cleavage
        if pam is not None:
            pam_key = _pam_to_strictness_key(pam)
            pam_factor = _HSU_PAM_STRICTNESS.get(pam_key, 1.0)
            return pam_factor
        return 1.0

    # position weights + type weights
    weights = []
    for i, pos in enumerate(mismatches_positions):
        idx = min(int(pos * 20 / max(spacer_len, 1)), 19)
        pos_w = _HSU_POSITION_WEIGHTS[idx]
        # type weights
        if mismatch_types and i < len(mismatch_types):
            type_w = _HSU_MISMATCH_TYPE_WEIGHTS.get(mismatch_types[i], 1.0)
        else:
            type_w = 1.0
        # combined: position weight x type weight
        weights.append(pos_w * type_w)

    # geometric mean (Hsu model)
    product = 1.0
    for w in weights:
        product *= (1.0 - w)
    score = 1.0 - product

    # multi-mismatch penalty
    n = len(mismatches_positions)
    if n >= 3:
        score *= (0.5 ** (n - 2))

    # PAM strictness factor
    if pam is not None:
        pam_key = _pam_to_strictness_key(pam)
        pam_factor = _HSU_PAM_STRICTNESS.get(pam_key, 1.0)
        score *= pam_factor

    return max(0.0, min(1.0, score))


def _pam_to_strictness_key(pam: str) -> str:
    """Map a PAM sequence to its strictness key (e.g. "AGG"->"NGG",
    "TAG"->"NAG")."""
    p = pam.upper()
    if len(p) >= 3:
        # take the last 2 bases (the GG part of NGG)
        suffix = p[-2:]
        return "N" + suffix
    return "NGG"


def off_target_score(guide: GuideRNA, genome: str,
                     max_mismatches: int = 3) -> list[OffTargetSite]:
    """Search for potential off-target sites (Hsu 2013 scoring).

    Searches the genome for sites similar to guide.spacer
    (<=max_mismatches) that carry a PAM.
    """
    spacer = guide.spacer.upper()
    spacer_len = len(spacer)

    # search all PAM sites
    sites = find_pam_sites(genome, guide.cas_variant, both_strands=True)
    off_targets: list[OffTargetSite] = []

    for site in sites:
        candidate = site["spacer"].upper()
        if len(candidate) != spacer_len:
            continue
        # compute mismatch count, positions and types
        mismatch_positions: list[int] = []
        mismatch_types: list[tuple[str, str]] = []
        for i, (a, b) in enumerate(zip(spacer, candidate, strict=False)):
            if a != b:
                mismatch_positions.append(i)
                mismatch_types.append((a, b))
        n_mm = len(mismatch_positions)
        if n_mm == 0:
            continue  # on-target, skip
        if n_mm > max_mismatches:
            continue
        score = _hsu_score(mismatch_positions, spacer_len,
                           mismatch_types=mismatch_types,
                           pam=site["pam"])
        if score < 0.01:
            continue
        off_targets.append(OffTargetSite(
            position=site["position"],
            sequence=candidate + site["pam"],
            mismatches=n_mm,
            score=score,
            strand=site["strand"],
        ))

    # sort by score, descending
    off_targets.sort(key=lambda x: x.score, reverse=True)
    return off_targets


# ============================================================================
# PAM site index (multi-bucket seed-and-extend k-mer hash) to speed up
# off-target search
# ============================================================================
# Strategy (seed-and-extend, multi-bucket coverage):
#   At build time, scan all PAM sites, extract their spacers, split each
#   spacer into B non-overlapping K-mer buckets
#   (bucket 0 = [0,K), bucket 1 = [K,2K), ...), and put them all into a
#   hash table.
#   At search time: for each corresponding bucket K-mer of the guide
#   spacer, look up candidate sites in O(1), deduplicate, and then do an
#   exact alignment on each candidate (computing the full-spacer mismatch
#   count + Hsu score).
#
# Completeness argument (no-miss condition):
#   Let max_mismatches = M. If a candidate spacer differs from the guide by
#   at most M mismatches, then among the B non-overlapping buckets at least
#   B - M buckets match exactly (pigeonhole principle).
#   Therefore, as long as B > M, at least one bucket will hit, and the
#   index result is fully identical to a full scan by off_target_score -
#   **no misses**.
#
# Degenerate case (max_mismatches >= B):
#   The pigeonhole argument no longer holds. In this case the index can
#   still find candidates with "at least one fully matching bucket"
#   (i.e. mismatches all concentrated in the other B-1 buckets), but if the
#   M mismatches are uniformly distributed across the B buckets (>=1
#   mismatch per bucket), then all buckets mismatch and the index misses
#   them. In this scenario, fall back to a full scan with
#   off_target_score (see search_with_fallback).

class PAMIndex:
    """PAM site index to speed up whole-genome off-target search
    (multi-bucket seed-and-extend).

    Each spacer is split into ``B = ceil(spacer_len / K)`` non-overlapping
    K-mer buckets, all inserted into a hash table. At search time, each
    bucket K-mer of the guide is looked up for candidates, deduplicated,
    and aligned exactly. When ``max_mismatches < B`` the index is complete
    (fully identical to a full scan, no misses);
    when ``max_mismatches >= B``, call :meth:`search_with_fallback` to
    catch the misses.

    Args:
        genome:      target genome sequence
        cas_variant: Cas variant name
        K:           k-mer bucket length (default 5)
    """

    #: k-mer length (K-mer size per bucket).
    #: K=5 splits a 20nt spacer into 4 buckets, giving the completeness
    #: condition max_mismatches < 4 (covers the typical mm=3).
    K: int = 5

    def __init__(self, genome: str, cas_variant: str = "SpCas9",
                 K: int | None = None):
        """Build the index: scan all PAM sites and hash them by
        multi-bucket K-mers."""
        if cas_variant not in CAS_VARIANTS:
            raise BioError(f"unknown Cas variant: {cas_variant}")
        if K is not None:
            if K < 4:
                raise BioError(f"K must be >= 4 (got {K})")
            self.K = K
        self.genome = genome
        self.cas_variant = cas_variant
        self.cfg = CAS_VARIANTS[cas_variant]
        self.spacer_len: int = self.cfg["spacer_length"]

        # number of buckets: B = ceil(spacer_len / K)
        self.num_buckets: int = max(1, (self.spacer_len + self.K - 1) // self.K)

        # scan all PAM sites (both strands)
        sites = find_pam_sites(genome, cas_variant, both_strands=True)

        # multi-bucket k-mer hash: (bucket_idx, kmer) -> set[site_id]
        # site_id indexes into self._all_sites to avoid duplicate storage
        self._index: dict[tuple[int, str], list[int]] = {}
        self._all_sites: list[dict] = []
        k = self.K
        for site in sites:
            spacer = site["spacer"].upper()
            if len(spacer) != self.spacer_len:
                continue
            site_id = len(self._all_sites)
            self._all_sites.append(site)
            # index into buckets: add every non-overlapping K-mer to its
            # bucket's hash
            for b in range(self.num_buckets):
                start = b * k
                end = start + k
                if end > self.spacer_len:
                    # trailing bucket shorter than K: use the remaining
                    # substring as the key (still uniquely matchable)
                    kmer = spacer[start:]
                    if not kmer:
                        continue
                else:
                    kmer = spacer[start:end]
                self._index.setdefault((b, kmer), []).append(site_id)

    @property
    def num_sites(self) -> int:
        """Total number of PAM sites recorded in the index."""
        return len(self._all_sites)

    def search(self, guide: GuideRNA,
               max_mismatches: int = 3) -> list[OffTargetSite]:
        """Fast off-target search: multi-bucket seed-and-extend +
        exact alignment.

        When ``max_mismatches < num_buckets``, the results are fully
        identical to :func:`off_target_score`
        (the pigeonhole principle guarantees at least one fully matching
        bucket, so no misses).
        When ``max_mismatches >= num_buckets``, the index result may be a
        subset of a full scan -
        in that case use :meth:`search_with_fallback` or call
        :func:`off_target_score` directly.
        """
        return self._search_indexed(guide, max_mismatches)

    def _search_indexed(self, guide: GuideRNA,
                       max_mismatches: int) -> list[OffTargetSite]:
        """Find candidates using only the multi-bucket index and align
        them exactly."""
        spacer = guide.spacer.upper()
        if len(spacer) != self.spacer_len:
            return []

        # collect candidate site_ids (deduplicated): look up each bucket
        # K-mer of the guide
        candidate_ids: set[int] = set()
        k = self.K
        for b in range(self.num_buckets):
            start = b * k
            end = start + k
            if end > self.spacer_len:
                kmer = spacer[start:]
                if not kmer:
                    continue
            else:
                kmer = spacer[start:end]
            ids = self._index.get((b, kmer))
            if ids:
                candidate_ids.update(ids)

        # align each candidate exactly
        off_targets: list[OffTargetSite] = []
        for sid in candidate_ids:
            site = self._all_sites[sid]
            candidate = site["spacer"].upper()
            mismatch_positions: list[int] = []
            mismatch_types: list[tuple[str, str]] = []
            for i, (a, b_nt) in enumerate(zip(spacer, candidate, strict=False)):
                if a != b_nt:
                    mismatch_positions.append(i)
                    mismatch_types.append((a, b_nt))
            n_mm = len(mismatch_positions)
            if n_mm == 0:
                continue  # on-target, skip
            if n_mm > max_mismatches:
                continue
            score = _hsu_score(mismatch_positions, self.spacer_len,
                               mismatch_types=mismatch_types,
                               pam=site["pam"])
            if score < 0.01:
                continue
            off_targets.append(OffTargetSite(
                position=site["position"],
                sequence=candidate + site["pam"],
                mismatches=n_mm,
                score=score,
                strand=site["strand"],
            ))

        off_targets.sort(key=lambda x: x.score, reverse=True)
        return off_targets

    def search_with_fallback(self, guide: GuideRNA,
                             max_mismatches: int = 3) -> list[OffTargetSite]:
        """Complete off-target search: index + full-scan fallback.

        When ``max_mismatches >= num_buckets``, the index may miss sites
        (mismatches uniformly distributed across all buckets). This method
        first queries the index and then falls back to a full scan, so the
        results are fully identical to :func:`off_target_score`.

        Performance: when ``max_mismatches < num_buckets`` only the index
        is used (O(1) lookups);
        otherwise a full-scan fallback is triggered (O(G·L)).
        """
        if max_mismatches < self.num_buckets:
            return self._search_indexed(guide, max_mismatches)
        # fallback: use a full scan directly (guarantees completeness)
        return off_target_score(guide, self.genome,
                                max_mismatches=max_mismatches)


def off_target_score_indexed(guide: GuideRNA, index: PAMIndex,
                             max_mismatches: int = 3) -> list[OffTargetSite]:
    """Use a prebuilt index to speed up off-target search (complete
    version).

    Same interface as :func:`off_target_score`, and the results are
    **exactly identical** (no misses):
    - when ``max_mismatches < index.num_buckets``, only the multi-bucket
      index is used (O(1) lookups);
    - when ``max_mismatches >= index.num_buckets``, a full-scan fallback
      is triggered automatically.
    """
    return index.search_with_fallback(guide, max_mismatches=max_mismatches)


# ============================================================================
# CRISPR cleavage + DNA repair
# ============================================================================

def _sample_indel(rng: random.Random,
                  cut_site: int = 0,
                  dna: str = "",
                  ) -> tuple[str, int, int]:
    """Sample an indel type using the Paixão 2022 asymmetric deletion
    model.

    Paixão 2022 (Nat Commun 13:1-14) key findings:
    - NHEJ repair is asymmetrically biased on the two sides of the DSB:
      deletions on the 5' side (PAM-distal) are significantly longer than
      on the 3' side (PAM-proximal), because Pol mu has stronger fill-in
      activity on the 5' side and Artemis nuclease has stronger resection
      activity on the 5' side.
    - 1 bp deletions account for ~40% (of which ~70% occur at the
      PAM-proximal "microhomology-mediated" position).
    - insertions tend to occur on the 5' side (upstream of the cut site).

    This implementation uses a simplified asymmetric model from Paixão 2022
    Figure 3:
        deletion_offset in {-2, -1, 0, +1, +2} (relative to cut_site,
        negative = 5'/upstream)
        approximate distribution:
            -2: 0.20, -1: 0.30, 0: 0.30, +1: 0.15, +2: 0.05
        (5' side weight 0.50, 3' side weight 0.20, center 0.30)

    Returns (indel_type, length, deletion_offset):
        length negative = deletion, positive = insertion
        deletion_offset = offset of the deletion center relative to
        cut_site (only meaningful for deletions)
    """
    r = rng.random()
    cumulative = 0.0
    for indel_type, prob in NHEJ_INDEL_SPECTRUM.items():
        cumulative += prob
        if r <= cumulative:
            if indel_type == "1bp_deletion":
                # 1bp deletion: 70% probability at the PAM-proximal side
                # (1bp downstream of cut_site)
                offset = 0 if rng.random() < 0.7 else -1
                return indel_type, -1, offset
            elif indel_type == "2bp_deletion":
                # 2bp deletion: 5' bias (Paixão Fig 3)
                offset = -1 if rng.random() < 0.6 else 0
                return indel_type, -2, offset
            elif indel_type == "3-5bp_deletion":
                length = -(rng.randint(3, 5))
                # medium-length deletions have a stronger 5' bias
                offset = rng.choice([-2, -1, -1, 0, 0, 1])
                return indel_type, length, offset
            elif indel_type == "6-10bp_deletion":
                length = -(rng.randint(6, 10))
                # long deletions show a more pronounced 5' offset
                offset = rng.choice([-2, -2, -1, 0, 1])
                return indel_type, length, offset
            elif indel_type == "1bp_insertion":
                # insertion: 5' side (upstream of cut_site) insertion bias
                return indel_type, 1, -1 if rng.random() < 0.65 else 0
            elif indel_type == "2bp_insertion":
                return indel_type, 2, -1 if rng.random() < 0.6 else 0
            else:  # larger_indel
                length = rng.choice([-15, -12, -8, 5, 10])
                offset = rng.choice([-3, -2, -1, 0, 1])
                return indel_type, length, offset
    return "1bp_deletion", -1, 0


def cut_dna(dna: str, guide: GuideRNA,
            repair: str = "NHEJ",
            template: str | None = None,
            rng: random.Random | None = None,
            hdr_efficiency: str = "typical") -> str:
    """Simulate CRISPR cleavage + DNA repair (Paixão 2022 asymmetric NHEJ
    model).

    repair="NHEJ": non-homologous end joining, introduces indels according
                   to the Paixão 2022 asymmetric deletion spectrum
    repair="HDR": homology-directed repair, uses the template for
                  homologous recombination, falls back to NHEJ on failure
    """
    if rng is None:
        rng = random.Random(0)
    cfg = CAS_VARIANTS[guide.cas_variant]
    cut_site = guide.target_position + cfg["cut_offset"]
    if cut_site < 0 or cut_site >= len(dna):
        return dna  # cut site out of range, no edit

    # HDR repair
    if repair == "HDR" and template is not None:
        efficiency = HDR_EFFICIENCY.get(hdr_efficiency, 0.05)
        if rng.random() < efficiency:
            # HDR success: replace the sequence with the template
            # simplified: replace the sequence near cut_site with the
            # template
            # assume the template length equals the replaced length
            cut_blunt = cut_site
            return dna[:cut_blunt] + template + dna[cut_blunt:]
        else:
            # HDR failed, fall back to NHEJ
            repair = "NHEJ"

    # NHEJ repair (Paixão 2022 asymmetric model)
    if repair == "NHEJ":
        indel_type, length, deletion_offset = _sample_indel(
            rng, cut_site=cut_site, dna=dna)
        if length < 0:
            # deletion: delete |length| bp centered at
            # cut_site + deletion_offset
            del_len = -length
            center = cut_site + deletion_offset
            start = max(0, center - del_len // 2)
            end = min(len(dna), start + del_len)
            actual_del = end - start
            if actual_del <= 0:
                return dna
            return dna[:start] + dna[end:]
        elif length > 0:
            # insertion: insert at cut_site + deletion_offset
            insert_pos = cut_site + deletion_offset
            insert_pos = max(0, min(len(dna), insert_pos))
            insert_bases = "".join(rng.choice("ACGT") for _ in range(length))
            return dna[:insert_pos] + insert_bases + dna[insert_pos:]

    return dna


# ============================================================================
# Complete gene editing pipeline
# ============================================================================

def edit_gene(dna: str, target_position: int, new_sequence: str,
              cas_variant: str = "SpCas9",
              template: str | None = None,
              rng: random.Random | None = None,
              use_index: bool = True,
              max_mismatches: int = 3) -> EditResult:
    """Complete gene editing pipeline: design guide -> cleavage -> HDR
    repair.

    target_position: desired edit position
    new_sequence: new sequence to insert
    use_index: when True, use the PAMIndex multi-bucket seed-and-extend
               to speed up off-target search
               (complete: when max_mismatches < num_buckets, only the
               index is used;
               otherwise a full-scan fallback is triggered automatically
               and the results match off_target_score)
    max_mismatches: maximum number of mismatches allowed in the
                    off-target search
    """
    if rng is None:
        rng = random.Random(0)

    # 1. design the guide RNA
    guide = design_guide(dna, cas_variant, position=target_position)

    # 2. detect off-targets
    if use_index:
        # PAMIndex multi-bucket seed-and-extend (complete version)
        index = PAMIndex(dna, cas_variant=cas_variant)
        off_targets = off_target_score_indexed(
            guide, index, max_mismatches=max_mismatches)
    else:
        off_targets = off_target_score(guide, dna, max_mismatches=max_mismatches)
    # exclude the on-target site
    off_targets = [ot for ot in off_targets
                   if ot.position != guide.target_position]

    # 3. cleavage + HDR repair
    if template is None:
        template = new_sequence

    edited = cut_dna(dna, guide, repair="HDR", template=template,
                     rng=rng, hdr_efficiency="typical")

    # determine whether the edit succeeded
    success = (edited != dna)
    if success and new_sequence in edited:
        edit_type = "HDR"
        edit_len = 0
    elif success:
        edit_type = "NHEJ"
        edit_len = len(edited) - len(dna)
    else:
        edit_type = "no_edit"
        edit_len = 0

    return EditResult(
        edited_dna=edited,
        original_dna=dna,
        guide=guide,
        repair="HDR",
        edit_position=guide.target_position,
        edit_type=edit_type,
        edit_length=edit_len,
        success=success,
        off_targets=off_targets,
    )
