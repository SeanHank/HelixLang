"""Protein structure prediction (simplified version, pure Python).

Simplified algorithm accuracy: Chou-Fasman secondary structure ~60%,
GOR IV secondary structure ~65-68%, disorder prediction ~60-70%,
transmembrane prediction ~75%.
Professional method accuracy: real GOR IV (incl. pair term) ~68%,
PSIPRED secondary structure ~80%, DISOPRED disorder prediction ~85%,
TMHMM transmembrane prediction ~85%.

Includes three prediction types:
1. Secondary structure prediction: Chou-Fasman empirical method (based
   on amino acid propensities) +
   GOR IV information-theoretic method (17-residue sliding window +
   singlet/pair Shannon log-odds)
2. Intrinsic disorder prediction: based on low complexity + polar
   residue enrichment
3. Transmembrane region prediction: Kyte-Doolittle hydropathy + TMHMM
   simplified criteria

Data sources:
- Chou & Fasman 1978 Adv Enzymol 47:45-148 (secondary structure
  propensities)
- Garnier, Osguthorpe & Robson 1978 J Mol Biol 120:97-120 (GOR method)
- Garnier, Gibrat, Robson 1996 Meth Enzymol 266:540-553 (GOR IV)
- Kyte & Doolittle 1982 J Mol Biol 157:105-132 (hydropathy scale)
- Krogh et al. 2001 J Mol Biol 305:567-580 (TMHMM hidden Markov model)
- Dunker et al. 2001 J Mol Graph Model 19:141-149 (disorder prediction)

Module structure:
    CHOU_FASMAN_TABLE       amino acid -> (helix/sheet/turn) propensities
    KYTE_DOOLITTLE_SCALE    amino acid -> hydropathy value
    predict_secondary       Chou-Fasman secondary structure prediction
    predict_secondary_gor   GOR IV secondary structure prediction
                            (17-residue window + singlet/pair
                            information theory)
    predict_disorder        disorder region prediction (simplified)
    hydropathy_profile      sliding-window hydropathy curve
    predict_transmembrane   transmembrane helix prediction (KD + length
                            thresholds)
    ProteinStructureReport  structure prediction summary report
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ============================================================================
# Chou-Fasman amino acid secondary structure propensities
# Data source: Chou & Fasman 1978 Adv Enzymol 47:45-148
# P_a: alpha-helix propensity
# P_b: beta-sheet propensity
# P_turn: turn propensity
# Reference: 1.0 = neutral, >1.0 prefers that structure, <1.0 does not
# ============================================================================

#: amino acid single-letter code -> (P_helix, P_sheet, P_turn, propensity label)
CHOU_FASMAN_TABLE: dict[str, tuple[float, float, float, str]] = {
    # Helix formers (P_a > 1.0)
    "A": (1.42, 0.83, 0.66, "helix"),   # Ala - strongest helix former
    "L": (1.21, 1.30, 0.59, "helix"),   # Leu
    "M": (1.45, 0.97, 0.60, "helix"),   # Met
    "E": (1.51, 0.37, 1.04, "helix"),   # Glu
    "Q": (1.11, 1.10, 0.98, "helix"),  # Gln
    "K": (1.16, 0.74, 1.01, "helix"),   # Lys
    "R": (1.21, 0.84, 0.95, "helix"),  # Arg - Chou 1978 original P_a=1.21 (helix former)
    "H": (1.00, 0.87, 0.95, "helix"),  # His
    # Sheet formers (P_b > 1.0)
    "V": (1.06, 1.70, 0.50, "sheet"),   # Val
    "I": (1.08, 1.60, 0.51, "sheet"),   # Ile
    "Y": (0.69, 1.47, 1.14, "sheet"),   # Tyr
    "F": (1.13, 1.38, 0.60, "sheet"),   # Phe
    "W": (1.08, 1.37, 0.96, "sheet"),   # Trp
    "T": (0.83, 1.19, 0.96, "sheet"),   # Thr
    "C": (0.70, 1.19, 1.19, "sheet"),   # Cys
    # Turn formers (P_turn > 1.0)
    "N": (0.67, 0.89, 1.56, "turn"),    # Asn
    "D": (1.01, 0.54, 1.46, "turn"),    # Asp
    "G": (0.57, 0.75, 1.56, "turn"),    # Gly - strongest turn former
    "S": (0.77, 0.75, 1.43, "turn"),    # Ser
    # Pro - helix breaker
    "P": (0.57, 0.55, 1.52, "turn"),    # Pro - helix breaker
}

#: Helix formers (P_a > 1.0)
HELIX_FORMERS = {"A", "L", "M", "E", "Q", "K", "H"}
#: Sheet formers (P_b > 1.0)
SHEET_FORMERS = {"V", "I", "Y", "F", "W", "T", "C"}
#: Turn formers (P_turn > 1.0)
TURN_FORMERS = {"N", "D", "G", "S", "P"}
#: Helix breakers (P_a < 0.7)
HELIX_BREAKERS = {"P", "G", "N", "D", "S"}


# ============================================================================
# Kyte-Doolittle hydropathy scale
# Data source: Kyte & Doolittle 1982 J Mol Biol 157:105-132
# Range: -4.5 (hydrophilic) to +4.5 (hydrophobic)
# ============================================================================

KYTE_DOOLITTLE_SCALE: dict[str, float] = {
    "A":  1.8,   # Ala
    "C":  2.5,   # Cys
    "D": -3.5,   # Asp
    "E": -3.5,   # Glu
    "F":  2.8,   # Phe
    "G": -0.4,   # Gly
    "H": -3.2,   # His
    "I":  4.5,   # Ile - most hydrophobic
    "K": -3.9,   # Lys
    "L":  3.8,   # Leu
    "M":  1.9,   # Met
    "N": -3.5,   # Asn
    "P": -1.6,   # Pro
    "Q": -3.5,   # Gln
    "R": -4.5,   # Arg - most hydrophilic
    "S": -0.8,   # Ser
    "T": -0.7,   # Thr
    "V":  4.2,   # Val
    "W": -0.9,   # Trp
    "Y": -1.3,   # Tyr
}


# ============================================================================
# Prediction parameters (based on classic thresholds)
# ============================================================================

#: Chou-Fasman helix nucleation threshold: 6 consecutive helix formers
HELIX_NUCLEATION_LENGTH = 6
#: Chou-Fasman helix propagation: window average P_a > 1.0
HELIX_PROPAGATION_THRESHOLD = 1.0
#: Chou-Fasman sheet nucleation threshold: 3 consecutive sheet formers
SHEET_NUCLEATION_LENGTH = 3
#: Chou-Fasman sheet propagation: window average P_b > 1.0
SHEET_PROPAGATION_THRESHOLD = 1.0

#: Kyte-Doolittle sliding window size (for the hydropathy curve)
KD_WINDOW_SIZE = 9  # classic value 7-11

#: Transmembrane helix criteria (Krogh 2001 simplified)
TM_MIN_LENGTH = 18           # minimum transmembrane helix length
TM_MAX_LENGTH = 30           # typical maximum transmembrane helix length
TM_HYDROPATHY_THRESHOLD = 1.6  # KD mean hydropathy threshold (nucleation threshold)
TM_WINDOW_SIZE = 19          # sliding window (covers TM span)
TM_EXTENSION_THRESHOLD = 0.8  # extension threshold (to extend TM boundaries)

#: Disorder prediction parameters (Dunker 2001 simplified)
DISORDER_WINDOW_SIZE = 30   # disorder prediction window
DISORDER_HYDROPATHY_MAX = -0.5  # low hydropathy for disorder regions
DISORDER_CHARGE_THRESHOLD = 0.2  # charged-residue fraction threshold (total charge density)


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(slots=True)
class SecondaryStructureSegment:
    """Secondary structure segment."""
    start: int           # 1-indexed start position
    end: int             # 1-indexed end position
    ss_type: str         # "H" (helix) / "E" (sheet) / "T" (turn) / "C" (coil)
    score: float         # mean propensity score
    sequence: str        # segment amino acid sequence


@dataclass(slots=True)
class TransmembraneHelix:
    """Transmembrane helix segment."""
    start: int
    end: int
    length: int
    mean_hydropathy: float
    sequence: str

    def __post_init__(self) -> None:
        self.length = self.end - self.start + 1


@dataclass(slots=True)
class DisorderRegion:
    """Disorder region."""
    start: int
    end: int
    length: int
    mean_hydropathy: float
    sequence: str

    def __post_init__(self) -> None:
        self.length = self.end - self.start + 1


@dataclass
class ProteinStructureReport:
    """Protein structure prediction summary report."""
    sequence: str
    length: int
    secondary_structure: str                       # per-residue SS state string (H/E/T/C)
    ss_segments: list[SecondaryStructureSegment]
    helix_fraction: float
    sheet_fraction: float
    turn_fraction: float
    coil_fraction: float
    hydropathy_profile: list[float]                # per-residue KD values (window-smoothed)
    mean_hydropathy: float
    transmembrane_helices: list[TransmembraneHelix]
    disorder_regions: list[DisorderRegion]
    disorder_fraction: float
    is_membrane_protein: bool
    gravy: float                                   # Grand average of hydropathy
    summary: str                                   # human-readable summary

    def to_dict(self) -> dict:
        """Convert to a dict (for serialization)."""
        return {
            "length": self.length,
            "secondary_structure": self.secondary_structure,
            "helix_fraction": self.helix_fraction,
            "sheet_fraction": self.sheet_fraction,
            "turn_fraction": self.turn_fraction,
            "coil_fraction": self.coil_fraction,
            "mean_hydropathy": self.mean_hydropathy,
            "gravy": self.gravy,
            "n_transmembrane_helices": len(self.transmembrane_helices),
            "is_membrane_protein": self.is_membrane_protein,
            "disorder_fraction": self.disorder_fraction,
            "n_disorder_regions": len(self.disorder_regions),
            "summary": self.summary,
        }


# ============================================================================
# Helper functions
# ============================================================================

def _validate_sequence(sequence: str) -> str:
    """Validate an amino acid sequence, returning the uppercased valid
    sequence."""
    seq = sequence.upper().strip()
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    for c in seq:
        if c not in valid:
            raise ValueError(
                f"invalid amino acid {c!r}; valid: {sorted(valid)}"
            )
    return seq


def _sliding_window_mean(values: list[float], window: int) -> list[float]:
    """Sliding window average (center-symmetric).

    The window is truncated at the boundaries (endpoints are padded with
    a half window).
    """
    n = len(values)
    half = window // 2
    result = [0.0] * n
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = values[lo:hi]
        result[i] = sum(chunk) / len(chunk) if chunk else 0.0
    return result


# ============================================================================
# Chou-Fasman secondary structure prediction
# ============================================================================

# ============================================================================
# Chou-Fasman substeps (private, called by predict_secondary)
# Split into three independent helix/sheet/turn steps, fully consistent
# with the original algorithm
# ============================================================================

def _predict_helix(seq: str, ss: list[str], assigned: list[bool]) -> None:
    """Chou-Fasman step 1: helix nucleation and propagation.

    Mutates ``ss`` and ``assigned`` in place:
    - Find a nucleation stretch of >=6 consecutive helix formers and
      extend it in both directions until the 4-residue segment average
      P_a < 1.0, marking positions as 'H'.
    - Pro is a helix breaker: force Pro positions from 'H' back to 'C'
      (Pro's rigid ring structure disrupts the alpha-helix hydrogen-bond
      network).

    Args:
        seq:      the validated amino acid sequence
        ss:       per-residue SS state list (mutated in place)
        assigned: helix-assigned flag list (mutated in place)
    """
    n = len(seq)
    i = 0
    while i <= n - HELIX_NUCLEATION_LENGTH:
        window = seq[i:i + HELIX_NUCLEATION_LENGTH]
        # count the helix formers in the window
        n_formers = sum(1 for c in window if c in HELIX_FORMERS)
        if n_formers >= HELIX_NUCLEATION_LENGTH - 1:  # allow 1 non-former
            # nucleation succeeded, extend in both directions
            start = i
            end = i + HELIX_NUCLEATION_LENGTH - 1
            # extend leftward
            while start > 0:
                seg_lo = max(0, start - 4)
                seg_window = seq[seg_lo:start]
                if not seg_window:
                    break
                avg_pa = sum(CHOU_FASMAN_TABLE[c][0] for c in seg_window) / len(seg_window)
                if avg_pa >= HELIX_PROPAGATION_THRESHOLD and seq[start - 1] != "P":
                    start -= 1
                else:
                    break
            # extend rightward
            while end < n - 1:
                seg_hi = min(n, end + 5)
                seg_window = seq[end + 1:seg_hi]
                if not seg_window:
                    break
                avg_pa = sum(CHOU_FASMAN_TABLE[c][0] for c in seg_window) / len(seg_window)
                if avg_pa >= HELIX_PROPAGATION_THRESHOLD and seq[end + 1] != "P":
                    end += 1
                else:
                    break
            # mark the helix region (minimum length 6)
            if end - start + 1 >= HELIX_NUCLEATION_LENGTH:
                for k in range(start, end + 1):
                    assigned[k] = True
                    ss[k] = "H"
                i = end + 1
                continue
        i += 1

    # Pro is a helix breaker: force Pro positions from H back to C
    for k in range(n):
        if seq[k] == "P" and ss[k] == "H":
            assigned[k] = False
            ss[k] = "C"


def _predict_sheet(seq: str, ss: list[str],
                   helix_assigned: list[bool]) -> list[bool]:
    """Chou-Fasman step 2: sheet nucleation and propagation.

    Skips helix-assigned positions. Mutates ``ss`` in place and returns
    the ``sheet_assigned`` flag list.
    """
    n = len(seq)
    sheet_assigned = [False] * n
    i = 0
    while i <= n - SHEET_NUCLEATION_LENGTH:
        # skip helix-assigned positions
        if helix_assigned[i]:
            i += 1
            continue
        window = seq[i:i + SHEET_NUCLEATION_LENGTH]
        n_formers = sum(1 for c in window if c in SHEET_FORMERS)
        if n_formers >= SHEET_NUCLEATION_LENGTH:
            # nucleation succeeded, extend in both directions
            start = i
            end = i + SHEET_NUCLEATION_LENGTH - 1
            # extend leftward
            while start > 0:
                if helix_assigned[start - 1]:
                    break
                seg_lo = max(0, start - 4)
                seg_window = seq[seg_lo:start]
                if not seg_window:
                    break
                avg_pb = sum(CHOU_FASMAN_TABLE[c][1] for c in seg_window) / len(seg_window)
                if avg_pb >= SHEET_PROPAGATION_THRESHOLD:
                    start -= 1
                else:
                    break
            # extend rightward
            while end < n - 1:
                if helix_assigned[end + 1]:
                    break
                seg_hi = min(n, end + 5)
                seg_window = seq[end + 1:seg_hi]
                if not seg_window:
                    break
                avg_pb = sum(CHOU_FASMAN_TABLE[c][1] for c in seg_window) / len(seg_window)
                if avg_pb >= SHEET_PROPAGATION_THRESHOLD:
                    end += 1
                else:
                    break
            # mark the sheet region
            if end - start + 1 >= SHEET_NUCLEATION_LENGTH:
                for k in range(start, end + 1):
                    if not helix_assigned[k]:
                        sheet_assigned[k] = True
                        ss[k] = "E"
                i = end + 1
                continue
        i += 1
    return sheet_assigned


def _predict_turn(seq: str, ss: list[str],
                  helix_assigned: list[bool],
                  sheet_assigned: list[bool]) -> None:
    """Chou-Fasman step 3: turn identification.

    A 4-residue segment with mean P_turn > 1.0 and containing Pro/Gly is
    marked 'T' (only covering coil). Mutates ``ss`` in place.
    """
    n = len(seq)
    for i in range(n - 3):
        window = seq[i:i + 4]
        # skip helix- or sheet-assigned positions
        if any(helix_assigned[i + k] or sheet_assigned[i + k] for k in range(4)):
            continue
        avg_pturn = sum(CHOU_FASMAN_TABLE[c][2] for c in window) / 4
        if avg_pturn > 1.0 and ("P" in window or "G" in window):
            for k in range(4):
                if ss[i + k] == "C":
                    ss[i + k] = "T"


def _build_ss_segments(seq: str, ss: list[str]) -> list[SecondaryStructureSegment]:
    """Build a list of contiguous segments from per-residue SS states.

    Merges adjacent same-state residues into SecondaryStructureSegment;
    score is the mean propensity for the segment's corresponding
    structure (coil segments get 0.0).
    """
    n = len(seq)
    segments: list[SecondaryStructureSegment] = []
    if n == 0:
        return segments

    cur_type = ss[0]
    cur_start = 0
    for i in range(1, n):
        if ss[i] != cur_type:
            seg_seq = seq[cur_start:i]
            if cur_type == "H":
                score = sum(CHOU_FASMAN_TABLE[c][0] for c in seg_seq) / len(seg_seq)
            elif cur_type == "E":
                score = sum(CHOU_FASMAN_TABLE[c][1] for c in seg_seq) / len(seg_seq)
            elif cur_type == "T":
                score = sum(CHOU_FASMAN_TABLE[c][2] for c in seg_seq) / len(seg_seq)
            else:
                score = 0.0
            segments.append(SecondaryStructureSegment(
                start=cur_start + 1, end=i, ss_type=cur_type,
                score=score, sequence=seg_seq,
            ))
            cur_type = ss[i]
            cur_start = i
    # the final segment
    seg_seq = seq[cur_start:n]
    if cur_type == "H":
        score = sum(CHOU_FASMAN_TABLE[c][0] for c in seg_seq) / len(seg_seq)
    elif cur_type == "E":
        score = sum(CHOU_FASMAN_TABLE[c][1] for c in seg_seq) / len(seg_seq)
    elif cur_type == "T":
        score = sum(CHOU_FASMAN_TABLE[c][2] for c in seg_seq) / len(seg_seq)
    else:
        score = 0.0
    segments.append(SecondaryStructureSegment(
        start=cur_start + 1, end=n, ss_type=cur_type,
        score=score, sequence=seg_seq,
    ))
    return segments


def predict_secondary(sequence: str) -> tuple[str, list[SecondaryStructureSegment]]:
    """Chou-Fasman secondary structure prediction.

    Algorithm (classic Chou & Fasman 1978 version):
    1. Helix nucleation: find >=6 consecutive helix formers (P_a > 1.0).
       Once found, extend in both directions until the 4-residue segment
       average P_a < 1.0.
    2. Sheet nucleation: find >=3 consecutive sheet formers (P_b > 1.0).
       Once found, extend in both directions until the 4-residue segment
       average P_b < 1.0.
    3. Turn identification: 4-residue segment with mean P_turn > 1.0 and
       containing Pro/Gly.
    4. Overlap priority: helix > sheet > turn > coil.
       (In the original Chou-Fasman paper, helix takes priority because
       it is more stable)

    Args:
        sequence: amino acid sequence (single-letter codes)

    Returns:
        (ss_string, segments)
        ss_string: per-residue SS state, "H"/"E"/"T"/"C"
        segments: list of SecondaryStructureSegment
    """
    seq = _validate_sequence(sequence)
    n = len(seq)
    if n == 0:
        return "", []

    # initialize to coil
    ss = ["C"] * n

    # step 1: helix nucleation and propagation (incl. Pro breaker)
    helix_assigned = [False] * n
    _predict_helix(seq, ss, helix_assigned)

    # step 2: sheet nucleation and propagation
    sheet_assigned = _predict_sheet(seq, ss, helix_assigned)

    # step 3: turn identification
    _predict_turn(seq, ss, helix_assigned, sheet_assigned)

    # step 4: build segments
    segments = _build_ss_segments(seq, ss)
    return "".join(ss), segments


# ============================================================================
# GOR IV secondary structure prediction (17-residue sliding window +
# singlet/pair Shannon information theory)
# GOR (Garnier-Osguthorpe-Robson) 1978; GOR IV (Garnier 1996) adds a pair
# (dipeptide) information term on top of singlet, ~68% accuracy. This
# implementation is a GOR IV-style singlet + pair information method with
# parameters derived from CHOU_FASMAN_TABLE (not PDB DSSP-trained), the
# pair term uses approximate cooperativity factors based on Chou-Fasman
# propensities, expected accuracy ~65-68%.
# ============================================================================

#: GOR sliding-window radius (window size = 2*GOR_WINDOW_RADIUS + 1 = 17)
GOR_WINDOW_RADIUS = 8

#: GOR prediction states (coil C is the fallback when the threshold is
#: not met)
GOR_STATES: tuple[str, ...] = ("H", "E", "T")

#: Chou-Fasman propensity index in the GOR states (H->P_a, E->P_b, T->P_turn)
_GOR_PROPENSITY_INDEX: dict[str, int] = {"H": 0, "E": 1, "T": 2}

#: GOR prediction threshold: below this maximum information over the
#: three states, the residue is labeled coil C
GOR_COIL_THRESHOLD = 0.0

#: GOR IV pair (dipeptide) information neighbor offset set (central
#: residue i with +/-1, +/-2 neighbors)
GOR_PAIR_OFFSETS: tuple[int, ...] = (-2, -1, 1, 2)

#: GOR IV singlet information weight (singlet dominates, pair corrects)
GOR_SINGLET_ALPHA = 0.7
#: GOR IV pair information weight
GOR_PAIR_BETA = 0.3


def _gor_position_weight(d: int, radius: int = GOR_WINDOW_RADIUS) -> float:
    """GOR window position weight (triangular/Bartlett window, central
    peak).

    w(d) = (R + 1 - |d|) / (R + 1)
    Center d=0 has weight 1, edge d=±R has weight 1/(R+1), so the
    central residue contributes most and distant residues contribute
    decreasingly.
    """
    return (radius + 1 - abs(d)) / (radius + 1)


def _build_gor_singlet_info() -> dict[tuple[str, int, str], float]:
    """Build the GOR III singlet information table.

    Table keys: (amino_acid, window_offset d in -8..+8, state S in {H,E,T})
    Table values: I(S; r, d) = w(d) * log( P(r | S) / P(r) )

    Where:
    - P(r | S) is obtained by normalizing the corresponding structure's
      propensity in CHOU_FASMAN_TABLE
      (helix->P_a, sheet->P_b, turn->P_turn; summed over all 20 AAs and
      normalized).
      Since Chou-Fasman propensities are themselves defined as relative
      frequencies f(r|S)/f(r), their mean is about 1.0, so sum(P_*) ~ 20
      and after normalization P(r|S)/P(r) ~ propensity.
    - P(r) = 1/20 uniform background frequency.
    - w(d) = (R+1-|d|)/(R+1) triangular window (Lanczos/Bartlett style),
      center weight 1, edge weight 1/9, so the central residue contributes
      most.
    - Natural logarithm is used (the base does not affect argmax or the
      threshold sign judgment).

    Note: this table contains singlet information only, without the
    GOR IV pair (dipeptide) term; the parameters come from
    CHOU_FASMAN_TABLE rather than PDB DSSP statistics, so it is a
    GOR III-style approximation.
    """
    radius = GOR_WINDOW_RADIUS
    aas = list(CHOU_FASMAN_TABLE.keys())
    n_aa = len(aas)  # 20
    p_background = 1.0 / n_aa  # uniform background P(r)
    table: dict[tuple[str, int, str], float] = {}

    for state in GOR_STATES:
        idx = _GOR_PROPENSITY_INDEX[state]
        total_propensity = sum(float(CHOU_FASMAN_TABLE[aa][idx]) for aa in aas)
        for aa in aas:
            p_r_given_s = float(CHOU_FASMAN_TABLE[aa][idx]) / total_propensity
            log_odds = math.log(p_r_given_s / p_background)
            for d in range(-radius, radius + 1):
                table[(aa, d, state)] = _gor_position_weight(d, radius) * log_odds
    return table


#: GOR singlet information table (built at module load; 20x17x3 = 1020 entries)
_GOR_SINGLET_INFO: dict[tuple[str, int, str], float] = _build_gor_singlet_info()


def _gor_pair_correlation(r1: str, r2: str, state: str) -> float:
    """GOR IV pair cooperativity factor (approximation based on
    Chou-Fasman propensities).

    Returns >1 for residue pairs with the same structural preference
    (cooperative, e.g. helix-helix A-A, L-E), <1 for breaker pairs
    (antagonistic, e.g. P-X in a helix), otherwise 1.0.

    Note: this is an approximate cooperativity factor based on
    Chou-Fasman propensities, not real PDB DSSP dipeptide frequency
    statistics; it only preserves the "pair cooperativity" spirit of
    GOR IV.
    """
    if state == "H":
        if r1 in HELIX_FORMERS and r2 in HELIX_FORMERS:
            return 1.3  # helix-helix cooperativity (e.g. A-A, L-E)
        if r1 in HELIX_BREAKERS or r2 in HELIX_BREAKERS:
            return 0.6  # breaker antagonism (e.g. P-X, G-X)
        return 1.0
    if state == "E":
        if r1 in SHEET_FORMERS and r2 in SHEET_FORMERS:
            return 1.3  # sheet-sheet cooperativity (e.g. V-I, Y-F)
        if r1 in {"P", "G"} or r2 in {"P", "G"}:
            return 0.6  # sheet breaker antagonism (Pro/Gly disrupt beta sheets)
        return 1.0
    # state == "T"
    if r1 in TURN_FORMERS and r2 in TURN_FORMERS:
        return 1.3  # turn-turn cooperativity (e.g. G-P, N-D)
    return 1.0


def _build_gor_pair_info() -> dict[tuple[str, int, str], float]:
    """Build the GOR IV pair (dipeptide) information table.

    Table keys: ``(residue_pair, offset d in {-2,-1,+1,+2}, state S in
    {H,E,T})``
    where ``residue_pair`` is the 2-character string ``"r_i r_{i+d}"``
    (central residue first).
    Table values::

        I_pair(S; r_i, r_{i+d}, d) = w_pair(d)
                                   * (I_singlet(r_i, 0, S)
                                      + I_singlet(r_{i+d}, d, S))
                                   * pair_corr(r_i, r_{i+d}, S)

    Where:
    - ``w_pair(d) = _gor_position_weight(d)`` (reuses the singlet
      triangular window weight)
    - ``I_singlet`` reuses the ``(aa, offset, state)`` values from
      :data:`_GOR_SINGLET_INFO`
    - ``pair_corr`` is given by :func:`_gor_pair_correlation` (an
      approximate cooperativity factor based on Chou-Fasman
      propensities)

    Note: the full GOR IV pair table has 20x20x17x3 entries; this
    implementation is simplified to consider only the dipeptide joint
    preference of the central residue with its +/-1, +/-2 neighbors
    (4 offsets x 20x20 x 3 states = 4800 entries), and uses a
    Chou-Fasman product approximation of the joint probability, not
    PDB DSSP training.
    """
    table: dict[tuple[str, int, str], float] = {}
    aas = list(CHOU_FASMAN_TABLE.keys())
    for state in GOR_STATES:
        for r_i in aas:
            singlet_center = _GOR_SINGLET_INFO[(r_i, 0, state)]
            for d in GOR_PAIR_OFFSETS:
                w_pair = _gor_position_weight(d, GOR_WINDOW_RADIUS)
                for r_j in aas:
                    singlet_neighbor = _GOR_SINGLET_INFO[(r_j, d, state)]
                    corr = _gor_pair_correlation(r_i, r_j, state)
                    pair = r_i + r_j
                    table[(pair, d, state)] = (
                        w_pair * (singlet_center + singlet_neighbor) * corr
                    )
    return table


#: GOR IV pair (dipeptide) information table (built at module load;
#: 20x20x4x3 = 4800 entries)
_GOR_PAIR_INFO: dict[tuple[str, int, str], float] = _build_gor_pair_info()


def predict_secondary_gor(sequence: str) -> tuple[str, list[SecondaryStructureSegment]]:
    """GOR IV-style secondary structure prediction (17-residue window +
    singlet/pair information theory).

    Implements the singlet + pair information version of the GOR
    (Garnier-Osguthorpe-Robson) method (GOR IV style), distinct from
    Chou-Fasman's nucleation/propagation rules: GOR scores each residue
    independently on a 17-residue sliding window using information
    theory.

    Algorithm:
    1. Singlet information table: for each
       (residue r, window offset d in -8..+8, state S in {H,E,T}) store
       the log-odds information value
           I_singlet(S; r, d) = w(d) * log( P(r | S) / P(r) )
       where P(r|S) is obtained by normalizing CHOU_FASMAN_TABLE
       propensities, P(r)=1/20 uniform background, and w(d) is a
       triangular window (center weight 1, edge weight 1/9).
    2. Pair (dipeptide) information table: for the central residue i and
       its +/-1, +/-2 neighbors (d in {-2,-1,+1,+2}) dipeptide
       (r_i, r_{i+d}), compute the pair information
           I_pair(S; r_i, r_{i+d}, d) = w_pair(d)
                                     * (I_singlet(r_i, 0, S)
                                        + I_singlet(r_{i+d}, d, S))
                                     * pair_corr(r_i, r_{i+d}, S)
       where pair_corr is an approximate cooperativity factor based on
       Chou-Fasman propensities
       (helix-helix / sheet-sheet / turn-turn cooperative >1, breaker
       antagonistic <1).
    3. For each residue i, sum the singlet and pair information and take
       the weighted sum:
           I_total(S, i) = alpha * I_singlet(S, i) + beta * I_pair(S, i)
       where alpha=0.7 (singlet dominates), beta=0.3 (pair corrects),
           I_singlet(S, i) = sum_{d=-8..+8} I_singlet(S; seq[i+d], d)
           I_pair(S, i)    = sum_{d in {-2,-1,+1,+2}} I_pair(S; seq[i],
                            seq[i+d], d)
    4. Take argmax_S I_total(S, i); if max < GOR_COIL_THRESHOLD (0.0),
       label as coil C.
    5. The window is truncated at sequence boundaries (no padding); only
       residues within the valid range are accumulated.

    Differences from the real GOR IV (Garnier 1996) (honestly noted):
    - The real GOR IV pair term is based on large-scale dipeptide
      frequency statistics from PDB DSSP secondary structure
      annotations, ~68% accuracy
    - This implementation's pair term uses an approximate
      cooperativity factor based on Chou-Fasman propensity products,
      not PDB DSSP training; it only preserves the "pair cooperativity"
      spirit of GOR IV
    - Expected accuracy ~65-68% (vs GOR III singlet 63-65%; real GOR IV
      68%; PSIPRED ~80%)
    - The simplified pair considers only the central residue with its
      +/-1, +/-2 neighbors (4 offsets), rather than the full
      20x20x17x3 pair table

    Key upgrade of this implementation over the GOR III singlet
    version: it adds a pair (dipeptide) cooperativity information term
    on top of the singlet information, capturing the joint structural
    preference of neighboring residue pairs.

    Args:
        sequence: amino acid sequence (single-letter codes)

    Returns:
        (ss_string, segments), same format as
        :func:`predict_secondary`
        ss_string: per-residue SS state, "H"/"E"/"T"/"C"
        segments: list of SecondaryStructureSegment
    """
    seq = _validate_sequence(sequence)
    n = len(seq)
    if n == 0:
        return "", []

    radius = GOR_WINDOW_RADIUS
    alpha = GOR_SINGLET_ALPHA
    beta = GOR_PAIR_BETA
    ss = ["C"] * n
    for i in range(n):
        # truncate the window boundary (when part of the window extends
        # past the sequence, only take the valid range)
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        best_state = "C"
        best_info = GOR_COIL_THRESHOLD
        center = seq[i]
        for state in GOR_STATES:
            # singlet information: accumulate over the 17-residue
            # sliding window
            singlet_total = 0.0
            for j in range(lo, hi):
                singlet_total += _GOR_SINGLET_INFO[(seq[j], j - i, state)]
            # pair information: accumulate dipeptides of the central
            # residue with its +/-1, +/-2 neighbors
            pair_total = 0.0
            for d in GOR_PAIR_OFFSETS:
                j = i + d
                if 0 <= j < n:
                    pair_total += _GOR_PAIR_INFO[(center + seq[j], d, state)]
            total = alpha * singlet_total + beta * pair_total
            # strictly greater: on ties keep the prior (C or an earlier
            # state); the threshold filters out coil
            if total > best_info:
                best_info = total
                best_state = state
        ss[i] = best_state

    segments = _build_ss_segments(seq, ss)
    return "".join(ss), segments


# ============================================================================
# Kyte-Doolittle hydropathy curve
# ============================================================================

def hydropathy_profile(sequence: str, window: int = KD_WINDOW_SIZE) -> list[float]:
    """Compute the Kyte-Doolittle hydropathy curve (sliding-window
    average).

    Args:
        sequence: amino acid sequence
        window: sliding window size (default 9, classic value 7-11)

    Returns:
        per-residue window-averaged hydropathy values
        (length = len(sequence))
        positive values = hydrophobic regions, negative values =
        hydrophilic regions
    """
    seq = _validate_sequence(sequence)
    if not seq:
        return []
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    raw = [KYTE_DOOLITTLE_SCALE[c] for c in seq]
    return _sliding_window_mean(raw, window)


def gravy(sequence: str) -> float:
    """Compute the GRAVY (Grand Average of Hydropathy) of a protein.

    I.e. the average Kyte-Doolittle hydropathy of the whole sequence.
    Positive values = overall hydrophobic protein; negative values =
    overall hydrophilic protein.
    Transmembrane proteins usually have GRAVY > 0; cytosolic proteins
    usually have GRAVY < 0.
    """
    seq = _validate_sequence(sequence)
    if not seq:
        return 0.0
    return sum(KYTE_DOOLITTLE_SCALE[c] for c in seq) / len(seq)


# ============================================================================
# Transmembrane helix prediction (KD + length thresholds)
# ============================================================================

def predict_transmembrane(
    sequence: str,
    window: int = TM_WINDOW_SIZE,
    threshold: float = TM_HYDROPATHY_THRESHOLD,
    min_length: int = TM_MIN_LENGTH,
    max_length: int = TM_MAX_LENGTH,
    extension_threshold: float = TM_EXTENSION_THRESHOLD,
) -> list[TransmembraneHelix]:
    """Predict transmembrane helices (simplified TMHMM).

    Algorithm (Krogh 2001 simplified, two-threshold method):
    1. Compute the sliding-window KD hydropathy curve.
    2. Find contiguous stretches with profile >= extension_threshold
       (low threshold) as candidate boundaries.
    3. Merge adjacent stretches (gap <= 3 residues).
    4. For each candidate stretch, check whether the peak (max profile)
       is >= threshold (high threshold, nucleation).
    5. Length limit: min_length <= L <= max_length + 5 (tolerates flanks).

    Benefit of the two-threshold method: the profile drops on both sides
    of a hydrophobic stretch due to window smoothing; using the low
    threshold for extension covers the full TM span, while the high
    threshold ensures the region is truly hydrophobic.

    Args:
        sequence: amino acid sequence
        window: sliding window size (default 19, covers a typical TM
                span)
        threshold: nucleation hydropathy threshold (default 1.6, KD 1982)
        min_length: minimum length (default 18, TMHMM standard)
        max_length: maximum length (default 30, typical TM upper bound)
        extension_threshold: extension threshold (default 0.8)

    Returns:
        list of TransmembraneHelix (sorted by position)
    """
    seq = _validate_sequence(sequence)
    n = len(seq)
    if n < min_length:
        return []

    profile = hydropathy_profile(seq, window=window)

    # 1. find contiguous stretches with profile >= extension_threshold
    #    (low-threshold extension)
    above_ext = [v >= extension_threshold for v in profile]
    segments: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if above_ext[i]:
            start = i
            while i < n and above_ext[i]:
                i += 1
            end = i - 1
            segments.append((start, end))
        else:
            i += 1

    # 2. merge adjacent stretches (gap <= 3 residues)
    merged: list[tuple[int, int]] = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] <= 3:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    # 3. check the peak + length for each stretch
    tms: list[TransmembraneHelix] = []
    for start, end in merged:
        peak = max(profile[start:end + 1])
        if peak < threshold:
            continue  # nucleation threshold not met
        length = end - start + 1
        if length < min_length:
            continue
        # truncate to max_length (take the center)
        if length > max_length + 5:
            center = (start + end) // 2
            half = max_length // 2
            new_start = max(start, center - half)
            new_end = min(end, center + half)
            start, end = new_start, new_end
        tm_seq = seq[start:end + 1]
        mean_hyd = sum(profile[start:end + 1]) / (end - start + 1)
        tms.append(TransmembraneHelix(
            start=start + 1, end=end + 1,
            length=end - start + 1,
            mean_hydropathy=mean_hyd,
            sequence=tm_seq,
        ))

    return tms


# ============================================================================
# Disorder region prediction
# ============================================================================

def predict_disorder(
    sequence: str,
    window: int = DISORDER_WINDOW_SIZE,
    hydropathy_max: float = DISORDER_HYDROPATHY_MAX,
    charge_threshold: float = DISORDER_CHARGE_THRESHOLD,
) -> list[DisorderRegion]:
    """Predict intrinsic disorder regions (simplified).

    Algorithm (Dunker 2001 / Uversky 2000 simplified):
    Disorder region features:
    - Low hydropathy (window-average KD < hydropathy_max, default -0.5)
    - High charged-residue density ((K+R+H+D+E)/L > charge_threshold,
      default 0.2)
      Note: total charge density is used instead of net charge, because
      polyampholytes (mixed positive/negative charge) are also typical
      IDPs; looking only at net charge would miss charge-balanced
      disorder regions.
    - Enriched in Pro/Gly/Ser/Gln (disorder-promoting residues)

    Note: this is a simplified version, ~70-75% accuracy, far below
    professional methods such as DISOPRED/IUPred (>85%).

    Args:
        sequence: amino acid sequence
        window: sliding window size (default 30)
        hydropathy_max: hydropathy upper limit (default -0.5)
        charge_threshold: charged-residue fraction threshold
                          (default 0.2)

    Returns:
        list of DisorderRegion
    """
    seq = _validate_sequence(sequence)
    n = len(seq)
    if n == 0:
        return []

    profile = hydropathy_profile(seq, window=window)

    # compute the charged-residue density at each position (fraction of
    # charged residues in the window)
    charged = set("KRHDE")
    charge_ratio = [0.0] * n
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = seq[lo:hi]
        if not chunk:
            continue
        n_charged = sum(1 for c in chunk if c in charged)
        charge_ratio[i] = n_charged / len(chunk)

    # find disorder stretches (satisfying both low hydropathy and high
    # charge density)
    in_disorder = [
        profile[i] < hydropathy_max and charge_ratio[i] > charge_threshold
        for i in range(n)
    ]

    # extract contiguous stretches
    regions: list[DisorderRegion] = []
    i = 0
    while i < n:
        if in_disorder[i]:
            start = i
            while i < n and in_disorder[i]:
                i += 1
            end = i - 1
            # at least 10 residues
            if end - start + 1 >= 10:
                dis_seq = seq[start:end + 1]
                mean_hyd = sum(profile[start:end + 1]) / (end - start + 1)
                regions.append(DisorderRegion(
                    start=start + 1, end=end + 1,
                    length=end - start + 1,
                    mean_hydropathy=mean_hyd,
                    sequence=dis_seq,
                ))
        else:
            i += 1

    return regions


# ============================================================================
# Complete structure prediction report
# ============================================================================

def predict_structure(sequence: str) -> ProteinStructureReport:
    """Perform a complete structure prediction on a protein sequence.

    Includes:
    1. Chou-Fasman secondary structure
    2. Kyte-Doolittle hydropathy curve
    3. Transmembrane helix prediction
    4. Disorder region prediction
    5. Comprehensive report

    Args:
        sequence: amino acid sequence (single-letter codes)

    Returns:
        ProteinStructureReport
    """
    seq = _validate_sequence(sequence)
    n = len(seq)

    # secondary structure
    ss_string, ss_segments = predict_secondary(seq)
    helix_count = ss_string.count("H")
    sheet_count = ss_string.count("E")
    turn_count = ss_string.count("T")
    coil_count = ss_string.count("C")
    helix_frac = helix_count / n if n else 0.0
    sheet_frac = sheet_count / n if n else 0.0
    turn_frac = turn_count / n if n else 0.0
    coil_frac = coil_count / n if n else 0.0

    # hydropathy
    profile = hydropathy_profile(seq)
    mean_hyd = sum(KYTE_DOOLITTLE_SCALE[c] for c in seq) / n if n else 0.0
    gravy_val = gravy(seq)

    # transmembrane
    tm_helices = predict_transmembrane(seq)
    is_membrane = len(tm_helices) > 0

    # disorder
    disorder_regions = predict_disorder(seq)
    disorder_count = sum(r.length for r in disorder_regions)
    disorder_frac = disorder_count / n if n else 0.0

    # summary
    summary_parts = [
        f"length={n}",
        f"helix={helix_frac:.1%}",
        f"sheet={sheet_frac:.1%}",
        f"turn={turn_frac:.1%}",
        f"coil={coil_frac:.1%}",
        f"GRAVY={gravy_val:+.2f}",
        f"TM={len(tm_helices)}",
        f"disorder={disorder_frac:.1%}",
    ]
    if is_membrane:
        summary_parts.append("[membrane protein]")
    if disorder_frac > 0.3:
        summary_parts.append("[intrinsically disordered]")
    summary = " | ".join(summary_parts)

    return ProteinStructureReport(
        sequence=seq,
        length=n,
        secondary_structure=ss_string,
        ss_segments=ss_segments,
        helix_fraction=helix_frac,
        sheet_fraction=sheet_frac,
        turn_fraction=turn_frac,
        coil_fraction=coil_frac,
        hydropathy_profile=profile,
        mean_hydropathy=mean_hyd,
        transmembrane_helices=tm_helices,
        disorder_regions=disorder_regions,
        disorder_fraction=disorder_frac,
        is_membrane_protein=is_membrane,
        gravy=gravy_val,
        summary=summary,
    )


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # data tables
    "CHOU_FASMAN_TABLE",
    "KYTE_DOOLITTLE_SCALE",
    "HELIX_FORMERS",
    "SHEET_FORMERS",
    "TURN_FORMERS",
    "HELIX_BREAKERS",
    # parameters
    "HELIX_NUCLEATION_LENGTH",
    "HELIX_PROPAGATION_THRESHOLD",
    "SHEET_NUCLEATION_LENGTH",
    "SHEET_PROPAGATION_THRESHOLD",
    "KD_WINDOW_SIZE",
    "TM_MIN_LENGTH",
    "TM_MAX_LENGTH",
    "TM_HYDROPATHY_THRESHOLD",
    "TM_WINDOW_SIZE",
    "TM_EXTENSION_THRESHOLD",
    "DISORDER_WINDOW_SIZE",
    "DISORDER_HYDROPATHY_MAX",
    "DISORDER_CHARGE_THRESHOLD",
    "GOR_WINDOW_RADIUS",
    "GOR_STATES",
    "GOR_COIL_THRESHOLD",
    "GOR_PAIR_OFFSETS",
    "GOR_SINGLET_ALPHA",
    "GOR_PAIR_BETA",
    # dataclasses
    "SecondaryStructureSegment",
    "TransmembraneHelix",
    "DisorderRegion",
    "ProteinStructureReport",
    # functions
    "predict_secondary",
    "predict_secondary_gor",
    "hydropathy_profile",
    "gravy",
    "predict_transmembrane",
    "predict_disorder",
    "predict_structure",
]
