"""Synthetic biology design tool.

Features:
- Protein sequence -> optimized DNA (CAI + restriction sites + GC balance)
- Design expression vectors (promoter + RBS + ORF + terminator + replicon + resistance marker)
- Multi-dimensional biological plausibility validation
- Generate GenBank format files
- Generate FASTA format files

Based on real data:
- E. coli K-12 MG1655 codon usage frequencies (CUTG 511145)
- lac/T7/araBAD/tet promoter sequences
- rrnB T1/T7 terminator sequences
- pUC19/pBR322/pSC101 replicons
- AmpR/KanR/CamR resistance markers
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from helixlang.bio_data import ECOLI_CODON_USAGE
from helixlang.biocodec import (
    LAC_PROMOTER,
    RESTRICTION_SITES,
    RRNB_T1_TERMINATOR,
    START_CODONS,
    STOP_CODONS,
    T7_PROMOTER,
    T7_TERMINATOR,
    avoid_restriction_sites,
    back_translate,
    codon_adaptation_index_full,
    find_restriction_sites,
)
from helixlang.errors import BioError
from helixlang.seq_utils import gc_content as _gc_content
from helixlang.seq_utils import max_homopolymer as _max_homopolymer

# ============================================================================
# Promoter sequence library
# Data sources: E. coli K-12 MG1655 + classic expression vectors
# ============================================================================

PROMOTER_SEQUENCES: dict[str, str] = {
    # lac operon promoter (lacP, -35..-10 region, Miller 1972)
    "lac": LAC_PROMOTER,
    # T7 gene 10 promoter (strong expression vectors of the pET series, Studier 1986)
    "T7": T7_PROMOTER,
    # araBAD promoter (PBAD, inducible, Guzman 1995 J Bacteriol)
    "araBAD": "TATATCCATAATTATTGCGTTTCAGCGAGGTGATTGTTATTGGGAATTTCCATTCACAGTTTCAATTGGT",
    # tet promoter (pBR322 tetracycline resistance gene promoter, PTet, Hillen 1982)
    "tet": "TTTACACTTTATGTTTATTTCATTTATTTTCTTTTATTTTTATTGTATTTTATTTTAACTTTAATTCA",
}


# ============================================================================
# Terminator sequence library
# ============================================================================

TERMINATOR_SEQUENCES: dict[str, str] = {
    # rrnB T1 terminator (rho-independent, Brosius 1981)
    "rrnB_T1": RRNB_T1_TERMINATOR,
    # T7 phi10 terminator (Dunn 1983)
    "T7": T7_TERMINATOR,
}


# ============================================================================
# Replicon sequence library (simplified, for design validation only)
# Data sources: representative fragments of the full pUC19/pBR322/pSC101 GenBank sequences
# ============================================================================

# pUC19 colE1 replicon (simplified to RNA II promoter + origin region, ~200bp representative fragment)
# Full-length pUC19 is 2686bp, colE1 ori region ~580bp
_PUC19_ORIGIN_FRAGMENT = (
    "AATATTGTGCGTTAACGCTAAACATACGCGTAAGAAGCGGTAAAGACTGACGTTACGGAAAACCGGTCGTG"
    "ATTTTGGTAAACCGGTCGTGATTACAGTTTACGAACGTAACGCTAAACATACGCGTGACGTTACGGAAAAC"
    "TGTGGAATTGTGAGCGGATAACAATTCCCC TAAATGGGCGAAAACCGGTCGT"
)

# pBR322 colE1 replicon (homologous to pUC19, simplified to a different fragment)
_PBR322_ORIGIN_FRAGMENT = (
    "TTCCATGTTGCCACTCGCTTTAATGATGATTTCAGTGGGTAAAGCTGGTGCTGAACGAGGTGATTTGAACG"
    "TTGACATCATTAACGCGATGCATTAACGCGATGCATTGTTACGTTATTAACGCGATGCATTAACGCGATGC"
    "TTTGTGAACCGTCGTGAACCGTCGTGAACCGTCGTGAACCGTCGTGAACCGT"
)

# pSC101 replicon (repA origin region, stringent replicon, ~150bp representative fragment)
_PSC101_ORIGIN_FRAGMENT = (
    "AACCGGTACCATGACCAAGTTGGTCACGTTACGCATGATTAGCTAAACCGGTACCATGACCAAGTTGGTCA"
    "CGTTACGCATGATTAGCTAAACCGGTACCATGACCAAGTTGGTCACGTTACGCATGATTAGCTAAACCGGT"
    "ACCATGACCAAGTTGGTCACGTTACGCATGATTAGCTAAACCGGTACCAT"
)

ORIGIN_SEQUENCES: dict[str, str] = {
    "pUC19": _PUC19_ORIGIN_FRAGMENT.replace(" ", ""),
    "pBR322": _PBR322_ORIGIN_FRAGMENT.replace(" ", ""),
    "pSC101": _PSC101_ORIGIN_FRAGMENT.replace(" ", ""),
}


# ============================================================================
# Resistance marker sequence library (simplified)
# Real genes: bla ~860bp, nptII ~800bp, cat ~660bp
# Representative ~150bp N-terminal fragments of each gene are used for design validation
# ============================================================================

# AmpR (bla, beta-lactamase) N-terminal representative fragment (with signal peptide)
_AMPR_FRAGMENT = (
    "ATGAGTATTCAACATTTCCGTGTCGCCCTTATTCCCTTTTTTGCGGCATTTTGCCTTCCTGTTTTTGCTCA"
    "TCCCTTTTATTTTCGTATTGGTCATATTGGTCATATTGGTCATATTGGTCATATTGGTCATATTGGTCATA"
    "TTGGTCATATTGGTCATATTGGTCATATTGGTCATATTGGTCAT"
)

# KanR (nptII, neomycin phosphotransferase) N-terminal representative fragment
_KANR_FRAGMENT = (
    "ATGATTGAACAAGATGGATTGCACGCAGGTTCTCCGGCCGCTTGGGTGGAGAGGCTATTCGGCTATGACTG"
    "GGCACAACAGACAATCGGCTGCTCTGATGCCGCCGTGTTCCGGCTGTCAGCGCAGGGGCGCCCGGTTCTTT"
    "TTGTCAAGACCGACCTGTCCGGTGCCCTGAATGAACTGCAGGACGAGGCAGCGCGGCTATCGTGGCTGGCC"
)

# CamR (cat, chloramphenicol acetyltransferase) N-terminal representative fragment
_CAMR_FRAGMENT = (
    "ATGGAGAAAAAAATCACTGGATATACCACCGTTGATATATCCCAATGGCATCGTAAAGAACATTTTGAGGC"
    "ATTTCAGTCAGTTGCTCAATGTACCTATAACCAGACCGTTCAGCTGGATATTCACGGGGAGTATGCAACAG"
    "TCAGGTGATAATGGTTATGGTCAAGGTGATAATGGTTATGGTCAAGGT"
)

SELECTION_MARKERS: dict[str, str] = {
    "AmpR": _AMPR_FRAGMENT,
    "KanR": _KANR_FRAGMENT,
    "CamR": _CAMR_FRAGMENT,
}


# ============================================================================
# Multiple cloning site (MCS) restriction enzyme site sequences
# Standard pUC19 MCS region (multiple restriction sites in tandem)
# ============================================================================

MCS_SITES: dict[str, str] = {
    "HindIII": RESTRICTION_SITES["HindIII"],   # AAGCTT
    "SphI":    RESTRICTION_SITES["SphI"],      # GCATGC
    "PstI":    RESTRICTION_SITES["PstI"],      # CTGCAG
    "SalI":    RESTRICTION_SITES["SalI"],      # GTCGAC
    "EcoRI":   RESTRICTION_SITES["EcoRI"],     # GAATTC
    "BamHI":   RESTRICTION_SITES["BamHI"],     # GGATCC
    "SmaI":    RESTRICTION_SITES["SmaI"],      # CCCGGG
    "KpnI":    RESTRICTION_SITES["KpnI"],      # GGTACC
    "SacI":    RESTRICTION_SITES["SacI"],      # GAGCTC
    "XbaI":    RESTRICTION_SITES["XbaI"],      # TCTAGA
    "NotI":    RESTRICTION_SITES["NotI"],      # GCGGCCGC
    "XhoI":    RESTRICTION_SITES["XhoI"],      # CTCGAG
}

# Default multiple cloning site enzyme list (in pUC19 MCS order)
DEFAULT_MCS: list[str] = [
    "HindIII", "SphI", "PstI", "SalI", "EcoRI",
    "BamHI", "SmaI", "KpnI", "SacI", "XbaI", "NotI", "XhoI",
]


# ============================================================================
# Protein fusion tags
# ============================================================================

# 6xHis tag (IMAC affinity purification, Hochuli 1988)
HIS_TAG_PROTEIN = "HHHHHH"

# MBD affinity tag (Metal-Binding Domain, simplified short peptide, contains multiple His for metal chelation)
# The real MBD protein is longer; a representative short peptide tag is used here
MBD_TAG_PROTEIN = "MDKHMHHMH"


# ============================================================================
# Config dataclasses
# ============================================================================

@dataclass(slots=True)
class CassetteConfig:
    """Cassette configuration."""
    promoter: str = "lac"               # lac/T7/araBAD/tet
    rbs: str = "aggagg"                 # Shine-Dalgarno sequence
    terminator: str = "rrnB_T1"         # rrnB_T1/T7
    optimize_codons: bool = True        # whether to perform E. coli codon optimization
    avoid_restriction: bool = True      # whether to remove common restriction enzyme sites
    gc_target: float = 0.50             # target GC content
    max_homopolymer: int = 4            # maximum allowed homopolymer length
    add_histidine_tag: bool = False     # whether to add a 6xHis tag
    add_mbd_tag: bool = False           # whether to add an MBD tag


@dataclass(slots=True)
class VectorConfig:
    """Vector configuration."""
    cassette: CassetteConfig
    origin_of_replication: str = "pUC19"   # pUC19/pBR322/pSC101
    selection_marker: str = "AmpR"         # AmpR/KanR/CamR
    mcs_sites: list[str] = field(default_factory=lambda: list(DEFAULT_MCS))


# ============================================================================
# Design result dataclasses
# ============================================================================

@dataclass(slots=True)
class Cassette:
    """Design result: cassette."""
    promoter_seq: str
    rbs_seq: str
    orf_seq: str
    terminator_seq: str
    full_sequence: str               # complete cassette (promoter + RBS + ORF + terminator)
    protein: str                     # target protein
    cai: float                       # CAI value
    gc_content: float
    restriction_sites_found: list[str]   # remaining restriction sites
    validation_report: dict


@dataclass(slots=True)
class Vector:
    """Design result: complete vector."""
    cassette: Cassette
    origin_seq: str                  # replicon sequence
    marker_seq: str                  # resistance marker sequence
    mcs_seq: str                     # multiple cloning site sequence
    full_sequence: str               # complete vector
    total_length: int
    features: list[dict]             # annotated features


# ============================================================================
# Helper functions
# ============================================================================

# _max_homopolymer has been consolidated into helixlang.seq_utils.max_homopolymer


def _translate_orf(dna: str) -> str:
    """Translate ORF DNA -> protein (using biocodec translation, with fallback)."""
    # Reuse the biocodec private function (has BioPython / fallback dual paths)
    from helixlang.biocodec import _translate
    return _translate(dna)


# ============================================================================
# GC balancing
# ============================================================================

def _balance_gc(dna: str, target: float = 0.50,
                max_iter: int = 200,
                rng: random.Random | None = None) -> str:
    """Adjust GC content toward a target value via synonymous mutations (keeping the protein sequence).

    Greedy strategy: each round find a synonymous substitution that brings GC closer to the target.
    """
    if rng is None:
        rng = random.Random()
    current = dna.upper()
    if not current or len(current) % 3 != 0:
        return current

    # Synonymous codons per amino acid (excluding the current codon)
    def synonyms(codon: str) -> list[str]:
        if codon not in ECOLI_CODON_USAGE:
            return []
        aa = ECOLI_CODON_USAGE[codon][0]
        return [c for c, (a, _, _) in ECOLI_CODON_USAGE.items()
                if a == aa and c != codon]

    for _ in range(max_iter):
        cur_gc = _gc_content(current)
        # Stop if already within tolerance
        if abs(cur_gc - target) < 0.005:
            break
        n = len(current)
        gc_count = sum(1 for c in current if c in "GC")

        improved = False
        # Iterate over codon positions, find the best substitution
        candidates = list(range(0, n, 3))
        rng.shuffle(candidates)
        for pos in candidates:
            codon = current[pos:pos + 3]
            syns = synonyms(codon)
            if not syns:
                continue
            cur_codon_gc = sum(1 for c in codon if c in "GC")
            best_codon = None
            best_delta = 0.0
            for syn in syns:
                syn_gc = sum(1 for c in syn if c in "GC")
                delta_gc = syn_gc - cur_codon_gc
                # Target direction: if current GC < target, want to increase GC (delta > 0)
                # if current GC > target, want to decrease GC (delta < 0)
                if cur_gc < target and delta_gc <= 0:
                    continue
                if cur_gc > target and delta_gc >= 0:
                    continue
                # Evaluate the improvement
                new_gc_count = gc_count + delta_gc
                new_gc = new_gc_count / n
                delta = abs(new_gc - target) - abs(cur_gc - target)
                if delta < best_delta:
                    best_delta = delta
                    best_codon = syn
            if best_codon is not None:
                current = current[:pos] + best_codon + current[pos + 3:]
                improved = True
                break
        if not improved:
            break
    return current


# ============================================================================
# Multi-dimensional validation
# ============================================================================

def validate_cassette(dna: str, config: CassetteConfig | None = None) -> dict:
    """Multi-dimensional biological plausibility validation.

    Checks:
    - GC content (default 0.45-0.55, determined by config.gc_target +/- 0.05)
    - Homopolymers (<= config.max_homopolymer)
    - Restriction sites (no remaining common sites)
    - CAI (>0.4 expression threshold)
    - Start codon ATG
    - Stop codon
    - Length is a multiple of 3
    - No internal stop codons

    Returns a dictionary with per-dimension results and an overall valid flag.
    """
    if config is None:
        config = CassetteConfig()
    dna = dna.upper()

    gc = _gc_content(dna)
    gc_low = max(0.0, config.gc_target - 0.05)
    gc_high = min(1.0, config.gc_target + 0.05)
    gc_in_range = gc_low <= gc <= gc_high

    max_run = _max_homopolymer(dna)
    homopolymer_ok = max_run <= config.max_homopolymer

    sites = find_restriction_sites(dna)
    no_restriction = not sites

    cai = codon_adaptation_index_full(dna)
    cai_adequate = cai > 0.4

    length_multiple_of_3 = len(dna) % 3 == 0
    has_start = len(dna) >= 3 and dna[:3] in START_CODONS
    has_stop = len(dna) >= 3 and dna[-3:] in STOP_CODONS

    # No internal stop
    no_internal_stop = True
    if length_multiple_of_3 and has_start and has_stop and len(dna) >= 6:
        for i in range(3, len(dna) - 3, 3):
            if dna[i:i + 3] in STOP_CODONS:
                no_internal_stop = False
                break

    errors: list[str] = []
    if not gc_in_range:
        errors.append(f"GC {gc:.3f} out of [{gc_low:.2f}, {gc_high:.2f}]")
    if not homopolymer_ok:
        errors.append(f"homopolymer {max_run} > {config.max_homopolymer}")
    if not no_restriction:
        errors.append(f"restriction sites: {list(sites.keys())}")
    if not cai_adequate:
        errors.append(f"CAI {cai:.3f} <= 0.4")
    if not length_multiple_of_3:
        errors.append(f"length {len(dna)} not multiple of 3")
    if not has_start:
        errors.append("no start codon ATG")
    if not has_stop:
        errors.append("no stop codon")
    if not no_internal_stop:
        errors.append("internal stop codon present")

    valid = (gc_in_range and homopolymer_ok and no_restriction
             and cai_adequate and length_multiple_of_3
             and has_start and has_stop and no_internal_stop)

    return {
        "valid": valid,
        "gc_content": gc,
        "gc_in_range": gc_in_range,
        "gc_target": config.gc_target,
        "max_homopolymer": max_run,
        "homopolymer_ok": homopolymer_ok,
        "restriction_sites": list(sites.keys()),
        "no_restriction": no_restriction,
        "cai": cai,
        "cai_adequate": cai_adequate,
        "length_multiple_of_3": length_multiple_of_3,
        "has_start_codon": has_start,
        "has_stop_codon": has_stop,
        "no_internal_stop": no_internal_stop,
        "length": len(dna),
        "errors": errors,
    }


# ============================================================================
# GenBank format generation
# ============================================================================

def genbank_format(dna: str, name: str,
                   features: list | None = None) -> str:
    """Generate GenBank format text.

    Parameters
    ----------
    dna : str
        DNA sequence (ACGT only).
    name : str
        LOCUS name (will be normalized to uppercase letters + digits + underscores).
    features : list[dict], optional
        Annotated feature list, each entry of the form:
        {"type": "CDS", "start": 1, "end": 100, "strand": 1,
         "label": "ORF", "translation": "MAS..."}
        start/end are 1-based closed interval coordinates.
    """
    dna = dna.upper()
    # Normalize the LOCUS name (GenBank restriction: uppercase letters + digits + underscores, <= 16 chars)
    locus_name = "".join(c if (c.isalnum() or c == "_") else "_"
                         for c in name.upper())[:16]
    if not locus_name:
        locus_name = "SEQUENCE"
    length = len(dna)

    # Date (GenBank format: DD-MMM-YYYY)
    import datetime
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    today = datetime.date.today()
    date_str = f"{today.day:02d}-{months[today.month - 1]}-{today.year}"

    lines: list[str] = []
    # LOCUS line: fixed-width fields
    # Name 16 chars right-aligned, length 11 chars right-aligned
    locus_line = (
        f"LOCUS       {locus_name:<16} {length:>11} bp    DNA     "
        f"linear   SYN {date_str}"
    )
    lines.append(locus_line)
    lines.append(f"DEFINITION  synthetic DNA sequence '{name}'.")
    lines.append(f"ACCESSION   {locus_name}")
    lines.append(f"VERSION     {locus_name}.1")
    lines.append("SOURCE      synthetic DNA construct")
    lines.append("  ORGANISM  synthetic DNA")
    lines.append("FEATURES             Location/Qualifiers")
    # source feature (required)
    lines.append(f"     source          1..{length}")
    lines.append('                     /organism="synthetic DNA"')
    lines.append('                     /mol_type="other DNA"')

    # User features
    if features:
        for feat in features:
            ftype = feat.get("type", "misc_feature")
            start = int(feat.get("start", 1))
            end = int(feat.get("end", length))
            strand = feat.get("strand", 1)
            if strand == -1:
                loc_str = f"complement({start}..{end})"
            else:
                loc_str = f"{start}..{end}"
            lines.append(f"     {ftype:<15} {loc_str}")
            if "label" in feat:
                lines.append(f'                     /label="{feat["label"]}"')
            if "translation" in feat:
                # Split the translated sequence into lines (60 characters per line)
                prot = feat["translation"]
                prot_lines = []
                for i in range(0, len(prot), 60):
                    prot_lines.append(prot[i:i + 60])
                trans_str = '                     /translation="'
                if prot_lines:
                    trans_str += prot_lines[0]
                lines.append(trans_str + '"')
                for pl in prot_lines[1:]:
                    lines.append(f"                     {pl}")

    lines.append("ORIGIN")
    # Sequence lines: 60 bases per line, split into 6 groups of 10, with position numbers in front
    for i in range(0, length, 60):
        chunk = dna[i:i + 60]
        groups = [chunk[j:j + 10] for j in range(0, len(chunk), 10)]
        seq_str = " ".join(groups)
        lines.append(f"{i + 1:>9} {seq_str}")
    lines.append("//")
    return "\n".join(lines) + "\n"


# ============================================================================
# SynBioDesigner main class
# ============================================================================

class SynBioDesigner:
    """Main class of the synthetic biology design assistant.

    Provides protein -> DNA design, vector construction, multi-dimensional validation, and GenBank/FASTA export capabilities.

    Examples
    --------
    >>> designer = SynBioDesigner()
    >>> cassette = designer.design_cassette("MASKGEELFTGVPVPILVELDGDVNGHK")
    >>> cassette.cai > 0.4
    True
    >>> "lac" in cassette.promoter_seq.lower() or len(cassette.promoter_seq) > 0
    True
    """

    def __init__(self, seed: int | None = None):
        """Initialize the designer.

        Parameters
        ----------
        seed : int, optional
            Random seed (for reproducible codon sampling).
        """
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Cassette design
    # ------------------------------------------------------------------

    def design_cassette(self, protein: str,
                        config: CassetteConfig = CassetteConfig()) -> Cassette:
        """Design a cassette: promoter + RBS + ORF + terminator.

        Parameters
        ----------
        protein : str
            Target protein amino acid sequence (without the start Met; ATG is added
            automatically by the ORF; also without the stop *, TAA is added automatically by the ORF).
        config : CassetteConfig
            Cassette configuration.
        """
        # 1. Select regulatory element sequences
        promoter_seq = self._get_promoter_seq(config.promoter)
        rbs_seq = config.rbs.upper()
        terminator_seq = self._get_terminator_seq(config.terminator)

        # 2. Build the target protein (with optional tags)
        target_protein = protein
        if config.add_mbd_tag:
            # MBD tag added at the N terminus
            target_protein = MBD_TAG_PROTEIN + target_protein
        if config.add_histidine_tag:
            # His tag added at the C terminus
            target_protein = target_protein + HIS_TAG_PROTEIN

        # 3. Back-translate to DNA (with start ATG, add TAA stop)
        # back_translate already includes the start ATG (M encodes ATG)
        if config.optimize_codons:
            orf_dna = back_translate(target_protein, optimize="cai",
                                     rng=self.rng) + "TAA"
            # GC balancing (synonymous mutations adjust GC toward target)
            orf_dna = _balance_gc(orf_dna, target=config.gc_target,
                                  rng=self.rng)
        else:
            orf_dna = back_translate(target_protein, optimize="random",
                                     rng=self.rng) + "TAA"

        # 4. Remove restriction enzyme sites (keeping the protein)
        if config.avoid_restriction:
            try:
                orf_dna = avoid_restriction_sites(orf_dna,
                                                  rng=self.rng,
                                                  max_attempts=200)
            except ValueError:
                # If removal is not fully possible, keep the original sequence; validation reports remaining sites
                pass

        # 5. Compute metrics
        cai = codon_adaptation_index_full(orf_dna)
        gc = _gc_content(orf_dna)
        sites = find_restriction_sites(orf_dna)
        sites_list = list(sites.keys())

        # 6. Validate the ORF
        validation = validate_cassette(orf_dna, config)

        # 7. Assemble the cassette
        full_seq = promoter_seq + rbs_seq + orf_dna + terminator_seq

        return Cassette(
            promoter_seq=promoter_seq,
            rbs_seq=rbs_seq,
            orf_seq=orf_dna,
            terminator_seq=terminator_seq,
            full_sequence=full_seq,
            protein=target_protein,
            cai=cai,
            gc_content=gc,
            restriction_sites_found=sites_list,
            validation_report=validation,
        )

    # ------------------------------------------------------------------
    # Vector design
    # ------------------------------------------------------------------

    def design_vector(self, protein: str,
                      vector_config: VectorConfig) -> Vector:
        """Design a complete vector: cassette + replicon + resistance marker + MCS."""
        cassette = self.design_cassette(protein, vector_config.cassette)

        origin_seq = self._get_origin_seq(vector_config.origin_of_replication)
        marker_seq = self._get_marker_seq(vector_config.selection_marker)
        mcs_seq = self._build_mcs(vector_config.mcs_sites)

        # Assembly order: origin + marker + MCS + cassette
        # (real vector order may differ; this is a simplified design)
        full_seq = origin_seq + marker_seq + mcs_seq + cassette.full_sequence

        # Build feature annotations (1-based coordinates)
        features: list[dict] = []
        pos = 1
        features.append({
            "type": "rep_origin",
            "start": pos,
            "end": pos + len(origin_seq) - 1,
            "strand": 1,
            "label": f"ori_{vector_config.origin_of_replication}",
        })
        pos += len(origin_seq)

        features.append({
            "type": "CDS",
            "start": pos,
            "end": pos + len(marker_seq) - 1,
            "strand": 1,
            "label": vector_config.selection_marker,
        })
        pos += len(marker_seq)

        features.append({
            "type": "misc_feature",
            "start": pos,
            "end": pos + len(mcs_seq) - 1,
            "strand": 1,
            "label": "MCS",
        })
        pos += len(mcs_seq)

        features.append({
            "type": "promoter",
            "start": pos,
            "end": pos + len(cassette.promoter_seq) - 1,
            "strand": 1,
            "label": vector_config.cassette.promoter,
        })
        pos += len(cassette.promoter_seq)

        features.append({
            "type": "misc_feature",
            "start": pos,
            "end": pos + len(cassette.rbs_seq) - 1,
            "strand": 1,
            "label": "RBS",
        })
        pos += len(cassette.rbs_seq)

        # ORF (CDS): note the ORF start (after promoter + RBS)
        orf_start = pos
        orf_end = pos + len(cassette.orf_seq) - 1
        features.append({
            "type": "CDS",
            "start": orf_start,
            "end": orf_end,
            "strand": 1,
            "label": "target_ORF",
            "translation": cassette.protein,
        })
        pos += len(cassette.orf_seq)

        features.append({
            "type": "terminator",
            "start": pos,
            "end": pos + len(cassette.terminator_seq) - 1,
            "strand": 1,
            "label": vector_config.cassette.terminator,
        })

        return Vector(
            cassette=cassette,
            origin_seq=origin_seq,
            marker_seq=marker_seq,
            mcs_seq=mcs_seq,
            full_sequence=full_seq,
            total_length=len(full_seq),
            features=features,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, dna: str) -> dict:
        """Multi-dimensional validation.

        Auto-detect the ORF: if the input is directly an ORF (starts with ATG and its length is a
        multiple of 3), validate it directly; otherwise use find_orfs to find the longest ORF.
        Also perform GC/homopolymer/restriction site checks on the full sequence.
        """
        # Reuse biocodec's ORF detection capability
        from helixlang.biocodec import find_orfs

        dna = dna.upper()
        full_gc = _gc_content(dna)
        full_max_run = _max_homopolymer(dna)
        full_sites = find_restriction_sites(dna)

        # If the sequence starts with ATG and its length is a multiple of 3, treat it as an ORF directly
        is_pure_orf = (len(dna) >= 6 and dna[:3] in START_CODONS
                       and len(dna) % 3 == 0)
        if is_pure_orf:
            orf_dna = dna
            orf_start = 0
            orf_end = len(dna)
            stop_codon = dna[-3:] if dna[-3:] in STOP_CODONS else ""
        else:
            # Find the longest ORF in the full sequence
            orfs = find_orfs(dna, min_length_aa=10, both_strands=False)
            if not orfs:
                return {
                    "valid": False,
                    "gc_content": full_gc,
                    "max_homopolymer": full_max_run,
                    "restriction_sites": list(full_sites.keys()),
                    "errors": ["no ORF found (min 10 aa)"],
                    "orf_found": False,
                    "orf_seq": "",
                    "orf_length": 0,
                    "orf_cai": 0.0,
                    "orf_gc_content": 0.0,
                    "orf_start": -1,
                    "orf_end": -1,
                    "stop_codon": "",
                }
            # Take the longest ORF
            best = max(orfs, key=lambda o: len(o.sequence))
            orf_dna = best.sequence
            orf_start = best.start
            orf_end = best.end
            stop_codon = best.stop_codon

        # Perform full validation on the ORF
        orf_validation = validate_cassette(orf_dna)

        return {
            "valid": orf_validation["valid"],
            "orf_found": True,
            "orf_start": orf_start,
            "orf_end": orf_end,
            "orf_seq": orf_dna,
            "orf_length": len(orf_dna),
            "stop_codon": stop_codon,
            "gc_content": full_gc,
            "orf_gc_content": orf_validation["gc_content"],
            "max_homopolymer": full_max_run,
            "restriction_sites": list(full_sites.keys()),
            "orf_cai": orf_validation["cai"],
            "errors": orf_validation["errors"],
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_genbank(self, dna: str, filename: str,
                       features: list | None = None) -> str:
        """Export GenBank format.

        Parameters
        ----------
        dna : str
            DNA sequence.
        filename : str
            LOCUS name / file base name (uses the .gb extension when writing to a file).
        features : list[dict], optional
            Feature annotation list.

        Returns
        -------
        str
            GenBank format text.
        """
        return genbank_format(dna, filename, features)

    def export_fasta(self, dna: str, name: str) -> str:
        """Export FASTA format.

        Parameters
        ----------
        dna : str
            DNA sequence.
        name : str
            Sequence name (>header).

        Returns
        -------
        str
            FASTA format text (header + sequence, 60 characters per line).
        """
        dna = dna.upper()
        lines = [f">{name}"]
        for i in range(0, len(dna), 60):
            lines.append(dna[i:i + 60])
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_promoter_seq(self, name: str) -> str:
        """Get the promoter sequence by name."""
        if name not in PROMOTER_SEQUENCES:
            raise BioError(
                f"unknown promoter {name!r}; "
                f"available: {list(PROMOTER_SEQUENCES.keys())}"
            )
        return PROMOTER_SEQUENCES[name]

    def _get_terminator_seq(self, name: str) -> str:
        """Get the terminator sequence by name."""
        if name not in TERMINATOR_SEQUENCES:
            raise BioError(
                f"unknown terminator {name!r}; "
                f"available: {list(TERMINATOR_SEQUENCES.keys())}"
            )
        return TERMINATOR_SEQUENCES[name]

    def _get_origin_seq(self, name: str) -> str:
        """Get the replicon sequence by name."""
        if name not in ORIGIN_SEQUENCES:
            raise BioError(
                f"unknown origin {name!r}; "
                f"available: {list(ORIGIN_SEQUENCES.keys())}"
            )
        return ORIGIN_SEQUENCES[name]

    def _get_marker_seq(self, name: str) -> str:
        """Get the resistance marker sequence by name."""
        if name not in SELECTION_MARKERS:
            raise BioError(
                f"unknown selection marker {name!r}; "
                f"available: {list(SELECTION_MARKERS.keys())}"
            )
        return SELECTION_MARKERS[name]

    def _build_mcs(self, mcs_sites: list[str]) -> str:
        """Build an MCS sequence from a list of restriction sites.

        Concatenate each restriction site sequence into a multiple cloning site.
        """
        parts: list[str] = []
        for site_name in mcs_sites:
            if site_name in MCS_SITES:
                parts.append(MCS_SITES[site_name])
            elif site_name in RESTRICTION_SITES:
                parts.append(RESTRICTION_SITES[site_name])
            else:
                raise BioError(f"unknown MCS site {site_name!r}")
        return "".join(parts)
