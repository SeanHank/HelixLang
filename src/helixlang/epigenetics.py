"""Epigenetic modification model: DNA methylation + histone
modifications.

Based on real data:
- E. coli Dam methylation: GATC sites (Marinus 1973)
- E. coli Dcm methylation: CCWGG sites (Marinus 1984)
- Eukaryotic CpG methylation: mammalian CG sites (Bird 2002 Cell
  109:1-8)
- Histone modifications: H3K4me3 (active promoters), H3K27me3
  (Polycomb repression), H3K36me3 (transcription elongation),
  H3K9me3 (heterochromatin)
- DNA methylation represses gene expression (~70% expression reduction,
  Bird 2002)
- CpG islands: GC>55%, length>200bp, observed/expected CpG >0.65
  (Takai 2002)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# ============================================================================
# Heuristic coefficient registry
# ============================================================================
# The accessibility/expression weights below are dimensionless heuristic
# coefficients, NOT measured kinetic constants. They are registered as
# named constants so future calibration against ChIP-seq occupancy /
# expression data is a one-line edit. Suggested primary sources for a
# future quantitative retune:
#   - DNMT processivity & TET oxidation kinetics: Jeltsch 2001
#     Biochemistry 40:4186-4198; Ito 2011 Science 333:1300-1303
#   - H3K27me3 / H3K4me3 occupancy-to-expression mapping: Young 2011
#     Nat Rev Genet 12:59-69; ENCODE ChIP-seq (Consortium 2012)
#   - DNA methylation repression strength (~70% reduction in promoter):
#     Bird 2002 Cell 109:1-8; Watt 1988 Mol Cell Biol 8:2770-2776

#: methylation weight on chromatin accessibility (0.4 = heuristic)
METHYLATION_ACCESSIBILITY_WEIGHT = 0.4
#: promoter-methylation expression repression (0.7 = heuristic ~70%,
#: Bird 2002)
PROMOTER_METHYLATION_REPRESSION = 0.7
#: gene-body-methylation expression repression (0.2 = heuristic)
GENE_BODY_METHYLATION_REPRESSION = 0.2
#: neutral chromatin base accessibility
BASE_ACCESSIBILITY = 0.5

# ============================================================================
# Histone modification type data
# ============================================================================
# The scores are relative expression/accessibility weights (heuristic,
# uncited linear mappings, see HEURISTIC_COEFFICIENT registry above);
# the mark->location->effect assignments follow the literature
# (H3K4me3 active promoters, H3K27me3 Polycomb repression, H3K36me3
# elongation, H3K9me3 heterochromatin, H3K27ac enhancers).

HISTONE_MARK_TYPES: dict[str, dict] = {
    "H3K4me3":  {"effect": "activating",      "location": "promoter",   "score": +0.5},
    "H3K27me3": {"effect": "repressing",      "location": "promoter",   "score": -0.7},
    "H3K36me3": {"effect": "elongation",      "location": "gene_body",  "score": +0.3},
    "H3K9me3":  {"effect": "heterochromatin", "location": "any",        "score": -0.9},
    "H3K27ac":  {"effect": "activating",      "location": "enhancer",   "score": +0.6},
}


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(slots=True)
class MethylationState:
    """DNA methylation state."""
    positions: dict[int, float] = field(default_factory=dict)  # position -> methylation probability [0,1]
    methylase: str = "custom"  # "dam" | "dcm" | "cpg" | "custom"
    total_sites: int = 0
    methylated_sites: int = 0


@dataclass(slots=True)
class HistoneMark:
    """Histone modification."""
    position: int   # position on the chromatin
    mark: str       # "H3K4me3" | "H3K27me3" | "H3K36me3" | "H3K9me3" | "H3K27ac"
    level: float    # modification level 0-1


@dataclass(slots=True)
class ChromatinState:
    """Chromatin state."""
    methylation: MethylationState
    histone_marks: list[HistoneMark] = field(default_factory=list)
    chromatin_accessibility: dict[int, float] = field(default_factory=dict)  # position -> accessibility 0-1
    expression_modifier: dict[str, float] = field(default_factory=dict)      # gene -> expression modifier


# ============================================================================
# Site search
# ============================================================================

def find_dam_sites(dna: str) -> list[int]:
    """Search for Dam methylation sites (GATC).

    The E. coli Dam methyltransferase methylates the N6 position of
    adenine in GATC sequences (Marinus 1973 MGG 115:248-250).

    Returns all GATC start positions (0-based, overlaps allowed).
    """
    dna = dna.upper()
    sites: list[int] = []
    i = dna.find("GATC")
    while i != -1:
        sites.append(i)
        i = dna.find("GATC", i + 1)
    return sites


def find_dcm_sites(dna: str) -> list[int]:
    """Search for Dcm methylation sites (CCWGG, W=A/T).

    The E. coli Dcm methyltransferase methylates the C5 position of
    cytosine at the second C of CCAGG/CCTGG sequences (Marinus 1984).

    Returns all CCWGG start positions (0-based, sorted).
    """
    dna = dna.upper()
    sites: list[int] = []
    for motif in ("CCAGG", "CCTGG"):
        i = dna.find(motif)
        while i != -1:
            sites.append(i)
            i = dna.find(motif, i + 1)
    sites.sort()
    return sites


def find_cpg_sites(dna: str) -> list[int]:
    """Search for CpG sites (CG dinucleotides).

    Mammalian DNMT1/3A/3B methylate the 5 position of cytosine in CpG
    dinucleotides (Bird 2002 Cell 109:1-8).

    Returns all CG start positions (0-based).
    """
    dna = dna.upper()
    sites: list[int] = []
    i = dna.find("CG")
    while i != -1:
        sites.append(i)
        i = dna.find("CG", i + 1)
    return sites


def find_cpg_islands(dna: str, min_length: int = 200,
                     min_gc: float = 0.55, min_oe: float = 0.65) -> list[dict]:
    """Search for CpG islands (Takai 2002 criteria).

    Takai 2002 PNAS 99:3720-3725 criteria:
    - length > 200 bp
    - GC content > 55%
    - observed/expected CpG ratio > 0.65

    O/E = (CG count x length) / (C count x G count)

    Scans with a min_length-sized sliding window and merges adjacent
    passing windows.
    Returns [{start, end, length, gc_content, cpg_oe}].
    """
    dna = dna.upper()
    n = len(dna)
    if n < min_length:
        return []

    # sliding-window scan
    passing: list[tuple[int, int]] = []
    for i in range(n - min_length + 1):
        window = dna[i:i + min_length]
        if _gc_fraction(window) >= min_gc and _cpg_oe(window) >= min_oe:
            passing.append((i, i + min_length))

    if not passing:
        return []

    # merge adjacent/overlapping windows
    islands: list[dict] = []
    cur_start, cur_end = passing[0]
    for s, e in passing[1:]:
        if s <= cur_end:  # overlapping or adjacent
            cur_end = max(cur_end, e)
        else:
            window = dna[cur_start:cur_end]
            islands.append({
                "start": cur_start,
                "end": cur_end,
                "length": cur_end - cur_start,
                "gc_content": _gc_fraction(window),
                "cpg_oe": _cpg_oe(window),
            })
            cur_start, cur_end = s, e
    # the last island
    window = dna[cur_start:cur_end]
    islands.append({
        "start": cur_start,
        "end": cur_end,
        "length": cur_end - cur_start,
        "gc_content": _gc_fraction(window),
        "cpg_oe": _cpg_oe(window),
    })

    return islands


def _gc_fraction(seq: str) -> float:
    """Compute the GC content."""
    if not seq:
        return 0.0
    gc = sum(1 for c in seq if c in "GC")
    return gc / len(seq)


def _cpg_oe(seq: str) -> float:
    """Compute the CpG observed/expected ratio.

    O/E = (CG count x length) / (C count x G count)
    """
    if not seq:
        return 0.0
    c_count = sum(1 for c in seq if c == "C")
    g_count = sum(1 for c in seq if c == "G")
    if c_count == 0 or g_count == 0:
        return 0.0
    cg_count = seq.count("CG")
    expected = (c_count * g_count) / len(seq)
    if expected == 0:
        return 0.0
    return cg_count / expected


# ============================================================================
# DNA methylation model
# ============================================================================

def methylate_dna(dna: str, cell_type: str = "ecoli",
                  methylase: str = "dam",
                  rng: random.Random | None = None) -> MethylationState:
    """DNA methylation model.

    - E. coli Dam: GATC fully methylated (>95%, Marinus 1973)
    - E. coli Dcm: CCWGG fully methylated (Marinus 1984)
    - Eukaryotic CpG: CpG islands low-methylated (<20%),
      non-island CpGs highly methylated (>70%)
      (Bird 2002 Cell 109:1-8)

    Returns a MethylationState; positions is position -> methylation
    probability.
    """
    if rng is None:
        rng = random.Random()
    dna = dna.upper()

    if cell_type == "ecoli":
        if methylase == "dam":
            sites = find_dam_sites(dna)
            # >95% methylation (Marinus 1973)
            positions: dict[int, float] = {s: rng.uniform(0.96, 0.99) for s in sites}
            methylated = sum(1 for v in positions.values() if v > 0.5)
            return MethylationState(
                positions=positions,
                methylase="dam",
                total_sites=len(sites),
                methylated_sites=methylated,
            )
        elif methylase == "dcm":
            sites = find_dcm_sites(dna)
            positions = {s: rng.uniform(0.95, 0.99) for s in sites}
            methylated = sum(1 for v in positions.values() if v > 0.5)
            return MethylationState(
                positions=positions,
                methylase="dcm",
                total_sites=len(sites),
                methylated_sites=methylated,
            )
        else:
            raise ValueError(
                f"unknown ecoli methylase {methylase!r}; expected 'dam' or 'dcm'"
            )
    elif cell_type in ("eukaryote", "mammal", "human"):
        if methylase != "cpg":
            raise ValueError(
                f"unknown eukaryotic methylase {methylase!r}; expected 'cpg'"
            )
        sites = find_cpg_sites(dna)
        islands = find_cpg_islands(dna)
        island_ranges = [(isl["start"], isl["end"]) for isl in islands]
        positions = {}
        methylated = 0
        for s in sites:
            in_island = any(start <= s < end for start, end in island_ranges)
            if in_island:
                # CpG islands low-methylated <20% (Bird 2002)
                prob = rng.uniform(0.05, 0.20)
            else:
                # non-island CpGs highly methylated >70% (Bird 2002)
                prob = rng.uniform(0.70, 0.95)
            positions[s] = prob
            if prob > 0.5:
                methylated += 1
        return MethylationState(
            positions=positions,
            methylase="cpg",
            total_sites=len(sites),
            methylated_sites=methylated,
        )
    else:
        raise ValueError(
            f"unknown cell_type {cell_type!r}; expected 'ecoli' or 'eukaryote'"
        )


# ============================================================================
# Histone modifications
# ============================================================================

def add_histone_marks(dna: str, gene_positions: list[dict],
                      cell_type: str = "eukaryote") -> list[HistoneMark]:
    """Add histone modifications.

    gene_positions elements look like:
        {"name": "geneA", "start": 100, "end": 500, "promoter": (50, 100)}

    - Promoter regions: H3K4me3 (active) or H3K27me3 (repressed)
    - Gene bodies: H3K36me3 (transcription elongation, active genes
      only)
    - Heterochromatin (both ends of the DNA): H3K9me3

    Cell types:
    - "eukaryote"/"mammal"/"human": apply histone modifications
    - "ecoli": bacteria have no histones, returns an empty list
    """
    if cell_type == "ecoli":
        # bacteria have no histones
        return []

    marks: list[HistoneMark] = []
    n = len(dna)

    for i, gene in enumerate(gene_positions):
        start = gene.get("start", 0)
        end = gene.get("end", n)
        promoter = gene.get("promoter")
        if promoter is None:
            # default promoter is 50 bp upstream of the gene
            promoter = (max(0, start - 50), start)
        p_start, p_end = promoter
        p_mid = (p_start + p_end) // 2

        # alternate active/repressed (for demonstration; real scenarios
        # need transcriptome data)
        active = (i % 2 == 0)
        if active:
            # active gene: promoter H3K4me3 + gene body H3K36me3
            marks.append(HistoneMark(position=p_mid, mark="H3K4me3", level=0.9))
            body_pos = (start + end) // 2
            if start <= body_pos < end:
                marks.append(HistoneMark(
                    position=body_pos, mark="H3K36me3", level=0.7
                ))
        else:
            # repressed gene: promoter H3K27me3
            marks.append(HistoneMark(position=p_mid, mark="H3K27me3", level=0.9))

    # heterochromatin: both ends of the DNA (telomere-like regions)
    if n > 100:
        marks.append(HistoneMark(position=10, mark="H3K9me3", level=0.8))
        marks.append(HistoneMark(position=n - 10, mark="H3K9me3", level=0.8))

    return marks


# ============================================================================
# Chromatin accessibility
# ============================================================================

def calculate_accessibility(chromatin: ChromatinState) -> dict[int, float]:
    """Calculate chromatin accessibility.

    - Methylation decreases accessibility (DNA methylation -> chromatin
      compaction)
    - H3K4me3/H3K27ac increase accessibility (active chromatin)
    - H3K27me3/H3K9me3 decrease accessibility (repressed chromatin)

    Accessibility ranges over [0, 1], 1 = fully open, 0 = fully
    compacted.
    """
    accessibility: dict[int, float] = {}

    # collect all marked positions
    all_positions: set[int] = set(chromatin.methylation.positions.keys())
    for mark in chromatin.histone_marks:
        all_positions.add(mark.position)

    for pos in all_positions:
        # base accessibility 0.5
        acc = BASE_ACCESSIBILITY
        # methylation decreases accessibility
        if pos in chromatin.methylation.positions:
            meth_prob = chromatin.methylation.positions[pos]
            acc -= METHYLATION_ACCESSIBILITY_WEIGHT * meth_prob
        # histone modifications
        for mark in chromatin.histone_marks:
            if mark.position == pos:
                mark_info = HISTONE_MARK_TYPES.get(mark.mark)
                if mark_info:
                    acc += mark_info["score"] * mark.level
        # clamp to [0, 1]
        accessibility[pos] = max(0.0, min(1.0, acc))

    return accessibility


# ============================================================================
# Gene expression modifiers
# ============================================================================

def calculate_expression_modifier(chromatin: ChromatinState,
                                  gene_positions: list[dict]) -> dict[str, float]:
    """Calculate gene expression modifiers.

    - DNA methylation -> expression reduced ~70% (Bird 2002 Cell
      109:1-8)
    - H3K4me3 -> expression increased (active promoters)
    - H3K27me3 -> expression repressed (Polycomb)
    - H3K9me3 -> strongly repressed (heterochromatin)
    - H3K36me3 -> slightly increased (transcription elongation)

    Returns {gene_name: modifier}, ranging over [0, 2], 1 = normal
    expression.
    """
    modifiers: dict[str, float] = {}

    for i, gene in enumerate(gene_positions):
        name = gene.get("name", f"gene_{i}")
        start = gene.get("start", 0)
        end = gene.get("end", 0)
        promoter = gene.get("promoter", (max(0, start - 50), start))
        p_start, p_end = promoter

        # initial modifier 1.0 (normal expression)
        modifier = 1.0

        # methylation reduces expression
        meth_reduction = 0.0
        meth_count = 0
        for pos, prob in chromatin.methylation.positions.items():
            in_promoter = p_start <= pos < p_end
            in_gene = start <= pos < end
            if in_promoter:
                # promoter methylation -> strong repression ~70%
                # (Bird 2002)
                meth_reduction += PROMOTER_METHYLATION_REPRESSION * prob
                meth_count += 1
            elif in_gene:
                # gene body methylation -> mild repression
                meth_reduction += GENE_BODY_METHYLATION_REPRESSION * prob
                meth_count += 1
        if meth_count > 0:
            meth_reduction /= meth_count
        modifier -= meth_reduction

        # histone modifications
        for mark in chromatin.histone_marks:
            mark_info = HISTONE_MARK_TYPES.get(mark.mark)
            if not mark_info:
                continue
            in_promoter = p_start <= mark.position < p_end
            in_gene = start <= mark.position < end
            # only affect this gene's expression when the mark is in the
            # promoter or gene body
            if in_promoter or in_gene:
                modifier += mark_info["score"] * mark.level

        # clamp to [0, 2]
        modifiers[name] = max(0.0, min(2.0, modifier))

    return modifiers
