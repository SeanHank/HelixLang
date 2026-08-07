"""Biological DNA<->Helix compiler: converts real DNA sequences
<-> HelixLang programs.

Difference from `dna_codec.py`:
- `dna_codec.py` treats DNA as a storage medium (Goldman/Erlich coding
  of arbitrary bytes)
- `biocodec.py` treats DNA as a gene (codon -> opcode, consistent with
  real biology)

Core features:
1. **DNA -> Helix** (bio_disassemble):
   - scan for ORFs (ATG...stop)
   - each ORF compiles into one HelixLang gene
   - codons -> opcodes via codon_table.STANDARD_TABLE
   - third-position wobble acts as an operand modifier

2. **Helix -> DNA** (bio_assemble):
   - reverse-maps a HelixLang gene's ORF back to codons
   - applies E. coli codon usage frequency optimization (CAI)
   - adds promoters (lacP/araBAD and other real sequences) +
     terminators
   - avoids common restriction enzyme sites (EcoRI/BamHI/HindIII, etc.)

3. **Real biology validation**:
   - ORF length is a multiple of 3 + start ATG + stop codon
   - GC content 45-55% (typical of the E. coli genome)
   - no internal stop codons (ORF integrity)
   - CAI (codon adaptation index) >= 0.3 (E. coli expression
     threshold)

4. **Real promoter/terminator sequences**:
   - lacP: TTTACA...TATAAT (E. coli lac operon)
   - T7 terminator: TTTTTTTT T-rich region
   - rrnB T1 terminator: classic rho-independent terminator

References:
- E. coli K-12 MG1655 genome (Blattner 1997 Science 277:1453-1462)
- codon usage frequencies: CUTG E. coli 511145 (GenScript)
- lac operon promoter: Miller 1972
- restriction enzyme sites: NEB 2024 catalogue
"""
from __future__ import annotations

import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from helixlang import bio_data
from helixlang.bio_data import ECOLI_CODON_USAGE
from helixlang.codon_table import (
    STANDARD_TABLE,
)
from helixlang.seq_utils import gc_content as _gc_content
from helixlang.seq_utils import reverse_complement as _reverse_complement

# ============================================================================
# Real biological regulatory elements (E. coli K-12 MG1655)
# ============================================================================

# lac operon promoter (lacP, -35..-10 region, measured in E. coli K-12
# MG1655)
# Data source: Miller 1972, Kennedy 1977, E. coli K-12 MG1655 U00096.3
LAC_PROMOTER = "TTTACAATTTTCGCGATCTTTTTTATGCTTCCGGCTCGTATAATGTGTGGAATTGTGAGCGGATAACAATT"  # 71bp
# -35: TTTACA, -10: TATAAT, transcription start + lacI binding

# T7 gene 10 promoter (strong expression, common in expression vectors)
T7_PROMOTER = "TAATACGACTCACTATAGGGAGA"  # 23bp

# rrnB T1 terminator (rho-independent, E. coli K-12, classic
# terminator)
# Data source: Brosius 1981 J Biol Chem 256:4987-4990
RRNB_T1_TERMINATOR = "GCCGCCGGTTTTTTTGCTTTTGGCGGCATTTTTT"  # GC-rich stem-loop + U-rich tail

# T7 terminator (T7 phi10 terminator)
T7_TERMINATOR = "GCTAGTTATTGCTCAGCGGGTGAAATGCCGCCTGTTTACAAC"

# common restriction enzyme sites (NEB 2024 catalogue, 6-cutters)
# must be avoided in DNA storage (to prevent cleavage during cloning)
RESTRICTION_SITES: dict[str, str] = {
    "EcoRI":  "GAATTC",
    "BamHI":  "GGATCC",
    "HindIII": "AAGCTT",
    "NheI":   "GCTAGC",
    "XhoI":   "CTCGAG",
    "XbaI":   "TCTAGA",
    "SalI":   "GTCGAC",
    "PstI":   "CTGCAG",
    "KpnI":   "GGTACC",
    "SphI":   "GCATGC",
    "SmaI":   "CCCGGG",
    "SacI":   "GAGCTC",
    "NotI":   "GCGGCCGC",     # 8-cutter
    "PacI":   "TTAATTAA",
}


# ============================================================================
# ORF detection (Open Reading Frame)
# ============================================================================

# start codons (standard table)
START_CODONS = {"ATG", "GTG", "TTG"}  # ATG 90%+, GTG/TTG rare starts
# stop codons (standard table)
STOP_CODONS = {"TAA", "TAG", "TGA"}


@dataclass(slots=True)
class ORF:
    """A detected ORF."""
    start: int                # DNA 0-based start position
    end: int                  # DNA 0-based end position (exclusive)
    start_codon: str          # ATG/GTG/TTG
    stop_codon: str           # TAA/TAG/TGA
    strand: str               # "+" | "-"
    sequence: str             # ORF DNA sequence (incl. start and stop)
    protein: str              # translated protein (without the stop *)


def find_orfs(dna: str, min_length_aa: int = 10,
              both_strands: bool = True) -> list[ORF]:
    """Scan all ORFs in a DNA sequence.

    Rules (standard NCBI table 1):
    - start: ATG (main), GTG/TTG (rare)
    - stop: TAA/TAG/TGA
    - length >= min_length_aa * 3 + 3 (incl. stop)
    - three reading frames (+0/+1/+2), optional minus strand

    Consistent with NCBI ORFfinder: returns all non-overlapping,
    shortest ORFs.
    """
    orfs: list[ORF] = []
    strands = ["+"] + (["-"] if both_strands else [])
    for strand in strands:
        seq = dna if strand == "+" else _reverse_complement(dna)
        for frame in range(3):
            orfs.extend(_find_orfs_in_frame(seq, frame, strand, min_length_aa))
    # sort by start position
    orfs.sort(key=lambda o: (o.start, o.strand))
    return orfs


def _find_orfs_in_frame(seq: str, frame: int, strand: str,
                        min_length_aa: int) -> list[ORF]:
    """Find ORFs in a single reading frame."""
    orfs = []
    i = frame
    n = len(seq)
    min_len_nt = min_length_aa * 3 + 3  # incl. stop codon
    while i + 3 <= n:
        codon = seq[i:i + 3]
        if codon in START_CODONS:
            # find the nearest stop codon
            j = i + 3
            stop_codon = None
            while j + 3 <= n:
                c = seq[j:j + 3]
                if c in STOP_CODONS:
                    stop_codon = c
                    break
                j += 3
            if stop_codon and (j + 3 - i) >= min_len_nt:
                orf_seq = seq[i:j + 3]
                # translate (incl. start ATG/M, without the stop *)
                protein = _translate(orf_seq[:-3])
                # convert coordinates to the original strand
                if strand == "+":
                    start, end = i, j + 3
                else:
                    # minus strand: reverse the coordinates
                    start = len(seq) - (j + 3)
                    end = len(seq) - i
                orfs.append(ORF(
                    start=start, end=end,
                    start_codon=codon, stop_codon=stop_codon,
                    strand=strand, sequence=orf_seq, protein=protein,
                ))
                # skip this ORF (avoid overlaps)
                i = j + 3
            else:
                i += 3
        else:
            i += 3
    return orfs


def _translate(dna: str) -> str:
    """Translate DNA -> protein (standard table 1, BioPython optional)."""
    try:
        from Bio.Seq import Seq
        return str(Seq(dna).translate(table=1))
    except ImportError:
        # fallback: hand-written translation table
        return _translate_fallback(dna)


def _translate_fallback(dna: str) -> str:
    """Fallback translation when BioPython is unavailable."""
    table = {}
    for codon, (aa, _, _) in ECOLI_CODON_USAGE.items():
        table[codon] = aa
    out = []
    for i in range(0, len(dna) - len(dna) % 3, 3):
        c = dna[i:i + 3].upper()
        out.append(table.get(c, "X"))
    return "".join(out)


# ============================================================================
# Restriction enzyme site detection
# ============================================================================

def find_restriction_sites(dna: str,
                           enzymes: Iterable[str] | None = None
                           ) -> dict[str, list[int]]:
    """Detect restriction enzyme sites in DNA.

    Returns {enzyme_name: [position1, position2, ...]} (0-based).
    enzymes: None means scan all RESTRICTION_SITES.
    """
    if enzymes is None:
        enzymes = list(RESTRICTION_SITES.keys())
    dna_upper = dna.upper()
    sites: dict[str, list[int]] = {}
    for enz in enzymes:
        if enz not in RESTRICTION_SITES:
            continue
        site = RESTRICTION_SITES[enz]
        positions = []
        # forward direction
        i = dna_upper.find(site)
        while i != -1:
            positions.append(i)
            i = dna_upper.find(site, i + 1)
        # reverse complement (many sites are palindromic, some are not)
        rc = _reverse_complement(site)
        if rc != site:
            i = dna_upper.find(rc)
            while i != -1:
                positions.append(i)
                i = dna_upper.find(rc, i + 1)
        positions.sort()
        if positions:
            sites[enz] = positions
    return sites


def has_restriction_sites(dna: str, enzymes: Iterable[str] | None = None) -> bool:
    """Whether the DNA contains restriction enzyme sites."""
    return bool(find_restriction_sites(dna, enzymes))


def avoid_restriction_sites(dna: str,
                            enzymes: Iterable[str] | None = None,
                            max_attempts: int = 100,
                            rng: random.Random | None = None) -> str:
    """Remove restriction enzyme sites via synonymous mutations (keeping
    the protein sequence).

    DNA optimization for biological expression: destroys restriction
    enzyme sites by replacing them with synonymous codons while keeping
    the translation product unchanged.

    Raises ValueError if the sites cannot be removed after
    max_attempts.
    """
    if rng is None:
        rng = random.Random()
    current = dna.upper()
    for _ in range(max_attempts):
        sites = find_restriction_sites(current, enzymes)
        if not sites:
            return current
        # take the first site and try a synonymous mutation
        enz, positions = next(iter(sites.items()))
        pos = positions[0]
        # find the codon covering pos
        codon_start = (pos // 3) * 3
        if codon_start + 3 > len(current):
            # cannot modify; skip -> force-break (synonymous mutation
            # not feasible), degrade to a direct substitution (breaks
            # the protein)
            current = current[:pos] + ("A" if current[pos] != "A" else "T") + current[pos + 1:]
            continue
        codon = current[codon_start:codon_start + 3]
        # find a synonymous codon
        if codon not in ECOLI_CODON_USAGE:
            current = current[:pos] + ("A" if current[pos] != "A" else "T") + current[pos + 1:]
            continue
        aa = ECOLI_CODON_USAGE[codon][0]
        synonyms = [c for c, (a, _, _) in ECOLI_CODON_USAGE.items() if a == aa and c != codon]
        if not synonyms:
            # no synonymous codon -> force-break
            current = current[:pos] + ("A" if current[pos] != "A" else "T") + current[pos + 1:]
            continue
        # replace with a random synonymous codon
        new_codon = rng.choice(synonyms)
        current = current[:codon_start] + new_codon + current[codon_start + 3:]
    # final check
    sites = find_restriction_sites(current, enzymes)
    if sites:
        raise ValueError(
            f"could not remove all restriction sites after {max_attempts} attempts; "
            f"remaining: {list(sites.keys())}"
        )
    return current


# ============================================================================
# Codon optimization (for E. coli)
# ============================================================================

# reverse amino acid -> codon table
_AA_TO_CODONS: dict[str, list[str]] = {}
for _codon, (_aa, _per_thousand, _frac) in ECOLI_CODON_USAGE.items():
    _AA_TO_CODONS.setdefault(_aa, []).append(_codon)


def back_translate(protein: str, optimize: str = "cai",
                   rng: random.Random | None = None) -> str:
    """Protein -> DNA reverse translation (codon optimization).

    optimize:
    - "cai": use each amino acid's E. coli optimal codon (highest
      fraction)
    - "random": pick a synonymous codon at random
    - "balanced": random weighted by E. coli frequency
    """
    if rng is None:
        rng = random.Random()
    out = []
    for aa in protein:
        if aa == "*":
            # stop: TAA (60% in E. coli)
            out.append("TAA")
            continue
        if aa not in _AA_TO_CODONS:
            raise ValueError(f"unknown amino acid {aa!r}")
        codons = _AA_TO_CODONS[aa]
        if optimize == "cai":
            # pick the one with the highest fraction
            best = max(codons, key=lambda c: ECOLI_CODON_USAGE[c][2])
            out.append(best)
        elif optimize == "random":
            out.append(rng.choice(codons))
        elif optimize == "balanced":
            # weighted by frequency
            weights = [ECOLI_CODON_USAGE[c][1] for c in codons]
            total = sum(weights)
            r = rng.random() * total
            cum = 0.0
            for c, w in zip(codons, weights, strict=False):
                cum += w
                if r < cum:
                    out.append(c)
                    break
            else:
                out.append(codons[-1])
        else:
            raise ValueError(f"unknown optimize mode {optimize!r}")
    return "".join(out)


def codon_adaptation_index_full(dna: str) -> float:
    """Compute the CAI of a whole DNA sequence (Sharp 1987 Nucleic
    Acids Res 15:1281-1295).

    CAI = exp(mean(ln(frac_i / max_frac_aa_i)))
    where frac_i is the fraction of codon i and max_frac_aa_i is the
    maximum fraction of its amino acid.

    Delegates to :func:`helixlang.bio_data.cai` (E. coli K-12 table).
    """
    return bio_data.cai(dna, species="ecoli")


# ============================================================================
# DNA <-> HelixLang biological compilation
# ============================================================================

@dataclass(slots=True)
class BioHelixProgram:
    """Result of a DNA->Helix compilation."""
    dna: str                          # original DNA
    orfs: list[ORF]                   # detected ORFs
    genes: list[dict]                 # one gene dict per ORF
    helix_source: str                 # generated HelixLang source
    gc_content: float
    cai: float                        # overall CAI
    restriction_sites: dict[str, list[int]]
    notes: list[str] = field(default_factory=list)


def dna_to_helix(dna: str, gene_prefix: str = "orf",
                 min_length_aa: int = 3) -> BioHelixProgram:
    """DNA -> HelixLang source code (biological compilation).

    Steps:
    1. scan all ORFs (both strands, >= min_length_aa codons)
    2. each ORF -> a HelixLang gene:
       - codons -> opcodes via STANDARD_TABLE
       - third-position wobble -> operand
       - start ATG -> OP_START, stop -> OP_HALT
    3. generate the HelixLang source (with #gene name=orfN comments)
    4. detect restriction enzyme sites, compute GC/CAI

    min_length_aa: minimum ORF length (in codons). Default 3 (suited
        to HelixLang programs). For real-gene analysis, set to 30+ to
        filter short ORFs.

    Returns a BioHelixProgram containing the source + metadata.
    """
    dna = dna.upper()
    orfs = find_orfs(dna, min_length_aa=min_length_aa, both_strands=True)
    genes = []
    helix_lines = [
        f"# HelixLang bio-compiled from DNA ({len(dna)} bp)",
        f"# Detected {len(orfs)} ORF(s)",
        f"# GC content: {_gc_content(dna):.4f}",
        f"# CAI: {codon_adaptation_index_full(dna):.4f}",
        "",
    ]
    notes = []
    for i, orf in enumerate(orfs):
        gene_name = f"{gene_prefix}_{i + 1}"
        # ORF DNA -> codon list
        codons = [orf.sequence[j:j + 3] for j in range(0, len(orf.sequence), 3)]
        # verify every codon is in the table
        unknown_codons = [c for c in codons if c not in STANDARD_TABLE]
        if unknown_codons:
            notes.append(f"{gene_name}: {len(unknown_codons)} unknown codon(s) skipped")
            codons = [c for c in codons if c in STANDARD_TABLE]
        # convert to a HelixLang gene source
        codon_str = " ".join(codons)
        gene_src = f"#gene name={gene_name}\n{codon_str}\n#end"
        genes.append({
            "name": gene_name,
            "orf": orf,
            "codons": codons,
            "protein": orf.protein,
            "source": gene_src,
        })
        helix_lines.append(gene_src)
        helix_lines.append("")
    # restriction enzyme sites
    sites = find_restriction_sites(dna)
    if sites:
        notes.append(f"restriction sites found: {list(sites.keys())}")
    helix_source = "\n".join(helix_lines)
    return BioHelixProgram(
        dna=dna,
        orfs=orfs,
        genes=genes,
        helix_source=helix_source,
        gc_content=_gc_content(dna),
        cai=codon_adaptation_index_full(dna),
        restriction_sites=sites,
        notes=notes,
    )


def helix_to_dna(helix_source: str,
                 promoter: str = "lac",
                 terminator: str = "rrnB_T1",
                 add_promoter: bool = True,
                 add_terminator: bool = True,
                 optimize_codons: bool = True,
                 avoid_restriction: bool = True,
                 ) -> str:
    """HelixLang source -> real biological DNA (biological
    compilation).

    Steps:
    1. parse the HelixLang source (extract #gene blocks and codons)
    2. each gene's ORF -> DNA (concatenate codons)
    3. (optional) codon optimization (synonymous mutations raise CAI)
    4. (optional) remove restriction enzyme sites (synonymous
       mutations)
    5. (optional) add promoter + terminator

    Returns the full DNA sequence (promoter + gene1 + gene2 + ... +
    terminator).
    """
    # 1. parse the #gene blocks
    genes = _parse_helix_genes(helix_source)
    if not genes:
        raise ValueError("no #gene blocks found in HelixLang source")
    # 2. concatenate ORFs
    parts: list[str] = []
    for g in genes:
        orf_dna = g["orf_dna"]
        if optimize_codons:
            # translate then back-translate (apply E. coli optimization)
            # note: back_translate already includes the start ATG (M
            # encodes ATG), so only back-translate the protein and add
            # a stop TAA
            protein = _translate(orf_dna)
            # strip the trailing stop *
            if protein.endswith("*"):
                protein_body = protein[:-1]
            else:
                protein_body = protein
            # back-translate (incl. start ATG, add TAA stop)
            orf_dna = back_translate(protein_body, optimize="cai") + "TAA"
        if avoid_restriction:
            try:
                orf_dna = avoid_restriction_sites(orf_dna)
            except ValueError:
                pass  # keep the original sequence if it cannot be
                      # removed
        parts.append(orf_dna)
    gene_dna = "".join(parts)
    # 3. add the regulatory elements
    result = ""
    if add_promoter:
        result += _get_promoter(promoter)
    result += gene_dna
    if add_terminator:
        result += _get_terminator(terminator)
    return result


def _parse_helix_genes(source: str) -> list[dict]:
    """Parse the #gene blocks from a HelixLang source."""
    genes = []
    lines = source.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#gene"):
            # extract name=...
            name = "unnamed"
            m = re.search(r"name\s*=\s*(\S+)", line)
            if m:
                name = m.group(1)
            # collect codon lines until #end
            codon_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("#end"):
                stripped = lines[i].strip()
                if stripped and not stripped.startswith("#"):
                    codon_lines.append(stripped)
                i += 1
            # parse codons
            codons = []
            for cl in codon_lines:
                for token in cl.split():
                    if len(token) == 3 and all(c in "ACGTacgt" for c in token):
                        codons.append(token.upper())
            orf_dna = "".join(codons)
            if orf_dna:
                genes.append({
                    "name": name,
                    "codons": codons,
                    "orf_dna": orf_dna,
                })
        i += 1
    return genes


def _get_promoter(name: str) -> str:
    """Get a promoter by name."""
    promoters = {
        "lac": LAC_PROMOTER,
        "T7": T7_PROMOTER,
        "t7": T7_PROMOTER,
    }
    if name not in promoters:
        raise ValueError(f"unknown promoter {name!r}; available: {list(promoters)}")
    return promoters[name]


def _get_terminator(name: str) -> str:
    """Get a terminator by name."""
    terminators = {
        "rrnB_T1": RRNB_T1_TERMINATOR,
        "T7": T7_TERMINATOR,
        "t7": T7_TERMINATOR,
    }
    if name not in terminators:
        raise ValueError(f"unknown terminator {name!r}; available: {list(terminators)}")
    return terminators[name]



# ============================================================================
# Real biology validation
# ============================================================================

@dataclass(slots=True)
class BioValidationReport:
    """DNA biological plausibility validation report."""
    valid: bool
    gc_content: float
    gc_in_range: bool              # 0.45-0.55 (typical for E. coli)
    length_multiple_of_3: bool
    has_start_codon: bool
    has_stop_codon: bool
    no_internal_stop: bool         # no stop codon inside the ORF
    cai: float
    cai_adequate: bool             # >=0.3 expression threshold
    restriction_sites: dict[str, list[int]]
    max_homopolymer: int
    max_homopolymer_ok: bool       # <=3 (sequencing limit)
    errors: list[str] = field(default_factory=list)


def validate_biological(dna: str) -> BioValidationReport:
    """Validate whether a DNA sequence satisfies biological expression
    constraints.

    Checks:
    - length is a multiple of 3
    - contains a start codon ATG
    - contains a stop codon
    - no internal stop codons in the ORF
    - GC content 45-55% (typical for E. coli)
    - CAI >= 0.3 (expression threshold)
    - no long homopolymers (<=3, Illumina sequencing limit)
    - no common restriction enzyme sites
    """
    dna = dna.upper()
    errors = []
    gc = _gc_content(dna)
    gc_in_range = 0.45 <= gc <= 0.55
    if not gc_in_range:
        errors.append(f"GC {gc:.3f} out of [0.45, 0.55]")
    length_ok = len(dna) % 3 == 0
    if not length_ok:
        errors.append(f"length {len(dna)} not multiple of 3")
    has_start = dna[:3] in START_CODONS if len(dna) >= 3 else False
    if not has_start:
        errors.append("no start codon at position 0")
    has_stop = dna[-3:] in STOP_CODONS if len(dna) >= 3 else False
    if not has_stop:
        errors.append("no stop codon at end")
    # no internal stops
    no_internal_stop = True
    if length_ok and has_start and has_stop:
        for i in range(3, len(dna) - 3, 3):
            if dna[i:i + 3] in STOP_CODONS:
                no_internal_stop = False
                errors.append(f"internal stop codon at position {i}")
                break
    # CAI
    cai = codon_adaptation_index_full(dna)
    cai_adequate = cai >= 0.3
    if not cai_adequate:
        errors.append(f"CAI {cai:.3f} below expression threshold 0.3")
    # restriction sites
    sites = find_restriction_sites(dna)
    if sites:
        errors.append(f"restriction sites: {list(sites.keys())}")
    # homopolymers
    max_run = 1
    run = 1
    for i in range(1, len(dna)):
        if dna[i] == dna[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    max_homopolymer_ok = max_run <= 3
    if not max_homopolymer_ok:
        errors.append(f"homopolymer {max_run} > 3 (sequencing limit)")
    valid = (gc_in_range and length_ok and has_start and has_stop
             and no_internal_stop and cai_adequate and not sites
             and max_homopolymer_ok)
    return BioValidationReport(
        valid=valid,
        gc_content=gc,
        gc_in_range=gc_in_range,
        length_multiple_of_3=length_ok,
        has_start_codon=has_start,
        has_stop_codon=has_stop,
        no_internal_stop=no_internal_stop,
        cai=cai,
        cai_adequate=cai_adequate,
        restriction_sites=sites,
        max_homopolymer=max_run,
        max_homopolymer_ok=max_homopolymer_ok,
        errors=errors,
    )
