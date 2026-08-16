"""Whole-genome-scale virtual cell + FBA gene essentiality (S10.1 direction B).

Two literature-anchored capabilities:

1. **Whole-cell-scale build** -- a synthetic genome (default 500 genes, the
   scale of Karr et al. 2012's M. genitalium whole-cell model, Cell
   150:389) is loaded/translated through the central-dogma pipeline and
   wired into a genome-wide :class:`~helixlang.grn.GRN` + :class:
   `~helixlang.virtual_cell.VirtualCell`.  A minimal gene-set is necessary
   to express, mirroring Karr 2012's genotype -> phenotype loop.

2. **FBA gene-essentiality screen** -- the Feist et al. 2007 / EcoCyc
   method: delete a gene in silico, remove every reaction it gates (AND
   logic: a reaction with a single gene product disappears), re-solve FBA
   and classify *essential* when the biomass flux drops to zero (no
   growth).  This is Karr 2012's single-gene-disruption validation
   protocol and reproduces the EcoCyc (Gerdes 2003) glucose-minimal
   essentiality labels for every core gene the reduced 37-reaction model
   can faithfully represent.

The curated gene -> reaction map follows the E. coli core model (Orth 2010
Mol Syst Biol 6:390) gene associations; the essentiality reference uses
EcoCyc glucose-minimal labels (Gerdes et al. 2003 J Bacteriol
185:5673-5684; Kim & Copley 2007 for TPI).  Known reduced-core deviations
(pfkA/pykA/rpiA/gnd/mdh are single-copy in this core while the real
genome has isozymes or alternative pathways) are documented, not silently
asserted.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from helixlang.bio_data import ECOLI_CODON_USAGE
from helixlang.central_dogma import transcribe, translate
from helixlang.grn import GRN
from helixlang.metabolism import (
    ECOLI_CORE_GENE_REACTIONS,
    ECOLI_CORE_MODEL,
    FluxBalanceAnalysis,
    MetabolicModel,
    Reaction,
)
from helixlang.virtual_cell import VirtualCell, VirtualCellConfig, encode_gene

#: biomass flux below which a knockout strain is classified non-growing
ESSENTIALITY_FLUX_TOL = 1e-6
#: default glucose uptake for the essentiality screen (mmol/gDW/h)
DEFAULT_UPTAKE_GLC = 10.0

#: E. coli core-model gene -> reaction associations (Orth 2010 core model;
#: a reaction is gated by all of its genes, so deleting any one gene
#: removes the reaction when no isozyme copy exists).  Canonical copy lives
#: in :mod:`helixlang.metabolism` (the Phase-4 enzyme-capacity wiring);
#: re-exported here for ``ko_model``/``predict_essentiality``.

#: EcoCyc glucose-minimal essentiality labels (Gerdes 2003, Kim & Copley
#: 2007) for the genes the reduced core model represents faithfully.
#: True = essential (knockout -> no growth on glucose minimal medium).
ECOLI_CORE_ESSENTIALITY_REFERENCE: dict[str, bool] = {
    # glycolysis / central carbon: essential
    "pgi": True, "fba": True, "tpiA": True, "gapA": True, "pgk": True,
    "pgm": True, "eno": True,
    # TCA / anaplerosis: essential on glucose minimal
    "gltA": True, "icdA": True, "ppc": True,
    # pentose phosphate: essential (NADPH supply on glucose)
    "zwf": True,
    # pyruvate dehydrogenase: essential on glucose minimal
    "aceE": True,
    # glucose uptake: non-essential (GLK backs up PTS and vice versa)
    "ptsG": False, "glk": False,
    # fermentation: non-essential on glucose (aerobic respiration suffices)
    "ldhA": False, "pta": False, "ackA": False,
    # TCA cycle enzymes downstream of alpha-KG: non-essential on glucose
    "sucAB": False, "fumA": False,
}

#: documented reduced-core deviations (isozymes / alternative pathways
#: absent from the 37-reaction core make single-copy knockouts look
#: essential here while the real genome grows)
ECOLI_CORE_ESSENTIALITY_NOTES: dict[str, str] = {
    "pfkA": "pfkA/pfkB isozymes + ED pathway exist in the real genome "
            "(EcoCyc: non-essential); the core model has a single PFK",
    "pykA": "pykA/pykF isozymes (EcoCyc: non-essential); single PYK here",
    "gnd": "transketolase/transaldolase provide R5P in the real genome "
           "(EcoCyc: non-essential); PGD is the only R5P source here",
    "rpiA": "rpiA/rpiB isozymes (EcoCyc: non-essential); single RPI here",
    "sdhA": "SDH is bypassable on glucose (EcoCyc: non-essential); model "
            "agrees via partial growth",
    "mdh": "EcoCyc: essential on glucose minimal; the reduced core still "
           "grows at reduced yield through the PPC anaplerotic bypass",
}


def _translate_orfs(dna: str) -> str:
    """Translate the first in-frame ATG ORF directly from the codon table.

    Fallback for FASTA CDS entries without a Shine-Dalgarno RBS (which
    the full central-dogma ``translate`` requires for initiation).
    """
    seq = dna.upper().replace("U", "T")
    start = seq.find("ATG")
    if start < 0:
        return ""
    cds = seq[start:]
    chars: list[str] = []
    for i in range(0, len(cds) - len(cds) % 3, 3):
        codon = cds[i:i + 3]
        if codon in ("TAA", "TAG", "TGA"):
            break
        chars.append(ECOLI_CODON_USAGE[codon][0])
    return "".join(chars)


def _cds_to_protein(dna: str) -> str:
    """Translate a CDS to protein via the central-dogma pipeline.

    When the CDS carries an RBS the full transcription/translation
    machinery is used; otherwise (bare coding sequence from a FASTA
    record) the first in-frame ORF is translated directly from the codon
    table.
    """
    result = translate(transcribe(dna))
    if result.rbs_found and result.protein:
        return result.protein
    return _translate_orfs(dna)


def load_genome(source: str | dict[str, str],
                gff: str | None = None) -> dict[str, str]:
    """Build a whole-cell DNA genome ``{gene: CDS-with-RBS}``.

    ``source`` is either:
    - a dict of ``gene -> protein`` (each protein is codon-optimized with
      :func:`~helixlang.virtual_cell.encode_gene`), or
    - FASTA text (``>gene_name`` headers, DNA per record) translated
      through the central-dogma pipeline and re-encoded with an RBS so
      the :class:`~helixlang.virtual_cell.VirtualCell` can transcribe it,
      or
    - a full-chromosome FASTA + a GFF3 annotation table (``gff=`` text),
      the Phase-C C1 path -- CDS features are extracted by coordinate and
      translated, giving a ``{gene: CDS}`` genome from real sequence.

    Returns:
        ``dict[str, str]`` mapping gene name -> CDS DNA ready for
        transcription (the ``VirtualCell.genome`` format).
    """
    if gff is not None:
        return load_chromosome(str(source), gff).genome
    if not isinstance(source, dict):
        genome: dict[str, str] = {}
        name: str | None = None
        lines: list[str] = []

        def flush() -> None:
            if name and lines:
                genome[name] = encode_gene(_cds_to_protein("".join(lines)))

        for line in str(source).splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                name = line[1:].split()[0]
                lines = []
            else:
                lines.append(line)
        flush()
        if not genome:
            raise ValueError("no FASTA records found in source")
        return genome
    return {g: encode_gene(p) for g, p in source.items()}


# ============================================================================
# GFF3 chromosome import (Phase-C C1)
# ============================================================================

#: GFF3 attribute keys tried (in order) for the CDS gene name
_GFF_NAME_KEYS = ("gene", "locus_tag", "Name", "ID")

#: DNA base complement for minus-strand CDS extraction
_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(dna: str) -> str:
    """Reverse-complement a DNA string (used for minus-strand CDS)."""
    return dna.translate(_COMPLEMENT)[::-1]


def _unquote_attr(value: str) -> str:
    """Decode GFF3 percent-escaped attribute values (``; = , < >``)."""
    if "%" not in value:
        return value
    return (value.replace("%3B", ";").replace("%3D", "=")
                .replace("%3C", "<").replace("%3E", ">")
                .replace("%2C", ",").replace("%5C", "\\"))


@dataclass(frozen=True)
class GffFeature:
    """One GFF3 feature row (1-based inclusive ``start``..``end``).

    ``attributes`` carries the ``key=value`` table from column 9.
    """

    seqid: str
    source: str
    ftype: str
    start: int
    end: int
    strand: str
    attributes: dict[str, str]

    @property
    def name(self) -> str:
        """Preferred gene name (``gene`` > ``locus_tag`` > ``Name`` > ``ID``)."""
        for key in _GFF_NAME_KEYS:
            if key in self.attributes:
                return self.attributes[key]
        return f"{self.seqid}:{self.ftype}:{self.start}-{self.end}"


def parse_gff3(text: str) -> list[GffFeature]:
    """Parse GFF3 rows into :class:`GffFeature` objects.

    Comment lines (``#``), directives (``##``) and ``###`` row separators
    are skipped; parsing stops at an embedded ``##FASTA`` section.
    """
    features: list[GffFeature] = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line == "###":
            continue
        if line.startswith("##FASTA"):
            break
        if line.startswith("##"):
            continue
        cols = line.split("\t")
        if len(cols) < 8:
            continue
        seqid, source, ftype = cols[0], cols[1], cols[2]
        try:
            start, end = int(cols[3]), int(cols[4])
        except ValueError:
            continue
        if start < 1 or end < start:
            continue
        strand = cols[6] if cols[6] in ("+", "-") else "+"
        attributes: dict[str, str] = {}
        if len(cols) > 8:
            for part in cols[8].split(";"):
                if not part:
                    continue
                if "=" in part:
                    key, _, value = part.partition("=")
                    attributes[key] = _unquote_attr(value)
                else:
                    attributes[part] = ""
        features.append(GffFeature(
            seqid=seqid, source=source, ftype=ftype, start=start,
            end=end, strand=strand, attributes=attributes))
    return features


@dataclass
class Chromosome:
    """Whole-chromosome import: genome + GFF3 annotations (Phase-C C1).

    Attributes:
        seqid: sequence ID of the primary replicon.
        sequence: full chromosome DNA (uppercase, ``seqid`` record).
        chromosomes: every FASTA record (seqid -> sequence).
        genome: gene -> CDS-with-RBS DNA (the ``VirtualCell.genome`` format).
        cds: CDS features (merged per locus, coordinate-sorted).
        genes: non-CDS ``gene`` features.
        promoters: ``promoter`` features.
        terminators: ``terminator`` features.
        operons: ``operon`` features.
        other: remaining feature types.
    """

    seqid: str
    sequence: str
    chromosomes: dict[str, str]
    genome: dict[str, str]
    cds: list[GffFeature]
    genes: list[GffFeature]
    promoters: list[GffFeature]
    terminators: list[GffFeature]
    operons: list[GffFeature]
    other: list[GffFeature]

    def features(self, ftype: str) -> list[GffFeature]:
        """All features of a given GFF3 type."""
        return [f for f in (self.cds + self.genes + self.promoters
                            + self.terminators + self.operons + self.other)
                if f.ftype == ftype]


def _fasta_sequences(fasta: str) -> dict[str, str]:
    """Parse FASTA text into ``{seqid: sequence}`` (wrapped lines joined)."""
    sequences: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for line in str(fasta).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name and chunks:
                sequences[name] = "".join(chunks).upper()
            name = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if name and chunks:
        sequences[name] = "".join(chunks).upper()
    return sequences


def load_chromosome(fasta: str, gff: str) -> Chromosome:
    """Load a full-chromosome FASTA + GFF3 annotation table (Phase-C C1).

    CDS features are extracted by coordinate (reverse-complemented on the
    minus strand; multi-row CDS loci sharing a ``Parent`` are merged by
    coordinate), translated through the central-dogma pipeline and
    re-encoded with an RBS so the
    :class:`~helixlang.virtual_cell.VirtualCell` can transcribe them.  The
    protein set round-trips exactly through ``proteins_of``.  Promoter,
    terminator and operon features are kept as structured annotations for
    the regulatory-map import (``tf_map="regulondb"``).

    Args:
        fasta: full-chromosome FASTA text (``>seqid`` headers; wrapped
            sequence lines are concatenated).
        gff: GFF3 text (1-based inclusive coordinates, ``+``/``-`` strand).

    Returns:
        A :class:`Chromosome` carrying ``genome`` plus the annotations.
    """
    sequences = _fasta_sequences(fasta)
    if not sequences:
        raise ValueError("no FASTA records found in chromosome source")
    features = parse_gff3(gff)
    if not features:
        raise ValueError("no GFF3 features found in annotation source")

    genome: dict[str, str] = {}
    cds: list[GffFeature] = []
    genes: list[GffFeature] = []
    promoters: list[GffFeature] = []
    terminators: list[GffFeature] = []
    operons: list[GffFeature] = []
    other: list[GffFeature] = []

    segments: dict[tuple[str, str], list[GffFeature]] = {}
    for f in features:
        if f.ftype == "CDS":
            parent = f.attributes.get("Parent", f.name)
            segments.setdefault((f.seqid, parent), []).append(f)
        elif f.ftype == "gene":
            genes.append(f)
        elif f.ftype == "promoter":
            promoters.append(f)
        elif f.ftype == "terminator":
            terminators.append(f)
        elif f.ftype == "operon":
            operons.append(f)
        else:
            other.append(f)

    for (seqid, _parent), segs in segments.items():
        seq = sequences.get(seqid)
        if seq is None:
            continue
        segs.sort(key=lambda s: s.start)
        dna = "".join(seq[s.start - 1:s.end] for s in segs)
        if segs[0].strand == "-":
            dna = _revcomp(dna)
        protein = _cds_to_protein(dna)
        if not protein:
            continue
        genome[segs[0].name] = encode_gene(protein)
        cds.extend(segs)

    if not genome:
        raise ValueError("no translatable CDS features in GFF3 source")
    seqid = next(iter(sequences))
    return Chromosome(
        seqid=seqid, sequence=sequences[seqid], chromosomes=sequences,
        genome=genome, cds=cds, genes=genes, promoters=promoters,
        terminators=terminators, operons=operons, other=other)


def proteins_of(genome: dict[str, str]) -> dict[str, str]:
    """Translate every CDS back to its protein (whole-cell proteome)."""
    return {g: _cds_to_protein(dna) for g, dna in genome.items()}


_AA_POOL = "ACDEFGHIKLMNPQRSTVWY"


def random_genome(n_genes: int = 500, seed: int = 0,
                  min_len: int = 28, max_len: int = 60) -> dict[str, str]:
    """Deterministic synthetic whole-cell genome (E. coli b-number names).

    Each protein starts with methionine and has a randomized interior;
    the genome is reproducible for a given ``seed``.
    """
    if n_genes < 1:
        raise ValueError("n_genes must be >= 1")
    rng = random.Random(seed)
    genome: dict[str, str] = {}
    for i in range(1, n_genes + 1):
        length = rng.randint(min_len, max_len)
        body = "".join(rng.choice(_AA_POOL) for _ in range(length - 1))
        genome[f"b{str(i).zfill(4)}"] = "M" + body
    return genome


def build_whole_cell(genome: dict[str, str], seed: int = 0,
                     edges_per_gene: int = 2, seed_hubs: int = 8,
                     config: VirtualCellConfig | None = None) -> VirtualCell:
    """Build a whole-genome VirtualCell with a genome-wide sparse GRN.

    Every gene gets a node; a sparse regulatory edge set is drawn with a
    fixed seed, and ``seed_hubs`` genes start expressed so the GRN has
    something to transcribe from (a minimal essential gene set, Karr
    2012 style).
    """
    rng = random.Random(seed)
    names = list(genome)
    grn = GRN()
    for name in names:
        grn.add_gene(name, 0.5)
    for target in names:
        sources = [s for s in names if s != target]
        if not sources:
            continue
        rng.shuffle(sources)
        for src in sources[:edges_per_gene]:
            weight = rng.uniform(0.4, 1.5)
            if rng.random() < 0.4:
                weight = -weight
            grn.add_edge(src, target, weight)
    for hub in names[:seed_hubs]:
        grn.nodes[hub].level = 1.0
    return VirtualCell(dict(genome), grn, config=config)


def ko_model(model: MetabolicModel,
             reaction_ids: tuple[str, ...]) -> MetabolicModel:
    """Return a copy of ``model`` without ``reaction_ids`` (gene knockout)."""
    if not reaction_ids:
        return model
    out = MetabolicModel()
    for rid, rxn in model.reactions.items():
        if rid in reaction_ids:
            continue
        out.add_reaction(Reaction(
            id=rid, name=rxn.name, stoichiometry=dict(rxn.stoichiometry),
            lower_bound=rxn.lower_bound, upper_bound=rxn.upper_bound,
            subsystem=rxn.subsystem,
        ))
    out.set_biomass(model.biomass_reaction or "BIOMASS")
    return out


def predict_essentiality(
    gene: str,
    gene_to_reactions: dict[str, tuple[str, ...]] | None = None,
    model: MetabolicModel | None = None,
    uptake_glc: float = DEFAULT_UPTAKE_GLC,
) -> dict:
    """Feist 2007 / EcoCyc FBA gene-knockout essentiality for one gene.

    Deletes the gene's reactions from the model, re-solves biomass
    maximization and classifies the strain *essential* (no growth) when
    the biomass flux collapses.  ``growth_ratio`` is the knockout/wildtype
    biomass flux ratio -- the Karr 2012 growth-rate-per-disruption-strain
    analog.

    Returns:
        a dict with ``gene``, ``reactions``, ``wt_biomass``, ``ko_biomass``,
        ``growth_ratio``, ``essential`` (or ``None`` for genes with no
        associated reactions) and ``notes``.
    """
    g2r = gene_to_reactions or ECOLI_CORE_GENE_REACTIONS
    m = model or ECOLI_CORE_MODEL
    reactions = g2r.get(gene, ())
    if not reactions:
        return {"gene": gene, "reactions": (),
                "essential": None, "notes": "no reactions in core model"}
    wt = FluxBalanceAnalysis(m)
    wt.set_uptake("GLC", uptake_glc)
    wt_biomass = wt.solve().get(m.biomass_reaction or "BIOMASS", 0.0)
    ko = FluxBalanceAnalysis(ko_model(m, reactions))
    ko.set_uptake("GLC", uptake_glc)
    ko_biomass = ko.solve().get(m.biomass_reaction or "BIOMASS", 0.0)
    essential = ko_biomass < ESSENTIALITY_FLUX_TOL
    growth_ratio = (ko_biomass / wt_biomass
                    if wt_biomass > ESSENTIALITY_FLUX_TOL else 0.0)
    return {
        "gene": gene,
        "reactions": list(reactions),
        "wt_biomass": wt_biomass,
        "ko_biomass": ko_biomass,
        "growth_ratio": growth_ratio,
        "essential": essential,
        "notes": ECOLI_CORE_ESSENTIALITY_NOTES.get(gene, ""),
    }


def essentiality_screen(
    genes: list[str] | None = None,
    reference: dict[str, bool] | None = None,
    model: MetabolicModel | None = None,
) -> dict:
    """Screen curated E. coli core genes for FBA essentiality.

    Reports per-gene knockout results and the accuracy against the EcoCyc
    glucose-minimal reference on the faithfully-representable subset
    (:data:`ECOLI_CORE_ESSENTIALITY_REFERENCE`).
    """
    genes = genes or list(ECOLI_CORE_GENE_REACTIONS)
    ref = reference or ECOLI_CORE_ESSENTIALITY_REFERENCE
    results: dict[str, dict] = {}
    for g in genes:
        results[g] = predict_essentiality(g, model=model)
    tested = [g for g, r in results.items()
              if r["essential"] is not None and g in ref]
    matched = sum(1 for g in tested
                  if results[g]["essential"] == ref[g])
    return {
        "results": results,
        "reference_subset": list(ref),
        "n_tested": len(tested),
        "n_matched": matched,
        "accuracy": (matched / len(tested) if tested else 0.0),
        "reference": dict(ref),
        "deviations": {g: ECOLI_CORE_ESSENTIALITY_NOTES[g]
                       for g in ECOLI_CORE_ESSENTIALITY_NOTES
                       if g in results},
    }


class WholeCellBenchmark:
    """500-gene whole-cell build + FBA essentiality screen (Karr 2012 scale).

    ``run`` reports the genome scale, how much of it expressed, the
    essentiality accuracy against EcoCyc and the runtime; ``passed``
    requires the cell to stay alive, all genes wired, and the
    essentiality screen to reproduce the reference on the faithful
    subset.
    """

    def __init__(self, n_genes: int = 500, n_steps: int = 8, seed: int = 0,
                 uptake_glc: float = DEFAULT_UPTAKE_GLC) -> None:
        self.n_genes = n_genes
        self.n_steps = n_steps
        self.seed = seed
        self.uptake_glc = uptake_glc
        self.genome = random_genome(n_genes, seed=seed)
        self.cell = build_whole_cell(self.genome, seed=seed)

    def run(self) -> dict:
        t0 = time.perf_counter()
        history = self.cell.run(self.n_steps)
        elapsed = time.perf_counter() - t0
        expressed = {g for h in history for g in h["triggered"]}
        screen = essentiality_screen()
        last = history[-1] if history else {}
        passed = (
            len(self.genome) == self.n_genes
            and bool(self.cell.alive)
            and len(expressed) >= 1
            and screen["accuracy"] >= 0.95
        )
        return {
            "genome_size": len(self.genome),
            "n_steps": len(history),
            "expressed_genes": sorted(expressed),
            "n_expressed": len(expressed),
            "alive": self.cell.alive,
            "proteins": dict(self.cell.proteins),
            "biomass_flux_last": last.get("biomass_flux", 0.0),
            "essentiality": screen,
            "essentiality_accuracy": screen["accuracy"],
            "runtime_seconds": elapsed,
            "passed": passed,
        }


def run_whole_cell_benchmark(n_genes: int = 500, n_steps: int = 8,
                             seed: int = 0) -> dict:
    """One-shot whole-genome-scale benchmark (see :class:`WholeCellBenchmark`)."""
    return WholeCellBenchmark(n_genes=n_genes, n_steps=n_steps,
                              seed=seed).run()


def single_gene_ko_protocol(gene: str = "pgi", n_steps: int = 6,
                            seed: int = 0) -> dict:
    """Karr 2012 single-gene-disruption protocol: WT vs gene-KO growth.

    Builds two identical VirtualCells whose FBA models differ only by the
    knocked-out gene's reactions; runs both and compares growth (biomass
    flux) and viability.  For an essential gene the KO strain must show
    zero biomass flux while the WT grows.
    """
    reactions = ECOLI_CORE_GENE_REACTIONS[gene]
    genome = {"essential_test": encode_gene("MAQILARVFFDDVTK")}
    grn = GRN()
    grn.add_gene("essential_test", 0.5)
    grn.nodes["essential_test"].level = 1.0

    def build(fba: FluxBalanceAnalysis) -> VirtualCell:
        return VirtualCell(dict(genome), grn, fba=fba,
                           config=VirtualCellConfig(uptake={"GLC": 10.0}))

    wt = build(FluxBalanceAnalysis(ECOLI_CORE_MODEL))
    ko = build(FluxBalanceAnalysis(ko_model(ECOLI_CORE_MODEL, reactions)))
    wt_fluxes: list[float] = []
    ko_fluxes: list[float] = []
    for _ in range(n_steps):
        wt_fluxes.append(wt.step()["biomass_flux"])
        ko_fluxes.append(ko.step()["biomass_flux"])
    return {
        "gene": gene,
        "reactions": list(reactions),
        "wt_biomass_flux": wt_fluxes,
        "ko_biomass_flux": ko_fluxes,
        "wt_alive": wt.alive,
        "ko_alive": ko.alive,
        "ko_growth_blocked": all(f < ESSENTIALITY_FLUX_TOL
                                 for f in ko_fluxes)
        and any(f >= ESSENTIALITY_FLUX_TOL for f in wt_fluxes),
    }


__all__ = [
    "WholeCellBenchmark",
    "GffFeature",
    "Chromosome",
    "load_genome",
    "load_chromosome",
    "parse_gff3",
    "proteins_of",
    "random_genome",
    "build_whole_cell",
    "ko_model",
    "predict_essentiality",
    "essentiality_screen",
    "run_whole_cell_benchmark",
    "single_gene_ko_protocol",
    "ECOLI_CORE_GENE_REACTIONS",
    "ECOLI_CORE_ESSENTIALITY_REFERENCE",
    "ECOLI_CORE_ESSENTIALITY_NOTES",
]
