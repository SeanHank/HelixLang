"""Whole-genome-scale virtual cell + FBA essentiality tests (S10.1 direction B).

Verification goals:
- ``load_genome``: protein dicts and FASTA DNA records (with and without
  RBS) round-trip through the central-dogma pipeline.
- ``random_genome``: deterministic synthetic whole-cell genomes.
- FBA gene essentiality (Feist 2007 / EcoCyc method): each knockout
  prediction reproduces the EcoCyc glucose-minimal label on the
  faithfully-representable core subset (100% agreement), and documented
  reduced-core deviations are reported, not asserted away.
- The Karr 2012 single-gene-disruption protocol: deleting an essential
  gene blocks biomass production in a VirtualCell while WT grows.
- The 500-gene WholeCellBenchmark passes end-to-end.

References:
- Karr et al. 2012 Cell 150:389 (single-gene knockout -> growth)
- Feist et al. 2007 Mol Syst Biol 3:121 (FBA gene essentiality)
- Gerdes et al. 2003 J Bacteriol 185:5673 (E. coli glucose-minimal
  essentiality); Kim & Copley 2007 (TPI essentiality)
"""
from __future__ import annotations

import pytest

from helixlang.plugins.apps.whole_cell_scale import (
    ECOLI_CORE_ESSENTIALITY_NOTES,
    ECOLI_CORE_ESSENTIALITY_REFERENCE,
    Chromosome,
    GffFeature,
    _revcomp,
    build_whole_cell,
    essentiality_screen,
    ko_model,
    load_chromosome,
    load_genome,
    parse_gff3,
    predict_essentiality,
    proteins_of,
    random_genome,
    run_whole_cell_benchmark,
    single_gene_ko_protocol,
)
from helixlang.plugins.runtime.metabolism import ECOLI_CORE_MODEL
from helixlang.plugins.runtime.virtual_cell import encode_gene

PROTEIN = "MAQILARVFFDDV"


# ============================================================================
# Genome loading / translation
# ============================================================================

def test_load_genome_protein_dict_roundtrip() -> None:
    genome = load_genome({"lacZ": PROTEIN, "galK": "MSSRPQAAASSWW"})
    assert set(genome) == {"lacZ", "galK"}
    assert proteins_of(genome) == {"lacZ": PROTEIN, "galK": "MSSRPQAAASSWW"}


def test_load_genome_fasta_roundtrip() -> None:
    dna = load_genome({"lacZ": PROTEIN})["lacZ"]
    bare = dna[dna.find("ATG"):]  # CDS without the RBS (GenBank-style)
    fasta = f">geneA\n{bare}\n>geneB\n{dna}\n"
    genome = load_genome(fasta)
    assert set(genome) == {"geneA", "geneB"}
    assert proteins_of(genome)["geneB"] == PROTEIN
    assert proteins_of(genome)["geneA"] == PROTEIN  # bare-CDS fallback


def test_load_genome_empty_fasta_raises() -> None:
    with pytest.raises(ValueError):
        load_genome(">only_header\n")


# ============================================================================
# GFF3 chromosome import (Phase-C C1)
# ============================================================================

_CHROM_PROTEINS = {
    "gltA": "MAQILARVFFDDV",
    "zwf": "MSSRPQAAASSWW",
    "aceE": "MKVLIVTGDVLDAA",
}
_SPACER = "GATCTAGCTAGC"


def _chromosome_fixture() -> tuple[str, str, dict[str, str]]:
    """Synthetic 3-gene chromosome: FASTA text, GFF3 text, proteins.

    ``gltA``/``zwf`` sit on the plus strand, ``aceE`` on the minus
    strand; intergenic spacers carry a promoter, the trailing spacer a
    terminator, and an operon spans the whole locus.
    """
    genes = {g: encode_gene(p) for g, p in _CHROM_PROTEINS.items()}
    placements = [("gltA", "+"), ("zwf", "+"), ("aceE", "-")]
    chrom_parts: list[str] = []
    cds_rows: list[tuple[int, int, str, str]] = []
    offset = 0
    for i, (name, strand) in enumerate(placements):
        dna = genes[name] if strand == "+" else _revcomp(genes[name])
        chrom_parts.append(dna)
        cds_rows.append((offset + 1, offset + len(dna), strand, name))
        offset += len(dna)
        if i < len(placements) - 1:
            chrom_parts.append(_SPACER)
            offset += len(_SPACER)
    chrom_parts.append(_SPACER)
    terminator_start = offset + 1
    offset += len(_SPACER)
    chrom = "".join(chrom_parts)

    lines = ["##gff-version 3"]
    for start, end, strand, name in cds_rows:
        lines.append(
            f"NC_000913\t.\tgene\t{start}\t{end}\t.\t{strand}\t.\t"
            f"ID=gene:{name};gene={name}")
        lines.append(
            f"NC_000913\t.\tCDS\t{start}\t{end}\t.\t{strand}\t0\t"
            f"ID=cds:{name};gene={name}")
    p_start, p_end = cds_rows[0][1] + 1, cds_rows[0][1] + len(_SPACER)
    lines.append(
        f"NC_000913\t.\tpromoter\t{p_start}\t{p_end}\t.\t+\t.\t"
        f"ID=prom:gltA;gene=gltA")
    lines.append(
        f"NC_000913\t.\tterminator\t{terminator_start}\t"
        f"{terminator_start + len(_SPACER) - 1}\t.\t+\t.\t"
        f"ID=term:aceE;gene=aceE")
    lines.append(
        f"NC_000913\t.\toperon\t1\t{offset}\t.\t+\t.\tID=op:glycolysis")
    gff = "\n".join(lines) + "\n"
    fasta = ">NC_000913\n" + "\n".join(
        chrom[i:i + 60] for i in range(0, len(chrom), 60)) + "\n"
    return fasta, gff, _CHROM_PROTEINS


def test_load_chromosome_roundtrip() -> None:
    """C1 gate: full-chromosome FASTA + GFF3 round-trips the proteome."""
    fasta, gff, proteins = _chromosome_fixture()
    genes = {g: encode_gene(p) for g, p in proteins.items()}

    chrom = load_chromosome(fasta, gff)
    assert isinstance(chrom, Chromosome)
    assert chrom.seqid == "NC_000913"
    assert chrom.sequence == "".join(fasta.splitlines()[1:])  # wrapped FASTA
    assert set(chrom.genome) == set(genes)
    # both strands round-trip exactly through the central-dogma pipeline
    assert chrom.genome == genes
    assert proteins_of(chrom.genome) == proteins
    # structured annotations survive the import
    assert len(chrom.cds) == 3
    assert len(chrom.genes) == 3
    assert len(chrom.promoters) == 1
    assert len(chrom.terminators) == 1
    assert len(chrom.operons) == 1
    assert chrom.promoters[0].name == "gltA"
    assert chrom.operons[0].attributes["ID"] == "op:glycolysis"
    assert all(isinstance(f, GffFeature) for f in chrom.features("promoter"))


def test_load_genome_accepts_gff() -> None:
    """``load_genome(fasta, gff=gff)`` uses the chromosome path (C1)."""
    fasta, gff, proteins = _chromosome_fixture()
    genome = load_genome(fasta, gff=gff)
    assert proteins_of(genome) == proteins


def test_load_chromosome_multisegment_cds_merged() -> None:
    """Split CDS rows sharing a Parent merge into one translatable gene."""
    dna = encode_gene("MPKKPLTSYQW")
    mid = len(dna) // 2
    gff = (
        "##gff-version 3\n"
        f"NC_000913\t.\tCDS\t1\t{mid}\t.\t+\t0\t"
        f"ID=cds:fadD;gene=fadD;Parent=trans:fadD\n"
        f"NC_000913\t.\tCDS\t{mid + 1}\t{len(dna)}\t.\t+\t0\t"
        f"ID=cds:fadD2;gene=fadD;Parent=trans:fadD\n")
    chrom = load_chromosome(f">NC_000913\n{dna}\n", gff)
    assert proteins_of(chrom.genome) == {"fadD": "MPKKPLTSYQW"}


def test_parse_gff3_handles_directives_and_escaping() -> None:
    text = (
        "##gff-version 3\n"
        "##sequence-region NC_000913 1 1000\n"
        "NC_000913\tRefSeq\tCDS\t10\t30\t.\t-\t0\t"
        "ID=cds:y1;gene=y1;note=has%3Bsemicolon%3Dequal\n"
        "NC_000913\t.\tCDS\t5\t3\t.\t+\t0\tID=bad\n"
        "##FASTA\n>NC_000913\nACGTACGT\n")
    feats = parse_gff3(text)
    assert len(feats) == 1
    assert feats[0].strand == "-"
    assert feats[0].start == 10 and feats[0].end == 30
    assert feats[0].attributes["note"] == "has;semicolon=equal"


def test_load_chromosome_empty_gff_raises() -> None:
    with pytest.raises(ValueError, match="no GFF3 features"):
        load_chromosome(">NC_000913\nACGTACGT\n", "##gff-version 3\n")



def test_random_genome_is_deterministic() -> None:
    g1 = random_genome(50, seed=7)
    g2 = random_genome(50, seed=7)
    g3 = random_genome(50, seed=8)
    assert g1 == g2
    assert g1 != g3
    assert len(g1) == 50
    assert all(name.startswith("b") for name in g1)
    assert all(seq.startswith("M") for seq in g1.values())
    # every generated protein is translatable
    genome_dna = load_genome(g1)
    assert proteins_of(genome_dna) == g1


# ============================================================================
# FBA gene essentiality (Feist 2007 / EcoCyc method)
# ============================================================================

def test_essentiality_screen_reproduces_ecocyc_reference() -> None:
    screen = essentiality_screen()
    assert screen["accuracy"] == pytest.approx(1.0)
    assert screen["n_matched"] == len(ECOLI_CORE_ESSENTIALITY_REFERENCE)
    assert screen["n_tested"] == len(ECOLI_CORE_ESSENTIALITY_REFERENCE)


@pytest.mark.parametrize(
    ("gene", "expected"),
    [
        # essential on glucose minimal (EcoCyc / Gerdes03 / Kim&Copley)
        ("pgi", True), ("fba", True), ("tpiA", True), ("gapA", True),
        ("pgk", True), ("pgm", True), ("eno", True), ("gltA", True),
        ("icdA", True), ("ppc", True), ("zwf", True), ("aceE", True),
        # non-essential on glucose minimal
        ("ptsG", False), ("glk", False), ("ldhA", False), ("pta", False),
        ("ackA", False), ("sucAB", False), ("fumA", False),
    ],
)
def test_essentiality_of_faithful_core_genes(gene: str, expected: bool) -> None:
    result = predict_essentiality(gene)
    assert result["essential"] == expected, gene
    assert ECOLI_CORE_ESSENTIALITY_REFERENCE[gene] == expected


def test_reduced_core_deviations_are_documented() -> None:
    # pfkA/pykA/gnd/rpiA look essential in the reduced core (single copy,
    # no isozyme/ED pathway) while the real genome grows; mdh deviates the
    # other way (EcoCyc essential, reduced core grows via PPC bypass) --
    # all are reported with a note, not silently asserted
    for gene in ("pfkA", "pykA", "gnd", "rpiA"):
        result = predict_essentiality(gene)
        assert result["essential"] is True
        assert result["notes"], gene
        assert gene in ECOLI_CORE_ESSENTIALITY_NOTES
    assert predict_essentiality("mdh")["essential"] is False
    assert "mdh" in ECOLI_CORE_ESSENTIALITY_NOTES


def test_unknown_gene_has_no_prediction() -> None:
    result = predict_essentiality("not_a_core_gene")
    assert result["essential"] is None
    assert result["reactions"] == ()


def test_ko_model_removes_only_target_reactions() -> None:
    ko = ko_model(ECOLI_CORE_MODEL, ("PGI",))
    assert "PGI" not in ko.reactions
    assert "PFK" in ko.reactions
    assert ko.biomass_reaction == ECOLI_CORE_MODEL.biomass_reaction
    assert len(ko.reactions) == len(ECOLI_CORE_MODEL.reactions) - 1


# ============================================================================
# Karr 2012 single-gene-disruption protocol
# ============================================================================

def test_single_gene_ko_blocks_essential_gene_growth() -> None:
    result = single_gene_ko_protocol("pgi")
    assert result["ko_growth_blocked"]
    assert result["wt_alive"] and result["ko_alive"]
    assert max(result["wt_biomass_flux"]) > 0.0
    assert max(result["ko_biomass_flux"]) == 0.0


# ============================================================================
# Whole-cell benchmark (Karr 2012 scale)
# ============================================================================

def test_whole_cell_benchmark_passes() -> None:
    result = run_whole_cell_benchmark(n_genes=500, n_steps=8, seed=0)
    assert result["passed"]
    assert result["genome_size"] == 500
    assert result["n_expressed"] >= 1
    assert result["alive"]
    assert result["essentiality_accuracy"] == pytest.approx(1.0)
    assert result["runtime_seconds"] < 60.0


def test_build_whole_cell_wires_all_genes() -> None:
    genome = random_genome(120, seed=1)
    cell = build_whole_cell(genome, seed=1, seed_hubs=4)
    assert len(cell.genome) == 120
    assert len(cell.grn.nodes) == 120
    assert sum(1 for h in cell.grn.nodes.values() if h.level > 0.0) == 4
    cell.run(5)
    assert cell.alive


def test_whole_cell_benchmark_config_validation() -> None:
    with pytest.raises(ValueError):
        random_genome(0)
