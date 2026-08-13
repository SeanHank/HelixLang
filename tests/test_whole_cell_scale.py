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

from helixlang.apps.whole_cell_scale import (
    ECOLI_CORE_ESSENTIALITY_NOTES,
    ECOLI_CORE_ESSENTIALITY_REFERENCE,
    build_whole_cell,
    essentiality_screen,
    ko_model,
    load_genome,
    predict_essentiality,
    proteins_of,
    random_genome,
    run_whole_cell_benchmark,
    single_gene_ko_protocol,
)
from helixlang.metabolism import ECOLI_CORE_MODEL

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
