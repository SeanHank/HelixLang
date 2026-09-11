"""Targeted coverage-close tests for ``plugins.annotation`` edge branches.

Flush the final uncovered branches in ``sequences.py`` genome loading (a bare
sequence line before the first header and a trailing header with no sequence).
"""
from helixlang.plugins.annotation.sequences import extract_proteins_from_gff3


def test_gff3_with_bare_line_and_partial_header(tmp_path):
    genome = tmp_path / "genome.fa"
    genome.write_text(
        "ATG\n"                       # bare sequence line before first header
        ">contig1\n"
        "ATGAAAGAATTTTAA\n"
        ">partial\n"                  # header with no trailing sequence
    )
    gff = tmp_path / "ann.gff3"
    gff.write_text("contig1\t.\tCDS\t1\t15\t.\t+\t0\tID=gene1\n")

    proteins = extract_proteins_from_gff3(str(genome), str(gff))
    assert "gene1" in proteins
    ps = proteins["gene1"]
    assert ps.gene_id == "gene1"
    assert ps.contig == "contig1"
    assert ps.start == 1
    assert ps.end == 15


def test_gff3_genome_without_any_header(tmp_path):
    genome = tmp_path / "genome.fa"
    genome.write_text("ATGAAAGAATTTTAA\n")
    gff = tmp_path / "ann.gff3"
    gff.write_text("contig1\t.\tCDS\t1\t15\t.\t+\t0\tID=gene1\n")

    proteins = extract_proteins_from_gff3(str(genome), str(gff))
    assert proteins == {}
