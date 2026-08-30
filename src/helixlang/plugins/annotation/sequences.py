"""Protein sequence extraction from genome (doc/20 §13.3.3)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Standard codon table (NCBI translation table 11) — single source of truth
# lives in core.codon_table (doc/38 §4); this module only imports it.
from helixlang.api.language import STANDARD_AMINO_ACIDS


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(complement.get(c.upper(), "N") for c in reversed(seq))


def translate(seq: str) -> str:
    """Translate a nucleotide sequence to protein using standard codon table."""
    protein = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i + 3].upper()
        aa = STANDARD_AMINO_ACIDS.get(codon, "X")
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


@dataclass
class ProteinSequence:
    """A protein sequence extracted from a genome."""

    gene_id: str
    sequence: str
    start: int
    end: int
    strand: str
    contig: str


def extract_proteins_from_fasta(
    fasta_path: str | Path,
) -> dict[str, str]:
    """Extract protein sequences directly from a FASTA file.

    Assumes the FASTA contains protein sequences (not nucleotide).
    Used when no GFF3 is provided.

    Parameters
    ----------
    fasta_path : path to protein FASTA file

    Returns
    -------
    dict mapping gene_id → protein sequence
    """
    proteins: dict[str, str] = {}
    fasta_path = Path(fasta_path)

    if not fasta_path.exists():
        return proteins

    current_id = ""
    current_seq: list[str] = []

    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_seq:
                    proteins[current_id] = "".join(current_seq)
                header = line[1:]
                current_id = header.split()[0] if header else ""
                current_seq = []
            elif current_id:
                current_seq.append(line.upper())

    if current_id and current_seq:
        proteins[current_id] = "".join(current_seq)

    return proteins


def extract_proteins_from_gff3(
    genome_fasta: str | Path,
    gff3_path: str | Path,
) -> dict[str, ProteinSequence]:
    """Extract protein sequences from genome using GFF3 coordinates.

    Parameters
    ----------
    genome_fasta : path to genome nucleotide FASTA
    gff3_path : path to GFF3 annotation file

    Returns
    -------
    dict mapping gene_id → ProteinSequence
    """
    proteins: dict[str, ProteinSequence] = {}
    genome_path = Path(genome_fasta)
    gff_path = Path(gff3_path)

    if not genome_path.exists() or not gff_path.exists():
        return proteins

    # Load genome sequences into memory
    contigs: dict[str, str] = {}
    current_contig = ""
    current_seq: list[str] = []

    with open(genome_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current_contig and current_seq:
                    contigs[current_contig] = "".join(current_seq)
                current_contig = line[1:].split()[0]
                current_seq = []
            elif current_contig:
                current_seq.append(line.upper())
        if current_contig and current_seq:
            contigs[current_contig] = "".join(current_seq)

    # Parse GFF3 for CDS features
    with open(gff_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            contig = parts[0]
            feature_type = parts[2]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attributes = parts[8]

            if feature_type != "CDS":
                continue

            # Parse gene_id from attributes
            gene_id = ""
            for attr in attributes.split(";"):
                if attr.startswith("ID="):
                    gene_id = attr[3:]
                    break
            if not gene_id:
                continue

            # Extract and translate
            if contig in contigs:
                nuc_seq = contigs[contig][start - 1:end]
                if strand == "-":
                    nuc_seq = reverse_complement(nuc_seq)
                protein_seq = translate(nuc_seq)

                proteins[gene_id] = ProteinSequence(
                    gene_id=gene_id,
                    sequence=protein_seq,
                    start=start,
                    end=end,
                    strand=strand,
                    contig=contig,
                )

    return proteins


def extract_protein_sequences(
    genome_fasta: str | Path,
    gff3_path: str | Path | None = None,
) -> dict[str, str]:
    """Extract protein sequences from genome.

    If GFF3 provided: use CDS coordinates to extract nucleotide,
    translate with standard codon table.
    If no GFF3: assume FASTA contains protein sequences directly.

    Parameters
    ----------
    genome_fasta : path to genome FASTA (nucleotide or protein)
    gff3_path : optional path to GFF3 annotation

    Returns
    -------
    dict mapping gene_id → protein sequence
    """
    if gff3_path:
        proteins = extract_proteins_from_gff3(genome_fasta, gff3_path)
        return {pid: ps.sequence for pid, ps in proteins.items()}
    else:
        return extract_proteins_from_fasta(genome_fasta)
