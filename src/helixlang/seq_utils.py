"""Sequence utility functions (unified implementation to eliminate cross-module duplication).

Provides basic tools such as GC content calculation and reverse complement for
DNA sequences. All biology modules should import these functions from this
module to avoid :func:`gc_content` / :func:`reverse_complement` being redefined
in multiple places, which causes inconsistent behavior.

Design notes:
- Pure Python implementation with no numpy / BioPython dependency, keeping the
  core biology modules free of external dependencies.
- ``gc_content`` returns ``0.0`` for an empty string and is case-insensitive
  (it uppercases input before counting).
- ``reverse_complement`` supports IUPAC ambiguous bases (R/Y/S/W/K/M/B/V/D/H/N);
  unknown characters map to ``N``, and input is uppercased.

References:
- IUPAC nucleotide nomenclature (Nomenclature Committee, Eur J Biochem 1985 150:1)
"""
from __future__ import annotations

# IUPAC base complement table (including ambiguous bases)
_COMPLEMENT: dict[str, str] = {
    "A": "T", "T": "A", "C": "G", "G": "C",
    "U": "A",  # RNA base (in case an RNA sequence is passed)
    "N": "N",  # any base
    # IUPAC ambiguous bases
    "R": "Y", "Y": "R",  # R=A/G, Y=C/T
    "S": "S", "W": "W",  # S=G/C, W=A/T
    "K": "M", "M": "K",  # K=G/T, M=A/C
    "B": "V", "V": "B",  # B=C/G/T, V=A/C/G
    "D": "H", "H": "D",  # D=A/G/T, H=A/C/T
}


def gc_content(seq: str) -> float:
    """Compute the GC content ratio of a DNA/RNA sequence.

    Args:
        seq: nucleic acid sequence (case-insensitive; may contain RNA U)

    Returns:
        GC ratio in [0.0, 1.0]; returns ``0.0`` for an empty string.

    Examples:
        >>> gc_content("ATGC")
        0.5
        >>> gc_content("GGCC")
        1.0
        >>> gc_content("")
        0.0
    """
    if not seq:
        return 0.0
    upper = seq.upper()
    gc = upper.count("G") + upper.count("C")
    return gc / len(upper)


def reverse_complement(seq: str) -> str:
    """Compute the reverse complement of a DNA sequence (supports IUPAC ambiguous bases).

    Args:
        seq: DNA sequence (case-insensitive; may contain IUPAC ambiguous bases)

    Returns:
        Reverse complement; unknown bases are represented as ``N``.

    Examples:
        >>> reverse_complement("ATGC")
        'GCAT'
        >>> reverse_complement("AAATTT")
        'AAATTT'
        >>> reverse_complement("ACGTNR")
        'YNACGT'
    """
    return "".join(
        _COMPLEMENT.get(b, "N") for b in reversed(seq.upper())
    )


__all__ = ["gc_content", "reverse_complement", "max_homopolymer",
           "stop_codons_from_table"]


def max_homopolymer(seq: str) -> int:
    """Compute the longest homopolymer run length in a sequence.

    Used for DNA storage synthesis constraints (overly long homopolymers are
    prone to sequencing/synthesis errors).

    Args:
        seq: nucleic acid sequence (case-insensitive)

    Returns:
        Length of the longest run of identical bases; returns ``0`` for an empty string.

    Examples:
        >>> max_homopolymer("AAATGC")
        3
        >>> max_homopolymer("ACGT")
        1
        >>> max_homopolymer("")
        0
    """
    if not seq:
        return 0
    upper = seq.upper()
    best = 1
    run = 1
    for i in range(1, len(upper)):
        if upper[i] == upper[i - 1]:
            run += 1
            if run > best:
                best = run
        else:
            run = 1
    return best


def stop_codons_from_table(table: dict) -> set:
    """Extract the set of stop codons from a codon->Op mapping table.

    Stop codons = codons mapped to ``OP_HALT``. Shared by lexer/parser and
    server/cli to eliminate duplicate definitions.

    Args:
        table: ``{codon: Op}`` mapping (from :func:`codon_table.get_table`)

    Returns:
        Set of stop codon strings.
    """
    from helixlang.codon_table import Op
    return {codon for codon, op in table.items() if op == Op.OP_HALT}
