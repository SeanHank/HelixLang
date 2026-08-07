"""HelixLang codon table: 64 codons -> opcode mapping + three translation tables.

Design principles:
- 64 codons = 6-bit instruction space, split into ~30 logical opcodes
- Degeneracy as aliasing: synonymous codons map to the same opcode
- Third-base wobble value serves as an operand modifier (A=0, C=1, G=2, T=3)
- Switchable translation tables: standard / mitochondrial / ciliate
"""
from __future__ import annotations

from enum import IntEnum

from helixlang.errors import HelixError


class Op(IntEnum):
    """Bytecode opcodes."""

    # Control / flow
    OP_START = 0x10
    OP_HALT = 0x11
    OP_RETURN = 0x12
    OP_NOP = 0x13

    # Stack
    OP_PUSH_CONST = 0x20
    OP_POP = 0x21
    OP_DUP = 0x22
    OP_SWAP = 0x23

    # Synthesis
    OP_BUILD_PROTEIN = 0x30
    OP_BUILD_MEMBRANE = 0x31
    OP_BUILD_PIGMENT = 0x32

    # Behavior
    OP_MOVE = 0x40
    OP_SIGNAL = 0x41
    OP_DIVIDE = 0x42
    OP_DIE = 0x43
    OP_FEED = 0x44

    # Morphology
    OP_GROW_LSYSTEM = 0x50
    OP_DIFFUSE = 0x51
    OP_REACT = 0x52
    OP_EMIT_MORPHOGEN = 0x53

    # Memory / regulation
    OP_READ_MEM = 0x60
    OP_WRITE_MEM = 0x61
    OP_MODIFY_STATE = 0x62
    OP_REGULATE = 0x63
    OP_BIND = 0x64

    # Calls
    OP_CALL_GENE = 0x70

    # VM-level control flow (not mapped from codons; used by compiler-generated inter-gene barriers + debugger/direct bytecode)
    OP_JUMP = 0x80
    OP_JUMP_IF_ZERO = 0x81

    # Arithmetic
    OP_ADD = 0x90
    OP_SUB = 0x91
    OP_MUL = 0x92
    OP_LT = 0x93
    OP_NOT = 0x94

    # System
    OP_TICK = 0xF0
    OP_DEBUG = 0xFE


# Number of operand bytes per instruction
OP_OPERAND_BYTES: dict[Op, int] = {
    Op.OP_START: 0, Op.OP_HALT: 0, Op.OP_RETURN: 0, Op.OP_NOP: 0,
    Op.OP_PUSH_CONST: 1, Op.OP_POP: 0, Op.OP_DUP: 0, Op.OP_SWAP: 0,
    Op.OP_BUILD_PROTEIN: 1, Op.OP_BUILD_MEMBRANE: 1, Op.OP_BUILD_PIGMENT: 0,
    Op.OP_MOVE: 1, Op.OP_SIGNAL: 1, Op.OP_DIVIDE: 1, Op.OP_DIE: 1, Op.OP_FEED: 1,
    Op.OP_GROW_LSYSTEM: 1, Op.OP_DIFFUSE: 1, Op.OP_REACT: 1, Op.OP_EMIT_MORPHOGEN: 1,
    Op.OP_READ_MEM: 1, Op.OP_WRITE_MEM: 1, Op.OP_MODIFY_STATE: 1,
    Op.OP_REGULATE: 1, Op.OP_BIND: 1,
    Op.OP_CALL_GENE: 2,  # u16 gene offset
    Op.OP_JUMP: 2, Op.OP_JUMP_IF_ZERO: 2,
    Op.OP_ADD: 0, Op.OP_SUB: 0, Op.OP_MUL: 0, Op.OP_LT: 0, Op.OP_NOT: 0,
    Op.OP_TICK: 0, Op.OP_DEBUG: 0,
}

# Third-base wobble value
WOBBLE_BITS: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}


# Standard translation table (NCBI table 1)
STANDARD_TABLE: dict[str, Op] = {
    # Start / Stop
    "ATG": Op.OP_START,
    "TAA": Op.OP_HALT, "TAG": Op.OP_HALT, "TGA": Op.OP_HALT,
    # Phe (F)
    "TTT": Op.OP_PUSH_CONST, "TTC": Op.OP_PUSH_CONST,
    # Leu (L) — 6 codons (CTN + TTA/TTG)
    "CTT": Op.OP_GROW_LSYSTEM, "CTC": Op.OP_GROW_LSYSTEM,
    "CTA": Op.OP_GROW_LSYSTEM, "CTG": Op.OP_GROW_LSYSTEM,
    "TTA": Op.OP_GROW_LSYSTEM, "TTG": Op.OP_GROW_LSYSTEM,
    # Ile (I)
    "ATT": Op.OP_READ_MEM, "ATC": Op.OP_READ_MEM, "ATA": Op.OP_READ_MEM,
    # Val (V)
    "GTT": Op.OP_MOVE, "GTC": Op.OP_MOVE, "GTA": Op.OP_MOVE, "GTG": Op.OP_MOVE,
    # Ser (S) — 6 codons (TCN + AGT/AGC)
    "TCT": Op.OP_SIGNAL, "TCC": Op.OP_SIGNAL, "TCA": Op.OP_SIGNAL, "TCG": Op.OP_SIGNAL,
    "AGT": Op.OP_SIGNAL, "AGC": Op.OP_SIGNAL,
    # Pro (P)
    "CCT": Op.OP_MODIFY_STATE, "CCC": Op.OP_MODIFY_STATE,
    "CCA": Op.OP_MODIFY_STATE, "CCG": Op.OP_MODIFY_STATE,
    # Thr (T)
    "ACT": Op.OP_DIFFUSE, "ACC": Op.OP_DIFFUSE,
    "ACA": Op.OP_DIFFUSE, "ACG": Op.OP_DIFFUSE,
    # Ala (A)
    "GCT": Op.OP_BUILD_PROTEIN, "GCC": Op.OP_BUILD_PROTEIN,
    "GCA": Op.OP_BUILD_PROTEIN, "GCG": Op.OP_BUILD_PROTEIN,
    # Tyr (Y)
    "TAT": Op.OP_WRITE_MEM, "TAC": Op.OP_WRITE_MEM,
    # His (H)
    "CAT": Op.OP_REGULATE, "CAC": Op.OP_REGULATE,
    # Gln (Q)
    "CAA": Op.OP_EMIT_MORPHOGEN, "CAG": Op.OP_EMIT_MORPHOGEN,
    # Asn (N)
    "AAT": Op.OP_DIVIDE, "AAC": Op.OP_DIVIDE,
    # Lys (K)
    "AAA": Op.OP_DIE, "AAG": Op.OP_DIE,
    # Asp (D)
    "GAT": Op.OP_REACT, "GAC": Op.OP_REACT,
    # Glu (E)
    "GAA": Op.OP_FEED, "GAG": Op.OP_FEED,
    # Cys (C)
    "TGT": Op.OP_BIND, "TGC": Op.OP_BIND,
    # Trp (W)
    "TGG": Op.OP_BUILD_PIGMENT,
    # Arg (R) — 6 codons (CGN + AGA/AGG)
    "CGT": Op.OP_CALL_GENE, "CGC": Op.OP_CALL_GENE,
    "CGA": Op.OP_CALL_GENE, "CGG": Op.OP_CALL_GENE,
    "AGA": Op.OP_CALL_GENE, "AGG": Op.OP_CALL_GENE,
    # Gly (G)
    "GGT": Op.OP_BUILD_MEMBRANE, "GGC": Op.OP_BUILD_MEMBRANE,
    "GGA": Op.OP_BUILD_MEMBRANE, "GGG": Op.OP_BUILD_MEMBRANE,
}

# Mitochondrial table (NCBI table 2): TGA->Trp, ATA->Met, AGA/AGG->Stop
MITO_VERTEBRATE_TABLE: dict[str, Op] = {
    **STANDARD_TABLE,
    "TGA": Op.OP_BUILD_PIGMENT,
    "ATA": Op.OP_START,
    "AGA": Op.OP_HALT, "AGG": Op.OP_HALT,
}

# Ciliate table (NCBI table 6): TAA/TAG->Gln
CILIATE_TABLE: dict[str, Op] = {
    **STANDARD_TABLE,
    "TAA": Op.OP_EMIT_MORPHOGEN, "TAG": Op.OP_EMIT_MORPHOGEN,
}

TABLES: dict[str, dict[str, Op]] = {
    "standard": STANDARD_TABLE,
    "mito_vertebrate": MITO_VERTEBRATE_TABLE,
    "ciliate": CILIATE_TABLE,
}


def get_table(name: str) -> dict[str, Op]:
    """Get a translation table by name."""
    if name not in TABLES:
        raise HelixError(f"unknown translation table: {name!r}")
    return TABLES[name]


def wobble(codon: str) -> int:
    """Return the third-base wobble value (A=0, C=1, G=2, T=3)."""
    return WOBBLE_BITS[codon[2].upper()]


# Invariant self-check
assert len(STANDARD_TABLE) == 64, f"STANDARD_TABLE must have 64 codons, got {len(STANDARD_TABLE)}"
assert all(c in STANDARD_TABLE for c in (
    "AAA", "AAC", "AAG", "AAT", "ACA", "ACC", "ACG", "ACT",
    "AGA", "AGC", "AGG", "AGT", "ATA", "ATC", "ATG", "ATT",
    "CAA", "CAC", "CAG", "CAT", "CCA", "CCC", "CCG", "CCT",
    "CGA", "CGC", "CGG", "CGT", "CTA", "CTC", "CTG", "CTT",
    "GAA", "GAC", "GAG", "GAT", "GCA", "GCC", "GCG", "GCT",
    "GGA", "GGC", "GGG", "GGT", "GTA", "GTC", "GTG", "GTT",
    "TAA", "TAC", "TAG", "TAT", "TCA", "TCC", "TCG", "TCT",
    "TGA", "TGC", "TGG", "TGT", "TTA", "TTC", "TTG", "TTT",
)), "STANDARD_TABLE missing some codon"
