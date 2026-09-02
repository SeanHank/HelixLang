"""Bytecode Chunk: opcode sequence + constant pool + line/codon mapping + gene offset table."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from helixlang.core.codon_table import Op

# ── Bytecode ABI version ──────────────────────────────────────────────────
# Frozen as of HelixLang 2026.9.0.
# or Chunk layout **must** bump this constant and update spec/bytecode-abi.md.
# Note: OP_USE_PLUGIN (doc/36 §3.2) is a compiler-generated opcode that does NOT
# change the codon→opcode mapping or operand layout of any prior op, so the ABI
# version is deliberately left at 1 (bumping would invalidate every golden
# result that records OPCODE_VERSION). The hxbc loader still enforces strict ABI
# equality on this constant (doc/36 F9).
OPCODE_VERSION: int = 1


@dataclass(slots=True)
class Chunk:
    """Bytecode container. All emit methods return the starting offset of the emission."""

    code: bytearray = field(default_factory=bytearray)
    constants: list[Any] = field(default_factory=list)
    lines: list[int] = field(default_factory=list)
    codon_indices: list[int] = field(default_factory=list)
    gene_offsets: dict[str, int] = field(default_factory=dict)

    def emit(self, op: Op, *operands: int,
             line: int = 0, codon_index: int = -1) -> int:
        """Emit one instruction. operands are variable-length byte values."""
        start = len(self.code)
        self.code.append(int(op))
        self.lines.append(line)
        self.codon_indices.append(codon_index)
        for b in operands:
            self.code.append(b & 0xFF)
            self.lines.append(line)
            self.codon_indices.append(codon_index)
        return start

    def emit_u16(self, op: Op, value: int,
                 line: int = 0, codon_index: int = -1) -> int:
        """Emit an instruction with a u16 operand."""
        return self.emit(op, (value >> 8) & 0xFF, value & 0xFF,
                         line=line, codon_index=codon_index)

    def add_constant(self, value: Any) -> int:
        """Add a value to the constant pool, returning its index (deduplicated)."""
        for i, c in enumerate(self.constants):
            if c == value:
                return i
        self.constants.append(value)
        return len(self.constants) - 1

    def read_u8(self, ip: int) -> tuple[int, int]:
        """Read a 1-byte operand, returning (value, new_ip)."""
        return self.code[ip], ip + 1

    def read_u16(self, ip: int) -> tuple[int, int]:
        """Read a 2-byte operand (big-endian), returning (value, new_ip)."""
        return (self.code[ip] << 8) | self.code[ip + 1], ip + 2

    def __len__(self) -> int:
        return len(self.code)
