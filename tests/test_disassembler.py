"""Disassembler unit tests."""
from __future__ import annotations

from helixlang.core.bytecode import Chunk
from helixlang.core.codon_table import Op
from helixlang.core.disassembler import disassemble


def _chunk() -> Chunk:
    chunk = Chunk()
    chunk.emit(Op.OP_START)
    chunk.emit(Op.OP_BUILD_PROTEIN, 2)
    chunk.emit(Op.OP_HALT)
    chunk.gene_offsets["g"] = 0
    chunk.constants = [("src", "x")]
    chunk.lines = [1, 1, 1]
    chunk.codon_indices = [0, 0, 0]
    return chunk


def test_disassemble_header_and_gene_offsets() -> None:
    out = disassemble(_chunk(), "prog")
    assert out.startswith("=== prog ===")
    assert "--- Gene Offsets ---" in out
    assert "g" in out
    assert "OP_START" in out and "OP_BUILD_PROTEIN" in out and "OP_HALT" in out
    assert "--- Constants ---" in out


def test_disassemble_empty_chunk_no_tables() -> None:
    out = disassemble(Chunk(), "empty")
    assert "=== empty ===" in out
    assert "--- Gene Offsets ---" not in out
    assert "--- Constants ---" not in out
    assert out.endswith("--- Code ---")


def test_disassemble_unknown_opcode() -> None:
    chunk = Chunk()
    chunk.code.append(0xFF)  # not a valid Op
    chunk.code.append(int(Op.OP_HALT))
    out = disassemble(chunk, "bad")
    assert "<unknown 0xFF>" in out
    assert "OP_HALT" in out


def test_disassemble_codon_and_line_annotation() -> None:
    chunk = Chunk()
    chunk.emit(Op.OP_START)
    chunk.lines = [3]
    chunk.codon_indices = [2]
    out = disassemble(chunk, "ann")
    assert "codon #2" in out
    assert "line 3" in out
