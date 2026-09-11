"""Bytecode Chunk unit tests.

Covers the Chunk emit/read/append surface not hit by compiler smoke tests:
- emit with multi-byte operands and per-byte line/codon metadata.
- emit_u16 big-endian encoding + read_u16 decoding.
- read_u8 single byte.
- add_constant deduplication.
- __len__.
- OPCODE_VERSION ABI constant.
"""
from __future__ import annotations

from helixlang.core.bytecode import OPCODE_VERSION, Chunk
from helixlang.core.codon_table import Op


def test_opcode_version() -> None:
    assert isinstance(OPCODE_VERSION, int)
    assert OPCODE_VERSION == 1


def test_empty_chunk_len() -> None:
    c = Chunk()
    assert len(c) == 0


def test_emit_and_len() -> None:
    c = Chunk()
    start = c.emit(Op.OP_PUSH_CONST, 0xAB, 0xCD, line=3, codon_index=7)
    assert start == 0
    assert len(c) == 3
    assert list(c.code) == [Op.OP_PUSH_CONST.value, 0xAB, 0xCD]
    assert c.lines == [3, 3, 3]
    assert c.codon_indices == [7, 7, 7]


def test_emit_returns_start_offset() -> None:
    c = Chunk()
    first = c.emit(Op.OP_NOP)
    second = c.emit(Op.OP_NOP)
    assert first == 0
    assert second == 1


def test_emit_u16_roundtrip() -> None:
    c = Chunk()
    start = c.emit_u16(Op.OP_CALL_GENE, 0x1234, line=5)
    assert start == 0
    assert list(c.code) == [Op.OP_CALL_GENE.value, 0x12, 0x34]
    value, new_ip = c.read_u16(1)
    assert value == 0x1234
    assert new_ip == 3


def test_read_u8() -> None:
    c = Chunk()
    c.emit(Op.OP_NOP, 0x55)
    value, new_ip = c.read_u8(1)
    assert value == 0x55
    assert new_ip == 2


def test_add_constant_deduplicated() -> None:
    c = Chunk()
    i0 = c.add_constant("hello")
    i1 = c.add_constant("hello")
    i2 = c.add_constant(5)
    assert i0 == i1 == 0
    assert i2 == 1
    assert c.constants == ["hello", 5]
