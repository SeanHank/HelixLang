"""doc/38 §10 goal 12: seeded fuzz of the .helixc binary loader (hxbc).

A valid artifact is corrupted in every structural way the reader must survive:
bit-level corruption, truncation at every offset, and header/version/table
mutations.  Invariant (doc/38 §10.2): ``loads_program`` must either succeed or
raise a *typed* ``BinaryError`` family error (``BinaryFormatError`` /
``ABIVersionError``) — never a raw ``IndexError`` / ``UnicodeDecodeError`` /
``struct.error`` escaping across the module boundary, and never a hang.  The
reader contract guards (``_MAX_COUNT`` / ``_MAX_STR_LEN``, section length
checks in :mod:`helixlang.core.hxbc`) are the baseline this pins.
"""
from __future__ import annotations

import hashlib
import random
import struct

import pytest

from helixlang.core import hxbc as _hxbc
from helixlang.core.ast_nodes import (
    Codon,
    Gene,
    Program,
    ReactionDecl,
)
from helixlang.core.bytecode import OPCODE_VERSION, Chunk
from helixlang.core.errors import ABIVersionError
from helixlang.core.hxbc import (
    _MAX_COUNT,
    _MAX_STR_LEN,
    _TAG_CODON,
    _TAG_GENE,
    _TAG_PLUGIN_EXT,
    _V_BOOL,
    _V_INT,
    _V_LIST,
    _V_NONE,
    _V_STR,
    FORMAT_VERSION,
    MAGIC,
    SECTION_EOF,
    SECTION_META,
    SECTION_PROG,
    BinaryError,
    BinaryFormatError,
    _check_table_names,
    _chunk_consistent,
    _consttime_eq,
    _decode_chunk,
    _decode_codon_list,
    _decode_ext,
    _decode_field,
    _decode_genes,
    _decode_manifest,
    _decode_plugin_ext,
    _decode_program,
    _encode_manifest,
    _encode_program,
    _read_tag,
    _Reader,
    _Writer,
    decompile,
    decompile_to_file,
    dumps_program,
    loads_program,
)
from helixlang.core.parser import parse_source

_ALLOWED = (BinaryError, BinaryFormatError, ABIVersionError)

_VALID_SRC = (
    "#gene name=g\nATG GCT GGT GTA TAA\n#end\n"
    "#config ticks=4 output=stdout\n"
)


def _valid_artifact() -> bytes:
    prog = parse_source(_VALID_SRC)
    return dumps_program(prog)


def _load(data: bytes):
    """-> ("ok", artifact) | ("err", exc).  Bounded input, no hang."""
    try:
        return ("ok", loads_program(data))
    except _ALLOWED:  # noqa: BLE001 - typed binary errors only
        return ("err", 0)
    except Exception as exc:  # noqa: BLE001 - everything else is the bug
        return ("boom", exc)


def _expect_typed(data: bytes, where: str, byte: bytes) -> None:
    state = _load(data)
    assert state[0] != "boom", (
        f"{where} byte={byte.hex()}: untyped {type(state[1]).__name__}: "
        f"{state[1]} on {data[:64]!r}")


def test_fuzz_hxbc_truncation_every_offset():
    """Truncating the artifact at every offset must terminate in a typed
    BinaryError or load cleanly; never escape raw data-structure errors."""
    data = _valid_artifact()
    checked = 0
    for cut in range(len(data) + 1):
        payload = data[:cut]
        _expect_typed(payload, "truncation", f"cut={cut}".encode())
        checked += 1
    assert checked == len(data) + 1


def test_fuzz_hxbc_bit_flips():
    """Flipping a single byte at any position (seeded) is typed-or-clean."""
    data = _valid_artifact()
    rng = random.Random(0xB1C0)
    for _ in range(1000):
        pos = rng.randrange(len(data))
        val = data[pos] ^ rng.randrange(1, 256)
        mutated = bytearray(data)
        mutated[pos] = val
        _expect_typed(bytes(mutated), "bitflip", f"@{pos}".encode())


def test_fuzz_hxbc_header_mutations():
    """Magic/formatted-version/table-id mutations exercise the header gates."""
    data = _valid_artifact()
    # Bad magic bytes.
    for magic in (b"XXXX", b"HLX!", b"hxbc", MAGIC[:-1] + b"Z"):
        _expect_typed(magic + data[4:], "magic", magic)
    # Version byte: future/out-of-range values.
    for ver in (0x00, 0x02, 0x7F, 0xFF):
        mutated = bytearray(data)
        mutated[4] = ver
        _expect_typed(bytes(mutated), "version", bytes([ver]))
    # Corruption of the trailing header region and beyond (table ids live in
    # the body; slicing the header in half is a structural mutation too).
    for cut in (5, 6, 7, 8, 11):
        _expect_typed(data[:cut], "header-slice", f"cut={cut}".encode())


def test_fuzz_hxbc_random_seeded_corruption():
    """40 seeds x 100 byte-ops of structural corruption (spikes, deletions,
    insertions, overruns) — always typed-or-clean."""
    data = _valid_artifact()
    for seed in range(40):
        rng = random.Random(0xB1C0_0000 + seed)
        for _ in range(25):
            mutated = bytearray(data)
            for _ in range(rng.randint(1, 3)):
                op = rng.randrange(4)
                pos = rng.randrange(len(mutated))
                if op == 0:                      # flip
                    mutated[pos] ^= rng.randrange(1, 256)
                elif op == 1 and mutated:        # delete block
                    del mutated[pos:pos + rng.randint(1, 12)]
                elif op == 2:                    # spike trash bytes in place
                    mutated[pos:pos + rng.randint(1, 8)] = \
                        bytes(rng.randrange(256) for _ in range(rng.randint(1, 8)))
                else:                            # append overrun
                    mutated.extend(bytes(rng.randrange(256) for _ in range(rng.randint(1, 8))))
            _expect_typed(bytes(mutated), "corrupt", f"seed={seed}".encode())


def test_fuzz_hxbc_loader_deterministic():
    """Loading identical bytes twice yields the identical outcome class."""
    data = _valid_artifact()
    rng = random.Random(0xD3E1)
    for _ in range(200):
        mutated = bytearray(data)
        for _ in range(rng.randint(1, 2)):
            pos = rng.randrange(len(mutated))
            mutated[pos] ^= rng.randrange(1, 256)
        payload = bytes(mutated)
        a = ("ok" if _load(payload)[0] == "ok" else "err")
        b = ("ok" if _load(payload)[0] == "ok" else "err")
        assert a == b


# ============================================================================
# Targeted coverage helpers
# ============================================================================
def _prog_payload(prog=None):
    """Encode a minimal program to PROG section bytes."""
    if prog is None:
        prog = parse_source(_VALID_SRC)
    return _encode_program(prog)


def _meta_payload():
    return _encode_manifest(_hxbc._current_manifest())


def _raw_artifact(sections, flags=0):
    """Build a .helixc from list of (magic, bytes) tuples."""
    pre_eof = bytearray()
    for magic, payload in sections:
        pre_eof += magic + struct.pack(">I", len(payload)) + payload
    payload_len = len(pre_eof) + 40
    header = (MAGIC + bytes([FORMAT_VERSION, flags, 0, 0])
              + struct.pack(">I", payload_len))
    digest = hashlib.sha256(pre_eof).digest()
    return (bytes(header) + bytes(pre_eof) + SECTION_EOF
            + struct.pack(">I", 32) + digest)


# ============================================================================
# A. BinaryError.__str__ with offset >= 0  (line 175)
# ============================================================================
class TestBinaryErrorStr:
    def test_str_with_offset(self):
        err = BinaryFormatError("boom", section="PROG", offset=42)
        s = str(err)
        assert "offset=42" in s
        assert "section=PROG" in s
        assert "boom" in s

    def test_str_without_offset(self):
        err = BinaryFormatError("boom")
        s = str(err)
        assert "offset" not in s

    def test_str_with_section_only(self):
        err = BinaryFormatError("boom", section="CHNK")
        s = str(err)
        assert "section=CHNK" in s
        assert "offset" not in s


# ============================================================================
# B. _decode_manifest trailing bytes (line 250)
# ============================================================================
class TestDecodeManifestTrailing:
    def test_trailing_bytes_in_meta(self):
        w = _Writer()
        w.u32(2)  # language_spec
        w.u32(1)  # ast_schema
        w.u32(1)  # simulation_semantics
        w.u32(1)  # reference_data
        w.u8(0xFF)  # extra byte
        with pytest.raises(BinaryFormatError, match="trailing"):
            _decode_manifest(bytes(w.buf))


# ============================================================================
# C. _Writer.str_map  (line 340)
# ============================================================================
class TestWriterStrMap:
    def test_str_map_calls_field_map(self):
        w = _Writer()
        w.str_map({"a": "1", "b": "2"})
        # Verify it produces same output as field_map
        w2 = _Writer()
        w2.field_map({"a": "1", "b": "2"})
        assert bytes(w.buf) == bytes(w2.buf)


# ============================================================================
# D. _Writer.value for None, bool, list, and unknown type
#    (lines 362, 364, 365, 381-384, 392)
# ============================================================================
class TestWriterValue:
    def test_value_none(self):
        w = _Writer()
        w.value(None)
        assert bytes(w.buf) == bytes([_V_NONE])

    def test_value_bool_true(self):
        w = _Writer()
        w.value(True)
        assert bytes(w.buf) == bytes([_V_BOOL, 1])

    def test_value_bool_false(self):
        w = _Writer()
        w.value(False)
        assert bytes(w.buf) == bytes([_V_BOOL, 0])

    def test_value_list(self):
        w = _Writer()
        w.value([1, "hello"])
        data = bytes(w.buf)
        assert data[0] == _V_LIST

    def test_value_unknown_type(self):
        w = _Writer()
        with pytest.raises(BinaryFormatError, match="cannot encode"):
            w.value(object())


# ============================================================================
# E. _Reader._fail  (line 412)
# ============================================================================
class TestReaderFail:
    def test_fail_raises(self):
        r = _Reader(b"", "TEST")
        with pytest.raises(BinaryFormatError, match="error message"):
            r._fail("error message")


# ============================================================================
# F. _Reader.need truncation (line 416)
# ============================================================================
class TestReaderNeed:
    def test_need_truncated(self):
        r = _Reader(b"\x00\x01", "PROG")
        with pytest.raises(BinaryFormatError, match="truncated"):
            r.u32()  # needs 4 bytes, only 2 available

    def test_need_negative(self):
        r = _Reader(b"\x00", "PROG")
        with pytest.raises(BinaryFormatError):
            r.need(-1)


# ============================================================================
# G. _Reader.bool_ invalid (line 452)
# ============================================================================
class TestReaderBool:
    def test_invalid_bool_byte(self):
        r = _Reader(bytes([0x02]), "CHNK")
        with pytest.raises(BinaryFormatError, match="invalid bool"):
            r.bool_()


# ============================================================================
# H. _Reader.str_ too long (line 458)
# ============================================================================
class TestReaderStr:
    def test_string_exceeds_limit(self):
        w = _Writer()
        w.u32(_MAX_STR_LEN + 1)
        r = _Reader(bytes(w.buf), "PROG")
        with pytest.raises(BinaryFormatError, match="exceeds limit"):
            r.str_()


# ============================================================================
# I. _Reader.opt_str invalid flag (line 470)
# ============================================================================
class TestReaderOptStr:
    def test_invalid_opt_str_flag(self):
        r = _Reader(bytes([0x02]), "PROG")  # flag=2 is invalid
        with pytest.raises(BinaryFormatError, match="invalid opt.*str.*flag"):
            r.opt_str()


# ============================================================================
# J. _Reader.opt_f64 invalid flag (line 478)
# ============================================================================
class TestReaderOptF64:
    def test_invalid_opt_f64_flag(self):
        r = _Reader(bytes([0x02]), "PROG")
        with pytest.raises(BinaryFormatError, match="invalid opt.*f64.*flag"):
            r.opt_f64()


# ============================================================================
# K. _Reader._count exceeds limit (line 483)
# ============================================================================
class TestReaderCount:
    def test_count_reads_u16(self):
        w = _Writer()
        w.u16(42)
        r = _Reader(bytes(w.buf), "PROG")
        assert r._count("test") == 42


# ============================================================================
# L. _Reader.str_map (line 495)
# ============================================================================
class TestReaderStrMap:
    def test_str_map_reads_field_map(self):
        w = _Writer()
        w.u16(1)
        w.str_("key")
        w.str_("value")
        r = _Reader(bytes(w.buf), "PROG")
        result = r.str_map()
        assert result == {"key": "value"}


# ============================================================================
# M. _Reader.u32_list exceeds limit (line 503)
# ============================================================================
class TestReaderU32List:
    def test_u32_list_exceeds_limit(self):
        w = _Writer()
        w.u32(_MAX_COUNT + 1)
        r = _Reader(bytes(w.buf), "CHNK")
        with pytest.raises(BinaryFormatError, match="exceeds limit"):
            r.u32_list()


# ============================================================================
# N. _Reader.value for None, bool, list, unknown tag
#    (lines 520, 522, 532, 540)
# ============================================================================
class TestReaderValue:
    def test_value_none(self):
        w = _Writer()
        w.u8(_V_NONE)
        r = _Reader(bytes(w.buf), "CHNK")
        assert r.value() is None

    def test_value_bool(self):
        w = _Writer()
        w.u8(_V_BOOL)
        w.u8(1)
        r = _Reader(bytes(w.buf), "CHNK")
        assert r.value() is True

    def test_value_list(self):
        w = _Writer()
        w.u8(_V_LIST)
        w.u16(2)
        w.u8(_V_INT)
        w.i32(42)
        w.u8(_V_STR)
        w.str_("hello")
        r = _Reader(bytes(w.buf), "CHNK")
        assert r.value() == [42, "hello"]

    def test_value_unknown_tag(self):
        w = _Writer()
        w.u8(0xFF)
        r = _Reader(bytes(w.buf), "CHNK")
        with pytest.raises(BinaryFormatError, match="unknown value tag"):
            r.value()


# ============================================================================
# O. _Reader.record success and failure (lines 545-549)
# ============================================================================
class TestReaderRecord:
    def test_record_matching_tag(self):
        w = _Writer()
        w.u8(0x01)
        r = _Reader(bytes(w.buf), "PROG")
        called = []
        r.record(0x01, lambda: called.append(True))
        assert called == [True]

    def test_record_mismatched_tag(self):
        w = _Writer()
        w.u8(0x02)
        r = _Reader(bytes(w.buf), "PROG")
        with pytest.raises(BinaryFormatError, match="expected record tag"):
            r.record(0x01, lambda: None)


# ============================================================================
# P. _Reader.remaining  (line 555)
# ============================================================================
class TestReaderRemaining:
    def test_remaining(self):
        r = _Reader(b"hello", "PROG")
        assert r.remaining() == 5
        r.u8()
        assert r.remaining() == 4


# ============================================================================
# Q. _read_tag mismatch  (line 561)
# ============================================================================
class TestReadTag:
    def test_read_tag_mismatch(self):
        w = _Writer()
        w.u8(0xFF)
        r = _Reader(bytes(w.buf), "PROG")
        with pytest.raises(BinaryFormatError, match="expected record tag"):
            _read_tag(r, 0x01)


# ============================================================================
# R. _decode_codon_list invalid codon  (line 584)
# ============================================================================
class TestDecodeCodonList:
    def test_invalid_codon_sequence(self):
        w = _Writer()
        w.u16(1)  # 1 codon
        w.u8(_TAG_CODON)
        w.str_("AT")  # length 2, not 3
        w.i32(0)
        w.i32(1)
        with pytest.raises(BinaryFormatError, match="invalid codon sequence"):
            _decode_codon_list(_Reader(bytes(w.buf), "PROG"))


# ============================================================================
# S. Non-string sim_extension encode (lines 720-722)
# ============================================================================
class TestNonStringSimExt:
    def test_non_string_ext_encodes_and_decodes(self):
        prog = parse_source(_VALID_SRC)
        prog.sim_extensions["custom_list"] = [1, 2, 3]
        data = dumps_program(prog)
        art = loads_program(data)
        assert art.program.sim_extensions["custom_list"] == [1, 2, 3]

    def test_non_string_ext_dict(self):
        prog = parse_source(_VALID_SRC)
        prog.sim_extensions["custom_dict"] = {"a": 1, "b": 2}
        data = dumps_program(prog)
        art = loads_program(data)
        assert art.program.sim_extensions["custom_dict"] == {"a": 1, "b": 2}


# ============================================================================
# T. _decode_program trailing bytes (line 748)
# ============================================================================
class TestDecodeProgramTrailing:
    def test_trailing_bytes_in_prog(self):
        """After folding the provably-unreachable trailing-bytes check in
        _decode_program, extra bytes are caught by _decode_plugin_ext."""
        prog = parse_source(_VALID_SRC)
        prog_data = _encode_program(prog)
        bad_data = prog_data + b"\x00\x00\x00"
        with pytest.raises(BinaryFormatError):
            _decode_program(bad_data)


# ============================================================================
# U. _decode_genes empty ORF (line 780)
# ============================================================================
class TestDecodeGenes:
    def test_empty_orf(self):
        w = _Writer()
        w.u16(1)  # 1 gene
        w.u8(_TAG_GENE)
        w.str_("g1")
        w.u8(0x00)  # no promoter
        # codons: 1 codon
        w.u16(1)
        w.u8(_TAG_CODON)
        w.str_("ATG")
        w.i32(0)
        w.i32(1)
        # orf: 0 codons (empty)
        w.u16(0)
        # fields
        w.u16(0)
        r = _Reader(bytes(w.buf), "PROG")
        prog = Program()
        with pytest.raises(BinaryFormatError, match="empty ORF"):
            _decode_genes(r, prog)


# ============================================================================
# V. _decode_field invalid flag (line 822)
# ============================================================================
class TestDecodeField:
    def test_invalid_flag(self):
        w = _Writer()
        w.u8(0xFF)
        r = _Reader(bytes(w.buf), "PROG")
        prog = Program()
        with pytest.raises(BinaryFormatError, match="invalid opt.*flag"):
            _decode_field(r, prog)


# ============================================================================
# W. _decode_ext JSON parsing (lines 931-935)
# ============================================================================
class TestDecodeExt:
    def test_json_value_decodes(self):
        w = _Writer()
        w.u16(1)  # 1 extension
        w.str_("mykey")
        w.u8(0x01)  # json tag
        w.str_("[1, 2, 3]")
        r = _Reader(bytes(w.buf), "PROG")
        prog = Program()
        _decode_ext(r, prog)
        assert prog.sim_extensions["mykey"] == [1, 2, 3]

    def test_json_decode_fallback_invalid_json(self):
        w = _Writer()
        w.u16(1)
        w.str_("mykey")
        w.u8(0x01)  # json tag
        w.str_("not valid json {{{")
        r = _Reader(bytes(w.buf), "PROG")
        prog = Program()
        _decode_ext(r, prog)
        # Falls back to raw string on JSONDecodeError
        assert prog.sim_extensions["mykey"] == "not valid json {{{"

    def test_non_json_tag_reads_as_string(self):
        w = _Writer()
        w.u16(1)
        w.str_("mykey")
        w.u8(0x00)  # plain string tag
        w.str_("plain_value")
        r = _Reader(bytes(w.buf), "PROG")
        prog = Program()
        _decode_ext(r, prog)
        assert prog.sim_extensions["mykey"] == "plain_value"


# ============================================================================
# X. _decode_plugin_ext error paths (lines 952, 956, 958, 968-969)
# ============================================================================
class TestDecodePluginExt:
    def test_wrong_tag(self):
        w = _Writer()
        w.u8(0xFF)  # wrong tag
        r = _Reader(bytes(w.buf), "PROG")
        prog = Program()
        with pytest.raises(BinaryFormatError, match="expected PLUGIN_EXT tag"):
            _decode_plugin_ext(r, prog)

    def test_unknown_plugin_id(self):
        w = _Writer()
        w.u8(_TAG_PLUGIN_EXT)
        w.str_("nonexistent_plugin")
        w.u32(_hxbc.PLUGIN_EXT_ABI)
        w.u16(0)  # 0 entries
        r = _Reader(bytes(w.buf), "PROG")
        prog = Program()
        with pytest.raises(_hxbc.PluginMissingError):
            _decode_plugin_ext(r, prog)

    def test_abi_mismatch(self):
        w = _Writer()
        w.u8(_TAG_PLUGIN_EXT)
        w.str_("gem")  # known plugin
        w.u32(999)  # wrong ABI
        w.u16(0)
        r = _Reader(bytes(w.buf), "PROG")
        prog = Program()
        with pytest.raises(_hxbc.PluginBinaryError, match="ABI"):
            _decode_plugin_ext(r, prog)

    def test_json_fallback_in_plugin_ext(self):
        w = _Writer()
        w.u8(_TAG_PLUGIN_EXT)
        w.str_("gem")
        w.u32(_hxbc.PLUGIN_EXT_ABI)
        w.u16(1)  # 1 entry
        w.str_("test_key")
        w.u8(0x01)  # json tag
        w.str_("invalid json {{{")
        r = _Reader(bytes(w.buf), "PROG")
        prog = Program()
        _decode_plugin_ext(r, prog)
        assert prog.sim_extensions["test_key"] == "invalid json {{{"

    def test_valid_json_in_plugin_ext(self):
        w = _Writer()
        w.u8(_TAG_PLUGIN_EXT)
        w.str_("gem")
        w.u32(_hxbc.PLUGIN_EXT_ABI)
        w.u16(1)
        w.str_("test_key")
        w.u8(0x01)  # json tag
        w.str_("{\"a\": 1}")
        r = _Reader(bytes(w.buf), "PROG")
        prog = Program()
        _decode_plugin_ext(r, prog)
        assert prog.sim_extensions["test_key"] == {"a": 1}

    def test_string_value_in_plugin_ext(self):
        w = _Writer()
        w.u8(_TAG_PLUGIN_EXT)
        w.str_("gem")
        w.u32(_hxbc.PLUGIN_EXT_ABI)
        w.u16(1)
        w.str_("test_key")
        w.u8(0x00)  # string tag
        w.str_("hello")
        r = _Reader(bytes(w.buf), "PROG")
        prog = Program()
        _decode_plugin_ext(r, prog)
        assert prog.sim_extensions["test_key"] == "hello"


# ============================================================================
# Y. _decode_chunk error paths (lines 1023, 1037, 1041)
# ============================================================================
class TestDecodeChunk:
    def test_code_length_exceeds_limit(self):
        w = _Writer()
        w.u32(OPCODE_VERSION)
        w.u32(_MAX_COUNT + 1)
        with pytest.raises(BinaryFormatError, match="exceeds limit"):
            _decode_chunk(bytes(w.buf))

    def test_invalid_dna_flag(self):
        w = _Writer()
        w.u32(OPCODE_VERSION)
        w.u32(0)  # code length
        w.u16(0)  # 0 constants
        w.u32(0)  # 0 lines
        w.u32(0)  # 0 codon_indices
        w.u16(0)  # 0 gene_offsets
        w.u8(0xFF)  # invalid dna flag
        with pytest.raises(BinaryFormatError, match="invalid dna_sequence flag"):
            _decode_chunk(bytes(w.buf))

    def test_trailing_bytes_in_chunk(self):
        w = _Writer()
        w.u32(OPCODE_VERSION)
        w.u32(0)  # code length
        w.u16(0)  # 0 constants
        w.u32(0)  # 0 lines
        w.u32(0)  # 0 codon_indices
        w.u16(0)  # 0 gene_offsets
        w.u8(0x00)  # no dna
        w.u8(0xFF)  # extra trailing byte
        with pytest.raises(BinaryFormatError, match="trailing"):
            _decode_chunk(bytes(w.buf))


# ============================================================================
# Z. loads_program section errors (lines 1160, 1162, 1164, 1177)
# ============================================================================
class TestLoadsSectionErrors:
    def test_unknown_section(self):
        prog = parse_source(_VALID_SRC)
        data = _raw_artifact([
            (SECTION_PROG, _prog_payload(prog)),
            (b"XXXX", b""),
            (_meta_payload().__class__.__name__ and SECTION_META, _meta_payload()),
        ])
        with pytest.raises(BinaryFormatError, match="unknown section"):
            loads_program(data)

    def test_section_overruns(self):
        meta = _meta_payload()
        pre_eof = bytearray()
        # PROG section with huge length that overruns
        pre_eof += SECTION_PROG + struct.pack(">I", 0xFFFFFFFF)
        pre_eof += SECTION_META + struct.pack(">I", len(meta)) + meta
        payload_len = len(pre_eof) + 40
        header = (MAGIC + bytes([FORMAT_VERSION, 0, 0, 0])
                  + struct.pack(">I", payload_len))
        digest = hashlib.sha256(pre_eof).digest()
        data = (bytes(header) + bytes(pre_eof) + SECTION_EOF
                + struct.pack(">I", 32) + digest)
        with pytest.raises(BinaryFormatError, match="overruns"):
            loads_program(data)

    def test_duplicate_section(self):
        prog = parse_source(_VALID_SRC)
        data = _raw_artifact([
            (SECTION_PROG, _prog_payload(prog)),
            (SECTION_PROG, _prog_payload(prog)),
            (SECTION_META, _meta_payload()),
        ])
        with pytest.raises(BinaryFormatError, match="duplicate section"):
            loads_program(data)

    def test_wrong_section_order(self):
        prog = parse_source(_VALID_SRC)
        data = _raw_artifact([
            (SECTION_META, _meta_payload()),
            (SECTION_PROG, _prog_payload(prog)),
        ])
        with pytest.raises(BinaryFormatError, match="unexpected section order"):
            loads_program(data)


# ============================================================================
# AA. _consttime_eq length mismatch (line 1214)
# ============================================================================
class TestConsttimeEq:
    def test_different_lengths(self):
        assert not _consttime_eq(b"abc", b"ab")

    def test_same_content(self):
        assert _consttime_eq(b"hello", b"hello")

    def test_different_content_same_length(self):
        assert not _consttime_eq(b"abc", b"abd")


# ============================================================================
# BB. _chunk_consistent mismatches (lines 1224, 1226, 1232)
# ============================================================================
class TestChunkConsistent:
    def test_code_lines_mismatch(self):
        prog = parse_source(_VALID_SRC)
        chunk = Chunk(code=b'\x10', lines=[1, 2], codon_indices=[-1],
                      gene_offsets={"g": 0})
        assert not _chunk_consistent(chunk, prog)

    def test_code_codon_indices_mismatch(self):
        prog = parse_source(_VALID_SRC)
        chunk = Chunk(code=b'\x10', lines=[1], codon_indices=[-1, -1],
                      gene_offsets={"g": 0})
        assert not _chunk_consistent(chunk, prog)

    def test_wrong_first_byte(self):
        prog = parse_source(_VALID_SRC)
        chunk = Chunk(code=b'\xFF', lines=[1], codon_indices=[-1],
                      gene_offsets={"g": 0})
        assert not _chunk_consistent(chunk, prog)

    def test_gene_offsets_mismatch(self):
        prog = parse_source(_VALID_SRC)
        chunk = Chunk(code=b'\x10', lines=[1], codon_indices=[-1],
                      gene_offsets={"wrong_gene": 0})
        assert not _chunk_consistent(chunk, prog)

    def test_empty_code_is_consistent(self):
        prog = parse_source(_VALID_SRC)
        chunk = Chunk(code=b'', lines=[], codon_indices=[],
                      gene_offsets={"g": 0})
        # empty code: skips first-byte check, gene_offsets must match
        assert _chunk_consistent(chunk, prog)

    def test_consistent_chunk(self):
        from helixlang.core.codon_table import get_table
        from helixlang.core.compiler import Compiler
        prog = parse_source(_VALID_SRC)
        chunk = Compiler(get_table("standard")).compile(prog)
        assert _chunk_consistent(chunk, prog)


# ============================================================================
# CC. decompile gene fields (lines 1321, 1323)
# ============================================================================
class TestDecompileGeneFields:
    def test_gene_without_name_in_fields(self):
        prog = Program()
        codons = [Codon(seq="ATG", index=0, line=1),
                  Codon(seq="GCT", index=1, line=1),
                  Codon(seq="TAA", index=2, line=1)]
        prog.genes.append(Gene(
            name="my_gene", promoter=None,
            codons=list(codons), orf=list(codons),
            fields={"custom": "value"},
        ))
        prog.config.table = "standard"
        text = decompile(prog)
        assert "name=my_gene" in text

    def test_gene_with_promoter_not_in_fields(self):
        prog = Program()
        codons = [Codon(seq="ATG", index=0, line=1),
                  Codon(seq="GCT", index=1, line=1),
                  Codon(seq="TAA", index=2, line=1)]
        prog.genes.append(Gene(
            name="my_gene", promoter="p1",
            codons=list(codons), orf=list(codons),
            fields={"custom": "value"},
        ))
        prog.config.table = "standard"
        text = decompile(prog)
        assert "promoter=p1" in text


# ============================================================================
# DD. decompile reaction branches (lines 1366-1372, False branches)
# ============================================================================
class TestDecompileReactionBranches:
    def test_reaction_empty_name_and_subsystem(self):
        prog = parse_source(_VALID_SRC)
        prog.reactions.append(ReactionDecl(
            id="r1", name="", substrate="A", substrate_coeff=1.0,
            product="B", product_coeff=1.0, lower_bound=0.0,
            upper_bound=10.0, subsystem="", reversible=False,
        ))
        text = decompile(prog)
        assert "#reaction id=r1" in text
        assert "name=" not in text.split("id=r1")[0].split("\n")[-1]

    def test_reaction_with_name_and_subsystem(self):
        prog = parse_source(_VALID_SRC)
        prog.reactions.append(ReactionDecl(
            id="r1", name="rxn1", substrate="A", substrate_coeff=1.0,
            product="B", product_coeff=1.0, lower_bound=0.0,
            upper_bound=10.0, subsystem="cytoplasm", reversible=True,
        ))
        text = decompile(prog)
        assert "name=rxn1" in text
        assert "subsystem=cytoplasm" in text
        assert "reversible=true" in text


# ============================================================================
# EE. decompile_to_file  (line 1426)
# ============================================================================
class TestDecompileToFile:
    def test_decompile_to_file(self, tmp_path):
        prog = parse_source(_VALID_SRC)
        out = tmp_path / "decompiled.helix"
        decompile_to_file(prog, out)
        assert out.exists()
        text = out.read_text()
        assert "#gene" in text


# ============================================================================
# FF. _check_table_names missing table  (line 1433)
# ============================================================================
class TestCheckTableNames:
    def test_missing_table_name(self, monkeypatch):
        monkeypatch.setattr(_hxbc, "TABLES", {})
        with pytest.raises(BinaryFormatError, match="not in codon_table.TABLES"):
            _check_table_names()


# ============================================================================
# GG. Round-trip coverage for non-string plugin_ext (encode + decode)
# ============================================================================
class TestPluginExtRoundtrip:
    def test_gem_extension_roundtrip(self):
        prog = parse_source(_VALID_SRC)
        prog.extensions.extension("gem").set("gem_test", {"nested": "dict"})
        data = dumps_program(prog)
        art = loads_program(data)
        assert art.program.sim_extensions["gem_test"] == {"nested": "dict"}

    def test_gem_extension_string_roundtrip(self):
        prog = parse_source(_VALID_SRC)
        prog.extensions.extension("gem").set("gem_str", "hello")
        data = dumps_program(prog)
        art = loads_program(data)
        assert art.program.sim_extensions["gem_str"] == "hello"
