"""HelixLang binary artifact (.helixc) codec.

Implements ``doc/11-helixc-binary-format.md``: a versioned container holding

- a ``PROG`` section: the serialized ``Program`` AST (authoritative payload),
- an optional ``CHNK`` section: the precompiled bytecode ``Chunk``,
- an optional ``SRC`` section: the original source text,
- an ``EOF`` trailer with a SHA-256 checksum of all preceding sections.

The container is a self-describing, typed, length-checked byte stream -- never
``pickle``/``marshal`` -- so loading an artifact never executes code and any
corruption fails fast with :class:`BinaryFormatError`.

Public API::

    # Write
    dumps_program(program, *, chunk=None, source=None) -> bytes
    save_program(program, path, *, chunk=None, source=None) -> None
    compile_file(src_path, out_path, ...) -> ArtifactInfo

    # Read / run
    loads_program(data) -> LoadedArtifact
    load_program(path)  -> LoadedArtifact

    # Debug / test
    decompile(program) -> str
    decompile_to_file(program, path) -> None
    verify(path) -> None

Round-trip invariants (tests/test_helixc.py):
- ``parse(decompile(p))`` recompiles to the same ``Chunk`` as ``p``.
- canonical source round-trips; an embedded ``SRC`` decompiles byte-for-byte.
- ``dumps_program`` is deterministic (sorted map keys, no timestamps).
"""
from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from helixlang.ast_nodes import (
    BioInstruction,
    Codon,
    EnzymeDecl,
    FieldDecl,
    Gene,
    LSystemDecl,
    MediaDecl,
    MorphogenFeedback,
    PoolDecl,
    Program,
    Promoter,
    Regulation,
)
from helixlang.bytecode import Chunk
from helixlang.codon_table import TABLES, get_table
from helixlang.errors import HelixError

__all__ = [
    "ArtifactInfo",
    "BinaryError",
    "BinaryFormatError",
    "BinaryVersionError",
    "FORMAT_VERSION",
    "LoadedArtifact",
    "compile_file",
    "decompile",
    "decompile_to_file",
    "dumps_program",
    "load_program",
    "loads_program",
    "save_program",
    "verify",
]

MAGIC = b"HLXC"
FORMAT_VERSION = 1
SECTION_PROG = b"PROG"
SECTION_CHNK = b"CHNK"
SECTION_SRC = b"SRC "
SECTION_EOF = b"EOF "

_TABLE_IDS: dict[str, int] = {"standard": 0, "mito_vertebrate": 1, "ciliate": 2}
_TABLE_NAMES: dict[int, str] = {v: k for k, v in _TABLE_IDS.items()}

# Record tags (doc/11-helixc-binary-format.md §4.3)
_TAG_CODON = 0x01
_TAG_GENE = 0x02
_TAG_PROMOTER = 0x03
_TAG_REGULATION = 0x04
_TAG_LSYSTEM = 0x05
_TAG_FIELD = 0x06
_TAG_MORPHOGEN = 0x07
_TAG_MEDIA = 0x08
_TAG_ENZYME = 0x09
_TAG_POOL = 0x0A
_TAG_CONFIG = 0x0B
_TAG_BIO_INSTR = 0x0C
_TAG_PROGRAM = 0x0D

# Value tags for the generic constant-pool codec
_V_NONE = 0x00
_V_BOOL = 0x01
_V_INT = 0x02
_V_FLOAT = 0x03
_V_STR = 0x04
_V_TUPLE = 0x05
_V_LIST = 0x06
_V_DICT = 0x07

_MISSING = -1
_MAX_STR_LEN = 1 << 24
_MAX_COUNT = 1 << 20


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class BinaryError(HelixError):
    """Base class for .helixc codec errors."""

    def __init__(self, msg: str, *, section: str = "", offset: int = -1):
        super().__init__(msg)
        self.section = section
        self.offset = offset

    def __str__(self) -> str:
        loc = f" section={self.section}" if self.section else ""
        if self.offset >= 0:
            loc += f" offset={self.offset}"
        return f"[{type(self).__name__} @{loc}] {self.msg}"


class BinaryFormatError(BinaryError):
    """Malformed/unsafe/corrupt .helixc artifact."""


class BinaryVersionError(BinaryFormatError):
    """Unknown .helixc format version."""


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class _Writer:
    __slots__ = ("buf",)

    def __init__(self) -> None:
        self.buf = bytearray()

    def u8(self, v: int) -> None:
        self.buf.append(v & 0xFF)

    def u16(self, v: int) -> None:
        self.buf += struct.pack(">H", v)

    def u32(self, v: int) -> None:
        self.buf += struct.pack(">I", v)

    def i32(self, v: int) -> None:
        self.buf += struct.pack(">i", v)

    def f64(self, v: float) -> None:
        self.buf += struct.pack(">d", v)

    def bool_(self, v: bool) -> None:
        self.u8(1 if v else 0)

    def str_(self, s: str) -> None:
        b = s.encode("utf-8")
        self.u32(len(b))
        self.buf += b

    def opt_str(self, s: str | None) -> None:
        if s is None:
            self.u8(0x00)
        else:
            self.u8(0x01)
            self.str_(s)

    def opt_f64(self, v: float | None) -> None:
        if v is None:
            self.u8(0x00)
        else:
            self.u8(0x01)
            self.f64(v)

    def field_map(self, d: dict[str, str]) -> None:
        items = sorted(d.items())
        self.u16(len(items))
        for k, v in items:
            self.str_(k)
            self.str_(v)

    def str_map(self, d: dict[str, str]) -> None:
        self.field_map(d)

    def str_list(self, lst: list[str]) -> None:
        self.u16(len(lst))
        for s in lst:
            self.str_(s)

    def u32_list(self, lst: list[int]) -> None:
        self.u32(len(lst))
        for x in lst:
            self.u32(x & 0xFFFFFFFF)

    def int_key_map(self, d: dict[int, dict[str, str]]) -> None:
        items = sorted(d.items())
        self.u16(len(items))
        for k, v in items:
            self.u32(k)
            self.field_map(v)

    def value(self, v: Any) -> None:
        """Encode an arbitrary constant-pool value (nested, tagged)."""
        if v is None:
            self.u8(_V_NONE)
        elif isinstance(v, bool):
            self.u8(_V_BOOL)
            self.u8(1 if v else 0)
        elif isinstance(v, int):
            self.u8(_V_INT)
            self.i32(v)
        elif isinstance(v, float):
            self.u8(_V_FLOAT)
            self.f64(v)
        elif isinstance(v, str):
            self.u8(_V_STR)
            self.str_(v)
        elif isinstance(v, tuple):
            self.u8(_V_TUPLE)
            self.u16(len(v))
            for x in v:
                self.value(x)
        elif isinstance(v, list):
            self.u8(_V_LIST)
            self.u16(len(v))
            for x in v:
                self.value(x)
        elif isinstance(v, dict):
            self.u8(_V_DICT)
            self.u16(len(v))
            for k, val in sorted(v.items(), key=lambda kv: str(kv[0])):
                self.value(k)
                self.value(val)
        else:
            raise BinaryFormatError(
                f"cannot encode constant of type {type(v).__name__!r}")

    def record(self, tag: int, body: Callable[[], None]) -> None:
        self.u8(tag)
        body()


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
class _Reader:
    __slots__ = ("data", "pos", "section")

    def __init__(self, data: bytes, section: str = "PROG"):
        self.data = data
        self.pos = 0
        self.section = section

    def _fail(self, msg: str) -> NoReturn:
        raise BinaryFormatError(msg, section=self.section, offset=self.pos)

    def need(self, n: int) -> None:
        if n < 0 or self.pos + n > len(self.data):
            self._fail(f"truncated section: need {n} bytes at offset "
                       f"{self.pos}, have {len(self.data) - self.pos}")

    def u8(self) -> int:
        self.need(1)
        b = self.data[self.pos]
        self.pos += 1
        return b

    def u16(self) -> int:
        self.need(2)
        v = cast(int, struct.unpack_from(">H", self.data, self.pos)[0])
        self.pos += 2
        return v

    def u32(self) -> int:
        self.need(4)
        v = cast(int, struct.unpack_from(">I", self.data, self.pos)[0])
        self.pos += 4
        return v

    def i32(self) -> int:
        self.need(4)
        v = cast(int, struct.unpack_from(">i", self.data, self.pos)[0])
        self.pos += 4
        return v

    def f64(self) -> float:
        self.need(8)
        v = cast(float, struct.unpack_from(">d", self.data, self.pos)[0])
        self.pos += 8
        return v

    def bool_(self) -> bool:
        b = self.u8()
        if b not in (0, 1):
            self._fail(f"invalid bool byte {b}")
        return b == 1

    def str_(self) -> str:
        n = self.u32()
        if n > _MAX_STR_LEN:
            self._fail(f"string length {n} exceeds limit {_MAX_STR_LEN}")
        self.need(n)
        s = self.data[self.pos:self.pos + n].decode("utf-8")
        self.pos += n
        return s

    def opt_str(self) -> str | None:
        flag = self.u8()
        if flag == 0x00:
            return None
        if flag == 0x01:
            return self.str_()
        self._fail(f"invalid opt<str> flag {flag}")

    def opt_f64(self) -> float | None:
        flag = self.u8()
        if flag == 0x00:
            return None
        if flag == 0x01:
            return self.f64()
        self._fail(f"invalid opt<f64> flag {flag}")

    def _count(self, what: str) -> int:
        n = self.u16()
        if n > _MAX_COUNT:
            self._fail(f"{what} count {n} exceeds limit {_MAX_COUNT}")
        return n

    def field_map(self) -> dict[str, str]:
        d: dict[str, str] = {}
        for _ in range(self._count("field-map")):
            k = self.str_()
            v = self.str_()
            d[k] = v
        return d

    def str_map(self) -> dict[str, str]:
        return self.field_map()

    def str_list(self) -> list[str]:
        return [self.str_() for _ in range(self._count("str-list"))]

    def u32_list(self) -> list[int]:
        n = self.u32()
        if n > _MAX_COUNT:
            self._fail(f"u32-list count {n} exceeds limit {_MAX_COUNT}")
        self.need(n * 4)
        out = list(struct.unpack_from(f">{n}I", self.data, self.pos))
        self.pos += n * 4
        return out

    def int_key_map(self) -> dict[int, dict[str, str]]:
        d: dict[int, dict[str, str]] = {}
        for _ in range(self._count("int-key-map")):
            k = self.u32()
            v = self.field_map()
            d[k] = v
        return d

    def value(self) -> Any:
        tag = self.u8()
        if tag == _V_NONE:
            return None
        if tag == _V_BOOL:
            return self.bool_()
        if tag == _V_INT:
            return self.i32()
        if tag == _V_FLOAT:
            return self.f64()
        if tag == _V_STR:
            return self.str_()
        if tag == _V_TUPLE:
            return tuple(self.value() for _ in range(self._count("tuple")))
        if tag == _V_LIST:
            return [self.value() for _ in range(self._count("list"))]
        if tag == _V_DICT:
            n = self._count("dict")
            d: dict[Any, Any] = {}
            for _ in range(n):
                k = self.value()
                d[k] = self.value()
            return d
        raise BinaryFormatError(
            f"unknown value tag 0x{tag:02X}",
            section=self.section, offset=self.pos - 1)

    def record(self, expected: int, body: Callable[[], None]) -> None:
        tag = self.u8()
        if tag != expected:
            self._fail(f"expected record tag 0x{expected:02X}, "
                       f"got 0x{tag:02X}")
        body()

    def at_end(self) -> bool:
        return self.pos == len(self.data)

    def remaining(self) -> int:
        return len(self.data) - self.pos


def _read_tag(r: _Reader, expected: int) -> None:
    tag = r.u8()
    if tag != expected:
        r._fail(f"expected record tag 0x{expected:02X}, got 0x{tag:02X}")


# ---------------------------------------------------------------------------
# PROG section encoding
# ---------------------------------------------------------------------------
def _encode_codon_list(w: _Writer, codons: list[Codon]) -> None:
    w.u16(len(codons))
    for c in codons:
        w.u8(_TAG_CODON)
        w.str_(c.seq)
        w.u32(c.index)
        w.u32(c.line)


def _decode_codon_list(r: _Reader) -> list[Codon]:
    out: list[Codon] = []
    for _ in range(r._count("codon-list")):  # noqa: SLF001
        _read_tag(r, _TAG_CODON)
        seq = r.str_()
        index = r.i32()
        line = r.i32()
        if len(seq) != 3:
            r._fail(f"invalid codon sequence {seq!r}")  # noqa: SLF001
        out.append(Codon(seq=seq, index=index, line=line))
    return out


def _encode_program(program: Program) -> bytes:
    w = _Writer()
    w.record(_TAG_PROGRAM, lambda: _encode_program_body(w, program))
    return bytes(w.buf)


def _encode_program_body(w: _Writer, prog: Program) -> None:
    # genes
    w.u16(len(prog.genes))
    for g in prog.genes:
        w.u8(_TAG_GENE)
        w.str_(g.name)
        w.opt_str(g.promoter)
        _encode_codon_list(w, g.codons)
        _encode_codon_list(w, g.orf)
        w.field_map(g.fields)
    # promoters
    w.u16(len(prog.promoters))
    for p in prog.promoters:
        w.u8(_TAG_PROMOTER)
        w.str_(p.name)
        w.f64(p.strength)
        w.field_map(p.fields)
    # regulations
    w.u16(len(prog.regulations))
    for reg in prog.regulations:
        w.u8(_TAG_REGULATION)
        w.str_(reg.source)
        w.str_(reg.target)
        w.f64(reg.strength)
    # lsystems (str-key map, sorted)
    lsys = sorted(prog.lsystems.items())
    w.u16(len(lsys))
    for name, decl in lsys:
        w.str_(name)
        w.u8(_TAG_LSYSTEM)
        w.str_(decl.axiom)
        w.int_key_map(decl.rules)
        w.f64(decl.angle)
        w.f64(decl.step)
    # field_decl (optional)
    if prog.field_decl is None:
        w.u8(0x00)
    else:
        w.u8(0x01)
        fd = prog.field_decl
        w.u8(_TAG_FIELD)
        w.u32(fd.size)
        w.f64(fd.F)
        w.f64(fd.k)
        w.f64(fd.Du)
        w.f64(fd.Dv)
    # morphogen_feedback
    w.u16(len(prog.morphogen_feedback))
    for fb in prog.morphogen_feedback:
        w.u8(_TAG_MORPHOGEN)
        w.str_(fb.gene)
        w.str_(fb.channel)
        w.f64(fb.gain)
    # config
    cfg = prog.config
    w.u8(_TAG_CONFIG)
    w.u32(cfg.ticks)
    w.str_list(cfg.output)
    w.str_(cfg.table)
    w.u32(cfg.ops_per_tick)
    w.u32(cfg.react_steps)
    w.bool_(cfg.use_central_dogma)
    w.str_(cfg.species)
    w.str_(cfg.backend)
    w.field_map(cfg.sim)
    # bio_instructions
    w.u16(len(prog.bio_instructions))
    for inst in prog.bio_instructions:
        w.u8(_TAG_BIO_INSTR)
        w.str_(inst.kind)
        w.str_(inst.target)
        w.field_map(inst.params)
        w.u32(inst.line)
    # type_annotations (sorted)
    type_map = sorted(prog.type_annotations.items())
    w.u16(len(type_map))
    for k, v in type_map:
        w.str_(k)
        w.str_(v)
    # media
    w.u16(len(prog.media))
    for md in prog.media:
        w.u8(_TAG_MEDIA)
        w.str_(md.nutrient)
        w.f64(md.concentration)
        w.opt_f64(md.diffusion_um2_s)
    # enzymes
    w.u16(len(prog.enzymes))
    for enz in prog.enzymes:
        w.u8(_TAG_ENZYME)
        w.str_(enz.gene)
        w.str_(enz.reaction)
        w.opt_f64(enz.kcat)
    # pools
    w.u16(len(prog.pools))
    for pl in prog.pools:
        w.u8(_TAG_POOL)
        w.str_(pl.name)
        w.f64(pl.init)
    # sim_extensions (sorted)
    ext = sorted(prog.sim_extensions.items())
    w.u16(len(ext))
    for k, v in ext:
        w.str_(k)
        w.str_(v)


def _decode_program(data: bytes) -> Program:
    r = _Reader(data, "PROG")
    prog = Program()
    _decode_program_body(r, prog)
    if not r.at_end():
        r._fail(f"trailing {r.remaining()} bytes in PROG section")  # noqa: SLF001
    return prog


def _decode_program_body(r: _Reader, prog: Program) -> None:
    _read_tag(r, _TAG_PROGRAM)
    _decode_genes(r, prog)
    _decode_promoters(r, prog)
    _decode_regulations(r, prog)
    _decode_lsystems(r, prog)
    _decode_field(r, prog)
    _decode_morphogens(r, prog)
    _decode_config(r, prog)
    _decode_bio(r, prog)
    _decode_types(r, prog)
    _decode_media(r, prog)
    _decode_enzymes(r, prog)
    _decode_pools(r, prog)
    _decode_ext(r, prog)


def _decode_genes(r: _Reader, prog: Program) -> None:
    for _ in range(r._count("genes")):  # noqa: SLF001
        _read_tag(r, _TAG_GENE)
        name = r.str_()
        promoter = r.opt_str()
        codons = _decode_codon_list(r)
        orf = _decode_codon_list(r)
        fields = r.field_map()
        if not orf:
            r._fail(f"gene {name!r} has an empty ORF")  # noqa: SLF001
        prog.genes.append(Gene(name=name, promoter=promoter, codons=codons,
                               orf=orf, fields=fields))


def _decode_promoters(r: _Reader, prog: Program) -> None:
    for _ in range(r._count("promoters")):  # noqa: SLF001
        _read_tag(r, _TAG_PROMOTER)
        name = r.str_()
        strength = r.f64()
        fields = r.field_map()
        prog.promoters.append(Promoter(name=name, strength=strength,
                                       fields=fields))


def _decode_regulations(r: _Reader, prog: Program) -> None:
    for _ in range(r._count("regulations")):  # noqa: SLF001
        _read_tag(r, _TAG_REGULATION)
        source = r.str_()
        target = r.str_()
        strength = r.f64()
        prog.regulations.append(Regulation(source=source, target=target,
                                           strength=strength))


def _decode_lsystems(r: _Reader, prog: Program) -> None:
    for _ in range(r._count("lsystems")):  # noqa: SLF001
        name = r.str_()
        _read_tag(r, _TAG_LSYSTEM)
        axiom = r.str_()
        rules = r.int_key_map()
        angle = r.f64()
        step = r.f64()
        prog.lsystems[name] = LSystemDecl(
            name=name, axiom=axiom, rules=rules, angle=angle, step=step)


def _decode_field(r: _Reader, prog: Program) -> None:
    flag = r.u8()
    if flag == 0x00:
        return
    if flag != 0x01:
        r._fail(f"invalid opt<FieldDecl> flag {flag}")  # noqa: SLF001
    _read_tag(r, _TAG_FIELD)
    size = r.u32()
    F = r.f64()
    k = r.f64()
    Du = r.f64()
    Dv = r.f64()
    prog.field_decl = FieldDecl(size=size, F=F, k=k, Du=Du, Dv=Dv)


def _decode_morphogens(r: _Reader, prog: Program) -> None:
    for _ in range(r._count("morphogen_feedback")):  # noqa: SLF001
        _read_tag(r, _TAG_MORPHOGEN)
        gene = r.str_()
        channel = r.str_()
        gain = r.f64()
        prog.morphogen_feedback.append(
            MorphogenFeedback(gene=gene, channel=channel, gain=gain))


def _decode_config(r: _Reader, prog: Program) -> None:
    _read_tag(r, _TAG_CONFIG)
    cfg = prog.config
    cfg.ticks = r.u32()
    cfg.output = r.str_list()
    cfg.table = r.str_()
    cfg.ops_per_tick = r.u32()
    cfg.react_steps = r.u32()
    cfg.use_central_dogma = r.bool_()
    cfg.species = r.str_()
    cfg.backend = r.str_()
    cfg.sim = r.field_map()


def _decode_bio(r: _Reader, prog: Program) -> None:
    for _ in range(r._count("bio_instructions")):  # noqa: SLF001
        _read_tag(r, _TAG_BIO_INSTR)
        kind = r.str_()
        target = r.str_()
        params = r.field_map()
        line = r.u32()
        prog.bio_instructions.append(
            BioInstruction(kind=kind, target=target, params=params, line=line))


def _decode_types(r: _Reader, prog: Program) -> None:
    for _ in range(r._count("type_annotations")):  # noqa: SLF001
        k = r.str_()
        v = r.str_()
        prog.type_annotations[k] = v


def _decode_media(r: _Reader, prog: Program) -> None:
    for _ in range(r._count("media")):  # noqa: SLF001
        _read_tag(r, _TAG_MEDIA)
        nutrient = r.str_()
        concentration = r.f64()
        diffusion = r.opt_f64()
        prog.media.append(MediaDecl(nutrient=nutrient,
                                    concentration=concentration,
                                    diffusion_um2_s=diffusion))


def _decode_enzymes(r: _Reader, prog: Program) -> None:
    for _ in range(r._count("enzymes")):  # noqa: SLF001
        _read_tag(r, _TAG_ENZYME)
        gene = r.str_()
        reaction = r.str_()
        kcat = r.opt_f64()
        prog.enzymes.append(EnzymeDecl(gene=gene, reaction=reaction,
                                       kcat=kcat))


def _decode_pools(r: _Reader, prog: Program) -> None:
    for _ in range(r._count("pools")):  # noqa: SLF001
        _read_tag(r, _TAG_POOL)
        name = r.str_()
        init = r.f64()
        prog.pools.append(PoolDecl(name=name, init=init))


def _decode_ext(r: _Reader, prog: Program) -> None:
    for _ in range(r._count("sim_extensions")):  # noqa: SLF001
        k = r.str_()
        v = r.str_()
        prog.sim_extensions[k] = v


# ---------------------------------------------------------------------------
# CHNK section encoding
# ---------------------------------------------------------------------------
def _dna_by_index(program: Program) -> list[str]:
    by_index: dict[int, str] = {}
    for g in program.genes:
        for c in g.codons:
            by_index.setdefault(c.index, c.seq)
    return [by_index[i] for i in sorted(by_index)]


def _encode_chunk(chunk: Chunk, program: Program) -> bytes:
    w = _Writer()
    w.u32(len(chunk.code))
    w.buf += bytes(chunk.code)
    w.u16(len(chunk.constants))
    for c in chunk.constants:
        w.value(c)
    w.u32_list(chunk.lines)
    w.u32_list(chunk.codon_indices)
    offsets = sorted(chunk.gene_offsets.items())
    w.u16(len(offsets))
    for name, off in offsets:
        w.str_(name)
        w.u32(off)
    dna = _dna_by_index(program)
    if dna:
        w.u8(0x01)
        w.str_list(dna)
    else:
        w.u8(0x00)
    return bytes(w.buf)


def _decode_chunk(data: bytes) -> Chunk:
    r = _Reader(data, "CHNK")
    ncode = r.u32()
    if ncode > _MAX_COUNT:
        r._fail(f"chunk code length {ncode} exceeds limit")
    r.need(ncode)
    code = bytearray(r.data[r.pos:r.pos + ncode])
    r.pos += ncode
    constants = [r.value() for _ in range(r._count("constants"))]
    lines = r.u32_list()
    codon_indices = r.u32_list()
    offsets: dict[str, int] = {}
    for _ in range(r._count("gene_offsets")):
        name = r.str_()
        off = r.u32()
        offsets[name] = off
    has_dna = r.u8()
    if has_dna not in (0, 1):
        r._fail(f"invalid dna_sequence flag {has_dna}")
    if has_dna:
        r.str_list()  # validated but redundant with PROG; dropped
    if not r.at_end():
        r._fail(f"trailing {r.remaining()} bytes in CHNK section")  # noqa: SLF001
    codon_indices_signed = [x if x != 0xFFFFFFFF else _MISSING
                            for x in codon_indices]
    return Chunk(
        code=code,
        constants=constants,
        lines=lines,
        codon_indices=codon_indices_signed,
        gene_offsets=offsets,
    )


# ---------------------------------------------------------------------------
# Container assembly / parsing
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class LoadedArtifact:
    """Decoded .helixc artifact ready to run/disassemble/decompile."""

    program: Program
    chunk: Chunk | None = None
    source: str | None = None
    table: str = "standard"
    chunk_stale: bool = False


@dataclass(slots=True)
class ArtifactInfo:
    """Result of :func:`compile_file`."""

    path: Path
    program: Program
    chunk: Chunk | None
    table: str
    source: str | None


def dumps_program(program: Program, *, chunk: Chunk | None = None,
                  source: str | None = None) -> bytes:
    """Serialize a program into a .helixc byte string (deterministic)."""
    table = program.config.table
    table_id = _TABLE_IDS.get(table, 0)
    sections: list[tuple[bytes, bytes]] = [
        (SECTION_PROG, _encode_program(program)),
    ]
    flags = 0
    if chunk is not None:
        flags |= 0x01
        sections.append((SECTION_CHNK, _encode_chunk(chunk, program)))
    if source is not None:
        flags |= 0x02
        sections.append((SECTION_SRC, source.encode("utf-8")))

    pre_eof = bytearray()
    for magic, payload in sections:
        pre_eof += magic + struct.pack(">I", len(payload)) + payload
    # EOF section: 4 (magic) + 4 (u32 length) + 32 (SHA-256 digest)
    payload_len = len(pre_eof) + 40
    header = (MAGIC + bytes([FORMAT_VERSION, flags, table_id, 0])
              + struct.pack(">I", payload_len))
    digest = hashlib.sha256(pre_eof).digest()
    return bytes(header) + bytes(pre_eof) + SECTION_EOF + struct.pack(
        ">I", len(digest)) + digest


def save_program(program: Program, path: str | Path, *,
                 chunk: Chunk | None = None,
                 source: str | None = None) -> None:
    """Write a program to a .helixc file."""
    Path(path).write_bytes(dumps_program(program, chunk=chunk, source=source))


def loads_program(data: bytes) -> LoadedArtifact:
    """Decode and validate a .helixc byte string."""
    if len(data) < 12:
        raise BinaryFormatError(
            f"artifact too short ({len(data)} bytes < 12-byte header)")
    if data[:4] != MAGIC:
        raise BinaryFormatError(f"bad magic {data[:4]!r}; not a .helixc artifact")
    version = data[4]
    if version != FORMAT_VERSION:
        raise BinaryVersionError(
            f"artifact uses .helixc format version {version}; this build "
            f"supports {FORMAT_VERSION} - recompile with --compile")
    flags = data[5]
    if flags & 0xFC:
        raise BinaryFormatError(f"invalid header flags 0x{flags:02X}")
    table_id = data[6]
    if table_id not in _TABLE_NAMES:
        raise BinaryFormatError(f"invalid table id {table_id}")
    if data[7] != 0:
        raise BinaryFormatError("reserved header byte must be zero")
    payload_len = struct.unpack_from(">I", data, 8)[0]
    if len(data) != 12 + payload_len:
        raise BinaryFormatError(
            f"length mismatch: header claims {payload_len} payload bytes, "
            f"file has {len(data) - 12}")

    pos = 12
    end = len(data) - 40  # EOF section is the last 4+4+32 bytes
    if end < pos or data[end:end + 4] != SECTION_EOF:
        raise BinaryFormatError("missing EOF trailer section")
    digest = data[end + 8:end + 8 + 32]
    if len(digest) != 32:
        raise BinaryFormatError("invalid EOF digest length")

    expected = hashlib.sha256(data[12:end]).digest()
    if not _consttime_eq(digest, expected):
        raise BinaryFormatError("checksum mismatch: artifact is corrupted")

    sections: dict[bytes, bytes] = {}
    order: list[bytes] = []
    while pos < end:
        magic = data[pos:pos + 4]
        length = struct.unpack_from(">I", data, pos + 4)[0]
        if magic not in (SECTION_PROG, SECTION_CHNK, SECTION_SRC):
            raise BinaryFormatError(f"unknown section {magic!r}")
        if pos + 8 + length > end:
            raise BinaryFormatError(f"section {magic!r} overruns trailer")
        if magic in sections:
            raise BinaryFormatError(f"duplicate section {magic!r}")
        sections[magic] = data[pos + 8:pos + 8 + length]
        order.append(magic)
        pos += 8 + length

    expected_order = [SECTION_PROG]
    if flags & 0x01:
        expected_order.append(SECTION_CHNK)
    if flags & 0x02:
        expected_order.append(SECTION_SRC)
    if order != expected_order:
        raise BinaryFormatError(f"unexpected section order {order!r}")

    program = _decode_program(sections[SECTION_PROG])
    chunk: Chunk | None = None
    if flags & 0x01:
        chunk = _decode_chunk(sections[SECTION_CHNK])
    source: str | None = None
    if flags & 0x02:
        source = sections[SECTION_SRC].decode("utf-8")

    # CHNK is a derived cache: structurally consistent + consistent with PROG,
    # otherwise drop it so the caller recompiles from the authoritative AST.
    chunk_stale = False
    if chunk is not None:
        if not _chunk_consistent(chunk, program):
            chunk = None
            chunk_stale = True

    return LoadedArtifact(
        program=program,
        chunk=chunk,
        source=source,
        table=_TABLE_NAMES[table_id],
        chunk_stale=chunk_stale,
    )


def _consttime_eq(a: bytes, b: bytes) -> bool:
    """Constant-time comparison for the section digest."""
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b, strict=True):
        result |= x ^ y
    return result == 0


def _chunk_consistent(chunk: Chunk, program: Program) -> bool:
    """Structural + provenance sanity check for the cached chunk."""
    if len(chunk.code) != len(chunk.lines):
        return False
    if len(chunk.code) != len(chunk.codon_indices):
        return False
    gene_names = {g.name for g in program.genes}
    if set(chunk.gene_offsets) != gene_names:
        return False
    if chunk.code:
        if chunk.code[0] != 0x10:  # first byte must be OP_START of gene 0
            return False
    return True


def load_program(path: str | Path) -> LoadedArtifact:
    """Read, validate, and decode a .helixc file."""
    return loads_program(Path(path).read_bytes())


def verify(path: str | Path) -> None:
    """Re-decode a .helixc file; raises BinaryFormatError on corruption."""
    load_program(path)


# ---------------------------------------------------------------------------
# compile_file: .helix source -> .helixc artifact
# ---------------------------------------------------------------------------
def compile_file(src_path: str | Path, out_path: str | Path, *,
                 include_chunk: bool = True, include_source: bool = True,
                 table: str = "standard") -> ArtifactInfo:
    """Parse + semantically check + compile a .helix source into a .helixc."""
    from helixlang.compiler import Compiler
    from helixlang.lexer import Lexer
    from helixlang.parser import Parser
    from helixlang.semantic import SemanticAnalyzer
    from helixlang.seq_utils import stop_codons_from_table

    source = Path(src_path).read_text()
    tbl = get_table(table)
    stop_codons = stop_codons_from_table(tbl)
    tokens = list(Lexer(source).tokens())
    program = Parser(tokens, stop_codons=stop_codons).parse()
    SemanticAnalyzer(program).check()
    chunk: Chunk | None = None
    if include_chunk:
        chunk = Compiler(tbl).compile(program)
    emb_source = source if include_source else None
    save_program(program, Path(out_path), chunk=chunk, source=emb_source)
    return ArtifactInfo(
        path=Path(out_path),
        program=program,
        chunk=chunk,
        table=table,
        source=emb_source,
    )


# ---------------------------------------------------------------------------
# Decompiler
# ---------------------------------------------------------------------------
def _fmt_float(x: float) -> str:
    s = repr(x)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _fmt_str(s: str) -> str:
    if any(c in s for c in " \t=#"):  # noqa: SIM300
        return f'"{s}"'
    return s


def _fmt_fields(d: dict[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(d.items()))


def _fmt_codons(codons: list[Codon], per_line: int = 12) -> str:
    return "\n".join(
        " ".join(c.seq for c in codons[i:i + per_line])
        for i in range(0, len(codons), per_line)
    )


def _fmt_rules(rules: dict[int, dict[str, str]]) -> str:
    parts: list[str] = []
    for gen, table in rules.items():
        pairs = ",".join(f"{sym}->{prod}" for sym, prod in table.items())
        parts.append(f"{gen}:{pairs}")
    return ";".join(parts)


def decompile(program: Program) -> str:
    """Regenerate canonical .helix source from a Program (round-trip R1/R2)."""
    lines: list[str] = []
    for p in program.promoters:
        lines.append("#promoter " + _fmt_fields(p.fields))

    for g in program.genes:
        if g.fields or not g.name.startswith("__anon"):
            fields = dict(g.fields)
            if "name" not in fields:
                fields = {"name": g.name, **fields}
            if "promoter" not in fields and g.promoter is not None:
                fields["promoter"] = g.promoter
            lines.append("#gene " + _fmt_fields(fields))
            lines.append(_fmt_codons(g.codons))
            lines.append("#end")
        else:
            lines.append(_fmt_codons(g.codons))

    for reg in program.regulations:
        lines.append(f"#regulate {reg.source} -> {reg.target} "
                     f"strength={_fmt_float(reg.strength)}")

    for name, decl in program.lsystems.items():
        lines.append(
            f"#lsystem name={name} axiom={_fmt_str(decl.axiom)} "
            f"rules={_fmt_str(_fmt_rules(decl.rules))} "
            f"angle={_fmt_float(decl.angle)} step={_fmt_float(decl.step)}")

    if program.field_decl is not None:
        f = program.field_decl
        lines.append(
            f"#field size={f.size} F={_fmt_float(f.F)} k={_fmt_float(f.k)} "
            f"Du={_fmt_float(f.Du)} Dv={_fmt_float(f.Dv)}")

    for fb in program.morphogen_feedback:
        lines.append(f"#morphogen gene={fb.gene} channel={fb.channel} "
                     f"gain={_fmt_float(fb.gain)}")

    for md in program.media:
        s = f"#media nutrient={md.nutrient} concentration={_fmt_float(md.concentration)}"
        if md.diffusion_um2_s is not None:
            s += f" diffusion_um2_s={_fmt_float(md.diffusion_um2_s)}"
        lines.append(s)

    for enz in program.enzymes:
        s = f"#enzyme gene={enz.gene} reaction={enz.reaction}"
        if enz.kcat is not None:
            s += f" kcat={_fmt_float(enz.kcat)}"
        lines.append(s)

    for pl in program.pools:
        lines.append(f"#metabolite name={pl.name} init={_fmt_float(pl.init)}")

    for name in sorted(program.type_annotations):
        lines.append(f"#type {name}={program.type_annotations[name]}")

    for inst in program.bio_instructions:
        lines.append("#" + inst.kind + " " + _fmt_fields(inst.params))

    cfg = program.config
    parts = [
        f"ticks={cfg.ticks}",
        f"output={','.join(cfg.output)}",
        f"table={cfg.table}",
        f"ops_per_tick={cfg.ops_per_tick}",
        f"react_steps={cfg.react_steps}",
        f"use_central_dogma={'true' if cfg.use_central_dogma else 'false'}",
        f"species={cfg.species}",
        f"backend={cfg.backend}",
    ]
    for k in sorted(cfg.sim):
        parts.append(f"{k}={cfg.sim[k]}")
    lines.append("#config " + " ".join(parts))

    for k in sorted(program.sim_extensions):
        lines.append(f"#sim {k}={program.sim_extensions[k]}")

    return "\n".join(lines) + "\n"


def decompile_to_file(program: Program, path: str | Path) -> None:
    """Write decompiled source to a file."""
    Path(path).write_text(decompile(program))


def _check_table_names() -> None:
    """Guard: the header table-id mapping must match codon_table.TABLES."""
    for name in _TABLE_IDS:
        if name not in TABLES:
            raise BinaryFormatError(f"table {name!r} not in codon_table.TABLES")


_check_table_names()
