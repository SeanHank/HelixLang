"""DNA physical codec module: converts arbitrary binary <-> synthetic
DNA sequences.

Implements two real schemes (coexisting with the existing virtual
opcode/codon mapping):

1. **Goldman 2013 rotating-key encoding** - Nature 494:77-80
   - bytes -> base-3 trits using the paper's published Huffman code
     (256 codewords of 5-6 trits, average 5.07 trits/byte; the table
     is taken verbatim from the corrected specification file
     `View_huff3.cd.new.correct` at
     https://www.ebi.ac.uk/goldman-srv/DNA-storage/orig_files/)
   - the message ends with the published EXTRA symbol (codeword
     `222020`) and the trit stream is zero-padded to a multiple of 25
   - each trit -> 1 nt, looked up in the rotating-key table as
     "previous base + trit" (mathematically guarantees no
     homopolymers)
   - 100 nt data segments, 25 nt step, 75 nt overlap -> 4x redundancy
   - 17 nt index header (base-3 rotating-key-encoded segment number +
     parity check)
   - adjacent segments alternate forward / reverse-complement
   - measured capacity ~0.28 bit/nt (incl. index overhead; the paper
     measures 0.29 bit/nt)

2. **Erlich-Zielinski 2017 DNA Fountain** - Science 355:950-954
   - LT fountain code (Luby Transform) splits the data into 32-byte
     source blocks
   - Robust Soliton Distribution samples the degree d, randomly selects
     d source blocks and XORs them
   - 4B seed (LFSR-style) + 32B payload + 2B Reed-Solomon = 38B ->
     152 nt
   - 2-bit DNA mapping (A=00, C=01, G=10, T=11)
   - GC 45-55% + homopolymers <=3, rejection sampling
   - BP / peeling decoder
   - measured capacity 1.57 bit/nt (close to 99% of the Shannon limit
     1.58 bit/nt)

PCR error model (Saiki 1988 / Potapov 2017):
   - Taq: substitution 1.5e-4/nt/cycle, indels 1-3% of total errors
   - Transitions : Transversions ~ 6:1 (keto-enol tautomerism)
   - 30 cycles accumulate ~0.45% substitution
   - Pfu/Q5/Phusion high-fidelity modes optional

Dependencies:
   - biopython: IUPAC strict validation + transcription/translation
   - reedsolo: Reed-Solomon error correction code

References:
   - Goldman et al. Nature 2013 494:77-80 (PMC3672958)
   - Erlich & Zielinski Science 2017 355:950-954
   - Saiki et al. Science 1988 239:487-491
   - Potapov & Ong PLoS ONE 2017 12(1):e0169774
   - Tindall & Kunkel Biochemistry 1988 27:6008-6013
"""
from __future__ import annotations

import math
import random
import struct
from collections.abc import Iterable
from dataclasses import dataclass

# BioPython optional (degrades to plain alphabet validation if missing)
try:
    from Bio.Seq import Seq
    _HAS_BIOPYTHON = True
except ImportError:
    _HAS_BIOPYTHON = False

class ReedSolomonError(Exception):
    """Placeholder exception type when reedsolo is missing (keeps the
    except clauses usable)."""
    pass


try:
    from reedsolo import ReedSolomonError as _RealReedSolomonError
    from reedsolo import RSCodec
    _HAS_REEDSOLO = True
except ImportError:
    _HAS_REEDSOLO = False
    _RealReedSolomonError = ReedSolomonError

# catch the real ReedSolomonError when reedsolo is available, fall back
# to the placeholder class when it is missing (erlich_* first raises a
# RuntimeError when reedsolo is missing, so the except clauses are not
# reached)
_RS_ERROR_TYPES: tuple[type[BaseException], ...] = (
    ValueError, IndexError, _RealReedSolomonError,
)

from helixlang.seq_utils import gc_content as _gc_content  # noqa: E402
from helixlang.seq_utils import reverse_complement as _reverse_complement  # noqa: E402

# ============================================================================
# Goldman 2013 rotating-key encoding
# ============================================================================
# Core innovation of the paper: each trit maps to 1 nt, and the mapping
# depends on the "previous base", choosing from the 3 bases other than
# it in a fixed cyclic order, which **mathematically guarantees no
# homopolymers** (>=2 identical bases).
#
# Rotating-key table (Wikipedia "DNA digital data storage", cross-
# checked against the Goldman 2013 paper):
#   cyclic order A->C->G->T->A. Given previous base P, the 3
#   candidates are the 3 bases after P (in cyclic order).
#
#   prev=T (idx 3): trit 0->A, 1->C, 2->G   (indices 0,1,2)
#   prev=G (idx 2): trit 0->T, 1->A, 2->C   (indices 3,0,1)
#   prev=C (idx 1): trit 0->G, 1->T, 2->A   (indices 2,3,0)
#   prev=A (idx 0): trit 0->C, 1->G, 2->T   (indices 1,2,3)
#
# i.e.: next_idx = (prev_idx + 1 + trit) % 4

_BASES = "ACGT"
_BASE_IDX = {b: i for i, b in enumerate(_BASES)}  # STATE: global (immutable lookup table)

# ============================================================================
# Goldman 2013 base-3 Huffman code (verbatim from the paper's corrected
# specification file)
# ============================================================================
# Source: View_huff3.cd.new.correct, published by the authors at
# https://www.ebi.ac.uk/goldman-srv/DNA-storage/orig_files/
# (Nature 494:77-80, 2013, doi:10.1038/nature11875).
# Each of the 256 byte values maps to a prefix-free codeword of 5 or 6
# trits; the average codeword length is 5.07 trits/byte (the paper
# quotes ~5.05; the exact average depends on the byte distribution).
# The code was validated by re-decoding the author-supplied encoded
# DNA (View_huff3.cd.new.dna) back to the specification file, which
# matches byte-for-byte up to the two known corruption errors in the
# published data.
_GOLDMAN_HUFFMAN_CODE: dict[int, str] = {
    0: '22201', 1: '00100', 2: '11220', 3: '00211', 4: '20222',
    5: '00222', 6: '02211', 7: '222110', 8: '22002', 9: '02100',
    10: '22001', 11: '222122', 12: '12001', 13: '02021', 14: '10100',
    15: '02010', 16: '20101', 17: '12211', 18: '12120', 19: '11111',
    20: '21211', 21: '21221', 22: '20220', 23: '00122', 24: '20022',
    25: '12121', 26: '21111', 27: '00221', 28: '00202', 29: '222202',
    30: '222102', 31: '00010', 32: '02212', 33: '10011', 34: '22011',
    35: '02221', 36: '21212', 37: '21021', 38: '11211', 39: '10111',
    40: '12220', 41: '22110', 42: '22101', 43: '11122', 44: '22022',
    45: '01210', 46: '00210', 47: '02122', 48: '10122', 49: '01011',
    50: '11101', 51: '01102', 52: '22112', 53: '12122', 54: '11012',
    55: '222112', 56: '02201', 57: '02011', 58: '20021', 59: '222021',
    60: '00022', 61: '222200', 62: '222120', 63: '21010', 64: '00121',
    65: '02022', 66: '20100', 67: '10211', 68: '21001', 69: '21210',
    70: '10212', 71: '222212', 72: '20110', 73: '20010', 74: '21220',
    75: '21022', 76: '21000', 77: '01211', 78: '10220', 79: '12002',
    80: '12011', 81: '11212', 82: '21100', 83: '12210', 84: '20112',
    85: '22200', 86: '22102', 87: '21222', 88: '21012', 89: '12101',
    90: '10120', 91: '01202', 92: '10200', 93: '02210', 94: '222211',
    95: '11201', 96: '00102', 97: '01112', 98: '22010', 99: '00012',
    100: '22100', 101: '20001', 102: '20202', 103: '02102', 104: '20200',
    105: '20210', 106: '20012', 107: '11100', 108: '02101', 109: '11021',
    110: '00021', 111: '02110', 112: '12102', 113: '01012', 114: '10101',
    115: '10222', 116: '10221', 117: '10002', 118: '01120', 119: '00201',
    120: '10020', 121: '222111', 122: '222220', 123: '02111', 124: '222222',
    125: '00000', 126: '10112', 127: '22121', 128: '02000', 129: '10000',
    130: '20111', 131: '00212', 132: '22021', 133: '21112', 134: '11022',
    135: '01220', 136: '11102', 137: '20011', 138: '22111', 139: '10021',
    140: '12212', 141: '11202', 142: '10201', 143: '02200', 144: '02002',
    145: '11120', 146: '20102', 147: '11110', 148: '11002', 149: '22000',
    150: '21002', 151: '21102', 152: '222221', 153: '11020', 154: '20221',
    155: '01002', 156: '11001', 157: '00120', 158: '02202', 159: '10202',
    160: '10012', 161: '22012', 162: '20211', 163: '21201', 164: '00220',
    165: '11222', 166: '21011', 167: '10110', 168: '20002', 169: '20122',
    170: '22122', 171: '20201', 172: '10022', 173: '21101', 174: '12110',
    175: '12222', 176: '00200', 177: '21202', 178: '10210', 179: '10010',
    180: '02012', 181: '12221', 182: '12022', 183: '02222', 184: '01100',
    185: '02121', 186: '01122', 187: '00112', 188: '01020', 189: '222100',
    190: '01222', 191: '21020', 192: '01201', 193: '00001', 194: '12021',
    195: '12010', 196: '20121', 197: '21120', 198: '00002', 199: '222201',
    200: '00011', 201: '01010', 202: '12112', 203: '11112', 204: '02120',
    205: '11010', 206: '01110', 207: '01212', 208: '20120', 209: '12000',
    210: '12100', 211: '11210', 212: '11011', 213: '21200', 214: '12200',
    215: '01111', 216: '01200', 217: '12012', 218: '10121', 219: '10102',
    220: '222210', 221: '00020', 222: '01000', 223: '20020', 224: '11121',
    225: '10001', 226: '02001', 227: '01101', 228: '222121', 229: '21121',
    230: '02220', 231: '01001', 232: '222101', 233: '01022', 234: '20212',
    235: '00101', 236: '222022', 237: '01021', 238: '00111', 239: '11200',
    240: '12201', 241: '11000', 242: '02112', 243: '01221', 244: '00110',
    245: '11221', 246: '01121', 247: '12111', 248: '12020', 249: '02020',
    250: '22020', 251: '20000', 252: '21110', 253: '22120', 254: '12202',
    255: '21122',
}

# End-of-message symbol (the 257th Huffman leaf in the published tree;
# never emitted for real bytes, marks the boundary between the message
# and the zero padding).
_GOLDMAN_EXTRA_CODE = "222020"


def _build_goldman_trie() -> dict:
    """Prefix-free trie over the Goldman Huffman codewords."""
    trie: dict = {}
    for _byte_val, code in _GOLDMAN_HUFFMAN_CODE.items():
        node = trie
        for t in code:
            node = node.setdefault(t, {})
        node["#"] = _byte_val
    # the EXTRA leaf terminates decoding
    node = trie
    for t in _GOLDMAN_EXTRA_CODE:
        node = node.setdefault(t, {})
    node["#"] = "EXTRA"
    return trie


_GOLDMAN_TRIE = _build_goldman_trie()


def _trit_to_base(prev: str, trit: int) -> str:
    """Rotating key: previous base + trit -> next base. Guaranteed !=
    prev."""
    return _BASES[(_BASE_IDX[prev] + 1 + trit) % 4]


def _base_to_trit(prev: str, curr: str) -> int:
    """Reverse: previous base + current base -> trit (0/1/2)."""
    diff = (_BASE_IDX[curr] - _BASE_IDX[prev] - 1) % 4
    return diff


# byte -> 5/6 trits (Goldman Huffman code, average 5.07 trits/byte)
def _byte_to_trits(b: int) -> str:
    """Byte -> Goldman Huffman codeword (5-6 trits)."""
    try:
        return _GOLDMAN_HUFFMAN_CODE[b]
    except KeyError:
        raise ValueError(f"byte {b} has no Goldman Huffman codeword") from None


def _trits_to_byte(trits: str) -> int:
    """Greedy Huffman decode of the next codeword -> byte.

    Returns the byte value, or -1 for the EXTRA end-of-message symbol,
    or raises ValueError if `trits` does not begin with a complete,
    valid codeword.
    """
    node: dict = _GOLDMAN_TRIE
    for i, t in enumerate(trits):
        child = node.get(t)
        if child is None:
            raise ValueError(f"invalid Goldman trit {t!r} at position {i}")
        node = child
        val = node.get("#")
        if val is not None:
            if val == "EXTRA":
                return -1
            return int(val)
    raise ValueError("incomplete Goldman Huffman codeword")


def _bytes_to_dna_goldman(data: bytes, start_base: str = "A") -> str:
    """Byte stream -> DNA (Huffman trits + rotating key, no
    homopolymers).

    The byte stream is encoded with the published Goldman Huffman
    code, terminated by the EXTRA symbol, and the trit stream is
    zero-padded to a multiple of 25 before rotating-key conversion
    (mirroring the author-supplied intermediate DNA files).

    start_base: the "previous base" context of the first base (default
    A, matching the author-supplied DNA files).
    """
    trits = "".join(_GOLDMAN_HUFFMAN_CODE[b] for b in data)
    trits += _GOLDMAN_EXTRA_CODE
    pad = (25 - len(trits) % 25) % 25
    trits += "0" * pad
    prev = start_base
    out = []
    for t in trits:
        nb = _trit_to_base(prev, int(t))
        out.append(nb)
        prev = nb
    return "".join(out)


def _dna_to_bytes_goldman(dna: str, start_base: str = "A") -> bytes:
    """DNA -> byte stream (reverse rotating key + Huffman decode).

    Decoding stops at the EXTRA end-of-message symbol; trailing
    padding is discarded. Fault tolerance: if a trit sequence does not
    form a valid codeword (e.g. after PCR/sequencing errors), the
    offending trit is skipped and decoding re-synchronises at the next
    trit.
    """
    prev = start_base
    out = bytearray()
    window: list[str] = []
    for curr in dna:
        window.append(str(_base_to_trit(prev, curr)))
        prev = curr
        # try to close a codeword as soon as one is complete
        while True:
            try:
                val = _trits_to_byte("".join(window))
            except ValueError:
                break
            if val == -1:
                return bytes(out)
            out.append(val)
            window.clear()
        # drop the first trit of a window that cannot close (re-sync)
        if len(window) > 6:
            window.pop(0)
    return bytes(out)


# index encoding: 17 nt carries the segment number + parity check
# first 13 trits encode the segment number (3^13 = 1,594,323, plenty),
# last 4 trits are the checksum
INDEX_NT = 17
INDEX_PAYLOAD_TRITS = 13
INDEX_PARITY_TRITS = 4
SEGMENT_NT = 100          # data segment length (Goldman standard)
SEGMENT_STEP_NT = 25      # step (= 4x redundancy)


def _encode_index(seg_idx: int, length: int = INDEX_NT,
                  start_base: str = "T") -> str:
    """Segment number -> 17 nt index header (rotating key + weighted
    checksum).

    The checksum is a position-weighted checksum:
    sum(trit_i * (i+1)) mod 3^PARITY, which ensures any single-trit
    change alters the checksum (covering all 13 payload trits).
    """
    # segment number -> 13 trits (most significant first)
    idx_trits = []
    v = seg_idx
    for _ in range(INDEX_PAYLOAD_TRITS):
        idx_trits.append(v % 3)
        v //= 3
    idx_trits.reverse()
    # weighted checksum: sum(trit_i * (i+1)) mod 3^PARITY_TRITS
    # any single-trit change alters the checksum -> detects single-base
    # errors
    checksum = sum(t * (i + 1) for i, t in enumerate(idx_trits))
    checksum %= (3 ** INDEX_PARITY_TRITS)
    par_trits = []
    p = checksum
    for _ in range(INDEX_PARITY_TRITS):
        par_trits.append(p % 3)
        p //= 3
    par_trits.reverse()
    all_trits = idx_trits + par_trits
    # rotating-key encoding
    prev = start_base
    out = []
    for t in all_trits:
        nb = _trit_to_base(prev, t)
        out.append(nb)
        prev = nb
    # pad up to length (if any remaining)
    while len(out) < length:
        out.append(_trit_to_base(prev, 0))
        prev = out[-1]
    return "".join(out[:length])


def _decode_index(dna: str, start_base: str = "T") -> int:
    """17 nt index header -> segment number (with weighted checksum
    validation)."""
    prev = start_base
    trits = []
    for curr in dna[:INDEX_PAYLOAD_TRITS + INDEX_PARITY_TRITS]:
        trits.append(_base_to_trit(prev, curr))
        prev = curr
    # segment number
    seg_idx = 0
    for t in trits[:INDEX_PAYLOAD_TRITS]:
        seg_idx = seg_idx * 3 + t
    # weighted checksum validation
    expected = sum(t * (i + 1) for i, t in enumerate(trits[:INDEX_PAYLOAD_TRITS]))
    expected %= (3 ** INDEX_PARITY_TRITS)
    parity = 0
    for t in trits[INDEX_PAYLOAD_TRITS:]:
        parity = parity * 3 + t
    if parity != expected:
        raise ValueError(
            f"index checksum mismatch: got {parity}, expected {expected} "
            f"(seg_idx={seg_idx})"
        )
    return seg_idx


@dataclass(slots=True)
class GoldmanOligo:
    """A Goldman oligo: index + payload + overhang (index)."""
    index: int            # segment number
    payload: str          # 100 nt data segment
    overhang: str         # 17 nt index header
    full: str             # overhang + payload (the actually-synthesized
                          # sequence, possibly RC)


def _align_payload(payload: str, consensus: str, nominal: int) -> list[int | None]:
    """Per-base consensus-column mapping of `payload` (indel-tolerant).

    Banded Smith-Waterman local alignment of `payload` against a
    window of `consensus` around the nominal offset (`index * 25`).
    Returns a list parallel to `payload` giving the consensus index
    each base maps to (None for payload bases that are insertions
    relative to the consensus, i.e. that align to a gap). This
    tolerates the insertions/deletions introduced by synthesis, PCR and
    sequencing, which would otherwise shift a position-based majority
    vote. Falls back to the nominal mapping when no confident alignment
    is found.
    """
    lo = max(0, nominal - SEGMENT_NT)
    hi = min(len(consensus), nominal + SEGMENT_NT)
    if hi <= lo:
        # no overlap with the consensus: append at its end
        return list(range(len(consensus), len(consensus) + len(payload)))
    window = consensus[lo:hi]
    length, mlen = len(payload), len(window)
    nominal_map: list[int | None] = list(range(nominal, nominal + length))
    if length == 0:
        return []
    # fast path: if the payload matches the consensus at its nominal
    # offset (allowing <= 1 substitution and no indels), no alignment
    # is needed. A real indel shifts every downstream base, so a clean
    # match implies the nominal offset is correct.
    overlap = min(hi - nominal, length)
    if overlap >= 60:
        mismatches = 0
        for k in range(overlap):
            if payload[k] != consensus[nominal + k]:
                mismatches += 1
                if mismatches > 1:
                    break
        if mismatches <= 1:
            return nominal_map
    # Semi-global alignment: every payload base must map onto a
    # consensus column (indels are placed as gaps), but the consensus
    # window may extend beyond the payload's true coverage in both
    # directions (the window is padded only with real consensus, never
    # with unknown 'N' columns). Padding the reference with 'N' would
    # let the aligner silently absorb an internal deletion as a
    # trailing truncation; requiring the full payload to be consumed
    # forces the deletion to be placed where the payload actually
    # disagrees with the consensus.
    mapping: list[int | None] = [None] * length
    match, mismatch, gap = 2, -3, -4
    band = 25
    diag = nominal - lo  # expected j - i for the true alignment
    score = [[0] * (mlen + 1) for _ in range(length + 1)]
    tb = [[0] * (mlen + 1) for _ in range(length + 1)]
    for i in range(1, length + 1):
        jmin = max(1, i + diag - band)
        jmax = min(mlen, i + diag + band)
        for j in range(jmin, jmax + 1):
            a, b = payload[i - 1], window[j - 1]
            s_diag = score[i - 1][j - 1] + (match if a == b else mismatch)
            s_up = score[i - 1][j] + gap      # gap in the window
            s_left = score[i][j - 1] + gap    # gap in the payload
            s = max(s_diag, s_up, s_left)
            score[i][j] = s
            if s == s_diag:
                tb[i][j] = 1
            elif s == s_up:
                tb[i][j] = 2
            else:
                tb[i][j] = 3
    # best end: payload fully consumed (row `length`), any column
    best = -10**9
    best_j = mlen
    for j in range(mlen + 1):
        if score[length][j] > best:
            best = score[length][j]
            best_j = j
    if best < match * 30:  # not a confident alignment
        return nominal_map
    i, j = length, best_j
    while i > 0 and j > 0:
        if tb[i][j] == 1:   # diagonal: payload[i-1] <-> window[j-1]
            mapping[i - 1] = lo + (j - 1)
            i -= 1
            j -= 1
        elif tb[i][j] == 2:  # payload[i-1] aligns to a gap in the window
            i -= 1
        else:                # window[j-1] aligns to a gap in the payload
            j -= 1
    if i > 0:
        # payload longer than the window's coverage: append the
        # remainder after the last mapped column
        next_col = lo + j
        for k in range(i, 0, -1):
            mapping[length - k] = next_col
            next_col += 1
    return mapping


def _columns_to_dna(columns: list[dict[str, int]]) -> str:
    """Consensus sequence from per-column base votes.

    Maximizes the total vote count subject to the Goldman rotating-key
    constraint (base[i] != base[i-1] along the whole stream). The
    constraint is a hard one - a homopolymer pair is invalid DNA for this
    codec - so a column's most-voted base is only chosen when it is
    compatible with the column before it. A column with no votes lets the
    state machine continue the key without penalty. A dynamic program over
    the 4 states (one per base) makes the choice globally optimal and
    immune to the error cascade a greedy left-to-right tie break suffers.
    """
    if not columns:
        return ""
    bases = ("A", "C", "G", "T")
    prev = [columns[0].get(b, 0) for b in bases]
    parent = []
    for col in columns[1:]:
        votes = [col.get(b, 0) for b in bases]
        cur: list[int] = []
        par: list[int] = []
        for x in range(4):
            best_val, best_y = max(
                (prev[y], y) for y in range(4) if y != x)
            cur.append(best_val + votes[x])
            par.append(best_y)
        parent.append(par)
        prev = cur
    last = max(range(4), key=lambda x: (prev[x], -x))
    out = [bases[last]]
    for c in range(len(columns) - 1, 0, -1):
        last = parent[c - 1][last]
        out.append(bases[last])
    out.reverse()
    return "".join(out)


def _merge_maps(seg_payloads: dict[int, str],
                maps: dict[int, list[int | None]]) -> list[dict[str, int]]:
    """Column-wise base counts using per-base consensus mappings."""
    columns: list[dict[str, int]] = []
    for seg_idx, payload in seg_payloads.items():
        for j, base in enumerate(payload):
            col = maps[seg_idx][j]
            if col is None or col < 0:
                continue
            while len(columns) <= col:
                columns.append({})
            columns[col][base] = columns[col].get(base, 0) + 1
    return columns


def _assemble_dna(seg_payloads: dict[int, str],
                  total_len: int | None = None) -> str:
    """Reassemble the long DNA stream from the overlapping segments.

    Segments are 100 nt payloads with a 25 nt step (75 nt overlap,
    4x redundancy). A naive per-position majority vote is fragile to
    insertions/deletions (one indel shifts every downstream vote), so
    the sequence is rebuilt the way the paper's decoder does it: by
    overlap assembly, i.e. aligning each segment (indel-tolerant
    Smith-Waterman) against the consensus assembled so far and merging
    it column-wise, followed by two refinement passes that re-align
    every segment against the full consensus.

    Returns the reconstructed DNA through the end of the last segment
    (or min(total_len * 6, ...) when total_len is given). The EXTRA
    end-of-message codeword truncates it during byte decoding.
    """
    if not seg_payloads:
        return ""
    # Nominal placement: each segment i covers stream columns
    # [25*i, 25*i + len(payload)) - the Goldman scheme encodes the
    # whole message as one continuous stream and every segment is a
    # 100 nt window of it at a 25 nt step, so the offsets are known a
    # priori. Indels shift a segment's *own* downstream votes, but the
    # 4x overlap means each column is voted by ~4 segments, and a
    # segment is only trusted where the vote is consistent.
    columns: list[dict[str, int]] = []
    for seg_idx, payload in seg_payloads.items():
        start = seg_idx * SEGMENT_STEP_NT
        for j, base in enumerate(payload):
            col = start + j
            while len(columns) <= col:
                columns.append({})
            columns[col][base] = columns[col].get(base, 0) + 1
    consensus = _columns_to_dna(columns)
    # refinement: re-align every segment against the global consensus
    # (semi-global, so indels are placed as internal gaps) and re-vote.
    # This relocates a segment's shifted votes to their true columns and
    # is what lets the vote win even where an indel made a majority
    # disagree with the true base. The rotating-key dynamic program in
    # _columns_to_dna keeps every refinement step globally consistent.
    for _ in range(2):
        maps = {
            i: _align_payload(p, consensus, i * SEGMENT_STEP_NT)
            for i, p in seg_payloads.items()
        }
        columns = _merge_maps(seg_payloads, maps)
        refined = _columns_to_dna(columns)
        if refined == consensus:
            break
        consensus = refined
    if total_len is not None:
        columns = columns[:total_len * 6]
    return _columns_to_dna(columns)


def goldman_encode(data: bytes) -> list[GoldmanOligo]:
    """Goldman 2013 encoding: binary -> list of DNA oligos.

    Steps (paper Methods + Fig.1):
    1. byte stream -> Huffman-coded trits + EXTRA marker, padded to a
       multiple of 25, then -> DNA (rotating key, no homopolymers)
    2. split DNA into 100 nt segments with a 25 nt step (4x overlapping
       redundancy)
    3. add a 17 nt index header to each segment (segment number + parity
       check)
    4. adjacent segments alternate reverse complement
    Total oligo length = 117 nt (informational part)
    """
    # 1. bytes -> DNA (Huffman + EXTRA + pad to multiple of 25)
    dna = _bytes_to_dna_goldman(data)
    # 2. segment (25 nt step, 100 nt segments)
    # pad the last segment to a full 100 nt using rotating-key
    # continuation (trit=0 cycle), which guarantees no homopolymers and
    # ~50% GC. Because dna length is a multiple of 25, this adds
    # exactly 75 nt to complete the final segment.
    n_segments = max(1, math.ceil(len(dna) / SEGMENT_STEP_NT))
    min_len = (n_segments - 1) * SEGMENT_STEP_NT + SEGMENT_NT
    pad_total = max(0, min_len - len(dna))
    if pad_total > 0:
        last_base = dna[-1] if dna else "T"
        pad_bases = []
        prev = last_base
        for _ in range(pad_total):
            nb = _trit_to_base(prev, 0)  # trit=0 continuation, cycles
                                         # A->C->G->T
            pad_bases.append(nb)
            prev = nb
        dna_padded = dna + "".join(pad_bases)
    else:
        dna_padded = dna

    oligos: list[GoldmanOligo] = []
    for seg_idx in range(n_segments):
        start = seg_idx * SEGMENT_STEP_NT
        payload = dna_padded[start:start + SEGMENT_NT]
        # 3. index header
        index_dna = _encode_index(seg_idx)
        # 4. alternating reverse complement (odd segments RC the whole
        #    oligo)
        full = index_dna + payload
        if seg_idx % 2 == 1:
            full = _reverse_complement(full)
        oligos.append(GoldmanOligo(
            index=seg_idx, payload=payload, overhang=index_dna, full=full
        ))
    return oligos


def goldman_decode(oligos: Iterable[GoldmanOligo],
                   total_len: int | None = None) -> bytes:
    """Goldman decoding: 4x overlapping vote restores the original
    bytes.

    Steps:
    1. each oligo: detect orientation (forward/RC), decode the index,
       extract the payload
    2. group by segment number
    3. restore the long DNA string by 4x overlapping vote (each
       position is covered by up to 4 segments)
    4. DNA -> bytes
    """
    # 1. collect each segment's payload
    # The encoder emits exactly one oligo per segment (indices
    # 0..n-1), so a decoded segment number outside that range cannot be
    # part of the message. This rejects the rare spurious checksum
    # passes (~1/81 chance per 17 nt index) that would otherwise
    # produce a bogus reconstruction length.
    oligo_list = list(oligos)
    n_segments = len(oligo_list)
    seg_payloads: dict[int, str] = {}
    for oligo in oligo_list:
        full = oligo.full
        # try decoding the index in forward orientation
        seg_idx = None
        payload = None
        try:
            idx = _decode_index(full[:INDEX_NT])
            # verify: even segments should be forward
            if idx % 2 == 0 and 0 <= idx < n_segments:
                seg_idx = idx
                payload = full[INDEX_NT:INDEX_NT + SEGMENT_NT]
        except (ValueError, KeyError):
            pass
        if seg_idx is None:
            # try RC
            try:
                rc = _reverse_complement(full)
                idx = _decode_index(rc[:INDEX_NT])
                if idx % 2 == 1 and 0 <= idx < n_segments:
                    seg_idx = idx
                    payload = rc[INDEX_NT:INDEX_NT + SEGMENT_NT]
            except (ValueError, KeyError):
                pass
        if seg_idx is None:
            # fall back to oligo.index + oligo.payload directly
            seg_idx = oligo.index
            payload = oligo.payload
        if payload:
            # vote among multiple copies: take the longest (or the
            # first)
            if seg_idx not in seg_payloads or len(payload) > len(seg_payloads[seg_idx]):
                seg_payloads[seg_idx] = payload

    if not seg_payloads:
        raise ValueError("no valid oligos decoded")

    # 3. restore the long DNA string by overlap-based reassembly
    #    (indel-tolerant; see _assemble_dna)
    dna = _assemble_dna(seg_payloads, total_len=total_len)

    # 4. DNA -> bytes
    data = _dna_to_bytes_goldman(dna)
    if total_len is not None:
        data = data[:total_len]
    return data


# ============================================================================
# Erlich-Zielinski 2017 DNA Fountain code
# ============================================================================
# Paper core: LT fountain code + RS inner code + 2-bit DNA mapping +
# constraint rejection sampling
#
# Architecture:
#   source data -> K source blocks of 32 bytes
#   each oligo = LT encoding (seed determines degree d and adjacent
#                source blocks; XOR yields the 32B payload)
#              + 4B seed header + 2B RS checksum = 38B
#              -> 2-bit DNA mapping -> 76 nt? no, 38*8/2 = 152 nt
#              -> GC/homopolymer constraint filtering (rejection
#                 sampling)

ERLICH_OLIGO_SIZE = 32       # source block size (bytes)
ERLICH_HEADER_SIZE = 4       # seed header (bytes)
ERLICH_RS_SIZE = 2           # RS inner-code checksum (bytes)
ERLICH_OLIGO_NT = 152        # 38B * 8 / 2 = 152 nt
ERLICH_GC_DEV = 0.05         # GC in [0.45, 0.55]
ERLICH_MAX_HOMOPOLYMER = 3   # >=4 consecutive identical bases not allowed

# 2-bit DNA mapping (Erlich 2017 utils.pyx)
_DNA_BIN = {"A": "00", "C": "01", "G": "10", "T": "11"}  # STATE: global (immutable lookup table)
_BIN_DNA = {v: k for k, v in _DNA_BIN.items()}  # STATE: global (immutable lookup table)


def _bytes_to_dna_2bit(data: bytes) -> str:
    """2-bit binary -> DNA (A=00, C=01, G=10, T=11)."""
    bits = "".join(f"{b:08b}" for b in data)
    if len(bits) % 2:
        bits += "0"
    return "".join(_BIN_DNA[bits[i:i + 2]] for i in range(0, len(bits), 2))


def _dna_to_bytes_2bit(dna: str) -> bytes:
    """DNA -> 2-bit binary -> bytes."""
    bits = "".join(_DNA_BIN[c] for c in dna if c in _DNA_BIN)
    n = (len(bits) // 8) * 8
    bits = bits[:n]
    return bytes(int(bits[i:i + 8], 2) for i in range(0, n, 8))


def _has_homopolymer(dna: str, max_run: int = ERLICH_MAX_HOMOPOLYMER) -> bool:
    """Check whether there is a homopolymer longer than max_run."""
    run = 1
    for i in range(1, len(dna)):
        if dna[i] == dna[i - 1]:
            run += 1
            if run > max_run:
                return True
        else:
            run = 1
    return False


def _satisfies_constraints(dna: str, gc_dev: float = ERLICH_GC_DEV,
                           max_homopolymer: int = ERLICH_MAX_HOMOPOLYMER) -> bool:
    """Check the GC and homopolymer constraints (Erlich 2017 defaults)."""
    gc = _gc_content(dna)
    if not (0.5 - gc_dev <= gc <= 0.5 + gc_dev):
        return False
    if _has_homopolymer(dna, max_homopolymer):
        return False
    return True


def robust_soliton_distribution(K: int, delta: float = 0.001,
                                c: float = 0.025) -> list[float]:
    """Robust Soliton Distribution (Erlich 2017 robust_solition.pyx).

    Returns a list of length K, prob[d-1] = P(degree=d), d=1..K.
    """
    if K <= 0:
        return []
    if K == 1:
        return [1.0]
    # Ideal Soliton rho
    rho = [0.0] * (K + 1)  # indices 0..K, use 1..K
    rho[1] = 1.0 / K
    for d in range(2, K + 1):
        rho[d] = 1.0 / (d * (d - 1))
    # Robust part tau
    S = c * math.log(K / delta) * math.sqrt(K)
    pivot = max(1, int(math.floor(K / S)))
    tau = [0.0] * (K + 1)
    for d in range(1, min(pivot, K + 1)):
        tau[d] = (S / K) * (1.0 / d)
    if pivot <= K:
        tau[pivot] = (S / K) * math.log(S / delta)
    # normalize
    mu = [rho[d] + tau[d] for d in range(K + 1)]
    Z = sum(mu[1:])  # d=1..K
    if Z <= 0:
        # degrade to the ideal soliton distribution
        return [rho[d] / sum(rho[1:]) for d in range(1, K + 1)]
    return [mu[d] / Z for d in range(1, K + 1)]


def _sample_degree(rsd: list[float], rng: random.Random) -> int:
    """Sample the degree d from the RSD."""
    r = rng.random()
    cum = 0.0
    for d, p in enumerate(rsd, 1):
        cum += p
        if r < cum:
            return d
    return len(rsd)


@dataclass(slots=True)
class ErlichOligo:
    """An Erlich-Zielinski oligo."""
    index: int            # oligo number (for tracking, not required for
                          # decoding)
    seed: int             # LT fountain-code seed (4 bytes, used to
                          # rebuild adjacency during decoding)
    payload: str          # DNA payload (152 nt, satisfies GC/homopolymer
                          # constraints)
    rs_oligo: bytes       # raw bytes after RS encoding (incl. RS
                          # checksum)


def erlich_encode(data: bytes, oligo_size: int = ERLICH_OLIGO_SIZE,
                  rs_num: int = ERLICH_RS_SIZE,
                  gc_dev: float = ERLICH_GC_DEV,
                  max_homopolymer: int = ERLICH_MAX_HOMOPOLYMER,
                  redundancy: float = 1.1,
                  delta: float = 0.001, c: float = 0.025,
                  seed_rng: int = 42,
                  whiten: bool = True) -> list[ErlichOligo]:
    """Erlich-Zielinski 2017 fountain-code encoding.

    Steps (paper Methods + encode.py):
    1. split data into K source blocks of oligo_size bytes (pad the last
       block with 0)
    2. compute the RSD
    3. generate K * (1 + redundancy) oligos:
       a. random seed (32-bit)
       b. seed -> PRNG -> sample degree d + select d source blocks
       c. XOR the d source blocks -> 32B payload
       d. (optional) whiten: XOR the payload with a seed-derived mask
          to raise byte entropy
       e. seed(4B) + payload(32B) -> RS encoding -> +2B checksum = 38B
       f. 2-bit DNA mapping -> 152 nt
       g. GC/homopolymer constraint filtering (rejection sampling,
          retry with a new seed)

    whiten: whether to whiten the payload (XOR with a seed-derived
    mask). The original paper assumes the data is already compressed
    (high entropy); whitening lets arbitrary data (including text) also
    satisfy the GC/homopolymer constraints.
    """
    if not _HAS_REEDSOLO:
        raise RuntimeError("erlich_encode requires reedsolo: pip install reedsolo")

    # 1. split into chunks
    if not data:
        data = b"\x00" * oligo_size
    chunks: list[bytes] = []
    for i in range(0, len(data), oligo_size):
        chunk = data[i:i + oligo_size]
        if len(chunk) < oligo_size:
            chunk = chunk + b"\x00" * (oligo_size - len(chunk))
        chunks.append(chunk)
    K = len(chunks)

    # 2. RSD
    rsd = robust_soliton_distribution(K, delta=delta, c=c)
    rs = RSCodec(rs_num)
    n_target = max(1, int(math.ceil(K * (1 + redundancy))))
    rng = random.Random(seed_rng)

    oligos: list[ErlichOligo] = []
    seen_seeds: set[int] = set()
    attempts = 0
    max_attempts = n_target * 100

    while len(oligos) < n_target and attempts < max_attempts:
        attempts += 1
        seed = rng.randint(0, 2 ** 31 - 1)
        if seed in seen_seeds:
            continue
        # sample degree d
        prng = random.Random(seed)
        d = _sample_degree(rsd, prng)
        # select d source blocks (without replacement)
        neighbors = prng.sample(range(K), min(d, K))
        # XOR source blocks
        payload = bytearray(oligo_size)
        for nb in neighbors:
            chunk = chunks[nb]
            for j in range(oligo_size):
                payload[j] ^= chunk[j]
        # whiten: XOR the payload with a seed-derived mask (raises byte
        # entropy -> satisfies GC/homopolymer constraints)
        if whiten:
            mask_prng = random.Random(seed ^ 0x5EEDBA11)
            mask = bytes(mask_prng.randint(0, 255) for _ in range(oligo_size))
            payload = bytearray(a ^ b for a, b in zip(payload, mask, strict=False))
        # RS encoding: seed(4B) + payload(oligo_size) -> +rs_num
        # checksum
        header = struct.pack(">I", seed)
        message = bytes(header) + bytes(payload)
        try:
            rs_encoded = bytes(rs.encode(message))
        except _RS_ERROR_TYPES:
            # Reed-Solomon encoding error (rare; usually the input is
            # out of the field range)
            continue
        # 2-bit DNA mapping
        dna = _bytes_to_dna_2bit(rs_encoded)
        # constraint filtering (rejection sampling)
        if _satisfies_constraints(dna, gc_dev, max_homopolymer):
            seen_seeds.add(seed)
            oligos.append(ErlichOligo(
                index=len(oligos), seed=seed,
                payload=dna, rs_oligo=rs_encoded
            ))

    if len(oligos) < n_target:
        raise RuntimeError(
            f"could not generate enough valid oligos: "
            f"{len(oligos)}/{n_target} (K={K}, attempts={attempts})"
        )
    return oligos


def erlich_decode(oligos: Iterable[ErlichOligo],
                  K: int, oligo_size: int = ERLICH_OLIGO_SIZE,
                  rs_num: int = ERLICH_RS_SIZE,
                  total_len: int | None = None,
                  delta: float = 0.001, c: float = 0.025,
                  whiten: bool = True) -> bytes:
    """Erlich-Zielinski fountain-code decoding (BP / peeling decoder).

    Steps (paper Methods + receiver.py + glass.pyx):
    1. each oligo: DNA -> bytes -> RS decode (correct or discard)
    2. extract seed + payload (if whitened at encoding, reverse-XOR to
       restore)
    3. seed -> PRNG -> rebuild degree d + adjacent source blocks
    4. build a bipartite graph, BP peeling decode
    """
    if not _HAS_REEDSOLO:
        raise RuntimeError("erlich_decode requires reedsolo")

    rs = RSCodec(rs_num)
    rsd = robust_soliton_distribution(K, delta=delta, c=c)

    # 1. decode each oligo
    decoded_droplets: list[tuple[int, bytes]] = []  # (seed, payload)
    for oligo in oligos:
        raw = _dna_to_bytes_2bit(oligo.payload)
        if len(raw) < ERLICH_HEADER_SIZE + oligo_size + rs_num:
            continue
        try:
            corrected, _, _ = rs.decode(raw)
            corrected = bytes(corrected)
        except _RS_ERROR_TYPES:
            continue  # RS failed (errors beyond correction capacity /
                      # invalid input format), discard
        seed = struct.unpack(">I", corrected[:ERLICH_HEADER_SIZE])[0]
        payload = bytearray(corrected[ERLICH_HEADER_SIZE:ERLICH_HEADER_SIZE + oligo_size])
        # reverse whiten: XOR to restore the original payload
        if whiten:
            mask_prng = random.Random(seed ^ 0x5EEDBA11)
            mask = bytes(mask_prng.randint(0, 255) for _ in range(oligo_size))
            payload = bytearray(a ^ b for a, b in zip(payload, mask, strict=False))
        decoded_droplets.append((seed, bytes(payload)))

    if not decoded_droplets:
        raise ValueError("no droplets survived RS decoding")

    # 2. build the bipartite graph: each droplet -> adjacent source
    #    blocks
    #    rebuild adjacency using the seed
    droplet_neighbors: list[list[int]] = []
    droplet_payloads: list[bytes] = []
    for seed, payload_bytes in decoded_droplets:
        prng = random.Random(seed)
        d = _sample_degree(rsd, prng)
        neighbors = prng.sample(range(K), min(d, K))
        droplet_neighbors.append(neighbors)
        droplet_payloads.append(payload_bytes)

    # 3. BP peeling decoding
    recovered: list[bytes | None] = [None] * K
    # each source block -> adjacent droplet indices
    block_to_droplets: dict[int, list[int]] = {i: [] for i in range(K)}
    for di, neighbors in enumerate(droplet_neighbors):
        for nb in neighbors:
            block_to_droplets[nb].append(di)

    # degree = number of unrecovered adjacent source blocks
    degree = [len(n) for n in droplet_neighbors]
    # queue: droplets of degree 1
    queue = [di for di, deg in enumerate(degree) if deg == 1]

    while queue:
        di = queue.pop(0)
        if degree[di] != 1:
            continue
        # find the only unrecovered source block
        unrecovered = [nb for nb in droplet_neighbors[di]
                       if recovered[nb] is None]
        if not unrecovered:
            degree[di] = 0
            continue
        target = unrecovered[0]
        recovered_target = bytes(droplet_payloads[di])
        recovered[target] = recovered_target
        # Peel: XOR the target out of every droplet adjacent to it
        for dj in block_to_droplets[target]:
            if degree[dj] > 0 and dj != di:
                # XOR
                new_payload = bytearray(droplet_payloads[dj])
                for j in range(oligo_size):
                    new_payload[j] ^= recovered_target[j]
                droplet_payloads[dj] = bytes(new_payload)
                droplet_neighbors[dj].remove(target)
                degree[dj] -= 1
                if degree[dj] == 1:
                    queue.append(dj)
        degree[di] = 0

    # 4. check whether everything was recovered
    if any(r is None for r in recovered):
        missing = [i for i, r in enumerate(recovered) if r is None]
        raise ValueError(
            f"LT decoding failed: {len(missing)}/{K} blocks unrecovered "
            f"(need more oligos or higher redundancy)"
        )

    result = b"".join(recovered)  # type: ignore[arg-type]
    if total_len is not None:
        result = result[:total_len]
    return result


# ============================================================================
# PCR error model + error-correction validation
# ============================================================================
# Saiki 1988 / Potapov 2017 measured error rates (per base per cycle)
# Taq DNA polymerase (no proofreading)
PCR_SUBSTITUTION_RATE = 1.5e-4    # Taq, Potapov 2017
PCR_INDEL_RATE = 4.5e-6           # Taq, indels make up ~1-3% of errors
DEFAULT_PCR_CYCLES = 30

# Fidelity of different polymerases (Potapov 2017, NEB data)
PCR_POLYMERASE_RATES: dict[str, dict[str, float]] = {
    "taq":     {"substitution": 1.5e-4, "indel": 4.5e-6},
    "pfu":     {"substitution": 5.1e-6, "indel": 1.5e-7},
    "q5":      {"substitution": 5.3e-7, "indel": 1.6e-8},
    "phusion": {"substitution": 3.9e-6, "indel": 1.2e-7},
}

# Transition / Transversion bias (Potapov 2017)
# Transitions (A<->G, C<->T) make up ~86%, Transversions ~14%
# Implementation: on substitution, pick transition vs transversion
# with probability 6:1
_TRANSITIONS = {"A": "G", "G": "A", "C": "T", "T": "C"}  # STATE: global (immutable lookup table)
_TRANSVERSIONS = {  # STATE: global (immutable lookup table)
    "A": ("C", "T"), "G": ("C", "T"),
    "C": ("A", "G"), "T": ("A", "G"),
}


def _substitute(base: str, rng: random.Random,
                transition_bias: float = 6.0 / 7.0) -> str:
    """Replace a base with transition/transversion bias.

    Args:
        transition_bias: probability of picking a transition
            (default 6/7 ~ 0.857, Potapov 2017).
    """
    if rng.random() < transition_bias:
        return _TRANSITIONS[base]
    else:
        return rng.choice(_TRANSVERSIONS[base])


def pcr_amplify(dna: str, cycles: int = DEFAULT_PCR_CYCLES,
                sub_rate: float = PCR_SUBSTITUTION_RATE,
                indel_rate: float = PCR_INDEL_RATE,
                rng: random.Random | None = None,
                polymerase: str | None = None) -> str:
    """Simulate errors introduced by PCR amplification.

    Per base, per cycle:
    - substitute with probability sub_rate
      (transition:transversion = 6:1, Potapov 2017)
    - insert a random base with probability indel_rate/2
    - delete with probability indel_rate/2

    Args:
        polymerase: named polymerase (taq/pfu/q5/phusion) overriding
            sub_rate/indel_rate.
    Returns:
        the final sequence as read by sequencing (one read).

    30 Taq cycles give cumulative substitution ~ 1-(1-1.5e-4)^30 ~ 0.45%.
    """
    if rng is None:
        rng = random.Random()
    if polymerase and polymerase.lower() in PCR_POLYMERASE_RATES:
        rates = PCR_POLYMERASE_RATES[polymerase.lower()]
        sub_rate = rates["substitution"]
        indel_rate = rates["indel"]
    bases = "ACGT"
    current = dna
    for _ in range(cycles):
        new_seq = []
        for c in current:
            r = rng.random()
            if r < indel_rate / 2:
                # insert a random base and keep the original base
                new_seq.append(rng.choice(bases))
                new_seq.append(c)
            elif r < indel_rate:
                # deletion
                continue
            elif r < indel_rate + sub_rate:
                # substitution (transition/transversion bias)
                new_seq.append(_substitute(c, rng))
            else:
                new_seq.append(c)
        current = "".join(new_seq)
    return current


# ============================================================================
# DNA chemical synthesis error model (Filges 2021 Clinical Chemistry
# 67:1384-1394)
# ============================================================================
# Unlike PCR errors: chemical synthesis (phosphoramidite method) error
# spectrum is dominated by deletions (~7:1)
# Coupling efficiency 98.5-99.5%/base, full-length fraction 10-50% for
# 140-mers
# Overall oligo accuracy 97.2% (Filges 2021, average over vendors)

# Filges 2021 measured defaults
SYNTHESIS_DELETION_RATE = 1.0e-2      # 1%/base (Filges 2021 typical)
SYNTHESIS_SUBSTITUTION_RATE = 1.4e-3  # ~deletion/7
SYNTHESIS_INSERTION_RATE = 1.0e-4     # rare


def synthesize_dna(dna: str,
                   deletion_rate: float = SYNTHESIS_DELETION_RATE,
                   substitution_rate: float = SYNTHESIS_SUBSTITUTION_RATE,
                   insertion_rate: float = SYNTHESIS_INSERTION_RATE,
                   rng: random.Random | None = None,
                   quality: str = "typical") -> str:
    """Simulate errors introduced by chemical synthesis
    (phosphoramidite method, Filges 2021).

    Filges 2021 measured IDT/Eurofins/Sigma-Aldrich/BioSearch:
    - deletions dominate (deletion:substitution ~ 7:1)
    - overall oligo accuracy 97.2%
    - full-length fraction 10-50% for 140-mers (98.5-99.5% coupling)

    Args:
        quality: "low" (98.5% coupling) | "typical" (99%) | "high"
            (99.5%), overriding deletion_rate/substitution_rate.
    Returns:
        the synthesized sequence (with errors).

    Differences from the PCR model:
    - PCR is substitution-dominated (Taq 1.5e-4, transition/transversion
      bias 6:1)
    - synthesis is deletion-dominated (1e-2, no transition/transversion
      bias)
    """
    if rng is None:
        rng = random.Random()
    # override parameters by quality
    if quality == "low":
        deletion_rate = 1.5e-2     # 98.5% coupling
        substitution_rate = 2.1e-3
    elif quality == "high":
        deletion_rate = 5.0e-3     # 99.5% coupling
        substitution_rate = 7.0e-4
    # "typical" uses the defaults
    bases = "ACGT"
    out = []
    for c in dna:
        r = rng.random()
        if r < deletion_rate:
            # deletion (Filges 2021 dominant error type)
            continue
        elif r < deletion_rate + insertion_rate:
            # insertion
            out.append(rng.choice(bases))
            out.append(c)
        elif r < deletion_rate + insertion_rate + substitution_rate:
            # substitution (no clear transition/transversion bias in
            # chemical synthesis)
            others = [b for b in bases if b != c]
            out.append(rng.choice(others))
        else:
            out.append(c)
    return "".join(out)


def synthesis_yield(oligo_length: int,
                    coupling_efficiency: float = 0.99) -> float:
    """Estimate the full-length oligo fraction (Filges 2021).

    full-length fraction = coupling_efficiency^(oligo_length - 1)
    Filges 2021: 140-mer @ 98.5% -> ~10%, @ 99.5% -> ~50%
    """
    if oligo_length <= 1:
        return 1.0
    return coupling_efficiency ** (oligo_length - 1)


# ============================================================================
# Sequencing error model (Ceze, Nivala, Strauss Nat Rev Genet 2019
# 20:456-466)
# ============================================================================
# Illumina SBS: substitution-dominated, Q30=1e-3
# PacBio HiFi: random errors, Q40+ ~ 1e-4
# ONT R10.4: indel-dominated, simplex ~1%

# Default parameters (Illumina HiSeq/NovaSeq Q30)
SEQUENCING_SUBSTITUTION_RATE = 1.0e-3
SEQUENCING_INDEL_RATE = 1.0e-4


def sequence_dna(dna: str,
                 platform: str = "illumina_hiseq_novaseq",
                 rng: random.Random | None = None) -> str:
    """Simulate sequencing errors (Ceze 2019 review, multi-platform
    parameters).

    platform options:
    - "illumina_hiseq_novaseq": Q30, sub=1e-3, indel=1e-4 (default)
    - "illumina_miseq": sub=5e-3, indel=5e-4
    - "pacbio_hifi": Q40+, sub=1e-4, indel=1e-4
    - "ont_r10_4_simplex": sub=5e-3, indel=1e-2 (indel-dominated)
    - "ont_r10_4_duplex": sub=5e-4, indel=1e-3

    Returns:
        the sequencing read (with errors).
    """
    if rng is None:
        rng = random.Random()
    # load platform parameters from bio_data (avoid redefinition)
    from helixlang.bio_data import SEQUENCING_PLATFORM_ERROR_RATES
    if platform not in SEQUENCING_PLATFORM_ERROR_RATES:
        raise ValueError(f"unknown platform {platform!r}; "
                         f"available: {list(SEQUENCING_PLATFORM_ERROR_RATES)}")
    rates = SEQUENCING_PLATFORM_ERROR_RATES[platform]
    sub_rate = float(rates["substitution"])
    indel_rate = float(rates["indel"])
    bases = "ACGT"
    out = []
    for c in dna:
        r = rng.random()
        if r < indel_rate / 2:
            # insertion
            out.append(rng.choice(bases))
            out.append(c)
        elif r < indel_rate:
            # deletion
            continue
        elif r < indel_rate + sub_rate:
            # substitution (Illumina substitution-dominated, no clear
            # transition/transversion bias)
            others = [b for b in bases if b != c]
            out.append(rng.choice(others))
        else:
            out.append(c)
    return "".join(out)


# ============================================================================
# DNA decay model (Allentoft 2012, Grass 2015)
# ============================================================================

def decay_dna(dna: str, years: float, temperature_c: float = 13.1,
              encapsulated: bool = False,
              rng: random.Random | None = None) -> str:
    """Simulate random degradation of DNA over the years (Allentoft 2012
    Arrhenius model).

    decay probability = 1 - exp(-years * ln2 / t_half)
    Each base independently "degrades" with the decay probability
    (replaced with N, mimicking deamination/hydrolysis damage).

    Allentoft 2012: bone DNA half-life 521 years at 13.1°C
    Grass 2015: silica-encapsulated half-life 2000 years at 70°C

    Returns:
        the degraded sequence (with Ns).
    """
    if rng is None:
        rng = random.Random()
    from helixlang.bio_data import dna_survival_fraction
    survival = dna_survival_fraction(years, temperature_c, encapsulated)
    # per-base survival probability = survival
    out = []
    for c in dna:
        if rng.random() < survival:
            out.append(c)
        else:
            # degrade to N (IUPAC ambiguous base)
            out.append("N")
    return "".join(out)


def validate_iupac_dna(dna: str) -> bool:
    """Strictly validate a DNA sequence with BioPython IUPAC.

    Accepts ACGT + IUPAC ambiguous bases (N/R/Y/etc.).
    """
    if not dna:
        return False
    if _HAS_BIOPYTHON:
        try:
            from Bio.Data.IUPACData import ambiguous_dna_values
            for c in dna.upper():
                if c not in ambiguous_dna_values:
                    return False
            return True
        except (ImportError, AttributeError):
            # BioPython data table missing or attribute changed —
            # fall back to pure-letter validation
            return False
    return all(c in "ACGTN" for c in dna.upper())


def translate_dna(dna: str, table: int = 1) -> str:
    """Translate DNA -> protein with the BioPython standard translation
    table.

    table=1 standard nuclear genes; 2 mitochondria; 6 ciliates.
    """
    if not _HAS_BIOPYTHON:
        raise RuntimeError("translate_dna requires biopython")
    seq = Seq(dna)
    return str(seq.translate(table=table))


def gc_stats(dna: str) -> dict:
    """GC statistics: content, max homopolymer length."""
    gc = _gc_content(dna)
    max_run = 1
    run = 1
    for i in range(1, len(dna)):
        if dna[i] == dna[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return {
        "gc_content": round(gc, 4),
        "max_homopolymer": max_run,
        "length": len(dna),
        "valid_iupac": validate_iupac_dna(dna),
    }


# ============================================================================
# Integration: helix <-> DNA physical codec
# ============================================================================

def helix_to_dna(helix_source: str, scheme: str = "goldman") -> dict:
    """Encode HelixLang source (text) as DNA.

    scheme: "goldman" | "erlich"
    Returns:
        a dict containing the oligo list + statistics.
    """
    data = helix_source.encode("utf-8")
    if scheme == "goldman":
        oligos = goldman_encode(data)
        # num_segments = number of 100 nt payload oligos produced (the
        # byte stream is Huffman-coded, so it is not simply 6/byte)
        num_segments = len(oligos)
        total_bp = sum(len(o.full) for o in oligos)
        return {
            "scheme": "goldman",
            "oligos": [{"index": o.index, "full": o.full,
                        "payload": o.payload, "overhang": o.overhang}
                       for o in oligos],
            "stats": {
                "num_oligos": len(oligos),
                "num_segments": num_segments,
                "data_bytes": len(data),
                "total_bp": total_bp,
                "density_bit_per_nt": round(len(data) * 8 / total_bp, 4)
                                      if total_bp > 0 else 0.0,
            },
        }
    elif scheme == "erlich":
        erlich_oligos = erlich_encode(data)
        total_bp = sum(len(o.payload) for o in erlich_oligos)
        return {
            "scheme": "erlich",
            "oligos": [{"index": o.index, "seed": o.seed, "payload": o.payload}
                       for o in erlich_oligos],
            "stats": {
                "num_oligos": len(erlich_oligos),
                "total_bp": total_bp,
                "data_bytes": len(data),
                "density_bit_per_nt": round(len(data) * 8 / total_bp, 4)
                                      if total_bp > 0 else 0.0,
                "K": max(1, math.ceil(len(data) / ERLICH_OLIGO_SIZE)),
            },
        }
    else:
        raise ValueError(f"unknown scheme {scheme!r}")


def dna_to_helix(oligos_data: dict, scheme: str = "goldman",
                 total_len: int | None = None) -> str:
    """Decode DNA oligos back into HelixLang source."""
    if scheme == "goldman":
        goldman_oligos = [
            GoldmanOligo(
                index=o["index"], payload=o["payload"],
                overhang=o.get("overhang", ""), full=o.get("full", "")
            ) for o in oligos_data["oligos"]
        ]
        orig_len = oligos_data.get("stats", {}).get("data_bytes")
        effective_len = total_len if total_len is not None else orig_len
        data = goldman_decode(goldman_oligos, total_len=effective_len)
        return data.decode("utf-8", errors="replace")
    elif scheme == "erlich":
        erlich_oligos = [
            ErlichOligo(index=o["index"], seed=o["seed"],
                        payload=o["payload"], rs_oligo=b"")
            for o in oligos_data["oligos"]
        ]
        K = oligos_data.get("stats", {}).get("K")
        if K is None:
            # infer K = num_oligos / (1 + redundancy)
            K = max(1, len(erlich_oligos) // 2)
        orig_len = oligos_data.get("stats", {}).get("data_bytes")
        effective_len = total_len if total_len is not None else orig_len
        data = erlich_decode(erlich_oligos, K=K, total_len=effective_len)
        return data.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
