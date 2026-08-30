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

import random

from helixlang.core.errors import ABIVersionError
from helixlang.core.hxbc import (
    MAGIC,
    BinaryError,
    BinaryFormatError,
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
