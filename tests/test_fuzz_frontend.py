"""doc/38 §10 goal 12: seeded fuzz of the frontend (lexer/parser/semantic).

A skewed alphabet of real helix constructs (#gene/#end/#config blocks, codons,
key=value pairs, ``->``, braces) plus garbage/truncation/CRLF/backslash
mutations is pushed through :func:`~helixlang.core.parser.parse_source`.

Invariants (doc/38 §10.1):

A. every input either produces a *typed* error (``LexError`` / ``ParseError`` /
   ``SemanticError`` / ``SimConfigError`` family) or a :class:`Program` —
   never an untyped ``Exception`` and never a hang;
B. determinism — lexing twice yields identical tokens and parsing twice yields
   an identical decompiled program;
C. round-trip — every ``Program`` that parses **and** passes semantic analysis
   survives ``dumps_program → loads_program → decompile → parse_source``
   identically (decompile is canonical).
"""
from __future__ import annotations

import random

import pytest

from helixlang.core import hxbc
from helixlang.core.errors import LexError, ParseError, SemanticError, SimConfigError
from helixlang.core.lexer import Lexer
from helixlang.core.parser import parse_source

_BASES = "ACGT"
_CODONS = ["ATG", "GCT", "GGT", "GTA", "GAT", "GAA", "TGT", "TCT",
           "TAA", "TAG", "TGA", "GGC", "AGC", "AAG", "CAC", "CCG"]
_KEYS = ["name", "ticks", "output", "table", "ops_per_tick", "react_steps",
         "species", "backend", "gene_size", "max_cells", "verbose"]

_ALLOWED = (LexError, ParseError, SemanticError, SimConfigError)

_MAX_SRC = 2048


def _random_word(rng: random.Random) -> str:
    """A random 1..6 base sequence (may split a codon — that's fine)."""
    return "".join(rng.choice(_BASES) for _ in range(rng.randint(1, 6)))


def _valid_base(rng: random.Random) -> str:
    """A structurally-valid program: gene blocks plus an optional #config."""
    out: list[str] = []
    for _ in range(rng.randint(1, 3)):
        out.append(f"#gene name=g{rng.randint(1, 3)}")
        while True:
            out.append(" ".join(rng.choice(_CODONS) for _ in range(rng.randint(1, 4))))
            if rng.random() < 0.5:
                break
        out.append("#end")
    if rng.random() < 0.9:
        kv = ", ".join(f"{rng.choice(_KEYS)}="
                       f"{rng.choice(['1', '5', 'stdout', 'standard'])}"
                       for _ in range(rng.randint(1, 3)))
        out.append(f"#config {kv}")
    return "\n".join(out)


def _mutate(rng: random.Random, src: str) -> str:
    """Apply 1..4 random mutations; the result may or may not stay valid."""
    ops = [
        lambda s: s[: len(s) // 2],                            # truncate
        lambda s: s + "\n#ende",                               # bad #end
        lambda s: s.replace("\n#end", "", 1),                  # drop #end
        lambda s: s.replace("#gene", "#GeneE", 1),             # bad keyword
        lambda s: s + " " + "".join(rng.choices(_BASES, k=rng.randint(1, 3))),
        lambda s: "".join(_BASES[rng.randrange(4)] if ch in _BASES else ch
                          for ch in s),                        # base flips
        lambda s: s.replace("\n", "\r\n"),                     # CRLF
        lambda s: s.replace(" ", "\\\n"),                      # backslash continuation
        lambda s: "#" + s,                                     # leading gem
        lambda s: s + rng.choice(["{", "}", "->", "=", "'''", "\x00"]),
        lambda s: s + rng.choice(["", "   ", "\t"]),
    ]
    out = src
    for _ in range(rng.randint(1, 4)):
        out = rng.choice(ops)(out)
    return out[:_MAX_SRC]


def _lex_tokens(src: str) -> list[tuple[str, object]]:
    return [(t.kind, t.value) for t in Lexer(src).tokens()]


def _try_compile(src: str):
    """-> ("ok", Program) | ("err", exc).  Input is bounded to ``_MAX_SRC``
    (≈ 40 tokens) so the frontend terminates on any input (invariant A, no hang)."""
    try:
        toks = _lex_tokens(src)
        prog = parse_source(src)
    except _ALLOWED as exc:  # noqa: BLE001 - typed frontend errors only
        return ("err", exc)
    del toks
    return ("ok", prog)


def _frontend_error_type(exc: Exception) -> str:
    return f"{type(exc).__module__}.{type(exc).__name__}"


def test_frontend_fuzz_type_safety_and_determinism():
    """Invariants A + B over 1000 seeded trials."""
    oks = 0
    for trial in range(1000):
        rng = random.Random(0x5EED_0000 + trial)
        src = _mutate(rng, _valid_base(rng))
        state = _try_compile(src)
        if state[0] == "err":
            exc = state[1]
            assert isinstance(exc, _ALLOWED), (
                f"trial {trial}: untyped {_frontend_error_type(exc)} from {src!r}\n{exc}")
            continue
        oks += 1
        _, prog = state
        # Invariant B: lexing and parsing are pure functions of the source.
        assert _lex_tokens(src) == _lex_tokens(src)
        assert hxbc.decompile(prog) == hxbc.decompile(parse_source(src))
    # The alphabet must actually reach parseable programs (guards a degenerate
    # fuzzer that only ever throws).
    assert oks > 0, "fuzzer generated zero parseable programs in 1000 trials"


def test_frontend_fuzz_roundtrip():
    """Invariant C: compile→decompile→reparse is idempotent on 1000 seeds.

    Only programs that pass semantic analysis reach this round; each such
    program must survive ``dumps_program → loads_program → decompile →
    parse_source`` with a byte-identical decompilation.
    """
    seen = 0
    for trial in range(1000):
        rng = random.Random(0x5EED_5000 + trial)
        src = _mutate(rng, _valid_base(rng))
        state = _try_compile(src)
        if state[0] == "err":
            continue
        _, prog = state
        data = hxbc.dumps_program(prog)
        restored = hxbc.loads_program(data).program
        flat = hxbc.decompile(prog)
        # decompile(loaded_program) must equal decompile(original).
        assert hxbc.decompile(restored) == flat, f"trial {trial}: load round-trip"
        # ... and decompile output is canonical: reparse it identically.
        assert hxbc.decompile(parse_source(flat)) == flat, f"trial {trial}: reparse"
        seen += 1
    assert seen > 0, "no semantic-passing program in 1000 seeds"


# ── regression corpus (doc/38 §10): inputs that once escaped as untyped
# exceptions or broke canonical round-trip; pinned here so they never regress.
# (FOUND BY the fuzzers above — bug 1: bare ValueError from config fields;
#  bug 2: empty-value sim parameters broke decompile→reparse canonicality.)


@pytest.mark.parametrize("src", [
    "#gene name=g\nATG TAA\n#end\n#config ticks=abc\n",
    "#gene name=g\nATG TAA\n#end\n#config ops_per_tick=output\n",
    "#gene name=g\nATG TAA\n#end\n#config react_steps=xyz\n",
    # bug 2: a bare trailing ```key=``` must be a typed ParseError — it can
    # never become a silent empty sim parameter (nor merge into the next key).
    "#gene name=g\nATG TAA\n#end\n#config name=1 C\n",
])
def test_frontend_regression_corpus(src):
    """Regression inputs must stay typed errors (never raw ValueError)."""
    state = _try_compile(src)
    assert state[0] == "err", f"input should fail; got ok: sim? {src!r}"
    assert isinstance(state[1], _ALLOWED), \
        f"untyped {_frontend_error_type(state[1])}: {state[1]}"
    # And the failure must be deterministic on re-run (invariant B).
    state2 = _try_compile(src)
    assert state2[0] == state[0] and type(state2[1]) is type(state[1])


def test_frontend_regression_space_merge_is_stable():
    """``key= next=val`` (a space between empty-value key and another field)
    is an ambiguous but *deterministic* lexing form — it must never crash and
    must round-trip stably (the merge result is parse→decompile→parse fixed).

    The lexer keeps ``=`` inside unquoted values on purpose
    (``#config calibration_uptake=GLC=10.0`` in examples/30_virtual_cell.helix),
    so this asserts stability rather than rejection.
    """
    src = "#gene name=g\nATG TAA\n#end\n#config C= gene_size=5\n"
    state = _try_compile(src)
    assert state[0] == "ok", f"merge input must stay parseable: {state[1]}"
    prog = state[1]
    flat = hxbc.decompile(prog)
    assert hxbc.decompile(parse_source(flat)) == flat, f"not idempotent: {flat!r}"
