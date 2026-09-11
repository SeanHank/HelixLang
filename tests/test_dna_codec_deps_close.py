"""Coverage-closure tests for dna_codec.py (branch/line closure).

Targets the reachable-but-uncovered branches in the Goldman 2013
rotating-key codec, the Erlich 2017 fountain codec, the synthesis /
sequencing / decay error models, and the optional-dependency fallbacks
(Bio.Seq, reedsolo), exercised in-process by re-executing the module
source with the target package blocked in sys.modules.
"""
from __future__ import annotations

import importlib.util
import pathlib
import random
import sys

import pytest

pytest.importorskip("Bio")
pytest.importorskip("reedsolo")

from reedsolo import ReedSolomonError

import helixlang
import helixlang.plugins.runtime.dna_codec as dc
from helixlang.plugins.runtime.dna_codec import (
    _GOLDMAN_HUFFMAN_CODE,
    ERLICH_OLIGO_SIZE,
    ErlichOligo,
    GoldmanOligo,
    _align_payload,
    _assemble_dna,
    _byte_to_trits,
    _dna_to_bytes_goldman,
    _encode_index,
    _trit_to_base,
    decay_dna,
    dna_to_helix,
    erlich_decode,
    erlich_encode,
    goldman_decode,
    goldman_encode,
    helix_to_dna,
    robust_soliton_distribution,
    sequence_dna,
    synthesis_yield,
    synthesize_dna,
    validate_iupac_dna,
)
from helixlang.plugins.runtime.seq_utils import reverse_complement

_CODEC_PATH = (pathlib.Path(helixlang.__file__).parent
               / "plugins/runtime/dna_codec.py")


class _ScriptRng:
    """Deterministic rng stand-in for error-model branch coverage."""

    def __init__(self, values):
        self._values = list(values)

    def random(self):
        return self._values.pop(0) if self._values else 0.5

    def choice(self, seq):
        return seq[0]


def _fresh_module_with_blocked(blocked):
    """Re-execute the dna_codec source under its real module name with
    the given package paths forced to None in sys.modules, then restore
    the original module object."""
    saved = {name: sys.modules.get(name) for name in blocked}
    original = sys.modules.get("helixlang.plugins.runtime.dna_codec")
    try:
        if original is not None:
            del sys.modules["helixlang.plugins.runtime.dna_codec"]
        for name in blocked:
            sys.modules[name] = None
        spec = importlib.util.spec_from_file_location(
            "helixlang.plugins.runtime.dna_codec", _CODEC_PATH)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["helixlang.plugins.runtime.dna_codec"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        if original is not None:
            sys.modules["helixlang.plugins.runtime.dna_codec"] = original
        else:
            sys.modules.pop("helixlang.plugins.runtime.dna_codec", None)
        for name, obj in saved.items():
            if obj is not None:
                sys.modules[name] = obj
            else:
                sys.modules.pop(name, None)


def _dna_for_trits(trits, prev="A"):
    out = []
    for t in trits:
        nb = _trit_to_base(prev, int(t))
        out.append(nb)
        prev = nb
    return "".join(out)


# --------------------------------------------------------------------
# Module-level optional-dependency fallbacks
# --------------------------------------------------------------------
def test_bio_fallback_module():
    mod = _fresh_module_with_blocked(["Bio.Seq"])
    assert mod._HAS_BIOPYTHON is False
    # degraded pure-letter validation
    assert mod.validate_iupac_dna("ACGTA") is True
    assert mod.validate_iupac_dna("ACGTN") is True
    assert mod.validate_iupac_dna("ACGTR") is False
    with pytest.raises(RuntimeError):
        mod.translate_dna("ATG")


def test_reedsolo_fallback_module():
    mod = _fresh_module_with_blocked(["reedsolo"])
    assert mod._HAS_REEDSOLO is False
    with pytest.raises(RuntimeError):
        mod.erlich_encode(b"data")
    with pytest.raises(RuntimeError):
        mod.erlich_decode([], K=1)


def test_iupac_biointernals_missing(monkeypatch):
    # _HAS_BIOPYTHON True but the IUPACData table fails to import ->
    # the except branch in validate_iupac_dna (independent of any prior
    # in-process reload that may have disabled BioPython)
    monkeypatch.setattr(dc, "_HAS_BIOPYTHON", True)
    key = "Bio.Data.IUPACData"
    saved = sys.modules.get(key)
    sys.modules[key] = None
    try:
        assert validate_iupac_dna("ACGT") is False
    finally:
        if saved is not None:
            sys.modules[key] = saved
        else:
            sys.modules.pop(key, None)


# --------------------------------------------------------------------
# Goldman rotating-key decode internals
# --------------------------------------------------------------------
def test_byte_to_trits_happy_path_and_error():
    assert _byte_to_trits(65) == _GOLDMAN_HUFFMAN_CODE[65]
    with pytest.raises(ValueError, match="no Goldman Huffman codeword"):
        _byte_to_trits(256)


def test_goldman_window_resync_and_invalid_trit():
    # trit stream 2220102: every depth-1..5 prefix is a non-leaf trie
    # node, the depth-6 prefix is invalid (raises inside
    # _trits_to_byte), and the growing window eventually exceeds 6 so
    # the re-sync pop fires.
    dna = _dna_for_trits("2220102")
    assert _dna_to_bytes_goldman(dna) == b"\x00"


def test_align_payload_smith_waterman_with_indel():
    # >1 mismatches forces the fast path to break into the banded SW
    # alignment (semi-global, indel-tolerant)
    consensus = "A" * 100 + "C" * 40
    payload = "A" * 50 + "T" + "A" * 49 + "C" * 40
    mapping = _align_payload(payload, consensus, 0)
    assert len(mapping) == len(payload)
    assert all(c is not None and c >= 0 for c in mapping)


def test_align_payload_remainder_appended():
    consensus = "A" * 40 + "C" * 30
    payload = "A" * 10 + "A" * 70 + "C" * 29 + "G"
    mapping = _align_payload(payload, consensus, 0)
    assert len(mapping) == len(payload)


def test_assemble_dna_assembles():
    oligos = goldman_encode(b"Hello, Goldman!")
    payloads = {o.index: o.payload for o in oligos}
    out = _assemble_dna(payloads)
    assert isinstance(out, str) and out


def test_assemble_dna_refinement_loop_exhausts():
    # staggered indels keep the two refinement passes diverging, so the
    # loop runs to exhaustion (no early break) and total_len slices
    payloads = {
        0: "TAGACACGTCGCTAGTGACTATCTTATGTGCGCAGCATGCTGTGAGTATCGATGACGATACTCTGTACAGCGAGCACTATACGTACGTACGTACGTACG",
        1: "TGTGCGCAGCATCTGTGAGTATCGATGACGAGTGACTCCTGTACAGACGAGCATATACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
        2: "ATGACGAGTGACTCTGTACAGACGAGCATATACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTATCGTACGTACGTAACGTACGTACGTA",
        3: "GCATATACGTACGTACGTACAGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAACGTACGTACGTCGTAGTACGTACGTAGTAC",
    }
    assert _assemble_dna(payloads)
    assert _assemble_dna(payloads, total_len=5)


def test_goldman_decode_orientation_fallbacks():
    # rc-even oligo: forward decode raises, RC decode succeeds but
    # yields an even / out-of-range index -> fallback path (698->703)
    big_even = 48
    o_even = GoldmanOligo(
        index=1, payload="C" * 80, overhang="",
        full=reverse_complement(_encode_index(big_even)) + "G" * 80,
    )
    # junk oligo: both orientations fail to decode (701-702) and the
    # payload is empty -> skipped in the vote (707->680)
    o_junk = GoldmanOligo(
        index=2, payload="", overhang="",
        full="AGTAACCGTACCCATCCAGAGACATCGAGT",
    )
    assert isinstance(goldman_decode([o_even, o_junk]), bytes)


def test_goldman_decode_total_len_truncation():
    oligos = goldman_encode(b"Hello, Goldman!")
    assert goldman_decode(oligos, total_len=6) == b"Hello,"


# --------------------------------------------------------------------
# Erlich fountain codec
# --------------------------------------------------------------------
def test_robust_soliton_bounds():
    assert robust_soliton_distribution(0) == []
    assert robust_soliton_distribution(1) == [1.0]
    rsd = robust_soliton_distribution(10)
    assert len(rsd) == 10 and abs(sum(rsd) - 1.0) < 1e-9


def test_erlich_encode_empty_input():
    oligos = erlich_encode(b"")
    assert erlich_decode(oligos, K=1) == b"\x00" * ERLICH_OLIGO_SIZE


def test_erlich_encode_no_whitening():
    # high-entropy deterministic data (without whitening the 2-bit DNA
    # only satisfies the GC/homopolymer constraints for near-random
    # payloads)
    data = bytes(random.Random(0).randbytes(128))
    oligos = erlich_encode(data, whiten=False, redundancy=2.0, seed_rng=0)
    assert erlich_decode(oligos, K=4, total_len=len(data),
                         whiten=False) == data


def test_erlich_rs_encode_error_keeps_retrying(monkeypatch):
    class _FailingRS:
        def __init__(self, n):
            self.n = n

        def encode(self, message):
            raise ReedSolomonError("synthetic RS failure")

    monkeypatch.setattr(dc, "RSCodec", _FailingRS)
    with pytest.raises(RuntimeError, match="could not generate enough"):
        erlich_encode(b"data" * 8)


def test_erlich_decode_short_or_corrupt_payloads():
    short = ErlichOligo(index=0, seed=1, payload="ACGT", rs_oligo=b"")
    with pytest.raises(ValueError, match="no droplets survived"):
        erlich_decode([short], K=1)

    # everything flipped -> every byte outside RS correction capacity
    good = erlich_encode(b"payload data" * 4, seed_rng=7)
    rng = random.Random(5)
    bad = [
        ErlichOligo(
            o.index, o.seed,
            "".join(rng.choice([b for b in "ACGT" if b != o.payload[i]])
                    for i in range(len(o.payload))),
            o.rs_oligo,
        )
        for o in good
    ]
    with pytest.raises(ValueError, match="no droplets survived"):
        erlich_decode(bad, K=len(bad))


def test_erlich_decode_lt_failure():
    data = b"fountain" * 16
    oligos = erlich_encode(data, seed_rng=9)
    with pytest.raises(ValueError, match="LT decoding failed"):
        erlich_decode(oligos[:2], K=10)
    # successful decode with total_len truncation
    assert erlich_decode(oligos, K=4, total_len=10) == data[:10]


# --------------------------------------------------------------------
# helix <-> DNA integration
# --------------------------------------------------------------------
def test_helix_to_dna_erlich_scheme():
    source = "let x = 42"
    out = helix_to_dna(source, scheme="erlich")
    assert out["scheme"] == "erlich"
    assert out["stats"]["K"] == 1
    assert dna_to_helix(out, scheme="erlich") == source


def test_dna_to_helix_erlich_k_inferred():
    source = "def foo(): return 'hi'"
    out = helix_to_dna(source, scheme="erlich")
    del out["stats"]["K"]
    assert dna_to_helix(out, scheme="erlich") == source


@pytest.mark.parametrize("fn,kwargs", [
    (helix_to_dna, {"helix_source": "x", "scheme": "nope"}),
    (dna_to_helix, {"oligos_data": {"oligos": []}, "scheme": "nope"}),
])
def test_unknown_scheme(fn, kwargs):
    with pytest.raises(ValueError, match="unknown scheme"):
        fn(**kwargs)


# --------------------------------------------------------------------
# Synthesis / sequencing / decay error models
# --------------------------------------------------------------------
def test_synthesize_dna_all_quality_levels():
    # "typical" defaults: del 1e-2, ins 1e-4, sub 1.4e-3
    rng = _ScriptRng([0.005, 0.01005, 0.011, 0.990, 0.005, 0.01005, 0.011, 0.990])
    out = synthesize_dna("ACGTACGT", rng=rng)
    assert isinstance(out, str) and out
    for quality in ("low", "typical", "high", "bogus"):
        assert synthesize_dna("ACGTACGTACGTACGT", quality=quality)
    # default rng branch
    assert synthesize_dna("ACGTACGTACGT" * 3)


def test_synthesis_yield():
    assert synthesis_yield(0) == 1.0
    assert synthesis_yield(1) == 1.0
    assert synthesis_yield(140) < 1.0


def test_sequence_dna_platforms_and_errors():
    # insertion / deletion / substitution / keep branches via a
    # scripted rng on the default platform (indel 1e-4, sub 1e-3)
    rng = _ScriptRng([0.00001, 0.00007, 0.0005, 0.99] * 4)
    assert sequence_dna("ACGTACGT" * 4, rng=rng)
    for platform in (
        "illumina_hiseq_novaseq", "illumina_miseq", "pacbio_hifi",
        "ont_r10_4_simplex", "ont_r10_4_duplex",
    ):
        assert sequence_dna("ACGTACGT" * 4, platform=platform)
    assert sequence_dna("ACGTACGT" * 4)  # default rng
    with pytest.raises(ValueError, match="unknown platform"):
        sequence_dna("ACGT", platform="nanopore_zebra")


def test_decay_dna_branches():
    # survival ~ exp(-1000*ln2/521) ~ 0.26 -> rng 0.1 keeps, 0.9 -> N
    out = decay_dna("AC", years=1000, rng=_ScriptRng([0.1, 0.9]))
    assert out[0] == "A"
    assert out[1] == "N"
    # default rng branch + encapsulated variant
    assert decay_dna("ACGT", years=10)
    assert decay_dna("ACGT", years=10, encapsulated=True)
