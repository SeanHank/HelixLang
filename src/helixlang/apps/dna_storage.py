"""DNA storage application: a complete file <-> DNA storage pipeline.

Features:
- File -> DNA encoding (Goldman/Erlich selectable)
- DNA -> file decoding
- Full-chain simulation (synthesis -> PCR -> sequencing -> decoding)
- Storage density/cost/durability analysis reports

DNA storage fidelity assumptions: the error model assumes independent errors (synthesis
deletion-dominated / PCR substitution-dominated / sequencing-platform-specific), though
real errors are correlated and clustered. Real systems need RS/fountain code redundancy
(Erlich built-in) + 4x overlapping redundancy (Goldman) for zero-error recovery; decay uses an Arrhenius single-exponential approximation.

Based on real data:
- Goldman 2013 density 0.29 bit/nt, cost ~$12,400/MB (Nature 494:77-80)
- Erlich 2017 density 1.57 bit/nt, Shannon limit 1.58 (Science 355:950-954)
- DNA synthesis cost ~$0.50/bp (2024 TWIST Bioscience pricing)
- DNA sequencing cost ~$0.01/bp (Illumina NovaSeq 2024)
- DNA storage durability >10,000 years @ -20°C (Allentoft 2012)
"""
from __future__ import annotations

import math
import random
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from helixlang.bio_data import (
    DNA_STORAGE_DENSITY_BENCHMARKS,
    DNA_STORAGE_SHANNON_LIMIT_BIT_PER_NT,
    dna_decay_half_life,
)
from helixlang.dna_codec import (
    ERLICH_OLIGO_SIZE,
    INDEX_NT,
    SEGMENT_NT,
    ErlichOligo,
    GoldmanOligo,
    decay_dna,
    erlich_decode,
    erlich_encode,
    goldman_decode,
    goldman_encode,
    pcr_amplify,
    sequence_dna,
    synthesize_dna,
)
from helixlang.errors import BioError
from helixlang.seq_utils import gc_content as _gc_content
from helixlang.seq_utils import max_homopolymer as _max_homopolymer

# ============================================================================
# Cost/durability constants (2024 market prices)
# ============================================================================
DNA_SYNTHESIS_COST_PER_BP_USD = 0.50   # TWIST Bioscience 2024
DNA_SEQUENCING_COST_PER_BP_USD = 0.01  # Illumina NovaSeq 2024


# ============================================================================
# Report dataclasses
# ============================================================================

@dataclass(slots=True)
class StorageReport:
    """Encoding report: returned by store()."""
    scheme: str                  # goldman / erlich
    oligos: list                 # oligo list
    total_bp: int                # total base count
    density_bit_per_nt: float    # actual storage density (bit/nt)
    num_oligos: int              # oligo count
    avg_oligo_length: float      # average oligo length
    encoding_time: float         # encoding time (seconds)
    data_len: int = 0            # original data length (bytes)
    K: int | None = None         # number of source blocks in the Erlich scheme (None for Goldman)
    # chunk metadata for parallel encoding: [(chunk_data_size, oligo_count), ...]
    # retrieve_concurrent uses this to split the oligo list and decode chunk by chunk
    parallel_chunks: list[tuple[int, int]] | None = None


@dataclass(slots=True)
class LifecycleReport:
    """Lifecycle report: returned by simulate_lifecycle()."""
    original_data: bytes         # original data
    recovered_data: bytes        # recovered data
    integrity: float             # data integrity 0-1
    error_rate: float            # total error rate
    synthesis_errors: int        # synthesis stage error count
    pcr_errors: int              # PCR stage error count
    sequencing_errors: int       # sequencing stage error count
    decay_damage: int            # decay-damaged base count
    success: bool                # whether recovery succeeded


@dataclass(slots=True)
class AnalysisReport:
    """Analysis report: returned by analyze()."""
    density_bit_per_nt: float    # storage density
    shannon_efficiency: float    # efficiency relative to 1.58 bit/nt
    estimated_cost_usd: float    # synthesis + sequencing cost (USD)
    durability_years: float      # estimated storage durability (years)
    oligo_count: int             # oligo count
    total_bp: int                # total base count
    gc_content: float            # average GC content
    max_homopolymer: int         # maximum homopolymer length
    comparison: dict             # comparison with paper benchmarks


# ============================================================================
# Public utility functions
# ============================================================================

def estimate_cost(total_bp: int, include_sequencing: bool = True) -> dict:
    """Estimate the DNA storage cost (synthesis + sequencing).

    Based on 2024 market prices:
    - Synthesis: $0.50/bp (TWIST Bioscience)
    - Sequencing: $0.01/bp (Illumina NovaSeq)

    Returns {synthesis_cost, sequencing_cost, total_cost, cost_per_mb}.
    cost_per_mb = cost per million bases.
    """
    synthesis_cost = total_bp * DNA_SYNTHESIS_COST_PER_BP_USD
    sequencing_cost = (total_bp * DNA_SEQUENCING_COST_PER_BP_USD
                       if include_sequencing else 0.0)
    total_cost = synthesis_cost + sequencing_cost
    cost_per_mb = total_cost / (total_bp / 1_000_000) if total_bp > 0 else 0.0
    return {
        "synthesis_cost": synthesis_cost,
        "sequencing_cost": sequencing_cost,
        "total_cost": total_cost,
        "cost_per_mb": cost_per_mb,
    }


def estimate_durability(temperature_c: float,
                        encapsulated: bool = False) -> float:
    """Estimate DNA storage durability (years).

    Based on the Arrhenius model (Allentoft 2012 / Grass 2015):
    - encapsulated=False: naked DNA / bone DNA model (Allentoft 2012, Ea=110 kJ/mol)
      13.1°C -> 521 years
    - encapsulated=True: silica-encapsulated model (Grass 2015, Ea=91 kJ/mol)
      70°C -> 2000 years, 9°C -> ~2 million years
    """
    return dna_decay_half_life(temperature_c, encapsulated=encapsulated)


def format_fasta(oligos: list, scheme: str) -> str:
    """Format an oligo list into FASTA format.

    Header format: >{scheme}_{index}
    Sequence: the oligo's DNA sequence (Goldman=full, Erlich=payload)
    """
    lines: list[str] = []
    for oligo in oligos:
        if scheme == "goldman":
            seq = oligo.full
        elif scheme == "erlich":
            seq = oligo.payload
        else:
            raise BioError(f"unknown scheme {scheme!r}; use 'goldman' or 'erlich'")
        lines.append(f">{scheme}_{oligo.index}")
        lines.append(seq)
    return "\n".join(lines) + "\n"


def parse_fasta(fasta_text: str) -> list:
    """Parse an oligo list from FASTA text.

    Automatically detect the scheme (goldman/erlich) and return the corresponding oligo object list.
    Header format: >{scheme}_{index}
    """
    oligos: list[Any] = []
    current_scheme: str | None = None
    current_index: int = 0
    current_seq_lines: list[str] = []

    def flush() -> None:
        nonlocal current_scheme, current_index, current_seq_lines
        if current_scheme is None or not current_seq_lines:
            return
        seq = "".join(current_seq_lines)
        if current_scheme == "goldman":
            oligos.append(GoldmanOligo(
                index=current_index, payload="",
                overhang="", full=seq
            ))
        elif current_scheme == "erlich":
            oligos.append(ErlichOligo(
                index=current_index, seed=0,
                payload=seq, rs_oligo=b""
            ))
        current_seq_lines = []

    for raw_line in fasta_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header = line[1:].strip()
            first_token = header.split()[0] if header else ""
            parts = first_token.split("_", 1)
            current_scheme = parts[0] if parts else None
            if len(parts) > 1 and parts[1].isdigit():
                current_index = int(parts[1])
            else:
                current_index = len(oligos)
            current_seq_lines = []
        else:
            current_seq_lines.append(line.upper())
    flush()
    return oligos


def compare_with_benchmarks(density: float) -> dict:
    """Compare storage density with paper benchmarks.

    Returns {goldman_2013, erlich_2017, organick_2018, shannon_limit, this_work}.
    """
    return {
        "goldman_2013": DNA_STORAGE_DENSITY_BENCHMARKS["goldman_2013"]["density_bit_per_nt"],
        "erlich_2017": DNA_STORAGE_DENSITY_BENCHMARKS["erlich_2017"]["density_bit_per_nt"],
        "organick_2018": DNA_STORAGE_DENSITY_BENCHMARKS["organick_2018"]["density_bit_per_nt"],
        "shannon_limit": DNA_STORAGE_SHANNON_LIMIT_BIT_PER_NT,
        "this_work": density,
    }


# ============================================================================
# Private helper functions
# ============================================================================

def _count_diffs(seq1: str, seq2: str) -> int:
    """Count bases that differ between two sequences at corresponding positions (ignoring length differences)."""
    return sum(1 for a, b in zip(seq1, seq2, strict=False) if a != b)


# _max_homopolymer has been consolidated into helixlang.seq_utils.max_homopolymer


# ============================================================================
# Reed-Solomon availability (mirrors the optional reedsolo dependency of
# the Erlich codec in helixlang.dna_codec)
# ============================================================================

try:
    from reedsolo import ReedSolomonError as _RealRSEncoderError
    from reedsolo import RSCodec as _RSCodec
    _HAS_REEDSOLO_BENCH = True
except ImportError:
    _RealRSEncoderError = ValueError
    _RSCodec = None
    _HAS_REEDSOLO_BENCH = False

_RS_ERROR_TYPES_BENCH: tuple[type[BaseException], ...] = (
    ValueError, IndexError, _RealRSEncoderError,
)


# ============================================================================
# Codec benchmark (S5): cost-robustness trade-off across code rates
# ============================================================================

@dataclass(slots=True)
class CodecBenchmarkRow:
    """One row of the codec benchmark table.

    Attributes:
        scheme: ``"fountain"`` (Erlich LT + RS inner code), ``"goldman"``
            (4x overlapping segments) or ``"rs"`` (block Reed-Solomon).
        target_density: requested density (bit/nt); None for Goldman's
            native rate.
        achieved_density: actual density of the encoded payload.
        redundancy: relative redundancy 1 + extra (fountain only).
        max_loss_fraction: largest fraction of dropped oligos that still
            decodes (erasure tolerance).
        max_error_rate: largest per-base substitution rate that still
            decodes (error tolerance).
        decode_time_s: time for one full decode of the surviving set.
        num_oligos: number of DNA oligos produced.
        total_bp: total bases produced.
        cost_per_gb_usd: synthesis + sequencing cost normalized per GB
            of stored data (2024 market prices).
    """

    scheme: str
    target_density: float | None
    achieved_density: float
    redundancy: float | None
    max_loss_fraction: float
    max_error_rate: float
    decode_time_s: float
    num_oligos: int
    total_bp: int
    cost_per_gb_usd: float


def _mutate_dna(seq: str, rate: float, rng: random.Random) -> str:
    """Introduce substitutions at per-base ``rate`` (no indels)."""
    bases = "ACGT"
    out: list[str] = []
    for b in seq:
        if rng.random() < rate:
            out.append(rng.choice([x for x in bases if x != b]))
        else:
            out.append(b)
    return "".join(out)


def _binary_search_max(ok: Callable[[float], bool],
                       lo: float, hi: float, iters: int = 7) -> float:
    """Largest ``f`` in [lo, hi] for which ``ok(f)`` holds."""
    best = lo
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if ok(mid):
            best = mid
            lo = mid
        else:
            hi = mid
    return best


# -- fountain (Erlich) -----------------------------------------------------

def _fountain_decode_ok(oligos: list, data: bytes,
                        fraction: float, drop: bool,
                        seed: int) -> bool:
    """Decode ``oligos`` after dropping/mutating a ``fraction``."""
    K = max(1, math.ceil(len(data) / ERLICH_OLIGO_SIZE))
    rng = random.Random(seed + round(fraction * 10_000))
    if drop:
        keep = max(1, int(len(oligos) * (1.0 - fraction)))
        sub = rng.sample(oligos, keep)
    else:
        sub = []
        for o in oligos:
            payload = _mutate_dna(o.payload, fraction, rng)
            sub.append(ErlichOligo(index=o.index, seed=o.seed,
                                   payload=payload, rs_oligo=b""))
    try:
        return erlich_decode(sub, K=K, total_len=len(data)) == data
    except Exception:
        return False


def _fountain_loss_max(oligos: list, data: bytes, seed: int) -> float:
    return _binary_search_max(
        lambda f: _fountain_decode_ok(oligos, data, f, True, seed),
        0.0, 0.95)


def _fountain_error_max(oligos: list, data: bytes, seed: int) -> float:
    return _binary_search_max(
        lambda f: _fountain_decode_ok(oligos, data, f, False, seed),
        0.0, 0.5)


def _benchmark_fountain(data: bytes, target_density: float,
                        seed: int) -> CodecBenchmarkRow:
    """Tune the Erlich redundancy to hit ``target_density`` and measure
    loss/error tolerance."""
    best_red = 1.05
    best_density = float("inf")
    for redundancy in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0,
                       1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0,
                       4.0, 5.0, 6.0, 8.0, 10.0):
        oligos = erlich_encode(data, redundancy=redundancy, seed_rng=seed)
        density = len(data) * 8 / sum(len(o.payload) for o in oligos)
        if abs(density - target_density) < abs(best_density - target_density):
            best_density = density
            best_red = redundancy
    oligos = erlich_encode(data, redundancy=best_red, seed_rng=seed)
    total_bp = sum(len(o.payload) for o in oligos)
    t0 = time.perf_counter()
    loss = _fountain_loss_max(oligos, data, seed)
    err = _fountain_error_max(oligos, data, seed)
    dt = time.perf_counter() - t0
    return CodecBenchmarkRow(
        scheme="fountain", target_density=target_density,
        achieved_density=best_density, redundancy=best_red,
        max_loss_fraction=loss, max_error_rate=err,
        decode_time_s=dt, num_oligos=len(oligos), total_bp=total_bp,
        cost_per_gb_usd=estimate_cost(total_bp)["total_cost"]
        / (len(data) / 1e9),
    )


# -- Goldman ---------------------------------------------------------------

def _goldman_decode_ok(oligos: list, data: bytes,
                       fraction: float, drop: bool, seed: int) -> bool:
    rng = random.Random(seed + round(fraction * 10_000))
    if drop:
        keep = max(1, int(len(oligos) * (1.0 - fraction)))
        sub = rng.sample(oligos, keep)
    else:
        sub = []
        for o in oligos:
            full = _mutate_dna(o.full, fraction, rng)
            sub.append(GoldmanOligo(index=o.index, payload="",
                                    overhang="", full=full))
    try:
        return goldman_decode(sub, total_len=len(data)) == data
    except Exception:
        return False


def _goldman_loss_max(oligos: list, data: bytes, seed: int) -> float:
    return _binary_search_max(
        lambda f: _goldman_decode_ok(oligos, data, f, True, seed),
        0.0, 0.95)


def _goldman_error_max(oligos: list, data: bytes, seed: int) -> float:
    return _binary_search_max(
        lambda f: _goldman_decode_ok(oligos, data, f, False, seed),
        0.0, 0.5)


def _benchmark_goldman(data: bytes, seed: int) -> CodecBenchmarkRow:
    """Goldman has a fixed native density (built-in 4x overlap)."""
    oligos = goldman_encode(data)
    total_bp = sum(len(o.full) for o in oligos)
    density = len(data) * 8 / total_bp
    t0 = time.perf_counter()
    loss = _goldman_loss_max(oligos, data, seed)
    err = _goldman_error_max(oligos, data, seed)
    dt = time.perf_counter() - t0
    return CodecBenchmarkRow(
        scheme="goldman", target_density=None,
        achieved_density=density, redundancy=None,
        max_loss_fraction=loss, max_error_rate=err,
        decode_time_s=dt, num_oligos=len(oligos), total_bp=total_bp,
        cost_per_gb_usd=estimate_cost(total_bp)["total_cost"]
        / (len(data) / 1e9),
    )


# -- block Reed-Solomon ----------------------------------------------------

def _rs_blocks(data: bytes, block_len: int, nsym: int
               ) -> tuple[list, Any, bytes]:
    """RS-encode ``data`` into per-block codewords (2-bit DNA mapping)."""
    codec = _RSCodec(nsym)
    padded = data + b"\x00" * ((block_len - len(data) % block_len) % block_len)
    blocks = [padded[i:i + block_len]
              for i in range(0, len(padded), block_len)]
    encoded = [bytes(codec.encode(b)) for b in blocks]
    return encoded, codec, data


def _rs_decode_ok(encoded: list, codec: Any, data: bytes,
                  fraction: float, drop: bool, block_len: int,
                  seed: int) -> bool:
    rng = random.Random(seed + round(fraction * 10_000))
    n_drop = int(len(encoded) * fraction) if drop else 0
    out = bytearray()
    for i, block in enumerate(encoded):
        if drop:
            if i < n_drop:
                return False  # a dropped block irrecoverably loses its
                              # data (RS corrects within, not across,
                              # blocks)
        elif fraction > 0.0:
            mutated = bytearray(block)
            for j in range(len(mutated)):
                if rng.random() < fraction:
                    mutated[j] = rng.randrange(256)
            block = bytes(mutated)
        try:
            dec, _, _ = codec.decode(block)
        except _RS_ERROR_TYPES_BENCH:
            return False
        out += bytes(dec)[:block_len]
    return bytes(out)[:len(data)] == data


def _rs_loss_max(encoded: list, codec: Any, data: bytes,
                 block_len: int, seed: int) -> float:
    return _binary_search_max(
        lambda f: _rs_decode_ok(encoded, codec, data, f, True,
                                block_len, seed),
        0.0, 0.95)


def _rs_error_max(encoded: list, codec: Any, data: bytes,
                  block_len: int, seed: int) -> float:
    return _binary_search_max(
        lambda f: _rs_decode_ok(encoded, codec, data, f, False,
                                block_len, seed),
        0.0, 0.5)


def _benchmark_rs(data: bytes, target_density: float,
                  seed: int) -> CodecBenchmarkRow:
    """Block Reed-Solomon tuned to ``target_density`` (density = 2R)."""
    block_len = 32
    rate = target_density / 2.0  # 2-bit DNA mapping: 8 bits / (4 nt)
    nsym = max(2, round(block_len * (1.0 / rate - 1.0)))
    encoded, codec, payload = _rs_blocks(data, block_len, nsym)
    total_bp = 4 * sum(len(e) for e in encoded)
    density = len(data) * 8 / total_bp
    t0 = time.perf_counter()
    loss = _rs_loss_max(encoded, codec, payload, block_len, seed)
    err = _rs_error_max(encoded, codec, payload, block_len, seed)
    dt = time.perf_counter() - t0
    return CodecBenchmarkRow(
        scheme="rs", target_density=target_density,
        achieved_density=density, redundancy=None,
        max_loss_fraction=loss, max_error_rate=err,
        decode_time_s=dt, num_oligos=len(encoded), total_bp=total_bp,
        cost_per_gb_usd=estimate_cost(total_bp)["total_cost"]
        / (len(data) / 1e9),
    )


def benchmark_codecs(data: bytes | None = None,
                     densities: tuple[float, ...] = (0.5, 1.0, 1.5),
                     schemes: tuple[str, ...] = ("goldman", "fountain", "rs"),
                     data_size: int = 512,
                     seed: int = 7) -> list[CodecBenchmarkRow]:
    """Compare codecs across code rates (S5; Nat Commun 2026 methodology).

    Scans the robustness of each scheme at fixed code rates (0.5 / 1.0 /
    1.5 bit/nt) and reports the erasure tolerance (max fraction of lost
    DNA molecules) and error tolerance (max per-base substitution rate)
    together with the per-GB cost and decode time.

    Literature anchor (Erlich & Zielinski 2017 Science 355:950): the
    fountain code decodes from a 1+redundancy fraction of the molecules,
    so the tolerated molecule loss grows as the code rate falls --
    at ~0.5 bit/nt a fountain scheme tolerates well above 60% sequence
    loss, while Goldman's 4x overlap tolerates roughly 3/4 loss at its
    low native density.

    Args:
        data: payload bytes (random by default).
        densities: target code rates in bit/nt.
        schemes: schemes to benchmark.
        data_size: payload size in bytes when ``data`` is None.
        seed: RNG seed for deterministic sampling.

    Returns:
        list of :class:`CodecBenchmarkRow` (one per scheme/rate).
    """
    if data is None:
        rng = random.Random(seed)
        data = bytes(rng.randint(0, 255) for _ in range(data_size))
    rows: list[CodecBenchmarkRow] = []
    for scheme in schemes:
        if scheme == "goldman":
            rows.append(_benchmark_goldman(data, seed))
        elif scheme == "fountain":
            for d in densities:
                rows.append(_benchmark_fountain(data, d, seed))
        elif scheme == "rs":
            if not _HAS_REEDSOLO_BENCH:
                raise BioError(
                    "rs benchmark requires reedsolo: pip install reedsolo")
            for d in densities:
                rows.append(_benchmark_rs(data, d, seed))
        else:
            raise BioError(f"unknown scheme {scheme!r}; "
                           f"use 'goldman', 'fountain' or 'rs'")
    return rows


def format_benchmark_table(rows: list[CodecBenchmarkRow]) -> str:
    """Render the benchmark rows as an aligned text table."""
    header = ("scheme   target  achieved  redun.  loss  err    decode  "
              "oligos  total_bp  cost/GB(USD)")
    lines = [header]
    for r in rows:
        tgt = f"{r.target_density:.2f}" if r.target_density is not None else "native"
        red = f"{r.redundancy:.2f}" if r.redundancy is not None else "-"
        lines.append(
            f"{r.scheme:<8} {tgt:>6} {r.achieved_density:>8.3f} {red:>6} "
            f"{r.max_loss_fraction:>5.2f} {r.max_error_rate:>5.2f} "
            f"{r.decode_time_s:>6.3f} {r.num_oligos:>6} "
            f"{r.total_bp:>9} {r.cost_per_gb_usd:>10.0f}")
    return "\n".join(lines)




class DNAStorage:
    """Main class of the DNA storage application.

    Wraps the Goldman 2013 / Erlich 2017 codecs into a complete file storage/retrieval/analysis application.

    Parameters
    ----------
    scheme : str
        Encoding scheme, "goldman" or "erlich" (default).
    """

    def __init__(self, scheme: str = "erlich"):
        if scheme not in ("goldman", "erlich"):
            raise BioError(f"unknown scheme {scheme!r}; use 'goldman' or 'erlich'")
        self.scheme = scheme

    def store(self, data: bytes, redundancy: float = 0.15) -> StorageReport:
        """Encode data as DNA oligos.

        Parameters
        ----------
        data : bytes
            The binary data to be stored.
        redundancy : float
            Redundancy (used only by the Erlich scheme; Goldman has built-in 4x overlapping redundancy).
        """
        t0 = time.perf_counter()
        if self.scheme == "goldman":
            # Goldman encoding has built-in 4x overlapping redundancy; the redundancy parameter is not applicable
            oligos: list = goldman_encode(data)
            total_bp = sum(len(o.full) for o in oligos)
            K = None
        else:  # erlich
            oligos = erlich_encode(data, redundancy=redundancy)
            total_bp = sum(len(o.payload) for o in oligos)
            K = max(1, math.ceil(len(data) / ERLICH_OLIGO_SIZE))

        encoding_time = time.perf_counter() - t0
        num_oligos = len(oligos)
        avg_oligo_length = total_bp / num_oligos if num_oligos else 0.0
        density = (len(data) * 8 / total_bp) if total_bp > 0 else 0.0

        return StorageReport(
            scheme=self.scheme,
            oligos=oligos,
            total_bp=total_bp,
            density_bit_per_nt=density,
            num_oligos=num_oligos,
            avg_oligo_length=avg_oligo_length,
            encoding_time=encoding_time,
            data_len=len(data),
            K=K,
        )

    def retrieve(self, oligos: list, total_len: int) -> bytes:
        """Decode data from DNA oligos.

        Parameters
        ----------
        oligos : list
            Oligo list (GoldmanOligo or ErlichOligo).
        total_len : int
            Original data length (bytes).
        """
        if self.scheme == "goldman":
            return goldman_decode(oligos, total_len=total_len)
        else:  # erlich
            K = max(1, math.ceil(total_len / ERLICH_OLIGO_SIZE)) if total_len > 0 else 1
            return erlich_decode(oligos, K=K, total_len=total_len)

    def store_parallel(self, data: bytes, redundancy: float = 0.15,
                       max_workers: int = 4,
                       chunk_size: int = 4096) -> StorageReport:
        """Encode data as DNA oligos in parallel.

        Split the data into multiple chunks and encode them in parallel with ThreadPoolExecutor.
        Each chunk is encoded independently, and the oligo index offsets are globally unique.

        - Erlich scheme: each chunk is independently fountain-code encoded, decoded chunk by chunk by retrieve_concurrent
        - Goldman scheme: 4x overlapping redundancy requires global segment indices, so it falls back to sequential store()

        Parameters
        ----------
        data : bytes
            The binary data to be stored.
        redundancy : float
            Redundancy (Erlich scheme only).
        max_workers : int
            Maximum number of parallel threads.
        chunk_size : int
            Bytes per chunk (Erlich scheme only).
        """
        t0 = time.perf_counter()

        # Goldman: 4x overlapping redundancy requires global segment indices, fall back to sequential encoding
        if self.scheme == "goldman":
            report = self.store(data, redundancy)
            report.encoding_time = time.perf_counter() - t0
            return report

        # Erlich: parallel encoding
        if not data:
            report = self.store(data, redundancy)
            report.encoding_time = time.perf_counter() - t0
            return report

        # Split into chunks
        chunks: list[bytes] = [data[i:i + chunk_size]
                               for i in range(0, len(data), chunk_size)]
        n_chunks = len(chunks)

        # Encode each chunk in parallel
        def _encode_chunk(chunk: bytes) -> tuple[list, int]:
            if self.scheme == "goldman":
                oligos: list = goldman_encode(chunk)
                bp = sum(len(o.full) for o in oligos)
            else:
                oligos = erlich_encode(chunk, redundancy=redundancy)
                bp = sum(len(o.payload) for o in oligos)
            return oligos, bp

        chunk_results: list[tuple[list, int]] = [None] * n_chunks  # type: ignore
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_encode_chunk, chunk): i
                             for i, chunk in enumerate(chunks)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                chunk_results[idx] = future.result()

        # Merge results: offset the oligo indices to be globally unique, record the chunk metadata
        all_oligos: list = []
        total_bp = 0
        parallel_chunks: list[tuple[int, int]] = []
        for i, (chunk_oligos, bp) in enumerate(chunk_results):
            offset = len(all_oligos)
            for oligo in chunk_oligos:
                oligo.index = offset + oligo.index
            all_oligos.extend(chunk_oligos)
            total_bp += bp
            parallel_chunks.append((len(chunks[i]), len(chunk_oligos)))

        encoding_time = time.perf_counter() - t0
        num_oligos = len(all_oligos)
        avg_oligo_length = total_bp / num_oligos if num_oligos else 0.0
        density = (len(data) * 8 / total_bp) if total_bp > 0 else 0.0
        K = max(1, math.ceil(len(data) / ERLICH_OLIGO_SIZE))

        return StorageReport(
            scheme=self.scheme,
            oligos=all_oligos,
            total_bp=total_bp,
            density_bit_per_nt=density,
            num_oligos=num_oligos,
            avg_oligo_length=avg_oligo_length,
            encoding_time=encoding_time,
            data_len=len(data),
            K=K,
            parallel_chunks=parallel_chunks,
        )

    def retrieve_concurrent(self, oligos: list,
                            parallel_chunks: list[tuple[int, int]]) -> bytes:
        """Decode data from DNA oligos encoded in parallel.

        Split the oligo list according to the parallel_chunks metadata returned by store_parallel,
        decode each chunk independently, then concatenate the results.

        Concurrency note (honest naming): this method does **not** use thread/process parallel decoding.
        ``store_parallel`` encodes each chunk in parallel with ThreadPoolExecutor during the encoding
        stage, but the decoding stage processes chunks sequentially (concurrent chunks, sequential decoding).
        The name ``retrieve_concurrent`` reflects that it handles the concurrently chunked data produced
        by ``store_parallel``, not that decoding itself is parallel. For parallel decoding, call ``retrieve``
        per chunk externally and manage the thread pool yourself.

        Parameters
        ----------
        oligos : list
            Oligo list (returned by store_parallel).
        parallel_chunks : list[tuple[int, int]]
            Chunk metadata [(chunk_data_size, oligo_count), ...],
            from StorageReport.parallel_chunks returned by store_parallel.
        """
        if not parallel_chunks:
            # No chunk metadata, fall back to sequential decoding
            total_len = sum(s for s, _ in parallel_chunks) if parallel_chunks else 0
            return self.retrieve(oligos, total_len=total_len)

        # Split the oligo list by oligo_count, decode each chunk sequentially
        decoded_parts: list[bytes] = []
        offset = 0
        for chunk_data_size, oligo_count in parallel_chunks:
            chunk_oligos = oligos[offset:offset + oligo_count]
            offset += oligo_count

            if self.scheme == "goldman":
                part = goldman_decode(chunk_oligos, total_len=chunk_data_size)
            else:  # erlich
                K = max(1, math.ceil(chunk_data_size / ERLICH_OLIGO_SIZE)) if chunk_data_size > 0 else 1
                part = erlich_decode(chunk_oligos, K=K, total_len=chunk_data_size)
            decoded_parts.append(part)

        return b"".join(decoded_parts)

    def retrieve_parallel(self, oligos: list,
                          parallel_chunks: list[tuple[int, int]]) -> bytes:
        """Deprecated alias, equivalent to :meth:`retrieve_concurrent`.

        .. deprecated::
            The old name ``retrieve_parallel`` implied that decoding was parallel, but in reality
            chunks were processed sequentially (only the encoding stage was parallel). It has been
            renamed to ``retrieve_concurrent`` to accurately reflect the behavior. Calling this
            alias emits a :class:`DeprecationWarning`.
        """
        warnings.warn(
            "retrieve_parallel is deprecated; use retrieve_concurrent "
            "(decoding is sequential, only encoding was parallel)",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.retrieve_concurrent(oligos, parallel_chunks)

    def simulate_lifecycle(self, data: bytes,
                           synthesis_quality: str = "typical",
                           pcr_cycles: int = 10,
                           polymerase: str = "taq",
                           sequencing_platform: str = "illumina_hiseq_novaseq",
                           storage_years: float = 0,
                           storage_temp: float = -20,
                           *, seed: int | None = None) -> LifecycleReport:
        """Simulate the complete storage lifecycle: encoding -> synthesis -> PCR -> decay -> sequencing -> decoding.

        Parameters
        ----------
        data : bytes
            Original data.
        synthesis_quality : str
            Synthesis quality ("low" / "typical" / "high" / "perfect").
            "perfect" skips synthesis errors (for noiseless pipeline validation).
        pcr_cycles : int
            PCR cycle count.
        polymerase : str
            Polymerase ("taq" / "pfu" / "q5" / "phusion").
        sequencing_platform : str
            Sequencing platform (see bio_data.SEQUENCING_PLATFORM_ERROR_RATES).
        storage_years : float
            Storage years (0 = no storage).
        storage_temp : float
            Storage temperature (deg C).
        seed : int, optional
            Random seed (for reproducible tests).
        """
        # 1. Encode
        report = self.store(data)
        original_oligos = report.oligos

        rng = random.Random(seed)
        synthesis_errors = 0
        pcr_errors = 0
        sequencing_errors = 0
        decay_damage = 0
        total_bases = 0

        processed_oligos: list[Any] = []
        for oligo in original_oligos:
            if self.scheme == "goldman":
                orig_seq = oligo.full
            else:
                orig_seq = oligo.payload
            total_bases += len(orig_seq)

            # 2. Chemical synthesis (phosphoramidite method, Filges 2021)
            if synthesis_quality == "perfect":
                synth_seq = orig_seq  # skip synthesis errors (noiseless pipeline validation)
            else:
                synth_seq = synthesize_dna(orig_seq, quality=synthesis_quality, rng=rng)
            synthesis_errors += _count_diffs(orig_seq, synth_seq)

            # 3. PCR amplification (Saiki 1988 / Potapov 2017)
            pcr_seq = pcr_amplify(synth_seq, cycles=pcr_cycles,
                                  polymerase=polymerase, rng=rng)
            pcr_errors += _count_diffs(synth_seq, pcr_seq)

            # 4. Sequencing (Ceze 2019)
            seq_seq = sequence_dna(pcr_seq, platform=sequencing_platform, rng=rng)
            sequencing_errors += _count_diffs(pcr_seq, seq_seq)

            # 5. Decay (Allentoft 2012 Arrhenius)
            if storage_years > 0:
                decayed_seq = decay_dna(seq_seq, years=storage_years,
                                        temperature_c=storage_temp, rng=rng)
                decay_damage += sum(1 for c in decayed_seq if c == "N")
            else:
                decayed_seq = seq_seq

            # 6. Rebuild the oligo (keep the original index/payload for the decoder fallback)
            if self.scheme == "goldman":
                # Extract the index overhang and payload region (fault-tolerant: the sequence may be shortened by indels)
                overhang = decayed_seq[:INDEX_NT]
                payload = decayed_seq[INDEX_NT:INDEX_NT + SEGMENT_NT]
                processed_oligos.append(GoldmanOligo(
                    index=oligo.index, payload=payload,
                    overhang=overhang, full=decayed_seq
                ))
            else:
                processed_oligos.append(ErlichOligo(
                    index=oligo.index, seed=oligo.seed,
                    payload=decayed_seq, rs_oligo=b""
                ))

        # 7. Decode
        try:
            recovered = self.retrieve(processed_oligos, total_len=len(data))
        except (ValueError, KeyError, IndexError, RuntimeError) as exc:
            # Decoding failed (insufficient oligos / corrupted indices / general decoding errors); record diagnostic information
            import logging
            logging.getLogger(__name__).debug(
                "DNA retrieve failed: %s: %s", type(exc).__name__, exc)
            recovered = b""
        except Exception as exc:
            # Fallback: third-party exceptions such as reedsolo.ReedSolomonError (inherit directly from Exception)
            import logging
            logging.getLogger(__name__).warning(
                "DNA retrieve unexpected error: %s: %s",
                type(exc).__name__, exc)
            recovered = b""

        # 8. Compute integrity
        if len(data) == 0:
            integrity = 1.0 if recovered == data else 0.0
        else:
            matches = sum(1 for a, b in zip(data, recovered, strict=False) if a == b)
            integrity = matches / len(data)

        total_errors = (synthesis_errors + pcr_errors
                        + sequencing_errors + decay_damage)
        error_rate = total_errors / total_bases if total_bases > 0 else 0.0
        success = recovered == data

        return LifecycleReport(
            original_data=data,
            recovered_data=recovered,
            integrity=integrity,
            error_rate=error_rate,
            synthesis_errors=synthesis_errors,
            pcr_errors=pcr_errors,
            sequencing_errors=sequencing_errors,
            decay_damage=decay_damage,
            success=success,
        )

    def analyze(self, oligos: list, data_len: int) -> AnalysisReport:
        """Generate a storage analysis report.

        Parameters
        ----------
        oligos : list
            Oligo list.
        data_len : int
            Original data length (bytes).
        """
        total_bp = 0
        gc_sum = 0.0  # weighted GC accumulation (gc_content * length)
        max_homopolymer = 0

        for oligo in oligos:
            if self.scheme == "goldman":
                seq = oligo.full
            else:
                seq = oligo.payload
            total_bp += len(seq)
            gc_sum += _gc_content(seq) * len(seq)
            h = _max_homopolymer(seq)
            if h > max_homopolymer:
                max_homopolymer = h

        density = (data_len * 8 / total_bp) if total_bp > 0 else 0.0
        shannon_eff = density / DNA_STORAGE_SHANNON_LIMIT_BIT_PER_NT
        cost = estimate_cost(total_bp, include_sequencing=True)
        durability = estimate_durability(-20, encapsulated=False)
        comparison = compare_with_benchmarks(density)
        avg_gc = gc_sum / total_bp if total_bp > 0 else 0.0

        return AnalysisReport(
            density_bit_per_nt=density,
            shannon_efficiency=shannon_eff,
            estimated_cost_usd=cost["total_cost"],
            durability_years=durability,
            oligo_count=len(oligos),
            total_bp=total_bp,
            gc_content=avg_gc,
            max_homopolymer=max_homopolymer,
            comparison=comparison,
        )
