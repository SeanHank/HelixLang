"""DIAMOND / BLAST+ local sequence search wrapper (doc/20 §5.1)."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Hit:
    """Single BLAST/DIAMOND alignment hit."""

    query_id: str
    subject_id: str
    identity: float
    alignment_length: int
    e_value: float
    bit_score: float
    stitle: str = ""


@dataclass
class SearchResult:
    """Container for DIAMOND/BLAST+ search results."""

    query_count: int = 0
    hits: list[Hit] = field(default_factory=list)

    def hits_for(self, query_id: str) -> list[Hit]:
        return [h for h in self.hits if h.query_id == query_id]


def run_diamond(
    query_fasta: str | Path,
    database: str | Path,
    *,
    outfmt: str = "6 qseqid sseqid pident length evalue bitscore stitle",
    evalue: float = 1e-5,
    max_target_seqs: int = 5,
    threads: int = 1,
    diamond_bin: str = "diamond",
) -> SearchResult:
    """Run DIAMOND blastp and parse results.

    Parameters
    ----------
    query_fasta : path to query FASTA (protein sequences)
    database : path to pre-built DIAMOND database (.dmnd)
    outfmt : DIAMOND output format string
    evalue : E-value threshold
    max_target_seqs : max targets per query
    threads : number of threads
    diamond_bin : path to diamond executable

    Returns
    -------
    SearchResult with parsed hits

    Raises
    ------
    FileNotFoundError if diamond binary not found
    RuntimeError if DIAMOND exits non-zero

    Notes
    -----
    Requires DIAMOND (Buchfink et al. 2021, Nat Methods 18:1613) to be
    installed.  If not available, the pipeline falls back to web BLAST
    or marks genes as unannotated.
    """
    result = SearchResult()
    query_path = Path(query_fasta)
    db_path = Path(database)

    if not query_path.exists():
        raise FileNotFoundError(f"Query FASTA not found: {query_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    cmd = [
        diamond_bin,
        "blastp",
        "--query", str(query_path),
        "--db", str(db_path),
        "--outfmt", *outfmt.split(),
        "--evalue", str(evalue),
        "--max-target-seqs", str(max_target_seqs),
        "--threads", str(threads),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"DIAMOND binary not found at '{diamond_bin}'. "
            "Install: conda install -c bioconda diamond"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"DIAMOND failed (rc={exc.returncode}): {exc.stderr}"
        ) from exc

    result.query_count = len({
        line.split("\t")[0]
        for line in proc.stdout.strip().splitlines()
        if line.strip()
    })

    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        result.hits.append(Hit(
            query_id=parts[0],
            subject_id=parts[1],
            identity=float(parts[2]),
            alignment_length=int(parts[3]),
            e_value=float(parts[4]),
            bit_score=float(parts[5]),
            stitle=parts[6] if len(parts) > 6 else "",
        ))

    return result


def build_database(
    input_fasta: str | Path,
    database: str | Path,
    *,
    diamond_bin: str = "diamond",
    mode: str = "prot",
) -> Path:
    """Build a DIAMOND database from a FASTA file.

    Parameters
    ----------
    input_fasta : source FASTA
    database : output database path (without .dmnd extension)
    diamond_bin : path to diamond executable
    mode : 'prot' for protein, 'dna' for nucleotide

    Returns
    -------
    Path to the created .dmnd file
    """
    db_path = Path(database)
    cmd = [
        diamond_bin,
        "makedb",
        "--in", str(input_fasta),
        "--db", str(db_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"DIAMOND binary not found at '{diamond_bin}'"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"DIAMOND makedb failed: {exc.stderr}"
        ) from exc

    return db_path.with_suffix(".dmnd")
