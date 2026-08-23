#!/usr/bin/env python3
"""Cross-platform downloader for HelixLang GEM reference databases.

Works on Windows, Linux, and macOS.  Only requires Python ≥ 3.9 and
standard-library modules (no third-party dependencies).

Usage:
    python scripts/download_data.py            # download everything
    python scripts/download_data.py pfam       # Pfam only
    python scripts/download_data.py ecoli      # E. coli reference only

Databases:
    Pfam-A.hmm           (CC0 public domain)   – EBI FTP
    E. coli K-12 MG1655  (public domain)       – NCBI / UniProt
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

PFAM_URL = "ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
ECOLI_GENOME_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nucleotide&id=U00096.3&rettype=fasta&retmode=text"
)
ECOLI_PROTEOME_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/reference_proteomes/Bacteria/"
    "UP000000625/UP000000625_83333.fasta.gz"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_dir() -> Path:
    """Return <project_root>/data, creating it if needed."""
    d = Path(__file__).resolve().parent.parent / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path, label: str) -> None:
    """Download *url* to *dest*, showing a simple progress indicator."""
    print(f"  Downloading {label} ...")
    try:
        urllib.request.urlretrieve(url, str(dest))  # noqa: S310
    except urllib.error.URLError as exc:
        sys.exit(f"  ERROR: failed to download {label}: {exc}")
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  Saved {dest.name}  ({size_mb:.1f} MB)")


def _gunzip(gz_path: Path, out_path: Path) -> None:
    """Decompress a .gz file."""
    print(f"  Decompressing {gz_path.name} ...")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {out_path.name}  ({size_mb:.1f} MB)")


def _run_hmmpress(hmm_path: Path) -> None:
    """Run hmmpress if available."""
    if shutil.which("hmmpress") is None:
        print("  WARNING: hmmpress not found – index files not built.")
        print("           Install HMMER:  conda install -c bioconda hmmer")
        return
    print("  Running hmmpress ...")
    subprocess.run(  # noqa: S603
        ["hmmpress", str(hmm_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("  hmmpress complete.")

# ---------------------------------------------------------------------------
# Download targets
# ---------------------------------------------------------------------------

def download_pfam() -> None:
    data = _data_dir()
    gz = data / "Pfam-A.hmm.gz"
    hmm = data / "Pfam-A.hmm"

    if hmm.exists():
        print("  Pfam-A.hmm already exists – skipping.")
        return

    _download(PFAM_URL, gz, "Pfam-A.hmm.gz")
    _gunzip(gz, hmm)
    _run_hmmpress(hmm)


def download_ecoli() -> None:
    data = _data_dir()
    genome = data / "ecoli_core_genome.fasta"
    proteome_gz = data / "ecoli_core_proteome.fasta.gz"
    proteome = data / "ecoli_core_proteome.fasta"

    if not genome.exists():
        _download(ECOLI_GENOME_URL, genome, "E. coli genome (U00096.3)")
    else:
        print("  ecoli_core_genome.fasta already exists – skipping.")

    if not proteome.exists():
        _download(ECOLI_PROTEOME_URL, proteome_gz, "E. coli proteome")
        _gunzip(proteome_gz, proteome)
    else:
        print("  ecoli_core_proteome.fasta already exists – skipping.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download reference databases for HelixLang GEM pipeline.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["all", "pfam", "ecoli"],
        help="What to download (default: all)",
    )
    args = parser.parse_args()

    actions = {
        "all":   [download_pfam, download_ecoli],
        "pfam":  [download_pfam],
        "ecoli": [download_ecoli],
    }

    for fn in actions[args.target]:
        print(f"\n=== {fn.__name__.replace('download_', '').upper()} ===")
        fn()

    print(f"\nDone.  Files in {_data_dir()}/")
    for f in sorted(_data_dir().iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            print(f"  {f.name:40s}  {size / 1024 / 1024:7.1f} MB")
        else:
            print(f"  {f.name:40s}  {size / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
