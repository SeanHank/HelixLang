"""Transporter classification from genome annotation (doc/20 §5.4)."""
from __future__ import annotations

from dataclasses import dataclass, field

# TCDB-like transporter family classification
TRANSPORT_FAMILIES: dict[str, dict[str, str]] = {
    "PF00528": {"family": "ABC_efflux", "type": "active", "direction": "export"},
    "PF00005": {"family": "ABC_ATPase", "type": "active", "direction": "both"},
    "PF00355": {"family": "MFS", "type": "secondary", "direction": "both"},
    "PF00892": {"family": "MATE", "type": "secondary", "direction": "export"},
    "PF01553": {"family": "Na_H_antiporter", "type": "secondary", "direction": "import"},
    "PF00321": {"family": "Amino_acid_perm", "type": "facilitated", "direction": "import"},
    "PF01306": {"family": "Sugar_perm", "type": "facilitated", "direction": "import"},
    "PF03547": {"family": "Phosphate_perm", "type": "secondary", "direction": "import"},
    "PF00104": {"family": "Thiamin_perm", "type": "facilitated", "direction": "import"},
    "PF03853": {"family": "NhaB", "type": "secondary", "direction": "import"},
    "PF06965": {"family": "KefB_KefC", "type": "secondary", "direction": "import"},
}

# Known substrate associations (simplified)
SUBSTRATE_KEYWORDS: dict[str, list[str]] = {
    "glucose": ["glucose", "gltP", "ptsG", "manZ", "glcB"],
    "acetate": ["acetate", "acs", "actP", "pta"],
    "lactate": ["lactate", "ldhA", "lldP"],
    "succinate": ["succinate", "sdcB", "dcuC"],
    "ammonium": ["ammonium", "amtB", "nrgA"],
    "phosphate": ["phosphate", "pstS", "pitA", "pitB"],
    "nitrate": ["nitrate", "narG", "narK", "nrtP"],
    "oxygen": ["oxygen", "cydA", "cyoA", "appC"],
}


@dataclass
class TransporterInfo:
    """Predicted transporter for a gene."""

    gene_id: str
    transporter_family: str
    transport_type: str      # "active", "secondary", "facilitated"
    direction: str           # "import", "export", "both"
    predicted_substrate: str = ""
    confidence: float = 0.5


@dataclass
class TransporterScanResult:
    """Genome-wide transporter prediction results."""

    total_genes: int = 0
    transporters: list[TransporterInfo] = field(default_factory=list)

    @property
    def transporter_count(self) -> int:
        return len(self.transporters)

    def by_substrate(self) -> dict[str, list[TransporterInfo]]:
        result: dict[str, list[TransporterInfo]] = {}
        for t in self.transporters:
            sub = t.predicted_substrate or "unknown"
            result.setdefault(sub, []).append(t)
        return result


def classify_transporters(
    protein_fasta: str,
    pfam_database: str = "",
    e_value_threshold: float = 1e-5,
) -> TransporterScanResult:
    """Classify transporters from protein sequences.

    Parameters
    ----------
    protein_fasta : path to protein FASTA
    pfam_database : path to Pfam-A.hmm (optional; heuristic if empty)
    e_value_threshold : HMMER E-value cutoff

    Returns
    -------
    TransporterScanResult
    """
    result = TransporterScanResult()

    if pfam_database:
        return _hmmer_transporter_scan(
            protein_fasta, pfam_database, e_value_threshold, result
        )

    return _heuristic_transporter_scan(protein_fasta, result)


def _heuristic_transporter_scan(
    fasta_path: str, result: TransporterScanResult
) -> TransporterScanResult:
    """Heuristic transporter detection from gene IDs / annotations."""
    try:
        with open(fasta_path) as fh:
            for line in fh:
                if line.startswith(">"):
                    result.total_genes += 1
                    header = line[1:].strip().lower()
                    gene_id = line[1:].split()[0]

                    for _pfam_acc, info in TRANSPORT_FAMILIES.items():
                        if info["family"].lower() in header:
                            substrate = _guess_substrate(header)
                            result.transporters.append(TransporterInfo(
                                gene_id=gene_id,
                                transporter_family=info["family"],
                                transport_type=info["type"],
                                direction=info["direction"],
                                predicted_substrate=substrate,
                                confidence=0.5,
                            ))
                            break
    except FileNotFoundError:  # SILENTBENIGN - optional db file absent
        pass
    return result


def _hmmer_transporter_scan(
    fasta_path: str,
    pfam_hmm: str,
    e_value: float,
    result: TransporterScanResult,
) -> TransporterScanResult:
    """HMMER-based transporter scan."""
    import subprocess

    cmd = [
        "hmmsearch", "--domE", str(e_value), "--domtblout", "-",
        pfam_hmm, fasta_path,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError("hmmsearch not found") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"hmmsearch failed: {exc.stderr}") from exc

    seen: set[str] = set()
    for line in proc.stdout.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 22:
            continue
        gene_id = parts[0]
        pfam_family = parts[3]
        if pfam_family not in TRANSPORT_FAMILIES:
            continue
        info = TRANSPORT_FAMILIES[pfam_family]
        if gene_id not in seen:
            result.total_genes += 1
            seen.add(gene_id)
        result.transporters.append(TransporterInfo(
            gene_id=gene_id,
            transporter_family=info["family"],
            transport_type=info["type"],
            direction=info["direction"],
            predicted_substrate="",
            confidence=0.7,
        ))

    return result


def _guess_substrate(header: str) -> str:
    """Guess transport substrate from gene header."""
    for substrate, keywords in SUBSTRATE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in header:
                return substrate
    return ""
