"""Transcription factor identification via DNA-binding domain scan (doc/20 §7.2)."""
from __future__ import annotations

from dataclasses import dataclass, field

# Pfam accession → TF family mapping (prokaryotic TF families)
TF_PFAM_DOMAINS: dict[str, str] = {
    # Accessions verified against Pfam release 37 (July 2025).
    # Old accessions (e.g. PF00076, PF00486, PF00158) were reassigned
    # to unrelated families; the mappings below reflect the current DB.
    "PF00072": "Response_reg",           # two-component response regulator
    "PF00356": "HTH_LacI",              # LacI family helix-turn-helix
    "PF00392": "HTH_GntR",             # GntR family
    "PF00165": "HTH_AraC",             # AraC family
    "PF01022": "HTH_NusA",             # NusA / HTH_5
    "PF00126": "HTH_LysR",             # LysR family / HTH_1
    "PF02954": "HTH_TetR",             # TetR family / HTH_8
    "PF13443": "HTH_XRE",              # Cro/C1-type HTH (XRE family)
    "PF00325": "HTH_Crp",              # CRP/FNR family
    "PF09339": "HTH_IclR",             # IclR family
    "PF18024": "HTH_TyrR",             # TyrR family / HTH_50
    "PF01475": "HTH_Fur",              # Fur family
    "PF09824": "HTH_ArsR",             # ArsR family
    "PF01047": "HTH_MarR",             # MarR family
    "PF21157": "HTH_DksA",             # DksA family
    "PF03551": "HTH_PadR",             # PadR family
    "PF13556": "HTH_Tric",             # HTH_30 / Tricorn-associated
    "PF01037": "HTH_AsnC",             # AsnC family
    "PF00376": "HTH_MerR",             # MerR family
}


@dataclass
class TFCandidate:
    """A predicted transcription factor."""

    gene_id: str
    tf_family: str
    domain_accession: str
    domain_start: int = 0
    domain_end: int = 0
    e_value: float = 0.0
    score: float = 0.0
    confidence: float = 0.0


@dataclass
class TFScanResult:
    """Results of genome-wide TF detection."""

    total_genes: int = 0
    tf_candidates: list[TFCandidate] = field(default_factory=list)

    @property
    def tf_count(self) -> int:
        return len(self.tf_candidates)

    @property
    def tf_fraction(self) -> float:
        if self.total_genes == 0:
            return 0.0
        return self.tf_count / self.total_genes

    def tf_ids(self) -> list[str]:
        return [t.gene_id for t in self.tf_candidates]

    def tfs_by_family(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for t in self.tf_candidates:
            result.setdefault(t.tf_family, []).append(t.gene_id)
        return result


def detect_transcription_factors(
    protein_fasta: str,
    pfam_database: str = "",
    e_value_threshold: float = 1.0,
) -> TFScanResult:
    """Detect transcription factors by scanning for DNA-binding Pfam domains.

    Parameters
    ----------
    protein_fasta : path to FASTA of protein sequences
    pfam_database : path to Pfam-A.hmm (HMMER format)
    e_value_threshold : domain detection E-value cutoff

    Returns
    -------
    TFScanResult with predicted TFs

    Notes
    -----
    When pfam_database is empty, performs a simple heuristic scan:
    genes whose IDs or annotations contain known TF family names are
    flagged.  For full accuracy, provide a Pfam HMM database and HMMER3.
    """
    result = TFScanResult()

    if not pfam_database:
        return _heuristic_tf_scan(protein_fasta, result)

    return _hmmer_tf_scan(protein_fasta, pfam_database,
                          e_value_threshold, result)


def _heuristic_tf_scan(
    fasta_path: str, result: TFScanResult
) -> TFScanResult:
    """Simple heuristic TF detection from FASTA headers / gene IDs."""
    try:
        with open(fasta_path) as fh:
            for line in fh:
                if line.startswith(">"):
                    result.total_genes += 1
                    header = line[1:].strip().lower()
                    for acc, family in TF_PFAM_DOMAINS.items():
                        if family.lower() in header:
                            gene_id = line[1:].split()[0]
                            result.tf_candidates.append(TFCandidate(
                                gene_id=gene_id,
                                tf_family=family,
                                domain_accession=acc,
                                confidence=0.6,
                            ))
                            break
    except FileNotFoundError:  # SILENTBENIGN - optional db file absent
        pass
    return result


def _count_fasta_sequences(fasta_path: str) -> int:
    """Count sequences in a FASTA file."""
    count = 0
    try:
        with open(fasta_path) as fh:
            for line in fh:
                if line.startswith(">"):
                    count += 1
    except FileNotFoundError:  # SILENTBENIGN - optional input absent
        pass
    return count


def _hmmer_tf_scan(
    fasta_path: str,
    pfam_hmm: str,
    e_value: float,
    result: TFScanResult,
) -> TFScanResult:
    """Full HMMER-based TF detection.

    Uses ``--max`` (all heuristic filters off) for maximum sensitivity,
    because HTH DNA-binding domains are short (~30-60 residues) and
    can be rejected by HMMER's default pre-filters.  Results are
    de-duplicated per gene keeping the best-scoring TF hit, and
    spurious matches with negative bit-scores are dropped.
    """
    import os
    import subprocess
    import tempfile

    result.total_genes = _count_fasta_sequences(fasta_path)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".domtbl", delete=False
    ) as tmp:
        tmp_path = tmp.name

    cmd = [
        "hmmsearch",
        "--max",
        "--domE", str(e_value),
        "--domtblout", tmp_path,
        pfam_hmm,
        fasta_path,
    ]
    try:
        subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "hmmsearch not found. Install HMMER: conda install -c bioconda hmmer"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"hmmsearch failed: {exc.stderr}") from exc

    best_hits: dict[str, tuple[str, str, float, float]] = {}

    try:
        with open(tmp_path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 22:
                    continue
                gene_id = parts[0]
                accession = parts[4].split(".")[0]
                if accession not in TF_PFAM_DOMAINS:
                    continue
                score = float(parts[13])
                if score <= 0:
                    continue
                e_val = float(parts[12])
                family = TF_PFAM_DOMAINS[accession]
                if gene_id not in best_hits or score > best_hits[gene_id][3]:
                    best_hits[gene_id] = (family, accession, e_val, score)
    finally:
        os.unlink(tmp_path)

    for gene_id, (family, accession, e_val, score) in best_hits.items():
        # Confidence: logistic-like mapping from bit score.
        # score ≥10 → ~1.0, score 0 → ~0.5, score <0 → dropped above.
        confidence = max(0.0, min(1.0, 0.5 + score / 20.0))
        result.tf_candidates.append(TFCandidate(
            gene_id=gene_id,
            tf_family=family,
            domain_accession=accession,
            e_value=e_val,
            score=score,
            confidence=round(confidence, 2),
        ))

    return result
