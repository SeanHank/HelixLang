"""ESM3-based 3D protein structure prediction (doc/26 Phase B).

Uses Facebook's ESM3 (Lin et al. 2023, Science 379:1123-1130) for end-to-end
protein structure prediction from sequence alone.  ESM3 uses evolutionary-scale
representations to predict 3D coordinates without requiring MSAs.

Falls back to Chou-Fasman (``protein_structure.py``) when ``esm`` is absent —
only when the caller explicitly opts into reduced fidelity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")

_KYTE_DOOHLITTLE: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}

_ESM_AVAILABLE = False
try:
    import esm  # noqa: F401
    import torch  # noqa: F401
    from esm.pretrained import ESM3_sm_open_v0  # noqa: F401
    _ESM_AVAILABLE = True
except ImportError:  # SILENTBENIGN - optional esm/torch capability probe
    pass

_model_cache: dict[str, Any] = {}


def is_available() -> bool:
    return _ESM_AVAILABLE


@dataclass
class TransmembraneHelix:
    start: int
    end: int
    hydropathy: float


@dataclass
class DisorderRegion:
    start: int
    end: int


@dataclass
class ProteinStructure3D:
    sequence: str
    coords: Any
    plddt: Any
    secondary_structure: str
    tm_helices: list[TransmembraneHelix] = field(default_factory=list)
    disorder: list[DisorderRegion] = field(default_factory=list)
    mean_plddt: float = 0.0
    ptm_score: float = 0.0


def _validate_sequence(sequence: str) -> str:
    seq = sequence.upper().strip()
    if not seq:
        raise ValueError("sequence must be non-empty")
    for i, aa in enumerate(seq):
        if aa not in _AMINO_ACIDS:
            raise ValueError(f"invalid amino acid {aa!r} at position {i}")
    return seq


def _get_model(model_name: str, device: str) -> Any:
    key = f"{model_name}:{device}"
    if key not in _model_cache:
        from esm.pretrained import ESM3_sm_open_v0
        _model_cache[key] = ESM3_sm_open_v0(device)
    return _model_cache[key]


def _derive_secondary_from_coords(coords: Any) -> str:
    import numpy as np

    if hasattr(coords, "detach"):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(coords)

    n = int(coords_np.shape[0])
    if n < 4:
        return "C" * n

    ca = coords_np[:, 0] if coords_np.ndim == 3 else coords_np
    if ca.ndim != 2 or ca.shape[1] != 3:
        return "C" * n

    ss = list("C" * n)
    for i in range(1, n - 1):
        v1 = ca[i] - ca[i - 1]
        v2 = ca[i + 1] - ca[i]
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        cos_angle = float(np.dot(v1, v2) / norm_product)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.degrees(math.acos(cos_angle))
        if 80 < angle < 150:
            ss[i] = "E"
        elif 40 < angle < 90:
            ss[i] = "H"
    return "".join(ss)


def _normalize_plddt(plddt: Any) -> Any:
    """Normalize plddt values to [0, 100] range regardless of input scale."""
    import numpy as np

    if hasattr(plddt, "detach"):
        arr = plddt.detach().cpu().numpy()
    else:
        arr = np.asarray(plddt, dtype=float)
    if arr.max() <= 1.0:
        arr = arr * 100.0
    return arr


def _detect_tm_helices(
    sequence: str,
    plddt: Any,
    window: int = 9,
    threshold: float = 1.6,
    min_len: int = 18,
    max_len: int = 30,
) -> list[TransmembraneHelix]:

    plddt_np = _normalize_plddt(plddt)

    seq = sequence.upper()
    n = len(seq)
    scores: list[float] = []
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        scores.append(
            sum(_KYTE_DOOHLITTLE.get(c, 0.0) for c in seq[start:end]) / (end - start)
        )

    helices: list[TransmembraneHelix] = []
    in_tm = False
    tm_start = 0
    for i, s in enumerate(scores):
        if s >= threshold and plddt_np[i] > 50.0:
            if not in_tm:
                in_tm = True
                tm_start = i
        else:
            if in_tm:
                length = i - tm_start
                if min_len <= length <= max_len:
                    avg_hyd = sum(scores[tm_start:i]) / length
                    helices.append(TransmembraneHelix(tm_start, i, avg_hyd))
                in_tm = False
    if in_tm:
        length = n - tm_start
        if min_len <= length <= max_len:
            avg_hyd = sum(scores[tm_start:]) / length
            helices.append(TransmembraneHelix(tm_start, n, avg_hyd))
    return helices


def _detect_disorder(plddt: Any, threshold: float = 50.0) -> list[DisorderRegion]:
    plddt_np = _normalize_plddt(plddt)

    regions: list[DisorderRegion] = []
    in_dis = False
    dis_start = 0
    for i, v in enumerate(plddt_np):
        if v < threshold:
            if not in_dis:
                in_dis = True
                dis_start = i
        else:
            if in_dis:
                regions.append(DisorderRegion(dis_start, i))
                in_dis = False
    if in_dis:
        regions.append(DisorderRegion(dis_start, len(plddt_np)))
    return regions


def predict_structure_esm(
    sequence: str,
    model_name: str = "esmfold_v1",
    device: str | None = None,
    num_steps: int = 10,
    max_residues: int = 700,
) -> ProteinStructure3D:
    """Predict 3D structure from amino acid sequence using ESM3.

    Parameters
    ----------
    sequence : amino acid sequence (standard 20 AAs + X)
    model_name : ESM3 model variant identifier
    device : compute device (auto-detect if None)
    num_steps : diffusion sampling steps (more = better quality)
    max_residues : truncate sequences longer than this

    Returns
    -------
    ProteinStructure3D with Ca coordinates, pLDDT, secondary structure,
    TM helix, and disorder annotations
    """
    if not _ESM_AVAILABLE:
        raise ImportError(  # noqa: TRY003 - third-party optional dep message
            "ESM3 structure prediction requires the optional 'esm' and 'torch' "
            "packages. Install them with: pip install 'helixlang[ml]'. To accept "
            "reduced-fidelity (Chou-Fasman) predictions instead, pass "
            "'--low-fidelity' (use NAME --low-fidelity)."
        )
    import numpy as np
    from esm.models.esm3 import ESMProtein, GenerationConfig

    seq = _validate_sequence(sequence)
    if len(seq) > max_residues:
        seq = seq[:max_residues]

    if device is None:
        device = "cuda" if _has_cuda() else "cpu"

    model = _get_model(model_name, device)
    protein = ESMProtein(sequence=seq)
    config = GenerationConfig(
        track="structure",
        schedule="cosine",
        strategy="random",
        num_steps=num_steps,
        temperature=1.0,
    )
    result = model.generate(protein, config)

    coords_raw = result.coordinates
    if hasattr(coords_raw, "detach"):
        coords_np = coords_raw.detach().cpu().numpy()
    else:
        coords_np = np.asarray(coords_raw)

    if coords_np.ndim == 3:
        ca_coords = coords_np[:, 0, :]
    elif coords_np.ndim == 2:
        ca_coords = coords_np
    else:
        ca_coords = np.zeros((len(seq), 3))

    plddt_raw = result.plddt
    if hasattr(plddt_raw, "detach"):
        plddt_np = plddt_raw.detach().cpu().numpy()
    else:
        plddt_np = np.asarray(plddt_raw)
    plddt_100 = plddt_np * 100.0

    ss = _derive_secondary_from_coords(ca_coords)
    tm_helices = _detect_tm_helices(seq, plddt_np)
    disorder = _detect_disorder(plddt_np)
    mean_plddt = float(np.mean(plddt_100))

    ptm_val = 0.0
    if result.ptm is not None:
        if hasattr(result.ptm, "mean"):
            ptm_val = float(result.ptm.mean().item())
        else:
            ptm_val = float(result.ptm)

    return ProteinStructure3D(
        sequence=seq,
        coords=ca_coords,
        plddt=plddt_100,
        secondary_structure=ss,
        tm_helices=tm_helices,
        disorder=disorder,
        mean_plddt=mean_plddt,
        ptm_score=ptm_val,
    )


def predict_structure_batch(
    sequences: list[str],
    model_name: str = "esmfold_v1",
    device: str | None = None,
    num_steps: int = 10,
    max_residues: int = 700,
) -> list[ProteinStructure3D]:
    """Batch structure prediction for multiple sequences."""
    return [
        predict_structure_esm(s, model_name, device, num_steps, max_residues)
        for s in sequences
    ]


def _has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


__all__ = [
    "DisorderRegion",
    "ProteinStructure3D",
    "TransmembraneHelix",
    "is_available",
    "predict_structure_batch",
    "predict_structure_esm",
]
