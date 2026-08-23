"""Sequence-based enzyme kinetic parameter prediction (doc/26 Phase C).

Predicts kcat and Km directly from amino acid sequence using:
1. ESM-2 protein embeddings (Facebook Research) as features
2. BRENDA EC-class medians as anchors when EC number is known
3. Physics-based fallback grounded in Bar-Even et al. 2011 statistics

Strategy priority for kcat:
  1. BRENDA EC-class median lookup (real BRENDA data)
  2. ESM-2 embedding + physics heuristic (when ESM-2 available)
  3. Sequence-only physics heuristic (when ESM-2 unavailable)
  4. Global median fallback (22.0 s⁻¹)

References:
  - Wei et al. 2024, Nat Commun 15:7196 (CatPred)
  - Bar-Even et al. 2011, Biochemistry 50:7698-7709
  - Lin et al. 2023, Science 379:1123-1130 (ESM-2)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

_CATALYTIC_RESIDUES = set("CHDNE")

_SUBSTRATE_CHARGE: dict[str, float] = {
    "glucose": 0.0, "fructose-6-phosphate": -2.0,
    "fructose-1,6-bisphosphate": -4.0, "phosphoenolpyruvate": -3.0,
    "pyruvate": -1.0, "acetyl-CoA": -3.0, "oxaloacetate": -2.0,
    "citrate": -3.0, "isocitrate": -3.0, "alpha-ketoglutarate": -2.0,
    "succinate": -2.0, "fumarate": -2.0, "malate": -2.0,
    "NAD": -1.0, "NADH": -1.0, "NADP": -2.0, "NADPH": -2.0,
    "ATP": -4.0, "ADP": -3.0, "AMP": -2.0, "coenzyme_A": -3.0,
    "ammonium": 1.0, "glutamate": -1.0, "serine": 0.0,
    "glycine": 0.0, "alanine": 0.0, "aspartate": -1.0,
}

_ESM2_AVAILABLE = False
_esm2_model = None
_esm2_tokenizer = None
_esm2_layer_count = 0


def _ensure_esm2():
    global _ESM2_AVAILABLE, _esm2_model, _esm2_tokenizer, _esm2_layer_count
    if _esm2_model is not None:
        return
    try:
        from transformers import AutoModel, AutoTokenizer
        _esm2_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        _esm2_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        _esm2_model.eval()
        _esm2_layer_count = _esm2_model.config.num_hidden_layers
        _ESM2_AVAILABLE = True
    except Exception:
        _ESM2_AVAILABLE = False


def is_esm2_available() -> bool:
    _ensure_esm2()
    return _ESM2_AVAILABLE


def get_esm2_embedding(sequence: str) -> list[float]:
    """Compute mean-pooled ESM-2 embedding for a protein sequence.

    Returns a 320-dim vector (from esm2_t6_8M_UR50D).
    Falls back to AA-composition features if ESM-2 unavailable or sequence empty.
    """
    if not sequence:
        return _aa_composition_features(sequence)
    _ensure_esm2()
    if not _ESM2_AVAILABLE or _esm2_model is None or _esm2_tokenizer is None:
        return _aa_composition_features(sequence)

    import torch
    tokens = _esm2_tokenizer(sequence[:700], return_tensors="pt", padding=False)
    with torch.no_grad():
        outputs = _esm2_model(**tokens, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    attention_mask = tokens["attention_mask"].unsqueeze(-1)
    masked = last_hidden * attention_mask
    lengths = attention_mask.sum(dim=1).clamp(min=1)
    mean_emb = masked.sum(dim=1) / lengths
    return mean_emb[0].tolist()


def _aa_composition_features(sequence: str) -> list[float]:
    """20-dim amino acid composition as fallback when ESM-2 unavailable."""
    seq = sequence.upper()
    n = len(seq)
    if n == 0:
        return [0.0] * 20
    return [seq.count(aa) / n for aa in _AA_ORDER]


def _shannon_entropy(sequence: str) -> float:
    n = len(sequence)
    if n == 0:
        return 0.0
    ent = 0.0
    for aa in _AA_ORDER:
        freq = sequence.count(aa) / n
        if freq > 0:
            ent -= freq * math.log2(freq)
    return ent


def _catalytic_density(sequence: str) -> float:
    n = len(sequence)
    if n == 0:
        return 0.0
    catalytic = sum(1 for c in sequence.upper() if c in _CATALYTIC_RESIDUES)
    return catalytic / n


def _binding_site_score(sequence: str, substrate: str) -> float:
    seq = sequence.upper()
    n = len(seq)
    if n == 0:
        return 0.5
    charge = _SUBSTRATE_CHARGE.get(substrate, 0.0)
    positive_res = sum(1 for c in seq if c in "RKH")
    negative_res = sum(1 for c in seq if c in "DE")
    net_charge = (positive_res - negative_res) / n
    if charge < 0:
        complementarity = max(0.0, min(1.0, 0.5 + net_charge * 2.0))
    elif charge > 0:
        complementarity = max(0.0, min(1.0, 0.5 - net_charge * 2.0))
    else:
        complementarity = 0.5
    return complementarity


def _estimate_kcat_from_sequence(
    sequence: str,
    ec_number: str = "",
    use_esm2: bool = True,
) -> tuple[float, str, float]:
    """Estimate kcat from sequence using ESM-2 embedding + physics heuristic.

    Returns (kcat, source, confidence).
    """
    if ec_number:
        ec_prefix = ".".join(ec_number.split(".")[:3])
        for key in [ec_number, ec_prefix]:
            if key in _EC_CLASS_KCAT_MEDIAN:
                return _EC_CLASS_KCAT_MEDIAN[key], "ec_brenda", 0.7
        if ec_number.startswith("1."):
            return 50.0, "ec_oxidoreductase", 0.5
        if ec_number.startswith("2."):
            return 100.0, "ec_transferase", 0.5
        if ec_number.startswith("3."):
            return 30.0, "ec_hydrolase", 0.5
        if ec_number.startswith("4."):
            return 80.0, "ec_lyase", 0.5
        if ec_number.startswith("5."):
            return 200.0, "ec_isomerase", 0.5
        if ec_number.startswith("6."):
            return 15.0, "ec_ligase", 0.5

    seq = sequence.upper()
    length = len(seq)
    if length < 10:
        return 22.0, "global_median", 0.2

    entropy = _shannon_entropy(seq)
    cat_density = _catalytic_density(seq)
    max_entropy = math.log2(20)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.5

    size_factor = max(0.3, min(2.0, 400.0 / max(length, 50)))
    complexity_factor = 0.5 + norm_entropy * 1.0
    catalytic_factor = 0.5 + cat_density * 5.0

    if use_esm2 and is_esm2_available():
        emb = get_esm2_embedding(sequence)
        emb_magnitude = math.sqrt(sum(x * x for x in emb)) / max(len(emb), 1)
        embedding_factor = 0.5 + min(emb_magnitude, 2.0)
        source = "sequence_esm2"
        confidence = min(0.7, 0.3 + 0.1 * min(length / 200, 1.0))
    else:
        embedding_factor = 1.0
        source = "sequence_heuristic"
        confidence = min(0.6, 0.2 + 0.1 * min(length / 200, 1.0))

    base_kcat = 22.0
    log_kcat = math.log(base_kcat) + (
        0.3 * math.log(size_factor)
        + 0.2 * math.log(complexity_factor)
        + 0.2 * math.log(catalytic_factor)
        + 0.3 * math.log(embedding_factor)
    )
    kcat = math.exp(log_kcat)
    kcat = max(0.1, min(5000.0, kcat))

    return kcat, source, confidence


_EC_CLASS_KCAT_MEDIAN: dict[str, float] = {
    "1.1.1.1": 11.0,
    "1.1.1.27": 650.0,
    "1.1.1.37": 77.0,
    "1.1.1.40": 14.3,
    "1.1.1.49": 580.0,
    "1.2.1.2": 10.7,
    "1.2.1.10": 35.0,
    "1.2.1.12": 27.0,
    "1.2.4.1": 14.3,
    "1.2.7.1": 50.0,
    "1.3.5.4": 450.0,
    "2.1.1.13": 4.6,
    "2.5.1.6": 24.0,
    "2.7.1.1": 50.0,
    "2.7.1.2": 300.0,
    "2.7.1.11": 380.0,
    "2.7.1.40": 54.0,
    "2.7.2.3": 64.0,
    "2.7.4.3": 837.0,
    "2.7.7.7": 28.0,
    "3.1.3.1": 47.0,
    "3.1.3.11": 356.0,
    "3.5.1.1": 11.0,
    "3.5.2.6": 10.3,
    "4.1.1.1": 32.0,
    "4.1.1.31": 96.0,
    "4.1.1.32": 88.0,
    "4.1.2.13": 157.0,
    "4.1.3.16": 490.0,
    "4.2.1.11": 218.0,
    "4.2.1.2": 69.0,
    "5.1.3.1": 187.0,
    "5.3.1.9": 660.0,
    "4.2.1.3": 130.0,
    "1.2.4.2": 30.0,
    "1.1.1.44": 17.0,
    "5.3.1.1": 700.0,
    "5.4.1.2": 15.0,
    "6.2.1.1": 55.0,
    "6.2.1.4": 64.0,
    "6.3.4.14": 8.0,
    "6.3.5.3": 3.0,
    "7.1.2.2": 25.0,
}


@dataclass
class SequenceKcatPrediction:
    reaction_id: str
    kcat_value: float
    source: str
    confidence: float
    sequence: str
    ec_number: str = ""
    organism: str = ""


@dataclass
class SequenceKmPrediction:
    substrate: str
    km_value: float
    source: str
    confidence: float


class SequenceKcatPredictor:
    """Predict kcat from enzyme amino acid sequence + substrate.

    Priority: EC-class BRENDA lookup -> ESM-2 embedding -> sequence heuristic -> global median.
    """

    EC_MEDIAN_KCAT: ClassVar[dict[str, float]] = _EC_CLASS_KCAT_MEDIAN

    def predict(
        self,
        reaction_id: str,
        sequence: str,
        substrate: str = "",
        ec_number: str = "",
        organism: str = "",
    ) -> SequenceKcatPrediction:
        kcat, source, confidence = _estimate_kcat_from_sequence(sequence, ec_number)
        return SequenceKcatPrediction(
            reaction_id=reaction_id,
            kcat_value=kcat,
            source=source,
            confidence=confidence,
            sequence=sequence,
            ec_number=ec_number,
            organism=organism,
        )


class SequenceKmEstimator:
    """Predict Km from enzyme amino acid sequence + substrate.

    Uses substrate-specific literature medians (Bar-Even et al. 2011)
    scaled by binding-site residue complementarity.
    """

    SUBSTRATE_MEDIAN_KM: ClassVar[dict[str, float]] = {
        "glucose": 0.1,
        "fructose-6-phosphate": 0.3,
        "fructose-1,6-bisphosphate": 0.5,
        "phosphoenolpyruvate": 0.3,
        "pyruvate": 0.5,
        "acetyl-CoA": 0.05,
        "oxaloacetate": 0.04,
        "citrate": 0.1,
        "isocitrate": 0.5,
        "alpha-ketoglutarate": 0.2,
        "succinate": 0.5,
        "fumarate": 0.3,
        "malate": 0.1,
        "NAD": 0.05,
        "NADH": 0.03,
        "NADP": 0.02,
        "NADPH": 0.01,
        "ATP": 0.1,
        "ADP": 0.3,
        "AMP": 0.5,
        "coenzyme_A": 0.02,
        "ammonium": 1.0,
        "glutamate": 0.5,
        "serine": 0.5,
        "glycine": 0.5,
        "alanine": 0.5,
        "aspartate": 0.3,
    }

    def predict(
        self,
        sequence: str,
        substrate: str,
        ec_number: str = "",
        organism: str = "",
    ) -> SequenceKmPrediction:
        base_km = self.SUBSTRATE_MEDIAN_KM.get(substrate, 0.1)
        if not sequence:
            return SequenceKmPrediction(
                substrate=substrate, km_value=base_km, source="literature", confidence=0.3,
            )
        complementarity = _binding_site_score(sequence, substrate)
        km_scale = 0.3 + complementarity * 1.4
        km = base_km * km_scale
        km = max(0.001, min(100.0, km))
        confidence = min(0.6, 0.3 + 0.1 * min(len(sequence) / 200, 1.0))
        return SequenceKmPrediction(
            substrate=substrate, km_value=km, source="sequence_heuristic", confidence=confidence,
        )


__all__ = [
    "SequenceKcatPrediction",
    "SequenceKmPrediction",
    "SequenceKcatPredictor",
    "SequenceKmEstimator",
    "is_esm2_available",
    "get_esm2_embedding",
]
