"""Protein fitness oracles for ML-guided directed evolution (T2.4, P8).

Zero-/few-shot fitness prediction that can guide rounds of mutagenesis
the way EVOLVEpro (Jiang et al. Science 2025 387:eadr6006), MULTI-evolve
(Tran et al. Science 2025 387:eadr1183) and FSFP (Otalora Ottó et al.
Nat Commun 2024 15:5566) do.  Two oracles are provided:

- :class:`BLOSUMOracle` — baseline substitution-matrix log-odds scoring
  (Henikoff & Henikoff 1992), zero dependencies.  A variant's fitness is
  the normalized sum of BLOSUM62 scores relative to the wild type; this is
  the classical *chemical-tolerance* baseline that PLMs are benchmarked
  against on DMS data (ProteinGym, Notin et al. 2024).
- :class:`ESM2Oracle` — optional zero-shot ESM-2 pseudo-likelihood oracle
  (Meier et al. PNAS 2021; Rives et al. PNAS 2021).  When ``transformers``
  + ``torch`` are installed, it masks each position, reads the model's
  marginal distribution over the masked token, and scores a variant with
  the wild-type-minus-variant log-likelihood difference (the Frazer et al.
  2021 / EVE zero-shot protocol).  Degrades gracefully (``available``
  False) when the optional extra is missing.

Both implement the :class:`FitnessOracle` protocol: ``score(reference,
variant) -> float`` with higher = fitter.  ``calculate_fitness`` in
:mod:`helixlang.plugins.runtime.evolution` can be pointed at any oracle through
``method="oracle"``.

References:
- Henikoff S & Henikoff JG. PNAS 1992 89:10915-10919 (BLOSUM matrices)
- Rives A et al. PNAS 2021 118:e2016239118 (ESM-1b)
- Meier J et al. PNAS 2021 118:e2016239118 (ESM-2 embeddings)
- Frazer J et al. PNAS 2021 118:e2012055118 (zero-shot variant effect)
- Notin P et al. Nat Commun 2024 15:5566 (ProteinGym DMS benchmark; FSFP)
- Jiang W et al. Science 2025 387:eadr6006 (EVOLVEpro)
- Tran H et al. Science 2025 387:eadr1183 (MULTI-evolve)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

#: canonical amino-acid ordering (alphabetical, matches BLOSUM62)
AA20: tuple[str, ...] = tuple("ARNDCQEGHILKMFPSTWYV")

#: BLOSUM62 substitution log-odds scores (Henikoff & Henikoff 1992).
#: BLOSUM62[a][b] = log-odds of observing a aligned to b, in half-bits.
BLOSUM62: dict[str, dict[str, int]] = {
    "A": {"A": 4, "R": -1, "N": -2, "D": -2, "C": 0, "Q": -1, "E": -1,
          "G": 0, "H": -2, "I": -1, "L": -1, "K": -1, "M": -1, "F": -2,
          "P": -1, "S": 1, "T": 0, "W": -3, "Y": -2, "V": 0},
    "R": {"A": -1, "R": 5, "N": 0, "D": -2, "C": -3, "Q": 1, "E": 0,
          "G": -2, "H": 0, "I": -3, "L": -2, "K": 2, "M": -1, "F": -3,
          "P": -2, "S": -1, "T": -1, "W": -3, "Y": -2, "V": -3},
    "N": {"A": -2, "R": 0, "N": 6, "D": 1, "C": -3, "Q": 0, "E": 0,
          "G": 0, "H": 1, "I": -3, "L": -3, "K": 0, "M": -2, "F": -3,
          "P": -2, "S": 1, "T": 0, "W": -4, "Y": -2, "V": -3},
    "D": {"A": -2, "R": -2, "N": 1, "D": 6, "C": -3, "Q": 0, "E": 2,
          "G": -1, "H": -1, "I": -3, "L": -4, "K": -1, "M": -3, "F": -3,
          "P": -1, "S": 0, "T": -1, "W": -4, "Y": -3, "V": -3},
    "C": {"A": 0, "R": -3, "N": -3, "D": -3, "C": 9, "Q": -3, "E": -4,
          "G": -3, "H": -3, "I": -1, "L": -1, "K": -3, "M": -1, "F": -2,
          "P": -3, "S": -1, "T": -1, "W": -2, "Y": -2, "V": -1},
    "Q": {"A": -1, "R": 1, "N": 0, "D": 0, "C": -3, "Q": 5, "E": 2,
          "G": -2, "H": 0, "I": -3, "L": -2, "K": 1, "M": 0, "F": -3,
          "P": -1, "S": 0, "T": -1, "W": -2, "Y": -1, "V": -2},
    "E": {"A": -1, "R": 0, "N": 0, "D": 2, "C": -4, "Q": 2, "E": 5,
          "G": -2, "H": 0, "I": -3, "L": -3, "K": 1, "M": -2, "F": -3,
          "P": -1, "S": 0, "T": -1, "W": -3, "Y": -2, "V": -2},
    "G": {"A": 0, "R": -2, "N": 0, "D": -1, "C": -3, "Q": -2, "E": -2,
          "G": 6, "H": -2, "I": -4, "L": -4, "K": -2, "M": -3, "F": -3,
          "P": -2, "S": 0, "T": -2, "W": -2, "Y": -3, "V": -3},
    "H": {"A": -2, "R": 0, "N": 1, "D": -1, "C": -3, "Q": 0, "E": 0,
          "G": -2, "H": 8, "I": -3, "L": -3, "K": -1, "M": -2, "F": -1,
          "P": -2, "S": -1, "T": -2, "W": -2, "Y": 2, "V": -3},
    "I": {"A": -1, "R": -3, "N": -3, "D": -3, "C": -1, "Q": -3, "E": -3,
          "G": -4, "H": -3, "I": 4, "L": 2, "K": -3, "M": 1, "F": 0,
          "P": -3, "S": -2, "T": -1, "W": -3, "Y": -1, "V": 3},
    "L": {"A": -1, "R": -2, "N": -3, "D": -4, "C": -1, "Q": -2, "E": -3,
          "G": -4, "H": -3, "I": 2, "L": 4, "K": -2, "M": 2, "F": 0,
          "P": -3, "S": -2, "T": -1, "W": -2, "Y": -1, "V": 1},
    "K": {"A": -1, "R": 2, "N": 0, "D": -1, "C": -3, "Q": 1, "E": 1,
          "G": -2, "H": -1, "I": -3, "L": -2, "K": 5, "M": -1, "F": -3,
          "P": -1, "S": 0, "T": -1, "W": -3, "Y": -2, "V": -2},
    "M": {"A": -1, "R": -1, "N": -2, "D": -3, "C": -1, "Q": 0, "E": -2,
          "G": -3, "H": -2, "I": 1, "L": 2, "K": -1, "M": 5, "F": 0,
          "P": -2, "S": -1, "T": -1, "W": -1, "Y": -1, "V": 1},
    "F": {"A": -2, "R": -3, "N": -3, "D": -3, "C": -2, "Q": -3, "E": -3,
          "G": -3, "H": -1, "I": 0, "L": 0, "K": -3, "M": 0, "F": 6,
          "P": -4, "S": -2, "T": -2, "W": 1, "Y": 3, "V": -1},
    "P": {"A": -1, "R": -2, "N": -2, "D": -1, "C": -3, "Q": -1, "E": -1,
          "G": -2, "H": -2, "I": -3, "L": -3, "K": -1, "M": -2, "F": -4,
          "P": 7, "S": -1, "T": -1, "W": -4, "Y": -3, "V": -2},
    "S": {"A": 1, "R": -1, "N": 1, "D": 0, "C": -1, "Q": 0, "E": 0,
          "G": 0, "H": -1, "I": -2, "L": -2, "K": 0, "M": -1, "F": -2,
          "P": -1, "S": 4, "T": 1, "W": -3, "Y": -2, "V": -2},
    "T": {"A": 0, "R": -1, "N": 0, "D": -1, "C": -1, "Q": -1, "E": -1,
          "G": -2, "H": -2, "I": -1, "L": -1, "K": -1, "M": -1, "F": -2,
          "P": -1, "S": 1, "T": 5, "W": -2, "Y": -2, "V": 0},
    "W": {"A": -3, "R": -3, "N": -4, "D": -4, "C": -2, "Q": -2, "E": -3,
          "G": -2, "H": -2, "I": -3, "L": -2, "K": -3, "M": -1, "F": 1,
          "P": -4, "S": -3, "T": -2, "W": 11, "Y": 2, "V": -3},
    "Y": {"A": -2, "R": -2, "N": -2, "D": -3, "C": -2, "Q": -1, "E": -2,
          "G": -3, "H": 2, "I": -1, "L": -1, "K": -2, "M": -1, "F": 3,
          "P": -3, "S": -2, "T": -2, "W": 2, "Y": 7, "V": -1},
    "V": {"A": 0, "R": -3, "N": -3, "D": -3, "C": -1, "Q": -2, "E": -2,
          "G": -3, "H": -3, "I": 3, "L": 1, "K": -2, "M": 1, "F": -1,
          "P": -2, "S": -2, "T": 0, "W": -3, "Y": -1, "V": 4},
}

#: minimum / maximum BLOSUM62 score per residue (for normalization)
_BLOSUM_MIN = {aa: min(row.values()) for aa, row in BLOSUM62.items()}


def _validate(sequence: str, name: str) -> None:
    if not sequence:
        raise ValueError(f"{name} must be a non-empty protein sequence")
    for i, aa in enumerate(sequence):
        if aa not in BLOSUM62:
            raise ValueError(f"invalid amino acid {aa!r} at {name} position {i}")


def blosum62_raw(reference: str, variant: str) -> float:
    """Sum of BLOSUM62 log-odds scores of ``variant`` vs ``reference``.

    Only the identities of the two aligned sequences matter (equal
    length required); each position contributes ``BLOSUM62[r_i][v_i]``.
    """
    if len(reference) != len(variant):
        raise ValueError("reference and variant must have equal length")
    _validate(reference, "reference")
    _validate(variant, "variant")
    return float(sum(BLOSUM62[r][v]
                     for r, v in zip(reference, variant, strict=True)))


def blosum62_normalized(reference: str, variant: str) -> float:
    """BLOSUM62 score normalized to [0, 1] (identity = 1.0).

    ``raw`` is scaled by the best possible score (all identities) and the
    worst possible score (each position mutates to the residue with the
    lowest BLOSUM score against the reference residue).
    """
    if len(reference) != len(variant):
        raise ValueError("reference and variant must have equal length")
    _validate(reference, "reference")
    _validate(variant, "variant")
    raw = 0.0
    best = 0.0
    worst = 0.0
    for r, v in zip(reference, variant, strict=True):
        raw += BLOSUM62[r][v]
        best += BLOSUM62[r][r]
        worst += _BLOSUM_MIN[r]
    if best == worst:
        return 1.0
    return max(0.0, min(1.0, (raw - worst) / (best - worst)))


class FitnessOracle(Protocol):
    """Pluggable fitness oracle: ``score(reference, variant) -> float``.

    Higher score = fitter variant; the reference is the wild type.
    """

    def score(self, reference: str, variant: str) -> float: ...

    @property
    def available(self) -> bool:
        """Whether this oracle can actually score (False when the
        optional model/dependency is missing)."""
        ...


@dataclass(slots=True)
class BLOSUMOracle:
    """BLOSUM62 baseline fitness oracle (no dependencies)."""

    def score(self, reference: str, variant: str) -> float:
        return blosum62_normalized(reference, variant)

    @property
    def available(self) -> bool:
        return True


def _load_esm(model_name: str) -> tuple[Any, Any]:
    """Lazily import torch + transformers and load the ESM-2 model.

    Returns ``(model, tokenizer)`` or raises ImportError.
    """
    try:
        import torch  # noqa: F401
        from transformers import EsmForMaskedLM, EsmTokenizer
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "ESM2Oracle requires the optional 'transformers' and 'torch' "
            "packages; install them or use BLOSUMOracle") from exc
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


class ESM2Oracle:
    """Zero-shot ESM-2 pseudo-likelihood fitness oracle (optional).

    Masks each residue in turn, reads the model's predictive distribution
    at that position, and returns the per-residue log-likelihood of the
    true residue (``sum_i log P(aa_i | context)``).  Variants are scored
    by the variant-minus-reference difference (Frazer et al. 2021
    zero-shot protocol).

    ``available`` is False when ``transformers``/``torch`` are not
    installed; scoring then raises a clear error.
    """

    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D",
                 device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._load_error: Exception | None = None
        self._ref_cache: dict[str, float] = {}
        try:
            self._model, self._tokenizer = _load_esm(model_name)
            if self.device is None:
                self.device = "cuda" if self._has_cuda() else "cpu"
            self._model = self._model.to(self.device)
        except Exception as exc:  # pragma: no cover - optional extra
            self._load_error = exc

    @staticmethod
    def _has_cuda() -> bool:  # pragma: no cover - optional extra
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return self._model is not None

    def _log_probs(self, sequence: str) -> list[float]:
        """Per-position log-likelihoods of the true residues."""
        import torch  # noqa: F401
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None
        seq = sequence.replace("X", "A")  # ESM-2 has no X token
        log_probs: list[float] = []
        with torch.no_grad():
            for i in range(len(seq)):
                masked = seq[:i] + "<mask>" + seq[i + 1:]
                tokens = tokenizer(masked, return_tensors="pt")
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                out = model(**tokens)
                probs = torch.softmax(out.logits[0, i + 1], dim=-1)
                token_id = tokenizer.convert_tokens_to_ids(seq[i])
                log_probs.append(math.log(float(probs[token_id].item()) + 1e-9))
        return log_probs

    def pseudo_log_likelihood(self, sequence: str) -> float:
        """Sum over residues of masked-marginal log-likelihoods.

        Results are cached per sequence to avoid redundant forward passes
        when scoring many variants against the same reference.
        """
        if not self.available:
            raise RuntimeError(
                f"ESM2Oracle unavailable ({self.model_name}): {self._load_error}")
        if sequence not in self._ref_cache:
            self._ref_cache[sequence] = float(sum(self._log_probs(sequence)))
        return self._ref_cache[sequence]

    def score(self, reference: str, variant: str) -> float:
        """Zero-shot variant effect: loglik(variant) - loglik(wt).

        Positive means the variant has higher pseudo-likelihood than
        the wild type (predicted fitter), matching the BLOSUMOracle
        convention where higher score = better.
        """
        if len(reference) != len(variant):
            raise ValueError("reference and variant must have equal length")
        return self.pseudo_log_likelihood(variant) - \
            self.pseudo_log_likelihood(reference)


# ============================================================================
# Dispatch + evolution integration
# ============================================================================

def oracle_score(reference: str, variant: str,
                 oracle: FitnessOracle | str | None = None) -> float:
    """Score a protein variant against a reference with a named oracle.

    ``oracle`` may be:
    - a :class:`FitnessOracle` instance,
    - ``"blosum62"`` (default) -> :class:`BLOSUMOracle`,
    - an :class:`ESM2Oracle` instance (or ``"esm2"``) for the optional
      PLM path.
    """
    if oracle is None or oracle == "blosum62":
        return BLOSUMOracle().score(reference, variant)
    if oracle == "esm2":
        esm = ESM2Oracle()
        if not esm.available:
            raise RuntimeError(
                "esm2 oracle requested but transformers/torch are not "
                "installed; fall back to 'blosum62'")
        return esm.score(reference, variant)
    if isinstance(oracle, str):
        raise ValueError(f"unknown oracle {oracle!r}; available: blosum62, esm2")
    return float(oracle.score(reference, variant))


def rank_variants(reference: str, variants: list[str],
                  oracle: FitnessOracle | str | None = None,
                  reverse: bool = False) -> list[tuple[str, float]]:
    """Rank a panel of variants by oracle fitness (descending by default).

    Returns ``[(variant, score), ...]`` sorted so the first entry is the
    recommended next round of mutagenesis.
    """
    scored = [(v, oracle_score(reference, v, oracle)) for v in variants]
    scored.sort(key=lambda t: t[1], reverse=not reverse)
    return scored


def protein_to_dna(protein: str) -> str:
    """Translate a protein sequence back to one coding DNA path."""
    from helixlang.plugins.runtime.biocodec import back_translate
    return back_translate(protein)


def dna_fitness(dna: str, reference_dna: str,
                oracle: FitnessOracle | str | None = None) -> float:
    """Fitness of a coding DNA variant vs a reference coding DNA.

    Both sequences are translated to protein (standard code) and scored
    with the oracle.  Requires complete coding frames (lengths that are
    multiples of three, no premature stops handled by back-translation).
    """
    from helixlang.plugins.runtime.dna_codec import translate_dna
    protein = translate_dna(dna)
    ref_protein = translate_dna(reference_dna)
    return oracle_score(ref_protein, protein, oracle)


__all__ = [
    "AA20", "BLOSUM62", "FitnessOracle",
    "BLOSUMOracle", "ESM2Oracle",
    "blosum62_raw", "blosum62_normalized",
    "oracle_score", "rank_variants",
    "protein_to_dna", "dna_fitness",
]
