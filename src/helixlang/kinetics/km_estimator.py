"""Km estimator: heuristic Km prediction from protein properties (doc/20 §8.2)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Protocol


class KmModel(Protocol):
    """Protocol for Km ML prediction models."""

    def predict(self, sequence: str, substrate: str) -> float:
        """Predict Km from sequence and substrate."""
        ...


@dataclass
class KmEstimator:
    """Km estimation engine using heuristic and ML strategies.

    Strategy priority:
    1. BRENDA database lookup (exact EC + substrate + organism)
    2. Heuristic estimation from protein properties
    3. ML prediction (if available)
    4. Global median fallback
    """

    target_organism: str = "Escherichia coli"
    ml_model: KmModel | None = None

    # Curated median Km values by substrate class (mM).
    # From Bar-Even et al. 2011, Biochemistry; Drew & Bhatt 2022.
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

    def estimate(
        self,
        reaction_id: str,
        substrate: str = "",
        sequence: str = "",
        molecular_weight: float = 0.0,
    ) -> float:
        """Estimate Km for a substrate.

        Parameters
        ----------
        reaction_id : reaction identifier
        substrate : substrate name
        sequence : protein sequence
        molecular_weight : enzyme molecular weight (Da)

        Returns
        -------
        Km in mM
        """
        # Strategy 1: Known substrate median
        if substrate and substrate.lower() in self.SUBSTRATE_MEDIAN_KM:
            km = self.SUBSTRATE_MEDIAN_KM[substrate.lower()]
            # Adjust for organism (prokaryotes typically lower Km)
            if self.target_organism not in (
                "Escherichia coli",
                "Bacillus subtilis",
                "Saccharomyces cerevisiae",
                "Homo sapiens",
            ):
                km *= 1.5  # uncharacterized organisms get penalty
            return km

        # Strategy 2: Heuristic from sequence properties
        if sequence:
            return self._heuristic_km(sequence, molecular_weight)

        # Strategy 3: ML prediction
        if self.ml_model and sequence:
            return self.ml_model.predict(sequence, substrate)

        # Strategy 4: Global median (mM)
        return 0.5

    def _heuristic_km(
        self, sequence: str, molecular_weight: float
    ) -> float:
        """Heuristic Km estimation from protein properties.

        Based on the empirical relationship:
        - Km correlates with active site accessibility
        - Smaller proteins tend to have lower Km
        - Charge distribution affects substrate binding
        """
        if not sequence:
            return 0.5

        seq_upper = sequence.upper()
        length = len(seq_upper)

        # Hydrophobicity-based correction (Kyte-Doolittle)
        hydrophobic = sum(
            1 for aa in seq_upper
            if aa in "AILMFWV"
        )
        hydrophobicity = hydrophobic / max(length, 1)

        # Charge-based correction
        positive = sum(1 for aa in seq_upper if aa in "RKH")
        negative = sum(1 for aa in seq_upper if aa in "DE")
        net_charge = (positive - negative) / max(length, 1)

        # Base Km (mM) from global median
        base_km = 0.5

        # Size correction: smaller enzymes → lower Km
        if molecular_weight > 0:
            size_factor = math.exp(-(molecular_weight - 40000) / 60000)
        else:
            size_factor = 1.0

        # Hydrophobicity correction: hydrophobic active sites → lower Km
        hydrophobicity_factor = math.exp(-2.0 * (hydrophobicity - 0.4))

        # Charge correction: net positive → lower Km (better binding)
        charge_factor = math.exp(-0.5 * net_charge)

        km = base_km * size_factor * hydrophobicity_factor * charge_factor
        return max(0.01, min(10.0, km))  # clamp to [0.01, 10.0] mM


def estimate_km(
    reaction_id: str,
    substrate: str = "",
    sequence: str = "",
    molecular_weight: float = 0.0,
    target_organism: str = "Escherichia coli",
) -> float:
    """Convenience function for Km estimation.

    Parameters
    ----------
    reaction_id : reaction identifier
    substrate : substrate name
    sequence : protein sequence
    molecular_weight : enzyme molecular weight (Da)
    target_organism : target organism

    Returns
    -------
    Km in mM
    """
    estimator = KmEstimator(target_organism=target_organism)
    return estimator.estimate(
        reaction_id, substrate, sequence, molecular_weight
    )
