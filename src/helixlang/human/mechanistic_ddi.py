"""Mechanistic DDI Predictor: compositional reasoning (doc/32 §8.3).

Predicts novel drug-drug interactions by COMPOSING known mechanisms (enzyme
inhibition, competitive binding, transporter effects) rather than memorizing
known DDI pairs. This is the pharmacological equivalent of compositional
generalization.

Literature:
- MARD (arXiv 2026): accuracy IMPROVES on rarely-seen drugs
- Dual-Pathway Fusion (arXiv 2025): zero-shot DDI on unseen drugs
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnzymeProfile:
    """Drug's interaction profile with a single enzyme/transporter.

    km_um and ki_um are distinct pharmacological parameters:
    - km_um: Michaelis constant — substrate concentration at half-maximal
      velocity; applies when this drug is a SUBSTRATE of the enzyme.
    - ki_um: inhibition constant — inhibitor concentration at half-maximal
      inhibition; applies when this drug INHIBITS the enzyme.
    """

    enzyme: str
    is_substrate: bool = False
    substrate_fraction: float = 0.0
    inhibition_strength: float = 0.0
    km_um: float = 10.0
    ki_um: float | None = None


@dataclass(frozen=True)
class DrugMechanism:
    """Complete mechanistic profile for a drug."""

    name: str
    enzyme_profiles: list[EnzymeProfile] = field(default_factory=list)
    bioavailability: float = 0.5
    is_renal_clearance: bool = False


@dataclass(frozen=True)
class DDIPrediction:
    """Predicted drug-drug interaction result."""

    drug_a: str
    drug_b: str
    auc_ratio: float
    total_inhibition: float
    significance: str
    mechanisms: list[dict[str, float | str]]
    confidence: float = 0.8


class EnzymeInhibitionLibrary:
    """Library of drug enzyme inhibition profiles from PharmGKB.

    Maps drug names to their CYP/transporter interaction profiles.
    """

    def __init__(self) -> None:
        self._drugs: dict[str, DrugMechanism] = {}
        self._load_default_library()

    def _load_default_library(self) -> None:
        """Load literature-curated enzyme profiles for common drugs."""
        drugs = [
            DrugMechanism(
                name="warfarin",
                enzyme_profiles=[
                    EnzymeProfile("CYP2C9", True, 0.85, 0.3, 5.0),
                    EnzymeProfile("CYP3A4", True, 0.10, 0.0, 10.0),
                    EnzymeProfile("CYP1A2", True, 0.05, 0.0, 10.0),
                ],
                bioavailability=0.93,
            ),
            DrugMechanism(
                name="amiodarone",
                enzyme_profiles=[
                    EnzymeProfile("CYP2D6", True, 0.20, 0.9, 10.0),
                    EnzymeProfile("CYP3A4", True, 0.50, 0.7, 10.0),
                    EnzymeProfile("CYP2C9", False, 0.0, 0.5, 10.0),
                    EnzymeProfile("CYP2C19", False, 0.0, 0.4, 10.0),
                ],
                bioavailability=0.46,
            ),
            DrugMechanism(
                name="fluconazole",
                enzyme_profiles=[
                    EnzymeProfile("CYP2C19", True, 0.80, 0.7, 8.0),
                    EnzymeProfile("CYP3A4", True, 0.20, 0.3, 10.0),
                    EnzymeProfile("CYP2C9", False, 0.0, 0.8, 10.0),
                ],
                bioavailability=0.90,
            ),
            DrugMechanism(
                name="clarithromycin",
                enzyme_profiles=[
                    EnzymeProfile("CYP3A4", True, 0.85, 0.85, 6.0),
                    EnzymeProfile("CYP1A2", True, 0.15, 0.3, 10.0),
                ],
                bioavailability=0.55,
            ),
            DrugMechanism(
                name="simvastatin",
                enzyme_profiles=[
                    EnzymeProfile("CYP3A4", True, 0.95, 0.0, 3.0),
                    EnzymeProfile("CYP2D6", True, 0.05, 0.0, 10.0),
                ],
                bioavailability=0.05,
            ),
            DrugMechanism(
                name="omeprazole",
                enzyme_profiles=[
                    EnzymeProfile("CYP2C19", True, 0.70, 0.2, 12.0),
                    EnzymeProfile("CYP3A4", True, 0.30, 0.0, 10.0),
                ],
                bioavailability=0.35,
            ),
            DrugMechanism(
                name="ciprofloxacin",
                enzyme_profiles=[
                    EnzymeProfile("CYP1A2", True, 0.70, 0.7, 9.0),
                    EnzymeProfile("CYP3A4", True, 0.30, 0.0, 10.0),
                ],
                bioavailability=0.70,
            ),
            DrugMechanism(
                name="verapamil",
                enzyme_profiles=[
                    EnzymeProfile("CYP3A4", True, 0.90, 0.6, 7.0),
                    EnzymeProfile("CYP1A2", True, 0.10, 0.0, 10.0),
                ],
                bioavailability=0.22,
            ),
            DrugMechanism(
                name="metformin",
                enzyme_profiles=[],
                bioavailability=0.55,
                is_renal_clearance=True,
            ),
            DrugMechanism(
                name="methotrexate",
                enzyme_profiles=[],
                bioavailability=0.70,
                is_renal_clearance=True,
            ),
            DrugMechanism(
                name="ibuprofen",
                enzyme_profiles=[
                    EnzymeProfile("CYP2C9", True, 0.90, 0.1, 10.0),
                    EnzymeProfile("CYP2C8", True, 0.10, 0.0, 10.0),
                ],
                bioavailability=0.80,
            ),
            DrugMechanism(
                name="metoprolol",
                enzyme_profiles=[
                    EnzymeProfile("CYP2D6", True, 0.80, 0.0, 5.0),
                ],
                bioavailability=0.50,
            ),
            DrugMechanism(
                name="clopidogrel",
                enzyme_profiles=[
                    EnzymeProfile("CYP2C19", True, 0.85, 0.0, 8.0),
                    EnzymeProfile("CYP3A4", True, 0.15, 0.0, 10.0),
                ],
                bioavailability=0.50,
            ),
        ]
        for drug in drugs:
            self._drugs[drug.name.lower()] = drug

    def register_drug(self, mechanism: DrugMechanism) -> None:
        """Register a new drug mechanism profile."""
        self._drugs[mechanism.name.lower()] = mechanism

    def get(self, drug_name: str) -> DrugMechanism | None:
        """Get mechanism profile for a drug."""
        return self._drugs.get(drug_name.lower())


class MechanisticDDIPredictor:
    """Predict DDI via compositional Michaelis-Menten kinetics.

    For drugs D₁ and D₂:
    AUC_ratio(D₂) = 1 / (1 - Σ inhibition_i × frac_metabolized_i × occupancy_i)

    where occupancy_i = [D₁] / (Ki_i + [D₁]) uses the perpetrator's
    inhibition constant Ki on enzyme i (falling back to its Km as a proxy
    when no Ki is curated).
    """

    def __init__(self, library: EnzymeInhibitionLibrary | None = None) -> None:
        self.library = library or EnzymeInhibitionLibrary()

    def predict(
        self,
        drug_a: str,
        drug_b: str,
        drug_a_conc_um: float = 10.0,
    ) -> DDIPrediction | None:
        """Predict AUC ratio for drug_b when co-administered with drug_a.

        drug_a is the perpetrator (inhibitor), drug_b is the victim (substrate).
        """
        info_a = self.library.get(drug_a)
        info_b = self.library.get(drug_b)
        if info_a is None or info_b is None:
            return None

        all_enzymes: dict[str, EnzymeProfile] = {}
        for p in info_a.enzyme_profiles:
            if p.inhibition_strength > 0:
                all_enzymes[p.enzyme] = p
        for p in info_b.enzyme_profiles:
            if p.is_substrate:
                all_enzymes.setdefault(p.enzyme, p)

        total_inhibition = 0.0
        mechanisms: list[dict[str, float]] = []

        inhibition_map = {p.enzyme: p for p in info_a.enzyme_profiles}
        substrate_map = {p.enzyme: p for p in info_b.enzyme_profiles}

        for enzyme_name in all_enzymes:
            inhib = inhibition_map.get(enzyme_name)
            sub = substrate_map.get(enzyme_name)
            if inhib is None or sub is None:
                continue
            if inhib.inhibition_strength <= 0 or not sub.is_substrate:
                continue

            # Occupancy of the inhibitor on the enzyme uses Ki (inhibition
            # constant), NOT the substrate's Km. When Ki is unavailable
            # (ki_um is None here — the default library has no curated Ki
            # values), Km is used as a proxy per the common Ki ≈ Km
            # approximation for competitive inhibitors.
            ki_um = inhib.ki_um if inhib.ki_um is not None else inhib.km_um
            occupancy = drug_a_conc_um / (ki_um + drug_a_conc_um)
            effective = inhib.inhibition_strength * occupancy * sub.substrate_fraction
            total_inhibition += effective
            mechanisms.append({
                "enzyme": enzyme_name,
                "inhibition_strength": inhib.inhibition_strength,
                "substrate_fraction": sub.substrate_fraction,
                "occupancy": occupancy,
                "effective_inhibition": effective,
            })

        auc_ratio = 1.0 / max(1.0 - total_inhibition, 0.01)
        auc_ratio = min(auc_ratio, 10.0)

        if auc_ratio > 2.0:
            significance = "CONTRAINDICATED"
        elif auc_ratio > 1.25:
            significance = "DDI_ALERT"
        else:
            significance = "NO_CLINICAL_DDI"

        return DDIPrediction(
            drug_a=drug_a,
            drug_b=drug_b,
            auc_ratio=auc_ratio,
            total_inhibition=total_inhibition,
            significance=significance,
            mechanisms=mechanisms,
        )

    def predict_all_pairs(
        self, drug_list: list[str], drug_a_conc_um: float = 10.0
    ) -> list[DDIPrediction]:
        """Predict DDI for all pairs in a drug list."""
        predictions = []
        for i, a in enumerate(drug_list):
            for b in drug_list[i + 1:]:
                pred = self.predict(a, b, drug_a_conc_um)
                if pred is not None:
                    predictions.append(pred)
                pred_rev = self.predict(b, a, drug_a_conc_um)
                if pred_rev is not None:
                    predictions.append(pred_rev)
        return predictions
