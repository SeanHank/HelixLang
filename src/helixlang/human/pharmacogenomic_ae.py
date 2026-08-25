"""Mechanistic Pharmacogenomic Adverse Event Prediction (doc/32 §7.6).

Traces the pathway: genotype → enzyme activity → drug metabolism rate →
toxic metabolite accumulation → cellular damage threshold → AE.

Unlike statistical AE prediction (requires N > 1/frequency), this approach
predicts rare AEs from mechanistic first principles, capturing genotype-dependent
heterogeneity that statistics average over.

References:
- DGANet, Frontiers 2025: pharmacogenomic ADR prediction AUROC 92.76%
- DrugBank + PharmGKB: known toxic metabolite pathways
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class AERisk(Enum):
    """Adverse event risk classification."""

    MINIMAL = "minimal"       # P(AE) < 1%
    LOW = "low"               # P(AE) 1-5%
    MODERATE = "moderate"     # P(AE) 5-15%
    HIGH = "high"             # P(AE) 15-30%
    VERY_HIGH = "very_high"   # P(AE) > 30%


class AEOrgan(Enum):
    """Target organ for adverse event."""

    LIVER = "liver"
    KIDNEY = "kidney"
    HEART = "heart"
    BONE_MARROW = "bone_marrow"
    BRAIN = "brain"
    SKIN = "skin"
    GI = "gi"


@dataclass
class ToxicMetabolite:
    """A known toxic metabolite of a parent drug.

    Attributes:
        name: metabolite identifier (e.g. "NAPQI")
        parent_drug: parent drug name
        producing_enzyme: CYP isoform that produces this metabolite
        eliminating_enzyme: CYP/UGT that detoxifies it (None if no detox pathway)
        km_production: Michaelis constant for production (µM)
        vmax_production: Vmax for production (µmol/min/mg protein)
        km_elimination: Michaelis constant for elimination (µM)
        vmax_elimination: Vmax for elimination (µmol/min/mg protein)
        ic50_target_organ: IC50 for target organ toxicity (µM)
        target_organ: primary organ affected
        half_life_h: metabolite elimination half-life (hours)
    """

    name: str
    parent_drug: str
    producing_enzyme: str
    eliminating_enzyme: str | None
    km_production: float
    vmax_production: float
    km_elimination: float = 10.0
    vmax_elimination: float = 5.0
    ic50_target_organ: float = 10.0
    target_organ: AEOrgan = AEOrgan.LIVER
    half_life_h: float = 2.0


@dataclass
class MetaboliteState:
    """Current state of a toxic metabolite in the body."""

    metabolite_name: str
    concentration_um: float = 0.0
    production_rate: float = 0.0
    elimination_rate: float = 0.0
    net_accumulation_rate: float = 0.0
    toxicity_score: float = 0.0  # [M] / IC50
    ae_probability: float = 0.0


@dataclass
class AEPrediction:
    """Adverse event prediction for a specific drug-genotype combination."""

    drug_name: str
    genotype_description: str
    metabolite: str
    target_organ: AEOrgan
    risk_level: AERisk
    ae_probability: float  # P(AE | genotype)
    metabolite_concentration: float  # predicted steady-state [M] (µM)
    ic50: float
    toxicity_ratio: float  # [M]/IC50
    enzyme_activity: float  # producing enzyme activity (fraction of normal)
    eliminating_activity: float  # eliminating enzyme activity
    mechanism: str  # human-readable mechanism
    confidence: float = 0.8  # prediction confidence


# ============================================================================
# Curated toxic metabolite database (DrugBank + PharmGKB)
# ============================================================================

TOXIC_METABOLITES: dict[str, ToxicMetabolite] = {
    "NAPQI": ToxicMetabolite(
        name="NAPQI",
        parent_drug="acetaminophen",
        producing_enzyme="CYP2E1",
        eliminating_enzyme="GST",
        km_production=500.0,
        vmax_production=8.0,
        km_elimination=100.0,
        vmax_elimination=12.0,
        ic50_target_organ=50.0,
        target_organ=AEOrgan.LIVER,
        half_life_h=1.0,
    ),
    "MTX_PG": ToxicMetabolite(
        name="MTX_PG",
        parent_drug="methotrexate",
        producing_enzyme="FPGS",
        eliminating_enzyme=None,
        km_production=50.0,
        vmax_production=2.0,
        km_elimination=1.0,
        vmax_elimination=0.1,
        ic50_target_organ=5.0,
        target_organ=AEOrgan.BONE_MARROW,
        half_life_h=24.0,
    ),
    "5FU": ToxicMetabolite(
        name="5-fluorouracil",
        parent_drug="fluorouracil",
        producing_enzyme="oral_administration",
        eliminating_enzyme="DPD",
        km_production=30.0,
        vmax_production=15.0,
        km_elimination=1.0,
        vmax_elimination=0.5,
        ic50_target_organ=10.0,
        target_organ=AEOrgan.HEART,
        half_life_h=0.5,
    ),
    "CISPLATIN_AQ": ToxicMetabolite(
        name="aqua-cisplatin",
        parent_drug="cisplatin",
        producing_enzyme="non_enzymatic",
        eliminating_enzyme=None,
        km_production=10.0,
        vmax_production=5.0,
        km_elimination=1.0,
        vmax_elimination=0.2,
        ic50_target_organ=2.0,
        target_organ=AEOrgan.KIDNEY,
        half_life_h=1.5,
    ),
    "IRINOTECAN_SN38": ToxicMetabolite(
        name="SN-38",
        parent_drug="irinotecan",
        producing_enzyme="CES2",
        eliminating_enzyme="UGT1A1",
        km_production=20.0,
        vmax_production=10.0,
        km_elimination=5.0,
        vmax_elimination=8.0,
        ic50_target_organ=1.0,
        target_organ=AEOrgan.BONE_MARROW,
        half_life_h=6.0,
    ),
    "MYCOPHENOLIC_ACID": ToxicMetabolite(
        name="mycophenolic acid",
        parent_drug="mycophenolate_mofetil",
        producing_enzyme="esterase",
        eliminating_enzyme="UGT1A9",
        km_production=50.0,
        vmax_production=20.0,
        km_elimination=30.0,
        vmax_elimination=15.0,
        ic50_target_organ=50.0,
        target_organ=AEOrgan.GI,
        half_life_h=8.0,
    ),
    "AMIKACIN_OD": ToxicMetabolite(
        name="amikacin-uptake",
        parent_drug="amikacin",
        producing_enzyme="renal_uptake",
        eliminating_enzyme=None,
        km_production=5.0,
        vmax_production=3.0,
        km_elimination=1.0,
        vmax_elimination=0.1,
        ic50_target_organ=100.0,
        target_organ=AEOrgan.KIDNEY,
        half_life_h=2.0,
    ),
}


class ToxicMetaboliteAccumulator:
    """Tracks toxic metabolite production, accumulation, and elimination.

    For each drug, computes:
        d[M]/dt = Vmax_prod × [D] / (Km_prod + [D]) × activity(prod_enzyme)
                 - Vmax_elim × [M] / (Km_elim + [M]) × activity(elim_enzyme)

    where [D] is parent drug concentration and activity() comes from genotype.
    """

    def __init__(self) -> None:
        self._states: dict[str, MetaboliteState] = {}
        self._drug_concs: dict[str, float] = {}

    def set_drug_concentration(self, drug_name: str, concentration_um: float) -> None:
        """Update parent drug concentration."""
        self._drug_concs[drug_name.lower()] = concentration_um

    def step(
        self,
        dt_h: float,
        genotype_activities: dict[str, float],
    ) -> dict[str, MetaboliteState]:
        """Advance all metabolite accumulations by one time step.

        Args:
            dt_h: time step in hours
            genotype_activities: {enzyme_name: activity_fraction} from genotype

        Returns:
            Updated metabolite states
        """
        for met_name, met in TOXIC_METABOLITES.items():
            drug_conc = self._drug_concs.get(met.parent_drug.lower(), 0.0)
            prod_activity = genotype_activities.get(met.producing_enzyme, 1.0)
            elim_activity = (
                genotype_activities.get(met.eliminating_enzyme, 1.0)
                if met.eliminating_enzyme else 0.1
            )

            # Michaelis-Menten production
            production = met.vmax_production * drug_conc / (met.km_production + drug_conc + 1e-10)
            production *= prod_activity

            # Michaelis-Menten elimination
            current_conc = self._states.get(met_name, MetaboliteState(metabolite_name=met_name)).concentration_um
            elimination = met.vmax_elimination * current_conc / (met.km_elimination + current_conc + 1e-10)
            elimination *= elim_activity

            # Accumulation
            net_rate = production - elimination
            new_conc = max(0.0, current_conc + net_rate * dt_h)

            # Toxicity score
            tox_score = new_conc / met.ic50_target_organ

            # AE probability (sigmoid of toxicity ratio)
            ae_prob = 1.0 / (1.0 + math.exp(-3.0 * (tox_score - 1.0)))

            self._states[met_name] = MetaboliteState(
                metabolite_name=met_name,
                concentration_um=new_conc,
                production_rate=production,
                elimination_rate=elimination,
                net_accumulation_rate=net_rate,
                toxicity_score=tox_score,
                ae_probability=ae_prob,
            )

        return dict(self._states)

    def get_states(self) -> dict[str, MetaboliteState]:
        return dict(self._states)


class GenotypeAEPredictor:
    """Predicts adverse events from genotype → enzyme → metabolite → threshold.

    Combines the ToxicMetaboliteAccumulator with genotype information to
    produce individualized AE risk predictions.
    """

    def __init__(self, accumulator: ToxicMetaboliteAccumulator | None = None) -> None:
        self.accumulator = accumulator or ToxicMetaboliteAccumulator()

    def predict_ae(
        self,
        drug_name: str,
        drug_concentration_um: float,
        genotype_activities: dict[str, float],
    ) -> list[AEPrediction]:
        """Predict adverse events for a drug at given genotype.

        Args:
            drug_name: name of the parent drug
            drug_concentration_um: steady-state parent drug concentration (µM)
            genotype_activities: {enzyme: activity} from GenotypeProfile

        Returns:
            List of AE predictions (one per known toxic metabolite)
        """
        drug_lower = drug_name.lower().replace(" ", "_").replace("-", "_")
        predictions: list[AEPrediction] = []

        for met_name, met in TOXIC_METABOLITES.items():
            if met.parent_drug.lower().replace(" ", "_") != drug_lower:
                continue

            prod_activity = genotype_activities.get(met.producing_enzyme, 1.0)
            elim_activity = (
                genotype_activities.get(met.eliminating_enzyme, 1.0)
                if met.eliminating_enzyme else 0.1
            )

            # Steady-state metabolite concentration
            production_ss = met.vmax_production * drug_concentration_um / (
                met.km_production + drug_concentration_um + 1e-10
            ) * prod_activity
            elimination_ss_rate = met.vmax_elimination / (met.km_elimination + 1e-10) * elim_activity
            ss_conc = production_ss / (elimination_ss_rate + 1e-10)

            tox_ratio = ss_conc / met.ic50_target_organ
            ae_prob = 1.0 / (1.0 + math.exp(-3.0 * (tox_ratio - 1.0)))

            if ae_prob >= 0.30:
                risk = AERisk.VERY_HIGH
            elif ae_prob >= 0.15:
                risk = AERisk.HIGH
            elif ae_prob >= 0.05:
                risk = AERisk.MODERATE
            elif ae_prob >= 0.01:
                risk = AERisk.LOW
            else:
                risk = AERisk.MINIMAL

            # Determine mechanism description
            if prod_activity < 0.5:
                mechanism = (
                    f"PM phenotype for {met.producing_enzyme}: reduced production of "
                    f"{met_name} → lower AE risk"
                )
            elif prod_activity > 1.5:
                mechanism = (
                    f"UM phenotype for {met.producing_enzyme}: increased production of "
                    f"{met_name} → higher AE risk"
                )
            elif met.eliminating_enzyme and elim_activity < 0.5:
                mechanism = (
                    f"PM phenotype for {met.eliminating_enzyme}: impaired detoxification "
                    f"of {met_name} → higher AE risk"
                )
            else:
                mechanism = f"Normal metabolism of {met_name} via {met.producing_enzyme}"

            predictions.append(AEPrediction(
                drug_name=drug_name,
                genotype_description=f"{met.producing_enzyme}: {prod_activity:.2f}",
                metabolite=met_name,
                target_organ=met.target_organ,
                risk_level=risk,
                ae_probability=ae_prob,
                metabolite_concentration=ss_conc,
                ic50=met.ic50_target_organ,
                toxicity_ratio=tox_ratio,
                enzyme_activity=prod_activity,
                eliminating_activity=elim_activity,
                mechanism=mechanism,
                confidence=0.8,
            ))

        return predictions

    def predict_all(
        self,
        drug_concentrations: dict[str, float],
        genotype_activities: dict[str, float],
    ) -> dict[str, list[AEPrediction]]:
        """Predict AE for all drugs in a regimen.

        Args:
            drug_concentrations: {drug_name: concentration (µM)}
            genotype_activities: {enzyme: activity} from GenotypeProfile

        Returns:
            {drug_name: [AEPrediction, ...]}
        """
        results = {}
        for drug, conc in drug_concentrations.items():
            preds = self.predict_ae(drug, conc, genotype_activities)
            if preds:
                results[drug] = preds
        return results
