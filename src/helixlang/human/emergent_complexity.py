"""Emergent Complexity: Epigenetics, Multi-Organ Feedback, Microbiome-Liver-Immune Axis.

Models biological complexity that emerges from the INTERACTION of multiple
organ systems, creating feedback loops and emergent behaviors not predictable
from any single subsystem:

1. **Epigenetic modulation**: CYP enzyme expression changes via DNA methylation
   in response to chronic drug exposure (time-scale: days-weeks)
2. **Liver-Gut-Microbiome feedback loop**: bile acids → gut bacteria → SCFA →
   hepatic metabolism → bile acid synthesis (enterohepatic circulation)
3. **Gut-Liver Immune Axis**: bacterial translocation → hepatic Kupffer cell
   activation → systemic inflammation → drug metabolism changes
4. **Stress-Immune-Endocrine triple feedback**: cortisol suppresses immune →
   less IL-6 → less acute phase → less CYP inhibition → more drug clearance
5. **Drug-induced epigenetic reprogramming**: chronic NSAID → COX-2 methylation →
   altered inflammatory response

These effects are REDUNDANT with simpler models at short time-scales (< 1 week)
but become dominant at long time-scales (> 2 weeks) and in chronic therapy.

References:
- Ingvorsen et al., Toxicol In Vitro 2021: epigenetic CYP modulation
- Swanson et al., Nat Rev Drug Discov 2024: gut-liver axis pharmacology
- Serhan et al., Nat Rev Immunol 2024: inflammation-resolution circuits
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ============================================================================
# Epigenetic Modulation of Drug-Metabolizing Enzymes
# ============================================================================

@dataclass
class EpigeneticState:
    """Methylation state of a single CpG island near a CYP gene."""

    gene: str
    methylation_level: float = 0.0  # 0=unmethylated (full expression), 1=fully methylated (silenced)
    baseline_methylation: float = 0.0
    expression_fraction: float = 1.0  # relative to unmethylated state
    time_constant_h: float = 168.0    # ~7 days for methylation change


@dataclass
class EpigeneticModulation:
    """Tracks epigenetic changes to drug-metabolizing enzymes.

    Chronic drug exposure can alter CYP expression via:
    - Direct CYP induction (PXR/CAR activation → demethylation → ↑ expression)
    - CYP inhibition feedback (metabolite accumulation → ↑ methylation → ↓ expression)
    - Inflammatory cytokines (IL-6/TNF-α → ↑ methylation → ↓ CYP expression)
    """

    def __init__(self) -> None:
        self._states: dict[str, EpigeneticState] = {
            "CYP1A2":  EpigeneticState("CYP1A2", baseline_methylation=0.1),
            "CYP2B6":  EpigeneticState("CYP2B6", baseline_methylation=0.15),
            "CYP2C8":  EpigeneticState("CYP2C8", baseline_methylation=0.1),
            "CYP2C9":  EpigeneticState("CYP2C9", baseline_methylation=0.08),
            "CYP2C19": EpigeneticState("CYP2C19", baseline_methylation=0.12),
            "CYP2D6":  EpigeneticState("CYP2D6", baseline_methylation=0.05),
            "CYP2E1":  EpigeneticState("CYP2E1", baseline_methylation=0.10),
            "CYP3A4":  EpigeneticState("CYP3A4", baseline_methylation=0.08),
            "CYP3A5":  EpigeneticState("CYP3A5", baseline_methylation=0.20),
            "UGT1A1":  EpigeneticState("UGT1A1", baseline_methylation=0.10),
        }
        self._drug_exposure: dict[str, float] = {}  # cumulative exposure (AUC-like)
        self._inflammatory_signal: float = 0.0

    def update(
        self,
        dt_h: float,
        drug_concentrations: dict[str, float],
        il6_pg_ml: float = 1.0,
        tnf_pg_ml: float = 5.0,
    ) -> dict[str, float]:
        """Update epigenetic states based on drug exposure and inflammation.

        Returns {gene: expression_modifier} to multiply on CYP activity.
        """
        # Accumulate drug exposure
        for drug, conc in drug_concentrations.items():
            self._drug_exposure[drug] = self._drug_exposure.get(drug, 0.0) + conc * dt_h

        # Inflammatory signal (IL-6 and TNF-α drive CYP suppression)
        self._inflammatory_signal = 0.5 * min(il6_pg_ml / 10.0, 1.0) + 0.5 * min(tnf_pg_ml / 50.0, 1.0)

        modifiers: dict[str, float] = {}
        for gene, state in self._states.items():
            # Methylation drift toward new steady state
            # Inflammation increases methylation (suppresses expression)
            target_methylation = state.baseline_methylation + 0.3 * self._inflammatory_signal

            # Drug-specific effects (some drugs induce, some suppress)
            for drug, exposure in self._drug_exposure.items():
                if "rifampicin" in drug or "phenobarbital" in drug:
                    target_methylation -= 0.1 * min(exposure / 100.0, 1.0)  # induction
                elif "isoniazid" in drug or "phenytoin" in drug:
                    target_methylation += 0.05 * min(exposure / 100.0, 1.0)  # suppression

            target_methylation = max(0.0, min(0.9, target_methylation))

            # Exponential relaxation toward target
            tau = state.time_constant_h
            rate = (target_methylation - state.methylation_level) / tau
            state.methylation_level += rate * dt_h
            state.methylation_level = max(0.0, min(0.95, state.methylation_level))

            # Expression fraction: sigmoidal relationship with methylation
            state.expression_fraction = 1.0 / (1.0 + math.exp(8.0 * (state.methylation_level - 0.4)))
            modifiers[gene] = state.expression_fraction

        return modifiers

    def get_expression_modifier(self, gene: str) -> float:
        """Get current expression modifier for a CYP gene."""
        state = self._states.get(gene)
        return state.expression_fraction if state else 1.0


# ============================================================================
# Liver-Gut-Microbiome Feedback Loop
# ============================================================================

@dataclass
class LiverGutFeedback:
    """Models the enterohepatic circulation and gut-liver axis.

    Loop: Liver → bile acids → gut → microbiome (BSH) → secondary bile acids
          → portal vein → liver → FXR activation → bile acid synthesis regulation

    Also: Liver → glucose/lactate → gut absorption → portal → liver
          Gut permeability → endotoxin (LPS) → Kupffer cells → IL-6/TNF-α
    """

    def __init__(self) -> None:
        self.bile_acid_pool: float = 3000.0  # µmol total body pool
        self.primary_fraction: float = 0.75   # cholate + chenodeoxycholate
        self.secondary_fraction: float = 0.25  # deoxycholate + lithocholate
        self.fxr_activation: float = 0.5      # farnesoid X receptor (regulates synthesis)
        self.cyp7a1_rate: float = 1.0         # cholesterol 7α-hydroxylase (rate-limiting BA synthesis)
        self.enterohepatic_circulation_rate: float = 0.95  # 95% reabsorbed
        self.gut_permeability: float = 0.02
        self.endotoxin_level: float = 0.1     # EU/mL
        self.kupffer_activation: float = 0.0

    def step(
        self,
        dt_h: float,
        bshe_activity: float = 1.0,
        inflammation: float = 0.0,
    ) -> dict[str, float]:
        """Advance liver-gut feedback by one time step.

        Args:
            dt_h: time step (hours)
            bshe_activity: bile salt hydrolase activity from microbiome
            inflammation: systemic inflammation score (0-1)

        Returns signals to feed back into other systems
        """
        # Bile acid deconjugation by gut bacteria
        deconjugation_rate = bshe_activity * 0.02  # fraction per hour
        deconjugated = self.bile_acid_pool * self.primary_fraction * deconjugation_rate * dt_h
        self.primary_fraction -= deconjugated / max(self.bile_acid_pool, 1.0)
        self.secondary_fraction += deconjugated / max(self.bile_acid_pool, 1.0)
        self.primary_fraction = max(0.1, min(0.9, self.primary_fraction))
        self.secondary_fraction = 1.0 - self.primary_fraction

        # FXR activation by bile acids (particularly CDCA)
        self.fxr_activation = min(1.0, self.primary_fraction * 1.5)

        # CYP7A1 regulation: FXR suppresses CYP7A1 (negative feedback)
        self.cyp7a1_rate = max(0.1, 1.0 - 0.8 * self.fxr_activation)

        # Synthesis of new bile acids (compensates for fecal loss)
        synthesis_rate = self.cyp7a1_rate * 50.0  # µmol/h
        fecal_loss = self.bile_acid_pool * (1.0 - self.enterohepatic_circulation_rate) * 0.01
        self.bile_acid_pool += (synthesis_rate - fecal_loss) * dt_h
        self.bile_acid_pool = max(1000.0, min(6000.0, self.bile_acid_pool))

        # Gut permeability increases with inflammation
        self.gut_permeability = 0.02 + 0.15 * inflammation
        self.gut_permeability = min(0.5, self.gut_permeability)

        # Endotoxin translocation (proportional to permeability)
        self.endotoxin_level = self.gut_permeability * 10.0

        # Kupffer cell activation by endotoxin
        self.kupffer_activation = min(1.0, self.endotoxin_level / 2.0)

        return {
            "bile_acid_pool": self.bile_acid_pool,
            "fxr_activation": self.fxr_activation,
            "cyp7a1_rate": self.cyp7a1_rate,
            "gut_permeability": self.gut_permeability,
            "endotoxin_level": self.endotoxin_level,
            "kupffer_activation": self.kupffer_activation,
        }


# ============================================================================
# Stress-Immune-Endocrine Triple Feedback
# ============================================================================

@dataclass
class StressImmuneEndocrine:
    """Triple feedback loop: HPA axis ↔ immune system ↔ metabolic state.

    Feedback paths:
    1. Stress → cortisol ↑ → immune suppression (anti-inflammatory)
    2. Immune activation → IL-6/TNF-α → HPA axis stimulation → cortisol ↑
    3. Cortisol → insulin resistance → glucose ↑ → metabolic stress → immune ↓
    4. Immune activation → CRP ↑ → fever → metabolic rate ↑ → cortisol ↑
    5. Chronic stress → immune exhaustion → ↑ infection susceptibility
    """

    def __init__(self) -> None:
        self.cortisol_stimulation: float = 0.0   # external stress input (0-1)
        self.cortisol_level: float = 12.0        # µg/dL (normal 5-25)
        self.cortisol_suppression: float = 0.0   # effect on immune (0-1)
        self.immune_activation: float = 0.0      # feedback from immune to HPA
        self.insulin_resistance: float = 0.0     # cortisol-driven
        self.glucose_elevation: float = 0.0      # mg/dL above baseline
        self.fever_contribution: float = 0.0     # °C added by inflammation
        self.metabolic_rate_multiplier: float = 1.0
        self.immune_exhaustion: float = 0.0      # chronic activation → exhaustion

    def step(
        self,
        dt_h: float,
        il6: float = 1.0,
        tnf: float = 5.0,
        immune_activation_input: float = 0.0,
        crp: float = 0.0,
    ) -> dict[str, float]:
        """Advance the stress-immune-endocrine loop.

        Returns signals to feed into other subsystems.
        """
        # Path 1: Immune activation → HPA stimulation
        immune_stimulation = 0.3 * min(il6 / 50.0, 1.0) + 0.2 * min(tnf / 100.0, 1.0)
        self.immune_activation = immune_stimulation

        # Path 2: Cortisol production (HPA axis)
        total_stimulation = self.cortisol_stimulation + self.immune_activation
        target_cortisol = 12.0 + 25.0 * total_stimulation  # normal 12, stressed up to 37
        self.cortisol_level += (target_cortisol - self.cortisol_level) * (1 - math.exp(-dt_h / 4.0))
        self.cortisol_level = max(3.0, min(50.0, self.cortisol_level))

        # Path 3: Cortisol → immune suppression
        self.cortisol_suppression = min(1.0, max(0.0, (self.cortisol_level - 20.0) / 30.0))

        # Path 4: Cortisol → insulin resistance → glucose
        self.insulin_resistance = min(0.8, (self.cortisol_level - 15.0) / 40.0)
        self.glucose_elevation = 20.0 * self.insulin_resistance

        # Path 5: Fever from CRP/inflammation
        self.fever_contribution = 0.005 * crp  # 1°C per 200 mg/L CRP
        self.fever_contribution = min(2.0, self.fever_contribution)

        # Path 6: Metabolic rate increases with fever
        self.metabolic_rate_multiplier = 1.0 + 0.13 * self.fever_contribution  # ~13% per °C

        # Path 7: Immune exhaustion from chronic activation
        chronic_load = min(total_stimulation, 1.0)
        exhaustion_rate = 0.001 * chronic_load  # slow accumulation
        recovery_rate = 0.005 * self.immune_exhaustion  # recovery when stimulus removed
        self.immune_exhaustion += (exhaustion_rate - recovery_rate) * dt_h
        self.immune_exhaustion = max(0.0, min(0.5, self.immune_exhaustion))

        return {
            "cortisol_ug_dl": self.cortisol_level,
            "cortisol_suppression": self.cortisol_suppression,
            "insulin_resistance": self.insulin_resistance,
            "glucose_elevation_mg_dl": self.glucose_elevation,
            "fever_c": self.fever_contribution,
            "metabolic_rate": self.metabolic_rate_multiplier,
            "immune_exhaustion": self.immune_exhaustion,
        }


# ============================================================================
# Emergent Complexity Orchestrator
# ============================================================================

class EmergentComplexityModel:
    """Orchestrates all emergent complexity subsystems.

    This module runs after the individual organ models and applies
    cross-system feedback effects that create emergent behavior.
    """

    def __init__(self) -> None:
        self.epigenetics = EpigeneticModulation()
        self.liver_gut = LiverGutFeedback()
        self.stress_immune = StressImmuneEndocrine()

        # State tracking
        self._cumulative_exposure: dict[str, float] = {}

    def step(
        self,
        dt_h: float,
        t_h: float,
        drug_concentrations: dict[str, float],
        il6: float = 1.0,
        tnf: float = 5.0,
        crp: float = 0.0,
        bshe_activity: float = 1.0,
        cortisol_input: float = 12.0,
        immune_activation: float = 0.0,
    ) -> dict[str, float]:
        """Run one step of all emergent complexity models.

        Args:
            dt_h: time step (hours)
            t_h: current time (hours)
            drug_concentrations: {drug: µM}
            il6, tnf, crp: current immune/inflammatory markers
            bshe_activity: bile salt hydrolase activity from microbiome
            cortisol_input: external cortisol stimulation
            immune_activation: immune → HPA feedback

        Returns dict of all emergent signals
        """
        # 1. Epigenetic CYP modulation
        cyp_modifiers = self.epigenetics.update(dt_h, drug_concentrations, il6, tnf)

        # 2. Liver-gut feedback loop
        lg_signals = self.liver_gut.step(dt_h, bshe_activity, crp / 180.0)

        # 3. Stress-immune-endocrine triple feedback
        se_signals = self.stress_immune.step(
            dt_h, il6, tnf, immune_activation, crp,
        )

        # Update cortisol stimulation
        self.stress_immune.cortisol_stimulation = cortisol_input / 37.0

        # Aggregate results
        return {
            # Epigenetic CYP modifiers
            **{f"epigenetic_{k}": v for k, v in cyp_modifiers.items()},
            # Liver-gut
            "bile_acid_pool": lg_signals["bile_acid_pool"],
            "fxr_activation": lg_signals["fxr_activation"],
            "gut_permeability": lg_signals["gut_permeability"],
            "endotoxin_level": lg_signals["endotoxin_level"],
            "kupffer_activation": lg_signals["kupffer_activation"],
            # Stress-immune-endocrine
            "cortisol_ug_dl": se_signals["cortisol_ug_dl"],
            "cortisol_suppression": se_signals["cortisol_suppression"],
            "glucose_elevation_mg_dl": se_signals["glucose_elevation_mg_dl"],
            "fever_c": se_signals["fever_c"],
            "metabolic_rate": se_signals["metabolic_rate"],
            "immune_exhaustion": se_signals["immune_exhaustion"],
        }

    def get_cyp_modifier(self, gene: str) -> float:
        """Get the epigenetic expression modifier for a CYP gene."""
        return self.epigenetics.get_expression_modifier(gene)
