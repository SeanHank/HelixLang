"""Per-disease ODE models for 8 major categories (doc/30 §§1-8).

Replaces the generic logistic-severity DiseaseProgressionModel with
mechanistic, disease-specific ODE systems that capture the actual
pathophysiology:

1. **Cardiovascular** — Guyton venous return / cardiac function
2. **Metabolic (T2D)** — Bergman β-cell + hepatic glucose output
3. **Cancer** — Gompertz tumor growth + immune surveillance + pathway-specific therapy
4. **Autoimmune (RA)** — joint inflammation + cytokine dynamics
5. **Neurological** — neurodegeneration + synaptic loss
6. **Renal** — nephron loss + compensatory hyperfiltration
7. **Hepatic** — fibrosis progression + synthetic function
8. **Hematological** — myelodysplasia + ineffective hematopoiesis

Each model exposes a uniform `step(dt_h, drug_effectiveness)` interface
and produces disease-specific state that feeds into labs, vitals, and
organ crosstalk.

Module structure:
    CardiovascularODE     Guyton-inspired CV model
    MetabolicT2DODE       β-cell + insulin-glucose dynamics
    CancerODE             Gompertz + immune surveillance + per-pathway therapy
    TumorClone            Clonal subpopulation model
    TumorHeterogeneity    Clonal dynamics + resistance tracking
    TumorBiopsy           Tumor molecular profiling
    AutoimmuneRAODE       Joint inflammation model
    NeurologicalODE       Neurodegeneration model
    RenalODE              Nephron loss model
    HepaticODE            Fibrosis progression model
    HematologicalODE      Myelodysplasia model
    DiseaseModelFactory   dispatch to correct ODE by disease name
    create_disease_model  convenience factory

References:
- Guyton AC, Hall JE. Textbook of Medical Physiology, 14th ed. 2020.
- Bergman RN et al. Physiologic evaluation of factors controlling glucose
  tolerance in man. J Clin Invest 1981;68:1456-1467.
- Gompertz B. Phil. Trans. R. Soc. 1825;115:513-583 (tumor growth, per Laird 1964).
- Mackey MC, Glass L. Science 1977;197:287-289 (hematological oscillations).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

__all__ = [
    "CardiovascularODE",
    "MetabolicT2DODE",
    "CancerODE",
    "TumorClone",
    "TumorHeterogeneity",
    "TumorBiopsy",
    "select_targeted_therapy",
    "TARGET_TO_PATHWAY",
    "AutoimmuneRAODE",
    "NeurologicalODE",
    "RenalODE",
    "HepaticODE",
    "HematologicalODE",
    "RespiratoryODE",
    "InfectiousDiseaseODE",
    "GastrointestinalODE",
    "EndocrineODE",
    "create_disease_model",
]


# ============================================================================
# 1. Cardiovascular (Guyton Venous Return)
# ============================================================================


@dataclass
class CardiovascularODE:
    """Simplified Guyton venous return / cardiac function model.

    States:
        MAP: mean arterial pressure (mmHg)
        CO: cardiac output (L/min)
        blood_volume: effective circulating volume (L)
        vascular_resistance: systemic vascular resistance

    Drug targets: ACEi/ARB (↓SVR), β-blockers (↓HR/CO), diuretics (↓volume),
    statins (↓atherosclerosis progression)
    """

    map_mmhg: float = 93.0       # mean arterial pressure
    co_l_min: float = 5.0        # cardiac output
    blood_volume_l: float = 5.0   # effective volume
    svr: float = 18.6            # systemic vascular resistance (mmHg/(L/min))

    # --- Rate constants ---
    volume_half_life_h: float = 48.0   # fluid balance half-life
    pressure_autoregulation: float = 0.02  # baroreflex gain

    # --- Disease modifiers ---
    atherosclerosis_severity: float = 0.0  # 0-1
    heart_failure_severity: float = 0.0    # 0-1 (reduces contractility)
    hypertension_severity: float = 0.0     # 0-1 (increases SVR)

    def step(self, dt_h: float, drug_svr_mod: float = 1.0,
             drug_volume_mod: float = 1.0) -> None:
        """Advance one hour."""
        # Cardiac output: CO = HR × SV (simplified)
        contractility = 1.0 - 0.6 * self.heart_failure_severity
        self.co_l_min = 5.0 * contractility * (self.blood_volume_l / 5.0)

        # MAP = CO × SVR
        effective_svr = self.svr * (1.0 + 0.5 * self.hypertension_severity) * drug_svr_mod
        effective_svr *= (1.0 + 0.3 * self.atherosclerosis_severity)
        self.map_mmhg = self.co_l_min * effective_svr

        # Volume dynamics (kidney regulation)
        k_vol = math.log(2) / (self.volume_half_life_h + 1e-12)
        target_volume = 5.0 * (1.0 - 0.1 * self.heart_failure_severity)
        self.blood_volume_l += dt_h * (
            -k_vol * (self.blood_volume_l - target_volume) * drug_volume_mod)

        self.svr = effective_svr
        self.blood_volume_l = max(3.0, min(7.0, self.blood_volume_l))
        self.map_mmhg = max(60.0, min(180.0, self.map_mmhg))


# ============================================================================
# 2. Metabolic T2D (β-cell + Insulin-Glucose)
# ============================================================================


@dataclass
class MetabolicT2DODE:
    """T2D-specific ODE: β-cell function + hepatic glucose output.

    Extends Bergman minimal model with β-cell failure progression
    and hepatic insulin resistance.

    States:
        beta_cell_function: 0-1 (remaining β-cell capacity)
        hepatic_glucose_output: mg/dL/h
        hba1c_target: derived from average glucose
    """

    beta_cell_function: float = 1.0  # 0=complete failure, 1=normal
    hepatic_glr: float = 2.0        # hepatic glucose release rate (mg/dL/h)
    peripheral_uptake: float = 1.0   # muscle/fat glucose disposal

    # --- Progression parameters ---
    beta_cell_loss_rate: float = 0.001  # per hour (slow decline)
    glucotoxicity_factor: float = 0.0001  # chronic hyperglycemia → β-cell loss

    # --- Disease severity ---
    t2d_severity: float = 0.0  # 0-1

    def step(self, dt_h: float, glucose_mg_dl: float = 100.0,
             insulin_uuml: float = 10.0,
             drug_effectiveness: float = 0.0) -> None:
        """Advance one hour.

        Args:
            glucose_mg_dl: current plasma glucose
            insulin_uuml: current plasma insulin
            drug_effectiveness: 0-1 (metformin/etc efficacy)
        """
        # β-cell loss accelerated by glucotoxicity
        if glucose_mg_dl > 180.0:
            glucotoxicity = self.glucotoxicity_factor * (
                glucose_mg_dl - 180.0)
        else:
            glucotoxicity = 0.0

        self.beta_cell_function -= dt_h * (
            self.beta_cell_loss_rate + glucotoxicity)
        self.beta_cell_function = max(0.05, self.beta_cell_function)

        # Hepatic glucose output: suppressed by insulin, increased by resistance
        insulin_suppression = insulin_uuml / (insulin_uuml + 50.0)
        resistance = 1.0 + 2.0 * self.t2d_severity * (
            1.0 - drug_effectiveness)
        self.hepatic_glr = 3.0 * resistance * (1.0 - 0.7 * insulin_suppression)

        # Drug effect: metformin reduces hepatic glucose output
        self.hepatic_glr *= (1.0 - 0.4 * drug_effectiveness)


# ============================================================================
# 3. Cancer (Gompertz + Immune Surveillance + Per-Pathway Therapy)
# ============================================================================

# Target protein → pathway mapping (doc/33 Phase 4 §12.1)
TARGET_TO_PATHWAY: dict[str, str] = {
    "EGFR": "egfr", "EGFRvIII": "egfr", "ERBB1": "egfr",
    "BRAF_V600E": "braf", "BRAF": "braf",
    "VEGFR2": "vegfr", "VEGFR1": "vegfr", "KDR": "vegfr", "VEGFR": "vegfr",
    "ALK": "alk", "EML4-ALK": "alk",
    "HER2": "her2", "ERBB2": "her2",
    "PD-L1": "pd_l1", "CD274": "pd_l1", "PDL1": "pd_l1",
    "CTLA4": "ctla4", "CTLA-4": "ctla4",
    "Tubulin": "microtubule", "Beta-tubulin": "microtubule", "TUBB": "microtubule",
    "TOP1": "topoisomerase", "TOP2A": "topoisomerase",
    "DNA": "dna_damage", "BRCA1": "dna_damage", "BRCA2": "dna_damage",
    "PARP1": "parp", "PARP": "parp",
    "BCR-ABL": "bcr_abl", "ABL1": "bcr_abl",
    "KRAS": "kras", "KRAS_G12C": "kras",
    "MET": "met", "HGFR": "met",
    "RET": "ret",
    "NTRK": "ntrk", "NTRK1": "ntrk", "NTRK2": "ntrk",
    "PI3K": "pi3k", "PIK3CA": "pi3k",
    "mTOR": "mtor",
    "CDK4": "cdk46", "CDK6": "cdk46",
    "IDH1": "idh", "IDH2": "idh",
    "FLT3": "flt3",
    "JAK2": "jak",
}


@dataclass
class TumorClone:
    """A clonal subpopulation within a heterogeneous tumor.

    Each clone has independent growth rate and drug sensitivities.
    Resistance mutations reduce drug sensitivity and incur a fitness cost.

    Attributes:
        name: clone identifier (e.g. "EGFR_L858R_parent", "EGFR_T790M_resistant")
        fraction: proportion of total tumor (0.0–1.0, all clones sum to 1.0)
        growth_rate: intrinsic Gompertz growth rate (per hour)
        drug_sensitivities: mapping pathway → sensitivity (0.0 = no effect, 1.0 = full kill)
        resistance_mutations: list of acquired resistance mutations
        fitness_cost: multiplicative fitness cost per resistance mutation (0.0–1.0)
    """

    name: str = "parent"
    fraction: float = 1.0
    growth_rate: float = 0.01
    drug_sensitivities: dict[str, float] = field(default_factory=dict)
    resistance_mutations: list[str] = field(default_factory=list)
    fitness_cost: float = 0.1

    def effective_growth(self) -> float:
        """Growth rate reduced by fitness cost of resistance mutations."""
        cost = 1.0 - self.fitness_cost * len(self.resistance_mutations)
        return self.growth_rate * max(0.1, cost)

    def drug_kill_rate(self, pathway_effects: dict[str, float],
                       drug_kill_capacity: float = 0.05) -> float:
        """Total drug-mediated kill rate for this clone.

        Pathway inhibition × sensitivity gives a dimensionless fraction
        (0–1) of maximum kill, scaled by *drug_kill_capacity* (1/h) to
        produce a pharmacologically realistic kill rate independent of the
        clone's intrinsic growth rate.
        """
        inhibition = 0.0
        for pathway, effect in pathway_effects.items():
            sensitivity = self.drug_sensitivities.get(pathway, 1.0)
            inhibition += effect * sensitivity
        return inhibition * drug_kill_capacity


@dataclass
class TumorHeterogeneity:
    """Clonal heterogeneity model for tumor evolution.

    Tracks multiple clones, computes aggregate tumor metrics, and
    simulates stochastic resistance mutation emergence.

    Attributes:
        clones: list of tumor clones (must sum to fraction ~1.0)
        resistance_rate: probability of resistance mutation per cell division per pathway
        carrying_capacity: maximum tumor volume (relative)
    """

    clones: list[TumorClone] = field(default_factory=lambda: [
        TumorClone(name="parent", fraction=1.0, growth_rate=0.01)
    ])
    resistance_rate: float = 1e-6
    carrying_capacity: float = 1.0
    _rng: random.Random = field(default_factory=lambda: random.Random(42), repr=False)

    def step(self, dt_h: float, pathway_effects: dict[str, float],
             drug_kill_capacity: float = 0.05) -> dict[str, float]:
        """Advance clonal dynamics one time step.

        Returns:
            Dictionary with aggregate tumor metrics:
            - total_volume: aggregate tumor volume
            - weighted_growth: volume-weighted average growth rate
            - weighted_kill: volume-weighted average drug kill rate
            - resistant_fraction: fraction of tumor with any resistance mutation
        """
        if not self.clones:
            return {"total_volume": 0.0, "weighted_growth": 0.0,
                    "weighted_kill": 0.0, "resistant_fraction": 0.0}

        total_volume = 0.0
        weighted_growth = 0.0
        weighted_kill = 0.0
        resistant_volume = 0.0

        for clone in self.clones:
            if clone.fraction <= 0.0:
                continue
            eff_growth = clone.effective_growth()
            kill = clone.drug_kill_rate(pathway_effects, drug_kill_capacity)
            net_growth = eff_growth - kill
            clone.fraction *= math.exp(dt_h * net_growth)
            total_volume += clone.fraction
            weighted_growth += clone.fraction * eff_growth
            weighted_kill += clone.fraction * kill
            if clone.resistance_mutations:
                resistant_volume += clone.fraction

        # Stochastic resistance mutation emergence
        new_clones: list[TumorClone] = []
        for clone in list(self.clones):
            if clone.fraction <= 0.01:
                continue
            for pathway in pathway_effects:
                if pathway_effects[pathway] > 0.01:
                    if self._rng.random() < self.resistance_rate * dt_h * clone.fraction:
                        new_sens = dict(clone.drug_sensitivities)
                        new_sens[pathway] = new_sens.get(pathway, 1.0) * 0.1
                        new_clone = TumorClone(
                            name=f"{clone.name}_R_{pathway}",
                            fraction=clone.fraction * 0.01,
                            growth_rate=clone.growth_rate,
                            drug_sensitivities=new_sens,
                            resistance_mutations=clone.resistance_mutations + [pathway],
                            fitness_cost=clone.fitness_cost,
                        )
                        clone.fraction *= 0.99
                        new_clones.append(new_clone)
        self.clones.extend(new_clones)

        # Renormalize fractions
        if total_volume > 0.0:
            scale = 1.0 / total_volume
            for clone in self.clones:
                clone.fraction *= scale
                weighted_growth *= scale
                weighted_kill *= scale
            resistant_volume *= scale
        else:
            resistant_volume = 0.0

        return {
            "total_volume": total_volume,
            "weighted_growth": weighted_growth,
            "weighted_kill": weighted_kill,
            "resistant_fraction": resistant_volume,
        }

    def get_clone_summary(self) -> list[dict[str, object]]:
        """Return a summary of current clone composition."""
        return [
            {
                "name": c.name,
                "fraction": c.fraction,
                "growth_rate": c.growth_rate,
                "resistance_mutations": list(c.resistance_mutations),
            }
            for c in self.clones
            if c.fraction > 0.001
        ]


@dataclass
class TumorBiopsy:
    """Tumor molecular profile from biopsy.

    Used for biomarker-driven therapy selection (doc/33 Phase 4 §12.3).

    Attributes:
        mutations: somatic mutations detected (e.g. ["EGFR_L858R", "TP53_R175H"])
        amplifications: gene amplifications (e.g. ["HER2"])
        deletions: gene deletions (e.g. ["CDKN2A"])
        pd_l1_expression: PD-L1 tumor proportion score (0.0–1.0)
        msi_status: microsatellite instability status ("MSI-H", "MSS", "MSI-L")
        tmb_per_mb: tumor mutational burden (mutations per megabase)
        hr_status: homologous recombination status ("HRD", "HRC")
        fusion_genes: fusion transcripts detected (e.g. ["EML4-ALK", "BCR-ABL"])
    """

    mutations: list[str] = field(default_factory=list)
    amplifications: list[str] = field(default_factory=list)
    deletions: list[str] = field(default_factory=list)
    pd_l1_expression: float = 0.0
    msi_status: str = "MSS"
    tmb_per_mb: float = 0.0
    hr_status: str = "HRC"
    fusion_genes: list[str] = field(default_factory=list)

    def has_mutation(self, pattern: str) -> bool:
        """Check if a mutation pattern is present (case-insensitive substring)."""
        pl = pattern.lower()
        return any(pl in m.lower() for m in self.mutations)

    def has_amplification(self, gene: str) -> bool:
        """Check if a gene amplification is present."""
        return gene.upper() in [a.upper() for a in self.amplifications]

    def has_fusion(self, pattern: str) -> bool:
        """Check if a fusion gene is present."""
        pl = pattern.lower()
        return any(pl in f.lower() for f in self.fusion_genes)


# Therapy guidelines based on molecular profile (NCCN-inspired)
_THERAPY_RULES: list[tuple[str, str, list[str], str]] = [
    # (rule_name, required_condition, drugs, rationale)
    ("EGFR_mutant", "mutation", ["erlotinib", "gefitinib", "osimertinib"],
     "EGFR-mutant NSCLC: first-line EGFR TKI"),
    ("ALK_fusion", "fusion", ["crizotinib", "alectinib", "lorlatinib"],
     "ALK-rearranged NSCLC: first-line ALK TKI"),
    ("HER2_amp", "amplification", ["trastuzumab", "t-dXd"],
     "HER2-amplified: anti-HER2 therapy"),
    ("BRAF_V600E", "mutation", ["dabrafenib", "trametinib"],
     "BRAF V600E: BRAF + MEK inhibitor"),
    ("KRAS_G12C", "mutation", ["sotorasib", "adagrasib"],
     "KRAS G12C: covalent KRAS inhibitor"),
    ("PD_L1_high", "pd_l1", ["pembrolizumab", "nivolumab", "atezolizumab"],
     "PD-L1 ≥50%: first-line anti-PD-(L)1"),
    ("MSI_high", "msi", ["pembrolizumab", "dostarlimab"],
     "MSI-H: anti-PD-1 (tissue-agnostic)"),
    ("TMB_high", "tmb", ["pembrolizumab"],
     "TMB ≥10 mut/Mb: anti-PD-1 (tissue-agnostic)"),
    ("BRCA_HRD", "hrd", ["olaparib", "niraparib"],
     "BRCA/HRD: PARP inhibitor"),
    ("BCR_ABL", "fusion", ["imatinib", "dasatinib"],
     "BCR-ABL: tyrosine kinase inhibitor"),
    ("FLT3_mutant", "mutation", ["midostaurin", "gilteritinib"],
     "FLT3-mutant AML: FLT3 inhibitor"),
    ("IDH_mutant", "mutation", ["ivosidenib", "enasidenib"],
     "IDH-mutant: IDH inhibitor"),
]


def select_targeted_therapy(biopsy: TumorBiopsy) -> list[dict[str, str]]:
    """Select appropriate targeted therapies based on tumor biomarkers.

    Implements NCCN-inspired molecular tumor board decision logic.

    Args:
        biopsy: Tumor molecular profile

    Returns:
        Ranked list of therapy recommendations with rationale.
        Each dict has keys: "drug", "pathway", "rationale", "priority"
    """
    recommendations: list[dict[str, str]] = []
    seen_drugs: set[str] = set()

    for rule_name, condition, drugs, rationale in _THERAPY_RULES:
        matched = False
        if condition == "mutation":
            matched = any(biopsy.has_mutation(d) for d in ["EGFR", "BRAF", "KRAS", "FLT3", "IDH"])
        elif condition == "fusion":
            matched = any(biopsy.has_fusion(f) for f in ["ALK", "BCR-ABL", "ROS1", "RET", "NTRK"])
        elif condition == "amplification":
            matched = any(biopsy.has_amplification(g) for g in ["HER2", "FGFR1", "MET"])
        elif condition == "pd_l1":
            matched = biopsy.pd_l1_expression >= 0.5
        elif condition == "msi":
            matched = biopsy.msi_status == "MSI-H"
        elif condition == "tmb":
            matched = biopsy.tmb_per_mb >= 10.0
        elif condition == "hrd":
            matched = biopsy.hr_status == "HRD"

        if matched:
            for drug in drugs:
                if drug not in seen_drugs:
                    seen_drugs.add(drug)
                    pathway = ""
                    drug_upper = drug.upper()
                    for tp, pw in TARGET_TO_PATHWAY.items():
                        if tp.upper().replace("-", "") in drug_upper.replace("-", ""):
                            pathway = pw
                            break
                    recommendations.append({
                        "drug": drug,
                        "pathway": pathway or "unknown",
                        "rationale": rationale,
                        "priority": "high" if rule_name in (
                            "EGFR_mutant", "ALK_fusion", "BCR_ABL",
                        ) else "medium",
                    })

    return recommendations


@dataclass
class CancerODE:
    """Tumor growth with Gompertz kinetics, immune surveillance, and per-pathway therapy.

    States:
        tumor_volume: relative tumor burden (0=none, 1=lethal)
        immune_surveillance: 0-1 (immune control of tumor)
        angiogenesis: 0-1 (new blood vessel formation)

    Drug targets: cytotoxic chemo (kills proliferating cells),
    targeted therapy (inhibits growth signaling), immunotherapy (boosts surveillance).

    Per-pathway effects (doc/33 Phase 4):
        pathway_effects: dict mapping pathway name → inhibition level (0.0–1.0)
        Supported pathways: egfr, braf, vegfr, alk, her2, microtubule,
        topoisomerase, dna_damage, pd_l1, ctla4, parp, bcr_abl, kras,
        met, ret, ntrk, pi3k, mtor, cdk46, idh, flt3, jak
    """

    tumor_volume: float = 0.01   # initial small tumor (normalized)
    immune_surveillance: float = 0.8
    angiogenesis: float = 0.5

    # --- Gompertz parameters ---
    growth_rate: float = 0.0003   # intrinsic growth rate (1/h); ~3-month doubling
                                  # Laird 1964: Gompertz analysis of tumor growth
    carrying_capacity: float = 1.0  # maximum tumor volume (normalized)
    doubling_time_h: float = 2310.0   # ~96 days (NSCLC typical;
                                      # Chaudhary et al. 2016: median 96 days)

    # --- Immune parameters ---
    immune_kill_rate: float = 0.0001  # immune-mediated tumor cell death
    tumor_escape_rate: float = 0.00002  # tumor immune evasion

    # --- Legacy drug effects (backward compatible) ---
    chemo_kill_rate: float = 0.0   # set by drug PD (legacy)
    targeted_inhibition: float = 0.0  # set by drug PD (legacy)
    immunotherapy_boost: float = 0.0

    # --- Per-pathway effects (doc/33 Phase 4 §12.1) ---
    pathway_effects: dict[str, float] = field(default_factory=dict)

    # --- Drug kill capacity (1/h) ---
    # Maximum drug-mediated kill rate; independent of intrinsic growth_rate.
    # For targeted therapy (EGFR/ALK/BRAF inhibitors) ~0.001-0.003/hr,
    # for cytotoxic chemo ~0.002-0.008/hr.
    # Shaked et al. 2010: tumor cell kill rates for cytotoxic agents.
    drug_kill_capacity: float = 0.001

    # --- Tumor heterogeneity (doc/33 Phase 4 §12.2) ---
    heterogeneity: TumorHeterogeneity | None = None

    def _total_drug_kill(self) -> float:
        """Compute total drug kill rate from legacy + per-pathway effects.

        Pathway effects are scaled by *drug_kill_capacity* (1/h) so that
        inhibition reduces tumor burden at a pharmacologically realistic
        rate independent of intrinsic growth.
        """
        legacy_kill = self.chemo_kill_rate + self.targeted_inhibition
        pathway_kill = sum(self.pathway_effects.values()) * self.drug_kill_capacity
        return legacy_kill + pathway_kill

    def step(self, dt_h: float) -> None:
        """Advance one hour."""
        V = self.tumor_volume
        K = self.carrying_capacity

        # Gompertz growth: dV/dt = r * V * ln(K/V)
        if V > 0 and V < K:
            growth = self.growth_rate * V * math.log(K / V)
        else:
            growth = 0.0

        # Immune killing
        immune_kill = self.immune_kill_rate * self.immune_surveillance * V

        # Drug killing (cytotoxic + targeted + per-pathway)
        drug_kill = self._total_drug_kill() * V

        # Tumor volume dynamics
        self.tumor_volume = max(0.0, V + dt_h * (
            growth - immune_kill - drug_kill))

        # Immune surveillance: tumor can escape, immunotherapy boosts
        immuno_boost = self.pathway_effects.get("pd_l1", 0.0) * 0.3 + \
            self.pathway_effects.get("ctla4", 0.0) * 0.3 + \
            self.immunotherapy_boost * 0.1
        self.immune_surveillance -= dt_h * (
            self.tumor_escape_rate * V - immuno_boost)
        self.immune_surveillance = max(0.0, min(1.0, self.immune_surveillance))

        # Angiogenesis: VEGF pathway inhibition reduces angiogenesis
        vegf_inhib = self.pathway_effects.get("vegfr", 0.0)
        self.angiogenesis = min(1.0, V * 0.8 * (1.0 - 0.7 * vegf_inhib))

        # Clonal heterogeneity dynamics (if enabled)
        if self.heterogeneity is not None:
            het_metrics = self.heterogeneity.step(
                dt_h, self.pathway_effects,
                drug_kill_capacity=self.drug_kill_capacity,
            )
            # total_volume is pre-normalization aggregate of clone fractions;
            # it reflects net growth/shrinkage (< 1.0 = kill > growth).
            het_growth_index = het_metrics["total_volume"]
            if het_growth_index > 0.0:
                self.tumor_volume = max(0.0, self.tumor_volume * het_growth_index)
            else:
                self.tumor_volume = 0.0


# ============================================================================
# 4. Autoimmune RA (Joint Inflammation)
# ============================================================================


@dataclass
class AutoimmuneRAODE:
    """Rheumatoid arthritis: joint inflammation + cytokine dynamics.

    States:
        joint_inflammation: 0-1 (mean across joints)
        synovial_tnf: pg/mL (local TNF in synovial fluid)
        erosive_damage: 0-1 (permanent joint damage)

    Drug targets: NSAIDs (↓inflammation), DMARDs (↓immune activation),
    biologics (anti-TNF, anti-IL-6)
    """

    joint_inflammation: float = 0.3
    synovial_tnf: float = 50.0   # pg/mL (elevated in RA)
    erosive_damage: float = 0.0

    # --- Dynamics ---
    inflammation_drive: float = 0.02   # spontaneous flare rate
    resolution_rate: float = 0.01      # natural resolution
    damage_accumulation: float = 0.0001  # erosion per inflammation-hour

    # --- Flare ceiling: caps post-drug rebound at baseline severity ---
    baseline_severity: float = 0.6  # set from config, prevents rebound to 1.0

    # --- Drug effects ---
    nsaid_effect: float = 0.0
    dmard_effect: float = 0.0
    biologic_effect: float = 0.0

    def step(self, dt_h: float) -> None:
        """Advance one hour."""
        I = self.joint_inflammation
        TNF = self.synovial_tnf

        # Drug suppresses the autoimmune activation drive (cytokine signaling),
        # not the inflammation directly.  Existing inflammation resolves via
        # natural resolution rate, giving a gradual clinical decline.
        effective_drive = self.inflammation_drive * max(0.0, 1.0 - self.dmard_effect)
        dI = (effective_drive * TNF / 50.0
              - self.resolution_rate * I
              - self.nsaid_effect * I)

        # Synovial TNF: JAK inhibitor suppresses production (cytokine signaling)
        dTNF = 10.0 * I * max(0.0, 1.0 - self.dmard_effect) - 2.0 * TNF / 50.0

        # Erosive damage accumulates
        self.erosive_damage += dt_h * self.damage_accumulation * I * TNF / 50.0

        new_I = max(0.0, min(1.0, I + dt_h * dI))
        # Flare ceiling: inflammation cannot exceed baseline severity * 1.1
        # once it has been reduced by treatment (prevents rebound to 1.0)
        if new_I > self.baseline_severity:
            new_I = min(new_I, self.baseline_severity * 1.1)
        self.joint_inflammation = new_I
        self.synovial_tnf = max(0.0, TNF + dt_h * dTNF)
        self.erosive_damage = min(1.0, self.erosive_damage)


# ============================================================================
# 5. Neurological (Neurodegeneration)
# ============================================================================


@dataclass
class NeurologicalODE:
    """Neurodegeneration model: synaptic density + neuroinflammation.

    States:
        synaptic_density: 0-1 (remaining synaptic connections)
        neuroinflammation: 0-1 (microglial activation)
        cognitive_score: 0-1 (MMSE-like, derived from synaptic density)

    Applicable to: Alzheimer's, Parkinson's, ALS
    Drug targets: cholinesterase inhibitors (symptomatic),
    neuroprotective agents (slow progression)
    """

    synaptic_density: float = 0.8
    neuroinflammation: float = 0.2
    cognitive_score: float = 0.8

    # --- Progression ---
    neurodegeneration_rate: float = 0.0002  # per hour
    inflammation_toxicity: float = 0.001    # inflammation damages synapses
    neuroprotection: float = 0.0            # drug effect

    # --- Drug effects ---
    cholinesterase_inhibition: float = 0.0  # symptomatic improvement
    disease_modifying_effect: float = 0.0   # slows progression

    def step(self, dt_h: float) -> None:
        """Advance one hour."""
        S = self.synaptic_density
        N = self.neuroinflammation

        # Synaptic loss: degeneration + inflammatory damage
        dS = -(self.neurodegeneration_rate * (1.0 - self.disease_modifying_effect)
               + self.inflammation_toxicity * N) * S

        # Neuroinflammation: self-perpetuating
        dN = 0.001 * N * (1.0 - N) - 0.01 * N

        self.synaptic_density = max(0.0, S + dt_h * dS)
        self.neuroinflammation = max(0.0, min(1.0, N + dt_h * dN))

        # Cognitive score: synaptic density + symptomatic boost
        self.cognitive_score = min(1.0, (
            self.synaptic_density + 0.1 * self.cholinesterase_inhibition))


# ============================================================================
# 6. Renal (Nephron Loss + CKD Progression)
# ============================================================================


@dataclass
class RenalODE:
    """Renal ODE: nephron loss with compensatory hyperfiltration.

    States:
        nephron_mass: 0-1 (remaining functional nephrons)
        single_nephron_gfr: compensatory increase per nephron
        proteinuria: g/day (marker of glomerular damage)

    Overlaps with renal_model.py but provides ODE dynamics
    for integration with organ crosstalk.
    """

    nephron_mass: float = 1.0
    single_nephron_gfr: float = 1.0  # compensatory multiplier
    proteinuria: float = 0.1         # g/day

    # --- Dynamics ---
    nephron_loss_rate: float = 0.0001  # per hour
    hyperfiltration_adaptation: float = 0.01  # compensation rate
    proteinuria_toxicity: float = 0.0001  # proteinuria → nephron loss

    # --- Drug effects ---
    acei_effect: float = 0.0   # reduces proteinuria, slows loss
    sglt2_effect: float = 0.0  # reduces hyperfiltration

    def step(self, dt_h: float) -> None:
        """Advance one hour."""
        N = self.nephron_mass

        # Nephron loss: baseline + proteinuria toxicity
        loss = (self.nephron_loss_rate + self.proteinuria_toxicity * self.proteinuria)
        loss *= (1.0 - self.acei_effect * 0.5)  # ACEi slows loss

        # Compensatory hyperfiltration
        if N < 1.0:
            target_sngfr = 1.0 / max(0.1, N)
            self.single_nephron_gfr += dt_h * (
                self.hyperfiltration_adaptation * (target_sngfr - self.single_nephron_gfr))
            # SGLT2 reduces hyperfiltration
            self.single_nephron_gfr *= (1.0 - 0.3 * self.sglt2_effect)

        self.nephron_mass = max(0.05, N - dt_h * loss)
        self.proteinuria = max(0.0, self.proteinuria + dt_h * (
            0.01 * (1.0 - N) - 0.1 * self.acei_effect * self.proteinuria))


# ============================================================================
# 7. Hepatic (Fibrosis Progression)
# ============================================================================


@dataclass
class HepaticODE:
    """Hepatic fibrosis: collagen deposition + synthetic function.

    States:
        fibrosis_stage: 0-4 (METAVIR scale, continuous)
        synthetic_function: 0-1 (albumin, clotting factor production)
        portal_pressure: mmHg (HVPG)

    Drug targets: antivirals (HBV/HCV), Ursodiol (cholestatic),
    diuretics (ascites), beta-blockers (portal hypertension)
    """

    fibrosis_stage: float = 0.0    # 0-4 continuous
    synthetic_function: float = 1.0
    portal_pressure: float = 5.0   # mmHg (normal <5)

    # --- Progression ---
    fibrosis_progression_rate: float = 0.00005  # per hour
    inflammation_to_fibrosis: float = 0.0001   # inflammation drives fibrosis
    regression_rate: float = 0.00001           # slow regression without cause

    # --- Inflammation driver ---
    hepatic_inflammation: float = 0.2  # ALT/AST driven

    # --- Drug effects ---
    antiviral_effect: float = 0.0
    anti_fibrotic_effect: float = 0.0

    def step(self, dt_h: float) -> None:
        """Advance one hour."""
        F = self.fibrosis_stage

        # Fibrosis progression: inflammation-driven + basal
        dF = (self.fibrosis_progression_rate
              + self.inflammation_to_fibrosis * self.hepatic_inflammation)
        dF *= (1.0 - self.antiviral_effect * 0.8)
        dF *= (1.0 - self.anti_fibrotic_effect * 0.5)
        dF -= self.regression_rate * (1.0 - self.antiviral_effect)

        self.fibrosis_stage = max(0.0, min(4.0, F + dt_h * dF))

        # Synthetic function: inversely related to fibrosis
        self.synthetic_function = max(0.1, 1.0 - 0.2 * F)

        # Portal pressure: increases with fibrosis
        self.portal_pressure = 5.0 + 5.0 * F / 4.0


# ============================================================================
# 8. Hematological (Myelodysplasia)
# ============================================================================


@dataclass
class HematologicalODE:
    """Myelodysplastic syndrome: ineffective hematopoiesis.

    States:
        stem_cell_pool: 0-1 (functional hematopoietic stem cells)
        ineffective_ratio: fraction of output that is dysplastic
        blast_percentage: % blasts (disease severity marker)

    Overlaps with hematology_model.py Friberg but captures the
    underlying stem cell defect.
    """

    stem_cell_pool: float = 0.8
    ineffective_ratio: float = 0.3
    blast_percentage: float = 5.0

    # --- Progression ---
    stem_cell_loss_rate: float = 0.00005
    blast_expansion_rate: float = 0.001

    # --- Drug effects ---
    hypomethylating_effect: float = 0.0  # azacitidine/decitabine
    growth_factor_effect: float = 0.0    # EPO/G-CSF

    def step(self, dt_h: float) -> None:
        """Advance one hour."""
        S = self.stem_cell_pool

        # Stem cell loss
        self.stem_cell_pool = max(0.05, S - dt_h * self.stem_cell_loss_rate *
                                  (1.0 - self.hypomethylating_effect * 0.6))

        # Blast expansion
        self.blast_percentage += dt_h * (
            self.blast_expansion_rate * (1.0 - S) * 100.0
            * (1.0 - self.hypomethylating_effect * 0.7))
        self.blast_percentage = max(0.0, min(90.0, self.blast_percentage))

        # Ineffective ratio: higher with worse disease
        self.ineffective_ratio = min(0.8, 0.3 + 0.1 * (1.0 - S))


# ============================================================================
# 9. Respiratory (Asthma / COPD)
# ============================================================================


@dataclass
class RespiratoryODE:
    """Respiratory disease model (asthma / COPD).

    States:
        airway_resistance: airway resistance (baseline 1.0)
        inflammation_score: airway inflammation (0-1)
        bronchodilation: bronchodilator responsiveness (0-1)
        fev1_percent: FEV1 as % predicted

    Drug targets: bronchodilators (↓resistance), corticosteroids (↓inflammation),
    leukotriene antagonists (↓inflammation), biologics (anti-IL5/IL13)
    """

    airway_resistance: float = 1.0
    inflammation_score: float = 0.0
    bronchodilation: float = 0.0
    fev1_percent: float = 80.0

    # Disease modifiers
    asthma_severity: float = 0.0
    copd_severity: float = 0.0
    smoking_effect: float = 0.0

    def step(self, dt_h: float, drug_bronchodilator: float = 1.0,
             drug_anti_inflammatory: float = 1.0) -> None:
        """Advance one hour."""
        # Airway resistance increases with inflammation and smoking
        base_resistance = 1.0 + 0.5 * self.asthma_severity + 0.8 * self.copd_severity
        base_resistance += 0.3 * self.smoking_effect
        self.airway_resistance = base_resistance * drug_bronchodilator

        # Inflammation resolves with anti-inflammatory drugs
        self.inflammation_score *= (1.0 - 0.02 * drug_anti_inflammatory)
        self.inflammation_score += 0.001 * (self.asthma_severity + self.copd_severity)
        self.inflammation_score = max(0.0, min(1.0, self.inflammation_score))

        # FEV1 depends on resistance and inflammation
        self.fev1_percent = max(15.0, 80.0 - 20.0 * self.airway_resistance
                                - 30.0 * self.inflammation_score)


# ============================================================================
# 10. Infectious Disease (HIV / TB)
# ============================================================================


@dataclass
class InfectiousDiseaseODE:
    """Infectious disease model (HIV / TB / bacterial).

    States:
        immune_function: overall immune function (0-1)
        viral_bacterial_load: pathogen burden (log scale)
        cd4_count: CD4+ T cell count (cells/uL)
        inflammation: systemic inflammation score (0-1)

    Drug targets: antiretrovirals/antibiotics (↓load), immunomodulators (↑function)
    """

    immune_function: float = 1.0
    viral_bacterial_load: float = 0.0  # log10 scale
    cd4_count: float = 800.0
    inflammation: float = 0.0

    # Disease modifiers
    hiv_severity: float = 0.0
    tb_severity: float = 0.0
    bacterial_severity: float = 0.0

    def step(self, dt_h: float, drug_effectiveness: float = 0.0) -> None:
        """Advance one hour."""
        # Pathogen load grows without treatment
        growth_rate = 0.005 * (self.hiv_severity + self.tb_severity + self.bacterial_severity)
        self.viral_bacterial_load += dt_h * growth_rate * (1.0 - drug_effectiveness * 0.95)

        # Drug also actively clears existing pathogen (kills bacteria directly)
        # clearance_rate scales with drug effectiveness: full effect → 0.15/h (~5h half-life)
        if drug_effectiveness > 0.0 and self.viral_bacterial_load > 0.0:
            clearance_rate = 0.15 * drug_effectiveness
            self.viral_bacterial_load -= dt_h * clearance_rate * self.viral_bacterial_load

        self.viral_bacterial_load = max(-2.0, min(8.0, self.viral_bacterial_load))

        # Immune function degrades with pathogen burden
        self.immune_function = max(0.05, 1.0 - 0.12 * self.viral_bacterial_load)

        # CD4 count tracks immune function
        self.cd4_count = max(20.0, 800.0 * self.immune_function)

        # Inflammation from pathogen and immune activation
        self.inflammation = min(1.0, 0.2 * self.viral_bacterial_load / 8.0)


# ============================================================================
# 11. Gastrointestinal (GERD / IBD)
# ============================================================================


@dataclass
class GastrointestinalODE:
    """Gastrointestinal disease model (GERD / IBD).

    States:
        acid_secretion: gastric acid output (relative)
        mucosal_integrity: mucosal barrier function (0-1)
        motility: GI motility (relative)
        pain_score: abdominal pain (0-10)

    Drug targets: PPIs (↓acid), H2 blockers (↓acid), biologics (↓inflammation)
    """

    acid_secretion: float = 1.0
    mucosal_integrity: float = 1.0
    motility: float = 1.0
    pain_score: float = 0.0

    # Disease modifiers
    gerd_severity: float = 0.0
    ibd_severity: float = 0.0

    def step(self, dt_h: float, drug_acid_suppression: float = 1.0,
             drug_anti_inflammatory: float = 1.0) -> None:
        """Advance one hour."""
        # Acid secretion increases with GERD
        self.acid_secretion = (1.0 + 0.5 * self.gerd_severity) * drug_acid_suppression
        self.acid_secretion = max(0.1, self.acid_secretion)

        # Mucosal integrity decreases with IBD and acid
        self.mucosal_integrity = max(0.1, 1.0 - 0.3 * self.ibd_severity
                                     - 0.2 * self.acid_secretion)
        self.mucosal_integrity *= (1.0 + 0.1 * drug_anti_inflammatory)
        self.mucosal_integrity = min(1.0, self.mucosal_integrity)

        # Pain from mucosal damage and inflammation
        self.pain_score = max(0.0, min(10.0,
            3.0 * self.gerd_severity + 5.0 * self.ibd_severity
            - 2.0 * drug_acid_suppression - 1.5 * drug_anti_inflammatory))


# ============================================================================
# 12. Endocrine (Thyroid / Adrenal)
# ============================================================================


@dataclass
class EndocrineODE:
    """Endocrine disease model (hypothyroidism / hyperthyroidism).

    States:
        t4_level: free T4 level (pM)
        tsh_level: TSH level (mIU/L)
        metabolic_rate: relative metabolic rate (1.0 = normal)

    Drug targets: levothyroxine (↑T4), methimazole (↓T4), beta-blockers (↓HR)
    """

    t4_level: float = 120.0  # pM
    tsh_level: float = 2.5   # mIU/L
    metabolic_rate: float = 1.0

    # Disease modifiers
    hypothyroid_severity: float = 0.0
    hyperthyroid_severity: float = 0.0

    def step(self, dt_h: float, drug_t4_supplement: float = 0.0,
             drug_antithyroid: float = 1.0) -> None:
        """Advance one hour."""
        # T4 level
        if self.hypothyroid_severity > 0:
            self.t4_level = max(20.0, 120.0 * (1.0 - 0.8 * self.hypothyroid_severity)
                                + drug_t4_supplement * 80.0)
        elif self.hyperthyroid_severity > 0:
            self.t4_level = min(400.0, 120.0 * (1.0 + 2.0 * self.hyperthyroid_severity)
                                * drug_antithyroid)
        else:
            self.t4_level = 120.0 + drug_t4_supplement * 80.0

        # TSH: inverse relationship with T4
        self.tsh_level = max(0.1, 2.5 * (120.0 / max(self.t4_level, 1.0)) ** 1.5)

        # Metabolic rate tracks T4
        self.metabolic_rate = max(0.3, min(2.5, self.t4_level / 120.0))


# ============================================================================
# Factory
# ============================================================================


def create_disease_model(
    disease_name: str,
    severity: float = 0.0,
    category: str = "",
) -> object:
    """Create appropriate ODE model for a disease category.

    First tries keyword matching on *disease_name*.  If no match is found
    and *category* is provided, falls back to category-based dispatch.
    Returns the disease-specific ODE model with severity applied.
    """
    name_lower = disease_name.lower().replace(" ", "_").replace("-", "_")

    if any(k in name_lower for k in ["cardiovascular", "hypertension", "heart_failure", "cv"]):
        cv = CardiovascularODE()
        cv.atherosclerosis_severity = severity
        cv.hypertension_severity = severity * 0.5
        cv.heart_failure_severity = severity * 0.3
        return cv

    elif any(k in name_lower for k in ["diabetes", "t2d", "metabolic", "type_2"]):
        t2d = MetabolicT2DODE()
        t2d.t2d_severity = severity
        t2d.beta_cell_function = max(0.1, 1.0 - severity * 0.8)
        return t2d

    elif any(k in name_lower for k in ["cancer", "tumor", "oncology", "nsclc"]):
        ca = CancerODE()
        ca.tumor_volume = min(0.5, severity * 0.5)
        ca.immune_surveillance = max(0.1, 0.8 - severity * 0.5)
        return ca

    elif any(k in name_lower for k in ["rheumatoid", "autoimmune", "lupus"]):
        ra = AutoimmuneRAODE()
        ra.joint_inflammation = severity
        ra.synovial_tnf = 50.0 + severity * 200.0
        ra.baseline_severity = severity
        return ra

    elif any(k in name_lower for k in ["alzheimer", "parkinson", "neurodegener", "neuro"]):
        neuro = NeurologicalODE()
        neuro.synaptic_density = max(0.2, 1.0 - severity * 0.7)
        neuro.neuroinflammation = severity * 0.5
        return neuro

    elif any(k in name_lower for k in ["ckd", "renal", "kidney"]):
        renal = RenalODE()
        renal.nephron_mass = max(0.1, 1.0 - severity * 0.8)
        renal.proteinuria = severity * 3.0
        return renal

    elif any(k in name_lower for k in ["cirrhosis", "hepatic", "liver", "fibrosis"]):
        hep = HepaticODE()
        hep.fibrosis_stage = severity * 4.0
        hep.synthetic_function = max(0.2, 1.0 - severity * 0.7)
        return hep

    elif any(k in name_lower for k in ["mds", "myelodysplastic", "leukemia", "hematological"]):
        heme = HematologicalODE()
        heme.stem_cell_pool = max(0.1, 1.0 - severity * 0.7)
        heme.blast_percentage = 5.0 + severity * 40.0
        return heme

    elif any(k in name_lower for k in ["asthma", "copd", "respiratory", "bronchial"]):
        resp = RespiratoryODE()
        resp.asthma_severity = severity if "asthma" in name_lower else 0.0
        resp.copd_severity = severity if "copd" in name_lower else 0.0
        resp.inflammation_score = severity * 0.5
        resp.fev1_percent = max(15.0, 80.0 - 65.0 * severity)
        return resp

    elif any(k in name_lower for k in ["hiv", "tb", "tuberculosis", "infectious", "bacterial"]):
        inf = InfectiousDiseaseODE()
        inf.hiv_severity = severity if "hiv" in name_lower else 0.0
        inf.tb_severity = severity if any(k in name_lower for k in ["tb", "tuberculosis"]) else 0.0
        inf.bacterial_severity = severity if "bacterial" in name_lower else 0.0
        inf.viral_bacterial_load = severity * 5.0
        inf.cd4_count = max(20.0, 800.0 * (1.0 - severity * 0.9))
        return inf

    elif any(k in name_lower for k in ["gerd", "gerd", "gastro", "reflux", "ibd", "crohn", "colitis"]):
        gi = GastrointestinalODE()
        gi.gerd_severity = severity if any(k in name_lower for k in ["gerd", "reflux"]) else 0.0
        gi.ibd_severity = severity if any(k in name_lower for k in ["ibd", "crohn", "colitis"]) else 0.0
        gi.acid_secretion = 1.0 + severity * 0.5
        gi.mucosal_integrity = max(0.1, 1.0 - severity * 0.6)
        return gi

    elif any(k in name_lower for k in ["thyroid", "hypothyroid", "hyperthyroid", "endocrine"]):
        endo = EndocrineODE()
        if "hypo" in name_lower:
            endo.hypothyroid_severity = severity
            endo.t4_level = max(20.0, 120.0 * (1.0 - 0.8 * severity))
        else:
            endo.hyperthyroid_severity = severity
            endo.t4_level = min(400.0, 120.0 * (1.0 + 2.0 * severity))
        endo.metabolic_rate = endo.t4_level / 120.0
        return endo

    # --- Category-based fallback (doc/33 robust dispatch) ---
    cat_lower = category.lower().strip() if category else ""
    if cat_lower:
        if cat_lower in ("cardiovascular",):
            cv = CardiovascularODE()
            cv.atherosclerosis_severity = severity
            cv.hypertension_severity = severity * 0.5
            cv.heart_failure_severity = severity * 0.3
            return cv
        elif cat_lower in ("respiratory",):
            resp = RespiratoryODE()
            resp.inflammation_score = severity * 0.5
            resp.fev1_percent = max(15.0, 80.0 - 65.0 * severity)
            return resp
        elif cat_lower in ("neurological",):
            neuro = NeurologicalODE()
            neuro.synaptic_density = max(0.2, 1.0 - severity * 0.7)
            neuro.neuroinflammation = severity * 0.5
            return neuro
        elif cat_lower in ("metabolic",):
            t2d = MetabolicT2DODE()
            t2d.t2d_severity = severity
            t2d.beta_cell_function = max(0.1, 1.0 - severity * 0.8)
            return t2d
        elif cat_lower in ("infectious",):
            inf = InfectiousDiseaseODE()
            inf.viral_bacterial_load = severity * 5.0
            inf.cd4_count = max(20.0, 800.0 * (1.0 - severity * 0.9))
            return inf
        elif cat_lower in ("hematological",):
            heme = HematologicalODE()
            heme.stem_cell_pool = max(0.1, 1.0 - severity * 0.7)
            heme.blast_percentage = 5.0 + severity * 40.0
            return heme
        elif cat_lower in ("autoimmune",):
            ra = AutoimmuneRAODE()
            ra.joint_inflammation = severity
            ra.synovial_tnf = 50.0 + severity * 200.0
            ra.baseline_severity = severity
            return ra
        elif cat_lower in ("endocrine",):
            endo = EndocrineODE()
            endo.hypothyroid_severity = severity
            endo.t4_level = max(20.0, 120.0 * (1.0 - 0.8 * severity))
            endo.metabolic_rate = endo.t4_level / 120.0
            return endo
        elif cat_lower in ("gastrointestinal",):
            gi = GastrointestinalODE()
            gi.ibd_severity = severity
            gi.acid_secretion = 1.0 + severity * 0.5
            gi.mucosal_integrity = max(0.1, 1.0 - severity * 0.6)
            return gi
        elif cat_lower in ("hepatic", "liver"):
            hep = HepaticODE()
            hep.fibrosis_stage = severity * 4.0
            hep.synthetic_function = max(0.2, 1.0 - severity * 0.7)
            return hep
        elif cat_lower in ("renal", "kidney"):
            renal = RenalODE()
            renal.nephron_mass = max(0.1, 1.0 - severity * 0.8)
            renal.proteinuria = severity * 3.0
            return renal
        elif cat_lower in ("oncology", "cancer"):
            ca = CancerODE()
            ca.tumor_volume = min(0.5, severity * 0.5)
            ca.immune_surveillance = max(0.1, 0.8 - severity * 0.5)
            return ca

        return _GenericDiseaseModel(severity=severity)

    else:
        return _GenericDiseaseModel(severity=severity)


@dataclass
class _GenericDiseaseModel:
    """Generic fallback for diseases without specific ODE models.

    Provides basic organ-system coupling so that even unknown diseases
    produce realistic lab/vital perturbations rather than a flat severity
    drift.  Organ function proxies feed into the ClinicalLabModel via
    the VP recording loop.
    """
    severity: float = 0.0
    progression_rate: float = 0.0001

    # Organ function proxies (0–1, 1 = healthy)
    liver_function: float = 1.0
    kidney_function: float = 1.0
    immune_activation: float = 0.0  # 0–1, systemic inflammation
    inflammation_score: float = 0.0  # 0–1, CRP driver

    def step(self, dt_h: float, drug_effectiveness: float = 0.0) -> None:
        self.severity += dt_h * self.progression_rate * (1.0 - drug_effectiveness * 0.5)
        self.severity = max(0.0, min(1.0, self.severity))
        # Organ function declines with severity
        self.liver_function = max(0.3, 1.0 - 0.4 * self.severity)
        self.kidney_function = max(0.3, 1.0 - 0.3 * self.severity)
        # Inflammation rises with severity, tempered by drug effect
        self.inflammation_score = min(1.0, self.severity * 0.6 * (1.0 - drug_effectiveness * 0.3))
        self.immune_activation = min(1.0, self.inflammation_score * 0.8)
