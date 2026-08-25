"""Rule-based innate immune agent-based model (doc/31 §2.4, §5.1 #3).

Population-level ODE approximation of the innate immune system, driving
CRP and WBC counts mechanistically instead of proxy formulas.  The model
tracks three compartments (tissue, blood, lymphoid organ) with agents
(macrophages, neutrophils, dendritic cells, T-cells) and soluble signals
(TNF-α, IL-1β, IL-6, IL-10).

Key output channels:
    - IL-6 → hepatic CRP production (replaces proxy in ClinicalLabModel)
    - Neutrophil count → WBC differential (replaces manual scaling)
    - TNF-α → systemic inflammation score (feeds vitals temperature)

Module structure:
    CytokinePool          soluble mediator concentrations
    ImmuneCellPopulation  ODE-tracked cell counts per type
    InnateImmuneModel     coupled cytokine + cell dynamics
    CRPDriver             IL-6 → CRP hepatic production
    create_immune_model   factory

References:
- BIS-class models: Candiani et al. 2024 (432-parameter innate immune ABM)
- IIRABM: Marina et al. 2024 (rule-based innate response)
- IL-6 → CRP: Volanakis NEJM 2001; Pepys & Hirschfield J Clin Invest 2003
- Neutrophil kinetics: Lord et al. Blood 1989;PRICE et al. J Clin Invest 1976
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "CytokinePool",
    "ImmuneCellPopulation",
    "InnateImmuneModel",
    "CRPDriver",
    "create_immune_model",
]


# ============================================================================
# Cytokine Pool (Soluble Mediators)
# ============================================================================


@dataclass
class CytokinePool:
    """Cytokine concentrations (pg/mL for cytokines, ng/mL for IL-6→CRP).

    Steady-state values represent healthy baseline.  Inflammation
    drives TNF-α, IL-1β, IL-6 up; IL-10 provides anti-inflammatory
    counterbalance.
    """

    tnf_alpha: float = 5.0       # pg/mL (healthy ~5)
    il1_beta: float = 2.0        # pg/mL (healthy ~2)
    il6: float = 1.0             # pg/mL (healthy ~1; drives CRP)
    il10: float = 5.0            # pg/mL (healthy ~5; anti-inflammatory)

    # Rate constants (1/h)
    tnf_half_life_h: float = 0.5   # TNF-α ~30 min
    il1_half_life_h: float = 1.0   # IL-1β ~1 h
    il6_half_life_h: float = 1.5   # IL-6 ~90 min
    il10_half_life_h: float = 2.0  # IL-10 ~2 h

    # Production rates (pg/mL/h) — activated by PAMPs/DAMPs
    tnf_production_rate: float = 10.0
    il1_production_rate: float = 8.0
    il6_production_rate: float = 12.0
    il10_production_rate: float = 6.0

    # PAMP/DAMP signal (0=none, 1=severe infection)
    pathogen_signal: float = 0.0

    def step(self, dt_h: float) -> None:
        """Advance cytokine dynamics one hour."""
        k_tnf = math.log(2) / (self.tnf_half_life_h + 1e-12)
        k_il1 = math.log(2) / (self.il1_half_life_h + 1e-12)
        k_il6 = math.log(2) / (self.il6_half_life_h + 1e-12)
        k_il10 = math.log(2) / (self.il10_half_life_h + 1e-12)

        # Production stimulated by pathogen + pro-inflammatory feedback
        stim = self.pathogen_signal
        self.tnf_alpha += dt_h * (
            self.tnf_production_rate * stim - k_tnf * self.tnf_alpha)
        self.il1_beta += dt_h * (
            self.il1_production_rate * stim - k_il1 * self.il1_beta)
        self.il6 += dt_h * (
            self.il6_production_rate * stim - k_il6 * self.il6)
        # IL-10 produced in response to pro-inflammatory cytokines
        anti_stim = stim + 0.1 * (self.tnf_alpha + self.il1_beta) / 10.0
        self.il10 += dt_h * (
            self.il10_production_rate * anti_stim - k_il10 * self.il10)

        # Floor
        self.tnf_alpha = max(0.0, self.tnf_alpha)
        self.il1_beta = max(0.0, self.il1_beta)
        self.il6 = max(0.0, self.il6)
        self.il10 = max(0.0, self.il10)


# ============================================================================
# Immune Cell Populations (ODE-Tracked)
# ============================================================================


@dataclass
class ImmuneCellPopulation:
    """Cell counts (×10³/µL) for innate immune populations.

    Populations:
        neutrophils: circulating neutrophils (major WBC component)
        macrophages: tissue-resident macrophages
        monocytes: blood monocytes (precursors to macrophages)
        dendritic_cells: antigen-presenting cells
        t_cells: adaptive T-lymphocytes (simplified)
    """

    neutrophils: float = 4.0      # ×10³/µL (normal 2-7)
    macrophages: float = 0.5      # ×10³/µL tissue equivalent
    monocytes: float = 0.4        # ×10³/µL (normal 0.2-0.8)
    dendritic_cells: float = 0.1  # ×10³/µL
    t_cells: float = 1.5          # ×10³/µL (normal 1-2)

    # Production rates (×10³/µL/h)
    neutrophil_production: float = 0.1   # bone marrow output
    monocyte_production: float = 0.02
    t_cell_production: float = 0.01

    # Clearance rates (1/h)
    neutrophil_clearance: float = 0.05   # ~14 h half-life in blood
    monocyte_clearance: float = 0.03
    macrophage_clearance: float = 0.01   # long-lived tissue cells

    # Mobilisation: cytokine-driven release from bone marrow
    gcsf_sensitivity: float = 0.5  # neutrophil response to G-CSF/IL-6

    def step(self, dt_h: float, il6: float, tnf: float) -> None:
        """Advance cell dynamics one hour.

        Args:
            il6: IL-6 level (pg/mL) — drives neutrophil mobilisation
            tnf: TNF-α level (pg/mL) — drives monocyte differentiation
        """
        # Neutrophil dynamics: production + mobilisation - clearance
        mobilisation = self.gcsf_sensitivity * max(0.0, il6 - 1.0) / 10.0
        self.neutrophils += dt_h * (
            self.neutrophil_production * (1.0 + mobilisation)
            - self.neutrophil_clearance * self.neutrophils)

        # Monocyte dynamics
        self.monocytes += dt_h * (
            self.monocyte_production * (1.0 + tnf / 20.0)
            - self.monocyte_clearance * self.monocytes)

        # Macrophage: differentiated from monocytes, slow turnover
        differentiation = 0.1 * self.monocytes  # monocyte → macrophage
        self.macrophages += dt_h * (
            differentiation - self.macrophage_clearance * self.macrophages)

        # Dendritic cells: tissue sentinels
        self.dendritic_cells += dt_h * (
            0.005 * (1.0 + tnf / 50.0)
            - 0.02 * self.dendritic_cells)

        # T-cells: slow adaptive response
        self.t_cells += dt_h * (
            self.t_cell_production * (1.0 + il6 / 20.0)
            - 0.005 * self.t_cells)

        # Floors
        self.neutrophils = max(0.1, self.neutrophils)
        self.macrophages = max(0.01, self.macrophages)
        self.monocytes = max(0.01, self.monocytes)
        self.dendritic_cells = max(0.01, self.dendritic_cells)
        self.t_cells = max(0.1, self.t_cells)

    def get_wbc_total(self) -> float:
        """Total WBC (×10³/µL) — approximate from populations."""
        return (self.neutrophils + self.monocytes
                + self.dendritic_cells + self.t_cells)


# ============================================================================
# Innate Immune Model (Coupled)
# ============================================================================


@dataclass
class InnateImmuneModel:
    """Coupled cytokine + cell population model.

    Represents the innate immune response to infection, injury, or
    sterile inflammation.  Drives CRP via IL-6 and WBC counts via
    cell dynamics.
    """

    cytokines: CytokinePool = field(default_factory=CytokinePool)
    cells: ImmuneCellPopulation = field(default_factory=ImmuneCellPopulation)

    # --- Cortisol suppression (from HPA axis) ---
    cortisol_suppression: float = 0.0  # 0=none, 1=complete suppression

    # --- Drug effects ---
    immunosuppression: float = 0.0  # e.g. chemotherapy, biologics
    anti_inflammatory: float = 0.0  # 0-1: PD-driven suppression of IL-6/TNF (JAK inhibitors etc.)

    # --- Disease modifiers ---
    autoimmune_activation: float = 0.0  # chronic inflammatory drive
    infection_severity: float = 0.0     # acute infection level

    # --- Base rates (saved once, restored each tick) ---
    _base_tnf_rate: float = field(default=10.0, repr=False)
    _base_il6_rate: float = field(default=12.0, repr=False)
    _base_monocyte_prod: float = field(default=0.5, repr=False)

    def __post_init__(self) -> None:
        self._base_tnf_rate = self.cytokines.tnf_production_rate
        self._base_il6_rate = self.cytokines.il6_production_rate
        self._base_monocyte_prod = self.cells.monocyte_production

    def step(self, dt_h: float) -> None:
        """Advance immune dynamics one hour."""
        # Set pathogen signal from infection
        self.cytokines.pathogen_signal = (
            self.infection_severity
            + self.autoimmune_activation * 0.5)

        # Restore base rates before applying modifiers (prevents compounding)
        self.cytokines.tnf_production_rate = self._base_tnf_rate
        self.cytokines.il6_production_rate = self._base_il6_rate
        self.cells.monocyte_production = self._base_monocyte_prod

        # Cortisol suppresses cytokine production
        if self.cortisol_suppression > 0:
            self.cytokines.tnf_production_rate *= (
                1.0 - self.cortisol_suppression * 0.7)
            self.cytokines.il6_production_rate *= (
                1.0 - self.cortisol_suppression * 0.6)

        # Immunosuppression dampens cell production
        if self.immunosuppression > 0:
            self.cells.neutrophil_production *= (
                1.0 - self.immunosuppression * 0.8)
            self.cells.monocyte_production *= (
                1.0 - self.immunosuppression * 0.7)

        # Anti-inflammatory drug effect (e.g. JAK inhibitors, NSAIDs)
        # suppresses IL-6 and TNF-α cytokine production
        if self.anti_inflammatory > 0:
            suppression = min(1.0, self.anti_inflammatory)
            self.cytokines.il6_production_rate *= (1.0 - suppression * 0.9)
            self.cytokines.tnf_production_rate *= (1.0 - suppression * 0.85)

        # Step cytokines
        self.cytokines.step(dt_h)

        # Step cells with cytokine signals
        self.cells.step(dt_h, self.cytokines.il6, self.cytokines.tnf_alpha)

    def get_il6(self) -> float:
        return self.cytokines.il6

    def get_tnf(self) -> float:
        return self.cytokines.tnf_alpha

    def get_il10(self) -> float:
        return self.cytokines.il10

    def get_neutrophils(self) -> float:
        return self.cells.neutrophils

    def get_wbc_total(self) -> float:
        return self.cells.get_wbc_total()


# ============================================================================
# CRP Driver (IL-6 → Hepatic CRP Production)
# ============================================================================


@dataclass
class CRPDriver:
    """Mechanistic IL-6 → CRP pathway.

    Hepatocyte CRP synthesis is driven by IL-6 via JAK/STAT3 signaling.
    CRP half-life ~19 h; peak CRP ~6 h after IL-6 stimulus.

    Replaces the proxy formula in ClinicalLabModel.
    """

    crp_mg_l: float = 0.5     # baseline CRP (mg/L)
    il6_threshold: float = 0.3  # IL-6 above baseline needed to drive CRP production
    crp_production_rate: float = 1.5  # mg/L per (pg/mL IL-6) per hour
    crp_clearance_rate: float = 0.036  # 1/h (~19 h half-life)
    max_crp: float = 200.0     # physiological ceiling (mg/L)

    def step(self, dt_h: float, il6_pg_ml: float) -> None:
        """Advance CRP dynamics one hour."""
        # IL-6 drives CRP production (sigmoidal response)
        effective_il6 = max(0.0, il6_pg_ml - self.il6_threshold)
        production = self.crp_production_rate * effective_il6
        clearance = self.crp_clearance_rate * self.crp_mg_l

        self.crp_mg_l += dt_h * (production - clearance)
        self.crp_mg_l = max(0.1, min(self.max_crp, self.crp_mg_l))


# ============================================================================
# Factory
# ============================================================================


def create_immune_model(
    infection_severity: float = 0.0,
    autoimmune_activation: float = 0.0,
    cortisol_level: float = 12.0,
    immunosuppression: float = 0.0,
) -> tuple[InnateImmuneModel, CRPDriver]:
    """Factory creating immune model + CRP driver.

    Args:
        infection_severity: 0-1 scale of acute infection
        autoimmune_activation: 0-1 scale of chronic autoimmunity
        cortisol_level: serum cortisol (µg/dL) for HPA suppression
        immunosuppression: 0-1 scale of drug-induced immunosuppression
    """
    immune = InnateImmuneModel()
    immune.infection_severity = infection_severity
    immune.autoimmune_activation = autoimmune_activation
    # Cortisol > 20 µg/dL suppresses immune response
    if cortisol_level > 20.0:
        immune.cortisol_suppression = min(1.0, (cortisol_level - 20.0) / 30.0)
    immune.immunosuppression = immunosuppression

    crp_driver = CRPDriver()

    return immune, crp_driver
