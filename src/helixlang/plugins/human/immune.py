"""Rule-based innate immune agent-based model (doc/31 §2.4, §5.1 #3).

Population-level ODE approximation of the innate immune system, driving
CRP and WBC counts mechanistically instead of proxy formulas.  The model
tracks three compartments (tissue, blood, lymphoid organ) with agents
(macrophages, neutrophils, dendritic cells, T-cells) and soluble signals
(TNF-α, IL-1β, IL-6, IL-10, Type I IFN).

Key output channels:
    - IL-6 → hepatic CRP production (replaces proxy in ClinicalLabModel)
    - Neutrophil count → WBC differential (replaces manual scaling)
    - TNF-α → systemic inflammation score (feeds vitals temperature)
    - IFN-α/β → antiviral state (doc/40 G1)

Module structure:
    CytokinePool          soluble mediator concentrations
    IFNPool               Type I interferon dynamics (doc/40 G1)
    ImmuneCellPopulation  ODE-tracked cell counts per type
    InnateImmuneModel     coupled cytokine + cell dynamics
    CRPDriver             IL-6 → CRP hepatic production (v2: saturating + lag, doc/40 G8/G9)
    create_immune_model   factory

References:
- BIS-class models: Candiani et al. 2024 (432-parameter innate immune ABM)
- IIRABM: Marina et al. 2024 (rule-based innate response)
- IL-6 → CRP: Volanakis NEJM 2001; Pepys & Hirschfield J Clin Invest 2003
- Neutrophil kinetics: Lord et al. Blood 1989;PRICE et al. J Clin Invest 1976
- Friberg granulopoiesis: Friberg et al. JCO 2002 (transit-chain neutropenia)
- Type I IFN: Pawelek et al. PLoS Comput Biol 2012 (influenza immune model)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is a project dependency
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# doc/40 Phase B: adaptive immunity + vaccination (G2/G3/G7/G12). Imported
# lazily-safe; adaptive.py itself imports nothing from immune, so no cycle.
from helixlang.plugins.human.adaptive import (  # noqa: E402
    AdaptiveImmuneModel as _AdaptiveImmuneModel,
    cohort_adaptive_step as _cohort_adaptive_step,
)

# doc/40 Phase C: reduced complement cascade (G5) and NK/mast pools (G6).
from helixlang.plugins.human.complement import (  # noqa: E402
    ComplementCascade as _ComplementCascade,
    cohort_complement_step as _cohort_complement_step,
)

# doc/40 Phase C: tissue vs blood pseudo-compartments (G10).
from helixlang.plugins.human.tissue_blood import (  # noqa: E402
    TissueBloodModel as _TissueBloodModel,
    cohort_tissue_blood_step as _cohort_tissue_blood_step,
)

__all__ = [
    "CytokinePool",
    "IFNPool",
    "ImmuneCellPopulation",
    "InnateImmuneModel",
    "CRPDriver",
    "create_immune_model",
    "cohort_immune_step",
    "run_cohort",
    "sample_virtual_population",
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

    # --- doc/40 Phase A G1+G11 (opt-in): saturating Hill production and a
    # genuine IL-10 (A) negative-feedback compartment, after the reduced
    # P/A/damage model (Reynolds et al. 2006, L2/L3 adoption).  Inert at
    # baseline; when enabled the O2 cohort kernel falls back to the scalar
    # path so parity is preserved by construction.
    hill_il10_feedback: bool = False
    hill_n: float = 2.0                 # Hill exponent on the PAMP/DAMP drive
    hill_kd: float = 0.5                # signal at half-maximal drive
    il10_feedback_half: float = 5.0     # pg/mL IL-10 at half-max suppression
    il10_feedback_n: float = 2.0        # Hill exponent of IL-10 suppression
    il10_p_half: float = 8.0            # pro-infl. load at half-max IL-10 induction
    il10_hill: float = 2.0              # Hill exponent of IL-10 induction

    # --- Cached decay constants (doc/39 O1, math-equivalent) ---
    # ``k = ln2/half_life`` is loop-invariant across steps; hoisted out of
    # ``step`` and recomputed only when a half-life field is mutated.
    _k_key: tuple[float, float, float, float] = field(init=False, repr=False)
    _decay: tuple[float, float, float, float] = field(init=False, repr=False)

    def _compute_decay(self) -> tuple[float, float, float, float]:
        return (
            math.log(2) / (self.tnf_half_life_h + 1e-12),
            math.log(2) / (self.il1_half_life_h + 1e-12),
            math.log(2) / (self.il6_half_life_h + 1e-12),
            math.log(2) / (self.il10_half_life_h + 1e-12),
        )

    def __post_init__(self) -> None:
        self._k_key = (self.tnf_half_life_h, self.il1_half_life_h,
                       self.il6_half_life_h, self.il10_half_life_h)
        self._decay = self._compute_decay()

    def _ensure_decay_cached(self) -> None:
        key = (self.tnf_half_life_h, self.il1_half_life_h,
               self.il6_half_life_h, self.il10_half_life_h)
        if key != self._k_key:
            self._k_key = key
            self._decay = self._compute_decay()

    def step(self, dt_h: float) -> None:
        """Advance cytokine dynamics one hour.

        Baseline (``hill_il10_feedback=False``) reproduces the reduced
        P/A/damage linear-production biomass exactly — behavior identical to
        the vectorized ``_cohort_cytokine_step`` kernel.  When enabled
        (doc/40 Phase A G1+G11), production of the pro-inflammatory cytokines
        is driven by a saturating Hill activation of the pathogen signal, and
        IL-10 becomes a genuine "L2/A"-compartment negative-feedback signal:
        induced by combined pro-inflammatory load, and feeding back to
        suppress TNF/IL-1/IL-6 production on its own Hill response.  This is
        the exclusive reality switch the cohort kernel deliberately does not
        mirror (it falls back to scalar stepping when enabled).
        """
        self._ensure_decay_cached()
        k_tnf, k_il1, k_il6, k_il10 = self._decay

        stim = self.pathogen_signal
        if not self.hill_il10_feedback:
            # --- baseline linear reduced model (unchanged) ---
            self.tnf_alpha += dt_h * (
                self.tnf_production_rate * stim - k_tnf * self.tnf_alpha)
            self.il1_beta += dt_h * (
                self.il1_production_rate * stim - k_il1 * self.il1_beta)
            self.il6 += dt_h * (
                self.il6_production_rate * stim - k_il6 * self.il6)
            anti_stim = stim + 0.1 * (self.tnf_alpha + self.il1_beta) / 10.0
            self.il10 += dt_h * (
                self.il10_production_rate * anti_stim - k_il10 * self.il10)
        else:
            # --- doc/40 Phase A: saturating Hill physiology ---
            n, kd = self.hill_n, self.hill_kd
            drive = stim ** n / (kd ** n + stim ** n + 1e-12)
            # IL-10 (A) feedback: more IL-10 → less pro-inflammatory drive.
            fb = (self.il10 ** self.il10_feedback_n) / (
                self.il10_feedback_half ** self.il10_feedback_n
                + self.il10 ** self.il10_feedback_n + 1e-12)
            self.tnf_alpha += dt_h * (
                self.tnf_production_rate * drive * (1.0 - fb)
                - k_tnf * self.tnf_alpha)
            self.il1_beta += dt_h * (
                self.il1_production_rate * drive * (1.0 - fb)
                - k_il1 * self.il1_beta)
            self.il6 += dt_h * (
                self.il6_production_rate * drive * (1.0 - fb)
                - k_il6 * self.il6)
            # IL-10 induction by pro-inflammatory load (Hill, G11).
            load = self.tnf_alpha + self.il1_beta + self.il6
            i10_stim = (load ** self.il10_hill) / (
                self.il10_p_half ** self.il10_hill
                + load ** self.il10_hill + 1e-12)
            self.il10 += dt_h * (
                self.il10_production_rate * i10_stim - k_il10 * self.il10)

        # Floor
        self.tnf_alpha = max(0.0, self.tnf_alpha)
        self.il1_beta = max(0.0, self.il1_beta)
        self.il6 = max(0.0, self.il6)
        self.il10 = max(0.0, self.il10)


# ============================================================================
# Type I Interferon Pool (doc/40 G1)
# ============================================================================


@dataclass
class IFNPool:
    """Type I interferon (IFN-α/β) antiviral loop (doc/40 G1).

    Driven by pathogen signal; suppresses pathogen replication via a
    saturating Hill-type antiviral feedback.  Default production near
    zero at baseline so existing InnateImmuneModel.step is unaffected.
    """

    ifn_alpha_beta: float = 0.0     # pg/mL (baseline ~0, rises with infection)

    # Production: Vmax * sig^h / (Kd^h + sig^h)  (Hill activation by pathogen)
    ifn_vmax: float = 5.0           # pg/mL/h max production rate
    ifn_kd: float = 0.3             # pg/mL half-maximal signal
    ifn_hill_n: float = 2.0         # Hill coefficient

    # Clearance: exponential decay
    ifn_half_life_h: float = 1.5    # ~90 min (type I IFN in serum)

    # Antiviral effect: pathogen_eff = pathogen * (1 - ε * IFN/(IFN + K_ifn))
    antiviral_efficiency: float = 0.6   # 0–1, fraction of pathogen suppressed
    antiviral_k_ifn: float = 5.0        # pg/mL IFN at half-maximal suppression

    # Cached decay constant
    _k_ifn: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._k_ifn = math.log(2) / (self.ifn_half_life_h + 1e-12)

    def effective_pathogen(self, pathogen: float) -> float:
        """Return pathogen signal after IFN-α/β antiviral suppression."""
        if self.ifn_alpha_beta <= 0.0:
            return pathogen
        suppression = self.antiviral_efficiency * self.ifn_alpha_beta / (
            self.antiviral_k_ifn + self.ifn_alpha_beta)
        return pathogen * (1.0 - suppression)

    def step(self, dt_h: float, pathogen_signal: float) -> None:
        """Advance IFN dynamics one hour."""
        self._k_ifn = math.log(2) / (self.ifn_half_life_h + 1e-12)
        sig = pathogen_signal
        production = self.ifn_vmax * sig**self.ifn_hill_n / (
            self.ifn_kd**self.ifn_hill_n + sig**self.ifn_hill_n)
        self.ifn_alpha_beta += dt_h * (production - self._k_ifn * self.ifn_alpha_beta)
        self.ifn_alpha_beta = max(0.0, self.ifn_alpha_beta)


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

    # --- G6 pools (doc/40 Phase C): NK, mast, eosinophil, basophil ---
    nk_cells: float = 0.3         # ×10³/µL (normal 0.15-0.6)
    mast_cells: float = 0.1       # ×10³/µL tissue-equivalent
    eosinophils: float = 0.1      # ×10³/µL (normal 0.02-0.5)
    basophils: float = 0.05       # ×10³/µL (normal 0.01-0.08)

    # Production rates (×10³/µL/h)
    neutrophil_production: float = 0.1   # bone marrow output
    monocyte_production: float = 0.02
    t_cell_production: float = 0.01

    # Clearance rates (1/h)
    neutrophil_clearance: float = 0.05   # ~14 h half-life in blood
    monocyte_clearance: float = 0.03
    macrophage_clearance: float = 0.01   # long-lived tissue cells

    # --- G6 kinetics ---
    nk_production: float = 0.005
    nk_clearance: float = 0.02
    mast_production: float = 0.002
    mast_clearance: float = 0.01
    eos_production: float = 0.002
    eos_clearance: float = 0.03
    baso_production: float = 0.001
    baso_clearance: float = 0.04
    # Mast-cell anaphylaxis (IgE cross-linking → histamine release, G6):
    # drives a transient systemic mediator spike used for the anaphylaxis flag.
    histamine_ng_ml: float = field(default=0.0, init=False, repr=False)
    igE_signal: float = 0.0       # 0-1 external anaphylaxis/IgE drive

    # Mobilisation: cytokine-driven release from bone marrow
    gcsf_sensitivity: float = 0.5  # neutrophil response to G-CSF/IL-6

    # --- Friberg transit chain (doc/40 G4) ---
    # 4-compartment maturation chain: precursor → T1 → T2 → T3 → circulating.
    # Transit time per compartment (hours).  Total maturation ~4 days.
    friberg_transit_time_h: float = 24.0
    # Feedback: proliferation rate = prolif_max / (1 + (ANC/K)^n) normalized so
    # at healthy ANC (= K) the proliferating input equals the baseline
    # neutrophil production, then drops as ANC rises (negative feedback) and
    # rises as ANC falls (nadir recovery).
    friberg_k_prolif: float = 4.0       # ×10³/µL ANC at half-maximal proliferation
    friberg_hill_n: float = 2.0         # Hill exponent for ANC feedback
    # Drug kill fraction (0–1) on proliferating cells (chemotherapy, etc.)
    friberg_drug_kill: float = 0.0

    # Transit state (init to steady-state: all compartments equal)
    _transit_t1: float = field(default=0.0, init=False, repr=False)
    _transit_t2: float = field(default=0.0, init=False, repr=False)
    _transit_t3: float = field(default=0.0, init=False, repr=False)
    _transit_t4: float = field(default=0.0, init=False, repr=False)

    def _friberg_baseline_prod(self) -> float:
        """Baseline proliferating-pool input (steady-state neutrophil production)."""
        return self.neutrophil_production

    def _init_transit_steady_state(self) -> None:
        """Set transit compartments to steady-state matching baseline production."""
        tau = self.friberg_transit_time_h + 1e-12
        ss = self.neutrophil_production * tau
        self._transit_t1 = ss
        self._transit_t2 = ss
        self._transit_t3 = ss
        self._transit_t4 = ss

    def step(self, dt_h: float, il6: float, tnf: float) -> None:
        """Advance cell dynamics one hour.

        Args:
            il6: IL-6 level (pg/mL) — drives neutrophil mobilisation
            tnf: TNF-α level (pg/mL) — drives monocyte differentiation
        """
        # --- Friberg transit chain (doc/40 G4) ---
        # Proliferation rate: normalized so at healthy ANC (= K_prolif) the
        # proliferating input equals baseline neutrophil production, with
        # negative feedback: high ANC suppresses, low ANC boosts (recovery).
        baseline_prod = self._friberg_baseline_prod()
        ratio_n = (self.neutrophils / (self.friberg_k_prolif + 1e-12)) \
            ** self.friberg_hill_n
        prolif_rate = baseline_prod * 2.0 / (1.0 + ratio_n)
        # Drug kills proliferating precursor cells
        prolif_rate *= (1.0 - min(1.0, max(0.0, self.friberg_drug_kill)))

        tau = self.friberg_transit_time_h + 1e-12

        # Initialize transit to steady-state on first step
        if self._transit_t1 <= 0.0 and self._transit_t2 <= 0.0:
            self._init_transit_steady_state()

        # Advance transit compartments (explicit Euler)
        self._transit_t1 += dt_h * (prolif_rate - self._transit_t1 / tau)
        self._transit_t2 += dt_h * (self._transit_t1 / tau - self._transit_t2 / tau)
        self._transit_t3 += dt_h * (self._transit_t2 / tau - self._transit_t3 / tau)
        out_t4 = self._transit_t3 / tau
        self._transit_t4 += dt_h * (self._transit_t3 / tau - self._transit_t4 / tau)

        # Transit chain output → neutrophil production (replaces linear)
        effective_neut_prod = out_t4

        # Neutrophil dynamics: production + mobilisation - clearance
        mobilisation = self.gcsf_sensitivity * max(0.0, il6 - 1.0) / 10.0
        self.neutrophils += dt_h * (
            effective_neut_prod * (1.0 + mobilisation)
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

        # --- G6 pools (doc/40 Phase C) ---
        # NK: IFN-γ-mediated antiviral surveillance (rises with innate signal).
        self.nk_cells += dt_h * (
            self.nk_production * (1.0 + tnf / 50.0)
            - self.nk_clearance * self.nk_cells)
        # Mast cells: tissue resident; IgE cross-linking releases histamine.
        self.mast_cells += dt_h * (
            self.mast_production - self.mast_clearance * self.mast_cells)
        # Eosinophils / basophils: parasitic/allergic arms.
        self.eosinophils += dt_h * (
            self.eos_production * (1.0 + self.igE_signal)
            - self.eos_clearance * self.eosinophils)
        self.basophils += dt_h * (
            self.baso_production * (1.0 + self.igE_signal)
            - self.baso_clearance * self.basophils)
        # Histamine (ng/mL): rapid release on IgE signal, ~10 min half-life
        # (acute anaphylaxis mediator that clears minutes after the trigger).
        self.histamine_ng_ml = self.histamine_ng_ml * math.exp(-6.0 * dt_h) \
            + self.igE_signal * 40.0 * min(1.0, dt_h)

        # Floors
        self.neutrophils = max(0.1, self.neutrophils)
        self.macrophages = max(0.01, self.macrophages)
        self.monocytes = max(0.01, self.monocytes)
        self.dendritic_cells = max(0.01, self.dendritic_cells)
        self.t_cells = max(0.1, self.t_cells)
        self.nk_cells = max(0.01, self.nk_cells)
        self.mast_cells = max(0.01, self.mast_cells)
        self.eosinophils = max(0.01, self.eosinophils)
        self.basophils = max(0.01, self.basophils)

    def get_wbc_total(self) -> float:
        """Total WBC (×10³/µL) — approximate from populations."""
        return (self.neutrophils + self.monocytes
                + self.dendritic_cells + self.t_cells)


# ============================================================================
# Innate Immune Model (Coupled)
# ============================================================================


@dataclass
class InnateImmuneModel:
    """Coupled cytokine + cell population model (doc/40 Phase A).

    Represents the innate immune response to infection, injury, or
    sterile inflammation.  Drives CRP via IL-6 and WBC counts via
    cell dynamics.

    Phase A additions:
        - IFN-α/β antiviral loop (G1)
        - Circadian cortisol modulation (G11)
        - Friberg transit chain in ImmuneCellPopulation (G4)
    """

    cytokines: CytokinePool = field(default_factory=CytokinePool)
    cells: ImmuneCellPopulation = field(default_factory=ImmuneCellPopulation)
    ifn: IFNPool = field(default_factory=IFNPool)

    # --- Adaptive immunity (doc/40 Phase B: G2/G3/G7/G12) ---
    # Additive and inert at baseline: no infection + no vaccine leaves every
    # adaptive pool at its naive baseline and antibody at the baseline titer.
    adaptive: _AdaptiveImmuneModel = field(
        default_factory=_AdaptiveImmuneModel)

    # --- Complement cascade (doc/40 Phase C: G5) ---
    # Additive and inert at baseline: no signal -> C3/C5 at 1.0, anaphylatoxins
    # and MAC ~0.  ``anti_c5_dose`` (0-1) simulates an anti-C5 agent that
    # suppresses the MAC arm while sparing C3b opsonization.
    complement: _ComplementCascade = field(
        default_factory=_ComplementCascade)

    # --- Tissue vs blood pseudo-compartments (doc/40 Phase C: G10) ---
    # Additive and inert at baseline: blood fields mirror the circulating
    # channels (so existing consumers are unchanged), tissue fields sit at
    # baseline, and the tissue-vs-blood divergence is ~0 with no signal.
    tissue_blood: _TissueBloodModel = field(
        default_factory=_TissueBloodModel)


    # --- Cortisol suppression (from HPA axis) ---
    cortisol_suppression: float = 0.0  # 0=none, 1=complete suppression

    # --- Circadian cortisol modulation (doc/40 G11) ---
    circadian_amplitude: float = 0.0   # fractional amplitude (0=no variation)
    circadian_phase_h: float = 8.0     # hour of peak cortisol (default ~08:00)
    _sim_hour: float = field(default=0.0, init=False, repr=False)

    # --- Drug effects ---
    immunosuppression: float = 0.0  # e.g. chemotherapy, biologics
    anti_inflammatory: float = 0.0  # 0-1: PD-driven suppression of IL-6/TNF (JAK inhibitors etc.)

    # --- Biologic anti-IL-6 occupancy (doc/40 Phase D, L10) ---
    # Target-mediated drug disposition-style occupancy (0-1) of a
    # tocilizumab/siltuximab-class anti-IL-6: directly suppresses IL-6
    # production more strongly than the generic anti_inflammatory pathway,
    # with partial tissue/lymph-node penetration per L10 (~10% less at the
    # tissue).  Inert at 0.
    il6_biologic_occupancy: float = 0.0

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

        # --- Circadian cortisol modulation (doc/40 G11) ---
        effective_cortisol = self.cortisol_suppression
        if self.circadian_amplitude > 0.0:
            import math as _m
            cortisol_var = self.circadian_amplitude * _m.sin(
                2.0 * _m.pi * (self._sim_hour - self.circadian_phase_h) / 24.0)
            effective_cortisol = max(0.0, min(1.0,
                self.cortisol_suppression * (1.0 + cortisol_var)))

        # --- IFN-α/β antiviral loop (doc/40 G11 → G1) ---
        self.ifn.step(dt_h, self.cytokines.pathogen_signal)
        effective_pathogen = self.ifn.effective_pathogen(
            self.cytokines.pathogen_signal)
        self.cytokines.pathogen_signal = effective_pathogen

        # Restore base rates before applying modifiers (prevents compounding)
        self.cytokines.tnf_production_rate = self._base_tnf_rate
        self.cytokines.il6_production_rate = self._base_il6_rate
        self.cells.monocyte_production = self._base_monocyte_prod

        # Cortisol suppresses cytokine production (uses circadian-modulated value)
        if effective_cortisol > 0:
            self.cytokines.tnf_production_rate *= (
                1.0 - effective_cortisol * 0.7)
            self.cytokines.il6_production_rate *= (
                1.0 - effective_cortisol * 0.6)

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

        # Biologic anti-IL-6 occupancy (doc/40 Phase D, L10): direct,
        # TMDD-style IL-6 neutralisation.  Inert at occupancy 0.
        occ = min(1.0, max(0.0, self.il6_biologic_occupancy))
        if occ > 0.0:
            # Near-complete neutralisation of circulating IL-6 at high
            # occupancy (tocilizumab/siltuximab class); tissue penetration is
            # partial (~10% lower per L10) but the dominant blood effect is
            # captured here and propagates to the tissue readout next tick.
            self.cytokines.il6_production_rate *= (1.0 - 0.97 * occ)

        # Step cytokines
        self.cytokines.step(dt_h)

        # Step cells with cytokine signals
        self.cells.step(dt_h, self.cytokines.il6, self.cytokines.tnf_alpha)

        # --- Adaptive immunity (doc/40 Phase B) ---
        # The adaptive layer sees the same antigen drive as the innate layer
        # (effective post-IFN pathogen signal), so a viral load that the
        # innate arm is not clearing grows a CD8/antibody response; a cleared
        # infection leaves memory. Inert when infection_severity == 0 and no
        # vaccine has been administered.
        self.adaptive.step(dt_h, max(0.0, effective_pathogen))

        # --- Complement cascade (doc/40 Phase C: G5) ---
        # Driven by the same post-IFN tissue/immune signal.  Inert at
        # baseline; an anti-C5 dose modulates the MAC arm additively.
        self.complement.step(dt_h, max(0.0, effective_pathogen))

        # --- Tissue vs blood split (doc/40 Phase C: G10) ---
        # Mirror the circulating channels into the blood compartment, then
        # apply migration/differencing under the same tissue signal.  Inert
        # at baseline (no signal -> divergence ~0).
        tb = self.tissue_blood
        tb.blood_il6 = self.cytokines.il6
        tb.blood_neutrophils = self.cells.neutrophils
        tb.blood_monocytes = self.cells.monocytes
        tb.step(dt_h, max(0.0, effective_pathogen))

        # Advance simulation clock for circadian modulation
        self._sim_hour = (self._sim_hour + dt_h) % 24.0

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

    # --- Adaptive immunity accessors (doc/40 Phase B) ---
    def get_igg(self) -> float:
        return self.adaptive.get_igg()

    def get_igm(self) -> float:
        return self.adaptive.get_igm()

    def get_total_antibody(self) -> float:
        return self.adaptive.get_total_antibody()

    def vaccinate(self, dose: float) -> None:
        """Administer a vaccine dose through the adaptive layer (G12)."""
        self.adaptive.vaccinate(dose)

    # --- PD-1 checkpoint accessors (doc/40 Phase D, G14) ---
    def get_effector_t(self) -> float:
        return self.adaptive.get_effector_t()

    def get_memory_t(self) -> float:
        return self.adaptive.get_memory_t()

    def set_checkpoint_blockade(self, blockade: float) -> None:
        """Set PD-1/PD-L1 checkpoint blockade (0-1) on the adaptive layer."""
        self.adaptive.checkpoint_blockade = min(
            1.0, max(0.0, float(blockade)))

    def get_checkpoint_blockade(self) -> float:
        return self.adaptive.checkpoint_blockade

    def set_il6_biologic_occupancy(self, occ: float) -> None:
        """Set biologic anti-IL-6 occupancy (0-1), L10."""
        self.il6_biologic_occupancy = min(1.0, max(0.0, float(occ)))

    # --- Complement accessors (doc/40 Phase C: G5) ---
    def get_c3a(self) -> float:
        return self.complement.get_c3a()

    def get_c5a(self) -> float:
        return self.complement.get_c5a()

    def get_mac(self) -> float:
        return self.complement.get_mac()

    def get_opsonization(self) -> float:
        return self.complement.get_opsonization()

    # --- G6 accessors ---
    def get_nk_cells(self) -> float:
        return self.cells.nk_cells

    def get_mast_cells(self) -> float:
        return self.cells.mast_cells

    def get_histamine(self) -> float:
        return self.cells.histamine_ng_ml

    def get_eosinophils(self) -> float:
        return self.cells.eosinophils

    def get_basophils(self) -> float:
        return self.cells.basophils

    # --- Tissue/blood accessors (doc/40 Phase C: G10) ---
    def get_blood_neutrophils(self) -> float:
        return self.tissue_blood.get_blood_neutrophils()

    def get_tissue_neutrophils(self) -> float:
        return self.tissue_blood.get_tissue_neutrophils()

    def get_tissue_il6(self) -> float:
        return self.tissue_blood.get_tissue_il6()

    def get_tissue_macrophages(self) -> float:
        return self.tissue_blood.get_tissue_macrophages()

    def get_tissue_blood_divergence(self) -> float:
        return self.tissue_blood.get_tissue_blood_divergence()


# ============================================================================
# CRP Driver (IL-6 → Hepatic CRP Production)
# ============================================================================


@dataclass
class CRPDriver:
    """Mechanistic IL-6 → CRP pathway (v2, doc/40 G8/G9).

    Hepatocyte CRP synthesis is driven by IL-6 via JAK/STAT3 signaling.
    Uses saturating Hill kinetics (not linear) so CRP reaches sepsis
    levels (up to 1000 µg/mL) while staying physiological at baseline.

    CRP half-life ~19 h; peak CRP ~6 h after IL-6 stimulus.
    Includes a lag compartment modelling the ~6 h transcriptional delay.
    """

    crp_mg_l: float = 0.5     # baseline CRP (mg/L)
    crp_clearance_rate: float = 0.036  # 1/h (~19 h half-life)
    max_crp: float = 1000.0    # clinical ceiling for severe sepsis (mg/L)

    # Saturating Hill production: Vmax * IL6_lag^n / (Kd^n + IL6_lag^n)
    crp_vmax: float = 40.0     # mg/L/h at saturating IL-6
    il6_kd: float = 5.0        # pg/mL IL-6 at half-maximal CRP production
    il6_hill_n: float = 2.0    # Hill exponent for IL-6→CRP

    # Lag compartment: models ~6 h transcriptional/post-translational delay
    il6_lag_tau_h: float = 6.0  # time constant of IL-6 lag (hours)
    _il6_lagged: float = field(default=0.0, init=False, repr=False)

    # --- APR panel (doc/40 G9) ---
    saa_mg_l: float = 0.0           # serum amyloid A (mg/L)
    ferritin_ng_ml: float = 50.0    # ferritin (ng/mL, normal 12-300)
    pct_ng_ml: float = 0.0          # procalcitonin (ng/mL, <0.1 normal)
    fibrinogen_g_l: float = 3.0     # fibrinogen (g/L, normal 2-4)

    def step(self, dt_h: float, il6_pg_ml: float) -> None:
        """Advance CRP dynamics one hour."""
        # Lag compartment: first-order toward current IL-6
        if self.il6_lag_tau_h > 0.0:
            alpha = 1.0 - math.exp(-dt_h / self.il6_lag_tau_h)
        else:
            alpha = 1.0
        self._il6_lagged += alpha * (il6_pg_ml - self._il6_lagged)

        # Saturating Hill production (doc/40 G9)
        il6_n = self._il6_lagged**self.il6_hill_n
        kd_n = self.il6_kd**self.il6_hill_n
        production = self.crp_vmax * il6_n / (kd_n + il6_n + 1e-12)
        clearance = self.crp_clearance_rate * self.crp_mg_l

        self.crp_mg_l += dt_h * (production - clearance)
        self.crp_mg_l = max(0.1, min(self.max_crp, self.crp_mg_l))

        # APR panel: SAA tracks CRP with amplification, others driven by IL-6
        self.saa_mg_l = self.crp_mg_l * 10.0  # SAA ~10× CRP in acute phase
        il6_norm = min(1.0, self._il6_lagged / 50.0)
        self.ferritin_ng_ml = 50.0 + 950.0 * il6_norm
        self.pct_ng_ml = il6_norm * 10.0  # 0–10 ng/mL
        self.fibrinogen_g_l = 3.0 + 2.0 * il6_norm  # 3–5 g/L


# ============================================================================
# Factory
# ============================================================================


def create_immune_model(
    infection_severity: float = 0.0,
    autoimmune_activation: float = 0.0,
    cortisol_level: float = 12.0,
    immunosuppression: float = 0.0,
    hill_il10_feedback: bool = False,
) -> tuple[InnateImmuneModel, CRPDriver]:
    """Factory creating immune model + CRP driver.

    Args:
        infection_severity: 0-1 scale of acute infection
        autoimmune_activation: 0-1 scale of chronic autoimmunity
        cortisol_level: serum cortisol (µg/dL) for HPA suppression
        immunosuppression: 0-1 scale of drug-induced immunosuppression
        hill_il10_feedback: opt in to saturating Hill cytokine production and
            the IL-10 negative-feedback compartment (doc/40 Phase A G1+G11).
            Off by default because it changes CytokinePool kinetics; a cohort
            kernel run falls back to scalar stepping while it is enabled.
    """
    immune = InnateImmuneModel()
    immune.infection_severity = infection_severity
    immune.autoimmune_activation = autoimmune_activation
    # Cortisol > 20 µg/dL suppresses immune response
    if cortisol_level > 20.0:
        immune.cortisol_suppression = min(1.0, (cortisol_level - 20.0) / 30.0)
    immune.immunosuppression = immunosuppression
    if hill_il10_feedback:
        immune.cytokines.hill_il10_feedback = True

    crp_driver = CRPDriver()

    return immune, crp_driver


# ============================================================================
# Cohort-vectorized immune stepping (doc/39 O2)
#
# Advances N independent virtual patients' innate-immune ODEs *simultaneously*
# with NumPy elementwise operations, mirroring the scalar
# ``InnateImmuneModel.step`` / ``CytokinePool.step`` / ``ImmuneCellPopulation.
# step`` equations exactly (same float arithmetic, vectorized over the cohort
# axis).  This is the 🟪 realism-better path: it opens virtual-population runs
# (per-patient parameter sampling) at far lower cost than N per-patient loops.
# When numpy is absent it falls back to a scalar loop over the model objects,
# so behavior is identical with or without the ``fast`` extra.
# ============================================================================


def _cohort_cytokine_step(
    tnf: Any, il1: Any, il6: Any, il10: Any,
    prod_tnf: Any, prod_il1: Any, prod_il6: Any, prod_il10: Any,
    k_tnf: Any, k_il1: Any, k_il6: Any, k_il10: Any,
    stim: Any, dt_h: float,
) -> tuple[Any, Any, Any, Any]:
    """One vectorized CytokinePool step (mirrors ``CytokinePool.step``)."""
    tnf = np_max(0.0, tnf + dt_h * (prod_tnf * stim - k_tnf * tnf))
    il1 = np_max(0.0, il1 + dt_h * (prod_il1 * stim - k_il1 * il1))
    il6 = np_max(0.0, il6 + dt_h * (prod_il6 * stim - k_il6 * il6))
    anti_stim = stim + 0.1 * (tnf + il1) / 10.0
    il10 = np_max(0.0, il10 + dt_h * (prod_il10 * anti_stim - k_il10 * il10))
    return tnf, il1, il6, il10


def _cohort_cells_step(
    neut: Any, macro: Any, mono: Any, dc: Any, tcells: Any,
    prod_neut: Any, prod_mono: Any, prod_tcell: Any,
    clr_neut: Any, clr_mono: Any, clr_macro: Any,
    gcsf: Any, il6: Any, tnf: Any, dt_h: float,
    t1: Any = 0.0, t2: Any = 0.0, t3: Any = 0.0, t4: Any = 0.0,
    tau: Any = 1.0, prolif: Any = 0.0, mob_scale: Any = 1.0,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """One vectorized ImmuneCellPopulation step (mirrors the scalar step).

    Returns (neut, macro, mono, dc, tcells, t1, t2, t3, t4).
    When tau is 0 or scalar 1.0 with no transit arrays, Friberg is skipped.
    """
    use_friberg = not (isinstance(tau, (int, float)) and tau <= 0.0)
    if use_friberg:
        # Friberg transit chain (doc/40 G4) — scalar path uses in-place
        # Gauss-Seidel: t1 updated first, t2 uses new t1, t3 uses new t2.
        inv_tau = 1.0 / (tau + 1e-12)
        t1_new = t1 + dt_h * (prolif - t1 * inv_tau)
        t2_new = t2 + dt_h * (t1_new * inv_tau - t2 * inv_tau)
        t3_new = t3 + dt_h * (t2_new * inv_tau - t3 * inv_tau)
        out_t4 = t3_new * inv_tau
        t4_new = t4 + dt_h * (t3_new * inv_tau - t4 * inv_tau)
        effective_neut_prod = out_t4
        t1, t2, t3, t4 = t1_new, t2_new, t3_new, t4_new
    else:
        effective_neut_prod = prod_neut

    mobil = gcsf * np_max(0.0, il6 - 1.0) / 10.0
    neut = np_max(0.1, neut + dt_h * (effective_neut_prod * (1.0 + mobil)
                                      - clr_neut * neut))
    mono = np_max(0.01, mono + dt_h * (prod_mono * (1.0 + tnf / 20.0)
                                       - clr_mono * mono))
    diff = 0.1 * mono
    macro = np_max(0.01, macro + dt_h * (diff - clr_macro * macro))
    dc = np_max(0.01, dc + dt_h * (0.005 * (1.0 + tnf / 50.0) - 0.02 * dc))
    tcells = np_max(0.1, tcells + dt_h * (prod_tcell * (1.0 + il6 / 20.0)
                                          - 0.005 * tcells))
    return neut, macro, mono, dc, tcells, t1, t2, t3, t4


def np_max(a: float, b: Any) -> Any:
    """Elementwise ``max(a, b)`` over a scalar-or-array ``b`` (doc/39 O2)."""
    if _HAS_NUMPY and isinstance(b, _np.ndarray):
        return _np.maximum(a, b)
    return max(a, b)


def cohort_immune_step(
    models: list[InnateImmuneModel],
    dt_h: float,
    *,
    use_numpy: bool | None = None,
) -> None:
    """Advance a cohort of ``InnateImmuneModel`` by ``dt_h`` (doc/39 O2).

    Equivalence guarantee: with ``use_numpy=True`` the cohort advances through
    an explicit NumPy system whose equations are term-for-term identical to
    :meth:`InnateImmuneModel.step`; with ``use_numpy=False`` (or when numpy is
    unavailable) it falls back to calling each model's scalar ``step``.  Either
    way every model ends in the same state as the scalar path.
    """
    if use_numpy is None:
        use_numpy = _HAS_NUMPY

    if not use_numpy or not _HAS_NUMPY:
        for model in models:
            model.step(dt_h)
        return

    # The vectorized kernel mirrors CytokinePool.step's default linear
    # production exactly.  The opt-in saturating-Hill + IL-10 A-compartment
    # physiology (doc/40 Phase A G1) is deliberately NOT mirrored there; a
    # cohort that enables it falls through to scalar stepping so the
    # equivalence guarantee in the docstring holds by construction.
    if any(m.cytokines.hill_il10_feedback for m in models):
        for model in models:
            model.step(dt_h)
        return

    np = _np
    n = len(models)
    if n == 0:
        return

    import math

    tnf = np.array([m.cytokines.tnf_alpha for m in models], dtype=float)
    il1 = np.array([m.cytokines.il1_beta for m in models], dtype=float)
    il6 = np.array([m.cytokines.il6 for m in models], dtype=float)
    il10 = np.array([m.cytokines.il10 for m in models], dtype=float)
    neut = np.array([m.cells.neutrophils for m in models], dtype=float)
    macro = np.array([m.cells.macrophages for m in models], dtype=float)
    mono = np.array([m.cells.monocytes for m in models], dtype=float)
    dc = np.array([m.cells.dendritic_cells for m in models], dtype=float)
    tcells = np.array([m.cells.t_cells for m in models], dtype=float)

    # Decay constants (O1: ln2/half-life, treated as fixed per cohort model).
    k_tnf = np.array([math.log(2) / (m.cytokines.tnf_half_life_h + 1e-12)
                      for m in models], dtype=float)
    k_il1 = np.array([math.log(2) / (m.cytokines.il1_half_life_h + 1e-12)
                      for m in models], dtype=float)
    k_il6 = np.array([math.log(2) / (m.cytokines.il6_half_life_h + 1e-12)
                      for m in models], dtype=float)
    k_il10 = np.array([math.log(2) / (m.cytokines.il10_half_life_h + 1e-12)
                       for m in models], dtype=float)

    prod_tnf = np.array([m.cytokines.tnf_production_rate for m in models],
                        dtype=float)
    prod_il1 = np.array([m.cytokines.il1_production_rate for m in models],
                        dtype=float)
    prod_il6 = np.array([m.cytokines.il6_production_rate for m in models],
                        dtype=float)
    prod_il10 = np.array([m.cytokines.il10_production_rate for m in models],
                         dtype=float)
    prod_neut = np.array([m.cells.neutrophil_production for m in models],
                         dtype=float)
    prod_mono = np.array([m.cells.monocyte_production for m in models],
                         dtype=float)
    prod_tcell = np.array([m.cells.t_cell_production for m in models],
                          dtype=float)
    clr_neut = np.array([m.cells.neutrophil_clearance for m in models],
                        dtype=float)
    clr_mono = np.array([m.cells.monocyte_clearance for m in models],
                        dtype=float)
    clr_macro = np.array([m.cells.macrophage_clearance for m in models],
                         dtype=float)
    gcsf = np.array([m.cells.gcsf_sensitivity for m in models], dtype=float)

    inf_sev = np.array([m.infection_severity for m in models], dtype=float)
    auto_sev = np.array([m.autoimmune_activation for m in models], dtype=float)
    cort_raw = np.array([m.cortisol_suppression for m in models], dtype=float)
    circ_amp = np.array([m.circadian_amplitude for m in models], dtype=float)
    circ_phase = np.array([m.circadian_phase_h for m in models], dtype=float)
    sim_hour = np.array([m._sim_hour for m in models], dtype=float)
    immuno = np.array([m.immunosuppression for m in models], dtype=float)
    anti = np.array([m.anti_inflammatory for m in models], dtype=float)

    # IFN state (doc/40 G1)
    ifn_val = np.array([m.ifn.ifn_alpha_beta for m in models], dtype=float)
    ifn_vmax = np.array([m.ifn.ifn_vmax for m in models], dtype=float)
    ifn_kd = np.array([m.ifn.ifn_kd for m in models], dtype=float)
    ifn_hn = np.array([m.ifn.ifn_hill_n for m in models], dtype=float)
    ifn_k_ifn = np.array([m.ifn.antiviral_k_ifn for m in models], dtype=float)
    ifn_eps = np.array([m.ifn.antiviral_efficiency for m in models], dtype=float)
    ifn_k_decay = np.array([math.log(2) / (m.ifn.ifn_half_life_h + 1e-12)
                            for m in models], dtype=float)

    stim = inf_sev + auto_sev * 0.5

    # --- IFN step: production from stim, then antiviral suppression ---
    ifn_kd_n = ifn_kd ** ifn_hn
    ifn_sig_n = stim ** ifn_hn
    ifn_prod = ifn_vmax * ifn_sig_n / (ifn_kd_n + ifn_sig_n + 1e-12)
    ifn_val = np_max(0.0, ifn_val + dt_h * (ifn_prod - ifn_k_decay * ifn_val))
    # Antiviral suppression: stim_eff = stim * (1 - eps * IFN/(IFN + K_ifn))
    suppression = ifn_eps * ifn_val / (ifn_k_ifn + ifn_val + 1e-12)
    stim_eff = stim * (1.0 - suppression)

    # --- Circadian cortisol (doc/40 G11) ---
    cort_eff = cort_raw.copy()
    has_circ = circ_amp > 0.0
    if np.any(has_circ):
        cortisol_var = circ_amp * np.sin(
            2.0 * math.pi * (sim_hour - circ_phase) / 24.0)
        cort_eff[has_circ] = np.clip(
            cort_raw[has_circ] * (1.0 + cortisol_var[has_circ]), 0.0, 1.0)

    # Per-model base rates (mirror __post_init__ snapshot), restored each step
    # so modifiers never compound.
    base_tnf = np.array([m._base_tnf_rate for m in models], dtype=float)
    base_il6 = np.array([m._base_il6_rate for m in models], dtype=float)
    base_mono = np.array([m._base_monocyte_prod for m in models], dtype=float)

    # Per-model modifiers, mirroring InnateImmuneModel.step exactly
    anti_c = np.minimum(anti, 1.0)
    prod_tnf = base_tnf * (1.0 - cort_eff * 0.7) * (1.0 - anti_c * 0.85)
    prod_il6 = base_il6 * (1.0 - cort_eff * 0.6) * (1.0 - anti_c * 0.9)
    prod_mono = base_mono * (1.0 - immuno * 0.7)
    prod_neut = prod_neut * (1.0 - immuno * 0.8)

    # Biologic anti-IL-6 occupancy (doc/40 Phase D, L10): direct TMDD-style
    # neutralisation of circulating IL-6 production, mirroring the scalar
    # step (immune.step applies a 0.97 factor at occupancy 1.0).
    occupancy = np.clip(
        np.array([m.il6_biologic_occupancy for m in models], dtype=float),
        0.0, 1.0)
    prod_il6 = prod_il6 * (1.0 - 0.97 * occupancy)

    # --- Friberg transit chain (doc/40 G4) ---
    t1 = np.array([m.cells._transit_t1 for m in models], dtype=float)
    t2 = np.array([m.cells._transit_t2 for m in models], dtype=float)
    t3 = np.array([m.cells._transit_t3 for m in models], dtype=float)
    t4 = np.array([m.cells._transit_t4 for m in models], dtype=float)
    fri_tau = np.array([m.cells.friberg_transit_time_h for m in models],
                       dtype=float)
    fri_k = np.array([m.cells.friberg_k_prolif for m in models], dtype=float)
    fri_hn = np.array([m.cells.friberg_hill_n for m in models], dtype=float)
    fri_kill = np.array([m.cells.friberg_drug_kill for m in models],
                        dtype=float)

    # Proliferation rate: normalized so at healthy ANC (= K) the input equals
    # baseline production, with negative feedback from circulating ANC.
    baseline_prod = prod_neut
    ratio_n = (neut / (fri_k + 1e-12)) ** fri_hn
    prolif_rate = baseline_prod * 2.0 / (1.0 + ratio_n)
    prolif_rate = prolif_rate * (1.0 - np.clip(fri_kill, 0.0, 1.0))

    # Initialize transit to steady-state on first step (check if all zeros)
    all_zero = np.all(t1 == 0.0) and np.all(t2 == 0.0)
    if all_zero:
        ss = prod_neut * (fri_tau + 1e-12)
        t1 = ss.copy()
        t2 = ss.copy()
        t3 = ss.copy()
        t4 = ss.copy()

    tnf, il1, il6, il10 = _cohort_cytokine_step(
        tnf, il1, il6, il10, prod_tnf, prod_il1, prod_il6, prod_il10,
        k_tnf, k_il1, k_il6, k_il10, stim_eff, dt_h)
    neut, macro, mono, dc, tcells, t1, t2, t3, t4 = _cohort_cells_step(
        neut, macro, mono, dc, tcells, prod_neut, prod_mono, prod_tcell,
        clr_neut, clr_mono, clr_macro, gcsf, il6, tnf, dt_h,
        t1, t2, t3, t4, fri_tau, prolif_rate)

    for i, m in enumerate(models):
        m.cytokines.tnf_alpha = float(tnf[i])
        m.cytokines.il1_beta = float(il1[i])
        m.cytokines.il6 = float(il6[i])
        m.cytokines.il10 = float(il10[i])
        m.cells.neutrophils = float(neut[i])
        m.cells.macrophages = float(macro[i])
        m.cells.monocytes = float(mono[i])
        m.cells.dendritic_cells = float(dc[i])
        m.cells.t_cells = float(tcells[i])
        m.cells.neutrophil_production = float(prod_neut[i])
        m.cells.monocyte_production = float(prod_mono[i])
        # Friberg transit state
        m.cells._transit_t1 = float(t1[i])
        m.cells._transit_t2 = float(t2[i])
        m.cells._transit_t3 = float(t3[i])
        m.cells._transit_t4 = float(t4[i])
        # IFN state
        m.ifn.ifn_alpha_beta = float(ifn_val[i])
        # Circadian clock
        m._sim_hour = (m._sim_hour + dt_h) % 24.0

    # --- Adaptive / complement / tissue-vs-blood sub-systems (doc/40) ---
    # The scalar ``step`` drives these by the same post-IFN effective antigen
    # signal it feeds the innate kernel.  Mirror them here with their own
    # vectorized O2/O9 cohort kernels so a cohort run exercises the full
    # doc/40 Phase B-F biology instead of silently skipping it.  Inert at
    # baseline (no infection/vaccine/biologic), so default trajectories are
    # unchanged; the blood compartments are refreshed from the just-written
    # circulating channels exactly as the scalar step does.
    epitope = list(np_max(0.0, stim_eff))
    _cohort_adaptive_step([m.adaptive for m in models], dt_h, epitope,
                          use_numpy=use_numpy)
    _cohort_complement_step([m.complement for m in models], dt_h, epitope,
                            use_numpy=use_numpy)
    _cohort_tissue_blood_step(
        [m.tissue_blood for m in models], dt_h, epitope,
        [m.cytokines.il6 for m in models],
        [m.cells.neutrophils for m in models],
        [m.cells.monocytes for m in models],
        use_numpy=use_numpy)


# ============================================================================
# Cohort runner (doc/39 O9-part-1 / doc/31 §2.4)
#
# Advances an entire cohort of virtual patients through T hours, feeding the
# O2 vectorized kernel.  The per-agent ODEs are independent (production scales
# are per-patient), so the cohort is embarrassingly parallel: contiguous model
# slabs are dispatched to distinct worker processes, each advancing its slab
# with ``cohort_immune_step``, then state is merged back.  Results are
# bit-identical to a single-process run regardless of ``workers``.
# ============================================================================


def _advance_slab(models: list[InnateImmuneModel], n_steps: int,
                  dt_h: float, use_numpy: bool | None
                  ) -> list[InnateImmuneModel]:
    """Advance a model slab for ``n_steps`` (worker entry point, O9).

    The slab is a pickled copy inside the worker process; the returned list
    carries the advanced state back to the parent for merging.
    """
    for _ in range(n_steps):
        cohort_immune_step(models, dt_h, use_numpy=use_numpy)
    return models


def _copy_state(src: InnateImmuneModel,
                dst: InnateImmuneModel) -> None:
    """Copy the full observable state of ``src`` onto ``dst`` in place."""
    dst.cytokines = src.cytokines
    dst.cells = src.cells
    dst.ifn = src.ifn
    dst.cortisol_suppression = src.cortisol_suppression
    dst.immunosuppression = src.immunosuppression
    dst.anti_inflammatory = src.anti_inflammatory
    dst.autoimmune_activation = src.autoimmune_activation
    dst.infection_severity = src.infection_severity
    dst._sim_hour = src._sim_hour
    dst.adaptive = src.adaptive
    dst.complement = src.complement
    dst.tissue_blood = src.tissue_blood
    dst.il6_biologic_occupancy = src.il6_biologic_occupancy


# ============================================================================
# Virtual population sampling (doc/40 Phase D, gap G13)
#
# Inter-individual variability per L6 (npj Sys Biol Appl 2023): baseline
# immune parameters are drawn per patient from log-normal distributions within
# physiologic variance bands, so a cohort produces a *distribution* of
# response rather than one representative run.  Drawn values are baked into
# each InnateImmuneModel's CytokinePool / ImmuneCellPopulation / IFNPool at
# construction; the resulting pool of models plugs straight into ``run_cohort``
# (doc/39 O2 vectorization).
#
# Determinism per doc/39 §5.3: the per-patient RNG stream is seeded as
# ``seed*1000003 + i`` — a pure function of (seed, patient index) — so cohort
# vectorization and multiprocessing slabting never change the per-patient
# trajectories.
# ============================================================================


def sample_virtual_population(
    n: int,
    seed: int = 0,
    *,
    sd_log: float = 0.15,
    rng: Any | None = None,
) -> list[InnateImmuneModel]:
    """Build ``n`` virtual patients with log-normal baseline variance (G13).

    Args:
        n: number of virtual patients (>= 1).
        seed: master seed; combined with patient index for deterministic,
            seed-independent-of-patient-order sampling.
        sd_log: log-normal standard deviation (symmetric ~±15% by default),
            giving a cohort spanning roughly the physiologic range.
        rng: optional caller-supplied ``random.Random``; if given, ``seed`` is
            ignored (used for the reduction sampling use-case).

    Returns:
        A list of ``n`` :class:`InnateImmuneModel` with sampled baselines.
    """
    n = max(1, int(n))
    import random as _rnd

    models: list[InnateImmuneModel] = []
    for i in range(n):
        local = rng if rng is not None else _rnd.Random(seed * 1000003 + i)

        def g(mean: float, local: Any = local, sd_log: float = sd_log) -> float:
            return math.exp(local.gauss(0.0, sd_log)) * mean

        m = InnateImmuneModel()

        # Cytokine baselines (pg/mL) around healthy reference values.
        m.cytokines.tnf_alpha = g(5.0)
        m.cytokines.il1_beta = g(2.0)
        m.cytokines.il6 = g(1.0)
        m.cytokines.il10 = g(5.0)

        # WBC / cell-pool baselines (±~30% around healthy).
        m.cells.neutrophils = g(4.0)
        m.cells.monocytes = g(0.4)
        m.cells.macrophages = g(0.5)
        m.cells.dendritic_cells = g(0.1)
        m.cells.t_cells = g(1.5)
        m.cells.nk_cells = g(0.3)
        m.cells.eosinophils = g(0.1)
        m.cells.basophils = g(0.05)

        # Production rates scale with the individual's set-point.
        m.cytokines.tnf_production_rate = g(10.0)
        m.cytokines.il6_production_rate = g(12.0)
        m.cytokines.il10_production_rate = g(6.0)
        m.cells.neutrophil_production = g(0.1)
        m.cells.monocyte_production = g(0.02)

        # IFN responsiveness.
        m.ifn.ifn_vmax = g(5.0)
        # Keep the Friberg transit chain steady-state WITHIN one patient
        # (transit compartments = production * tau) so a new individual is at
        # its own healthy ANC, not the population mean.
        m.cells._init_transit_steady_state()

        # Fix baselines captured at construction (so __post_init__ caches decay
        # constants against the sampled half-lives are consistent; re-seed the
        # per-field cached decay keys).
        m.cytokines.__post_init__()
        m.cells._init_transit_steady_state()

        models.append(m)
    return models


def run_cohort(
    models: list[InnateImmuneModel],
    n_steps: int,
    dt_h: float = 1.0,
    *,
    workers: int = 1,
    use_numpy: bool | None = None,
) -> None:
    """Advance a cohort of ``InnateImmuneModel`` for ``n_steps`` hours (O9).

    When ``workers > 1`` the cohort is split into contiguous slabs and each
    slab is advanced in its own ``multiprocessing`` worker using the O2
    vectorized kernel; per-agent ODEs are independent so the merge is
    bit-identical to a single-process run.  Falls back to the single-process
    loop when ``workers`` is 1 or process dispatch is unavailable.
    """
    if n_steps <= 0 or not models:
        return
    if workers <= 1:
        _advance_slab(models, n_steps, dt_h, use_numpy)
        return

    import multiprocessing as _mp

    worker_count = max(2, int(workers))
    slabs: list[list[InnateImmuneModel]] = [models[i::worker_count]
                                            for i in range(worker_count)]
    slabs = [s for s in slabs if s]
    if len(slabs) <= 1:
        _advance_slab(models, n_steps, dt_h, use_numpy)
        return
    ctx = _mp.get_context("spawn")
    with ctx.Pool(processes=len(slabs)) as pool:
        merged = pool.starmap(
            _advance_slab,
            [(slab, n_steps, dt_h, use_numpy) for slab in slabs],
        )
    # Merge advanced state back onto the caller's objects, keeping the
    # original list identity and element order.
    for slab_idx, slab in enumerate(merged):
        position = 0
        for i in range(slab_idx, len(models), worker_count):
            if position >= len(slab):
                break
            _copy_state(slab[position], models[i])
            position += 1
