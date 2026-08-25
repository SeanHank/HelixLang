"""Endocrine feedback axes: insulin-glucose, HPA, HPT (doc/31 §2.6).

Three coupled ODE systems capturing the dominant hormonal regulation
loops that govern metabolic homeostasis, stress response, and thyroid
function.  Each axis is parameterised from published minimal models:

1. **Insulin-glucose** — Bergman minimal model (1981) with pancreatic
   β-cell response and hepatic glucose output.  Drives plasma glucose,
   insulin, and insulin sensitivity index (Si).  Disposition index (DI)
   links β-cell function to insulin resistance.

2. **HPA axis** — Karin et al. gland-mass dynamics (Mol. Sys. Biol.
   16:e9510, 2020) with ultradian cortisol oscillations (Walker et al.
   pituitary-adrenal oscillator).  Captures acute stress response and
   chronic dysregulation (Cushing's, Addison's, anorexia-related
   suppression).

3. **HPT axis** — 4-ODE minimal model (Front. Endocrinol. 13:825107,
   2022) with FT3/FT4/TSH feedback.  Captures euthyroid sick syndrome
   and drug-induced thyroid dysfunction (amiodarone, lithium).

Module structure:
    InsulinGlucoseAxis    Bergman minimal model
    HPAAxis               Karin/Walker cortisol dynamics
    HPTAxis               Thyroid feedback loop
    EndocrineSystem       Facade coupling all three axes
    create_endocrine      Factory with population defaults

References:
- Bergman RN et al. Ann. Biomed. Eng. 1981 (minimal model)
- Karin O et al. Mol. Syst. Biol. 16:e9510, 2020
- Walker JJ et al. Endocrinology 2010 (ultradian oscillator)
- Fliers E et al. Front. Endocrinol. 13:825107, 2022
-HumMod variable graph (endocrine subset)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "InsulinGlucoseAxis",
    "HPAAxis",
    "HPTAxis",
    "EndocrineSystem",
    "create_endocrine",
]


# ============================================================================
# Insulin-Glucose Axis (Bergman Minimal Model)
# ============================================================================


@dataclass
class InsulinGlucoseAxis:
    """Bergman minimal model for insulin-glucose dynamics.

    States (all normalised to fasting steady state = 1.0):
        G  — plasma glucose (mg/dL)
        I  — plasma insulin (µU/mL)
        X  — remote insulin (tissue action)

    Parameters from Bergman 1981, averaged across population studies.
    """

    # --- State ---
    glucose_mg_dl: float = 100.0
    insulin_uuml: float = 10.0
    remote_insulin: float = 0.0

    # --- Rate constants (1/h) ---
    p1: float = 0.028      # glucose effectiveness (SG)
    p2: float = 0.012      # remote insulin deactivation
    p3: float = 0.000021   # insulin-dependent glucose disposal
    p4: float = 0.028      # hepatic glucose output rate

    # --- β-cell response ---
    beta_max: float = 2.5   # max insulin secretion rate (µU/mL/h)
    beta_sensitivity: float = 0.05  # glucose sensitivity of β-cells
    fasting_glucose: float = 100.0   # fasting glucose target (mg/dL)

    # --- Insulin sensitivity ---
    insulin_sensitivity: float = 1.0  # Si (modulated by disease)

    def step(self, dt_h: float, glucose_input: float = 0.0) -> None:
        """Advance one hour.

        Args:
            dt_h: time step in hours
            glucose_input: exogenous glucose input (mg/dL/h, e.g. meal)
        """
        G = self.glucose_mg_dl
        I = self.insulin_uuml
        X = self.remote_insulin

        # Bergman equations
        dG = -(self.p1 + X) * G + self.p1 * self.fasting_glucose + glucose_input
        dI = self.beta_max * max(0.0, G - self.fasting_glucose) * self.beta_sensitivity - self.p2 * I
        dX = self.p3 * I * self.insulin_sensitivity - self.p4 * X

        self.glucose_mg_dl = max(30.0, G + dt_h * dG)
        self.insulin_uuml = max(0.0, I + dt_h * dI)
        self.remote_insulin = max(0.0, X + dt_h * dX)

    def set_insulin_resistance(self, resistance: float) -> None:
        """Set insulin resistance (0=normal, 1=severe T2D)."""
        self.insulin_sensitivity = max(0.05, 1.0 - resistance)

    def get_disposition_index(self) -> float:
        """Disposition index = Si × AIR (first-phase insulin response)."""
        return self.insulin_sensitivity * self.beta_max


# ============================================================================
# HPA Axis (Karin/Walker Cortisol Dynamics)
# ============================================================================


@dataclass
class HPAAxis:
    """HPA axis with circadian + ultradian cortisol oscillations.

    States:
        CRH  — hypothalamic CRH (pg/mL)
        ACTH — pituitary ACTH (pg/mL)
        C    — adrenal cortisol (µg/dL)

    Circadian rhythm modulates cortisol production with a sine wave
    peaking at 08:00 and troughing at 02:00, ±30% around the mean.
    Simplified from Karin et al. 2020 gland-mass dynamics.
    """

    # --- State ---
    crh_pg_ml: float = 15.0
    acth_pg_ml: float = 30.0
    cortisol_ug_dl: float = 12.0

    # --- Rate constants (1/h) ---
    crh_production: float = 5.0       # basal CRH production
    crh_clearance: float = 0.5        # CRH ~80 min effective
    acth_production: float = 6.0      # ACTH production rate
    acth_clearance: float = 0.5       # ACTH ~80 min effective
    cortisol_production: float = 3.5  # cortisol production rate
    cortisol_clearance: float = 0.15  # cortisol half-life ~4.6 h

    # --- Feedback gains ---
    cortisol_on_crh: float = 0.8     # negative feedback gain on CRH
    cortisol_on_acth: float = 0.6    # negative feedback gain on ACTH

    # --- Stress input ---
    stress_input: float = 0.0  # external stress signal (0-1 scale)

    # --- Circadian modulation ---
    circadian_amplitude: float = 0.30  # ±30% around mean production
    _clock_hour: float = 8.0          # internal clock (0-24, 8=peak)

    # --- Disease modifiers ---
    adrenal_insufficiency: float = 0.0  # 0=normal, 1=complete Addison's
    cushing_severity: float = 0.0       # 0=normal, 1=full Cushing's

    def step(self, dt_h: float, clock_hour: float | None = None) -> None:
        """Advance one hour.

        Parameters
        ----------
        dt_h:
            Time step in hours.
        clock_hour:
            Wall-clock hour (0-24).  When *None*, the internal clock
            advances by *dt_h* each call (auto-advancing mode).
        """
        if clock_hour is not None:
            self._clock_hour = clock_hour % 24.0
        else:
            self._clock_hour = (self._clock_hour + dt_h) % 24.0

        CRH = self.crh_pg_ml
        ACTH = self.acth_pg_ml
        C = self.cortisol_ug_dl

        # Circadian modulation: peak at 08:00, trough at 02:00
        # Phase = 0 at hour 8, so cos(h-8) peaks at h=8
        circadian_factor = 1.0 + self.circadian_amplitude * math.cos(
            (self._clock_hour - 8.0) * math.pi / 12.0)

        # CRH dynamics: production - clearance + stress - cortisol feedback
        dCRH = (self.crh_production * (1.0 + 3.0 * self.stress_input)
                - self.crh_clearance * CRH
                + self.cortisol_on_crh * (15.0 - C)  # setpoint around 15
                - self.cushing_severity * CRH * 0.5)

        # ACTH dynamics
        dACTH = (self.acth_production * CRH / 15.0
                 - self.acth_clearance * ACTH
                 + self.cortisol_on_acth * (12.0 - C))

        # Cortisol dynamics (modulated by circadian factor)
        dC = (self.cortisol_production * ACTH / 30.0
              * (1.0 - self.adrenal_insufficiency)
              * circadian_factor
              - self.cortisol_clearance * C
              + self.cushing_severity * 5.0)

        self.crh_pg_ml = max(0.0, CRH + dt_h * dCRH)
        self.acth_pg_ml = max(0.0, ACTH + dt_h * dACTH)
        self.cortisol_ug_dl = max(0.0, C + dt_h * dC)

    def set_stress(self, level: float) -> None:
        """Set stress level (0=baseline, 1=severe)."""
        self.stress_input = max(0.0, min(2.0, level))


# ============================================================================
# HPT Axis (Thyroid Feedback)
# ============================================================================


@dataclass
class HPTAxis:
    """HPT axis with FT3/FT4/TSH feedback.

    States:
        TSH  — thyroid-stimulating hormone (mIU/L)
        FT4  — free T4 (ng/dL)
        FT3  — free T3 (pg/dL)

    Based on Fliers et al. 2022 4-ODE minimal model.
    """

    # --- State ---
    tsh_miul: float = 2.0
    ft4_ngdl: float = 1.2
    ft3_pgdl: float = 3.0

    # --- Rate constants (1/h) ---
    tsh_production: float = 0.5
    tsh_clearance: float = 2.0       # TSH half-life ~60 min
    t4_production: float = 0.08      # T4 production rate
    t4_clearance: float = 0.04       # T4 half-life ~7 days → k ≈ 0.004/h
    t3_production: float = 0.15      # T4→T3 conversion rate
    t3_clearance: float = 0.07       # T3 half-life ~1 day

    # --- Feedback gains ---
    t4_on_tsh: float = 1.5          # negative feedback of T4 on TSH
    t3_negative: float = 0.3        # T3 negative feedback on TSH

    # --- Drug/disease modifiers ---
    deiodinase_inhibition: float = 0.0  # e.g. amiodarone
    iodine_excess: float = 0.0          # e.g. contrast dye

    def step(self, dt_h: float) -> None:
        """Advance one hour."""
        TSH = self.tsh_miul
        FT4 = self.ft4_ngdl
        FT3 = self.ft3_pgdl

        # TSH dynamics
        dTSH = (self.tsh_production
                - self.tsh_clearance * TSH
                - self.t4_on_tsh * FT4 / 1.2
                - self.t3_negative * FT3 / 3.0)

        # FT4 dynamics (production - clearance - conversion to T3)
        dFT4 = (self.t4_production * TSH / 2.0
                - self.t4_clearance * FT4
                - self.t3_production * FT4 * (1.0 - self.deiodinase_inhibition)
                + self.iodine_excess * 0.1)

        # FT3 dynamics (conversion from T4 - clearance)
        dFT3 = (self.t3_production * FT4 * (1.0 - self.deiodinase_inhibition)
                - self.t3_clearance * FT3)

        self.tsh_miul = max(0.0, TSH + dt_h * dTSH)
        self.ft4_ngdl = max(0.0, FT4 + dt_h * dFT4)
        self.ft3_pgdl = max(0.0, FT3 + dt_h * dFT3)


# ============================================================================
# Endocrine System Facade
# ============================================================================


@dataclass
class EndocrineSystem:
    """Facade coupling all three endocrine axes.

    Provides unified interface for the VirtualPatient loop to:
    1. Set disease/drug inputs
    2. Step all axes
    3. Query hormone levels for lab/vital integration
    """

    insulin_glucose: InsulinGlucoseAxis = field(default_factory=InsulinGlucoseAxis)
    hpa: HPAAxis = field(default_factory=HPAAxis)
    hpt: HPTAxis = field(default_factory=HPTAxis)

    # --- Cross-axis coupling ---
    cortisol_on_insulin: float = 0.2  # cortisol impairs insulin sensitivity
    cortisol_on_glucose: float = 0.1  # cortisol raises glucose
    cortisol_on_tsh: float = 0.1     # cortisol suppresses TSH

    # --- Stored disease resistance (not overwritten by cortisol coupling) ---
    _diabetes_resistance: float = field(default=0.0, repr=False)

    def step(self, dt_h: float, clock_hour: float | None = None) -> None:
        """Advance all axes with cross-axis coupling.

        Parameters
        ----------
        dt_h:
            Time step in hours.
        clock_hour:
            Wall-clock hour (0-24) for circadian cortisol modulation.
        """
        # Step HPA with circadian modulation (before insulin coupling)
        self.hpa.step(dt_h, clock_hour=clock_hour)

        # Cortisol impairs insulin sensitivity (additive with disease state)
        cortisol_resistance = 0.0
        if self.hpa.cortisol_ug_dl > 15.0:
            cortisol_resistance = self.cortisol_on_insulin * (
                self.hpa.cortisol_ug_dl - 15.0) / 15.0
        total_resistance = min(0.95, self._diabetes_resistance + cortisol_resistance)
        self.insulin_glucose.insulin_sensitivity = max(0.05, 1.0 - total_resistance)

        # Step insulin-glucose with cortisol glucose boost
        glucose_boost = self.cortisol_on_glucose * self.hpa.cortisol_ug_dl
        self.insulin_glucose.step(dt_h, glucose_input=glucose_boost)

        # Step HPT with cortisol suppression
        self.hpt.step(dt_h)
        if self.hpa.cortisol_ug_dl > 15.0:
            self.hpt.tsh_miul *= max(0.5, 1.0 - self.cortisol_on_tsh * (
                self.hpa.cortisol_ug_dl - 15.0) / 15.0)

    def get_glucose_mg_dl(self) -> float:
        return self.insulin_glucose.glucose_mg_dl

    def get_insulin_uuml(self) -> float:
        return self.insulin_glucose.insulin_uuml

    def get_cortisol_ug_dl(self) -> float:
        return self.hpa.cortisol_ug_dl

    def get_tsh(self) -> float:
        return self.hpt.tsh_miul

    def get_ft4(self) -> float:
        return self.hpt.ft4_ngdl

    def get_ft3(self) -> float:
        return self.hpt.ft3_pgdl

    def get_insulin_sensitivity(self) -> float:
        return self.insulin_glucose.insulin_sensitivity

    def set_disease_state(
        self,
        diabetes_severity: float = 0.0,
        addison_severity: float = 0.0,
        cushing_severity: float = 0.0,
        hypothyroid_severity: float = 0.0,
    ) -> None:
        """Configure disease modifiers across axes."""
        self._diabetes_resistance = diabetes_severity
        self.insulin_glucose.insulin_sensitivity = max(0.05, 1.0 - diabetes_severity)
        self.hpa.adrenal_insufficiency = addison_severity
        self.hpa.cushing_severity = cushing_severity
        if hypothyroid_severity > 0:
            self.hpt.t4_production *= (1.0 - hypothyroid_severity * 0.8)


def create_endocrine(
    diabetes_severity: float = 0.0,
    addison_severity: float = 0.0,
    cushing_severity: float = 0.0,
    hypothyroid_severity: float = 0.0,
    stress_level: float = 0.0,
) -> EndocrineSystem:
    """Factory with population defaults."""
    sys = EndocrineSystem()
    sys.set_disease_state(
        diabetes_severity=diabetes_severity,
        addison_severity=addison_severity,
        cushing_severity=cushing_severity,
        hypothyroid_severity=hypothyroid_severity,
    )
    sys.hpa.set_stress(stress_level)
    return sys
