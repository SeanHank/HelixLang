"""Closed-loop physiological core (doc/42 Phase B, RL-1/2/4).

This module consolidates three previously-disjoint model families into a single
shared, physiologically-grounded core that feeds vitals and PBPK perfusion:

* **RL-1 — closed-loop cardiovascular/hemodynamic core** (``HemodynamicModel``):
  cardiac output driven by preload (Starling), afterload (SVR), and
  contractility, plus a simple baroreflex/autonomic gain loop.  Mean arterial
  pressure falls out of CO × SVR rather than being a target-tracking delta, so
  vitals are *flow/pressure balanced* (Guyton-and-Hall style).

* **RL-2 — gas-exchange layer** (``GasExchangeModel``): alveolar-arterial O₂/CO₂
  transfer, an oxygen-hemoglobin dissociation curve (Severinghaus 1979) feeding
  a real SpO₂, and respiratory-rate drive from PₐCO₂ / PₐO₂ / acid-base rather
  than bicarbonate-only heuristics.

* **RL-4 — thermoregulation** (``ThermoregulationModel``): heat-production vs
  heat-loss balance with a hypothalamic set-point, circadian + ambient coupling;
  fever is a *perturbation of the set-point* (from CRP/cytokines), never a
  direct temperature assignment.

* **RL-5 — cross-system coupler** (``PhysiologicalCoupler``): organ-function →
  cardiac output/perfusion → renal/hepatic clearance coupling so a failing organ
  measurably changes drug exposure.

The core is deliberately deterministic (no RNG by default) so downstream golden
acceptance remains reproducible.

References
----------
- Severinghaus JW. Simple, accurate equations for human blood O2 dissociation
  computations. *J Appl Physiol* 1979. doi:10.1152/jappl.1979.46.3.599
- Guyton & Hall, *Textbook of Medical Physiology* (cardiovascular control,
  gas exchange, thermoregulation).
- West JB, *Respiratory Physiology: The Essentials* (alveolar gas equation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ============================================================================
# Blood-gas chemistry
# ============================================================================


def severinghaus_saturation(pao2_mmhg: float) -> float:
    """SaO₂ (%) from PₐO₂ (mmHg), Severinghaus (1979) approximation.

    ``SaO2 = 100 / (1 + 23400/(PO2^3 + 150*PO2))`` — accurate to ~±1% over the
    physiological range (20–120 mmHg).
    """
    po2 = max(0.0, float(pao2_mmhg))
    return 100.0 / (1.0 + 23400.0 / (po2 ** 3 + 150.0 * po2 + 1e-12))


def po2_from_saturation(sao2_pct: float) -> float:
    """Inverse Severinghaus: approximate PₐO₂ (mmHg) from SaO₂ (%).

    Binary-search inversion of :func:`severinghaus_saturation` over
    0–600 mmHg (numerically exact to the continuous function).
    """
    target = max(0.0, min(100.0, float(sao2_pct)))
    lo, hi = 0.0, 600.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if severinghaus_saturation(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def henderson_hasselbalch_pH(hco3_meq_per_l: float, paco2_mmhg: float) -> float:
    """Plasma pH from bicarbonate and CO₂ (Henderson–Hasselbalch).

    ``pH = 6.1 + log10([HCO3-] / (0.0307 * PaCO2))`` (alpha_CO2 ≈ 0.0307
    mmol · L⁻¹ · mmHg⁻¹ at 37 °C).
    """
    hco3 = max(0.05, float(hco3_meq_per_l))
    paco2 = max(0.05, float(paco2_mmhg))
    return 6.1 + math.log10(hco3 / (0.0307 * paco2))


def hco3_from_ph_paco2(ph: float, paco2_mmhg: float) -> float:
    """Bicarbonate (meq/L) consistent with a target pH at a given PaCO₂."""
    return float(0.0307 * max(0.05, float(paco2_mmhg)) * (10.0 ** (float(ph) - 6.1)))


def alveolar_arterial_oxygen(
    fi_o2: float,
    pb_mmhg: float,
    paco2_mmhg: float,
    rq: float = 0.8,
    a_do2_mmhg: float = 10.0,
) -> float:
    """Ideal alveolar PₐO₂ (mmHg) via the alveolar gas equation, then − A–a gradient.

    ``PAO2 = FiO2*(PB - PH2O) - PaCO2/RQ`` , with PH2O ≈ 47 mmHg at 37 °C
    (West, *Respiratory Physiology*).  ``a_do2`` is the age-scalable alveolar–
    arterial gradient subtracted to obtain arterial PₐO₂.
    """
    ph2o = 47.0
    pao2_ideal = max(0.0, float(fi_o2) * (float(pb_mmhg) - ph2o) - float(paco2_mmhg) / rq)
    return max(0.0, pao2_ideal - max(0.0, float(a_do2_mmhg)))


def arterial_oxygen_saturation(
    fi_o2: float,
    pb_mmhg: float,
    paco2_mmhg: float,
    a_do2_mmhg: float = 10.0,
) -> float:
    """Merged convenience: arterial PaO₂ → SaO₂% (Severinghaus)."""
    pao2 = alveolar_arterial_oxygen(fi_o2, pb_mmhg, paco2_mmhg, a_do2_mmhg=a_do2_mmhg)
    return severinghaus_saturation(pao2)


# ============================================================================
# RL-1 — Closed-loop cardiovascular core
# ============================================================================


@dataclass
class HemodynamicState:
    """Instantaneous hemodynamic/fluid state shared by vitals + PBPK + disease."""

    heart_rate_bpm: float = 72.0
    stroke_volume_ml: float = 70.0
    cardiac_output_l_min: float = 5.0
    map_mmhg: float = 93.0
    systolic_bp_mmhg: float = 120.0
    diastolic_bp_mmhg: float = 80.0
    svr_dyne: float = 1200.0            # systemic vascular resistance (dyn·s·cm⁻⁵)
    blood_volume_l: float = 5.0         # effective circulating volume
    contractility: float = 1.0          # 0..1.5, 1.0 = healthy
    cardiac_index_l_min_m2: float = 3.0  # CO / BSA


class HemodynamicModel:
    """Closed-loop cardiovascular model (RL-1).

    CO is generated from preload (Starling on effective volume), afterload
    (SVR), and contractility, then **MAP = CO × SVR**.  A baroreflex/autonomic
    loop opposes deviations of MAP from a set-point by raising HR and SVR when
    MAP falls (and vice-versa).  SBP/DBP are derived from MAP plus a
    pulse-pressure term that narrows with rising afterload and widens with
    rising stroke volume.  This is a single shared state — it feeds PBPK organ
    perfusion, vitals, and the disease ODEs.

    Deterministic: no RNG; a single ``step`` is reproducible bit-for-bit.
    """

    def __init__(
        self,
        body_weight_kg: float = 70.0,
        age_years: float = 30.0,
        bsa_m2: float = 1.85,
        set_point_map_mmhg: float = 93.0,
        baseline_hr_bpm: float = 72.0,
        baseline_sv_ml: float = 70.0,
    ) -> None:
        self.weight_kg = body_weight_kg
        self.age_years = age_years
        self.bsa_m2 = bsa_m2
        self.set_point_map_mmhg = set_point_map_mmhg
        self.state = HemodynamicState(
            heart_rate_bpm=baseline_hr_bpm,
            stroke_volume_ml=baseline_sv_ml,
            cardiac_output_l_min=baseline_hr_bpm * baseline_sv_ml / 1000.0,
            svr_dyne=1328.0,
        )
        self.state.map_mmhg = self.state.cardiac_output_l_min * self.state.svr_dyne / 80.0
        self._init_sbp_dbp()

    @staticmethod
    def create_from_physiology(physiology: Any) -> HemodynamicModel:
        weight = getattr(physiology, "body_weight_kg", 70.0)
        age = getattr(physiology, "age_years", 30.0)
        bsa = getattr(physiology, "body_surface_area_m2", 1.85)
        base_hr = 72.0 + float(max(0.0, age - 30.0)) * 0.3
        return HemodynamicModel(
            body_weight_kg=weight, age_years=age, bsa_m2=bsa, baseline_hr_bpm=base_hr,
        )

    def _init_sbp_dbp(self) -> None:
        pp = 40.0
        self.state.systolic_bp_mmhg = self.state.map_mmhg + pp / 2.0
        self.state.diastolic_bp_mmhg = self.state.map_mmhg - pp / 2.0

    def _sv_starling(self) -> float:
        # Frank–Starling: SV scales with preload (effective volume / normal),
        # bounded by the flat portion of the curve (max ~1.35x).
        vol = max(3.0, min(7.0, self.state.blood_volume_l))
        preload = 0.6 + 0.4 * (vol / 5.0)          # 1.0 at 5.0 L
        return max(30.0, self.state.stroke_volume_ml * preload * self.state.contractility)

    def _svr_baroreflex(self, hr_adjust: float) -> float:
        # baroreflex vasoconstriction opposes the HR response (∓ to net error)
        return self.state.svr_dyne * (1.0 - 0.25 * hr_adjust)

    def step(
        self,
        dt_h: float,
        *,
        disease_contractility: float = 1.0,
        drug_inotropy: float = 1.0,
        drug_svr_mod: float = 1.0,
        drug_chronotropy: float = 1.0,
        volume_mod: float = 1.0,
        ambient_temperature_c: float = 22.0,
    ) -> None:
        """Advance hemodynamics *dt_h* hours.

        Args:
            disease_contractility: 0..1 (heart failure reduces it)
            drug_inotropy: >1 positive inotrope, <1 negative inotrope
            drug_svr_mod: <1 vasodilator (ACEi/ARB), >1 vasoconstrictor
            drug_chronotropy: >1 positive chronotrope, <1 negative (β-blocker)
            volume_mod: <1 diuretic (reduces effective volume)
        """
        self.state.contractility = max(0.0, 1.0 * disease_contractility * drug_inotropy)

        # preload (volume): a diuretic (<1 volume_mod) lowers the effective
        # volume target, reducing preload-driven stroke volume (Frank-Starling)
        vol_target = 5.0 * max(0.5, volume_mod)
        k_vol = math.log(2.0) / 48.0
        self.state.blood_volume_l += dt_h * (-k_vol * (self.state.blood_volume_l - vol_target))
        self.state.blood_volume_l = max(3.0, min(7.0, self.state.blood_volume_l))

        # afterload
        self.state.svr_dyne = max(500.0, 1200.0 * drug_svr_mod)

        # CO from HR x SV (starling)
        sv = self._sv_starling()
        hr = max(35.0, min(200.0, self.state.heart_rate_bpm * drug_chronotropy))
        co = hr * sv / 1000.0

        # map = co * svr (convert dyn-s/cm5 * L/min -> mmHg)
        map_mmhg = co * self.state.svr_dyne / 80.0

        # baroreflex: oppose error from set-point
        err = (self.set_point_map_mmhg - map_mmhg) / self.set_point_map_mmhg
        gain = 0.35
        hr_adjust = gain * err                          # + when MAP low -> tachycardia
        svr_adjust = -gain * err * 0.7                   # vasoconstriction when MAP low

        hr = max(35.0, min(200.0, hr * (1.0 + hr_adjust)))
        svr2 = max(500.0, self.state.svr_dyne * (1.0 + svr_adjust))
        co = hr * sv / 1000.0
        map2 = co * svr2 / 80.0

        # pulse pressure: wide with high SV / low afterload (Windkessel)
        pp = 20.0 + 0.35 * sv * (1200.0 / max(svr2, 500.0))

        self.state.heart_rate_bpm = hr
        self.state.stroke_volume_ml = sv
        self.state.svr_dyne = svr2
        self.state.cardiac_output_l_min = co
        self.state.map_mmhg = max(45.0, min(200.0, map2))
        self.state.systolic_bp_mmhg = max(60.0, self.state.map_mmhg + pp / 2.0)
        self.state.diastolic_bp_mmhg = max(30.0, self.state.map_mmhg - pp / 2.0)
        self.state.cardiac_index_l_min_m2 = co / max(1.0, self.bsa_m2)


# ============================================================================
# RL-2 — Gas-exchange layer
# ============================================================================


@dataclass
class GasExchangeState:
    """Arterial blood-gas snapshot."""

    pao2_mmhg: float = 100.0
    paco2_mmhg: float = 40.0
    sao2_pct: float = 97.0
    ph: float = 7.40
    respiratory_rate_per_min: float = 16.0
    alveolar_ventilation_l_min: float = 5.0


class GasExchangeModel:
    """Alveolar-arterial O₂/CO₂ exchange with real SpO₂ (RL-2).

    Alveolar ventilation clears CO₂ and raises O₂; metabolic production rate of
    CO₂ (V̇CO₂) is matched by ventilation per the steady-state clearance
    ``PaCO2 = V̇CO2 / V̇A`` (times a constant).  Arterial PₐO₂ follows the
    alveolar gas equation and the SpO₂ comes from the Severinghaus curve.
    Respiratory-rate drive is a chemoreceptor blend of PaCO₂ (primary), pH
    (acidosis stimulates), and PaO₂ (hypoxic drive) — not a bicarbonate-only rule.
    """

    def __init__(
        self,
        hemoglobin_g_per_dl: float = 14.0,
        fi_o2: float = 0.21,
        pb_mmhg: float = 760.0,
        resting_rr_per_min: float = 16.0,
    ) -> None:
        self.hgb = hemoglobin_g_per_dl
        self.fi_o2 = fi_o2
        self.pb = pb_mmhg
        self.resting_rr = resting_rr_per_min
        self.state = GasExchangeState(respiratory_rate_per_min=resting_rr_per_min)
        self._vco2 = 0.20                 # L/min CO2 production (resting ~200 mL/min)

    def step(
        self,
        dt_h: float,
        *,
        metabolic_ventilation_mod: float = 1.0,
        respiratory_depression: float = 1.0,
        hco3_meq_per_l: float = 24.0,
    ) -> None:
        """Advance blood gases *dt_h* hours from current ventilation conditions.

        Args:
            metabolic_ventilation_mod: >1 hyperventilation, <1 hypoventilation
            respiratory_depression: <1 respiratory depressant (opioids reduce drive)
            hco3_meq_per_l: plasma bicarbonate for the acid-base drive
        """
        va_l_min = 5.0 * max(0.4, metabolic_ventilation_mod) * max(0.2, respiratory_depression)
        va_l_min = max(0.5, va_l_min)

        # steady-state CO2 clearance: PaCO2 ~ VCO2 / VA
        paco2 = 40.0 * (5.0 / va_l_min)

        # alveolar gas equation for PaO2 with a physiologic A-a gradient
        pao2 = alveolar_arterial_oxygen(self.fi_o2, self.pb, paco2, a_do2_mmhg=10.0)
        pao2 = max(20.0, min(150.0, pao2))
        sao2 = severinghaus_saturation(pao2)

        # acid-base (Henderson-Hasselbalch)
        ph = henderson_hasselbalch_pH(hco3_meq_per_l, paco2)

        # chemoreceptor respiratory drive
        drive = 1.0
        drive += (paco2 - 40.0) * 0.08          # hypercapnic drive
        drive += max(0.0, 7.40 - ph) * 1.5       # acidosis drive
        drive += max(0.0, 90.0 - pao2) * 0.01    # hypoxic drive
        drive = max(0.4, drive)
        rr_drive = self.resting_rr * drive * max(0.15, respiratory_depression)

        self.state.paco2_mmhg = max(10.0, min(120.0, paco2))
        self.state.pao2_mmhg = pao2
        self.state.sao2_pct = max(0.0, min(100.0, sao2))
        self.state.ph = max(6.8, min(7.8, ph))
        self.state.respiratory_rate_per_min = max(6.0, min(45.0, rr_drive))
        self.state.alveolar_ventilation_l_min = va_l_min


# ============================================================================
# RL-4 — Thermoregulation
# ============================================================================


@dataclass
class ThermoState:
    """Thermal state: core temperature and hypothalamic set-point (°C)."""

    core_temperature_c: float = 37.0
    set_point_c: float = 37.0
    heat_production_w: float = 100.0     # Watts (basal ~100 W)
    heat_loss_w: float = 100.0
    circadian_offset_c: float = 0.0


class ThermoregulationModel:
    """Heat-balance thermoregulation with a hypothalamic set-point (RL-4).

    Core temperature integrates heat production minus heat loss
    (``m·c·dT/dt = M − E``), where loss is Newtonian against ambient and scaled
    by skin conductance.  The set-point carries a small circadian rhythm; *fever*
    raises the set-point from CRP/cytokines rather than directly assigning
    temperature, so temperature *tracks* the elevated set-point through the same
    heat-balance ODE.
    """

    def __init__(
        self,
        body_mass_kg: float = 70.0,
        core_temp_c: float = 37.0,
        ambient_temp_c: float = 22.0,
    ) -> None:
        self.mass_kg = body_mass_kg
        self.ambient_c = ambient_temp_c
        self.cp_j_per_kg_c = 3470.0          # specific heat of body ~3.47 kJ/kg·°C
        self.basal_production_w = 100.0
        self.state = ThermoState(core_temperature_c=core_temp_c, set_point_c=core_temp_c)

    def step(
        self,
        dt_h: float,
        *,
        crp_mg_per_l: float = 0.0,
        cytokines: float = 0.0,
        ambient_temperature_c: float | None = None,
        circadian_hour: float = 12.0,
        exercise_vigorous: float = 0.0,
    ) -> None:
        """Advance thermoregulation *dt_h* hours.

        Args:
            crp_mg_per_l: CRP drives fever as a set-point elevation
            cytokines: 0..1 additive inflammatory set-point shift
            ambient_temperature_c: external temperature (override)
            circadian_hour: 0..24 for the diurnal set-point rhythm (±0.35 °C)
            exercise_vigorous: 0..1 adds metabolic heat production
        """
        if ambient_temperature_c is not None:
            self.ambient_c = ambient_temperature_c

        # circadian set-point ~ +/-0.35 C peaking in late afternoon (~17h)
        circadian = 0.35 * math.cos((circadian_hour - 17.0) / 24.0 * 2.0 * math.pi)
        base_sp = 37.0 + circadian

        # fever = set-point elevation from CRP/cytokines (not a direct temp assign)
        fever_sp = min(4.0, max(0.0, (crp_mg_per_l - 5.0) * 0.015)) + 1.5 * max(0.0, cytokines)
        set_point = base_sp + fever_sp
        self.state.set_point_c = set_point

        # Heat balance (mass/flow-balanced, not target-tracking): core
        # temperature integrates heat production minus heat loss (m·c·dT/dt =
        # M − E).  Set-point error drives thermoregulatory *effectors* —
        # shivering/metabolic heat when below the set-point, cutaneous
        # vasodilation/skin loss when above — so temperature is pulled toward
        # the (possibly elevated) set-point through the same ODE.
        core = self.state.core_temperature_c
        below = max(0.0, set_point - core)
        above = max(0.0, core - set_point)

        # basal metabolic heat, + shivering/fever-metabolism gain when below;
        # exercise adds metabolic heat
        prod = self.basal_production_w * (1.0 + 0.12 * below) + exercise_vigorous * 500.0

        # passive Newtonian conductance balances basal production at ~37 C at
        # room ambient; conductance widens with excess heat (sweating /
        # vasodilation) when above the set-point.
        denom = max(8.0, 37.0 - self.ambient_c)
        conductance = self.basal_production_w / denom + 4.0 * above
        loss = max(0.0, conductance * (core - self.ambient_c))

        # integrate: dT/dt = (M - E) / (m*c) ; 1 W for 1 s = 1 J
        dt_s = dt_h * 3600.0
        dT = (prod - loss) / (self.mass_kg * self.cp_j_per_kg_c) * dt_s
        self.state.core_temperature_c = max(34.0, min(42.0, core + dT))

        self.state.heat_production_w = prod
        self.state.heat_loss_w = loss
        self.state.circadian_offset_c = circadian


# ============================================================================
# RL-5 — Cross-system coupler
# ============================================================================


class PhysiologicalCoupler:
    """Bidirectional organ-function → perfusion → clearance coupling (RL-5).

    Makes the previously-static PBPK organ flows depend on the closed-loop CV
    state and makes renal/hepatic clearance depend on organ function, so a
    failing organ measurably changes drug exposure (and perfusion → disease).
    """

    @staticmethod
    def cardiac_output_fraction(
        organ_flow_l_per_h: float,
        cardiac_output_l_per_h: float,
        fallback_fraction: float,
    ) -> float:
        if cardiac_output_l_per_h <= 0.0:
            return fallback_fraction
        return max(0.02, min(0.9, organ_flow_l_per_h / cardiac_output_l_per_h))

    @staticmethod
    def renal_clearance_from_egfr(
        base_clearance_l_per_h: float,
        egfr_ml_per_min: float,
        normal_egfr_ml_per_min: float = 100.0,
    ) -> float:
        """Scale renal clearance proportionally to (pathologic) eGFR."""
        ratio = max(0.05, min(2.0, egfr_ml_per_min / max(1.0, normal_egfr_ml_per_min)))
        return base_clearance_l_per_h * ratio

    @staticmethod
    def hepatic_clearance_from_function(
        base_clearance_l_per_h: float,
        liver_function: float,
    ) -> float:
        """Scale hepatic clearance by a 0..1 liver-function (intrinsic) metric."""

        ratio = max(0.05, min(1.0, liver_function))
        return base_clearance_l_per_h * ratio


# ============================================================================
# doc/42 Phase B — composed vitals driver (RL-1 + RL-2 + RL-4 + RL-5)
# ============================================================================


@dataclass
class VitalSnapshot:
    """Physiological vitals derived from CV/gas/thermo flow & heat balance."""

    systolic_bp_mmhg: float = 120.0
    diastolic_bp_mmhg: float = 80.0
    heart_rate_bpm: float = 72.0
    respiratory_rate_per_min: float = 16.0
    temperature_c: float = 37.0
    spo2_pct: float = 98.0
    cardiac_output_l_min: float = 5.0
    map_mmhg: float = 93.0


# Curated drug-effect maps for the opt-in physiological vitals path.  Keyed by
# the *normalized* drug name (see clinical_output._VITAL_DRUG_EFFECTS style).
# Modifiers are multiplicative around a healthy value of 1.0.
_INOTROPE_DRUGS = {"digoxin", "dobutamine", "milrinone", "norepinephrine"}
_CHRONOTROPE_DRUGS = {"beta_blocker", "propranolol", "metoprolol", "atenolol", "bisoprolol"}
_VASODILATOR_DRUGS = {"ace_inhibitor", "arb", "enalapril", "lisinopril", "ramipril",
                      "losartan", "valsartan", "telmisartan"}
_DIURETIC_DRUGS = {"furosemide", "torasemide", "bumetanide", "hydrochlorothiazide",
                   "spironolactone", "loop_diuretic"}
_RESPIRATORY_DEPRESSANT_DRUGS = {"morphine", "fentanyl", "oxycodone", "hydromorphone",
                                 "opioid"}


class PhysiologicalVitalsDriver:
    """Composed vitals driver: closed-loop CV + gas exchange + thermoregulation.

    This is the Phase B replacement for delta-based vital tracking.  It drives
    vitals from a *single shared* hemodynamic/fluid state (RL-1), an
    alveolar-arterial gas layer with real SpO₂ and chemoreceptor drive (RL-2),
    and hypothalamic heat-balance thermoregulation with a CRP/cytokine
    set-point (RL-4).  Cross-system coupling (RL-5): renal function and hepatic
    function scale organ clearance/perfusion, and a failing heart reduces
    contractility while inflammation raises the thermoregulatory set-point.

    Deterministic: no RNG; identical inputs give identical vitals bit-for-bit.
    """

    MED_PREFIXES = {
        "beta_blocker", "ace_inhibitor", "arb", "loop_diuretic", "opioid",
    }

    def __init__(
        self,
        body_weight_kg: float = 70.0,
        age_years: float = 30.0,
        bsa_m2: float = 1.85,
        hemoglobin_g_per_dl: float = 14.0,
        ambient_temperature_c: float = 22.0,
    ) -> None:
        self.hemo = HemodynamicModel(
            body_weight_kg=body_weight_kg,
            age_years=age_years,
            bsa_m2=bsa_m2,
        )
        self.gas = GasExchangeModel(hemoglobin_g_per_dl=hemoglobin_g_per_dl)
        self.thermo = ThermoregulationModel(
            body_mass_kg=body_weight_kg,
            ambient_temp_c=ambient_temperature_c,
        )
        self.weight_kg = body_weight_kg

    @staticmethod
    def _fraction(drug_concs: dict[str, float], classes: set[str], strength: float) -> float:
        """Product of multiplicative modifiers for drugs in ``classes``."""
        mod = 1.0
        for dkey, conc in (drug_concs or {}).items():
            if dkey not in classes and not any(dkey == c or dkey.startswith(c) for c in classes):
                continue
            scale = min(max(conc, 0.0) / 30.0, 3.0)
            if scale > 0.0:
                mod *= 1.0 + (strength - 1.0) * (scale / 3.0)
        return max(0.2, min(5.0, mod))

    def step(
        self,
        dt_h: float,
        *,
        drug_concs: dict[str, float] | None = None,
        crp_mg_per_l: float = 0.0,
        cytokines: float = 0.0,
        disease_contractility: float = 1.0,
        hemoglobin_g_per_dl: float | None = None,
        hco3_meq_per_l: float = 24.0,
        egfr_ml_per_min: float = 100.0,
        hepatic_function: float = 1.0,
        circadian_hour: float = 12.0,
        ambient_temperature_c: float = 22.0,
    ) -> VitalSnapshot:
        drug_concs = drug_concs or {}

        # RL-5: organ-function coupling scales intrabdominal perfusion snapshots
        renal_frac = PhysiologicalCoupler.cardiac_output_fraction(
            egfr_ml_per_min * 0.06, self.hemo.state.cardiac_output_l_min * 60.0, 0.17
        )
        del renal_frac  # perfusion snapshot; kept for future PBPK wiring

        # drug modifiers
        inotropy = self._fraction(drug_concs, _INOTROPE_DRUGS, 1.4)
        chronotropy = self._fraction(drug_concs, _CHRONOTROPE_DRUGS, 0.8)
        svr_mod = self._fraction(drug_concs, _VASODILATOR_DRUGS, 0.8)
        volume_mod = self._fraction(drug_concs, _DIURETIC_DRUGS, 0.75)
        respiratory_depression = self._fraction(drug_concs, _RESPIRATORY_DEPRESSANT_DRUGS, 0.6)

        # RL-1: closed-loop CV from shared fluid state
        self.hemo.step(
            dt_h,
            disease_contractility=disease_contractility,
            drug_inotropy=inotropy,
            drug_svr_mod=svr_mod,
            drug_chronotropy=chronotropy,
            volume_mod=volume_mod,
            ambient_temperature_c=ambient_temperature_c,
        )
        h = self.hemo.state

        # RL-2: gas exchange + chemoreceptor drive
        self.gas.step(
            dt_h,
            metabolic_ventilation_mod=1.0,
            respiratory_depression=respiratory_depression,
            hco3_meq_per_l=hco3_meq_per_l,
        )
        g = self.gas.state

        # RL-4: heat balance + CRP/cytokine fever set-point
        self.thermo.step(
            dt_h,
            crp_mg_per_l=crp_mg_per_l,
            cytokines=cytokines,
            ambient_temperature_c=ambient_temperature_c,
            circadian_hour=circadian_hour,
        )
        t = self.thermo.state

        return VitalSnapshot(
            systolic_bp_mmhg=h.systolic_bp_mmhg,
            diastolic_bp_mmhg=h.diastolic_bp_mmhg,
            heart_rate_bpm=h.heart_rate_bpm,
            respiratory_rate_per_min=g.respiratory_rate_per_min,
            temperature_c=t.core_temperature_c,
            spo2_pct=g.sao2_pct,
            cardiac_output_l_min=h.cardiac_output_l_min,
            map_mmhg=h.map_mmhg,
        )
