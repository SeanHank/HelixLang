"""Renal function and CKD progression model (doc/30 §6).

Implements the regulatory-endorsed **eGFR-slope formalism** for chronic
kidney disease progression inside the virtual patient:

* **CKD-EPI 2021** race-free creatinine equation converts serum
  creatinine into reported eGFR; its numerical inverse maps the true
  filtering capacity back to a creatinine steady state.
* **Two-slope decomposition** (acute dip + chronic slope) reproduces
  exactly what DAPA-CKD / EMPA-KIDNEY / FIDELIO-DKD report: RAAS-
  blockade and SGLT2-inhibitor initiation cause a small acute
  hemodynamic dip while improving the *chronic* slope ~30-50%.
* **Serum-creatinine kinetics** are first-order approach to the
  steady state implied by current GFR (constant generation,
  clearance proportional to GFR), so observed creatinine -- and hence
  reported eGFR -- lags true function during AKI, as in real patients.
* **KFRE** 4-variable equation (Tangri et al. JAMA 2011; multinational
  recalibration Tangri et al. JAMA 2016) gives 2-/5-year kidney-failure
  risk from age, sex, eGFR, UACR.

Drug channels: SGLT2 inhibitors, RAAS blockade (ACEi/ARB/MRA), NSAIDs
(with ACEi+diuretic "triple whammy"), nephrotoxins via AKI induction.

Module structure:
    ckd_epi_2021          creatinine-only CKD-EPI 2021 eGFR
    inverse_ckd_epi       creatinine implied by a target GFR
    RenalFunctionModel    hourly-step renal model
    create_renal_model    factory

References:
- Inker LA et al. JASN 2019 (eGFR slope as regulatory surrogate)
- KDIGO 2024 CKD guideline (G/A categories, risk heatmap)
- DAPA-CKD NEJM 2020; EMPA-KIDNEY NEJM 2023; FIDELIO-DKD (acute dip +
  chronic slope calibration anchors)
- Tangri N et al. JAMA 2011;305:1553 (KFRE derivation)
- Tangri N et al. JAMA 2016;315:164 (multinational recalibration;
  baseline survivals used here)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from helixlang.human.disease_progression import DiseaseStage

__all__ = [
    "ckd_epi_2021",
    "inverse_ckd_epi",
    "RenalFunctionModel",
    "create_renal_model",
]

_HOURS_PER_YEAR = 24.0 * 365.25

#: KFRE 4-variable linear predictor coefficients (centered covariates)
_KFRE_COEF_AGE = -0.2201        # per age/10 minus 7.036
_KFRE_CENTER_AGE = 7.036
_KFRE_COEF_MALE = 0.2467        # male(1/0) minus 0.5642
_KFRE_CENTER_MALE = 0.5642
_KFRE_COEF_EGFR = -0.5567       # per eGFR/5 minus 7.222
_KFRE_CENTER_EGFR = 7.222
_KFRE_COEF_LNACR = 0.4510       # ln(UACR mg/g) minus 5.137
_KFRE_CENTER_LNACR = 5.137
#: baseline survival by horizon and cohort calibration
_KFRE_S0 = {
    (2, True): 0.9751, (5, True): 0.9240,           # North American
    (2, False): 0.9832, (5, False): 0.9365,         # non-NA recalibration
}


def ckd_epi_2021(
    creatinine_mg_dl: float,
    age_years: float,
    is_female: bool,
) -> float:
    """Return CKD-EPI 2021 creatinine-only eGFR (mL/min/1.73m^2).

    ``eGFR = 142 * min(Scr/k,1)^a * max(Scr/k,1)^(-1.200) *
    0.9938**age * (1.012 if female)`` with k = 0.7 (F) / 0.9 (M),
    alpha = -0.241 (F) / -0.302 (M).
    """
    if creatinine_mg_dl <= 0.0:
        raise ValueError("creatinine_mg_dl must be > 0")
    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    ratio = creatinine_mg_dl / kappa
    term_low = min(ratio, 1.0) ** alpha
    term_high = max(ratio, 1.0) ** -1.200
    egfr = (
        142.0 * term_low * term_high
        * 0.9938 ** age_years
        * (1.012 if is_female else 1.0)
    )
    return max(egfr, 1.0)


def inverse_ckd_epi(
    target_egfr: float,
    age_years: float,
    is_female: bool,
) -> float:
    """Return the creatinine consistent with *target_egfr*.

    Bisection on the monotone-decreasing CKD-EPI curve over
    [0.15, 40] mg/dL; accurate to <0.05 mg/dL.
    """
    if target_egfr <= 0.0:
        raise ValueError("target_egfr must be > 0")
    lo, hi = 0.15, 40.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if ckd_epi_2021(mid, age_years, is_female) > target_egfr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ============================================================================
# Model
# ============================================================================

@dataclass(slots=True)
class _AkiEpisode:
    """Active acute kidney injury episode."""

    loss_fraction: float
    recovery_fraction: float
    remaining_injury_h: float
    recovery_tau_h: float
    in_recovery: bool = False


class RenalFunctionModel:
    """Hourly renal-function model with CKD progression and AKI.

    The *functional* GFR follows the two-slope decomposition::

        d(eGFR)/yr = chronic_slope + d(acute_offset)/dt + AKI terms

    where ``chronic_slope`` combines an age-related component, fixed
    comorbidity components, and an albuminuria-driven pathological
    component that drug therapy can decelerate.  Serum creatinine then
    equilibrates toward the CKD-EPI value implied by current effective
    GFR with a clearance-dependent time constant, producing the
    characteristic lag between true GFR change and observed labs.
    """

    def __init__(
        self,
        age_years: float = 55.0,
        is_female: bool = False,
        diabetes: bool = False,
        hypertension: bool = False,
        weight_kg: float = 70.0,
        initial_egfr: float | None = None,
        initial_uacr_mg_g: float = 15.0,
    ) -> None:
        if age_years <= 0.0:
            raise ValueError("age_years must be > 0")
        if weight_kg <= 0.0:
            raise ValueError("weight_kg must be > 0")
        self.age_years = age_years
        self.is_female = is_female
        self.diabetes = diabetes
        self.hypertension = hypertension
        self.weight_kg = weight_kg

        self._baseline_egfr = (
            initial_egfr
            if initial_egfr is not None
            else max(ckd_epi_2021(0.9 if not is_female else 0.8, age_years, is_female), 60.0)
        )
        #: true filtering capacity (mL/min/1.73m^2)
        self.functional_egfr = self._baseline_egfr
        #: persistent hemodynamic offset from RAAS/SGLT2i initiation
        self.acute_offset = 0.0
        self.serum_creatinine = inverse_ckd_epi(
            self.functional_egfr, age_years, is_female
        )
        self.uacr_mg_g = min(max(initial_uacr_mg_g, 1.0), 20_000.0)

        # drug channels (toggled by start_*/stop_* methods)
        self.sglt2i_active = False
        self.raas_blockade_active = False
        self.nsaid_active = False
        self.loop_diuretic_active = False

        self._aki: _AkiEpisode | None = None
        #: pending permanent-GFR loss applied gradually post-AKI
        self._aki_recovery_pool = 0.0
        #: creatinine distribution volume (~total body water)
        self._creatinine_vd_l = 0.6 * weight_kg

    # ------------------------------------------------------------------
    # Intervention channels
    # ------------------------------------------------------------------

    def start_sglt2i(self) -> None:
        """Begin an SGLT2 inhibitor (chronic slope ~50% better + dip)."""
        self.sglt2i_active = True

    def stop_sglt2i(self) -> None:
        """Discontinue the SGLT2 inhibitor."""
        self.sglt2i_active = False

    def start_raas_blockade(self) -> None:
        """Begin ACEi/ARB/MRA therapy (slope benefit + hemodynamic dip)."""
        self.raas_blockade_active = True

    def stop_raas_blockade(self) -> None:
        """Discontinue RAAS blockade."""
        self.raas_blockade_active = False

    def set_nsaid(self, active: bool) -> None:
        """Set NSAID exposure (afferent arteriole constriction)."""
        self.nsaid_active = active

    def set_loop_diuretic(self, active: bool) -> None:
        """Set loop-diuretic exposure (completes the triple whammy)."""
        self.loop_diuretic_active = active

    def induce_aki(
        self,
        fractional_loss: float,
        recovery_fraction: float = 0.7,
        injury_duration_h: float = 72.0,
        recovery_tau_h: float = 96.0,
    ) -> None:
        """Trigger an AKI episode.

        Args:
            fractional_loss: peak fraction of functional GFR lost
                during the injury window in [0.01, 0.95].
            recovery_fraction: portion of the lost function that
                returns after the insult clears; the rest is permanent.
            injury_duration_h: hours the full insult persists.
            recovery_tau_h: recovery time constant afterwards.
        """
        if not 0.01 <= fractional_loss <= 0.95:
            raise ValueError("fractional_loss must be within [0.01, 0.95]")
        if not 0.0 <= recovery_fraction <= 1.0:
            raise ValueError("recovery_fraction must be within [0, 1]")
        self._aki = _AkiEpisode(
            loss_fraction=fractional_loss,
            recovery_fraction=recovery_fraction,
            remaining_injury_h=injury_duration_h,
            recovery_tau_h=max(recovery_tau_h, 1.0),
        )

    # ------------------------------------------------------------------
    # Slope composition
    # ------------------------------------------------------------------

    def _albuminuria_acceleration(self) -> float:
        """Pathological slope component driven by UACR (mL/min/yr).

        Zero at UACR <= 30 (A1), full magnitude at 300 (A3), linear in
        log10-UACR between; calibrated against trial placebo arms
        (CREDENCE placebo -4.71, FIDELIO placebo -3.97 mL/min/yr).
        """
        uacr = max(self.uacr_mg_g, 1.0)
        frac = (math.log10(uacr) - math.log10(30.0)) / math.log10(10.0)
        frac = min(max(frac, 0.0), 1.3)
        return 4.0 * frac

    def chronic_slope_ml_per_year(self) -> float:
        """Current net chronic eGFR slope (negative = declining)."""
        base_age = -0.9 if self.age_years >= 30 else -0.3
        comorbidity = (0.5 if self.diabetes else 0.0) + (
            0.3 if self.hypertension else 0.0
        )
        benefit = (1.0 - 0.50 * self.sglt2i_active) * (
            1.0 - 0.30 * self.raas_blockade_active
        )
        patho = self._albuminuria_acceleration() * benefit
        return base_age - comorbidity - patho

    def _nsaid_penalty_fraction(self) -> float:
        """Functional-GFR multiplier loss from prostaglandin blockade."""
        penalty = 0.15 if self.nsaid_active else 0.0
        if self.nsaid_active and self.raas_blockade_active:
            penalty += 0.08
        if self.nsaid_active and self.raas_blockade_active and self.loop_diuretic_active:
            penalty += 0.12  # triple whammy total ~35%
        return penalty

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def step(self, dt_h: float) -> float:
        """Advance the kidney *dt_h* hours; returns reported eGFR.

        Reported eGFR is recomputed from serum creatinine via
        CKD-EPI 2021, so it lags :attr:`functional_egfr` during rapid
        changes (AKI onset/recovery) exactly as clinical staging does.
        """
        if dt_h < 0.0:
            raise ValueError(f"dt_h must be >= 0, got {dt_h}")
        reported = ckd_epi_2021(
            self.serum_creatinine, self.age_years, self.is_female
        )
        remaining = dt_h
        while remaining > 1e-12:
            h = min(remaining, 0.5)

            years = h / _HOURS_PER_YEAR
            self.functional_egfr += self.chronic_slope_ml_per_year() * years

            dip_target = -(2.5 * self.raas_blockade_active + 2.0 * self.sglt2i_active)
            tau_dip = 336.0
            d_offset = (dip_target - self.acute_offset) / tau_dip
            self.acute_offset += d_offset * h

            aki_effect = 0.0
            if self._aki is not None:
                episode = self._aki
                baseline = self._baseline_egfr
                if not episode.in_recovery:
                    if episode.remaining_injury_h > 0.0:
                        burn = min(h, episode.remaining_injury_h)
                        episode.remaining_injury_h -= burn
                        aki_effect = -episode.loss_fraction * baseline
                    else:
                        # injury window over: the temporary loss ends
                        # here; bank the permanent fraction into a
                        # pending pool applied gradually so observed
                        # function recovers along a smooth exponential
                        episode.in_recovery = True
                        self._aki_recovery_pool = (
                            episode.loss_fraction
                            * (1.0 - episode.recovery_fraction)
                            * baseline
                        )
                elif self._aki_recovery_pool > 1e-3:
                    transfer = min(
                        self._aki_recovery_pool,
                        self._aki_recovery_pool * h / episode.recovery_tau_h,
                    )
                    self._aki_recovery_pool -= transfer
                    self.functional_egfr -= transfer
                else:
                    self._aki_recovery_pool = 0.0
                    self._aki = None

            nsaid_mult = 1.0 - self._nsaid_penalty_fraction()
            effective_egfr = max(
                (self.functional_egfr + self.acute_offset + aki_effect) * nsaid_mult,
                1.5,
            )
            self.functional_egfr = min(max(self.functional_egfr, 1.0), 160.0)

            target_scr = inverse_ckd_epi(
                effective_egfr, self.age_years, self.is_female
            )
            clearance_l_h = max(effective_egfr, 3.0) * 0.06
            tau_scr = 0.693 * self._creatinine_vd_l / clearance_l_h
            self.serum_creatinine += (target_scr - self.serum_creatinine) * (
                h / tau_scr
            )

            growth_yr = 0.02 + (0.15 if self.diabetes else 0.0) + (
                0.08 if self.hypertension else 0.0
            )
            reduction_yr = (
                0.35 * self.raas_blockade_active + 0.30 * self.sglt2i_active
            )
            d_ln_uacr = (growth_yr - reduction_yr) * h / _HOURS_PER_YEAR
            self.uacr_mg_g = min(
                max(self.uacr_mg_g * math.exp(d_ln_uacr), 1.0), 20_000.0
            )

            reported = ckd_epi_2021(
                self.serum_creatinine, self.age_years, self.is_female
            )
            remaining -= h

        return reported

    # ------------------------------------------------------------------
    # Clinical outputs
    # ------------------------------------------------------------------

    def reported_egfr(self) -> float:
        """eGFR as a clinician would compute it (CKD-EPI on creatinine)."""
        return ckd_epi_2021(self.serum_creatinine, self.age_years, self.is_female)

    def kdigo_g_stage(self) -> str:
        """KDIGO G category of the *reported* eGFR (``"G1"``...``"G5"``)."""
        egfr = self.reported_egfr()
        if egfr >= 90.0:
            return "G1"
        if egfr >= 60.0:
            return "G2"
        if egfr >= 45.0:
            return "G3a"
        if egfr >= 30.0:
            return "G3b"
        if egfr >= 15.0:
            return "G4"
        return "G5"

    def kdigo_a_category(self) -> str:
        """KDIGO albuminuria category from UACR (``"A1"``, ``"A2"``, ``"A3"``)."""
        if self.uacr_mg_g < 30.0:
            return "A1"
        if self.uacr_mg_g < 300.0:
            return "A2"
        return "A3"

    def to_disease_stage(self) -> DiseaseStage:
        """Map onto the shared doc/28 severity ladder."""
        stage = self.kdigo_g_stage()
        if stage in ("G1", "G2"):
            return DiseaseStage.PRECLINICAL
        if stage == "G3a":
            return DiseaseStage.MILD
        if stage == "G3b":
            return DiseaseStage.MODERATE
        if stage == "G4":
            return DiseaseStage.SEVERE
        return DiseaseStage.CRITICAL

    def time_to_krt_years(self) -> float:
        """Linear projection to kidney replacement therapy (eGFR 15).

        Returns ``math.inf`` when the chronic slope is non-negative or
        eGFR is already below the threshold.
        """
        egfr = self.reported_egfr()
        slope = self.chronic_slope_ml_per_year()
        if egfr <= 15.0 or slope >= -0.01:
            return math.inf
        return (egfr - 15.0) / -slope

    def kfre_risk(self, years: int = 5, north_american: bool = False) -> float:
        """4-variable KFRE probability of treated kidney failure.

        Args:
            years: prediction horizon, 2 or 5.
            north_american: use original North American calibration;
                default is the 2016 multinational recalibration.

        Validated for adults with eGFR < 60; values outside that range
        are extrapolation and callers should treat them qualitatively.
        """
        if years not in (2, 5):
            raise ValueError("years must be 2 or 5")
        egfr = min(max(self.reported_egfr(), 2.0), 120.0)
        acr = max(self.uacr_mg_g, 5.0)
        linear_predictor = (
            _KFRE_COEF_AGE * (self.age_years / 10.0 - _KFRE_CENTER_AGE)
            + _KFRE_COEF_MALE * ((0.0 if self.is_female else 1.0) - _KFRE_CENTER_MALE)
            + _KFRE_COEF_EGFR * (egfr / 5.0 - _KFRE_CENTER_EGFR)
            + _KFRE_COEF_LNACR * (math.log(acr) - _KFRE_CENTER_LNACR)
        )
        s0 = _KFRE_S0[(years, north_american)]
        return 1.0 - s0 ** math.exp(linear_predictor)

    def ckd_heatmap_cell(self) -> str:
        """KDIGO risk-heatmap cell label (``"G4·A3"`` style)."""
        return f"{self.kdigo_g_stage()}·{self.kdigo_a_category()}"

    def lab_values(self) -> dict[str, float]:
        """Snapshot formatted for ClinicalLabs-style integration."""
        return {
            "egfr_ml_min_1_73m2": self.reported_egfr(),
            "functional_egfr_ml_min_1_73m2": self.functional_egfr,
            "creatinine_mg_dl": self.serum_creatinine,
            "uacr_mg_g": self.uacr_mg_g,
            "chronic_slope_ml_min_per_year": self.chronic_slope_ml_per_year(),
            "kfre_5y_risk": self.kfre_risk(years=5),
        }


def create_renal_model(
    age_years: float = 55.0,
    is_female: bool = False,
    diabetes: bool = False,
    hypertension: bool = False,
    initial_egfr: float | None = None,
    initial_uacr_mg_g: float = 15.0,
) -> RenalFunctionModel:
    """Build a :class:`RenalFunctionModel` for a virtual patient."""
    return RenalFunctionModel(
        age_years=age_years,
        is_female=is_female,
        diabetes=diabetes,
        hypertension=hypertension,
        initial_egfr=initial_egfr,
        initial_uacr_mg_g=initial_uacr_mg_g,
    )
