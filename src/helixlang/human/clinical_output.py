"""Clinical laboratory values and vital-signs dynamics (doc/28).

Literature-anchored models for every standard clinical analyte, hepatic
and renal injury kinetics, bone-marrow suppression, haemodynamic and
vital-signs dynamics with drug and disease modifiers.

References:
    - CKD-EPI 2021 (Levey et al. 2021)
    - Hy's Law (FDA Guidance 2009)
    - Hyaluronic-acid / MELD scoring
    - Guyton & Hall 2016 (haemodynamics)
    - Bateman / Derendorf PD models
"""
from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ClinicalLabs",
    "ClinicalLabModel",
    "VitalSigns",
    "VitalsModel",
    "REFERENCE_RANGES",
]

# ============================================================================
# Reference ranges (healthy adult, male unless noted)
# ============================================================================

REFERENCE_RANGES: dict[str, dict[str, Any]] = {
    "alt_u_per_l": {"low": 7.0, "high": 56.0, "unit": "U/L"},
    "ast_u_per_l": {"low": 10.0, "high": 40.0, "unit": "U/L"},
    "alp_u_per_l": {"low": 44.0, "high": 147.0, "unit": "U/L"},
    "ggt_u_per_l": {"low": 8.0, "high": 61.0, "unit": "U/L"},
    "bilirubin_total_mg_per_dl": {"low": 0.1, "high": 1.2, "unit": "mg/dL"},
    "bilirubin_direct_mg_per_dl": {"low": 0.0, "high": 0.3, "unit": "mg/dL"},
    "albumin_g_per_dl": {"low": 3.5, "high": 5.5, "unit": "g/dL"},
    "inr": {"low": 0.8, "high": 1.2, "unit": ""},
    "creatinine_mg_per_dl": {"low": 0.7, "high": 1.3, "unit": "mg/dL"},
    "bun_mg_per_dl": {"low": 7.0, "high": 20.0, "unit": "mg/dL"},
    "egfr_ml_per_min": {"low": 60.0, "high": 150.0, "unit": "mL/min"},
    "cystatin_c_mg_per_l": {"low": 0.6, "high": 1.0, "unit": "mg/L"},
    "wbc_per_ul": {"low": 4500.0, "high": 11000.0, "unit": "/uL"},
    "rbc_million_per_ul": {"low": 4.5, "high": 5.9, "unit": "10^6/uL"},
    "hemoglobin_g_per_dl": {"low": 13.5, "high": 17.5, "unit": "g/dL"},
    "hematocrit_pct": {"low": 38.3, "high": 48.6, "unit": "%"},
    "platelets_per_ul": {"low": 150000.0, "high": 400000.0, "unit": "/uL"},
    "mcv_fl": {"low": 80.0, "high": 100.0, "unit": "fL"},
    "glucose_mg_per_dl": {"low": 70.0, "high": 100.0, "unit": "mg/dL"},
    "hba1c_pct": {"low": 4.0, "high": 5.7, "unit": "%"},
    "sodium_meq_per_l": {"low": 136.0, "high": 145.0, "unit": "mEq/L"},
    "potassium_meq_per_l": {"low": 3.5, "high": 5.0, "unit": "mEq/L"},
    "chloride_meq_per_l": {"low": 98.0, "high": 106.0, "unit": "mEq/L"},
    "bicarbonate_meq_per_l": {"low": 22.0, "high": 29.0, "unit": "mEq/L"},
    "calcium_mg_per_dl": {"low": 8.5, "high": 10.5, "unit": "mg/dL"},
    "phosphate_mg_per_dl": {"low": 2.5, "high": 4.5, "unit": "mg/dL"},
    "total_cholesterol_mg_per_dl": {"low": 0.0, "high": 200.0, "unit": "mg/dL"},
    "ldl_mg_per_dl": {"low": 0.0, "high": 100.0, "unit": "mg/dL"},
    "hdl_mg_per_dl": {"low": 40.0, "high": 999.0, "unit": "mg/dL"},
    "triglycerides_mg_per_dl": {"low": 0.0, "high": 150.0, "unit": "mg/dL"},
    "crp_mg_per_l": {"low": 0.0, "high": 3.0, "unit": "mg/L"},
    "esr_mm_per_h": {"low": 0.0, "high": 15.0, "unit": "mm/h"},
    "lactate_mmol_per_l": {"low": 0.5, "high": 2.2, "unit": "mmol/L"},
}

# Half-lives for lab-value recovery toward baseline (hours)
_RECOVERY_HALF_LIFE_H: dict[str, float] = {
    "alt_u_per_l": 47.0,
    "ast_u_per_l": 35.0,
    "creatinine_mg_per_dl": 168.0,
    "wbc_per_ul": 336.0,
    "hemoglobin_g_per_dl": 672.0,
    "platelets_per_ul": 240.0,
    "crp_mg_per_l": 19.0,
    "bilirubin_total_mg_per_dl": 72.0,
    "inr": 48.0,
}

# Hepatotoxic drugs: {drug_key: (alt_rate_per_hour, ast_rate_per_hour)}
# ALT rises ~47 h half-life; maximum ~20x ULN for severe DILI
_HEPATOTOXIC_DRUGS: dict[str, tuple[float, float]] = {
    "ibuprofen": (0.5, 0.4),
    "metformin": (0.3, 0.2),
    "cisplatin": (1.5, 1.2),
    "tamoxifen": (0.8, 0.6),
    "methotrexate": (0.9, 0.7),
}

# Nephrotoxic drugs: {drug_key: creatinine_rise_rate_mg_per_dl_per_day}
_NEPHROTOXIC_DRUGS: dict[str, float] = {
    "cisplatin": 0.4,
    "ibuprofen": 0.15,
    "metformin": 0.05,
    "methotrexate": 0.25,
}

# Myelosuppressive drugs: {drug_key: wbc_suppression_fraction_per_day}
# Fraction of circulating WBC suppressed per day of active therapy
_MYELOSUPPRESSIVE_DRUGS: dict[str, float] = {
    "cisplatin": 0.15,
    "tamoxifen": 0.03,
    "imatinib": 0.05,
    "methotrexate": 0.12,
}

#: Reference unbound concentration (µM) at which each toxicity pathway's
#: rate scaling equals 1.0.  Drug concentrations arrive from the PBPK layer
#: already MW-converted to µM (mg/L × 1000 / MW_da), so these thresholds
#: are directly comparable across molecules of any molecular weight.
_POTENCY_REFERENCE_UM: dict[str, float] = {
    "hepatotoxicity": 50.0,
    "nephrotoxicity": 40.0,
    "myelosuppression": 30.0,
}


def _potency_scale(conc_um: float, reference_um: float, max_scale: float) -> float:
    """Normalized potency factor for toxicity-rate scaling.

    Linear in the µM concentration relative to *reference_um* (unity at the
    reference) and saturating at *max_scale*, mirroring a low-dose-linear
    Emax relation.
    """
    if conc_um <= 0.0 or reference_um <= 0.0:
        return 0.0
    return min(conc_um / reference_um, max_scale)


# ============================================================================
# Clinical lab snapshot
# ============================================================================


@dataclass
class ClinicalLabs:
    """Complete clinical laboratory snapshot with healthy-adult defaults.

    Every field represents a standard clinical analyte.  Values outside
    reference ranges trigger pathological behaviour in downstream models
    (progression staging, vital-signs feedback, toxicity checks).
    """

    # --- Hepatic ---
    alt_u_per_l: float = 25.0
    ast_u_per_l: float = 25.0
    alp_u_per_l: float = 70.0
    ggt_u_per_l: float = 40.0
    bilirubin_total_mg_per_dl: float = 0.7
    bilirubin_direct_mg_per_dl: float = 0.1
    albumin_g_per_dl: float = 4.5
    inr: float = 1.0

    # --- Renal ---
    creatinine_mg_per_dl: float = 1.0
    bun_mg_per_dl: float = 15.0
    egfr_ml_per_min: float = 120.0
    cystatin_c_mg_per_l: float = 0.8

    # --- Haematologic ---
    wbc_per_ul: float = 7000.0
    rbc_million_per_ul: float = 5.0
    hemoglobin_g_per_dl: float = 15.0
    hematocrit_pct: float = 45.0
    platelets_per_ul: float = 250000.0
    mcv_fl: float = 90.0

    # --- Metabolic ---
    glucose_mg_per_dl: float = 90.0
    hba1c_pct: float = 5.5
    sodium_meq_per_l: float = 140.0
    potassium_meq_per_l: float = 4.0
    chloride_meq_per_l: float = 102.0
    bicarbonate_meq_per_l: float = 24.0
    calcium_mg_per_dl: float = 9.5
    phosphate_mg_per_dl: float = 3.5
    lactate_mmol_per_l: float = 1.2

    # --- Lipid ---
    total_cholesterol_mg_per_dl: float = 200.0
    ldl_mg_per_dl: float = 100.0
    hdl_mg_per_dl: float = 50.0
    triglycerides_mg_per_dl: float = 150.0

    # --- Inflammatory ---
    crp_mg_per_l: float = 1.0
    esr_mm_per_h: float = 10.0

    # --- Demographic context (needed for eGFR etc.) ---
    age_years: float = 30.0
    sex: str = "male"

    def is_abnormal(self, field_name: str) -> bool:
        """Return True if *field_name* is outside the reference range."""
        rng = REFERENCE_RANGES.get(field_name)
        if rng is None:
            return False
        val = getattr(self, field_name, None)
        if val is None:
            return False
        return bool(val < rng["low"] or val > rng["high"])

    def abnormal_count(self) -> int:
        """Count how many analytes are outside reference ranges."""
        return sum(
            1 for name in REFERENCE_RANGES if self.is_abnormal(name)
        )

    def to_progression_labs(self) -> dict[str, Any]:
        """Export fields compatible with disease_progression.ClinicalLabs."""
        return {
            "age_years": self.age_years,
            "egfr_ml_min_1_73m2": self.egfr_ml_per_min,
            "creatinine_mg_dl": self.creatinine_mg_per_dl,
            "alt_u_l": self.alt_u_per_l,
            "ast_u_l": self.ast_u_per_l,
            "total_bilirubin_mg_dl": self.bilirubin_total_mg_per_dl,
            "albumin_g_dl": self.albumin_g_per_dl,
            "inr": self.inr,
            "platelets_per_ul": self.platelets_per_ul,
            "hba1c_percent": self.hba1c_pct,
            "tumor_marker_ng_ml": 0.5,
            "ascites_grade": 0,
            "encephalopathy_grade": 0,
        }

    def snapshot(self) -> ClinicalLabs:
        """Return a deep copy for historical recording."""
        return copy(self)


# ============================================================================
# Clinical lab dynamic model
# ============================================================================


class ClinicalLabModel:
    """Computes clinical lab values over time given physiology, disease and drugs.

    The model starts from a *baseline* snapshot (derived from physiology +
    disease state) and advances analyte-by-analyte at each time step,
    applying:
      - Drug-induced hepatotoxicity / nephrotoxicity / myelosuppression
      - Disease-driven derangements (diabetes → glucose, Gaucher → ALP)
      - First-order exponential return toward baseline for recovery
    """

    def __init__(
        self,
        baseline: ClinicalLabs,
        physiology: Any | None = None,
    ) -> None:
        self.baseline = baseline.snapshot()
        self.current = baseline.snapshot()
        self.physiology = physiology
        self._hours_since_start: float = 0.0
        self._drug_smiles: dict[str, str] = {}
        self._toxicity_cache: dict[str, dict[str, float]] = {}

    def register_drug_smiles(self, drug_key: str, smiles: str) -> None:
        """Register a SMILES string for a drug to enable structure-based toxicity prediction."""
        self._drug_smiles[drug_key] = smiles

    def _get_structure_toxicity(self, drug_key: str) -> dict[str, float] | None:
        """Get toxicity scores from SMILES-based prediction for unknown drugs."""
        smiles = self._drug_smiles.get(drug_key)
        if not smiles:
            return None
        if drug_key in self._toxicity_cache:
            return self._toxicity_cache[drug_key]
        try:
            from helixlang.human.molecular_toxicity import MolecularToxicityPredictor
            predictor = MolecularToxicityPredictor()
            profile = predictor.predict_toxicity(smiles)
            if profile.confidence <= 0.0:
                return None
            result = {
                "hepatotoxicity": profile.hepatotoxicity_score,
                "nephrotoxicity": profile.nephrotoxicity_score,
                "myelosuppression": profile.myelosuppression_score,
                "cardiotoxicity": profile.cardiotoxicity_score,
            }
            self._toxicity_cache[drug_key] = result
            return result
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Baseline derivation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_baseline_from_physiology(
        physiology: Any,
        disease: Any | None = None,
    ) -> ClinicalLabs:
        """Derive baseline labs from *physiology* + optional *disease*."""
        labs = ClinicalLabs()
        labs.age_years = physiology.age_years
        labs.sex = physiology.sex

        # --- Body-size adjustments ---
        weight = getattr(physiology, "body_weight_kg", 70.0)
        labs.hematocrit_pct = physiology.hematocrit * 100.0
        labs.hemoglobin_g_per_dl = physiology.hematocrit * 33.0
        labs.albumin_g_per_dl = physiology.albumin_g_per_dL

        # --- eGFR from creatinine (CKD-EPI 2021 simplified) ---
        # Baseline creatinine for a 70 kg male ≈ 1.0; adjust for muscle mass
        muscle_factor = weight / 70.0
        if physiology.sex == "female":
            muscle_factor *= 0.85
        labs.creatinine_mg_per_dl = 1.0 * muscle_factor
        labs.egfr_ml_per_min = _ckd_epi_2021(
            labs.creatinine_mg_per_dl, physiology.age_years, physiology.sex,
        )

        # --- CYP-dependent adjustments ---
        cyp = getattr(physiology, "cytochrome_p450_activity", {})
        # Low CYP1A2 → higher ALT baseline (less efficient detox)
        if cyp.get("CYP1A2", 0.2) < 0.1:
            labs.alt_u_per_l *= 1.3

        # --- Disease modifiers ---
        if disease is not None:
            labs = _apply_disease_to_labs(labs, disease, physiology)

        return labs

    # ------------------------------------------------------------------
    # Time advance
    # ------------------------------------------------------------------

    def update(
        self,
        dt_h: float,
        drug_concentrations: dict[str, float] | None = None,
        disease_severity: float = 0.0,
        disease_name: str = "",
    ) -> ClinicalLabs:
        """Advance all analytes by *dt_h* hours and return updated snapshot.

        ``drug_concentrations`` values are expected in µM (MW-converted by
        the PBPK layer); toxicity scaling uses the normalized potency
        factors in :data:`_POTENCY_REFERENCE_UM`.
        """
        drug_conc = drug_concentrations or {}
        self._hours_since_start += dt_h

        # --- Hepatotoxicity ---
        self._apply_hepatotoxicity(dt_h, drug_conc)

        # --- Nephrotoxicity ---
        self._apply_nephrotoxicity(dt_h, drug_conc)

        # --- Myelosuppression ---
        self._apply_myelosuppression(dt_h, drug_conc)

        # --- Disease-driven analytes ---
        self._apply_disease_analytes(dt_h, disease_severity, drug_conc, disease_name)

        # --- Recovery toward baseline ---
        self._apply_recovery(dt_h)

        # --- Derived values ---
        self.current.egfr_ml_per_min = _ckd_epi_2021(
            self.current.creatinine_mg_per_dl,
            self.current.age_years,
            self.current.sex,
        )
        self.current.hematocrit_pct = self.current.hemoglobin_g_per_dl / 33.0 * 100.0
        self.current.bun_mg_per_dl = max(
            5.0,
            self.current.creatinine_mg_per_dl * 15.0,
        )

        return self.current.snapshot()

    def get_current(self) -> ClinicalLabs:
        return self.current.snapshot()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_hepatotoxicity(
        self, dt_h: float, drug_conc: dict[str, float],
    ) -> None:
        for drug_key, (alt_rate, ast_rate) in _HEPATOTOXIC_DRUGS.items():
            conc = drug_conc.get(drug_key, 0.0)
            if conc <= 0.0:
                continue
            scale = _potency_scale(
                conc, _POTENCY_REFERENCE_UM["hepatotoxicity"], 3.0,
            )
            self.current.alt_u_per_l = min(
                1500.0,
                self.current.alt_u_per_l + alt_rate * scale * dt_h,
            )
            self.current.ast_u_per_l = min(
                1500.0,
                self.current.ast_u_per_l + ast_rate * scale * dt_h,
            )
            if (
                self.current.alt_u_per_l > 3.0 * 56.0
                and self.current.bilirubin_total_mg_per_dl > 2.0 * 1.2
            ):
                self.current.bilirubin_total_mg_per_dl = min(
                    15.0,
                    self.current.bilirubin_total_mg_per_dl + 0.05 * dt_h,
                )
        # SMILES-driven toxicity fallback for unknown drugs
        for drug_key, conc in drug_conc.items():
            if conc <= 0.0 or drug_key in _HEPATOTOXIC_DRUGS:
                continue
            tox = self._get_structure_toxicity(drug_key)
            if tox is not None and tox["hepatotoxicity"] > 0.01:
                rate = tox["hepatotoxicity"] * 1.5
                scale = _potency_scale(conc, _POTENCY_REFERENCE_UM["hepatotoxicity"], 3.0)
                self.current.alt_u_per_l = min(
                    1500.0, self.current.alt_u_per_l + rate * scale * dt_h,
                )
                self.current.ast_u_per_l = min(
                    1500.0, self.current.ast_u_per_l + rate * 0.7 * scale * dt_h,
                )

    def _apply_nephrotoxicity(
        self, dt_h: float, drug_conc: dict[str, float],
    ) -> None:
        for drug_key, rate in _NEPHROTOXIC_DRUGS.items():
            conc = drug_conc.get(drug_key, 0.0)
            if conc <= 0.0:
                continue
            scale = _potency_scale(
                conc, _POTENCY_REFERENCE_UM["nephrotoxicity"], 3.0,
            )
            self.current.creatinine_mg_per_dl = min(
                15.0,
                self.current.creatinine_mg_per_dl + rate * scale * (dt_h / 24.0),
            )
        for drug_key, conc in drug_conc.items():
            if conc <= 0.0 or drug_key in _NEPHROTOXIC_DRUGS:
                continue
            tox = self._get_structure_toxicity(drug_key)
            if tox is not None and tox["nephrotoxicity"] > 0.01:
                rate = tox["nephrotoxicity"] * 0.5
                scale = _potency_scale(conc, _POTENCY_REFERENCE_UM["nephrotoxicity"], 3.0)
                self.current.creatinine_mg_per_dl = min(
                    15.0, self.current.creatinine_mg_per_dl + rate * scale * (dt_h / 24.0),
                )

    def _apply_myelosuppression(
        self, dt_h: float, drug_conc: dict[str, float],
    ) -> None:
        for drug_key, frac_per_day in _MYELOSUPPRESSIVE_DRUGS.items():
            conc = drug_conc.get(drug_key, 0.0)
            if conc <= 0.0:
                continue
            scale = _potency_scale(
                conc, _POTENCY_REFERENCE_UM["myelosuppression"], 2.0,
            )
            suppression = frac_per_day * scale * (dt_h / 24.0)
            self.current.wbc_per_ul = max(
                500.0,
                self.current.wbc_per_ul * (1.0 - suppression),
            )
            self.current.platelets_per_ul = max(
                10000.0,
                self.current.platelets_per_ul * (1.0 - suppression * 0.7),
            )
            self.current.hemoglobin_g_per_dl = max(
                5.0,
                self.current.hemoglobin_g_per_dl * (1.0 - suppression * 0.3),
            )
        for drug_key, conc in drug_conc.items():
            if conc <= 0.0 or drug_key in _MYELOSUPPRESSIVE_DRUGS:
                continue
            tox = self._get_structure_toxicity(drug_key)
            if tox is not None and tox["myelosuppression"] > 0.01:
                frac = tox["myelosuppression"] * 0.15
                scale = _potency_scale(conc, _POTENCY_REFERENCE_UM["myelosuppression"], 2.0)
                suppression = frac * scale * (dt_h / 24.0)
                self.current.wbc_per_ul = max(500.0, self.current.wbc_per_ul * (1.0 - suppression))
                self.current.platelets_per_ul = max(
                    10000.0, self.current.platelets_per_ul * (1.0 - suppression * 0.7),
                )
                self.current.hemoglobin_g_per_dl = max(
                    5.0, self.current.hemoglobin_g_per_dl * (1.0 - suppression * 0.3),
                )

    def _apply_disease_analytes(
        self,
        dt_h: float,
        severity: float,
        drug_conc: dict[str, float],
        disease_name: str = "",
    ) -> None:
        """Adjust analytes based on ongoing disease severity."""
        # Glucose rises with diabetes severity (severity 0-1 → glucose 90-300)
        disease_glucose = 90.0 + severity * 210.0
        self.current.glucose_mg_per_dl = 0.95 * self.current.glucose_mg_per_dl + 0.05 * disease_glucose

        # HbA1c slowly tracks glucose (τ ~ 120 days ≈ 2880 h)
        target_hba1c = 5.5 + severity * 4.5  # up to 10%
        tau_hba1c = 2880.0
        self.current.hba1c_pct += (target_hba1c - self.current.hba1c_pct) * (dt_h / tau_hba1c)

        # Inflammation (CRP) rises with severity
        target_crp = 1.0 + severity * 50.0
        self.current.crp_mg_per_l = 0.99 * self.current.crp_mg_per_l + 0.01 * target_crp

        # Cancer: lactate rises (Warburg), albumin drops
        target_lactate = 1.2 + severity * 8.0
        self.current.lactate_mmol_per_l = 0.995 * self.current.lactate_mmol_per_l + 0.005 * target_lactate
        target_albumin = max(1.5, 4.5 - severity * 3.0)
        self.current.albumin_g_per_dl = 0.999 * self.current.albumin_g_per_dl + 0.001 * target_albumin

        # --- Electrolyte dynamics ---
        self._apply_electrolyte_dynamics(dt_h, severity, drug_conc)

        # --- Lipid dynamics ---
        self._apply_lipid_dynamics(dt_h, severity)

        # --- Coagulation ---
        self._apply_coagulation(dt_h, severity, drug_conc)

    def _apply_electrolyte_dynamics(
        self,
        dt_h: float,
        severity: float,
        drug_conc: dict[str, float],
    ) -> None:
        """Dynamic electrolytes driven by disease, renal function, drugs."""
        egfr = self.current.egfr_ml_per_min

        # --- Sodium: dehydration from disease, cisplatin hyponatremia ---
        target_na = 140.0 - severity * 8.0  # dehydration
        cisplatin = drug_conc.get("cisplatin", 0.0)
        if cisplatin > 0.0:
            target_na -= 5.0 * min(cisplatin / 30.0, 1.0)
        tau_na = 48.0  # hours to equilibrate
        self.current.sodium_meq_per_l += (target_na - self.current.sodium_meq_per_l) * (dt_h / tau_na)

        # --- Potassium: renal function, cisplatin wasting ---
        target_k = 4.0
        if egfr < 60.0:
            target_k += (60.0 - egfr) * 0.02  # hyperkalemia, max ~5.2
        if cisplatin > 0.0:
            target_k -= 0.5 * min(cisplatin / 30.0, 1.0)  # hypokalemia from wasting
        target_k = max(2.5, min(7.0, target_k))
        tau_k = 24.0
        self.current.potassium_meq_per_l += (target_k - self.current.potassium_meq_per_l) * (dt_h / tau_k)

        # --- Calcium: albumin-adjusted ---
        target_ca = 9.5 + 0.8 * (4.5 - self.current.albumin_g_per_dl)
        tau_ca = 72.0
        self.current.calcium_mg_per_dl += (target_ca - self.current.calcium_mg_per_dl) * (dt_h / tau_ca)

        # --- Phosphate: CKD-driven hyperphosphatemia ---
        target_phos = 3.5
        if egfr < 60.0:
            target_phos += (60.0 - egfr) * 0.05  # up to ~6.5
        tau_phos = 96.0
        self.current.phosphate_mg_per_dl += (target_phos - self.current.phosphate_mg_per_dl) * (dt_h / tau_phos)

        # --- Chloride: tracks sodium/anion gap ---
        self.current.chloride_meq_per_l = self.current.sodium_meq_per_l - 140.0 + 102.0

        # --- Bicarbonate: from lactate + renal acidosis ---
        target_hco3 = 24.0 - max(0.0, self.current.lactate_mmol_per_l - 2.2) * 2.0
        if egfr < 30.0:
            target_hco3 -= 4.0  # renal acidosis
        target_hco3 = max(10.0, min(30.0, target_hco3))
        tau_hco3 = 48.0
        self.current.bicarbonate_meq_per_l += (target_hco3 - self.current.bicarbonate_meq_per_l) * (dt_h / tau_hco3)

    def _apply_lipid_dynamics(self, dt_h: float, severity: float) -> None:
        """Lipids shift slowly with disease (T2DM → dyslipidemia)."""
        tau_lipid = 2880.0  # ~120 days, slow turnover
        target_ldl = 100.0 + severity * 50.0
        target_hdl = max(30.0, 50.0 - severity * 15.0)
        target_tg = 150.0 + severity * 100.0
        self.current.ldl_mg_per_dl += (target_ldl - self.current.ldl_mg_per_dl) * (dt_h / tau_lipid)
        self.current.hdl_mg_per_dl += (target_hdl - self.current.hdl_mg_per_dl) * (dt_h / tau_lipid)
        self.current.triglycerides_mg_per_dl += (target_tg - self.current.triglycerides_mg_per_dl) * (dt_h / tau_lipid)
        self.current.total_cholesterol_mg_per_dl = (
            self.current.ldl_mg_per_dl + self.current.hdl_mg_per_dl
            + self.current.triglycerides_mg_per_dl / 5.0
        )

    def _apply_coagulation(
        self, dt_h: float, severity: float, drug_conc: dict[str, float],
    ) -> None:
        """INR from liver synthetic function + drug effects."""
        target_inr = 1.0
        # Liver disease → INR rises
        if severity > 0.3:
            target_inr += (severity - 0.3) * 3.0  # up to ~3.1 at severity=1.0
        # Ibuprofen: mild antiplatelet effect (slight INR rise for vulnerable)
        ibuprofen = drug_conc.get("ibuprofen", 0.0)
        if ibuprofen > 0.0:
            target_inr += 0.1 * min(ibuprofen / 40.0, 1.0)
        target_inr = min(5.0, max(0.8, target_inr))
        tau_inr = 72.0
        self.current.inr += (target_inr - self.current.inr) * (dt_h / tau_inr)

    def _apply_recovery(self, dt_h: float) -> None:
        """Exponential return of each analyte toward its baseline value."""
        for field_name, half_life in _RECOVERY_HALF_LIFE_H.items():
            current_val = getattr(self.current, field_name, None)
            baseline_val = getattr(self.baseline, field_name, None)
            if current_val is None or baseline_val is None:
                continue
            if half_life <= 0 or half_life >= 1e12:
                continue
            k = math.log(2.0) / half_life
            decay = math.exp(-k * dt_h)
            new_val = baseline_val + (current_val - baseline_val) * decay
            setattr(self.current, field_name, new_val)


# ============================================================================
# Vital signs
# ============================================================================


@dataclass
class VitalSigns:
    """Standard vital signs snapshot."""

    systolic_bp_mmhg: float = 120.0
    diastolic_bp_mmhg: float = 80.0
    heart_rate_bpm: float = 72.0
    respiratory_rate_per_min: float = 16.0
    temperature_c: float = 37.0
    spo2_pct: float = 98.0
    weight_kg: float = 70.0
    qt_interval_ms: float = 380.0
    qtc_ms: float = 400.0

    @property
    def map_mmhg(self) -> float:
        """Mean arterial pressure = 1/3·SBP + 2/3·DBP."""
        return self.systolic_bp_mmhg / 3.0 + 2.0 * self.diastolic_bp_mmhg / 3.0

    @property
    def pulse_pressure(self) -> float:
        return self.systolic_bp_mmhg - self.diastolic_bp_mmhg

    def snapshot(self) -> VitalSigns:
        return VitalSigns(
            systolic_bp_mmhg=self.systolic_bp_mmhg,
            diastolic_bp_mmhg=self.diastolic_bp_mmhg,
            heart_rate_bpm=self.heart_rate_bpm,
            respiratory_rate_per_min=self.respiratory_rate_per_min,
            temperature_c=self.temperature_c,
            spo2_pct=self.spo2_pct,
            weight_kg=self.weight_kg,
            qt_interval_ms=self.qt_interval_ms,
            qtc_ms=self.qtc_ms,
        )


# Drug effects on haemodynamics: {drug: (sbp_delta, dbp_delta, hr_delta, weight_delta)}
_VITAL_DRUG_EFFECTS: dict[str, tuple[float, float, float, float]] = {
    "metformin": (-3.0, -2.0, 0.0, 0.0),
    "ibuprofen": (4.0, 2.0, 1.0, 0.5),
    "cisplatin": (-5.0, -3.0, 3.0, -0.3),
    "tamoxifen": (0.0, 0.0, 4.0, 0.2),
    "imatinib": (-2.0, -1.0, 2.0, 1.0),
    "imiglucerase": (0.0, 0.0, 0.0, 0.0),
}


class VitalsModel:
    """Dynamic vital-signs model with drug and disease modifiers."""

    def __init__(
        self,
        base_vitals: VitalSigns,
        physiology: Any | None = None,
    ) -> None:
        self.baseline = base_vitals.snapshot()
        self.current = base_vitals.snapshot()
        self.physiology = physiology

    @staticmethod
    def create_from_physiology(physiology: Any) -> VitalsModel:
        """Create vitals model from a HumanPhysiology instance."""
        vs = VitalSigns(
            weight_kg=physiology.body_weight_kg,
        )
        # Adjust resting HR for age and fitness
        age = physiology.age_years
        vs.heart_rate_bpm = 72.0 + (age - 30.0) * 0.3

        # OBESITY → ↑BP, ↑HR
        bmi = getattr(physiology, "bmi", 24.2)
        if bmi > 30.0:
            excess = (bmi - 30.0) * 1.5
            vs.systolic_bp_mmhg += excess
            vs.diastolic_bp_mmhg += excess * 0.6
            vs.heart_rate_bpm += excess * 0.2

        return VitalsModel(base_vitals=vs, physiology=physiology)

    def update(
        self,
        dt_h: float,
        drug_concentrations: dict[str, float] | None = None,
        labs: ClinicalLabs | None = None,
        disease_severity: float = 0.0,
    ) -> VitalSigns:
        """Advance vitals by *dt_h* hours and return updated snapshot."""
        drug_conc = drug_concentrations or {}

        # --- Drug effects ---
        sbp_delta = 0.0
        dbp_delta = 0.0
        hr_delta = 0.0
        wt_delta = 0.0
        temp_delta = 0.0
        for drug_key, (s, d, h, w) in _VITAL_DRUG_EFFECTS.items():
            conc = drug_conc.get(drug_key, 0.0)
            if conc <= 0.0:
                continue
            # Scale effect by concentration (EC50 ~ 30 uM, max at 3x EC50)
            scale = min(conc / 30.0, 3.0) / 3.0
            sbp_delta += s * scale
            dbp_delta += d * scale
            hr_delta += h * scale
            wt_delta += w * scale * (dt_h / 24.0)

        # --- Disease effects ---
        if disease_severity > 0.3:
            # Fever from inflammation
            if labs is not None and labs.crp_mg_per_l > 10.0:
                temp_delta += min(2.0, (labs.crp_mg_per_l - 10.0) * 0.02)
            # Hypertension from renal disease / diabetes
            if labs is not None and labs.egfr_ml_per_min < 60.0:
                renal_htn = (60.0 - labs.egfr_ml_per_min) * 0.15
                sbp_delta += renal_htn
                dbp_delta += renal_htn * 0.5
            # Tachycardia from anaemia
            if labs is not None and labs.hemoglobin_g_per_dl < 10.0:
                hr_delta += (10.0 - labs.hemoglobin_g_per_dl) * 1.5
            # Weight loss from cancer / severe disease
            wt_delta -= disease_severity * 0.02 * (dt_h / 24.0)

        # --- Apply deltas ---
        self.current.systolic_bp_mmhg = max(
            60.0,
            min(220.0, self.baseline.systolic_bp_mmhg + sbp_delta),
        )
        self.current.diastolic_bp_mmhg = max(
            30.0,
            min(130.0, self.baseline.diastolic_bp_mmhg + dbp_delta),
        )
        self.current.heart_rate_bpm = max(
            40.0,
            min(180.0, self.baseline.heart_rate_bpm + hr_delta),
        )
        self.current.temperature_c = max(
            34.0,
            min(42.0, self.baseline.temperature_c + temp_delta),
        )
        self.current.weight_kg = max(
            30.0, self.current.weight_kg + wt_delta,
        )

        # --- Electrolyte-driven vital effects ---
        if labs is not None:
            # Severe hypokalemia → tachycardia
            if labs.potassium_meq_per_l < 3.0:
                hr_delta += (3.0 - labs.potassium_meq_per_l) * 8.0
            # Severe hyperkalemia → bradycardia
            if labs.potassium_meq_per_l > 6.0:
                hr_delta -= (labs.potassium_meq_per_l - 6.0) * 10.0
            # Severe hyponatremia → hypotension
            if labs.sodium_meq_per_l < 125.0:
                na_drop = (125.0 - labs.sodium_meq_per_l) * 0.5
                sbp_delta -= na_drop
                dbp_delta -= na_drop * 0.3

        # --- QTc prolongation ---
        # Bazett: QTc = QT / sqrt(RR)
        rr_s = 60.0 / max(self.current.heart_rate_bpm, 40.0)
        qt_drug_delta = 0.0
        _QT_DRUG_EFFECTS: dict[str, float] = {
            "tamoxifen": 8.0,
            "cisplatin": 5.0,
            "ibuprofen": 2.0,
            "imatinib": 3.0,
        }
        for drug_key, qt_delta in _QT_DRUG_EFFECTS.items():
            conc = drug_conc.get(drug_key, 0.0)
            if conc > 0.0:
                scale = min(conc / 30.0, 1.0)
                qt_drug_delta += qt_delta * scale
        # Hypocalcemia prolongs QT
        if labs is not None and labs.calcium_mg_per_dl < 7.0:
            qt_drug_delta += (7.0 - labs.calcium_mg_per_dl) * 5.0
        # Hypokalemia prolongs QT
        if labs is not None and labs.potassium_meq_per_l < 3.5:
            qt_drug_delta += (3.5 - labs.potassium_meq_per_l) * 10.0

        base_qt = 380.0
        self.current.qt_interval_ms = base_qt + qt_drug_delta
        self.current.qtc_ms = self.current.qt_interval_ms / (rr_s ** 0.5)

        # --- Dynamic SpO₂: anemia-driven ---
        if labs is not None:
            if labs.hemoglobin_g_per_dl < 8.0:
                self.current.spo2_pct = max(85.0, 98.0 - (8.0 - labs.hemoglobin_g_per_dl) * 3.0)
            else:
                self.current.spo2_pct = min(99.0, self.baseline.spo2_pct)

        # --- Dynamic respiratory rate: acid-base compensation ---
        if labs is not None:
            if labs.bicarbonate_meq_per_l < 18.0:
                # Metabolic acidosis → Kussmaul breathing
                self.current.respiratory_rate_per_min = min(
                    35.0, 16.0 + (18.0 - labs.bicarbonate_meq_per_l) * 1.0,
                )
            elif labs.bicarbonate_meq_per_l > 30.0:
                # Metabolic alkalosis → hypoventilation
                self.current.respiratory_rate_per_min = max(
                    10.0, 16.0 - (labs.bicarbonate_meq_per_l - 30.0) * 0.5,
                )
            else:
                self.current.respiratory_rate_per_min = 16.0

        return self.current.snapshot()

    def get_current(self) -> VitalSigns:
        return self.current.snapshot()


# ============================================================================
# Helpers
# ============================================================================


def _ckd_epi_2021(
    creatinine_mg_dl: float,
    age: float,
    sex: str,
) -> float:
    """CKD-EPI 2021 eGFR equation (race-free version).

    eGFR = 142 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^(-1.200) × 0.9938^age × (1.012 if female)

    κ = 0.9 (male), 0.7 (female); α = -0.302 (male), -0.241 (female)
    """
    kappa = 0.7 if sex == "female" else 0.9
    alpha = -0.241 if sex == "female" else -0.302
    sex_factor = 1.012 if sex == "female" else 1.0
    scr_kappa = creatinine_mg_dl / kappa
    egfr = (
        142.0
        * (min(scr_kappa, 1.0) ** alpha)
        * (max(scr_kappa, 1.0) ** (-1.200))
        * (0.9938 ** age)
        * sex_factor
    )
    return float(max(1.0, min(200.0, egfr)))


def _apply_disease_to_labs(
    labs: ClinicalLabs,
    disease: Any,
    physiology: Any,
) -> ClinicalLabs:
    """Modify baseline labs based on disease state."""
    name = getattr(disease, "name", "").lower()
    severity = getattr(disease, "severity", 0.0)

    if "gaucher" in name:
        # Elevated ALP, hepatomegaly effect, mild anemia
        labs.alp_u_per_l = 70.0 + severity * 300.0
        labs.alt_u_per_l = 25.0 + severity * 60.0
        labs.hemoglobin_g_per_dl = max(8.0, 15.0 - severity * 5.0)

    elif "pku" in name:
        # Hyperphenylalaninemia — not directly in standard labs, but
        # secondary metabolic acidosis effect
        labs.phenylalanine_mmol_per_l = 0.09 + severity * 2.31  # type: ignore[attr-defined]

    elif "diabetes" in name or "t2d" in name:
        labs.glucose_mg_per_dl = 90.0 + severity * 210.0
        labs.hba1c_pct = 5.5 + severity * 4.5

    elif "warburg" in name or "cancer" in name:
        labs.crp_mg_per_l = 1.0 + severity * 40.0
        labs.albumin_g_per_dl = max(1.5, 4.5 - severity * 3.0)
        labs.lactate_mmol_per_l = 1.2 + severity * 8.0

    elif "fabry" in name:
        labs.creatinine_mg_per_dl = 1.0 + severity * 1.5
        labs.egfr_ml_per_min = _ckd_epi_2021(
            labs.creatinine_mg_per_dl, labs.age_years, labs.sex,
        )

    elif "msud" in name:
        # Leucine accumulation → metabolic crisis markers
        labs.alt_u_per_l = 25.0 + severity * 80.0
        labs.lactate_mmol_per_l = 1.2 + severity * 4.0

    return labs
