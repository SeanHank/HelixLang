"""Dynamic disease staging and progression over time (doc/28).

Diseases advance along the shared :class:`DiseaseStage` ladder
(preclinical -> mild -> moderate -> severe -> critical) driven by two
competing annual rates from :class:`ProgressionRate`:

- ``progression_rate_per_year`` — natural worsening (severity units/yr)
  under no treatment.
- ``treatment_response_rate`` — improvement (severity units/yr) under a
  fully effective treatment.  The net change each step is::

      d(severity)/yr = progression_rate * (1 - E) - response_rate * E

  where ``E`` is the supplied drug effectiveness in [0, 1].  Diseases
  whose standard of care only *slows* decline without reversing it
  (e.g. CKD under ACEi/ARB) are encoded with a small or zero response
  rate and rely on the ``(1 - E)`` term: an ACEi modeled at E = 0.75
  turns the CKD rate of 0.012 severity/yr (= 1.2 mL/min/1.73m2 eGFR
  decline over a 100-unit span) into ~0.3 mL/min/yr.

Staging uses laboratory values when available (eGFR for CKD, FIB-4 /
Child-Pugh for cirrhosis, HbA1c for diabetes, tumor markers for cancer)
and falls back to the continuous severity score otherwise.

Module structure:
    DiseaseStage             five-stage severity ladder
    ClinicalLabs             lab snapshot with derived indices (FIB-4,
                             Child-Pugh)
    ProgressionRate          literature-anchored rate parameters
    DiseaseProgressionModel  stateful stepper over severity/damage/stage
    PROGRESSION_PROFILES     built-in profiles (CKD, cirrhosis, T2DM,
                             cancer)
    create_progression_model factory resolving names/aliases

References:
- KDIGO 2012. Kidney Int Suppl (eGFR categories G1-G5)
- Sterling RK et al. Hepatology 2006 (FIB-4 index)
- Pugh RNH et al. Br J Surg 1973; Child CG 1964 (Child-Pugh score)
- American Diabetes Association. Standards of Care in Diabetes 2024
- Eisenhauer EA et al. Eur J Cancer 2009 (RECIST 1.1, tumor response)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "DiseaseStage",
    "ClinicalLabs",
    "ProgressionRate",
    "DiseaseProgressionModel",
    "PROGRESSION_PROFILES",
    "create_progression_model",
]

#: hours per average year (Julian year), used for dt_h -> year conversion
_HOURS_PER_YEAR = 24.0 * 365.25


class DiseaseStage(Enum):
    """Shared disease-severity ladder used by every progression model."""

    PRECLINICAL = "preclinical"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


#: strict ordering of stages, ascending severity
_STAGE_ORDER: dict[DiseaseStage, int] = {
    DiseaseStage.PRECLINICAL: 0,
    DiseaseStage.MILD: 1,
    DiseaseStage.MODERATE: 2,
    DiseaseStage.SEVERE: 3,
    DiseaseStage.CRITICAL: 4,
}


# ============================================================================
# Laboratory snapshot
# ============================================================================

@dataclass(slots=True)
class ClinicalLabs:
    """Snapshot of laboratory values feeding progression staging.

    All fields carry healthy adult defaults so callers may construct a
    labs object with only the analytes their scenario overrides.

    Attributes:
        age_years: patient age (needed for FIB-4).
        egfr_ml_min_1_73m2: estimated GFR, KDIGO categories.
        creatinine_mg_dl: serum creatinine.
        alt_u_l / ast_u_l: serum transaminases.
        total_bilirubin_mg_dl: total bilirubin.
        albumin_g_dl: serum albumin.
        inr: international normalized ratio.
        platelets_per_ul: platelet count per microliter.
        hba1c_percent: glycated hemoglobin.
        tumor_marker_ng_ml: generic normalized tumor marker burden
            (CA-125, PSA, AFP, ... rescaled to a common axis).
        ascites_grade: 0 none / 1 mild / 2 refractory (Child-Pugh).
        encephalopathy_grade: 0 none / 1-2 / 3-4 (Child-Pugh).
    """

    age_years: float = 50.0
    egfr_ml_min_1_73m2: float = 95.0
    creatinine_mg_dl: float = 0.9
    alt_u_l: float = 25.0
    ast_u_l: float = 22.0
    total_bilirubin_mg_dl: float = 0.8
    albumin_g_dl: float = 4.0
    inr: float = 1.0
    platelets_per_ul: float = 250_000.0
    hba1c_percent: float = 5.4
    tumor_marker_ng_ml: float = 0.5
    ascites_grade: int = 0
    encephalopathy_grade: int = 0

    def __post_init__(self) -> None:
        if self.platelets_per_ul <= 0.0:
            raise ValueError("platelets_per_ul must be > 0 for FIB-4")
        if self.age_years <= 0.0:
            raise ValueError("age_years must be > 0")

    def fib4_index(self) -> float:
        """Return the FIB-4 liver-fibrosis index (Sterling 2006).

        ``FIB-4 = age * AST / (platelets[10^9/L] * sqrt(AST))``.
        Interpretation: < 1.45 excludes advanced fibrosis; > 3.25
        indicates advanced fibrosis/cirrhosis.
        """
        platelets_10e9 = self.platelets_per_ul / 1000.0
        return float(
            self.age_years * self.ast_u_l
            / (platelets_10e9 * (self.ast_u_l ** 0.5))
        )

    def child_pugh_score(self) -> int:
        """Return the Child-Pugh score (5-15; A <= 6, B 7-9, C >= 10)."""
        bili = (
            1 if self.total_bilirubin_mg_dl < 2.0
            else 2 if self.total_bilirubin_mg_dl < 3.0
            else 3
        )
        albumin = (
            1 if self.albumin_g_dl > 3.5
            else 2 if self.albumin_g_dl >= 2.8
            else 3
        )
        inr_pts = (
            1 if self.inr < 1.7
            else 2 if self.inr <= 2.3
            else 3
        )
        ascites = min(max(self.ascites_grade, 0) + 1, 3)
        encephalopathy = (
            1 if self.encephalopathy_grade == 0
            else 2 if self.encephalopathy_grade <= 2
            else 3
        )
        return bili + albumin + inr_pts + ascites + encephalopathy


# ============================================================================
# Rate parameters and model
# ============================================================================

@dataclass(slots=True)
class ProgressionRate:
    """Literature-based progression rates for a specific disease.

    Attributes:
        disease_name: canonical profile name (e.g. ``"CKD"``).
        stage_thresholds: stage value -> parameter threshold on the
            staging metric (e.g. eGFR for CKD, HbA1c for diabetes).
            Direction of comparison is set by
            :attr:`higher_parameter_is_worse`.
        progression_rate_per_year: natural worsening per year in
            severity units (see module docstring for unit conversions).
        treatment_response_rate: improvement per year under a fully
            effective treatment.
        relapse_probability_per_year: probability of relapse after
            stopping (or losing) effective treatment.
        reversibility: fraction of accumulated damage that is reversible
            within [0, 1].
        plateau_time_years: expected time to end-stage if untreated.
        higher_parameter_is_worse: True when rising metric values mean
            worse disease (HbA1c, tumor markers); False when falling
            values do (eGFR).
    """

    disease_name: str
    stage_thresholds: dict[str, float]
    progression_rate_per_year: float
    treatment_response_rate: float
    relapse_probability_per_year: float
    reversibility: float
    plateau_time_years: float = 10.0
    higher_parameter_is_worse: bool = False

    def __post_init__(self) -> None:
        if self.progression_rate_per_year < 0.0:
            raise ValueError("progression_rate_per_year must be >= 0")
        if self.treatment_response_rate < 0.0:
            raise ValueError("treatment_response_rate must be >= 0")
        if not 0.0 <= self.relapse_probability_per_year <= 1.0:
            raise ValueError("relapse_probability_per_year must be in [0, 1]")
        if not 0.0 <= self.reversibility <= 1.0:
            raise ValueError("reversibility must be in [0, 1]")
        if self.plateau_time_years <= 0.0:
            raise ValueError("plateau_time_years must be > 0")


@dataclass
class DiseaseProgressionModel:
    """Tracks disease progression over time.

    Attributes:
        disease_name: canonical disease key (see PROGRESSION_PROFILES).
        current_stage: stage at simulation start / last step.
        current_severity: continuous severity in [0, 1].
        progression_rate: rate parameters; when absent the model still
            stages from severity/labs but does not progress.
        cumulative_damage: non-reversible damage accrued so far in
            [0, inf) — drives permanent organ-function loss.
    """

    disease_name: str
    current_stage: DiseaseStage = DiseaseStage.MILD
    current_severity: float = 0.3
    progression_rate: ProgressionRate | None = None
    cumulative_damage: float = 0.0
    _elapsed_time_h: float = 0.0
    _was_treated: bool = field(default=False, repr=False)
    rng: random.Random | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Staging helpers
    # ------------------------------------------------------------------

    def _stage_from_severity(self, severity: float) -> DiseaseStage:
        """Map continuous severity onto the stage ladder."""
        if severity < 0.05:
            return DiseaseStage.PRECLINICAL
        if severity < 0.35:
            return DiseaseStage.MILD
        if severity < 0.65:
            return DiseaseStage.MODERATE
        if severity < 0.90:
            return DiseaseStage.SEVERE
        return DiseaseStage.CRITICAL

    def _metric_from_labs(self, labs: ClinicalLabs) -> float | None:
        """Resolve this disease's staging metric from a lab snapshot."""
        key = self.disease_name.upper()
        if key == "CKD":
            return labs.egfr_ml_min_1_73m2
        if key == "LIVER_CIRRHOSIS":
            return labs.fib4_index()
        if key == "DIABETES_T2":
            return labs.hba1c_percent
        if key == "CANCER_GENERIC":
            return labs.tumor_marker_ng_ml
        return None

    def _stage_from_metric(self, metric: float) -> DiseaseStage:
        """Apply the profile's ``stage_thresholds`` to a metric value."""
        rate = self.progression_rate
        if rate is None or not rate.stage_thresholds:
            return self._stage_from_severity(self.current_severity)
        valid_values = {member.value for member in DiseaseStage}
        thresholds = {
            stage: value
            for stage, value in rate.stage_thresholds.items()
            if stage in valid_values
        }
        if not thresholds:
            return self._stage_from_severity(self.current_severity)

        # Always walk the ladder in ascending severity; only the
        # comparison direction depends on which way "worse" runs.
        ordered = sorted(
            thresholds.items(),
            key=lambda kv: _STAGE_ORDER[DiseaseStage(kv[0])],
        )
        for stage_value, cutoff in ordered:
            stage = DiseaseStage(stage_value)
            if rate.higher_parameter_is_worse and metric < cutoff:
                return stage
            if not rate.higher_parameter_is_worse and metric >= cutoff:
                return stage
        return DiseaseStage.CRITICAL

    def _restage(self, labs: ClinicalLabs | None) -> DiseaseStage:
        """Recompute the current stage from labs when available."""
        if labs is None:
            return self._stage_from_severity(self.current_severity)
        metric = self._metric_from_labs(labs)
        if metric is None:
            return self._stage_from_severity(self.current_severity)
        return self._stage_from_metric(metric)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        dt_h: float,
        drug_effectiveness: float,
        labs: ClinicalLabs | None = None,
    ) -> DiseaseStage:
        """Advance disease by *dt_h* hours. Returns the new stage.

        Args:
            dt_h: integration interval in hours (> 0).
            drug_effectiveness: net treatment effectiveness in [0, 1]
                (0 = untreated natural history, 1 = fully effective).
            labs: optional lab snapshot; when given, restaging uses the
                disease-specific clinical metric instead of raw severity.

        Relapse handling: when drug effectiveness drops below 0.5 while
        the previous step was treated, a relapse roll is made against
        ``relapse_probability_per_year`` scaled by the elapsed interval;
        on success a severity bump proportional to lost reversibility is
        applied immediately.
        """
        if dt_h < 0.0:
            raise ValueError(f"dt_h must be >= 0, got {dt_h}")
        if not 0.0 <= drug_effectiveness <= 1.0:
            raise ValueError(
                f"drug_effectiveness must be within [0, 1], "
                f"got {drug_effectiveness}"
            )

        years = dt_h / _HOURS_PER_YEAR
        delta = 0.0
        treated_now = drug_effectiveness >= 0.5

        if self.progression_rate is not None:
            rate = self.progression_rate
            delta += rate.progression_rate_per_year * years * (
                1.0 - drug_effectiveness
            )
            delta -= rate.treatment_response_rate * years * drug_effectiveness

            if self._was_treated and not treated_now and dt_h > 0.0:
                p_relapse = 1.0 - (1.0 - rate.relapse_probability_per_year) ** years
                rng = self.rng or random.Random(0)
                if rng.random() < p_relapse:
                    delta += min(
                        0.25,
                        rate.progression_rate_per_year
                        * max(rate.plateau_time_years * 0.05, 0.5)
                        * (1.0 - rate.reversibility),
                    )

        previous = self.current_severity
        new_severity = min(1.0, max(0.0, previous + delta))

        if delta > 0.0:
            self.cumulative_damage += delta * (1.0 - self.reversibility_fraction())
        elif delta < 0.0:
            recovered = min(
                self.cumulative_damage, abs(delta) * self.reversibility_fraction()
            )
            self.cumulative_damage -= recovered

        self.current_severity = new_severity
        self.current_stage = self._restage(labs)
        self._elapsed_time_h += dt_h
        self._was_treated = treated_now
        return self.current_stage

    def reversibility_fraction(self) -> float:
        """Return the profile's reversibility (0.0 when unprofiled)."""
        return self.progression_rate.reversibility if self.progression_rate else 0.0

    def get_severity(self) -> float:
        """Return current continuous severity clamped to [0, 1]."""
        return min(1.0, max(0.0, self.current_severity))

    def get_organ_function(self, organ: str) -> float:
        """Return [0, 1] organ-specific function fraction (1.0 = healthy).

        Sensitivity weights encode how strongly each disease erodes a
        given organ's function; organs outside the table are unaffected
        and return 1.0.  Cumulative (non-reversible) damage subtracts a
        permanent floor on top of the reversible component.
        """
        sensitivities = _ORGAN_SENSITIVITY.get(self.disease_name.upper(), {})
        sensitivity = sensitivities.get(organ.strip().lower(), 0.0)
        reversible_loss = (
            (self.get_severity() - min(self.cumulative_damage, self.get_severity()))
            * sensitivity
        )
        permanent_loss = self.cumulative_damage * sensitivity
        return min(1.0, max(0.0, 1.0 - reversible_loss - permanent_loss))


# ============================================================================
# Organ sensitivity tables
# ============================================================================

#: organ -> sensitivity in [0, 1] per disease family; missing organs are
#: considered unaffected (sensitivity 0.0)
_ORGAN_SENSITIVITY: dict[str, dict[str, float]] = {
    "CKD": {
        "kidney": 1.0, "cardiovascular": 0.45, "vascular": 0.40,
        "bone": 0.30, "hematologic": 0.35, "immune": 0.20,
    },
    "LIVER_CIRRHOSIS": {
        "liver": 1.0, "coagulation": 0.60, "brain": 0.35,
        "spleen": 0.50, "gastrointestinal": 0.40, "immune": 0.25,
    },
    "DIABETES_T2": {
        "endocrine_pancreas": 1.0, "vascular": 0.55, "kidney": 0.45,
        "eye": 0.55, "peripheral_nerve": 0.50, "heart": 0.40,
    },
    "CANCER_GENERIC": {
        "primary_tumor": 1.0, "immune": 0.60, "bone_marrow": 0.50,
        "liver": 0.35, "lung": 0.35,
    },
}


# ============================================================================
# Predefined profiles and factory
# ============================================================================

#: Literature-anchored progression profiles keyed by canonical name.
PROGRESSION_PROFILES: dict[str, ProgressionRate] = {
    "CKD": ProgressionRate(
        disease_name="CKD",
        stage_thresholds={
            "preclinical": 90.0, "mild": 60.0, "moderate": 30.0, "severe": 15.0,
        },
        progression_rate_per_year=0.012,
        treatment_response_rate=0.0,
        relapse_probability_per_year=0.10,
        reversibility=0.15,
        plateau_time_years=45.0,
        higher_parameter_is_worse=False,
    ),
    "LIVER_CIRRHOSIS": ProgressionRate(
        disease_name="LIVER_CIRRHOSIS",
        stage_thresholds={
            "preclinical": 1.45, "mild": 2.50, "moderate": 3.25, "severe": 6.00,
        },
        progression_rate_per_year=0.10,
        treatment_response_rate=0.02,
        relapse_probability_per_year=0.15,
        reversibility=0.30,
        plateau_time_years=10.0,
        higher_parameter_is_worse=True,
    ),
    "DIABETES_T2": ProgressionRate(
        disease_name="DIABETES_T2",
        stage_thresholds={
            "preclinical": 5.7, "mild": 6.5, "moderate": 7.5, "severe": 9.5,
        },
        progression_rate_per_year=0.05,
        treatment_response_rate=0.04,
        relapse_probability_per_year=0.20,
        reversibility=0.40,
        plateau_time_years=20.0,
        higher_parameter_is_worse=True,
    ),
    "CANCER_GENERIC": ProgressionRate(
        disease_name="CANCER_GENERIC",
        stage_thresholds={
            "preclinical": 1.0, "mild": 10.0, "moderate": 100.0, "severe": 500.0,
        },
        progression_rate_per_year=0.25,
        treatment_response_rate=0.50,
        relapse_probability_per_year=0.25,
        reversibility=0.60,
        plateau_time_years=5.0,
        higher_parameter_is_worse=True,
    ),
}

#: accepted synonyms for :func:`create_progression_model` lookup
_DISEASE_ALIASES: dict[str, str] = {
    "ckd": "CKD",
    "chronic kidney disease": "CKD",
    "kidney disease": "CKD",
    "renal failure": "CKD",
    "cirrhosis": "LIVER_CIRRHOSIS",
    "liver cirrhosis": "LIVER_CIRRHOSIS",
    "hepatic cirrhosis": "LIVER_CIRRHOSIS",
    "liver fibrosis": "LIVER_CIRRHOSIS",
    "diabetes": "DIABETES_T2",
    "diabetes_t2": "DIABETES_T2",
    "type 2 diabetes": "DIABETES_T2",
    "t2dm": "DIABETES_T2",
    "cancer": "CANCER_GENERIC",
    "cancer_generic": "CANCER_GENERIC",
    "tumor": "CANCER_GENERIC",
    "solid_tumor": "CANCER_GENERIC",
}


def create_progression_model(disease_name: str) -> DiseaseProgressionModel:
    """Build a :class:`DiseaseProgressionModel` for *disease_name*.

    Resolution is case-insensitive and accepts common aliases (see
    ``_DISEASE_ALIASES``).  Unknown diseases raise ``ValueError`` with
    the list of known keys.
    """
    key = disease_name.strip().lower()
    canonical = _DISEASE_ALIASES.get(key, key.upper())
    rate = PROGRESSION_PROFILES.get(canonical)
    if rate is None:
        known = ", ".join(sorted(PROGRESSION_PROFILES))
        raise ValueError(f"Unknown disease {disease_name!r}; known: {known}")
    initial_severity = 0.3 if canonical != "CANCER_GENERIC" else 0.2
    return DiseaseProgressionModel(
        disease_name=canonical,
        current_stage=DiseaseStage.MILD,
        current_severity=initial_severity,
        progression_rate=rate,
        cumulative_damage=initial_severity * (1.0 - rate.reversibility),
    )
