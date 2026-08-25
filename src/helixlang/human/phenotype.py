"""External traits to physiological parameter scaling (doc/28 human stack).

Folds a :class:`~helixlang.human.genotype.GenotypeProfile` and observable
:class:`ExternalTraits` (age, sex, anthropometrics, smoking, pregnancy,
exercise, ethnicity) into a fully scaled
:class:`~helixlang.human.physiology.HumanPhysiology` and the final hepatic
CYP450 activity panel consumed by the PBPK layer.

Scaling rules are literature-anchored:

- Age: cardiac output declines ~1%/yr after 40 (Guyton & Hall 2016);
  GFR declines ~1 mL/min per year after 40 (KDIGO 2013, CKD
  epidemiology); hepatic CYP3A4/CYP2C9 intrinsic clearance declines
  measurably after 60 (Sotaniemi et al. Clin Pharmacol Ther 1995);
  CYP2D6 is stable to ~70 then declines.
- Sex: females show ~85% of male cardiac output, ~80% skeletal muscle
  mass, higher body-fat fraction (Karastergiou et al. Int J Biol Sci
  2012), ~15% lower GFR at matched weight, and ~20% higher hepatic
  CYP3A4 content (Wolbold et al. J Pharmacol Exp Ther 2003).
- Obesity: liver volume rises ~0.67% per kg excess body weight
  (Molnar et al. Hepatology 2003); cardiac output rises with excess
  adiposity (de Divitiis et al. Obes Res 2001); GFR per unit BSA falls;
  adipose compartment expands ~0.8 L per kg excess weight.
- Smoking: 2-3x induction of hepatic CYP1A2 and mild CYP2E1 induction
  in current smokers (Benowitz et al. Clin Pharmacokinet 2003; Oneta
  et al. Alcohol Clin Exp Res 1998), partially reversible on quitting.
- Ethnicity: CYP2C19 poor-metabolizer frequency ~15% East Asian vs ~2%
  European and CYP2D6 PM ~1% East Asian vs ~2-5% European (Bernard et
  al. Br J Clin Pharmacol 2006); UGT1A1*28 allele frequency highest in
  African populations -> irinotecan severe-neutropenia risk (Innocenti
  et al. J Clin Oncol 2009).
- Pregnancy: term cardiac output +50%, GFR +50%, plasma volume +45%,
  albumin -15%, CYP3A4 induced toward 2x while CYP2D6 capacity falls
  (model specification doc/28).
- Exercise: vigorous training raises resting cardiac output (~+8%) and
  hepatic blood flow (~+15%) and lowers resting heart rate.

Module contents:
    ExternalTraits          observable patient traits
    PhenotypeCalculator     genotype + traits -> HumanPhysiology
    create_default_traits   reference 70 kg / 170 cm / 30 yr male
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from helixlang.human.physiology import (
    DEFAULT_CYP450_ACTIVITY,
    TISSUE_PROFILES,
    HumanPhysiology,
    OrganSpec,
)

if TYPE_CHECKING:
    from helixlang.human.genotype import GenotypeProfile

__all__ = [
    "ExternalTraits",
    "PhenotypeCalculator",
    "create_default_traits",
]


# ============================================================================
# Literature-anchored scaling constants
# ============================================================================

#: annual fractional decline of resting cardiac output after age 40
AGE_CARDIAC_DECLINE_PER_YEAR = 0.01
#: age after which cardiac output decline begins
AGE_CARDIAC_ONSET_YEARS = 40.0
#: young-adult reference GFR (mL/min) and its linear post-40 decline
REFERENCE_GFR_ML_PER_MIN = 120.0
GFR_ANNUAL_DECLINE_ML_PER_MIN = 1.0

#: female/male ratios at matched anthropometrics
FEMALE_CARDIAC_OUTPUT_FRACTION = 0.85
FEMALE_MUSCLE_VOLUME_FRACTION = 0.80
FEMALE_GFR_FRACTION = 0.85
FEMALE_LIVER_VOLUME_FRACTION = 0.85
FEMALE_KIDNEY_VOLUME_FRACTION = 0.90
FEMALE_PLASMA_VOLUME_FRACTION = 0.90
FEMALE_ADPOSE_MULTIPLIER = 1.33
FEMALE_CYP3A4_FACTOR = 1.20
FEMALE_HEMATOCRIT = 0.41

#: obesity scalings relative to a BMI-25 weight-matched reference
REFERENCE_BMI_FOR_EXCESS = 25.0
LIVER_VOLUME_GAIN_FRACTION_PER_EXCESS_KG = 0.0067
CARDIAC_OUTPUT_GAIN_PER_EXCESS_KG = 0.014
ADIPOSE_ML_PER_EXCESS_KG = 800.0
OBESITY_GFR_DROP_PER_BMI_ABOVE_30 = 0.003
PLASMA_VOLUME_WEIGHT_EXPONENT = 0.5

#: sarcopenia: fractional muscle loss per year after 60
MUSCLE_ANNUAL_LOSS_AFTER_60 = 0.01

#: smoking multipliers (Benowitz 2003; Oneta 1998)
SMOKING_CYP1A2_BASE = 2.0
SMOKING_CYP1A2_PACK_YEAR_GAIN = min(1.0, 1.0) * 1.0
SMOKING_CYP1A2_FORMER = 1.2
SMOKING_CYP2E1_CURRENT = 1.5
SMOKING_CYP2E1_FORMER = 1.1
HEAVY_DRINKING_THRESHOLD_DRINKS_PER_WEEK = 14.0
MODERATE_DRINKING_THRESHOLD_DRINKS_PER_WEEK = 7.0
ALCOHOL_CYP2E1_HEAVY = 1.5
ALCOHOL_CYP2E1_MODERATE = 1.2

#: hepatic CYP declines after 60 (Sotaniemi 1995 approximation)
AGE_CYP_DECLINE_PER_YEAR_AFTER_60 = 0.007
AGE_CYP_MIN_FACTOR = 0.60
AGE_CYP2D6_ONSET_YEARS = 70.0
AGE_CYP2D6_DECLINE_PER_YEAR = 0.005

#: population-mean activity factors derived from PM frequency differences
#: (east_asian CYP2C19 PM ~15% vs european ~2% => x0.87 mean activity;
#: east_asian CYP2D6 PM ~1% vs european ~5% => x1.04; south_asian
#: CYP2D6 PM ~7% => x0.98)
ETHNIC_CYP_FACTORS: dict[str, dict[str, float]] = {
    "european": {},
    "african": {},
    "east_asian": {"CYP2C19": 0.87, "CYP2D6": 1.04},
    "south_asian": {"CYP2D6": 0.98},
    "hispanic": {},
}

#: UGT1A1*28 allele frequencies (Innocenti 2009) driving irinotecan
#: toxicity-risk flags by ethnicity
ETHNIC_UGT1A1_RISK: dict[str, float] = {
    "african": 0.45,
    "european": 0.30,
    "hispanic": 0.30,
    "south_asian": 0.25,
    "east_asian": 0.12,
}

#: pregnancy scalings at term (gestational week 40)
PREGNANCY_CARDIAC_OUTPUT_GAIN = 0.50
PREGNANCY_GFR_GAIN = 0.50
PREGNANCY_PLASMA_VOLUME_GAIN = 0.45
PREGNANCY_ALBUMIN_DROP_FRACTION = 0.15
PREGNANCY_HEMATOCRIT_DROP_FRACTION = 0.15
PREGNANCY_CYP3A4_MAX_FACTOR = 2.0
PREGNANCY_CYP2D6_MAX_DROP = 0.30
PREGNANCY_TERM_WEEKS = 40.0

#: exercise adaptations at rest
EXERCISE_FACTORS: dict[str, tuple[float, float]] = {
    # (cardiac-output factor, hepatic-flow factor)
    "sedentary": (0.97, 1.00),
    "light": (1.00, 1.00),
    "moderate": (1.02, 1.05),
    "vigorous": (1.08, 1.15),
}

#: baseline cardiac output and organ flow shares of the reference adult
#: (TISSUE_PROFILES flows divided by the 5000 mL/min reference output)
_REFERENCE_CARDIAC_OUTPUT = 5000.0


def _flow_fraction(organ_name: str) -> float:
    """Resting flow share of cardiac output for a TISSUE_PROFILES organ."""
    return float(TISSUE_PROFILES[organ_name]["blood_flow_ml_per_min"]) / _REFERENCE_CARDIAC_OUTPUT


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into the closed interval ``[lo, hi]``."""
    return max(lo, min(hi, value))


# ============================================================================
# Data structures
# ============================================================================

@dataclass(slots=True)
class ExternalTraits:
    """Observable external traits of a virtual patient.

    Attributes:
        age_years: chronological age.
        sex: ``"male"`` | ``"female"``.
        body_weight_kg: measured body mass.
        height_cm: standing height.
        ethnicity: continental ancestry group used for pharmacogenomic
            population priors.
        smoking_status: ``"never"`` | ``"former"`` | ``"current"``.
        pack_years: cumulative tobacco exposure (packs/day x years).
        alcohol_drinks_per_week: standard drinks per week.
        exercise_level: ``"sedentary"`` | ``"light"`` | ``"moderate"``
            | ``"vigorous"``.
        pregnant: whether the patient is pregnant.
        gestational_weeks: gestational age when ``pregnant``.
        comorbidities: free-text comorbidity labels (reserved for
            downstream disease-layer coupling).
    """

    age_years: float = 30.0
    sex: str = "male"
    body_weight_kg: float = 70.0
    height_cm: float = 170.0
    ethnicity: str = "european"
    smoking_status: str = "never"
    pack_years: float = 0.0
    alcohol_drinks_per_week: float = 0.0
    exercise_level: str = "moderate"
    pregnant: bool = False
    gestational_weeks: float = 0.0
    comorbidities: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.body_weight_kg <= 0.0:
            raise ValueError("body_weight_kg must be positive")
        if self.height_cm <= 0.0:
            raise ValueError("height_cm must be positive")
        if self.age_years < 0.0:
            raise ValueError("age_years must be non-negative")
        if self.sex not in ("male", "female"):
            raise ValueError("sex must be 'male' or 'female'")
        if self.ethnicity not in ETHNIC_CYP_FACTORS:
            raise ValueError(
                f"ethnicity must be one of {sorted(ETHNIC_CYP_FACTORS)}, "
                f"got {self.ethnicity!r}"
            )
        if self.smoking_status not in ("never", "former", "current"):
            raise ValueError(
                "smoking_status must be never/former/current, "
                f"got {self.smoking_status!r}"
            )
        if self.exercise_level not in EXERCISE_FACTORS:
            raise ValueError(
                f"exercise_level must be one of {sorted(EXERCISE_FACTORS)}, "
                f"got {self.exercise_level!r}"
            )
        if self.pack_years < 0.0 or self.alcohol_drinks_per_week < 0.0:
            raise ValueError("pack_years and alcohol_drinks_per_week must be >= 0")
        if self.gestational_weeks < 0.0:
            raise ValueError("gestational_weeks must be non-negative")
        if self.pregnant and self.gestational_weeks <= 0.0:
            raise ValueError("gestational_weeks must be positive when pregnant")

    @property
    def bmi(self) -> float:
        """Body mass index (kg/m^2)."""
        height_m = self.height_cm / 100.0
        return self.body_weight_kg / (height_m * height_m)

    @property
    def is_obese(self) -> bool:
        """True when BMI reaches the WHO obesity threshold of 30 kg/m^2."""
        return self.bmi >= 30.0


@dataclass
class PhenotypeCalculator:
    """Calculates physiological parameters from genotype + external traits.

    Attributes:
        genotype: pharmacogenomic profile supplying genetic CYP
            activity scores.
        traits: observable external traits driving demographic,
            lifestyle, and obstetric scaling.
    """

    genotype: GenotypeProfile
    traits: ExternalTraits

    # ------------------------------------------------------------------
    # CYP450 panel
    # ------------------------------------------------------------------

    def compute_cyp_activity(self) -> dict[str, float]:
        """Combine genetic + demographic + lifestyle CYP450 modulation.

        Returns a dict of enzyme -> activity multiplier for every
        isoform in :data:`DEFAULT_CYP450_ACTIVITY`, where 1.0 equals the
        reference 30-year male extensive metabolizer. Multipliers fold
        together (in order): genotype activity score, age-related
        decline, sex, smoking, alcohol, ethnicity priors, and pregnancy.

        Returns:
            enzyme -> dimensionless activity multiplier, each clipped to
            [0.05, 4.0].
        """
        t = self.traits
        activity: dict[str, float] = {}
        for enzyme in DEFAULT_CYP450_ACTIVITY:
            value = self.genotype.get_cyp_activity(enzyme)
            value *= _age_cyp_factor(enzyme, t.age_years)
            if enzyme == "CYP3A4" and t.sex == "female":
                value *= FEMALE_CYP3A4_FACTOR
            value *= _smoking_cyp_factor(enzyme, t)
            value *= _alcohol_cyp_factor(enzyme, t.alcohol_drinks_per_week)
            value *= ETHNIC_CYP_FACTORS.get(t.ethnicity, {}).get(enzyme, 1.0)
            if t.pregnant:
                value *= _pregnancy_cyp_factor(enzyme, t.gestational_weeks)
            activity[enzyme] = round(_clamp(value, 0.05, 4.0), 4)
        return activity

    # ------------------------------------------------------------------
    # Renal function estimate
    # ------------------------------------------------------------------

    def estimate_gfr_ml_per_min(self) -> float:
        """Estimate absolute GFR (mL/min) from age, sex, BMI, pregnancy.

        Starts from the 120 mL/min young-adult reference, subtracts
        ~1 mL/min per year past 40, then applies the female, obesity,
        and pregnancy adjustments.
        """
        t = self.traits
        gfr = REFERENCE_GFR_ML_PER_MIN
        if t.age_years > AGE_CARDIAC_ONSET_YEARS:
            gfr -= GFR_ANNUAL_DECLINE_ML_PER_MIN * (
                t.age_years - AGE_CARDIAC_ONSET_YEARS
            )
        gfr = _clamp(gfr, 0.25 * REFERENCE_GFR_ML_PER_MIN, gfr)
        gfr *= FEMALE_GFR_FRACTION if t.sex == "female" else 1.0
        if t.bmi > 30.0:
            gfr *= max(
                0.85, 1.0 - OBESITY_GFR_DROP_PER_BMI_ABOVE_30 * (t.bmi - 30.0)
            )
        if t.pregnant:
            gfr *= 1.0 + PREGNANCY_GFR_GAIN * _gestational_progress(t)
        return round(gfr, 1)

    # ------------------------------------------------------------------
    # Whole-body physiology assembly
    # ------------------------------------------------------------------

    def compute_physiology(self) -> HumanPhysiology:
        """Return a fully scaled :class:`HumanPhysiology`.

        Builds the six-compartment reference adult from
        :data:`TISSUE_PROFILES`, then applies every demographic,
        obstetric, lifestyle, and genetic scaling rule documented at
        module level. The returned physiology carries the final CYP450
        activity panel from :meth:`compute_cyp_activity` and passes
        :meth:`HumanPhysiology.validate`.
        """
        t = self.traits
        excess_kg = max(0.0, t.body_weight_kg - _reference_weight_kg(t))
        is_female = t.sex == "female"
        co_factor, hepatic_flow_factor = EXERCISE_FACTORS[t.exercise_level]

        cardiac_output = _REFERENCE_CARDIAC_OUTPUT
        if t.age_years > AGE_CARDIAC_ONSET_YEARS:
            cardiac_output *= (1.0 - AGE_CARDIAC_DECLINE_PER_YEAR) ** (
                t.age_years - AGE_CARDIAC_ONSET_YEARS
            )
        cardiac_output *= FEMALE_CARDIAC_OUTPUT_FRACTION if is_female else 1.0
        cardiac_output *= 1.0 + CARDIAC_OUTPUT_GAIN_PER_EXCESS_KG * excess_kg
        if t.pregnant:
            cardiac_output *= 1.0 + PREGNANCY_CARDIAC_OUTPUT_GAIN
        cardiac_output *= co_factor

        gfr_fraction = _clamp(
            self.estimate_gfr_ml_per_min() / REFERENCE_GFR_ML_PER_MIN, 0.15, 1.6
        )

        muscle_factor = 1.0
        if is_female:
            muscle_factor *= FEMALE_MUSCLE_VOLUME_FRACTION
        if t.age_years > 60.0:
            muscle_factor *= max(
                0.60, 1.0 - MUSCLE_ANNUAL_LOSS_AFTER_60 * (t.age_years - 60.0)
            )
        if t.exercise_level == "vigorous":
            muscle_factor *= 1.05

        liver_volume = TISSUE_PROFILES["liver"]["volume_ml"]
        liver_volume *= FEMALE_LIVER_VOLUME_FRACTION if is_female else 1.0
        liver_volume *= 1.0 + LIVER_VOLUME_GAIN_FRACTION_PER_EXCESS_KG * excess_kg

        kidney_volume = TISSUE_PROFILES["kidney"]["volume_ml"]
        kidney_volume *= FEMALE_KIDNEY_VOLUME_FRACTION if is_female else 1.0

        muscle_volume = TISSUE_PROFILES["muscle"]["volume_ml"] * muscle_factor
        adipose_volume = TISSUE_PROFILES["adipose"]["volume_ml"] * (
            FEMALE_ADPOSE_MULTIPLIER if is_female else 1.0
        ) + ADIPOSE_ML_PER_EXCESS_KG * excess_kg

        organs = {
            "liver": OrganSpec.from_profile(
                "liver",
                {
                    **TISSUE_PROFILES["liver"],
                    "volume_ml": round(liver_volume, 1),
                    "blood_flow_ml_per_min": round(
                        _flow_fraction("liver") * cardiac_output * hepatic_flow_factor,
                        1,
                    ),
                },
            ),
            "kidney": OrganSpec.from_profile(
                "kidney",
                {
                    **TISSUE_PROFILES["kidney"],
                    "volume_ml": round(kidney_volume, 1),
                    "blood_flow_ml_per_min": round(
                        _flow_fraction("kidney") * cardiac_output * gfr_fraction,
                        1,
                    ),
                },
            ),
            "brain": OrganSpec.from_profile(
                "brain",
                {
                    **TISSUE_PROFILES["brain"],
                    "blood_flow_ml_per_min": round(
                        _flow_fraction("brain") * cardiac_output, 1
                    ),
                },
            ),
            "heart": OrganSpec.from_profile(
                "heart",
                {
                    **TISSUE_PROFILES["heart"],
                    "blood_flow_ml_per_min": round(
                        _flow_fraction("heart") * cardiac_output, 1
                    ),
                },
            ),
            "muscle": OrganSpec.from_profile(
                "muscle",
                {
                    **TISSUE_PROFILES["muscle"],
                    "volume_ml": round(muscle_volume, 1),
                    "blood_flow_ml_per_min": round(
                        _flow_fraction("muscle") * cardiac_output * muscle_factor,
                        1,
                    ),
                },
            ),
            "adipose": OrganSpec.from_profile(
                "adipose",
                {
                    **TISSUE_PROFILES["adipose"],
                    "volume_ml": round(adipose_volume, 1),
                    "blood_flow_ml_per_min": round(
                        _flow_fraction("adipose") * cardiac_output, 1
                    ),
                },
            ),
        }

        plasma_volume = TISSUE_PROFILES.get("kidney", {}).get("volume_ml", 300.0)
        plasma_volume = _REFERENCE_PLASMA_VOLUME_ML * (
            (t.body_weight_kg / 70.0) ** PLASMA_VOLUME_WEIGHT_EXPONENT
        )
        plasma_volume *= FEMALE_PLASMA_VOLUME_FRACTION if is_female else 1.0
        if t.pregnant:
            plasma_volume *= 1.0 + PREGNANCY_PLASMA_VOLUME_GAIN

        hematocrit = FEMALE_HEMATOCRIT if is_female else 0.45
        albumin = 4.5
        if t.pregnant:
            progress = _gestational_progress(t)
            hematocrit *= 1.0 - PREGNANCY_HEMATOCRIT_DROP_FRACTION * progress
            albumin *= 1.0 - PREGNANCY_ALBUMIN_DROP_FRACTION * progress

        return HumanPhysiology(
            body_weight_kg=t.body_weight_kg,
            height_cm=t.height_cm,
            age_years=t.age_years,
            sex=t.sex,
            cardiac_output_ml_per_min=round(cardiac_output, 1),
            organs=organs,
            plasma_volume_ml=round(plasma_volume, 1),
            hematocrit=round(hematocrit, 3),
            albumin_g_per_dL=round(albumin, 2),
            cytochrome_p450_activity=self.compute_cyp_activity(),
        )


# ============================================================================
# Scaling helper functions
# ============================================================================

_REFERENCE_PLASMA_VOLUME_ML = 3000.0


def _reference_weight_kg(traits: ExternalTraits) -> float:
    """Weight (kg) of a BMI-25 adult of the same height."""
    height_m = traits.height_cm / 100.0
    return REFERENCE_BMI_FOR_EXCESS * height_m * height_m


def _gestational_progress(traits: ExternalTraits) -> float:
    """Gestational progress fraction in [0, 1] toward term (week 40)."""
    return _clamp(traits.gestational_weeks / PREGNANCY_TERM_WEEKS, 0.0, 1.0)


def _age_cyp_factor(enzyme: str, age_years: float) -> float:
    """Age-related intrinsic-clearance multiplier for one enzyme."""
    if enzyme == "CYP2D6":
        if age_years <= AGE_CYP2D6_ONSET_YEARS:
            return 1.0
        return max(
            AGE_CYP_MIN_FACTOR,
            1.0 - AGE_CYP2D6_DECLINE_PER_YEAR * (age_years - AGE_CYP2D6_ONSET_YEARS),
        )
    if enzyme in ("CYP3A4", "CYP2C9"):
        if age_years <= 60.0:
            return 1.0
        return max(
            AGE_CYP_MIN_FACTOR,
            1.0 - AGE_CYP_DECLINE_PER_YEAR_AFTER_60 * (age_years - 60.0),
        )
    return 1.0


def _smoking_cyp_factor(enzyme: str, traits: ExternalTraits) -> float:
    """Tobacco-smoke induction multiplier for one enzyme."""
    if enzyme == "CYP1A2":
        if traits.smoking_status == "current":
            return SMOKING_CYP1A2_BASE + SMOKING_CYP1A2_PACK_YEAR_GAIN * min(
                traits.pack_years, 40.0
            ) / 40.0
        if traits.smoking_status == "former":
            return SMOKING_CYP1A2_FORMER
        return 1.0
    if enzyme == "CYP2E1":
        if traits.smoking_status == "current":
            return SMOKING_CYP2E1_CURRENT
        if traits.smoking_status == "former":
            return SMOKING_CYP2E1_FORMER
        return 1.0
    return 1.0


def _alcohol_cyp_factor(enzyme: str, drinks_per_week: float) -> float:
    """Alcohol-mediated CYP2E1 induction multiplier (Oneta 1998)."""
    if enzyme != "CYP2E1":
        return 1.0
    if drinks_per_week >= HEAVY_DRINKING_THRESHOLD_DRINKS_PER_WEEK:
        return ALCOHOL_CYP2E1_HEAVY
    if drinks_per_week >= MODERATE_DRINKING_THRESHOLD_DRINKS_PER_WEEK:
        return ALCOHOL_CYP2E1_MODERATE
    return 1.0


def _pregnancy_cyp_factor(enzyme: str, gestational_weeks: float) -> float:
    """Pregnancy multiplier for one enzyme, scaled by gestational progress."""
    progress = _clamp(gestational_weeks / PREGNANCY_TERM_WEEKS, 0.0, 1.0)
    if enzyme == "CYP3A4":
        return 1.0 + (PREGNANCY_CYP3A4_MAX_FACTOR - 1.0) * progress
    if enzyme == "CYP2D6":
        return 1.0 - PREGNANCY_CYP2D6_MAX_DROP * progress
    return 1.0


def create_default_traits() -> ExternalTraits:
    """Create the doc/27 reference traits (70 kg / 170 cm / 30 yr male).

    Never-smoker, teetotal, moderately active, non-pregnant European
    adult; each call returns a fresh, independently mutable instance.
    """
    return ExternalTraits()
