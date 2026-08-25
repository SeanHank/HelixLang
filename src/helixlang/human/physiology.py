"""Human physiology specification for doc/27 (Stage A domain layer).

Defines the anatomical and physiological parameter sets consumed by the
PBPK (:mod:`helixlang.human.pharmacokinetics`), pathology, and long-term
simulation modules of the doc/27 human pathology & drug simulation stack.
Every value is anchored to published literature; nothing is invented.

Contents:
    OrganSpec               single organ compartment (volume, blood flow,
                            parenchymal fraction, tissue GEM handle)
    HumanPhysiology         whole-body physiology (anthropometrics, cardiac
                            output, blood/plasma volumes, organ map, plasma
                            protein level, hepatic CYP450 activity panel)
    TISSUE_PROFILES         literature-anchored per-organ parameter tables
                            for liver, kidney, brain, heart, muscle, adipose
    DEFAULT_HUMAN           reference 70 kg / 170 cm / 30 year male
    create_default_physiology()
                            factory returning a fresh, independently
                            mutable physiology instance

Parameter provenance (doc/27 §4.3-§4.5):
- Liver volume 1500 mL (Katykhin 2020); hepatic blood flow 1500 mL/min
  = 25% of cardiac output (Guyton & Hall 2016, Textbook of Medical
  Physiology)
- Kidney volume 300 mL (Nyengaard 1992); renal blood flow 1200 mL/min
  = 20% of cardiac output (Guyton 2016)
- Brain volume 1400 mL; cerebral blood flow 750 mL/min = 12.5% of cardiac
  output; brain glucose consumption ~120 g/day (Mergenthaler 2013)
- Heart volume 300 mL; coronary flow 250 mL/min = 5% of cardiac output at
  rest; myocardial O2 consumption 56 mL/kg/min, the highest of any organ
  (Staniszewski 2020)
- Skeletal muscle volume 24000 mL (~30% of body weight); resting muscle
  perfusion 750 mL/min = 12.5% of cardiac output (Guyton 2016)
- Adipose volume 15000 mL; adipose perfusion 200 mL/min (Guyton 2016)
- Cardiac output 5000 mL/min; plasma volume ~3000 mL; hematocrit 0.45;
  plasma albumin 4.5 g/dL (Guyton 2016; LEVINE 2013 Crit Care)
- Hepatic O2 consumption 44 mL/kg/min (Wilke 1999); basal hepatic glucose
  uptake 1.5 mmol/kg/min (Krabbe 2015)
- Summed organ O2 uptakes predict ~266 mL/min resting whole-body VO2,
  consistent with the canonical ~250 mL/min (Guyton 2016)
- Relative hepatic CYP450 abundances follow Rowland et al.,
  Br J Clin Pharmacol (CYP3A4 dominant at ~28% of total CYP content)

Primary source: Guyton AC, Hall JE. Textbook of Medical Physiology,
14th ed., 2016; remaining citations inline above.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helixlang.metabolism import MetabolicModel


#: average mammalian/human cell wet density (g/mL), doc/27 §4.2
CELL_DENSITY_G_PER_ML = 1.05


TISSUE_PROFILES: dict[str, dict] = {
    "liver": {
        "volume_ml": 1500.0,
        "blood_flow_ml_per_min": 1500.0,
        "tissue_fraction": 0.80,
        "key_reactions": [
            "CYP3A4",
            "CYP2D6",
            "CYP2C9",
            "UGT1A1",
            "SULT2A1",
            "GSTA1",
            "ALDOB",
            "PYGM",
            "PCK1",
            "G6PC",
        ],
        "oxygen_consumption_ml_per_kg_per_min": 44.0,
        "glucose_uptake_mmol_per_kg_per_min": 1.5,
        "lactate_production": True,
        "gluconeogenesis": True,
        "bile_acid_synthesis": True,
    },
    "kidney": {
        "volume_ml": 300.0,
        "blood_flow_ml_per_min": 1200.0,
        "tissue_fraction": 0.85,
        "key_reactions": ["SLC22A6", "SLC22A8", "CYP3A5", "CYP2B6", "GLUL"],
        "oxygen_consumption_ml_per_kg_per_min": 16.0,
        "glucose_uptake_mmol_per_kg_per_min": 0.5,
        "amino_acid_reabsorption": True,
        "urea_cycle_participation": True,
    },
    "brain": {
        "volume_ml": 1400.0,
        "blood_flow_ml_per_min": 750.0,
        "tissue_fraction": 0.80,
        "key_reactions": ["GPI", "PFKM", "LDHA", "CS", "IDH3A", "OGDH"],
        "oxygen_consumption_ml_per_kg_per_min": 38.0,
        "glucose_uptake_mmol_per_kg_per_min": 1.0,
        "ketone_body_utilization": True,
        "blood_brain_barrier": True,
    },
    "heart": {
        "volume_ml": 300.0,
        "blood_flow_ml_per_min": 250.0,
        "tissue_fraction": 0.80,
        "key_reactions": ["CPT1A", "ACADL", "HADHA", "LDHA", "CKM"],
        "oxygen_consumption_ml_per_kg_per_min": 56.0,
        "glucose_uptake_mmol_per_kg_per_min": 0.3,
        "fatty_acid_oxidation": True,
        "lactate_uptake": True,
    },
    "muscle": {
        "volume_ml": 24000.0,
        "blood_flow_ml_per_min": 750.0,
        "tissue_fraction": 0.80,
        "key_reactions": ["PYGM", "LDHA", "CPT1A", "ACADM", "CKM", "PFKM"],
        "oxygen_consumption_ml_per_kg_per_min": 6.0,
        "glucose_uptake_mmol_per_kg_per_min": 0.3,
        "glycogen_storage": True,
        "insulin_dependent_uptake": True,
    },
    "adipose": {
        "volume_ml": 15000.0,
        "blood_flow_ml_per_min": 200.0,
        "tissue_fraction": 0.85,
        "key_reactions": ["LPL", "FASN", "ACACA", "DGAT1", "ADIPOQ"],
        "oxygen_consumption_ml_per_kg_per_min": 2.0,
        "glucose_uptake_mmol_per_kg_per_min": 0.1,
        "lipolysis": True,
        "lipogenesis": True,
    },
}


#: fraction of total hepatic CYP450 content per isoform (Rowland et al.);
#: residual ~26% is distributed across minor isoforms
DEFAULT_CYP450_ACTIVITY: dict[str, float] = {
    "CYP3A4": 0.28,
    "CYP2C9": 0.20,
    "CYP1A2": 0.13,
    "CYP2E1": 0.07,
    "CYP2D6": 0.04,
    "CYP2C19": 0.02,
}


def _default_cyp450_activity() -> dict[str, float]:
    return dict(DEFAULT_CYP450_ACTIVITY)


_PROFILE_CORE_KEYS = (
    "volume_ml",
    "blood_flow_ml_per_min",
    "tissue_fraction",
    "key_reactions",
    "oxygen_consumption_ml_per_kg_per_min",
    "glucose_uptake_mmol_per_kg_per_min",
)


@dataclass(slots=True)
class OrganSpec:
    """Specification of a human organ for simulation."""

    name: str
    volume_ml: float
    blood_flow_ml_per_min: float
    tissue_fraction: float
    key_reactions: list[str] = field(default_factory=list)
    oxygen_consumption_ml_per_kg_per_min: float = 0.0
    glucose_uptake_mmol_per_kg_per_min: float = 0.0
    properties: dict[str, bool] = field(default_factory=dict)
    model: MetabolicModel | None = None
    tissue_profile: dict | None = None

    @classmethod
    def from_profile(cls, name: str, profile: dict | None = None) -> OrganSpec:
        """Build an :class:`OrganSpec` from a ``TISSUE_PROFILES`` entry.

        Uses ``profile`` verbatim when given, otherwise looks up ``name``
        in :data:`TISSUE_PROFILES` (raising ``KeyError`` for unknown
        organs). Boolean profile flags outside the core keys are stored
        in :attr:`properties`.
        """
        prof = dict(profile if profile is not None else TISSUE_PROFILES[name])
        properties = {k: v for k, v in prof.items() if k not in _PROFILE_CORE_KEYS}
        return cls(
            name=name,
            volume_ml=float(prof["volume_ml"]),
            blood_flow_ml_per_min=float(prof["blood_flow_ml_per_min"]),
            tissue_fraction=float(prof["tissue_fraction"]),
            key_reactions=list(prof.get("key_reactions", [])),
            oxygen_consumption_ml_per_kg_per_min=float(
                prof.get("oxygen_consumption_ml_per_kg_per_min", 0.0)
            ),
            glucose_uptake_mmol_per_kg_per_min=float(
                prof.get("glucose_uptake_mmol_per_kg_per_min", 0.0)
            ),
            properties=properties,
            tissue_profile=prof,
        )

    @property
    def parenchymal_volume_ml(self) -> float:
        """Volume occupied by functional parenchymal cells (mL)."""
        return self.volume_ml * self.tissue_fraction

    @property
    def parenchymal_mass_kg(self) -> float:
        """Functional tissue mass (kg) at 1.05 g/mL cell density."""
        return self.parenchymal_volume_ml * CELL_DENSITY_G_PER_ML / 1000.0

    @property
    def oxygen_consumption_ml_per_min(self) -> float:
        """Whole-organ O2 uptake (mL O2/min)."""
        return self.oxygen_consumption_ml_per_kg_per_min * self.parenchymal_mass_kg

    @property
    def glucose_uptake_mmol_per_min(self) -> float:
        """Whole-organ glucose uptake (mmol/min)."""
        return self.glucose_uptake_mmol_per_kg_per_min * self.parenchymal_mass_kg

    def flow_fraction(self, cardiac_output_ml_per_min: float) -> float:
        """Fraction of the given cardiac output received by this organ."""
        if cardiac_output_ml_per_min <= 0.0:
            raise ValueError("cardiac_output_ml_per_min must be positive")
        return self.blood_flow_ml_per_min / cardiac_output_ml_per_min

    def has_property(self, key: str) -> bool:
        """Return True when the tissue profile flag ``key`` is enabled."""
        return bool(self.properties.get(key, False))


@dataclass(slots=True)
class HumanPhysiology:
    """Complete human physiology specification.

    Defaults describe the reference 70 kg / 170 cm / 30 year adult male
    of doc/27 §4.5 (Guyton & Hall 2016). Use
    :func:`create_default_physiology` for instances pre-populated with
    the six :data:`TISSUE_PROFILES` organs.
    """

    body_weight_kg: float = 70.0
    height_cm: float = 170.0
    age_years: float = 30.0
    sex: str = "male"
    cardiac_output_ml_per_min: float = 5000.0
    organs: dict[str, OrganSpec] = field(default_factory=dict)
    plasma_volume_ml: float = 3000.0
    hematocrit: float = 0.45
    albumin_g_per_dL: float = 4.5
    cytochrome_p450_activity: dict[str, float] = field(
        default_factory=_default_cyp450_activity
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` when any physiological parameter is invalid."""
        if self.body_weight_kg <= 0.0:
            raise ValueError("body_weight_kg must be positive")
        if self.height_cm <= 0.0:
            raise ValueError("height_cm must be positive")
        if self.age_years < 0.0:
            raise ValueError("age_years must be non-negative")
        if self.sex not in ("male", "female"):
            raise ValueError("sex must be 'male' or 'female'")
        if self.cardiac_output_ml_per_min <= 0.0:
            raise ValueError("cardiac_output_ml_per_min must be positive")
        if not 0.0 < self.hematocrit < 1.0:
            raise ValueError("hematocrit must lie strictly between 0 and 1")
        if self.plasma_volume_ml <= 0.0:
            raise ValueError("plasma_volume_ml must be positive")
        if self.albumin_g_per_dL < 0.0:
            raise ValueError("albumin_g_per_dL must be non-negative")
        for organ_name, organ in self.organs.items():
            if not 0.0 <= organ.tissue_fraction <= 1.0:
                raise ValueError(
                    f"{organ_name}: tissue_fraction must lie in [0, 1]"
                )

    @property
    def bmi(self) -> float:
        """Body mass index (kg/m^2)."""
        height_m = self.height_cm / 100.0
        return self.body_weight_kg / (height_m * height_m)

    @property
    def body_surface_area_m2(self) -> float:
        """Body surface area (m^2) by the Du Bois & Du Bois 1916 formula."""
        return float(0.007184 * self.height_cm**0.725 * self.body_weight_kg**0.425)

    @property
    def blood_volume_ml(self) -> float:
        """Total blood volume (mL) from plasma volume and hematocrit."""
        return self.plasma_volume_ml / (1.0 - self.hematocrit)

    @property
    def red_cell_volume_ml(self) -> float:
        """Red cell volume (mL) = blood volume x hematocrit."""
        return self.blood_volume_ml * self.hematocrit

    @property
    def total_organ_volume_ml(self) -> float:
        """Summed volume of all modeled organs (mL)."""
        return sum(organ.volume_ml for organ in self.organs.values())

    @property
    def total_organ_blood_flow_ml_per_min(self) -> float:
        """Summed blood flow across all modeled organs (mL/min)."""
        return sum(o.blood_flow_ml_per_min for o in self.organs.values())

    @property
    def organ_flow_fractions(self) -> dict[str, float]:
        """Per-organ share of cardiac output (dimensionless)."""
        return {
            name: organ.flow_fraction(self.cardiac_output_ml_per_min)
            for name, organ in self.organs.items()
        }

    @property
    def total_oxygen_consumption_ml_per_min(self) -> float:
        """Summed whole-organ O2 uptake (mL O2/min), ~250 at rest."""
        return sum(
            organ.oxygen_consumption_ml_per_min for organ in self.organs.values()
        )

    @property
    def total_glucose_uptake_mmol_per_min(self) -> float:
        """Summed whole-organ glucose uptake (mmol/min)."""
        return sum(
            organ.glucose_uptake_mmol_per_min for organ in self.organs.values()
        )

    def has_organ(self, name: str) -> bool:
        """Return True when an organ with the given name is modeled."""
        return name in self.organs

    def get_organ(self, name: str) -> OrganSpec:
        """Return the :class:`OrganSpec` for ``name``.

        Raises ``KeyError`` listing the modeled organs when absent.
        """
        try:
            return self.organs[name]
        except KeyError:
            available = ", ".join(sorted(self.organs)) or "<none>"
            raise KeyError(
                f"unknown organ '{name}'; modeled organs: {available}"
            ) from None

    def organ_flow_fraction(self, name: str) -> float:
        """Fraction of cardiac output delivered to organ ``name``."""
        return self.get_organ(name).flow_fraction(self.cardiac_output_ml_per_min)

    def with_organ(self, organ: OrganSpec) -> HumanPhysiology:
        """Return a copy of this physiology with ``organ`` added/replaced."""
        return replace(self, organs={**self.organs, organ.name: organ})


def create_default_physiology(
    body_weight_kg: float = 70.0,
    height_cm: float = 170.0,
    age_years: float = 30.0,
    sex: str = "male",
) -> HumanPhysiology:
    """Create a :class:`HumanPhysiology` populated from ``TISSUE_PROFILES``.

    Returns the doc/27 reference adult male (70 kg, 170 cm, 30 years,
    5000 mL/min cardiac output) with liver, kidney, brain, heart, muscle,
    and adipose compartments attached. Each call returns a fresh,
    independently mutable instance.
    """
    organs = {
        name: OrganSpec.from_profile(name) for name in TISSUE_PROFILES
    }
    return HumanPhysiology(
        body_weight_kg=body_weight_kg,
        height_cm=height_cm,
        age_years=age_years,
        sex=sex,
        organs=organs,
    )


DEFAULT_HUMAN = create_default_physiology()


__all__ = [
    "CELL_DENSITY_G_PER_ML",
    "DEFAULT_CYP450_ACTIVITY",
    "DEFAULT_HUMAN",
    "OrganSpec",
    "HumanPhysiology",
    "TISSUE_PROFILES",
    "create_default_physiology",
]
