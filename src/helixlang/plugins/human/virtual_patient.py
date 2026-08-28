"""Virtual Patient — unified facade for full-body simulation (doc/28).

Accepts a person's genome, external traits, disease state, and drug regimen,
then runs an hourly integration loop coupling:
  - Genotype → CYP450 phenotype → drug clearance modifiers
  - External traits → organ volumes, blood flows, cardiac output
  - PBPK per drug → concentration-vs-time in every compartment
  - PD → metabolic-flux scaling + clinical-toxicity triggers
  - Disease progression → dynamic severity, staging, organ-function fractions
  - Drug-drug interactions → clearance modifiers, additive-toxicity alerts
  - Clinical labs (liver, kidney, CBC, metabolic, inflammatory)
  - Vital signs (BP, HR, temperature, weight, SpO₂)
  - Post-treatment recovery → biomarker return, rebound, sequelae

Output: *VirtualPatientResult* — full time-series of every parameter.

References:
    - Guyton & Hall 2016 (human physiology)
    - Rowland & Tozer (Clinical Pharmacokinetics, 5th ed.)
    - CKD-EPI 2021 (eGFR)
    - FDA Guidance for DILI (2009)
    - CPIC guidelines (pharmacogenomics)
"""
from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from helixlang.plugins.human.clinical_output import ClinicalLabModel, ClinicalLabs, VitalsModel
from helixlang.plugins.human.ddi import DDIModel, assess_additive_toxicity
from helixlang.plugins.human.disease import DiseaseState
from helixlang.plugins.human.disease_progression import (
    DiseaseProgressionModel,
    DiseaseStage,
    ProgressionRate,
    create_progression_model,
)
from helixlang.plugins.human.drug import IV, IV_INFUSION, Drug
from helixlang.plugins.human.genotype import (
    CORE_CYP_ENZYMES,
    GenotypeProfile,
    create_default_genotype,
)
from helixlang.plugins.human.organ_crosstalk import apply_crosstalk, create_crosstalk
from helixlang.plugins.human.pharmacodynamics import (
    Pharmacodynamics,
    get_predefined_pd,
    hill_equation,
    infer_pd_from_drug,
)
from helixlang.plugins.human.pharmacokinetics import (
    DEFAULT_FLOW_FRACTIONS,
    FIRST_ORDER_ROUTES,
    ML_PER_MIN_TO_L_PER_H,
    ORGAN_NAMES,
    PBPKConfig,
)
from helixlang.plugins.human.phenotype import ExternalTraits, PhenotypeCalculator
from helixlang.plugins.human.physiology import HumanPhysiology
from helixlang.plugins.human.recovery import RecoveryModel, create_recovery_model

try:
    from helixlang.plugins.human.gem_human import HumanGEMLoader
except ImportError:  # pragma: no cover
    HumanGEMLoader = None  # type: ignore[assignment,misc]

try:
    from helixlang.plugins.human.endocrine import EndocrineSystem
    from helixlang.plugins.human.hematology_model import HematologySystem
    from helixlang.plugins.human.immune import CRPDriver, InnateImmuneModel
    from helixlang.plugins.human.qsp_binding import QSPBindingSystem
    from helixlang.plugins.human.renal_model import RenalFunctionModel
except ImportError:  # pragma: no cover
    EndocrineSystem = None  # type: ignore[assignment,misc]
    QSPBindingSystem = None  # type: ignore[assignment,misc]
    InnateImmuneModel = None  # type: ignore[assignment,misc]
    CRPDriver = None  # type: ignore[assignment,misc]
    HematologySystem = None  # type: ignore[assignment,misc]
    RenalFunctionModel = None  # type: ignore[assignment,misc]

# doc/32 modules — imported lazily to avoid circular imports
_stochastic_ode = None
_bayesian_denoiser = None
_mechanistic_ddi = None
_tissue_gem = None
_reduced_order = None
_pharmacogenomic_ae = None
_proteome_binding = None
_microbiome = None
_emergent_complexity = None

def _import_doc32() -> None:
    """Lazy-import doc/32 modules on first use."""
    global _stochastic_ode, _bayesian_denoiser, _mechanistic_ddi  # noqa: PLW0603
    global _tissue_gem, _reduced_order, _pharmacogenomic_ae  # noqa: PLW0603
    global _proteome_binding, _microbiome, _emergent_complexity  # noqa: PLW0603
    if _stochastic_ode is None:
        try:
            from helixlang.plugins.human.stochastic_ode import SDEConfig
            _stochastic_ode = SDEConfig
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "VirtualPatient needs helixlang.plugins.human.stochastic_ode. "
                "Reinstall with: pip install --force-reinstall helixlang"
            ) from exc
    if _bayesian_denoiser is None:
        try:
            from helixlang.plugins.human.bayesian_denoiser import BayesianDenoiser
            _bayesian_denoiser = BayesianDenoiser
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "VirtualPatient needs helixlang.plugins.human.bayesian_denoiser. "
                "Reinstall with: pip install --force-reinstall helixlang"
            ) from exc
    if _mechanistic_ddi is None:
        try:
            from helixlang.plugins.human.mechanistic_ddi import (
                EnzymeInhibitionLibrary,
                MechanisticDDIPredictor,
            )
            _mechanistic_ddi = (MechanisticDDIPredictor, EnzymeInhibitionLibrary)
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "VirtualPatient needs helixlang.plugins.human.mechanistic_ddi. "
                "Reinstall with: pip install --force-reinstall helixlang"
            ) from exc
    if _tissue_gem is None:
        try:
            from helixlang.plugins.human.tissue_gem import GEMDecomposer, OrganGEMCoupler
            _tissue_gem = (GEMDecomposer, OrganGEMCoupler)
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "VirtualPatient needs helixlang.plugins.human.tissue_gem. "
                "Reinstall with: pip install --force-reinstall helixlang"
            ) from exc
    if _reduced_order is None:
        try:
            from helixlang.plugins.human.reduced_order_organ import PODModeGenerator
            _reduced_order = PODModeGenerator
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "VirtualPatient needs helixlang.plugins.human.reduced_order_organ. "
                "Reinstall with: pip install --force-reinstall helixlang"
            ) from exc
    if _pharmacogenomic_ae is None:
        try:
            from helixlang.plugins.human.pharmacogenomic_ae import GenotypeAEPredictor
            _pharmacogenomic_ae = GenotypeAEPredictor
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "VirtualPatient needs helixlang.plugins.human.pharmacogenomic_ae. "
                "Reinstall with: pip install --force-reinstall helixlang"
            ) from exc
    if _proteome_binding is None:
        try:
            from helixlang.plugins.human.proteome_binding import ProteomeBindingCascade
            _proteome_binding = ProteomeBindingCascade
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "VirtualPatient needs helixlang.plugins.human.proteome_binding. "
                "Reinstall with: pip install --force-reinstall helixlang"
            ) from exc
    if _microbiome is None:
        try:
            from helixlang.plugins.human.microbiome import MicrobiomeCompartment
            _microbiome = MicrobiomeCompartment
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "VirtualPatient needs helixlang.plugins.human.microbiome. "
                "Reinstall with: pip install --force-reinstall helixlang"
            ) from exc
    if _emergent_complexity is None:
        try:
            from helixlang.plugins.human.emergent_complexity import EmergentComplexityModel
            _emergent_complexity = EmergentComplexityModel
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "VirtualPatient needs helixlang.plugins.human.emergent_complexity. "
                "Reinstall with: pip install --force-reinstall helixlang"
            ) from exc

__all__ = [
    "VirtualPatient",
    "VirtualPatientConfig",
    "VirtualPatientResult",
]

# ============================================================================
# Disease progression helpers (used by _build_disease_model)
# ============================================================================

# Canonical profiles that support custom severity override
_DISEASE_PROGRESSION_PROFILES: dict[str, ProgressionRate] = {
    "CKD": ProgressionRate(
        disease_name="CKD",
        stage_thresholds={"preclinical": 90.0, "mild": 60.0, "moderate": 30.0, "severe": 15.0},
        progression_rate_per_year=0.012, treatment_response_rate=0.0,
        relapse_probability_per_year=0.10, reversibility=0.15,
        plateau_time_years=45.0, higher_parameter_is_worse=False,
    ),
    "LIVER_CIRRHOSIS": ProgressionRate(
        disease_name="LIVER_CIRRHOSIS",
        stage_thresholds={"preclinical": 1.45, "mild": 2.50, "moderate": 3.25, "severe": 6.00},
        progression_rate_per_year=0.10, treatment_response_rate=0.02,
        relapse_probability_per_year=0.15, reversibility=0.30,
        plateau_time_years=10.0, higher_parameter_is_worse=True,
    ),
    "DIABETES_T2": ProgressionRate(
        disease_name="DIABETES_T2",
        stage_thresholds={"preclinical": 5.7, "mild": 6.5, "moderate": 7.5, "severe": 9.5},
        progression_rate_per_year=0.05, treatment_response_rate=0.04,
        relapse_probability_per_year=0.20, reversibility=0.40,
        plateau_time_years=20.0, higher_parameter_is_worse=True,
    ),
    "CANCER_GENERIC": ProgressionRate(
        disease_name="CANCER_GENERIC",
        stage_thresholds={"preclinical": 1.0, "mild": 10.0, "moderate": 100.0, "severe": 500.0},
        progression_rate_per_year=0.25, treatment_response_rate=0.50,
        relapse_probability_per_year=0.25, reversibility=0.60,
        plateau_time_years=5.0, higher_parameter_is_worse=True,
    ),
    "INFECTIOUS": ProgressionRate(
        disease_name="INFECTIOUS",
        stage_thresholds={"preclinical": 0.05, "mild": 0.30, "moderate": 0.60, "severe": 0.85},
        progression_rate_per_year=0.50, treatment_response_rate=2.0,
        relapse_probability_per_year=0.30, reversibility=0.90,
        plateau_time_years=0.1, higher_parameter_is_worse=True,
    ),
    "AUTOIMMUNE": ProgressionRate(
        disease_name="AUTOIMMUNE",
        stage_thresholds={"preclinical": 0.05, "mild": 0.30, "moderate": 0.60, "severe": 0.85},
        progression_rate_per_year=0.15, treatment_response_rate=0.20,
        relapse_probability_per_year=0.40, reversibility=0.50,
        plateau_time_years=10.0, higher_parameter_is_worse=True,
    ),
    "CARDIOVASCULAR": ProgressionRate(
        disease_name="CARDIOVASCULAR",
        stage_thresholds={"preclinical": 0.05, "mild": 0.30, "moderate": 0.60, "severe": 0.85},
        progression_rate_per_year=0.10, treatment_response_rate=0.05,
        relapse_probability_per_year=0.20, reversibility=0.30,
        plateau_time_years=15.0, higher_parameter_is_worse=True,
    ),
    "RESPIRATORY": ProgressionRate(
        disease_name="RESPIRATORY",
        stage_thresholds={"preclinical": 0.05, "mild": 0.30, "moderate": 0.60, "severe": 0.85},
        progression_rate_per_year=0.10, treatment_response_rate=0.15,
        relapse_probability_per_year=0.35, reversibility=0.40,
        plateau_time_years=15.0, higher_parameter_is_worse=True,
    ),
    "NEUROLOGICAL": ProgressionRate(
        disease_name="NEUROLOGICAL",
        stage_thresholds={"preclinical": 0.05, "mild": 0.30, "moderate": 0.60, "severe": 0.85},
        progression_rate_per_year=0.08, treatment_response_rate=0.02,
        relapse_probability_per_year=0.10, reversibility=0.20,
        plateau_time_years=20.0, higher_parameter_is_worse=True,
    ),
    "HEMATOLOGICAL": ProgressionRate(
        disease_name="HEMATOLOGICAL",
        stage_thresholds={"preclinical": 0.05, "mild": 0.30, "moderate": 0.60, "severe": 0.85},
        progression_rate_per_year=0.20, treatment_response_rate=0.10,
        relapse_probability_per_year=0.20, reversibility=0.30,
        plateau_time_years=10.0, higher_parameter_is_worse=True,
    ),
    "GASTROINTESTINAL": ProgressionRate(
        disease_name="GASTROINTESTINAL",
        stage_thresholds={"preclinical": 0.05, "mild": 0.30, "moderate": 0.60, "severe": 0.85},
        progression_rate_per_year=0.10, treatment_response_rate=0.15,
        relapse_probability_per_year=0.30, reversibility=0.40,
        plateau_time_years=15.0, higher_parameter_is_worse=True,
    ),
    "ENDOCRINE": ProgressionRate(
        disease_name="ENDOCRINE",
        stage_thresholds={"preclinical": 0.05, "mild": 0.30, "moderate": 0.60, "severe": 0.85},
        progression_rate_per_year=0.10, treatment_response_rate=0.15,
        relapse_probability_per_year=0.25, reversibility=0.50,
        plateau_time_years=15.0, higher_parameter_is_worse=True,
    ),
}


def _build_disease_progression_model(canonical: str, severity: float) -> DiseaseProgressionModel:
    """Build a DiseaseProgressionModel with custom severity."""
    rate = _DISEASE_PROGRESSION_PROFILES[canonical]
    # Map severity [0,1] to initial stage
    if severity < 0.05:
        stage = DiseaseStage.PRECLINICAL
    elif severity < 0.35:
        stage = DiseaseStage.MILD
    elif severity < 0.65:
        stage = DiseaseStage.MODERATE
    elif severity < 0.90:
        stage = DiseaseStage.SEVERE
    else:
        stage = DiseaseStage.CRITICAL
    return DiseaseProgressionModel(
        disease_name=canonical,
        current_stage=stage,
        current_severity=severity,
        progression_rate=rate,
        cumulative_damage=severity * (1.0 - rate.reversibility),
    )


def _build_generic_progression_model(severity: float) -> DiseaseProgressionModel:
    """Generic fallback with custom severity (replaces hardcoded 0.2)."""
    if severity < 0.05:
        stage = DiseaseStage.PRECLINICAL
    elif severity < 0.35:
        stage = DiseaseStage.MILD
    elif severity < 0.65:
        stage = DiseaseStage.MODERATE
    elif severity < 0.90:
        stage = DiseaseStage.SEVERE
    else:
        stage = DiseaseStage.CRITICAL
    rate = ProgressionRate(
        disease_name="GENERIC",
        stage_thresholds={"preclinical": 0.05, "mild": 0.30, "moderate": 0.60, "severe": 0.85},
        progression_rate_per_year=0.10,
        treatment_response_rate=0.10,
        relapse_probability_per_year=0.20,
        reversibility=0.40,
        plateau_time_years=10.0,
        higher_parameter_is_worse=True,
    )
    return DiseaseProgressionModel(
        disease_name="GENERIC",
        current_stage=stage,
        current_severity=severity,
        progression_rate=rate,
        cumulative_damage=severity * (1.0 - rate.reversibility),
    )


# ============================================================================
# Anti-inflammatory PD target filter
# ============================================================================
# Only PD targets in this set are considered anti-inflammatory for immune
# suppression. Prevents antibiotics/chemo from falsely suppressing IL-6/CRP.
ANTI_INFLAMMATORY_PD_TARGETS: frozenset[str] = frozenset({
    "inflammation", "COX1", "COX2", "prostaglandin_synthesis",
    "leukotriene_synthesis", "TNF", "IL6R", "JAK",
    "NF_kB", "TLR4", "complement", "phospholipase_A2",
})

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class VirtualPatientConfig:
    """Complete input specification for a virtual-patient simulation."""

    # --- Person ---
    genotype: GenotypeProfile = field(default_factory=create_default_genotype)
    traits: ExternalTraits = field(default_factory=ExternalTraits)

    # --- Disease ---
    disease: DiseaseState | None = None
    disease_profile_name: str = ""

    # --- Cancer therapy (doc/33 Phase 4) ---
    tumor_biopsy: dict[str, Any] | None = None

    # --- Drugs ---
    drugs: list[Drug] = field(default_factory=list)
    pharmacodynamics: dict[str, Pharmacodynamics] = field(default_factory=dict)
    ddi_model: DDIModel | None = None

    # --- Simulation control ---
    total_duration_days: float = 30.0
    dfa_dt_h: float = 1.0
    output_time_resolution_h: float = 1.0

    # --- doc/30-31 extensions ---
    qsp_bindings: list[dict[str, Any]] = field(default_factory=list)
    endocrine_configs: list[dict[str, Any]] = field(default_factory=list)
    immune_configs: list[dict[str, Any]] = field(default_factory=list)

    # --- Tracking flags ---
    track_vitals: bool = True
    track_labs: bool = True
    track_drug_levels: bool = True
    track_fluxes: bool = True
    track_metabolites: bool = True
    track_biomarkers: bool = True

    # --- Model paths ---
    base_model_path: str = ""

    def __post_init__(self) -> None:
        if self.disease_profile_name and self.disease is None:
            from helixlang.plugins.human.disease import DISEASE_PROFILES
            prof = DISEASE_PROFILES.get(self.disease_profile_name)
            if prof is not None:
                self.disease = deepcopy(prof)

        if self.drugs:
            for drug in self.drugs:
                key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                if key not in self.pharmacodynamics:
                    pd = get_predefined_pd(drug.molecule.name)
                    if pd is not None:
                        self.pharmacodynamics[key] = pd
                    else:
                        # Infer PD from molecular properties (doc/33 gap-2)
                        self.pharmacodynamics[key] = infer_pd_from_drug(
                            drug.molecule.name,
                            target_protein=drug.molecule.target_protein,
                            binding_kd_um=drug.molecule.binding_affinity_kd_um,
                            mw_da=drug.molecule.molecular_weight_da,
                        )

        if self.ddi_model is None and len(self.drugs) > 1:
            from helixlang.plugins.human.ddi import create_default_ddi_model
            self.ddi_model = create_default_ddi_model()


# ============================================================================
# Result
# ============================================================================


@dataclass
class VirtualPatientResult:
    """Complete time-series output of a virtual-patient simulation."""

    time_h: list[float] = field(default_factory=list)

    # Vitals
    systolic_bp: list[float] = field(default_factory=list)
    diastolic_bp: list[float] = field(default_factory=list)
    heart_rate: list[float] = field(default_factory=list)
    temperature: list[float] = field(default_factory=list)
    weight_kg: list[float] = field(default_factory=list)
    spo2_pct: list[float] = field(default_factory=list)
    respiratory_rate: list[float] = field(default_factory=list)

    # Clinical labs
    alt: list[float] = field(default_factory=list)
    ast: list[float] = field(default_factory=list)
    creatinine: list[float] = field(default_factory=list)
    egfr: list[float] = field(default_factory=list)
    wbc: list[float] = field(default_factory=list)
    hemoglobin: list[float] = field(default_factory=list)
    platelets: list[float] = field(default_factory=list)
    glucose: list[float] = field(default_factory=list)
    hba1c: list[float] = field(default_factory=list)
    crp: list[float] = field(default_factory=list)
    bilirubin: list[float] = field(default_factory=list)
    albumin: list[float] = field(default_factory=list)
    inr: list[float] = field(default_factory=list)
    sodium: list[float] = field(default_factory=list)
    potassium: list[float] = field(default_factory=list)
    lactate: list[float] = field(default_factory=list)
    # Electrolytes
    calcium: list[float] = field(default_factory=list)
    phosphate: list[float] = field(default_factory=list)
    chloride: list[float] = field(default_factory=list)
    bicarbonate: list[float] = field(default_factory=list)
    # Lipids
    ldl: list[float] = field(default_factory=list)
    hdl: list[float] = field(default_factory=list)
    triglycerides: list[float] = field(default_factory=list)
    # ECG
    qtc_ms: list[float] = field(default_factory=list)

    # doc/30-31 new channels
    cortisol: list[float] = field(default_factory=list)
    insulin: list[float] = field(default_factory=list)
    glucose_endocrine: list[float] = field(default_factory=list)
    tsh: list[float] = field(default_factory=list)
    ft4: list[float] = field(default_factory=list)
    il6: list[float] = field(default_factory=list)
    tnf_alpha: list[float] = field(default_factory=list)
    neutrophils: list[float] = field(default_factory=list)
    tumor_volume: list[float] = field(default_factory=list)
    tumor_clone_fractions: list[dict[str, float]] = field(default_factory=list)
    resistance_mutations: list[list[str]] = field(default_factory=list)
    nephron_mass: list[float] = field(default_factory=list)
    fibrosis_stage: list[float] = field(default_factory=list)
    beta_cell_function: list[float] = field(default_factory=list)
    # doc/33 new disease-specific state channels
    fev1_percent: list[float] = field(default_factory=list)
    cd4_count: list[float] = field(default_factory=list)
    acid_secretion: list[float] = field(default_factory=list)
    mucosal_integrity: list[float] = field(default_factory=list)
    t4_level: list[float] = field(default_factory=list)
    # doc/33 CV/Neuro ODE feedback channels
    cardiac_output: list[float] = field(default_factory=list)
    map_mmhg: list[float] = field(default_factory=list)
    synaptic_density: list[float] = field(default_factory=list)
    cognitive_score: list[float] = field(default_factory=list)

    # Drug concentrations (per drug)
    drug_concentrations: dict[str, list[float]] = field(default_factory=dict)

    # Disease
    disease_severity: list[float] = field(default_factory=list)
    disease_stage: list[str] = field(default_factory=list)

    # Metabolism
    metabolite_pools: dict[str, list[float]] = field(default_factory=dict)

    # Events
    clinical_events: list[dict[str, Any]] = field(default_factory=list)
    ddi_alerts: list[dict[str, Any]] = field(default_factory=list)

    # Summary metrics
    auc_plasma: dict[str, float] = field(default_factory=dict)
    time_in_therapeutic_range_fraction: float = 0.0
    overall_efficacy_score: float = 0.0
    total_toxicity_events: int = 0
    max_alt: float = 0.0
    max_creatinine: float = 0.0
    min_wbc: float = 0.0
    min_egfr: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dictionary."""
        return {
            "time_h": self.time_h,
            "vitals": {
                "systolic_bp_mmhg": self.systolic_bp,
                "diastolic_bp_mmhg": self.diastolic_bp,
                "heart_rate_bpm": self.heart_rate,
                "temperature_c": self.temperature,
                "weight_kg": self.weight_kg,
                "spo2_pct": self.spo2_pct,
                "respiratory_rate": self.respiratory_rate,
                "qtc_ms": self.qtc_ms,
            },
            "endocrine": {
                "cortisol_ug_dl": self.cortisol,
                "insulin_uuml": self.insulin,
                "glucose_endocrine_mg_dl": self.glucose_endocrine,
                "tsh_miul": self.tsh,
                "ft4_ngdl": self.ft4,
            },
            "immune": {
                "il6_pg_ml": self.il6,
                "tnf_alpha_pg_ml": self.tnf_alpha,
                "neutrophils_x1000": self.neutrophils,
            },
            "disease_ode": {
                "tumor_volume": self.tumor_volume,
                "tumor_clone_fractions": self.tumor_clone_fractions,
                "resistance_mutations": self.resistance_mutations,
                "nephron_mass": self.nephron_mass,
                "fibrosis_stage": self.fibrosis_stage,
                "beta_cell_function": self.beta_cell_function,
                "fev1_percent": self.fev1_percent,
                "cd4_count": self.cd4_count,
                "acid_secretion": self.acid_secretion,
                "mucosal_integrity": self.mucosal_integrity,
                "t4_level": self.t4_level,
                "cardiac_output_l_min": self.cardiac_output,
                "map_mmhg": self.map_mmhg,
                "synaptic_density": self.synaptic_density,
                "cognitive_score": self.cognitive_score,
            },
            "labs": {
                "alt_u_per_l": self.alt,
                "ast_u_per_l": self.ast,
                "creatinine_mg_per_dl": self.creatinine,
                "egfr_ml_per_min": self.egfr,
                "wbc_per_ul": self.wbc,
                "hemoglobin_g_per_dl": self.hemoglobin,
                "platelets_per_ul": self.platelets,
                "glucose_mg_per_dl": self.glucose,
                "hba1c_pct": self.hba1c,
                "crp_mg_per_l": self.crp,
                "bilirubin_mg_per_dl": self.bilirubin,
                "albumin_g_per_dl": self.albumin,
                "inr": self.inr,
                "lactate_mmol_per_l": self.lactate,
                "calcium_mg_per_dl": self.calcium,
                "phosphate_mg_per_dl": self.phosphate,
                "chloride_meq_per_l": self.chloride,
                "bicarbonate_meq_per_l": self.bicarbonate,
                "ldl_mg_per_dl": self.ldl,
                "hdl_mg_per_dl": self.hdl,
                "triglycerides_mg_per_dl": self.triglycerides,
            },
            "drug_concentrations": self.drug_concentrations,
            "disease": {
                "severity": self.disease_severity,
                "stage": self.disease_stage,
            },
            "metabolite_pools": self.metabolite_pools,
            "clinical_events": self.clinical_events,
            "ddi_alerts": self.ddi_alerts,
            "summary": {
                "auc_plasma": self.auc_plasma,
                "time_in_therapeutic_range_fraction": self.time_in_therapeutic_range_fraction,
                "overall_efficacy_score": self.overall_efficacy_score,
                "total_toxicity_events": self.total_toxicity_events,
                "max_alt": self.max_alt,
                "max_creatinine": self.max_creatinine,
                "min_wbc": self.min_wbc,
                "min_egfr": self.min_egfr,
            },
        }

    def summary(self) -> str:
        """Human-readable one-page summary."""
        lines = [
            "=== Virtual Patient Simulation Summary ===",
            f"Duration: {self.time_h[-1]:.1f} h ({self.time_h[-1] / 24:.1f} days)" if self.time_h else "No data",
            "",
            "--- Vitals ---",
        ]
        if self.systolic_bp:
            lines.append(f"  BP range: {min(self.systolic_bp):.0f}-{max(self.systolic_bp):.0f} / "
                         f"{min(self.diastolic_bp):.0f}-{max(self.diastolic_bp):.0f} mmHg")
            lines.append(f"  HR range: {min(self.heart_rate):.0f}-{max(self.heart_rate):.0f} bpm")
            lines.append(f"  Temp range: {min(self.temperature):.1f}-{max(self.temperature):.1f} °C")
            lines.append(f"  Weight: {self.weight_kg[0]:.1f} → {self.weight_kg[-1]:.1f} kg" if self.weight_kg else "")

        lines += ["", "--- Labs ---"]
        if self.alt:
            lines.append(f"  ALT: {self.alt[0]:.0f} → max {self.max_alt:.0f} U/L")
        if self.creatinine:
            lines.append(f"  Creatinine: {self.creatinine[0]:.2f} → max {self.max_creatinine:.2f} mg/dL")
        if self.egfr:
            lines.append(f"  eGFR: {self.egfr[0]:.0f} → min {self.min_egfr:.0f} mL/min")
        if self.wbc:
            lines.append(f"  WBC: {self.wbc[0]:.0f} → min {self.min_wbc:.0f} /uL")

        lines += ["", "--- Disease ---"]
        if self.disease_stage:
            lines.append(f"  Stage: {self.disease_stage[0]} → {self.disease_stage[-1]}")
        if self.disease_severity:
            lines.append(f"  Severity: {self.disease_severity[0]:.2f} → {self.disease_severity[-1]:.2f}")

        lines += ["", "--- Drug Exposure ---"]
        for drug_name, concs in self.drug_concentrations.items():
            if concs:
                cmax = max(concs)
                lines.append(f"  {drug_name}: Cmax={cmax:.2f} uM")

        lines += ["", "--- Events ---"]
        lines.append(f"  Total toxicity events: {self.total_toxicity_events}")
        lines.append(f"  DDI alerts: {len(self.ddi_alerts)}")
        lines.append(f"  Efficacy score: {self.overall_efficacy_score:.2f}")
        return "\n".join(lines)


# ============================================================================
# Virtual Patient Engine
# ============================================================================


class VirtualPatient:
    """Unified virtual-patient simulation engine.

    Couples genotype → physiology → PBPK → PD → clinical labs → vital signs
    → disease progression → DDI → recovery into a single hourly integration loop.

    doc/30-31 extensions:
    - Endocrine axes (insulin-glucose, HPA, HPT)
    - QSP binding (mass-action, TMDD, competitive)
    - Innate immune ABM (cytokines, CRP/WBC)
    - Organ crosstalk (cross-organ coupling)
    - Per-disease ODE models (8 categories)
    - Hematology/renal integration
    """

    def __init__(self, config: VirtualPatientConfig) -> None:
        self.config = config
        self._physiology = self._build_physiology()
        self._labs_model = self._build_labs_model()
        self._vitals_model = VitalsModel.create_from_physiology(self._physiology)
        self._disease_model = self._build_disease_model()
        self._recovery_model = self._build_recovery_model()
        self._ddi_model = config.ddi_model or DDIModel()
        self._drug_engine: dict[str, _DrugPBPK] = {}
        self._biomarker_state: dict[str, float] = {}
        self._toxicity_flags: dict[str, list[float]] = {}
        self._treatment_active = True
        # doc/30-31 new modules
        self._endocrine = self._build_endocrine()
        self._immune: InnateImmuneModel | None = None
        self._crp_driver: CRPDriver | None = None
        self._qsp_binding = self._build_qsp_binding()
        self._organ_crosstalk = create_crosstalk()
        self._disease_ode = self._build_disease_ode()
        self._hematology: HematologySystem | None = None
        self._renal: RenalFunctionModel | None = None
        # doc/32 — physiology constraints for state validation
        from helixlang.plugins.human.physiology_constraints import PhysiologyConstraints
        self._physiology_constraints = PhysiologyConstraints()
        # doc/32 — stochastic ODE for inter-individual variability
        self._sde_config: Any = None
        self._stochastic_active = False
        self._sde_seed: int | None = None
        self._rng: Any = None
        # doc/32 §8.1 — Kalman denoising of reported series (opt-in)
        self._denoise_outputs = False
        self._assay_cv = 0.15
        # doc/32 — mechanistic DDI predictor (augments rule-based DDIModel)
        self._mech_ddi_predictor: Any = None
        # doc/32 §7.4 — tissue-specific GEM organ coupler
        self._gem_coupler: Any = None
        # doc/32 §7.5 — reduced-order organ models for spatial gradients
        self._ro_organs: dict[str, Any] = {}
        # doc/32 §7.6 — pharmacogenomic AE predictor
        self._ae_predictor: Any = None
        # doc/33 — WBC recovery tracking (bone marrow kinetics)
        self._wbc_current: float = 0.0
        self._wbc_target: float = 0.0
        self._wbc_initialized = False

    # ------------------------------------------------------------------
    # doc/32 public simulation controls
    # ------------------------------------------------------------------

    def enable_stochastic(
        self, config: Any | None = None, seed: int | None = None
    ) -> None:
        """Activate SDE lab noise (doc/32 §8.2) for inter-individual variability."""
        if config is None:
            if _stochastic_ode is None:
                _import_doc32()
            if _stochastic_ode is not None:
                config = _stochastic_ode()
        self._sde_config = config
        self._stochastic_active = config is not None
        self._sde_seed = seed

    def disable_stochastic(self) -> None:
        """Deactivate SDE lab noise."""
        self._sde_config = None
        self._stochastic_active = False
        self._sde_seed = None
        self._rng = None

    def enable_denoising(self, assay_cv: float = 0.15) -> None:
        """Kalman-denoise reported alt/creatinine/wbc series (doc/32 §8.1)."""
        self._denoise_outputs = True
        self._assay_cv = assay_cv

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_physiology(self) -> HumanPhysiology:
        calc = PhenotypeCalculator(
            genotype=self.config.genotype,
            traits=self.config.traits,
        )
        return calc.compute_physiology()

    def _build_labs_model(self) -> ClinicalLabModel:
        baseline = ClinicalLabModel.compute_baseline_from_physiology(
            self._physiology,
            disease=self.config.disease,
        )
        # Apply disease metabolite perturbations from #disease_metabolite
        # annotations: interpolate between normal and target based on severity.
        if self.config.disease is not None:
            severity = self.config.disease.severity
            # metabolite_perturbations store values in mmol/L (the _mm suffix).
            # Clinical lab fields may use different units — conversion factors
            # are applied where needed (e.g. glucose: mmol/L → mg/dL × 18.016).
            _GLUCOSE_MMG_TO_MGDL = 18.016
            met_map = {
                "crp": ("crp_mg_per_l", 1.0),
                "lactate": ("lactate_mmol_per_l", 1.0),
                "procalcitonin": ("crp_mg_per_l", 1.0),
                "wbc": ("wbc_per_ul", 1000.0),  # ×1000: mmol→cells/µL approximation
                "hemoglobin": ("hemoglobin_g_per_dl", 1.0),
                "platelets": ("platelets_per_ul", 1000.0),
                "alt": ("alt_u_per_l", 1.0),
                "creatinine": ("creatinine_mg_per_dl", 0.0113),  # mmol→mg/dL
                "glucose": ("glucose_mg_per_dl", _GLUCOSE_MMG_TO_MGDL),
                "bilirubin": ("bilirubin_total_mg_per_dl", 0.0585),  # mmol→mg/dL
                "albumin": ("albumin_g_per_dl", 10.0),  # mmol→g/dL approx
            }
            for mp in self.config.disease.metabolite_perturbations:
                mapping = met_map.get(mp.metabolite_id)
                if mapping is None:
                    continue
                field_name, conv = mapping
                target_mm = mp.normal_concentration_mm + severity * (
                    mp.initial_concentration_mm - mp.normal_concentration_mm
                )
                target_lab = target_mm * conv
                if mp.metabolite_id == "procalcitonin":
                    # PCT tracks with CRP — boost CRP proportionally
                    current = getattr(baseline, field_name, 1.0)
                    setattr(baseline, field_name, max(current, target_lab))
                else:
                    setattr(baseline, field_name, target_lab)
        return ClinicalLabModel(baseline=baseline, physiology=self._physiology)

    def _build_disease_model(self) -> DiseaseProgressionModel | None:
        if self.config.disease is None:
            return None
        name = self.config.disease.name.lower().replace(" ", "_").replace("-", "_")
        cat = (self.config.disease.category or "").lower().strip()
        # Try canonical profiles by name
        for key in ("CKD", "DIABETES_T2", "LIVER_CIRRHOSIS", "CANCER_GENERIC"):
            if key.lower() in name or name in key.lower():
                return create_progression_model(key)
        # Category-based dispatch
        cat_to_profile = {
            "infectious": "INFECTIOUS",
            "autoimmune": "AUTOIMMUNE",
            "cardiovascular": "CARDIOVASCULAR",
            "respiratory": "RESPIRATORY",
            "neurological": "NEUROLOGICAL",
            "metabolic": "DIABETES_T2",
            "oncology": "CANCER_GENERIC",
            "hepatic": "LIVER_CIRRHOSIS",
            "liver": "LIVER_CIRRHOSIS",
            "renal": "CKD",
            "kidney": "CKD",
            "hematological": "HEMATOLOGICAL",
            "gastrointestinal": "GASTROINTESTINAL",
            "endocrine": "ENDOCRINE",
        }
        canonical = cat_to_profile.get(cat)
        if canonical and canonical in _DISEASE_PROGRESSION_PROFILES:
            return _build_disease_progression_model(canonical, self.config.disease.severity)
        # Generic fallback: use config severity instead of hardcoded 0.2
        return _build_generic_progression_model(self.config.disease.severity)

    def _build_recovery_model(self) -> RecoveryModel | None:
        if not self.config.drugs:
            return None
        drug_names = [d.molecule.name for d in self.config.drugs]
        baseline = {
            "ALT": self._labs_model.baseline.alt_u_per_l,
            "creatinine": self._labs_model.baseline.creatinine_mg_per_dl,
            "WBC": self._labs_model.baseline.wbc_per_ul,
            "hemoglobin": self._labs_model.baseline.hemoglobin_g_per_dl,
        }
        return create_recovery_model(drug_names, baseline)

    def _build_endocrine(self) -> EndocrineSystem | None:
        """Build endocrine axes based on disease state and config."""
        from helixlang.plugins.human.endocrine import create_endocrine
        diabetes_sev = 0.0
        addison_sev = 0.0
        cushing_sev = 0.0
        stress_level = 0.0
        if self.config.disease is not None:
            name = self.config.disease.name.lower()
            if "diabetes" in name or "t2d" in name:
                diabetes_sev = self.config.disease.severity
            elif "addison" in name:
                addison_sev = self.config.disease.severity
            elif "cushing" in name:
                cushing_sev = self.config.disease.severity
        for cfg in self.config.endocrine_configs:
            axis = cfg.get("axis", "")
            severity = cfg.get("severity", 0.0)
            if "diabetes" in axis:
                diabetes_sev = max(diabetes_sev, severity)
            elif "addison" in axis:
                addison_sev = max(addison_sev, severity)
            elif "cushing" in axis:
                cushing_sev = max(cushing_sev, severity)
            elif "stress" in axis:
                stress_level = cfg.get("level", 0.0)
        return create_endocrine(
            diabetes_severity=diabetes_sev,
            addison_severity=addison_sev,
            cushing_severity=cushing_sev,
            stress_level=stress_level,
        )

    def _build_qsp_binding(self) -> QSPBindingSystem | None:
        """Build QSP binding models from config and DSL annotations."""
        from helixlang.plugins.human.qsp_binding import create_qsp_binding
        sys = create_qsp_binding()
        for binding in self.config.qsp_bindings:
            drug = binding.get("drug", "")
            kind = binding.get("kind", "mass_action")
            if kind == "tmdd":
                sys.add_tmdd(drug, kss_nM=binding.get("kss_nM", 5.0),
                             emax=binding.get("emax", 1.0))
            elif kind == "mass_action":
                sys.add_mass_action(drug, kd_nM=binding.get("kd_nM", 10.0),
                                    emax=binding.get("emax", 1.0))
            elif kind == "competitive":
                sys.add_competitive(drug,
                                    kd_agonist_nM=binding.get("kd_agonist_nM", 10.0),
                                    ki_antagonist_nM=binding.get("ki_antagonist_nM", 5.0),
                                    emax=binding.get("emax", 1.0))
        if sys.models:
            return sys
        return None

    def _build_disease_ode(self) -> Any:
        """Build per-disease ODE model."""
        from helixlang.plugins.human.disease_ode_models import create_disease_model
        if self.config.disease is None:
            return None
        ode = create_disease_model(
            self.config.disease.name,
            self.config.disease.severity,
            category=self.config.disease.category,
        )
        # Wire tumor heterogeneity from biopsy (doc/33 Phase 4)
        if hasattr(ode, 'tumor_volume') and self.config.tumor_biopsy is not None:
            from helixlang.plugins.human.disease_ode_models import (
                CancerODE,
                TumorClone,
                TumorHeterogeneity,
            )
            assert isinstance(ode, CancerODE)
            biopsy = self.config.tumor_biopsy
            mutations = biopsy.get("mutations", [])
            amplifications = biopsy.get("amplifications", [])
            fusion_genes = biopsy.get("fusion_genes", [])
            pd_l1 = biopsy.get("pd_l1_expression", 0.0)
            msi = biopsy.get("msi_status", "MSS")
            tmb = biopsy.get("tmb_per_mb", 0.0)
            clones = []
            parent_sens: dict[str, float] = {}
            if any("EGFR" in m for m in mutations):
                parent_sens["egfr"] = 0.9
            if any("BRAF" in m for m in mutations):
                parent_sens["braf"] = 0.85
            if any("KRAS" in m for m in mutations):
                parent_sens["kras"] = 0.1
            if any("ALK" in f for f in fusion_genes):
                parent_sens["alk"] = 0.9
            if any("HER2" in a for a in amplifications):
                parent_sens["her2"] = 0.7
            if pd_l1 >= 0.5:
                parent_sens["pd_l1"] = 0.8
            if msi == "MSI-H":
                parent_sens["pd_l1"] = max(parent_sens.get("pd_l1", 0.0), 0.85)
            if any("BRCA" in m for m in mutations):
                parent_sens["parp"] = 0.9
            parent_growth = ode.growth_rate * (1.0 + 0.001 * tmb)
            clones.append(TumorClone(
                name="parent",
                fraction=1.0,
                growth_rate=parent_growth,
                drug_sensitivities=parent_sens,
            ))
            ode.heterogeneity = TumorHeterogeneity(clones=clones)
        return ode

    def _init_immune_if_needed(self) -> None:
        """Lazily initialize immune model (needs cortisol from endocrine)."""
        if self._immune is not None:
            return
        from helixlang.plugins.human.immune import create_immune_model
        cortisol = self._endocrine.get_cortisol_ug_dl() if self._endocrine else 12.0
        infection = 0.0
        autoimmune = 0.0
        immunosuppression = 0.0
        if self.config.disease is not None:
            name = self.config.disease.name.lower()
            category = self.config.disease.category.lower()
            if ("infection" in name or "sepsis" in name or "pneumonia" in name
                    or category == "infectious"):
                infection = self.config.disease.severity
            elif "autoimmune" in name or "lupus" in name or "vasculitis" in name:
                autoimmune = self.config.disease.severity
        for cfg in self.config.immune_configs:
            infection = max(infection, cfg.get("infection_severity", 0.0))
            autoimmune = max(autoimmune, cfg.get("autoimmune_activation", 0.0))
            immunosuppression = max(immunosuppression, cfg.get("immunosuppression", 0.0))
        self._immune, self._crp_driver = create_immune_model(
            infection_severity=infection,
            autoimmune_activation=autoimmune,
            cortisol_level=cortisol,
            immunosuppression=immunosuppression,
        )
        # Store base values for dynamic feedback during simulation
        self._immune._base_infection_severity = infection  # type: ignore[attr-defined]
        self._immune._base_autoimmune_activation = autoimmune  # type: ignore[attr-defined]
        # Seed CRP driver from disease metabolite perturbations
        if self._crp_driver is not None and self.config.disease is not None:
            crp_seeded = False
            for mp in self.config.disease.metabolite_perturbations:
                if mp.metabolite_id == "crp":
                    severity = self.config.disease.severity
                    target_crp = mp.normal_concentration_mm + severity * (
                        mp.initial_concentration_mm - mp.normal_concentration_mm
                    )
                    self._crp_driver.crp_mg_l = target_crp
                    crp_seeded = True
                    break
            # Fallback: seed CRP for autoimmune/inflammatory diseases without
            # explicit #disease_metabolite id=crp annotation
            if not crp_seeded and autoimmune > 0.0:
                severity = self.config.disease.severity
                self._crp_driver.crp_mg_l = 2.0 + severity * 45.0  # 2-47 mg/L

    def _init_hematology_renal_if_needed(self) -> None:
        """Lazily initialize hematology + renal models."""
        if self._hematology is not None:
            return
        try:
            from helixlang.plugins.human.hematology_model import create_hematology_system
            from helixlang.plugins.human.renal_model import create_renal_model
            self._hematology = create_hematology_system()
            self._renal = create_renal_model()
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "VirtualPatient needs helixlang.plugins.human.hematology_model and "
                "helixlang.plugins.human.renal_model. Reinstall with: "
                "pip install --force-reinstall helixlang"
            ) from exc

    def _init_drug_engines(self) -> None:
        for drug in self.config.drugs:
            key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
            self._drug_engine[key] = _DrugPBPK(drug, self._physiology)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> VirtualPatientResult:
        """Run the full virtual-patient simulation and return time-series."""
        total_h = self.config.total_duration_days * 24.0
        dt_h = self.config.dfa_dt_h
        record_interval = self.config.output_time_resolution_h
        n_steps = int(total_h / dt_h)

        self._init_drug_engines()
        self._init_immune_if_needed()
        self._init_hematology_renal_if_needed()
        # doc/32 — initialize mechanistic DDI predictor
        _import_doc32()
        if _mechanistic_ddi is not None:
            lib_cls, predictor_cls = _mechanistic_ddi[1], _mechanistic_ddi[0]
            lib = lib_cls()
            self._mech_ddi_predictor = predictor_cls(library=lib)
        # doc/32 §7.4 — tissue GEM coupler
        if _tissue_gem is not None:
            _gem_classes = _tissue_gem
            coupler_cls = _gem_classes[1]
            self._gem_coupler = coupler_cls()
        # doc/32 §7.5 — reduced-order organ models (spatial gradients)
        if _reduced_order is not None:
            _pod_gen = _reduced_order
            self._ro_organs = _pod_gen.generate_all()
        # doc/32 §7.6 — pharmacogenomic AE predictor
        if _pharmacogenomic_ae is not None:
            _ae_cls = _pharmacogenomic_ae
            self._ae_predictor = _ae_cls()
        # doc/32 §7.7 — proteome-wide binding cascade
        self._proteome_cascade = None
        if _proteome_binding is not None:
            self._proteome_cascade = _proteome_binding()
        # microbiome-drug interaction compartment
        self._microbiome_compartment: Any = None
        if _microbiome is not None:
            self._microbiome_compartment = _microbiome()
        # emergent complexity (epigenetics + liver-gut + stress-immune-endocrine)
        self._emergent_complexity = None
        if _emergent_complexity is not None:
            self._emergent_complexity = _emergent_complexity()
        # Register drug SMILES with labs model for structure-based toxicity prediction
        for drug in self.config.drugs:
            key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
            smiles = getattr(drug.molecule, "smiles", None)
            if smiles:
                self._labs_model.register_drug_smiles(key, smiles)
        result = VirtualPatientResult()
        self._pending_clearance_scale: dict[str, float] = {}
        self._rng = (
            random.Random(self._sde_seed)
            if self._stochastic_active and self._sde_config is not None
            else None
        )

        t_h = 0.0
        next_record = 0.0

        for step_i in range(n_steps + 1):
            t_h = step_i * dt_h

            # --- Endocrine axes (with circadian cortisol modulation) ---
            if self._endocrine is not None:
                clock_h = t_h % 24.0  # circadian cycle from elapsed time
                self._endocrine.step(dt_h, clock_hour=clock_h)

            # --- DDI ---
            cyp_profile = {
                enz: self.config.genotype.get_cyp_activity(enz)
                for enz in CORE_CYP_ENZYMES
            }
            drug_names = [d.molecule.name for d in self.config.drugs]
            # Pass an empty CYP profile: genotype-only effects are applied
            # separately via _compute_genetic_cyp_modifier below, so the DDI
            # model must report only true drug-drug interaction modifiers
            # (enzyme-state rules and the genotype second pass stay inert).
            clearance_mods = self._ddi_model.compute_clearance_modifiers(
                drug_names, {},
            )

            # --- doc/32 §8.3: Mechanistic DDI augmentation ---
            if self._mech_ddi_predictor is not None and len(drug_names) > 1:
                mech_names = [
                    n.lower().replace(" ", "_").replace("-", "_")
                    for n in drug_names
                ]
                mech_ddi_results = self._mech_ddi_predictor.predict_all_pairs(
                    mech_names,
                )
                for pred in mech_ddi_results:
                    if pred.significance not in ("DDI_ALERT", "CONTRAINDICATED"):
                        continue
                    # propagate AUC ratio as clearance reduction for the victim
                    victim = pred.drug_b.lower()
                    for dname in drug_names:
                        key = dname.lower().replace(" ", "_").replace("-", "_")
                        if key == victim:
                            clearance_mods[dname] = (
                                clearance_mods.get(dname, 1.0)
                                / max(pred.auc_ratio, 1.0)
                            )

            # --- Genetic CYP modulation on each drug ---
            for drug in self.config.drugs:
                key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                engine = self._drug_engine.get(key)
                if engine is None:
                    continue
                # Apply CYP-based clearance scaling, plus any crosstalk/ROM
                # effects queued on the previous tick (they are computed after
                # the PBPK advance and take effect one tick later).
                cyp_mod = _compute_genetic_cyp_modifier(drug, self.config.genotype)
                tp_mod = _compute_transporter_modifier(drug, self.config.genotype)
                nc_mod = _compute_non_cyp_modifier(drug, self.config.genotype)
                ddi_mod = clearance_mods.get(drug.molecule.name, 1.0)
                pending = self._pending_clearance_scale.pop(key, 1.0)
                engine.clearance_modifier = cyp_mod * tp_mod * nc_mod * ddi_mod * pending

            # --- PBPK advance ---
            drug_concs: dict[str, float] = {}
            for drug in self.config.drugs:
                key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                engine = self._drug_engine.get(key)
                if engine is None:
                    continue
                engine.advance(dt_h, t_h)
                drug_concs[key] = engine.get_central_concentration()
            # Invalidate GEM pool on dose events (doc/33)
            if self._gem_coupler is not None and any(
                len(e._doses_given) > getattr(e, '_last_dose_count', 0)
                for e in self._drug_engine.values()
            ):
                self._gem_coupler.invalidate_on_dose()
            for e in self._drug_engine.values():
                e._last_dose_count = len(e._doses_given)

            # --- PD flux modifiers ---
            pd_multipliers: dict[str, float] = {}
            for drug in self.config.drugs:
                key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                pd = self.config.pharmacodynamics.get(key)
                if pd is not None and pd.effects:
                    for eff in pd.effects:
                        conc = drug_concs.get(key, 0.0)
                        frac = hill_equation(
                            conc, eff.ec50_um, eff.emax,
                            eff.hill_coefficient, eff.baseline_effect,
                        )
                        if eff.effect_type == "inhibition":
                            pd_multipliers[eff.target_reaction] = pd_multipliers.get(
                                eff.target_reaction, 1.0,
                            ) * (1.0 - frac)
                        else:
                            pd_multipliers[eff.target_reaction] = pd_multipliers.get(
                                eff.target_reaction, 0.0,
                            ) + frac * eff.emax

            # --- Disease progression ---
            disease_sev = 0.0
            disease_stage_str = "healthy"
            if self._disease_model is not None and self.config.disease is not None:
                effectiveness = self._compute_treatment_effectiveness(
                    drug_concs, pd_multipliers,
                )
                labs_snap = self._labs_model.get_current()
                prog_labs = labs_snap.to_progression_labs()
                from helixlang.plugins.human.disease_progression import ClinicalLabs as ProgLabs
                prog_labs_obj = ProgLabs(**{
                    k: v for k, v in prog_labs.items()
                    if hasattr(ProgLabs, k)
                })
                stage = self._disease_model.step(dt_h, effectiveness, prog_labs_obj)
                disease_sev = self._disease_model.get_severity()
                disease_stage_str = stage.value

            # --- Clinical labs update ---
            labs = self._labs_model.update(
                dt_h, drug_concs, disease_sev,
                disease_name=self.config.disease_profile_name,
            )

            # --- Physiology constraints validation (doc/32 §7.3) ---
            if self._physiology_constraints is not None:
                vit = getattr(self._vitals_model, "current", None)
                state = {
                    "ph": 7.4,
                    "glucose": labs.glucose_mg_per_dl,
                    "creatinine": labs.creatinine_mg_per_dl,
                    "wbc": labs.wbc_per_ul,
                    "alt": labs.alt_u_per_l,
                    "ast": labs.ast_u_per_l,
                }
                if vit is not None:
                    # one-tick-lag vitals: the update runs later in this tick
                    state["temperature"] = vit.temperature_c
                    state["spo2"] = vit.spo2_pct
                    state["heart_rate"] = vit.heart_rate_bpm
                    state["map"] = vit.map_mmhg
                if self._endocrine is not None:
                    state["cortisol"] = self._endocrine.get_cortisol_ug_dl()
                if self._immune is not None:
                    state["tnf_alpha"] = self._immune.get_tnf()
                    state["il6"] = self._immune.get_il6()
                check = self._physiology_constraints.check(state)
                if not check.is_valid:
                    corrected = self._physiology_constraints.project_to_feasible(state)
                    labs.creatinine_mg_per_dl = corrected.get("creatinine", labs.creatinine_mg_per_dl)
                    labs.wbc_per_ul = corrected.get("wbc", labs.wbc_per_ul)
                    labs.alt_u_per_l = corrected.get("alt", labs.alt_u_per_l)

            # --- doc/32 §8.2: Stochastic ODE noise ---
            if (
                self._stochastic_active
                and self._sde_config is not None
                and self._rng is not None
            ):
                sigma_i = self._sde_config.sigma_intrinsic
                sigma_e = self._sde_config.sigma_extrinsic
                # intrinsic noise: σ√|X| for each lab
                for attr_name in ('alt_u_per_l', 'creatinine_mg_per_dl', 'wbc_per_ul'):
                    val = getattr(labs, attr_name, 0.0)
                    sigma = sigma_i * math.sqrt(abs(val) + 1e-6) + sigma_e * abs(val)
                    setattr(labs, attr_name, max(0.0, val + self._rng.gauss(0.0, sigma)))

            # --- Vital signs update ---
            vitals = self._vitals_model.update(dt_h, drug_concs, labs, disease_sev)

            # --- Hematology system (Friberg myelosuppression + erythropoiesis) ---
            if self._hematology is not None:
                heme_exposures: dict[str, float] = {}
                for drug in self.config.drugs:
                    dkey = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                    if dkey in drug_concs:
                        heme_exposures[dkey] = drug_concs[dkey]
                heme = self._hematology.step(dt_h, heme_exposures or None)
                labs.wbc_per_ul = max(100.0, heme["anc_x10e3_ul"] * 1000.0)
                labs.platelets_per_ul = max(1000.0, heme["platelets_x10e3_ul"] * 1000.0)
                labs.hemoglobin_g_per_dl = max(4.0, heme["hemoglobin_g_dl"])

            # --- Renal function model (CKD-EPI eGFR + AKI dynamics) ---
            if self._renal is not None:
                for drug in self.config.drugs:
                    dkey = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                    conc = drug_concs.get(dkey, 0.0)
                    # Only induce AKI if no active episode (prevents timer
                    # reset every hour which blocks recovery)
                    has_active_aki = (
                        hasattr(self._renal, "_aki")
                        and getattr(self._renal, "_aki", None) is not None
                    )
                    if conc > 1.0 and not has_active_aki and hasattr(self._renal, "induce_aki"):
                        _frac = max(0.01, min(0.30, conc * 0.003))
                        self._renal.induce_aki(
                            fractional_loss=_frac,
                            injury_duration_h=48.0,
                            recovery_fraction=0.85,
                        )
                reported_egfr = self._renal.step(dt_h)
                labs.egfr_ml_per_min = max(1.0, reported_egfr)
                labs.creatinine_mg_per_dl = max(0.3, self._renal.serum_creatinine)

            # --- Recovery ---
            # Check if all drugs have finished their treatment duration
            all_drugs_done = all(
                t_h >= d.duration_days * 24.0 for d in self.config.drugs
            ) if self.config.drugs else True

            if all_drugs_done and self._treatment_active:
                self._treatment_active = False
                if self._recovery_model is not None:
                    self._seed_recovery_from_labs(labs)
                    self._recovery_model.set_treatment_inactive()

            # Run recovery step
            if self._recovery_model is not None:
                recovery_biomarkers = self._recovery_model.step(dt_h, t_h)
                # Feed recovery biomarkers back into labs model
                self._feed_recovery_biomarkers(recovery_biomarkers)

            # --- Immune system ---
            if self._immune is not None:
                # Update cortisol from endocrine for immune suppression
                if self._endocrine is not None:
                    self._immune.cortisol_suppression = min(1.0, max(0.0,
                        (self._endocrine.get_cortisol_ug_dl() - 20.0) / 30.0))
                # Wire PD inflammation multiplier into immune suppression:
                # anti-inflammatory drugs (JAK inhibitors, NSAIDs, etc.)
                # suppress IL-6/TNF-α cytokine production.
                # Only count PD targets known to be anti-inflammatory;
                # prevents antibiotics/chemo from falsely suppressing IL-6/CRP.
                anti_inflam = 0.0
                for _pk, _pv in pd_multipliers.items():
                    if _pk in ANTI_INFLAMMATORY_PD_TARGETS and _pv < 1.0:
                        anti_inflam = max(anti_inflam, 1.0 - _pv)
                self._immune.anti_inflammatory = max(0.0, anti_inflam)
                self._immune.step(dt_h)
                # Wire immune neutrophil mobilisation into WBC during infection
                if self._hematology is not None and self.config.disease is not None:
                    heme_wbc = labs.wbc_per_ul
                    # Initialize tracking on first call
                    if not self._wbc_initialized:
                        self._wbc_current = heme_wbc
                        self._wbc_initialized = True
                    # Infection drives leukocytosis: scale WBC by disease severity
                    inf_sev = getattr(self._immune, 'infection_severity', 0.0)
                    auto_sev = getattr(self._immune, 'autoimmune_activation', 0.0)
                    active_sev = max(inf_sev, auto_sev * 0.5)
                    if active_sev > 0.1:
                        # WBC scales from 7000 (mild) to 25000 (severe sepsis)
                        self._wbc_target = 7000.0 + active_sev * 18000.0
                    else:
                        # Resolution: WBC recovers toward baseline via bone marrow
                        # Recovery half-life ~18h (fast for WBC vs slower for RBC)
                        self._wbc_target = 7000.0
                    # Bone marrow recovery kinetics: exponential approach
                    # Max production rate ~2500 cells/µL/h (neutrophil burst)
                    rate = max(0.01, min(0.08, dt_h / 18.0))  # ~18h half-life
                    self._wbc_current += (self._wbc_target - self._wbc_current) * rate
                    # Clamp floor
                    self._wbc_current = max(500.0, self._wbc_current)
                    labs.wbc_per_ul = max(heme_wbc, self._wbc_current)
                if self._crp_driver is not None:
                    self._crp_driver.step(dt_h, self._immune.get_il6())
                    # Override CRP from labs with immune-driven value
                    labs.crp_mg_per_l = self._crp_driver.crp_mg_l
                    # Cap CRP at disease-driven target to prevent overshoot for
                    # chronic autoimmune/inflammatory diseases.  Use explicit
                    # #disease_metabolite id=crp target when available, otherwise
                    # fall back to severity-based formula.
                    if self.config.disease is not None:
                        _crp_target = None
                        for _mp in self.config.disease.metabolite_perturbations:
                            if _mp.metabolite_id == "crp":
                                _sev = self.config.disease.severity
                                _crp_target = _mp.normal_concentration_mm + _sev * (
                                    _mp.initial_concentration_mm - _mp.normal_concentration_mm
                                )
                                break
                        if _crp_target is None:
                            # Fallback: severity-based CRP target
                            _crp_target = 1.0 + self.config.disease.severity * 50.0
                        # Allow 20% overshoot above disease target
                        labs.crp_mg_per_l = min(
                            labs.crp_mg_per_l, _crp_target * 1.2
                        )

            # --- QSP binding ---
            if self._qsp_binding is not None:
                for drug in self.config.drugs:
                    key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                    conc_nM = drug_concs.get(key, 0.0) * 1000.0  # µM → nM
                    self._qsp_binding.set_drug_concentration(key, conc_nM)
                self._qsp_binding.step(dt_h)
                # Feed QSP occupancy effects into PD multipliers
                for model_key, model in self._qsp_binding.models.items():
                    effect = model.compute_effect()
                    if effect > 0.0:
                        pd_multipliers[f"qsp_{model_key}"] = effect

            # --- Organ crosstalk ---
            self._organ_crosstalk = apply_crosstalk(
                self._organ_crosstalk,
                glucose_mg_dl=labs.glucose_mg_per_dl,
                egfr=labs.egfr_ml_per_min,
                cortisol_ug_dl=self._endocrine.get_cortisol_ug_dl() if self._endocrine else 12.0,
                albumin_g_dl=labs.albumin_g_per_dl,
                inr=labs.inr,
                il6_pg_ml=self._immune.get_il6() if self._immune else 1.0,
                tnf_pg_ml=self._immune.get_tnf() if self._immune else 5.0,
                phosphate_mg_dl=labs.phosphate_mg_per_dl,
            )
            # Apply crosstalk effects to labs
            if self._organ_crosstalk.clearance_modifier_from_liver != 1.0:
                for drug in self.config.drugs:
                    key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                    self._pending_clearance_scale[key] = (
                        self._pending_clearance_scale.get(key, 1.0)
                        * self._organ_crosstalk.clearance_modifier_from_liver
                    )

            # --- doc/32 §7.4: Tissue GEM metabolite exchange ---
            if self._gem_coupler is not None:
                organ_concs: dict[str, dict[str, float]] = {
                    "liver": {"glucose": labs.glucose_mg_per_dl / 18.0, "lactate": 1.0, "ammonia": 0.03},
                    "kidney": {"glucose": labs.glucose_mg_per_dl / 18.0, "creatinine": labs.creatinine_mg_per_dl * 0.01},
                    "brain": {"glucose": labs.glucose_mg_per_dl / 18.0 * 0.6, "ammonia": 0.02},
                    "muscle": {"glucose": labs.glucose_mg_per_dl / 18.0 * 0.8, "lactate": 1.5},
                    "adipose": {"glucose": labs.glucose_mg_per_dl / 18.0 * 0.5, "free_fatty_acids": 0.3},
                    "gi": {"glucose": labs.glucose_mg_per_dl / 18.0 * 0.7, "glutamine": 0.5},
                }
                updated_concs = self._gem_coupler.step(dt_h, organ_concs)
                # Multi-metabolite write-back from GEM exchange (doc/33)
                liver_glc = updated_concs.get("liver", {}).get("glucose", labs.glucose_mg_per_dl / 18.0)
                labs.glucose_mg_per_dl = max(40.0, min(500.0, liver_glc * 18.0))
                # Lactate from muscle→liver exchange
                muscle_lact = updated_concs.get("muscle", {}).get("lactate", 1.5)
                gem_lact = max(0.3, min(15.0, muscle_lact))
                # When disease metabolite perturbations set lactate (e.g. sepsis),
                # don't let the GEM coupler fully overwrite it. Use disease value
                # as the dominant signal, with GEM providing minor modulation.
                if self.config.disease is not None and self.config.disease.metabolite_perturbations:
                    has_lact_mp = any(
                        mp.metabolite_id == "lactate"
                        for mp in self.config.disease.metabolite_perturbations
                    )
                    if has_lact_mp:
                        labs.lactate_mmol_per_l = 0.8 * labs.lactate_mmol_per_l + 0.2 * gem_lact
                    else:
                        labs.lactate_mmol_per_l = gem_lact
                else:
                    labs.lactate_mmol_per_l = gem_lact

            # --- doc/32 §7.5: Reduced-order organ spatial effects ---
            for organ_name, ro_model in self._ro_organs.items():
                if hasattr(ro_model, 'step'):
                    drug_input = 0.0
                    for drug in self.config.drugs:
                        key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                        drug_input += drug_concs.get(key, 0.0) * 1000.0  # µM → nmol/h proxy
                    ro_model.step(dt_h, drug_input)
                    # Feed portal-central gradient into liver clearance modifier
                    if organ_name == "liver" and hasattr(ro_model, 'get_gradient'):
                        gradient = ro_model.get_gradient()
                        # Higher gradient → higher extraction ratio → modified clearance
                        if abs(gradient) > 0.1:
                            rom_factor = max(0.5, min(2.0, 1.0 + 0.1 * gradient))
                            for drug in self.config.drugs:
                                key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                                self._pending_clearance_scale[key] = (
                                    self._pending_clearance_scale.get(key, 1.0)
                                    * rom_factor
                                )

            # --- doc/32 §7.6: Pharmacogenomic AE prediction ---
            if self._ae_predictor is not None:
                cyp_activities = {
                    enz: self.config.genotype.get_cyp_activity(enz)
                    for enz in CORE_CYP_ENZYMES
                }
                ae_results = self._ae_predictor.predict_all(drug_concs, cyp_activities)
                for _drug_name, ae_preds in ae_results.items():
                    for pred in ae_preds:
                        if pred.ae_probability > 0.15:
                            # High AE risk → feed into organ-specific toxicity
                            if pred.target_organ.value == "liver":
                                labs.alt_u_per_l += dt_h * 0.05 * pred.toxicity_ratio
                            elif pred.target_organ.value == "kidney":
                                labs.creatinine_mg_per_dl += dt_h * 0.002 * pred.toxicity_ratio
                            elif pred.target_organ.value == "bone_marrow":
                                labs.wbc_per_ul *= (1.0 - dt_h * 0.001 * pred.toxicity_ratio)

            # --- doc/32 §7.7: Proteome-wide binding cascade ---
            if self._proteome_cascade is not None and self.config.drugs:
                for drug in self.config.drugs:
                    key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                    smiles = getattr(drug.molecule, "smiles", "")
                    if key in drug_concs and smiles:
                        binding_profile = self._proteome_cascade.screen_drug(key, smiles, drug_concs[key])
                        # UGT1A1 inhibition → reduced glucuronidation → ↑ drug levels
                        ugt_eff = binding_profile.inhibition_dict.get("UGT1A1", 0.0)
                        if ugt_eff > 0.1:
                            scale = max(0.5, 1.0 - 0.5 * ugt_eff)
                            drug_concs[key] *= scale
                # Inter-drug proteome DDI
                if len(self.config.drugs) >= 2:
                    d0 = self.config.drugs[0]
                    d1 = self.config.drugs[1]
                    k0 = d0.molecule.name.lower().replace(" ", "_").replace("-", "_")
                    k1 = d1.molecule.name.lower().replace(" ", "_").replace("-", "_")
                    s0 = getattr(d0.molecule, "smiles", "")
                    s1 = getattr(d1.molecule, "smiles", "")
                    if k0 in drug_concs and k1 in drug_concs and s0 and s1:
                        ddi_pred = self._proteome_cascade.predict_ddi(
                            k0, s0, drug_concs[k0], k1, s1, drug_concs[k1],
                        )
                        if ddi_pred.auc_ratio > 1.25:
                            drug_concs[k1] *= ddi_pred.auc_ratio
                # Write proteome-modified concentrations back into PBPK engines
                for _pw_drug in self.config.drugs:
                    _pk = _pw_drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                    if _pk in drug_concs and _pk in self._drug_engine:
                        self._drug_engine[_pk].conc_um["central"] = drug_concs[_pk]

            # --- Microbiome-drug interaction ---
            if self._microbiome_compartment is not None:
                for drug in self.config.drugs:
                    key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                    if key in drug_concs:
                        self._microbiome_compartment.set_drug_concentration(key, drug_concs[key])
                mi_effects = self._microbiome_compartment.step(dt_h)
                for dk, effect in mi_effects.items():
                    if dk in drug_concs:
                        drug_concs[dk] *= effect.bioavailability_modifier
                # Portal fluxes → liver impact
                portal = self._microbiome_compartment.get_portal_fluxes()
                ammonia_flux = portal.get("ammonia", 0.0)
                if ammonia_flux > 1.0:
                    labs.alt_u_per_l += dt_h * 0.01 * ammonia_flux
                scfa_flux = portal.get("scfa", 0.0)
                if scfa_flux > 5.0:
                    labs.alt_u_per_l -= dt_h * 0.005 * scfa_flux
                # Write microbiome-modified concentrations back into PBPK engines
                for dk in mi_effects:
                    if dk in drug_concs and dk in self._drug_engine:
                        self._drug_engine[dk].conc_um["central"] = drug_concs[dk]

            # --- Emergent complexity (epigenetics + liver-gut + stress-immune) ---
            if self._emergent_complexity is not None:
                il6_val = 5.0
                tnf_val = 10.0
                crp_val = 3.0
                if self._immune is not None:
                    il6_val = self._immune.get_il6()
                    tnf_val = self._immune.get_tnf()
                if self._crp_driver is not None:
                    crp_val = self._crp_driver.crp_mg_l
                bshe_act = 1.0
                if self._microbiome_compartment is not None:
                    bshe_act = self._microbiome_compartment.state.bile_salt_hydrolase_activity
                cortisol_val = 12.0
                if self._endocrine is not None and hasattr(self._endocrine, "get_cortisol_ug_dl"):
                    cortisol_val = self._endocrine.get_cortisol_ug_dl()
                emergent_signals = self._emergent_complexity.step(
                    dt_h=dt_h,
                    t_h=t_h,
                    drug_concentrations=drug_concs,
                    il6=il6_val,
                    tnf=tnf_val,
                    crp=crp_val,
                    bshe_activity=bshe_act,
                    cortisol_input=cortisol_val,
                )
                # Liver-gut feedback → ALT
                bile_acid_pool = emergent_signals.get("bile_acid_pool", 10.0)
                labs.alt_u_per_l += dt_h * 0.002 * (bile_acid_pool - 10.0)
                # Endotoxin → liver inflammation
                endotoxin = emergent_signals.get("endotoxin_level", 0.01)
                labs.alt_u_per_l += dt_h * 0.05 * endotoxin
                # Cortisol modulation of WBC
                cortisol_suppression = emergent_signals.get("cortisol_suppression", 0.0)
                labs.wbc_per_ul *= (1.0 - dt_h * 0.001 * cortisol_suppression)
                # Fever → metabolic rate
                fever = emergent_signals.get("fever_c", 0.0)
                if fever > 0.5:
                    labs.glucose_mg_per_dl += dt_h * 0.1 * fever
                # Wire epigenetic CYP modifiers into clearance (one-tick-lag)
                for _epi_gene in ("CYP1A2", "CYP2B6", "CYP2C8", "CYP2C9",
                                   "CYP2C19", "CYP2D6", "CYP2E1", "CYP3A4",
                                   "CYP3A5", "UGT1A1"):
                    _epi_val = emergent_signals.get(f"epigenetic_{_epi_gene}", 1.0)
                    for _epi_drug in self.config.drugs:
                        _epi_frac = _epi_drug.cyp_metabolism.get(_epi_gene, 0.0)
                        if _epi_frac > 0.0:
                            _dk = _epi_drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                            self._pending_clearance_scale[_dk] = (
                                self._pending_clearance_scale.get(_dk, 1.0)
                                * (0.5 + 0.5 * _epi_val)
                            )

            # --- Disease ODE model ---
            if self._disease_ode is not None:
                drug_eff = self._compute_treatment_effectiveness(drug_concs, pd_multipliers)
                if hasattr(self._disease_ode, 'step'):
                    if hasattr(self._disease_ode, 'glucose_mg_dl'):
                        # MetabolicT2DODE
                        self._disease_ode.step(dt_h, labs.glucose_mg_per_dl,
                                              self._endocrine.get_insulin_uuml() if self._endocrine else 10.0,
                                              drug_eff)
                        # Feed disease ODE state back into labs
                        hepatic_impairment = 1.0 - 0.3 * self._disease_ode.t2d_severity
                        labs.glucose_mg_per_dl = max(70.0, min(500.0,
                            labs.glucose_mg_per_dl * hepatic_impairment))
                    elif hasattr(self._disease_ode, 'nephron_mass'):
                        # RenalODE — wire ACEi/SGLT2 effects from PD multipliers
                        for _pk, _pv in pd_multipliers.items():
                            kl = _pk.lower()
                            if 'ace' in kl or 'angiotensin' in kl:
                                self._disease_ode.acei_effect = max(
                                    self._disease_ode.acei_effect, 1.0 - _pv)
                            if 'sglt2' in kl or 'glucose_reabsorption' in kl:
                                self._disease_ode.sglt2_effect = max(
                                    self._disease_ode.sglt2_effect, 1.0 - _pv)
                        self._disease_ode.step(dt_h)
                        # Nephron loss → rising creatinine, falling eGFR
                        nephron_frac = self._disease_ode.nephron_mass
                        labs.creatinine_mg_per_dl = max(0.5, labs.creatinine_mg_per_dl / max(0.1, nephron_frac))
                        labs.egfr_ml_per_min = max(5.0, labs.egfr_ml_per_min * nephron_frac)
                    elif hasattr(self._disease_ode, 'fibrosis_stage'):
                        # HepaticODE — wire antiviral/anti-fibrotic effects from PD
                        for _pk, _pv in pd_multipliers.items():
                            kl = _pk.lower()
                            if 'antiviral' in kl or 'viral' in kl or 'ns5a' in kl or 'protease' in kl:
                                self._disease_ode.antiviral_effect = max(
                                    self._disease_ode.antiviral_effect, 1.0 - _pv)
                            if 'anti_fibrotic' in kl or 'fibrosis' in kl or 'ppar' in kl:
                                self._disease_ode.anti_fibrotic_effect = max(
                                    self._disease_ode.anti_fibrotic_effect, 1.0 - _pv)
                        self._disease_ode.step(dt_h)
                        fib = self._disease_ode.fibrosis_stage
                        synth = self._disease_ode.synthetic_function
                        # Fibrosis → elevated liver enzymes, reduced albumin, rising INR
                        labs.alt_u_per_l += dt_h * 0.01 * fib
                        labs.ast_u_per_l += dt_h * 0.008 * fib
                        labs.albumin_g_per_dl = max(1.5, labs.albumin_g_per_dl * synth)
                        if fib > 2.0:
                            labs.inr = min(3.0, labs.inr + dt_h * 0.001 * (fib - 2.0))
                    elif hasattr(self._disease_ode, 'atherosclerosis_severity'):
                        # CardiovascularODE — wire drug effects on SVR & volume
                        drug_svr_mod = 1.0
                        drug_volume_mod = 1.0
                        for _cv_key, _cv_val in pd_multipliers.items():
                            k_lower = _cv_key.lower()
                            if 'svr' in k_lower or 'vasodilat' in k_lower or 'ace' in k_lower:
                                drug_svr_mod *= (1.0 - min(0.8, _cv_val))
                            if 'diuretic' in k_lower or 'volume' in k_lower:
                                drug_volume_mod *= (1.0 - min(0.5, _cv_val))
                        self._disease_ode.step(dt_h, drug_svr_mod, drug_volume_mod)
                        # CO deficit → elevated lactate (tissue hypoperfusion)
                        co_val = self._disease_ode.co_l_min
                        labs.lactate_mmol_per_l = max(0.5,
                            labs.lactate_mmol_per_l + dt_h * 0.01 * (1.0 - co_val / 5.0))
                    elif hasattr(self._disease_ode, 'tumor_volume'):
                        # CancerODE — per-pathway effects (doc/33 Phase 4)
                        from helixlang.plugins.human.disease_ode_models import TARGET_TO_PATHWAY
                        pathway_effects: dict[str, float] = {}
                        for _pk, _pv in pd_multipliers.items():
                            pathway = TARGET_TO_PATHWAY.get(_pk, "")
                            if not pathway:
                                kl = _pk.lower()
                                for tp, pw in TARGET_TO_PATHWAY.items():
                                    if tp.lower() in kl or kl in tp.lower():
                                        pathway = pw
                                        break
                            if pathway and _pv < 1.0:
                                inh = 1.0 - _pv
                                pathway_effects[pathway] = max(
                                    pathway_effects.get(pathway, 0.0), inh)
                        if pathway_effects:
                            self._disease_ode.pathway_effects = pathway_effects
                        self._disease_ode.step(dt_h)
                        # Record clone fractions and resistance
                        het = getattr(self._disease_ode, 'heterogeneity', None)
                        if het is not None:
                            clone_summary = het.get_clone_summary()
                            result.tumor_clone_fractions.append({
                                c["name"]: c["fraction"] for c in clone_summary
                            })
                            all_resist = []
                            for c in het.clones:
                                all_resist.extend(c.resistance_mutations)
                            result.resistance_mutations.append(sorted(set(all_resist)))
                        else:
                            result.tumor_clone_fractions.append({})
                            result.resistance_mutations.append([])
                    elif hasattr(self._disease_ode, 'joint_inflammation'):
                        # AutoimmuneRAODE
                        # Wire DMARD effect from PD multipliers into disease ODE
                        # Use anti-inflammatory PD targets (JAK, TNF, IL6R, COX2, etc.)
                        dmard_eff = 0.0
                        for _pk, _pv in pd_multipliers.items():
                            if _pk in ANTI_INFLAMMATORY_PD_TARGETS and _pv < 1.0:
                                dmard_eff = max(dmard_eff, 1.0 - _pv)
                        self._disease_ode.dmard_effect = min(1.0, dmard_eff)
                        self._disease_ode.step(dt_h)
                        # RA inflammation → systemic effects on labs
                        inflam = self._disease_ode.joint_inflammation
                        tnf_local = self._disease_ode.synovial_tnf
                        # Wire drug effect back into immune model so IL-6/CRP
                        # respond to disease activity reduction
                        if self._immune is not None:
                            base_auto = getattr(self._immune, '_base_autoimmune_activation',
                                                self.config.disease.severity if self.config.disease else 0.0)
                            self._immune.autoimmune_activation = base_auto * inflam
                        # CRP driven by inflammation (skip if CRP driver owns this channel)
                        if self._crp_driver is None:
                            labs.crp_mg_per_l = max(0.1, 0.5 + 5.0 * inflam * tnf_local / 50.0)
                        # Anemia of chronic disease (inflammation suppresses erythropoiesis)
                        labs.hemoglobin_g_per_dl = max(8.0, 14.0 - 2.5 * inflam)
                        # Low-grade fever from autoimmune inflammation
                        if inflam > 0.2:
                            vitals.temperature_c = min(38.5,
                                37.0 + 0.8 * inflam)
                    elif hasattr(self._disease_ode, 'synaptic_density'):
                        # NeurologicalODE — wire cholinesterase inhibition from PD
                        che_inhib = 0.0
                        neuroprotect = 0.0
                        for _neu_key, _neu_val in pd_multipliers.items():
                            k_lower = _neu_key.lower()
                            if 'cholinesterase' in k_lower or 'ache' in k_lower:
                                che_inhib = max(che_inhib, min(1.0, _neu_val))
                            if 'neuroprotect' in k_lower or 'nmda' in k_lower:
                                neuroprotect = max(neuroprotect, min(1.0, _neu_val))
                        self._disease_ode.cholinesterase_inhibition = che_inhib
                        self._disease_ode.disease_modifying_effect = neuroprotect
                        self._disease_ode.step(dt_h)
                        # Neuroinflammation → systemic CRP contribution (skip if CRP driver owns it)
                        if self._crp_driver is None:
                            neuro_inflam = self._disease_ode.neuroinflammation
                            labs.crp_mg_per_l = max(0.1,
                                labs.crp_mg_per_l + dt_h * 0.05 * neuro_inflam)
                    elif hasattr(self._disease_ode, 'stem_cell_pool'):
                        # HematologicalODE
                        self._disease_ode.step(dt_h)
                        # MDS → cytopenias
                        heme = self._disease_ode
                        labs.wbc_per_ul = max(500.0, 7000.0 * heme.stem_cell_pool)
                        labs.hemoglobin_g_per_dl = max(5.0, 14.0 * heme.stem_cell_pool)
                        labs.platelets_per_ul = max(10000.0, 250000.0 * heme.stem_cell_pool)
                    elif hasattr(self._disease_ode, 'airway_resistance'):
                        # RespiratoryODE (asthma / COPD)
                        # Map PD targets to respiratory drug effects
                        bronchod = 1.0
                        anti_inflam = 1.0
                        for _pk, _pv in pd_multipliers.items():
                            kl = _pk.lower()
                            if 'bronchodilat' in kl or 'beta2' in kl or 'adrenergic' in kl:
                                bronchod *= (1.0 - min(0.8, _pv))
                            if _pk in ANTI_INFLAMMATORY_PD_TARGETS and _pv < 1.0:
                                anti_inflam *= (1.0 - min(0.8, _pv))
                        self._disease_ode.step(dt_h, drug_bronchodilator=bronchod,
                                               drug_anti_inflammatory=anti_inflam)
                        # FEV1 → SpO2 → respiratory rate
                        fev1 = self._disease_ode.fev1_percent
                        vitals.spo2_pct = max(70.0, min(100.0, 60.0 + 0.5 * fev1))
                        vitals.respiratory_rate_per_min = max(8.0, min(35.0,
                            12.0 + 15.0 * self._disease_ode.inflammation_score))
                    elif hasattr(self._disease_ode, 'viral_bacterial_load'):
                        # InfectiousDiseaseODE (HIV / TB / bacterial)
                        # Drug effectiveness: detect inhibition effects by
                        # multiplier value < 1.0 (inhibition = target * (1-frac))
                        antimicrobial_eff = 0.0
                        for _pk, _pv in pd_multipliers.items():
                            if _pv < 1.0:
                                antimicrobial_eff = max(antimicrobial_eff, 1.0 - _pv)
                        self._disease_ode.step(dt_h, drug_effectiveness=min(1.0, antimicrobial_eff))
                        # Wire drug effect back into immune model so IL-6/CRP/WBC
                        # respond to infection clearance (not just static severity)
                        if self._immune is not None:
                            # Reduce effective infection_severity as pathogen is cleared
                            load_frac = max(0.0, self._disease_ode.viral_bacterial_load / 5.0)
                            base_inf = getattr(self._immune, '_base_infection_severity',
                                               self.config.disease.severity if self.config.disease else 0.0)
                            self._immune.infection_severity = base_inf * load_frac
                        # Pathogen burden → WBC and CRP
                        # Skip overwrites when specialized models own these channels
                        if self._hematology is None:
                            labs.wbc_per_ul = max(500.0, 8000.0 * self._disease_ode.immune_function)
                        if self._crp_driver is None:
                            labs.crp_mg_per_l = max(0.1, 0.5 + 3.0 * self._disease_ode.inflammation)
                        # Fever from systemic inflammation, attenuated by drug
                        if self._disease_ode.inflammation > 0.3:
                            eff_inflam = self._disease_ode.inflammation * (
                                1.0 - 0.8 * min(1.0, antimicrobial_eff))
                            vitals.temperature_c = min(41.0,
                                37.0 + 2.0 * max(0.0, eff_inflam))
                    elif hasattr(self._disease_ode, 'acid_secretion'):
                        # GastrointestinalODE (GERD / IBD)
                        # Map PD targets to GI drug effects
                        acid_suppr = 1.0
                        anti_inflam = 1.0
                        for _pk, _pv in pd_multipliers.items():
                            kl = _pk.lower()
                            if 'proton_pump' in kl or 'acid' in kl or 'ppi' in kl:
                                acid_suppr *= (1.0 - min(0.9, _pv))
                            if _pk in ANTI_INFLAMMATORY_PD_TARGETS and _pv < 1.0:
                                anti_inflam *= (1.0 - min(0.8, _pv))
                        self._disease_ode.step(dt_h, drug_acid_suppression=acid_suppr,
                                               drug_anti_inflammatory=anti_inflam)
                        # Mucosal damage → albumin loss, pain → vitals
                        mucosal = self._disease_ode.mucosal_integrity
                        labs.albumin_g_per_dl = max(1.5, 4.0 * mucosal)
                    elif hasattr(self._disease_ode, 't4_level'):
                        # EndocrineODE (thyroid)
                        t4_supp = pd_multipliers.get("thyroid_replacement", 0.0)
                        antithyroid = 1.0 - pd_multipliers.get("antithyroid", 0.0)
                        self._disease_ode.step(dt_h, drug_t4_supplement=t4_supp,
                                               drug_antithyroid=antithyroid)
                        # Thyroid → HR, temperature, metabolic rate
                        rate = self._disease_ode.metabolic_rate
                        vitals.heart_rate_bpm = max(45.0, min(130.0, 72.0 * rate))
                        vitals.temperature_c = max(35.0, min(39.5, 37.0 * rate))
                    else:
                        self._disease_ode.step(dt_h, drug_eff)
                        # Generic model: feed organ function proxies into labs
                        if hasattr(self._disease_ode, 'liver_function'):
                            liver_fn = self._disease_ode.liver_function
                            kidney_fn = self._disease_ode.kidney_function
                            inflam = self._disease_ode.inflammation_score
                            # Liver impairment → ALT/AST elevation
                            labs.alt_u_per_l += dt_h * 0.5 * (1.0 - liver_fn)
                            labs.ast_u_per_l += dt_h * 0.4 * (1.0 - liver_fn)
                            # Kidney impairment → creatinine rise, eGFR fall
                            labs.creatinine_mg_per_dl = max(0.5,
                                labs.creatinine_mg_per_dl * (2.0 - kidney_fn))
                            labs.egfr_ml_per_min = max(5.0,
                                labs.egfr_ml_per_min * kidney_fn)
                            # Inflammation → CRP, WBC (skip if specialized models own them)
                            if self._crp_driver is None:
                                labs.crp_mg_per_l = max(0.1, 0.5 + 8.0 * inflam)
                            if self._hematology is None:
                                labs.wbc_per_ul = max(500.0,
                                    7000.0 * (1.0 + 0.5 * inflam))

            # --- Toxicity checks ---
            self._check_toxicities(drug_concs, labs, t_h)

            # --- ODE severity feedback ---
            # Override disease_sev from the mechanistic ODE state so the reported
            # severity tracks actual disease dynamics, not just the generic model.
            if self._disease_ode is not None:
                if hasattr(self._disease_ode, 'viral_bacterial_load'):
                    # Infectious: severity scales with pathogen load (0-8 log scale → 0-1)
                    disease_sev = max(0.0, min(1.0,
                        self._disease_ode.viral_bacterial_load / 5.0))
                elif hasattr(self._disease_ode, 'joint_inflammation'):
                    # Autoimmune: severity = joint inflammation (0-1)
                    disease_sev = self._disease_ode.joint_inflammation
                elif hasattr(self._disease_ode, 'tumor_volume'):
                    # Cancer: severity = tumor volume (0-0.5 → 0-1)
                    disease_sev = min(1.0, self._disease_ode.tumor_volume * 2.0)
                elif hasattr(self._disease_ode, 'nephron_mass'):
                    # Renal: severity = 1 - nephron mass
                    disease_sev = max(0.0, 1.0 - self._disease_ode.nephron_mass)
                elif hasattr(self._disease_ode, 'fibrosis_stage'):
                    # Hepatic: severity = fibrosis / 4
                    disease_sev = self._disease_ode.fibrosis_stage / 4.0
                elif hasattr(self._disease_ode, 't2d_severity'):
                    disease_sev = self._disease_ode.t2d_severity
                elif hasattr(self._disease_ode, 'atherosclerosis_severity'):
                    disease_sev = self._disease_ode.atherosclerosis_severity
                elif hasattr(self._disease_ode, 'inflammation_score'):
                    disease_sev = self._disease_ode.inflammation_score
                elif hasattr(self._disease_ode, 'severity'):
                    disease_sev = self._disease_ode.severity

            # --- Fever resolution ---
            # After ODE processing, if disease severity is dropping (drug working),
            # decay temperature back toward baseline.  CRP-driven fever from the
            # VitalsModel was already applied earlier; now we attenuate it when
            # the mechanistic ODE says the disease is resolving.
            if self._disease_ode is not None and self.config.disease is not None:
                base_temp = 37.0
                fever_excess = vitals.temperature_c - base_temp
                if fever_excess > 0.0 and disease_sev < 0.3:
                    # Decay rate scales with how far severity has dropped
                    decay = max(0.0, (0.3 - disease_sev) / 0.3)
                    vitals.temperature_c = base_temp + fever_excess * max(0.0, 1.0 - decay)

            # --- Record at output resolution ---
            if t_h >= next_record:
                result.time_h.append(t_h)

                result.systolic_bp.append(vitals.systolic_bp_mmhg)
                result.diastolic_bp.append(vitals.diastolic_bp_mmhg)
                result.heart_rate.append(vitals.heart_rate_bpm)
                result.temperature.append(vitals.temperature_c)
                result.weight_kg.append(vitals.weight_kg)
                result.spo2_pct.append(vitals.spo2_pct)
                result.respiratory_rate.append(vitals.respiratory_rate_per_min)

                result.alt.append(labs.alt_u_per_l)
                result.ast.append(labs.ast_u_per_l)
                result.creatinine.append(labs.creatinine_mg_per_dl)
                result.egfr.append(labs.egfr_ml_per_min)
                result.wbc.append(labs.wbc_per_ul)
                result.hemoglobin.append(labs.hemoglobin_g_per_dl)
                result.platelets.append(labs.platelets_per_ul)
                result.glucose.append(labs.glucose_mg_per_dl)
                result.hba1c.append(labs.hba1c_pct)
                result.crp.append(labs.crp_mg_per_l)
                result.bilirubin.append(labs.bilirubin_total_mg_per_dl)
                result.albumin.append(labs.albumin_g_per_dl)
                result.inr.append(labs.inr)
                result.sodium.append(labs.sodium_meq_per_l)
                result.potassium.append(labs.potassium_meq_per_l)
                result.lactate.append(labs.lactate_mmol_per_l)
                # Electrolytes
                result.calcium.append(labs.calcium_mg_per_dl)
                result.phosphate.append(labs.phosphate_mg_per_dl)
                result.chloride.append(labs.chloride_meq_per_l)
                result.bicarbonate.append(labs.bicarbonate_meq_per_l)
                # Lipids
                result.ldl.append(labs.ldl_mg_per_dl)
                result.hdl.append(labs.hdl_mg_per_dl)
                result.triglycerides.append(labs.triglycerides_mg_per_dl)
                # ECG
                result.qtc_ms.append(vitals.qtc_ms)

                # doc/30-31 new channels
                if self._endocrine is not None:
                    result.cortisol.append(self._endocrine.get_cortisol_ug_dl())
                    result.insulin.append(self._endocrine.get_insulin_uuml())
                    result.glucose_endocrine.append(self._endocrine.get_glucose_mg_dl())
                    result.tsh.append(self._endocrine.get_tsh())
                    result.ft4.append(self._endocrine.get_ft4())
                else:
                    result.cortisol.append(12.0)
                    result.insulin.append(10.0)
                    result.glucose_endocrine.append(100.0)
                    result.tsh.append(2.0)
                    result.ft4.append(1.2)
                if self._immune is not None:
                    result.il6.append(self._immune.get_il6())
                    result.tnf_alpha.append(self._immune.get_tnf())
                    result.neutrophils.append(self._immune.get_neutrophils())
                else:
                    result.il6.append(1.0)
                    result.tnf_alpha.append(5.0)
                    result.neutrophils.append(4.0)
                if self._disease_ode is not None and hasattr(self._disease_ode, 'tumor_volume'):
                    result.tumor_volume.append(self._disease_ode.tumor_volume)
                else:
                    result.tumor_volume.append(0.0)
                if self._disease_ode is not None and hasattr(self._disease_ode, 'nephron_mass'):
                    result.nephron_mass.append(self._disease_ode.nephron_mass)
                else:
                    result.nephron_mass.append(1.0)
                if self._disease_ode is not None and hasattr(self._disease_ode, 'fibrosis_stage'):
                    result.fibrosis_stage.append(self._disease_ode.fibrosis_stage)
                else:
                    result.fibrosis_stage.append(0.0)
                if self._disease_ode is not None and hasattr(self._disease_ode, 'beta_cell_function'):
                    result.beta_cell_function.append(self._disease_ode.beta_cell_function)
                else:
                    result.beta_cell_function.append(1.0)
                if self._disease_ode is not None and hasattr(self._disease_ode, 'fev1_percent'):
                    result.fev1_percent.append(self._disease_ode.fev1_percent)
                else:
                    result.fev1_percent.append(80.0)
                if self._disease_ode is not None and hasattr(self._disease_ode, 'cd4_count'):
                    result.cd4_count.append(self._disease_ode.cd4_count)
                else:
                    result.cd4_count.append(800.0)
                if self._disease_ode is not None and hasattr(self._disease_ode, 'acid_secretion'):
                    result.acid_secretion.append(self._disease_ode.acid_secretion)
                else:
                    result.acid_secretion.append(1.0)
                if self._disease_ode is not None and hasattr(self._disease_ode, 'mucosal_integrity'):
                    result.mucosal_integrity.append(self._disease_ode.mucosal_integrity)
                else:
                    result.mucosal_integrity.append(1.0)
                if self._disease_ode is not None and hasattr(self._disease_ode, 't4_level'):
                    result.t4_level.append(self._disease_ode.t4_level)
                else:
                    result.t4_level.append(120.0)
                # doc/33 CV/Neuro ODE feedback channels
                if self._disease_ode is not None and hasattr(self._disease_ode, 'co_l_min'):
                    result.cardiac_output.append(self._disease_ode.co_l_min)
                    result.map_mmhg.append(self._disease_ode.map_mmhg)
                else:
                    result.cardiac_output.append(5.0)
                    result.map_mmhg.append(93.0)
                if self._disease_ode is not None and hasattr(self._disease_ode, 'synaptic_density'):
                    result.synaptic_density.append(self._disease_ode.synaptic_density)
                    result.cognitive_score.append(self._disease_ode.cognitive_score)
                else:
                    result.synaptic_density.append(1.0)
                    result.cognitive_score.append(1.0)

                result.disease_severity.append(disease_sev)
                result.disease_stage.append(disease_stage_str)

                for dk, cv in drug_concs.items():
                    result.drug_concentrations.setdefault(dk, []).append(cv)

                next_record += record_interval

        # --- DDI alerts ---
        ddi_alerts = self._ddi_model.get_clinical_alerts(drug_names, cyp_profile)
        tox_alerts = assess_additive_toxicity(drug_names)
        result.ddi_alerts = ddi_alerts + tox_alerts

        # --- doc/32 §8.1: Kalman denoising of reported series (opt-in) ---
        # ke_prior=0.0 → random-walk state model, so rising lab trends are
        # smoothed without the decay bias a nonzero ke would introduce.
        if self._denoise_outputs and _bayesian_denoiser is not None:
            denoiser = _bayesian_denoiser()
            for series_name in ("alt", "creatinine", "wbc"):
                raw = list(getattr(result, series_name))
                dr = denoiser.denoise(
                    list(result.time_h),
                    raw,
                    ke_prior=0.0,
                    assay_cv=self._assay_cv,
                )
                setattr(result, f"raw_{series_name}", raw)
                setattr(result, series_name, list(dr.denoised_values))

        # --- Finalize ---
        self._finalize(result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seed_recovery_from_labs(self, labs: ClinicalLabs) -> None:
        """Copy live lab values into the recovery model's biomarker state.

        The recovery model only decays *deviations from baseline* captured
        at treatment stop, so without this seed it would relax from its own
        (never-updated) initial values instead of the actual post-treatment
        labs.
        """
        if self._recovery_model is None:
            return
        seeds = {
            "ALT": labs.alt_u_per_l,
            "creatinine": labs.creatinine_mg_per_dl,
            "WBC": labs.wbc_per_ul,
            "hemoglobin": labs.hemoglobin_g_per_dl,
        }
        for key, value in seeds.items():
            if key in self._recovery_model.current_biomarkers:
                self._recovery_model.current_biomarkers[key] = value

    def _feed_recovery_biomarkers(self, recovery_biomarkers: dict[str, float]) -> None:
        """Blend post-treatment recovery biomarkers back into the labs model.

        Active treatment keeps lab dynamics drug-dominated; once treatment
        is inactive, the organ-specific first-order recovery trajectories
        drive the corresponding analytes.
        """
        if self._treatment_active or not recovery_biomarkers:
            return
        mapping = {
            "ALT": "alt_u_per_l",
            "creatinine": "creatinine_mg_per_dl",
            "WBC": "wbc_per_ul",
            "hemoglobin": "hemoglobin_g_per_dl",
        }
        current = self._labs_model.current
        for bio_key, field_name in mapping.items():
            value = recovery_biomarkers.get(bio_key)
            if value is not None:
                setattr(current, field_name, value)

    def _compute_treatment_effectiveness(
        self,
        drug_concs: dict[str, float],
        pd_multipliers: dict[str, float],
    ) -> float:
        """Returns [0,1] treatment effectiveness (1 = fully controlled)."""
        if not pd_multipliers:
            return 0.0
        # Average reduction from PD multipliers
        vals = list(pd_multipliers.values())
        avg_reduction = sum(max(0, 1.0 - v) for v in vals) / max(len(vals), 1)
        return min(1.0, avg_reduction)

    def _check_toxicities(
        self,
        drug_concs: dict[str, float],
        labs: ClinicalLabs,
        t_h: float,
    ) -> None:
        if labs.alt_u_per_l > 3.0 * 56.0:
            self._toxicity_flags.setdefault("hepatotoxicity", []).append(t_h)
        if labs.creatinine_mg_per_dl > 2.0:
            self._toxicity_flags.setdefault("nephrotoxicity", []).append(t_h)
        if labs.wbc_per_ul < 2000.0:
            self._toxicity_flags.setdefault("myelosuppression", []).append(t_h)
        if labs.alt_u_per_l > 3.0 * 56.0 and labs.bilirubin_total_mg_per_dl > 2.0 * 1.2:
            self._toxicity_flags.setdefault("hys_law", []).append(t_h)

    def _finalize(self, result: VirtualPatientResult) -> None:
        """Compute summary metrics."""
        for events in self._toxicity_flags.values():
            result.total_toxicity_events += len(events)
            for t in events:
                result.clinical_events.append({
                    "time_h": t,
                    "type": "toxicity",
                    "details": events,
                })

        for dk, concs in result.drug_concentrations.items():
            if concs:
                result.auc_plasma[dk] = _trapz(concs, result.time_h)

        if result.alt:
            result.max_alt = max(result.alt)
        if result.creatinine:
            result.max_creatinine = max(result.creatinine)
        if result.wbc:
            result.min_wbc = min(result.wbc)
        if result.egfr:
            result.min_egfr = min(result.egfr)

        # Efficacy: fraction of disease-severity values below threshold
        if result.disease_severity:
            controlled = sum(1 for s in result.disease_severity if s < 0.3)
            result.overall_efficacy_score = controlled / len(result.disease_severity)


def _trapz(values: list[float], time_h: list[float]) -> float:
    """Trapezoidal integration."""
    if len(values) < 2 or len(time_h) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(values)):
        dt = time_h[i] - time_h[i - 1]
        total += (values[i] + values[i - 1]) * 0.5 * dt
    return max(0.0, total)


def _compute_genetic_cyp_modifier(drug: Drug, genotype: GenotypeProfile) -> float:
    """Compute clearance multiplier from genotype CYP profile.

    Returns a multiplier on drug clearance. < 1.0 = reduced clearance.
    """
    cyp_map = drug.cyp_metabolism
    if not cyp_map:
        return 1.0

    # Weighted average of CYP contributions
    total_weight = sum(cyp_map.values())
    if total_weight <= 0:
        return 1.0

    modifier = 0.0
    for enzyme, weight in cyp_map.items():
        activity = genotype.get_cyp_activity(enzyme)
        modifier += (activity / 1.0) * (weight / total_weight)
        # Normalized to 1.0 = normal metabolizer
    return max(0.1, min(5.0, modifier))


def _compute_transporter_modifier(drug: Drug, genotype: GenotypeProfile) -> float:
    """Compute clearance multiplier from transporter genotype.

    Transporters like SLCO1B1 affect hepatic uptake; reduced function
    → higher systemic exposure. ABCB1 (P-gp) affects efflux.
    Returns a multiplier on drug clearance. < 1.0 = reduced clearance.
    """
    tp_map = drug.transporter_affected
    if not tp_map:
        return 1.0
    modifier = 0.0
    for transporter, weight in tp_map.items():
        activity = genotype.get_transporter_activity(transporter)
        # Reduced transporter function → reduced clearance
        modifier += (activity / 1.0) * (weight / sum(tp_map.values()))
    return max(0.1, min(5.0, modifier))


def _compute_non_cyp_modifier(drug: Drug, genotype: GenotypeProfile) -> float:
    """Compute clearance multiplier from non-CYP phase-II genotype.

    Enzymes like UGT1A1 (glucuronidation), TPMT (thiopurine methylation),
    DPYD (pyrimidine catabolism) can drastically alter drug levels.
    Returns a multiplier on drug clearance. < 1.0 = reduced clearance.
    """
    nc_map = drug.non_cyp_metabolism
    if not nc_map:
        return 1.0
    modifier = 0.0
    for enzyme, weight in nc_map.items():
        activity = genotype.get_non_cyp_activity(enzyme)
        modifier += (activity / 1.0) * (weight / sum(nc_map.values()))
    return max(0.1, min(5.0, modifier))


# ============================================================================
# Internal drug-PBPK wrapper
# ============================================================================


class _DrugPBPK:
    """Stateful whole-body PBPK engine for a single drug (doc/28).

    Unlike :class:`PBPKModel`, which integrates a full batch horizon from a
    fresh initial state on every ``run()``, this engine keeps its
    compartment concentrations between :meth:`advance` calls so repeated
    steps produce one continuous concentration-time course with the dosing
    schedule layered on top. All concentrations are tracked and reported in
    µM (``conc_uM = conc_mg_per_L * 1000 / molecular_weight_da``).

    Structure: well-stirred, perfusion-limited compartments (central plasma
    pool plus liver, kidney, brain, muscle, adipose) integrated with an
    explicit Euler sub-stepping scheme:

        dC_central/dt = (sum(Q_o * Kp_o * C_o) - Q_total * C_central) / Vc
                        - k_renal * C_central - k_hepatic * C_central + Input(t)
        dC_organ/dt   = Q_o * (C_central - Kp_o * C_organ) / V_o

    Renal clearance is ``f_renal * CL_total / Vc``, hepatic clearance the
    well-stirred extraction relation ``E_h * Q_liver / Vc``; trans-membrane
    routes absorb through a first-order gut depot
    (``dDepot/dt = -ka * Depot``, ``Input += ka * Depot * F / Vc``).
    """

    #: tissue:plasma partition coefficients (Kp); adipose uses 2.0 only for
    #: lipophilic drugs (logP > 2), else 1.0
    _DEFAULT_KP: dict[str, float] = {
        "liver": 3.5,
        "kidney": 1.0,
        "brain": 0.3,  # blood-brain barrier
        "muscle": 0.8,
    }
    _ADIPOSE_KP_LIPHOPHILIC = 2.0
    _NEUTRAL_KP = 1.0
    _LIPOPHILIC_LOGP_THRESHOLD = 2.0
    _EULER_SAFETY = 0.4
    _MAX_SUBSTEPS = 2000
    _INFUSION_DURATION_H = 1.0
    _DOSE_EPSILON_H = 0.01

    # Biologic MW thresholds for Kp scaling (doc/33 biologics)
    _BIOLOGIC_MW_THRESHOLD = 30_000.0  # Da — above this, MW-gated Kp applies
    _BIOLOGIC_LARGE_MW = 150_000.0     # Da — full-size mAb
    _FCRN_K_RECYCLING = 0.003          # FcRn rescue rate (per hour)
    _FCRN_SATURATION_UM = 50.0         # µM — FcRn binding half-saturation

    def __init__(self, drug: Drug, physiology: HumanPhysiology) -> None:
        self.drug = drug
        self.physiology = physiology
        self.clearance_modifier = 1.0
        self.sim_time_h = 0.0

        # Unit conversion: mg/L -> µM via molecular weight
        mw_da = max(drug.molecule.molecular_weight_da, 1.0)
        self._um_per_mg_per_l = 1000.0 / mw_da

        geometry = PBPKConfig()
        self.vc_l = geometry.plasma_volume_l
        self.organ_volumes_l: dict[str, float] = {
            name: getattr(geometry, f"{name}_volume_l") for name in ORGAN_NAMES
        }

        co_l_per_h = physiology.cardiac_output_ml_per_min * ML_PER_MIN_TO_L_PER_H
        self.organ_flows_l_per_h: dict[str, float] = {}
        for name in ORGAN_NAMES:
            organ = physiology.organs.get(name)
            if organ is not None and organ.blood_flow_ml_per_min > 0.0:
                self.organ_flows_l_per_h[name] = (
                    organ.blood_flow_ml_per_min * ML_PER_MIN_TO_L_PER_H
                )
            else:
                self.organ_flows_l_per_h[name] = (
                    co_l_per_h * DEFAULT_FLOW_FRACTIONS[name]
                )
        self.q_total_l_per_h = sum(self.organ_flows_l_per_h.values())

        lipophilic = drug.molecule.log_p > self._LIPOPHILIC_LOGP_THRESHOLD
        is_biologic = (mw_da > self._BIOLOGIC_MW_THRESHOLD)
        self.partition_ratios: dict[str, float] = {}
        for name in ORGAN_NAMES:
            if is_biologic:
                # MW-gated Kp: large molecules are confined to plasma + interstitial;
                # tissue penetration scales inversely with MW.
                base_kp = self._DEFAULT_KP.get(name, self._NEUTRAL_KP)
                mw_penalty = min(1.0, self._BIOLOGIC_MW_THRESHOLD / mw_da)
                if name == "brain":
                    mw_penalty *= 0.2  # BBB is essentially impermeable to mAbs
                self.partition_ratios[name] = max(0.05, base_kp * mw_penalty)
            else:
                self.partition_ratios[name] = self._DEFAULT_KP.get(name, self._NEUTRAL_KP)
        self.partition_ratios["adipose"] = (
            self._ADIPOSE_KP_LIPHOPHILIC if lipophilic else self._NEUTRAL_KP
        )
        if is_biologic:
            self.partition_ratios["adipose"] = 0.05  # hydrophilic: minimal adipose

        # --- Kp calibration: scale tissue Kp values to match target Vd ---
        target_vd = getattr(drug, 'volume_distribution_l', 0.0)
        if target_vd and target_vd > self.vc_l:
            vd_ss_current = self.vc_l + sum(
                self.organ_volumes_l[n] * self.partition_ratios[n]
                for n in ORGAN_NAMES
            )
            if vd_ss_current > self.vc_l:
                scale = (target_vd - self.vc_l) / (vd_ss_current - self.vc_l)
                for name in ORGAN_NAMES:
                    # Scale deviation from neutral (Kp=1.0) toward target
                    self.partition_ratios[name] = max(
                        0.05,
                        1.0 + (self.partition_ratios[name] - 1.0) * scale,
                    )

        # Elimination rate constants (per hour, acting on central plasma)
        # The specified renal_fraction and hepatic_extraction may not sum to
        # total clearance (e.g. osimertinib: renal 10% + hepatic E*Q_liver
        # ≈ 2.7 L/h but actual CL = 14.2 L/h).  We compute the shortfall
        # and add it as a lumped "other" elimination on the central compartment.
        cl_total_l_per_h = drug.clearance_ml_per_min * ML_PER_MIN_TO_L_PER_H
        renal_fraction = min(max(drug.renal_fraction, 0.0), 1.0)
        extraction = min(max(drug.hepatic_extraction_ratio, 0.0), 1.0)
        q_liver = self.organ_flows_l_per_h["liver"]

        renal_cl = renal_fraction * cl_total_l_per_h
        hepatic_cl = extraction * q_liver if q_liver > 0.0 else 0.0

        if cl_total_l_per_h > 0.0:
            if renal_fraction <= 0.0 and extraction <= 0.0:
                # No pathway fractions specified — apply full CL to central
                renal_cl = cl_total_l_per_h
                hepatic_cl = 0.0
            elif extraction <= 0.0 and q_liver > 0.0:
                # Only renal specified — derive hepatic from remainder
                remaining = cl_total_l_per_h - renal_cl
                if remaining > 0.0:
                    derived_extraction = min(1.0, remaining / q_liver)
                    hepatic_cl = derived_extraction * q_liver

        # Account for any remaining clearance not covered by named pathways
        # (extrahepatic metabolism, biliary, protein-binding-mediated, etc.)
        accounted_cl = renal_cl + hepatic_cl
        other_cl = max(0.0, cl_total_l_per_h - accounted_cl)

        self.k_renal_per_h = renal_cl / self.vc_l
        self.k_hepatic_per_h = (hepatic_cl + other_cl) / self.vc_l

        # Persistent compartment state (µM) + oral depot (mg)
        self.conc_um: dict[str, float] = {
            "central": 0.0,
            **{name: 0.0 for name in ORGAN_NAMES},
        }
        self.depot_mg = 0.0
        self._doses_given: list[float] = []
        self._last_dose_count: int = 0

    # ------------------------------------------------------------------
    # Dosing
    # ------------------------------------------------------------------

    def _administer_dose(self) -> None:
        """Apply one scheduled dose according to the route."""
        route = self.drug.route
        available_mg = self.drug.bioavailability * self.drug.dose_mg
        if route == IV:
            # Instantaneous bolus into the central compartment
            self.conc_um["central"] += (
                available_mg * self._um_per_mg_per_l / self.vc_l
            )
        elif route in FIRST_ORDER_ROUTES:
            # Oral / subcutaneous / intramuscular: fill the absorption depot
            self.depot_mg += available_mg
        # IV_INFUSION enters as a zero-order input inside _central_input_um_per_h

    def _central_input_um_per_h(
        self,
        t_abs_h: float,
        absorbed_mg_per_h: float,
    ) -> float:
        """Route-specific additive input to dC_central/dt (µM/h)."""
        flux = absorbed_mg_per_h * self._um_per_mg_per_l / self.vc_l
        if self.drug.route == IV_INFUSION and self._doses_given:
            t_start = self._doses_given[-1]
            if t_start <= t_abs_h <= t_start + self._INFUSION_DURATION_H:
                infusion_rate_mg_per_h = (
                    self.drug.bioavailability
                    * self.drug.dose_mg
                    / self._INFUSION_DURATION_H
                )
                flux += infusion_rate_mg_per_h * self._um_per_mg_per_l / self.vc_l
        return flux

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def _max_rate_constant(self, ka_per_h: float, k_elim_per_h: float) -> float:
        """Largest first-order rate constant of the network (per hour)."""
        rate = self.q_total_l_per_h / self.vc_l + abs(k_elim_per_h)
        for name in ORGAN_NAMES:
            rate += (
                self.organ_flows_l_per_h[name]
                * self.partition_ratios[name]
                / self.organ_volumes_l[name]
            )
        if self.drug.route in FIRST_ORDER_ROUTES:
            rate += ka_per_h
        return max(rate, 1e-9)

    def _euler_step(
        self,
        h_sub_h: float,
        t_abs_h: float,
        ka_per_h: float,
        k_elim_per_h: float,
    ) -> None:
        """One explicit Euler sub-step from the current state."""
        c = self.conc_um
        central = c["central"]

        organ_derivatives: dict[str, float] = {}
        for name in ORGAN_NAMES:
            organ_derivatives[name] = (
                self.organ_flows_l_per_h[name]
                * (central - self.partition_ratios[name] * c[name])
                / self.organ_volumes_l[name]
            )

        recirculation = sum(
            self.organ_flows_l_per_h[name]
            * self.partition_ratios[name]
            * c[name]
            for name in ORGAN_NAMES
        )
        absorbed_mg_per_h = ka_per_h * self.depot_mg
        d_central = (
            (recirculation - self.q_total_l_per_h * central) / self.vc_l
            - k_elim_per_h * central
            + self._central_input_um_per_h(t_abs_h, absorbed_mg_per_h)
        )

        c["central"] = max(0.0, central + d_central * h_sub_h)
        for name in ORGAN_NAMES:
            c[name] = max(0.0, c[name] + organ_derivatives[name] * h_sub_h)
        self.depot_mg = max(0.0, self.depot_mg - absorbed_mg_per_h * h_sub_h)

        # FcRn-mediated recycling for biologics (doc/33 biologics)
        if (self.drug.molecule.molecular_weight_da > self._BIOLOGIC_MW_THRESHOLD
                and central > self._FCRN_SATURATION_UM * 0.1):
            # FcRn rescues albumin/IgG from lysosomal degradation:
            # recycling rate scales with concentration (saturable Michaelis)
            fcrn_frac = (central / (central + self._FCRN_SATURATION_UM))
            rescue_rate = self._FCRN_K_RECYCLING * fcrn_frac
            # Apply rescue to central compartment (reduces effective elimination)
            c["central"] = max(0.0, c["central"] * (1.0 + rescue_rate * h_sub_h))

    def advance(self, dt_h: float, current_time_h: float) -> None:
        """Advance PBPK by *dt_h*, handling the dosing schedule."""
        if dt_h <= 0.0:
            return

        interval_h = max(self.drug.dosing_interval_h, 1e-6)
        treatment_end_h = self.drug.duration_days * 24.0

        # Give dose at t=0 and every interval within the treatment duration
        if current_time_h < treatment_end_h:
            due = not self._doses_given or (
                current_time_h - self._doses_given[-1]
                >= interval_h - self._DOSE_EPSILON_H
            )
            if due:
                self._doses_given.append(current_time_h)
                self._administer_dose()

        ka_per_h = 1.0 / max(self.drug.absorption_rate_h, 1e-6)
        k_elim_per_h = (
            (self.k_renal_per_h + self.k_hepatic_per_h) * self.clearance_modifier
        )

        max_substep_h = self._EULER_SAFETY / self._max_rate_constant(ka_per_h, k_elim_per_h)
        n_substeps = min(self._MAX_SUBSTEPS, max(1, math.ceil(dt_h / max_substep_h)))
        substep_h = dt_h / n_substeps
        for i in range(n_substeps):
            self._euler_step(
                substep_h,
                current_time_h + i * substep_h,
                ka_per_h,
                k_elim_per_h,
            )

        self.sim_time_h += dt_h

    def get_central_concentration(self) -> float:
        """Current central-compartment (plasma) concentration in µM."""
        return self.conc_um["central"]

    def get_concentrations(self) -> dict[str, float]:
        """Current concentration snapshot (µM) per compartment."""
        return dict(self.conc_um)
