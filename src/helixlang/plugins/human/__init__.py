"""Human physiology, pathology, and drug simulation plugin (doc/36 §7:
``human/*`` -> ``plugins/human/``).

This package is the canonical home of the human virtual-patient / pharmacology
stack.  It aggregates every ``helixlang.plugins.human.*`` submodule for import
convenience and exposes the :data:`PLUGIN` provider so the registry discovers
the ``human`` plugin (``#use human`` / ``#person``/``#drug``/... keywords,
doc/36 §3.1/§7).


Modules:
    physiology   - organ volumes, blood flows, tissue composition
    gem_human    - human GEM loader (Recon3D, tissue overlays)
    disease      - disease state modeling (gene knockout, metabolite perturbation)
    drug         - drug molecule specification (organic/inorganic/biologic)
    pharmacokinetics - PBPK compartmental model (ADME)
    pharmacodynamics  - dose-response, IC50, Hill equation
    simulation   - long-term integration (dFBA + PK/PD)
    genotype     - genome → pharmacogenomic phenotype (CYP450 mapping)
    phenotype    - external traits → organ-parameter scaling
    clinical_output - clinical lab values and vital-signs dynamics
    disease_progression - disease staging and dynamic severity
    ddi          - drug-drug interaction modeling (CYP competition)
    recovery     - post-treatment recovery, rebound, sequelae
    virtual_patient - unified facade for full-body patient simulation
    hematology_model - Friberg myelosuppression + erythropoiesis (doc/30 §8)
    renal_model  - eGFR-slope CKD progression + AKI (doc/30 §6)
    endocrine    - insulin-glucose, HPA, HPT axes (doc/31 §2.6)
    qsp_binding  - mass-action, TMDD, competitive PD (doc/31 §2.5)
    immune       - innate immune ABM, CRP/WBC (doc/31 §2.4)
    organ_crosstalk - organ-organ coupling (doc/30 §9)
    disease_ode_models - per-disease ODE systems (doc/30 §§1-8)

Baseline: literature-influenced physiological models, not faithful reproductions.
Referenced physiology and pharmacology constants are anchored to published sources
where validated (see validation/report.md levels and doc/42). **Not** a medical device
and **not** clinical decision support — see DISCLAIMER.md. Some subsystems (e.g. gas
exchange, thermoregulation, unified cardiovascular coupling) are simplified/partial in
2026 baseline and tracked in doc/42 (Phases A-B).
"""
from helixlang.plugins.human.bayesian_denoiser import BayesianDenoiser
from helixlang.plugins.human.calibration_cascade import CalibrationCascade
from helixlang.plugins.human.clinical_output import (
    ClinicalLabModel,
    ClinicalLabs,
    VitalSigns,
    VitalsModel,
)
from helixlang.plugins.human.ddi import DDIModel, DDIRule
from helixlang.plugins.human.disease import (
    DISEASE_PROFILES,
    DiseaseState,
    GenePerturbation,
    MetabolitePerturbation,
    apply_disease_state,
)
from helixlang.plugins.human.disease_ode_models import (
    AutoimmuneRAODE,
    CancerODE,
    CardiovascularODE,
    HematologicalODE,
    HepaticODE,
    MetabolicT2DODE,
    NeurologicalODE,
    RenalODE,
    create_disease_model,
)
from helixlang.plugins.human.disease_progression import (
    DiseaseProgressionModel,
    DiseaseStage,
    create_progression_model,
)
from helixlang.plugins.human.dose_optimizer import DoseOptimizer
from helixlang.plugins.human.drug import (
    PREDEFINED_DRUGS,
    Drug,
    DrugMolecule,
    get_predefined_drug,
    parse_drug_smiles,
)
from helixlang.plugins.human.emergent_complexity import EmergentComplexityModel
from helixlang.plugins.human.endocrine import EndocrineSystem, create_endocrine
from helixlang.plugins.human.gem_human import HumanGEMConfig, HumanGEMLoader
from helixlang.plugins.human.genotype import (
    CYPStatus,
    GenotypeProfile,
    Variant,
    create_default_genotype,
)
from helixlang.plugins.human.hematology_model import (
    HematologySystem,
    MyelosuppressionParams,
    create_hematology_system,
)
from helixlang.plugins.human.immune import CRPDriver, InnateImmuneModel, create_immune_model
from helixlang.plugins.human.mechanistic_ddi import EnzymeInhibitionLibrary, MechanisticDDIPredictor
from helixlang.plugins.human.microbiome import MicrobiomeCompartment

# doc/32 — advanced pharmacological modeling
from helixlang.plugins.human.molecular_toxicity import MolecularToxicityPredictor
from helixlang.plugins.human.organ_crosstalk import (
    OrganCrosstalk,
    apply_crosstalk,
    create_crosstalk,
)
from helixlang.plugins.human.pharmacodynamics import (
    PREDEFINED_PD,
    PDEffect,
    Pharmacodynamics,
    get_predefined_pd,
)
from helixlang.plugins.human.pharmacogenomic_ae import (
    TOXIC_METABOLITES,
    AEOrgan,
    AEPrediction,
    AERisk,
    GenotypeAEPredictor,
    ToxicMetabolite,
    ToxicMetaboliteAccumulator,
)
from helixlang.plugins.human.pharmacokinetics import PBPKConfig, PBPKModel, PBPKResult
from helixlang.plugins.human.phenotype import ExternalTraits, PhenotypeCalculator
from helixlang.plugins.human.physiology import (
    TISSUE_PROFILES,
    HumanPhysiology,
    OrganSpec,
    create_default_physiology,
)
from helixlang.plugins.human.physiology_constraints import PhysiologyConstraints
from helixlang.plugins.human.proteome_binding import ProteomeBindingCascade
from helixlang.plugins.human.qsp_binding import QSPBindingSystem, create_qsp_binding
from helixlang.plugins.human.recovery import RecoveryModel, Sequela
from helixlang.plugins.human.reduced_order_organ import (
    PODMode,
    PODModeGenerator,
    ReducedOrderOrgan,
)
from helixlang.plugins.human.renal_model import RenalFunctionModel, create_renal_model
from helixlang.plugins.human.simulation import (
    HumanSimulation,
    HumanSimulationConfig,
    HumanSimulationResult,
)
from helixlang.plugins.human.tissue_gem import (
    TISSUE_REACTION_SETS,
    GEMDecomposer,
    OrganGEMCoupler,
    TissueGEM,
)
from helixlang.plugins.human.virtual_4dvar import Virtual4DVar
from helixlang.plugins.human.virtual_patient import (
    VirtualPatient,
    VirtualPatientConfig,
    VirtualPatientResult,
)

__all__ = [
    # physiology (doc/27)
    "HumanPhysiology", "OrganSpec", "TISSUE_PROFILES", "create_default_physiology",
    "HumanGEMLoader", "HumanGEMConfig",
    "DiseaseState", "GenePerturbation", "MetabolitePerturbation", "apply_disease_state", "DISEASE_PROFILES",
    "Drug", "DrugMolecule", "parse_drug_smiles", "PREDEFINED_DRUGS", "get_predefined_drug",
    "PBPKModel", "PBPKConfig", "PBPKResult",
    "Pharmacodynamics", "PDEffect", "PREDEFINED_PD", "get_predefined_pd",
    "HumanSimulation", "HumanSimulationConfig", "HumanSimulationResult",
    # doc/28 — virtual patient
    "Variant", "CYPStatus", "GenotypeProfile", "create_default_genotype",
    "ExternalTraits", "PhenotypeCalculator",
    "ClinicalLabs", "ClinicalLabModel", "VitalSigns", "VitalsModel",
    "DiseaseStage", "DiseaseProgressionModel", "create_progression_model",
    "DDIRule", "DDIModel",
    "RecoveryModel", "Sequela",
    "VirtualPatient", "VirtualPatientConfig", "VirtualPatientResult",
    # doc/30 wave-1 disease modules
    "HematologySystem", "MyelosuppressionParams", "create_hematology_system",
    "RenalFunctionModel", "create_renal_model",
    # doc/30-31 new modules
    "EndocrineSystem", "create_endocrine",
    "QSPBindingSystem", "create_qsp_binding",
    "InnateImmuneModel", "CRPDriver", "create_immune_model",
    "OrganCrosstalk", "apply_crosstalk", "create_crosstalk",
    "CardiovascularODE", "MetabolicT2DODE", "CancerODE", "AutoimmuneRAODE",
    "NeurologicalODE", "RenalODE", "HepaticODE", "HematologicalODE",
    "create_disease_model",
    # doc/32 — advanced pharmacological modeling
    "MolecularToxicityPredictor",
    "CalibrationCascade",
    "BayesianDenoiser",
    "DoseOptimizer",
    "PhysiologyConstraints",
    "MechanisticDDIPredictor",
    "EnzymeInhibitionLibrary",
    "Virtual4DVar",
    # doc/32 §7.4 — tissue-specific GEM decomposition
    "GEMDecomposer", "OrganGEMCoupler", "TissueGEM", "TISSUE_REACTION_SETS",
    # doc/32 §7.5 — multi-scale reduced-order models
    "PODMode", "PODModeGenerator", "ReducedOrderOrgan",
    # doc/32 §7.6 — mechanistic pharmacogenomic AE prediction
    "AEPrediction", "AERisk", "AEOrgan",
    "GenotypeAEPredictor", "ToxicMetaboliteAccumulator",
    "ToxicMetabolite", "TOXIC_METABOLITES",
    # doc/32 §7.7 — proteome-wide binding cascade
    "ProteomeBindingCascade",
    # microbiome-drug interaction modeling
    "MicrobiomeCompartment",
    # emergent complexity (epigenetics, multi-organ feedback)
    "EmergentComplexityModel",
]

# ---------------------------------------------------------------------------
# Plugin contract (doc/36 §3.1/§3.3): make the package a discoverable plugin.
# ---------------------------------------------------------------------------
from collections.abc import Callable

from helixlang.api.registry import PluginProvider


def _check(pkg: str) -> bool:
    def _probe() -> bool:
        try:
            __import__(pkg)
            return True
        except ImportError:
            return False
    return _probe()


def _make_backend(cfg: dict | None = None) -> type:
    from helixlang.plugins.human.virtual_patient import VirtualPatient
    return VirtualPatient


def _load() -> Callable[[dict | None], type]:
    # numpy is the hard baseline for the human physiology/PK/PD numerics.
    if not _check("numpy"):
        from helixlang.core.errors import PluginDependencyError
        raise PluginDependencyError("human", "numpy", "human")
    return _make_backend


PLUGIN = PluginProvider(
    name="human",
    extra="human",
    keywords=(
        "person", "drug", "pd_effect", "qsp_binding",
        "endocrine_config", "immune_config", "tumor_biopsy", "disease",
    ),
    native=None,
    capability_flags=("--low-fidelity",),
    checks={"numpy": lambda: _check("numpy")},
    load=_load,
)
