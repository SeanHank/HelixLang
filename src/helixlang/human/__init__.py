"""Human physiology, pathology, and drug simulation (doc/27-31).

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

Baseline: 100% real. Every parameter anchored to published literature.
"""
from helixlang.human.bayesian_denoiser import BayesianDenoiser
from helixlang.human.calibration_cascade import CalibrationCascade
from helixlang.human.clinical_output import ClinicalLabModel, ClinicalLabs, VitalSigns, VitalsModel
from helixlang.human.ddi import DDIModel, DDIRule
from helixlang.human.disease import (
    DISEASE_PROFILES,
    DiseaseState,
    GenePerturbation,
    MetabolitePerturbation,
    apply_disease_state,
)
from helixlang.human.disease_ode_models import (
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
from helixlang.human.disease_progression import (
    DiseaseProgressionModel,
    DiseaseStage,
    create_progression_model,
)
from helixlang.human.dose_optimizer import DoseOptimizer
from helixlang.human.drug import (
    PREDEFINED_DRUGS,
    Drug,
    DrugMolecule,
    get_predefined_drug,
    parse_drug_smiles,
)
from helixlang.human.emergent_complexity import EmergentComplexityModel
from helixlang.human.endocrine import EndocrineSystem, create_endocrine
from helixlang.human.gem_human import HumanGEMConfig, HumanGEMLoader
from helixlang.human.genotype import CYPStatus, GenotypeProfile, Variant, create_default_genotype
from helixlang.human.hematology_model import (
    HematologySystem,
    MyelosuppressionParams,
    create_hematology_system,
)
from helixlang.human.immune import CRPDriver, InnateImmuneModel, create_immune_model
from helixlang.human.mechanistic_ddi import EnzymeInhibitionLibrary, MechanisticDDIPredictor
from helixlang.human.microbiome import MicrobiomeCompartment

# doc/32 — advanced pharmacological modeling
from helixlang.human.molecular_toxicity import MolecularToxicityPredictor
from helixlang.human.organ_crosstalk import OrganCrosstalk, apply_crosstalk, create_crosstalk
from helixlang.human.pharmacodynamics import (
    PREDEFINED_PD,
    PDEffect,
    Pharmacodynamics,
    get_predefined_pd,
)
from helixlang.human.pharmacogenomic_ae import (
    TOXIC_METABOLITES,
    AEOrgan,
    AEPrediction,
    AERisk,
    GenotypeAEPredictor,
    ToxicMetabolite,
    ToxicMetaboliteAccumulator,
)
from helixlang.human.pharmacokinetics import PBPKConfig, PBPKModel, PBPKResult
from helixlang.human.phenotype import ExternalTraits, PhenotypeCalculator
from helixlang.human.physiology import (
    TISSUE_PROFILES,
    HumanPhysiology,
    OrganSpec,
    create_default_physiology,
)
from helixlang.human.physiology_constraints import PhysiologyConstraints
from helixlang.human.proteome_binding import ProteomeBindingCascade
from helixlang.human.qsp_binding import QSPBindingSystem, create_qsp_binding
from helixlang.human.recovery import RecoveryModel, Sequela
from helixlang.human.reduced_order_organ import (
    PODMode,
    PODModeGenerator,
    ReducedOrderOrgan,
)
from helixlang.human.renal_model import RenalFunctionModel, create_renal_model
from helixlang.human.simulation import HumanSimulation, HumanSimulationConfig, HumanSimulationResult
from helixlang.human.tissue_gem import (
    TISSUE_REACTION_SETS,
    GEMDecomposer,
    OrganGEMCoupler,
    TissueGEM,
)
from helixlang.human.virtual_4dvar import Virtual4DVar
from helixlang.human.virtual_patient import (
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
