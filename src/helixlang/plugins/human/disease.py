"""Disease state modeling for human pathology simulation (doc/27 Stage B).

Disease states are modeled as *perturbations to a human GEM* (constraint-
based disease modeling, Schuster 2002 Biochimie; Jamshidi & Palsson 2007
Mol Syst Biol):

- gene knockout -> zero flux through every reaction gated by the gene
- gene downregulation -> flux upper bound scaled by residual activity
- gene overexpression -> flux capacity fold increase (Warburg reprogramming)
- metabolite accumulation/deficiency -> pathological pool initial values
- transport restriction -> exchange-reaction bound scaling

:func:`apply_disease_state` deep-copies the input
:class:`~helixlang.plugins.runtime.metabolism.MetabolicModel`, applies every perturbation,
and returns the diseased copy (the input model is never mutated).  The
returned model carries a ``metabolite_pool_initials`` attribute — a
``{metabolite_id: concentration}`` dict for seeding a
:class:`~helixlang.plugins.runtime.metabolism.MetabolitePool`.

Module structure:
    GenePerturbation          single gene-level perturbation
    MetabolitePerturbation    single metabolite-level perturbation
    DiseaseState              complete disease specification
    apply_disease_state       apply perturbations to a MetabolicModel
    initial_metabolite_pools  pool-seeding dict for a DiseaseState
    DISEASE_PROFILES          pre-defined literature-anchored profiles

References (doc/27 §5):
- Beutler E et al. Mol Genet Metab 2004 (Gaucher disease type 1, GBA1)
- Blau N et al. J Inherit Metab Dis 2010 (phenylketonuria, PAH)
- Zarate YA & Hopkin RJ. Mol Genet Metab 2017 (Fabry disease, GLA)
- Strauss KA et al. GeneReviews 2016 (maple syrup urine disease, BCKDHA)
- DeFronzo RA et al. Diabetes 2015 (type 2 diabetes, IRS1/INSR)
- Vander Heiden MG et al. Science 2009 (Warburg effect, HK2/PDK1)
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

from helixlang.plugins.runtime.metabolism import ECOLI_CORE_GENE_REACTIONS, MetabolicModel

#: valid :attr:`GenePerturbation.perturbation_type` values
_GENE_PERTURBATION_TYPES = ("knockout", "downregulate", "overexpress")
#: valid :attr:`MetabolitePerturbation.perturbation_type` values
_METABOLITE_PERTURBATION_TYPES = ("accumulate", "deplete", "block_export")
#: effective residual activity below which a reaction is fully closed
_MIN_EFFECTIVE_FRACTION = 1e-6
#: upper cap on overexpression fold change (guards activity_fraction ~ 0)
_MAX_OVEREXP_FOLD = 100.0
#: name of the attribute attached to diseased models by
#: :func:`apply_disease_state`, holding ``{metabolite_id: concentration}``
#: for seeding a :class:`~helixlang.plugins.runtime.metabolism.MetabolitePool`
_METABOLITE_POOL_ATTR = "metabolite_pool_initials"


# ============================================================================
# Data structures (doc/27 §5.3)
# ============================================================================

@dataclass(slots=True)
class GenePerturbation:
    """A single gene-level perturbation in a disease state.

    Attributes:
        gene_id: gene identifier (e.g. ``"GBA1"``, ``"HEXA"``, ``"PAH"``).
        perturbation_type: one of ``"knockout"``, ``"downregulate"`` or
            ``"overexpress"``.
        activity_fraction: residual catalytic capacity relative to wild
            type in [0, 1] (0.0 = complete loss, 0.05 = 5% residual).
            For ``"overexpress"`` the reciprocal is used as the flux-cap
            fold change (activity_fraction 0.5 -> 2x baseline capacity).
        affected_reactions: explicit reaction IDs gated by this gene.
            When empty they are resolved automatically from the model's
            gene-reaction association by :func:`apply_disease_state`.
    """

    gene_id: str
    perturbation_type: str = "knockout"
    activity_fraction: float = 0.0
    affected_reactions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.perturbation_type not in _GENE_PERTURBATION_TYPES:
            raise ValueError(
                f"perturbation_type must be one of "
                f"{_GENE_PERTURBATION_TYPES}, got {self.perturbation_type!r}"
            )
        if not 0.0 <= self.activity_fraction <= 1.0:
            raise ValueError(
                f"activity_fraction must be within [0, 1], "
                f"got {self.activity_fraction}"
            )


@dataclass(slots=True)
class MetabolitePerturbation:
    """Metabolite-level perturbation (accumulation or deficiency).

    Attributes:
        metabolite_id: pool/metabolite identifier (e.g. ``"phe"``,
            ``"glucosylceramide"``, ``"lactate"``).  Concentrations are
            mM for small metabolites; lipid second-messenger pools use
            uM values as reported in the source literature.
        perturbation_type: ``"accumulate"`` | ``"deplete"`` |
            ``"block_export"`` (export-capable exchange reactions are
            fully closed regardless of ``transport_restriction``).
        initial_concentration_mm: pathological pool concentration at
            simulation start.
        normal_concentration_mm: healthy reference concentration;
            severity interpolates between normal and pathological.
        transport_restriction: multiplier applied to exchange-reaction
            bounds for this metabolite (1.0 = normal transport,
            0.0 = fully blocked).
    """

    metabolite_id: str
    perturbation_type: str = "accumulate"
    initial_concentration_mm: float = 0.0
    normal_concentration_mm: float = 0.0
    transport_restriction: float = 1.0

    def __post_init__(self) -> None:
        if self.perturbation_type not in _METABOLITE_PERTURBATION_TYPES:
            raise ValueError(
                f"perturbation_type must be one of "
                f"{_METABOLITE_PERTURBATION_TYPES}, "
                f"got {self.perturbation_type!r}"
            )
        if not 0.0 <= self.transport_restriction <= 1.0:
            raise ValueError(
                f"transport_restriction must be within [0, 1], "
                f"got {self.transport_restriction}"
            )


@dataclass(slots=True)
class DiseaseState:
    """Complete disease specification.

    Attributes:
        name: human-readable disease name
            (e.g. ``"Gaucher disease type 1"``).
        category: mechanism class — ``"enzyme_deficiency"`` |
            ``"transporter_defect"`` | ``"metabolic_overload"`` |
            ``"receptor_dysfunction"`` | ``"cancer_metabolism"``.
        gene_perturbations: gene-level perturbations to apply.
        metabolite_perturbations: metabolite-level perturbations.
        severity: expression level in [0, 1]; 0.0 = healthy phenotype,
            1.0 = full disease expression.  Scales both the gene
            perturbation effect and the pool interpolation between
            normal and pathological concentrations.
        onset_age_years: typical age of clinical onset.
        description: free-text summary with literature anchors.
    """

    name: str
    category: str = "enzyme_deficiency"
    gene_perturbations: list[GenePerturbation] = field(default_factory=list)
    metabolite_perturbations: list[MetabolitePerturbation] = field(
        default_factory=list)
    severity: float = 1.0
    onset_age_years: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(
                f"severity must be within [0, 1], got {self.severity}")


# ============================================================================
# Gene-reaction association resolution
# ============================================================================

def _resolve_affected_reactions(model: MetabolicModel,
                                gene_id: str) -> list[str]:
    """Resolve the reactions gated by *gene_id* on *model*.

    Resolution order:

    1. ``model.genes`` registry (``Gene.protein_reaction_rules`` lists)
    2. an optional ``model.gene_reactions`` dict, in either direction
       (``gene -> [reactions]`` or ``reaction -> [genes]``)
    3. scan of every reaction's GPR string for the gene token
    4. the curated :data:`~helixlang.plugins.runtime.metabolism.ECOLI_CORE_GENE_REACTIONS`
       table (fallback when the model carries no association data)
    """
    found: list[str] = []

    genes = getattr(model, "genes", None)
    if isinstance(genes, dict):
        gene_entry = genes.get(gene_id)
        if gene_entry is not None:
            rules = getattr(gene_entry, "protein_reaction_rules", None)
            if rules:
                found.extend(str(rid) for rid in rules)

    raw_map = getattr(model, "gene_reactions", None)
    if isinstance(raw_map, dict):
        direct = raw_map.get(gene_id)
        if isinstance(direct, (list, tuple)):
            found.extend(str(rid) for rid in direct)
        else:
            for rxn_id, associated_genes in raw_map.items():
                if gene_id in (associated_genes or []):
                    found.append(str(rxn_id))

    if not found:
        pattern = re.compile(rf"(?<![\w.]){re.escape(gene_id)}(?![\w.])")
        for rxn_id, rxn in model.reactions.items():
            gpr = rxn.gene_reaction_rule
            if gpr and pattern.search(gpr):
                found.append(rxn_id)

    if not found:
        found.extend(ECOLI_CORE_GENE_REACTIONS.get(gene_id, ()))

    return sorted(dict.fromkeys(found))


def _gpr_blocked(rule: str, ko_genes: set[str]) -> bool:
    """Evaluate whether a GPR rule is inactive given knocked-out genes.

    Standard boolean semantics: the rule is a disjunction (OR) of
    conjunctions (AND); it stays active when at least one OR-alternative
    has no knocked-out gene (an intact isozyme compensates).
    """
    alternatives = [alt for alt in re.split(r"\bor\b", rule) if alt.strip()]
    if not alternatives:
        return True
    for alternative in alternatives:
        term_genes = {
            token.strip().strip("()'\"")
            for token in re.split(r"\band\b", alternative)
            if token.strip()
        }
        term_genes.discard("")
        if term_genes and not (term_genes & ko_genes):
            return False
    return True


# ============================================================================
# Disease application to a metabolic model (doc/27 §5.4)
# ============================================================================

def initial_metabolite_pools(disease: DiseaseState) -> dict[str, float]:
    """Return the pool-seeding concentrations for *disease*.

    Each metabolite pool starts at the severity-interpolated value
    between ``normal_concentration_mm`` (severity 0) and
    ``initial_concentration_mm`` (severity 1).  The returned dict feeds
    ``MetabolitePool(model, initial=...)`` pool initialization.
    """
    return {
        mp.metabolite_id: mp.normal_concentration_mm
        + disease.severity
        * (mp.initial_concentration_mm - mp.normal_concentration_mm)
        for mp in disease.metabolite_perturbations
    }


def apply_disease_state(model: MetabolicModel,
                        disease: DiseaseState) -> MetabolicModel:
    """Apply the perturbations of *disease* to a human GEM.

    The input model is never mutated: a deep copy receives all changes
    and is returned.

    Gene perturbations resolve each gene's affected reactions via
    ``affected_reactions`` (explicit) or the model gene-reaction
    association (registry / ``gene_reactions`` map / GPR scan), then:

    - knockout/downregulation: effective residual fraction is
      ``1 - severity * (1 - activity_fraction)``; at (near) zero the
      reaction upper bound becomes 0 (and the lower bound too when the
      reaction is reversible), otherwise bounds are scaled by the
      fraction.  GPR boolean logic is honored: reactions with an intact
      OR-isozyme are left untouched.
    - overexpression: upper bound scaled by
      ``1 + severity * (1/activity_fraction - 1)``
      (activity_fraction 0.5, severity 1 -> 2x capacity).

    Metabolite perturbations seed ``metabolite_pool_initials`` on the
    returned model (see :func:`initial_metabolite_pools`) and scale the
    absolute exchange-reaction bounds by ``transport_restriction``
    (``block_export`` closes export entirely).

    Args:
        model: a :class:`~helixlang.plugins.runtime.metabolism.MetabolicModel`
            (e.g. a human GEM loaded through ``helixlang.plugins.human.gem_human``).
        disease: the :class:`DiseaseState` to express.

    Returns:
        a new, diseased :class:`~helixlang.plugins.runtime.metabolism.MetabolicModel`.
    """
    new_model = copy.deepcopy(model)
    ko_genes = {
        gp.gene_id
        for gp in disease.gene_perturbations
        if gp.perturbation_type == "knockout"
    }

    for gp in disease.gene_perturbations:
        targets = gp.affected_reactions or _resolve_affected_reactions(
            new_model, gp.gene_id)
        for rxn_id in targets:
            rxn = new_model.reactions.get(rxn_id)
            if rxn is None:
                continue
            if gp.perturbation_type == "overexpress":
                fold = min(
                    1.0 / max(gp.activity_fraction, 1.0 / _MAX_OVEREXP_FOLD),
                    _MAX_OVEREXP_FOLD,
                )
                rxn.upper_bound *= 1.0 + disease.severity * (fold - 1.0)
                continue
            gpr = rxn.gene_reaction_rule
            if gpr and not _gpr_blocked(gpr, ko_genes):
                continue
            fraction = (
                1.0 - disease.severity * (1.0 - gp.activity_fraction))
            if fraction <= _MIN_EFFECTIVE_FRACTION:
                rxn.upper_bound = 0.0
                if rxn.lower_bound < 0.0:
                    rxn.lower_bound = 0.0
            else:
                rxn.upper_bound *= fraction
                rxn.lower_bound *= fraction

    for mp in disease.metabolite_perturbations:
        restriction = (
            0.0
            if mp.perturbation_type == "block_export"
            else mp.transport_restriction
        )
        if restriction < 1.0:
            for rxn_id, rxn in new_model.reactions.items():
                is_exchange = (
                    rxn.subsystem == "exchange" or rxn_id.startswith("EX_"))
                if is_exchange and mp.metabolite_id in rxn.stoichiometry:
                    rxn.upper_bound *= restriction
                    rxn.lower_bound *= restriction

    setattr(new_model, _METABOLITE_POOL_ATTR, initial_metabolite_pools(disease))
    return new_model


# ============================================================================
# Pre-defined disease profiles (doc/27 §5.5)
# ============================================================================

#: Literature-anchored disease profiles.  Concentration units follow the
#: sources: amino acids and glucose/lactate are mM; sphingolipid storage
#: pools (glucosylceramide, Gb3) keep their published uM magnitudes.
DISEASE_PROFILES: dict[str, DiseaseState] = {
    "GAUCHER": DiseaseState(
        name="Gaucher disease type 1",
        category="enzyme_deficiency",
        gene_perturbations=[GenePerturbation("GBA1", "knockout", 0.0)],
        metabolite_perturbations=[
            MetabolitePerturbation("glucosylceramide", "accumulate", 25.0, 0.5),
        ],
        onset_age_years=30.0,
        description=(
            "Lysosomal glucocerebrosidase (GBA1) deficiency; "
            "glucosylceramide accumulates in macrophage lysosomes "
            "(uM-scale lipid pool). Beutler 2004, Mol Genet Metab."
        ),
    ),
    "PKU": DiseaseState(
        name="Phenylketonuria",
        category="enzyme_deficiency",
        gene_perturbations=[GenePerturbation("PAH", "downregulate", 0.05)],
        metabolite_perturbations=[
            MetabolitePerturbation("phenylalanine", "accumulate", 2.4, 0.09),
        ],
        description=(
            "Phenylalanine hydroxylase deficiency (~5% residual "
            "activity in classic PKU); plasma phenylalanine rises from "
            "~90 uM to >1.2 mM. Blau 2010, J Inherit Metab Dis."
        ),
    ),
    "FABRY": DiseaseState(
        name="Fabry disease",
        category="enzyme_deficiency",
        gene_perturbations=[GenePerturbation("GLA", "knockout", 0.0)],
        metabolite_perturbations=[
            MetabolitePerturbation("Gb3", "accumulate", 15.0, 0.5),
        ],
        onset_age_years=10.0,
        description=(
            "Alpha-galactosidase A (GLA) deficiency; globotriaosylceramide "
            "(Gb3, uM-scale lipid pool) deposits in vascular endothelium. "
            "Zarate & Hopkin 2017, Mol Genet Metab."
        ),
    ),
    "MSUD": DiseaseState(
        name="Maple syrup urine disease",
        category="enzyme_deficiency",
        gene_perturbations=[GenePerturbation("BCKDHA", "knockout", 0.0)],
        metabolite_perturbations=[
            MetabolitePerturbation("leucine", "accumulate", 3.0, 0.15),
        ],
        onset_age_years=0.02,
        description=(
            "Branched-chain alpha-ketoacid dehydrogenase (BCKDHA) "
            "deficiency; neonatal leucine elevation to >3 mM with "
            "maple-syrup odor. Strauss 2016, GeneReviews."
        ),
    ),
    "DIABETES_T2": DiseaseState(
        name="Type 2 diabetes mellitus (simplified)",
        category="receptor_dysfunction",
        gene_perturbations=[
            GenePerturbation("IRS1", "downregulate", 0.3),
            GenePerturbation("INSR", "downregulate", 0.3),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("glucose", "accumulate", 7.0, 5.0),
        ],
        onset_age_years=45.0,
        description=(
            "Insulin signaling resistance (IRS1/INSR at ~30% signaling "
            "capacity reduces GLUT4 recruitment); fasting plasma glucose "
            "at the diabetic threshold of 7 mM vs 5 mM normal. "
            "DeFronzo 2015, Diabetes."
        ),
    ),
    "WARBURG_CANCER": DiseaseState(
        name="Cancer metabolism (Warburg effect)",
        category="cancer_metabolism",
        gene_perturbations=[
            GenePerturbation("HK2", "overexpress", 0.5),
            GenePerturbation("PDK1", "overexpress", 0.5),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("lactate", "accumulate", 10.0, 1.8),
        ],
        onset_age_years=60.0,
        description=(
            "Oncogene-driven glycolytic reprogramming: HK2 hexokinase "
            "and PDK1 pyruvate dehydrogenase kinase overexpressed 2x, "
            "diverting pyruvate to lactate (~10 mM intratumoral vs "
            "~1.8 mM plasma). Vander Heiden 2009, Science."
        ),
    ),
    "HYPERTENSION": DiseaseState(
        name="Essential hypertension",
        category="cardiovascular",
        gene_perturbations=[
            GenePerturbation("ACE", "overexpress", 0.5),
            GenePerturbation("AGTR1", "overexpress", 0.5),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("angiotensin_II", "accumulate", 0.08, 0.03),
        ],
        onset_age_years=35.0,
        description=(
            "Essential hypertension: elevated angiotensin II (0.08 vs "
            "0.03 mM normal) with ACE/AGTR1 overexpression driving "
            "vasoconstriction and sodium retention. Whelton 2018, "
            "ACC/AHA Hypertension Guidelines."
        ),
    ),
    "ASTHMA": DiseaseState(
        name="Bronchial asthma",
        category="respiratory",
        gene_perturbations=[
            GenePerturbation("IL4", "overexpress", 0.5),
            GenePerturbation("IL13", "overexpress", 0.5),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("histamine", "accumulate", 0.005, 0.001),
        ],
        onset_age_years=5.0,
        description=(
            "Type 2 inflammatory asthma: IL-4/IL-13 overexpression driving "
            "IgE production and eosinophil recruitment; mast cell histamine "
            "release (5 vs 1 uM normal) causing bronchospasm. "
            "Lambrecht 2015, Nat Rev Immunol."
        ),
    ),
    "COPD": DiseaseState(
        name="Chronic obstructive pulmonary disease",
        category="respiratory",
        gene_perturbations=[
            GenePerturbation("SERPINA1", "downregulate", 0.3),
            GenePerturbation("MMP9", "overexpress", 0.5),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("leukotriene_B4", "accumulate", 0.003, 0.0005),
        ],
        onset_age_years=50.0,
        description=(
            "COPD: alpha-1 antitrypsin deficiency (SERPINA1 at 30% "
            "residual) with MMP9 overexpression causing emphysematous "
            "destruction; elevated LTB4 (3 vs 0.5 uM) driving neutrophilic "
            "inflammation. Barnes 2016, Nat Rev Dis Primers."
        ),
    ),
    "DEPRESSION": DiseaseState(
        name="Major depressive disorder",
        category="neurological",
        gene_perturbations=[
            GenePerturbation("SLC6A4", "downregulate", 0.5),
            GenePerturbation("HTR2A", "downregulate", 0.4),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("serotonin", "deplete", 0.0001, 0.0003),
        ],
        onset_age_years=25.0,
        description=(
            "Major depressive disorder: serotonin transporter (SLC6A4) "
            "downregulated 50% with 5-HT2A receptor downregulation; "
            "synaptic 5-HT reduced to 0.1 vs 0.3 uM normal. "
            "Duman 2018, Biol Psychiatry."
        ),
    ),
    "EPILEPSY": DiseaseState(
        name="Epilepsy (generalized)",
        category="neurological",
        gene_perturbations=[
            GenePerturbation("SCN1A", "downregulate", 0.4),
            GenePerturbation("GABRA1", "downregulate", 0.4),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("glutamate", "accumulate", 12.0, 8.0),
        ],
        onset_age_years=10.0,
        description=(
            "Generalized epilepsy: Nav1.1 (SCN1A) and GABA-A receptor "
            "(GABRA1) downregulation causing excitatory/inhibitory imbalance; "
            "elevated extracellular glutamate (12 vs 8 mM normal) "
            "lowering seizure threshold. Brodie 2018, Lancet Neurol."
        ),
    ),
    "OSTEOPOROSIS": DiseaseState(
        name="Postmenopausal osteoporosis",
        category="metabolic",
        gene_perturbations=[
            GenePerturbation("RANKL", "overexpress", 0.5),
            GenePerturbation("OPG", "downregulate", 0.5),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("calcium", "accumulate", 2.8, 2.4),
        ],
        onset_age_years=55.0,
        description=(
            "Postmenopausal osteoporosis: RANKL overexpression with OPG "
            "downregulation driving osteoclastogenesis; slightly elevated "
            "serum calcium (2.8 vs 2.4 mM) reflecting net bone resorption. "
            "Compston 2019, Lancet."
        ),
    ),
    "HIV": DiseaseState(
        name="HIV-1 infection",
        category="infectious",
        gene_perturbations=[
            GenePerturbation("CD4", "downregulate", 0.3),
            GenePerturbation("CCR5", "overexpress", 0.5),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("viral_load", "accumulate", 100000.0, 0.0),
        ],
        onset_age_years=30.0,
        description=(
            "HIV-1 infection: CD4+ T cell depletion (CD4 at 30% residual) "
            "with CCR5 co-receptor overexpression; viral load rises to "
            "100,000 copies/mL from undetectable. "
            "Deeks 2015, Lancet."
        ),
    ),
    "TUBERCULOSIS": DiseaseState(
        name="Pulmonary tuberculosis",
        category="infectious",
        gene_perturbations=[
            GenePerturbation("IFNG", "downregulate", 0.4),
            GenePerturbation("TNF", "overexpress", 0.5),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("interferon_gamma", "deplete", 0.01, 0.05),
        ],
        onset_age_years=25.0,
        description=(
            "Pulmonary tuberculosis: IFN-gamma deficiency (0.01 vs 0.05 "
            "uM) impairing macrophage activation; TNF-alpha overexpression "
            "driving granulomatous inflammation. "
            "Pai 2016, Lancet Infect Dis."
        ),
    ),
    "ANEMIA": DiseaseState(
        name="Iron deficiency anemia",
        category="hematological",
        gene_perturbations=[],
        metabolite_perturbations=[
            MetabolitePerturbation("iron", "deplete", 5.0, 20.0),
            MetabolitePerturbation("hemoglobin", "deplete", 70.0, 140.0),
        ],
        onset_age_years=20.0,
        description=(
            "Iron deficiency anemia: serum iron depleted to 5 from 20 uM; "
            "hemoglobin reduced to 70 from 140 g/L causing tissue hypoxia. "
            "Kassebaum 2016, Lancet Haematol."
        ),
    ),
    "CROHNS": DiseaseState(
        name="Crohn's disease",
        category="autoimmune",
        gene_perturbations=[
            GenePerturbation("NOD2", "downregulate", 0.3),
            GenePerturbation("TNF", "overexpress", 0.6),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("TNF_alpha", "accumulate", 0.015, 0.003),
        ],
        onset_age_years=25.0,
        description=(
            "Crohn's disease: NOD2 loss-of-function (30% residual) "
            "impairing innate immune response to gut bacteria; TNF-alpha "
            "elevated to 15 vs 3 pg/mL driving transmural inflammation. "
            "Baumgart 2012, Lancet."
        ),
    ),
    "ULCERATIVE_COLITIS": DiseaseState(
        name="Ulcerative colitis",
        category="autoimmune",
        gene_perturbations=[
            GenePerturbation("IL10", "downregulate", 0.3),
            GenePerturbation("TNF", "overexpress", 0.5),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("TNF_alpha", "accumulate", 0.012, 0.003),
        ],
        onset_age_years=35.0,
        description=(
            "Ulcerative colitis: IL-10 deficiency (30% residual) with "
            "TNF-alpha overexpression driving mucosal inflammation; "
            "restricted to colonic mucosa. Ungaro 2017, Lancet."
        ),
    ),
    "PSORIASIS": DiseaseState(
        name="Psoriasis vulgaris",
        category="autoimmune",
        gene_perturbations=[
            GenePerturbation("IL17A", "overexpress", 0.6),
            GenePerturbation("IL23", "overexpress", 0.5),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("IL17", "accumulate", 0.01, 0.002),
        ],
        onset_age_years=30.0,
        description=(
            "Psoriasis vulgaris: IL-17A/IL-23 axis overactivation driving "
            "keratinocyte hyperproliferation; IL-17 elevated to 10 vs "
            "2 pg/mL. Boehncke 2015, Lancet."
        ),
    ),
    "SCHIZOPHRENIA": DiseaseState(
        name="Schizophrenia",
        category="neurological",
        gene_perturbations=[
            GenePerturbation("DRD2", "overexpress", 0.5),
            GenePerturbation("NRG1", "downregulate", 0.4),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("dopamine", "accumulate", 0.008, 0.003),
        ],
        onset_age_years=22.0,
        description=(
            "Schizophrenia: DRD2 overexpression with neuregulin 1 "
            "downregulation; mesolimbic dopamine elevated to 8 vs "
            "3 uM driving positive symptoms. Howes 2017, Lancet."
        ),
    ),
    "HYPERLIPIDEMIA": DiseaseState(
        name="Hyperlipidemia (mixed)",
        category="metabolic",
        gene_perturbations=[
            GenePerturbation("LDLR", "downregulate", 0.4),
            GenePerturbation("PCSK9", "overexpress", 0.5),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("LDL_cholesterol", "accumulate", 5.0, 3.0),
        ],
        onset_age_years=45.0,
        description=(
            "Mixed hyperlipidemia: LDL receptor downregulated (40% residual) "
            "with PCSK9 overexpression; LDL-C elevated to 5 from 3 mM. "
            "Ference 2017, Eur Heart J."
        ),
    ),
    "GOUT": DiseaseState(
        name="Gout (hyperuricemia)",
        category="metabolic",
        gene_perturbations=[
            GenePerturbation("SLC2A9", "downregulate", 0.3),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("uric_acid", "accumulate", 0.54, 0.25),
        ],
        onset_age_years=40.0,
        description=(
            "Gout: URAT1/GLUT9 (SLC2A9) dysfunction causing renal urate "
            "underexcretion; serum urate elevated to 540 vs 250 uM "
            "with monosodium urate crystal deposition. "
            "Richette 2017, Lancet."
        ),
    ),
    "HYPOTHYROIDISM": DiseaseState(
        name="Hypothyroidism (primary)",
        category="endocrine",
        gene_perturbations=[
            GenePerturbation("TPO", "downregulate", 0.3),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("T4", "deplete", 30.0, 120.0),
        ],
        onset_age_years=40.0,
        description=(
            "Primary hypothyroidism: thyroid peroxidase (TPO) at 30% "
            "residual activity; free T4 reduced to 30 from 120 pM "
            "causing metabolic slowing. Chaker 2017, Lancet."
        ),
    ),
    "GERD": DiseaseState(
        name="Gastroesophageal reflux disease",
        category="gastrointestinal",
        gene_perturbations=[],
        metabolite_perturbations=[
            MetabolitePerturbation("gastric_acid", "accumulate", 50.0, 10.0),
        ],
        onset_age_years=35.0,
        description=(
            "GERD: gastric acid hypersecretion (50 vs 10 mM) with "
            "lower esophageal sphincter dysfunction causing chronic "
            "acid reflux and mucosal injury. "
            "Katz 2022, Gastroenterology."
        ),
    ),
    "CHRONIC_PAIN": DiseaseState(
        name="Chronic pain syndrome",
        category="neurological",
        gene_perturbations=[
            GenePerturbation("SCN9A", "overexpress", 0.5),
            GenePerturbation("COMT", "downregulate", 0.4),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("substance_P", "accumulate", 0.005, 0.001),
        ],
        onset_age_years=35.0,
        description=(
            "Chronic pain syndrome: Nav1.7 (SCN9A) overexpression with "
            "COMT downregulation causing pain hypersensitivity; substance P "
            "elevated to 5 vs 1 pM driving central sensitization. "
            "Colloca 2017, Nat Rev Dis Primers."
        ),
    ),
    "ALLERGIC_RHINITIS": DiseaseState(
        name="Allergic rhinitis",
        category="immune",
        gene_perturbations=[
            GenePerturbation("IL4", "overexpress", 0.5),
            GenePerturbation("IL13", "overexpress", 0.5),
        ],
        metabolite_perturbations=[
            MetabolitePerturbation("histamine", "accumulate", 0.003, 0.0005),
        ],
        onset_age_years=10.0,
        description=(
            "Allergic rhinitis: IL-4/IL-13 driven IgE-mediated "
            "hypersensitivity; mast cell histamine release (3 vs "
            "0.5 uM) causing nasal congestion and rhinorrhea. "
            "Bousquet 2020, Allergy."
        ),
    ),
}
