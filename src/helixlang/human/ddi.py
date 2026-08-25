"""Drug-drug interaction modeling via CYP450 enzyme competition (doc/28).

Interactions are modeled as rules between a *substrate* (the drug being
affected) and a *perpetrator* (the interacting drug or an enzyme-state
condition).  Each rule carries a ``fold_change`` multiplier applied to the
substrate's systemic clearance when the rule fires:

- ``fold_change < 1`` — reduced clearance (accumulation / overdose risk),
  typical for enzyme **inhibition** or impaired prodrug activation.
- ``fold_change > 1`` — increased clearance (sub-therapeutic risk),
  typical for enzyme **induction**.

Two trigger mechanisms are supported:

1. **Co-administration** — ``interacting_drug`` is another drug name and
   fires whenever both drugs appear in the simulated regimen.
2. **Enzyme-state** — ``interacting_drug`` names the enzyme itself
   (e.g. ``"CYP2D6"``); the rule fires from the supplied
   ``cyp_profiles`` (enzyme -> fractional activity) when the enzyme is
   deficient (< 0.25 residual activity, i.e. a poor metabolizer) for
   inhibitory rules, or ultrarapid (> 1.5) for inducing rules.

Drug matching is case-insensitive; returned dictionaries use the
lower-cased drug name as key.

Module structure:
    DDIRule                   single interaction rule
    DDIModel                  rule database + clearance/alert computation
    DEFAULT_DDI_RULES         curated literature-anchored rule list
    create_default_ddi_model  convenience DDIModel factory
    assess_additive_toxicity  class-based additive toxicity screening

References:
- Lynch T & Price A. Am Fam Physician 2007 (CYP450 interactions primer)
- Relling MV et al. Clin Pharmacol Ther 2014 (CPIC: CYP2D6 tamoxifen)
- van Erp NP et al. J Clin Oncol 2007 (CYP3A4-mediated imatinib exposure)
- Perazella MA. Am J Med Sci 2000 (NSAID + cisplatin nephrotoxicity)
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "DDIRule",
    "DDIModel",
    "DEFAULT_DDI_RULES",
    "create_default_ddi_model",
    "assess_additive_toxicity",
]

#: valid :attr:`DDIRule.interaction_type` values ("additive_toxicity" and
#: "monitoring" carry no clearance change and surface only as alerts)
_INTERACTION_TYPES = ("inhibition", "induction", "additive_toxicity", "monitoring")
#: valid :attr:`DDIRule.severity` values
_SEVERITY_LEVELS = ("mild", "moderate", "severe", "contraindicated")
#: ranking used to sort alerts (higher = more urgent)
_SEVERITY_RANK = {"mild": 0, "moderate": 1, "severe": 2, "contraindicated": 3}
#: residual fractional activity at or below which an enzyme counts as
#: deficient (CPIC poor-metabolizer territory) for enzyme-state rules
_ENZYME_DEFICIENT_FRACTION = 0.25
#: fractional activity at or above which an enzyme counts as ultrarapid
#: for enzyme-state induction rules
_ENZYME_ULTRARAPID_FRACTION = 1.5
#: bounds applied to the compounded clearance multiplier
_MIN_MULTIPLIER = 0.01
_MAX_MULTIPLIER = 20.0
#: recommendation string per severity level
_SEVERITY_RECOMMENDATIONS = {
    "mild": "Monitor for symptoms; dose adjustment usually not required.",
    "moderate": "Monitor therapeutic response; consider dose adjustment.",
    "severe": "Avoid combination if possible; otherwise reduce dose and monitor closely.",
    "contraindicated": "Do not co-administer; select an alternative therapy.",
}


# ============================================================================
# Data structures
# ============================================================================

@dataclass(slots=True)
class DDIRule:
    """A single drug-drug interaction rule.

    Attributes:
        substrate: drug being affected (lower-case canonical name).
        interacting_drug: the perpetrator.  Either another drug name or,
            for enzyme-state rules, the enzyme name itself (e.g.
            ``"CYP2D6"``) denoting a loss-of-function / ultrarapid
            metabolizer phenotype resolved via ``cyp_profiles``.
        enzyme: CYP (or transport) protein mediating the interaction.
        interaction_type: ``"inhibition"``, ``"induction"``,
            ``"additive_toxicity"`` (PD synergy, no clearance change) or
            ``"monitoring"`` (informational, e.g. renally cleared drug).
        fold_change: multiplier on substrate clearance when the rule
            fires (0.2 = 80% inhibition; 3.0 = 3x induction).
        severity: one of ``"mild"``, ``"moderate"``, ``"severe"``,
            ``"contraindicated"``.
        clinical_effect: human-readable description shown in alerts.
    """

    substrate: str
    interacting_drug: str
    enzyme: str
    interaction_type: str
    fold_change: float
    severity: str = "moderate"
    clinical_effect: str = ""

    def __post_init__(self) -> None:
        if self.interaction_type not in _INTERACTION_TYPES:
            raise ValueError(
                f"interaction_type must be one of {_INTERACTION_TYPES}, "
                f"got {self.interaction_type!r}"
            )
        if self.severity not in _SEVERITY_LEVELS:
            raise ValueError(
                f"severity must be one of {_SEVERITY_LEVELS}, got {self.severity!r}"
            )
        if self.fold_change <= 0.0:
            raise ValueError(f"fold_change must be > 0, got {self.fold_change}")
        self.substrate = self.substrate.strip().lower()
        self.interacting_drug = self.interacting_drug.strip().lower()
        self.enzyme = self.enzyme.strip()

    @property
    def is_enzyme_state_rule(self) -> bool:
        """True when the perpetrator is an enzyme phenotype, not a drug."""
        return self.interacting_drug == self.enzyme.lower()


@dataclass
class DDIModel:
    """Computes drug-drug interactions for co-administered drugs.

    Attributes:
        rules: interaction rules to evaluate.
    """

    rules: list[DDIRule] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rule_triggered(
        self,
        rule: DDIRule,
        name_set: set[str],
        cyp_profiles: dict[str, float],
    ) -> bool:
        """Return whether *rule* fires for the given regimen and phenotypes."""
        if rule.is_enzyme_state_rule:
            activity = cyp_profiles.get(rule.enzyme, 1.0)
            if rule.interaction_type == "induction":
                return activity >= _ENZYME_ULTRARAPID_FRACTION
            return 0.0 < activity < _ENZYME_DEFICIENT_FRACTION
        return rule.interacting_drug in name_set

    def _fallback_effect(self, rule: DDIRule) -> str:
        """Generate a human-readable effect string when none was authored."""
        if rule.is_enzyme_state_rule:
            direction = (
                "limits" if rule.fold_change < 1.0 else "accelerates"
            )
            return (
                f"Altered {rule.enzyme} activity {direction} "
                f"{rule.substrate} metabolism (clearance x{rule.fold_change:g})"
            )
        verb = (
            "inhibits" if rule.interaction_type == "inhibition" else "induces"
        )
        return (
            f"{rule.interacting_drug} {verb} {rule.enzyme} -> "
            f"{rule.substrate} clearance x{rule.fold_change:g}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_clearance_modifiers(
        self,
        drug_names: list[str],
        cyp_profiles: dict[str, float],
    ) -> dict[str, float]:
        """Return ``{drug_name: clearance_multiplier}`` for all drugs.

        Multipliers compound multiplicatively across every triggered rule
        and are capped to ``[0.01, 20]``.  A multiplier < 1.0 means
        reduced clearance (accumulation risk); > 1.0 means increased
        clearance (sub-therapeutic risk).

        Genotype-derived enzyme capacity is intentionally NOT applied
        here; it is owned by the calling facade (doc/29 §5).  Enzyme-state
        rules (``interacting_drug == enzyme``) remain permitted: their
        ``fold_change`` is an explicit literature-anchored interaction
        magnitude, not a blanket activity copy.

        Args:
            drug_names: drugs in the active regimen (any letter case).
            cyp_profiles: enzyme -> fractional activity (1.0 = normal);
                consumed only by enzyme-state rules.

        Returns:
            ``{lower_case_drug_name: multiplier}`` covering exactly the
            input drugs.
        """
        name_set = {name.strip().lower() for name in drug_names}
        modifiers = {name: 1.0 for name in name_set}

        for rule in self.rules:
            if rule.interaction_type in ("additive_toxicity", "monitoring"):
                continue
            if rule.substrate not in name_set:
                continue
            if not self._rule_triggered(rule, name_set, cyp_profiles):
                continue
            modifiers[rule.substrate] *= rule.fold_change

        floor, ceiling = _MIN_MULTIPLIER, _MAX_MULTIPLIER
        return {
            name: min(max(mult, floor), ceiling)
            for name, mult in modifiers.items()
        }

    def get_clinical_alerts(
        self,
        drug_names: list[str],
        cyp_profiles: dict[str, float],
    ) -> list[dict]:
        """Return triggered interactions sorted by descending severity.

        Args:
            drug_names: drugs in the active regimen (any letter case).
            cyp_profiles: enzyme -> fractional activity (1.0 = normal).

        Returns:
            List of dicts with keys ``severity``, ``drugs``, ``enzyme``,
            ``effect`` and ``recommendation``.
        """
        name_set = {name.strip().lower() for name in drug_names}
        alerts: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for rule in self.rules:
            if rule.substrate not in name_set:
                continue
            if not self._rule_triggered(rule, name_set, cyp_profiles):
                continue
            key = (rule.substrate, rule.interacting_drug)
            if key in seen:
                continue
            seen.add(key)
            drugs = [rule.substrate]
            if not rule.is_enzyme_state_rule:
                drugs.append(rule.interacting_drug)
            alerts.append({
                "severity": rule.severity,
                "drugs": drugs,
                "enzyme": rule.enzyme,
                "effect": rule.clinical_effect or self._fallback_effect(rule),
                "recommendation": _SEVERITY_RECOMMENDATIONS[rule.severity],
            })

        alerts.sort(key=lambda a: _SEVERITY_RANK[a["severity"]], reverse=True)
        return alerts


# ============================================================================
# Additive (class-based) toxicity screening
# ============================================================================

#: drugs associated with QT-interval prolongation / torsades de pointes
_QT_PROLONGING_DRUGS = {
    "tamoxifen", "cisplatin", "ondansetron", "haloperidol", "methadone",
    "ciprofloxacin", "levofloxacin", "amitriptyline", "quetiapine",
    "hydroxychloroquine", "azithromycin",
}
#: non-steroidal anti-inflammatory drugs
_NSAIDS = {
    "ibuprofen", "naproxen", "diclofenac", "ketorolac", "celecoxib",
    "indomethacin", "aspirin", "meloxicam",
}
#: direct nephrotoxins (NSAIDs handled separately)
_DIRECT_NEPHROTOXINS = {
    "cisplatin", "carboplatin", "gentamicin", "tobramycin", "vancomycin",
    "amphotericin_b", "cyclosporine", "tacrolimus",
}
#: high-extraction CYP3A4 substrates contributing to hepatic load
_CYP3A4_SUBSTRATES = {
    "imatinib", "tamoxifen", "atorvastatin", "simvastatin", "lovastatin",
    "carbamazepine", "cyclosporine", "tacrolimus", "midazolam",
    "clarithromycin", "erythromycin",
}

_TOXICITY_RANK = {"info": 0, "mild": 1, "moderate": 2, "severe": 3}


def assess_additive_toxicity(drug_names: list[str]) -> list[dict]:
    """Screen a regimen for class-based additive toxicities.

    Complements pairwise DDI rules with three pharmacologic-class checks:

    - **QT prolongation** — >= 2 QT-prolonging drugs raises torsades risk.
    - **Nephrotoxicity** — cisplatin + NSAIDs is severely additive;
      metformin adds lactic-acidosis risk once renal injury is plausible.
    - **Hepatotoxicity** — >= 2 CYP3A4 substrates indicates hepatic load.

    Args:
        drug_names: drugs in the active regimen (any letter case).

    Returns:
        List of dicts with keys ``toxicity_type``, ``drugs``,
        ``severity``, ``mechanism`` and ``recommendation``, sorted by
        descending severity.
    """
    names = {name.strip().lower() for name in drug_names}
    alerts: list[dict] = []

    qt_hits = sorted(names & _QT_PROLONGING_DRUGS)
    if len(qt_hits) >= 2:
        alerts.append({
            "toxicity_type": "qt_prolongation",
            "drugs": qt_hits,
            "severity": "severe",
            "mechanism": "Additive cardiac K+-channel blockade (hERG)",
            "recommendation": "Obtain baseline ECG, correct K+/Mg2+, avoid additional QT drugs.",
        })
    elif len(qt_hits) == 1:
        alerts.append({
            "toxicity_type": "qt_prolongation",
            "drugs": qt_hits,
            "severity": "moderate",
            "mechanism": "Single-agent QT prolongation risk",
            "recommendation": "Monitor ECG if risk factors present.",
        })

    nsaid_hits = names & _NSAIDS
    renal_hits = sorted((names & _DIRECT_NEPHROTOXINS) | nsaid_hits)
    has_metformin = "metformin" in names
    if "cisplatin" in names and nsaid_hits:
        alerts.append({
            "toxicity_type": "nephrotoxicity",
            "drugs": sorted({"cisplatin"} | nsaid_hits),
            "severity": "severe",
            "mechanism": "Additive proximal-tubule injury; NSAIDs blunt compensatory prostaglandin vasodilation",
            "recommendation": "Avoid NSAIDs during cisplatin cycles; ensure aggressive hydration.",
        })
    elif len(renal_hits) >= 2:
        alerts.append({
            "toxicity_type": "nephrotoxicity",
            "drugs": renal_hits,
            "severity": "moderate",
            "mechanism": "Multiple renal stressors with additive tubular injury",
            "recommendation": "Monitor serum creatinine and urine output.",
        })
    if has_metformin and renal_hits:
        alerts.append({
            "toxicity_type": "nephrotoxicity",
            "drugs": sorted({"metformin", *renal_hits}),
            "severity": "moderate",
            "mechanism": "Renal injury reduces metformin clearance (lactic acidosis risk)",
            "recommendation": "Hold metformin if eGFR falls below 45 mL/min/1.73m2.",
        })

    hepatic_hits = sorted(names & _CYP3A4_SUBSTRATES)
    if len(hepatic_hits) >= 2:
        alerts.append({
            "toxicity_type": "hepatotoxicity",
            "drugs": hepatic_hits,
            "severity": "moderate" if len(hepatic_hits) == 2 else "severe",
            "mechanism": f"{len(hepatic_hits)} concurrent CYP3A4 substrates increase hepatic metabolic load",
            "recommendation": "Monitor ALT/AST periodically; review hepatic clearance capacity.",
        })

    alerts.sort(key=lambda a: _TOXICITY_RANK[a["severity"]], reverse=True)
    return alerts


# ============================================================================
# Built-in rule database
# ============================================================================

#: Curated literature-anchored interaction rules shipped with HelixLang.
#: Drug names are lower-case canonical identifiers; enzyme-state rules
#: repeat the enzyme name in ``interacting_drug``.
DEFAULT_DDI_RULES: list[DDIRule] = [
    DDIRule(
        substrate="ibuprofen",
        interacting_drug="metformin",
        enzyme="CYP2C9",
        interaction_type="inhibition",
        fold_change=0.7,
        severity="mild",
        clinical_effect=(
            "Competitive CYP2C9 inhibition modestly reduces ibuprofen "
            "clearance; clinically insignificant for short courses."
        ),
    ),
    DDIRule(
        substrate="tamoxifen",
        interacting_drug="CYP2D6",
        enzyme="CYP2D6",
        interaction_type="inhibition",
        fold_change=0.3,
        severity="severe",
        clinical_effect=(
            "CYP2D6 poor-metabolizer phenotype cripples conversion of "
            "tamoxifen (prodrug) to active metabolite endoxifen; "
            "anti-estrogenic efficacy substantially reduced."
        ),
    ),
    DDIRule(
        substrate="imatinib",
        interacting_drug="ketoconazole",
        enzyme="CYP3A4",
        interaction_type="inhibition",
        fold_change=0.3,
        severity="severe",
        clinical_effect=(
            "Potent CYP3A4 inhibition raises imatinib exposure ~3-fold; "
            "edema, cytopenias and hepatotoxicity risk."
        ),
    ),
    DDIRule(
        substrate="imatinib",
        interacting_drug="rifampin",
        enzyme="CYP3A4",
        interaction_type="induction",
        fold_change=3.0,
        severity="severe",
        clinical_effect=(
            "CYP3A4 induction collapses imatinib exposure; sub-therapeutic "
            "levels and treatment failure."
        ),
    ),
    DDIRule(
        substrate="cisplatin",
        interacting_drug="ibuprofen",
        enzyme="kidney_prostaglandins",
        interaction_type="additive_toxicity",
        fold_change=1.0,
        severity="severe",
        clinical_effect=(
            "Additive nephrotoxicity: NSAID-prostaglandin blockade plus "
            "cisplatin proximal-tubule injury."
        ),
    ),
    DDIRule(
        substrate="metformin",
        interacting_drug="fluconazole",
        enzyme="CYP2C9",
        interaction_type="monitoring",
        fold_change=1.0,
        severity="mild",
        clinical_effect=(
            "No CYP-mediated interaction: metformin is cleared unchanged "
            "by the kidney; CYP2C9 inhibition is irrelevant to its PK."
        ),
    ),
    DDIRule(
        substrate="tamoxifen",
        interacting_drug="fluoxetine",
        enzyme="CYP2D6",
        interaction_type="inhibition",
        fold_change=0.4,
        severity="moderate",
        clinical_effect=(
            "Fluoxetine (potent CYP2D6 inhibitor) reduces endoxifen "
            "formation ~60%; switch to venlafaxine or citalopram."
        ),
    ),
    DDIRule(
        substrate="warfarin",
        interacting_drug="fluconazole",
        enzyme="CYP2C9",
        interaction_type="inhibition",
        fold_change=0.3,
        severity="severe",
        clinical_effect=(
            "CYP2C9 inhibition of S-warfarin clearance sharply raises INR; "
            "bleeding risk."
        ),
    ),
    DDIRule(
        substrate="simvastatin",
        interacting_drug="clarithromycin",
        enzyme="CYP3A4",
        interaction_type="inhibition",
        fold_change=0.15,
        severity="contraindicated",
        clinical_effect=(
            "CYP3A4 inhibition massively elevates simvastatin exposure; "
            "rhabdomyolysis risk. Contraindicated combination."
        ),
    ),
    DDIRule(
        substrate="ethinylestradiol",
        interacting_drug="carbamazepine",
        enzyme="CYP3A4",
        interaction_type="induction",
        fold_change=2.5,
        severity="severe",
        clinical_effect=(
            "Enzyme induction accelerates oral-contraceptive clearance; "
            "contraceptive failure risk. Use a non-hormonal method."
        ),
    ),
    DDIRule(
        substrate="clopidogrel",
        interacting_drug="omeprazole",
        enzyme="CYP2C19",
        interaction_type="inhibition",
        fold_change=0.6,
        severity="moderate",
        clinical_effect=(
            "CYP2C19 inhibition blunts clopidogrel bioactivation to its "
            "thiol metabolite; reduced antiplatelet effect."
        ),
    ),
    DDIRule(
        substrate="methotrexate",
        interacting_drug="ibuprofen",
        enzyme="OAT1/3",
        interaction_type="inhibition",
        fold_change=0.6,
        severity="severe",
        clinical_effect=(
            "NSAID competition for renal organic-anion transporters "
            "reduces methotrexate secretion; myelosuppression risk at "
            "antirheumatic doses."
        ),
    ),
    DDIRule(
        substrate="digoxin",
        interacting_drug="verapamil",
        enzyme="P-gp",
        interaction_type="inhibition",
        fold_change=0.4,
        severity="severe",
        clinical_effect=(
            "P-glycoprotein inhibition reduces renal and biliary digoxin "
            "secretion; toxicity (arrhythmia) risk."
        ),
    ),
]


def create_default_ddi_model() -> DDIModel:
    """Return a :class:`DDIModel` seeded with :data:`DEFAULT_DDI_RULES`."""
    return DDIModel(rules=list(DEFAULT_DDI_RULES))
