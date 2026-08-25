"""Pharmacodynamic models: dose-response, Hill equation, flux correction (doc/27)."""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PDEffect:
    """A single pharmacodynamic effect on a metabolic target."""

    target_reaction: str
    target_gene: str = ""
    effect_type: str = "inhibition"
    ec50_um: float = 1.0
    emax: float = 1.0
    hill_coefficient: float = 1.0
    baseline_effect: float = 0.0

    def __post_init__(self) -> None:
        if self.effect_type not in ("inhibition", "activation", "induction"):
            raise ValueError(f"Unknown effect_type: {self.effect_type}")
        if self.ec50_um <= 0:
            raise ValueError("ec50_um must be > 0")
        if not 0.0 <= self.emax <= 1.0:
            raise ValueError("emax must be in [0, 1]")


@dataclass
class Pharmacodynamics:
    """Complete PD model for a drug."""

    drug_name: str
    effects: list[PDEffect] = field(default_factory=list)
    dose_response_model: str = "hill"
    target_biomarkers: dict[str, float] = field(default_factory=dict)
    toxicity_concentration_um: float = 100.0
    therapeutic_window: tuple[float, float] = (1.0, 50.0)

    def __post_init__(self) -> None:
        lo, hi = self.therapeutic_window
        if lo >= hi:
            raise ValueError("therapeutic_window[0] must be < [1]")


# ---------------------------------------------------------------------------
# Hill equation core
# ---------------------------------------------------------------------------

def hill_equation(
    concentration: float,
    ec50: float,
    emax: float = 1.0,
    n: float = 1.0,
    e0: float = 0.0,
) -> float:
    """E = E0 + (Emax - E0) * C^n / (EC50^n + C^n).

    Parameters
    ----------
    concentration : drug concentration (same units as ec50)
    ec50 : concentration for half-maximal effect
    emax : maximal effect
    n : Hill coefficient (steepness)
    e0 : baseline effect at zero concentration

    Returns
    -------
    Effect value in [e0, e0 + (Emax - e0)].
    """
    if concentration <= 0.0:
        return float(e0)
    cn = float(concentration ** n)
    ec50n = float(ec50 ** n)
    return float(e0 + (emax - e0) * cn / (ec50n + cn))


def inhibition_fraction(concentration: float, ic50: float, n: float = 1.0) -> float:
    """Fraction of enzyme activity remaining after inhibition.

    Returns value in [0, 1] where 1 = fully active, 0 = fully inhibited.
    """
    if concentration <= 0.0:
        return 1.0
    cn = float(concentration ** n)
    ic50n = float(ic50 ** n)
    return float(ic50n / (ic50n + cn))


def activation_fraction(concentration: float, ec50: float, n: float = 1.0,
                        baseline: float = 0.0) -> float:
    """Fraction of activity restored by an activator.

    Returns value in [baseline, 1.0] where baseline = residual disease activity.
    """
    return baseline + (1.0 - baseline) * hill_equation(concentration, ec50, 1.0, n)


# ---------------------------------------------------------------------------
# PD effect computation
# ---------------------------------------------------------------------------

def compute_pd_effects(
    concentration_um: float,
    pd: Pharmacodynamics,
) -> dict[str, float]:
    """Compute fractional multipliers for each target reaction.

    Returns {reaction_id: multiplier} where:
    - inhibition: multiplier in [0, 1] (1 = full activity)
    - activation/induction: multiplier in [baseline, 1+fold_induction]
    """
    results: dict[str, float] = {}
    for eff in pd.effects:
        if eff.effect_type == "inhibition":
            mult = inhibition_fraction(concentration_um, eff.ec50_um, eff.hill_coefficient)
        elif eff.effect_type in ("activation", "induction"):
            mult = activation_fraction(
                concentration_um, eff.ec50_um, eff.hill_coefficient, eff.baseline_effect
            )
        else:
            mult = 1.0
        results[eff.target_reaction] = mult
    return results


def compute_pd_effects_over_time(
    concentrations: dict[str, list[float]],
    time_h: list[float],
    pd: Pharmacodynamics,
    target_organ: str = "liver",
) -> dict[str, list[float]]:
    """Compute PD effects at every recorded time point.

    Parameters
    ----------
    concentrations : organ -> [C(t)] in µM
    time_h : time points in hours
    pd : Pharmacodynamics model
    target_organ : which organ concentration to use for PD

    Returns
    -------
    {reaction_id: [multiplier(t)]} with same length as time_h.
    """
    organ_conc = concentrations.get(target_organ, concentrations.get("central", []))
    n = len(time_h)
    if not organ_conc or not pd.effects:
        return {e.target_reaction: [1.0] * n for e in pd.effects}

    result: dict[str, list[float]] = {e.target_reaction: [] for e in pd.effects}
    for i in range(n):
        c = organ_conc[i] if i < len(organ_conc) else organ_conc[-1]
        effects = compute_pd_effects(c, pd)
        for eff in pd.effects:
            result[eff.target_reaction].append(effects.get(eff.target_reaction, 1.0))
    return result


def apply_pd_to_flux_bounds(
    pd_multipliers: dict[str, float],
    original_bounds: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Apply PD multipliers to flux bounds.

    For inhibition: multiply upper_bound by multiplier.
    For activation: multiply upper_bound by multiplier (>1 for overexpression).
    """
    corrected = dict(original_bounds)
    for rxn_id, mult in pd_multipliers.items():
        if rxn_id in corrected:
            lo, hi = corrected[rxn_id]
            corrected[rxn_id] = (lo, hi * mult)
    return corrected


# ---------------------------------------------------------------------------
# Pre-defined PD profiles
# ---------------------------------------------------------------------------

PREDEFINED_PD: dict[str, Pharmacodynamics] = {
    "imiglucerase_gaucher": Pharmacodynamics(
        drug_name="imiglucerase",
        effects=[
            PDEffect(target_reaction="GBA", target_gene="GBA1",
                     effect_type="activation", ec50_um=0.5, emax=0.95,
                     hill_coefficient=1.2, baseline_effect=0.05),
        ],
        target_biomarkers={"glucosylceramide": 0.5},
    ),
    "ibuprofen_cox": Pharmacodynamics(
        drug_name="ibuprofen",
        effects=[
            PDEffect(target_reaction="COX1", effect_type="inhibition",
                     ec50_um=5.0, emax=0.9, hill_coefficient=1.0),
            PDEffect(target_reaction="COX2", effect_type="inhibition",
                     ec50_um=10.0, emax=0.8, hill_coefficient=1.0),
        ],
    ),
    "metformin_complex1": Pharmacodynamics(
        drug_name="metformin",
        effects=[
            PDEffect(target_reaction="NADH_DQ", effect_type="inhibition",
                     ec50_um=20.0, emax=0.3, hill_coefficient=1.0),
        ],
        target_biomarkers={"glucose_plasma": 5.0},
    ),
    "cisplatin_dna": Pharmacodynamics(
        drug_name="cisplatin",
        effects=[
            PDEffect(target_reaction="DNA_REPLICATION", effect_type="inhibition",
                     ec50_um=1.0, emax=0.95, hill_coefficient=1.5),
        ],
    ),
    "tamoxifen_esr1": Pharmacodynamics(
        drug_name="tamoxifen",
        effects=[
            PDEffect(target_reaction="ESR1", effect_type="inhibition",
                     ec50_um=0.05, emax=0.9, hill_coefficient=1.0),
        ],
    ),
    "imatinib_bcr_abl": Pharmacodynamics(
        drug_name="imatinib",
        effects=[
            PDEffect(target_reaction="BCR_ABL", effect_type="inhibition",
                     ec50_um=0.1, emax=0.95, hill_coefficient=1.0),
        ],
    ),
}


def get_predefined_pd(drug_name: str) -> Pharmacodynamics | None:
    """Look up a pre-defined PD profile by drug name (case-insensitive).

    Tries exact match first, then falls back to prefix matching
    (e.g. ``"metformin"`` matches ``"metformin_complex1"``).
    """
    key = drug_name.lower().replace(" ", "_").replace("-", "_")
    pd = PREDEFINED_PD.get(key)
    if pd is not None:
        return pd
    # Prefix fallback: "metformin" -> "metformin_complex1"
    for pd_key, pd_val in PREDEFINED_PD.items():
        if pd_key.startswith(key):
            return pd_val
    return None


# ---------------------------------------------------------------------------
# PD inference for novel drugs (doc/33 gap-2 closure)
# ---------------------------------------------------------------------------

# Target-protein → default PD mapping for common drug targets
_TARGET_PD_MAP: dict[str, tuple[str, str, float, float]] = {
    # (target_reaction, effect_type, ec50_um, emax)
    "HMGCR":      ("cholesterol_synthesis", "inhibition", 0.01, 0.85),
    "ACE":        ("angiotensin_II_production", "inhibition", 0.1, 0.80),
    "CACNA1C":    ("calcium_influx", "inhibition", 0.05, 0.75),
    "PTGS2":      ("prostaglandin_synthesis", "inhibition", 0.5, 0.90),
    "SLC6A4":     ("serotonin_reuptake", "inhibition", 0.1, 0.80),
    "ABAT":       ("gaba_metabolism", "inhibition", 1.0, 0.70),
    "CYP2C19":    ("clopidogrel_activation", "inhibition", 0.5, 0.85),
    "CYP2C9":     ("warfarin_clearance", "inhibition", 0.2, 0.80),
    "CYP3A4":     ("simvastatin_clearance", "inhibition", 0.5, 0.75),
    "CYP2D6":     ("tramadol_activation", "inhibition", 0.3, 0.80),
    "EGFR":       ("tumor_growth", "inhibition", 0.05, 0.70),
    "BRAF":       ("tumor_growth", "inhibition", 0.01, 0.80),
    "BCL2":       ("apoptosis_resistance", "inhibition", 0.1, 0.75),
    "CD20":       ("b_cell_survival", "inhibition", 0.05, 0.85),
    "TNF":        ("inflammation", "inhibition", 0.1, 0.90),
    "IL6R":       ("inflammation", "inhibition", 0.05, 0.85),
    "JAK":        ("inflammation", "inhibition", 0.1, 0.80),
    "DPP4":       ("incretin_degradation", "inhibition", 0.1, 0.75),
    "SGLT2":      ("glucose_reabsorption", "inhibition", 0.5, 0.80),
    "PPAR":       ("lipid_metabolism", "activation", 0.1, 0.70),
    "COX1":       ("platelet_aggregation", "inhibition", 0.5, 0.85),
    "THROMBIN":   ("coagulation", "inhibition", 0.05, 0.90),
    "FA10":       ("coagulation", "inhibition", 0.05, 0.85),
    "VKORC1":     ("vitamin_k_cycle", "inhibition", 0.01, 0.90),
    "DHFR":       ("folate_synthesis", "inhibition", 0.1, 0.85),
    "TUBB":       ("microtubule_assembly", "inhibition", 0.05, 0.80),
    "TOP1":       ("dna_relication", "inhibition", 0.1, 0.75),
    "TOP2":       ("dna_relication", "inhibition", 0.1, 0.80),
    "KDR":        ("tumor_growth", "inhibition", 0.05, 0.70),
    "PDGFRA":     ("tumor_growth", "inhibition", 0.05, 0.65),
}


def infer_pd_from_drug(drug_name: str, target_protein: str = "",
                       binding_kd_um: float = 1.0,
                       mw_da: float = 300.0) -> Pharmacodynamics:
    """Infer a basic PD profile for a novel drug from its molecular properties.

    This provides a reasonable default when no hand-curated PD exists.
    Uses target-protein matching when available, falls back to MW-based
    heuristics (larger molecules tend to have more specific targets).

    Parameters
    ----------
    drug_name : str
    target_protein : str
        Gene/protein symbol (e.g. ``"HMGCR"``, ``"EGFR"``).
    binding_kd_um : float
        Binding affinity in µM.
    mw_da : float
        Molecular weight in Daltons.

    Returns
    -------
    Pharmacodynamics
        A PD model with at least one inferred PDEffect.
    """
    effects: list[PDEffect] = []

    # Try target-based lookup
    if target_protein:
        key = target_protein.strip().upper()
        if key in _TARGET_PD_MAP:
            rxn, etype, ec50, emax = _TARGET_PD_MAP[key]
            # Scale EC50 by binding affinity (tighter binding → lower EC50)
            ec50_adj = max(0.001, min(100.0, binding_kd_um * 2.0))
            effects.append(PDEffect(
                target_reaction=rxn,
                target_gene=key,
                effect_type=etype,
                ec50_um=ec50_adj,
                emax=min(0.95, emax),
                hill_coefficient=1.2,
            ))
        else:
            # Unknown target: generic inhibition with moderate efficacy
            effects.append(PDEffect(
                target_reaction=f"{key.lower()}_activity",
                target_gene=key,
                effect_type="inhibition",
                ec50_um=max(0.1, binding_kd_um * 3.0),
                emax=0.5,
                hill_coefficient=1.0,
            ))
    elif mw_da > 5000:
        # Biologics: assume target-specific inhibition
        effects.append(PDEffect(
            target_reaction="target_activity",
            effect_type="inhibition",
            ec50_um=0.1,
            emax=0.8,
            hill_coefficient=1.5,
        ))
    else:
        # Small molecule without known target: weak off-target inhibition
        effects.append(PDEffect(
            target_reaction="generic_metabolism",
            effect_type="inhibition",
            ec50_um=50.0,
            emax=0.3,
            hill_coefficient=1.0,
        ))

    return Pharmacodynamics(
        drug_name=drug_name,
        effects=effects,
        toxicity_concentration_um=max(50.0, binding_kd_um * 100.0),
        therapeutic_window=(max(0.1, binding_kd_um), max(10.0, binding_kd_um * 50.0)),
    )


__all__ = [
    "PDEffect",
    "Pharmacodynamics",
    "hill_equation",
    "inhibition_fraction",
    "activation_fraction",
    "compute_pd_effects",
    "compute_pd_effects_over_time",
    "apply_pd_to_flux_bounds",
    "PREDEFINED_PD",
    "get_predefined_pd",
    "infer_pd_from_drug",
]
