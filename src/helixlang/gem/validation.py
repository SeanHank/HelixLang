"""Model validation: FBA consistency, mass balance, gene essentiality (doc/20 §6.1)."""
from __future__ import annotations

from dataclasses import dataclass, field

from helixlang.gem.consensus import ConsensusResult
from helixlang.metabolism import FluxBalanceAnalysis, MetabolicModel, Reaction


@dataclass
class GemValidationResult:
    """Result of genome-scale model validation."""

    predicted_growth_rate: float = 0.0
    experimental_growth_rate: float | None = None
    growth_rate_error: float | None = None
    essential_genes_correct: float | None = None
    auxotrophy_correct: float | None = None
    blocked_reactions: list[str] = field(default_factory=list)
    mass_balance_violations: list[str] = field(default_factory=list)
    validation_passed: bool = False


def _parse_equation_to_stoich(equation: str) -> dict[str, float]:
    """Parse a simplified equation string into a stoichiometry dict."""
    stoich: dict[str, float] = {}
    if "<=>" in equation:
        sides = equation.split("<=>")
    elif "->" in equation:
        sides = equation.split("->")
    else:
        return stoich
    for side_sign, side in enumerate(sides):
        sign = -1.0 if side_sign == 0 else 1.0
        for term in side.split("+"):
            term = term.strip()
            if not term:
                continue
            parts = term.split()
            if len(parts) == 1:
                met, coeff = parts[0], 1.0
            elif len(parts) == 2 and _is_numeric(parts[0]):
                coeff, met = float(parts[0]), parts[1]
            else:
                met = parts[-1]
                coeff = 1.0
            stoich[met] = stoich.get(met, 0.0) + sign * coeff
    return stoich


def _is_numeric(s: str) -> bool:
    """Return True if *s* can be parsed as a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _build_model_from_consensus(
    consensus: ConsensusResult,
) -> MetabolicModel:
    """Build a MetabolicModel from a ConsensusResult."""
    model = MetabolicModel()
    for rxn in consensus.reactions:
        stoich = _parse_equation_to_stoich(rxn.equation)
        if not stoich:
            continue
        model.add_reaction(Reaction(
            id=rxn.reaction_id,
            name=rxn.reaction_id,
            stoichiometry=stoich,
            lower_bound=-1000.0 if rxn.confidence >= 0.6 else 0.0,
            upper_bound=1000.0,
            subsystem="gem_reconstructed",
        ))
    if consensus.reactions:
        model.set_biomass(consensus.reactions[0].reaction_id)
    return model


def check_mass_balance(model: MetabolicModel, fluxes: dict[str, float]) -> list[str]:
    """Check steady-state mass balance: S·v = 0 for each metabolite.

    Parameters
    ----------
    model : metabolic model with stoichiometry
    fluxes : {reaction_id: flux} from an FBA solution

    Returns
    -------
    List of metabolite IDs where |S·v| > 1e-6 (mass balance violations).
    """
    violations: list[str] = []
    for met in sorted(model.metabolites):
        net = 0.0
        for rxn_id, rxn in model.reactions.items():
            coef = rxn.stoichiometry.get(met, 0.0)
            if coef != 0.0:
                net += coef * fluxes.get(rxn_id, 0.0)
        if abs(net) > 1e-6:
            violations.append(met)
    return violations


def gene_essentiality_test(
    model: MetabolicModel,
    consensus: ConsensusResult,
    biomass_rxn: str | None = None,
) -> dict[str, bool]:
    """Perform in silico gene knockout essentiality analysis.

    For each gene in the consensus GPR rules, knock it out by setting
    the upper bound of all reactions it gates to 0, re-run FBA, and
    compare the growth rate to the wild-type.

    Parameters
    ----------
    model : metabolic model (used as the base for knockouts)
    consensus : consensus result with GPR rules
    biomass_rxn : biomass reaction id (None = use model.biomass_reaction)

    Returns
    -------
    dict mapping gene_id -> is_essential (True if knockout reduces
    growth rate to < 1% of wild-type).
    """
    bm = biomass_rxn or model.biomass_reaction
    if bm is None:
        return {}

    # wild-type growth rate
    fba_wt = FluxBalanceAnalysis(model)
    try:
        wt_fluxes = fba_wt.solve(objective=bm)
        wt_growth = wt_fluxes.get(bm, 0.0)
    except Exception:
        return {}

    if wt_growth < 1e-9:
        return {}

    # collect gene -> reaction mappings from GPR rules
    gene_rxns: dict[str, list[str]] = {}
    for rxn in consensus.reactions:
        if rxn.gpr and rxn.gpr.gene_ids:
            for gene in rxn.gpr.gene_ids:
                gene_rxns.setdefault(gene, []).append(rxn.reaction_id)

    results: dict[str, bool] = {}
    for gene, rids in gene_rxns.items():
        # save original bounds
        saved: dict[str, tuple[float, float]] = {}
        for rid in rids:
            if rid in model.reactions:
                rxn_obj = model.reactions[rid]
                saved[rid] = (rxn_obj.lower_bound, rxn_obj.upper_bound)
                rxn_obj.upper_bound = 0.0

        try:
            fba_ko = FluxBalanceAnalysis(model)
            ko_fluxes = fba_ko.solve(objective=bm)
            ko_growth = ko_fluxes.get(bm, 0.0)
        except Exception:
            ko_growth = 0.0

        results[gene] = (ko_growth / wt_growth) < 0.01

        # restore bounds
        for rid, (lb, ub) in saved.items():
            if rid in model.reactions:
                model.reactions[rid].lower_bound = lb
                model.reactions[rid].upper_bound = ub

    return results


def validate_model(
    model: MetabolicModel | None = None,
    consensus: ConsensusResult | None = None,
    medium: dict[str, float] | None = None,
    expected_growth_rate: float | None = None,
) -> GemValidationResult:
    """Validate a genome-scale metabolic model.

    Steps:
      1. Build MetabolicModel from consensus if not provided.
      2. Apply medium uptake constraints.
      3. Run FBA to get predicted growth rate.
      4. If expected_growth_rate given, compute absolute error.
      5. Check mass balance S·v = 0 for all metabolites.
      6. Find blocked reactions (zero flux in optimal solution).
      7. Return GemValidationResult.

    Parameters
    ----------
    model : pre-built MetabolicModel (optional; built from consensus)
    consensus : consensus reconstruction result (used if model is None)
    medium : {metabolite_id: max_uptake_rate} for exchange constraints
    expected_growth_rate : experimental growth rate for comparison

    Returns
    -------
    GemValidationResult with all validation metrics
    """
    if model is None and consensus is not None:
        model = _build_model_from_consensus(consensus)
    if model is None:
        return GemValidationResult()

    result = GemValidationResult()

    # apply medium constraints
    if medium:
        for met, uptake in medium.items():
            ex_id = f"EX_{met}"
            if ex_id in model.reactions:
                model.reactions[ex_id].upper_bound = uptake

    # run FBA
    bm = model.biomass_reaction
    if bm is None:
        return result

    fba = FluxBalanceAnalysis(model)
    try:
        fluxes = fba.solve(objective=bm)
        result.predicted_growth_rate = fluxes.get(bm, 0.0)
    except Exception:
        result.predicted_growth_rate = 0.0
        fluxes = {}

    # growth rate error
    if expected_growth_rate is not None:
        result.experimental_growth_rate = expected_growth_rate
        result.growth_rate_error = abs(
            result.predicted_growth_rate - expected_growth_rate
        )

    # mass balance check
    result.mass_balance_violations = check_mass_balance(model, fluxes)

    # blocked reactions
    result.blocked_reactions = [
        rxn_id for rxn_id, v in fluxes.items() if abs(v) < 1e-12
    ]

    # overall pass: biomass > 0 and no mass balance violations
    result.validation_passed = (
        result.predicted_growth_rate > 1e-9
        and len(result.mass_balance_violations) == 0
    )

    return result
