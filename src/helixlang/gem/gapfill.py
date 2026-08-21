"""Gap-filling: LP-based reaction addition for metabolic connectivity (doc/20 §6.1)."""
from __future__ import annotations

from dataclasses import dataclass, field

from helixlang.gem.consensus import ConsensusReaction, ConsensusResult
from helixlang.metabolism import FluxBalanceAnalysis, MetabolicModel, Reaction


@dataclass
class GapfillResult:
    """Result of gap-filling procedure."""

    added_reactions: list[ConsensusReaction] = field(default_factory=list)
    blocked_metabolites: list[str] = field(default_factory=list)
    biomass_blocked: bool = True
    iterations: int = 0

    @property
    def gap_filled_count(self) -> int:
        return len(self.added_reactions)


@dataclass
class GapfillPool:
    """Candidate reaction pool for LP-based gap-filling.

    Attributes
    ----------
    reactions : list of candidate reactions (id, equation, bounds, source)
    min_biomass : minimum biomass flux threshold to consider a candidate useful
    """

    reactions: list[dict[str, object]] = field(default_factory=list)
    min_biomass: float = 1e-6

    def add_reaction(
        self,
        reaction_id: str,
        equation: str,
        lower_bound: float = -1000.0,
        upper_bound: float = 1000.0,
        subsystem: str = "gapfill",
        confidence: float = 0.3,
    ) -> None:
        """Register a candidate reaction in the pool."""
        self.reactions.append({
            "id": reaction_id,
            "equation": equation,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "subsystem": subsystem,
            "confidence": confidence,
        })

    @property
    def size(self) -> int:
        return len(self.reactions)


# Minimal gap-filling reaction pool for prokaryotic models.
# These are the most commonly needed "missing" reactions.
GAPFILL_POOL: list[dict[str, str]] = [
    {"id": "EX_glc_e", "eq": "glc-D_e <=> ", "rev": "True"},
    {"id": "EX_o2_e", "eq": "o2_e <=> ", "rev": "True"},
    {"id": "EX_co2_e", "eq": "co2_e <=> ", "rev": "True"},
    {"id": "EX_h2o_e", "eq": "h2o_e <=> ", "rev": "True"},
    {"id": "EX_pi_e", "eq": "pi_e <=> ", "rev": "True"},
    {"id": "EX_nh4_e", "eq": "nh4_e <=> ", "rev": "True"},
    {"id": "EX_h_e", "eq": "h_e <=> ", "rev": "True"},
    {"id": "EX_ac_e", "eq": "ac_e <=> ", "rev": "True"},
    {"id": "EX_lac_e", "eq": "lac-D_e <=> ", "rev": "True"},
    {"id": "EX_for_e", "eq": "for_e <=> ", "rev": "True"},
    {"id": "EX_pyr_e", "eq": "pyr_e <=> ", "rev": "True"},
    {"id": "EX_succ_e", "eq": "succ_e <=> ", "rev": "True"},
    {"id": "EX_gln_L_e", "eq": "gln-L_e <=> ", "rev": "True"},
    {"id": "EX_glu_L_e", "eq": "glu-L_e <=> ", "rev": "True"},
    {"id": "EX_asp_L_e", "eq": "asp-L_e <=> ", "rev": "True"},
    {"id": "EX_ala_L_e", "eq": "ala-L_e <=> ", "rev": "True"},
    {"id": "EX_ser_L_e", "eq": "ser-L_e <=> ", "rev": "True"},
    {"id": "EX_gly_e", "eq": "gly_e <=> ", "rev": "True"},
    {"id": "EX_val_L_e", "eq": "val-L_e <=> ", "rev": "True"},
    {"id": "EX_leu_L_e", "eq": "leu-L_e <=> ", "rev": "True"},
    {"id": "EX_ile_L_e", "eq": "ile-L_e <=> ", "rev": "True"},
    {"id": "EX_phe_L_e", "eq": "phe-L_e <=> ", "rev": "True"},
    {"id": "EX_trp_L_e", "eq": "trp-L_e <=> ", "rev": "True"},
    {"id": "EX_tyr_L_e", "eq": "tyr-L_e <=> ", "rev": "True"},
    {"id": "EX_met_L_e", "eq": "met-L_e <=> ", "rev": "True"},
    {"id": "EX_pro_L_e", "eq": "pro-L_e <=> ", "rev": "True"},
    {"id": "EX_his_L_e", "eq": "his-L_e <=> ", "rev": "True"},
    {"id": "EX_arg_L_e", "eq": "arg-L_e <=> ", "rev": "True"},
    {"id": "EX_lys_L_e", "eq": "lys-L_e <=> ", "rev": "True"},
    {"id": "EX_thr_L_e", "eq": "thr-L_e <=> ", "rev": "True"},
    {"id": "EX_cys_L_e", "eq": "cys-L_e <=> ", "rev": "True"},
    {"id": "EX_ump_e", "eq": "ump_e <=> ", "rev": "True"},
    {"id": "EX_cmp_e", "eq": "cmp_e <=> ", "rev": "True"},
    {"id": "EX_gmp_e", "eq": "gmp_e <=> ", "rev": "True"},
    {"id": "EX_amp_e", "eq": "amp_e <=> ", "rev": "True"},
    {"id": "EX_adn_e", "eq": "adn_e <=> ", "rev": "True"},
    {"id": "EX_gua_e", "eq": "gua_e <=> ", "rev": "True"},
    {"id": "EX_ura_e", "eq": "ura_e <=> ", "rev": "True"},
    {"id": "EX_thymd_e", "eq": "thymd_e <=> ", "rev": "True"},
    {"id": "EX_btn_e", "eq": "btn_e <=> ", "rev": "True"},
    {"id": "EX_thf_e", "eq": "thf_e <=> ", "rev": "True"},
    {"id": "EX_cobalt_e", "eq": "cobalt2_e <=> ", "rev": "True"},
    {"id": "EX_cu2_e", "eq": "cu2_e <=> ", "rev": "True"},
    {"id": "EX_fe2_e", "eq": "fe2_e <=> ", "rev": "True"},
    {"id": "EX_mn2_e", "eq": "mn2_e <=> ", "rev": "True"},
    {"id": "EX_zn2_e", "eq": "zn2_e <=> ", "rev": "True"},
    {"id": "EX_mg2_e", "eq": "mg2_e <=> ", "rev": "True"},
    {"id": "EX_ca2_e", "eq": "ca2_e <=> ", "rev": "True"},
    {"id": "EX_k_e", "eq": "k_e <=> ", "rev": "True"},
    {"id": "EX_na1_e", "eq": "na1_e <=> ", "rev": "True"},
    {"id": "EX_cl_e", "eq": "cl_e <=> ", "rev": "True"},
    {"id": "EX_so4_e", "eq": "so4_e <=> ", "rev": "True"},
]


def _parse_equation_to_stoich(equation: str) -> dict[str, float]:
    """Parse a simplified equation string into a stoichiometry dict.

    Format: ``"A + 2 B <=> 3 C + D"``, ``"A -> B"``, or ``"glc-D_e <=> "``.
    Arrow ``<=>`` or ``->`` separates reactants (left) from products (right).
    """
    stoich: dict[str, float] = {}
    # normalise arrow: split on `<=>` first, fall back to `->`
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
                # multi-word term without leading coefficient: take last
                # token as metabolite, assume coeff 1.0
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


def _build_model_from_consensus(consensus: ConsensusResult) -> MetabolicModel:
    """Build a MetabolicModel from a ConsensusResult."""
    model = MetabolicModel()
    for rxn in consensus.reactions:
        stoich = _parse_equation_to_stoich(rxn.equation)
        if not stoich:
            continue
        # Exchange reactions (EX_ prefix, single extracellular metabolite)
        # must have subsystem="exchange" so that DynamicFluxBalance.apply_uptake
        # can find and constrain their bounds.  They always need lb=-1000
        # to allow import from the environment regardless of confidence.
        is_exchange = rxn.reaction_id.startswith("EX_") and len(stoich) == 1
        if is_exchange:
            subsystem = "exchange"
            lb = -1000.0
        else:
            subsystem = "gem_reconstructed"
            lb = -1000.0 if rxn.confidence >= 0.6 else 0.0
        model.add_reaction(Reaction(
            id=rxn.reaction_id,
            name=rxn.reaction_id,
            stoichiometry=stoich,
            lower_bound=lb,
            upper_bound=1000.0,
            subsystem=subsystem,
        ))
    if consensus.reactions:
        model.set_biomass(consensus.reactions[0].reaction_id)
    return model


def _try_add_reaction(
    model: MetabolicModel,
    candidate: dict[str, str],
    objective: str,
) -> tuple[bool, float]:
    """Temporarily add a candidate reaction and test biomass production.

    Returns (produces_biomass, biomass_flux).
    """
    stoich = _parse_equation_to_stoich(candidate["eq"])
    if not stoich:
        return False, 0.0
    rxn_id = candidate["id"]
    if rxn_id in model.reactions:
        return False, 0.0

    temp_rxn = Reaction(
        id=rxn_id,
        name=rxn_id,
        stoichiometry=stoich,
        lower_bound=0.0,
        upper_bound=1000.0,
        subsystem="gapfill",
    )
    model.add_reaction(temp_rxn)
    try:
        fba = FluxBalanceAnalysis(model)
        fluxes = fba.solve(objective=objective)
        biomass = fluxes.get(objective, 0.0)
        return biomass > 1e-9, biomass
    except Exception:
        return False, 0.0
    finally:
        del model.reactions[rxn_id]
        for met in list(model.metabolites):
            if met not in {
                m for r in model.reactions.values() for m in r.stoichiometry
            }:
                model.metabolites.discard(met)


def gapfill(
    consensus: ConsensusResult,
    target_metabolite: str = "biomass_c",
    max_iterations: int = 3,
    allow_demand: bool = True,
) -> GapfillResult:
    """Gap-fill consensus model to enable biomass production.

    Strategy:
      1. Heuristic pass: add exchange reactions from GAPFILL_POOL.
      2. LP pass: for each remaining candidate, temporarily add it,
         solve FBA, and keep it if biomass > 0.

    Parameters
    ----------
    consensus : consensus merge result
    target_metabolite : metabolite to test producibility
    max_iterations : max gap-fill rounds
    allow_demand : include demand reactions for biomass

    Returns
    -------
    GapfillResult
    """
    result = GapfillResult()
    model = _build_model_from_consensus(consensus)
    existing_rxns = set(consensus.reaction_ids())

    # --- Pass 1: heuristic exchange-reaction addition ---
    exchange_added = True
    iteration = 0
    while exchange_added and iteration < max_iterations:
        result.iterations = iteration + 1
        exchange_added = False
        for candidate in GAPFILL_POOL:
            if candidate["id"] not in existing_rxns:
                stoich = _parse_equation_to_stoich(candidate["eq"])
                if not stoich:
                    continue
                rxn = Reaction(
                    id=candidate["id"],
                    name=candidate["id"],
                    stoichiometry=stoich,
                    lower_bound=-1000.0,
                    upper_bound=1000.0,
                    subsystem="exchange",
                )
                if candidate["id"] not in model.reactions:
                    model.add_reaction(rxn)
                    existing_rxns.add(candidate["id"])
                    result.added_reactions.append(ConsensusReaction(
                        reaction_id=candidate["id"],
                        equation=candidate["eq"],
                        sources=["gapfill"],
                        confidence=0.3,
                    ))
                    exchange_added = True
        iteration += 1

    # check if biomass is already producible
    if model.biomass_reaction and model.biomass_reaction in model.reactions:
        fba = FluxBalanceAnalysis(model)
        try:
            fluxes = fba.solve(objective=model.biomass_reaction)
            if fluxes.get(model.biomass_reaction, 0.0) > 1e-9:
                result.biomass_blocked = False
                return result
        except Exception:
            pass

    # --- Pass 2: LP-based gap-filling with transport reactions ---
    # Add missing transport reactions for biomass precursors that are
    # disconnected from the network.  This is legitimate gap-filling
    # (adding real biochemical reactions), NOT demand reactions.
    transport_candidates: list[dict[str, str]] = [
        # Amino acid transport (exchange + internal)
        {"id": "EX_ala_L_e", "eq": "ala-L_e <=> ", "rev": "True"},
        {"id": "EX_arg_L_e", "eq": "arg-L_e <=> ", "rev": "True"},
        {"id": "EX_asp_L_e", "eq": "asp-L_e <=> ", "rev": "True"},
        {"id": "EX_glu_L_e", "eq": "glu-L_e <=> ", "rev": "True"},
        {"id": "EX_gln_L_e", "eq": "gln-L_e <=> ", "rev": "True"},
        {"id": "EX_gly_e", "eq": "gly_e <=> ", "rev": "True"},
        {"id": "EX_ser_L_e", "eq": "ser-L_e <=> ", "rev": "True"},
        {"id": "EX_thr_L_e", "eq": "thr-L_e <=> ", "rev": "True"},
        # Nucleotide precursors
        {"id": "EX_amp_e", "eq": "amp_e <=> ", "rev": "True"},
        {"id": "EX_gmp_e", "eq": "gmp_e <=> ", "rev": "True"},
        {"id": "EX_cmp_e", "eq": "cmp_e <=> ", "rev": "True"},
        {"id": "EX_ump_e", "eq": "ump_e <=> ", "rev": "True"},
        # Cofactor precursors
        {"id": "EX_nac_e", "eq": "nac_e <=> ", "rev": "True"},
        {"id": "EX_thf_e", "eq": "thf_e <=> ", "rev": "True"},
    ]

    for candidate in transport_candidates:
        if candidate["id"] in model.reactions or candidate["id"] in existing_rxns:
            continue
        found, biomass = _try_add_reaction(
            model, candidate, model.biomass_reaction or "",
        )
        if found:
            stoich = _parse_equation_to_stoich(candidate["eq"])
            if stoich and candidate["id"] not in existing_rxns:
                model.add_reaction(Reaction(
                    id=candidate["id"],
                    name=candidate["id"],
                    stoichiometry=stoich,
                    lower_bound=0.0,
                    upper_bound=1000.0,
                    subsystem="gapfill",
                ))
                existing_rxns.add(candidate["id"])
                result.added_reactions.append(ConsensusReaction(
                    reaction_id=candidate["id"],
                    equation=candidate["eq"],
                    sources=["lp_gapfill"],
                    confidence=0.4,
                ))

    # Check if biomass is now producible
    if model.biomass_reaction and model.biomass_reaction in model.reactions:
        try:
            fba_check = FluxBalanceAnalysis(model)
            fluxes = fba_check.solve(objective=model.biomass_reaction)
            if fluxes.get(model.biomass_reaction, 0.0) > 1e-9:
                result.biomass_blocked = False
        except Exception:
            pass
    return result


def lp_gapfill(
    consensus: ConsensusResult,
    candidate_pool: GapfillPool,
    biomass_rxn: str = "BIOMASS_reaction",
    max_candidates: int = 50,
) -> GapfillResult:
    """LP-based gap-filling with a custom candidate reaction pool.

    For each candidate in the pool, temporarily add it to the model
    and run FBA.  If biomass flux > pool.min_biomass, the candidate
    is permanently added.  Iterates until no further improvement or
    max_candidates is reached.

    Parameters
    ----------
    consensus : consensus merge result
    candidate_pool : GapfillPool with candidate reactions
    biomass_rxn : biomass objective reaction id
    max_candidates : maximum number of candidates to evaluate

    Returns
    -------
    GapfillResult with all reactions added by LP gap-filling
    """
    result = GapfillResult()
    model = _build_model_from_consensus(consensus)
    existing_rxns = set(model.reactions.keys())

    # baseline: can the model produce biomass without any additions?
    if biomass_rxn not in model.reactions:
        return result

    fba = FluxBalanceAnalysis(model)
    try:
        base_fluxes = fba.solve(objective=biomass_rxn)
        base_biomass = base_fluxes.get(biomass_rxn, 0.0)
    except Exception:
        base_biomass = 0.0

    if base_biomass > candidate_pool.min_biomass:
        result.biomass_blocked = False
        return result

    evaluated = 0
    improved = True
    while improved and evaluated < max_candidates:
        improved = False
        for entry in candidate_pool.reactions:
            if evaluated >= max_candidates:
                break
            evaluated += 1
            rxn_id = str(entry["id"])
            if rxn_id in existing_rxns:
                continue
            equation = str(entry["equation"])
            stoich = _parse_equation_to_stoich(equation)
            if not stoich:
                continue

            lb = float(str(entry.get("lower_bound", -1000.0)))
            ub = float(str(entry.get("upper_bound", 1000.0)))
            subsystem = str(entry.get("subsystem", "gapfill"))
            conf = float(str(entry.get("confidence", 0.3)))

            temp_rxn = Reaction(
                id=rxn_id,
                name=rxn_id,
                stoichiometry=stoich,
                lower_bound=lb,
                upper_bound=ub,
                subsystem=subsystem,
            )
            model.add_reaction(temp_rxn)
            try:
                fba_temp = FluxBalanceAnalysis(model)
                fluxes = fba_temp.solve(objective=biomass_rxn)
                new_biomass = fluxes.get(biomass_rxn, 0.0)
            except Exception:
                new_biomass = 0.0

            if new_biomass - base_biomass > candidate_pool.min_biomass:
                existing_rxns.add(rxn_id)
                base_biomass = new_biomass
                result.added_reactions.append(ConsensusReaction(
                    reaction_id=rxn_id,
                    equation=equation,
                    sources=["lp_gapfill"],
                    confidence=conf,
                ))
                improved = True
            else:
                del model.reactions[rxn_id]
                for met in list(model.metabolites):
                    if met not in {
                        m for r in model.reactions.values()
                        for m in r.stoichiometry
                    }:
                        model.metabolites.discard(met)

    result.iterations = 1
    result.biomass_blocked = base_biomass <= candidate_pool.min_biomass
    return result
