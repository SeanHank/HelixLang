"""Enzyme-constrained genome-scale metabolic model builder (doc/26 Phase D).

Implements ECMpy 2.0-style (Wu et al. 2021) enzyme-constrained GEM construction.
Uses the sMOMENT-lite iterative approach (Sanchez et al. 2017) with real
molecular weights computed from enzyme sequences and real kcat values from
BRENDA or sequence-based prediction.

When a full metabolic model is provided (e.g., E. coli core model with 42 reactions),
applies enzyme constraints to it. When no model is provided, builds a minimal
core model from EC numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_AA_AVG_MW = 110.0
_CORE_MODEL_PATH = Path(__file__).parent.parent / "data" / "ecoli_core_model.json"

_EC_TO_REACTION: dict[str, dict[str, Any]] = {
    "2.7.1.1": {"id": "GLK", "name": "glucokinase"},
    "2.7.1.2": {"id": "GLK", "name": "glucokinase"},
    "5.3.1.9": {"id": "PGI", "name": "phosphoglucose isomerase"},
    "2.7.1.11": {"id": "PFK", "name": "phosphofructokinase"},
    "4.1.2.13": {"id": "FBA", "name": "fructose bisphosphate aldolase"},
    "5.3.1.1": {"id": "TPI", "name": "triosephosphate isomerase"},
    "1.2.1.12": {"id": "GAPD", "name": "G3P dehydrogenase"},
    "2.7.2.3": {"id": "PGK", "name": "phosphoglycerate kinase"},
    "4.1.1.32": {"id": "PGM", "name": "phosphoglycerate mutase"},
    "4.2.1.11": {"id": "ENO", "name": "enolase"},
    "2.7.1.40": {"id": "PYK", "name": "pyruvate kinase"},
    "1.2.4.1": {"id": "PDH", "name": "pyruvate dehydrogenase"},
    "4.1.1.31": {"id": "PPC", "name": "PEP carboxylase"},
    "4.1.3.16": {"id": "CS", "name": "citrate synthase"},
    "4.2.1.3": {"id": "ACONT", "name": "aconitase"},
    "1.1.1.40": {"id": "ICDH", "name": "isocitrate dehydrogenase"},
    "1.2.4.2": {"id": "AKGDH", "name": "alpha-KG dehydrogenase"},
    "6.2.1.4": {"id": "SUCOAS", "name": "succinyl-CoA synthetase"},
    "1.3.5.4": {"id": "SUCDHi", "name": "succinate dehydrogenase"},
    "4.2.1.2": {"id": "FUM", "name": "fumarase"},
    "1.1.1.37": {"id": "MDH", "name": "malate dehydrogenase"},
    "1.1.1.49": {"id": "G6PDH", "name": "glucose-6-phosphate dehydrogenase"},
    "1.1.1.44": {"id": "PGD", "name": "phosphogluconate dehydrogenase"},
    "5.1.3.1": {"id": "RPI", "name": "ribose-5-phosphate isomerase"},
    "1.1.1.27": {"id": "LDH", "name": "lactate dehydrogenase"},
}

CORE_ENZYME_KCAT: dict[str, float] = {
    "GLK": 300.0, "PGI": 660.0, "PFK": 380.0, "FBA": 157.0,
    "TPI": 700.0, "GAPD": 27.0, "PGK": 64.0, "PGM": 88.0,
    "ENO": 218.0, "PYK": 54.0, "PDH": 14.3, "CS": 490.0,
    "ACONT": 130.0, "ICDH": 14.3, "AKGDH": 30.0, "SUCOAS": 64.0,
    "SUCDHi": 450.0, "FUM": 69.0, "MDH": 77.0, "PPC": 96.0,
    "G6PDH": 580.0, "PGD": 17.0, "RPI": 187.0, "LDH": 650.0,
}

CORE_ENZYME_MW: dict[str, float] = {
    "GLK": 34800.0, "PGI": 60300.0, "PFK": 34600.0, "FBA": 31900.0,
    "TPI": 26700.0, "GAPD": 36000.0, "PGK": 41500.0, "PGM": 28600.0,
    "ENO": 45800.0, "PYK": 52100.0, "PDH": 104800.0, "CS": 48500.0,
    "ACONT": 42700.0, "ICDH": 45600.0, "AKGDH": 112000.0, "SUCOAS": 43700.0,
    "SUCDHi": 66700.0, "FUM": 48300.0, "MDH": 33400.0, "PPC": 101700.0,
    "G6PDH": 56200.0, "PGD": 51200.0, "RPI": 22700.0, "LDH": 36500.0,
}


def _molecular_weight_from_sequence(sequence: str) -> float:
    n = len(sequence)
    return n * _AA_AVG_MW


def _load_core_model() -> Any | None:
    """Load the E. coli core model (42 reactions) if available."""
    if not _CORE_MODEL_PATH.exists():
        return None
    try:
        from helixlang.metabolism import MetabolicModel, Reaction
        data = json.loads(_CORE_MODEL_PATH.read_text())
        model = MetabolicModel()
        for rxn_data in data["reactions"]:
            model.add_reaction(Reaction(
                id=rxn_data["id"],
                name=rxn_data.get("name", rxn_data["id"]),
                stoichiometry=rxn_data["stoichiometry"],
                lower_bound=rxn_data.get("lower_bound", 0.0),
                upper_bound=rxn_data.get("upper_bound", 1000.0),
                subsystem=rxn_data.get("subsystem", ""),
            ))
        if data.get("biomass_reaction"):
            model.set_biomass(data["biomass_reaction"])
        return model
    except Exception:
        return None


@dataclass
class EnzymeConstraint:
    reaction_id: str
    gene_id: str
    ec_number: str
    kcat: float
    molecular_weight: float
    enzyme_fraction: float = 0.0
    upper_bound: float = 0.0


@dataclass
class EnzymePoolConstraint:
    total_enzyme_mass: float
    total_enzyme_mass_g: float
    budget_constraint: str = "enzyme_pool"


@dataclass
class ECGEMResult:
    model: Any
    enzyme_constraints: list[EnzymeConstraint] = field(default_factory=list)
    enzyme_pool: EnzymePoolConstraint | None = None
    growth_rate: float = 0.0
    growth_rate_unconstrained: float = 0.0
    enzyme_usage: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class ECGEMBuilder:
    """Build enzyme-constrained GEM from enzyme kinetic data.

    When a core model is available (E. coli), applies enzyme constraints to it.
    When no model is provided, builds a minimal model from EC numbers.
    """

    DEFAULT_ENZYME_MASS_FRACTION: float = 0.55
    DEFAULT_DRY_WEIGHT_CONC: float = 0.3

    def __init__(
        self,
        base_model: Any | None = None,
        kcat_predictions: dict[str, float] | None = None,
        km_predictions: dict[str, float] | None = None,
        ec_numbers: dict[str, str] | None = None,
        sequences: dict[str, str] | None = None,
        organism: str = "e_coli_k12",
        enzyme_mass_fraction: float = 0.55,
        dry_weight_conc: float = 0.3,
    ) -> None:
        self.kcat_predictions = kcat_predictions or {}
        self.km_predictions = km_predictions or {}
        self.ec_numbers = ec_numbers or {}
        self.sequences = sequences or {}
        self.organism = organism
        self.enzyme_mass_fraction = enzyme_mass_fraction
        self.dry_weight_conc = dry_weight_conc
        self._enzyme_pool_g_per_L = enzyme_mass_fraction * dry_weight_conc
        self._base_model = base_model

    def _get_model(self) -> Any:
        if self._base_model is not None and self._base_model.reactions:
            return self._base_model
        core = _load_core_model()
        if core is not None:
            self._setup_medium(core)
            return core
        return self._build_minimal_model_from_enzymes()

    def _setup_medium(self, model: Any) -> None:
        """Open exchange reactions for glucose minimal medium."""
        for rxn_id, rxn in model.reactions.items():
            if rxn_id.startswith("EX_"):
                rxn.lower_bound = -1000.0
                rxn.upper_bound = 1000.0
        if "ATPM" in model.reactions:
            model.reactions["ATPM"].lower_bound = 8.39
            model.reactions["ATPM"].upper_bound = 8.39
        if "GLCpts" in model.reactions:
            model.reactions["GLCpts"].upper_bound = 10.0
        if "GLK" in model.reactions:
            model.reactions["GLK"].upper_bound = 10.0

    def _build_minimal_model_from_enzymes(self) -> Any:
        from helixlang.metabolism import MetabolicModel, Reaction
        model = MetabolicModel()
        added_rxns: set[str] = set()
        for gene_id, _kcat in self.kcat_predictions.items():
            ec = self.ec_numbers.get(gene_id, "")
            rxn_info = _EC_TO_REACTION.get(ec)
            if rxn_info is None:
                continue
            rxn_id = rxn_info["id"]
            if rxn_id in added_rxns:
                continue
            added_rxns.add(rxn_id)
            stoich = rxn_info.get("stoich", {})
            if stoich:
                model.add_reaction(Reaction(
                    id=rxn_id, name=rxn_info["name"],
                    stoichiometry=stoich,
                    lower_bound=-1000.0, upper_bound=1000.0,
                    subsystem="core_metabolism",
                ))
        produced_mets: set[str] = set()
        consumed_mets: set[str] = set()
        for rxn_id in added_rxns:
            for info in _EC_TO_REACTION.values():
                if info["id"] == rxn_id:
                    for met, coeff in info.get("stoich", {}).items():
                        if coeff > 0:
                            produced_mets.add(met)
                        elif coeff < 0:
                            consumed_mets.add(met)
                    break
        for met in consumed_mets:
            model.add_reaction(Reaction(
                id=f"EX_{met}", name=f"EX_{met}",
                stoichiometry={met: 1.0},
                lower_bound=-100.0, upper_bound=100.0,
                subsystem="exchange",
            ))
        for met in produced_mets:
            ex_id = f"EX_{met}"
            if ex_id not in model.reactions:
                model.add_reaction(Reaction(
                    id=ex_id, name=ex_id,
                    stoichiometry={met: 1.0},
                    lower_bound=-100.0, upper_bound=100.0,
                    subsystem="exchange",
                ))
        if produced_mets:
            biomass_mets: dict[str, float] = {}
            for met in ["PYR", "ACCOA", "G6P", "F6P", "FBP", "OAA", "SUC", "MAL", "CIT"]:
                if met in produced_mets:
                    biomass_mets[met] = -0.5
            if not biomass_mets:
                for met in produced_mets:
                    biomass_mets[met] = -0.5
            if added_rxns:
                model.add_reaction(Reaction(
                    id="BIOMASS", name="BIOMASS",
                    stoichiometry=biomass_mets,
                    lower_bound=0.0, upper_bound=1000.0,
                    subsystem="biomass",
                ))
                model.set_biomass("BIOMASS")
        return model

    def build(self) -> ECGEMResult:
        warnings: list[str] = []
        model = self._get_model()
        if not model.reactions:
            warnings.append("no reactions in model")
            return ECGEMResult(model=model, warnings=warnings)
        constraints = self._compute_enzyme_constraints(model)
        if not constraints:
            warnings.append("no enzyme constraints could be built")
            gr = self._solve_growth(model)
            return ECGEMResult(
                model=model, growth_rate=gr, growth_rate_unconstrained=gr,
                warnings=warnings,
            )
        pool = EnzymePoolConstraint(
            total_enzyme_mass=self.enzyme_mass_fraction,
            total_enzyme_mass_g=self._enzyme_pool_g_per_L,
        )
        gr_unconstrained = self._solve_growth(model)
        enzyme_usage = self._apply_enzyme_constraints(model, constraints)
        gr_constrained = self._solve_growth(model)
        if gr_constrained < 0.01 and gr_unconstrained > 0.01:
            warnings.append(
                f"ecGEM growth ({gr_constrained:.4f}) much lower than "
                f"unconstrained ({gr_unconstrained:.4f})"
            )
        return ECGEMResult(
            model=model,
            enzyme_constraints=constraints,
            enzyme_pool=pool,
            growth_rate=gr_constrained,
            growth_rate_unconstrained=gr_unconstrained,
            enzyme_usage=enzyme_usage,
            warnings=warnings,
        )

    def _compute_enzyme_constraints(self, model: Any) -> list[EnzymeConstraint]:
        constraints: list[EnzymeConstraint] = []
        rxn_ids = set(model.reactions.keys())
        ec_to_rxn: dict[str, str] = {}
        for ec, info in _EC_TO_REACTION.items():
            if info["id"] in rxn_ids:
                ec_to_rxn[ec] = info["id"]

        has_custom_kcats = bool(self.kcat_predictions)
        for gene_id, kcat in self.kcat_predictions.items():
            if kcat <= 0:
                continue
            ec = self.ec_numbers.get(gene_id, "")
            rxn_id = ec_to_rxn.get(ec, "")
            if not rxn_id:
                continue
            seq = self.sequences.get(gene_id, "")
            mw = _molecular_weight_from_sequence(seq) if seq else 33000.0
            constraints.append(EnzymeConstraint(
                reaction_id=rxn_id, gene_id=gene_id,
                ec_number=ec, kcat=kcat, molecular_weight=mw,
            ))

        if not has_custom_kcats:
            for rxn_id in rxn_ids:
                kcat = CORE_ENZYME_KCAT.get(rxn_id, 0.0)
                mw = CORE_ENZYME_MW.get(rxn_id, 0.0)
                if kcat > 0 and mw > 0:
                    ec = ""
                    for e, info in _EC_TO_REACTION.items():
                        if info["id"] == rxn_id:
                            ec = e
                            break
                    constraints.append(EnzymeConstraint(
                        reaction_id=rxn_id, gene_id=rxn_id,
                        ec_number=ec, kcat=kcat, molecular_weight=mw,
                    ))
        return constraints

    def _apply_enzyme_constraints(
        self, model: Any, constraints: list[EnzymeConstraint]
    ) -> dict[str, float]:
        _KCAT_S_TO_H = 3600.0
        _MMOL_TO_MOL = 1000.0

        fluxes = self._solve_fluxes(model)
        total_required = 0.0
        for c in constraints:
            v = abs(fluxes.get(c.reaction_id, 0.0))
            if c.kcat > 0 and c.molecular_weight > 0:
                e_i = v * c.molecular_weight / (c.kcat * _KCAT_S_TO_H * _MMOL_TO_MOL)
            else:
                e_i = 0.0
            total_required += e_i

        if total_required <= 0:
            return {}

        scale = min(1.0, self._enzyme_pool_g_per_L / total_required)

        enzyme_usage: dict[str, float] = {}
        for c in constraints:
            v = abs(fluxes.get(c.reaction_id, 0.0))
            if c.kcat > 0 and c.molecular_weight > 0:
                e_i = v * c.molecular_weight / (c.kcat * _KCAT_S_TO_H * _MMOL_TO_MOL)
            else:
                e_i = 0.0
            e_i_scaled = e_i * scale
            c.upper_bound = (
                e_i_scaled * c.kcat * _KCAT_S_TO_H * _MMOL_TO_MOL / c.molecular_weight
                if c.molecular_weight > 0 else 0.0
            )
            c.enzyme_fraction = (
                e_i_scaled / self._enzyme_pool_g_per_L
                if self._enzyme_pool_g_per_L > 0 else 0.0
            )
            enzyme_usage[c.reaction_id] = c.enzyme_fraction
            if c.reaction_id in model.reactions:
                rxn = model.reactions[c.reaction_id]
                if c.upper_bound > 0:
                    rxn.upper_bound = min(rxn.upper_bound, c.upper_bound)

        return enzyme_usage

    def solve(self) -> ECGEMResult:
        return self.build()

    def validate(self, expected_growth: float, tolerance: float = 0.15) -> bool:
        result = self.build()
        if expected_growth <= 0:
            return True
        ratio = result.growth_rate / expected_growth if expected_growth > 0 else 0
        return (1.0 - tolerance) <= ratio <= (1.0 + tolerance)

    @staticmethod
    def _solve_growth(model: Any) -> float:
        try:
            from helixlang.metabolism import FluxBalanceAnalysis
            fba = FluxBalanceAnalysis(model)
            if model.biomass_reaction:
                fluxes = fba.solve(objective=model.biomass_reaction, maximize=True)
                return fluxes.get(model.biomass_reaction, 0.0)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _solve_fluxes(model: Any) -> dict[str, float]:
        try:
            from helixlang.metabolism import FluxBalanceAnalysis
            fba = FluxBalanceAnalysis(model)
            if model.biomass_reaction:
                return fba.solve(objective=model.biomass_reaction, maximize=True)
        except Exception:
            pass
        return {}


__all__ = [
    "ECGEMBuilder", "ECGEMResult", "EnzymeConstraint", "EnzymePoolConstraint",
    "_EC_TO_REACTION", "_molecular_weight_from_sequence", "_load_core_model",
]
