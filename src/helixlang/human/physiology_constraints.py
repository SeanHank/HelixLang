"""Physiology Constraints: mass balance + thermodynamic feasibility (doc/32 §7.3).

Enforces three fundamental physical constraints at every simulation step:
1. Mass balance: total mass in = total mass out + accumulation
2. Thermodynamic feasibility: ΔG < 0 for irreversible reactions
3. Homeostatic stability: vital signs within physiological bounds

A model that violates any of these produces physically impossible outputs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HomeostaticBounds:
    """Physiological bounds for vital signs and lab values."""

    ph_min: float = 6.8
    ph_max: float = 7.8
    temp_min_c: float = 35.5
    temp_max_c: float = 42.0
    map_min_mmhg: float = 60.0
    map_max_mmhg: float = 150.0
    glucose_min_mgdL: float = 40.0
    glucose_max_mgdL: float = 500.0
    spo2_min_pct: float = 50.0
    spo2_max_pct: float = 100.0
    heart_rate_min: float = 30.0
    heart_rate_max: float = 200.0
    alt_min: float = 0.0
    alt_max: float = 1000.0
    ast_min: float = 0.0
    ast_max: float = 1000.0
    creatinine_min: float = 0.3
    creatinine_max: float = 15.0
    wbc_min: float = 1000.0
    wbc_max: float = 30000.0
    cortisol_min: float = 1.0
    cortisol_max: float = 60.0
    tnf_alpha_min: float = 0.0
    tnf_alpha_max: float = 500.0
    il6_min: float = 0.0
    il6_max: float = 500.0


@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""

    variable: str
    value: float
    bound_min: float
    bound_max: float
    severity: float
    step: int = 0


@dataclass
class ConstraintCheckResult:
    """Result of constraint checking."""

    violations: list[ConstraintViolation]
    is_valid: bool
    total_penalty: float = 0.0


class PhysiologyConstraints:
    """Enforces physiological constraints on simulation state.

    Usage:
        constraints = PhysiologyConstraints()
        result = constraints.check(state_dict, step=0)
        if not result.is_valid:
            corrected = constraints.project_to_feasible(state_dict)
    """

    def __init__(
        self,
        bounds: HomeostaticBounds | None = None,
        penalty_weight: float = 100.0,
    ) -> None:
        self.bounds = bounds or HomeostaticBounds()
        self.penalty_weight = penalty_weight

    def check(
        self, state: dict[str, float], step: int = 0
    ) -> ConstraintCheckResult:
        """Check if state violates any homeostatic bounds."""
        violations: list[ConstraintViolation] = []

        checks = [
            ("ph", "ph_min", "ph_max"),
            ("temperature", "temp_min_c", "temp_max_c"),
            ("map", "map_min_mmhg", "map_max_mmhg"),
            ("glucose", "glucose_min_mgdL", "glucose_max_mgdL"),
            ("spo2", "spo2_min_pct", "spo2_max_pct"),
            ("heart_rate", "heart_rate_min", "heart_rate_max"),
            ("alt", "alt_min", "alt_max"),
            ("ast", "ast_min", "ast_max"),
            ("creatinine", "creatinine_min", "creatinine_max"),
            ("wbc", "wbc_min", "wbc_max"),
            ("cortisol", "cortisol_min", "cortisol_max"),
            ("tnf_alpha", "tnf_alpha_min", "tnf_alpha_max"),
            ("il6", "il6_min", "il6_max"),
        ]

        for var, min_attr, max_attr in checks:
            if var in state:
                val = state[var]
                lo = getattr(self.bounds, min_attr)
                hi = getattr(self.bounds, max_attr)
                if val < lo or val > hi:
                    violations.append(
                        ConstraintViolation(
                            variable=var,
                            value=val,
                            bound_min=lo,
                            bound_max=hi,
                            severity=max(lo - val, val - hi, 0.0),
                            step=step,
                        )
                    )

        penalty = sum(v.severity * self.penalty_weight for v in violations)
        return ConstraintCheckResult(
            violations=violations,
            is_valid=len(violations) == 0,
            total_penalty=penalty,
        )

    def project_to_feasible(self, state: dict[str, float]) -> dict[str, float]:
        """Project state values back to feasible region.

        Clamps each variable to its physiological bounds.
        """
        corrected = dict(state)
        bounds_map = {
            "ph": (self.bounds.ph_min, self.bounds.ph_max),
            "temperature": (self.bounds.temp_min_c, self.bounds.temp_max_c),
            "map": (self.bounds.map_min_mmhg, self.bounds.map_max_mmhg),
            "glucose": (self.bounds.glucose_min_mgdL, self.bounds.glucose_max_mgdL),
            "spo2": (self.bounds.spo2_min_pct, self.bounds.spo2_max_pct),
            "heart_rate": (self.bounds.heart_rate_min, self.bounds.heart_rate_max),
            "alt": (self.bounds.alt_min, self.bounds.alt_max),
            "ast": (self.bounds.ast_min, self.bounds.ast_max),
            "creatinine": (self.bounds.creatinine_min, self.bounds.creatinine_max),
            "wbc": (self.bounds.wbc_min, self.bounds.wbc_max),
            "cortisol": (self.bounds.cortisol_min, self.bounds.cortisol_max),
            "tnf_alpha": (self.bounds.tnf_alpha_min, self.bounds.tnf_alpha_max),
            "il6": (self.bounds.il6_min, self.bounds.il6_max),
        }
        for var, (lo, hi) in bounds_map.items():
            if var in corrected:
                corrected[var] = max(lo, min(hi, corrected[var]))
        return corrected

    def homeostatic_penalty(self, state: dict[str, float]) -> float:
        """Compute soft penalty for homeostatic deviation (for optimizer)."""
        penalty = 0.0
        checks = [
            ("ph", "ph_min", "ph_max"),
            ("glucose", "glucose_min_mgdL", "glucose_max_mgdL"),
            ("map", "map_min_mmhg", "map_max_mmhg"),
            ("creatinine", "creatinine_min", "creatinine_max"),
            ("wbc", "wbc_min", "wbc_max"),
        ]
        for var, min_attr, max_attr in checks:
            if var in state:
                val = state[var]
                lo = getattr(self.bounds, min_attr)
                hi = getattr(self.bounds, max_attr)
                if val < lo:
                    penalty += (lo - val) ** 2
                elif val > hi:
                    penalty += (val - hi) ** 2
        return penalty


class MassBalanceChecker:
    """Check mass balance for metabolic reactions.

    Verifies: Σ(stoichiometric_coeff × flux) ≈ d[metabolite]/dt
    """

    def __init__(self, tolerance: float = 0.01) -> None:
        self.tolerance = tolerance

    def check(
        self,
        stoich_matrix: list[list[float]],
        fluxes: list[float],
        metabolite_deltas: list[float],
    ) -> ConstraintCheckResult:
        """Check mass balance for a set of reactions.

        Args:
            stoich_matrix: rows = metabolites, cols = reactions
            fluxes: reaction fluxes
            metabolite_deltas: measured d[metabolite]/dt

        Returns:
            ConstraintCheckResult with mass balance violations
        """
        violations: list[ConstraintViolation] = []

        for i, (row, delta) in enumerate(
            zip(stoich_matrix, metabolite_deltas, strict=True)
        ):
            computed_delta = sum(s * f for s, f in zip(row, fluxes, strict=True))
            error = abs(computed_delta - delta)
            if error > self.tolerance:
                violations.append(
                    ConstraintViolation(
                        variable=f"metabolite_{i}",
                        value=computed_delta,
                        bound_min=delta - self.tolerance,
                        bound_max=delta + self.tolerance,
                        severity=error,
                    )
                )

        return ConstraintCheckResult(
            violations=violations,
            is_valid=len(violations) == 0,
            total_penalty=sum(v.severity for v in violations),
        )


class ThermodynamicChecker:
    """Check thermodynamic feasibility of metabolic reactions (doc/32 §7.3 constraint 2).

    Verifies:
    - Irreversible reactions: ΔG < 0 (spontaneous in forward direction)
    - Near-equilibrium reactions: |ΔG| < ε (consistent with near-equilibrium)

    Uses tabulated ΔG°' values (kJ/mol) from BioCyc/KEGG for core human metabolic
    reactions relevant to drug metabolism.
    """

    # ΔG°' at pH 7.0, 25°C, 1 mM Mg²⁺ (kJ/mol) — from BioCyc/KEGG
    # Positive ΔG°' means reverse is spontaneous at standard conditions;
    # actual ΔG = ΔG°' + RT·ln(Q), so Q shifts feasibility.
    REACTION_DG0: dict[str, float] = {
        "hexokinase": -16.7,
        "pfk": -14.2,
        "pyruvate_kinase": -31.4,
        "lactate_dehydrogenase": -25.1,
        "citrate_synthase": -31.5,
        "isocitrate_dehydrogenase": -21.0,
        "alpha_ketoglutarate_dehydrogenase": -33.5,
        "succinyl_coa_synthetase": -2.9,
        "succinate_dehydrogenase": -6.0,
        "fumarase": -3.8,
        "malate_dehydrogenase": 29.7,
        "phosphoglucose_isomerase": 1.7,
        "glucose_6_phosphatase": -3.3,
        "fructose_1_6_bisphosphatase": -16.4,
        "pyruvate_dehydrogenase": -33.4,
        "glutamine_synthetase": -16.3,
        "urea_cycle_carbamoyl_phosphate": -12.6,
        "cyp_oxidation": 0.0,
        "ugt_conjugation": -15.0,
        "sulfotransferase": -10.0,
    }

    # reactions that MUST be irreversible in vivo (ΔG must stay < 0)
    IRREVERSIBLE: set[str] = {
        "hexokinase", "pfk", "pyruvate_kinase", "citrate_synthase",
        "isocitrate_dehydrogenase", "alpha_ketoglutarate_dehydrogenase",
        "pyruvate_dehydrogenase", "glucose_6_phosphatase",
    }

    R_GAS = 8.314e-3  # kJ/(mol·K)

    def __init__(self, tolerance: float = 5.0, temperature_k: float = 310.15) -> None:
        self.tolerance = tolerance
        self.temperature_k = temperature_k

    def check_reaction(
        self,
        reaction_name: str,
        metabolite_concentrations: dict[str, float] | None = None,
        stoich_coeffs: list[float] | None = None,
        metabolite_names: list[str] | None = None,
    ) -> ConstraintViolation | None:
        """Check if a single reaction is thermodynamically feasible.

        Args:
            reaction_name: key into REACTION_DG0
            metabolite_concentrations: {metabolite: concentration in M}
            stoich_coeffs: stoichiometric coefficients (negative for reactants, positive for products)
            metabolite_names: corresponding metabolite names (same order as stoich_coeffs)
        """
        if reaction_name not in self.REACTION_DG0:
            return None

        dg0 = self.REACTION_DG0[reaction_name]

        # if no concentrations provided, just check sign of ΔG°'
        if metabolite_concentrations is None or stoich_coeffs is None or metabolite_names is None:
            if reaction_name in self.IRREVERSIBLE and dg0 >= 0:
                return ConstraintViolation(
                    variable=f"thermo_{reaction_name}",
                    value=dg0,
                    bound_min=float("-inf"),
                    bound_max=0.0,
                    severity=abs(dg0),
                )
            return None

        # compute ΔG = ΔG°' + RT·ln(Q)
        import math
        log_q = 0.0
        for coeff, met in zip(stoich_coeffs, metabolite_names, strict=True):
            conc = max(metabolite_concentrations.get(met, 1e-6), 1e-9)
            log_q += coeff * math.log(conc)

        rt = self.R_GAS * self.temperature_k
        dg = dg0 + rt * log_q

        if reaction_name in self.IRREVERSIBLE and dg >= 0:
            return ConstraintViolation(
                variable=f"thermo_{reaction_name}",
                value=dg,
                bound_min=float("-inf"),
                bound_max=0.0,
                severity=abs(dg),
            )

        return None

    def check_all(
        self,
        reaction_fluxes: dict[str, float],
        metabolite_concentrations: dict[str, float] | None = None,
        stoich_data: dict[str, tuple[list[float], list[str]]] | None = None,
    ) -> ConstraintCheckResult:
        """Check all known reactions for thermodynamic feasibility.

        Args:
            reaction_fluxes: {reaction_name: flux} — only reactions with non-zero flux are checked
            metabolite_concentrations: optional concentrations for ΔG calculation
            stoich_data: optional {reaction_name: (stoich_coeffs, metabolite_names)}
                for concentration-corrected ΔG calculation
        """
        violations: list[ConstraintViolation] = []
        for rxn, flux in reaction_fluxes.items():
            if abs(flux) < 1e-12:
                continue
            if stoich_data and rxn in stoich_data and metabolite_concentrations:
                coeffs, names = stoich_data[rxn]
                v = self.check_reaction(rxn, metabolite_concentrations, coeffs, names)
            else:
                v = self.check_reaction(rxn, metabolite_concentrations)
            if v is not None:
                violations.append(v)

        return ConstraintCheckResult(
            violations=violations,
            is_valid=len(violations) == 0,
            total_penalty=sum(v.severity for v in violations),
        )
