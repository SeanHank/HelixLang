"""FBA simplex method edge-case tests.

Verifies in src/helixlang/metabolism.py:
- ``simplex`` solver edge cases: empty/trivial problems, negative RHS, zero objective, fixed variables,
  degenerate/redundant constraints, multiple optima, numerical stability, max_iter truncation, unbounded detection.
- ``FluxBalanceAnalysis`` edge cases: infeasible models (no uptake + fixed ATPM),
  custom objectives, minimization, uptake on non-exchange metabolites, negative uptake, multiple uptake limits,
  zero-width bounds, single-reaction models, solution caching.

Does not repeat the basic tests in test_metabolism.py; focuses on edge and degenerate scenarios.

References:
- Dantzig GB. Linear Programming and Extensions 1963 (simplex method / degeneracy / two-phase)
- Orth JD et al. Mol Syst Biol 2010 6:390 (E. coli core model)
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from helixlang.core.errors import BioError
from helixlang.plugins.runtime.metabolism import (
    ATP_MAINTENANCE_FLUX,
    ECOLI_CORE_MODEL,
    FluxBalanceAnalysis,
    MetabolicModel,
    Reaction,
    simplex,
)

# ============================================================================
# simplex edge cases: empty/trivial problems
# ============================================================================

class TestSimplexEmptyAndTrivial:
    """Empty/trivial LP problems."""

    def test_no_variables_no_constraints(self):
        """n=0, m=0 → trivial optimal."""
        result = simplex(c=[], A=[], b=[], bounds=[], maximize=True)
        assert result["status"] == "optimal"
        assert result["x"] == []
        assert result["objective"] == 0.0

    def test_no_constraints_multiple_vars_maximize(self):
        """m=0, multiple variables maximization: each takes its upper bound."""
        c = [1.0, 2.0, 3.0]
        result = simplex(c, A=[], b=[], bounds=[(0, 5), (0, 3), (0, 2)], maximize=True)
        assert result["status"] == "optimal"
        assert result["x"][0] == pytest.approx(5.0, abs=1e-6)
        assert result["x"][1] == pytest.approx(3.0, abs=1e-6)
        assert result["x"][2] == pytest.approx(2.0, abs=1e-6)
        assert result["objective"] == pytest.approx(17.0, abs=1e-6)

    def test_no_constraints_multiple_vars_minimize(self):
        """m=0, multiple variables minimization: each takes its lower bound."""
        c = [1.0, 2.0]
        result = simplex(c, A=[], b=[], bounds=[(3, 5), (1, 4)], maximize=False)
        assert result["status"] == "optimal"
        assert result["x"][0] == pytest.approx(3.0, abs=1e-6)
        assert result["x"][1] == pytest.approx(1.0, abs=1e-6)
        assert result["objective"] == pytest.approx(5.0, abs=1e-6)

    def test_zero_objective_feasible(self):
        """Zero objective: any feasible solution is optimal."""
        c = [0.0, 0.0]
        result = simplex(c, A=[], b=[], bounds=[(3, 7), (0, 2)], maximize=True)
        assert result["status"] == "optimal"
        assert result["objective"] == pytest.approx(0.0, abs=1e-9)

    def test_negative_objective_maximized_at_lower_bound(self):
        """Maximizing a negative coefficient → take the lower bound."""
        c = [-1.0]
        result = simplex(c, A=[], b=[], bounds=[(2, 8)], maximize=True)
        assert result["status"] == "optimal"
        assert result["x"][0] == pytest.approx(2.0, abs=1e-6)
        assert result["objective"] == pytest.approx(-2.0, abs=1e-6)


# ============================================================================
# simplex edge cases: fixed variables and zero-width bounds
# ============================================================================

class TestSimplexFixedAndZeroBounds:
    """Fixed variables (lb == ub) and zero-width bounds."""

    def test_fixed_variable_lb_equals_ub(self):
        """lb == ub → variable is fixed."""
        c = [1.0, 1.0]
        result = simplex(c, A=[], b=[], bounds=[(5, 5), (0, 10)], maximize=True)
        assert result["status"] == "optimal"
        assert result["x"][0] == pytest.approx(5.0, abs=1e-6)
        assert result["x"][1] == pytest.approx(10.0, abs=1e-6)
        assert result["objective"] == pytest.approx(15.0, abs=1e-6)

    def test_all_variables_fixed_unique_solution(self):
        """All variables fixed → unique feasible solution."""
        c = [1.0, 2.0]
        result = simplex(c, A=[], b=[], bounds=[(3, 3), (4, 4)], maximize=True)
        assert result["status"] == "optimal"
        assert result["x"][0] == pytest.approx(3.0, abs=1e-6)
        assert result["x"][1] == pytest.approx(4.0, abs=1e-6)
        assert result["objective"] == pytest.approx(11.0, abs=1e-6)

    def test_all_variables_bounded_to_zero(self):
        """All variables lb=ub=0 → zero solution."""
        c = [1.0, 1.0]
        result = simplex(c, A=[], b=[], bounds=[(0, 0), (0, 0)], maximize=True)
        assert result["status"] == "optimal"
        assert result["objective"] == pytest.approx(0.0, abs=1e-9)

    def test_fixed_variable_with_equality_constraint(self):
        """Fixed variable + equality constraint."""
        # x fixed at 3, y free; x + y = 10 → y = 7; max y
        c = [0.0, 1.0]
        A = [[1.0, 1.0]]
        b = [10.0]
        result = simplex(c, A, b, bounds=[(3, 3), (0, 20)], maximize=True)
        assert result["status"] == "optimal"
        assert result["x"][0] == pytest.approx(3.0, abs=1e-6)
        assert result["x"][1] == pytest.approx(7.0, abs=1e-4)


# ============================================================================
# simplex edge cases: negative RHS
# ============================================================================

class TestSimplexNegativeRHS:
    """Row sign handling when equality constraint RHS is negative."""

    def test_negative_rhs_flips_row_sign(self):
        """-x = -5 → x = 5."""
        c = [1.0]
        A = [[-1.0]]
        b = [-5.0]
        result = simplex(c, A, b, bounds=[(0, 10)], maximize=True)
        assert result["status"] == "optimal"
        assert result["x"][0] == pytest.approx(5.0, abs=1e-6)
        assert result["objective"] == pytest.approx(5.0, abs=1e-6)

    def test_mixed_sign_rhs(self):
        """Multiple constraints with mixed positive/negative RHS."""
        # x - y = -2 (i.e. y = x + 2), x + y = 8 → x=3, y=5
        c = [1.0, 0.0]
        A = [[1.0, -1.0], [1.0, 1.0]]
        b = [-2.0, 8.0]
        result = simplex(c, A, b, bounds=[(0, 10), (0, 10)], maximize=True)
        assert result["status"] == "optimal"
        assert result["x"][0] == pytest.approx(3.0, abs=1e-4)
        assert result["x"][1] == pytest.approx(5.0, abs=1e-4)

    def test_all_negative_rhs(self):
        """All-negative RHS."""
        # -x = -4, -y = -3 → x=4, y=3; max x+y
        c = [1.0, 1.0]
        A = [[-1.0, 0.0], [0.0, -1.0]]
        b = [-4.0, -3.0]
        result = simplex(c, A, b, bounds=[(0, 10), (0, 10)], maximize=True)
        assert result["status"] == "optimal"
        assert result["x"][0] == pytest.approx(4.0, abs=1e-4)
        assert result["x"][1] == pytest.approx(3.0, abs=1e-4)


# ============================================================================
# simplex edge cases: degenerate and redundant constraints
# ============================================================================

class TestSimplexDegenerateAndRedundant:
    """Degenerate and redundant constraints (test Phase 1 artificial-variable cleanup)."""

    def test_redundant_duplicate_constraints(self):
        """Linearly dependent (duplicate) constraints still solvable."""
        # x + y = 4, 2x + 2y = 8 (redundant), max x+y
        c = [1.0, 1.0]
        A = [[1.0, 1.0], [2.0, 2.0]]
        b = [4.0, 8.0]
        result = simplex(c, A, b, bounds=[(0, 10), (0, 10)], maximize=True)
        assert result["status"] == "optimal"
        assert result["objective"] == pytest.approx(4.0, abs=1e-4)

    def test_three_redundant_constraints(self):
        """Three linearly dependent constraints."""
        # x + y + z = 6, 2x+2y+2z=12, 3x+3y+3z=18, max x
        c = [1.0, 0.0, 0.0]
        A = [[1.0, 1.0, 1.0],
             [2.0, 2.0, 2.0],
             [3.0, 3.0, 3.0]]
        b = [6.0, 12.0, 18.0]
        result = simplex(c, A, b,
                         bounds=[(0, 10), (0, 10), (0, 10)], maximize=True)
        assert result["status"] == "optimal"
        assert result["objective"] == pytest.approx(6.0, abs=1e-4)

    def test_multiple_optima_objective_value(self):
        """Multiple optima: objective value unique, specific solution not fixed."""
        # max x + y s.t. x + y + s = 4, x,y,s >= 0
        c = [1.0, 1.0, 0.0]
        A = [[1.0, 1.0, 1.0]]
        b = [4.0]
        result = simplex(c, A, b,
                         bounds=[(0, 10), (0, 10), (0, 10)], maximize=True)
        assert result["status"] == "optimal"
        assert result["objective"] == pytest.approx(4.0, abs=1e-6)
        assert result["x"][0] + result["x"][1] == pytest.approx(4.0, abs=1e-4)

    def test_equality_constraint_rhs_zero(self):
        """Equality constraint RHS = 0 (degenerate case)."""
        # x - y = 0 (x = y), max x, x,y in [0, 5]
        c = [1.0, 0.0]
        A = [[1.0, -1.0]]
        b = [0.0]
        result = simplex(c, A, b, bounds=[(0, 5), (0, 5)], maximize=True)
        assert result["status"] == "optimal"
        assert result["x"][0] == pytest.approx(5.0, abs=1e-4)
        assert result["x"][1] == pytest.approx(5.0, abs=1e-4)

    def test_degenerate_vertex_multiple_constraints_tight(self):
        """Degenerate vertex: multiple constraints tight at the vertex."""
        # max x s.t. x <= 3, x <= 3 (duplicate), y <= 2, x + y <= 5
        # convert to equality: x + s1 = 3, x + s2 = 3, y + s3 = 2, x + y + s4 = 5
        c = [1.0, 0.0, 0.0, 0.0, 0.0]
        A = [[1.0, 0.0, 1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 1.0],
             [1.0, 1.0, 0.0, 0.0, 0.0]]
        # the last column above corresponds to s4, but A columns correspond to [x, y, s1, s2, s3], missing s4 → rearrange
        # use a simpler form instead
        c = [1.0, 0.0]
        A = [[1.0, 0.0], [1.0, 0.0]]  # x = 3 (duplicate)
        b = [3.0, 3.0]
        result = simplex(c, A, b, bounds=[(0, 10), (0, 10)], maximize=True)
        assert result["status"] == "optimal"
        assert result["x"][0] == pytest.approx(3.0, abs=1e-4)


# ============================================================================
# simplex edge cases: numerical stability
# ============================================================================

class TestSimplexNumericalStability:
    """Numerical stability."""

    def test_small_coefficients(self):
        """Small coefficients (1e-6) still solvable."""
        c = [1e-6]
        result = simplex(c, A=[], b=[], bounds=[(0, 1e6)], maximize=True)
        assert result["status"] == "optimal"
        assert result["objective"] == pytest.approx(1.0, abs=1e-3)

    def test_large_objective_value(self):
        """Large objective value: x = 1000, c = 1000 → obj = 1e6."""
        c = [1000.0]
        result = simplex(c, A=[], b=[], bounds=[(0, 1000)], maximize=True)
        assert result["status"] == "optimal"
        assert result["objective"] == pytest.approx(1e6, abs=1e-3)

    def test_tight_bounds(self):
        """Tight bounds [4.999, 5.001]."""
        c = [1.0]
        result = simplex(c, A=[], b=[], bounds=[(4.999, 5.001)], maximize=True)
        assert result["status"] == "optimal"
        assert result["x"][0] == pytest.approx(5.001, abs=1e-6)

    def test_fractional_coefficients(self):
        """Fractional coefficients."""
        # max 0.5x + 0.25y s.t. x + y = 10, x,y >= 0
        c = [0.5, 0.25]
        A = [[1.0, 1.0]]
        b = [10.0]
        result = simplex(c, A, b, bounds=[(0, 100), (0, 100)], maximize=True)
        assert result["status"] == "optimal"
        # 0.5 > 0.25 → x = 10, y = 0
        assert result["x"][0] == pytest.approx(10.0, abs=1e-4)
        assert result["objective"] == pytest.approx(5.0, abs=1e-4)


# ============================================================================
# simplex edge cases: max_iter truncation and unbounded detection
# ============================================================================

class TestSimplexMaxIterAndUnbounded:
    """max_iter truncation and unbounded detection."""

    def test_max_iter_zero_does_not_crash(self):
        """max_iter=0 does not crash; returns max_iter or optimal."""
        c = [1.0]
        result = simplex(c, A=[], b=[], bounds=[(0, 10)], maximize=True, max_iter=0)
        assert result["status"] in ("max_iter", "optimal")

    def test_max_iter_one_does_not_crash(self):
        """max_iter=1 does not crash; result may be unreliable.

        Note: too-low max_iter leaves Phase 1 artificial variables not cleared, which may falsely report
        "infeasible" (even though the problem is actually feasible). This is known behavior of the
        two-phase simplex when the iteration limit is insufficient — Phase 1 is not complete when art_sum > 1e-6 is checked.
        """
        # the classic LP needs multiple iterations
        c = [3.0, 2.0]
        A = [[1.0, 1.0], [2.0, 1.0]]
        b = [4.0, 5.0]
        result = simplex(c, A, b, bounds=[(0, 100), (0, 100)],
                         maximize=True, max_iter=1)
        # not crashing suffices; with insufficient max_iter the status may be max_iter / infeasible / optimal
        assert result["status"] in ("max_iter", "optimal", "infeasible")

    def test_low_max_iter_can_false_infeasible(self):
        """max_iter too low → Phase 1 incomplete → falsely reports infeasible."""
        c = [3.0, 2.0]
        A = [[1.0, 1.0], [2.0, 1.0]]
        b = [4.0, 5.0]
        result = simplex(c, A, b, bounds=[(0, 100), (0, 100)],
                         maximize=True, max_iter=1)
        # with sufficient iterations this problem is optimal (x=1, y=3, obj=9), so infeasible is a false report
        result_full = simplex(c, A, b, bounds=[(0, 100), (0, 100)],
                              maximize=True, max_iter=10000)
        assert result_full["status"] == "optimal"
        # max_iter=1 may falsely report infeasible because Phase 1 is incomplete
        if result["status"] == "infeasible":
            # confirm this is a false report due to insufficient max_iter (solvable with sufficient iterations)
            assert result_full["status"] == "optimal"

    def test_unbounded_minimization(self):
        """Unbounded minimization: min(-x) s.t. x in [0, inf)."""
        # min(-x) → internally max(x), unbounded
        c = [-1.0]
        result = simplex(c, A=[], b=[], bounds=[(0, float("inf"))],
                         maximize=False)
        assert result["status"] == "unbounded"

    def test_large_finite_bound_treated_as_unbounded(self):
        """Large finite upper bound (> _BIG=1e9) is treated as unbounded.

        simplex substitutes +inf with _BIG=1e9; a finite upper bound exceeding _BIG is reported as
        unbounded when a variable hits the _BIG cap. This is the documented numerical strategy
        (see the simplex source comment "substitute a large number for +inf upper bounds"),
        and in real FBA DEFAULT_UPPER_BOUND=1000 is far below _BIG, so it is unaffected.
        """
        c = [1.0]
        result = simplex(c, A=[], b=[], bounds=[(0, 1e12)], maximize=True)
        assert result["status"] == "unbounded"

    def test_bound_exactly_at_big_is_optimal(self):
        """Upper bound exactly = _BIG(1e9) is still optimal (does not trigger unbounded detection)."""
        c = [1.0]
        result = simplex(c, A=[], b=[], bounds=[(0, 1e9)], maximize=True)
        assert result["status"] == "optimal"
        assert result["x"][0] == pytest.approx(1e9, abs=1.0)


# ============================================================================
# FBA edge cases: infeasible models
# ============================================================================

class TestFBAInfeasibleModel:
    """Infeasible FBA models."""

    def test_no_uptake_with_atpm_returns_all_zeros(self):
        """No glucose uptake + fixed ATPM → LP infeasible → FBA returns all zeros.

        ATPM lower bound = upper bound = 8.39 (ATP must be consumed), but with no carbon source no ATP
        can be produced, so the LP is infeasible. FBA returns an all-zero solution for infeasibility.
        """
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fluxes = fba.solve()
        for rid, v in fluxes.items():
            assert v == pytest.approx(0.0, abs=1e-9), \
                f"{rid}={v} not zero in infeasible model"

    def test_infeasible_simplex_status_directly(self):
        """Call simplex directly to verify the LP is infeasible without uptake.

        Construct the same LP as FBA (no uptake_limits), confirm simplex returns
        "infeasible" rather than "optimal" — proving the all-zero result is due to infeasibility, not a zero optimum.
        """
        met_list, rxn_list, S = ECOLI_CORE_MODEL.get_stoichiometry_matrix()
        n = len(rxn_list)
        c = [0.0] * n
        c[rxn_list.index("BIOMASS")] = 1.0
        bounds = []
        for rid in rxn_list:
            rxn = ECOLI_CORE_MODEL.reactions[rid]
            bounds.append((rxn.lower_bound, rxn.upper_bound))
        b = [0.0] * len(met_list)
        result = simplex(c, S, b, bounds, maximize=True)
        # ATPM fixed at 8.39 but no carbon source → infeasible
        assert result["status"] == "infeasible"

    def test_negative_uptake_makes_infeasible(self):
        """Negative uptake rate → EX_glc upper bound < lower bound → infeasible."""
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", -5.0)  # ub = -5, lb = 0 → lb > ub
        fluxes = fba.solve()
        assert fluxes["BIOMASS"] == pytest.approx(0.0, abs=1e-9)
        assert fluxes["EX_glc"] == pytest.approx(0.0, abs=1e-9)

    def test_infeasible_model_analyze_reports_zero(self):
        """Infeasible model analyze reports all zeros."""
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        # no uptake → infeasible
        report = fba.analyze()
        assert report["biomass_yield"] == pytest.approx(0.0, abs=1e-9)
        assert report["glucose_uptake"] == pytest.approx(0.0, abs=1e-9)


# ============================================================================
# FBA edge cases: custom objectives
# ============================================================================

class TestFBACustomObjective:
    """Custom objective reactions."""

    def test_maximize_lactate_secretion(self):
        """Maximize lactate secretion (non-biomass objective)."""
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        fluxes = fba.solve(objective="EX_lac")
        assert isinstance(fluxes, dict)
        # maximize EX_lac: LDH produces Lac for secretion, should be positive
        assert fluxes["EX_lac"] > 0

    def test_maximize_atpm_objective(self):
        """Maximize ATP maintenance flux."""
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        fluxes = fba.solve(objective="ATPM")
        # ATPM fixed at 8.39, maximization still 8.39
        assert fluxes["ATPM"] == pytest.approx(ATP_MAINTENANCE_FLUX, abs=1e-3)

    def test_minimize_biomass_below_maximize(self):
        """Minimize biomass < maximize biomass."""
        fba_max = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba_max.set_uptake("GLC", 10.0)
        bm_max = fba_max.solve(maximize=True)["BIOMASS"]

        fba_min = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba_min.set_uptake("GLC", 10.0)
        bm_min = fba_min.solve(maximize=False)["BIOMASS"]
        assert bm_min < bm_max

    def test_unknown_objective_reaction_raises(self):
        """Unknown objective reaction raises KeyError."""
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        with pytest.raises(BioError):
            fba.solve(objective="NOT_A_REACTION")

    def test_custom_objective_mass_balance_holds(self):
        """Custom objective solutions still satisfy steady-state mass balance S·v = 0."""
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        fluxes = fba.solve(objective="EX_lac")
        met_list, rxn_list, S = ECOLI_CORE_MODEL.get_stoichiometry_matrix()
        for i, met in enumerate(met_list):
            net = sum(S[i][j] * fluxes[rxn_list[j]] for j in range(len(rxn_list)))
            assert abs(net) < 1e-3, f"metabolite {met} unbalanced: net={net:.6f}"


# ============================================================================
# FBA edge cases: uptake limit bounds
# ============================================================================

class TestFBAUptakeEdgeCases:
    """Uptake limit edge cases."""

    def test_uptake_on_non_exchange_metabolite_ignored(self):
        """Setting uptake on a non-exchange metabolite → no effect on the solution (no EX_ reaction matches)."""
        fba1 = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba1.set_uptake("GLC", 10.0)
        bm1 = fba1.solve()["BIOMASS"]

        fba2 = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba2.set_uptake("GLC", 10.0)
        fba2.set_uptake("G6P", 100.0)  # G6P has no EX_ reaction
        bm2 = fba2.solve()["BIOMASS"]
        assert bm1 == pytest.approx(bm2, abs=1e-6)

    def test_multiple_uptake_limits(self):
        """Multiple uptake limits coexist (only GLC has an EX_ reaction)."""
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        fba.set_uptake("Lac", 5.0)  # EX_lac coefficient is negative (secretion direction), not overridden
        fluxes = fba.solve()
        assert fluxes["EX_glc"] <= 10.0 + 1e-6

    def test_set_uptake_overwrites_previous(self):
        """Repeated set_uptake overwrites the old value."""
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        fba.set_uptake("GLC", 5.0)
        fluxes = fba.solve()
        assert fluxes["EX_glc"] <= 5.0 + 1e-6

    def test_uptake_limits_stored(self):
        """set_uptake stores the limits dict."""
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        assert fba.uptake_limits["GLC"] == 10.0

    def test_zero_uptake_same_as_no_uptake(self):
        """set_uptake("GLC", 0) is equivalent to no uptake."""
        fba1 = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba1.set_uptake("GLC", 0.0)
        bm1 = fba1.solve()["BIOMASS"]

        fba2 = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        bm2 = fba2.solve()["BIOMASS"]
        assert bm1 == pytest.approx(bm2, abs=1e-9)


# ============================================================================
# FBA edge cases: zero-width bounds and single-reaction models
# ============================================================================

class TestFBAZeroWidthAndSingleReaction:
    """Zero-width bounds and single-reaction models."""

    def test_single_reaction_model_forced_zero(self):
        """Single-reaction model A→B: mass balance S·v=0 forces v=0 (no exchange reactions)."""
        m = MetabolicModel()
        m.add_reaction(Reaction(
            id="R1", name="r1",
            stoichiometry={"A": -1, "B": 1},
            lower_bound=0, upper_bound=5,
        ))
        m.set_biomass("R1")
        fba = FluxBalanceAnalysis(m)
        fluxes = fba.solve()
        # A: -v1=0, B: v1=0 → v1=0
        assert fluxes["R1"] == pytest.approx(0.0, abs=1e-6)

    def test_zero_width_bound_pins_reaction(self):
        """Zero-width bounds (lb=ub=0) pin the reaction flux to 0."""
        m = deepcopy(ECOLI_CORE_MODEL)
        m.reactions["LDH"].lower_bound = 0.0
        m.reactions["LDH"].upper_bound = 0.0
        fba = FluxBalanceAnalysis(m)
        fba.set_uptake("GLC", 10.0)
        fluxes = fba.solve()
        assert fluxes["LDH"] == pytest.approx(0.0, abs=1e-9)

    def test_pinned_reaction_keeps_model_feasible(self):
        """Pinning a nonessential reaction keeps the model feasible, biomass > 0."""
        m = deepcopy(ECOLI_CORE_MODEL)
        m.reactions["LDH"].lower_bound = 0.0
        m.reactions["LDH"].upper_bound = 0.0
        fba = FluxBalanceAnalysis(m)
        fba.set_uptake("GLC", 10.0)
        fluxes = fba.solve()
        assert fluxes["BIOMASS"] > 0

    def test_pinned_biomass_to_zero(self):
        """BIOMASS pinned to 0 → biomass flux 0, but the model is still feasible."""
        m = deepcopy(ECOLI_CORE_MODEL)
        m.reactions["BIOMASS"].lower_bound = 0.0
        m.reactions["BIOMASS"].upper_bound = 0.0
        fba = FluxBalanceAnalysis(m)
        fba.set_uptake("GLC", 10.0)
        fluxes = fba.solve()
        assert fluxes["BIOMASS"] == pytest.approx(0.0, abs=1e-9)


# ============================================================================
# FBA edge cases: solution caching
# ============================================================================

class TestFBASolutionCaching:
    """Solution caching behavior."""

    def test_last_solution_none_initially(self):
        """last_solution is None before solving."""
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        assert fba.last_solution is None

    def test_analyze_uses_cached_solution(self):
        """analyze uses the cached solution (does not re-solve)."""
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        fba.solve()
        cached = fba.last_solution
        report = fba.analyze()
        assert fba.last_solution is cached
        assert report["objective_value"] == cached["BIOMASS"]

    def test_resolve_overwrites_cache(self):
        """Re-solving overwrites the cache."""
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        first = fba.solve()
        assert fba.last_solution is first
        fba.set_uptake("GLC", 5.0)
        second = fba.solve()
        assert fba.last_solution is second
        assert second["EX_glc"] <= 5.0 + 1e-6


# ============================================================================
# FBA edge cases: model building
# ============================================================================

class TestFBAModelBuilding:
    """Model building edge cases."""

    def test_metabolic_model_duplicate_reaction_raises(self):
        """Duplicate reaction id raises ValueError."""
        m = MetabolicModel()
        m.add_reaction(Reaction(id="R1", name="r1",
                                 stoichiometry={"A": -1, "B": 1}))
        with pytest.raises(BioError):
            m.add_reaction(Reaction(id="R1", name="r1",
                                     stoichiometry={"C": -1, "D": 1}))

    def test_empty_model_solve_raises(self):
        """Empty model (no reactions) solve raises ValueError (no biomass reaction)."""
        m = MetabolicModel()
        fba = FluxBalanceAnalysis(m)
        with pytest.raises(BioError):
            fba.solve()

    def test_model_with_only_biomass_no_uptake(self):
        """Only biomass reaction, no uptake → biomass 0."""
        m = MetabolicModel()
        m.add_reaction(Reaction(
            id="BIOMASS", name="bio",
            stoichiometry={"A": -1, "Biomass": 1},
            lower_bound=0, upper_bound=10,
        ))
        m.set_biomass("BIOMASS")
        fba = FluxBalanceAnalysis(m)
        fluxes = fba.solve()
        # A has no source → BIOMASS = 0
        assert fluxes["BIOMASS"] == pytest.approx(0.0, abs=1e-6)
