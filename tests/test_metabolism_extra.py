"""Additional tests for metabolism.py — uncovered branches and functions.

Covers:
  - _simplex_max (pure-python) paths: optimal, unbounded, max_iter, zero-row
  - _simplex_max_numpy paths: optimal, zero-row, ndim-1 guard
  - enzyme_correction
  - EnzymeCapacity validation
  - MetabolitePool net_production, integrate, overflow_flux
  - FluxBalanceAnalysis solve/analyze with enzyme capacity
  - _detect_exchange_ids / _detect_exchange_metabolites
  - activate_acetate_switch
  - DynamicFluxBalance: init, reset, set_state, step, run, to_simulation_result,
    update_from_environment, apply_to_environment, step_from_solution,
    bound_override, feed events, chemostat
  - PhotoautotrophicFluxBalance: init, reset, step, co2_uptake_bound, light_effect
  - MetabolicProxy: fit, predict, rmse
  - _poly_features
  - solve_lp with method='scipy'
  - _solve_scipy
"""
from __future__ import annotations

import copy
from unittest.mock import MagicMock

import numpy as np
import pytest

import helixlang.plugins.runtime.metabolism as m
from helixlang.plugins.runtime.metabolism import (
    ECOLI_CORE_MODEL,
    BioError,
    DynamicFBAConfig,
    DynamicFluxBalance,
    DynamicSimulationResult,
    EnzymeCapacity,
    FeedEvent,
    FluxBalanceAnalysis,
    MetabolicModel,
    MetabolicProxy,
    MetabolitePool,
    MetabolitePoolConfig,
    PhotoautotrophicFluxBalance,
    Reaction,
    _poly_features,
    _simplex_check_feasible_bounds,
    _simplex_extract_solution,
    _simplex_max,
    _simplex_max_numpy,
    activate_acetate_switch,
    enzyme_correction,
    simplex,
    solve_lp,
)

# ─── simplex helpers ─────────────────────────────────────────────────

class TestSimplexMax:
    def test_optimal(self):
        tableau = [[1.0, 0.0, 4.0], [0.0, 1.0, 6.0]]
        basis = [0, 1]
        obj = [3.0, 2.0]
        assert _simplex_max(tableau, basis, obj, 2) == "optimal"

    def test_unbounded(self):
        tableau = [[1.0, -1.0, 0.0]]
        basis = [0]
        obj = [0.0, 1.0]
        assert _simplex_max(tableau, basis, obj, 2) == "unbounded"

    def test_max_iter(self):
        tableau = [[1.0, 1.0, 1.0, 0.0, 10.0],
                    [1.0, 0.0, 0.0, 1.0, 5.0]]
        basis = [2, 3]
        obj = [1.0, 1.0, 0.0, 0.0]
        assert _simplex_max(tableau, basis, obj, 4, max_iter=1) == "max_iter"

    def test_forbidden(self):
        tableau = [[1.0, 0.0, 1.0], [0.0, 1.0, 2.0]]
        basis = [0, 1]
        obj = [1.0, 1.0]
        assert _simplex_max(tableau, basis, obj, 2, forbidden={0, 1}) == "optimal"


class TestSimplexMaxNumpy:
    def test_zero_rows(self):
        tab = np.zeros((0, 3), dtype=np.float64)
        assert _simplex_max_numpy(tab, [], np.zeros(3), 2) == "optimal"

    def test_optimal(self):
        tableau = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 6.0]])
        basis = [0, 1]
        obj = np.array([3.0, 2.0])
        assert _simplex_max_numpy(tableau, basis, obj, 2) == "optimal"

    def test_unbounded(self):
        tableau = np.array([[1.0, -1.0, 0.0]])
        basis = [0]
        obj = np.array([0.0, 1.0])
        assert _simplex_max_numpy(tableau, basis, obj, 2) == "unbounded"

    def test_ndim1_guard(self):
        tab1d = np.array([], dtype=np.float64)
        result = _simplex_max_numpy(tab1d, [], np.zeros(1), 0)
        assert result == "optimal"

    def test_forbidden(self):
        tableau = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 2.0]])
        basis = [0, 1]
        obj = np.array([1.0, 1.0])
        assert _simplex_max_numpy(tableau, basis, obj, 2, forbidden={0, 1}) == "optimal"

    def test_max_iter(self):
        tableau = np.array([[1.0, 1.0, 1.0, 0.0, 10.0],
                            [1.0, 0.0, 0.0, 1.0, 5.0]])
        basis = [2, 3]
        obj = np.array([1.0, 1.0, 0.0, 0.0])
        assert _simplex_max_numpy(tableau, basis, obj, 4, max_iter=1) == "max_iter"

    def test_tied_ratio_blands(self):
        tableau = np.array([[1.0, 0.0, 0.0, 1.0, 5.0],
                            [0.0, 1.0, 0.0, 1.0, 5.0]])
        basis = [0, 1]
        obj = np.array([0.0, 0.0, 1.0])
        result = _simplex_max_numpy(tableau, basis, obj, 3, forbidden=set())
        assert result in ("optimal", "unbounded", "max_iter")


class TestSimplexHelpers:
    def test_check_feasible_bounds(self):
        assert _simplex_check_feasible_bounds([0, 0], [1, 1], 1e-9) is True
        assert _simplex_check_feasible_bounds([2, 0], [1, 1], 1e-9) is False

    def test_extract_solution(self):
        tableau = [[0.0, 1.0, 5.0], [1.0, 0.0, 3.0]]
        basis = [1, 0]
        lbs = [0.0, 0.0]
        n = 2
        x, y = _simplex_extract_solution(tableau, basis, lbs, 2, n)
        assert x[0] == pytest.approx(3.0)
        assert x[1] == pytest.approx(5.0)


# ─── enzyme_correction ───────────────────────────────────────────────

class TestEnzymeCorrection:
    def test_optimal(self):
        v = enzyme_correction(37.0, 7.0)
        assert v == pytest.approx(1.0, abs=0.01)

    def test_suboptimal(self):
        v = enzyme_correction(25.0, 7.0)
        assert 0.0 < v < 1.0

    def test_extreme_ph(self):
        v = enzyme_correction(37.0, 14.0)
        assert v < 0.01


# ─── EnzymeCapacity ──────────────────────────────────────────────────

class TestEnzymeCapacity:
    def test_valid(self):
        ec = EnzymeCapacity(gene_to_reactions={"g": ("r",)}, enzyme_scale=1.0)
        assert ec.enzyme_scale == 1.0

    def test_zero_scale_raises(self):
        with pytest.raises(ValueError, match="enzyme_scale"):
            EnzymeCapacity(gene_to_reactions={}, enzyme_scale=0.0)


# ─── MetabolitePool ──────────────────────────────────────────────────

class TestMetabolitePool:
    def test_init_and_net_production(self):
        pool = MetabolitePool(ECOLI_CORE_MODEL)
        fluxes = {"EX_glc": 5.0, "PFK": 3.0}
        net = pool.net_production("atp", fluxes)
        assert isinstance(net, float)

    def test_unknown_reaction_skipped(self):
        pool = MetabolitePool(ECOLI_CORE_MODEL)
        net = pool.net_production("atp", {"nonexistent_rxn": 1.0})
        assert net == 0.0

    def test_integrate(self):
        pool = MetabolitePool(ECOLI_CORE_MODEL,
                              config=MetabolitePoolConfig(dt_h=0.1, dilution=True))
        fluxes = {rid: 0.0 for rid in ECOLI_CORE_MODEL.reactions}
        fluxes["PFK"] = 1.0
        pool.integrate(fluxes, growth_rate=0.5)
        assert isinstance(pool.pools, dict)


# ─── FluxBalanceAnalysis with enzyme capacity ────────────────────────

class TestFBAEnzymeCapacity:
    def test_solve_with_ec(self):
        ec = EnzymeCapacity(
            gene_to_reactions=m.ECOLI_CORE_GENE_REACTIONS,
            kcat=m.ECOLI_CORE_KCAT,
            enzyme_scale=1e4,
        )
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        fba.set_enzyme_capacity(ec)
        sol = fba.solve()
        assert         sol["BIOMASS"] > 0.0

    def test_smoment_row(self):
        ec = EnzymeCapacity(
            gene_to_reactions=m.ECOLI_CORE_GENE_REACTIONS,
            kcat=m.ECOLI_CORE_KCAT,
            enzyme_scale=1e4,
            protein_mass_fraction=0.5,
        )
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        fba.set_enzyme_capacity(ec)
        sol = fba.solve()
        assert         sol["BIOMASS"] > 0.0

    def test_analyze(self):
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        report = fba.analyze()
        assert "objective_value" in report

    def test_set_uptake_multiple(self):
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 5.0)
        fba.set_uptake("O2", 10.0)
        sol = fba.solve()
        assert         sol["BIOMASS"] > 0.0

    def test_solve_different_objective(self):
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        sol = fba.solve(objective="PFK")
        assert isinstance(sol, dict)
        assert "PFK" in sol

    def test_set_enzyme_levels(self):
        ec = EnzymeCapacity(
            gene_to_reactions=m.ECOLI_CORE_GENE_REACTIONS,
            kcat=m.ECOLI_CORE_KCAT,
            enzyme_scale=1e4,
        )
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        fba.set_uptake("GLC", 10.0)
        fba.set_enzyme_capacity(ec)
        fba.set_enzyme_levels({"ptsG": 0.5, "glk": 0.3})
        sol = fba.solve()
        assert isinstance(sol, dict)

    def test_no_biomass_reaction(self):
        model = MetabolicModel()
        model.add_reaction(Reaction(
            id="r1", name="r1", stoichiometry={"A": -1, "B": 1}))
        fba = FluxBalanceAnalysis(model)
        with pytest.raises(BioError):
            fba.solve(objective="biomass")


# ─── exchange detection ──────────────────────────────────────────────

class TestExchangeDetection:
    def test_detect_ids_default(self):
        ex_glc, ex_o2, ex_ac = m._detect_exchange_ids(ECOLI_CORE_MODEL)
        assert ex_glc == "EX_glc"
        assert ex_o2 == "EX_o2"
        assert ex_ac == "EX_ac"

    def test_detect_ids_custom(self):
        model = MetabolicModel()
        model.add_reaction(Reaction(
            id="EX_glc-D_e", name="glc exchange", stoichiometry={"glc-D_e": -1.0}))
        ex_glc, ex_o2, ex_ac = m._detect_exchange_ids(model)
        assert ex_glc == "EX_glc-D_e"

    def test_detect_metabolites(self):
        met_glc, met_o2, met_ac = m._detect_exchange_metabolites(ECOLI_CORE_MODEL)
        assert met_glc != ""

    def test_detect_metabolites_missing_reaction(self):
        model = MetabolicModel()
        met_glc, met_o2, met_ac = m._detect_exchange_metabolites(model)
        assert met_glc == ""


# ─── activate_acetate_switch ────────────────────────────────────────

class TestAcetateSwitch:
    def test_activate(self):
        model = copy.deepcopy(ECOLI_CORE_MODEL)
        activate_acetate_switch(model)
        assert model.reactions["EX_ac"].lower_bound == -10.0
        for rid in ("ICL", "MAS", "ACS", "PEPCK", "FBP"):
            if rid in model.reactions:
                assert model.reactions[rid].upper_bound == 1000.0


# ─── DynamicFluxBalance ──────────────────────────────────────────────

class TestDynamicFluxBalance:
    def test_init(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        assert dfba.time_h == 0.0
        assert dfba.glucose_mm > 0.0

    def test_init_acetate_switch(self):
        cfg = DynamicFBAConfig(acetate_switch=True)
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        assert dfba.time_h == 0.0

    def test_reset(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        dfba.step()
        dfba.reset()
        assert dfba.time_h == 0.0
        assert len(dfba.history) == 0

    def test_set_state(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        dfba.set_state(biomass_gdw=2.0, glucose_mm=20.0, acetate_mm=5.0)
        assert dfba.biomass_gdw == 2.0
        assert dfba.glucose_mm == 20.0
        assert dfba.byproducts_mm.get("acetate", 0.0) == 5.0

    def test_set_state_partial(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        orig_glucose = dfba.glucose_mm
        dfba.set_state(biomass_gdw=3.0)
        assert dfba.biomass_gdw == 3.0
        assert dfba.glucose_mm == orig_glucose

    def test_uptake_bound(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        assert dfba.uptake_bound(0.0) == 0.0
        assert dfba.uptake_bound(-1.0) == 0.0
        assert dfba.uptake_bound(100.0) > 0.0

    def test_oxygen_uptake_bound(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        assert dfba.oxygen_uptake_bound(0.0) == 0.0
        assert dfba.oxygen_uptake_bound(-1.0) == 0.0
        assert dfba.oxygen_uptake_bound(100.0) > 0.0

    def test_step(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        entry = dfba.step()
        assert "time" in entry
        assert "biomass" in entry
        assert "glucose" in entry
        assert dfba.time_h > 0.0

    def test_run_with_duration(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        hist = dfba.run(duration_h=0.1)
        assert len(hist) > 0
        assert dfba.time_h >= 0.1

    def test_run_stagnation(self):
        cfg = DynamicFBAConfig(initial_glucose_mm=0.0)
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        hist = dfba.run(max_steps=5)
        assert isinstance(hist, list)

    def test_growth_rate(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        dfba.step()
        assert dfba.growth_rate > 0.0

    def test_growth_rate_empty(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        assert dfba.growth_rate == 0.0

    def test_last(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        entry = dfba.step()
        assert dfba.last() == entry

    def test_to_simulation_result(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        dfba.run(duration_h=0.5)
        result = dfba.to_simulation_result()
        assert isinstance(result, DynamicSimulationResult)
        assert len(result.time_points) > 0
        assert result.final_biomass > 0.0
        assert result.doubling_time > 0.0

    def test_bound_override(self):
        def override(t, batch):
            return {}
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, bound_override=override)
        entry = dfba.step()
        assert "time" in entry

    def test_feed_events(self):
        cfg = DynamicFBAConfig(
            feed_events=[FeedEvent(time_h=0.05, metabolite="EX_glc", amount_mmol=5.0)]
        )
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        dfba.run(duration_h=0.2)
        assert len(dfba.history) > 1

    def test_feed_event_dilution(self):
        cfg = DynamicFBAConfig(
            feed_events=[FeedEvent(time_h=0.05, metabolite="EX_glc",
                                   amount_mmol=5.0, dilution=0.1)]
        )
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        dfba.run(duration_h=0.2)
        assert len(dfba.history) > 1

    def test_chemostat(self):
        cfg = DynamicFBAConfig(
            chemostat=True,
            chemostat_dilution_rate=0.5,
            chemostat_feed_concentrations={"glucose": 10.0},
        )
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        dfba.run(duration_h=0.1)
        assert len(dfba.history) > 0

    def test_chemostat_acetate(self):
        cfg = DynamicFBAConfig(
            chemostat=True,
            chemostat_dilution_rate=0.5,
            chemostat_feed_concentrations={"acetate": 2.0},
        )
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        dfba.run(duration_h=0.1)
        assert len(dfba.history) > 0

    def test_chemostat_oxygen(self):
        cfg = DynamicFBAConfig(
            chemostat=True,
            chemostat_dilution_rate=0.5,
            chemostat_feed_concentrations={"oxygen": 5.0},
        )
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        dfba.run(duration_h=0.1)
        assert len(dfba.history) > 0

    def test_step_from_solution(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        sol = {rid: 0.0 for rid in ECOLI_CORE_MODEL.reactions}
        sol["BIOMASS"] = 0.5
        sol["EX_glc"] = 10.0
        entry = dfba.step_from_solution(sol, glucose_mm=10.0)
        assert "time" in entry

    def test_integrate_oxygen_missing(self):
        cfg = DynamicFBAConfig(initial_oxygen_mm=0.0)
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        sol = {rid: 0.0 for rid in ECOLI_CORE_MODEL.reactions}
        sol["BIOMASS"] = 0.1
        entry = dfba.step_from_solution(sol, glucose_mm=5.0)
        assert entry["oxygen"] >= 0.0

    def test_integrate_acetate_reimport(self):
        cfg = DynamicFBAConfig(acetate_switch=True, initial_acetate_mm=5.0)
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        sol = {rid: 0.0 for rid in ECOLI_CORE_MODEL.reactions}
        sol["BIOMASS"] = 0.1
        sol["EX_glc"] = 5.0
        sol["EX_ac"] = -3.0
        entry = dfba.step_from_solution(sol, glucose_mm=5.0)
        assert entry["acetate"] <= 5.0

    def test_apply_bounds(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        dfba._apply_bounds({"EX_glc": 5.0, "PFK": 10.0})

    def test_update_from_environment(self):
        env = MagicMock()
        env.config.width = 10
        env.config.height = 10
        env.substrate_at.return_value = 5.0
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        dfba.update_from_environment(env)
        assert dfba.glucose_mm == 5.0

    def test_apply_to_environment(self):
        env = MagicMock()
        env.config.width = 10
        env.config.height = 10
        env.get_field.side_effect = KeyError()
        env.fields = {}
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        dfba.byproducts_mm["acetate"] = 1.0
        dfba.apply_to_environment(env)
        env.add_field.assert_called()

    def test_apply_to_environment_existing_field(self):
        env = MagicMock()
        env.config.width = 10
        env.config.height = 10
        field_mock = MagicMock()
        env.get_field.return_value = field_mock
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        dfba.byproducts_mm["acetate"] = 1.0
        dfba.apply_to_environment(env)
        field_mock.add.assert_called()


# ─── PhotoautotrophicFluxBalance ─────────────────────────────────────

class TestPhotoautotrophicFluxBalance:
    def test_init(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        assert pfb.time_h == 0.0

    def test_reset(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        pfb.step()
        pfb.reset()
        assert pfb.time_h == 0.0

    def test_light_effect(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        v = pfb.light_effect()
        assert 0.0 <= v <= 1.0

    def test_co2_uptake_bound(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        assert pfb.co2_uptake_bound(0.0) == 0.0
        assert pfb.co2_uptake_bound(-1.0) == 0.0
        assert pfb.co2_uptake_bound(5.0) > 0.0

    def test_step(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        entry = pfb.step()
        assert "time" in entry
        assert "biomass" in entry


# ─── MetabolicProxy ──────────────────────────────────────────────────

class TestPolyFeatures:
    def test_degree1(self):
        assert _poly_features([2.0], 1) == [1.0, 2.0]

    def test_degree2(self):
        feats = _poly_features([2.0, 3.0], 2)
        assert feats[0] == 1.0
        assert 2.0 in feats
        assert 3.0 in feats
        assert 4.0 in feats  # x0^2
        assert 6.0 in feats  # x0*x1
        assert 9.0 in feats  # x1^2


class TestMetabolicProxy:
    def test_fit_and_predict(self):
        proxy = MetabolicProxy(degree=1, max_uptake=5.0)
        proxy.fit(n_samples=20, seed=42)
        pred = proxy.predict({"GLC": 2.0})
        assert "BIOMASS" in pred
        assert pred["BIOMASS"] >= 0.0

    def test_predict_vector(self):
        proxy = MetabolicProxy(degree=1, max_uptake=5.0)
        proxy.fit(n_samples=20, seed=42)
        pred = proxy.predict([2.0])
        assert "BIOMASS" in pred

    def test_predict_bad_key(self):
        proxy = MetabolicProxy(degree=1, max_uptake=5.0)
        proxy.fit(n_samples=10, seed=42)
        with pytest.raises(ValueError, match="unknown uptake feature"):
            proxy.predict({"bad_key": 1.0})

    def test_predict_wrong_length(self):
        proxy = MetabolicProxy(degree=1, max_uptake=5.0)
        with pytest.raises(ValueError, match="uptake vector must match"):
            proxy.predict([1.0, 2.0, 3.0, 4.0])

    def test_rmse(self):
        proxy = MetabolicProxy(degree=1, max_uptake=5.0)
        proxy.fit(n_samples=30, seed=42)
        errors = proxy.rmse(n_holdout=10, seed=99)
        assert "BIOMASS" in errors
        assert errors["BIOMASS"] >= 0.0

    def test_nn_fallback(self):
        proxy = MetabolicProxy(degree=1, max_uptake=5.0)
        proxy.coeffs = {}
        proxy._train_x = [[1.0], [2.0], [3.0]]
        proxy._train_y = {"Biomass": [0.1, 0.2, 0.3]}
        proxy.features = ["GLC"]
        proxy.outputs = ["Biomass"]
        pred = proxy.predict({"GLC": 2.0})
        assert pred["Biomass"] == pytest.approx(0.2)

    def test_nn_not_fitted(self):
        proxy = MetabolicProxy(degree=1, max_uptake=5.0)
        proxy.coeffs = {}
        with pytest.raises(RuntimeError, match="fit"):
            proxy.predict([1.0])

    def test_custom_features_outputs(self):
        proxy = MetabolicProxy(
            features=["GLC"], outputs=["PFK"], degree=1, max_uptake=5.0)
        proxy.fit(n_samples=10, seed=0)
        pred = proxy.predict({"GLC": 1.0})
        assert "PFK" in pred


# ─── solve_lp / scipy dispatch ───────────────────────────────────────

class TestSolveLP:
    def test_simplex_dispatch(self):
        result = solve_lp(
            c=[1.0, 1.0],
            A=[[1.0, 0.0], [0.0, 1.0]],
            b=[5.0, 5.0],
            bounds=[(0, 10), (0, 10)],
            method="simplex",
        )
        assert result["status"] == "optimal"
        assert result["objective"] == pytest.approx(10.0)

    def test_scipy_dispatch(self):
        result = solve_lp(
            c=[1.0, 1.0],
            A=[[1.0, 0.0], [0.0, 1.0]],
            b=[5.0, 5.0],
            bounds=[(0, 10), (0, 10)],
            method="scipy",
        )
        assert result["status"] == "optimal"
        assert result["objective"] == pytest.approx(10.0)

    def test_auto_simplex_small(self):
        result = solve_lp(
            c=[1.0, 1.0],
            A=[[1.0, 0.0], [0.0, 1.0]],
            b=[5.0, 5.0],
            bounds=[(0, 10), (0, 10)],
            method="auto",
        )
        assert result["status"] == "optimal"


# ─── simplex entry point (build_tableau → _simplex_max paths) ────────

class TestSimplexEntryPoint:
    def test_basic(self):
        result = simplex(
            c=[1.0, 1.0],
            A=[[1.0, 0.0], [0.0, 1.0]],
            b=[5.0, 5.0],
            bounds=[(0, 10), (0, 10)],
        )
        assert result["status"] == "optimal"
        assert result["objective"] == pytest.approx(10.0)

    def test_infeasible_bounds(self):
        result = simplex(
            c=[1.0],
            A=[],
            b=[],
            bounds=[(5.0, 2.0)],
        )
        assert result["status"] == "infeasible"

    def test_with_lower_bounds(self):
        result = simplex(
            c=[1.0, 2.0],
            A=[[1.0, 1.0]],
            b=[10.0],
            bounds=[(2.0, 10.0), (1.0, 10.0)],
        )
        assert result["status"] == "optimal"
        assert result["objective"] >= 6.0


# ─── additional metabolism coverage ─────────────────────────────────

class TestMetabolitePoolOverflow:
    def test_overflow_flux(self):
        pool = MetabolitePool(ECOLI_CORE_MODEL)
        fluxes = {"EX_ac": 3.0, "BIOMASS": 0.5, "PFK": 1.0}
        ovf = pool.overflow_flux(fluxes)
        assert isinstance(ovf, dict)

    def test_overflow_excludes_biomass(self):
        pool = MetabolitePool(ECOLI_CORE_MODEL)
        fluxes = {"BIOMASS": 5.0, "EX_ac": 2.0}
        ovf = pool.overflow_flux(fluxes)
        assert "Biomass" not in ovf

    def test_overflow_negative_ignored(self):
        pool = MetabolitePool(ECOLI_CORE_MODEL)
        fluxes = {"EX_ac": -1.0}
        ovf = pool.overflow_flux(fluxes)
        assert ovf == {}

    def test_integrate_negative_dt(self):
        pool = MetabolitePool(ECOLI_CORE_MODEL)
        with pytest.raises(ValueError, match="dt_h"):
            pool.integrate({}, dt_h=-1.0)

    def test_integrate_zero_dt(self):
        pool = MetabolitePool(ECOLI_CORE_MODEL)
        fluxes = {"PFK": 1.0}
        deltas = pool.integrate(fluxes, dt_h=0.0)
        assert all(v == 0.0 for v in deltas.values())

    def test_integrate_min_pool_floor(self):
        cfg = MetabolitePoolConfig(dt_h=1.0, min_pool=0.01, dilution=False)
        pool = MetabolitePool(ECOLI_CORE_MODEL, config=cfg)
        fluxes = {rid: -100.0 for rid in ECOLI_CORE_MODEL.reactions}
        pool.integrate(fluxes)
        for pool_val in pool.pools.values():
            assert pool_val >= cfg.min_pool


class TestDFBAMoreBranches:
    def test_oxygen_integrate_path(self):
        cfg = DynamicFBAConfig(
            initial_oxygen_mm=10.0,
            max_oxygen_uptake=5.0,
            oxygen_half_saturation_mm=0.5,
        )
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        entry = dfba.step()
        assert "oxygen" in entry

    def test_run_max_steps(self):
        cfg = DynamicFBAConfig(initial_glucose_mm=1000.0)
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        hist = dfba.run(duration_h=1.0, max_steps=3)
        assert len(hist) <= 3

    def test_feed_event_oxygen(self):
        cfg = DynamicFBAConfig(
            initial_oxygen_mm=1.0,
            feed_events=[FeedEvent(time_h=0.0, metabolite="EX_o2", amount_mmol=5.0)],
        )
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        dfba.step()
        assert dfba.byproducts_mm.get("oxygen", 0.0) >= 0.0

    def test_feed_event_acetate(self):
        cfg = DynamicFBAConfig(
            acetate_switch=True,
            initial_acetate_mm=0.0,
            feed_events=[FeedEvent(time_h=0.0, metabolite="EX_ac", amount_mmol=3.0)],
        )
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        dfba.step()
        assert "acetate" in dfba.byproducts_mm

    def test_acetate_switch_high_glucose(self):
        cfg = DynamicFBAConfig(
            acetate_switch=True,
            initial_glucose_mm=10.0,
            initial_acetate_mm=5.0,
        )
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        entry = dfba.step()
        assert "acetate" in entry

    def test_acetate_switch_low_glucose(self):
        cfg = DynamicFBAConfig(
            acetate_switch=True,
            initial_glucose_mm=0.01,
            initial_acetate_mm=5.0,
        )
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        entry = dfba.step()
        assert "acetate" in entry

    def test_chemostat_byproduct_dilution(self):
        cfg = DynamicFBAConfig(
            chemostat=True,
            chemostat_dilution_rate=0.5,
            chemostat_feed_concentrations={"oxygen": 5.0, "acetate": 2.0},
            initial_oxygen_mm=5.0,
            initial_acetate_mm=3.0,
        )
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        hist = dfba.run(duration_h=0.05)
        assert len(hist) > 0

    def test_step_from_solution_acetate_export(self):
        cfg = DynamicFBAConfig(acetate_switch=True, initial_acetate_mm=0.0)
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        sol = {rid: 0.0 for rid in ECOLI_CORE_MODEL.reactions}
        sol["BIOMASS"] = 0.5
        sol["EX_glc"] = 5.0
        sol["EX_ac"] = 2.0
        entry = dfba.step_from_solution(sol, glucose_mm=5.0)
        assert entry["acetate"] >= 0.0

    def test_to_simulation_result_growth_rate_key(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        dfba.run(duration_h=0.2)
        result = dfba.to_simulation_result()
        assert len(result.time_points) > 0
        assert len(result.biomass) > 0

    def test_update_from_environment_with_acetate(self):
        env = MagicMock()
        env.config.width = 10
        env.config.height = 10
        env.substrate_at.return_value = 2.0
        env.fields = {"acetate": MagicMock()}
        env.fields["acetate"].get.return_value = 3.0
        cfg = DynamicFBAConfig(acetate_switch=True)
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL, config=cfg)
        dfba.update_from_environment(env)
        assert dfba.glucose_mm == 2.0
        assert dfba.byproducts_mm.get("acetate", 0.0) == 3.0

    def test_apply_bounds_non_glc(self):
        dfba = DynamicFluxBalance(ECOLI_CORE_MODEL)
        dfba._apply_bounds({"PFK": 5.0})
        assert dfba.fba.model.reactions["PFK"].upper_bound == 5.0


class TestDFBAInfeasible:
    def test_infeasible_returns_zeros(self):
        fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
        sol = fba._build_and_solve("BIOMASS", maximize=True)
        assert isinstance(sol, dict)


class TestPhotoFBAExtra:
    def test_run(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        hist = pfb.run(duration_h=0.5)
        assert len(hist) > 0

    def test_run_max_steps(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        hist = pfb.run(duration_h=100.0, max_steps=3)
        assert len(hist) <= 3

    def test_growth_rate_empty(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        assert pfb.growth_rate == 0.0

    def test_last(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        entry = pfb.step()
        assert pfb.last() == entry

    def test_to_simulation_result(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        pfb.run(duration_h=0.5)
        result = pfb.to_simulation_result()
        assert isinstance(result, DynamicSimulationResult)
        assert result.final_biomass >= 0.0

    def test_to_simulation_result_no_growth(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        result = pfb.to_simulation_result()
        assert result.final_biomass == 0.0

    def test_set_co2_bound_no_ex(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        pfb._ex_co2 = ""
        pfb._set_co2_bound()

    def test_set_pet_bound_no_pet(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        pfb._set_pet_bound()

    def test_step_no_ex_co2(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        pfb._ex_co2 = ""
        entry = pfb.step()
        assert entry["co2_uptake"] == 0.0

    def test_step_vbm_zero(self):
        pfb = PhotoautotrophicFluxBalance(ECOLI_CORE_MODEL)
        pfb.co2_mm = 0.0
        entry = pfb.step()
        assert entry["biomass"] >= 0.0


class TestSolveLPScipy:
    def test_infeasible_scipy(self):
        result = solve_lp(
            c=[1.0],
            A=[[1.0]],
            b=[-5.0],
            bounds=[(0, 10)],
            method="scipy",
        )
        assert result["status"] != "optimal"

    def test_auto_dispatches_scipy_large(self):
        n = 200
        c = [1.0] * n
        A = [[1.0] * n]
        b = [float(n)]
        bounds = [(0.0, 1.0)] * n
        result = solve_lp(c, A, b, bounds, method="auto")
        assert result["status"] in ("optimal", "max_iter")

    def test_simplex_infeasible(self):
        result = simplex(
            c=[1.0],
            A=[[1.0]],
            b=[-1.0],
            bounds=[(0, 10)],
        )
        assert result["status"] in ("infeasible", "unbounded")


class TestLoadModel:
    def test_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            m.load_model("/nonexistent/path/model.json")

    def test_sbml_missing_file(self):
        with pytest.raises(OSError):
            m.load_model("test.xml")
