"""Coverage closure for helixlang.plugins.runtime.metabolism."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import helixlang.plugins.runtime.metabolism as metab
from helixlang.plugins.runtime.environment import (
    ACETATE_DIFFUSION_UM2_S,
    ConcentrationField,
    Environment,
)
from helixlang.plugins.runtime.metabolism import (
    BioError,
    DynamicFBAConfig,
    DynamicFluxBalance,
    EnzymeCapacity,
    FeedEvent,
    FluxBalanceAnalysis,
    MetabolicModel,
    MetabolicProxy,
    PhotoautotrophicFluxBalance,
    Reaction,
    _simplex_extract_solution,
    _simplex_max,
    _simplex_max_numpy,
    activate_acetate_switch,
    load_model,
    load_model_from_json,
    simplex,
)


def _mini_model(include_ac: bool = True, include_switch: bool = True):
    """Tiny GEM-style model: EX_S uptake (coef -1), biomass, optional ion."""
    m = MetabolicModel()
    m.add_reaction(Reaction(
        "EX_S", "S exchange", {"S": -1.0}, lower_bound=-10.0, upper_bound=0.0,
        subsystem="exchange"))
    m.add_reaction(Reaction(
        "BIO", "biomass", {"S": -1.0, "B": 1.0}, lower_bound=0.0,
        upper_bound=1000.0, subsystem="biomass"))
    m.add_reaction(Reaction(
        "EX_B", "B exchange", {"B": -1.0}, lower_bound=-1000.0,
        upper_bound=0.0, subsystem="exchange"))
    if include_ac:
        m.add_reaction(Reaction(
            "EX_ac", "acetate exchange", {"ac": -1.0}, lower_bound=-10.0,
            upper_bound=0.0, subsystem="exchange"))
    m.add_reaction(Reaction(
        "EX_SEC", "S secretion", {"L": 1.0}, lower_bound=-10.0,
        upper_bound=0.0, subsystem="exchange"))
    m.add_reaction(Reaction(
        "EX_Z", "Z zero-coef exchange", {"Z": 0.0}, lower_bound=-10.0,
        upper_bound=10.0, subsystem="exchange"))
    if include_switch:
        m.add_reaction(Reaction(
            "PEPCK", "pepck", {"B": -1.0, "A": 1.0}, lower_bound=0.0,
            upper_bound=0.0, subsystem="gluconeogenesis"))
    m.set_biomass("BIO")
    return m


# ---------------------------------------------------------------------------
# model loading / cobra conversion
# ---------------------------------------------------------------------------
def test_load_model_json_with_and_without_biomass(tmp_path):
    rxn = {"id": "R1", "stoichiometry": {"A": 1.0}, "lower_bound": 0.0,
           "upper_bound": 10.0, "subsystem": "other"}
    p1 = tmp_path / "m_no_bio.json"
    p1.write_text(json.dumps({"reactions": [rxn]}))
    m = load_model_from_json(p1)
    assert m.biomass_reaction is None
    p2 = tmp_path / "m_bio.json"
    p2.write_text(json.dumps({"reactions": [dict(rxn, id="BIO")],
                              "biomass_reaction": "BIO"}))
    m2 = load_model_from_json(p2)
    assert m2.biomass_reaction == "BIO"


class _MetId:
    """Hashable stand-in for a cobrapy Metabolite (has ``.id``)."""
    __slots__ = ("id",)

    def __init__(self, name: str) -> None:
        self.id = name

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _MetId) and other.id == self.id


def _met(name: str) -> _MetId:
    return _MetId(name)


def _stub_rxn(rid, stoich=None, lb=0.0, ub=1000.0, subsystem="exchange",
              gpr=None, obj_coef=0.0, name=None):
    mets = {_met(k): v for k, v in
            (stoich if stoich is not None else {"GLC": -1.0}).items()}
    return SimpleNamespace(
        id=rid, name=name or rid, metabolites=mets,
        lower_bound=lb, upper_bound=ub, subsystem=subsystem,
        gene_reaction_rule=gpr, objective_coefficient=obj_coef)


def _stub_cobra(obj_coef_rxn=True, objective=None, preserve_genes=True,
                include_biomass=True):
    r1 = _stub_rxn("EX_glc_e", {"glc_e": -1.0},
                   obj_coef=1.0 if obj_coef_rxn else 0.0)
    r2 = _stub_rxn("PDH", {"pyr": -1.0, "acCoA": 1.0}, subsystem="central",
                   gpr="G1 and G2")
    r3 = _stub_rxn("PFK", gpr="None")
    r4 = _stub_rxn("BGLS", gpr="")
    r5 = _stub_rxn("BiomassEcoli", {"Biomass": 1.0})
    genes = []
    if preserve_genes:
        genes.append(SimpleNamespace(id="G1", name="g1", reactions=[r2, r3]))
    obj = objective
    if obj is None and obj_coef_rxn:
        obj = SimpleNamespace(expression=None)
    rxns = [r1, r2, r3, r4]
    if include_biomass:
        rxns.append(r5)
    return SimpleNamespace(reactions=rxns,
                           genes=genes, objective=obj)


def test_from_cobra_strategy_1():
    m = metab._from_cobra_model(_stub_cobra())
    assert m.biomass_reaction == "EX_glc_e"
    assert m.genes["G1"].protein_reaction_rules == ["PDH", "PFK"]
    assert m.reactions["PDH"].gene_reaction_rule == "G1 and G2"
    assert m.reactions["PFK"].gene_reaction_rule is None
    assert m.reactions["BGLS"].gene_reaction_rule is None


def test_from_cobra_strategy_2_expression():
    expr = SimpleNamespace(args=[
        SimpleNamespace(variable=SimpleNamespace(name="reverse_test")),
        SimpleNamespace(variable=SimpleNamespace(name="PDH")),
    ])
    obj = SimpleNamespace(expression=expr)
    m = metab._from_cobra_model(_stub_cobra(obj_coef_rxn=False, objective=obj))
    assert m.biomass_reaction == "PDH"


def test_from_cobra_strategy_3_name_fallback():
    obj = SimpleNamespace(expression=SimpleNamespace(
        args=[SimpleNamespace(variable=SimpleNamespace(name="reverse_thing"))]))
    m = metab._from_cobra_model(_stub_cobra(obj_coef_rxn=False, objective=obj))
    assert m.biomass_reaction == "BiomassEcoli"


def test_from_cobra_no_objective_no_genes():
    m = metab._from_cobra_model(_stub_cobra(obj_coef_rxn=False, objective=None),
                                preserve_gpr=False)
    assert m.biomass_reaction is None
    assert m.genes == {}
    assert m.reactions["EX_glc_e"].gene_reaction_rule is None


def test_from_cobra_no_expression_no_biomass():
    obj = SimpleNamespace(expression=None)
    m = metab._from_cobra_model(
        _stub_cobra(obj_coef_rxn=False, objective=obj,
                    include_biomass=False))
    assert m.biomass_reaction is None


def test_from_cobra_term_without_variable():
    expr = SimpleNamespace(args=[
        SimpleNamespace(),  # no .variable attribute
        SimpleNamespace(variable=SimpleNamespace(name="PDH")),
    ])
    obj = SimpleNamespace(expression=expr)
    m = metab._from_cobra_model(
        _stub_cobra(obj_coef_rxn=False, objective=obj))
    assert m.biomass_reaction == "PDH"


def test_from_cobra_default_bounds():
    sbml = SimpleNamespace(reactions=[_stub_rxn("R_null", lb=None, ub=None)],
                           genes=[], objective=None)
    m = metab._from_cobra_model(sbml)
    rxn = m.reactions["R_null"]
    assert rxn.lower_bound == 0.0 and rxn.upper_bound == 1000.0


def test_load_sbml_via_cobra(monkeypatch, tmp_path):
    seen = {}

    def fake_read(path, *a, **k):
        seen["path"] = str(path)
        return _stub_cobra()

    cobra = importlib.import_module("cobra")
    monkeypatch.setattr(cobra.io, "read_sbml_model", fake_read)
    p = tmp_path / "tiny.sbml"
    p.write_text("<model/>")
    m = load_model(p)
    assert m.biomass_reaction == "EX_glc_e"
    assert seen["path"] == str(p)


def test_load_model_cobra_import_error():
    sys.modules["cobra"] = None
    try:
        with pytest.raises(BioError):
            load_model(Path("tiny.sbml"))
        with pytest.raises(BioError):
            load_model("iJO1366")
    finally:
        del sys.modules["cobra"]
    assert load_model() is metab.ECOLI_CORE_MODEL


def test_load_identifier_success_and_failure(monkeypatch):
    cobra = importlib.import_module("cobra")

    def fake_load(identifier, *a, **k):
        return _stub_cobra()

    monkeypatch.setattr(cobra.io, "load_model", fake_load)
    assert load_model("iJO1366").biomass_reaction == "EX_glc_e"

    def failing_load(identifier, *a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(cobra.io, "load_model", failing_load)
    with pytest.raises(BioError):
        load_model("iML1515")


# ---------------------------------------------------------------------------
# simplex kernel paths
# ---------------------------------------------------------------------------
def test_simplex_pure_python_fallback(monkeypatch):
    monkeypatch.setattr(metab, "_HAS_NUMPY", False)
    res = simplex([1.0], [[1.0]], [1.0], [(0.0, 10.0)])
    assert res["status"] == "optimal"


def test_simplex_native_optin(monkeypatch):
    monkeypatch.setenv("HELIX_ACCEL_SIMPLEX", "native")
    res = simplex([1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], [2.0, 3.0],
                  [(0.0, 10.0), (0.0, 10.0)])
    assert res["status"] == "optimal"
    assert res["objective"] == pytest.approx(5.0)


def test_simplex_phase2_statuses(monkeypatch):
    orig = metab._simplex_max_dispatch
    state = {"n": 0}

    def make_fake(status):
        def fake(tab, basis, obj_ar, n_vars, eps, max_iter, forbidden=None):
            state["n"] += 1
            if state["n"] == 1:
                return orig(tab, basis, obj_ar, n_vars, eps, max_iter,
                            forbidden)
            return status
        return fake

    monkeypatch.setattr(metab, "_simplex_max_dispatch", make_fake("unbounded"))
    res = simplex([1.0, 1.0], [[1.0, 0.0]], [2.0], [(0.0, 10.0), (0.0, 10.0)])
    assert res["status"] == "unbounded"

    state["n"] = 0
    monkeypatch.setattr(metab, "_simplex_max_dispatch", make_fake("max_iter"))
    res2 = simplex([1.0, 1.0], [[1.0, 0.0]], [2.0], [(0.0, 10.0), (0.0, 10.0)])
    assert res2["status"] == "max_iter"


def test_simplex_elimination_zero_factor():
    tableau = [[1.0, 1.0, 2.0], [0.0, 1.0, 3.0]]
    basis = [3, 4]
    obj = [2.0, 3.0, 0.0, 0.0, 0.0]
    assert _simplex_max(tableau, basis, obj, 2) == "optimal"


def test_simplex_extract_basis_including_nonvariable():
    x, y = _simplex_extract_solution([[1.0, 1.0, 2.0], [0.0, 1.0, 3.0]],
                                     [5, 0], [1.0], 2, 1)
    assert x == [4.0]
    assert y == [3.0, 0.0]


def test_simplex_max_python_unbounded():
    status = _simplex_max([[0.0, 1.0, 1.0]], [1], [1.0, 1.0, 0.0], 2)
    assert status == "unbounded"


def test_simplex_max_python_max_iter():
    tableau = [[1.0, 1.0, 2.0], [0.0, 1.0, 3.0]]
    basis = [3, 4]
    obj = [2.0, 3.0, 0.0, 0.0, 0.0]
    assert _simplex_max(tableau, basis, obj, 2, max_iter=1) == "max_iter"


def test_scipy_unavailable_fallback(monkeypatch):
    import importlib.util
    real_opt = sys.modules.get("scipy.optimize")
    sys.modules["scipy.optimize"] = None
    try:
        spec = importlib.util.spec_from_file_location(
            "_metab_no_scipy", metab.__file__)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_metab_no_scipy"] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            del sys.modules["_metab_no_scipy"]
        assert mod._HAS_SCIPY is False
    finally:
        if real_opt is None:
            sys.modules.pop("scipy.optimize", None)
        else:
            sys.modules["scipy.optimize"] = real_opt
    monkeypatch.setattr(metab, "_HAS_SCIPY", False)
    res = metab._solve_scipy([1.0], [[1.0]], [1.0], [(0.0, 10.0)])
    assert res["status"] == "infeasible"


def test_scipy_solver_statuses():
    if not metab._HAS_SCIPY:
        pytest.skip("scipy unavailable")
    got = metab._solve_scipy([1.0], [[1.0]], [1.0], [(0.0, 10.0)],
                             maximize=True)
    assert got["status"] == "optimal"
    got2 = metab._solve_scipy([1.0], [[1.0]], [1.0], [(0.0, 10.0)],
                              maximize=False)
    assert got2["status"] == "optimal"
    inf = metab._solve_scipy([1.0], [[1.0]], [1.0], [(10.0, 0.0)])
    assert inf["status"] == "infeasible"


def test_solve_lp_scipy_dispatch():
    if not metab._HAS_SCIPY:
        pytest.skip("scipy unavailable")
    res = metab.solve_lp([1.0], [[1.0]], [1.0], [(0.0, 10.0)],
                         maximize=True, method="scipy")
    assert res["status"] == "optimal"


def test_simplex_max_numpy_edges():
    empty = _simplex_max_numpy(np.zeros((0, 5)), [], np.zeros(5), 5)
    assert empty == "optimal"
    unbounded = _simplex_max_numpy(
        np.array([[-1.0, 0.0, 1.0, 0.0, 0.0, 10.0],
                  [0.0, -1.0, 0.0, 1.0, 0.0, 10.0]]),
        [2, 3], np.array([2.0, 3.0, 0.0, 0.0, 0.0]), 5)
    assert unbounded == "unbounded"
    table = [
        [1.0, 1.0, 0.0, 0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 10.0],
    ]
    obj = np.array([2.0, 3.0, 0.0, 0.0, 0.0])
    assert _simplex_max_numpy(np.array(table), [4, 2, 3], obj, 5) == "optimal"
    max_iter = _simplex_max_numpy(np.array(table), [4, 2, 3], obj, 5,
                                  max_iter=1)
    assert max_iter == "max_iter"


def test_simplex_max_numpy_forbidden():
    table = [
        [1.0, 1.0, 0.0, 0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 10.0],
    ]
    obj = np.array([2.0, 3.0, 0.0, 0.0, 0.0])
    status = _simplex_max_numpy(np.array(table), [4, 2, 3], obj, 5,
                                forbidden={0, 1})
    assert status == "optimal"


def test_enzyme_correction():
    assert metab.enzyme_correction(37.0, 7.0) == pytest.approx(1.0)
    assert 0.0 < metab.enzyme_correction(20.0, 5.0) < 1.0


def test_metabolite_pool_net_production():
    pool = metab.MetabolitePool(metab.ECOLI_CORE_MODEL,
                                initial={"o2": 2.0, "glc": 1.0})
    net = pool.net_production("o2", {"EX_o2_e": -1.0, "EX_unknown": 5.0})
    assert net == 0.0  # unknown rxn skipped; EX_o2_e not in model
    deltas = pool.integrate({"EX_glc_e": -1.0}, growth_rate=0.5)
    assert deltas and all(v == 0.0 for v in deltas.values())


# ---------------------------------------------------------------------------
# FBA bounds / MOMENT / analyze
# ---------------------------------------------------------------------------
def test_fba_uptake_negative_coef():
    fba = FluxBalanceAnalysis(_mini_model())
    fba.set_uptake("S", 5.0)
    fba.set_uptake("L", 3.0)
    fba.set_uptake("Z", 1.0)
    sol = fba.solve()
    assert "BIO" in sol and "EX_S" in sol and "EX_SEC" in sol
    assert fba.last_solution is not None


def test_fba_moment_gene_branches():
    fba = FluxBalanceAnalysis(_mini_model())
    ec = EnzymeCapacity(
        gene_to_reactions={"G": ("BIO",), "H": ("BIO",), "K": ("UNUSED",)},
        kcat={"BIO": 10.0, "EX_S": 0.0, "EX_SEC": 5.0},
        enzyme_scale=1.0,
        protein_mass_fraction=0.5,
    )
    fba.set_enzyme_capacity(ec)
    fba.set_enzyme_levels({"G": 2.0, "H": 1.0})
    fba.set_uptake("S", 10.0)
    assert "BIO" in fba.solve()


def test_analyze_twice():
    fba = FluxBalanceAnalysis(metab.ECOLI_CORE_MODEL)
    fba.set_uptake("glc", 10.0)
    fba.solve()
    first = fba.analyze()
    second = fba.analyze()
    assert first["objective_value"] == second["objective_value"]


# ---------------------------------------------------------------------------
# acetate switch
# ---------------------------------------------------------------------------
def test_activate_acetate_switch_missing_reactions():
    m = _mini_model(include_ac=False, include_switch=False)
    activate_acetate_switch(m)
    assert "PEPCK" not in m.reactions
    partial = _mini_model(include_ac=True, include_switch=True)
    activate_acetate_switch(partial)
    assert partial.reactions["PEPCK"].upper_bound == 1000.0
    assert partial.reactions["EX_ac"].lower_bound == -10.0


def _dfba_core(**cfg):
    return DynamicFluxBalance(metab._build_ecoli_core_model(),
                              DynamicFBAConfig(**cfg))


def test_dynamic_fba_o2_and_analyze_edges():
    dfb = _dfba_core(dt_h=0.5, initial_glucose_mm=5.0, initial_oxygen_mm=1.0)
    dfb.set_state(biomass_gdw=0.4, acetate_mm=0.1)
    assert dfb.biomass_gdw == pytest.approx(0.4)
    assert dfb.byproducts_mm["acetate"] == pytest.approx(0.1)
    dfb.run(duration_h=1.0)
    assert dfb.to_simulation_result().time_points
    dfb.history.append({"time": 1.0, "note": "x", "biomass": 0.7})
    asyn2 = dfb.to_simulation_result()
    assert asyn2.time_points
    dfb.history.append({"time": 2.0, "biomass": 0.8, "growth_rate": 0.1,
                        "total_S": 3.0})
    asyn3 = dfb.to_simulation_result()
    assert asyn3.substrates.get("S") == [3.0]
    assert asyn3.final_biomass >= asyn3.biomass[0]


def test_dynamic_fba_run_edges_and_queries():
    dfb = _dfba_core(dt_h=0.5, initial_glucose_mm=0.0)
    assert dfb.growth_rate == 0.0
    dfb.run(max_steps=0)
    assert not dfb.history
    assert dfb.to_simulation_result().biomass == []
    depleted = _dfba_core(dt_h=0.5, initial_glucose_mm=0.0)
    depleted.run()
    assert any(h for h in depleted.history)
    assert depleted.growth_rate >= 0.0
    assert depleted.last()["glucose"] < 1e-9
    dec = _dfba_core(dt_h=0.5)
    dec.history = [{"time": 0.0, "biomass": 1.0},
                   {"time": 1.0, "biomass": 0.5}]
    assert dec.to_simulation_result().doubling_time == 0.0


def test_dynamic_fba_ex_o2_bound_missing_and_zero():
    dfb = _dfba_core(dt_h=0.5)
    dfb._set_oxygen_uptake_bound(1.0)
    zero = _dfba_core(dt_h=0.5, initial_glucose_mm=0.0,
                      initial_oxygen_mm=0.0)
    assert zero.oxygen_uptake_bound(0.0) == 0.0
    assert zero.oxygen_uptake_bound(2.0) > 0.0


def test_dynamic_fba_o2_exchange_signs():
    def make_model(coef):
        m = MetabolicModel()
        m.add_reaction(Reaction(
            "EX_glc", "glc ex", {"glc": 1.0}, lower_bound=0.0,
            upper_bound=10.0, subsystem="exchange"))
        m.add_reaction(Reaction(
            "EX_o2", "o2 ex", {"o2": float(coef)}, lower_bound=-20.0,
            upper_bound=20.0, subsystem="exchange"))
        m.add_reaction(Reaction(
            "BIO", "biomass", {"glc": -1.0, "o2": -1.0, "B": 1.0},
            lower_bound=0.0, upper_bound=1000.0, subsystem="biomass"))
        m.set_biomass("BIO")
        return m

    def make_stub(m, ex_o2_flux):
        class Stub:
            model = m

            def solve(self, *a, **k):
                return {"BIO": 1.0, "EX_glc": 2.0, "EX_o2": ex_o2_flux}
        return Stub()

    for coef, flux in ((-1.0, -0.5), (1.0, 0.5)):
        m = make_model(coef)
        fba = FluxBalanceAnalysis(m)
        stub = make_stub(m, flux)
        fba.solve = stub.solve
        dfb = DynamicFluxBalance(m, DynamicFBAConfig(dt_h=0.25), fba=fba)
        entry = dfb.step()
        assert entry["biomass"] > 0.0


def test_dynamic_fba_dup_byproduct_pool():
    m = MetabolicModel()
    m.add_reaction(Reaction(
        "EX_glc", "glc ex", {"glc": 1.0}, lower_bound=0.0, upper_bound=10.0,
        subsystem="exchange"))
    m.add_reaction(Reaction(
        "EX_ac1", "ac ex 1", {"Ac": -1.0}, lower_bound=-10.0, upper_bound=0.0,
        subsystem="exchange"))
    m.add_reaction(Reaction(
        "EX_ac2", "ac ex 2", {"Ac": -1.0}, lower_bound=-10.0, upper_bound=0.0,
        subsystem="exchange"))
    m.add_reaction(Reaction(
        "BIO", "biomass", {"glc": -1.0, "Ac": -1.0, "B": 1.0},
        lower_bound=0.0, upper_bound=1000.0, subsystem="biomass"))
    m.set_biomass("BIO")
    dfb = DynamicFluxBalance(m, DynamicFBAConfig(dt_h=0.25))
    assert sorted(dfb._byproduct_pools) == ["acetate"]
    assert dfb._byproduct_ex["EX_ac2"] == "acetate"


def test_dynamic_fba_o2_empty_stoich_consumption():
    m = MetabolicModel()
    m.add_reaction(Reaction(
        "EX_glc", "glc ex", {"glc": 1.0}, lower_bound=0.0, upper_bound=10.0,
        subsystem="exchange"))
    m.add_reaction(Reaction(
        "EX_o2", "o2 ex", {}, lower_bound=-1000.0, upper_bound=1000.0,
        subsystem="exchange"))
    m.add_reaction(Reaction(
        "BIO", "biomass", {"glc": -1.0, "B": 1.0}, lower_bound=0.0,
        upper_bound=1000.0, subsystem="biomass"))
    m.set_biomass("BIO")

    class Stub:
        model = m

        def solve(self, *a, **k):
            return {"BIO": 1.0, "EX_glc": 2.0, "EX_o2": 0.3}

    fba = FluxBalanceAnalysis(m)
    fba.solve = Stub().solve
    dfb = DynamicFluxBalance(m, DynamicFBAConfig(dt_h=0.25), fba=fba)
    entry = dfb.step()
    assert entry["biomass"] > 0.0


def test_dynamic_fba_feed_and_dilution():
    cfg = DynamicFBAConfig(
        dt_h=0.25, initial_glucose_mm=5.0,
        feed_events=[
            FeedEvent(time_h=0.0, metabolite="EX_glc", amount_mmol=2.0),
            FeedEvent(time_h=0.25, metabolite="EX_ac", amount_mmol=3.0,
                      dilution=0.5),
            FeedEvent(time_h=0.25, metabolite="EX_o2", amount_mmol=1.0),
            FeedEvent(time_h=0.25, metabolite="EX_weird", amount_mmol=9.0),
        ])
    dfb = DynamicFluxBalance(metab._build_ecoli_core_model(), cfg)
    dfb.step()
    dfb.step()
    assert dfb.byproducts_mm["acetate"] > 0.0


def test_dynamic_fba_chemostat():
    cfg = DynamicFBAConfig(
        dt_h=0.1, initial_glucose_mm=5.0, chemostat=True,
        chemostat_dilution_rate=0.5,
        chemostat_feed_concentrations={"glucose": 2.0, "oxygen": 1.0,
                                       "acetate": 0.5, "other": 1.0})
    dfb = DynamicFluxBalance(metab._build_ecoli_core_model(), cfg)
    dfb.step()
    assert dfb.history


def test_dynamic_fba_acetate_switch_edges():
    low = _dfba_core(acetate_switch=True, dt_h=0.25, initial_glucose_mm=0.01,
                     acetate_switch_threshold_mm=0.5,
                     initial_acetate_mm=1.0)
    low.set_state(glucose_mm=0.001)
    low.step()
    assert low.byproducts_mm["acetate"] >= 0.0
    high = _dfba_core(acetate_switch=True, dt_h=0.25, initial_glucose_mm=10.0)
    high.step()


def test_dynamic_fba_acetate_switch_no_ex_ac():
    no_ac = DynamicFluxBalance(
        _mini_model(include_ac=False, include_switch=False),
        DynamicFBAConfig(acetate_switch=True, dt_h=0.25))
    no_ac.step()
    assert no_ac.history
    tiny = DynamicFluxBalance(_mini_model(include_ac=True, include_switch=False),
                              DynamicFBAConfig(acetate_switch=True, dt_h=0.25))
    tiny.set_state(glucose_mm=0.001)
    tiny.step()
    assert tiny.byproducts_mm
    assert all(x >= 0.0 for x in tiny.byproducts_mm.values())


def test_dynamic_fba_acetate_switch_missing_rxns():
    tiny = DynamicFluxBalance(_mini_model(include_ac=True, include_switch=False),
                              DynamicFBAConfig(acetate_switch=True, dt_h=0.25))
    tiny.set_state(glucose_mm=10.0)
    tiny.step()
    assert tiny.history


def test_dynamic_fba_apply_bounds_and_unknown():
    dfb = _dfba_core(dt_h=0.25, initial_glucose_mm=5.0)
    dfb.bound_override = lambda t, b: {
        dfb._ex_glc: 10.0, "PYK": 5.0, "UNKNOWN_RXN": 1.0}
    dfb.step()
    assert dfb.history


def test_dynamic_fba_no_glc_exchange():
    dfb = DynamicFluxBalance(_mini_model(),
                             DynamicFBAConfig(dt_h=0.25))
    dfb.step_from_solution({"BIO": 1.0, "EX_S": -2.0}, 5.0)
    dfb.step_from_solution({}, 0.0)
    assert dfb.history


def test_dynamic_fba_gem_o2_signs():
    m = MetabolicModel()
    m.add_reaction(Reaction(
        "EX_glc_e", "glc ex", {"glc_e": -1.0}, lower_bound=-10.0,
        upper_bound=0.0, subsystem="exchange"))
    m.add_reaction(Reaction(
        "EX_o2_e", "o2 ex", {"o2_e": -1.0}, lower_bound=-20.0,
        upper_bound=0.0, subsystem="exchange"))
    m.add_reaction(Reaction(
        "BIOMASS2", "biomass", {"glc_e": -1.0, "o2_e": -1.0, "X": 1.0},
        lower_bound=0.0, upper_bound=1000.0, subsystem="biomass"))
    m.set_biomass("BIOMASS2")
    dfb = DynamicFluxBalance(m, DynamicFBAConfig(dt_h=0.25))
    sol = {"BIOMASS2": 0.8, "EX_glc_e": -1.0, "EX_o2_e": -0.5}
    entry = dfb.step_from_solution(sol, 3.0)
    assert entry["glucose_uptake"] == pytest.approx(1.0)
    assert dfb.byproducts_mm["oxygen"] > 0.0


def test_dynamic_fba_environment_coupling():
    cfg = DynamicFBAConfig(acetate_switch=True, dt_h=0.25,
                           initial_glucose_mm=0.0)
    dfb = DynamicFluxBalance(metab._build_ecoli_core_model(), cfg)
    env = Environment()
    cx = env.config.width // 2
    cy = env.config.height // 2
    env.fields["glucose"].add(cx, cy, 3.0)
    dfb.update_from_environment(env, cx, cy)
    assert dfb.glucose_mm == pytest.approx(4.0)
    dfb.byproducts_mm["acetate"] = 0.7
    dfb.apply_to_environment(env, cx, cy)
    assert "acetate" in env.fields
    dfb.update_from_environment(env, cx, cy)
    assert dfb.byproducts_mm["acetate"] == pytest.approx(0.7)


def test_dynamic_fba_apply_to_environment_existing_field():
    dfb = _dfba_core(dt_h=0.25)
    env = Environment()
    field = ConcentrationField("acetate", env.config.width, env.config.height,
                               ACETATE_DIFFUSION_UM2_S, 0.0)
    env.add_field("acetate", field)
    dfb.apply_to_environment(env)
    assert env.get_field("acetate") is not None


def test_dynamic_fba_to_sim_result_no_growth():
    dfb = _dfba_core(dt_h=0.25, initial_glucose_mm=0.0)
    dfb.step()
    res = dfb.to_simulation_result()
    assert res.doubling_time == 0.0


# ---------------------------------------------------------------------------
# PhotoautotrophicFluxBalance edges
# ---------------------------------------------------------------------------
def _photo_stub_fba():
    m = MetabolicModel()
    m.add_reaction(Reaction(
        "GAPDH", "glyceraldehyde", {"GAP": -1.0, "BPG": 1.0},
        subsystem="calvin"))
    m.add_reaction(Reaction(
        "EX_co2_e", "co2 ex", {"o2": -1.0}, subsystem="exchange"))
    m.add_reaction(Reaction(
        "EX_glc", "glc ex", {"glc": -1.0}, subsystem="exchange"))
    m.add_reaction(Reaction(
        "EX_two", "multi-met exchange", {"A": -1.0, "B": -1.0},
        subsystem="exchange"))
    m.add_reaction(Reaction(
        "PET", "photosynthetic ET", {"H2O": -0.5, "NADPH": 1.0},
        subsystem="pet"))
    m.add_reaction(Reaction(
        "BIO", "photobiomass", {"BPG": -1.0, "B": 1.0}, subsystem="biomass"))
    m.set_biomass("BIO")

    class StubFBA:
        model = m

        def solve(self, *a, **k):
            return {"BIO": 1.0, "EX_co2_e": -2.0}

    return StubFBA()


def test_photoautotrophic_edges():
    cfg = DynamicFBAConfig(dt_h=0.5, co2_initial_mm=0.4, initial_biomass_gdw=1.0)
    pa = PhotoautotrophicFluxBalance(_mini_model(), cfg, fba=_photo_stub_fba())
    fresh = PhotoautotrophicFluxBalance(_mini_model(), cfg,
                                        fba=_photo_stub_fba())
    assert fresh.growth_rate == 0.0
    empty = PhotoautotrophicFluxBalance(_mini_model(), cfg,
                                        fba=_photo_stub_fba())
    assert empty.to_simulation_result().biomass == []
    assert pa._ex_co2 == "EX_co2_e"
    assert 0.0 < pa.light_effect() < 1.0
    entry = pa.step()
    assert entry["growth_rate"] > 0.0
    pa.step()
    assert pa.growth_rate > 0.0
    assert pa.last()["co2"] <= pa.config.co2_initial_mm
    assert pa.to_simulation_result().biomass
    pa.history.append({"time": 0.1, "biomass": 0.1, "growth_rate": 0.0,
                       "co2": 0.0, "co2_uptake": 0.0, "note": "x"})
    assert pa.to_simulation_result()


def test_photoautotrophic_coef_positive_and_missing_rxn():
    def make_model():
        m = MetabolicModel()
        m.add_reaction(Reaction(
            "EX_co2", "co2 ex", {"co2": 1.0}, subsystem="exchange"))
        m.add_reaction(Reaction(
            "BIO", "biomass", {"co2": -1.0, "B": 1.0}, subsystem="biomass"))
        m.set_biomass("BIO")
        return m

    def make_stub(m):
        class Stub:
            model = m

            def solve(self, *a, **k):
                return {"BIO": 1.0, "EX_co2": 2.0}
        return Stub()

    pa_export = PhotoautotrophicFluxBalance(
        make_model(), DynamicFBAConfig(dt_h=0.5), fba=make_stub(make_model()))
    assert pa_export.step()["co2_uptake"] == 2.0

    m_missing = make_model()
    stub_missing = make_stub(m_missing)
    pa = PhotoautotrophicFluxBalance(m_missing, DynamicFBAConfig(dt_h=0.5),
                                     fba=stub_missing)
    pa.model.reactions.pop("EX_co2")
    entry = pa.step()  # EX_co2 rxn removed -> raw flux passthrough
    assert entry["co2_uptake"] == 2.0


def test_photoautotrophic_run_to_exhaustion():
    cfg = DynamicFBAConfig(dt_h=0.25, co2_initial_mm=0.05, initial_biomass_gdw=1.0)
    pa = PhotoautotrophicFluxBalance(_mini_model(), cfg, fba=_photo_stub_fba())
    pa.run(duration_h=20.0)
    assert pa.co2_mm <= 1e-9
    pa2 = PhotoautotrophicFluxBalance(_mini_model(), cfg, fba=_photo_stub_fba())
    pa2.run()
    assert pa2.history
    pa3 = PhotoautotrophicFluxBalance(_mini_model(), cfg, fba=_photo_stub_fba())
    pa3.run(max_steps=0)
    assert pa3.history == []
    pa4 = PhotoautotrophicFluxBalance(_mini_model(), cfg, fba=_photo_stub_fba())
    pa4.run(duration_h=0.0)
    assert pa4.history == []


def test_photoautotrophic_no_co2_rxn():
    m = MetabolicModel()
    m.add_reaction(Reaction("R1", "central", {"A": 1.0}, subsystem="central"))
    m.set_biomass("R1")

    def make_stub(bio_flux):
        class Stub:
            model = m

            def solve(self, *a, **k):
                return {"R1": bio_flux}
        return Stub()

    cfg = DynamicFBAConfig(dt_h=0.5)
    pa = PhotoautotrophicFluxBalance(m, cfg, fba=make_stub(1.0))
    assert pa._ex_co2 == ""
    assert pa.step()["co2_uptake"] == 0.0
    pa2 = PhotoautotrophicFluxBalance(m, cfg, fba=make_stub(0.0))
    pa2.step()
    assert pa2.biomass_gdw == pytest.approx(cfg.initial_biomass_gdw)


# ---------------------------------------------------------------------------
# MetabolicProxy defaults & errors
# ---------------------------------------------------------------------------
def test_metabolic_proxy_default_construction():
    proxy = MetabolicProxy()
    assert proxy.features
    assert metab.ECOLI_CORE_MODEL.biomass_reaction in proxy.outputs


def test_metabolic_proxy_no_exchanges():
    m = MetabolicModel()
    m.add_reaction(Reaction("R1", "central", {"A": 1.0}))
    with pytest.raises(ValueError):
        MetabolicProxy(model=m)


def test_metabolic_proxy_fit_predict():
    proxy = MetabolicProxy(features=["glc"], degree=1, max_uptake=5.0,
                           outputs=["EX_ac", "BIO"])
    proxy.fit(n_samples=20, seed=0)
    pred = proxy.predict([2.0])
    assert set(pred) == set(proxy.outputs)
    assert proxy.predict({"glc": 2.0})["BIO"] == pytest.approx(pred["BIO"])
    assert len(proxy.coeffs) == len(proxy.outputs)
    with pytest.raises(ValueError):
        proxy.predict({"nonsense": 1.0})
    with pytest.raises(ValueError):
        proxy.predict([1.0, 2.0])
    stats = proxy.rmse(n_holdout=3)
    assert set(stats) == set(proxy.outputs)


def test_metabolic_proxy_nn_fallback():
    proxy = MetabolicProxy(features=["glc"], outputs=["BIO", "EX_ac"],
                           degree=1, max_uptake=5.0)
    with pytest.raises(RuntimeError):
        proxy.predict([1.0])
    proxy._train_x = [[1.0], [3.0]]
    proxy._train_y = {"BIO": [0.1, 0.3], "EX_ac": [0.0, 1.0]}
    out = proxy.predict([2.0])
    assert out["BIO"] == 0.1
    assert out["EX_ac"] == 0.0
