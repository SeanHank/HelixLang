"""Branch-completion tests for helixlang.plugins.human.simulation."""
from __future__ import annotations

import copy
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from helixlang.plugins.human.disease import (
    DiseaseState,
    MetabolitePerturbation,
)
from helixlang.plugins.human.drug import (
    BIOLOGIC,
    INTRAMUSCULAR,
    INTRATHECAL,
    IV,
    IV_INFUSION,
    OLIGONUCLEOTIDE,
    ORAL,
    SMALL_MOLECULE,
    SUBCUTANEOUS,
    Drug,
    DrugMolecule,
)
from helixlang.plugins.human.physiology import create_default_physiology
from helixlang.plugins.human.simulation import (
    HumanSimulation,
    HumanSimulationConfig,
    _hill_response,
    _PBPKEngine,
    _tissue_access_factor,
    _trapz,
)
from helixlang.plugins.runtime.metabolism import ECOLI_CORE_MODEL


def _load_blocking_imports(module_path: str, blocked: set[str]):
    import builtins

    spec = importlib.util.find_spec(module_path)
    throwaway = importlib.util.spec_from_file_location(
        "_no_" + module_path.rsplit(".", 1)[-1], spec.origin)
    old = sys.modules.get(throwaway.name)
    sys.modules[throwaway.name] = mod = importlib.util.module_from_spec(throwaway)
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in blocked:
            raise ImportError(f"{name} blocked")
        return real_import(name, *a, **k)

    builtins.__import__ = fake_import
    try:
        throwaway.loader.exec_module(mod)  # type: ignore[union-attr]
    finally:
        builtins.__import__ = real_import
        if old is None:
            sys.modules.pop(throwaway.name, None)
        else:
            sys.modules[throwaway.name] = old
    return mod


def _drug(name: str, route: str = ORAL, mw: float = 300.0,
          log_p: float = 1.0,
          drug_type: str = SMALL_MOLECULE, **kw) -> Drug:
    defaults = dict(
        route=route,
        dose_mg=1000.0, dosing_interval_h=24.0, duration_days=1.0,
        bioavailability=1.0, half_life_h=6.0, renal_fraction=0.5,
    )
    defaults.update(kw)
    return Drug(
        molecule=DrugMolecule(
            name=name, drug_type=drug_type, molecular_weight_da=mw,
            log_p=log_p, binding_affinity_kd_um=1.0,
        ),
        **defaults,
    )


def _sim(drugs, **over) -> HumanSimulation:
    base = dict(
        drugs=drugs,
        total_duration_days=1.0,
        dfa_dt_h=1.0,
        pbpk_dt_min=1.0,
        output_time_resolution_h=1.0,
        track_fluxes=False,
    )
    base.update(over)
    return HumanSimulation(HumanSimulationConfig(**base))


def _engine(drug=None, phys=None, dt_min=1.0) -> _PBPKEngine:
    if drug is None:
        drug = _drug("engine_drug")
    if phys is None:
        phys = create_default_physiology()
    return _PBPKEngine(drug, phys, dt_min)


class TestImportFallbacks:
    def test_scipy_and_pd_fallbacks(self):
        mod = _load_blocking_imports(
            "helixlang.plugins.human.simulation",
            {"scipy.integrate", "helixlang.plugins.human.pharmacodynamics",
             "helixlang.plugins.human.pharmacokinetics"},
        )
        assert mod._HAS_SCIPY is False
        assert mod.solve_ivp is None
        assert mod.PBPKConfig.__module__.endswith("_no_simulation")
        cfg = mod.PBPKConfig()
        assert cfg.dt_min == 1.0
        assert cfg.total_time_h == 24.0
        assert cfg.n_compartments == 6
        effect = mod.PDEffect(target_reaction="R")
        assert effect.ec50_um == 1.0
        pd = mod.Pharmacodynamics(drug_name="x")
        assert pd.toxicity_concentration_um == 100.0
        assert pd.dose_response_model == "hill"

    def test_numpy_fallback(self):
        mod = _load_blocking_imports(
            "helixlang.plugins.human.simulation", {"numpy"})
        assert mod._HAS_NUMPY is False
        assert mod._np is None


class TestHelpers:
    def test_hill_response_zero_ec50(self):
        assert _hill_response(5.0, 0.0, 1.0) == 1.0
        assert _hill_response(0.0, 0.0, 1.0) == 0.0

    def test_hill_response_zero_conc(self):
        assert _hill_response(0.0, 1.0, 1.0) == 0.0
        assert 0.0 < _hill_response(2.0, 1.0, 1.0) < 1.0

    def test_trapz_short(self):
        assert _trapz([], 1.0) == 0.0
        assert _trapz([1.0], 1.0) == 0.0

    def test_tissue_access_factor(self):
        assert _tissue_access_factor(
            _drug("bio", mw=200.0, drug_type=BIOLOGIC)) == 0.12
        assert _tissue_access_factor(_drug("large", mw=40000.0)) == 0.12
        assert _tissue_access_factor(
            _drug("oligo", mw=2000.0, drug_type=OLIGONUCLEOTIDE)) == 0.25
        assert _tissue_access_factor(_drug("mid", mw=10000.0)) == 0.25
        assert _tissue_access_factor(_drug("small")) == 1.0


class TestPBPKEngineRoutes:
    def test_apply_due_doses_break_at_horizon(self):
        engine = _engine(_drug(
            "capped", dosing_interval_h=0.5, duration_days=0.02,
            route=SUBCUTANEOUS))
        engine.apply_due_doses(100.0)
        assert engine.depot_umol > 0.0

    def test_iv_bolus(self):
        engine = _engine(_drug("iv", route=IV))
        engine.apply_due_doses(0.0)
        assert engine.conc_um["central"] > 0.0

    def test_iv_infusion(self):
        engine = _engine(_drug("inf", route=IV_INFUSION))
        engine.apply_due_doses(0.0)
        assert engine.infusion_rate_umol_h > 0.0
        assert engine.infusion_end_h > engine.time_h

    def test_intrathecal(self):
        engine = _engine(_drug("it", route=INTRATHECAL, mw=400.0))
        engine.apply_due_doses(0.0)
        assert engine.conc_um["brain"] > 0.0

    def test_unknown_route_falls_through_chain(self):
        engine = _engine(_drug("weird", route="nasal"))
        engine.apply_due_doses(0.0)
        assert engine.depot_umol == 0.0
        assert engine.conc_um["central"] == 0.0

    def test_oral_uses_depot(self):
        engine = _engine(_drug("oral", route=ORAL))
        engine.apply_due_doses(0.0)
        assert engine.depot_umol > 0.0

    def test_intramuscular_uses_depot(self):
        engine = _engine(_drug("im", route=INTRAMUSCULAR))
        engine.apply_due_doses(0.0)
        assert engine.depot_umol > 0.0

    def test_partial_organs_skip_missing(self):
        from helixlang.plugins.human.simulation import _TISSUES
        phys = create_default_physiology()
        del phys.organs["liver"]
        del phys.organs["brain"]
        engine = _engine(drug=_drug("partial"), phys=phys)
        present = {t for t in _TISSUES if t in engine.flows_l_h}
        assert present == {"kidney", "muscle", "adipose"}


def math_isfinite(x: float) -> bool:
    import math
    return math.isfinite(x)


class TestPBPKEngineAdvance:
    def test_euler_fallback_without_scipy(self, monkeypatch):
        import helixlang.plugins.human.simulation as sim
        monkeypatch.setattr(sim, "_HAS_SCIPY", False)
        engine = _engine(_drug("eul", route=IV))
        engine.apply_due_doses(0.0)
        engine.advance(1.0)
        assert engine.time_h == pytest.approx(1.0)
        assert math_isfinite(engine.conc_um["central"])

    def test_euler_fallback_solver_failure(self, monkeypatch):
        import helixlang.plugins.human.simulation as sim
        monkeypatch.setattr(
            sim, "solve_ivp",
            lambda *a, **k: SimpleNamespace(success=False))
        engine = _engine(_drug("fail", route=IV))
        engine.apply_due_doses(0.0)
        engine.advance(1.0)
        assert engine.time_h == pytest.approx(1.0)

    def test_scalar_derivatives_np(self):
        engine = _engine(_drug("np", route=IV))
        keys = engine._order()
        y = np.array([1.0] * len(keys) + [2.0], dtype=float)
        out = engine._derivatives_np(0.5, y, np)
        assert out.shape == (len(keys) + 1,)

    def test_target_concentration_unknown_tissue(self):
        engine = _engine(_drug("tgt"))
        assert engine.target_concentration("plasma") == 0.0
        assert engine.target_concentration("nonsense") == 0.0

    def test_advance_batch_requires_numpy(self, monkeypatch):
        import helixlang.plugins.human.simulation as sim
        monkeypatch.setattr(sim, "_HAS_NUMPY", False)
        a = _engine(_drug("a", route=IV))
        b = _engine(_drug("b", route=IV))
        assert a.advance_batch(1.0, [b]) is False

    def test_advance_batch_requires_scipy(self, monkeypatch):
        import helixlang.plugins.human.simulation as sim
        monkeypatch.setattr(sim, "_HAS_SCIPY", False)
        a = _engine(_drug("a", route=IV))
        b = _engine(_drug("b", route=IV))
        assert a.advance_batch(1.0, [b]) is False

    def test_advance_batch_desync_falls_back(self):
        a = _engine(_drug("a", route=IV))
        b = _engine(_drug("b", route=IV))
        a.advance(0.5)
        assert a.advance_batch(1.0, [b]) is False

    def test_advance_batch_solver_failure(self, monkeypatch):
        import helixlang.plugins.human.simulation as sim
        monkeypatch.setattr(
            sim, "solve_ivp",
            lambda *a, **k: SimpleNamespace(success=False))
        a = _engine(_drug("a", route=IV))
        b = _engine(_drug("b", route=IV))
        assert a.advance_batch(1.0, [b]) is False


class TestHumanSimulationBuild:
    def test_pd_none_becomes_empty(self):
        config = HumanSimulationConfig(drugs=[], pharmacodynamics=None)
        sim = HumanSimulation(config)
        assert sim.config.pharmacodynamics == {}

    def test_invalid_drug_rejected(self):
        with pytest.raises(ValueError, match="invalid drug"):
            HumanSimulation(
                HumanSimulationConfig(
                    drugs=[_drug("bad", route="nasal")]))

    def test_nonpositive_dfa_dt_rejected(self):
        with pytest.raises(ValueError, match="dfa_dt_h"):
            HumanSimulation(
                HumanSimulationConfig(
                    drugs=[], dfa_dt_h=0.0))

    def test_nonpositive_pbpk_dt_rejected(self):
        with pytest.raises(ValueError, match="pbpk_dt_min"):
            HumanSimulation(
                HumanSimulationConfig(
                    drugs=[], pbpk_dt_min=-1.0))

    def test_pd_for_case_insensitive(self):
        from helixlang.plugins.human.pharmacodynamics import (
            PDEffect,
            Pharmacodynamics,
        )
        sim = _sim([])
        sim.config.pharmacodynamics = {
            "Metformin": Pharmacodynamics(drug_name="Metformin", effects=[
                PDEffect(target_reaction="PAH")])}
        assert sim._pd_for("metformin").drug_name == "Metformin"
        assert sim._pd_for("aspirin") is None

    def test_engine_for_unknown_returns_dummy(self):
        sim = _sim([])
        engine = sim._engine_for(_drug("ghost"))
        assert engine.conc_um["central"] == 0.0


class TestLoadBaseModel:
    def test_missing_path_raises_model_missing(self, monkeypatch):
        monkeypatch.delenv("HELIX_ALLOW_LOW_FIDELITY", raising=False)
        from helixlang.core.errors import ModelMissingError
        sim = _sim([])
        sim.config.base_model_path = "/nonexistent/model.json"
        with pytest.raises(ModelMissingError):
            sim._load_base_model()

    def test_missing_path_opt_in_falls_back(self):
        monkeypatch_module = pytest.MonkeyPatch()
        monkeypatch_module.setattr(
            "helixlang.api.capabilities.opt_in", lambda *a, **k: True)
        sim = _sim([])
        sim.config.base_model_path = "/nonexistent/model.json"
        try:
            model = sim._load_base_model()
        finally:
            monkeypatch_module.undo()
        assert model is not None

    def test_load_success_returns_model(self, monkeypatch):
        import helixlang.plugins.human.simulation as sim_mod
        monkeypatch.setattr(
            sim_mod, "load_model_from_json",
            lambda path: copy.deepcopy(ECOLI_CORE_MODEL))
        sim = _sim([])
        sim.config.base_model_path = "/any/model.json"
        model = sim._load_base_model()
        assert model.reactions

    def test_load_success_without_biomass(self, monkeypatch):
        import helixlang.plugins.human.simulation as sim_mod

        class FakeModel:
            biomass_reaction = None
            reactions = {
                "T1": SimpleNamespace(name="transport"),
                "BIOMASS_ecli": SimpleNamespace(name="synthesis reaction"),
            }

            def set_biomass(self, rxn_id):
                self.biomass_reaction = rxn_id

        fake = FakeModel()
        monkeypatch.setattr(
            sim_mod, "load_model_from_json", lambda path: fake)
        sim = _sim([])
        sim.config.base_model_path = "/any/model.json"
        assert sim._load_base_model().biomass_reaction == "BIOMASS_ecli"

    def test_ensure_biomass_no_match(self):
        from helixlang.plugins.human.simulation import (
            HumanSimulation,
        )
        sim = _sim([])
        sim.model = SimpleNamespace(
            biomass_reaction=None,
            reactions={
                "T1": SimpleNamespace(name="transport"),
                "GLK": SimpleNamespace(name="glucokinase"),
            },
        )
        HumanSimulation._ensure_biomass(sim.model)
        assert sim.model.biomass_reaction is None

    def test_build_dfba_exception_returns_none(self, monkeypatch):
        import helixlang.plugins.human.simulation as sim_mod

        def boom(*a, **k):
            raise RuntimeError("no LP available")

        monkeypatch.setattr(sim_mod, "DynamicFluxBalance", boom)
        sim = _sim([])
        assert sim._build_dfba() is None


class TestPdMechanisms:
    def _pd(self, effects):
        from helixlang.plugins.human.pharmacodynamics import Pharmacodynamics
        return Pharmacodynamics(drug_name="metformin", effects=effects)

    def test_compute_effect_skips_missing_rxn(self):
        from helixlang.plugins.human.pharmacodynamics import PDEffect
        sim = _sim([])
        sim.config.pharmacodynamics = {"metformin": self._pd([
            PDEffect(target_reaction="NOT_A_REACTION")])}
        assert sim._compute_pd_effect("metformin", {"liver": 10.0}, 1.0) == {}

    def test_compute_effect_activation_then_inhibition(self):
        sim = _sim([])
        pd = Pharmacodynamics_effects_two()
        sim.config.pharmacodynamics = {"metformin": pd}
        out = sim._compute_pd_effect("metformin", {"liver": 100.0}, 1.0)
        assert out["BIOMASS"][0] == "inhibition"
        assert 0.0 < out["BIOMASS"][1] <= 1.0

    def test_apply_pd_bounds_both_kinds(self):
        from helixlang.plugins.human.pharmacodynamics import (
            PDEffect,
            Pharmacodynamics,
        )
        sim = _sim([])
        rxn_id = "BIOMASS"
        healthy = sim._healthy_bounds[rxn_id]
        # build a pd whose two effects target different reactions to
        # exercise both the inhibition and activation branches
        sim.config.pharmacodynamics = {
            "drug_x": Pharmacodynamics(drug_name="drug_x", effects=[
                PDEffect(target_reaction=rxn_id, effect_type="inhibition",
                         ec50_um=1.0, emax=1.0),
            ])}
        sim._apply_pd_bounds(0.0, {"drug_x": {"liver": 100.0}})
        assert sim._pd_bound_overrides[rxn_id] <= healthy[1]

        sim.config.pharmacodynamics = {
            "drug_y": Pharmacodynamics(drug_name="drug_y", effects=[
                PDEffect(target_reaction=rxn_id, effect_type="activation",
                         ec50_um=1.0, emax=1.0),
            ])}
        sim._apply_pd_bounds(0.0, {"drug_y": {"liver": 100.0, "central": 0.0}})
        assert sim._pd_bound_overrides[rxn_id] > healthy[1] * 0.99


class TestDiseaseBiomarkers:
    def _glc_disease(self):
        return DiseaseState(
            name="glc test",
            category="metabolic_overload",
            metabolite_perturbations=[
                MetabolitePerturbation(
                    metabolite_id="GLC", perturbation_type="accumulate",
                    initial_concentration_mm=20.0,
                    normal_concentration_mm=5.0,
                )],
            severity=0.5,
        )

    def test_init_biomarker_direct_pool(self):
        sim = _sim([], disease=self._glc_disease())
        state = sim._init_biomarkers()
        assert "GLC" in state
        assert state["GLC"]["pool_key"] == "GLC"

    def test_biomarker_pool_following_update(self):
        sim = _sim([], disease=self._glc_disease())
        state = sim._init_biomarkers()
        sim._pool_start = dict(sim.pools.pools)
        sim.pools.pools["GLC"] = 5.0
        sim._pool_start["GLC"] = 4.0
        before = state["GLC"]["value"]
        sim._update_biomarkers(state, strength=0.1, dt_h=1.0)
        assert state["GLC"]["value"] == pytest.approx(before * 5.0 / 4.0)

    def test_biomarker_non_pool_uses_zero_start(self):
        sim = _sim([], disease=self._glc_disease())
        state = sim._init_biomarkers()
        sim._pool_start = dict(sim.pools.pools)
        sim._pool_start["GLC"] = 0.0
        sim._update_biomarkers(state, strength=1.0, dt_h=1.0)
        assert math_isfinite(state["GLC"]["value"])

    def test_biomarker_exponential_update_no_pool(self):
        disease = DiseaseState(
            name="orphan",
            category="metabolic_overload",
            metabolite_perturbations=[
                MetabolitePerturbation(
                    metabolite_id="orphan_met", perturbation_type="accumulate",
                    initial_concentration_mm=10.0,
                    normal_concentration_mm=1.0,
                )],
            severity=1.0,
        )
        sim = _sim([], disease=disease)
        state = sim._init_biomarkers()
        info = state["orphan_met"]
        sim._update_biomarkers(state, strength=1.0, dt_h=1.0)
        assert abs(info["value"] - 10.0) < 10.0

    def test_normalization_zero_span(self):
        sim = _sim([])
        info = {"pathological": 2.0, "normal": 2.0, "value": 2.0}
        assert sim._normalization(info) == 0.0

    def test_perfuse_restores_oxygen(self):
        sim = _sim([])
        assert sim.dfba is not None
        if "oxygen" not in sim.dfba.byproducts_mm:
            sim.dfba.byproducts_mm["oxygen"] = 5.0
        sim.dfba.byproducts_mm["lactate"] = 3.0
        sim.dfba.byproducts_mm["acetate"] = 2.0
        sim._perfuse()
        assert sim.dfba.byproducts_mm["oxygen"] == pytest.approx(20.0)
        assert sim.dfba.byproducts_mm["lactate"] < 3.0

    def test_perfuse_without_oxygen(self):
        sim = _sim([])
        assert sim.dfba is not None
        sim.dfba.byproducts_mm.pop("oxygen", None)
        sim._perfuse()
        assert "oxygen" not in sim.dfba.byproducts_mm

    def test_check_response_records_time(self):
        sim = _sim([])
        biomarkers = {
            "m": {"pathological": 10.0, "normal": 0.0, "value": 2.0},
        }
        result = _make_result()
        seen: set[str] = set()
        sim._check_response(result, biomarkers, seen, 5.0)
        assert result.therapeutic_response_time_h == 5.0
        sim._check_response(result, biomarkers, seen, 7.0)
        assert result.therapeutic_response_time_h == 5.0
        result.therapeutic_response_time_h = -1.0
        sim._check_response(result, biomarkers, seen, 9.0)
        assert result.therapeutic_response_time_h == -1.0

    def test_check_response_marks_seen_resets_record_time(self):
        sim = _sim([])
        biomarkers = {
            "m": {"pathological": 10.0, "normal": 0.0, "value": 2.0},
        }
        result = _make_result()
        result.therapeutic_response_time_h = 3.0
        sim._check_response(result, biomarkers, set(), 5.0)
        assert result.therapeutic_response_time_h == 3.0

    def test_check_response_not_normalized(self):
        sim = _sim([])
        result = _make_result()
        biomarkers = {
            "m": {"pathological": 10.0, "normal": 0.0, "value": 9.0},
        }
        sim._check_response(result, biomarkers, set(), 5.0)
        assert result.therapeutic_response_time_h == -1.0

    def test_count_therapeutic_range_variants(self):
        from helixlang.plugins.human.pharmacodynamics import (
            Pharmacodynamics,
        )
        sim = _sim([], pharmacodynamics={})
        wc = Pharmacodynamics(
            drug_name="c", therapeutic_window=(1.0, 50.0))
        assert sim._count_therapeutic_range([]) == (0, 0)
        assert sim._count_therapeutic_range(
            [(None, None)]) == (0, 0)
        assert sim._count_therapeutic_range(
            [(_drug("c", route=IV), wc)])[0] == 1

    def test_finalize_without_biomarkers(self):
        sim = _sim([_drug("f", route=IV)])
        result = sim.run()
        assert result.time_in_therapeutic_range_fraction >= 0.0
        assert result.overall_efficacy_score >= 0.0


def _make_result():
    from helixlang.plugins.human.simulation import HumanSimulationResult
    return HumanSimulationResult()


def _pd_no_window():
    from helixlang.plugins.human.pharmacodynamics import Pharmacodynamics
    return Pharmacodynamics(drug_name="nw")


class TestRunVariants:
    def test_run_tracks_fluxes(self):
        sim = _sim([_drug("met", route=IV)], track_fluxes=True)
        result = sim.run()
        assert result.time_h
        assert result.flux_history

    def test_run_no_metabolite_tracking(self):
        sim = _sim([_drug("met", route=IV)], track_metabolites=False)
        result = sim.run()
        assert result.time_h
        assert result.metabolite_pools == {}

    def test_run_without_dfba(self, monkeypatch):
        import helixlang.plugins.human.simulation as sim_mod

        def boom(*a, **k):
            raise RuntimeError("no LP available")

        monkeypatch.setattr(sim_mod, "DynamicFluxBalance", boom)
        sim = _sim([_drug("met", route=IV)])
        assert sim.dfba is None
        result = sim.run()
        assert result.overall_efficacy_score >= 0.0

    def test_run_multi_drug_batch(self):
        sim = _sim([_drug("a", route=IV), _drug("b", route=IV)],
                   pharmacodynamics={})
        result = sim.run()
        assert len(result.drug_concentrations) == 2


class TestResultOutput:
    def test_save_csv(self, tmp_path):
        from helixlang.plugins.human.simulation import HumanSimulationResult
        result = HumanSimulationResult()
        result.time_h = [0.0, 1.0, 2.0]
        result.plasma_concentration = [1.0, 2.0, 3.0]
        result.drug_concentrations = {"a": [1.0, 2.0, 3.0]}
        result.biomarker_history = {"bg": [5.0, 5.0, 6.0]}
        out = tmp_path / "sim.csv"
        result.save_csv(str(out))
        assert out.exists()

    def test_record_flux_with_track(self):
        sim = _sim([_drug("r", route=IV)], track_fluxes=True)
        result = _make_result()
        biomarkers = {
            "m": {"pathological": 10.0, "normal": 0.0, "value": 5.0,
                  "pool_key": None},
        }
        sim._record(
            result, 1.0, biomarkers,
            {"r": sim.engines[0].concentrations()},
            {"GLC": 3.0}, 0)
        assert result.flux_history
        assert "GLC" in result.metabolite_pools
        assert "m" in result.biomarker_history

    def test_record_empty_pools_track_metabolites(self):
        sim = _sim([_drug("r", route=IV)], track_metabolites=True)
        sim.pools.pools = {}
        result = _make_result()
        sim._record(
            result, 1.0, {},
            {"r": sim.engines[0].concentrations()},
            {}, 0)
        assert result.time_h == [1.0]

    def test_record_biomarkers_off(self):
        sim = _sim([_drug("r", route=IV)], track_metabolites=False,
                   track_biomarkers=False)
        result = _make_result()
        sim._record(
            result, 1.0, {},
            {"r": sim.engines[0].concentrations()},
            {}, 0)
        assert result.time_h == [1.0]

    def test_record_pads_series(self):
        sim = _sim([_drug("r", route=IV)], track_metabolites=True)
        result = _make_result()
        result.drug_concentrations.setdefault("r", [1.0])
        sim._record(
            result, 1.0, {},
            {"r": sim.engines[0].concentrations()},
            {}, 4)
        assert result.drug_concentrations["r"] == [1.0, 0.0, 0.0, 0.0,
                                                   pytest.approx(0.0)]

    def test_record_pads_biomarker_history(self):
        sim = _sim([_drug("r", route=IV)])
        result = _make_result()
        result.biomarker_history.setdefault("x", [])
        result.biomarker_history["x"].extend([1.0] * 3)
        sim._record(
            result, 1.0, {"x": {"value": 2.0}},
            {"r": sim.engines[0].concentrations()},
            {}, 5)
        assert result.biomarker_history["x"][-1] == 2.0


def Pharmacodynamics_effects_two():
    from helixlang.plugins.human.pharmacodynamics import (
        PDEffect,
        Pharmacodynamics,
    )
    return Pharmacodynamics(drug_name="metformin", effects=[
        PDEffect(target_reaction="BIOMASS", effect_type="activation",
                 ec50_um=1.0, emax=1.0),
        PDEffect(target_reaction="BIOMASS", effect_type="inhibition",
                 ec50_um=1.0, emax=1.0),
    ])
