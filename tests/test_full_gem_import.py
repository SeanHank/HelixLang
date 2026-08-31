"""Tests for doc/24 — full genome-scale model import."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

cobra = pytest.importorskip("cobra", reason="requires cobra package")
scipy = pytest.importorskip("scipy", reason="requires scipy package")

from helixlang.plugins.gem.bridge import build_functional_model_full  # noqa: E402
from helixlang.plugins.gem.full_model import FullModelAdapter  # noqa: E402
from helixlang.plugins.gem.sbml_import import (  # noqa: E402
    detect_compartments,
    detect_exchange_reactions,
    get_model_info,
    load_bigg_cobra_model,
    load_sbml_model,
)
from helixlang.plugins.runtime.metabolism import _HAS_SCIPY, simplex, solve_lp  # noqa: E402

_CACHE_DIR = Path.home() / ".helixlang" / "gem_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cached_model(bigg_id: str) -> Path:
    """Return a local SBML file for ``bigg_id`` (doc/41).

    Offline-first: the doc/41 loader prefers a vendored copy in
    validation/references/models/; the SBML is cached under ~/.helixlang.
    If the model cannot be obtained (no vendored copy, no network) the test is
    skipped rather than failed — consistent with the release SKIP policy.
    """
    sbml_path = _CACHE_DIR / f"{bigg_id}.xml"
    if sbml_path.exists():
        return sbml_path
    try:
        model = load_bigg_cobra_model(bigg_id)
        cobra.io.write_sbml_model(model, str(sbml_path))
    except Exception as exc:  # noqa: BLE001 - model unavailability → skip
        pytest.skip(f"BiGG model {bigg_id} unavailable: {exc}")
    return sbml_path


# ====================================================================
# Phase A: SBML Import
# ====================================================================

class TestSBMLImport:
    def test_load_bigg_iml1515(self):
        sbml_path = _get_cached_model("iML1515")
        model = load_sbml_model(sbml_path)
        info = get_model_info(model)
        assert info["n_reactions"] > 2000
        assert info["n_metabolites"] > 1500
        assert info["biomass_reaction"] is not None

    def test_gpr_preservation(self):
        sbml_path = _get_cached_model("iML1515")
        model = load_sbml_model(sbml_path, preserve_gpr=True)
        gpr_count = sum(1 for r in model.reactions.values() if r.gene_reaction_rule)
        assert gpr_count > 2000
        assert len(model.genes) > 1000

    def test_gene_registry(self):
        sbml_path = _get_cached_model("iML1515")
        model = load_sbml_model(sbml_path, preserve_gpr=True)
        assert len(model.genes) > 1000

    def test_biomass_detection(self):
        sbml_path = _get_cached_model("iML1515")
        model = load_sbml_model(sbml_path)
        assert model.biomass_reaction is not None
        assert "BIOMASS" in model.biomass_reaction

    def test_exchange_detection(self):
        sbml_path = _get_cached_model("iML1515")
        model = load_sbml_model(sbml_path)
        exchanges = detect_exchange_reactions(model)
        assert len(exchanges) > 100
        assert "EX_glc__D_e" in exchanges

    def test_compartment_detection(self):
        sbml_path = _get_cached_model("iML1515")
        model = load_sbml_model(sbml_path)
        compartments = detect_compartments(model)
        assert "c" in compartments
        assert "e" in compartments

    def test_from_cobra_roundtrip(self):
        sbml_path = _get_cached_model("iML1515")
        model = load_sbml_model(sbml_path)
        assert len(model.reactions) > 2000


# ====================================================================
# Phase B: Solver Dispatch
# ====================================================================

class TestSolverDispatch:
    def test_has_scipy(self):
        assert _HAS_SCIPY

    def test_simplex_small(self):
        c = [1.0, 1.0]
        A = [[1.0, 1.0]]
        b = [1.0]
        bounds = [(0.0, 10.0), (0.0, 10.0)]
        result = simplex(c, A, b, bounds, maximize=True)
        assert result["status"] == "optimal"
        assert abs(result["objective"] - 1.0) < 1e-6

    def test_scipy_solver(self):
        c = [1.0, 1.0]
        A = [[1.0, 1.0]]
        b = [1.0]
        bounds = [(0.0, 10.0), (0.0, 10.0)]
        result = solve_lp(c, A, b, bounds, maximize=True, method="scipy")
        assert result["status"] == "optimal"
        assert abs(result["objective"] - 1.0) < 1e-6

    def test_dispatch_auto(self):
        c = [1.0] * 1000
        A = [[1.0] * 1000]
        b = [500.0]
        bounds = [(0.0, 10.0)] * 1000
        result = solve_lp(c, A, b, bounds, maximize=True, method="auto")
        assert result["status"] == "optimal"
        assert abs(result["objective"] - 500.0) < 1.0

    def test_solver_agreement(self):
        c = [2.0, 3.0]
        A = [[1.0, 2.0]]
        b = [4.0]
        bounds = [(0.0, 10.0), (0.0, 10.0)]
        r1 = simplex(c, A, b, bounds, maximize=True)
        r2 = solve_lp(c, A, b, bounds, maximize=True, method="scipy")
        assert abs(r1["objective"] - r2["objective"]) < 1e-4


# ====================================================================
# Phase C: FullModelAdapter
# ====================================================================

class TestFullModelAdapter:
    @pytest.fixture
    def ecoli_adapter(self):
        sbml_path = _get_cached_model("iML1515")
        return FullModelAdapter.from_sbml(sbml_path, "e_coli_k12")

    def test_exchange_detection(self, ecoli_adapter):
        assert len(ecoli_adapter.exchange_reactions) > 100
        assert "EX_glc__D_e" in ecoli_adapter.exchange_reactions

    def test_transport_detection(self, ecoli_adapter):
        assert len(ecoli_adapter.transport_reactions) > 100

    def test_internal_detection(self, ecoli_adapter):
        assert len(ecoli_adapter.internal_reactions) > 1000

    def test_medium_glucose_minimal(self, ecoli_adapter):
        ecoli_adapter.apply_medium("glucose_minimal")
        ecoli_adapter.solve()
        assert ecoli_adapter.growth_rate > 0.80

    def test_summary(self, ecoli_adapter):
        s = ecoli_adapter.summary()
        assert s["n_reactions"] > 2000
        assert s["n_exchange"] > 100
        assert s["organism"] == "e_coli_k12"

    def test_get_exchange_fluxes(self, ecoli_adapter):
        ecoli_adapter.apply_medium("glucose_minimal")
        fluxes = ecoli_adapter.solve()
        ex = ecoli_adapter.get_exchange_fluxes(fluxes)
        assert "EX_glc__D_e" in ex
        assert ex["EX_glc__D_e"] < -1.0


# ====================================================================
# Phase E: E. coli iML1515 Validation
# ====================================================================

class TestIColiML1515:
    def test_growth_rate(self):
        sbml_path = _get_cached_model("iML1515")
        adapter = FullModelAdapter.from_sbml(sbml_path, "e_coli_k12")
        adapter.apply_medium("glucose_minimal")
        adapter.solve()
        assert adapter.growth_rate > 0.80
        assert adapter.growth_rate < 1.0

    def test_growth_rate_original_bounds(self):
        sbml_path = _get_cached_model("iML1515")
        adapter = FullModelAdapter.from_sbml(sbml_path, "e_coli_k12")
        adapter.solve()
        assert abs(adapter.growth_rate - 0.877) < 0.01

    def test_glucose_uptake(self):
        sbml_path = _get_cached_model("iML1515")
        adapter = FullModelAdapter.from_sbml(sbml_path, "e_coli_k12")
        adapter.apply_medium("glucose_minimal")
        fluxes = adapter.solve()
        ex = adapter.get_exchange_fluxes(fluxes)
        assert abs(ex["EX_glc__D_e"] - (-10.0)) < 0.1

    def test_solve_time(self):
        sbml_path = _get_cached_model("iML1515")
        adapter = FullModelAdapter.from_sbml(sbml_path, "e_coli_k12")
        adapter.apply_medium("glucose_minimal")
        t0 = time.time()
        adapter.solve()
        elapsed = time.time() - t0
        assert elapsed < 5.0

    def test_gpr_rules_present(self):
        sbml_path = _get_cached_model("iML1515")
        adapter = FullModelAdapter.from_sbml(sbml_path, "e_coli_k12")
        gpr_count = sum(1 for r in adapter.model.reactions.values() if r.gene_reaction_rule)
        assert gpr_count > 2000


# ====================================================================
# Phase E: Synechocystis iJN678 Validation
# ====================================================================

class TestSynechocystis:
    def test_photoautotrophic_growth(self):
        sbml_path = _get_cached_model("iJN678")
        adapter = FullModelAdapter.from_sbml(
            sbml_path, "synechocystis_pcc6803",
            biomass_rxn="BIOMASS_Ec_SynAuto",
        )
        adapter.apply_medium("bg11")
        adapter.solve()
        assert adapter.growth_rate > 0.10

    def test_co2_uptake(self):
        sbml_path = _get_cached_model("iJN678")
        adapter = FullModelAdapter.from_sbml(
            sbml_path, "synechocystis_pcc6803",
            biomass_rxn="BIOMASS_Ec_SynAuto",
        )
        adapter.apply_medium("bg11")
        fluxes = adapter.solve()
        ex = adapter.get_exchange_fluxes(fluxes)
        assert ex.get("EX_co2_e", 0) < -0.1


# ====================================================================
# Phase D: build_functional_model_full
# ====================================================================

class TestBuildFunctionalModelFull:
    def test_ecoli(self):
        model = build_functional_model_full(
            "e_coli_k12", "glucose_minimal",
            sbml_path=str(_get_cached_model("iML1515")),
        )
        assert model._growth_rate > 0.80  # type: ignore[attr-defined]
        assert len(model._fba_fluxes) > 2000  # type: ignore[attr-defined]
        assert len(model.genes) > 1000

    def test_has_adapter(self):
        model = build_functional_model_full(
            "e_coli_k12", "glucose_minimal",
            sbml_path=str(_get_cached_model("iML1515")),
        )
        assert hasattr(model, "_adapter")
        assert model._adapter.summary()["organism"] == "e_coli_k12"  # type: ignore[attr-defined]
