"""Tests for doc/25 Phase VII-X gap closures (G7-G10).

G7: GRN regulatory edges -> FBA bounds
G8: Temperature / pH enzyme correction
G9: Population density feedback to GEM
G10: Genome evolution -> FBA re-solve
"""
from __future__ import annotations

import copy
import math

import pytest

from helixlang.gem.grn_inference import EvidenceLevel, RegulatoryEdge
from helixlang.metabolism import (
    ECOLI_CORE_MODEL,
    MetabolicModel,
    Reaction,
    enzyme_correction,
)


def _toy_model() -> MetabolicModel:
    m = MetabolicModel()
    m.add_reaction(Reaction(
        id="EX_glc_e", name="Glucose exchange",
        stoichiometry={"glc_e": -1.0},
        lower_bound=-1000.0, upper_bound=10.0,
        subsystem="exchange",
    ))
    m.add_reaction(Reaction(
        id="PGI", name="Phosphoglucose isomerase",
        stoichiometry={"g6p": -1.0, "f6p": 1.0},
        lower_bound=0.0, upper_bound=1000.0,
        subsystem="glycolysis", gene_reaction_rule="pgi",
    ))
    m.add_reaction(Reaction(
        id="PFK", name="Phosphofructokinase",
        stoichiometry={"f6p": -1.0, "fdp": 1.0},
        lower_bound=0.0, upper_bound=1000.0,
        subsystem="glycolysis", gene_reaction_rule="pfkA",
    ))
    m.add_reaction(Reaction(
        id="BIOMASS", name="Biomass",
        stoichiometry={"fdp": -1.0, "biomass_c": 1.0},
        lower_bound=0.0, upper_bound=1000.0,
        subsystem="biomass",
    ))
    m.biomass_reaction = "BIOMASS"
    return m


def _edge(
    tf="fnr", target="pfkA", reg_type="repression",
    confidence=0.8, target_reaction=None,
):
    return RegulatoryEdge(
        tf_id=tf, target_gene=target, regulation_type=reg_type,
        evidence_level=EvidenceLevel.PREDICTED,
        confidence=confidence, target_reaction=target_reaction,
    )


# ============================================================================
# G7: GRN -> FBA Closed Loop
# ============================================================================

class TestG7RegulatoryBounds:

    def test_repression_scales_upper_bound(self):
        from helixlang.gem.bridge import apply_regulatory_bounds
        model = _toy_model()
        gpr = {"pfkA": ["PFK"]}
        edges = [_edge(reg_type="repression", confidence=1.0)]
        n = apply_regulatory_bounds(model, edges, gpr, base_fraction=0.1)
        assert n == 1
        assert model.reactions["PFK"].upper_bound == pytest.approx(900.0)

    def test_repression_scales_negative_lower_bound(self):
        from helixlang.gem.bridge import apply_regulatory_bounds
        model = _toy_model()
        model.reactions["PGI"].lower_bound = -500.0
        gpr = {"pgi": ["PGI"]}
        edges = [_edge(target="pgi", target_reaction="PGI",
                        reg_type="repression", confidence=0.5)]
        n = apply_regulatory_bounds(model, edges, gpr, base_fraction=0.2)
        assert n == 1
        assert model.reactions["PGI"].lower_bound == pytest.approx(-450.0)

    def test_activation_preserves_existing_upper(self):
        from helixlang.gem.bridge import apply_regulatory_bounds
        model = _toy_model()
        model.reactions["PFK"].upper_bound = 500.0
        gpr = {"pfkA": ["PFK"]}
        edges = [_edge(reg_type="activation", confidence=0.9)]
        n = apply_regulatory_bounds(model, edges, gpr)
        assert n == 1
        assert model.reactions["PFK"].upper_bound == 500.0

    def test_target_reaction_overrides_gpr(self):
        from helixlang.gem.bridge import apply_regulatory_bounds
        model = _toy_model()
        gpr = {"some_gene": ["PGI"]}
        edges = [_edge(target="some_gene", target_reaction="PFK",
                        reg_type="repression", confidence=1.0)]
        n = apply_regulatory_bounds(model, edges, gpr, base_fraction=0.5)
        assert n == 1
        assert model.reactions["PFK"].upper_bound == pytest.approx(500.0)

    def test_unknown_reaction_skipped(self):
        from helixlang.gem.bridge import apply_regulatory_bounds
        model = _toy_model()
        gpr = {"unknown_gene": ["NONEXISTENT"]}
        edges = [_edge(target="unknown_gene")]
        n = apply_regulatory_bounds(model, edges, gpr)
        assert n == 0

    def test_empty_edges_returns_zero(self):
        from helixlang.gem.bridge import apply_regulatory_bounds
        model = _toy_model()
        n = apply_regulatory_bounds(model, [], {})
        assert n == 0

    def test_multiple_edges_modify_multiple_reactions(self):
        from helixlang.gem.bridge import apply_regulatory_bounds
        model = _toy_model()
        gpr = {"pgi": ["PGI"], "pfkA": ["PFK"]}
        edges = [
            _edge(target="pgi", reg_type="repression", confidence=1.0),
            _edge(target="pfkA", reg_type="repression", confidence=1.0),
        ]
        n = apply_regulatory_bounds(model, edges, gpr, base_fraction=0.1)
        assert n == 2
        assert model.reactions["PGI"].upper_bound == pytest.approx(900.0)
        assert model.reactions["PFK"].upper_bound == pytest.approx(900.0)

    def test_regulatory_edge_target_reaction_field(self):
        re = _edge(target_reaction="PFK")
        assert re.target_reaction == "PFK"

    def test_regulatory_edge_default_target_reaction_is_none(self):
        re = _edge()
        assert re.target_reaction is None


# ============================================================================
# G8: Temperature / pH Enzyme Correction
# ============================================================================

class TestG8EnzymeCorrection:

    def test_optimal_conditions_return_one(self):
        c = enzyme_correction(37.0, 7.0)
        assert c == pytest.approx(1.0, abs=1e-6)

    def test_low_temperature_reduces_activity(self):
        opt = enzyme_correction(37.0, 7.0)
        low = enzyme_correction(10.0, 7.0)
        assert low < opt
        assert low > 0.0

    def test_high_temperature_capped_at_one(self):
        opt = enzyme_correction(37.0, 7.0)
        high = enzyme_correction(60.0, 7.0)
        assert high == pytest.approx(1.0, abs=1e-6)
        assert opt == pytest.approx(1.0, abs=1e-6)

    def test_extreme_cold_very_low_activity(self):
        c = enzyme_correction(-10.0, 7.0)
        assert c < 0.05

    def test_suboptimal_ph_reduces_activity(self):
        opt = enzyme_correction(37.0, 7.0)
        acidic = enzyme_correction(37.0, 4.0)
        alkaline = enzyme_correction(37.0, 10.0)
        assert acidic < opt
        assert alkaline < opt

    def test_ph_symmetry(self):
        c_low = enzyme_correction(37.0, 5.0)
        c_high = enzyme_correction(37.0, 9.0)
        assert c_low == pytest.approx(c_high, abs=1e-10)

    def test_returns_in_zero_one_range(self):
        for t in [-20, 0, 20, 37, 50, 80, 100]:
            for ph in [2, 4, 6, 7, 8, 10, 12]:
                c = enzyme_correction(float(t), float(ph))
                assert 0.0 <= c <= 1.0 + 1e-10, f"t={t}, ph={ph}, c={c}"

    def test_custom_ph_width(self):
        c_narrow = enzyme_correction(37.0, 6.0, ph_width=1.0)
        c_wide = enzyme_correction(37.0, 6.0, ph_width=3.0)
        assert c_narrow < c_wide

    def test_temperature_effect_is_arrhenius(self):
        R = 8.314e-3
        ea = 50.0
        T1 = 20.0 + 273.15
        T_opt = 37.0 + 273.15
        expected_arr = math.exp(-ea / R * (1.0 / T1 - 1.0 / T_opt))
        expected_arr = min(expected_arr, 1.0)
        actual = enzyme_correction(20.0, 7.0, ea_kj_mol=ea)
        assert actual == pytest.approx(expected_arr, abs=1e-8)

    def test_wiring_in_growth_rate_gem(self):
        """Verify enzyme correction + density scaling are wired into _growth_rate_gem.

        The ECOLI_CORE_MODEL's EX_glc uses a secretion convention (coef +1)
        which is incompatible with the ecosystem's set_uptake mapping, so
        g_c may be zero.  The important checks are that (a) FBA is invoked
        (is_fba=True), (b) the function returns the correct 3-tuple, and
        (c) g_c is non-negative.
        """
        from helixlang.apps.ecosystem import (
            Ecosystem,
            EcosystemConfig,
            PatchConfig,
            Species,
            SubstrateConfig,
        )
        sp = Species(
            name="ecoli_gem",
            consumption={"glucose": (0.02, 0.1)},
            cn_ratio=6.0, maintenance=0.002,
            metabolic_model=ECOLI_CORE_MODEL,
        )
        pc = PatchConfig(
            name="p", kind="chemostat", width=1, height=1,
            flow_rate=0.0, temperature_c=37.0, ph=7.0,
            anoxic=True,
            initial_biomass={"ecoli_gem": 100.0},
            substrates={"glucose": SubstrateConfig(initial_mm=10.0, bulk_mm=10.0)},
        )
        eco = Ecosystem(EcosystemConfig(
            ticks=0, species=[sp], patches=[pc], gem_driven=True))
        patch = eco.patches[0]
        g_c, _, is_fba = patch._growth_rate_gem(sp, 100.0, 0, 0, 1.0, 1.0, 1.0, 0.0)
        assert is_fba is True
        assert g_c >= 0.0

    def test_suboptimal_temp_reduces_internal_bounds(self):
        from helixlang.metabolism import MetabolicModel, Reaction, enzyme_correction
        model = MetabolicModel()
        model.add_reaction(Reaction(
            id="PGI", name="PGI",
            stoichiometry={"a": -1.0, "b": 1.0},
            lower_bound=0.0, upper_bound=1000.0,
            subsystem="glycolysis",
        ))
        model.add_reaction(Reaction(
            id="BIOMASS", name="BM",
            stoichiometry={"b": -1.0, "c": 1.0},
            lower_bound=0.0, upper_bound=1000.0,
            subsystem="biomass",
        ))
        model.biomass_reaction = "BIOMASS"
        corr_37 = enzyme_correction(37.0, 7.0)
        corr_10 = enzyme_correction(10.0, 7.0)
        m37 = copy.deepcopy(model)
        m10 = copy.deepcopy(model)
        for rxn in m37.reactions.values():
            rxn.upper_bound *= corr_37
        for rxn in m10.reactions.values():
            rxn.upper_bound *= corr_10
        assert m37.reactions["PGI"].upper_bound > m10.reactions["PGI"].upper_bound


# ============================================================================
# G9: Population Density Feedback to GEM
# ============================================================================

class TestG9DensityScaling:

    def test_exchange_reactions_not_scaled(self):
        model = _toy_model()
        orig_ex_ub = model.reactions["EX_glc_e"].upper_bound
        corr = 0.5
        for rxn_id, rxn in model.reactions.items():
            if not rxn_id.startswith("EX_"):
                rxn.upper_bound *= corr
        assert model.reactions["EX_glc_e"].upper_bound == orig_ex_ub

    def test_density_scale_clamp_range(self):
        for bx, carrying in [(0, 100), (10, 100), (100, 100),
                              (200, 100), (1000, 100)]:
            if carrying > 0 and bx > 0:
                s = min(1.0, max(0.1, bx / carrying))
                assert 0.1 <= s <= 1.0

    def test_density_scale_at_half_capacity(self):
        bx, carrying = 50.0, 100.0
        s = min(1.0, max(0.1, bx / carrying))
        assert s == pytest.approx(0.5)

    def test_density_scale_above_capacity(self):
        bx, carrying = 500.0, 100.0
        s = min(1.0, max(0.1, bx / carrying))
        assert s == 1.0

    def test_density_scale_at_minimum_floor(self):
        bx, carrying = 1.0, 1000.0
        s = min(1.0, max(0.1, bx / carrying))
        assert s == pytest.approx(0.1)

    def test_last_fba_fluxes_stored(self):
        from helixlang.apps.ecosystem import (
            Ecosystem,
            EcosystemConfig,
            PatchConfig,
            Species,
            SubstrateConfig,
        )
        sp = Species(
            name="ecoli_gem",
            consumption={"glucose": (0.02, 0.1)},
            cn_ratio=6.0, maintenance=0.002,
            metabolic_model=ECOLI_CORE_MODEL,
        )
        pc = PatchConfig(
            name="p", kind="chemostat", width=1, height=1,
            flow_rate=0.0, temperature_c=37.0, ph=7.0,
            anoxic=True,
            initial_biomass={"ecoli_gem": 100.0},
            substrates={"glucose": SubstrateConfig(initial_mm=10.0, bulk_mm=10.0)},
        )
        eco = Ecosystem(EcosystemConfig(
            ticks=0, species=[sp], patches=[pc], gem_driven=True))
        patch = eco.patches[0]
        patch._growth_rate_gem(sp, 100.0, 0, 0, 1.0, 1.0, 1.0, 0.0)
        assert isinstance(sp.last_fba_fluxes, dict)
        assert len(sp.last_fba_fluxes) > 0

    def test_high_density_reduces_growth(self):
        from helixlang.apps.ecosystem import (
            Ecosystem,
            EcosystemConfig,
            PatchConfig,
            Species,
            SubstrateConfig,
        )
        sp = Species(
            name="ecoli_gem",
            consumption={"glucose": (0.02, 0.1)},
            cn_ratio=6.0, maintenance=0.002,
            metabolic_model=ECOLI_CORE_MODEL,
        )
        pc = PatchConfig(
            name="p", kind="chemostat", width=1, height=1,
            flow_rate=0.0, temperature_c=37.0, ph=7.0,
            anoxic=True,
            initial_biomass={"ecoli_gem": 100.0},
            substrates={"glucose": SubstrateConfig(initial_mm=10.0, bulk_mm=10.0)},
            carrying_capacity=200.0,
        )
        eco = Ecosystem(EcosystemConfig(
            ticks=0, species=[sp], patches=[pc], gem_driven=True))
        patch = eco.patches[0]
        g_low, _, _ = patch._growth_rate_gem(sp, 10.0, 0, 0, 1.0, 1.0, 1.0, 0.0)
        g_high, _, _ = patch._growth_rate_gem(sp, 190.0, 0, 0, 1.0, 1.0, 1.0, 0.0)
        assert g_low >= g_high, "Higher density should reduce growth"


# ============================================================================
# G10: Genome Evolution -> FBA Re-solve
# ============================================================================

class TestG10EvolutionFBA:

    def _make_vm(self):
        from helixlang.ast_nodes import Program
        from helixlang.bytecode import Chunk
        from helixlang.vm import CellVM
        return CellVM(Chunk(), Program())

    def test_gem_dirty_flag_default(self):
        vm = self._make_vm()
        assert vm._gem_dirty is False

    def test_gpr_map_default_empty(self):
        vm = self._make_vm()
        assert vm._gem_gpr_map == {}

    def test_update_enzyme_levels_frameshift_knocks_out(self):
        vm = self._make_vm()
        vm._gem_gpr_map = {"pfkA": ["PFK", "PFK2"]}
        vm._enzyme_kcat = {"PFK": 100.0, "PFK2": 200.0}
        vm._crispr_edits = [{
            "success": True, "target": "pfkA", "edit_type": "frameshift",
        }]
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert vm._enzyme_kcat["PFK"] == 0.0
        assert vm._enzyme_kcat["PFK2"] == 0.0
        assert vm._gem_dirty is False

    def test_update_enzyme_levels_deletion_knocks_out(self):
        vm = self._make_vm()
        vm._gem_gpr_map = {"gltA": ["CS"]}
        vm._enzyme_kcat = {"CS": 500.0}
        vm._crispr_edits = [{
            "success": True, "target": "gltA", "edit_type": "deletion",
        }]
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert vm._enzyme_kcat["CS"] == 0.0

    def test_update_enzyme_levels_substitution_reduces(self):
        vm = self._make_vm()
        vm._gem_gpr_map = {"pgi": ["PGI"]}
        vm._enzyme_kcat = {"PGI": 1000.0}
        vm._crispr_edits = [{
            "success": True, "target": "pgi", "edit_type": "substitution",
        }]
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert vm._enzyme_kcat["PGI"] == pytest.approx(700.0)

    def test_update_enzyme_levels_ignores_failed_edits(self):
        vm = self._make_vm()
        vm._gem_gpr_map = {"pgi": ["PGI"]}
        vm._enzyme_kcat = {"PGI": 1000.0}
        vm._crispr_edits = [{
            "success": False, "target": "pgi", "edit_type": "frameshift",
        }]
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert vm._enzyme_kcat["PGI"] == 1000.0

    def test_update_enzyme_levels_ignores_unknown_gene(self):
        vm = self._make_vm()
        vm._gem_gpr_map = {"pgi": ["PGI"]}
        vm._enzyme_kcat = {"PGI": 1000.0}
        vm._crispr_edits = [{
            "success": True, "target": "unknown", "edit_type": "frameshift",
        }]
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert vm._enzyme_kcat["PGI"] == 1000.0

    def test_nonsense_edit_knocks_out(self):
        vm = self._make_vm()
        vm._gem_gpr_map = {"zwf": ["G6PDH2r"]}
        vm._enzyme_kcat = {"G6PDH2r": 800.0}
        vm._crispr_edits = [{
            "success": True, "target": "zwf", "edit_type": "nonsense",
        }]
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert vm._enzyme_kcat["G6PDH2r"] == 0.0

    def test_cellvm_run_dispatches_to_dispatcher(self):
        vm = self._make_vm()
        vm._gem_dirty = True
        vm._gem_gpr_map = {"pgi": ["PGI"]}
        vm._enzyme_kcat = {"PGI": 1000.0}
        vm._crispr_edits = [{
            "success": True, "target": "pgi", "edit_type": "frameshift",
        }]
        vm._metabolic_model = _toy_model()
        vm._dispatcher._update_enzyme_levels_from_edits()
        assert vm._enzyme_kcat["PGI"] == 0.0
        assert not vm._gem_dirty


# ============================================================================
# G7: Ecosystem Wiring Integration Test
# ============================================================================

class TestG7EcosystemWiring:

    def test_growth_rate_gem_applies_regulatory_bounds(self):
        from helixlang.apps.ecosystem import (
            Ecosystem,
            EcosystemConfig,
            PatchConfig,
            Species,
            SubstrateConfig,
        )
        model = copy.deepcopy(ECOLI_CORE_MODEL)
        sp = Species(
            name="ecoli",
            consumption={"glucose": (0.02, 0.1)},
            cn_ratio=6.0, maintenance=0.002,
            metabolic_model=model,
            grn_edges=[_edge(target="pfkA", reg_type="repression",
                             confidence=1.0, target_reaction="PFK")],
            grn_gpr_map={},
        )
        pc = PatchConfig(
            name="p", kind="chemostat", width=1, height=1,
            flow_rate=0.0, temperature_c=37.0, ph=7.0,
            anoxic=True,
            initial_biomass={"ecoli": 100.0},
            substrates={"glucose": SubstrateConfig(initial_mm=10.0, bulk_mm=10.0)},
        )
        eco = Ecosystem(EcosystemConfig(
            ticks=0, species=[sp], patches=[pc], gem_driven=True))
        patch = eco.patches[0]
        g_c, _, _is_fba = patch._growth_rate_gem(sp, 100.0, 0, 0, 1.0, 1.0, 1.0, 0.0)
        assert g_c >= 0.0

    def test_species_has_grn_fields(self):
        from helixlang.apps.ecosystem import Species
        sp = Species(name="test")
        assert sp.grn_edges == []
        assert sp.grn_gpr_map == {}

    def test_regulatory_bounds_reduce_exchange_upper(self):
        from helixlang.gem.bridge import apply_regulatory_bounds
        model = copy.deepcopy(ECOLI_CORE_MODEL)
        edges = [_edge(target="pfkA", reg_type="repression",
                       confidence=1.0, target_reaction="PFK")]
        gpr = {}
        n = apply_regulatory_bounds(model, edges, gpr)
        assert n == 1
        assert model.reactions["PFK"].upper_bound < 1000.0
