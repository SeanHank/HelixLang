"""Ecosystem coverage-closure tests (app-level residual branches).

Fills the remaining uncovered lines/branches of
:mod:`helixlang.plugins.apps.ecosystem`: genotype->trait decoding,
``gem_to_species`` attribute-present/absent paths, FBA-backed growth
(synthetic ``MetabolicModel`` fixtures in the ecosystem exchange
convention), the GEM growth bookkeeping loop, CENTURY/N-cycle edge
guards, patch query helpers, scheduler forcing, dispersal edge cases,
analytics fallbacks, the invasion-fitness evolution loop and the
``build_multi_species_ecosystem`` genome-based constructor.
"""
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from helixlang.plugins.apps.ecosystem import (
    CenturyPools,
    DiurnalForcing,
    Ecosystem,
    EcosystemConfig,
    Metapopulation,
    NitrogenCycle,
    PatchConfig,
    ScalarConfig,
    Species,
    SpeciesTraitParams,
    SubstrateConfig,
    acetotroph,
    build_multi_species_ecosystem,
    gem_to_species,
    phototroph,
)
from helixlang.plugins.gem.grn_inference import EvidenceLevel, RegulatoryEdge
from helixlang.plugins.runtime.metabolism import (
    MetabolicModel,
    Reaction,
)

# ============================================================================
# Synthetic GEM fixtures (ecosystem exchange convention)
#
# coef=-1 exchange reactions (uptake convention) with a pre-set negative
# lower bound: ``set_uptake`` clamps the bound to ``[-limit, limit]`` so
# negative flux = substrate uptake for glc/o2, positive flux = product
# release for co2, matching the ecosystem's EX-flux sign expectations.
# ============================================================================

def _mk_ex(rid: str, met: str) -> Reaction:
    return Reaction(id=rid, name=rid, stoichiometry={met: -1.0},
                    lower_bound=-10.0, upper_bound=1000.0,
                    subsystem="exchange")


def _m1_aerobic_model() -> MetabolicModel:
    """Glucose + O2 -> CO2 + biomass (growth; co2 released, o2 taken up)."""
    m = MetabolicModel()
    m.add_reaction(_mk_ex("EX_glc__D_e", "glc__D_e"))
    m.add_reaction(_mk_ex("EX_o2_e", "o2_e"))
    m.add_reaction(_mk_ex("EX_co2_e", "co2_e"))
    m.add_reaction(Reaction(
        id="RESP", name="respiration",
        stoichiometry={"glc__D_e": -1.0, "o2_e": -1.0,
                       "co2_e": 1.0, "biomass_met": 1.0},
        upper_bound=1000.0))
    m.add_reaction(Reaction(id="BIOMASS", name="biomass drain",
                            stoichiometry={"biomass_met": -1.0},
                            upper_bound=1000.0))
    m.set_biomass("BIOMASS")
    return m


def _m2_photo_model() -> MetabolicModel:
    """CO2 + light -> O2 + biomass (growth; co2 taken up, o2 released)."""
    m = MetabolicModel()
    m.add_reaction(_mk_ex("EX_co2_e", "co2_e"))
    m.add_reaction(_mk_ex("EX_o2_e", "o2_e"))
    m.add_reaction(Reaction(
        id="PHOTOSYN", name="photosynthesis",
        stoichiometry={"co2_e": -1.0, "o2_e": 1.0, "biomass_met": 1.0},
        upper_bound=1000.0))
    m.add_reaction(Reaction(id="BIOMASS", name="biomass drain",
                            stoichiometry={"biomass_met": -1.0},
                            upper_bound=1000.0))
    m.set_biomass("BIOMASS")
    return m


def _m_no_biomass() -> MetabolicModel:
    """Valid reaction set but no biomass objective -> FBA fallback."""
    m = MetabolicModel()
    m.add_reaction(_mk_ex("EX_glc__D_e", "glc__D_e"))
    m.add_reaction(Reaction(
        id="RESP", name="respiration",
        stoichiometry={"glc__D_e": -1.0, "biomass_met": 1.0},
        upper_bound=1000.0))
    return m


def _m_biomass_rxn_named() -> MetabolicModel:
    """No ``set_biomass`` but a reaction literally named BIOMASS_reaction:
    exercises the ``solve(objective="BIOMASS_reaction")`` branch."""
    m = MetabolicModel()
    m.add_reaction(_mk_ex("EX_glc__D_e", "glc__D_e"))
    m.add_reaction(Reaction(
        id="RESP", name="respiration",
        stoichiometry={"glc__D_e": -1.0, "biomass_met": 1.0},
        upper_bound=1000.0))
    m.add_reaction(Reaction(id="BIOMASS_reaction", name="biomass drain",
                            stoichiometry={"biomass_met": -1.0},
                            upper_bound=1000.0))
    return m


def _m3_ferment_model() -> MetabolicModel:
    """Anaerobic fermentation: glucose -> biomass with NO oxygen/CO2
    exchange at all (both excise the o2/co2 branches in the step loop)."""
    m = MetabolicModel()
    m.add_reaction(_mk_ex("EX_glc__D_e", "glc__D_e"))
    m.add_reaction(Reaction(
        id="FERM", name="fermentation",
        stoichiometry={"glc__D_e": -1.0, "etoh_e": 2.0, "biomass_met": 1.0},
        upper_bound=1000.0))
    m.add_reaction(Reaction(id="EX_etoh_e", name="ethanol export",
                            stoichiometry={"etoh_e": -1.0},
                            lower_bound=0.0, upper_bound=1000.0))
    m.add_reaction(Reaction(id="BIOMASS", name="biomass drain",
                            stoichiometry={"biomass_met": -1.0},
                            upper_bound=1000.0))
    m.set_biomass("BIOMASS")
    return m


def _gm_species(model, name="gm", sub="glucose",
                photo=False, photo_vmax=0.0) -> Species:
    return Species(name=name, consumption={sub: (0.02, 0.1)},
                   cn_ratio=6.0, maintenance=0.002, photo=photo,
                   photo_vmax=photo_vmax,
                   metabolic_model=model)


def _gm_patch(species, anoxic=True, glu=10.0, nh4=50.0,
              co2=None, scalars=None,
              fluctuation_period=0, fluctuation_amplitude=0.0) -> PatchConfig:
    subs = {"glucose": SubstrateConfig(initial_mm=glu, carbon_per_mol=6)}
    if co2 is not None:
        subs["co2"] = SubstrateConfig(initial_mm=co2, carbon_per_mol=1)
    return PatchConfig(
        name="p", kind="chemostat", width=1, height=1, flow_rate=0.0,
        anoxic=anoxic, initial_biomass={species.name: 10.0},
        initial_nh4_mm=nh4, substrates=subs, scalars=scalars or {},
        fluctuation_period=fluctuation_period,
        fluctuation_amplitude=fluctuation_amplitude)


class TestCoverageBranches:
    # -- genotype -> phenotype (A4) --------------------------------------
    def test_genome_decodes_full_trait_vector(self):
        sp = Species(name="s", genome="ACGTACGTACGTACGTACGTACGTACGT")
        t = sp.traits
        assert t is not sp
        assert 0.5 <= t.uptake_gain <= 1.5
        assert 0.7 <= t.growth_rate_gain <= 1.3
        assert 0.3 <= t.yield_c <= 0.7
        assert 1.5 <= t.q10 <= 3.0
        assert 0.0 <= t.switching_cost <= 0.5

    def test_genome_short_uses_empty_slice_defaults(self):
        # a 1-nt genome: most 1/6th slices are empty -> _bit_mean returns 1.0
        t = SpeciesTraitParams().from_genome("A")
        assert t.uptake_gain == 0.5
        assert t.q10 == 1.0
        assert t.switching_cost == 1.0  # empty slice -> _bit_mean 1.0

    def test_genome_empty_returns_unchanged(self):
        t = SpeciesTraitParams()
        assert t.from_genome("") is t

    # -- gem_to_species attribute paths ---------------------------------
    def test_gem_to_species_full_attributes(self):
        class _BiomassComponent:
            def __init__(self, mid, c):
                self.metabolite_id = mid
                self.coefficient = c

        class _BiomassReaction:
            components = [
                _BiomassComponent("atp_c", -5.0),      # cofactor -> skipped
                _BiomassComponent("glc__D_e", -1.0),   # C-bearing (no N)
                _BiomassComponent("nh4_e", -2.0),      # N-bearing
                _BiomassComponent("pi_e", -3.0),       # neither C nor N
                _BiomassComponent("cys__L_e", 1.0),    # product -> ignored
            ]

        class _Result:
            fba_fluxes = {"EX_glc__D_e": -12.0, "EX_succ_e": 3.0, "ATPM": 8.0}
            growth_rate = 0.87
            kcat_predictions = [
                SimpleNamespace(reaction_id="R1", kcat=50.0),
                SimpleNamespace(reaction_id="R2"),       # no kcat -> skipped
            ]
            km_estimates = {"EX_glc_e": 0.15}
            biomass_reaction = _BiomassReaction()

        params = gem_to_species(_Result(), organism="e_coli_k12")
        assert params["vmax"] == pytest.approx(12.0)
        assert params["ks"] == pytest.approx(0.15)
        assert params["secretion"]["succinate"] > 0.0
        assert params["cn_ratio"] == pytest.approx(3.0)  # clamped 0.5 -> 3
        assert params["maintenance"] == pytest.approx(8.0e-4, abs=1e-12)

    def test_gem_to_species_absent_attributes_fall_back(self):
        class _Bare:  # none of the optional attributes present
            pass

        params = gem_to_species(_Bare())
        assert params["vmax"] == pytest.approx(0.02)
        assert params["ks"] == pytest.approx(0.1)
        assert params["yield_c"] == pytest.approx(0.5)
        assert params["cn_ratio"] == pytest.approx(6.0)
        assert params["maintenance"] == pytest.approx(0.001)

    def test_gem_to_species_unmapped_glucose_flux(self):
        class _Result:
            fba_fluxes = {"GLCPTS": -2.0}
            kcat_predictions = []
            km_estimates = {"EX_ac_e": 0.5, "GLCPTS": 0.3}
            biomass_reaction = None

        params = gem_to_species(_Result())
        assert params["vmax"] == pytest.approx(2.0)
        assert params["ks"] == pytest.approx(0.3)

    def test_gem_to_species_biomass_without_nitrogen(self):
        class _C:
            def __init__(self, mid, c):
                self.metabolite_id = mid
                self.coefficient = c

        class _BM:
            components = [_C("glc__D_e", -1.0), _C("succ_e", -1.0)]

        class _R:
            fba_fluxes = {"EX_glc__D_e": -1.0}
            kcat_predictions = []
            km_estimates = {}
            biomass_reaction = _BM()

        # only-carbon biomass -> cn_ratio stays at the 6.0 default
        assert gem_to_species(_R())["cn_ratio"] == pytest.approx(6.0)

    # -- Levins / CENTURY / nitrogen tightly-repeatable guards -------------
    def test_metapopulation_zero_colonization(self):
        meta = Metapopulation(8, 0.0, 0.1, seed=1)
        assert meta.levins_equilibrium() == 0.0

    def test_century_nonpositive_litter_and_ticks(self):
        c = CenturyPools()
        c.add_litter(0.0)
        c.add_litter(-5.0)
        c.step(0, 1.0, 1.0)
        assert c.litter_in_c == 0.0
        assert c.respired_c == 0.0

    def test_nitrogen_edge_guards(self):
        n = NitrogenCycle()
        n.immobilize(0.0)
        n.excrete(0.0)
        n.step(0.0, 0, 1.0, 1.0, True)
        assert n.nh4_mm == 0.0

    def test_nitrogen_ch4_oxidation(self):
        n = NitrogenCycle()
        n.ch4_mm = 1.0
        n.step(0.0, 10, 1.0, 1.0, True)
        assert n.ch4_mm < 1.0

    # -- Patch construction / query helpers ---------------------------------
    def test_patch_autocreates_glucose_for_chemostat(self):
        sp = Species(name="c", consumption={"acetate": (0.01, 0.1)})
        pc = PatchConfig(
            name="p", kind="chemostat", width=1, height=1,
            substrates={"acetate": SubstrateConfig(initial_mm=10.0)})
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        patch = eco.patches[0]
        assert "glucose" in patch.fields
        assert patch.field_mm("glucose") == pytest.approx(1.0)

    def test_patch_initial_biomass_unknown_species(self):
        sp = Species(name="c", consumption={"glucose": (0.01, 0.1)})
        pc = PatchConfig(
            name="p", kind="chemostat", width=1, height=1,
            initial_biomass={"ghost": 5.0},
            substrates={"glucose": SubstrateConfig(initial_mm=10.0)})
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        patch = eco.patches[0]
        assert patch.totals()["ghost"] == pytest.approx(5.0)

    def test_patch_query_helpers(self):
        sp = Species(name="c", consumption={"glucose": (0.01, 0.1)})
        pc = PatchConfig(
            name="p", kind="chemostat", width=1, height=1,
            initial_biomass={"c": 4.0},
            substrates={"glucose": SubstrateConfig(initial_mm=7.0)})
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        patch = eco.patches[0]
        assert patch.total_biomass_of("missing") == 0.0
        assert patch.local_field("glucose", 0, 0) == pytest.approx(7.0)
        assert patch.pool_totals() == patch.century.pools
        assert patch._cpm("bogus") == 6

    def test_carbon_balance_skips_zero_carbon_field(self):
        sp = Species(name="c", consumption={"glucose": (0.01, 0.1)})
        pc = PatchConfig(
            name="p", kind="chemostat", width=1, height=1, anoxic=False,
            substrates={"glucose": SubstrateConfig(initial_mm=10.0)})
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        patch = eco.patches[0]
        assert "oxygen" in patch.fields  # cpm 0 -> carbon_balance skips it
        c0 = patch.carbon_balance()
        assert c0 > 0.0

    # -- environmental drivers: stress / fluctuation / photo-without-co2 ------
    def test_stress_tolerance_gates_growth(self):
        sp = Species(name="s", consumption={"glucose": (0.02, 0.1)})
        pc = _gm_patch(sp, anoxic=True, glu=100.0, nh4=50.0,
                       scalars={"toxin": ScalarConfig("toxin", 0.5)})
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        patch = eco.patches[0]
        assert patch._stress_level() == pytest.approx(0.5)
        patch.step(1, random.Random(1))
        assert patch._stress_level() == pytest.approx(0.5)

    def test_fluctuation_phase_alternates(self):
        sp = Species(name="s", consumption={"glucose": (0.02, 0.1)})
        pc = _gm_patch(sp, anoxic=True, glu=100.0, nh4=50.0,
                       fluctuation_period=1, fluctuation_amplitude=2.0)
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        patch = eco.patches[0]
        patch.step(1, random.Random(1))
        patch.step(1, random.Random(1))  # tick 1 -> phase 1 -> amplitude cap
        assert patch.tick == 2

    def test_photo_rate_zero_with_exhausted_co2(self):
        sp = phototroph("producer")
        pc = PatchConfig(
            name="w", kind="water", width=1, height=1, anoxic=True,
            initial_biomass={"producer": 100.0},
            substrates={"co2": SubstrateConfig(initial_mm=0.0,
                                               carbon_per_mol=1)},
            scalars={"light": ScalarConfig("light", 200.0)})
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        patch = eco.patches[0]
        # co2 at the floor: photosynthesis_rate -> 0 -> no photo component
        g_c, comps = patch._growth_rate(sp, 100.0, 0, 0, 1.0, 1.0, 1.0, 200.0)
        assert g_c == 0.0
        assert all(s != "__photo__" for s, _, _ in comps)

    def test_oxygen_limited_growth_capped(self):
        sp = Species(name="s", consumption={"glucose": (0.02, 0.1)},
                     cn_ratio=6.0, maintenance=0.002)
        pc = _gm_patch(sp, anoxic=False, glu=100.0, nh4=50.0)
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        patch = eco.patches[0]
        patch.fields["oxygen"].set_all(0.005)
        c0 = patch.carbon_balance()
        patch.step(1, random.Random(1))
        assert patch.carbon_balance() == pytest.approx(c0, rel=1e-9)

    # -- FBA-backed growth (synthetic GEMs) --------------------------------
    def test_growth_rate_gem_aerobic_components(self):
        sp = _gm_species(_m1_aerobic_model())
        eco = Ecosystem(EcosystemConfig(
            ticks=0, species=[sp], patches=[_gm_patch(sp)], gem_driven=True))
        patch = eco.patches[0]
        g_c, comps, is_fba = patch._growth_rate_gem(
            sp, 5.0, 0, 0, 1.0, 1.0, 1.0, 0.0)
        assert is_fba
        assert g_c > 0.0
        subs = [s for s, _, _ in comps]
        assert "glucose" in subs  # EX_glc__D_e negative flux -> uptake comp

    def test_growth_rate_gem_ignores_grn_noop_activation(self):
        sp = _gm_species(_m1_aerobic_model())
        sp.grn_edges = [RegulatoryEdge(
            tf_id="t1", target_gene="g1", regulation_type="activation",
            evidence_level=EvidenceLevel.DATABASE, confidence=0.0,
            target_reaction="RESP")]
        sp.grn_gpr_map = {"g1": ["RESP"]}
        eco = Ecosystem(EcosystemConfig(
            ticks=0, species=[sp], patches=[_gm_patch(sp)], gem_driven=True))
        patch = eco.patches[0]
        g_c, _comps, is_fba = patch._growth_rate_gem(
            sp, 5.0, 0, 0, 1.0, 1.0, 1.0, 0.0)
        assert is_fba and g_c > 0.0

    def test_growth_rate_gem_zero_biomass_at_zero_density(self):
        sp = _gm_species(_m1_aerobic_model())
        eco = Ecosystem(EcosystemConfig(
            ticks=0, species=[sp], patches=[_gm_patch(sp)], gem_driven=True))
        patch = eco.patches[0]
        g_c, comps, is_fba = patch._growth_rate_gem(
            sp, 0.0, 0, 0, 1.0, 1.0, 1.0, 0.0)  # bx <= 0 -> no density scale
        assert is_fba and g_c > 0.0
        assert "glucose" in [s for s, _, _ in comps]

    def test_growth_rate_gem_exhausted_substrate(self):
        sp = _gm_species(_m1_aerobic_model())
        pc = _gm_patch(sp, glu=0.005)  # <= 5% ks floor
        eco = Ecosystem(EcosystemConfig(
            ticks=0, species=[sp], patches=[pc], gem_driven=True))
        patch = eco.patches[0]
        g_c, comps, is_fba = patch._growth_rate_gem(
            sp, 5.0, 0, 0, 1.0, 1.0, 1.0, 0.0)
        assert is_fba
        assert g_c == 0.0
        assert comps == []

    def test_growth_rate_gem_photo_metabolite_branch(self):
        sp = _gm_species(_m2_photo_model(), sub="co2",
                         photo=True, photo_vmax=0.012)
        pc = _gm_patch(sp, anoxic=False, glu=0.0, co2=5.0)
        eco = Ecosystem(EcosystemConfig(
            ticks=0, species=[sp], patches=[pc], gem_driven=True))
        patch = eco.patches[0]
        g_c, comps, is_fba = patch._growth_rate_gem(
            sp, 5.0, 0, 0, 1.0, 1.0, 1.0, 300.0)
        assert is_fba and g_c > 0.0
        assert "co2" in [s for s, _, _ in comps]

    def test_growth_rate_gem_missing_biomass_falls_back(self, capsys):
        sp = _gm_species(_m_no_biomass())
        eco = Ecosystem(EcosystemConfig(
            ticks=0, species=[sp], patches=[_gm_patch(sp)], gem_driven=True))
        patch = eco.patches[0]
        g_c, comps, is_fba = patch._growth_rate_gem(
            sp, 5.0, 0, 0, 1.0, 1.0, 1.0, 0.0)
        assert not is_fba
        assert g_c > 0.0  # Monod fallback
        assert "glucose" in [s for s, _, _ in comps]
        assert "[GEM-FALLBACK]" in capsys.readouterr().err

    def test_growth_rate_gem_biomass_reaction_named(self):
        sp = _gm_species(_m_biomass_rxn_named())
        eco = Ecosystem(EcosystemConfig(
            ticks=0, species=[sp], patches=[_gm_patch(sp)], gem_driven=True))
        patch = eco.patches[0]
        g_c, comps, is_fba = patch._growth_rate_gem(
            sp, 5.0, 0, 0, 1.0, 1.0, 1.0, 0.0)
        assert is_fba and g_c > 0.0
        assert "glucose" in [s for s, _, _ in comps]

    # -- GEM growth bookkeeping inside the population loop ------------------
    def test_gem_step_aerobic_books_carbon_and_oxygen(self):
        sp = _gm_species(_m1_aerobic_model())
        eco = Ecosystem(EcosystemConfig(
            ticks=0, seed=1, species=[sp],
            patches=[_gm_patch(sp, anoxic=False, glu=10.0, nh4=50.0)],
            gem_driven=True))
        patch = eco.patches[0]
        patch.step(5, random.Random(1))
        assert patch.totals()["gm"] > 0.0
        assert patch.fields["glucose"].total_mm() < 10.0
        assert patch.fields["co2"].total_mm() > 1.0  # respired CO2 added
        assert patch.consumed_c["gm"] > 0.0

    def test_gem_step_photo_produces_oxygen(self):
        sp = _gm_species(_m2_photo_model(), sub="co2",
                         photo=True, photo_vmax=0.012)
        pc = _gm_patch(sp, anoxic=False, glu=0.0, co2=5.0, nh4=50.0,
                       scalars={"light": ScalarConfig("light", 300.0)})
        eco = Ecosystem(EcosystemConfig(
            ticks=0, seed=1, species=[sp], patches=[pc], gem_driven=True))
        patch = eco.patches[0]
        patch.step(5, random.Random(1))
        assert patch.fields["oxygen"].total_mm() > 0.21  # produced
        assert patch.respired_c["gm"] == 0.0  # no CO2 secretion flux

    def test_gem_step_nitrogen_starved_no_consumption(self):
        sp = _gm_species(_m1_aerobic_model())
        eco = Ecosystem(EcosystemConfig(
            ticks=0, seed=1, species=[sp],
            patches=[_gm_patch(sp, anoxic=False, glu=10.0, nh4=0.0)],
            gem_driven=True))
        patch = eco.patches[0]
        patch.step(1, random.Random(1))
        # N-limited: g == 0 -> deplete returns 0 -> no growth bookkeeping
        assert patch.totals()["gm"] > 0.0
        assert patch.consumed_c["gm"] == 0.0

    def test_gem_step_ferment_emits_no_co2_or_o2(self):
        sp = _gm_species(_m3_ferment_model())
        eco = Ecosystem(EcosystemConfig(
            ticks=0, seed=1, species=[sp],
            patches=[_gm_patch(sp, anoxic=False, glu=10.0, nh4=50.0)],
            gem_driven=True))
        patch = eco.patches[0]
        o2_0 = patch.fields["oxygen"].total_mm()
        patch.step(2, random.Random(1))
        assert patch.consumed_c["gm"] > 0.0  # FBA block entered
        assert patch.respired_c["gm"] == 0.0  # no EX_co2_e flux
        assert patch.fields["oxygen"].total_mm() == pytest.approx(o2_0)

    def test_gem_step_anoxic_skips_oxygen_blocks(self):
        sp = _gm_species(_m1_aerobic_model())
        eco = Ecosystem(EcosystemConfig(
            ticks=0, seed=1, species=[sp],
            patches=[_gm_patch(sp, anoxic=True, glu=10.0, nh4=50.0)],
            gem_driven=True))
        patch = eco.patches[0]
        assert "oxygen" not in patch.fields
        patch.step(2, random.Random(1))
        assert patch.consumed_c["gm"] > 0.0
        assert patch.respired_c["gm"] > 0.0  # co2 produced, no o2 blocks

    # -- _apply_growth direct branches ----------------------------------------
    def test_apply_growth_photo_without_oxygen_field(self):
        sp = Species(name="s", cn_ratio=6.0)
        pc = _gm_patch(sp, anoxic=True, co2=5.0, glu=0.0)
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        patch = eco.patches[0]
        growth = patch._apply_growth(
            sp, 100.0, 0, 0, [("__photo__", 1, 0.01)], 0.01, 0.01,
            SpeciesTraitParams())
        assert growth > 0.0
        assert "oxygen" not in patch.fields

    def test_apply_growth_photo_exhausted_co2(self):
        sp = Species(name="s", cn_ratio=6.0)
        pc = _gm_patch(sp, anoxic=False, co2=0.0, glu=0.0)
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        patch = eco.patches[0]
        growth = patch._apply_growth(
            sp, 100.0, 0, 0, [("__photo__", 1, 0.01)], 0.01, 0.01,
            SpeciesTraitParams())
        assert growth == 0.0

    def test_apply_growth_yield_one_no_respiration(self):
        sp = Species(name="s", cn_ratio=6.0)
        pc = _gm_patch(sp, anoxic=False, glu=10.0)
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        patch = eco.patches[0]
        traits = SpeciesTraitParams(yield_c=1.0)
        growth = patch._apply_growth(
            sp, 100.0, 0, 0, [("glucose", 6, 0.01)], 0.1, 0.1, traits)
        assert growth > 0.0
        assert patch.respired_c["s"] == 0.0
        assert patch.consumed_c["s"] > 0.0

    # -- predation / biomass transfer guards ----------------------------
    def test_predation_skips_extinct_prey(self):
        prey = Species(name="prey", consumption={"glucose": (0.02, 0.1)})
        pred = Species(name="predator", consumption={"glucose": (0.02, 0.1)})
        pred.diet["prey"] = 0.5
        pred.attack_rate["prey"] = 0.001
        pc = PatchConfig(
            name="c", kind="chemostat", width=1, height=1, flow_rate=0.0,
            anoxic=True,
            initial_biomass={"prey": 0.0, "predator": 5.0},
            substrates={"glucose": SubstrateConfig(initial_mm=50.0,
                                                   carbon_per_mol=6)},
            initial_nh4_mm=20.0)
        eco = Ecosystem(EcosystemConfig(
            ticks=0, species=[prey, pred], patches=[pc]))
        eco.patches[0].step(1, random.Random(1))

    def test_subtract_and_add_biomass_guards(self):
        sp = Species(name="c", consumption={"glucose": (0.02, 0.1)})
        pc = PatchConfig(
            name="p", kind="chemostat", width=1, height=1,
            initial_biomass={"c": 0.0},
            substrates={"glucose": SubstrateConfig(initial_mm=10.0)})
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        patch = eco.patches[0]
        patch._subtract_biomass("c", 5.0)      # total 0 -> no-op
        patch._add_biomass("ghost", 5.0)       # unknown name -> no-op
        assert patch.total_biomass_of("c") == 0.0
        # empty grid -> per-site add is skipped
        patch.biomass["c"] = []
        patch._add_biomass("c", 5.0)
        assert patch.total_biomass_of("c") == 0.0

    # -- capacity overflow recycling ------------------------------------
    def test_capacity_overflow_recycled_into_litter(self):
        sp = Species(name="s", consumption={"glucose": (0.02, 0.1)},
                     cn_ratio=6.0, maintenance=0.001)
        pc = PatchConfig(
            name="p", kind="chemostat", width=1, height=1, flow_rate=0.0,
            anoxic=True, carrying_capacity=50.0,
            initial_biomass={"s": 100.0}, initial_nh4_mm=50.0,
            substrates={"glucose": SubstrateConfig(
                initial_mm=100.0, carbon_per_mol=6)})
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        patch = eco.patches[0]
        patch._apply_capacity(50.0)
        assert patch.total_biomass_c() == pytest.approx(50.0)
        assert patch.century.litter_in_c > 0.0

    def test_step_nitrogen_clamps_negative_som(self):
        sp = Species(name="s", consumption={"glucose": (0.02, 0.1)})
        pc = _gm_patch(sp, anoxic=True, glu=10.0, nh4=20.0)
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        patch = eco.patches[0]
        patch._som_n = 0.0
        patch.century.respired_c = 100.0
        patch._prev_respired = 0.0
        patch._step_nitrogen(1.0, 1.0)
        assert patch._som_n == 0.0

    # -- scheduler forcing ----------------------------------------------
    def test_fast_forward_uses_forcing_delta(self):
        sp = phototroph("producer", genome="ACGTACGT")
        pc = PatchConfig(
            name="w", kind="water", width=1, height=1, flow_rate=0.0,
            anoxic=False, initial_biomass={"producer": 100.0},
            scalars={
                "light": ScalarConfig("light", 500.0,
                                      DiurnalForcing(500.0, 500.0, lo=0.0)),
                "temperature": ScalarConfig("temperature", 25.0),
            })
        eco = Ecosystem(EcosystemConfig(
            ticks=0, seed=1, fast_forward=True, species=[sp], patches=[pc]))
        advance = eco.step()
        assert advance >= 1
        assert eco.tick == advance
        assert eco._forcing_delta() >= 0.0

    # -- dispersal edge cases -------------------------------------------
    def test_disperse_rejects_missing_neighbor(self):
        sp = Species(name="c", consumption={"glucose": (0.02, 0.1)})
        pc = _gm_patch(sp, glu=10.0)
        pc.dispersal["nowhere"] = 0.1
        eco = Ecosystem(EcosystemConfig(ticks=1, species=[sp], patches=[pc]))
        with pytest.raises(ValueError):
            eco.run()

    def test_disperse_zero_flux_is_noop(self):
        sp = Species(name="c", consumption={"glucose": (0.02, 0.1)})
        pa = _gm_patch(sp, glu=10.0)
        pa.name = "a"
        pa.initial_biomass = {"c": 0.0}
        pb = _gm_patch(sp, glu=10.0)
        pb.name = "b"
        pb.initial_biomass = {"c": 0.0}
        pa.dispersal["b"] = 0.2
        eco = Ecosystem(EcosystemConfig(
            ticks=2, seed=1, fast_forward=False, species=[sp], patches=[pa, pb]))
        eco.run()  # flux == 0 -> dispersal skipped without error

    # -- analytics fallbacks --------------------------------------------
    def test_record_skips_zero_carbon_field(self):
        sp = Species(name="c", consumption={"glucose": (0.02, 0.1)})
        pc = _gm_patch(sp, anoxic=False, glu=10.0)
        pc.substrates["ammonia"] = SubstrateConfig(initial_mm=1.0,
                                                   carbon_per_mol=0)
        eco = Ecosystem(EcosystemConfig(
            ticks=1, seed=1, fast_forward=False, species=[sp], patches=[pc]))
        rows = eco.run()
        assert "p:ammonia" not in rows[-1]
        assert "p:glucose" in rows[-1]

    def test_trophic_efficiency_counts_photo_producers(self):
        prod = Species(name="producer", photo=True, photo_vmax=0.01,
                       cn_ratio=8.0, maintenance=0.001)
        pred = Species(name="predator", consumption={"glucose": (0.02, 0.1)})
        pred.diet["producer"] = 0.5
        pred.attack_rate["producer"] = 0.001
        pc = PatchConfig(
            name="c", kind="chemostat", width=1, height=1, flow_rate=0.0,
            anoxic=True,
            initial_biomass={"producer": 80.0, "predator": 20.0},
            substrates={"glucose": SubstrateConfig(
                initial_mm=10.0, carbon_per_mol=6)})
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[prod, pred],
                                        patches=[pc]))
        assert eco.trophic_efficiency()["c"] == pytest.approx(0.25)

    def test_neutral_vs_niche_empty_community(self):
        sp = Species(name="c", consumption={"glucose": (0.02, 0.1)})
        pc = _gm_patch(sp, glu=10.0)
        pc.initial_biomass = {"c": 0.0}
        eco = Ecosystem(EcosystemConfig(ticks=0, species=[sp], patches=[pc]))
        assert eco.neutral_vs_niche() == {"label": 0.0, "deviance": 0.0}

    # -- invasion fitness / growth-rate fallbacks -------------------------
    def test_evaluate_growth_extinction_and_appearance(self):
        doomed = Species(name="doomed", consumption={}, cn_ratio=6.0,
                         maintenance=0.002)
        spark = Species(name="spark",
                        consumption={"glucose": (0.05, 0.1)},
                        cn_ratio=6.0, maintenance=0.001)
        pc = PatchConfig(
            name="p", kind="chemostat", width=1, height=1, flow_rate=0.0,
            anoxic=True, initial_biomass={"doomed": 5e-10, "spark": 5e-10},
            initial_nh4_mm=50.0,
            substrates={"glucose": SubstrateConfig(
                initial_mm=100.0, carbon_per_mol=6)})
        eco = Ecosystem(EcosystemConfig(
            ticks=0, seed=3, fast_forward=False,
            species=[doomed, spark], patches=[pc]))
        rates = eco._evaluate_growth(200)
        assert rates["doomed"] == -10.0
        assert rates["spark"] == 10.0

    def test_run_generations_substitutes_fitter_mutant(self):
        rows = self._run_generations("A" * 36, eval_ticks=2, nt=40)
        assert rows[0]["s:substituted"] == 1.0
        assert rows[0]["s:mutant_growth"] > rows[0]["s:resident_growth"]

    def test_run_generations_keeps_resident_when_not_fitter(self):
        rows = self._run_generations("T" * 36, eval_ticks=30, nt=36)
        assert rows[0]["s:substituted"] == 0.0
        assert rows[0]["s:resident_growth"] > 0.0

    @staticmethod
    def _run_generations(genome: str, eval_ticks: int,
                         nt: int) -> list[dict]:
        sp = Species(name="s", genome=genome,
                     consumption={"glucose": (0.02, 0.1)},
                     cn_ratio=6.0, maintenance=0.002)
        pc = PatchConfig(
            name="p", kind="chemostat", width=1, height=1, flow_rate=0.0,
            anoxic=True, initial_biomass={"s": 100.0}, initial_nh4_mm=50.0,
            substrates={"glucose": SubstrateConfig(
                initial_mm=100.0, carbon_per_mol=6)})
        cfg = EcosystemConfig(
            ticks=0, seed=0, fast_forward=False, species=[sp], patches=[pc],
            generations=1, substitution_rate=1.0, indel_rate=0.0,
            genome_length_nt=nt, evaluation_ticks=eval_ticks)
        return Ecosystem(cfg).run_generations()

    # -- presets ----------------------------------------------------------
    def test_phototroph_and_acetotroph_presets(self):
        p = phototroph(genome="ACGT")
        assert p.photo and p.consumption == {}
        a = acetotroph("a")
        assert "acetate" in a.consumption


def _build_fake_pipeline(fba_fluxes, growth_rate=None, name="sp"):
    """A GemPipelineResult stand-in; ``growth_rate`` conditionally absent."""
    attrs = {"fba_fluxes": fba_fluxes,
             "kcat_predictions": [],
             "km_estimates": {},
             "metabolic_model": None}
    if growth_rate is not None:
        attrs["growth_rate"] = growth_rate
    return type(name, (), attrs)()


class TestBuildMultiSpeciesEcosystem:
    def test_build_photoautotroph_from_inline_sequence(self, monkeypatch):
        captured = {}

        def _fake(genome_fasta, organism, medium):
            captured["path"] = genome_fasta
            return _build_fake_pipeline({"EX_co2_e": -5.0, "ATPM": 3.0},
                                        growth_rate=0.1)

        monkeypatch.setattr(
            "helixlang.plugins.apps.gem_pipeline.run_gem_pipeline", _fake)

        eco = build_multi_species_ecosystem(
            {"cell": "ATGCATGCATGCATGCATGCATGCATCGAACGT" * 2}, ticks=0)
        sp = eco.species_map["cell"]
        assert sp.photo
        assert sp.consumption["co2"][0] > 0.0
        # the inline DNA was written to a temp FASTA and cleaned up
        assert not Path(captured["path"]).exists()
        assert eco.patches[0].local_field("co2", 0, 0) == pytest.approx(5.0)

    def test_build_heterotroph_from_fasta_file(self, monkeypatch, tmp_path):
        fasta = tmp_path / "sp.fa"
        fasta.write_text(">sp\nACGTACGTACGTACGTACGTACGTACGTACGTACGT\n")
        monkeypatch.setattr(
            "helixlang.plugins.apps.gem_pipeline.run_gem_pipeline",
            lambda genome_fasta, organism, medium: _build_fake_pipeline(
                {"EX_glc__D_e": -8.0, "EX_ac_e": 2.0},
                growth_rate=0.5))

        eco = build_multi_species_ecosystem({"sp": str(fasta)}, ticks=0)
        sp = eco.species_map["sp"]
        assert not sp.photo
        assert "glucose" in sp.consumption
        assert sp.secretion["acetate"] > 0.0

    def test_build_two_consumers_share_glucose_substrate(self, monkeypatch):
        monkeypatch.setattr(
            "helixlang.plugins.apps.gem_pipeline.run_gem_pipeline",
            lambda genome_fasta, organism, medium: _build_fake_pipeline(
                {"EX_glc__D_e": -6.0}, growth_rate=0.4))

        eco = build_multi_species_ecosystem(
            {"a": "ACGTACGTACGTACGTACGTACGTACGTACGTACGT",
             "b": "TGCTTGCTTGCTTGCTTGCTTGCTTGCTTGCTTGCTTGCT"}, ticks=0)
        assert set(eco.species_map) == {"a", "b"}
        # both glucose consumers -> the glucose substrate registered once
        subs = eco.patches[0].config.substrates
        assert set(subs) == {"glucose"}

    def test_build_requires_species(self):
        with pytest.raises(ValueError):
            build_multi_species_ecosystem({})
