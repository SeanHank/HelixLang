"""Branch-coverage closure tests for sim_runtime/backends/pipelines.py.

Drives every `_run_*` executor edge that the existing long-tail backend
tests do not reach, using local monkeypatches of the heavy plugin entry
points so each case stays fast and hermetic (no network, no BiGG).
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

import helixlang.sim_runtime._coerce as _c
import helixlang.sim_runtime.backends.pipelines as pl
from helixlang.core.ast_nodes import EnzymeDecl, Gene
from helixlang.core.errors import SimConfigError
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser
from helixlang.plugins.runtime.metabolism import (
    ECOLI_CORE_MODEL,
    FluxBalanceAnalysis,
    Reaction,
)
from helixlang.sim_runtime import run
from helixlang.sim_runtime._types import SimResult


def parse(src: str):
    return Parser(list(Lexer(src).tokens())).parse()


def _prog(src: str = "#config backend=fba"):
    return parse(src)


# ============================================================================
# Small fakes shared by the GEM tests
# ============================================================================
class _FakeRxn:
    def __init__(self, stoich, subsystem="exchange"):
        self.reaction_id = ""
        self.stoichiometry = dict(stoich)
        self.subsystem = subsystem
        self.lower_bound = -1000.0
        self.upper_bound = 1000.0


class _FakeModel:
    def __init__(self, reactions=None, metabolites=None):
        self.reactions = reactions or {}
        self.metabolites = set(metabolites or [])
        self.biomass_reaction = None

    def add_reaction(self, rxn):
        rid = getattr(rxn, "reaction_id", getattr(rxn, "id", None))
        self.reactions[rid] = rxn
        for met in rxn.stoichiometry:
            self.metabolites.add(met)

    def set_biomass(self, rid):
        self.biomass_reaction = rid


class _FakeFBA:
    def __init__(self, model, growth=1.2, flux_keys=None):
        self.model = model
        self.growth = growth
        self._fluxes = flux_keys or {
            "BIOMASS": 1.0, "EX_glc__D_e": -10.0, "PDH": 5.0}
        self.levels = None
        self.capacity = None

    def solve(self, objective="biomass", maximize=True):
        return dict(self._fluxes)

    def analyze(self):
        return {
            "growth_rate_per_hour": self.growth,
            "biomass_per_glucose": 0.21,
            "key_fluxes": dict(self._fluxes),
        }

    def set_uptake(self, met, rate):
        self.model.metabolites.add(met)

    def set_enzyme_levels(self, levels):
        self.levels = levels

    def set_enzyme_capacity(self, capacity):
        self.capacity = capacity


class _FakeBatch:
    def __init__(self, entries, **kwargs):
        self._entries = list(entries)
        self.config = SimpleNamespace()

    def step(self):
        if not self._entries:
            return {"growth_rate": 0.0}
        return self._entries.pop(0)


def _gem_result(consensus=True, gapfill=True, grn=True, kcat=True,
                ann=4, stages=5, miss_annotations=False):
    from helixlang.plugins.gem.grn_inference import (
        EvidenceLevel,
        GRNInferenceResult,
        RegulatoryEdge,
    )

    class _Consensus:
        gene_reaction_rules = {
            "R_glk": ["g_glk", "g_zwf"],
            "R_pgi": "g_pgi",
        }

    edges = [
        RegulatoryEdge(
            tf_id="g_cra", target_gene="g_glk", regulation_type="repression",
            evidence_level=EvidenceLevel.PREDICTED, target_reaction="R_glk"),
    ]
    out = dict(
        annotated_genes=ann,
        consensus=_Consensus() if consensus else None,
        gapfill=(SimpleNamespace(added_reactions=[
            SimpleNamespace(equation="2 h2o_e + atp_c -> 2 h2o + adp_c + pi_c",
                            reaction_id="R_TST"),
        ]) if gapfill else None),
        grn=GRNInferenceResult(regulatory_edges=edges) if grn else None,
        kcat_predictions=[{"reaction": "R_glk", "kcat": 100.0}] if kcat else [],
        km_estimates=[{"reaction": "R_glk", "km": 0.5}] if kcat else [],
        final_reaction_count=12,
        stages_completed=stages,
        warnings=[],
        errors=[],
        summary=lambda: {"genome": "x", "gaps": 0},
    )
    if not miss_annotations:
        out["annotations"] = []
    return SimpleNamespace(**out)


def _bind_module(monkeypatch, module, name, value):
    monkeypatch.setattr(_import(module), name, value)


def _import(module):
    import importlib
    return importlib.import_module(module)


# ============================================================================
# whole_cell / dfba
# ============================================================================
def test_whole_cell_rejects_enzyme_kcat_override():
    with pytest.raises(SimConfigError, match="kcat overrides"):
        run(parse(
            "#config backend=whole_cell\n"
            "#enzyme gene=g1 reaction=R1 kcat=1.0\n"
            "#gene name=g1\nATG TAA\n#end\n"))


def test_dfba_oxygen_override_sets_bound(monkeypatch):
    model = deepcopy(ECOLI_CORE_MODEL)
    model.add_reaction(Reaction(
        id="EX_o2", name="O2 exchange",
        stoichiometry={"o2_e": -1.0},
        lower_bound=-1000.0, upper_bound=1000.0,
        subsystem="exchange"))
    fba = FluxBalanceAnalysis(model)
    prog = parse(
        "#config backend=fba\n"
        "#config fba_oxygen_max=3.0 fba_steps=2 fba_dt_h=0.1\n"
        "#config fba_initial_biomass_gdw=0.05 fba_glucose_mm=5.0\n")
    result = pl._run_dfba(prog, fba, model)
    assert result.backend == "fba"
    assert result.meta["dynamic"] is True
    assert model.reactions["EX_o2"].lower_bound != 0.0


# ============================================================================
# Long-tail #sim backends: missing branch edges
# ============================================================================
def test_consortium_default_output_columns():
    result = run(parse("""
#config backend=fba
#sim kind=consortium grid_width=8 grid_height=8
#sim sensors=4 producers=4 actuators=4 steps=5 seed=1
"""))
    assert result is not None and result.backend == "consortium"
    assert result.meta["kind"] == "consortium"


def test_digital_evolution_default_target():
    result = run(parse("""
#config backend=fba
#sim kind=digital_evolution population_size=20 genome_length=6
#sim substitution_rate=0.02 generations=5 seed=3
"""))
    assert result is not None and result.backend == "digital_evolution"
    assert len(result.meta["fittest_genome"]) > 0


def test_stochastic_fano_mode():
    result = run(parse("""
#config backend=fba
#sim kind=stochastic mode=fano k_on=1.0 k_off=1.0 burst_size=5.0
#sim output=mode,fano,on_fraction
"""))
    assert result is not None and result.backend == "stochastic"
    assert result.rows[0]["mode"] == "fano"


def test_codec_benchmark_invalid_scheme():
    with pytest.raises(SimConfigError, match="schemes"):
        run(parse("#sim kind=codec_benchmark schemes=bogus\n"
                  "#config backend=fba"))


def test_synbio_design_requires_protein():
    with pytest.raises(SimConfigError, match="requires a protein"):
        run(parse("#sim kind=synbio_design\n#config backend=fba") or parse(
            "#config backend=fba\n#sim kind=synbio_design"))


def test_protein_fitness_requires_reference():
    with pytest.raises(SimConfigError, match="requires reference"):
        run(parse("#config backend=fba\n#sim kind=protein_fitness "
                  "variants=A,B"))


def test_protein_fitness_requires_variants():
    with pytest.raises(SimConfigError, match="requires variants"):
        run(parse("#config backend=fba\n#sim kind=protein_fitness "
                  "reference=MAEAEAE"))


def test_protein_fitness_oracle_fallback(monkeypatch):
    calls = []

    def _rank(ref, variants, oracle):
        calls.append(oracle)
        if oracle == "esm2":
            raise RuntimeError("offline")
        return [("MAEAEAE", 1.0), ("MAEAEAK", 0.9)]

    monkeypatch.setattr(pl, "rank_variants", _rank)
    result = run(parse(
        "#config backend=fba\n"
        "#sim kind=protein_fitness reference=MAEAEAE variants=MAEAEAE,MAEAEAK "
        "oracle=esm2\n"))
    assert result is not None and result.meta["oracle"] == "blosum62"
    assert calls == ["esm2", "blosum62"]


def test_protein_structure_requires_sequence():
    with pytest.raises(SimConfigError, match="requires sequence"):
        run(parse("#config backend=fba\n#sim kind=protein_structure"))


def test_fate_switching_mode():
    result = run(parse("""
#config backend=fba
#sim kind=fate_analysis mode=switching scan_w=7.0 switch_resources=0.0,1.0
#sim n_trajectories=20 n_ticks=8 seed=1
#sim output=w,n_stable_states
"""))
    assert result is not None and result.backend == "fate_analysis"
    assert "switching_rates" in result.meta


def test_fate_slowing_mode():
    result = run(parse("""
#config backend=fba
#sim kind=fate_analysis mode=slowing w_values=1.0,2.0,3.0,4.0
#sim n_trajectories=20 n_ticks=8 seed=1
#sim output=w,n_stable_states
"""))
    assert result is not None and result.backend == "fate_analysis"
    assert "critical_slowing_down" in result.meta


class _FakeDirected:
    oracle_name = "blosum62"
    initial_fitness = 0.5
    guided_recovery = 0.8
    guided_gain = 2.0
    baseline_gain = 1.0
    spearman_rho = 0.9
    final_best_sequence = "ATG"
    guided_cumulative_best = 3.0
    baseline_cumulative_best = 1.5


def test_directed_evolution_oracle_fallback(monkeypatch):
    def _guide(oracle=None, **kwargs):
        if oracle == "esm2":
            raise RuntimeError("offline")
        return _FakeDirected()

    monkeypatch.setattr(pl, "guided_directed_evolution", _guide)
    result = run(parse(
        "#config backend=fba\n"
        "#sim kind=directed_evolution oracle=esm2 rounds=2 library_size=10\n"))
    assert result is not None
    assert result.meta["kind"] == "directed_evolution"
    assert result.rows[0]["oracle"] == "blosum62"


def test_3d_morphology_invalid_preset():
    with pytest.raises(SimConfigError, match="preset"):
        run(parse("#config backend=fba\n"
                  "#sim kind=3d_morphology preset=nope iterations=2"))


def test_3d_morphology_custom_rules():
    result = run(parse(
        "#config backend=fba\n"
        "#sim kind=3d_morphology preset=tree3d rules=F=F+F;x=FF "
        "iterations=3\n"))
    assert result is not None and result.backend == "3d_morphology"
    assert result.rows[0]["n_vertices"] > 0


def test_codon_usage_requires_orf():
    with pytest.raises(SimConfigError, match="requires at least one #gene"):
        run(parse("#config backend=fba\n#sim kind=codon_usage species=ecoli"))


# ============================================================================
# cardiology (sim_extensions-driven) error branches
# ============================================================================
def _cardiology_prog(cycles):
    prog = _prog()
    prog.sim_extensions.update({"kind": "cardiology", "cardiac_cycle": cycles})
    return prog


def test_cardiology_requires_cycle_annotations():
    with pytest.raises(SimConfigError, match="cardiac_cycle"):
        pl._run_cardiology(_cardiology_prog([]))


def test_cardiology_period_out_of_range():
    with pytest.raises(SimConfigError, match="period"):
        pl._run_cardiology(_cardiology_prog([{"period": "7.0"}]))


# ============================================================================
# ode_model
# ============================================================================
def _ode_prog(**extra):
    prog = _prog()
    base = {
        "ode_model": [{"name": "tw", "k1": "0.5", "k2": "0.2",
                       "t_end": "1", "steps": "4"}],
        "ode_species": [
            {"name": "A", "initial": "10", "units": "uM"},
            {"name": "B", "initial": "1", "units": "uM"},
            {"name": "", "initial": "1"},
        ],
        "ode_reaction": [
            {"species": "A", "expr": "(k1*A) + (k2*B)"},
            {"species": "B", "expr": "0.0"},
            {"species": "", "expr": "(k1*A)"},
        ],
    }
    base.update(extra)
    prog.sim_extensions.update({"kind": "ode_model", **base})
    return prog


def test_ode_model_rk4_integrates():
    result = pl._run_ode_model(_ode_prog())
    assert result.backend == "ode_model"
    by_sp = {r["species"]: r for r in result.rows}
    assert set(by_sp) == {"A", "B"}
    assert by_sp["B"]["final"] == by_sp["B"]["initial"]


def test_ode_model_requires_model():
    with pytest.raises(SimConfigError, match="#model"):
        pl._run_ode_model(_ode_prog(ode_model=[]))


def test_ode_model_requires_numeric_k():
    prog = _ode_prog()
    prog.sim_extensions["ode_model"] = [{"name": "x"}]
    with pytest.raises(SimConfigError, match="k1 and k2"):
        pl._run_ode_model(prog)


def test_ode_model_requires_species():
    with pytest.raises(SimConfigError, match="#species"):
        pl._run_ode_model(_ode_prog(ode_species=[]))


def test_ode_model_requires_named_species():
    with pytest.raises(SimConfigError, match="species list is empty"):
        pl._run_ode_model(_ode_prog(ode_species=[{"initial": "1"}]))


# ============================================================================
# human_simulation dispatch: legacy + virtual patient
# ============================================================================
class _FakeLegacyHuman:
    def run(self):
        return self

    auc_plasma = 42.0
    time_in_therapeutic_range_fraction = 0.8
    overall_efficacy_score = 0.9
    toxicity_events = []
    therapeutic_response_time_h = 3.0
    time_h = [0.0, 1.0]


class _FakeVpResult:
    def __init__(self, n=2):
        self.time_h = [0.0, 1.0][:n]
        self.systolic_bp = [120.0] * n
        self.diastolic_bp = [80.0] * n
        self.heart_rate = [70.0] * n
        self.temperature = [36.9] * n
        self.spo2_pct = [98.0] * n
        self.respiratory_rate = [14.0] * n
        self.qtc_ms = [420.0] * n
        self.alt = [20.0] * n
        self.ast = [21.0] * n
        self.creatinine = [0.9] * n
        self.egfr = [95.0] * n
        self.wbc = [6.0] * n
        self.hemoglobin = [14.0] * n
        self.platelets = [250.0] * n
        self.glucose = [90.0] * n
        self.hba1c = [5.2] * n
        self.crp = [1.0] * n
        self.bilirubin = [0.6] * n
        self.albumin = [4.2] * n
        self.inr = [1.0] * n
        self.sodium = [140.0] * n
        self.potassium = [4.2] * n
        self.lactate = [1.1] * n
        self.calcium = [9.4] * n
        self.phosphate = [3.4] * n
        self.chloride = [103.0] * n
        self.bicarbonate = [24.0] * n
        self.ldl = [100.0] * n
        self.hdl = [50.0] * n
        self.triglycerides = [120.0] * n
        self.disease_severity = [0.4] * n
        self.weight_kg = [70.0] * n
        self.cortisol = [10.0] * n
        self.insulin = [8.0] * n
        self.glucose_endocrine = [90.0] * n
        self.tsh = [2.1] * n
        self.ft4 = [1.2] * n
        self.il6 = [0.4] * n
        self.tnf_alpha = [0.3] * n
        self.neutrophils = [4.0] * n
        self.tumor_volume = [0.0] * n
        self.nephron_mass = [1.0] * n
        self.fibrosis_stage = [0.0] * n
        self.beta_cell_function = [1.0] * n
        self.drug_concentrations = {"warfarin": [1.0, 0.5][:n]}
        self.ddi_alerts = []
        self.clinical_events = []
        self.overall_efficacy_score = 0.7
        self.total_toxicity_events = 0

    def summary(self):
        return {"peak": 1.0}


def test_human_simulation_legacy_dispatch(monkeypatch):
    _bind_module(monkeypatch, "helixlang.plugins.human.simulation",
                 "HumanSimulation", lambda config: _FakeLegacyHuman())
    prog = _prog()
    prog.sim_extensions.update({"kind": "human", "duration_days": "2"})
    result = pl._run_human_simulation(prog)
    assert result.backend == "human"
    assert result.rows[0]["auc_plasma"] == 42.0


def test_human_simulation_virtual_patient(monkeypatch):
    _bind_module(monkeypatch, "helixlang.plugins.human.genotype",
                 "create_default_genotype", lambda: SimpleNamespace())
    _bind_module(monkeypatch, "helixlang.plugins.human.virtual_patient",
                 "VirtualPatientConfig",
                 lambda **kwargs: SimpleNamespace(**kwargs))
    _bind_module(monkeypatch, "helixlang.plugins.human.virtual_patient",
                 "VirtualPatient", lambda config: SimpleNamespace(run=lambda: _FakeVpResult(n=2)))
    monkeypatch.setattr(pl, "_build_genotype_from_helix", lambda *a, **k: None)
    monkeypatch.setattr(pl, "_build_traits_from_helix", lambda ext: {})
    monkeypatch.setattr(pl, "_build_disease_from_helix", lambda ext: None)
    monkeypatch.setattr(pl, "_build_drugs_from_helix", lambda ext: [])
    monkeypatch.setattr(pl, "_build_pd_from_helix", lambda ext: {})
    monkeypatch.setattr(pl, "_build_qsp_bindings_from_helix",
                        lambda ext: {"qsp_bindings": []})
    monkeypatch.setattr(pl, "_build_endocrine_config_from_helix",
                        lambda ext: {"endocrine_configs": []})
    monkeypatch.setattr(pl, "_build_immune_config_from_helix",
                        lambda ext: {"immune_configs": []})
    monkeypatch.setattr(pl, "_build_tumor_biopsy_from_helix", lambda ext: None)

    prog = _prog()
    prog.sim_extensions.update({"kind": "human", "person_age": "35",
                                "duration_days": "2"})
    result = pl._run_human_simulation(prog)
    assert result.backend == "human"
    assert result.meta["kind"] == "human_virtual_patient"
    assert "drug_warfarin" in result.columns
    assert result.rows[0]["drug_warfarin"] == 1.0


def test_human_virtual_patient_auto_infers_pd(monkeypatch):
    _bind_module(monkeypatch, "helixlang.plugins.human.genotype",
                 "create_default_genotype", lambda: SimpleNamespace())
    _bind_module(monkeypatch, "helixlang.plugins.human.virtual_patient",
                 "VirtualPatientConfig",
                 lambda **kwargs: SimpleNamespace(**kwargs))
    _bind_module(monkeypatch, "helixlang.plugins.human.virtual_patient",
                 "VirtualPatient", lambda config: SimpleNamespace(run=lambda: _FakeVpResult(n=1)))
    monkeypatch.setattr(pl, "_build_genotype_from_helix", lambda *a, **k: None)
    monkeypatch.setattr(pl, "_build_traits_from_helix", lambda ext: {})
    monkeypatch.setattr(pl, "_build_disease_from_helix", lambda ext: None)
    monkeypatch.setattr(pl, "_build_drugs_from_helix",
                        lambda ext: [SimpleNamespace(
                            molecule=SimpleNamespace(
                                name="war", target_protein="PTGS1",
                                binding_affinity_kd_um=1.0,
                                molecular_weight_da=180.0))])
    monkeypatch.setattr(pl, "_build_pd_from_helix", lambda ext: {})
    monkeypatch.setattr(pl, "_build_qsp_bindings_from_helix",
                        lambda ext: {"qsp_bindings": []})
    monkeypatch.setattr(pl, "_build_endocrine_config_from_helix",
                        lambda ext: {"endocrine_configs": []})
    monkeypatch.setattr(pl, "_build_immune_config_from_helix",
                        lambda ext: {"immune_configs": []})
    monkeypatch.setattr(pl, "_build_tumor_biopsy_from_helix", lambda ext: None)

    prog = _prog()
    prog.sim_extensions.update({"kind": "human", "person_age": "35",
                                "duration_days": "1"})
    result = pl._run_human_simulation(prog)
    assert result.backend == "human"
    assert result.meta["time_points"] == 1


# ============================================================================
# population: GEM auto-attach branches + trace_streaming
# ============================================================================
class _FakePop:
    def __init__(self, cells, config, seed=None):
        self.config = config
        self.trace = [("t", {"x": 1})]

    def step(self):
        return {"alive": 1, "divisions": 0}

    def dfba_stratification(self):
        return {"core_glucose_mm": 10.0, "edge_glucose_mm": 9.0}

    def colony_observables(self):
        return {"doubling_times_h": [1.0]}

    @staticmethod
    def dfba_stratification_static():  # pragma: no cover
        return {}


def test_population_gem_auto_attach_success(monkeypatch):
    _bind_module(monkeypatch, "helixlang.plugins.apps.gem_pipeline",
                 "run_gem_pipeline",
                 lambda **kwargs: SimpleNamespace(
                     metabolic_model=ECOLI_CORE_MODEL, growth_rate=0.8))
    monkeypatch.setattr(pl, "_seed_cells", lambda config, size: [1, 2])
    monkeypatch.setattr(pl, "CellPopulation3D", _FakePop)
    prog = _prog("#config backend=population #config dfba=true "
                 "#media nutrient=GLC concentration=10.0 "
                 "#config sim genome=nm #config ticks=2 dfba=true")
    prog.sim_extensions["genome"] = "nm"
    result = pl._run_population(prog)
    assert result.backend == "population"
    assert result.meta["colony_observables"]["doubling_times_h"] == [1.0]
    assert result.meta["genome"]["genes"] > 0


def test_population_gem_auto_attach_failure_raises(monkeypatch):
    _bind_module(monkeypatch, "helixlang.plugins.apps.gem_pipeline",
                 "run_gem_pipeline",
                 lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(pl, "_seed_cells", lambda config, size: [1])
    monkeypatch.setattr(pl, "CellPopulation3D", _FakePop)
    prog = _prog("#config backend=population #config dfba=true "
                 "#media nutrient=GLC concentration=10.0 "
                 "#config ticks=2 dfba=true")
    prog.sim_extensions["genome"] = "nm"
    with pytest.raises(Exception, match="GEM for genome"):
        pl._run_population(prog)


def test_population_trace_streaming_meta(monkeypatch):
    _bind_module(monkeypatch, "helixlang.plugins.apps.gem_pipeline",
                 "run_gem_pipeline",
                 lambda **kwargs: SimpleNamespace(
                     metabolic_model=ECOLI_CORE_MODEL, growth_rate=0.5))
    monkeypatch.setattr(pl, "_seed_cells", lambda config, size: [1])
    monkeypatch.setattr(pl, "CellPopulation3D", _FakePop)
    prog = _prog("#config backend=population\n"
                 "#media nutrient=GLC concentration=10.0\n"
                 "#config ticks=1 dfba=true trace_streaming=true\n")
    prog.sim_extensions["genome"] = "nm"
    result = pl._run_population(prog)
    assert result.meta["trace"] == [("t", {"x": 1})]


# ============================================================================
# ecosystem
# ============================================================================
class _FakeEco:
    def __init__(self, config, rows=None):
        self.config = config
        self._rows = rows if rows is not None else [
            {"tick": 0, "a__biomass": 1.0}]

    def run(self):
        return list(self._rows)

    def run_generations(self):
        return list(self._rows)

    def neutral_vs_niche(self):
        return {"niche": 0.6}

    def summary(self):
        return {"species": 2}


def _ecosystem_prog(species=True, patches=True, ticks=True):
    prog = _prog("#config backend=fba" + ("\n#config ticks=2" if ticks else ""))
    if species:
        prog.sim_extensions.update({
            "species.a.substrate": "glucose",
            "species.a.vmax": "0.05",
            "species.b.diet": "a:0.8",
            "species.c.photo": "true",
        })
    if patches:
        prog.sim_extensions.update({
            "patch.p.width": "8",
            "patch.p.height": "8",
            "patch.p.substrate.nh4.initial": "5",
        })
    return prog


def test_ecosystem_default_run(monkeypatch):
    monkeypatch.setattr(pl, "Ecosystem", _FakeEco)
    result = pl._run_ecosystem(_ecosystem_prog())
    assert result.backend == "ecosystem"
    assert result.meta["neutral_niche"] == {"niche": 0.6}


def test_ecosystem_run_generations(monkeypatch):
    monkeypatch.setattr(pl, "Ecosystem", _FakeEco)
    prog = _ecosystem_prog()
    prog.sim_extensions.update({"generations": "2"})
    result = pl._run_ecosystem(prog)
    assert result.backend == "ecosystem"
    assert result.meta["summary"] == {"species": 2}


def test_ecosystem_requires_species(monkeypatch):
    monkeypatch.setattr(pl, "Ecosystem", _FakeEco)
    with pytest.raises(SimConfigError, match="at least one #species"):
        pl._run_ecosystem(_ecosystem_prog(species=False))


def test_ecosystem_requires_patches(monkeypatch):
    monkeypatch.setattr(pl, "Ecosystem", _FakeEco)
    with pytest.raises(SimConfigError, match="at least one #patch"):
        pl._run_ecosystem(_ecosystem_prog(patches=False))


def test_ecosystem_empty_rows_defaults_columns(monkeypatch):
    monkeypatch.setattr(pl, "Ecosystem",
                        lambda config: _FakeEco(config, rows=[]))
    result = pl._run_ecosystem(_ecosystem_prog())
    assert result.backend == "ecosystem"
    assert result.rows == []
    assert len(result.columns) > 0


# ============================================================================
# population_dbtl
# ============================================================================
class _FakeDbtl:
    def __init__(self, config):
        self.config = config

    def run(self):
        return {
            "rounds": [
                {"round": 0, "best_growth": 0.4, "mean_growth": 0.3,
                 "best_genome": "ATG", "n_tested": 8,
                 "surrogate_best_trait": "t1"},
                {"round": 1, "best_growth": 0.6, "mean_growth": 0.5,
                 "best_genome": "ATGx", "n_tested": 8,
                 "surrogate_best_trait": "t2"},
            ],
            "designed_strain": "A", "round0_growth": 0.4,
            "final_growth": 0.6, "improved": True, "fold_improvement": 1.5,
        }


def test_population_dbtl_loop(monkeypatch):
    monkeypatch.setattr(pl, "PopulationDbtl", _FakeDbtl)
    prog = _prog("#config backend=fba")
    prog.sim_extensions.update({
        "kind": "population_dbtl", "n_rounds": "2",
        "evaluation_ticks": "10",
    })
    result = pl._run_population_dbtl(prog)
    assert result.backend == "population_dbtl"
    assert result.meta["improved"] is True
    assert len(result.rows) == 2


# ============================================================================
# GEM full-model path (_run_gem_full_model)
# ============================================================================
def _gem_full_model(ex_reactions):
    model = _FakeModel(reactions={
        rid: _FakeRxn(stoich, subsystem)
        for rid, (stoich, subsystem) in ex_reactions.items()
    })
    return model


def _patch_full_model(monkeypatch, model, growth=1.2,
                      dynamic_driver=None):
    _bind_module(monkeypatch, "helixlang.plugins.gem.bridge",
                 "build_functional_model_full",
                 lambda organism=None, medium=None: model)
    _bind_module(monkeypatch, "helixlang.plugins.runtime.metabolism",
                 "FluxBalanceAnalysis",
                 lambda m: _FakeFBA(m, growth=growth))
    _bind_module(monkeypatch, "helixlang.plugins.runtime.metabolism",
                 "DynamicFBAConfig",
                 lambda **kw: SimpleNamespace(**kw))
    _bind_module(monkeypatch, "helixlang.plugins.runtime.metabolism",
                 "PhotoautotrophicFluxBalance", dynamic_driver)


def test_gem_full_model_static_override_bounds(monkeypatch):
    model = _gem_full_model({
        "EX_glc_e": ({"glc-D_e": -1.0}, "exchange"),
        "EX_prod_e": ({"prod_e": 1.0}, "exchange"),
        "EX_two_e": ({"a_e": -1.0, "b_e": 1.0}, "exchange"),
        "EX_unrelated_e": ({"cold_e": -1.0}, "exchange"),
        "O2t": ({"o2_e": -1.0, "o2_c": 1.0}, "transport"),
    })
    _patch_full_model(monkeypatch, model)
    result = pl._run_gem_full_model(
        organism="e_coli_k12", medium_name="glucose_minimal",
        dynamic=False, duration=24.0, dt=0.05,
        medium_override={"glc-D_e": -12.0, "prod_e": 5.0},
    )
    assert result.backend == "gem"
    assert result.meta["use_full_model"] is True
    assert model.reactions["EX_glc_e"].lower_bound == -12.0
    assert model.reactions["EX_prod_e"].upper_bound == 5.0
    assert model.reactions["EX_two_e"].upper_bound == 1000.0
    assert model.reactions["EX_unrelated_e"].lower_bound == -1000.0


def test_gem_full_model_max_growth_override(monkeypatch):
    model = _gem_full_model({"EX_glc_e": ({"glc-D_e": -1.0}, "exchange")})
    _patch_full_model(monkeypatch, model, growth=5.0)
    result = pl._run_gem_full_model(
        organism="e_coli_k12", medium_name="glucose_minimal",
        dynamic=False, duration=24.0, dt=0.05,
        medium_override=None, max_growth_rate=0.5,
    )
    assert result.rows[-1]["status"] == "ok"
    assert result.meta["growth_rate_per_hour"] == pytest.approx(0.5)


def test_gem_full_model_dynamic_photo(monkeypatch):
    model = _gem_full_model({"EX_co2_e": ({"co2_e": -1.0}, "exchange")})
    entries = [
        {"co2": 9.0, "growth_rate": 0.12, "biomass": 0.02},
        {"co2": 0.0, "growth_rate": 0.0, "biomass": 0.04},
    ]
    _patch_full_model(monkeypatch, model,
                      dynamic_driver=lambda *a, **kw: _FakeBatch(entries))
    result = pl._run_gem_full_model(
        organism="synechocystis", medium_name="bg11",
        dynamic=True, duration=1.0, dt=0.5,
    )
    assert result.meta["dynamic"] is True
    assert result.meta["growth_rate_per_hour"] == pytest.approx(0.12, abs=1e-6)


def test_gem_full_model_exception(monkeypatch):
    _bind_module(monkeypatch, "helixlang.plugins.gem.bridge",
                 "build_functional_model_full",
                 lambda organism=None, medium=None: (_ for _ in ()).throw(
                     RuntimeError("no BiGG")))
    result = pl._run_gem_full_model(
        organism="e_coli_k12", medium_name="glucose_minimal",
        dynamic=False, duration=1.0, dt=0.1,
    )
    assert result.rows[0]["status"] == "failed"
    assert "no BiGG" in result.meta["errors"][0]


# ============================================================================
# GEM reconstruction path (_run_gem)
# ============================================================================
def _gem_prog(extras, genes=None, enzymes=None, no_genome=False):
    prog = _prog("#config backend=fba")
    base = {} if no_genome else {"gem_genome": "attrs/x.fasta"}
    base["gem_organism"] = "e_coli_k12"
    prog.sim_extensions.update(base)
    prog.sim_extensions.update(extras)
    if genes is not None:
        prog.genes = genes
    if enzymes is not None:
        prog.enzymes = enzymes
    return prog


def _patch_gem(monkeypatch, result=None, model=None, fba=None,
               dflux=None, pflux=None, infer=None, expr_levels=None,
               additions=True):
    if result is None:
        result = _gem_result()
    if model is None:
        model = _FakeModel(metabolites=["ala-L_c", "h2o", "o2_c"])
    if fba is None:
        fba = _FakeFBA(model)
    _bind_module(monkeypatch, "helixlang.plugins.apps.gem_pipeline",
                 "run_gem_pipeline", lambda **kw: result)
    _bind_module(monkeypatch, "helixlang.plugins.gem.bridge",
                 "consensus_to_metabolic_model", lambda consensus: model)
    _bind_module(monkeypatch, "helixlang.plugins.gem.bridge",
                 "_parse_equation_to_stoich",
                 lambda eq: {"h2o": 1.0, "atp_c": -1.0, "adp_c": 1.0})
    _bind_module(monkeypatch, "helixlang.plugins.gem.biomass",
                 "build_biomass_reaction",
                 lambda organism: SimpleNamespace(components=[
                     SimpleNamespace(metabolite_id="ala-L_c", coefficient=-1.0),
                     SimpleNamespace(metabolite_id="xyz_m", coefficient=1.0),
                     SimpleNamespace(metabolite_id="h2o", coefficient=0.0),
                 ]))
    _bind_module(monkeypatch, "helixlang.plugins.runtime.metabolism",
                 "FluxBalanceAnalysis", lambda m: fba)
    _bind_module(monkeypatch, "helixlang.plugins.runtime.metabolism",
                 "DynamicFBAConfig", lambda **kw: SimpleNamespace(**kw))
    _bind_module(monkeypatch, "helixlang.plugins.runtime.metabolism",
                 "DynamicFluxBalance", dflux)
    _bind_module(monkeypatch, "helixlang.plugins.runtime.metabolism",
                 "PhotoautotrophicFluxBalance", pflux)
    _bind_module(monkeypatch, "helixlang.plugins.omics.expression_inference",
                 "ExpressionModel",
                 lambda: SimpleNamespace(promoter_strength={}, rbs_strength={}))
    _bind_module(monkeypatch, "helixlang.plugins.omics.expression_inference",
                 "infer_expression",
                 lambda **kw: dict(expr_levels or {"g_glk": 2.0}))
    _bind_module(monkeypatch, "helixlang.plugins.gem.bridge",
                 "build_enzyme_capacity",
                 lambda consensus, predictions:
                 SimpleNamespace(kcat={"R_glk": 100.0}))
    _bind_module(monkeypatch, "helixlang.plugins.gem.bridge",
                 "apply_regulatory_bounds",
                 lambda model, edges, gpr: len(gpr))
    if additions:
        monkeypatch.setattr(pl, "_add_gem_core_reactions", lambda m: None)
        monkeypatch.setattr(pl, "_add_gem_transport_reactions", lambda m: None)
    return result, model, fba


def test_gem_no_consensus_builds_failed_rows(monkeypatch):
    result, model, fba = _patch_gem(
        monkeypatch, result=_gem_result(consensus=False, grn=False))
    prog = _gem_prog({"gem_medium": "glucose_minimal"})
    out = pl._run_gem(prog)
    assert out.backend == "gem"
    assert out.rows[1]["status"] == "failed"
    assert out.rows[2]["status"] == "skipped"


def test_gem_static_full_pipeline(monkeypatch):
    result, model, fba = _patch_gem(
        monkeypatch,
        result=_gem_result(grn=False, kcat=False))
    prog = _gem_prog({"gem_medium": "glucose_minimal"})
    out = pl._run_gem(prog)
    assert out.rows[-1]["status"] == "ok"
    assert out.meta["growth_rate_per_hour"] == pytest.approx(0.87, abs=1e-6)


def test_gem_dynamic_glucose_expression_and_bounds(monkeypatch):
    entries = [
        {"glucose": 5.0, "growth_rate": 0.8, "biomass": 0.2, "co2": 0.1},
        {"glucose": 0.0, "growth_rate": 0.0, "biomass": 0.5},
    ]
    result, model, fba = _patch_gem(
        monkeypatch, dflux=lambda *a, **kw: _FakeBatch(entries))
    genes = [Gene(name="g_glk", promoter=None, codons=[], orf=[],
                  fields={"expression_level": "2.5"}),
             Gene(name="g_bad", promoter=None, codons=[], orf=[],
                  fields={"expression_level": "abc"}),
             Gene(name="g_nofield", promoter=None, codons=[], orf=[],
                  fields={})]
    enz = [EnzymeDecl(gene="g_e", reaction="R_glk", kcat=250.0),
           EnzymeDecl(gene="g_f", reaction="R_nope", kcat=1.0)]
    prog = _gem_prog({"gem_medium": "glucose_minimal",
                      "gem_dynamic": "true",
                      "gem_expression": "true",
                      "gem_duration": "1",
                      "gem_dt": "0.5"},
                     genes=genes, enzymes=enz)
    out = pl._run_gem(prog)
    assert out.meta["dynamic"] is True
    assert out.meta["final_biomass"] == pytest.approx(0.2)
    assert out.meta["trajectory_steps"] == 2
    assert out.meta["grn_bounds_applied"] == 2
    assert out.meta["growth_rate_per_hour"] == pytest.approx(0.8, abs=1e-6)


def test_gem_dynamic_glucose_no_growing_entry(monkeypatch):
    entries = [{"glucose": 1.0, "growth_rate": 0.0, "biomass": 0.01}]
    result, model, fba = _patch_gem(
        monkeypatch, dflux=lambda *a, **kw: _FakeBatch(entries))
    prog = _gem_prog({"gem_medium": "glucose_minimal",
                      "gem_dynamic": "true",
                      "gem_duration": "1",
                      "gem_dt": "1"})
    out = pl._run_gem(prog)
    assert out.meta["final_biomass"] == pytest.approx(0.01)


def test_gem_photoautotrophic_bg11(monkeypatch):
    entries = [
        {"co2": 8.0, "growth_rate": 0.12, "biomass": 0.03},
        {"co2": 0.0, "growth_rate": 0.0, "biomass": 0.06},
    ]
    result, model, fba = _patch_gem(
        monkeypatch, pflux=lambda *a, **kw: _FakeBatch(entries))
    prog = _gem_prog({"gem_medium": "bg11",
                      "gem_dynamic": "true",
                      "gem_expression": "true",
                      "gem_duration": "1",
                      "gem_dt": "1"})
    out = pl._run_gem(prog)
    assert out.meta["dynamic"] is True
    assert out.meta["final_biomass"] == pytest.approx(0.03)
    assert out.meta["growth_rate_per_hour"] == pytest.approx(0.12, abs=1e-6)


def test_gem_inline_genome_and_override_parsing(monkeypatch):
    result, model, fba = _patch_gem(monkeypatch)
    prog = _gem_prog({
        "gem_inline_genome": "ATGCGTACGTTAA",
        "gem_inline_genes": [("x1", "ATG"), "notatuple"],
        "gem_medium": "glucose_minimal",
        "gem_medium_override": "fe3_e:0.5,o2:bad,glc_e",
        "gem_max_growth_rate": "0.5",
    }, no_genome=True)
    out = pl._run_gem(prog)
    assert out.rows[-1]["status"] == "ok"
    assert out.meta["growth_rate_per_hour"] == pytest.approx(0.5, abs=1e-6)


def test_gem_species_fallback_genome(monkeypatch):
    result, model, fba = _patch_gem(
        monkeypatch, result=_gem_result(grn=False))
    prog = _prog("#config backend=fba")
    prog.sim_extensions.update({
        "species.a.genome": "attr/sp.fa",
    })
    out = pl._run_gem(prog)
    assert out.meta["organism"] == "a"
    assert out.meta["stages_completed"] == 5


def test_gem_missing_genome_raises(monkeypatch):
    result, model, fba = _patch_gem(monkeypatch)
    prog = _prog("#config backend=fba")
    with pytest.raises(SimConfigError, match="GEM backend requires"):
        pl._run_gem(prog)


def test_gem_species_genome_loop_with_and_without_genome(monkeypatch):
    result, model, fba = _patch_gem(monkeypatch, result=_gem_result(grn=False))
    prog = _prog("#config backend=fba")
    prog.sim_extensions.update({
        "species.z.substrate": "glucose",
        "species.a.genome": "attr/sp.fa",
    })
    out = pl._run_gem(prog)
    assert out.meta["organism"] == "a"
    assert out.rows[-1]["status"] == "ok"


def test_gem_species_genome_no_match_raises(monkeypatch):
    result, model, fba = _patch_gem(monkeypatch, result=_gem_result(grn=False))
    prog = _prog("#config backend=fba")
    prog.sim_extensions.update({
        "species.z.substrate": "glucose",
        "species.w.substrate": "ammonia",
    })
    with pytest.raises(SimConfigError, match="GEM backend requires"):
        pl._run_gem(prog)


def test_gem_species_genome_keeps_organism(monkeypatch):
    result, model, fba = _patch_gem(monkeypatch, result=_gem_result(grn=False))
    prog = _prog("#config backend=fba")
    prog.sim_extensions.update({
        "gem_organism": "myorg",
        "species.a.genome": "attr/sp.fa",
    })
    out = pl._run_gem(prog)
    assert out.meta["organism"] == "myorg"
    assert out.rows[-1]["status"] == "ok"


def test_gem_single_inline_genome_no_genes(monkeypatch):
    result, model, fba = _patch_gem(monkeypatch, result=_gem_result(grn=False))
    prog = _gem_prog({
        "gem_inline_genome": "ATGCGTACGTTAA",
        "gem_medium": "glucose_minimal",
    }, no_genome=True)
    out = pl._run_gem(prog)
    assert out.rows[-1]["status"] == "ok"


def test_gem_use_full_model_dispatch(monkeypatch):
    monkeypatch.setattr(pl, "_run_gem_full_model",
                        lambda **kw: SimpleNamespace(
                            backend="gem", rows=[], columns=[], meta={}))
    result, model, fba = _patch_gem(monkeypatch)
    prog = _gem_prog({"gem_use_full_model": "true"})
    out = pl._run_gem(prog)
    assert out.backend == "gem"


def test_gem_invalid_max_growth_override(monkeypatch):
    result, model, fba = _patch_gem(monkeypatch, result=_gem_result(grn=False))
    prog = _gem_prog({"gem_medium": "glucose_minimal",
                      "gem_max_growth_rate": "notanumber"})
    out = pl._run_gem(prog)
    assert out.rows[-1]["status"] == "ok"


def test_gem_gapfill_empty(monkeypatch):
    result, model, fba = _patch_gem(
        monkeypatch, result=_gem_result(gapfill=False, grn=False, kcat=False))
    prog = _gem_prog({"gem_medium": "glucose_minimal"})
    out = pl._run_gem(prog)
    assert out.rows[-1]["status"] == "ok"


def test_gem_gapfill_skips_unparsable_equation(monkeypatch):
    result, model, fba = _patch_gem(monkeypatch, result=_gem_result(grn=False))
    _bind_module(monkeypatch, "helixlang.plugins.gem.bridge",
                 "_parse_equation_to_stoich", lambda eq: {})
    prog = _gem_prog({"gem_medium": "glucose_minimal"})
    out = pl._run_gem(prog)
    assert out.rows[-1]["status"] == "ok"


def test_gem_biomass_empty_stoich(monkeypatch):
    result, model, fba = _patch_gem(
        monkeypatch, model=_FakeModel(metabolites=[]),
        result=_gem_result(grn=False, kcat=False))
    prog = _gem_prog({"gem_medium": "glucose_minimal"})
    out = pl._run_gem(prog)
    assert out.rows[-1]["status"] == "ok"


def test_gem_expression_without_grn(monkeypatch):
    result, model, fba = _patch_gem(monkeypatch, result=_gem_result(grn=False))
    prog = _gem_prog({"gem_medium": "glucose_minimal",
                      "gem_expression": "true"})
    out = pl._run_gem(prog)
    assert out.rows[-1]["status"] == "ok"


def test_gem_expression_without_annotations(monkeypatch):
    entries = [{"glucose": 5.0, "growth_rate": 0.5, "biomass": 0.1}]
    result, model, fba = _patch_gem(monkeypatch, result=_gem_result(
        miss_annotations=True),
        dflux=lambda *a, **kw: _FakeBatch(entries))
    prog = _gem_prog({"gem_medium": "glucose_minimal",
                      "gem_expression": "true",
                      "gem_dynamic": "true",
                      "gem_duration": "1",
                      "gem_dt": "1"})
    out = pl._run_gem(prog)
    assert out.rows[-1]["status"] == "ok"
    assert any("Expression inference skipped" in w
               for w in out.meta["warnings"])


def test_gem_kcat_capacity_raise(monkeypatch):
    result, model, fba = _patch_gem(monkeypatch)
    _bind_module(monkeypatch, "helixlang.plugins.gem.bridge",
                 "build_enzyme_capacity",
                 lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kcat")))
    prog = _gem_prog({"gem_medium": "glucose_minimal",
                      "gem_expression": "true"})
    out = pl._run_gem(prog)
    assert out.rows[-1]["status"] == "ok"
    assert any("Enzyme capacity setup skipped" in w
               for w in out.meta["warnings"])


def test_gem_grn_without_consensus(monkeypatch):
    result, model, fba = _patch_gem(
        monkeypatch, result=_gem_result(consensus=False, kcat=False))
    prog = _gem_prog({"gem_medium": "glucose_minimal"})
    out = pl._run_gem(prog)
    assert out.rows[1]["status"] == "failed"
    assert out.rows[-1]["status"] == "skipped"


def test_gem_grn_bounds_raise(monkeypatch):
    result, model, fba = _patch_gem(monkeypatch)
    _bind_module(monkeypatch, "helixlang.plugins.gem.bridge",
                 "apply_regulatory_bounds",
                 lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    prog = _gem_prog({"gem_medium": "glucose_minimal"})
    out = pl._run_gem(prog)
    assert out.rows[-1]["status"] == "ok"
    assert any("GRN regulatory bounds skipped" in w
               for w in out.meta["warnings"])


def test_gem_fba_simulation_raises(monkeypatch):
    result, model, fba = _patch_gem(monkeypatch, result=_gem_result(grn=False))
    _bind_module(monkeypatch, "helixlang.plugins.gem.bridge",
                 "consensus_to_metabolic_model",
                 lambda c: (_ for _ in ()).throw(RuntimeError("bad model")))
    prog = _gem_prog({"gem_medium": "glucose_minimal"})
    out = pl._run_gem(prog)
    assert out.rows[-1]["status"].startswith("failed")
    assert any("FBA simulation failed" in w for w in out.meta["warnings"])

# ============================================================================
# whole_cell / ode / spatial_dfba / population / coerce extra branches
# ============================================================================
class _FakeVirtualCell:
    def __init__(self, genome, grn, config=None):
        self.name = "vc"

    def run(self, ticks):
        return [{"time": 0.0, "biomass": 1.0}]


def test_whole_cell_enzyme_without_kcat(monkeypatch):
    monkeypatch.setattr(pl, "VirtualCell", _FakeVirtualCell)
    monkeypatch.setattr(pl, "_build_grn", lambda *a, **k: {})
    monkeypatch.setattr(pl, "_build_virtual_cell_config",
                        lambda program: SimpleNamespace(seed=1))
    result = run(parse(
        "#config backend=whole_cell\n"
        "#config ticks=1\n"
        "#enzyme gene=g1 reaction=R1\n"
        "#gene name=g1\nATG TAA\n#end\n"))
    assert result.backend == "whole_cell"
    assert len(result.rows) == 1


def test_codon_usage_early_stop():
    result = run(parse(
        "#sim kind=codon_usage\n"
        "#gene name=g1\nATG TAA GCG\n#end\n"))
    assert result.backend == "codon_usage"
    assert result.rows[0]["protein"] == "M"


def test_cardiology_success_cycle():
    prog = _prog("#config backend=fba")
    prog.sim_extensions["cardiac_cycle"] = [
        {"period": "0.8", "conduction": "block"},
        {"period": "1.1"},
    ]
    result = pl._run_cardiology(prog)
    assert result.backend == "cardiology"
    assert len(result.rows) == 2
    assert result.rows[0]["conduction"] == "block"
    assert result.meta["cardiac_cycles"] == 2


def test_ecosystem_no_ticks_config(monkeypatch):
    monkeypatch.setattr(pl, "Ecosystem", _FakeEco)
    prog = _ecosystem_prog(ticks=False)
    prog.config.ticks = None
    result = pl._run_ecosystem(prog)
    assert result.backend == "ecosystem"
    assert result.meta["summary"] == {"species": 2}


def test_ecosystem_gem_driven(monkeypatch):
    _bind_module(monkeypatch, "helixlang.plugins.apps.gem_pipeline",
                 "run_gem_pipeline",
                 lambda **kw: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(pl, "Ecosystem", _FakeEco)
    prog = _ecosystem_prog()
    prog.sim_extensions.update({"gem_driven": "true",
                                "species.a.genome": "attrs/x.fasta"})
    result = pl._run_ecosystem(prog)
    assert result.backend == "ecosystem"


def test_gem_full_photo_no_growing_entry(monkeypatch):
    model = _gem_full_model({"EX_co2_e": ({"co2_e": -1.0}, "exchange")})
    entries = [
        {"co2": 0.0, "growth_rate": 0.0, "biomass": 0.05},
        {"co2": 0.004, "growth_rate": 0.0, "biomass": 0.06},
    ]
    _patch_full_model(monkeypatch, model,
                      dynamic_driver=lambda *a, **kw: _FakeBatch(entries))
    result = pl._run_gem_full_model(
        organism="synechocystis", medium_name="bg11",
        dynamic=True, duration=2.0, dt=1.0,
    )
    assert result.meta["growth_rate_per_hour"] == 0.0


def test_gem_photo_no_growing_entry(monkeypatch):
    entries = [{"co2": 0.002, "growth_rate": 0.0, "biomass": 0.05}]
    result, model, fba = _patch_gem(
        monkeypatch, pflux=lambda *a, **kw: _FakeBatch(entries))
    prog = _gem_prog({"gem_medium": "bg11",
                      "gem_dynamic": "true",
                      "gem_expression": "true",
                      "gem_duration": "1",
                      "gem_dt": "1"})
    out = pl._run_gem(prog)
    assert out.meta["final_biomass"] == 0.05


def test_spatial_dfba_default_columns():
    prog = _prog("#config backend=fba")
    prog.sim_extensions.update({"kind": "spatial_dfba", "steps": "1"})
    result = pl._run_spatial_dfba(prog)
    assert result.backend == "fba"
    assert "(default)" not in str(result.columns)


def test_population_gem_attach_optin_silent(monkeypatch):
    _bind_module(monkeypatch, "helixlang.plugins.apps.gem_pipeline",
                 "run_gem_pipeline",
                 lambda **kw: (_ for _ in ()).throw(RuntimeError("offline")))
    import helixlang.core.fidelity as fid
    monkeypatch.setattr(fid, "opt_in", lambda *a, **k: True)
    monkeypatch.setattr(pl, "_seed_cells", lambda config, size: [1])
    monkeypatch.setattr(pl, "CellPopulation3D", _FakePop)
    prog = _prog("#config backend=population\n"
                 "#media nutrient=GLC concentration=10.0\n"
                 "#config ticks=1 dfba=true\n")
    prog.sim_extensions["genome"] = "attrs/x.fasta"
    result = pl._run_population(prog)
    assert result.backend == "population"


def test_ode_species_without_reaction():
    prog = _prog("#config backend=fba")
    prog.sim_extensions["ode_model"] = [{"name": "m1", "k1": "0.1", "k2": "0.2"}]
    prog.sim_extensions["ode_species"] = [
        {"name": "a", "initial": "1.0"},
        {"name": "b", "initial": "0.5"},
    ]
    prog.sim_extensions["ode_reaction"] = [
        {"species": "a", "expr": "a * b"},
    ]
    result = pl._run_ode_model(prog)
    assert result.backend == "ode_model"
    assert result.rows[0]["final"] > 0.0
    assert result.rows[1]["final"] == pytest.approx(0.5)


def test_human_vp_auto_infer_drug_without_target(monkeypatch):
    _bind_module(monkeypatch, "helixlang.plugins.human.genotype",
                 "create_default_genotype", lambda: SimpleNamespace())
    _bind_module(monkeypatch, "helixlang.plugins.human.virtual_patient",
                 "VirtualPatientConfig",
                 lambda **kwargs: SimpleNamespace(**kwargs))
    _bind_module(monkeypatch, "helixlang.plugins.human.virtual_patient",
                 "VirtualPatient",
                 lambda config: SimpleNamespace(run=lambda: _FakeVpResult(n=1)))
    _bind_module(monkeypatch, "helixlang.plugins.human.pharmacodynamics",
                 "infer_pd_from_drug", lambda *a, **k: {"source": "infer"})
    monkeypatch.setattr(pl, "_build_genotype_from_helix", lambda *a, **k: None)
    monkeypatch.setattr(pl, "_build_traits_from_helix", lambda ext: {})
    monkeypatch.setattr(pl, "_build_disease_from_helix", lambda ext: None)
    monkeypatch.setattr(pl, "_build_drugs_from_helix",
                        lambda ext: [SimpleNamespace(
                            molecule=SimpleNamespace(
                                name="war", target_protein="PTGS1",
                                binding_affinity_kd_um=1.0,
                                molecular_weight_da=180.0)),
                            SimpleNamespace(
                            molecule=SimpleNamespace(
                                name="placebo", target_protein=None))])
    monkeypatch.setattr(pl, "_build_pd_from_helix", lambda ext: {})
    monkeypatch.setattr(pl, "_build_qsp_bindings_from_helix",
                        lambda ext: {"qsp_bindings": []})
    monkeypatch.setattr(pl, "_build_endocrine_config_from_helix",
                        lambda ext: {"endocrine_configs": []})
    monkeypatch.setattr(pl, "_build_immune_config_from_helix",
                        lambda ext: {"immune_configs": []})
    monkeypatch.setattr(pl, "_build_tumor_biopsy_from_helix", lambda ext: None)

    prog = _prog()
    prog.sim_extensions.update({"kind": "human", "person_age": "35",
                                "duration_days": "1"})
    result = pl._run_human_simulation(prog)
    assert result.backend == "human"
    assert result.meta["kind"] == "human_virtual_patient"


# ============================================================================
# _coerce / _types module coercion helpers
# ============================================================================


def test_coerce_invalid_bool_value():
    with pytest.raises(SimConfigError, match="expected true/false"):
        _c._opt_bool({"x": "maybe"}, "x", False)


def test_opt_none_variants():
    assert _c._opt_int_or_none({"x": "none"}, "x", 5) is None
    assert _c._opt_int_or_none({}, "x", 5) == 5
    assert _c._opt_float_or_none({"x": "none"}, "x", 5.0) is None
    assert _c._opt_float_or_none({}, "x", 5.0) == 5.0


def test_coerce_float_dict_errors():
    with pytest.raises(SimConfigError, match="k=v pairs"):
        _c._opt_float_dict({"x": "a=1.0,noeq"}, "x", {})
    with pytest.raises(SimConfigError, match="empty name"):
        _c._opt_float_dict({"x": "=1.0"}, "x", {})


def test_coerce_float_list_skips_empty_segments():
    assert _c._opt_float_list({"x": "1.0,,2.5"}, "x", ()) == (1.0, 2.5)


def test_coerce_replicon_specs():
    from helixlang.plugins.runtime.virtual_cell import RepliconSpec
    assert _c._opt_replicon_specs({"x": "pBR322:20,,pUC19:500"}, "x") == {
        "pBR322": RepliconSpec(kind="plasmid", copy_number=20),
        "pUC19": RepliconSpec(kind="plasmid", copy_number=500),
    }
    with pytest.raises(SimConfigError, match="copy number"):
        _c._opt_replicon_specs({"x": "p:0"}, "x")


def test_select_columns_fallback_union():
    prog = _prog("#config backend=fba")
    cols = _c._select_columns(prog, [{"a": 1, "b": 2}, {"b": 3}])
    assert cols == ["a", "b"]


def test_sim_result_to_dict():
    r = SimResult(backend="fba", columns=["x"], rows=[{"x": 1}],
                  meta={"k": 2})
    assert r.to_dict() == {
        "backend": "fba", "columns": ["x"], "rows": [{"x": 1}],
        "meta": {"k": 2}, "provenance": {},
    }
