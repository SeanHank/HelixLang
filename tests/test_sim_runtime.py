"""Simulation-backend adapter tests (doc/helix-language-wiring.md §13).

Covers the W-2 (whole_cell + fba), W-3 (population), W-4
(calibration/benchmark + /api/sim/run) and W-6 (#sim long-tail backends)
gates: the sim_runtime adapter maps parsed programs onto the quantitative
simulation stack.
"""
from __future__ import annotations

import pytest

from helixlang.errors import SimConfigError
from helixlang.lexer import Lexer
from helixlang.parser import Parser
from helixlang.sim_runtime import (
    _build_grn,
    _build_virtual_cell_config,
    run,
)


def parse(src: str):
    return Parser(list(Lexer(src).tokens())).parse()


# ============================================================================
# Dispatch
# ============================================================================
def test_classic_backend_returns_none():
    """backend=classic (and the default) keep the bytecode path."""
    assert run(parse("#config backend=classic")) is None
    assert run(parse("#config ticks=4")) is None


def test_unknown_backend_raises_sim_config_error():
    with pytest.raises(SimConfigError, match="unknown backend"):
        run(parse("#config backend=transcriptomics"))


# ============================================================================
# W-2: whole_cell backend (VirtualCell, Phases 1-4)
# ============================================================================
_WHOLE_CELL_SRC = """
#promoter name=p_constitutive strength=-0.4
#gene name=gltA promoter=p_constitutive chromosome=0.1
ATG GCT GGT GCT TAA
#end
#gene name=zwf promoter=p_constitutive chromosome=0.7 initial_level=1.0
ATG GCT GGT GCT TAA
#end
#regulate gltA -> zwf strength=+0.6
#media nutrient=GLC concentration=10.0
#config backend=whole_cell
#config division_rule=adder adder_volume_um3=1.6 adder_noise_std=0.1
#config replication_mode=cooper_helmstetter doubling_time_min=30
#config c_period_min=20 d_period_min=10
#config protein_maturation_mode=chaperone frac_cotranslational_fold=0.7
#config enzyme_capacity=true
#config k_fold=0.7
#config seed=0
#config ticks=40
#config output=age,energy,alive,divisions,volume_um3,added_volume_um3,phase
"""


def test_whole_cell_maps_annotations():
    """#gene/#promoter/#regulate/#media map onto the VirtualCell."""
    prog = parse(_WHOLE_CELL_SRC)
    cfg = _build_virtual_cell_config(prog)
    grn = _build_grn(prog, cfg.seed)

    assert cfg.division_rule == "adder"
    assert cfg.adder_volume_um3 == pytest.approx(1.6)
    assert cfg.replication_mode == "cooper_helmstetter"
    assert cfg.protein_maturation_mode == "chaperone"
    assert cfg.enzyme_capacity_enabled is True
    assert cfg.uptake == {"GLC": 10.0}
    assert cfg.chromosome_map == {"gltA": 0.1, "zwf": 0.7}
    # k_fold=0.7 -> fold_rate_per_min derived from the equilibrium
    assert cfg.fold_rate_per_min is not None

    assert "gltA" in grn.nodes and "zwf" in grn.nodes
    assert grn.nodes["zwf"].level == pytest.approx(1.0)
    assert any(e.source == "gltA" and e.target == "zwf" for e in grn.edges)


_WHOLE_CELL_ADDER_SRC = """
#promoter name=p_constitutive strength=-0.4
#gene name=gltA promoter=p_constitutive chromosome=0.1
ATG GCT GGT GCT TAA
#end
#gene name=zwf promoter=p_constitutive chromosome=0.7 initial_level=1.0
ATG GCT GGT GCT TAA
#end
#regulate gltA -> zwf strength=+0.6
#media nutrient=GLC concentration=10.0
#config backend=whole_cell
#config division_rule=adder adder_volume_um3=1.6
#config replication_mode=cooper_helmstetter doubling_time_min=30
#config c_period_min=20 d_period_min=10
#config seed=0
#config ticks=40
#config output=age,energy,alive,divisions,volume_um3,added_volume_um3,phase
"""


def test_whole_cell_adder_history():
    """division_rule=adder emits added_volume_um3 and divides within the
    simulation window."""
    prog = parse(_WHOLE_CELL_ADDER_SRC)
    result = run(prog)
    assert result is not None
    assert result.backend == "whole_cell"
    assert "added_volume_um3" in result.columns
    assert "divisions" in result.columns
    assert max(row["divisions"] for row in result.rows) >= 1


def test_whole_cell_requires_gene():
    """A pure-config whole_cell program is rejected with the offending key."""
    with pytest.raises(SimConfigError, match="at least one #gene"):
        run(parse("#config backend=whole_cell #config ticks=2"))


# ============================================================================
# W-2: fba backend (FluxBalanceAnalysis / DynamicFluxBalance)
# ============================================================================
def test_fba_backend_static_fluxes():
    src = """
#gene name=glk
ATG GCT GCT GCT GCT TAA
#end
#enzyme gene=glk reaction=GLK
#media nutrient=GLC concentration=10.0
#config backend=fba
#config output=BIOMASS,EX_glc,EX_ac,EX_lac,growth_rate_per_hour
"""
    result = run(parse(src))
    assert result is not None and result.backend == "fba"
    assert len(result.rows) == 1
    assert result.rows[0]["BIOMASS"] > 0.0
    assert result.meta["dynamic"] is False


def test_fba_backend_dfba_batch():
    """dynfba=true returns a batch trace (the diauxie curve)."""
    src = """
#media nutrient=GLC concentration=10.0
#config backend=fba
#config fba_model=core dynfba=true
#config fba_dt_h=0.25 fba_oxygen_max=20.0 fba_steps=8
#config output=time_h,biomass,glucose,co2,growth_rate
"""
    result = run(parse(src))
    assert result is not None and result.backend == "fba"
    assert result.meta["dynamic"] is True
    assert len(result.rows) == 8
    assert result.rows[-1]["time_h"] == pytest.approx(2.0)
    assert result.rows[-1]["biomass"] > result.rows[0]["biomass"]


def test_sim_config_error():
    """Malformed enum/float values name the offending sim key."""
    gene = "#gene name=g\nATG TAA\n#end\n"
    with pytest.raises(SimConfigError, match="division_rule"):
        run(parse(gene + "#config backend=whole_cell "
                   "#config division_rule=size #config ticks=1"))
    with pytest.raises(SimConfigError, match="adder_volume_um3"):
        run(parse(gene + "#config backend=whole_cell "
                   "#config adder_volume_um3=hot #config ticks=1"))
    with pytest.raises(SimConfigError, match="seed"):
        run(parse("#config backend=population #config seed=abc"))


# ============================================================================
# W-3: population backend (CellPopulation3D, Phase 5)
# ============================================================================
_POP_SRC = """
#promoter name=p_housekeeping strength=-0.4
#gene name=adhE promoter=p_housekeeping
ATG GCT GGT GCT TAA
#end
#media nutrient=GLC concentration=10.0 diffusion_um2_s=5.0
#media nutrient=O2 concentration=0.25 diffusion_um2_s=5.0
#config backend=population
#config population_size=25 grid_width=20 grid_height=20
#config dfba=true dfba_dt_h=0.25
#config dfba_oxygen_max_uptake=20.0
#config signaling=true signal_diffusion=0.3 signal_threshold=20.0
#config mechanics=shove crowding=true
#config seed=0
#config ticks=10
#config output=core_glucose_mm,edge_glucose_mm,core_oxygen_mm,edge_oxygen_mm,core_acetate_mm,edge_acetate_mm
"""


def test_population_backend_colony():
    """dfba=true returns colony_observables with metabolic stratification
    (centre glucose/O2 below the edge; acetate accumulates)."""
    result = run(parse(_POP_SRC))
    assert result is not None and result.backend == "population"
    assert len(result.rows) == 10
    last = result.rows[-1]
    assert last["core_glucose_mm"] < last["edge_glucose_mm"]
    assert last["core_oxygen_mm"] < last["edge_oxygen_mm"]
    assert last["core_acetate_mm"] > 0.0
    obs = result.meta["colony_observables"]
    assert "doubling_times_h" in obs and "radial_density" in obs


def test_population_dfba_requires_media():
    with pytest.raises(SimConfigError, match="#media"):
        run(parse("#config backend=population #config dfba=true "
                  "#config ticks=2"))


def test_population_determinism():
    """Same source + same seed => identical observables."""
    a = run(parse(_POP_SRC))
    b = run(parse(_POP_SRC))
    assert a is not None and b is not None
    assert a.rows == b.rows
    assert (a.meta["colony_observables"]
            == b.meta["colony_observables"])


def test_spatial_dfba_extension():
    """The #sim long-tail hook runs kind=spatial_dfba (example 24)."""
    src = """
#config backend=fba
#sim kind=spatial_dfba length=8
#sim inlet_glucose_mm=5.0 initial_glucose_mm=5.0
#sim initial_biomass_gdw=0.05 max_biomass_gdw=2.0
#sim glucose_diffusion_um2_s=2.0 steps=6
#sim output=depletion_front,co2_overflow
"""
    result = run(parse(src))
    assert result is not None
    assert result.meta["kind"] == "spatial_dfba"
    assert result.columns == ["depletion_front", "co2_overflow"]
    assert len(result.rows) == 6
    assert result.rows[-1]["co2_overflow"] > 0.0


# ============================================================================
# W-4: calibration / benchmark backends
# ============================================================================
def test_calibration_backend_recovers_parameters():
    """backend=calibration recovers the hidden parameters within tolerance."""
    src = """
#config backend=calibration
#config fit_seed=1 n_samples=20 refine_rounds=1
#config minutes=10 n_cells=1
#config output=best,passed
"""
    result = run(parse(src))
    assert result is not None and result.backend == "calibration"
    assert result.rows[0]["passed"] is True
    best = result.rows[0]["best"]
    assert best["adder_volume_um3"] == pytest.approx(1.6, abs=0.16)
    assert result.meta["recovered"]["enzyme_scale"] is True


def test_benchmark_backend_gates():
    """backend=benchmark returns the four gates passing."""
    src = """
#config backend=benchmark
#config truth_biomass_to_atp=5e6 truth_maintenance_atp_per_min=2.5e7
#config calibration_uptake=GLC=10.0 calibration_minutes=10
#config prediction_uptake=GLC=20.0 prediction_minutes=15
#config n_samples=60 fit_seed=0 refine_rounds=2
#config output=scores,passed,all_passed
"""
    result = run(parse(src))
    assert result is not None and result.backend == "benchmark"
    row = result.rows[0]
    assert row["passed"] is True
    assert row["all_passed"] is True
    scores = row["scores"]
    assert set(scores) == {"calibration_recovered", "prediction_matches",
                           "energy_rel_error", "biomass_to_atp_rel_error"}
    assert scores["calibration_recovered"] is True
    assert scores["prediction_matches"] is True
    assert scores["energy_rel_error"] < 0.05
    assert scores["biomass_to_atp_rel_error"] < 0.05


# ============================================================================
# W-6: long-tail #sim backends (wiring.md §19)
# ============================================================================
def test_sim_kind_overrides_classic_default():
    """#sim kind= overrides even backend=classic (wiring.md §8.6)."""
    result = run(parse("""
#config backend=classic
#sim kind=3d_morphology preset=bush iterations=2
"""))
    assert result is not None
    assert result.backend == "3d_morphology"


def test_sim_kind_consortium():
    """Quorum consensus run reaches the composition-ratio equilibrium."""
    result = run(parse("""
#config backend=fba
#sim kind=consortium grid_width=16 grid_height=16
#sim sensors=8 producers=8 actuators=8 steps=30 seed=1
#sim output=alive,consensus_fraction,consensus_reached,cumulative_output
"""))
    assert result is not None and result.backend == "consortium"
    assert result.meta["kind"] == "consortium"
    assert set(result.columns) == {
        "alive", "consensus_fraction", "consensus_reached",
        "cumulative_output"}
    last = result.rows[-1]
    assert last["alive"] > 0
    assert 0.0 <= last["consensus_fraction"] <= 1.0
    assert set(result.meta["composition"]) == {"producer", "sensor",
                                               "actuator"}


def test_sim_kind_digital_evolution():
    """Selection raises the fittest organism's signal-fitness."""
    result = run(parse("""
#config backend=fba
#sim kind=digital_evolution target=1010 population_size=40
#sim genome_length=8 substitution_rate=0.02 generations=40 seed=3
#sim output=generation,mean_fitness,max_fitness
"""))
    assert result is not None and result.backend == "digital_evolution"
    assert result.rows[-1]["max_fitness"] > result.rows[0]["max_fitness"]
    assert result.rows[-1]["mean_fitness"] > result.rows[0]["mean_fitness"]
    assert len(result.meta["fittest_genome"]) > 0


def test_sim_kind_stochastic():
    """Gillespie SSA reproduces the analytic telegraph-promoter Fano."""
    result = run(parse("""
#config backend=fba
#sim kind=stochastic mode=gillespie k_on=1.0 k_off=1.0 burst_size=5.0
#sim degradation_rate=0.14 n_replicates=4000 t_max=40.0 seed=1
#sim output=mode,mean,variance,fano,analytic_fano
"""))
    assert result is not None and result.backend == "stochastic"
    row = result.rows[0]
    assert row["mode"] == "gillespie"
    assert abs(row["fano"] - row["analytic_fano"]) < 0.15


def test_sim_kind_codec_benchmark():
    """Fountain codec tolerates more molecule loss at lower density."""
    result = run(parse("""
#config backend=fba
#sim kind=codec_benchmark densities=0.5,1.0 data_size=128 seed=7
#sim output=scheme,target_density,max_loss_fraction,max_error_rate
"""))
    assert result is not None and result.backend == "codec_benchmark"
    fountain = {r["target_density"]: r["max_loss_fraction"]
                for r in result.rows if r["scheme"] == "fountain"}
    assert set(fountain) == {0.5, 1.0}
    assert fountain[0.5] > fountain[1.0]


def test_sim_kind_synbio_design():
    """Cassette auto-design returns a validated in-frame ORF."""
    result = run(parse("""
#config backend=fba
#gene name=goi
ATG GCT GCT GGT GCT GGT GCT TAA
#end
#sim kind=synbio_design promoter=lac rbs=aggagg terminator=rrnB_T1
#sim optimize_codons=true avoid_restriction=true gc_target=0.5
#sim output=protein,orf_length,full_length,cai,gc_content,valid
"""))
    assert result is not None and result.backend == "synbio_design"
    row = result.rows[0]
    assert row["valid"] is True
    assert row["orf_length"] % 3 == 0
    assert 0.0 < row["gc_content"] < 1.0


def test_sim_kind_protein_fitness():
    """BLOSUM oracle ranks the reference itself first."""
    result = run(parse("""
#config backend=fba
#sim kind=protein_fitness reference=MAEAEAE
#sim variants=MAEAEAE,MAEAEAK,MAEAEAY oracle=blosum62
#sim output=rank,variant,score
"""))
    assert result is not None and result.backend == "protein_fitness"
    assert result.meta["oracle"] == "blosum62"
    assert result.rows[0]["variant"] == "MAEAEAE"
    assert result.rows[0]["score"] == pytest.approx(1.0)


def test_sim_kind_morphogen_gradient():
    """The decaying gradient carves a monotone three-band French flag."""
    result = run(parse("""
#config backend=fba
#sim kind=morphogen_gradient length=32 steps=60
#sim output=tick,source_um,mid_um,far_um,monotone,n_domains
"""))
    assert result is not None and result.backend == "morphogen_gradient"
    assert result.meta["monotone"] is True
    assert result.meta["domains"]["near"][0] == 0
    assert result.rows[-1]["n_domains"] == 3


def test_sim_kind_protein_structure():
    """The #gene ORF drives a real secondary-structure report."""
    result = run(parse("""
#config backend=fba
#gene name=helix_demo
ATG GCT GAA CTG GCT GAA CTG GCT GAA CTG TAA
#end
#sim kind=protein_structure
#sim output=length,helix_fraction,sheet_fraction,coil_fraction
"""))
    assert result is not None and result.backend == "protein_structure"
    row = result.rows[0]
    assert row["length"] > 0
    assert 0.0 <= row["helix_fraction"] <= 1.0
    assert row["helix_fraction"] > row["sheet_fraction"]


def test_sim_kind_fate_analysis():
    """Bistability appears as repression strength crosses the saddle-node."""
    result = run(parse("""
#config backend=fba
#sim kind=fate_analysis mode=bifurcation w_values=2.0,4.0,6.0,8.0
#sim output=w,n_stable_states,fate_a_level,fate_b_level
"""))
    assert result is not None and result.backend == "fate_analysis"
    by_w = {r["w"]: r["n_stable_states"] for r in result.rows}
    assert by_w[2.0] < by_w[6.0]


def test_sim_kind_directed_evolution():
    """Oracle-guided rounds beat the random-screening baseline."""
    result = run(parse("""
#config backend=fba
#sim kind=directed_evolution oracle=blosum62 rounds=3 library_size=20
#sim top_k=3 n_crippled_sites=8 seed=7
#sim output=oracle,initial_fitness,guided_recovery,guided_gain,baseline_gain
"""))
    assert result is not None and result.backend == "directed_evolution"
    row = result.rows[0]
    assert row["oracle"] == "blosum62"
    assert row["guided_recovery"] > row["baseline_gain"]


def test_sim_kind_3d_morphology():
    """LSystem3D preset grows a non-degenerate 3D bounding box."""
    result = run(parse("""
#config backend=fba
#sim kind=3d_morphology preset=tree3d iterations=4
#sim output=preset,iterations,n_lines,n_vertices,span_x,span_y,span_z
"""))
    assert result is not None and result.backend == "3d_morphology"
    row = result.rows[0]
    assert row["n_vertices"] > 0
    assert row["span_z"] > 0.0


def test_sim_kind_omics_calibration():
    """Calibrate-then-predict recovers the perturbation responses."""
    result = run(parse("""
#config backend=fba
#sim kind=omics_calibration n_genes=120 n_perturbations=16
#sim n_train=10 seed=0 fit_seed=1
#sim output=response_correlation,de_sign_agreement,passed
"""))
    assert result is not None and result.backend == "omics_calibration"
    row = result.rows[0]
    assert row["passed"] is True
    assert row["response_correlation"] > 0.5


def test_sim_kind_cello_workflow():
    """Cello closed loop validates the predicted truth table."""
    result = run(parse("""
#config backend=fba
#sim kind=cello_workflow function=nand backbone=pUC19 marker=AmpR
#sim time_course_min=60.0 time_step_min=10.0
#sim output=function,matches_target,gate_count,plasmid_length
"""))
    assert result is not None and result.backend == "cello_workflow"
    row = result.rows[0]
    assert row["matches_target"] is True
    assert row["gate_count"] >= 1
    assert row["plasmid_length"] > 0


def test_sim_kind_codon_usage():
    """A gene codon-optimized for E. coli scores 1.0 there, less elsewhere."""
    result = run(parse("""
#config backend=fba
#gene name=gene_ecoli
ATG GCG CTG GAA GTG ATT TTT GGC AGC TAA
#end
#sim kind=codon_usage species=ecoli,yeast,human
#sim output=gene,species,cai,orf_length,protein
"""))
    assert result is not None and result.backend == "codon_usage"
    by_species = {r["species"]: r["cai"] for r in result.rows}
    assert by_species["ecoli"] == pytest.approx(1.0)
    assert by_species["ecoli"] > by_species["yeast"]
    assert by_species["ecoli"] > by_species["human"]
