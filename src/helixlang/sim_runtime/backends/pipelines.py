"""Executor pipelines (doc/38 \u00a79 engine split).

Moved verbatim out of ``_engine.py``: one ``_run_*`` function per
sim backend, plus the plugin/runtime symbols they need.  ``_engine``
keeps orchestration only (``run()`` + shared config/state helpers); the
:class:`~helixlang.api.backend.Backend` adapters in ``backends/core.py``
delegate here.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from helixlang.plugins.apps.consortium import (
    QUORUM_SIGNAL_THRESHOLD,
    ROLES,
    SIGNAL_DIFFUSION_UM2_S,
    SIGNAL_EMISSION_PER_STEP,
    ConsortiumConfig,
    ConsortiumSimulator,
)
from helixlang.plugins.apps.digital_evolution import (
    DigitalEvolution,
    DigitalEvolutionConfig,
)
from helixlang.plugins.apps.dna_storage import benchmark_codecs
from helixlang.plugins.apps.ecosystem import (
    Ecosystem,
    EcosystemConfig,
)
from helixlang.plugins.apps.fate_analysis import (
    bistability_scan,
    critical_slowing_down,
    switching_rate,
)
from helixlang.plugins.apps.morphogen_gradient import MorphogenGradient
from helixlang.plugins.apps.omics_calibration import run_omics_calibration_benchmark
from helixlang.plugins.apps.population_calibration import run_population_calibration
from helixlang.plugins.apps.population_dbtl import (
    DbtlConfig,
    PopulationDbtl,
)
from helixlang.plugins.apps.protein_evolution import guided_directed_evolution
from helixlang.plugins.apps.spatial_dfba import (
    SpatialDFBA,
    SpatialDFBAConfig,
)
from helixlang.plugins.apps.spatial_evolution import (
    SpatialEvolution,
    SpatialEvolutionConfig,
)
from helixlang.plugins.apps.synbio_automation import run_cello_workflow
from helixlang.plugins.apps.synbio_designer import (
    CassetteConfig,
    SynBioDesigner,
)
from helixlang.plugins.apps.virtual_cell_bench import (
    VirtualCellBenchConfig,
    run_virtual_cell_benchmark,
)
from helixlang.plugins.apps.whole_cell_calibration import run_whole_cell_calibration
from helixlang.plugins.runtime.bio_data import (
    cai,
    get_codon_usage,
)
from helixlang.plugins.runtime.metabolism import (
    DEFAULT_GLC_UPTAKE,
    DynamicFBAConfig,
    DynamicFluxBalance,
    FluxBalanceAnalysis,
)
from helixlang.plugins.runtime.morphology_3d import (
    PLANT_PRESETS,
    LSystem3D,
)
from helixlang.plugins.runtime.population import CellPopulation3D
from helixlang.plugins.runtime.protein_fitness import rank_variants
from helixlang.plugins.runtime.protein_structure import predict_structure
from helixlang.plugins.runtime.stochastic import (
    TelegraphPromoter,
    gillespie_telegraph,
)
from helixlang.plugins.runtime.virtual_cell import VirtualCell

from .._coerce import (
    _opt_enum,
    _opt_float_list,
    _opt_str_list,
    _project,
    _select_columns,
)
from .._types import (
    ColonyResult,
    FluxResult,
    HistoryResult,
    ScoreResult,
)

if TYPE_CHECKING:
    pass

from helixlang.sim_runtime._engine import (
    _3D_MORPHOLOGY_COLUMNS,
    _CELLO_COLUMNS,
    _CODEC_DEFAULT_COLUMNS,
    _CODON_USAGE_COLUMNS,
    _CONSORTIUM_DEFAULT_COLUMNS,
    _DFBA_DEFAULT_COLUMNS,
    _DIGITAL_EVO_DEFAULT_COLUMNS,
    _DIRECTED_EVO_COLUMNS,
    _ECOSYSTEM_DEFAULT_COLUMNS,
    _FATE_DEFAULT_COLUMNS,
    _FBA_DEFAULT_COLUMNS,
    _MEDIUM_PRESETS,
    _MORPHOGEN_DEFAULT_COLUMNS,
    _OMICS_CALIBRATION_COLUMNS,
    _ORGANISM_MAX_GROWTH_RATE,
    _POP_DEFAULT_COLUMNS,
    _PROTEIN_FITNESS_COLUMNS,
    _PROTEIN_STRUCTURE_COLUMNS,
    _SIM_DFBA_DEFAULT_COLUMNS,
    _SPATIAL_EVO_DEFAULT_COLUMNS,
    _STOCHASTIC_FANO_COLUMNS,
    _STOCHASTIC_GILLESPIE_COLUMNS,
    _SYNBIO_DEFAULT_COLUMNS,
    _VC_DEFAULT_COLUMNS,
    Any,
    MetabolicModel,
    MorphogenGradientConfig,
    Program,
    SimConfigError,
    SimResult,
    _add_gem_core_reactions,
    _add_gem_transport_reactions,
    _attach_gem_to_ecosystem_species,
    _build_disease_from_helix,
    _build_drugs_from_helix,
    _build_ecosystem_patches,
    _build_ecosystem_species,
    _build_endocrine_config_from_helix,
    _build_genotype_from_helix,
    _build_grn,
    _build_immune_config_from_helix,
    _build_pd_from_helix,
    _build_population_config,
    _build_qsp_bindings_from_helix,
    _build_traits_from_helix,
    _build_tumor_biopsy_from_helix,
    _build_virtual_cell_config,
    _cello_truth_table,
    _enzyme_capacity,
    _first_gene_protein,
    _gene_orfs,
    _group_prefixed,
    _load_fba_model,
    _opt_bool,
    _opt_float,
    _opt_float_dict,
    _opt_float_or_none,
    _opt_int,
    _opt_int_or_none,
    _opt_morphogen_genes,
    _opt_morphogen_repression,
    _parse_bit_sequence,
    _parse_lsystem_rules,
    _seed_cells,
    _set_gem_medium,
    cast,
    translate_dna,
)


def _run_whole_cell(program: Program) -> HistoryResult:
    for e in program.enzymes:
        if e.kcat is not None:
            raise SimConfigError(
                "#enzyme kcat overrides require backend=fba; the whole_cell "
                f"backend uses the canonical ECOLI tables "
                f"(offending gene {e.gene!r})")
    genome = {g.name: "".join(c.seq for c in g.codons) for g in program.genes}
    if not genome:
        raise SimConfigError(
            "backend=whole_cell requires at least one #gene declaration")
    cfg = _build_virtual_cell_config(program)
    cell = VirtualCell(genome, _build_grn(program, cfg.seed), config=cfg)
    history = cell.run(max(1, program.config.ticks))
    columns = _select_columns(program, history, default=_VC_DEFAULT_COLUMNS)
    return HistoryResult(
        backend="whole_cell",
        columns=columns,
        rows=[_project(row, columns) for row in history],
        meta={"name": cell.name},
    )

def _run_static_fba(program: Program, fba: FluxBalanceAnalysis) -> FluxResult:
    sim = program.config.sim
    if _opt_bool(sim, "enzyme_capacity", False) or program.enzymes:
        fba.set_enzyme_capacity(_enzyme_capacity(program, sim))
        fba.set_enzyme_levels({e.gene: 1.0 for e in program.enzymes})
    sol = fba.solve(objective="biomass")
    report = fba.analyze()
    row: dict[str, Any] = dict(sol)
    row["growth_rate_per_hour"] = report["growth_rate_per_hour"]
    columns = _select_columns(program, [row], default=_FBA_DEFAULT_COLUMNS)
    return FluxResult(
        backend="fba", columns=columns,
        rows=[_project(row, columns)],
        meta={"objective": "biomass", "dynamic": False},
    )

def _run_dfba(program: Program, fba: FluxBalanceAnalysis,
              model: MetabolicModel) -> FluxResult:
    sim = program.config.sim
    glc_media = next((m.concentration for m in program.media
                      if m.nutrient == "GLC"), None)
    ac_media = next((m.concentration for m in program.media
                     if m.nutrient == "AC"), None)
    cfg = DynamicFBAConfig(
        dt_h=_opt_float(sim, "fba_dt_h", 0.05),
        initial_biomass_gdw=_opt_float(sim, "fba_initial_biomass_gdw", 0.05),
        initial_glucose_mm=_opt_float(sim, "fba_glucose_mm",
                                      glc_media if glc_media is not None
                                      else 10.0),
        initial_acetate_mm=_opt_float(sim, "fba_acetate_mm",
                                      ac_media if ac_media is not None
                                      else 0.0),
        max_glucose_uptake=_opt_float(sim, "fba_max_glucose_uptake",
                                      DEFAULT_GLC_UPTAKE),
        acetate_switch=_opt_bool(sim, "acetate_switch", False),
        acetate_switch_threshold_mm=_opt_float(
            sim, "acetate_switch_threshold_mm", 0.5),
    )
    batch = DynamicFluxBalance(model, config=cfg, fba=fba)
    o2_cap = _opt_float(sim, "fba_oxygen_max", 0.0)
    if o2_cap > 0.0 and "EX_o2" in model.reactions:
        def _cap_o2(_t: float, _b: DynamicFluxBalance) -> dict[str, float]:
            return {"EX_o2": o2_cap}
        batch.bound_override = _cap_o2
    steps = _opt_int(sim, "fba_steps", max(1, program.config.ticks))
    for _ in range(steps):
        batch.step()
    rows = [dict(entry) for entry in batch.history]
    for row in rows:
        row["time_h"] = row["time"]
    columns = _select_columns(program, rows, default=_DFBA_DEFAULT_COLUMNS)
    return FluxResult(
        backend="fba", columns=columns,
        rows=[_project(row, columns) for row in rows],
        meta={"objective": "biomass", "dynamic": True,
              "fba_steps": steps},
    )

def _run_fba(program: Program) -> FluxResult:
    sim = program.config.sim
    model = _load_fba_model(sim)
    fba = FluxBalanceAnalysis(model)
    for m in program.media:
        fba.set_uptake(m.nutrient, m.concentration)
    if _opt_bool(sim, "dynfba", False):
        return _run_dfba(program, fba, model)
    return _run_static_fba(program, fba)

def _run_spatial_dfba(program: Program) -> FluxResult:
    """``#sim kind=spatial_dfba`` — 1-D dFBA strip (examples/24).

    Every ``SpatialDFBAConfig`` field maps 1:1 to an ``#sim`` key; the
    ``steps`` / ``output`` keys control the batch length and columns.
    """
    ext = program.extensions
    config = SpatialDFBAConfig(
        length=_opt_int(ext, "length", 32),
        glucose_diffusion_um2_s=_opt_float(ext, "glucose_diffusion_um2_s", 2.0),
        initial_glucose_mm=_opt_float(ext, "initial_glucose_mm", 5.0),
        inlet_glucose_mm=_opt_float_or_none(ext, "inlet_glucose_mm", 5.0),
        initial_biomass_gdw=_opt_float(ext, "initial_biomass_gdw", 0.05),
        max_biomass_gdw=_opt_float_or_none(ext, "max_biomass_gdw", None),
        dt_h=_opt_float(ext, "dt_h", 0.05),
        max_glucose_uptake=_opt_float_or_none(ext, "max_glucose_uptake", None),
        seed=_opt_int_or_none(ext, "seed", None),
    )
    sim = SpatialDFBA(config)
    steps = _opt_int(ext, "steps", max(1, program.config.ticks))
    rows: list[dict[str, Any]] = []
    for _ in range(steps):
        snap = sim.step()
        rows.append({
            "tick": snap["tick"],
            "time_h": snap["time_h"],
            "total_glucose": sum(snap["glucose"]),
            "total_biomass": sum(snap["biomass"]),
            "total_acetate": sum(snap["acetate"]),
            "total_consumed": sum(snap["consumed"]),
            "depletion_front": sim.depletion_front(),
            "co2_overflow": sim.total_byproduct("co2"),
        })
    requested = ext.get("output")
    if requested:
        columns = [c.strip() for c in requested.split(",") if c.strip()]
    else:
        columns = _SIM_DFBA_DEFAULT_COLUMNS
    return FluxResult(
        backend="fba", columns=columns,
        rows=[_project(row, columns) for row in rows],
        meta={"kind": "spatial_dfba", "length": config.length,
              "dynamic": True},
    )

def _run_consortium(program: Program) -> ColonyResult:
    """``#sim kind=consortium`` — quorum consensus + composition control."""
    ext = program.extensions
    target_ratios = _opt_float_dict(ext, "target_ratios", {})
    config = ConsortiumConfig(
        grid_width=_opt_int(ext, "grid_width", 40),
        grid_height=_opt_int(ext, "grid_height", 40),
        signal_threshold_um=_opt_float(ext, "signal_threshold_um",
                                       QUORUM_SIGNAL_THRESHOLD),
        emission_um_per_tick=_opt_float(ext, "emission_um_per_tick",
                                        SIGNAL_EMISSION_PER_STEP),
        signal_decay_per_tick=_opt_float(ext, "signal_decay_per_tick", 0.1),
        signal_diffusion_um2_s=_opt_float(ext, "signal_diffusion_um2_s",
                                          SIGNAL_DIFFUSION_UM2_S),
        metabolic_cost=_opt_float(ext, "metabolic_cost", 0.0),
        energy_intake=_opt_float_dict(ext, "energy_intake", {}),
        division_threshold=_opt_float(ext, "division_threshold", 1.5e9),
        death_threshold=_opt_float(ext, "death_threshold", 0.0),
        initial_energy=_opt_float(ext, "initial_energy", 1.0e9),
        max_size=_opt_int(ext, "max_size", 4000),
        consensus_fraction=_opt_float(ext, "consensus_fraction", 0.5),
        output_per_actuator=_opt_float(ext, "output_per_actuator", 1.0),
        ratio_control_gain=_opt_float(ext, "ratio_control_gain", 1.0),
        target_ratios=target_ratios or None,
        seed=_opt_int_or_none(ext, "seed", None),
    )
    sim = ConsortiumSimulator(config)
    for role in ROLES:
        sim.add_cells(_opt_int(ext, f"{role}s", 30), role)
    steps = _opt_int(ext, "steps", max(1, program.config.ticks))
    history = sim.run(steps)
    columns = _select_columns(program, history, default=_CONSORTIUM_DEFAULT_COLUMNS)
    last = history[-1]
    return ColonyResult(
        backend="consortium", columns=columns,
        rows=[_project(row, columns) for row in history],
        meta={
            "kind": "consortium",
            "consensus_reached": bool(last["consensus_reached"]),
            "composition": {r: last[f"{r}_fraction"] for r in ROLES},
            "max_signal_um": last["max_signal"],
            "output_units": last["cumulative_output"],
        },
    )

def _run_digital_evolution(program: Program) -> ScoreResult:
    """``#sim kind=digital_evolution`` — Avida-style instruction genomes."""
    ext = program.extensions
    if "target" in ext:
        target = _parse_bit_sequence(ext["target"])
    else:
        target = DigitalEvolutionConfig().target
    config = DigitalEvolutionConfig(
        population_size=_opt_int(ext, "population_size", 100),
        genome_length=_opt_int(ext, "genome_length", 12),
        target=target,
        substitution_rate=_opt_float(ext, "substitution_rate", 0.01),
        insertion_rate=_opt_float(ext, "insertion_rate", 0.001),
        deletion_rate=_opt_float(ext, "deletion_rate", 0.001),
        step_limit=_opt_int(ext, "step_limit", 32),
        selection_enabled=_opt_bool(ext, "selection_enabled", True),
        generations=_opt_int(ext, "generations", 200),
        seed=_opt_int_or_none(ext, "seed", None),
    )
    evo = DigitalEvolution(config)
    rows = evo.run()
    columns = _select_columns(program, rows, default=_DIGITAL_EVO_DEFAULT_COLUMNS)
    return ScoreResult(
        backend="digital_evolution", columns=columns,
        rows=[_project(row, columns) for row in rows],
        meta={
            "kind": "digital_evolution",
            "final_mean": evo.mean_fitness(),
            "final_max": evo.max_fitness(),
            "fittest_genome": list(evo.fittest_genome()),
            "generations": evo.generation,
        },
    )

def _run_spatial_evolution(program: Program) -> ScoreResult:
    """``#sim kind=spatial_evolution`` — dual-loop range-expansion evolution
    (doc/18-programmable-cell-population-simulation.md §13 Design 1; Bosshard et al. 2020 BMC Genomics 21:232)."""
    ext = program.extensions
    config = SpatialEvolutionConfig(
        generations=_opt_int(ext, "generations", 10),
        population_size=_opt_int(ext, "population_size", 10),
        genome_length_nt=_opt_int(ext, "genome_length", 30),
        substitution_rate=_opt_float(ext, "substitution_rate", 0.05),
        indel_rate=_opt_float(ext, "indel_rate", 0.0),
        recombination_rate=_opt_float(ext, "recombination_rate", 0.2),
        selection_fraction=_opt_float(ext, "selection_fraction", 0.25),
        metabolic_cost=_opt_float(ext, "metabolic_cost", 0.05),
        seed=_opt_int_or_none(ext, "seed", None),
        grid_width=_opt_int(ext, "grid_width", 24),
        grid_height=_opt_int(ext, "grid_height", 24),
        colonization_ticks=_opt_int(ext, "colonization_ticks", 25),
        inner_population_size=_opt_int(ext, "inner_population_size", 40),
        energy_intake=_opt_float(ext, "energy_intake", 5.0e7),
        base_division_threshold=_opt_float(ext, "base_division_threshold",
                                            1.8e9),
        signaling=_opt_bool(ext, "signaling", True),
    )
    evo = SpatialEvolution(config)
    rows = evo.run()
    columns = _select_columns(program, rows,
                              default=_SPATIAL_EVO_DEFAULT_COLUMNS)
    return ScoreResult(
        backend="spatial_evolution", columns=columns,
        rows=[_project(row, columns) for row in rows],
        meta={
            "kind": "spatial_evolution",
            "final_mean": evo.mean_fitness(),
            "final_max": evo.max_fitness(),
            "fittest_genome": evo.best_genome(),
            "generations": evo.generation,
        },
    )

def _run_stochastic(program: Program) -> ScoreResult:
    """``#sim kind=stochastic`` — telegraph-promoter Fano / Gillespie SSA."""
    ext = program.extensions
    mode = _opt_enum(ext, "mode", "gillespie",
                     frozenset({"gillespie", "fano"}))
    k_on = _opt_float(ext, "k_on", 1.0)
    k_off = _opt_float(ext, "k_off", 1.0)
    burst_size = _opt_float(ext, "burst_size", 5.0)
    degradation_rate = _opt_float(ext, "degradation_rate", 0.14)
    expression_scale = _opt_float(ext, "expression_scale", 100.0)
    promoter = TelegraphPromoter(
        k_on, k_off, burst_size, degradation_rate, expression_scale)
    analytic = {
        "fano": promoter.fano_factor(),
        "on_fraction": promoter.on_fraction,
        "transcription_rate": promoter.transcription_rate,
        "mean": (promoter.transcription_rate / degradation_rate)
        * promoter.on_fraction,
    }
    if mode == "fano":
        row = {
            "mode": "fano",
            "fano": analytic["fano"],
            "on_fraction": analytic["on_fraction"],
            "transcription_rate": analytic["transcription_rate"],
            "k_on": k_on,
            "k_off": k_off,
            "burst_size": burst_size,
            "degradation_rate": degradation_rate,
        }
        columns = _select_columns(program, [row],
                                  default=_STOCHASTIC_FANO_COLUMNS)
        return ScoreResult(
            backend="stochastic", columns=columns,
            rows=[_project(row, columns)],
            meta={"kind": "stochastic", "mode": "fano", "promoter": analytic},
        )
    result = gillespie_telegraph(
        k_on, k_off, burst_size, degradation_rate,
        t_max=_opt_float(ext, "t_max", 60.0),
        n_replicates=_opt_int(ext, "n_replicates", 2000),
        seed=_opt_int_or_none(ext, "seed", None),
    )
    row = {
        "mode": "gillespie",
        "mean": result["mean"],
        "variance": result["variance"],
        "fano": result["fano"],
        "analytic_fano": analytic["fano"],
        "n_replicates": _opt_int(ext, "n_replicates", 2000),
        "t_max": _opt_float(ext, "t_max", 60.0),
    }
    columns = _select_columns(program, [row], default=_STOCHASTIC_GILLESPIE_COLUMNS)
    return ScoreResult(
        backend="stochastic", columns=columns, rows=[_project(row, columns)],
        meta={"kind": "stochastic", "mode": "gillespie", "promoter": analytic},
    )

def _run_codec_benchmark(program: Program) -> ScoreResult:
    """``#sim kind=codec_benchmark`` — DNA-storage codec robustness scan."""
    ext = program.extensions
    schemes = _opt_str_list(ext, "schemes", ("goldman", "fountain", "rs"))
    known = ("goldman", "fountain", "rs")
    for s in schemes:
        if s not in known:
            raise SimConfigError(
                f"sim key 'schemes': expected one of {known}, got {s!r}")
    rows_out = benchmark_codecs(
        densities=_opt_float_list(ext, "densities", (0.5, 1.0, 1.5)),
        schemes=schemes,
        data_size=_opt_int(ext, "data_size", 512),
        seed=_opt_int(ext, "seed", 7),
    )
    rows = [
        {
            "scheme": r.scheme,
            "target_density": r.target_density,
            "achieved_density": r.achieved_density,
            "redundancy": r.redundancy,
            "max_loss_fraction": r.max_loss_fraction,
            "max_error_rate": r.max_error_rate,
            "decode_time_s": r.decode_time_s,
            "num_oligos": r.num_oligos,
            "total_bp": r.total_bp,
            "cost_per_gb_usd": r.cost_per_gb_usd,
        }
        for r in rows_out
    ]
    columns = _select_columns(program, rows, default=_CODEC_DEFAULT_COLUMNS)
    return ScoreResult(
        backend="codec_benchmark", columns=columns,
        rows=[_project(r, columns) for r in rows],
        meta={"kind": "codec_benchmark"},
    )

def _run_synbio_design(program: Program) -> ScoreResult:
    """``#sim kind=synbio_design`` — cassette auto-design (promoter+RBS+ORF+term)."""
    ext = program.extensions
    protein = ext.get("protein", _first_gene_protein(program))
    if not protein:
        raise SimConfigError(
            "#sim kind=synbio_design requires a protein= sequence or a "
            "#gene ORF to design from")
    designer = SynBioDesigner(seed=_opt_int_or_none(ext, "seed", None))
    config = CassetteConfig(
        promoter=_opt_enum(ext, "promoter", "lac",
                           frozenset({"lac", "T7", "araBAD", "tet"})),
        rbs=ext.get("rbs", "aggagg").upper(),
        terminator=_opt_enum(ext, "terminator", "rrnB_T1",
                             frozenset({"rrnB_T1", "T7"})),
        optimize_codons=_opt_bool(ext, "optimize_codons", True),
        avoid_restriction=_opt_bool(ext, "avoid_restriction", True),
        gc_target=_opt_float(ext, "gc_target", 0.50),
        max_homopolymer=_opt_int(ext, "max_homopolymer", 4),
        add_histidine_tag=_opt_bool(ext, "add_histidine_tag", False),
        add_mbd_tag=_opt_bool(ext, "add_mbd_tag", False),
    )
    cassette = designer.design_cassette(protein, config=config)
    validation = cassette.validation_report
    row = {
        "protein": cassette.protein,
        "orf_length": len(cassette.orf_seq),
        "full_length": len(cassette.full_sequence),
        "cai": cassette.cai,
        "gc_content": cassette.gc_content,
        "n_restriction_sites": len(cassette.restriction_sites_found),
        "valid": bool(validation.get("valid")),
        "promoter": config.promoter,
    }
    columns = _select_columns(program, [row], default=_SYNBIO_DEFAULT_COLUMNS)
    return ScoreResult(
        backend="synbio_design", columns=columns,
        rows=[_project(row, columns)],
        meta={
            "kind": "synbio_design",
            "validation": validation,
            "promoter_seq": cassette.promoter_seq,
            "terminator_seq": cassette.terminator_seq,
            "orf_seq": cassette.orf_seq,
            "full_sequence": cassette.full_sequence,
        },
    )

def _run_protein_fitness(program: Program) -> ScoreResult:
    """``#sim kind=protein_fitness`` — oracle ranking of protein variants."""
    ext = program.extensions
    reference = ext.get("reference", _first_gene_protein(program))
    if not reference:
        raise SimConfigError(
            "#sim kind=protein_fitness requires reference= or a #gene ORF")
    raw_variants = ext.get("variants")
    if not raw_variants:
        raise SimConfigError(
            "#sim kind=protein_fitness requires variants=a,b,c")
    variants = [v.strip() for v in raw_variants.split(",") if v.strip()]
    oracle = _opt_enum(ext, "oracle", "blosum62",
                       frozenset({"blosum62", "esm2"}))
    try:
        ranked = rank_variants(reference, variants, oracle)
    except RuntimeError:
        ranked = rank_variants(reference, variants, "blosum62")
        oracle = "blosum62"
    rows = [
        {"rank": i + 1, "variant": v, "score": s}
        for i, (v, s) in enumerate(ranked)
    ]
    columns = _select_columns(program, rows, default=_PROTEIN_FITNESS_COLUMNS)
    return ScoreResult(
        backend="protein_fitness", columns=columns,
        rows=[_project(r, columns) for r in rows],
        meta={"kind": "protein_fitness", "oracle": oracle,
              "reference": reference, "best": ranked[0][0]},
    )

def _run_morphogen_gradient(program: Program) -> HistoryResult:
    """``#sim kind=morphogen_gradient`` — 1-D French-flag patterning strip."""
    ext = program.extensions
    config = MorphogenGradientConfig(
        length=_opt_int(ext, "length", 64),
        diffusion_um2_s=_opt_float(ext, "diffusion_um2_s", 10.0),
        decay_per_tick=_opt_float(ext, "decay_per_tick", 0.05),
        source_strength_um=_opt_float(ext, "source_strength_um", 20.0),
        response_steepness=_opt_float(ext, "response_steepness", 0.5),
        genes=_opt_morphogen_genes(ext),
        repression=_opt_morphogen_repression(ext),
        seed=_opt_int_or_none(ext, "seed", None),
    )
    grad = MorphogenGradient(config)
    steps = _opt_int(ext, "steps", max(1, program.config.ticks))
    grad.run(steps)
    rows: list[dict[str, Any]] = []
    for tick in range(1, len(grad.history) + 1):
        c = grad.concentration
        dom = grad.domains()
        rows.append({
            "tick": tick,
            "source_um": c[0] if c else 0.0,
            "mid_um": c[len(c) // 2] if c else 0.0,
            "far_um": c[-1] if c else 0.0,
            "monotone": 1.0 if grad.is_monotone_decreasing() else 0.0,
            "n_domains": sum(1 for s, e in dom.values() if s >= 0),
        })
    columns = _select_columns(program, rows, default=_MORPHOGEN_DEFAULT_COLUMNS)
    return HistoryResult(
        backend="morphogen_gradient", columns=columns,
        rows=[_project(r, columns) for r in rows],
        meta={
            "kind": "morphogen_gradient",
            "monotone": grad.is_monotone_decreasing(),
            "domains": {k: list(v) for k, v in grad.domains().items()},
            "boundary_positions": grad.boundary_positions(),
            "gradient_length_scale": grad.gradient_length_scale(),
            "concentration": grad.concentration,
        },
    )

def _run_protein_structure(program: Program) -> ScoreResult:
    """``#sim kind=protein_structure`` — predict_structure report summary."""
    ext = program.extensions
    sequence = ext.get("sequence", _first_gene_protein(program))
    if not sequence:
        raise SimConfigError(
            "#sim kind=protein_structure requires sequence= or a #gene ORF")
    report = predict_structure(sequence)
    data = report.to_dict()
    row = {k: data[k] for k in _PROTEIN_STRUCTURE_COLUMNS}
    columns = _select_columns(program, [row], default=_PROTEIN_STRUCTURE_COLUMNS)
    return ScoreResult(
        backend="protein_structure", columns=columns,
        rows=[_project(row, columns)],
        meta={
            "kind": "protein_structure",
            "secondary_structure": report.secondary_structure,
            "ss_segments": [asdict(s) for s in report.ss_segments],
            "transmembrane_helices": [
                asdict(h) for h in report.transmembrane_helices],
            "disorder_regions": [asdict(d) for d in report.disorder_regions],
            "summary": report.summary,
        },
    )

def _run_fate_analysis(program: Program) -> ScoreResult:
    """``#sim kind=fate_analysis`` — toggle bistability / switching / slowing."""
    ext = program.extensions
    mode = _opt_enum(ext, "mode", "full",
                     frozenset({"full", "bifurcation", "switching", "slowing"}))
    w_values = _opt_float_list(ext, "w_values",
                               (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
    scan_w = _opt_float(ext, "scan_w", 7.0)
    switch_resources = _opt_float_list(ext, "switch_resources",
                                       (0.0, 0.5, 1.0))
    n_trajectories = _opt_int(ext, "n_trajectories", 600)
    n_ticks = _opt_int(ext, "n_ticks", 200)
    noise_fano = _opt_float(ext, "noise_fano", 2.0)
    expression_scale = _opt_float(ext, "expression_scale", 1000.0)
    decay = _opt_float(ext, "decay", 0.5)
    seed = _opt_int_or_none(ext, "seed", 0)

    scan = bistability_scan(w_values)
    rows = [
        {
            "w": p.parameter,
            "n_stable_states": p.n_stable_states,
            "fate_a_level": p.stable_states[-1].a if p.stable_states else 0.0,
            "fate_b_level": p.stable_states[0].a if p.stable_states else 0.0,
        }
        for p in scan
    ]
    meta: dict[str, Any] = {
        "kind": "fate_analysis", "mode": mode, "w_values": list(w_values),
    }
    if mode in ("full", "switching"):
        meta["switching_rates"] = {
            r: switching_rate(
                scan_w, resource_strength=r, n_trajectories=n_trajectories,
                n_ticks=n_ticks, noise_fano=noise_fano,
                expression_scale=expression_scale, decay=decay, seed=seed)
            for r in switch_resources
        }
    if mode in ("full", "slowing"):
        meta["critical_slowing_down"] = {
            w: critical_slowing_down(
                w, n_ticks=n_ticks, noise_fano=noise_fano,
                expression_scale=expression_scale, decay=decay, seed=seed)
            for w in (3.0, 5.0, 5.3, 5.5)
        }
    columns = _select_columns(program, rows, default=_FATE_DEFAULT_COLUMNS)
    return ScoreResult(
        backend="fate_analysis", columns=columns,
        rows=[_project(r, columns) for r in rows], meta=meta,
    )

def _run_directed_evolution(program: Program) -> ScoreResult:
    """``#sim kind=directed_evolution`` — oracle-guided GB1 campaign."""
    ext = program.extensions
    oracle = _opt_enum(ext, "oracle", "blosum62",
                       frozenset({"blosum62", "esm2"}))
    kwargs = dict(
        rounds=_opt_int(ext, "rounds", 8),
        library_size=_opt_int(ext, "library_size", 60),
        top_k=_opt_int(ext, "top_k", 5),
        n_crippled_sites=_opt_int(ext, "n_crippled_sites", 8),
        cripple_seed=_opt_int(ext, "cripple_seed", 1),
        seed=_opt_int(ext, "seed", 7),
    )
    try:
        result = guided_directed_evolution(oracle=oracle, **kwargs)
    except RuntimeError:
        result = guided_directed_evolution(oracle="blosum62", **kwargs)
        oracle = "blosum62"
    row = {
        "oracle": result.oracle_name,
        "initial_fitness": result.initial_fitness,
        "guided_recovery": result.guided_recovery,
        "guided_gain": result.guided_gain,
        "baseline_gain": result.baseline_gain,
        "spearman_rho": result.spearman_rho,
        "final_best_sequence": result.final_best_sequence,
    }
    columns = _select_columns(program, [row], default=_DIRECTED_EVO_COLUMNS)
    return ScoreResult(
        backend="directed_evolution", columns=columns,
        rows=[_project(row, columns)],
        meta={
            "kind": "directed_evolution",
            "guided_cumulative_best": result.guided_cumulative_best,
            "baseline_cumulative_best": result.baseline_cumulative_best,
        },
    )

def _run_3d_morphology(program: Program) -> ScoreResult:
    """``#sim kind=3d_morphology`` — LSystem3D preset mesh statistics."""
    ext = program.extensions
    preset = ext.get("preset", "tree3d")
    spec = PLANT_PRESETS.get(preset)
    if spec is None:
        raise SimConfigError(
            f"sim key 'preset': expected one of {sorted(PLANT_PRESETS)}, "
            f"got {preset!r}")
    if "rules" in ext:
        rules = _parse_lsystem_rules(ext["rules"])
    else:
        rules = dict(spec["rules"])
    lsys = LSystem3D(
        axiom=ext.get("axiom", spec["axiom"]),
        rules=rules,
        angle=_opt_float(ext, "angle", float(spec["angle"])),
        step=_opt_float(ext, "step", float(spec["step"])),
    )
    iterations = _opt_int(ext, "iterations", max(1, program.config.ticks))
    lines = lsys.draw(iterations)
    points = lsys.get_points(iterations)
    bounds = lsys.get_bounds(iterations)
    size = bounds["size"]
    row = {
        "preset": preset,
        "iterations": iterations,
        "n_lines": len(lines),
        "n_vertices": len(points),
        "span_x": size.x,
        "span_y": size.y,
        "span_z": size.z,
    }
    columns = _select_columns(program, [row], default=_3D_MORPHOLOGY_COLUMNS)
    return ScoreResult(
        backend="3d_morphology", columns=columns,
        rows=[_project(row, columns)],
        meta={
            "kind": "3d_morphology",
            "bounds": {
                "min": [bounds["min"].x, bounds["min"].y, bounds["min"].z],
                "max": [bounds["max"].x, bounds["max"].y, bounds["max"].z],
                "center": [
                    bounds["center"].x, bounds["center"].y, bounds["center"].z],
                "size": [size.x, size.y, size.z],
            },
            "points": [[p.x, p.y, p.z] for p in points],
        },
    )

def _run_omics_calibration(program: Program) -> ScoreResult:
    """``#sim kind=omics_calibration`` — calibrate-then-predict perturb-seq."""
    ext = program.extensions
    result = run_omics_calibration_benchmark(
        n_genes=_opt_int(ext, "n_genes", 200),
        n_perturbations=_opt_int(ext, "n_perturbations", 24),
        n_train=_opt_int(ext, "n_train", 16),
        seed=_opt_int(ext, "seed", 0),
        fit_seed=_opt_int(ext, "fit_seed", 1),
    )
    row = {
        "response_correlation": result["response_correlation"],
        "de_sign_agreement": result["de_sign_agreement"],
        "mae_improvement_vs_baseline": result["mae_improvement_vs_baseline"],
        "n_holdout": len(result["holdout_perturbations"]),
        "passed": result["passed"],
    }
    columns = _select_columns(program, [row], default=_OMICS_CALIBRATION_COLUMNS)
    return ScoreResult(
        backend="omics_calibration", columns=columns,
        rows=[_project(row, columns)],
        meta={
            "kind": "omics_calibration",
            "truth_coupling": result["truth_coupling"],
            "fitted_coupling": result["fitted_coupling"],
            "mae_per_perturbation": result["mae_per_perturbation"],
            "baseline_mae_per_perturbation": result["baseline_mae_per_perturbation"],
            "fit_best": result["fit"]["best"],
        },
    )

def _run_population_calibration(program: Program) -> ScoreResult:
    """``#sim kind=population_calibration``: recover the dFBA colony
    parameters from colony-level mixed observables (doc/18-programmable-cell-population-simulation.md §13 Design 4)."""
    ext = program.extensions
    result = run_population_calibration(
        n_samples=_opt_int(ext, "n_samples", 12),
        refine_rounds=_opt_int(ext, "refine_rounds", 2),
        fit_seed=_opt_int(ext, "fit_seed", 0),
        division_ticks=_opt_int(ext, "division_ticks", 12),
        n_cells=_opt_int(ext, "n_cells", 4),
    )
    row: dict[str, Any] = {
        "best": result["fitted"],
        "sse": result["fit"]["sse"],
        "n_samples": result["fit"]["n_samples"],
        "passed": result["passed"],
    }
    columns = _select_columns(program, [row], default=list(row))
    return ScoreResult(
        backend="population_calibration", columns=columns,
        rows=[_project(row, columns)],
        meta={"relative_error": result["relative_error"],
              "recovered": result["recovered"],
              "truth": result["truth"]},
    )

def _run_cello_workflow(program: Program) -> ScoreResult:
    """``#sim kind=cello_workflow`` — truth table -> DNA + SBOL3 closed loop."""
    ext = program.extensions
    function = _opt_enum(ext, "function", "not",
                         frozenset({"not", "nand", "xor"}))
    report = run_cello_workflow(
        table=_cello_truth_table(function),
        backbone=ext.get("backbone", "pUC19"),
        marker=ext.get("marker", "AmpR"),
        time_course_min=_opt_float(ext, "time_course_min", 60.0),
        time_step_min=_opt_float(ext, "time_step_min", 5.0),
    )
    validation = report.validation
    row = {
        "function": function,
        "matches_target": bool(validation["matches_target"]),
        "gate_count": int(validation["gate_count"]),
        "plasmid_length": int(validation["plasmid_length"]),
        "sbol3_component_count": int(validation["sbol3_component_count"]),
        "time_curve_count": int(validation["time_curve_count"]),
    }
    columns = _select_columns(program, [row], default=_CELLO_COLUMNS)
    return ScoreResult(
        backend="cello_workflow", columns=columns,
        rows=[_project(row, columns)],
        meta={
            "kind": "cello_workflow",
            "validation": validation,
            "gate_order": report.gate_order,
            "time_curves": report.time_curves,
            "plasmid_dna": report.plasmid_dna,
        },
    )

def _run_codon_usage(program: Program) -> ScoreResult:
    """``#sim kind=codon_usage`` — per-gene CAI across species tables."""
    ext = program.extensions
    species_raw = ext.get("species", program.config.species or "ecoli")
    species_list = [s.strip().lower() for s in species_raw.split(",")
                    if s.strip()]
    for s in species_list:
        get_codon_usage(s)
    orfs = _gene_orfs(program)
    if not orfs:
        raise SimConfigError(
            "#sim kind=codon_usage requires at least one #gene ORF")
    rows: list[dict[str, Any]] = []
    for name, dna in orfs:
        protein = translate_dna(dna)
        stop = protein.find("*")
        if stop != -1:
            protein = protein[:stop]
        for s in species_list:
            rows.append({
                "gene": name,
                "species": s,
                "cai": cai(dna, s),
                "orf_length": len(dna),
                "protein": protein,
            })
    columns = _select_columns(program, rows, default=_CODON_USAGE_COLUMNS)
    return ScoreResult(
        backend="codon_usage", columns=columns,
        rows=[_project(r, columns) for r in rows],
        meta={"kind": "codon_usage", "species": species_list},
    )

_CARDIOLOGY_COLUMNS = [
    "period_s",
    "heart_rate_bpm",
    "stroke_volume_ml",
    "cardiac_output_l_min",
    "cardiac_index_l_min_m2",
    "map_mmhg",
    "systolic_bp_mmhg",
    "diastolic_bp_mmhg",
    "conduction",
]

def _run_cardiology(program: Program) -> ScoreResult:
    """``#sim kind=cardiology`` — closed-loop cardiac cycle (doc/42 Phase E).

    Drives the Phase B RL-1 closed-loop cardiovascular core
    (:class:`~helixlang.plugins.human.physiological_core.HemodynamicModel`) from
    each ``#cardiac_cycle`` annotation (``period`` in seconds, ``conduction``),
    so the directive produces real hemodynamic output instead of only validating
    a period.  Deterministic (no RNG), so the result is golden-verifiable.
    """
    from helixlang.plugins.human.physiological_core import HemodynamicModel

    ext = program.sim_extensions or {}
    cycles = ext.get("cardiac_cycle") or []
    if not isinstance(cycles, list) or not cycles:
        raise SimConfigError(
            "#sim kind=cardiology requires at least one #cardiac_cycle annotation")

    default_conduction = "normal"
    rows: list[dict[str, Any]] = []
    for entry in cycles:
        period_s = _opt_float({"period": entry.get("period", "0.8")}, "period", 0.8)
        if not 0.0 < period_s <= 5.0:
            raise SimConfigError(
                f"#cardiac_cycle period={entry.get('period')!r} outside (0, 5] s")
        conduction = str(entry.get("conduction", default_conduction) or default_conduction)

        # warm the closed-loop core to steady state, then drive for one period.
        h = HemodynamicModel(baseline_hr_bpm=72.0, baseline_sv_ml=70.0)
        for _ in range(24):
            h.step(1.0)
        dt_h = period_s / 3600.0
        h.step(dt_h)
        s = h.state
        rows.append({
            "period_s": round(period_s, 4),
            "heart_rate_bpm": round(s.heart_rate_bpm, 2),
            "stroke_volume_ml": round(s.stroke_volume_ml, 2),
            "cardiac_output_l_min": round(s.cardiac_output_l_min, 3),
            "cardiac_index_l_min_m2": round(s.cardiac_index_l_min_m2, 3),
            "map_mmhg": round(s.map_mmhg, 2),
            "systolic_bp_mmhg": round(s.systolic_bp_mmhg, 1),
            "diastolic_bp_mmhg": round(s.diastolic_bp_mmhg, 1),
            "conduction": conduction,
        })
    columns = _select_columns(program, rows, default=_CARDIOLOGY_COLUMNS)
    return ScoreResult(
        backend="cardiology", columns=columns,
        rows=[_project(r, columns) for r in rows],
        meta={"kind": "cardiology", "cardiac_cycles": len(rows)},
    )

def _run_ecosystem(program: Program) -> ScoreResult:
    """``#sim kind=ecosystem`` — multi-species/multi-patch ecosystem with
    dispersal, biogeochemistry, environmental forcing and an invasion-fitness
    evolution loop (doc/19 §5.3-§5.6; L1-L13).

    When ``gem_driven=true`` in ``#sim``, each ``#species`` with a
    ``genome=`` field has its GEM reconstructed (doc/20) and the resulting
    ``MetabolicModel`` + FBA fluxes are attached to the species so the
    ecosystem tick loop uses FBA-backed growth instead of Monod kinetics
    (doc/21 §3.3).

    Reads ``#sim`` keys (``ticks``/``seed``/``generations``/``fast_forward``/
    ``community_fba``/``sample_every``/``evaluation_ticks``/``gem_driven``/...),
    the ``#species`` table and the ``#patch`` table from ``program.extensions``.
    """
    ext = program.extensions
    # Merge #config sim params into ext so ecosystem can read them
    merged = dict(program.config.sim)
    merged.update(ext)
    ext = merged
    # Propagate program.config.ticks (set by CLI --ticks or #config ticks=)
    # into ext so the ecosystem uses the user-requested tick count.
    if program.config.ticks is not None:
        ext["ticks"] = str(program.config.ticks)
    species = _build_ecosystem_species(ext)
    patches = _build_ecosystem_patches(ext)
    gem_driven = _opt_bool(ext, "gem_driven", False)
    if not species:
        raise SimConfigError(
            "#sim kind=ecosystem requires at least one #species name=...")
    if not patches:
        raise SimConfigError(
            "#sim kind=ecosystem requires at least one #patch name=...")

    # GEM bridge (doc/21): reconstruct metabolic models for species with genomes
    if gem_driven:
        _attach_gem_to_ecosystem_species(species, ext)

    config = EcosystemConfig(
        ticks=_opt_int(ext, "ticks", 4320),
        seed=_opt_int_or_none(ext, "seed", None),
        fast_forward=_opt_bool(ext, "fast_forward", True),
        scheduler_max_step=_opt_int(ext, "scheduler_max_step", 480),
        scheduler_change_threshold=_opt_float(
            ext, "scheduler_change_threshold", 1e-4),
        community_fba=_opt_bool(ext, "community_fba", False),
        gem_driven=gem_driven,
        sample_every=_opt_int(ext, "sample_every", 1),
        species=species,
        patches=patches,
        generations=_opt_int(ext, "generations", 1),
        population_size=_opt_int(ext, "population_size", 6),
        substitution_rate=_opt_float(ext, "substitution_rate", 0.05),
        indel_rate=_opt_float(ext, "indel_rate", 0.0),
        recombination_rate=_opt_float(ext, "recombination_rate", 0.0),
        genome_length_nt=_opt_int(ext, "genome_length", 36),
        evaluation_ticks=_opt_int(ext, "evaluation_ticks", 200),
        evolution_enabled=_opt_bool(ext, "evolution_enabled", False),
        stress_field=ext.get("stress_field", "toxin"),
        stress_level=_opt_float(ext, "stress_level", 0.0),
    )
    eco = Ecosystem(config)
    if config.generations > 1 or config.evolution_enabled:
        rows = eco.run_generations()
    else:
        rows = eco.run()
    columns = _select_columns(program, rows, default=(
        list(rows[0]) if rows else _ECOSYSTEM_DEFAULT_COLUMNS))
    meta: dict[str, Any] = {
        "kind": "ecosystem",
        "species": [s.name for s in species],
        "patches": [p.name for p in patches],
        "neutral_niche": eco.neutral_vs_niche(),
        "summary": eco.summary(),
    }
    return ScoreResult(
        backend="ecosystem", columns=columns,
        rows=[_project(row, columns) for row in rows],
        meta=meta,
    )

def _run_population_dbtl(program: Program) -> ScoreResult:
    """``#sim kind=population_dbtl`` — Design→Build→Test→Learn loop over a
    population of strains (synbio_designer DNA → Ecosystem → fit_parameters).

    Reads ``#sim`` keys ``n_rounds``/``population_size``/``genome_length``/
    ``evaluation_ticks``/``seed``/``substrate``/``mutation_rate``/
    ``bias_fraction``/``n_candidates``; the gate is a designed strain whose
    growth strictly improves across the loop.
    """
    ext = program.extensions
    dbtl = PopulationDbtl(DbtlConfig(
        n_rounds=_opt_int(ext, "n_rounds", 4),
        population_size=_opt_int(ext, "population_size", 6),
        genome_length_nt=_opt_int(ext, "genome_length", 30),
        substrate=ext.get("substrate", "glucose"),
        vmax=_opt_float(ext, "vmax", 0.02),
        ks=_opt_float(ext, "ks", 0.1),
        substrate_mm=_opt_float(ext, "substrate_mm", 10000.0),
        initial_nh4_mm=_opt_float(ext, "initial_nh4_mm", 10000.0),
        carrying_capacity=_opt_float(ext, "carrying_capacity", 1e5),
        evaluation_ticks=_opt_int(ext, "evaluation_ticks", 80),
        seed=_opt_int_or_none(ext, "seed", None) or 0,
        target_protein=ext.get("target_protein", "MGTKDFYEAVRS"),
        mutation_rate=_opt_float(ext, "mutation_rate", 0.05),
        bias_fraction=_opt_float(ext, "bias_fraction", 0.6),
        n_candidates=_opt_int(ext, "n_candidates", 8),
    ))
    result = dbtl.run()
    rows: list[dict[str, Any]] = [
        {
            "round": float(r["round"]),
            "best_growth": r["best_growth"],
            "mean_growth": r["mean_growth"],
            "best_genome": r["best_genome"],
            "n_tested": float(r["n_tested"]),
            "surrogate_best_trait": r["surrogate_best_trait"],
        }
        for r in result["rounds"]
    ]
    columns = _select_columns(program, rows, default=(
        ["round", "best_growth", "mean_growth", "best_genome",
         "surrogate_best_trait"]))
    return ScoreResult(
        backend="population_dbtl", columns=columns,
        rows=[_project(row, columns) for row in rows],
        meta={
            "kind": "population_dbtl",
            "designed_strain": result["designed_strain"],
            "round0_growth": result["round0_growth"],
            "final_growth": result["final_growth"],
            "improved": result["improved"],
            "fold_improvement": result["fold_improvement"],
        },
    )

def _run_gem_full_model(
    organism: str,
    medium_name: str,
    dynamic: bool,
    duration: float,
    dt: float,
    medium_override: dict[str, float] | None = None,
    max_growth_rate: float | None = None,
) -> SimResult:
    """Run FBA on a full genome-scale model from BiGG (doc/24).

    This is the early-exit path for ``#gem ... use_full_model=true``.
    It loads a pre-built GEM (e.g. iML1515, iJN678) via
    :func:`build_functional_model_full`, applies medium, and runs
    static or dynamic FBA — skipping genome reconstruction entirely.
    """
    from helixlang.plugins.gem.bridge import build_functional_model_full
    from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis

    max_mu = _ORGANISM_MAX_GROWTH_RATE.get(organism, _ORGANISM_MAX_GROWTH_RATE["default"])
    if max_growth_rate is not None:
        max_mu = max_growth_rate

    try:
        model = build_functional_model_full(
            organism=organism,
            medium=medium_name,
        )

        # doc/25 Phase E: apply medium_override on full model
        if medium_override:
            for rid, rxn in model.reactions.items():
                if rxn.subsystem == "exchange" and rid.startswith("EX_"):
                    if len(rxn.stoichiometry) == 1:
                        met = next(iter(rxn.stoichiometry))
                        if met in medium_override:
                            coef = rxn.stoichiometry[met]
                            rate = medium_override[met]
                            if coef < 0:
                                rxn.lower_bound = -abs(rate)
                            else:
                                rxn.upper_bound = abs(rate)

        fba = FluxBalanceAnalysis(model)

        growth_rate = 0.0
        fba_status = "skipped"
        key_fluxes: dict[str, float] = {}
        analysis: dict[str, Any] = {}
        _extra_meta: dict[str, Any] = {}

        if dynamic and medium_name in ("bg11", "photoautotrophic"):
            from helixlang.plugins.runtime.metabolism import (
                DynamicFBAConfig,
                PhotoautotrophicFluxBalance,
            )
            photo_cfg = DynamicFBAConfig(
                substrate_type="co2",
                dt_h=dt,
                initial_biomass_gdw=0.01,
                co2_initial_mm=10.0,
                co2_max_uptake=50.0,
                co2_half_saturation_mm=0.5,
                max_growth_rate=max_mu,
                max_biomass_gdw=50.0,
            )
            batch = PhotoautotrophicFluxBalance(
                model=model, config=photo_cfg, fba=fba)
            n_steps = max(1, int(duration / dt))
            dyn_trajectory: list[dict] = []
            for _ in range(n_steps):
                dyn_trajectory.append(batch.step())
            if dyn_trajectory:
                growth_rate = max(
                    e.get("growth_rate", 0.0) for e in dyn_trajectory)
                _end = dyn_trajectory[-1]
                for e in reversed(dyn_trajectory):
                    if e.get("co2", 0.0) > 0.01 and e.get("growth_rate", 0.0) > 1e-6:
                        _end = e
                        break
                key_fluxes = {
                    k: round(v, 4)
                    for k, v in _end.items()
                    if isinstance(v, (int, float)) and abs(v) > 1e-6
                }
            fba_status = "ok"
            _extra_meta["dynamic"] = True
            _extra_meta["duration_h"] = duration
            _extra_meta["dt_h"] = dt
        else:
            fba.solve(objective="biomass", maximize=True)
            analysis = fba.analyze()
            growth_rate = analysis.get("growth_rate_per_hour", 0.0)
            growth_rate = min(growth_rate, max_mu)
            key_fluxes = {
                k: round(v, 4)
                for k, v in analysis.get("key_fluxes", {}).items()
                if abs(v) > 1e-6
            }
            fba_status = "ok"

        n_rxns = len(model.reactions)

        return SimResult(
            backend="gem",
            columns=["stage", "status", "reactions_total", "growth_rate"],
            rows=[
                {
                    "stage": "full_model_load",
                    "status": "ok",
                    "reactions_total": n_rxns,
                    "growth_rate": round(growth_rate, 4),
                },
                {
                    "stage": "fba_simulation",
                    "status": fba_status,
                    "reactions_total": n_rxns,
                    "growth_rate": round(growth_rate, 4),
                },
            ],
            meta={
                "organism": organism,
                "medium": medium_name,
                "use_full_model": True,
                "reactions_total": n_rxns,
                "growth_rate_per_hour": round(growth_rate, 4),
                "key_fluxes": key_fluxes,
                **_extra_meta,
            },
        )
    except Exception as exc:
        return SimResult(
            backend="gem",
            columns=["stage", "status", "error"],
            rows=[{
                "stage": "full_model_load",
                "status": "failed",
                "error": str(exc),
            }],
            meta={
                "organism": organism,
                "medium": medium_name,
                "use_full_model": True,
                "growth_rate_per_hour": 0.0,
                "errors": [str(exc)],
            },
        )

def _run_gem(program: Program) -> SimResult:
    """Run GEM reconstruction pipeline from #gem or #species genome data.

    After reconstruction completes, builds a MetabolicModel from the
    consensus result, sets exchange uptake bounds from the growth medium,
    and runs FBA to predict growth rate and flux distribution.

    Supports inline DNA sequences (doc/20 §12): if #gem has a DNA block
    instead of genome=, the sequence is written to a temp FASTA file.
    """
    import tempfile
    from pathlib import Path

    from helixlang.plugins.apps.gem_pipeline import run_gem_pipeline

    ext = program.extensions

    # Priority: #gem fields > #species fields
    genome_fasta = ext.get("gem_genome", "")
    organism = ext.get("gem_organism", "e_coli_k12")
    use_database = ext.get("gem_use_database", "true").lower() in ("true", "1", "yes")
    include_spontaneous = ext.get("gem_include_spontaneous", "true").lower() in ("true", "1", "yes")
    run_gapfill = ext.get("gem_gapfill", "true").lower() in ("true", "1", "yes")
    target_organism = ext.get("gem_target_organism", "Escherichia coli")
    medium_name = ext.get("gem_medium", "glucose_minimal")
    dynamic = ext.get("gem_dynamic", "false").lower() in ("true", "1", "yes")
    duration = float(ext.get("gem_duration", "24.0"))
    dt = float(ext.get("gem_dt", "0.05"))
    expression_enabled = ext.get("gem_expression", "false").lower() in ("true", "1", "yes")

    # doc/25 Phase E: medium_override — comma-separated met:value pairs
    # that override individual metabolite uptake rates in the preset.
    medium_override_str = ext.get("gem_medium_override", "")
    medium_override: dict[str, float] = {}
    if medium_override_str:
        for pair in medium_override_str.split(","):
            pair = pair.strip()
            if ":" in pair:
                met, val = pair.split(":", 1)
                try:
                    medium_override[met.strip()] = float(val.strip())
                except ValueError:  # SILENTBENIGN - skip non-numeric override
                    pass

    # doc/25 Phase F: max_growth_rate DSL override
    _max_growth_rate_override: float | None = None
    _mgr_str = ext.get("gem_max_growth_rate", "")
    if _mgr_str:
        try:
            _max_growth_rate_override = float(_mgr_str)
        except ValueError:  # SILENTBENIGN - keep default growth rate
            pass

    # Handle inline DNA sequence (doc/20 §12)
    inline_genome: str = ext.get("gem_inline_genome", "")
    inline_genes_raw = ext.get("gem_inline_genes", [])
    inline_genes: list[tuple[str, str]] = inline_genes_raw if isinstance(inline_genes_raw, list) else []
    _tmp_fasta: Path | None = None
    if inline_genome and not genome_fasta:
        # Write inline DNA to a temp FASTA file
        _tmp_fasta = Path(tempfile.mktemp(suffix=".fasta"))
        if inline_genes and isinstance(inline_genes, list):
            # Multi-gene FASTA: each gene gets its own header
            lines: list[str] = []
            for entry in inline_genes:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    gene_id, seq = entry
                    lines.append(f">{gene_id}\n{seq}")
                else:
                    lines.append(f">gene_{len(lines)}\n{entry}")
            _tmp_fasta.write_text("\n".join(lines) + "\n")
        else:
            _tmp_fasta.write_text(f">{organism}\n{inline_genome}\n")
        genome_fasta = str(_tmp_fasta)

    use_full = ext.get("gem_use_full_model", "false").lower() in ("true", "1", "yes")

    # ---- Full-model early path (doc/24) ----
    # When use_full_model=true, skip genome reconstruction entirely and
    # load a pre-built genome-scale model from BiGG instead.
    if use_full:
        return _run_gem_full_model(
            organism=organism,
            medium_name=medium_name,
            dynamic=dynamic,
            duration=duration,
            dt=dt,
            medium_override=medium_override or None,
            max_growth_rate=_max_growth_rate_override,
        )

    # Fallback: check #species for genome
    if not genome_fasta:
        groups = _group_prefixed(ext, "species.")
        for name, attrs in groups.items():
            if "genome" in attrs:
                genome_fasta = attrs["genome"]
                if organism == "e_coli_k12":
                    organism = name
                break

    if not genome_fasta:
        raise SimConfigError(
            "GEM backend requires either:\n"
            "  #gem organism=<name> genome=<path>\n"
            "  #gem organism=<name> (with inline DNA block)\n"
            "  or #species name=<name> genome=<path>")

    result = run_gem_pipeline(
        genome_fasta=genome_fasta,
        organism=organism,
        use_database_interactions=use_database,
        include_spontaneous=include_spontaneous,
        run_gapfill=run_gapfill,
        target_organism=target_organism,
    )

    # ---- Stage 6: FBA simulation ----
    growth_rate = 0.0
    fba_status = "skipped"
    key_fluxes: dict[str, float] = {}
    analysis: dict[str, Any] = {}
    _extra_meta: dict[str, Any] = {}

    if result.consensus is not None:
        try:
            from helixlang.plugins.gem.biomass import build_biomass_reaction
            from helixlang.plugins.gem.bridge import (
                _parse_equation_to_stoich,
                build_enzyme_capacity,
                consensus_to_metabolic_model,
            )
            from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis, Reaction

            model = consensus_to_metabolic_model(result.consensus)

            # Add gapfill reactions (exchange reactions from gap-filling)
            if result.gapfill:
                for rxn in result.gapfill.added_reactions:
                    stoich = _parse_equation_to_stoich(rxn.equation)
                    if stoich and rxn.reaction_id not in model.reactions:
                        model.add_reaction(Reaction(
                            id=rxn.reaction_id,
                            name=rxn.reaction_id,
                            stoichiometry=stoich,
                            lower_bound=-1000.0,
                            upper_bound=1000.0,
                            subsystem="exchange",
                    ))

            # Add essential core reactions that bottom-up may miss
            # (genes not in FASTA or EC mapping incomplete)
            # NOTE: must run BEFORE transport so that core metabolites (o2,
            # co2, etc.) exist in model.metabolites when transport reactions
            # are checked.
            _add_gem_core_reactions(model)

            # Add transport reactions to connect exchange and internal compartments
            _add_gem_transport_reactions(model)

            # Add biomass reaction to the model
            biomass = build_biomass_reaction(organism)
            biomass_stoich: dict[str, float] = {}
            for c in biomass.components:
                met = c.metabolite_id
                # Try multiple name variants to match model metabolites:
                # 1. Original (e.g. "ala-L_c")
                # 2. Stripped compartment (e.g. "ala-L")
                # 3. With _e suffix (e.g. "ala-L_e") — extracellular pool
                candidates = [met]
                if met.endswith(("_c", "_e", "_p")):
                    candidates.append(met[:-2])
                    candidates.append(met[:-2] + "_e")
                matched_met = next(
                    (m for m in candidates if m in model.metabolites),
                    None,
                )
                if matched_met is not None:
                    biomass_stoich[matched_met] = (
                        biomass_stoich.get(matched_met, 0.0) + c.coefficient
                    )
            # Remove zero-coeff metabolites
            biomass_stoich = {k: v for k, v in biomass_stoich.items() if abs(v) > 1e-12}
            if biomass_stoich:
                model.add_reaction(Reaction(
                    id="BIOMASS_reaction",
                    name="BIOMASS_reaction",
                    stoichiometry=biomass_stoich,
                    lower_bound=0.0,
                    upper_bound=1000.0,
                    subsystem="biomass",
                ))
                model.set_biomass("BIOMASS_reaction")

            fba = FluxBalanceAnalysis(model)

            # Wire expression inference if enabled (doc/20 §14)
            _enzyme_levels: dict[str, float] = {}
            if expression_enabled:
                try:
                    from helixlang.plugins.gem.grn_inference import GRNInferenceResult
                    from helixlang.plugins.omics.expression_inference import (
                        ExpressionModel,
                        infer_expression,
                    )
                    if isinstance(result.grn, GRNInferenceResult) and \
                            result.grn.regulatory_edges:
                        # Collect DSL expression_level overrides from #gene
                        _expr_overrides: dict[str, float] = {}
                        for g in program.genes:
                            elvl = g.fields.get("expression_level")
                            if elvl is not None:
                                try:
                                    _expr_overrides[g.name] = float(elvl)
                                except ValueError:  # SILENTBENIGN - keep default expression
                                    pass
                        # Build expression model with DSL overrides
                        _expr_model = ExpressionModel()
                        if _expr_overrides:
                            for gid, lvl in _expr_overrides.items():
                                _expr_model.promoter_strength[gid] = lvl
                                _expr_model.rbs_strength[gid] = 1.0
                        _enzyme_levels = infer_expression(
                            grn_result=result.grn,
                            annotations=result.annotations,
                            model=_expr_model if _expr_overrides else None,
                        )
                        # Apply DSL overrides directly (take precedence
                        # over inferred values)
                        _enzyme_levels.update(_expr_overrides)
                        fba.set_enzyme_levels(_enzyme_levels)
                except Exception as exc:
                    result.warnings.append(
                        f"Expression inference skipped: {exc}")

            # Wire enzyme constraints if kcat predictions exist AND we
            # have expression data (enzyme levels).  Without expression
            # data the enzyme_levels dict is empty, causing _build_and_solve
            # to cap every enzyme-gated reaction at ub = kc * 0 = 0,
            # which flips glycolysis/TCA to the reverse direction and
            # produces garbage FBA results.
            if result.kcat_predictions and _enzyme_levels:
                try:
                    from helixlang.plugins.gem.bridge import build_enzyme_capacity
                    ec = build_enzyme_capacity(
                        result.consensus,
                        result.kcat_predictions,
                    )
                    # Phase C: apply DSL #enzyme kcat overrides (doc/25 G3)
                    for e_decl in program.enzymes:
                        if e_decl.kcat is not None and e_decl.reaction in ec.kcat:
                            ec.kcat[e_decl.reaction] = e_decl.kcat
                    fba.set_enzyme_capacity(ec)
                except Exception as exc:
                    result.warnings.append(
                        f"Enzyme capacity setup skipped: {exc}")

            # Phase VII: GRN -> FBA closed loop (doc/25 G7)
            # Apply regulatory edges as FBA bounds: repressed genes reduce
            # reaction upper bounds; activated genes ensure minimum flux.
            if result.grn is not None and result.grn.regulatory_edges:
                try:
                    from helixlang.plugins.gem.bridge import apply_regulatory_bounds
                    _gpr_map: dict[str, list[str]] = {}
                    if result.consensus is not None:
                        for rxn_id, genes in getattr(
                                result.consensus, "gene_reaction_rules",
                                {}).items():
                            if isinstance(genes, list):
                                for g in genes:
                                    _gpr_map.setdefault(cast(str, g), []).append(rxn_id)
                    _n = apply_regulatory_bounds(
                        model, result.grn.regulatory_edges, _gpr_map)
                    _extra_meta["grn_bounds_applied"] = _n
                except Exception as exc:
                    result.warnings.append(
                        f"GRN regulatory bounds skipped: {exc}")

            # Set exchange uptake bounds from medium preset or #media
            _set_gem_medium(fba, medium_name, program, model,
                            medium_override=medium_override or None)

            # Determine organism-specific growth rate cap
            _org_key = organism.lower().replace(" ", "_").replace(".", "")
            max_mu = _ORGANISM_MAX_GROWTH_RATE.get(
                _org_key,
                _ORGANISM_MAX_GROWTH_RATE["default"],
            )
            # doc/25 Phase F: DSL max_growth_rate override
            if _max_growth_rate_override is not None:
                max_mu = _max_growth_rate_override
            # Determine initial substrate from medium preset
            _medium_uptake = _MEDIUM_PRESETS.get(
                medium_name, _MEDIUM_PRESETS["glucose_minimal"])
            _init_glucose = _medium_uptake.get("glc-D_e", 0.0)

            if dynamic and _init_glucose > 0.0:
                # ---- Dynamic FBA path (doc/20 §15) ----
                # Only used for glucose-based media; photoautotrophic
                # media use PhotoautotrophicFluxBalance (doc/22 §7).
                from helixlang.plugins.runtime.metabolism import (
                    DynamicFBAConfig,
                    DynamicFluxBalance,
                )
                dyn_cfg = DynamicFBAConfig(
                    dt_h=dt,
                    initial_biomass_gdw=0.01,
                    initial_glucose_mm=_init_glucose,
                    initial_acetate_mm=0.0,
                    max_glucose_uptake=_medium_uptake.get("glc-D_e", 10.0),
                    max_growth_rate=max_mu,
                )
                batch = DynamicFluxBalance(model, config=dyn_cfg, fba=fba)
                n_steps = max(1, int(duration / dt))
                dyn_trajectory: list[dict] = []
                for _ in range(n_steps):
                    dyn_trajectory.append(batch.step())
                # Summarise dynamic results
                if dyn_trajectory:
                    # Report the maximum (exponential-phase) growth rate
                    # from the trajectory, not the final step which may
                    # reflect post-substrate-depletion stationary phase
                    # (doc/22 §6 Step 6: target 0.7–0.9 h⁻¹).
                    growth_rate = max(
                        _row.get("growth_rate", _row.get("mu", 0.0))
                        for _row in dyn_trajectory
                    )
                    # Find the last trajectory entry where substrate
                    # (glucose) is still available AND the model is
                    # actively growing — this is the end of the
                    # productive exponential phase.  The final biomass
                    # and key_fluxes come from this entry, not from
                    # post-depletion stationary phase or infeasible FBA.
                    _glucose_key = "glucose"
                    _gr_key = "growth_rate"
                    _end_entry = dyn_trajectory[-1]
                    for _row in reversed(dyn_trajectory):
                        if (_row.get(_glucose_key, 0.0) > 0.01
                                and _row.get(_gr_key, 0.0) > 1e-6):
                            _end_entry = _row
                            break
                    final_biomass = _end_entry.get(
                        "biomass", _end_entry.get("total_biomass", 0.0))
                    key_fluxes = {
                        k: round(v, 4)
                        for k, v in _end_entry.items()
                        if isinstance(v, (int, float)) and abs(v) > 1e-6
                    }
                else:
                    final_biomass = 0.0
                    growth_rate = 0.0
                    key_fluxes = {}
                fba_status = "ok"
                _extra_meta["dynamic"] = True
                _extra_meta["duration_h"] = duration
                _extra_meta["dt_h"] = dt
                _extra_meta["final_biomass"] = final_biomass
                _extra_meta["trajectory_steps"] = len(dyn_trajectory)
            elif dynamic and medium_name in ("bg11", "photoautotrophic"):
                # ---- Photoautotrophic dFBA (doc/22 §7) ----
                from helixlang.plugins.runtime.metabolism import (
                    DynamicFBAConfig,
                    PhotoautotrophicFluxBalance,
                )
                photo_cfg = DynamicFBAConfig(
                    substrate_type="co2",
                    dt_h=dt,
                    initial_biomass_gdw=0.01,
                    # Batch CO₂ pool for photoautotrophic growth.
                    # 10 mM CO₂ at 5% sparging gives sustained carbon
                    # supply (Stumm & Morgan 1996: 5% CO₂ ≈ 5 mM
                    # saturated; 10 mM allows longer exponential phase
                    # before depletion).
                    co2_initial_mm=10.0,
                    # Calvin cycle capacity raised to 50 mmol/gDW/h to
                    # match literature CO₂ fixation rates for
                    # Synechocystis at 200 µmol photons/m²/s
                    # (Knoop 2013, est. 40–60 mmol/gDW/h).
                    co2_max_uptake=50.0,
                    co2_half_saturation_mm=0.5,
                    max_growth_rate=max_mu,
                    max_biomass_gdw=50.0,
                )
                batch = PhotoautotrophicFluxBalance(  # type: ignore[assignment]
                    model=model, config=photo_cfg, fba=fba)
                n_steps = max(1, int(duration / dt))
                dyn_trajectory = []
                for _ in range(n_steps):
                    dyn_trajectory.append(batch.step())
                if dyn_trajectory:
                    # Report the maximum (exponential-phase) growth rate
                    # from the trajectory, not the final step which may
                    # reflect post-CO₂-depletion stationary phase
                    # (doc/22 §7.5: target 0.14 h⁻¹ during growth).
                    growth_rate = max(
                        _row.get("growth_rate", 0.0)
                        for _row in dyn_trajectory
                    )
                    # Find the last entry where CO₂ is still available
                    # for productive growth AND the model is actively
                    # growing (doc/22 §7.5).
                    _co2_key = "co2"
                    _gr_key = "growth_rate"
                    _end_entry = dyn_trajectory[-1]
                    for _row in reversed(dyn_trajectory):
                        if (_row.get(_co2_key, 0.0) > 0.01
                                and _row.get(_gr_key, 0.0) > 1e-6):
                            _end_entry = _row
                            break
                    final_biomass = _end_entry.get("biomass", 0.0)
                    key_fluxes = {
                        k: round(v, 4)
                        for k, v in _end_entry.items()
                        if isinstance(v, (int, float)) and abs(v) > 1e-6
                    }
                else:
                    final_biomass = 0.0
                    growth_rate = 0.0
                    key_fluxes = {}
                fba_status = "ok"
                _extra_meta["dynamic"] = True
                _extra_meta["duration_h"] = duration
                _extra_meta["dt_h"] = dt
                _extra_meta["final_biomass"] = final_biomass
                _extra_meta["trajectory_steps"] = len(dyn_trajectory)
            else:
                # ---- Static FBA path ----
                fba.solve(objective="biomass", maximize=True)
                analysis = fba.analyze()
                growth_rate = analysis.get("growth_rate_per_hour", 0.0)
                # Cap growth rate at organism-specific maximum to
                # compensate for simplified model overestimation
                growth_rate = min(growth_rate, max_mu)
                key_fluxes = {
                    k: round(v, 4)
                    for k, v in analysis.get("key_fluxes", {}).items()
                    if abs(v) > 1e-6
                }
                fba_status = "ok"
        except Exception as exc:
            fba_status = f"failed: {exc}"
            result.warnings.append(f"FBA simulation failed: {exc}")

    # Build output rows from pipeline stages
    columns = [
        "stage", "status", "genes_annotated", "reactions_total",
        "grn_edges", "kcat_predictions", "km_estimates",
    ]
    rows: list[dict[str, Any]] = [
        {
            "stage": "annotation",
            "status": "ok" if result.annotated_genes > 0 else "failed",
            "genes_annotated": result.annotated_genes,
            "reactions_total": 0,
            "grn_edges": 0,
            "kcat_predictions": 0,
            "km_estimates": 0,
        },
        {
            "stage": "reconstruction",
            "status": "ok" if result.consensus else "failed",
            "genes_annotated": result.annotated_genes,
            "reactions_total": result.final_reaction_count,
            "grn_edges": 0,
            "kcat_predictions": 0,
            "km_estimates": 0,
        },
        {
            "stage": "grn",
            "status": "ok" if result.grn else "skipped",
            "genes_annotated": result.annotated_genes,
            "reactions_total": result.final_reaction_count,
            "grn_edges": result.grn.total_edges if result.grn else 0,
            "kcat_predictions": 0,
            "km_estimates": 0,
        },
        {
            "stage": "kinetics",
            "status": "ok",
            "genes_annotated": result.annotated_genes,
            "reactions_total": result.final_reaction_count,
            "grn_edges": result.grn.total_edges if result.grn else 0,
            "kcat_predictions": len(result.kcat_predictions),
            "km_estimates": len(result.km_estimates),
        },
        {
            "stage": "fba_simulation",
            "status": fba_status,
            "genes_annotated": result.annotated_genes,
            "reactions_total": result.final_reaction_count,
            "grn_edges": result.grn.total_edges if result.grn else 0,
            "kcat_predictions": len(result.kcat_predictions),
            "km_estimates": len(result.km_estimates),
        },
    ]

    return SimResult(
        backend="gem",
        columns=columns,
        rows=rows,
        meta={
            "organism": organism,
            "medium": medium_name,
            "stages_completed": result.stages_completed,
            "growth_rate_per_hour": round(growth_rate, 4),
            "biomass_yield": round(analysis.get("biomass_per_glucose", 0.0), 4),
            "key_fluxes": key_fluxes,
            "warnings": result.warnings,
            "errors": result.errors,
            "summary": result.summary(),
            **_extra_meta,
        },
    )

def _run_human_simulation(program: Program) -> SimResult:
    """``#sim kind=human`` — virtual-patient simulation (doc/27+28+29).

    Parses ``#person``, ``#trait``, ``#disease``, ``#disease_gene``,
    ``#disease_metabolite``, ``#drug``, ``#pd_effect`` annotations from
    ``program.extensions``, builds a
    :class:`~helixlang.plugins.human.virtual_patient.VirtualPatientConfig`, runs
    the full PBPK→PD→labs→vitals→recovery integration loop, and returns
    the complete time-series as a ``SimResult`` with one row per hour.

    Falls back to the legacy
    :class:`~helixlang.plugins.human.simulation.HumanSimulation` when no
    person/drug annotations are present (backward compatible).
    """
    ext = program.extensions

    # --- Detect which backend to use ---
    has_person = any(k.startswith("person_") for k in ext)
    has_drugs = "drugs" in ext and ext["drugs"]

    if not has_person and not has_drugs:
        return _run_human_simulation_legacy(program)

    return _run_virtual_patient(program)

def _run_human_simulation_legacy(program: Program) -> SimResult:
    """Legacy ``#sim kind=human`` backend using HumanSimulation (doc/27)."""
    from helixlang.plugins.human.simulation import (
        HumanSimulation,
        HumanSimulationConfig,
    )

    ext = program.extensions
    duration_days = _opt_float(ext, "duration_days", 30.0)
    config = HumanSimulationConfig(
        total_duration_days=duration_days,
        dfa_dt_h=_opt_float(ext, "dfa_dt_h", 1.0),
        target_tissue=ext.get("target_tissue", "liver"),
        base_model_path=ext.get("base_model_path", ""),
    )
    result = HumanSimulation(config).run()
    row = {
        "duration_days": duration_days,
        "auc_plasma": result.auc_plasma,
        "time_in_range_fraction": result.time_in_therapeutic_range_fraction,
        "efficacy_score": result.overall_efficacy_score,
        "toxicity_events": len(result.toxicity_events),
    }
    columns = _select_columns(program, [row], default=[
        "duration_days",
        "auc_plasma",
        "time_in_range_fraction",
        "efficacy_score",
        "toxicity_events",
    ])
    return SimResult(
        backend="human", columns=columns, rows=[_project(row, columns)],
        meta={
            "kind": "human",
            "therapeutic_response_time_h": result.therapeutic_response_time_h,
            "time_points": len(result.time_h),
        },
    )

def _run_virtual_patient(program: Program) -> SimResult:
    """Full virtual-patient simulation using VirtualPatient (doc/28+29).

    Builds a VirtualPatientConfig from helix annotations and runs the
    complete PBPK→PD→labs→vitals→recovery integration loop.
    """
    from helixlang.plugins.human.genotype import create_default_genotype
    from helixlang.plugins.human.virtual_patient import VirtualPatient, VirtualPatientConfig

    ext = program.extensions

    # --- Build genotype ---
    genotype = create_default_genotype()
    _build_genotype_from_helix(genotype, ext)

    # --- Build traits ---
    traits = _build_traits_from_helix(ext)

    # --- Build disease ---
    disease = _build_disease_from_helix(ext)
    disease_name = ext.get("disease_name", "")

    # --- Build drugs ---
    drugs = _build_drugs_from_helix(ext)

    # --- Build PD effects ---
    pd_effects = _build_pd_from_helix(ext)

    # --- Build QSP binding models ---
    qsp_bindings = _build_qsp_bindings_from_helix(ext)

    # --- Build endocrine config ---
    endocrine_config = _build_endocrine_config_from_helix(ext)

    # --- Build immune config ---
    immune_config = _build_immune_config_from_helix(ext)

    # --- Build tumor biopsy (doc/33 Phase 4) ---
    tumor_biopsy = _build_tumor_biopsy_from_helix(ext)

    # --- Auto-infer PD from drug target when no explicit #pd_effect ---
    if drugs and not pd_effects:
        from helixlang.plugins.human.pharmacodynamics import infer_pd_from_drug as _infer_pd
        for drug in drugs:
            tp = drug.molecule.target_protein
            if tp:
                key = drug.molecule.name.lower().replace(" ", "_").replace("-", "_")
                pd_effects[key] = _infer_pd(
                    drug.molecule.name,
                    target_protein=tp,
                    binding_kd_um=drug.molecule.binding_affinity_kd_um,
                    mw_da=drug.molecule.molecular_weight_da,
                )

    # --- Simulation parameters ---
    duration_days = _opt_float(ext, "duration_days", 30.0)
    dt_h = _opt_float(ext, "dfa_dt_h", 1.0)
    output_res_h = _opt_float(ext, "output_resolution_h", 1.0)

    config = VirtualPatientConfig(
        genotype=genotype,
        traits=traits,
        disease=disease,
        disease_profile_name=disease_name,
        drugs=drugs,
        pharmacodynamics=pd_effects,
        qsp_bindings=qsp_bindings.get("qsp_bindings", []),
        endocrine_configs=endocrine_config.get("endocrine_configs", []),
        immune_configs=immune_config.get("immune_configs", []),
        tumor_biopsy=tumor_biopsy,
        total_duration_days=duration_days,
        dfa_dt_h=dt_h,
        output_time_resolution_h=output_res_h,
    )

    patient = VirtualPatient(config)
    result = patient.run()

    # --- Build output rows (one row per recorded time point) ---
    n = len(result.time_h)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {
            "time_h": result.time_h[i],
            "systolic_bp": result.systolic_bp[i] if i < len(result.systolic_bp) else 0.0,
            "diastolic_bp": result.diastolic_bp[i] if i < len(result.diastolic_bp) else 0.0,
            "heart_rate": result.heart_rate[i] if i < len(result.heart_rate) else 0.0,
            "temperature": result.temperature[i] if i < len(result.temperature) else 0.0,
            "spo2_pct": result.spo2_pct[i] if i < len(result.spo2_pct) else 0.0,
            "respiratory_rate": result.respiratory_rate[i] if i < len(result.respiratory_rate) else 0.0,
            "qtc_ms": result.qtc_ms[i] if i < len(result.qtc_ms) else 0.0,
            "alt": result.alt[i] if i < len(result.alt) else 0.0,
            "ast": result.ast[i] if i < len(result.ast) else 0.0,
            "creatinine": result.creatinine[i] if i < len(result.creatinine) else 0.0,
            "egfr": result.egfr[i] if i < len(result.egfr) else 0.0,
            "wbc": result.wbc[i] if i < len(result.wbc) else 0.0,
            "hemoglobin": result.hemoglobin[i] if i < len(result.hemoglobin) else 0.0,
            "platelets": result.platelets[i] if i < len(result.platelets) else 0.0,
            "glucose": result.glucose[i] if i < len(result.glucose) else 0.0,
            "hba1c": result.hba1c[i] if i < len(result.hba1c) else 0.0,
            "crp": result.crp[i] if i < len(result.crp) else 0.0,
            "bilirubin": result.bilirubin[i] if i < len(result.bilirubin) else 0.0,
            "albumin": result.albumin[i] if i < len(result.albumin) else 0.0,
            "inr": result.inr[i] if i < len(result.inr) else 0.0,
            "sodium": result.sodium[i] if i < len(result.sodium) else 0.0,
            "potassium": result.potassium[i] if i < len(result.potassium) else 0.0,
            "lactate": result.lactate[i] if i < len(result.lactate) else 0.0,
            "calcium": result.calcium[i] if i < len(result.calcium) else 0.0,
            "phosphate": result.phosphate[i] if i < len(result.phosphate) else 0.0,
            "chloride": result.chloride[i] if i < len(result.chloride) else 0.0,
            "bicarbonate": result.bicarbonate[i] if i < len(result.bicarbonate) else 0.0,
            "ldl": result.ldl[i] if i < len(result.ldl) else 0.0,
            "hdl": result.hdl[i] if i < len(result.hdl) else 0.0,
            "triglycerides": result.triglycerides[i] if i < len(result.triglycerides) else 0.0,
            "disease_severity": result.disease_severity[i] if i < len(result.disease_severity) else 0.0,
            "weight_kg": result.weight_kg[i] if i < len(result.weight_kg) else 0.0,
            # doc/30-31 new channels
            "cortisol": result.cortisol[i] if i < len(result.cortisol) else 0.0,
            "insulin": result.insulin[i] if i < len(result.insulin) else 0.0,
            "glucose_endocrine": result.glucose_endocrine[i] if i < len(result.glucose_endocrine) else 0.0,
            "tsh": result.tsh[i] if i < len(result.tsh) else 0.0,
            "ft4": result.ft4[i] if i < len(result.ft4) else 0.0,
            "il6": result.il6[i] if i < len(result.il6) else 0.0,
            "tnf_alpha": result.tnf_alpha[i] if i < len(result.tnf_alpha) else 0.0,
            "neutrophils": result.neutrophils[i] if i < len(result.neutrophils) else 0.0,
            "tumor_volume": result.tumor_volume[i] if i < len(result.tumor_volume) else 0.0,
            "nephron_mass": result.nephron_mass[i] if i < len(result.nephron_mass) else 0.0,
            "fibrosis_stage": result.fibrosis_stage[i] if i < len(result.fibrosis_stage) else 0.0,
            "beta_cell_function": result.beta_cell_function[i] if i < len(result.beta_cell_function) else 0.0,
        }
        # Drug concentrations
        for dk, dc in result.drug_concentrations.items():
            row[f"drug_{dk}"] = dc[i] if i < len(dc) else 0.0
        rows.append(row)

    # --- Select columns ---
    default_cols = [
        "time_h", "systolic_bp", "diastolic_bp", "heart_rate",
        "temperature", "spo2_pct", "respiratory_rate", "qtc_ms",
        "alt", "ast", "creatinine", "egfr", "wbc", "hemoglobin",
        "platelets", "glucose", "hba1c", "crp",
        "bilirubin", "albumin", "inr",
        "sodium", "potassium", "lactate",
        "calcium", "phosphate", "chloride", "bicarbonate",
        "ldl", "hdl", "triglycerides",
        "disease_severity", "weight_kg",
        "cortisol", "insulin", "glucose_endocrine", "tsh", "ft4",
        "il6", "tnf_alpha", "neutrophils",
        "tumor_volume", "nephron_mass", "fibrosis_stage", "beta_cell_function",
    ]
    # Add drug concentration columns
    for dk in result.drug_concentrations:
        default_cols.append(f"drug_{dk}")

    columns = _select_columns(program, rows, default=default_cols)

    # --- Summary meta ---
    summary = result.summary()
    meta: dict[str, Any] = {
        "kind": "human_virtual_patient",
        "duration_days": duration_days,
        "time_points": n,
        "summary": summary,
        "ddi_alerts": result.ddi_alerts,
        "clinical_events": result.clinical_events,
        "overall_efficacy_score": result.overall_efficacy_score,
        "total_toxicity_events": result.total_toxicity_events,
    }

    return SimResult(
        backend="human",
        columns=columns,
        rows=[_project(r, columns) for r in rows],
        meta=meta,
    )

def _run_population(program: Program) -> ColonyResult:
    config = _build_population_config(program)
    if config.dfba_enabled and not program.media:
        raise SimConfigError(
            "backend=population with dfba=true requires at least one "
            "#media declaration (shared substrate fields)")

    # Phase G: auto-attach GEM model if a genome is specified
    ext = program.extensions
    genome_path = ext.get("genome", "")
    if genome_path and config.dfba_enabled:
        try:
            from helixlang.plugins.apps.gem_pipeline import run_gem_pipeline
            _gem_result = run_gem_pipeline(
                genome_fasta=genome_path,
                organism=ext.get("organism", "e_coli_k12"),
                medium=ext.get("medium", "glucose_minimal"),
            )
            if (_gem_result.metabolic_model is not None
                    and getattr(_gem_result, "growth_rate", 0.0) > 0):
                config.metabolic_model = _gem_result.metabolic_model
        except Exception as exc:  # noqa: BLE001
            # A requested GEM must not silently degrade to the E. coli core
            # proxy (doc/36 §3ξ.3, F3) without an explicit opt-in.
            from helixlang.core import fidelity
            if not fidelity.opt_in("--low-fidelity"):
                from helixlang.core.errors import ModelMissingError
                raise ModelMissingError(
                    f"GEM for genome {genome_path!r}", "apps",
                    detail=f"run_gem_pipeline failed: {exc}",
                ) from exc

    cells = _seed_cells(config, config.max_size)
    population = CellPopulation3D(
        cells, config, seed=_opt_int_or_none(program.config.sim, "seed", None))
    rows: list[dict[str, Any]] = []
    for _ in range(max(1, program.config.ticks)):
        stats = population.step()
        strat = population.dfba_stratification()
        rows.append({**stats, **strat})
    columns = _select_columns(program, rows, default=_POP_DEFAULT_COLUMNS)
    meta: dict[str, Any] = {
        "colony_observables": population.colony_observables(),
    }
    if config.genome is not None:
        meta["genome"] = {
            "genes": config.genome.n_genes,
            "edges": config.genome.n_edges,
            "tf_map": config.genome.tf_map,
            "grn_mode": config.genome.grn_mode,
            "active_gene_budget": config.genome.active_gene_budget,
        }
    if config.trace_streaming:
        meta["trace"] = population.trace
    return ColonyResult(
        backend="population", columns=columns,
        rows=[_project(row, columns) for row in rows],
        meta=meta,
    )

def _run_calibration(program: Program) -> ScoreResult:
    sim = program.config.sim
    result = run_whole_cell_calibration(
        minutes=_opt_int(sim, "minutes", 60),
        n_samples=_opt_int(sim, "n_samples", 60),
        refine_rounds=_opt_int(sim, "refine_rounds", 2),
        fit_seed=_opt_int(sim, "fit_seed", 0),
        adder_noise_std=_opt_float(sim, "adder_noise_std", 0.0),
        n_cells=_opt_int(sim, "n_cells", 1),
    )
    row: dict[str, Any] = {
        "best": result["fitted"],
        "sse": result["fit"]["sse"],
        "n_samples": result["fit"]["n_samples"],
        "passed": result["passed"],
    }
    columns = _select_columns(program, [row], default=list(row))
    return ScoreResult(
        backend="calibration", columns=columns,
        rows=[_project(row, columns)],
        meta={"relative_error": result["relative_error"],
              "recovered": result["recovered"],
              "truth": result["truth"]},
    )

def _run_benchmark(program: Program) -> ScoreResult:
    sim = program.config.sim
    config = VirtualCellBenchConfig(
        truth_biomass_to_atp=_opt_float(sim, "truth_biomass_to_atp", 5.0e6),
        truth_maintenance_atp_per_min=_opt_float(
            sim, "truth_maintenance_atp_per_min", 2.5e7),
        calibration_uptake=_opt_float_dict(sim, "calibration_uptake",
                                           {"GLC": 10.0}),
        prediction_uptake=_opt_float_dict(sim, "prediction_uptake",
                                          {"GLC": 20.0}),
        calibration_minutes=_opt_int(sim, "calibration_minutes", 20),
        prediction_minutes=_opt_int(sim, "prediction_minutes", 60),
        n_samples=_opt_int(sim, "n_samples", 150),
        fit_seed=_opt_int(sim, "fit_seed", 0),
        refine_rounds=_opt_int(sim, "refine_rounds", 3),
    )
    result = run_virtual_cell_benchmark(config)
    row: dict[str, Any] = {
        "scores": {
            "calibration_recovered": result["calibration_recovered"],
            "prediction_matches": result["prediction_matches"],
            "energy_rel_error": result["energy_rel_error"],
            "biomass_to_atp_rel_error": result["biomass_to_atp_rel_error"],
        },
        "passed": result["passed"],
        "all_passed": result["passed"],
    }
    columns = _select_columns(program, [row], default=list(row))
    return ScoreResult(
        backend="benchmark", columns=columns,
        rows=[_project(row, columns)],
        meta={"fitted_biomass_to_atp": result["fitted_biomass_to_atp"]},
    )


_ODE_MODEL_COLUMNS = [
    "name",
    "species",
    "t_end",
    "initial",
    "final",
    "conserved_sum",
    "max_abs_rate",
]


def _run_ode_model(program: Program) -> ScoreResult:
    """``#sim kind=ode_model`` — integrate a user-authored ODE model.

    Phase D (doc/42 RT-1): this is the "author biology in Helix" backend.
    It consumes ``#model`` / ``#species`` / ``#reaction`` annotations from
    :attr:`Program.sim_extensions`, builds an explicit-order RK4 integrator,
    and reports each species' initial/final concentration on an arbitrary
    well-mixed model.  Deterministic (no RNG), so the output is
    golden-verifiable.

    Species not covered by any ``#reaction`` are treated as held constant
    (rate zero).  The ``final`` state is reported at ``t = t_end``.
    """
    sim = program.sim_extensions or {}
    model = sim.get("ode_model")
    if not isinstance(model, list) or not model:
        raise SimConfigError(
            "#sim kind=ode_model requires a #model declaration")
    meta = model[0]
    name = str(meta.get("name", "ode"))
    try:
        params = {
            k: float(meta[k]) for k in ("k1", "k2")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise SimConfigError("ode model needs numeric k1 and k2") from exc

    species_rows = sim.get("ode_species") or []
    if not isinstance(species_rows, list) or not species_rows:
        raise SimConfigError("ode model needs at least one #species")
    initial: dict[str, float] = {}
    units: dict[str, str] = {}
    for sp in species_rows:
        sname = str(sp.get("name", ""))
        if not sname:
            continue
        initial[sname] = float(sp.get("initial", 0.0))
        units[sname] = str(sp.get("units", "") or "")

    rate_exprs: dict[str, str] = {}
    for rxn in sim.get("ode_reaction") or []:
        target = str(rxn.get("species", ""))
        expr = str(rxn.get("expr", ""))
        if target and expr:
            rate_exprs[target] = expr

    species_order = sorted(initial)
    if not species_order:
        raise SimConfigError("ode model species list is empty")
    state = [initial[s] for s in species_order]
    get: dict[str, Any] = {"__builtins__": {"pow": pow}}
    get.update(params)
    compiled = {s: _compile_rate(expr, species_order, get)
                for s, expr in rate_exprs.items()}
    for s in species_order:
        if s not in compiled:
            compiled[s] = ("0.0", get)

    t_end = _opt_float(meta, "t_end", 10.0)
    steps = _opt_int(meta, "steps", 100)
    dt = t_end / steps

    max_abs_rate: dict[str, float] = {}
    for _ in range(steps):
        rates_now = {s: _eval_rate(tpl, state, species_order)
                     for s, tpl in compiled.items()}
        k1 = [rates_now[s] for s in species_order]
        if max_abs_rate == {}:
            for s in species_order:
                max_abs_rate[s] = abs(k1[species_order.index(s)])
        # RK4 advance, holding un-compiled species constant.
        k2 = [_eval_rate(tpl, [v + 0.5 * dt * k1[i] for i, v in enumerate(state)],
                         species_order) for i, (s, tpl) in enumerate(compiled.items())]
        k3 = [_eval_rate(tpl, [v + 0.5 * dt * k2[i] for i, v in enumerate(state)],
                         species_order) for i, (s, tpl) in enumerate(compiled.items())]
        k4 = [_eval_rate(tpl, [v + dt * k3[i] for i, v in enumerate(state)],
                         species_order) for i, (s, tpl) in enumerate(compiled.items())]
        for i, s in enumerate(species_order):
            if s in compiled:
                state[i] += dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])

    final = {s: state[species_order.index(s)] for s in species_order}
    rows = [{
        "name": name,
        "species": s,
        "t_end": round(t_end, 6),
        "initial": round(initial[s], 6),
        "final": round(final[s], 6),
        "conserved_sum": round(sum(final.values()), 6),
        "max_abs_rate": round(max_abs_rate[s], 6),
    } for s in species_order]

    columns = _select_columns(program, rows, default=_ODE_MODEL_COLUMNS)
    return ScoreResult(
        backend="ode_model", columns=columns,
        rows=[_project(r, columns) for r in rows],
        meta={"model": name, "units": units, "rates": dict(rate_exprs)},
    )


def _compile_rate(expr: str, order: list[str], get: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return (rate_expr, eval_namespace) — expr is kept as a string to avoid
    exec of arbitrary code beyond the allowed math surface."""
    return (expr, get)


def _eval_rate(tpl: tuple[str, dict[str, Any]], state: list[float],
               order: list[str]) -> float:
    expr, ns = tpl
    if expr == "0.0":
        return 0.0
    env = dict(ns)
    for name, value in zip(order, state, strict=True):
        env[name] = value
    return float(eval(expr, {"__builtins__": {"pow": pow}}, env))  # noqa: S307


