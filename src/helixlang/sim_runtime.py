"""Simulation-library adapter (``doc/helix-language-wiring.md`` §8).

Maps a parsed :class:`~helixlang.ast_nodes.Program` onto the quantitative
simulation stack, selected by ``#config backend``:

- ``whole_cell``  -> :class:`~helixlang.virtual_cell.VirtualCell` (Phases 1-4)
- ``population``  -> :class:`~helixlang.population.CellPopulation3D` (Phase 5)
- ``fba``         -> :class:`~helixlang.metabolism.FluxBalanceAnalysis` /
                     :class:`~helixlang.metabolism.DynamicFluxBalance`
- ``calibration`` -> :func:`helixlang.apps.whole_cell_calibration.run_whole_cell_calibration`
- ``benchmark``   -> :func:`helixlang.apps.virtual_cell_bench.run_virtual_cell_benchmark`
- ``#sim kind=...`` (W-6) -> one of the registered long-tail backends in
  :data:`_SIM_BACKENDS` (consortium, digital_evolution, stochastic,
  codec_benchmark, synbio_design, protein_fitness, morphogen_gradient,
  protein_structure, fate_analysis, directed_evolution, 3d_morphology,
  omics_calibration, cello_workflow, codon_usage).

``backend=classic`` (the default) returns ``None`` so the CLI keeps the
existing compile -> CellVM path.  A registered ``#sim kind=...`` overrides
even the classic default.  The classic pipeline never reads
``Program.config.sim`` and never sees the structural annotations, so this
module is purely additive.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from helixlang.apps.consortium import (
    QUORUM_SIGNAL_THRESHOLD,
    ROLES,
    SIGNAL_DIFFUSION_UM2_S,
    SIGNAL_EMISSION_PER_STEP,
    ConsortiumConfig,
    ConsortiumSimulator,
)
from helixlang.apps.digital_evolution import (
    DigitalEvolution,
    DigitalEvolutionConfig,
)
from helixlang.apps.dna_storage import benchmark_codecs
from helixlang.apps.fate_analysis import (
    bistability_scan,
    critical_slowing_down,
    switching_rate,
)
from helixlang.apps.morphogen_gradient import (
    MorphogenGene,
    MorphogenGradient,
    MorphogenGradientConfig,
)
from helixlang.apps.omics_calibration import run_omics_calibration_benchmark
from helixlang.apps.protein_evolution import guided_directed_evolution
from helixlang.apps.spatial_dfba import SpatialDFBA, SpatialDFBAConfig
from helixlang.apps.synbio_automation import (
    TruthTable,
    run_cello_workflow,
)
from helixlang.apps.synbio_designer import CassetteConfig, SynBioDesigner
from helixlang.apps.virtual_cell_bench import (
    VirtualCellBenchConfig,
    run_virtual_cell_benchmark,
)
from helixlang.apps.whole_cell_calibration import (
    _fold_rate_from_k_fold,
    run_whole_cell_calibration,
)
from helixlang.ast_nodes import Program
from helixlang.bio_data import cai, get_codon_usage
from helixlang.codon_table import STANDARD_TABLE
from helixlang.compiler import Compiler
from helixlang.dna_codec import translate_dna
from helixlang.environment import (
    ConcentrationField,
    Environment,
    EnvironmentConfig,
)
from helixlang.errors import SimConfigError
from helixlang.grn import GRN
from helixlang.metabolism import (
    DEFAULT_ENZYME_SCALE,
    DEFAULT_GLC_UPTAKE,
    ECOLI_CORE_GENE_REACTIONS,
    ECOLI_CORE_KCAT,
    ECOLI_CORE_MODEL,
    DynamicFBAConfig,
    DynamicFluxBalance,
    EnzymeCapacity,
    FluxBalanceAnalysis,
    MetabolicModel,
    load_model,
)
from helixlang.morphology_3d import PLANT_PRESETS, LSystem3D
from helixlang.population import (
    CellPopulation3D,
    PopulationCell,
    PopulationConfig,
)
from helixlang.protein_fitness import rank_variants
from helixlang.protein_structure import predict_structure
from helixlang.stochastic import (
    TelegraphPromoter,
    gillespie_telegraph,
)
from helixlang.virtual_cell import VirtualCell, VirtualCellConfig

__all__ = [
    "BACKENDS",
    "ColonyResult",
    "FluxResult",
    "HistoryResult",
    "ScoreResult",
    "SimResult",
    "run",
]

BACKENDS = frozenset({
    "classic", "whole_cell", "population", "fba", "calibration", "benchmark",
})


# ============================================================================
# Result types
# ============================================================================
@dataclass
class SimResult:
    """Uniform sim-backend output: selected columns + rows (+ metadata).

    ``columns`` names the CSV/JSON columns; ``rows`` are one record each
    (whole-cell minutes, dFBA batch steps, population ticks, or a single
    score row for ``calibration``/``benchmark``).  ``meta`` carries
    non-tabular payloads (colony observables, per-cell traces).
    """
    backend: str
    columns: list[str]
    rows: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HistoryResult(SimResult):
    """``whole_cell`` history: one row per minute of simulated time."""


@dataclass
class FluxResult(SimResult):
    """``fba``: a static flux vector row or a dFBA batch trace."""


@dataclass
class ColonyResult(SimResult):
    """``population``: per-tick colony statistics + observables in meta."""


@dataclass
class ScoreResult(SimResult):
    """``calibration`` / ``benchmark``: fitted/score table."""


# ============================================================================
# Type coercion (helix-language-wiring.md §6.3)
# ============================================================================
_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}


def _coerce_float(key: str, raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        raise SimConfigError(
            f"sim key {key!r}: expected a number, got {raw!r}") from None


def _coerce_int(key: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError:
        raise SimConfigError(
            f"sim key {key!r}: expected an integer, got {raw!r}") from None


def _coerce_bool(key: str, raw: str) -> bool:
    low = raw.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise SimConfigError(
        f"sim key {key!r}: expected true/false/1/0/yes/no, got {raw!r}")


def _coerce_enum(key: str, raw: str, allowed: frozenset[str]) -> str:
    if raw in allowed:
        return raw
    raise SimConfigError(
        f"sim key {key!r}: expected one of {sorted(allowed)}, got {raw!r}")


def _opt_float(sim: dict[str, str], key: str, default: float) -> float:
    return _coerce_float(key, sim[key]) if key in sim else default


def _opt_int(sim: dict[str, str], key: str, default: int) -> int:
    return _coerce_int(key, sim[key]) if key in sim else default


def _opt_bool(sim: dict[str, str], key: str, default: bool) -> bool:
    return _coerce_bool(key, sim[key]) if key in sim else default


def _opt_enum(sim: dict[str, str], key: str, default: str,
              allowed: frozenset[str]) -> str:
    return _coerce_enum(key, sim[key], allowed) if key in sim else default


def _opt_int_or_none(sim: dict[str, str], key: str,
                     default: int | None) -> int | None:
    if key not in sim:
        return default
    if sim[key].strip() == "none":
        return None
    return _coerce_int(key, sim[key])


def _opt_float_or_none(sim: dict[str, str], key: str,
                       default: float | None) -> float | None:
    if key not in sim:
        return default
    if sim[key].strip() == "none":
        return None
    return _coerce_float(key, sim[key])


def _opt_float_dict(sim: dict[str, str], key: str,
                    default: dict[str, float]) -> dict[str, float]:
    """Coerce ``"a=1.0,b=2.5"`` -> ``{"a": 1.0, "b": 2.5}`` (§6.3 dict)."""
    if key not in sim:
        return dict(default)
    out: dict[str, float] = {}
    for pair in sim[key].split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            raise SimConfigError(
                f"sim key {key!r}: expected comma-separated k=v pairs, "
                f"got {sim[key]!r}")
        k, v = pair.split("=", 1)
        k = k.strip()
        if not k:
            raise SimConfigError(
                f"sim key {key!r}: empty name in {pair!r}")
        out[k] = _coerce_float(f"{key}.{k}", v.strip())
    return out


def _opt_float_list(sim: dict[str, str], key: str,
                    default: tuple[float, ...]) -> tuple[float, ...]:
    """Coerce ``"0.5,1.0,1.5"`` -> ``(0.5, 1.0, 1.5)``."""
    if key not in sim:
        return tuple(default)
    out: list[float] = []
    for part in sim[key].split(","):
        part = part.strip()
        if part:
            out.append(_coerce_float(key, part))
    return tuple(out)


def _opt_str_list(sim: dict[str, str], key: str,
                  default: tuple[str, ...]) -> tuple[str, ...]:
    """Coerce ``"goldman,fountain"`` -> ``("goldman", "fountain")``."""
    if key not in sim:
        return tuple(default)
    return tuple(p.strip() for p in sim[key].split(",") if p.strip())


# ============================================================================
# Dispatch
# ============================================================================
def run(program: Program, backend: str | None = None) -> SimResult | None:
    """Run the backend selected by ``backend`` or ``#config backend``.

    Returns ``None`` for ``classic`` (the CLI keeps the bytecode path).
    """
    name = backend or program.config.backend
    # Long-tail extension point (wiring.md §8.6): an #sim kind=... annotation
    # overrides the first-class backend, including the classic default.
    kind = program.sim_extensions.get("kind")
    if kind in _SIM_BACKENDS:
        return _SIM_BACKENDS[kind](program)
    if name == "classic":
        return None
    if name == "whole_cell":
        return _run_whole_cell(program)
    if name == "population":
        return _run_population(program)
    if name == "fba":
        return _run_fba(program)
    if name == "calibration":
        return _run_calibration(program)
    if name == "benchmark":
        return _run_benchmark(program)
    raise SimConfigError(
        f"unknown backend {name!r}; expected one of "
        "classic, whole_cell, population, fba, calibration, benchmark")


def _select_columns(program: Program, rows: list[dict[str, Any]],
                    default: list[str] | None = None) -> list[str]:
    """``#config output=`` / ``#sim output=`` column selection (§6.7);
    else ``default`` or the first-seen union of the row keys."""
    requested = program.config.output
    ext_output = program.sim_extensions.get("output")
    if ext_output:
        requested = [c.strip() for c in ext_output.split(",") if c.strip()]
    if requested and requested != ["stdout"]:
        return requested
    if default is not None:
        return default
    cols: list[str] = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    return cols


def _project(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    return {k: row.get(k) for k in columns}


# ============================================================================
# whole_cell backend (VirtualCell, Phases 1-4)
# ============================================================================
_VC_FLOAT_FIELDS = (
    "division_energy", "adder_volume_um3", "adder_noise_std",
    "volume_init_um3", "biomass_to_volume_pg_per_min",
    "cell_density_dry_pg_um3", "surface_exponent", "c_period_min",
    "d_period_min", "doubling_time_min", "energy_init",
    "maintenance_atp_per_min", "biomass_to_atp", "transcription_atp_per_nt",
    "translation_atp_per_aa", "protein_yield_per_mrna", "minutes_per_step",
    "enzyme_scale", "protein_mass_fraction", "frac_cotranslational_fold",
    "folding_atp_per_protein", "misfold_rate_per_min",
    "aggregation_rate_per_min", "degraded_rate_per_min",
    "protein_half_life_min",
)

# sim key -> VirtualCellConfig field (names differ for the _enabled booleans)
_VC_BOOL_FIELDS = (
    ("surface_scaling", "surface_scaling"),
    ("enzyme_capacity", "enzyme_capacity_enabled"),
    ("metabolite_pools", "metabolite_pools_enabled"),
)

_VC_ENUM_FIELDS = (
    ("division_rule", frozenset({"energy", "adder"})),
    ("replication_mode", frozenset({"flat", "cooper_helmstetter"})),
    ("protein_maturation_mode", frozenset({"instant", "chaperone"})),
)

_VC_DEFAULT_COLUMNS = [
    "age", "energy", "alive", "divisions", "mass", "volume_um3",
    "added_volume_um3", "biomass_flux", "phase",
]


def _build_grn(program: Program, noise_seed: int | None = None) -> GRN:
    """GRN from #gene/#promoter/#regulate (mirrors ``CellVM``).

    Promoter strengths become thresholds; a negative promoter strength is
    constitutive (active from tick 0).  Per-gene ``threshold=`` /
    ``initial_level=`` fields override the promoter-derived defaults
    (helix-language-wiring.md §7.2).
    """
    grn = GRN(noise_enabled=False, noise_seed=noise_seed)
    prom_by_name = {p.name: p for p in program.promoters}
    for p in program.promoters:
        initial = 1.0 if p.strength < 0 else 0.0
        grn.add_gene(p.name, threshold=p.strength, initial_level=initial)
    for g in program.genes:
        if g.promoter and g.promoter in prom_by_name:
            threshold = float(
                g.fields.get("threshold", prom_by_name[g.promoter].strength))
            initial = 0.0
        else:
            threshold = float(g.fields.get("threshold", -1.0))
            initial = 1.0
        if "initial_level" in g.fields:
            initial = float(g.fields["initial_level"])
        grn.add_gene(g.name, threshold, initial_level=initial)
    for r in program.regulations:
        grn.add_edge(r.source, r.target, r.strength)
    return grn


def _build_virtual_cell_config(program: Program) -> VirtualCellConfig:
    sim = program.config.sim
    kwargs: dict[str, Any] = {}
    for key in _VC_FLOAT_FIELDS:
        if key in sim:
            kwargs[key] = _coerce_float(key, sim[key])
    for key, target in _VC_BOOL_FIELDS:
        if key in sim:
            kwargs[target] = _coerce_bool(key, sim[key])
    for key, allowed in _VC_ENUM_FIELDS:
        if key in sim:
            kwargs[key] = _coerce_enum(key, sim[key], allowed)
    if "seed" in sim:
        kwargs["seed"] = _opt_int_or_none(sim, "seed", None)
    # #media -> FBA uptake bounds (concentration == exchange upper bound)
    kwargs["uptake"] = {m.nutrient: m.concentration for m in program.media}
    # chromosome_map: #config sim key + per-gene chromosome= passthrough
    cmap = _opt_float_dict(sim, "chromosome_map", {})
    for g in program.genes:
        if "chromosome" in g.fields:
            cmap[g.name] = _coerce_float(
                f"#gene {g.name} chromosome", g.fields["chromosome"])
    kwargs["chromosome_map"] = cmap
    # k_fold -> fold_rate_per_min (derived via the equilibrium folding
    # fraction, wiring.md §6.2)
    if "k_fold" in sim:
        misfold = kwargs.get("misfold_rate_per_min", 0.3)
        kwargs["fold_rate_per_min"] = _fold_rate_from_k_fold(
            _coerce_float("k_fold", sim["k_fold"]), float(misfold))
    return VirtualCellConfig(**kwargs)


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


# ============================================================================
# fba backend (FluxBalanceAnalysis / DynamicFluxBalance)
# ============================================================================
_FBA_DEFAULT_COLUMNS = [
    "BIOMASS", "EX_glc", "EX_ac", "EX_lac", "EX_co2",
    "growth_rate_per_hour",
]

_DFBA_DEFAULT_COLUMNS = [
    "time_h", "biomass", "glucose", "co2", "growth_rate",
]


def _load_fba_model(sim: dict[str, str]) -> MetabolicModel:
    value = sim.get("fba_model", "core")
    if value == "core":
        return ECOLI_CORE_MODEL
    return load_model(value)


def _enzyme_capacity(program: Program,
                     sim: dict[str, str]) -> EnzymeCapacity:
    """#enzyme bindings (or the canonical tables as fallback, §6.5)."""
    scale = _opt_float(sim, "enzyme_scale", DEFAULT_ENZYME_SCALE)
    mass = _opt_float_or_none(sim, "protein_mass_fraction", None)
    if program.enzymes:
        gene_to_reactions: dict[str, list[str]] = {}
        kcat: dict[str, float] = {}
        for e in program.enzymes:
            gene_to_reactions.setdefault(e.gene, []).append(e.reaction)
            if e.kcat is not None:
                kcat[e.reaction] = e.kcat
            else:
                kcat.setdefault(e.reaction,
                                ECOLI_CORE_KCAT.get(e.reaction, 1.0))
        return EnzymeCapacity(
            {g: tuple(rs) for g, rs in gene_to_reactions.items()},
            kcat=kcat, enzyme_scale=scale, protein_mass_fraction=mass)
    return EnzymeCapacity(
        dict(ECOLI_CORE_GENE_REACTIONS), kcat=dict(ECOLI_CORE_KCAT),
        enzyme_scale=scale, protein_mass_fraction=mass)


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
    cfg = DynamicFBAConfig(
        dt_h=_opt_float(sim, "fba_dt_h", 0.25),
        initial_biomass_gdw=_opt_float(sim, "fba_initial_biomass_gdw", 0.05),
        initial_glucose_mm=_opt_float(sim, "fba_glucose_mm",
                                      glc_media if glc_media is not None
                                      else 10.0),
        max_glucose_uptake=_opt_float(sim, "fba_max_glucose_uptake",
                                      DEFAULT_GLC_UPTAKE),
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


# ============================================================================
# Long-tail extension point (wiring.md §8.6): #sim kind=spatial_dfba
# ============================================================================
_SIM_DFBA_DEFAULT_COLUMNS = [
    "tick", "time_h", "total_glucose", "total_biomass", "total_acetate",
    "depletion_front", "co2_overflow",
]


def _run_spatial_dfba(program: Program) -> FluxResult:
    """``#sim kind=spatial_dfba`` — 1-D dFBA strip (examples/24).

    Every ``SpatialDFBAConfig`` field maps 1:1 to an ``#sim`` key; the
    ``steps`` / ``output`` keys control the batch length and columns.
    """
    ext = program.sim_extensions
    config = SpatialDFBAConfig(
        length=_opt_int(ext, "length", 32),
        glucose_diffusion_um2_s=_opt_float(ext, "glucose_diffusion_um2_s", 2.0),
        initial_glucose_mm=_opt_float(ext, "initial_glucose_mm", 5.0),
        inlet_glucose_mm=_opt_float_or_none(ext, "inlet_glucose_mm", 5.0),
        initial_biomass_gdw=_opt_float(ext, "initial_biomass_gdw", 0.05),
        max_biomass_gdw=_opt_float_or_none(ext, "max_biomass_gdw", None),
        dt_h=_opt_float(ext, "dt_h", 0.25),
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


# ============================================================================
# W-6 long-tail backends (wiring.md §19): one #sim kind= per Python app.
# Each maps its keys 1:1 onto the underlying app's config dataclass and
# emits the app's results through the standard SimResult/column machinery.
# ============================================================================

def _gene_orfs(program: Program) -> list[tuple[str, str]]:
    """``(gene_name, coding_dna)`` for every ``#gene`` ORF."""
    return [(g.name, "".join(c.seq for c in g.codons)) for g in program.genes]


def _first_gene_protein(program: Program) -> str:
    """Protein translated from the first ``#gene`` ORF ('' when absent)."""
    for _name, dna in _gene_orfs(program):
        protein = translate_dna(dna)
        stop = protein.find("*")
        if stop != -1:
            protein = protein[:stop]
        if protein:
            return protein
    return ""


# -- kind=consortium (apps/consortium.py) -----------------------------------

_CONSORTIUM_DEFAULT_COLUMNS = [
    "tick", "alive", "consensus_fraction", "consensus_reached",
    "output_rate", "cumulative_output", "active_actuators", "avg_signal",
    "producer_fraction", "sensor_fraction", "actuator_fraction",
]


def _run_consortium(program: Program) -> ColonyResult:
    """``#sim kind=consortium`` — quorum consensus + composition control."""
    ext = program.sim_extensions
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


# -- kind=digital_evolution (apps/digital_evolution.py) ---------------------

_DIGITAL_EVO_DEFAULT_COLUMNS = ["generation", "mean_fitness", "max_fitness"]


def _parse_bit_sequence(raw: str) -> tuple[int, ...]:
    """``"1,0,1,0"`` or ``"1010"`` -> ``(1, 0, 1, 0)``."""
    cleaned = raw.strip().replace(",", "").replace(" ", "")
    if not cleaned or any(ch not in "01" for ch in cleaned):
        raise SimConfigError(
            f"sim key 'target': expected a bit sequence like '1,0,1,0' "
            f"or '1010', got {raw!r}")
    return tuple(int(ch) for ch in cleaned)


def _run_digital_evolution(program: Program) -> ScoreResult:
    """``#sim kind=digital_evolution`` — Avida-style instruction genomes."""
    ext = program.sim_extensions
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


# -- kind=stochastic (stochastic.py) ------------------------------------------

_STOCHASTIC_FANO_COLUMNS = [
    "mode", "fano", "on_fraction", "transcription_rate",
]
_STOCHASTIC_GILLESPIE_COLUMNS = [
    "mode", "mean", "variance", "fano", "analytic_fano",
]


def _run_stochastic(program: Program) -> ScoreResult:
    """``#sim kind=stochastic`` — telegraph-promoter Fano / Gillespie SSA."""
    ext = program.sim_extensions
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


# -- kind=codec_benchmark (apps/dna_storage.py) -------------------------------

_CODEC_DEFAULT_COLUMNS = [
    "scheme", "target_density", "achieved_density", "redundancy",
    "max_loss_fraction", "max_error_rate", "decode_time_s",
    "num_oligos", "total_bp", "cost_per_gb_usd",
]


def _run_codec_benchmark(program: Program) -> ScoreResult:
    """``#sim kind=codec_benchmark`` — DNA-storage codec robustness scan."""
    ext = program.sim_extensions
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


# -- kind=synbio_design (apps/synbio_designer.py) ------------------------------

_SYNBIO_DEFAULT_COLUMNS = [
    "protein", "orf_length", "full_length", "cai", "gc_content",
    "n_restriction_sites", "valid",
]


def _run_synbio_design(program: Program) -> ScoreResult:
    """``#sim kind=synbio_design`` — cassette auto-design (promoter+RBS+ORF+term)."""
    ext = program.sim_extensions
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


# -- kind=protein_fitness (protein_fitness.py) ---------------------------------

_PROTEIN_FITNESS_COLUMNS = ["rank", "variant", "score"]


def _run_protein_fitness(program: Program) -> ScoreResult:
    """``#sim kind=protein_fitness`` — oracle ranking of protein variants."""
    ext = program.sim_extensions
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


# -- kind=morphogen_gradient (apps/morphogen_gradient.py) ----------------------

_MORPHOGEN_DEFAULT_COLUMNS = [
    "tick", "source_um", "mid_um", "far_um", "monotone", "n_domains",
]


def _opt_morphogen_genes(ext: dict[str, str]) -> tuple[MorphogenGene, ...]:
    """``genes=near:12.0,mid:6.0,far:2.0`` -> MorphogenGene tuple."""
    raw = ext.get("genes")
    if not raw:
        return MorphogenGradientConfig().genes
    out: list[MorphogenGene] = []
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, threshold = pair.partition(":")
        if not threshold:
            raise SimConfigError(
                f"sim key 'genes': expected name:threshold pairs, got {raw!r}")
        out.append(MorphogenGene(
            name.strip(),
            _coerce_float("genes", threshold.strip())))
    return tuple(out)


def _opt_morphogen_repression(ext: dict[str, str],
                              ) -> tuple[tuple[str, str], ...]:
    """``repression=near,mid;near,far`` -> repression edge tuple."""
    raw = ext.get("repression")
    if not raw:
        return MorphogenGradientConfig().repression
    out: list[tuple[str, str]] = []
    for edge in raw.split(";"):
        edge = edge.strip()
        if not edge:
            continue
        a, _, b = edge.partition(",")
        if not b:
            raise SimConfigError(
                f"sim key 'repression': expected 'a,b;c,d' edges, got {raw!r}")
        out.append((a.strip(), b.strip()))
    return tuple(out)


def _run_morphogen_gradient(program: Program) -> HistoryResult:
    """``#sim kind=morphogen_gradient`` — 1-D French-flag patterning strip."""
    ext = program.sim_extensions
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


# -- kind=protein_structure (protein_structure.py) ------------------------------

_PROTEIN_STRUCTURE_COLUMNS = [
    "length", "helix_fraction", "sheet_fraction", "turn_fraction",
    "coil_fraction", "mean_hydropathy", "gravy",
    "n_transmembrane_helices", "is_membrane_protein",
    "disorder_fraction", "n_disorder_regions",
]


def _run_protein_structure(program: Program) -> ScoreResult:
    """``#sim kind=protein_structure`` — predict_structure report summary."""
    ext = program.sim_extensions
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


# -- kind=fate_analysis (apps/fate_analysis.py) ---------------------------------

_FATE_DEFAULT_COLUMNS = ["w", "n_stable_states", "fate_a_level", "fate_b_level"]


def _run_fate_analysis(program: Program) -> ScoreResult:
    """``#sim kind=fate_analysis`` — toggle bistability / switching / slowing."""
    ext = program.sim_extensions
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


# -- kind=directed_evolution (apps/protein_evolution.py) ------------------------

_DIRECTED_EVO_COLUMNS = [
    "oracle", "initial_fitness", "guided_recovery", "guided_gain",
    "baseline_gain", "spearman_rho", "final_best_sequence",
]


def _run_directed_evolution(program: Program) -> ScoreResult:
    """``#sim kind=directed_evolution`` — oracle-guided GB1 campaign."""
    ext = program.sim_extensions
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


# -- kind=3d_morphology (morphology_3d.py) --------------------------------------

_3D_MORPHOLOGY_COLUMNS = [
    "preset", "iterations", "n_lines", "n_vertices",
    "span_x", "span_y", "span_z",
]


def _parse_lsystem_rules(raw: str) -> dict[str, str]:
    """``"X=F+[[X]-X];F=FF"`` -> ``{"X": "...", "F": "FF"}``."""
    rules: dict[str, str] = {}
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        lhs, _, rhs = pair.partition("=")
        if not rhs:
            raise SimConfigError(
                f"sim key 'rules': expected 'A=...;B=...' productions, "
                f"got {raw!r}")
        rules[lhs.strip()] = rhs.strip()
    if not rules:
        raise SimConfigError(
            f"sim key 'rules': no productions parsed from {raw!r}")
    return rules


def _run_3d_morphology(program: Program) -> ScoreResult:
    """``#sim kind=3d_morphology`` — LSystem3D preset mesh statistics."""
    ext = program.sim_extensions
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


# -- kind=omics_calibration (apps/omics_calibration.py) --------------------------

_OMICS_CALIBRATION_COLUMNS = [
    "response_correlation", "de_sign_agreement",
    "mae_improvement_vs_baseline", "n_holdout", "passed",
]


def _run_omics_calibration(program: Program) -> ScoreResult:
    """``#sim kind=omics_calibration`` — calibrate-then-predict perturb-seq."""
    ext = program.sim_extensions
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


# -- kind=cello_workflow (apps/synbio_automation.py) -----------------------------

_CELLO_COLUMNS = [
    "function", "matches_target", "gate_count", "plasmid_length",
    "sbol3_component_count", "time_curve_count",
]


def _cello_truth_table(function: str) -> TruthTable:
    if function == "not":
        return TruthTable.from_function(
            ["a"], ["y"], lambda v: (not v[0],))
    if function == "nand":
        return TruthTable.from_function(
            ["a", "b"], ["y"], lambda v: (not (v[0] and v[1]),))
    return TruthTable.from_function(
        ["a", "b"], ["y"], lambda v: (v[0] != v[1],))


def _run_cello_workflow(program: Program) -> ScoreResult:
    """``#sim kind=cello_workflow`` — truth table -> DNA + SBOL3 closed loop."""
    ext = program.sim_extensions
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


# -- kind=codon_usage (bio_data.py) ---------------------------------------------

_CODON_USAGE_COLUMNS = ["gene", "species", "cai", "orf_length", "protein"]


def _run_codon_usage(program: Program) -> ScoreResult:
    """``#sim kind=codon_usage`` — per-gene CAI across species tables."""
    ext = program.sim_extensions
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


_SIM_BACKENDS: dict[str, Callable[[Program], SimResult]] = {
    "3d_morphology": _run_3d_morphology,
    "codec_benchmark": _run_codec_benchmark,
    "codon_usage": _run_codon_usage,
    "cello_workflow": _run_cello_workflow,
    "consortium": _run_consortium,
    "digital_evolution": _run_digital_evolution,
    "directed_evolution": _run_directed_evolution,
    "fate_analysis": _run_fate_analysis,
    "morphogen_gradient": _run_morphogen_gradient,
    "omics_calibration": _run_omics_calibration,
    "protein_fitness": _run_protein_fitness,
    "protein_structure": _run_protein_structure,
    "spatial_dfba": _run_spatial_dfba,
    "stochastic": _run_stochastic,
    "synbio_design": _run_synbio_design,
}


# ============================================================================
# population backend (CellPopulation3D, Phase 5)
# ============================================================================
_POP_MECHANICS = {
    "none": None,
    "shove": "shoving",
    "shoving": "shoving",
    "force": "force",
}

_POP_DEFAULT_COLUMNS = [
    "generation", "alive_count", "avg_energy", "diversity_index",
    "division_rate", "death_rate",
]


def _environment(program: Program, width: int, height: int) -> Environment:
    env = Environment(EnvironmentConfig(width=width, height=height))
    for m in program.media:
        if m.nutrient in ("GLC", "glucose"):
            env.glucose.set_all(m.concentration)
            if m.diffusion_um2_s is not None:
                env.glucose.diffusion_um2_s = m.diffusion_um2_s
        elif m.nutrient in ("O2", "oxygen"):
            env.oxygen.set_all(m.concentration)
            if m.diffusion_um2_s is not None:
                env.oxygen.diffusion_um2_s = m.diffusion_um2_s
        else:
            env.add_field(m.nutrient, ConcentrationField(
                m.nutrient, width, height,
                m.diffusion_um2_s if m.diffusion_um2_s is not None else 300.0,
                m.concentration))
    return env


def _build_population_config(program: Program) -> PopulationConfig:
    sim = program.config.sim
    mechanics_raw = sim.get("mechanics", "none")
    mechanics = _POP_MECHANICS.get(mechanics_raw)
    if mechanics is None and mechanics_raw != "none":
        raise SimConfigError(
            f"sim key 'mechanics': expected one of "
            f"{sorted(_POP_MECHANICS)}, got {mechanics_raw!r}")
    config = PopulationConfig(
        max_size=_opt_int(sim, "population_size", 1000),
        grid_width=_opt_int(sim, "grid_width", 32),
        grid_height=_opt_int(sim, "grid_height", 32),
        grid_depth=_opt_int(sim, "grid_depth", 1),
        division_threshold=_opt_float(sim, "division_threshold", 1.8e9),
        death_threshold=_opt_float(sim, "death_threshold", 0.0),
        signaling_enabled=_opt_bool(sim, "signaling", True),
        signal_diffusion=_opt_float(sim, "signal_diffusion", 0.4),
        signal_threshold=_opt_float(sim, "signal_threshold", 5.0),
        crowding_enabled=_opt_bool(sim, "crowding", False),
        mechanics=mechanics,
        noise_enabled=_opt_bool(sim, "noise_enabled", False),
        noise_seed=_opt_int_or_none(sim, "noise_seed", None),
        trace_streaming=_opt_bool(sim, "trace_streaming", False),
        dfba_enabled=_opt_bool(sim, "dfba", False),
        dfba_dt_h=_opt_float(sim, "dfba_dt_h", 0.25),
        dfba_energy_scale=_opt_float(sim, "dfba_energy_scale", 1.25e8),
        dfba_initial_biomass_gdw=_opt_float(sim, "dfba_initial_biomass_gdw",
                                            0.05),
        dfba_glucose_half_saturation_mm=_opt_float(
            sim, "dfba_glucose_half_saturation_mm", 0.1),
        dfba_oxygen_max_uptake=_opt_float(sim, "dfba_oxygen_max_uptake", 40.0),
        dfba_oxygen_half_saturation_mm=_opt_float(
            sim, "dfba_oxygen_half_saturation_mm", 0.01),
        program=program,
        chunk=Compiler(STANDARD_TABLE).compile(program),
        ops_per_tick=program.config.ops_per_tick,
    )
    config.environment = _environment(
        program, config.grid_width, config.grid_height)
    return config


def _seed_cells(config: PopulationConfig, n: int) -> list[PopulationCell]:
    """Pack ``n`` cells as a centered colony block (like the apps do)."""
    w, h = config.grid_width, config.grid_height
    side = 1
    while side * side < n:
        side += 1
    off_x = max(0, (w - side) // 2)
    off_y = max(0, (h - side) // 2)
    cells: list[PopulationCell] = []
    for i in range(n):
        cells.append(PopulationCell(
            id=i,
            x=min(off_x + i % side, w - 1),
            y=min(off_y + (i // side) % side, h - 1),
        ))
    return cells


def _run_population(program: Program) -> ColonyResult:
    config = _build_population_config(program)
    if config.dfba_enabled and not program.media:
        raise SimConfigError(
            "backend=population with dfba=true requires at least one "
            "#media declaration (shared substrate fields)")
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
    if config.trace_streaming:
        meta["trace"] = population.trace
    return ColonyResult(
        backend="population", columns=columns,
        rows=[_project(row, columns) for row in rows],
        meta=meta,
    )


# ============================================================================
# calibration / benchmark backends (apps)
# ============================================================================
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
