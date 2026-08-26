"""Simulation-library adapter (``doc/12-helix-language-wiring.md`` §8).

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
  omics_calibration, population_calibration, cello_workflow, codon_usage).

``backend=classic`` (the default) returns ``None`` so the CLI keeps the
existing compile -> CellVM path.  A registered ``#sim kind=...`` overrides
even the classic default.  The classic pipeline never reads
``Program.config.sim`` and never sees the structural annotations, so this
module is purely additive.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from helixlang.human.disease import DiseaseState
    from helixlang.human.drug import Drug
    from helixlang.human.genotype import GenotypeProfile
    from helixlang.human.pharmacodynamics import Pharmacodynamics
    from helixlang.human.phenotype import ExternalTraits

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
from helixlang.apps.ecosystem import (
    Ecosystem,
    EcosystemConfig,
    PatchConfig,
    ScalarConfig,
    Species,
    SubstrateConfig,
)
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
from helixlang.apps.population_calibration import (
    run_population_calibration,
)
from helixlang.apps.population_dbtl import (
    DbtlConfig,
    PopulationDbtl,
)
from helixlang.apps.protein_evolution import guided_directed_evolution
from helixlang.apps.spatial_dfba import SpatialDFBA, SpatialDFBAConfig
from helixlang.apps.spatial_evolution import (
    SpatialEvolution,
    SpatialEvolutionConfig,
)
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
    DiurnalForcing,
    Environment,
    EnvironmentConfig,
    SeasonalForcing,
)
from helixlang.errors import SimConfigError
from helixlang.flow import FlowField, channel_poiseuille, stagnant
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
from helixlang.units import LATTICE_SPACING_UM
from helixlang.virtual_cell import RepliconSpec, VirtualCell, VirtualCellConfig

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
    "gem", "ecosystem",
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
    ``provenance`` records reproducibility metadata (seed, backend,
    source hash, dependency versions, runtime).
    """
    backend: str
    columns: list[str]
    rows: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

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
# Type coercion (12-helix-language-wiring.md §6.3)
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


def _opt_replicon_specs(sim: dict[str, str], key: str,
                        ) -> dict[str, RepliconSpec]:
    """Coerce ``"pBR322:20,pUC19:500"`` -> replicon specs (Phase-C C2).

    Replicons declared here are plasmids (constant base copy number);
    the chromosome replicon is implicit and fork-driven.
    """
    if key not in sim:
        return {}
    out: dict[str, RepliconSpec] = {}
    for pair in sim[key].split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise SimConfigError(
                f"sim key {key!r}: expected 'name:copy' pairs, got {pair!r}")
        name, _, copy_s = pair.partition(":")
        name = name.strip()
        copy = _coerce_int(f"{key}.{name}", copy_s.strip())
        if copy < 1:
            raise SimConfigError(
                f"sim key {key!r}: copy number must be >= 1, got {copy}")
        out[name] = RepliconSpec(kind="plasmid", copy_number=copy)
    return out


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
    result: SimResult | None = None
    if kind in _SIM_BACKENDS:
        result = _SIM_BACKENDS[kind](program)
    elif name == "classic":
        return None
    elif name == "whole_cell":
        result = _run_whole_cell(program)
    elif name == "population":
        result = _run_population(program)
    elif name == "fba":
        result = _run_fba(program)
    elif name == "calibration":
        result = _run_calibration(program)
    elif name == "benchmark":
        result = _run_benchmark(program)
    elif name == "gem":
        result = _run_gem(program)
    elif name == "ecosystem":
        result = _run_ecosystem(program)
    else:
        raise SimConfigError(
            f"unknown backend {name!r}; expected one of "
            "classic, whole_cell, population, fba, calibration, benchmark")
    if result is not None and not result.provenance:
        from helixlang.provenance import build_provenance
        seed = getattr(program.config, "seed", None)
        result.provenance = build_provenance(
            seed=seed,
            backend=name,
            source=getattr(program, "_source_text", None),
        )
    return result


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
    (12-helix-language-wiring.md §7.2).
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
    # replicons (Phase-C C2): #config sim replicons=name:copy,... plus
    # per-gene replicon= assignment onto those replicons
    kwargs["replicons"] = _opt_replicon_specs(sim, "replicons")
    gene_replicons: dict[str, str] = {}
    for g in program.genes:
        if "replicon" in g.fields:
            gene_replicons[g.name] = str(g.fields["replicon"]).strip()
    kwargs["gene_replicons"] = gene_replicons
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
        km: dict[str, float] = {}
        for e in program.enzymes:
            gene_to_reactions.setdefault(e.gene, []).append(e.reaction)
            if e.kcat is not None:
                kcat[e.reaction] = e.kcat
            else:
                kcat.setdefault(e.reaction,
                                ECOLI_CORE_KCAT.get(e.reaction, 1.0))
            if e.km is not None:
                km[e.reaction] = e.km
        return EnzymeCapacity(
            {g: tuple(rs) for g, rs in gene_to_reactions.items()},
            kcat=kcat, km=km, enzyme_scale=scale,
            protein_mass_fraction=mass)
    return EnzymeCapacity(
        dict(ECOLI_CORE_GENE_REACTIONS), kcat=dict(ECOLI_CORE_KCAT),
        enzyme_scale=scale, protein_mass_fraction=mass)


def _build_model_from_reactions(program: Program) -> MetabolicModel | None:
    """Build a MetabolicModel from DSL #reaction declarations.

    Returns a MetabolicModel if at least one #reaction block is present,
    otherwise None (caller should fall back to a loaded model).
    """
    if not program.reactions:
        return None
    from helixlang.metabolism import MetabolicModel
    from helixlang.metabolism import Reaction as Rxn
    model = MetabolicModel()
    for decl in program.reactions:
        stoich: dict[str, float] = {}
        if decl.substrate:
            stoich[decl.substrate] = decl.substrate_coeff
        if decl.product:
            stoich[decl.product] = decl.product_coeff
        rxn = Rxn(
            id=decl.id,
            name=decl.name,
            stoichiometry=stoich,
            lower_bound=decl.lower_bound,
            upper_bound=decl.upper_bound,
            subsystem=decl.subsystem,
        )
        model.add_reaction(rxn)
    return model


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


# -- kind=spatial_evolution (apps/spatial_evolution.py) -----------------------

_SPATIAL_EVO_DEFAULT_COLUMNS = [
    "generation", "mean_fitness", "max_fitness", "best_radius_sites",
    "best_survival", "mean_uptake_gain",
]


def _run_spatial_evolution(program: Program) -> ScoreResult:
    """``#sim kind=spatial_evolution`` — dual-loop range-expansion evolution
    (doc/18-programmable-cell-population-simulation.md §13 Design 1; Bosshard et al. 2020 BMC Genomics 21:232)."""
    ext = program.sim_extensions
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


def _run_population_calibration(program: Program) -> ScoreResult:
    """``#sim kind=population_calibration``: recover the dFBA colony
    parameters from colony-level mixed observables (doc/18-programmable-cell-population-simulation.md §13 Design 4)."""
    ext = program.sim_extensions
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


# -- kind=ecosystem (apps/ecosystem.py, doc/19 Phases A-D) ---------------------

_ECOSYSTEM_DEFAULT_COLUMNS = [
    "tick", "generation",
    "water:producer", "water:consumer", "water:oxygen", "water:co2",
]


def _group_prefixed(ext: dict[str, str], prefix: str) -> dict[str, dict[str, str]]:
    """Group ``<prefix><name>.<attr>`` extension keys into per-name tables."""
    groups: dict[str, dict[str, str]] = {}
    for k, v in ext.items():
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix):]
        name, _, attr = rest.partition(".")
        if not name or not attr:
            continue
        groups.setdefault(name, {})[attr] = v
    return groups


def _split_pair(value: str, key: str) -> tuple[str, float]:
    """``"glucose:0.02"`` -> ``("glucose", 0.02)``; else raises."""
    if ":" not in value:
        raise SimConfigError(
            f"sim key {key!r}: expected '<name>:<number>', got {value!r}")
    name, _, num = value.partition(":")
    if not name:
        raise SimConfigError(
            f"sim key {key!r}: empty name in {value!r}")
    return name, _coerce_float(f"{key}.{name}", num)


def _build_ecosystem_species(ext: dict[str, str]) -> list[Species]:
    """Build the ``#species`` table from ``species.<name>.<field>`` keys."""
    groups = _group_prefixed(ext, "species.")
    out: list[Species] = []
    for name, attrs in sorted(groups.items()):
        sp = Species(name=name)
        if "genome" in attrs:
            sp.genome = attrs["genome"]
        if "photo" in attrs:
            sp.photo = _coerce_bool(f"species.{name}.photo", attrs["photo"])
        if "photo_vmax" in attrs:
            sp.photo_vmax = _coerce_float(
                f"species.{name}.photo_vmax", attrs["photo_vmax"])
        if "cn_ratio" in attrs:
            sp.cn_ratio = _coerce_float(
                f"species.{name}.cn_ratio", attrs["cn_ratio"])
        if "maintenance" in attrs:
            sp.maintenance = _coerce_float(
                f"species.{name}.maintenance", attrs["maintenance"])
        # dotted consumption.<sub>.vmax / .ks
        for k, v in attrs.items():
            if not k.startswith("consumption."):
                continue
            sub = k[len("consumption."):]
            subname, _, subattr = sub.partition(".")
            if not subname:
                continue
            cur = sp.consumption.get(subname, (0.0, 0.1))
            if subattr == "vmax":
                sp.consumption[subname] = (
                    _coerce_float(f"species.{name}.{k}", v), cur[1])
            elif subattr == "ks":
                sp.consumption[subname] = (
                    cur[0], _coerce_float(f"species.{name}.{k}", v))
        # flat substrate/vmax/ks (+ second-substrate forms)
        for suffix, subkey in (("", "substrate"), ("2", "substrate2")):
            if subkey not in attrs:
                continue
            vmax_key = f"vmax{suffix}"
            ks_key = f"ks{suffix}"
            sub = attrs[subkey]
            vmax = (_coerce_float(f"species.{name}.{vmax_key}",
                                  attrs[vmax_key])
                    if vmax_key in attrs else 0.02)
            ks = (_coerce_float(f"species.{name}.{ks_key}",
                                attrs[ks_key])
                  if ks_key in attrs else 0.1)
            sp.consumption[sub] = (vmax, ks)
        # secretion=sub:rate / diet=prey:eff / attack=prey:rate and their
        # dotted ``<table>.<name>`` forms
        for k, v in attrs.items():
            if k == "secretion":
                sub, rate = _split_pair(v, f"species.{name}.secretion")
                sp.secretion[sub] = rate
            elif k == "diet":
                prey, eff = _split_pair(v, f"species.{name}.diet")
                sp.diet[prey] = eff
            elif k == "attack":
                prey, rate = _split_pair(v, f"species.{name}.attack")
                sp.attack_rate[prey] = rate
            elif k.startswith("secretion."):
                sp.secretion[k[len("secretion."):]] = _coerce_float(
                    f"species.{name}.{k}", v)
            elif k.startswith("diet."):
                sp.diet[k[len("diet."):]] = _coerce_float(
                    f"species.{name}.{k}", v)
            elif k.startswith("attack."):
                sp.attack_rate[k[len("attack."):]] = _coerce_float(
                    f"species.{name}.{k}", v)
        out.append(sp)
    return out


def _build_ecosystem_patches(ext: dict[str, str]) -> list[PatchConfig]:
    """Build the ``#patch`` table from ``patch.<name>.<field>`` keys."""
    groups = _group_prefixed(ext, "patch.")
    out: list[PatchConfig] = []
    for name, attrs in sorted(groups.items()):
        pc = PatchConfig(name=name)
        pc.kind = attrs.get("kind", pc.kind)
        pc.width = _opt_int(attrs, "width", pc.width)
        pc.height = _opt_int(attrs, "height", pc.height)
        pc.carrying_capacity = _opt_float(
            attrs, "carrying_capacity", pc.carrying_capacity)
        pc.anoxic = _opt_bool(attrs, "anoxic", pc.anoxic)
        pc.moisture = _opt_float(attrs, "moisture", pc.moisture)
        pc.clay = _opt_float(attrs, "clay", pc.clay)
        pc.cn_som = _opt_float(attrs, "cn_som", pc.cn_som)
        pc.cn_species = _opt_float(attrs, "cn_species", pc.cn_species)
        pc.initial_nh4_mm = _opt_float(attrs, "initial_nh4_mm", pc.initial_nh4_mm)
        pc.initial_no3_mm = _opt_float(attrs, "initial_no3_mm", pc.initial_no3_mm)
        pc.flow_rate = _opt_float(attrs, "flow_rate", pc.flow_rate)
        pc.temperature_c = _opt_float(attrs, "temperature", pc.temperature_c)
        pc.ph = _opt_float(attrs, "ph", pc.ph)
        pc.fluctuation_period = _opt_int(
            attrs, "fluctuation_period", pc.fluctuation_period)
        pc.fluctuation_amplitude = _opt_float(
            attrs, "fluctuation_amplitude", pc.fluctuation_amplitude)
        for k, v in attrs.items():
            if k.startswith("initial."):
                pc.initial_biomass[k[len("initial."):]] = _coerce_float(
                    f"patch.{name}.{k}", v)
        substrates: dict[str, SubstrateConfig] = {}
        for k, v in attrs.items():
            if not k.startswith("substrate."):
                continue
            sub = k[len("substrate."):]
            subname, _, subattr = sub.partition(".")
            if not subname:
                continue
            sc = substrates.setdefault(subname, SubstrateConfig(initial_mm=0.0))
            if subattr == "initial":
                sc.initial_mm = _coerce_float(f"patch.{name}.{k}", v)
            elif subattr == "bulk":
                sc.bulk_mm = _coerce_float(f"patch.{name}.{k}", v)
            elif subattr == "carbon_per_mol":
                sc.carbon_per_mol = int(_coerce_float(f"patch.{name}.{k}", v))
            elif subattr == "diffusion":
                sc.diffusion_um2_s = _coerce_float(f"patch.{name}.{k}", v)
        pc.substrates.update(substrates)
        scalars: dict[str, ScalarConfig] = {}
        amplitudes: dict[str, float] = {}
        force_defs: dict[str, str] = {}
        for k, v in attrs.items():
            if not k.startswith("scalar."):
                continue
            sub = k[len("scalar."):]
            sname, _, sattr = sub.partition(".")
            if not sname:
                continue
            scl = scalars.setdefault(sname, ScalarConfig())
            if sattr == "kind":
                scl.kind = v
            elif sattr == "initial":
                scl.initial = _coerce_float(f"patch.{name}.{k}", v)
            elif sattr == "amplitude":
                amplitudes[sname] = _coerce_float(f"patch.{name}.{k}", v)
            elif sattr == "forcing":
                force_defs[sname] = v.strip().lower()
        for sname, forcing in force_defs.items():
            if forcing not in ("diurnal", "seasonal"):
                raise SimConfigError(
                    f"patch {name!r} scalar {sname!r} forcing: expected "
                    f"'diurnal' or 'seasonal', got {forcing!r}")
            mean = scalars[sname].initial
            amp = amplitudes.get(sname, mean if sname == "light" else 3.0)
            if forcing == "diurnal":
                scalars[sname].forcing = DiurnalForcing(
                    mean, amp,
                    lo=(0.0 if sname == "light" else None))
            else:
                scalars[sname].forcing = SeasonalForcing(mean, amp)
        pc.scalars.update(scalars)
        for k, v in attrs.items():
            if k.startswith("dispersal."):
                pc.dispersal[k[len("dispersal."):]] = _coerce_float(
                    f"patch.{name}.{k}", v)
        out.append(pc)
    return out


def _attach_gem_to_ecosystem_species(
    species: list[Species],
    ext: dict[str, str],
) -> None:
    """Run GEM pipeline for each species with a genome and attach the
    resulting MetabolicModel + FBA fluxes (doc/21 §3.4).

    For each species where ``species.<name>.genome`` is set, this runs
    ``run_gem_pipeline()`` (bottom-up) or ``build_functional_model_full()``
    (when ``use_full_model=true``), extracts metabolic parameters via
    ``gem_to_species()``, and attaches the model to the species so the
    ecosystem tick loop can use FBA-backed growth.
    """
    import tempfile
    from pathlib import Path

    from helixlang.apps.ecosystem import gem_to_species

    groups = _group_prefixed(ext, "species.")
    for sp in species:
        attrs = groups.get(sp.name, {})
        genome_path = attrs.get("genome", "")
        use_full = attrs.get("use_full_model", "false").lower() in (
            "true", "1", "yes")

        # Full-model path (doc/24): load pre-built BiGG model directly
        # when use_full_model=true and no genome path is given.
        if use_full and not genome_path:
            try:
                from helixlang.gem.bridge import build_functional_model_full
                from helixlang.metabolism import FluxBalanceAnalysis

                medium_name = ext.get("gem_medium", "glucose_minimal")
                # Try the DSL gem_organism first, then sp.name
                gem_org = ext.get("gem_organism", "")
                org_candidates = ([gem_org] if gem_org else []) + [
                    sp.name, sp.name.replace(" ", "_"),
                ]
                model = None
                for org in org_candidates:
                    if not org:
                        continue
                    try:
                        model = build_functional_model_full(
                            organism=org, medium=medium_name)
                        break
                    except Exception:
                        continue
                if model is None:
                    continue
                sp.metabolic_model = model
                fba = FluxBalanceAnalysis(model)
                fluxes = fba.solve()
                sp.gem_fluxes = dict(fluxes) if fluxes else {}
                params = gem_to_species(type("R", (), {
                    "metabolic_model": model,
                    "fba_fluxes": sp.gem_fluxes,
                    "consensus": None,
                })(), organism=org)
                vmax = float(params.get("vmax", 0))
                if vmax > 0:
                    primary = sp.preferred_substrate() or "glucose"
                    ks = float(params.get("ks", 0.1))
                    sp.consumption[primary] = (vmax, ks)
                # Add oxygen as consumed substrate for FBA exchange
                # bounds so the FBA solver caps growth by O2 availability.
                if "oxygen" not in sp.consumption and not sp.photo:
                    sp.consumption["oxygen"] = (0.5, 0.01)
                max_mu = float(params.get("max_growth_rate", 0.87))
                mgr_str = attrs.get("max_growth_rate", "")
                if mgr_str:
                    try:
                        max_mu = float(mgr_str)
                    except ValueError:
                        pass
                sp.traits.max_growth_rate = max_mu
            except Exception:
                pass
            continue

        if not genome_path:
            continue

        # Handle inline DNA (from #species DNA block → stored in genome field)
        _tmp_fasta: Path | None = None
        if not Path(genome_path).exists() and len(genome_path) > 10:
            # Looks like raw sequence, write to temp FASTA
            _tmp_fasta = Path(tempfile.mktemp(suffix=".fasta"))
            _tmp_fasta.write_text(f">{sp.name}\n{genome_path}\n")
            genome_path = str(_tmp_fasta)

        try:
            from helixlang.apps.gem_pipeline import run_gem_pipeline
            result = run_gem_pipeline(
                genome_fasta=genome_path,
                organism=sp.name,
                target_organism=sp.name,
            )
            if result.metabolic_model is not None:
                sp.metabolic_model = result.metabolic_model
                sp.gem_fluxes = dict(result.fba_fluxes)
                # Extract parameters from GEM
                params = gem_to_species(result, organism=sp.name)
                # Update species consumption from GEM vmax/ks
                vmax = float(params.get("vmax", 0))
                if vmax > 0:
                    # Map primary substrate
                    primary = sp.preferred_substrate() or "glucose"
                    ks = float(params.get("ks", 0.1))
                    sp.consumption[primary] = (vmax, ks)
                yield_c = float(params.get("yield_c", 0))
                if yield_c > 0:
                    sp.traits.yield_c = yield_c
                secretion = params.get("secretion")
                if isinstance(secretion, dict):
                    sp.secretion.update(secretion)
                cn_ratio = float(params.get("cn_ratio", 6.0))
                if cn_ratio != 6.0:
                    sp.cn_ratio = cn_ratio
                maintenance = float(params.get("maintenance", 0.001))
                if maintenance != 0.001:
                    sp.maintenance = maintenance
                max_mu = float(params.get("max_growth_rate", 0.87))
                mgr_str = attrs.get("max_growth_rate", "")
                if mgr_str:
                    try:
                        max_mu = float(mgr_str)
                    except ValueError:
                        pass
                sp.traits.max_growth_rate = max_mu
        except Exception:
            pass  # non-fatal: fall back to Monod if GEM fails
        finally:
            if _tmp_fasta is not None:
                _tmp_fasta.unlink(missing_ok=True)


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
    the ``#species`` table and the ``#patch`` table from ``sim_extensions``.
    """
    ext = program.sim_extensions
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


# -- kind=population_dbtl (apps/population_dbtl.py, doc/19 §5.6 D3) -----------

def _run_population_dbtl(program: Program) -> ScoreResult:
    """``#sim kind=population_dbtl`` — Design→Build→Test→Learn loop over a
    population of strains (synbio_designer DNA → Ecosystem → fit_parameters).

    Reads ``#sim`` keys ``n_rounds``/``population_size``/``genome_length``/
    ``evaluation_ticks``/``seed``/``substrate``/``mutation_rate``/
    ``bias_fraction``/``n_candidates``; the gate is a designed strain whose
    growth strictly improves across the loop.
    """
    ext = program.sim_extensions
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


# ---------------------------------------------------------------------------
# _run_gem_full_model — full genome-scale model path (doc/24)
# ---------------------------------------------------------------------------

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
    from helixlang.gem.bridge import build_functional_model_full
    from helixlang.metabolism import FluxBalanceAnalysis

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
            from helixlang.metabolism import (
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


# ============================================================================
# GEM reconstruction backend (doc/20)
# ============================================================================
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

    from helixlang.apps.gem_pipeline import run_gem_pipeline

    ext = program.sim_extensions

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
                except ValueError:
                    pass

    # doc/25 Phase F: max_growth_rate DSL override
    _max_growth_rate_override: float | None = None
    _mgr_str = ext.get("gem_max_growth_rate", "")
    if _mgr_str:
        try:
            _max_growth_rate_override = float(_mgr_str)
        except ValueError:
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
            from helixlang.gem.biomass import build_biomass_reaction
            from helixlang.gem.bridge import (
                _parse_equation_to_stoich,
                build_enzyme_capacity,
                consensus_to_metabolic_model,
            )
            from helixlang.metabolism import FluxBalanceAnalysis, Reaction

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
                    from helixlang.gem.grn_inference import GRNInferenceResult
                    from helixlang.omics.expression_inference import (
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
                                except ValueError:
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
                    from helixlang.gem.bridge import build_enzyme_capacity
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
                    from helixlang.gem.bridge import apply_regulatory_bounds
                    _gpr_map: dict[str, list[str]] = {}
                    if result.consensus is not None:
                        for rxn_id, genes in getattr(
                                result.consensus, "gene_reaction_rules",
                                {}).items():
                            if isinstance(genes, list):
                                for g in genes:
                                    _gpr_map.setdefault(g, []).append(rxn_id)
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
                from helixlang.metabolism import (
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
                from helixlang.metabolism import (
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


# ---------------------------------------------------------------------------
# Medium presets: metabolite -> uptake rate (mmol/gDW/h)
# ---------------------------------------------------------------------------
_MEDIUM_PRESETS: dict[str, dict[str, float]] = {
    "glucose_minimal": {
        "glc-D_e": 10.0,
        "o2_e": 20.0,
        "nh4_e": 1000.0,
        "pi_e": 1000.0,
        "h2o_e": 1000.0,
        "h_e": 1000.0,
    },
    "m9_minimal": {
        "glc-D_e": 10.0,
        "o2_e": 20.0,
        "nh4_e": 1000.0,
        "pi_e": 1000.0,
        "h2o_e": 1000.0,
        "h_e": 1000.0,
        "so4_e": 1000.0,
    },
    "lb": {
        # LB broth: no glucose; carbon from tryptone/yeast extract peptides
        # Amino acid uptake caps approximated from typical LB composition
        "o2_e": 20.0,
        "nh4_e": 1000.0,
        "pi_e": 1000.0,
        "h2o_e": 1000.0,
        "h_e": 1000.0,
        "so4_e": 1000.0,
        "phe-L_e": 5.0,
        "trp-L_e": 1.0,
        "cys-L_e": 0.5,
        "met-L_e": 1.5,
        "tyr-L_e": 1.0,
        "leu-L_e": 5.0,
        "ile-L_e": 3.0,
        "val-L_e": 4.0,
        "ala-L_e": 5.0,
        "gly_e": 3.0,
        "ser-L_e": 3.0,
        "thr-L_e": 3.0,
        "asp-L_e": 4.0,
        "glu-L_e": 7.0,
        "gln-L_e": 3.0,
        "arg-L_e": 3.0,
        "lys-L_e": 4.0,
        "his-L_e": 1.0,
        "pro-L_e": 3.0,
    },
    "bg11": {
        # BG-11 for cyanobacteria (Rippka et al. 1979);
        # Nitrate as nitrogen source (NaNO3 17.65 mM), not ammonium.
        # Also include ammonium as fallback for models lacking nitrate
        # transport (e.g. some Synechocystis reconstructions).
        # CO2 as sole carbon source (photoautotrophic)
        "co2_e": 1000.0,
        "o2_e": 20.0,
        "no3_e": 1000.0,
        "nh4_e": 1000.0,
        "pi_e": 1000.0,
        "h2o_e": 1000.0,
        "h_e": 1000.0,
        "so4_e": 1000.0,
        "mg2_e": 1000.0,
        "ca2_e": 1000.0,
        "na1_e": 1000.0,
        "k1_e": 1000.0,
        "fe3_e": 0.1,
        "cl_e": 1000.0,
    },
    "photoautotrophic": {
        # Alias for bg11
        "co2_e": 1000.0,
        "o2_e": 20.0,
        "no3_e": 1000.0,
        "nh4_e": 1000.0,
        "pi_e": 1000.0,
        "h2o_e": 1000.0,
        "h_e": 1000.0,
        "so4_e": 1000.0,
    },
}

# Organism-specific growth rate caps (h^-1) for numerical safety
_ORGANISM_MAX_GROWTH_RATE: dict[str, float] = {
    "e_coli_k12": 0.87,
    "e_coli": 0.87,
    "synechocystis": 0.14,
    "synechocystis_pcc6803": 0.14,
    "b_subtilis": 0.58,
    "s_cerevisiae": 0.56,
    "s_aureus": 0.69,
    "m_tuberculosis": 0.03,
    "p_aeruginosa": 0.53,
    "k_pneumoniae": 0.53,
    "l_lactis": 0.57,
    "default": 0.87,
}


def _set_gem_medium(
    fba: Any,
    medium_name: str,
    program: Program | None = None,
    model: Any = None,
    organism: str = "e_coli_k12",
    medium_override: dict[str, float] | None = None,
) -> None:
    """Set exchange uptake bounds on a FluxBalanceAnalysis model.

    If ``medium_name`` is a known preset, use it directly.
    If ``medium_name`` is ``custom``, read ``#media`` annotations.
    Otherwise fall back to ``glucose_minimal``.

    ``medium_override`` (doc/25 Phase E): dict of
    ``metabolite_id -> uptake_rate`` that overrides individual entries
    in the preset (e.g. ``{"fe3_e": 0.5, "co2_e": 500}``).

    Non-essential exchange reactions (amino acids, nucleotides, organic
    acids) are capped at ``_TRACE_IMPORT_UB`` mmol/gDW/h — a small rate
    that prevents the LP from exploiting unlimited import while keeping
    the model feasible.  Medium-specified nutrients are opened at their
    full rate.
    """
    if medium_name in _MEDIUM_PRESETS:
        uptake = dict(_MEDIUM_PRESETS[medium_name])
    elif medium_name == "custom" and program is not None and program.media:
        uptake = {m.nutrient: m.concentration for m in program.media}
    else:
        uptake = dict(_MEDIUM_PRESETS.get(medium_name, _MEDIUM_PRESETS["glucose_minimal"]))

    # doc/25 Phase E: apply medium_override on top of preset
    if medium_override:
        uptake.update(medium_override)

    _TRACE_IMPORT_UB = 0.1  # mmol/gDW/h trace cap for non-essential imports

    # Cap all single-metabolite exchange reactions at trace import rate
    for rid, rxn in fba.model.reactions.items():
        if rxn.subsystem == "exchange" and rid.startswith("EX_"):
            if len(rxn.stoichiometry) == 1:
                met = next(iter(rxn.stoichiometry))
                coef = rxn.stoichiometry[met]
                # Close export direction, cap import at trace level
                if coef < 0:
                    rxn.lower_bound = -_TRACE_IMPORT_UB
                else:
                    rxn.upper_bound = _TRACE_IMPORT_UB

    # Open the metabolites specified in the medium at full rate
    for met, rate in uptake.items():
        for rid, rxn in fba.model.reactions.items():
            if rxn.subsystem == "exchange" and rid.startswith("EX_"):
                if met in rxn.stoichiometry:
                    coef = rxn.stoichiometry[met]
                    if coef < 0:
                        rxn.lower_bound = -abs(rate)
                    else:
                        rxn.upper_bound = abs(rate)
        fba.set_uptake(met, rate)

    # Close PET (photosynthetic electron transport) for non-photoautotrophic
    # media.  PET produces NADPH from light; it is only meaningful when the
    # organism grows photoautotrophically.  Leaving it open in heterotrophic
    # mode gives the FBA solver "free" NADPH, distorting flux distributions.
    _photo_media = {"bg11", "photoautotrophic"}
    if medium_name not in _photo_media:
        # Close Calvin cycle (CBB) and PET reactions — these are only
        # relevant for photoautotrophic growth.  Leaving them open in
        # heterotrophic mode creates artificial CO₂ fixation pathways
        # that distort the FBA flux distribution and reduce biomass.
        for _calvin_id in ("PET", "RBPC", "PRUK", "GAPD2",
                           "FBPASE", "SBPaldo", "SBPase"):
            if _calvin_id in fba.model.reactions:
                fba.model.reactions[_calvin_id].lower_bound = 0.0
                fba.model.reactions[_calvin_id].upper_bound = 0.0


def _add_gem_transport_reactions(model: Any) -> None:
    """Add transport reactions connecting extracellular exchange to internal metabolites.

    The gapfill adds exchange reactions using ``_e`` metabolites, but internal
    reactions use bare names (no compartment suffix).  Transport reactions
    bridge the two compartments so FBA can route flux from uptake into
    central metabolism.

    Also adds internal pools and transport for biomass precursors (amino
    acids, nucleotides, cofactors) that exist only in the extracellular
    compartment after gapfill.
    """
    from helixlang.metabolism import Reaction

    # ── 1. Core metabolite transport (central carbon + inorganic) ──────
    transport_reactions = [
        ("GLCtex", {"glc-D_e": -1.0, "g6p": 1.0}),
        ("NH4t", {"nh4_e": -1.0, "nh4": 1.0}),
        ("NO3t", {"no3_e": -1.0, "no3": 1.0}),
        ("PIt2r", {"pi_e": -1.0, "pi": 1.0}),
        ("H2Ot", {"h2o_e": -1.0, "h2o": 1.0}),
        ("O2t", {"o2_e": -1.0, "o2": 1.0}),
        ("CO2t", {"co2": -1.0, "co2_e": 1.0}),
    ]

    # ── 2. Biomass-precursor transport: _e ↔ internal ────────────────
    # Amino acids, nucleotides, lipids, and cofactors that the biomass
    # reaction consumes.  Each gets an internal metabolite (if missing)
    # and a bidirectional transport reaction connecting the _e pool to
    # the internal pool.  This mirrors biological amino-acid permeases
    # (e.g. Brøndsted & Bhatt 1992).
    _BIOMASS_TRANSPORT: list[tuple[str, str]] = [
        # amino acids  (_e_id, internal_id)
        ("ala-L_e", "ala-L"),
        ("arg-L_e", "arg-L"),
        ("asp-L_e", "asp-L"),
        ("cys-L_e", "cys-L"),
        ("gln-L_e", "gln-L"),
        ("glu-L_e", "glu-L"),
        ("gly_e", "gly"),
        ("his-L_e", "his-L"),
        ("ile-L_e", "ile-L"),
        ("leu-L_e", "leu-L"),
        ("lys-L_e", "lys-L"),
        ("met-L_e", "met-L"),
        ("phe-L_e", "phe-L"),
        ("pro-L_e", "pro-L"),
        ("ser-L_e", "ser-L"),
        ("thr-L_e", "thr-L"),
        ("trp-L_e", "trp-L"),
        ("tyr-L_e", "tyr-L"),
        ("val-L_e", "val-L"),
        # nucleotides (internal pools for RNA / DNA precursors)
        ("ump_e", "ump"),
        ("cmp_e", "cmp"),
        ("gmp_e", "gmp"),
        ("amp_e", "amp"),
        # extracellular pyruvate
        ("pyr_e", "pyr"),
        # extracellular succinate
        ("succ_e", "succ"),
    ]

    # ── 3. Ensure EX_co2_e exchange reaction exists ──────────────────
    if "EX_co2_e" not in model.reactions and "co2_e" in model.metabolites:
        model.add_reaction(Reaction(
            id="EX_co2_e", name="CO2 exchange",
            stoichiometry={"co2_e": -1.0},
            lower_bound=0.0, upper_bound=1000.0,
            subsystem="exchange",
        ))
    elif "EX_co2_e" not in model.reactions and "co2_e" not in model.metabolites:
        model.metabolites.add("co2_e")
        model.add_reaction(Reaction(
            id="EX_co2_e", name="CO2 exchange",
            stoichiometry={"co2_e": -1.0},
            lower_bound=0.0, upper_bound=1000.0,
            subsystem="exchange",
        ))

    # ── 4. Add core transport reactions ──────────────────────────────
    for rxn_id, stoich in transport_reactions:
        met_set = set(stoich.keys()) & model.metabolites
        if len(met_set) >= 2:
            if rxn_id not in model.reactions:
                filtered = {k: v for k, v in stoich.items() if k in model.metabolites}
                if filtered:
                    model.add_reaction(Reaction(
                        id=rxn_id, name=rxn_id,
                        stoichiometry=filtered,
                        lower_bound=-1000.0, upper_bound=1000.0,
                        subsystem="transport",
                    ))

    # ── 5. Add biomass-precursor transport (e → internal) ───────────
    for ext_met, int_met in _BIOMASS_TRANSPORT:
        # Skip if external metabolite doesn't exist in the model
        if ext_met not in model.metabolites:
            continue
        # Register internal metabolite if missing
        if int_met not in model.metabolites:
            model.metabolites.add(int_met)
        # Add bidirectional transport reaction if not already present
        rxn_id = f"{int_met}_tex"
        if rxn_id not in model.reactions:
            model.add_reaction(Reaction(
                id=rxn_id, name=f"{int_met} transport",
                stoichiometry={ext_met: -1.0, int_met: 1.0},
                lower_bound=-1000.0, upper_bound=1000.0,
                subsystem="transport",
            ))

    # ── 6. Energy coupling: ATP synthase + thiamine-folate transport ──
    # The respiratory chain (CYTBD) creates a proton motive force that
    # drives ATP synthase.  In reduced models we collapse this into a
    # single lumped reaction: ADP + Pi → ATP (+ H2O).
    if "ATPs4r" not in model.reactions:
        atp_mets = {"adp", "pi", "atp", "h2o"}
        if atp_mets.issubset(model.metabolites):
            model.add_reaction(Reaction(
                id="ATPs4r", name="ATP synthase (simplified)",
                stoichiometry={"adp": -1.0, "pi": -1.0, "atp": 1.0, "h2o": 1.0},
                lower_bound=0.0, upper_bound=1000.0,
                subsystem="energy",
            ))

    # THF transport (cofactor for one-carbon metabolism)
    if "thf_e" in model.metabolites and "thf" not in model.metabolites:
        model.metabolites.add("thf")
    if "thf_e" in model.metabolites and "thf" in model.metabolites:
        if "thf_tex" not in model.reactions:
            model.add_reaction(Reaction(
                id="thf_tex", name="THF transport",
                stoichiometry={"thf_e": -1.0, "thf": 1.0},
                lower_bound=-1000.0, upper_bound=1000.0,
                subsystem="transport",
            ))


def _add_gem_core_reactions(model: Any) -> None:
    """Add core metabolism, amino acid / nucleotide / cofactor biosynthesis.

    The consensus model from genome analysis typically only provides
    exchange reactions and a few gap-filled reactions.  This function
    injects a minimal but stoichiometrically complete core metabolism
    covering glycolysis, PPP, TCA, ETC, 19 amino acid biosynthesis
    pathways, nucleotide biosynthesis, and cofactor transport.

    All metabolites referenced by the reactions are auto-added to the
    model if missing.  Reactions that already exist are replaced.
    """
    from helixlang.metabolism import Reaction

    core_reactions = [
        # --- Glycolysis (glucose → pyruvate) ---
        ("GLCtex", {"glc-D_e": -1.0, "glc-D": 1.0}),
        ("HEX1", {"glc-D": -1.0, "atp": -1.0, "g6p": 1.0, "adp": 1.0}),
        ("PGI", {"g6p": -1.0, "f6p": 1.0}),
        ("PFK", {"f6p": -1.0, "atp": -1.0, "fbp": 1.0, "adp": 1.0}),
        ("FBA", {"fbp": -1.0, "g3p": 1.0, "dhap": 1.0}),
        ("TPI", {"dhap": -1.0, "g3p": 1.0}),
        ("GAPD", {"g3p": -1.0, "nad": -1.0, "pi": -1.0, "13pg": 1.0, "nadh": 1.0}),
        ("PGK", {"13pg": -1.0, "adp": -1.0, "3pg": 1.0, "atp": 1.0}),
        ("PGM", {"3pg": -1.0, "2pg": 1.0}),
        ("ENO", {"2pg": -1.0, "pep": 1.0, "h2o": -1.0}),
        ("PK", {"pep": -1.0, "adp": -1.0, "pyr": 1.0, "atp": 1.0}),
        # --- Pyruvate dehydrogenase (link to TCA) ---
        ("PDH", {"pyr": -1.0, "coa": -1.0, "nad": -1.0, "accoa": 1.0, "co2": 1.0, "nadh": 1.0}),
        # --- TCA cycle ---
        ("CS", {"accoa": -1.0, "oaa": -1.0, "h2o": -1.0, "cit": 1.0, "coa": 1.0}),
        ("ACONa", {"cit": -1.0, "acon-C": 1.0, "h2o": 1.0}),
        ("ACONb", {"acon-C": -1.0, "h2o": -1.0, "icit": 1.0}),
        ("ICDH", {"icit": -1.0, "nadp": -1.0, "akg": 1.0, "co2": 1.0, "nadph": 1.0}),
        ("AKGDH", {"akg": -1.0, "coa": -1.0, "nad": -1.0, "succoa": 1.0, "co2": 1.0, "nadh": 1.0}),
        ("SCoAS", {"succoa": -1.0, "adp": -1.0, "pi": -1.0, "succ": 1.0, "atp": 1.0, "coa": 1.0}),
        ("SDH", {"succ": -1.0, "fum": 1.0}),
        ("FUM", {"fum": -1.0, "h2o": -1.0, "mal-L": 1.0}),
        ("MDH", {"mal-L": -1.0, "nad": -1.0, "oaa": 1.0, "nadh": 1.0}),
        # --- Anaplerosis (OAA replenishment) ---
        ("PPC", {"pep": -1.0, "co2": -1.0, "oaa": 1.0, "pi": 1.0}),
        ("PCK", {"oaa": -1.0, "atp": -1.0, "pep": 1.0, "adp": 1.0, "co2": 1.0}),
        # --- Pentose phosphate (NADPH supply) ---
        ("G6PDH2r", {"g6p": -1.0, "nadp": -1.0, "6pgl": 1.0, "nadph": 1.0}),
        ("PGL", {"6pgl": -1.0, "h2o": -1.0, "6pgc": 1.0}),
        ("GND", {"6pgc": -1.0, "nadp": -1.0, "ru5p-D": 1.0, "co2": 1.0, "nadph": 1.0}),
        ("RPE", {"ru5p-D": -1.0, "xu5p-D": 1.0}),
        ("RPI", {"ru5p-D": -1.0, "r5p": 1.0}),
        ("TKT1", {"xu5p-D": -1.0, "r5p": -1.0, "s7p": 1.0, "g3p": 1.0}),
        ("TALA", {"s7p": -1.0, "g3p": -1.0, "f6p": 1.0, "e4p": 1.0}),
        ("TKT2", {"xu5p-D": -1.0, "e4p": -1.0, "f6p": 1.0, "g3p": 1.0}),
        # --- Electron transport chain ---
        ("CYTBD", {"nadh": -1.0, "o2": -0.5, "nad": 1.0, "h2o": 0.5}),
        ("THD2", {"nadph": -1.0, "nad": -1.0, "nadp": 1.0, "nadh": 1.0}),
        # --- ATP synthase ---
        ("ATPs4r", {"adp": -1.0, "pi": -1.0, "atp": 1.0, "h2o": 1.0}),

        # ================================================================
        #  CALVIN-BENSON-BASSHAM CYCLE (CBB / Calvin cycle)
        #  CO₂ fixation for photoautotrophic growth (doc/22 §7)
        # ================================================================

        # RuBisCO: ribulose-1,5-bisphosphate + CO₂ -> 2 × 3-phosphoglycerate
        ("RBPC", {"rbp": -1.0, "co2": -1.0, "3pg": 2.0}),
        # PRK: ribulose-5-phosphate + ATP -> ribulose-1,5-bisphosphate + ADP
        ("PRUK", {"ru5p-D": -1.0, "atp": -1.0, "rbp": 1.0, "adp": 1.0}),
        # FBPase (gluconeogenic): fructose-1,6-bisphosphate -> fructose-6-P + Pi
        ("FBPASE", {"fbp": -1.0, "h2o": -1.0, "f6p": 1.0, "pi": 1.0}),
        # SBP aldolase: dihydroxyacetone-P + erythrose-4-P -> sedoheptulose-1,7-bisP
        ("SBPaldo", {"dhap": -1.0, "e4p": -1.0, "sbp": 1.0}),
        # SBPase: sedoheptulose-1,7-bisphosphate -> sedoheptulose-7-P + Pi
        ("SBPase", {"sbp": -1.0, "h2o": -1.0, "s7p": 1.0, "pi": 1.0}),
        # NADP-dependent GAPDH (Calvin cycle reduction: 1,3BPG -> G3P using NADPH)
        ("GAPD2", {"13pg": -1.0, "nadph": -1.0, "g3p": 1.0, "nadp": 1.0, "pi": 1.0}),
        # Photosynthetic electron transport (linear e- flow: H₂O -> NADPH + O₂)
        # Represents PSII + PSI combined:  H₂O + NADP⁺ → NADPH + ½O₂
        # Upper bound is set by light intensity in PhotoautotrophicFluxBalance.
        ("PET", {"h2o": -1.0, "nadp": -1.0, "nadph": 1.0, "o2": 0.5}),

        # ================================================================
        #  AMINO ACID BIOSYNTHESIS (E. coli K-12 / iML1515)
        # ================================================================

        # --- Glutamate family (from a-ketoglutarate) ---
        ("GLUDy", {"akg": -1.0, "nadph": -1.0, "nh4": -1.0,
                   "glu-L": 1.0, "nadp": 1.0, "h2o": 1.0}),
        ("GLNS", {"glu-L": -1.0, "atp": -1.0, "nh4": -1.0,
                  "gln-L": 1.0, "adp": 1.0, "pi": 1.0}),
        ("P5CD", {"gln-L": -1.0, "nadph": -2.0, "h2o": -1.0,
                  "pro-L": 1.0, "nadp": 2.0, "nh4": 1.0}),

        # --- Aspartate family (from oxaloacetate) ---
        ("ASPTA", {"oaa": -1.0, "glu-L": -1.0,
                   "asp-L": 1.0, "akg": 1.0}),
        ("ASPK", {"asp-L": -1.0, "atp": -1.0,
                  "4pasp": 1.0, "adp": 1.0}),
        ("HSD", {"4pasp": -1.0, "nadph": -1.0, "h2o": -1.0,
                 "hom-L": 1.0, "nadp": 1.0, "pi": 1.0}),
        ("THRA", {"hom-L": -1.0, "atp": -1.0, "h2o": -1.0,
                  "thr-L": 1.0, "adp": 1.0, "pi": 1.0}),
        ("ILE_syn", {"thr-L": -1.0, "pyr": -1.0, "glu-L": -1.0,
                     "nadph": -1.0,
                     "ile-L": 1.0, "akg": 1.0, "nadp": 1.0,
                     "nh4": 1.0, "h2o": 1.0}),
        ("MET_syn", {"hom-L": -1.0, "cys-L": -1.0, "succoa": -1.0,
                     "atp": -1.0,
                     "met-L": 1.0, "succ": 1.0, "coa": 1.0,
                     "adp": 1.0, "pi": 1.0}),
        ("LYSa", {"asp-L": -1.0, "atp": -1.0, "pyr": -1.0,
                  "nadph": -1.0,
                  "lys-L": 1.0, "adp": 1.0, "pi": 1.0,
                  "nadp": 1.0, "co2": 1.0, "h2o": 1.0}),
        ("ASNS", {"asp-L": -1.0, "atp": -1.0, "gln-L": -1.0,
                  "h2o": -1.0,
                  "asn-L": 1.0, "adp": 1.0, "pi": 1.0,
                  "glu-L": 1.0, "amp": 1.0, "ppi": 1.0}),

        # --- Pyruvate family (from pyruvate) ---
        ("ALAT", {"pyr": -1.0, "glu-L": -1.0,
                  "ala-L": 1.0, "akg": 1.0}),
        ("VAL_syn", {"pyr": -2.0, "nadph": -1.0, "glu-L": -1.0,
                     "val-L": 1.0, "akg": 1.0, "nadp": 1.0,
                     "co2": 1.0, "h2o": 1.0}),
        ("LEU_syn", {"pyr": -3.0, "accoa": -1.0, "glu-L": -1.0,
                     "leu-L": 1.0, "coa": 1.0, "akg": 1.0,
                     "co2": 2.0, "h2o": 1.0}),

        # --- Serine family (from 3-phosphoglycerate) ---
        ("PGDH", {"3pg": -1.0, "nad": -1.0,
                  "3php": 1.0, "nadh": 1.0}),
        ("PSAT", {"3php": -1.0, "glu-L": -1.0,
                  "ser-L": 1.0, "akg": 1.0}),
        ("SHMT", {"ser-L": -1.0, "thf": -1.0,
                  "gly": 1.0, "methf": 1.0}),
        ("CYS_syn", {"ser-L": -1.0, "accoa": -1.0, "h2s": -1.0,
                     "cys-L": 1.0, "coa": 1.0, "h2o": 1.0}),

        # --- Aromatic family (PEP + E4P -> chorismate -> Phe/Tyr/Trp) ---
        ("SHIKK", {"pep": -2.0, "e4p": -1.0, "atp": -1.0,
                   "chorismate": 1.0, "adp": 1.0, "pi": 2.0, "h2o": 1.0}),
        ("PPAra", {"chorismate": -1.0, "glu-L": -1.0,
                   "phe-L": 1.0, "akg": 1.0, "co2": 1.0, "h2o": 1.0}),
        ("TYR_syn", {"chorismate": -1.0, "akg": -1.0, "nadph": -1.0,
                     "tyr-L": 1.0, "glu-L": 1.0, "nadp": 1.0, "h2o": 1.0}),
        ("TRP_syn", {"chorismate": -1.0, "gln-L": -1.0, "ser-L": -1.0,
                     "trp-L": 1.0, "pyr": 1.0, "glu-L": 1.0,
                     "g3p": 1.0, "h2o": 1.0, "ppi": 1.0}),

        # --- Histidine (from PRPP + ATP, condensed) ---
        ("HIS_syn", {"atp": -1.0, "prpp": -2.0, "glu-L": -1.0,
                     "his-L": 1.0, "akg": 1.0, "fum": 1.0,
                     "adp": 1.0, "ppi": 1.0, "h2o": 2.0}),

        # ================================================================
        #  NUCLEOTIDE BIOSYNTHESIS
        # ================================================================

        ("PRPPS", {"r5p": -1.0, "atp": -1.0, "prpp": 1.0, "adp": 1.0}),

        # Purine: PRPP -> IMP -> AMP/GMP
        ("IMPS", {"prpp": -1.0, "asp-L": -1.0, "gln-L": -1.0,
                  "atp": -1.0,
                  "imp": 1.0, "glu-L": 1.0, "fum": 1.0,
                  "adp": 1.0, "ppi": 1.0}),
        ("ADSS", {"imp": -1.0, "asp-L": -1.0, "gtp": -1.0,
                  "adp": 1.0, "fum": 1.0, "ppi": 1.0}),
        ("IMPDH", {"imp": -1.0, "nad": -1.0, "h2o": -1.0,
                   "xmp": 1.0, "nadh": 1.0}),
        ("GMPS", {"xmp": -1.0, "gln-L": -1.0, "atp": -1.0,
                  "gmp": 1.0, "glu-L": 1.0, "amp": 1.0, "ppi": 1.0}),
        ("NDPKat", {"atp": -1.0, "adp": 1.0, "gdp": -1.0, "gtp": 1.0}),
        ("NDPKut", {"atp": -1.0, "adp": 1.0, "udp": -1.0, "utp": 1.0}),
        ("NDPKct", {"atp": -1.0, "adp": 1.0, "cdp": -1.0, "ctp": 1.0}),

        # Pyrimidine: OAA + Gln -> UMP -> UTP/CTP
        ("PYRsyn", {"oaa": -1.0, "gln-L": -1.0, "atp": -1.0,
                    "dho": 1.0, "glu-L": 1.0, "adp": 1.0, "pi": 1.0}),
        ("DHORD", {"dho": -1.0, "nad": -1.0, "h2o": -1.0,
                   "orot": 1.0, "nadh": 1.0}),
        ("OPRT", {"orot": -1.0, "prpp": -1.0,
                  "omp": 1.0, "ppi": 1.0}),
        ("UMPsyn", {"omp": -1.0, "ump": 1.0, "co2": 1.0}),
        ("UTPS", {"ump": -1.0, "atp": -1.0, "utp": 1.0, "adp": 1.0}),
        ("CTPS", {"utp": -1.0, "gln-L": -1.0, "atp": -1.0,
                  "ctp": 1.0, "glu-L": 1.0, "adp": 1.0, "pi": 1.0}),

        # Deoxyribonucleotide (NTP -> dNTP via ribonucleotide reductase)
        ("RNRa", {"atp": -1.0, "nadph": -1.0,
                  "datp": 1.0, "nadp": 1.0, "h2o": 1.0}),
        ("RNRb", {"gtp": -1.0, "nadph": -1.0,
                  "dgtp": 1.0, "nadp": 1.0, "h2o": 1.0}),
        ("RNRc", {"ctp": -1.0, "nadph": -1.0,
                  "dctp": 1.0, "nadp": 1.0, "h2o": 1.0}),
        ("RNRd", {"utp": -1.0, "nadph": -1.0,
                  "dttp": 1.0, "nadp": 1.0, "h2o": 1.0}),

        # ================================================================
        #  COFACTOR / VITAMIN TRANSPORT
        # ================================================================

        # THF (vitamin B9) transport
        ("THFtex", {"thf_e": -1.0, "thf": 1.0}),
        # Thiamine pyrophosphate (vitamin B1) - thmpp transport
        ("TMPPtex", {"thmpp_e": -1.0, "thmpp": 1.0}),
        # Pyridoxal-5-phosphate (vitamin B6) - plp transport
        ("PLPtex", {"plp_e": -1.0, "plp": 1.0}),
        # Coenzyme A synthesis from pantothenate + cysteine
        ("COAsynth", {"pant_e": -1.0, "cys-L": -1.0, "atp": -2.0,
                      "coa": 1.0, "adp": 2.0, "pi": 2.0,
                      "co2": 1.0, "h2o": 1.0}),
        # Pantothenate (vitamin B5) exchange – needed for CoA biosynthesis
        ("EX_pant_e", {"pant_e": -1.0}),
        # H2S transport (for cysteine synthesis)
        ("H2Stex", {"h2s_e": -1.0, "h2s": 1.0}),
        # Arginine biosynthesis (glutamate → arginine, condensed)
        ("ARG_syn", {"glu-L": -2.0, "asp-L": -1.0, "atp": -3.0,
                     "nh4": -1.0,
                     "arg-L": 1.0, "akg": 1.0, "fum": 1.0,
                     "adp": 3.0, "pi": 3.0, "h2o": 2.0}),
        # Thiamine pyrophosphate exchange – biomass trace cofactor
        ("EX_thmpp_e", {"thmpp_e": -1.0}),
        # Folate cycle closure: methf → thf (regenerates THF)
        ("FOLateCYCLE", {"methf": -1.0, "h2o": -1.0, "thf": 1.0}),
        # NAD / NADP transport + exchange – break pool conservation deadlock
        ("NADtex", {"nad_e": -1.0, "nad": 1.0}),
        ("EX_nad_e", {"nad_e": -1.0}),
        ("NADPtex", {"nadp_e": -1.0, "nadp": 1.0}),
        ("EX_nadp_e", {"nadp_e": -1.0}),
    ]

    for rxn_id, stoich in core_reactions:
        # Remove existing reaction with same id if present
        if rxn_id in model.reactions:
            del model.reactions[rxn_id]
        # Ensure all metabolites exist in the model
        for met in stoich:
            if met not in model.metabolites:
                model.metabolites.add(met)
        model.add_reaction(Reaction(
            id=rxn_id,
            name=rxn_id,
            stoichiometry=stoich,
            lower_bound=-1000.0,
            upper_bound=1000.0,
            subsystem="core",
        ))


def _run_human_simulation(program: Program) -> SimResult:
    """``#sim kind=human`` — virtual-patient simulation (doc/27+28+29).

    Parses ``#person``, ``#trait``, ``#disease``, ``#disease_gene``,
    ``#disease_metabolite``, ``#drug``, ``#pd_effect`` annotations from
    ``program.sim_extensions``, builds a
    :class:`~helixlang.human.virtual_patient.VirtualPatientConfig`, runs
    the full PBPK→PD→labs→vitals→recovery integration loop, and returns
    the complete time-series as a ``SimResult`` with one row per hour.

    Falls back to the legacy
    :class:`~helixlang.human.simulation.HumanSimulation` when no
    person/drug annotations are present (backward compatible).
    """
    ext = program.sim_extensions

    # --- Detect which backend to use ---
    has_person = any(k.startswith("person_") for k in ext)
    has_drugs = "drugs" in ext and ext["drugs"]

    if not has_person and not has_drugs:
        return _run_human_simulation_legacy(program)

    return _run_virtual_patient(program)


def _run_human_simulation_legacy(program: Program) -> SimResult:
    """Legacy ``#sim kind=human`` backend using HumanSimulation (doc/27)."""
    from helixlang.human.simulation import (
        HumanSimulation,
        HumanSimulationConfig,
    )

    ext = program.sim_extensions
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
    from helixlang.human.genotype import create_default_genotype
    from helixlang.human.virtual_patient import VirtualPatient, VirtualPatientConfig

    ext = program.sim_extensions

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

    # --- Auto-infer PD from drug target when no explicit #pd_effect ---
    if drugs and not pd_effects:
        from helixlang.human.pharmacodynamics import infer_pd_from_drug as _infer_pd
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


# ---------------------------------------------------------------------------
# Annotation → config builders
# ---------------------------------------------------------------------------

def _build_genotype_from_helix(
    genotype: GenotypeProfile, ext: dict[str, Any],
) -> None:
    """Populate genotype from #gene annotations stored in sim_extensions.

    Routes pharmacogenes to the correct status dict:
    - CYP enzymes (CYP2D6, CYP3A4, ...) → ``cyp_status`` via star-allele
    - Transporters (SLCO1B1, ABCB1, ...) → ``transporter_status``
    - Non-CYP enzymes (UGT1A1, TPMT, ...) → ``non_cyp_enzyme_status``
    - Other genes → ``gene_variants`` (generic storage)
    """
    from helixlang.human.genotype import (
        CYP_ALLELE_ACTIVITIES,
        NON_CYP_ENZYMES,
        TRANSPORTER_ALLELE_EFFECTS,
        VALID_ZYGOSITY,
        Variant,
        _AlleleCall,
        _summarize_enzyme,
        _summarize_non_cyp,
        _summarize_transporter,
    )

    gene_list = ext.get("genes", [])
    if not isinstance(gene_list, list):
        return

    # Collect allele calls per gene
    cyp_calls: dict[str, list[_AlleleCall]] = {}
    transporter_calls: dict[str, list[_AlleleCall]] = {}
    non_cyp_calls: dict[str, list[_AlleleCall]] = {}
    other_genes: dict[str, list[Variant]] = {}

    for entry in gene_list:
        if not isinstance(entry, dict):
            continue
        gene_name = entry.get("name", "").strip()
        allele = entry.get("allele", "").strip()
        zygosity = entry.get("zygosity", "het")
        if zygosity not in VALID_ZYGOSITY:
            zygosity = "het"
        if not gene_name or not allele:
            continue

        # Import star-allele resolution here for clarity
        from helixlang.human.genotype import _resolve_star_allele

        star, dup_count = _resolve_star_allele(gene_name, allele)

        variant = Variant(
            gene_id=gene_name,
            chromosome="0",
            position=0,
            ref=".",
            alt=allele,
            zygosity=zygosity,
        )

        call = _AlleleCall(
            variant=variant,
            star=star,
            duplication_count=dup_count,
            copy_number=None,
        )

        gene_upper = gene_name.upper()
        if gene_upper in CYP_ALLELE_ACTIVITIES:
            cyp_calls.setdefault(gene_upper, []).append(call)
        elif gene_upper in TRANSPORTER_ALLELE_EFFECTS:
            transporter_calls.setdefault(gene_upper, []).append(call)
        elif gene_upper in NON_CYP_ENZYMES:
            non_cyp_calls.setdefault(gene_upper, []).append(call)
        else:
            other_genes.setdefault(gene_name, []).append(variant)

    # Summarize CYP enzymes
    for gene, calls in cyp_calls.items():
        genotype.cyp_status[gene] = _summarize_enzyme(gene, calls)

    # Summarize transporters
    for gene, calls in transporter_calls.items():
        genotype.transporter_status[gene] = _summarize_transporter(gene, calls)

    # Summarize non-CYP enzymes
    for gene, calls in non_cyp_calls.items():
        genotype.non_cyp_enzyme_status[gene] = _summarize_non_cyp(gene, calls)

    # Store non-pharmacogene variants in gene_variants
    for gene_name, variants in other_genes.items():
        for v in variants:
            genotype.add_gene_variant(gene_name, v)


def _build_traits_from_helix(ext: dict[str, Any]) -> ExternalTraits:
    """Build ExternalTraits from #person + #trait annotations."""
    from helixlang.human.phenotype import ExternalTraits as _Ext
    return _Ext(
        age_years=_opt_float(ext, "person_age", 30.0),
        sex=ext.get("person_sex", "male"),
        body_weight_kg=_opt_float(ext, "person_weight", 70.0),
        height_cm=_opt_float(ext, "person_height", 170.0),
        ethnicity=ext.get("person_ethnicity", "european"),
        smoking_status=ext.get("trait_smoking", "never"),
        pack_years=_opt_float(ext, "trait_pack_years", 0.0),
        alcohol_drinks_per_week=_opt_float(ext, "trait_alcohol", 0.0),
        exercise_level=ext.get("trait_exercise", "moderate"),
        pregnant=ext.get("trait_pregnant", "false").lower() == "true",
    )


def _build_disease_from_helix(ext: dict[str, Any]) -> DiseaseState | None:
    """Build DiseaseState from #disease + #disease_gene + #disease_metabolite."""
    from helixlang.human.disease import (
        DiseaseState as _DS,
    )
    from helixlang.human.disease import (
        GenePerturbation as _GP,
    )
    from helixlang.human.disease import (
        MetabolitePerturbation as _MP,
    )
    disease_name = ext.get("disease_name", "")
    if not disease_name:
        return None

    gene_perts = []
    for entry in ext.get("disease_genes", []):
        if isinstance(entry, dict) and entry.get("gene"):
            gene_perts.append(_GP(
                gene_id=entry["gene"],
                perturbation_type=entry.get("type", "downregulate"),
                activity_fraction=_opt_float(entry, "activity", 0.0),
            ))

    met_perts = []
    for entry in ext.get("disease_metabolites", []):
        if isinstance(entry, dict) and entry.get("id"):
            met_perts.append(_MP(
                metabolite_id=entry["id"],
                perturbation_type=entry.get("type", "accumulate"),
                initial_concentration_mm=_opt_float(entry, "concentration", 0.0),
                normal_concentration_mm=_opt_float(entry, "normal", 0.0),
            ))

    return _DS(
        name=disease_name,
        category=ext.get("disease_category", "metabolic_overload"),
        gene_perturbations=gene_perts,
        metabolite_perturbations=met_perts,
        severity=_opt_float(ext, "disease_severity", 0.5),
        onset_age_years=_opt_float(ext, "disease_onset_age", 40.0),
        description=ext.get("disease_description", ""),
    )


def _build_drugs_from_helix(ext: dict[str, Any]) -> list[Drug]:
    """Build Drug objects from #drug annotations.

    Supports all Drug fields including:
      - drug_type, target_protein, binding_affinity_kd
      - cyp_metabolism (e.g. cyp_metabolism="CYP3A4:0.5,CYP2D6:0.3")
      - transporter_affected (e.g. transporter_affected="SLCO1B1:0.6,ABCB1:0.15")
      - non_cyp_metabolism (e.g. non_cyp_metabolism="UGT1A1:0.7")

    When SMILES is provided and MW/ADME are not all explicit, auto-infers
    properties via ``parse_drug_smiles`` (MW/formula/LogP) and
    ``smiles_to_adme`` (bioavailability, Vd, CL, half-life, etc.).
    PD effects are auto-inferred from ``target_protein`` when no explicit
    ``#pd_effect`` is provided.
    """
    from helixlang.human.drug import SMALL_MOLECULE, parse_drug_smiles, smiles_to_adme
    from helixlang.human.drug import Drug as _Drug
    from helixlang.human.drug import DrugMolecule as _DM

    def _parse_fraction_map(raw: str) -> dict[str, float]:
        """Parse 'CYP3A4:0.5,CYP2D6:0.3' → {'CYP3A4': 0.5, 'CYP2D6': 0.3}."""
        result: dict[str, float] = {}
        if not raw:
            return result
        for part in raw.split(","):
            part = part.strip()
            if ":" in part:
                key, val = part.split(":", 1)
                try:
                    result[key.strip()] = float(val.strip())
                except ValueError:
                    pass
        return result

    drugs = []
    for entry in ext.get("drugs", []):
        if not isinstance(entry, dict) or not entry.get("name"):
            continue

        name = entry["name"]
        smiles = entry.get("smiles", "")
        drug_type = entry.get("drug_type", SMALL_MOLECULE)
        target_protein = str(entry.get("target_protein", entry.get("target", "")))
        binding_kd = _opt_float(entry, "binding_affinity_kd", 0.0)

        # --- Molecule: prefer parse_drug_smiles when SMILES is available ---
        if smiles:
            mol = parse_drug_smiles(smiles, name=name, drug_type=drug_type)
            if target_protein:
                mol.target_protein = target_protein
            if binding_kd > 0:
                mol.binding_affinity_kd_um = binding_kd
            # Allow explicit mw to override SMILES-parsed value
            explicit_mw = _opt_float(entry, "mw", 0.0)
            if explicit_mw > 0:
                mol.molecular_weight_da = explicit_mw
        else:
            mol = _DM(
                name=name,
                drug_type=drug_type,
                smiles=smiles,
                molecular_weight_da=_opt_float(entry, "mw", 0.0),
                formula=entry.get("formula", ""),
                target_protein=target_protein,
                binding_affinity_kd_um=binding_kd,
                protein_binding_fraction=_opt_float(entry, "protein_binding", 0.0),
            )

        # --- ADME inference from SMILES when not all explicit ---
        explicit_cl = "cl" in entry
        explicit_vd = "vd" in entry
        explicit_hl = "half_life" in entry
        if smiles and not (explicit_cl and explicit_vd and explicit_hl):
            try:
                inferred = smiles_to_adme(
                    smiles,
                    drug_type=drug_type,
                    mw_da=mol.molecular_weight_da,
                )
                # Fill in only missing (zero-valued) fields
                if mol.molecular_weight_da <= 0:
                    mol.molecular_weight_da = inferred.get("molecular_weight_da", 0.0)
                if mol.log_p <= 0:
                    mol.log_p = inferred.get("log_p", 0.0)
                if not explicit_cl:
                    entry.setdefault("cl", str(round(inferred.get("clearance_ml_per_min", 100.0), 2)))
                if not explicit_vd:
                    entry.setdefault("vd", str(round(inferred.get("volume_distribution_l", 50.0), 2)))
                if not explicit_hl:
                    entry.setdefault("half_life", str(round(inferred.get("half_life_h", 6.0), 2)))
                entry.setdefault("bioavailability", str(round(inferred.get("bioavailability", 0.8), 3)))
                entry.setdefault("protein_binding", str(round(inferred.get("protein_binding", 0.5), 3)))
                entry.setdefault("absorption_rate", str(round(inferred.get("absorption_rate_h", 1.0), 2)))
                entry.setdefault("hepatic_eh", str(round(inferred.get("hepatic_extraction_ratio", 0.3), 3)))
                entry.setdefault("renal_fraction", str(round(inferred.get("renal_fraction", 0.2), 3)))
            except Exception:
                pass  # Graceful degradation: use hand-specified values

        # --- Build Drug ---
        drug = _Drug(
            molecule=mol,
            dose_mg=_opt_float(entry, "dose", 0.0),
            dosing_interval_h=_opt_float(entry, "interval", 24.0),
            route=entry.get("route", "oral"),
            duration_days=_opt_float(entry, "duration", 30.0),
            bioavailability=_opt_float(entry, "bioavailability", 1.0),
            absorption_rate_h=_opt_float(entry, "absorption_rate", 1.0),
            volume_distribution_l=_opt_float(entry, "vd", 50.0),
            clearance_ml_per_min=_opt_float(entry, "cl", 100.0),
            half_life_h=_opt_float(entry, "half_life", 6.0),
            hepatic_extraction_ratio=_opt_float(entry, "hepatic_eh", 0.0),
            renal_fraction=_opt_float(entry, "renal_fraction", 0.0),
            cyp_metabolism=_parse_fraction_map(entry.get("cyp_metabolism", "")),
            transporter_affected=_parse_fraction_map(entry.get("transporter_affected", "")),
            non_cyp_metabolism=_parse_fraction_map(entry.get("non_cyp_metabolism", "")),
        )
        drugs.append(drug)
    return drugs


def _build_pd_from_helix(ext: dict[str, Any]) -> dict[str, Pharmacodynamics]:
    """Build PD effects from #pd_effect annotations."""
    from helixlang.human.pharmacodynamics import PDEffect as _PE
    from helixlang.human.pharmacodynamics import Pharmacodynamics as _PD
    effects_by_drug: dict[str, list[_PE]] = {}
    for entry in ext.get("pd_effects", []):
        if not isinstance(entry, dict) or not entry.get("drug"):
            continue
        drug_key = entry["drug"].lower().replace(" ", "_").replace("-", "_")
        eff = _PE(
            target_reaction=entry.get("target", "BIOMASSReaction"),
            effect_type=entry.get("type", "inhibition"),
            ec50_um=_opt_float(entry, "ec50", 1.0),
            emax=_opt_float(entry, "emax", 1.0),
            hill_coefficient=_opt_float(entry, "hill", 1.0),
        )
        effects_by_drug.setdefault(drug_key, []).append(eff)

    result = {}
    for drug_key, effs in effects_by_drug.items():
        result[drug_key] = _PD(
            drug_name=drug_key,
            effects=effs,
        )
    return result


def _build_qsp_bindings_from_helix(ext: dict[str, Any]) -> dict[str, Any]:
    """Build QSP binding models from #qsp_binding annotations."""
    bindings = []
    for entry in ext.get("qsp_bindings", []):
        if not isinstance(entry, dict) or not entry.get("drug"):
            continue
        bindings.append({
            "drug": entry["drug"],
            "kind": entry.get("kind", "mass_action"),
            "kd_nM": _opt_float(entry, "kd_nM", 10.0),
            "kss_nM": _opt_float(entry, "kss_nM", 5.0),
            "emax": _opt_float(entry, "emax", 1.0),
            "kd_agonist_nM": _opt_float(entry, "kd_agonist", 10.0),
            "ki_antagonist_nM": _opt_float(entry, "ki", 5.0),
        })
    return {"qsp_bindings": bindings}


def _build_endocrine_config_from_helix(ext: dict[str, Any]) -> dict[str, Any]:
    """Build endocrine config from #endocrine_config annotations."""
    configs = []
    for entry in ext.get("endocrine_configs", []):
        if not isinstance(entry, dict) or not entry.get("axis"):
            continue
        configs.append({
            "axis": entry["axis"],
            "severity": _opt_float(entry, "severity", 0.0),
            "level": _opt_float(entry, "level", 0.0),
        })
    return {"endocrine_configs": configs}


def _build_immune_config_from_helix(ext: dict[str, Any]) -> dict[str, Any]:
    """Build immune config from #immune_config annotations."""
    configs = []
    for entry in ext.get("immune_configs", []):
        if not isinstance(entry, dict):
            continue
        configs.append({
            "infection_severity": _opt_float(entry, "infection_severity", 0.0),
            "autoimmune_activation": _opt_float(entry, "autoimmune_activation", 0.0),
            "immunosuppression": _opt_float(entry, "immunosuppression", 0.0),
        })
    return {"immune_configs": configs}


_SIM_BACKENDS: dict[str, Callable[[Program], SimResult]] = {
    "3d_morphology": _run_3d_morphology,
    "codec_benchmark": _run_codec_benchmark,
    "codon_usage": _run_codon_usage,
    "cello_workflow": _run_cello_workflow,
    "consortium": _run_consortium,
    "digital_evolution": _run_digital_evolution,
    "directed_evolution": _run_directed_evolution,
    "ecosystem": _run_ecosystem,
    "fate_analysis": _run_fate_analysis,
    "human": _run_human_simulation,
    "morphogen_gradient": _run_morphogen_gradient,
    "omics_calibration": _run_omics_calibration,
    "population_calibration": _run_population_calibration,
    "population_dbtl": _run_population_dbtl,
    "protein_fitness": _run_protein_fitness,
    "protein_structure": _run_protein_structure,
    "spatial_dfba": _run_spatial_dfba,
    "spatial_evolution": _run_spatial_evolution,
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
    "contact": "contact",
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
    # #sim keys merge over #config keys (the same open extension point,
    # wiring.md §8.6); the population backend reads both so e.g.
    # `#sim lbm=true` and `#config mechanics=contact` combine.
    sim = {**program.config.sim, **program.sim_extensions}
    mechanics_raw = sim.get("mechanics", "none")
    mechanics = _POP_MECHANICS.get(mechanics_raw)
    if mechanics is None and mechanics_raw != "none":
        raise SimConfigError(
            f"sim key 'mechanics': expected one of "
            f"{sorted(_POP_MECHANICS)}, got {mechanics_raw!r}")
    cell_shape = sim.get("cell_shape")
    if cell_shape not in (None, "rod"):
        raise SimConfigError(
            f"sim key 'cell_shape': expected 'rod' or unset, got "
            f"{cell_shape!r}")
    width = _opt_int(sim, "grid_width", 32)
    height = _opt_int(sim, "grid_height", 32)
    depth = _opt_int(sim, "grid_depth", 1)
    flow = _build_pop_flow(sim, width, height)
    lbm = _build_pop_lbm(sim, width, height, depth)
    lbm_3d = _opt_bool(sim, "lbm_3d", False)
    if lbm_3d:
        if _opt_bool(sim, "lbm", False):
            raise SimConfigError(
                "sim keys 'lbm' and 'lbm_3d' are mutually exclusive: "
                "use one or the other as the flow driver")
        if flow is not None:
            raise SimConfigError(
                "sim keys 'flow' and 'lbm_3d' are mutually exclusive: "
                "use one or the other as the flow driver")
        if depth < 2:
            raise SimConfigError(
                "sim key 'lbm_3d' requires grid_depth > 1 (a 3D lattice)")
    elif flow is not None and lbm is not None:
        raise SimConfigError(
            "sim keys 'flow' and 'lbm' are mutually exclusive: "
            "use one or the other as the flow driver")
    # #genome source=... (Design 5): fields merge into sim_extensions under a
    # genome_ prefix (the same open extension point as #sim); build the
    # shared sparse template once per run.
    ext = program.sim_extensions
    genome = None
    if ext.get("genome") or any(k.startswith("genome_") for k in ext):
        from helixlang.apps.genome_scale import build_genome
        genome = build_genome(
            source=str(ext.get("genome_source", "synth-4300")),
            tf_map=str(ext.get("genome_tf_map", "regulon")),
            grn_mode=str(ext.get("genome_grn_mode", "sparse")),
            seed=_opt_int(ext, "genome_seed", 7),
            active_gene_budget=_opt_int(
                ext, "genome_active_gene_budget", 512),
            noise_seed=_opt_int_or_none(sim, "noise_seed", None),
        )
    config = PopulationConfig(
        max_size=_opt_int(sim, "population_size", 1000),
        grid_width=width,
        grid_height=height,
        grid_depth=depth,
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
        dfba_dt_h=_opt_float(sim, "dfba_dt_h", 0.05),
        dfba_energy_scale=_opt_float(sim, "dfba_energy_scale", 1.25e8),
        dfba_initial_biomass_gdw=_opt_float(sim, "dfba_initial_biomass_gdw",
                                            0.05),
        dfba_glucose_half_saturation_mm=_opt_float(
            sim, "dfba_glucose_half_saturation_mm", 0.1),
        dfba_oxygen_max_uptake=_opt_float(sim, "dfba_oxygen_max_uptake", 40.0),
        dfba_oxygen_half_saturation_mm=_opt_float(
            sim, "dfba_oxygen_half_saturation_mm", 0.01),
        dfba_shared_batch=_opt_bool(sim, "dfba_shared", False),
        acetate_switch=_opt_bool(sim, "acetate_switch", False),
        acetate_switch_threshold_mm=_opt_float(
            sim, "acetate_switch_threshold_mm", 0.5),
        program=program,
        chunk=Compiler(STANDARD_TABLE).compile(program),
        ops_per_tick=program.config.ops_per_tick,
        genome=genome,
        flow=flow,
        cell_shape=cell_shape,
        cell_length_um=_opt_float(
            sim, "length_um", _opt_float(sim, "cell_length_um", 2.0)),
        cell_diameter_um=_opt_float(
            sim, "diameter_um", _opt_float(sim, "cell_diameter_um", 1.0)),
        contact_stiffness=_opt_float(sim, "contact_stiffness", 1.0e3),
        fluid_viscosity_mpas=_opt_float(sim, "fluid_viscosity_mpas", 1.0),
        lbm=lbm,
        flow_substeps=_opt_int(sim, "lbm_substeps", 1),
    )
    config.environment = _environment(
        program, config.grid_width, config.grid_height)
    return config


def _build_pop_flow(sim: dict[str, str], width: int,
                    height: int) -> FlowField | None:
    """Build the analytic Level-1 flow field from ``#sim flow=...``."""
    raw = sim.get("flow")
    if raw is None:
        return None
    if raw == "channel_poiseuille":
        direction = sim.get("direction", "E")
        mean = _opt_float(sim, "mean_velocity_um_s", 50.0)
        return channel_poiseuille(width, height, mean, direction)
    if raw == "stagnant":
        return stagnant(width, height)
    raise SimConfigError(
        f"sim key 'flow': expected 'channel_poiseuille' or 'stagnant', "
        f"got {raw!r}")


def _build_pop_lbm(sim: dict[str, str], width: int, height: int,
                   depth: int = 1) -> object | None:
    """Build the Level-2 LBM solver from ``#sim lbm=true``/``lbm_3d=true``.

    ``lbm=true`` builds the 2D D2Q9 solver (``apps.lattice_boltzmann``);
    ``lbm_3d=true`` builds the 3D D3Q19 solver
    (``apps.lattice_boltzmann_3d``) over the ``width x height x depth``
    volume.  Both share the same key family (``relaxation_omega``,
    ``lbm_inlet_density``, ``lbm_outlet_density``, ``lbm_substeps``).
    """
    lbm_3d = _opt_bool(sim, "lbm_3d", False)
    if not (_opt_bool(sim, "lbm", False) or lbm_3d):
        return None
    try:
        import numpy as np  # noqa: F401
    except ImportError as e:
        raise SimConfigError(
            "sim keys 'lbm'/'lbm_3d' require numpy (lattice Boltzmann "
            "solver)") from e
    omega = _opt_float(sim, "relaxation_omega", 1.2 if lbm_3d else 1.5)
    inlet_velocity = _opt_float(sim, "lbm_inlet_velocity", 0.0)
    inlet_density = _opt_float(sim, "lbm_inlet_density", 1.001)
    outlet_density = _opt_float(sim, "lbm_outlet_density", 0.999)
    if lbm_3d:
        from helixlang.apps.lattice_boltzmann_3d import LatticeBoltzmann3D

        return LatticeBoltzmann3D(
            width=width,
            height=height,
            depth=depth,
            omega=omega,
            inlet_velocity=inlet_velocity,
            inlet_density=inlet_density,
            outlet_density=outlet_density,
        )
    from helixlang.apps.lattice_boltzmann import LatticeBoltzmann

    return LatticeBoltzmann(
        width=width,
        height=height,
        omega=omega,
        inlet_velocity=inlet_velocity,
        inlet_density=inlet_density,
        outlet_density=outlet_density,
    )


def _seed_cells(config: PopulationConfig, n: int) -> list[PopulationCell]:
    """Pack ``n`` cells as a centered colony block (like the apps do).

    With ``grid_depth > 1`` the block fills a centered cuboid of
    ``side x side x side`` layers so the 3D D3Q19 colony occupies all
    three axes (cells default to z = 0 otherwise).
    """
    w, h, d = config.grid_width, config.grid_height, config.grid_depth
    side = 1
    layers = 1
    while side * side * layers < n:
        side += 1
        if side * side * layers >= n:
            break
        layers = min(side, d)
        if side * side * layers >= n:
            break
        side += 1
    side = max(1, side)
    layers = max(1, min(layers, d))
    off_x = max(0, (w - side) // 2)
    off_y = max(0, (h - side) // 2)
    off_z = max(0, (d - layers) // 2)
    cells: list[PopulationCell] = []
    for i in range(n):
        x = min(off_x + i % side, w - 1)
        y = min(off_y + (i // side) % side, h - 1)
        z = min(off_z + (i // (side * side)) % layers, d - 1)
        body = None
        if config.cell_shape == "rod":
            from helixlang.cell_body import CellBody

            body = CellBody(
                x=(x + 0.5) * LATTICE_SPACING_UM,
                y=(y + 0.5) * LATTICE_SPACING_UM,
                length_um=config.cell_length_um,
                diameter_um=config.cell_diameter_um,
                angle=0.0,
            )
        cells.append(PopulationCell(id=i, x=x, y=y, z=z, body=body))
    return cells


def _run_population(program: Program) -> ColonyResult:
    config = _build_population_config(program)
    if config.dfba_enabled and not program.media:
        raise SimConfigError(
            "backend=population with dfba=true requires at least one "
            "#media declaration (shared substrate fields)")

    # Phase G: auto-attach GEM model if a genome is specified
    ext = program.sim_extensions
    genome_path = ext.get("genome", "")
    if genome_path and config.dfba_enabled:
        try:
            from helixlang.apps.gem_pipeline import run_gem_pipeline
            _gem_result = run_gem_pipeline(
                genome_fasta=genome_path,
                organism=ext.get("organism", "e_coli_k12"),
                medium=ext.get("medium", "glucose_minimal"),
            )
            if (_gem_result.metabolic_model is not None
                    and getattr(_gem_result, "growth_rate", 0.0) > 0):
                config.metabolic_model = _gem_result.metabolic_model
        except Exception:
            pass  # non-fatal: fall back to ECOLI_CORE_MODEL

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
