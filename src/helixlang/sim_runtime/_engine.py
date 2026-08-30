"""Sim backend engine: orchestration, config builders, shared state helpers.

doc/38 §9 split: the per-backend ``_run_*`` executors moved verbatim to
:mod:`helixlang.sim_runtime.backends.pipelines`; backends registered in
:mod:`helixlang.sim_runtime.backends.core` delegate there.  This module keeps
``run()`` and the shared config/state helpers only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from helixlang.plugins.human.disease import DiseaseState
    from helixlang.plugins.human.drug import Drug
    from helixlang.plugins.human.genotype import GenotypeProfile
    from helixlang.plugins.human.pharmacodynamics import Pharmacodynamics
    from helixlang.plugins.human.phenotype import ExternalTraits
from helixlang.api.ast import ProgramView
from helixlang.api.backend import EffectiveConfig, RunRequest
from helixlang.core.ast_nodes import Program
from helixlang.core.codon_table import STANDARD_TABLE
from helixlang.core.compiler import Compiler
from helixlang.core.errors import PluginMissingError, SimConfigError
from helixlang.core.units import LATTICE_SPACING_UM
from helixlang.plugins.apps.ecosystem import (
    PatchConfig,
    ScalarConfig,
    Species,
    SubstrateConfig,
)
from helixlang.plugins.apps.morphogen_gradient import (
    MorphogenGene,
    MorphogenGradientConfig,
)
from helixlang.plugins.apps.synbio_automation import (
    TruthTable,
)
from helixlang.plugins.apps.whole_cell_calibration import (
    _fold_rate_from_k_fold,
)
from helixlang.plugins.runtime.dna_codec import translate_dna
from helixlang.plugins.runtime.environment import (
    ConcentrationField,
    DiurnalForcing,
    Environment,
    EnvironmentConfig,
    SeasonalForcing,
)
from helixlang.plugins.runtime.flow import FlowField, channel_poiseuille, stagnant
from helixlang.plugins.runtime.grn import GRN
from helixlang.plugins.runtime.metabolism import (
    DEFAULT_ENZYME_SCALE,
    ECOLI_CORE_GENE_REACTIONS,
    ECOLI_CORE_KCAT,
    ECOLI_CORE_MODEL,
    EnzymeCapacity,
    MetabolicModel,
    load_model,
)
from helixlang.plugins.runtime.population import (
    PopulationCell,
    PopulationConfig,
)
from helixlang.plugins.runtime.virtual_cell import VirtualCellConfig

from ._coerce import (
    _coerce_bool,
    _coerce_enum,
    _coerce_float,
    _opt_bool,
    _opt_float,
    _opt_float_dict,
    _opt_float_or_none,
    _opt_int,
    _opt_int_or_none,
    _opt_replicon_specs,
)
from ._types import (
    SimResult,
)


def run(program: Program, backend: str | None = None) -> SimResult | None:
    """Run the backend selected by ``backend`` or ``#config backend``.

    Returns ``None`` for ``classic`` (the CLI keeps the bytecode path).

    Dispatch (doc/38 §6.5): resolves through the
    :class:`~helixlang.api.backend.BackendRegistry` — ``#sim kind=...`` wins
    over ``#config backend=...`` / the ``backend`` argument, both map to the
    same ``Backend`` objects, and an unknown name is a hard error.
    """
    from helixlang.sim_runtime.backends import get_backend_registry

    name = backend or program.config.backend
    kind = program.extensions.get("kind")
    registry = get_backend_registry()
    if kind is not None and registry.has(kind=kind):
        resolved = registry.resolve(kind=kind)
    elif name is None or name == "classic":
        return None
    else:
        try:
            resolved = registry.resolve(backend=name)
        except PluginMissingError:
            raise SimConfigError(
                f"unknown backend {name!r}; expected one of "
                + ", ".join(sorted(registry.ids()) + ["classic"])) from None
    result = resolved.run(RunRequest(
        # E3 shim: the executors still take the raw Program; E4 replaces this
        # with a real ProgramView wrapper (doc/38 §6.5).
        program=cast(ProgramView, program),
        config=_effective_config(program),
        registry=registry,
        seed=getattr(program.config, "seed", None),
        source=getattr(program, "_source_text", None),
    ))
    if result is not None and not result.provenance:
        from helixlang.core.provenance import build_provenance
        seed = getattr(program.config, "seed", None)
        result.provenance = build_provenance(
            seed=seed,
            backend=resolved.id,
            source=getattr(program, "_source_text", None),
        )
    return result
def _effective_config(program: Program) -> EffectiveConfig:
    """Typed view of ``#config`` for a :class:`~helixlang.api.backend.RunRequest`.

    Mirrors the legacy adapter exactly; ``classic`` maps to ``backend=None``.
    """
    cfg = program.config
    return EffectiveConfig(
        kind=program.extensions.get("kind") or "classic",
        backend=None if cfg.backend == "classic" else cfg.backend,
        table=cfg.table,
        ticks=cfg.ticks,
        ops_per_tick=cfg.ops_per_tick,
        react_steps=cfg.react_steps,
        use_central_dogma=cfg.use_central_dogma,
        species=cfg.species,
        output=cfg.output[0] if cfg.output else "stdout",
    )
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
    from helixlang.plugins.runtime.metabolism import MetabolicModel
    from helixlang.plugins.runtime.metabolism import Reaction as Rxn
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
_SIM_DFBA_DEFAULT_COLUMNS = [
    "tick", "time_h", "total_glucose", "total_biomass", "total_acetate",
    "depletion_front", "co2_overflow",
]
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
_CONSORTIUM_DEFAULT_COLUMNS = [
    "tick", "alive", "consensus_fraction", "consensus_reached",
    "output_rate", "cumulative_output", "active_actuators", "avg_signal",
    "producer_fraction", "sensor_fraction", "actuator_fraction",
]
_DIGITAL_EVO_DEFAULT_COLUMNS = ["generation", "mean_fitness", "max_fitness"]
def _parse_bit_sequence(raw: str) -> tuple[int, ...]:
    """``"1,0,1,0"`` or ``"1010"`` -> ``(1, 0, 1, 0)``."""
    cleaned = raw.strip().replace(",", "").replace(" ", "")
    if not cleaned or any(ch not in "01" for ch in cleaned):
        raise SimConfigError(
            f"sim key 'target': expected a bit sequence like '1,0,1,0' "
            f"or '1010', got {raw!r}")
    return tuple(int(ch) for ch in cleaned)
_SPATIAL_EVO_DEFAULT_COLUMNS = [
    "generation", "mean_fitness", "max_fitness", "best_radius_sites",
    "best_survival", "mean_uptake_gain",
]
_STOCHASTIC_FANO_COLUMNS = [
    "mode", "fano", "on_fraction", "transcription_rate",
]
_STOCHASTIC_GILLESPIE_COLUMNS = [
    "mode", "mean", "variance", "fano", "analytic_fano",
]
_CODEC_DEFAULT_COLUMNS = [
    "scheme", "target_density", "achieved_density", "redundancy",
    "max_loss_fraction", "max_error_rate", "decode_time_s",
    "num_oligos", "total_bp", "cost_per_gb_usd",
]
_SYNBIO_DEFAULT_COLUMNS = [
    "protein", "orf_length", "full_length", "cai", "gc_content",
    "n_restriction_sites", "valid",
]
_PROTEIN_FITNESS_COLUMNS = ["rank", "variant", "score"]
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
_PROTEIN_STRUCTURE_COLUMNS = [
    "length", "helix_fraction", "sheet_fraction", "turn_fraction",
    "coil_fraction", "mean_hydropathy", "gravy",
    "n_transmembrane_helices", "is_membrane_protein",
    "disorder_fraction", "n_disorder_regions",
]
_FATE_DEFAULT_COLUMNS = ["w", "n_stable_states", "fate_a_level", "fate_b_level"]
_DIRECTED_EVO_COLUMNS = [
    "oracle", "initial_fitness", "guided_recovery", "guided_gain",
    "baseline_gain", "spearman_rho", "final_best_sequence",
]
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
_OMICS_CALIBRATION_COLUMNS = [
    "response_correlation", "de_sign_agreement",
    "mae_improvement_vs_baseline", "n_holdout", "passed",
]
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
_CODON_USAGE_COLUMNS = ["gene", "species", "cai", "orf_length", "protein"]
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

    from helixlang.plugins.apps.ecosystem import gem_to_species

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
                from helixlang.plugins.gem.bridge import build_functional_model_full
                from helixlang.plugins.runtime.metabolism import FluxBalanceAnalysis

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
                    except ValueError:  # SILENTBENIGN - keep default growth rate
                        pass
                sp.traits.max_growth_rate = max_mu
            except Exception:  # SILENTBENIGN - non-fatal; skip species row
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
            from helixlang.plugins.apps.gem_pipeline import run_gem_pipeline
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
                    except ValueError:  # SILENTBENIGN - keep default growth rate
                        pass
                sp.traits.max_growth_rate = max_mu
        except Exception:  # SILENTBENIGN - fall back to Monod if GEM fails
            pass  # non-fatal: fall back to Monod if GEM fails
        finally:
            if _tmp_fasta is not None:
                _tmp_fasta.unlink(missing_ok=True)
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
    from helixlang.plugins.runtime.metabolism import Reaction

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
    from helixlang.plugins.runtime.metabolism import Reaction

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
def _build_genotype_from_helix(
    genotype: GenotypeProfile, ext: dict[str, Any],
) -> None:
    """Populate genotype from #gene annotations stored in the `#gene`/genotype sections.

    Routes pharmacogenes to the correct status dict:
    - CYP enzymes (CYP2D6, CYP3A4, ...) → ``cyp_status`` via star-allele
    - Transporters (SLCO1B1, ABCB1, ...) → ``transporter_status``
    - Non-CYP enzymes (UGT1A1, TPMT, ...) → ``non_cyp_enzyme_status``
    - Other genes → ``gene_variants`` (generic storage)
    """
    from helixlang.plugins.human.genotype import (
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
        from helixlang.plugins.human.genotype import _resolve_star_allele

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
    from helixlang.plugins.human.phenotype import ExternalTraits as _Ext
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
    from helixlang.plugins.human.disease import (
        DiseaseState as _DS,
    )
    from helixlang.plugins.human.disease import (
        GenePerturbation as _GP,
    )
    from helixlang.plugins.human.disease import (
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
def _build_tumor_biopsy_from_helix(ext: dict[str, Any]) -> dict[str, Any] | None:
    """Build tumor biopsy dict from #tumor_biopsy annotation.

    Returns dict with keys: mutations, amplifications, deletions, fusion_genes,
    pd_l1_expression, msi_status, tmb_per_mb, hr_status.
    Returns None if no #tumor_biopsy annotation present.
    """
    raw = ext.get("tumor_biopsy")
    if not raw or not isinstance(raw, dict):
        return None

    def _split_csv(val: str) -> list[str]:
        if not val:
            return []
        return [v.strip() for v in val.split(",") if v.strip()]

    return {
        "mutations": _split_csv(raw.get("mutation", "")),
        "amplifications": _split_csv(raw.get("amplification", "")),
        "deletions": _split_csv(raw.get("deletion", "")),
        "fusion_genes": _split_csv(raw.get("fusion", "")),
        "pd_l1_expression": _opt_float(raw, "pd_l1_expression", 0.0),
        "msi_status": raw.get("msi_status", "MSS"),
        "tmb_per_mb": _opt_float(raw, "tmb_per_mb", 0.0),
        "hr_status": raw.get("hr_status", "HRC"),
    }
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
    from helixlang.plugins.human.drug import SMALL_MOLECULE, parse_drug_smiles, smiles_to_adme
    from helixlang.plugins.human.drug import Drug as _Drug
    from helixlang.plugins.human.drug import DrugMolecule as _DM

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
                except ValueError:  # SILENTBENIGN - skip non-numeric param
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
            except Exception:  # SILENTBENIGN - use hand-specified values
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
    from helixlang.plugins.human.pharmacodynamics import PDEffect as _PE
    from helixlang.plugins.human.pharmacodynamics import Pharmacodynamics as _PD
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
    sim = {**program.config.sim, **program.extensions}
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
    # #genome source=... (Design 5): fields merge into the population section under a
    # genome_ prefix (the same open extension point as #sim); build the
    # shared sparse template once per run.
    ext = program.extensions
    genome = None
    if ext.get("genome") or any(k.startswith("genome_") for k in ext):
        from helixlang.plugins.apps.genome_scale import build_genome
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
        from helixlang.plugins.apps.lattice_boltzmann_3d import LatticeBoltzmann3D

        return LatticeBoltzmann3D(
            width=width,
            height=height,
            depth=depth,
            omega=omega,
            inlet_velocity=inlet_velocity,
            inlet_density=inlet_density,
            outlet_density=outlet_density,
        )
    from helixlang.plugins.apps.lattice_boltzmann import LatticeBoltzmann

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
            from helixlang.plugins.runtime.cell_body import CellBody

            body = CellBody(
                x=(x + 0.5) * LATTICE_SPACING_UM,
                y=(y + 0.5) * LATTICE_SPACING_UM,
                length_um=config.cell_length_um,
                diameter_um=config.cell_diameter_um,
                angle=0.0,
            )
        cells.append(PopulationCell(id=i, x=x, y=y, z=z, body=body))
    return cells
