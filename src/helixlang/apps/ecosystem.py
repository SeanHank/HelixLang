"""Ecosystem runtime: multi-species, multi-population, multi-environment (doc/19 Phases A-D).

This is the ⑧-⑨-⑩ layer that closes the whole-organism loop: species
identity, patches (habitats) connected by dispersal, inter-species
interactions (competition, cross-feeding, predation, metabolic
switching with a cost), decomposition and nutrient cycles, environmental
drivers (temperature/light/pH/toxin via ``ScalarField``), a community
(FBA) objective, an event scheduler for year-scale runs, ecosystem
analytics, and an invasion-fitness evolution loop.

Method -> design-element mapping (doc/19-whole-organism-lifecycle-simulation.md §5.2):

- L1  particle-based species identity       -> ``Species`` + per-species traits
- L2  condition-dependent switching cost    -> ``switching_cost`` + mode recovery
- L3  OptCom community objective            -> ``CommunityFBA`` (inner LPs / outer goal)
- L5  neutral drift vs. niche partitioning  -> per-species uptake coefficients +
                                              ``neutral_vs_niche`` diagnostic
- L6  predation, Lotka-Volterra validation  -> ``lotka_volterra_step`` +
                                              ``lotka_volterra_conserved``
- L7  Levins metapopulation / source-sink   -> ``Metapopulation`` + dispersal
- L8  eco-evolutionary feedbacks            -> invasion-fitness outer loop
- L10 Q10 / Arrhenius (DAMM) temperature    -> ``q10_rate_modifier`` / ``damm_rate``
- L11 CENTURY pool decomposition            -> ``CenturyPools`` (Sierra 2012 k)
- L12 DAYCENT daily N-gas fluxes            -> ``NitrogenCycle``
- L13 SoilR reference parameters            -> ``CenturyPools`` weekly k

Units follow the rest of the runtime: one tick = 1 min, one lattice site
= (10 µm)^3 = 1e-12 L, substrate fields in mM, and biomass in **mmol of
carbon** so the C/N budgets close exactly (a sealed microcosm conserves
its total carbon and nitrogen to numerical tolerance).
"""
from __future__ import annotations

import copy
import math
import random
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from helixlang.environment import (
    ConcentrationField,
    DiurnalForcing,
    ScalarField,
    monod_uptake,
    photosynthesis_rate,
    q10_rate_modifier,
)
from helixlang.evolution import mutate
from helixlang.metabolism import simplex
from helixlang.units import TIME_TICK_MIN

# ============================================================================
# Genotype -> phenotype (A4: continuous trait axes, Ferriere & Legendre 2013)
# ============================================================================

# Conversion factor: FBA fluxes are in mmol/gDW/h (or 1/h for biomass).
# The ecosystem ticks are minutes (TIME_TICK_MIN == 1.0), so FBA rates
# must be scaled by _H_PER_TICK to match per-tick conventions.
_H_PER_TICK: float = TIME_TICK_MIN / 60.0

_NT_BITS: dict[str, tuple[int, int]] = {
    "A": (0, 0), "C": (0, 1), "G": (1, 0), "T": (1, 1),
    "a": (0, 0), "c": (0, 1), "g": (1, 0), "t": (1, 1),
}


def _bit_vector(genome: str) -> list[int]:
    out: list[int] = []
    for ch in genome:
        b = _NT_BITS[ch]
        out.extend(b)
    return out


def _bit_mean(bits: list[int], lo: float, hi: float) -> float:
    if not bits:
        return 1.0
    return lo + (hi - lo) * sum(bits) / len(bits)


@dataclass(slots=True)
class SpeciesTraitParams:
    """Continuous functional traits decoded from a genotype (Fisher 2021).

    Traits are the axes of eco-evolutionary selection (A4): a genotype
    with a different trait vector experiences a different selective
    environment, so a change in the environment changes which genome
    is fittest -- the niche-construction arm of the loop.
    """

    uptake_gain: float = 1.0        # multiplier on Monod uptake (0.5..1.5)
    growth_rate_gain: float = 1.0   # multiplier on yield growth (0.7..1.3)
    yield_c: float = 0.5            # biomass-C per substrate-C (0.3..0.7)
    tolerance: float = 0.5          # stress (toxin/pH/heat) tolerance (0..1)
    q10: float = 2.0                # Q10 temperature sensitivity (1.5..3.0)
    switching_cost: float = 0.05    # growth penalty for a metabolic switch (0..0.5)
    max_growth_rate: float = 2.0    # h^-1; organism-specific cap (doc/20 §15)

    def from_genome(self, genome: str) -> SpeciesTraitParams:
        bits = _bit_vector(genome)
        n = len(bits)
        if n == 0:
            return self
        sixth = max(1, n // 6)
        self.uptake_gain = _bit_mean(bits[0:sixth], 0.5, 1.5)
        self.growth_rate_gain = _bit_mean(bits[sixth:2 * sixth], 0.7, 1.3)
        self.yield_c = _bit_mean(bits[2 * sixth:3 * sixth], 0.3, 0.7)
        self.tolerance = _bit_mean(bits[3 * sixth:4 * sixth], 0.0, 1.0)
        self.q10 = _bit_mean(bits[4 * sixth:5 * sixth], 1.5, 3.0)
        self.switching_cost = _bit_mean(bits[5 * sixth:], 0.0, 0.5)
        return self


# ============================================================================
# Species + patch configuration
# ============================================================================

@dataclass(slots=True)
class Species:
    """A species: genome + interaction table + decoded traits (L1, L2, L3).

    Args:
        name: species key used by patches and observations.
        genome: DNA genotype; traits are decoded from it.
        consumption: ``substrate -> (vmax, Ks)`` in mM; the growth
            substrate(s) with per-species uptake coefficients (niche
            separation, L5).
        secretion: ``substrate -> mM per unit biomass C per tick``
            (cross-feeding / syntrophy, L3).
        diet: ``prey species -> conversion efficiency`` (predation, L6).
        attack_rate: ``prey species -> mass-action attack rate``
            (1/(biomass C * tick)).
        photo: light-gated photoautotrophy (B3).
        photo_vmax: max CO2 fixation rate (biomass-C per biomass-C per
            tick at saturating light).
        cn_ratio: biomass C:N ratio used by the N cycle (CENTURY, L11).
        maintenance: per-tick biomass fraction respired (maintenance +
            natural mortality).
        traits: continuous functional traits (A4).
        metabolic_model: optional GEM from doc/20 pipeline; when set
            and ``gem_driven=True``, growth uses FBA instead of Monod.
        gem_fluxes: FBA optimal fluxes (from GEM pipeline Stage 6).
        gem_kcat: per-reaction kcat predictions.
        gem_km: per-reaction Km estimates.
    """

    name: str
    genome: str = ""
    consumption: dict[str, tuple[float, float]] = field(default_factory=dict)
    secretion: dict[str, float] = field(default_factory=dict)
    diet: dict[str, float] = field(default_factory=dict)
    attack_rate: dict[str, float] = field(default_factory=dict)
    photo: bool = False
    photo_vmax: float = 0.01
    cn_ratio: float = 6.0
    maintenance: float = 0.001
    traits: SpeciesTraitParams = field(default_factory=SpeciesTraitParams)
    metabolic_model: object | None = field(default=None, repr=False)
    gem_fluxes: dict[str, float] = field(default_factory=dict)
    gem_kcat: dict[str, float] = field(default_factory=dict)
    gem_km: dict[str, float] = field(default_factory=dict)
    last_fba_fluxes: dict[str, float] = field(default_factory=dict)
    grn_edges: list = field(default_factory=list)
    grn_gpr_map: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.genome:
            self.traits = SpeciesTraitParams().from_genome(self.genome)

    def preferred_substrate(self) -> str | None:
        if self.consumption:
            return next(iter(self.consumption))
        return None


# ============================================================================
# GEM → Species bridge (doc/21 §3.1)
# ============================================================================

# Mapping of GEM exchange reaction prefixes to ecosystem substrate names
_EXCHANGE_SUBSTRATE_MAP: dict[str, str] = {
    "EX_glc__D_e": "glucose", "EX_glc_e": "glucose",
    "EX_ac_e": "acetate", "EX_acald_e": "acetaldehyde",
    "EX_etoh_e": "ethanol", "EX_lac__D_e": "lactate",
    "EX_pyr_e": "pyruvate", "EX_succ_e": "succinate",
    "EX_fum_e": "fumarate", "EX_mal__L_e": "malate",
    "EX_cit_e": "citrate", "EX_for_e": "formate",
    "EX_co2_e": "co2", "EX_o2_e": "oxygen",
    "EX_nh4_e": "ammonia", "EX_no3_e": "nitrate",
    "EX_pi_e": "phosphate",
}

# Reverse mapping: ecosystem substrate -> GEM exchange metabolite
_ECOSYSTEM_TO_GEM_EXCHANGE: dict[str, str] = {
    # Keys must use the BiGG double-underscore convention so that
    # FluxBalanceAnalysis.set_uptake / _build_and_solve can match
    # metabolites in the model's stoichiometry dict.
    "glucose": "glc__D_e", "acetate": "ac_e", "oxygen": "o2_e",
    "co2": "co2_e", "ammonia": "nh4_e", "nitrate": "no3_e",
    "phosphate": "pi_e",
    "ethanol": "etoh_e", "lactate": "lac__D_e", "pyruvate": "pyr_e",
    "succinate": "succ_e", "formate": "for_e", "citrate": "cit_e",
}


def gem_to_species(
    pipeline_result: object,
    organism: str = "e_coli_k12",
    medium: str = "glucose_minimal",
) -> dict[str, Any]:
    """Extract ecosystem-compatible parameters from a GEM pipeline result.

    Reads FBA fluxes, kcat predictions, and Km estimates from a
    ``GemPipelineResult`` and returns a dict suitable for populating a
    ``Species`` via ``Species.from_gem_params()``.

    Returns dict with keys: vmax, ks, yield_c, secretion, cn_ratio,
    maintenance, max_growth_rate.
    """

    fluxes: dict[str, float] = {}
    kcat_map: dict[str, float] = {}
    km_map: dict[str, float] = {}

    # Extract from GemPipelineResult attributes
    if hasattr(pipeline_result, "fba_fluxes"):
        fluxes = dict(pipeline_result.fba_fluxes)
    if hasattr(pipeline_result, "kcat_predictions"):
        for kp in pipeline_result.kcat_predictions:
            if hasattr(kp, "reaction_id") and hasattr(kp, "kcat"):
                kcat_map[kp.reaction_id] = kp.kcat
    if hasattr(pipeline_result, "km_estimates"):
        km_map = dict(pipeline_result.km_estimates)

    # Determine vmax from glucose exchange flux
    vmax = 0.0
    ks = 0.1
    for ex_id, substrate in _EXCHANGE_SUBSTRATE_MAP.items():
        if substrate == "glucose" and ex_id in fluxes:
            # FBA flux is negative for uptake; take absolute value
            vmax = abs(fluxes[ex_id])
            break
    # Also check metabolite-based naming
    if vmax == 0.0:
        for rxn_id, flux in fluxes.items():
            if "glc" in rxn_id.lower() and flux < 0:
                vmax = abs(flux)
                break
    if vmax == 0.0:
        vmax = 0.02  # fallback default

    # Estimate Km from the glucose exchange reaction
    for rxn_id, km_val in km_map.items():
        if "glc" in rxn_id.lower():
            ks = km_val
            break

    # Compute yield from biomass flux / glucose uptake
    growth_rate = 0.0
    if hasattr(pipeline_result, "growth_rate"):
        growth_rate = pipeline_result.growth_rate
    yield_c = 0.5  # default
    if vmax > 0 and growth_rate > 0:
        # growth_rate is in 1/h; vmax is mmol/gDW/h
        # yield = growth_rate / (vmax * carbon_per_substrate)
        # For glucose (6C): yield = mu / (vmax * 6)
        yield_c = min(0.7, max(0.1, growth_rate / (vmax * 6.0)))

    # Detect secretion (non-zero export fluxes)
    secretion: dict[str, float] = {}
    for rxn_id, flux in fluxes.items():
        if flux > 1e-6:  # positive flux = export
            substrate = _EXCHANGE_SUBSTRATE_MAP.get(rxn_id, "")
            if substrate and substrate not in ("co2", "oxygen",
                                                "ammonia", "phosphate"):
                secretion[substrate] = flux * 0.01  # scale to per-biomass-C

    # Estimate cn_ratio from biomass reaction if available
    cn_ratio = 6.0
    if hasattr(pipeline_result, "biomass_reaction"):
        bm = pipeline_result.biomass_reaction
        if bm is not None and hasattr(bm, "components"):
            c_total = 0.0
            n_total = 0.0
            for comp in bm.components:
                met = comp.metabolite_id.lower() if hasattr(comp, "metabolite_id") else ""
                coeff = comp.coefficient if hasattr(comp, "coefficient") else 0.0
                if coeff < 0:  # consumed
                    if any(m in met for m in ("atp", "nad", "nadp", "coa")):
                        continue  # skip cofactors
                    if "c" in met and "n" not in met:
                        c_total += abs(coeff)
                    elif "n" in met:
                        n_total += abs(coeff)
            if n_total > 0:
                cn_ratio = max(3.0, min(15.0, c_total / n_total))

    # Maintenance from ATPM reaction
    maintenance = 0.001
    if "ATPM" in fluxes and fluxes["ATPM"] > 0:
        # ATPM flux is mmol/gDW/h; convert to per-tick fraction
        # assuming ~10 mmol ATP/gDW/h maintenance ≈ 0.001 per tick
        maintenance = min(0.01, fluxes["ATPM"] * 0.0001)

    # Organism-specific growth rate caps (h^-1, literature values)
    _ORG_MAX_GROWTH: dict[str, float] = {
        "e_coli": 0.87, "e_coli_k12": 0.87,
        "synechocystis": 0.14, "synechocystis_pcc6803": 0.14,
        "b_subtilis": 0.58, "s_cerevisiae": 0.56, "s_aureus": 0.69,
        "m_tuberculosis": 0.03, "p_aeruginosa": 0.53,
        "k_pneumoniae": 0.53, "l_lactis": 0.57,
    }
    _org_key = organism.lower().replace(" ", "_").replace(".", "")
    max_mu = _ORG_MAX_GROWTH.get(_org_key, _ORG_MAX_GROWTH.get("e_coli", 0.87))

    return {
        "vmax": vmax,
        "ks": ks,
        "yield_c": yield_c,
        "secretion": secretion,
        "cn_ratio": cn_ratio,
        "maintenance": maintenance,
        "max_growth_rate": max_mu,
    }


#: carbon atoms per molecule for common substrates (used for the C
#: budget and growth bookkeeping; O2 carries no carbon)
_CARBON_PER_MOL: dict[str, int] = {
    "co2": 1, "oxygen": 0, "acetate": 2, "ethanol": 2, "methanol": 1,
    "pyruvate": 3, "citrate": 6, "succinate": 4, "formate": 1,
    "glucose": 6, "fructose": 6, "lactate": 3, "ammonia": 0, "nh4": 0,
}


def default_carbon_per_mol(substrate: str) -> int:
    """Carbon atoms per molecule of ``substrate`` (6 for unknown)."""
    return _CARBON_PER_MOL.get(substrate.lower(), 6)


@dataclass(slots=True)
class SubstrateConfig:
    """Initial substrate field on a patch.

    ``carbon_per_mol`` defaults to 0, meaning "use the per-substrate
    chemical default" (:func:`default_carbon_per_mol`: co2=1, acetate=2,
    glucose=6, ...); set it explicitly to override.
    """

    initial_mm: float
    bulk_mm: float | None = None
    diffusion_um2_s: float = 300.0
    carbon_per_mol: int = 0

@dataclass(slots=True)
class ScalarConfig:
    """Scalar driver (temperature/light/pH/toxin) on a patch (B1)."""

    kind: str = "temperature"
    initial: float = 25.0
    forcing: Callable[[int], float] | None = None
    diffusion_um2_s: float = 0.0


@dataclass(slots=True)
class PatchConfig:
    """A habitat (chemostat, biofilm, water column, sediment, soil).

    Args:
        name: patch key; ``dispersal`` edges reference neighbor names.
        kind: habitat kind (drives default fields / flow).
        width, height: lattice sites (well-mixed dynamics use 1x1).
        carrying_capacity: total biomass-C (mmol) density cap.
        anoxic: no initial oxygen.
        moisture: water-filled pore-space fraction (soil; DAMM, L10).
        clay: clay fraction slowing active->slow SOM transfer (CENTURY).
        flow_rate: chemostat refresh (0 = sealed/batch).
        temperature_c: environmental temperature (°C, default 25).
        ph: environmental pH (default 7.0).
        initial_biomass: ``species -> biomass-C (mmol)``.
        substrates: ``name -> SubstrateConfig``.
        scalars: ``name -> ScalarConfig``.
        dispersal: ``neighbor patch -> per-tick migration fraction``.
        fluctuation_period, fluctuation_amplitude: periodic substrate
            supply forcing (L2; 0 disables).
    """

    name: str = "water"
    kind: str = "water"
    width: int = 1
    height: int = 1
    carrying_capacity: float = 1e5
    anoxic: bool = False
    moisture: float = 1.0
    clay: float = 0.2
    cn_som: float = 12.0
    cn_species: float = 6.0
    initial_nh4_mm: float = 0.0
    initial_no3_mm: float = 0.0
    flow_rate: float = 0.0
    temperature_c: float = 25.0
    ph: float = 7.0
    initial_biomass: dict[str, float] = field(default_factory=dict)
    substrates: dict[str, SubstrateConfig] = field(default_factory=dict)
    scalars: dict[str, ScalarConfig] = field(default_factory=dict)
    dispersal: dict[str, float] = field(default_factory=dict)
    fluctuation_period: int = 0
    fluctuation_amplitude: float = 0.0


@dataclass(slots=True)
class EcosystemConfig:
    """Ecosystem run configuration.

    Args:
        ticks: total simulation minutes to run.
        seed: master RNG seed.
        fast_forward: enable the event scheduler (D1).
        scheduler_max_step: maximum tick jump at equilibrium.
        scheduler_change_threshold: per-tick state delta treated as
            quiescent (equilibrium -> skip).
        community_fba: reconcile per-species uptake with an OptCom
            community objective each tick (L3).
        sample_every: record an observation row every N ticks.
        species: list of :class:`Species`.
        patches: list of :class:`PatchConfig`.
        generations, population_size, substitution_rate, indel_rate,
            recombination_rate, genome_length_nt, evaluation_ticks:
            outer-loop invasion-fitness evolution (A4, L8).
        evolution_enabled: run the outer loop in ``run_generations``.
        stress_field: patch scalar name used to gate stress tolerance
            selection.
        stress_level: applied toxin/heat stress for selection tests.
        gem_driven: when True, species with a ``metabolic_model`` use
            FBA-backed growth instead of Monod kinetics.
    """

    ticks: int = 1000
    seed: int | None = None
    fast_forward: bool = True
    scheduler_max_step: int = 480
    scheduler_change_threshold: float = 1e-4
    community_fba: bool = False
    gem_driven: bool = False
    sample_every: int = 1
    species: list[Species] = field(default_factory=list)
    patches: list[PatchConfig] = field(default_factory=list)
    generations: int = 1
    population_size: int = 6
    substitution_rate: float = 0.05
    indel_rate: float = 0.0
    recombination_rate: float = 0.0
    genome_length_nt: int = 36
    evaluation_ticks: int = 200
    evolution_enabled: bool = False
    stress_field: str = "toxin"
    stress_level: float = 0.0


# ============================================================================
# Lotka-Volterra predation (L6) -- analytic validation target
# ============================================================================

def lotka_volterra_step(prey: float, predator: float,
                        alpha: float, beta: float,
                        delta: float, gamma: float,
                        dt: float = 1.0,
                        substeps: int = 16) -> tuple[float, float]:
    """One predator-prey update (Euler with ``substeps`` inner steps).

    The continuous Lotka-Volterra system (Volterra 1926; Lotka 1925)::

        dx/dt = alpha*x - beta*x*y     (prey)
        dy/dt = delta*x*y - gamma*y    (predator)

    Integrated on the tick grid with ``substeps`` Euler sub-steps.  Near
    the centre ``(gamma/delta, alpha/beta)`` orbits are ellipses with
    period ``T ~= 2*pi/sqrt(alpha*gamma)`` (Hsu 1983) and the quantity
    ``V = delta*x - gamma*ln x + beta*y - alpha*ln y`` is conserved on
    closed orbits -- both are checked by tests.
    """
    dt_sub = dt / substeps
    x, y = prey, predator
    for _ in range(substeps):
        x += (alpha * x - beta * x * y) * dt_sub
        y += (delta * x * y - gamma * y) * dt_sub
        x = max(x, 0.0)
        y = max(y, 0.0)
    return x, y


def lotka_volterra_conserved(prey: float, predator: float,
                             alpha: float, beta: float,
                             delta: float, gamma: float) -> float:
    """The conserved quantity ``V = delta*x - gamma*ln x + beta*y -
    alpha*ln y`` (Volterra 1926)."""
    return (delta * prey - gamma * math.log(prey)
            + beta * predator - alpha * math.log(predator))


# ============================================================================
# Levins metapopulation + source-sink (L7)
# ============================================================================

class Metapopulation:
    """Stochastic Levins metapopulation emerging from per-patch dispersal.

    A graph of ``n`` patches, each either occupied by a single species or
    empty.  Per tick every occupied patch colonizes each neighboring
    empty patch with probability ``m`` (dispersal) and goes extinct with
    probability ``e``.  The steady-state occupancy fraction converges to
    the Levins equilibrium ``p* = 1 - e/m`` (Levins 1969); ``m <= e``
    drives global extinction.  The agent-level loop is per-cell
    dispersal (a colonist is a cell that migrates along a dispersal edge
    and establishes), so the ODE equilibrium is *emergent*, not imposed.
    """

    def __init__(self, n_patches: int, colonization_rate: float,
                 extinction_rate: float, seed: int | None = None,
                 graph: list[list[int]] | None = None,
                 initial_fraction: float = 0.5) -> None:
        if n_patches <= 0:
            raise ValueError("n_patches must be > 0")
        if not 0.0 <= extinction_rate <= 1.0:
            raise ValueError("extinction_rate must be in [0, 1]")
        if not 0.0 <= colonization_rate <= 1.0:
            raise ValueError("colonization_rate must be in [0, 1]")
        self.n = n_patches
        self.m = colonization_rate
        self.e = extinction_rate
        self.rng = random.Random(seed)
        # graph=None -> classic well-mixed Levins; a graph gives spatial
        # (per-neighbor) dispersal
        self.graph = graph
        self.occupied = [
            self.rng.random() < initial_fraction for _ in range(n_patches)]
        self.tick = 0
        self.history: list[float] = [self.occupancy()]

    def occupancy(self) -> float:
        return sum(self.occupied) / self.n

    def step(self) -> float:
        """One tick: colonization of empty patches, then extinction.

        The default (``graph=None``) uses the classic Levins mass-action
        colonization ``m * occupancy``: each empty patch is colonized
        with probability ``m*P`` and each occupied patch goes extinct
        with probability ``e``, whose steady state is exactly
        ``P* = 1 - e/m`` (Levins 1969).  When a ``graph`` was supplied
        (spatial dispersal), colonization is instead per-neighbor: each
        occupied neighbor seeds an empty patch with probability ``m``.
        """
        occ = self.occupancy()
        if self.graph is not None:
            for i in range(self.n):
                if (not self.occupied[i]
                        and any(self.occupied[j] for j in self.graph[i])
                        and self.rng.random() < self.m):
                    self.occupied[i] = True
        else:
            for i in range(self.n):
                if (not self.occupied[i]
                        and self.rng.random() < self.m * occ):
                    self.occupied[i] = True
        for i in range(self.n):
            if self.occupied[i] and self.rng.random() < self.e:
                self.occupied[i] = False
        self.tick += 1
        occ = self.occupancy()
        self.history.append(occ)
        return occ

    def run(self, ticks: int) -> list[float]:
        for _ in range(ticks):
            self.step()
        return self.history

    def levins_equilibrium(self) -> float:
        """The analytic steady state ``p* = 1 - e/m`` (Levins 1969)."""
        if self.m <= 0.0:
            return 0.0
        return max(0.0, 1.0 - self.e / self.m)


def source_sink_equilibrium(immigration: float, death_rate: float,
                            emigration: float) -> float:
    """Sink equilibrium abundance ``i / (d + x)`` (Pulliam 1988).

    A sink patch has negative intrinsic growth; its steady-state
    abundance is sustained by immigration ``i`` balanced against
    mortality ``d`` (plus emigration).  The source patch maintains the
    sink through this immigration flux.
    """
    if death_rate + emigration <= 0.0:
        raise ValueError("death_rate + emigration must be > 0")
    return immigration / (death_rate + emigration)


# ============================================================================
# CENTURY decomposition pools (L11/L13, Parton 1987, Sierra 2012)
# ============================================================================

#: reference first-order decay constants per week (Sierra et al. 2012,
#: SoilR CenturyModel): structural/metabolic litter, active/slow/passive
#: SOM.  k_week -> per-minute constant ``1 - (1-k)^(1/10080)``.
_CENTURY_K_WEEK: dict[str, float] = {
    "structural": 0.076,   # STR.surface
    "metabolic": 0.28,     # MET.surface
    "active": 0.14,        # ACT
    "slow": 0.0038,        # SLW
    "passive": 0.00013,    # PAS
}

_WEEK_MIN = 7 * 1440


def century_k_per_min(k_week: float) -> float:
    """Per-minute first-order constant for a per-week CENTURY rate."""
    if k_week >= 1.0:
        return 1.0
    return float(1.0 - (1.0 - k_week) ** (1.0 / _WEEK_MIN))


#: default CENTURY transfer fractions (structural/metabolic -> SOM)
_CENTURY_TRANSFERS: dict[str, dict[str, float]] = {
    "structural": {"active": 0.35, "slow": 0.20, "respired": 0.45},
    "metabolic": {"active": 0.45, "respired": 0.55},
    "active": {"slow_frac": 0.40},
    "slow": {"passive_frac": 0.45},
    "passive": {"respired": 1.0},
}


class CenturyPools:
    """First-order pool decomposition with the CENTURY structure (L11).

    Structural vs. metabolic litter is partitioned by a ``lignin`` proxy
    (lignin:N, Parton 1987); litter feeds the active/slow/passive SOM
    cascade.  Every pool decays first order with the Sierra et al. 2012
    per-week constants, modulated by temperature (Q10, L10), moisture
    and clay.  The carbon budget is tracked exactly::

        total_C = litter_in - respired_C  ==  sum(pools)

    ``step`` advances over ``ticks`` minutes with the exact exponential
    form ``exp(-k*t)``, so fast-forwarded and tick-by-tick runs agree.
    """

    def __init__(self, clay: float = 0.2, lignin: float = 0.2) -> None:
        self.clay = float(clay)
        self.lignin = float(lignin)
        self.pools: dict[str, float] = {p: 0.0 for p in _CENTURY_K_WEEK}
        self.litter_in_c: float = 0.0
        self.respired_c: float = 0.0
        self.k_min: dict[str, float] = {
            p: century_k_per_min(k) for p, k in _CENTURY_K_WEEK.items()}

    def add_litter(self, carbon_mmol: float) -> None:
        """Feed dead biomass carbon into structural/metabolic litter."""
        if carbon_mmol <= 0.0:
            return
        self.litter_in_c += carbon_mmol
        self.pools["structural"] += carbon_mmol * self.lignin
        self.pools["metabolic"] += carbon_mmol * (1.0 - self.lignin)

    def _modifier(self, t_mod: float, moisture: float) -> float:
        # Q10 temperature term (L10) * moisture gate (DAMM)
        return max(0.0, t_mod) * max(0.0, min(1.0, moisture))

    def step(self, ticks: int, t_mod: float, moisture: float) -> None:
        """Advance the pools by ``ticks`` minutes (exact exponential)."""
        if ticks <= 0:
            return
        mod = self._modifier(t_mod, moisture)
        resp = 0.0
        pools = self.pools
        transfers: dict[str, float] = {}
        for pool, k in self.k_min.items():
            amount = pools[pool]
            if amount <= 0.0:
                continue
            decayed = amount * (1.0 - math.exp(-k * mod * ticks))
            pools[pool] = amount - decayed
            t = _CENTURY_TRANSFERS[pool]
            if pool == "active":
                to_slow = decayed * t["slow_frac"] * (1.0 - 0.75 * self.clay)
                transfers["slow"] = transfers.get("slow", 0.0) + to_slow
                resp += decayed - to_slow
            elif pool == "slow":
                to_passive = decayed * t["passive_frac"]
                transfers["passive"] = transfers.get("passive", 0.0) + to_passive
                resp += decayed - to_passive
            else:
                for dst, frac in t.items():
                    if dst == "respired":
                        resp += decayed * frac
                    elif dst in pools:
                        transfers[dst] = transfers.get(dst, 0.0) + decayed * frac
        for dst, amount in transfers.items():
            pools[dst] += amount
        self.respired_c += resp

    def total(self) -> float:
        return sum(self.pools.values())

    def carbon_balance(self) -> float:
        """``litter_in - respired - total`` (== 0 to fp tolerance)."""
        return self.litter_in_c - self.respired_c - self.total()

    def equilibrium(self, litter_rate_per_tick: float,
                    t_mod: float = 1.0, moisture: float = 1.0) -> dict[str, float]:
        """Closed-form steady-state pools for constant litter input
        (Bolker et al. 1998 linear analysis).

        At steady state every pool satisfies ``input_i - k_i*C_i = 0``
        (transfers counted in the input).  Structural/metabolic pools
        receive the litter split directly; active/slow/passive follow
        the transfer fractions.  Returns the analytic pool vector used
        as the long-run validation target.
        """
        mod = self._modifier(t_mod, moisture)
        k = self.k_min
        lit_s = litter_rate_per_tick * self.lignin
        lit_m = litter_rate_per_tick * (1.0 - self.lignin)
        structural_eq = lit_s / max(k["structural"] * mod, 1e-30)
        metabolic_eq = lit_m / max(k["metabolic"] * mod, 1e-30)
        in_active = (metabolic_eq * k["metabolic"] * mod * 0.45
                     + structural_eq * k["structural"] * mod * 0.35)
        active_eq = in_active / max(k["active"] * mod, 1e-30)
        in_slow = (structural_eq * k["structural"] * mod * 0.20
                   + active_eq * k["active"] * mod
                   * 0.40 * (1.0 - 0.75 * self.clay))
        slow_eq = in_slow / max(k["slow"] * mod, 1e-30)
        in_passive = slow_eq * k["slow"] * mod * 0.45
        passive_eq = in_passive / max(k["passive"] * mod, 1e-30)
        return {
            "structural": structural_eq,
            "metabolic": metabolic_eq,
            "active": active_eq,
            "slow": slow_eq,
            "passive": passive_eq,
        }


# ============================================================================
# Nitrogen cycle (L12, DAYCENT-inspired: nitrification / denitrification)
# ============================================================================

class NitrogenCycle:
    """N submodel (Parton et al. 1994 DAYCENT, reduced).

    Ties N flow to C flow through C:N ratios: decomposed SOM C releases
    mineral N; microbial growth immobilizes N (NH4 first, then NO3);
    nitrification oxidizes NH4 -> NO3; anoxic denitrification reduces
    NO3 -> N2 gas; fixation adds N.  Growth is capped by the available
    mineral N in :meth:`Patch._step_populations`, so a sealed microcosm
    (``cn_som == cn_litter == species.cn_ratio``, no fixation, no
    denitrification) closes its N budget exactly.
    """

    def __init__(self, cn_som: float = 12.0,
                 cn_litter: float = 25.0,
                 nitrification_rate: float = 1.0e-4,
                 denitrification_rate: float = 1.0e-4) -> None:
        self.cn_som = float(cn_som)
        self.cn_litter = float(cn_litter)
        self.k_nitrif = float(nitrification_rate)
        self.k_denitrif = float(denitrification_rate)
        self.nh4_mm: float = 0.0
        self.no3_mm: float = 0.0
        self.ch4_mm: float = 0.0
        self.fixed_n: float = 0.0
        self.gas_n2: float = 0.0
        self.mineralized_n: float = 0.0
        self.immobilized_n: float = 0.0

    def available_n(self) -> float:
        """Mineral N available for immobilization (NH4 + NO3)."""
        return self.nh4_mm + self.no3_mm

    def immobilize(self, amount_n: float) -> None:
        """Draw ``amount_n`` from NH4 then NO3 (immobilization)."""
        if amount_n <= 0.0:
            return
        take = min(self.nh4_mm, amount_n)
        self.nh4_mm -= take
        rem = amount_n - take
        if rem > 0.0:
            take2 = min(self.no3_mm, rem)
            self.no3_mm -= take2
        self.immobilized_n += amount_n

    def excrete(self, amount_n: float) -> None:
        """Return nitrogen carried by respired consumption to NH4
        (predator/consumer excretion closes the trophic N loop)."""
        if amount_n > 0.0:
            self.nh4_mm += amount_n

    def step(self, decayed_c_mmol: float,
             ticks: int, t_mod: float,
             moisture: float, anoxic: bool) -> None:
        """Advance N pools by ``ticks`` minutes."""
        if ticks <= 0:
            return
        dt = float(ticks)
        # mineralization (net): decayed SOM C / C:N -> NH4
        mineralized = decayed_c_mmol / max(self.cn_som, 1e-30)
        self.nh4_mm += mineralized
        self.mineralized_n += mineralized
        # nitrification: NH4 -> NO3 (temperature + moisture modulated)
        nitr = self.nh4_mm * (1.0 - math.exp(-self.k_nitrif * t_mod
                                             * moisture * dt))
        self.nh4_mm -= nitr
        self.no3_mm += nitr
        # denitrification: NO3 -> N2 gas (requires anoxia)
        if anoxic and self.no3_mm > 0.0:
            denit = self.no3_mm * (1.0 - math.exp(-self.k_denitrif
                                                  * moisture * dt))
            self.no3_mm -= denit
            self.gas_n2 += denit
        # CH4 oxidation (methanotrophy) is a first-order sink
        if self.ch4_mm > 0.0:
            self.ch4_mm *= math.exp(-1.0e-3 * dt)

    def fix_n(self, amount_mm: float) -> None:
        """Biological N fixation into NH4 (legumes / diazotrophs)."""
        self.nh4_mm += amount_mm
        self.fixed_n += amount_mm

    def budget(self) -> float:
        """Sealed N budget check (== 0 when closed): mineralized N should
        equal immobilization + gas + current mineral pools."""
        return (self.mineralized_n
                - (self.immobilized_n + self.nh4_mm + self.no3_mm
                   + self.gas_n2))


# ============================================================================
# Community FBA (L3, OptCom: inner species LPs, outer community objective)
# ============================================================================

class CommunityFBA:
    """OptCom-style two-level community objective (Zomorrodi & Maranas 2012).

    The inner level is each species' substrate demand (its maximal
    uptake under the shared resource pool); the outer level maximizes a
    community goal -- total community biomass growth -- subject to the
    shared substrate budget::

        max  sum_i  w_i * yield_i * demand_i * u_i
        s.t. sum_i  demand_i * u_i <= budget,    0 <= u_i <= 1

    ``u_i`` is the fraction of species i's maximal demand the community
    can grant; it becomes the uptake/deposit bound reconciling the
    agent loop with the community metabolic optimum each tick.  Solved
    with the pure-Python simplex (:func:`helixlang.metabolism.simplex`),
    so no scipy dependency.
    """

    @staticmethod
    def solve(yields: list[float], demands: list[float],
              budget: float,
              weights: list[float] | None = None) -> dict:
        """Community-optimal uptake fractions.

        Args:
            yields: biomass-C per substrate-C per species.
            demands: maximal substrate draw per species (mM-equivalent).
            budget: shared substrate available this tick.
            weights: species weights in the community objective.

        Returns the simplex result with ``x`` = uptake fractions ``u_i``.
        """
        n = len(yields)
        if n == 0:
            return {"status": "optimal", "x": [], "objective": 0.0}
        if len(demands) != n:
            raise ValueError("yields and demands must have equal length")
        w = weights or [1.0] * n
        c = [w[i] * yields[i] * demands[i] for i in range(n)]
        A = [[demands[i] for i in range(n)]]
        b = [budget]
        bounds = [(0.0, 1.0)] * n
        return simplex(c, A, b, bounds, maximize=True)


# ============================================================================
# Scheduler (D1, G7): advance quiescent epochs exactly
# ============================================================================

@dataclass(slots=True)
class Scheduler:
    """Event scheduler: skip ticks while the state is at a fixed point.

    ``next_advance`` returns the number of ticks to advance: ``max_step``
    when every monitored state delta is below ``change_threshold`` and no
    forcing driver is changing (the system is at equilibrium, so the
    skipped ticks provably change nothing), otherwise 1.  Skipping is
    therefore *tick-equivalent* by construction: at equilibrium the
    per-tick step is a no-op.  Slow pool dynamics (CENTURY) are advanced
    with their exact exponential form over the jumped window.
    """

    max_step: int = 480
    change_threshold: float = 1e-4

    def next_advance(self, deltas: list[float],
                     forcing_delta: float = 0.0) -> int:
        if forcing_delta >= self.change_threshold:
            return 1
        if deltas and all(d < self.change_threshold for d in deltas):
            return max(1, self.max_step)
        return 1


# ============================================================================
# Patch: one habitat with its own fields, biomass and pools
# ============================================================================

class Patch:
    """A spatial habitat: substrate/scalar fields + per-species biomass.

    Each site is a (10 µm)^3 volume; growth consumes substrate at the
    local site (Monod, per-species uptake coefficients), and the field
    diffuses/advects.  Well-mixed patches are 1x1 (chemostat); spatial
    patches (biofilm/water/sediment) carry gradients that segregate
    niches.  Dead biomass feeds the CENTURY pools, which re-mineralize
    nutrients -- closing the C loop inside the patch.
    """

    def __init__(self, config: PatchConfig,
                 species: dict[str, Species],
                 gem_driven: bool = False) -> None:
        self.config = config
        self.species = species
        self.gem_driven = gem_driven
        w, h = config.width, config.height
        self.fields: dict[str, ConcentrationField] = {}
        self.scalars: dict[str, ScalarField] = {}
        for name, sc in config.substrates.items():
            self.fields[name] = ConcentrationField(
                name, w, h, sc.diffusion_um2_s, sc.initial_mm)
        for name, scl in config.scalars.items():
            self.scalars[name] = ScalarField(
                name, w, h, scl.kind, scl.initial, scl.forcing,
                scl.diffusion_um2_s)
        # O2/CO2 defaults for water-like patches
        if "oxygen" not in self.fields and not config.anoxic:
            bulk = None if config.flow_rate <= 0.0 else 0.21
            self.fields["oxygen"] = ConcentrationField(
                "oxygen", w, h, 2500.0, 0.21 if bulk is None else bulk)
        if "co2" not in self.fields:
            # ~1 mM dissolved inorganic carbon baseline (freshwater DIC
            # is 0.5-2 mM); CO2 is refilled by respiration and, in
            # flowing patches, by exchange with the bulk water
            self.fields["co2"] = ConcentrationField(
                "co2", w, h, 1600.0, 1.0)
        if "glucose" not in self.fields and config.kind in (
                "chemostat", "biofilm"):
            self.fields["glucose"] = ConcentrationField(
                "glucose", w, h, 600.0, 1.0)
        # fields any species consumes or secretes must exist (start at 0
        # unless configured): keeps cross-feeding and uptake well-posed
        for sp in species.values():
            for sub in list(sp.consumption) + list(sp.secretion):
                if sub not in self.fields:
                    self.fields[sub] = ConcentrationField(
                        sub, w, h, 300.0, 0.0)
                    config.substrates[sub] = SubstrateConfig(
                        initial_mm=0.0,
                        carbon_per_mol=default_carbon_per_mol(sub))
        # biomass per species per site [y][x]; every declared species owns
        # a grid (zero when not seeded) so per-species stepping, dispersal
        # and colonization are always well-posed
        self.biomass: dict[str, list[list[float]]] = {}
        for sname in species:
            self.biomass[sname] = [[0.0] * w for _ in range(h)]
        for name, amount in config.initial_biomass.items():
            if name not in self.biomass:
                self.biomass[name] = [[0.0] * w for _ in range(h)]
            grid = self.biomass[name]
            for y in range(h):
                for x in range(w):
                    grid[y][x] = amount / (w * h)
        self.mode: dict[str, str] = {s: (sp.preferred_substrate() or "none")
                                     for s, sp in species.items()}
        self.switch_recovery: dict[str, int] = {}
        self.century = CenturyPools(clay=config.clay)
        self.nitrogen = NitrogenCycle(cn_som=config.cn_som,
                                      cn_litter=config.cn_species)
        self.nitrogen.nh4_mm = config.initial_nh4_mm
        self.nitrogen.no3_mm = config.initial_no3_mm
        self.tick = 0
        self._pool_total_before: float = 0.0
        self._growth_c: float = 0.0
        self._prev_respired: float = 0.0
        self._som_n: float = 0.0
        self._totals: dict[str, float] = {
            s: sum(sum(row) for row in grid)
            for s, grid in self.biomass.items()}
        self.npp: dict[str, float] = {s: 0.0 for s in species}
        self.consumed_c: dict[str, float] = {s: 0.0 for s in species}
        self.respired_c: dict[str, float] = {s: 0.0 for s in species}
        self.predation_c: dict[str, float] = {s: 0.0 for s in species}

    # -- queries ----------------------------------------------------------

    def totals(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, grid in self.biomass.items():
            out[name] = sum(sum(row) for row in grid)
        return out

    def total_biomass_c(self) -> float:
        return sum(self.totals().values())

    def total_biomass_of(self, name: str) -> float:
        grid = self.biomass.get(name)
        if grid is None:
            return 0.0
        return sum(sum(row) for row in grid)

    def field_mm(self, name: str) -> float:
        return self.fields[name].total_mm()

    def scalar_mean(self, name: str) -> float:
        return self.scalars[name].mean()

    def local_field(self, name: str, x: int, y: int) -> float:
        return self.fields[name].get(x, y)

    def pool_totals(self) -> dict[str, float]:
        return dict(self.century.pools)

    # -- element budgets ---------------------------------------------------

    def _cpm(self, name: str) -> int:
        sc = self.config.substrates.get(name)
        if sc is not None and sc.carbon_per_mol > 0:
            return sc.carbon_per_mol
        return default_carbon_per_mol(name)

    def c_in_field(self, name: str) -> float:
        """Total carbon (C-units) held in a substrate field.

        Field totals are summed ``mM`` over the sites; carbon is counted
        in C-units where ``1 mM*site of substrate X == carbon_per_mol(X)
        C-units``.  Biomass, litter and the CENTURY pools use the same
        C-units, so the sealed-patch budget closes exactly.
        """
        return self.field_mm(name) * self._cpm(name)

    def carbon_balance(self) -> float:
        """Total carbon in the patch: fields + biomass + CENTURY pools.

        Conserved across ticks when the patch is sealed (``flow_rate ==
        0``) and the capacity cap recycles overflow into litter: every
        C flow -- growth, respiration, predation, death -> litter ->
        re-mineralization -- is tracked in the same C-units.
        """
        total = 0.0
        for name in self.fields:
            cpm = self._cpm(name)
            if cpm > 0:
                total += self.c_in_field(name)
        total += sum(self.totals().values())
        total += self.century.total()
        return total

    def nitrogen_budget(self) -> float:
        """Total N (mmol) in the patch: mineral pools + biomass + SOM.

        Closes exactly for any C:N configuration: SOM nitrogen is
        tracked explicitly (litter in at each species' C:N, released at
        ``cn_som``), so growth, death, predation and decomposition all
        move N 1:1 with their carbon flows.
        """
        bio_n = 0.0
        for sname, sp in self.species.items():
            if sname in self.biomass:
                bio_n += self.total_biomass_of(sname) / max(sp.cn_ratio, 1e-30)
        return (self.nitrogen.nh4_mm + self.nitrogen.no3_mm
                + bio_n + self._som_n)

    def carbon_in_biomass(self) -> float:
        return sum(self.totals().values())

    # -- dynamics ----------------------------------------------------------

    def _temperature(self) -> float:
        if "temperature" in self.scalars:
            return self.scalars["temperature"].mean()
        return 25.0

    def _light(self) -> float:
        if "light" in self.scalars:
            return self.scalars["light"].mean()
        return 0.0

    def _modifiers(self) -> tuple[float, float]:
        t = self._temperature()
        moisture = self.config.moisture
        t_mod = q10_rate_modifier(t, 2.0, 25.0)
        return t_mod, moisture

    def _fluctuation_factor(self) -> float:
        cfg = self.config
        if cfg.fluctuation_period <= 0:
            return 1.0
        phase = (self.tick // cfg.fluctuation_period) % 2
        if phase == 1:
            return max(0.0, 1.0 - cfg.fluctuation_amplitude)
        return 1.0

    def _update_totals(self) -> None:
        self._totals = self.totals()

    def step(self, ticks: int,
             rng: random.Random) -> dict[str, float]:
        """Advance the patch ``ticks`` minutes; returns per-species and
        field deltas for the scheduler."""
        before = self.totals()
        before_fields = {n: f.total_mm() for n, f in self.fields.items()}
        t_mod, moisture = self._modifiers()
        fluct = self._fluctuation_factor()
        stress = self._stress_level()

        for _ in range(ticks):
            self._step_populations(t_mod, moisture, fluct, stress, rng)
            self._step_fields(t_mod, moisture, ticks)
            self.century.step(1, t_mod, moisture)
            self._step_nitrogen(t_mod, moisture)
            self.tick += 1

        after = self.totals()
        after_fields = {n: f.total_mm() for n, f in self.fields.items()}
        deltas: dict[str, float] = {}
        for s in before:
            deltas[f"biomass:{s}"] = abs(after.get(s, 0.0) - before[s])
        for n in before_fields:
            deltas[f"field:{n}"] = abs(after_fields[n] - before_fields[n])
        deltas["pools"] = abs(self.century.total()
                              - self._pool_total_before)
        return deltas

    def _stress_level(self) -> float:
        for name in ("toxin", "ph"):
            if name in self.scalars:
                return self.scalars[name].mean()
        return 0.0

    def _step_populations(self, t_mod: float, moisture: float,
                          fluct: float, stress: float,
                          rng: random.Random) -> None:
        cfg = self.config
        w, h = cfg.width, cfg.height
        light = self._light()
        capacity = cfg.carrying_capacity
        litter = 0.0

        for name, sp in self.species.items():
            grid = self.biomass[name]
            traits = sp.traits
            temp_ok = self._temperature_ok(traits, stress)
            for y in range(h):
                row = grid[y]
                for x in range(w):
                    bx = row[x]
                    if bx <= 0.0:
                        continue
                    is_fba = False
                    if self.gem_driven and sp.metabolic_model is not None:
                        g_c, comps, is_fba = self._growth_rate_gem(
                            sp, bx, x, y, t_mod, moisture, fluct, light)
                    else:
                        g_c, comps = self._growth_rate(
                            sp, bx, x, y, t_mod, moisture, fluct, light)
                    g_c *= temp_ok
                    # N-limited growth (Redfield / Liebig)
                    scale = 1.0
                    demand_n = bx * g_c / sp.cn_ratio
                    if demand_n > 0.0:
                        avail = self.nitrogen.available_n()
                        if demand_n > avail:
                            scale = avail / demand_n
                    g = g_c * scale
                    # aerobic O2 cap
                    if "oxygen" in self.fields and sp.consumption:
                        o2 = self.fields["oxygen"].get(x, y)
                        respired_g = g * (1.0 - traits.yield_c) / traits.yield_c
                        demand_o2 = bx * respired_g
                        if demand_o2 > o2:
                            g *= o2 / demand_o2
                    # switching cost
                    rec = self.switch_recovery.get(name, 0)
                    if rec > 0:
                        g *= (1.0 - traits.switching_cost)
                        self.switch_recovery[name] = rec - 1
                    if is_fba and comps:
                        # GEM path: FBA exchange fluxes drive substrate
                        # depletion directly; growth comes from g*bx.
                        growth_c = 0.0
                        respired_total = 0.0
                        consumed_total = 0.0
                        expected_total = 0.0
                        _fba_fluxes = sp.last_fba_fluxes or {}
                        for sub, cpm, rate in comps:
                            field = self.fields.get(sub)
                            if field is None:
                                continue
                            expected = rate * g / g_c * bx if g_c > 0 else 0.0
                            expected_total += expected * cpm
                            removed = field.deplete(x, y, expected) if g_c > 0 else 0.0
                            if removed <= 0.0:
                                continue
                            c_units = removed * cpm
                            consumed_total += c_units
                        # Gate growth AND CO2/O2 on actual substrate
                        # consumption.  When no substrate was removed
                        # (all at floor), nothing should grow or emit.
                        if consumed_total > 0.0 and expected_total > 0.0:
                            consumption_ratio = min(
                                1.0, consumed_total / expected_total)
                            growth_c = g * bx * consumption_ratio
                            # CO2 produced by FBA (positive EX_co2_e
                            # flux), scaled by consumption ratio.
                            if "co2" in self.fields and _fba_fluxes:
                                co2_flux = max(
                                    0.0, _fba_fluxes.get("EX_co2_e", 0.0))
                                if co2_flux > 0 and g_c > 0:
                                    co2_rate = (co2_flux * _H_PER_TICK
                                                * g / g_c * bx
                                                * consumption_ratio)
                                    self.fields["co2"].add(x, y, co2_rate)
                                    respired_total += (
                                        co2_rate * self._cpm("co2"))
                            # O2 consumed by FBA (negative EX_o2_e
                            # flux), scaled by consumption ratio.
                            if "oxygen" in self.fields and _fba_fluxes:
                                o2_flux = min(
                                    0.0, _fba_fluxes.get("EX_o2_e", 0.0))
                                if o2_flux < 0 and g_c > 0:
                                    o2_rate = (abs(o2_flux) * _H_PER_TICK
                                               * g / g_c * bx
                                               * consumption_ratio)
                                    self.fields["oxygen"].deplete(
                                        x, y, o2_rate)
                            # O2 produced by FBA (positive EX_o2_e
                            # flux, e.g. photosynthesis), scaled by
                            # consumption ratio.
                            if "oxygen" in self.fields and _fba_fluxes:
                                o2_prod = max(
                                    0.0, _fba_fluxes.get("EX_o2_e", 0.0))
                                if o2_prod > 0 and g_c > 0:
                                    o2_rate = (o2_prod * _H_PER_TICK
                                               * g / g_c * bx
                                               * consumption_ratio)
                                    self.fields["oxygen"].add(
                                        x, y, o2_rate)
                        self.consumed_c[sp.name] += consumed_total
                        self.respired_c[sp.name] += respired_total
                    else:
                        growth_c = self._apply_growth(
                            sp, bx, x, y, comps, g, g_c, traits)
                    if growth_c > 0.0:
                        self.nitrogen.immobilize(growth_c / sp.cn_ratio)
                    self._growth_c += growth_c
                    self.npp[name] += growth_c
                    # DSL secretion suppressed when GEM-driven (FBA
                    # already accounts for all exchange fluxes).
                    sec_total = 0.0
                    gem_active = is_fba and g_c > 0
                    for sub, rate in sp.secretion.items():
                        if sub in self.fields:
                            sec = rate * bx if not gem_active else 0.0
                            self.fields[sub].add(x, y, sec)
                            sec_total += sec * self._cpm(sub)
                    if sec_total > 0.0:
                        self.nitrogen.excrete(sec_total / sp.cn_ratio)
                    new = (bx * (1.0 - sp.maintenance)
                           + growth_c - sec_total)
                    row[x] = max(0.0, new)
                    litter += bx * sp.maintenance
                    self._som_n += (bx * sp.maintenance) / sp.cn_ratio
        if litter > 0.0:
            self.century.add_litter(litter)
        self._step_predation(rng)
        self._apply_capacity(capacity)
        self._update_totals()

    def _temperature_ok(self, traits: SpeciesTraitParams,
                        stress: float) -> float:
        """Stress tolerance (A4, L8/L9): high tolerance survives high
        stress; tolerance 0 dies at any stress."""
        if stress <= 0.0:
            return 1.0
        survive = max(0.0, min(1.0, traits.tolerance / max(stress, 1e-9)))
        return survive

    def _growth_rate(self, sp: Species, bx: float, x: int, y: int,
                     t_mod: float, moisture: float,
                     fluct: float, light: float) -> tuple[float, list]:
        """C-limited per-biomass growth and its substrate components.

        Returns ``(g_c, components)`` where ``g_c`` is the un-capped
        specific growth rate and ``components`` is a list of
        ``(substrate, carbon_per_mol, rate)`` tuples (``"__photo__"``
        for light-gated CO2 fixation).  No state is mutated here; the
        caller applies the N cap and books the actual C flows.
        """
        traits = sp.traits
        components: list[tuple] = []
        g_c = 0.0
        for sub, (vmax, ks) in sp.consumption.items():
            field = self.fields.get(sub)
            if field is None:
                continue
            cpm = self._cpm(sub)
            s = field.get(x, y) * fluct
            uptake = monod_uptake(vmax, s, ks)
            rate = (uptake * t_mod * moisture
                    * traits.uptake_gain * traits.growth_rate_gain)
            if rate <= 0.0:
                continue
            components.append((sub, cpm, rate))
            g_c += rate * cpm * traits.yield_c
        if sp.photo and light > 0.0 and "co2" in self.fields:
            co2 = self.fields["co2"].get(x, y)
            rate = photosynthesis_rate(light, co2, sp.photo_vmax)
            if rate > 0.0:
                components.append(("__photo__", 1, rate))
                g_c += rate
        return g_c, components

    def _growth_rate_gem(
        self, sp: Species, bx: float, x: int, y: int,
        t_mod: float, moisture: float, fluct: float, light: float,
    ) -> tuple[float, list, bool]:
        """FBA-backed growth rate for a GEM-equipped species (doc/21 §3.3).

        Solves a per-site FBA with exchange bounds set from local
        substrate concentrations, then returns the biomass flux as the
        growth rate.  Falls back to Monod if FBA is infeasible.

        Design (v2 – unit-clean):
          FBA fluxes are in mmol/gDW/h while the ecosystem fields are in
          mM with an arbitrary biomass unit ``bx``.  Converting between
          them requires a biomass-density parameter the ecosystem does
          not track.  Instead we use FBA **only** for the growth rate
          (the biomass flux) and delegate substrate consumption to the
          existing Monod pathway which is already in correct ecosystem
          units.  This keeps the two models cleanly separated:
            * FBA → metabolically optimal growth rate
            * Monod → field-scale substrate consumption rates
        """
        from helixlang.metabolism import FluxBalanceAnalysis, MetabolicModel

        if sp.metabolic_model is None or not isinstance(sp.metabolic_model, MetabolicModel):
            _monod_result = self._growth_rate(sp, bx, x, y, t_mod, moisture,
                                     fluct, light)
            return _monod_result[0], _monod_result[1], False
        model = copy.deepcopy(sp.metabolic_model)

        try:
            fba = FluxBalanceAnalysis(model)

            # Phase VII: GRN → FBA closed loop (doc/25 G7)
            if sp.grn_edges:
                from helixlang.gem.bridge import apply_regulatory_bounds
                apply_regulatory_bounds(model, sp.grn_edges, sp.grn_gpr_map)

            # Phase VIII: Temperature/pH enzyme correction (doc/25 G8)
            from helixlang.metabolism import enzyme_correction
            _corr = enzyme_correction(
                self.config.temperature_c, self.config.ph)
            for rxn_id, rxn in model.reactions.items():
                if not rxn_id.startswith("EX_"):
                    rxn.upper_bound *= _corr
                    if rxn.lower_bound < 0:
                        rxn.lower_bound *= _corr

            # Phase IX: Density-dependent enzyme scaling (doc/25 G9)
            # Logistic-style: full capacity at low density, decreasing
            # toward zero as biomass approaches carrying capacity.
            carrying = self.config.carrying_capacity
            if carrying > 0 and bx > 0:
                _density_scale = max(0.01, 1.0 - bx / carrying)
                for rxn_id, rxn in model.reactions.items():
                    if not rxn_id.startswith("EX_"):
                        rxn.upper_bound *= _density_scale
                        if rxn.lower_bound < 0:
                            rxn.lower_bound *= _density_scale

            # Set exchange bounds from local substrate concentrations.
            # Monod uptake is in per-tick (per-minute) units; FBA
            # expects mmol/gDW/h, so we scale up by _H_PER_TICK.
            # When substrate concentration is at/below the floor, set
            # uptake to zero so FBA doesn't report phantom growth.
            # Use a dynamic floor: if concentration is <= 5% of ks,
            # Monod gives < 5% of vmax — treat as exhausted.
            for sub, (vmax_ks, ks_val) in sp.consumption.items():
                field = self.fields.get(sub)
                if field is None:
                    continue
                conc = field.get(x, y) * fluct
                gem_met = _ECOSYSTEM_TO_GEM_EXCHANGE.get(sub, sub)
                # Dynamic floor: substrate at or below this concentration
                # is treated as exhausted.  Use the larger of 5% of ks
                # (so Monod < 5% of vmax) and a practical floor above the
                # field's numerical minimum (~0.01).  This prevents
                # phantom growth when substrates are depleted.
                _floor = max(0.05 * ks_val if ks_val > 0 else 1e-4, 0.02)
                if conc <= _floor:
                    fba.set_uptake(gem_met, 0.0)
                else:
                    uptake = monod_uptake(vmax_ks, conc, ks_val)
                    rate = uptake * t_mod * moisture * sp.traits.uptake_gain
                    # Photo species: use photo_vmax for CO2 uptake
                    # (the photosynthetic rate, not the generic vmax).
                    if sp.photo and sub == "co2" and sp.photo_vmax > 0:
                        rate = monod_uptake(
                            sp.photo_vmax, conc, ks_val
                        ) * t_mod * moisture * sp.traits.uptake_gain
                        # Modulate by light saturation (Michaelis-Menten
                        # on PAR intensity).
                        _light_ks = 150.0
                        light_sat = (light / (_light_ks + light)
                                     if light > 0 else 0.0)
                        rate *= light_sat
                    fba.set_uptake(gem_met, max(0.0, rate / _H_PER_TICK))

            # Solve for biomass
            bm_rxn = model.biomass_reaction
            if bm_rxn and bm_rxn in model.reactions:
                fluxes = fba.solve(objective=bm_rxn)
                bm_flux = fluxes.get(bm_rxn, 0.0)
            else:
                fluxes = fba.solve(objective="BIOMASS_reaction")
                bm_flux = fluxes.get("BIOMASS_reaction", 0.0)

            # Store FBA result on species for cross-tick persistence (G9)
            sp.last_fba_fluxes = dict(fluxes)

            # Convert FBA 1/h to per-tick (per-minute) specific growth
            g_c = max(0.0, bm_flux * _H_PER_TICK * sp.traits.growth_rate_gain)
            g_c = min(g_c, sp.traits.max_growth_rate)

            if g_c > 0:
                # Build FBA exchange components for substrate accounting.
                # Each component is (field_name, cpm, rate) where rate
                # is in per-tick field-depletion units.
                components: list[tuple] = []
                for rxn_id, flux in fluxes.items():
                    if not rxn_id.startswith("EX_"):
                        continue
                    if flux >= -1e-8:
                        continue
                    substrate = _EXCHANGE_SUBSTRATE_MAP.get(rxn_id, "")
                    if substrate and substrate in self.fields:
                        cpm = self._cpm(substrate)
                        # mmol/gDW/h → per-tick field units:
                        # divide by 60 (_H_PER_TICK) to get per-tick,
                        # then scale by the ecosystem-specific unit
                        # factor so that _apply_growth depletes the
                        # correct amount of substrate per unit biomass.
                        components.append((substrate, cpm,
                                           abs(flux) * _H_PER_TICK))
                # Flag: components are FBA-sourced, not Monod.
                # The caller must use g_c*bx as growth_c (not the
                # yield_c-weighted sum) and handle CO2 separately.
                return g_c, components, True
            # FBA solved but biomass=0 (substrates exhausted).
            # Do NOT fall through to Monod — the GEM is authoritative.
            return 0.0, [], True
        except Exception as _gem_err:
            import sys
            print(f"[GEM-FALLBACK] {sp.name}: {_gem_err}",
                  file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

        _monod_result = self._growth_rate(sp, bx, x, y, t_mod, moisture,
                                            fluct, light)
        return _monod_result[0], _monod_result[1], False

    def _apply_growth(self, sp: Species, bx: float, x: int, y: int,
                      comps: list, g: float, g_c: float,
                      traits: SpeciesTraitParams) -> float:
        """Book the actual C flows of growth ``g`` and return the C
        gained (in C-units).

        Uses the substrate the site actually released (``deplete``
        clamps to availability), so the sealed budget closes exactly:
        each ``r`` mM removed carries ``r * carbon_per_mol`` C-units,
        ``yield`` becomes biomass, ``1 - yield`` is respired CO2 (1 C
        per molecule -> C-units map 1:1 to mM).
        """
        rate_scale = g / g_c if g_c > 0.0 else 0.0
        growth_c = 0.0
        for sub, cpm, rate in comps:
            if sub == "__photo__":
                removed = self.fields["co2"].deplete(x, y, rate * rate_scale * bx)
                if removed > 0.0:
                    if "oxygen" in self.fields:
                        self.fields["oxygen"].add(x, y, removed)
                    growth_c += removed
                continue
            field = self.fields.get(sub)
            if field is None:
                continue
            removed = field.deplete(x, y, rate * rate_scale * bx)
            if removed <= 0.0:
                continue
            c_units = removed * cpm
            respired = c_units * (1.0 - traits.yield_c)
            if respired > 0.0:
                if "co2" in self.fields:
                    self.fields["co2"].add(x, y, respired)
                if "oxygen" in self.fields:
                    self.fields["oxygen"].deplete(x, y, respired)
                self.respired_c[sp.name] += respired
            self.consumed_c[sp.name] += c_units
            growth_c += c_units * traits.yield_c
        return growth_c

    def _step_predation(self, rng: random.Random) -> None:
        """Mass-action predation (L6, Lotka-Volterra)."""
        totals = self.totals()
        for pred_name, pred in self.species.items():
            for prey_name, conversion in pred.diet.items():
                attack = pred.attack_rate.get(prey_name, 1e-4)
                xp = totals[pred_name]
                xq = totals[prey_name]
                if xp <= 0.0 or xq <= 0.0:
                    continue
                eaten = attack * xp * xq
                # subtract prey C proportionally across sites
                self._subtract_biomass(prey_name, eaten)
                # predator gains conversion fraction; the rest is
                # respired (trophic inefficiency, Lindeman 1942)
                gain = eaten * conversion
                self._add_biomass(pred_name, gain)
                self.predation_c[pred_name] += gain
                self.respired_c[pred_name] += eaten * (1.0 - conversion)
                if "co2" in self.fields:
                    self.fields["co2"].add(
                        0, 0, eaten * (1.0 - conversion))
                # the C respired from prey carries N (egestion/excretion,
                # closes the trophic N loop at equal C:N ratios)
                prey_sp = self.species.get(prey_name)
                prey_cn = prey_sp.cn_ratio if prey_sp else 6.0
                self.nitrogen.excrete(
                    eaten * (1.0 - conversion) / max(prey_cn, 1e-30))

    def _subtract_biomass(self, name: str, amount: float) -> None:
        grid = self.biomass[name]
        total = sum(sum(row) for row in grid)
        if total <= 0.0 or amount <= 0.0:
            return
        frac = amount / total
        for row in grid:
            for i in range(len(row)):
                row[i] = max(0.0, row[i] * (1.0 - frac))

    def _add_biomass(self, name: str, amount: float) -> None:
        if name not in self.biomass:
            return
        grid = self.biomass[name]
        n = sum(len(row) for row in grid)
        if n == 0:
            return
        per = amount / n
        for row in grid:
            for i in range(len(row)):
                row[i] += per

    def _apply_capacity(self, capacity: float) -> None:
        total = self.total_biomass_c()
        if total > capacity:
            scale = capacity / total
            removed = 0.0
            for sname, grid in self.biomass.items():
                sp = self.species[sname]
                for row in grid:
                    for i in range(len(row)):
                        rem = row[i] * (1.0 - scale)
                        row[i] = max(0.0, row[i] * scale)
                        removed += rem
                        self._som_n += rem / max(sp.cn_ratio, 1e-30)
            # recycle the overflow into litter so the sealed C budget
            # still closes
            self.century.add_litter(removed)

    def _step_fields(self, t_mod: float, moisture: float,
                     ticks: int) -> None:
        cfg = self.config
        for fld in self.fields.values():
            fld.diffuse()
        if cfg.flow_rate > 0.0:
            self._replenish()
        for scalar in self.scalars.values():
            scalar.step(self.tick)

    def _replenish(self) -> None:
        cfg = self.config
        for name, fld in self.fields.items():
            sc = cfg.substrates.get(name)
            bulk = sc.bulk_mm if sc else None
            if bulk is None:
                continue
            for y in range(fld.height):
                row = fld.concentration[y]
                for x in range(fld.width):
                    row[x] = row[x] + cfg.flow_rate * (bulk - row[x])

    def _step_nitrogen(self, t_mod: float, moisture: float) -> None:
        decayed = self.century.respired_c - self._prev_respired
        self._prev_respired = self.century.respired_c
        self.nitrogen.step(decayed, 1, t_mod, moisture, self.config.anoxic)
        # the decomposed SOM releases its nitrogen to the mineral pool
        self._som_n -= decayed / max(self.config.cn_som, 1e-30)
        if self._som_n < 0.0:
            self._som_n = 0.0
        # decomposition respiration leaves the pools as CO2 -> the co2
        # field (1 C per molecule: C-units map 1:1 to mM)
        if decayed > 0.0 and "co2" in self.fields:
            co2 = self.fields["co2"]
            n = co2.width * co2.height
            per = decayed / n if n else 0.0
            for y in range(co2.height):
                for x in range(co2.width):
                    co2.add(x, y, per)


# ============================================================================
# Ecosystem: patches + species + scheduler + analytics + evolution loop
# ============================================================================

class Ecosystem:
    """Integrated ecosystem runtime (doc/19 §5.3-§5.6).

    Owns the patches, the species, the scheduler (D1), the analytics
    (D4) and the invasion-fitness evolution loop (A4).
    """

    def __init__(self, config: EcosystemConfig | None = None) -> None:
        self.config = config or EcosystemConfig()
        self.rng = random.Random(self.config.seed)
        self.species_map: dict[str, Species] = {
            s.name: s for s in self.config.species}
        if not self.species_map:
            raise ValueError("Ecosystem needs at least one Species")
        self.patches: list[Patch] = [
            Patch(pc, self.species_map, gem_driven=self.config.gem_driven)
            for pc in self.config.patches]
        if not self.patches:
            raise ValueError("Ecosystem needs at least one Patch")
        self.scheduler = Scheduler(
            max_step=self.config.scheduler_max_step,
            change_threshold=self.config.scheduler_change_threshold)
        self.tick = 0
        self.history: list[dict[str, float]] = []
        self._last_deltas: list[float] = [1.0]

    # -- scheduler helpers -------------------------------------------------

    def _forcing_delta(self) -> float:
        """Max per-tick change of any scalar forcing over the next
        ``max_step`` minutes (0 when a driver is constant)."""
        nxt = self.tick + self.config.scheduler_max_step
        worst = 0.0
        for patch in self.patches:
            for sc in patch.config.scalars.values():
                if sc.forcing is not None:
                    worst = max(worst, abs(sc.forcing(nxt) - sc.forcing(self.tick)))
        return worst

    # -- main loop ---------------------------------------------------------

    def step(self) -> int:
        """Advance the ecosystem by one step; returns the number of ticks
        actually advanced.

        When ``config.fast_forward`` is set the event scheduler (D1)
        skips quiescent epochs; otherwise the step is exactly one tick.
        """
        if self.config.fast_forward:
            advance = self.scheduler.next_advance(
                self._last_deltas, self._forcing_delta())
        else:
            advance = 1
        for patch in self.patches:
            patch._pool_total_before = patch.century.total()
            deltas = patch.step(advance, self.rng)
            self._last_deltas = list(deltas.values())
        self._disperse(advance)
        self.tick += advance
        return advance

    def _disperse(self, ticks: int) -> None:
        """Per-cell migration along patch dispersal edges (L7)."""
        by_patch = {p.config.name: p for p in self.patches}
        moved: dict[tuple[str, str], float] = {}
        for patch in self.patches:
            for neighbor_name, rate in patch.config.dispersal.items():
                neighbor = by_patch.get(neighbor_name)
                if neighbor is None:
                    raise ValueError(
                        f"patch {patch.config.name!r} dispersal target "
                        f"{neighbor_name!r} not found")
                for sp_name, amount in patch.totals().items():
                    flux = amount * rate * ticks
                    if flux <= 0.0:
                        continue
                    patch._subtract_biomass(sp_name, flux)
                    neighbor._add_biomass(sp_name, flux)
                    moved[(patch.config.name, neighbor_name)] = (
                        moved.get((patch.config.name, neighbor_name), 0.0)
                        + flux)

    def run(self) -> list[dict[str, float]]:
        """Run ``config.ticks`` minutes; returns the observation rows."""
        cfg = self.config
        while self.tick < cfg.ticks:
            self.step()
            if self.tick % cfg.sample_every == 0:
                self.history.append(self.record())
        return self.history

    # -- observations / analytics (D4) -------------------------------------

    def record(self) -> dict[str, float]:
        row: dict[str, float] = {"tick": float(self.tick)}
        for patch in self.patches:
            prefix = patch.config.name
            for sp_name, amount in patch.totals().items():
                row[f"{prefix}:{sp_name}"] = amount
            for fname in patch.fields:
                if patch._cpm(fname) > 0 or fname == "oxygen":
                    row[f"{prefix}:{fname}"] = patch.field_mm(fname)
            for sname in patch.scalars:
                row[f"{prefix}:{sname}"] = patch.scalar_mean(sname)
        return row

    def abundances(self) -> dict[str, float]:
        """Per-species total abundance across all patches."""
        out: dict[str, float] = {s: 0.0 for s in self.species_map}
        for patch in self.patches:
            for name, amount in patch.totals().items():
                out[name] += amount
        return out

    def energy_flow(self) -> dict[str, dict[str, float]]:
        """Per-patch energy flow: NPP -> consumption -> decomposition.

        Reports gross NPP (biomass C fixed), predation consumption and
        respired CO2 per patch -- the production/consumption/decomposition
        ledger of the ecosystem (Lindeman 1942; D4).
        """
        out: dict[str, dict[str, float]] = {}
        for patch in self.patches:
            npp = sum(patch.npp.values())
            consumption = sum(patch.predation_c.values())
            respiration = sum(patch.respired_c.values()) + patch.century.respired_c
            out[patch.config.name] = {
                "npp": npp,
                "consumption": consumption,
                "respiration": respiration,
                "decomposition": patch.century.litter_in_c,
            }
        return out

    def trophic_efficiency(self) -> dict[str, float]:
        """Consumer-biomass / producer-biomass per link (Lindeman 1942)."""
        out: dict[str, float] = {}
        for patch in self.patches:
            totals = patch.totals()
            producers = 0.0
            consumers = 0.0
            for name, sp in self.species_map.items():
                if sp.photo:
                    producers += totals.get(name, 0.0)
                elif sp.diet:
                    consumers += totals.get(name, 0.0)
            out[patch.config.name] = (
                consumers / producers if producers > 0.0 else 0.0)
        return out

    def neutral_vs_niche(self) -> dict[str, float]:
        """Neutral-vs-niche diagnostic (Hubbell 2001; D4).

        Compares the observed per-species abundance distribution against
        the neutral (random multinomial) null: under pure drift the
        species-abundance variance matches the multinomial expectation
        ``N*p*(1-p)``; a deviance far above the null signals niche
        structure (partitioned resources stabilize abundances).
        """
        counts = list(self.abundances().values())
        total = sum(counts)
        if total <= 0.0:
            return {"label": 0.0, "deviance": 0.0}
        n_species = len(counts)
        p = 1.0 / n_species if n_species else 0.0
        expected_var = total * p * (1.0 - p)
        observed_var = sum((c - total / n_species) ** 2 for c in counts) / n_species
        deviance = observed_var / max(expected_var, 1e-30)
        # label: 1 = niche-structured (variance exceeds the neutral null
        # several-fold), 0 = drift-dominated
        label = 1.0 if deviance > 4.0 else 0.0
        return {"label": label, "deviance": deviance}

    def summary(self) -> dict[str, float]:
        flow = self.energy_flow()
        out: dict[str, float] = {
            "total_biomass_c": sum(self.abundances().values()),
            "total_npp": sum(f["npp"] for f in flow.values()),
            "total_respiration": sum(f["respiration"] for f in flow.values()),
            "neutral_niche": self.neutral_vs_niche()["label"],
        }
        return out

    # -- invasion-fitness evolution loop (A4, L8) --------------------------

    def _evaluate_growth(self, ticks: int) -> dict[str, float]:
        """Long-term per-tick log-growth rate of each species (invasion
        fitness proxy: Ferriere & Legendre 2013)."""
        start = self.abundances()
        for _ in range(ticks):
            self.step()
        end = self.abundances()
        out: dict[str, float] = {}
        for name, s0 in start.items():
            s1 = end.get(name, 0.0)
            if s0 > 1e-9 and s1 > 1e-9:
                out[name] = (math.log(s1) - math.log(s0)) / max(1, ticks)
            elif s1 <= 1e-9:
                out[name] = -10.0
            else:
                out[name] = 10.0
        return out

    def run_generations(self) -> list[dict[str, float]]:
        """Outer-loop evolution: mutant genomes invade when their
        long-term growth rate in the resident community exceeds the
        resident's (adaptive dynamics / invasion fitness)."""
        cfg = self.config
        rows: list[dict[str, float]] = []
        for gen in range(cfg.generations):
            resident = self._evaluate_growth(cfg.evaluation_ticks)
            row: dict[str, float] = {
                "generation": float(gen), "tick": float(self.tick)}
            for name in self.species_map:
                sp = self.species_map[name]
                base = (cfg.seed if cfg.seed is not None else 0)
                seed = base + zlib.crc32(sp.genome.encode("ascii")) % (2 ** 31)
                mutant_genome, _ = mutate(
                    sp.genome, mutation_rate=cfg.substitution_rate,
                    indel_rate=cfg.indel_rate,
                    rng=random.Random(seed + gen))
                if len(mutant_genome) != cfg.genome_length_nt:
                    mutant_genome = (mutant_genome + "A" * cfg.genome_length_nt)[
                        :cfg.genome_length_nt]
                mutant = Species(
                    name=sp.name, genome=mutant_genome,
                    consumption=dict(sp.consumption),
                    secretion=dict(sp.secretion), diet=dict(sp.diet),
                    attack_rate=dict(sp.attack_rate), photo=sp.photo,
                    photo_vmax=sp.photo_vmax, cn_ratio=sp.cn_ratio,
                    maintenance=sp.maintenance)
                r_growth = resident.get(name, -10.0)
                clone = self._clone_with(name, mutant)
                m_growth = clone._evaluate_growth(cfg.evaluation_ticks).get(
                    name, -10.0)
                row[f"{name}:resident_growth"] = r_growth
                row[f"{name}:mutant_growth"] = m_growth
                if m_growth > r_growth:
                    sp.genome = mutant_genome
                    sp.traits = mutant.traits
                    row[f"{name}:substituted"] = 1.0
                else:
                    row[f"{name}:substituted"] = 0.0
            rows.append(row)
            self.history.append(row)
        return rows

    def _clone_with(self, name: str, mutant: Species) -> Ecosystem:
        """A replicate ecosystem with ``mutant`` replacing species
        ``name``, seeded from the current state (same patches/fields)."""
        cfg = self.config
        species = [mutant if s.name == name else s for s in cfg.species]
        clone = Ecosystem(EcosystemConfig(
            ticks=0, seed=cfg.seed, fast_forward=False,
            community_fba=cfg.community_fba, gem_driven=cfg.gem_driven,
            species=species, patches=cfg.patches))
        # copy current biomass/fields into the clone
        by_name = {p.config.name: p for p in self.patches}
        for dst in clone.patches:
            src = by_name[dst.config.name]
            for sname, grid in src.biomass.items():
                dst.biomass[sname] = [row[:] for row in grid]
            for fname, fld in src.fields.items():
                dst.fields[fname].concentration = fld.snapshot()
            dst.century.pools = dict(src.century.pools)
            dst.century.respired_c = src.century.respired_c
            dst.century.litter_in_c = src.century.litter_in_c
            dst._som_n = src._som_n
            dst.nitrogen.nh4_mm = src.nitrogen.nh4_mm
            dst.nitrogen.no3_mm = src.nitrogen.no3_mm
            dst._prev_respired = src._prev_respired
        return clone


# ============================================================================
# Common species/patch presets
# ============================================================================

def phototroph(name: str = "producer", genome: str = "") -> Species:
    """Light-gated photoautotroph (B3): CO2 + light -> biomass + O2."""
    return Species(
        name=name, genome=genome, photo=True, photo_vmax=0.01,
        consumption={}, cn_ratio=8.0, maintenance=0.001)


def heterotroph(name: str, substrate: str = "glucose",
                vmax: float = 0.02, ks: float = 0.1,
                genome: str = "") -> Species:
    """Aerobe consuming a dissolved substrate (Monod, per-species Ks)."""
    return Species(
        name=name, genome=genome,
        consumption={substrate: (vmax, ks)}, cn_ratio=6.0,
        maintenance=0.002)


def acetotroph(name: str, vmax: float = 0.012, ks: float = 0.05,
               genome: str = "") -> Species:
    """Acetate consumer (cross-feeding partner, L3)."""
    return Species(
        name=name, genome=genome,
        consumption={"acetate": (vmax, ks)}, cn_ratio=6.0,
        maintenance=0.0015)


def water_patch(name: str = "water",
                phototroph_amount: float = 100.0,
                consumer_amount: float = 10.0,
                initial_nh4_mm: float = 0.0,
                initial_no3_mm: float = 0.0) -> PatchConfig:
    """Diurnal water column: phototroph + heterotroph consumer.

    Light is a diurnal sine (full-day mean 500 µmol m⁻² s⁻¹); the patch
    is sealed (no flow), so daytime photosynthesis supersaturates O2 and
    night-time respiration sags it (B4 gate).
    """
    return PatchConfig(
        name=name, kind="water",
        carrying_capacity=1e5,
        initial_biomass={"producer": phototroph_amount,
                         "consumer": consumer_amount},
        initial_nh4_mm=initial_nh4_mm,
        initial_no3_mm=initial_no3_mm,
        substrates={},
        scalars={
            "light": ScalarConfig("light", 500.0,
                                  DiurnalForcing(500.0, 500.0, lo=0.0)),
            "temperature": ScalarConfig("temperature", 25.0,
                                        DiurnalForcing(25.0, 3.0)),
        },
        dispersal={},
    )


__all__ = [
    "SpeciesTraitParams", "Species", "SubstrateConfig", "ScalarConfig",
    "PatchConfig", "EcosystemConfig",
    "lotka_volterra_step", "lotka_volterra_conserved",
    "Metapopulation", "source_sink_equilibrium",
    "century_k_per_min", "CenturyPools", "NitrogenCycle",
    "CommunityFBA", "Scheduler", "Patch", "Ecosystem",
    "phototroph", "heterotroph", "acetotroph", "water_patch",
    "gem_to_species",
    "build_multi_species_ecosystem",
]


# ============================================================================
# Multi-species ecosystem from genomes (doc/22 §18, Phase I)
# ============================================================================

def build_multi_species_ecosystem(
    species_genomes: dict[str, str],
    medium: str = "glucose_minimal",
    ticks: int = 1000,
    width: int = 10,
    height: int = 10,
    flow_rate: float = 0.001,
    initial_biomass: float = 10.0,
    substrate_initial_mm: float = 10.0,
    substrate_bulk_mm: float = 10.0,
    gem_dt: float = 0.05,
) -> Ecosystem:
    """Build an ecosystem from genome FASTA files.

    For each species:
    1. Run GEM pipeline → functional MetabolicModel
    2. Extract Monod parameters via gem_to_species
    3. Create Species with metabolic_model attached
    4. Build Ecosystem with gem_driven=True

    Parameters
    ----------
    species_genomes:
        Mapping of ``name → FASTA_path``.
    medium:
        GEM medium preset (``"glucose_minimal"``, ``"bg11"``, etc.).
    ticks:
        Number of ecosystem tick-loop iterations.
    width, height:
        Patch grid dimensions.
    flow_rate:
        Chemostat dilution rate.
    initial_biomass:
        Initial biomass (gDW/L) seeded per species.
    substrate_initial_mm, substrate_bulk_mm:
        Initial and bulk substrate concentration (mM).
    gem_dt:
        Time step (h) passed to the GEM pipeline for dFBA.
    """
    import tempfile
    from pathlib import Path

    from helixlang.apps.gem_pipeline import run_gem_pipeline

    species_list: list[Species] = []
    substrates: dict[str, SubstrateConfig] = {}

    for name, fasta_path in species_genomes.items():
        # Handle inline DNA sequences (not file paths)
        _tmp_fasta: Path | None = None
        _path = fasta_path
        if not Path(_path).exists() and len(_path) > 10:
            _tmp_fasta = Path(tempfile.mktemp(suffix=".fasta"))
            _tmp_fasta.write_text(f">{name}\n{_path}\n")
            _path = str(_tmp_fasta)

        try:
            result = run_gem_pipeline(
                genome_fasta=_path,
                organism=name,
                medium=medium,
            )
        finally:
            if _tmp_fasta is not None:
                _tmp_fasta.unlink(missing_ok=True)

        params = gem_to_species(result, organism=name, medium=medium)

        # Detect primary substrate from fluxes
        primary_sub = "glucose"
        fluxes = dict(result.fba_fluxes) if result.fba_fluxes else {}
        for rxn_id, flux in fluxes.items():
            if flux < 0 and "co2" in rxn_id.lower():
                primary_sub = "co2"
                break

        # Build consumption dict
        consumption: dict[str, tuple[float, float]] = {}
        vmax = float(params.get("vmax", 0))
        ks = float(params.get("ks", 0.1))
        if vmax > 0:
            consumption[primary_sub] = (vmax, ks)

        # Photoautotrophs
        is_photo = primary_sub == "co2"

        sp = Species(
            name=name,
            metabolic_model=result.metabolic_model,
            consumption=consumption,
            photo=is_photo,
            photo_vmax=0.01 if is_photo else 0.0,
            cn_ratio=float(params.get("cn_ratio", 6.0)),
            maintenance=float(params.get("maintenance", 0.002)),
            traits=SpeciesTraitParams(
                yield_c=float(params.get("yield_c", 0.5)),
                max_growth_rate=float(params.get("max_growth_rate", 0.87)),
            ),
            gem_fluxes=fluxes,
        )
        # Propagate secretion from FBA
        secretion = params.get("secretion")
        if isinstance(secretion, dict):
            sp.secretion.update(secretion)

        species_list.append(sp)

        # Register substrate if not yet present
        if primary_sub not in substrates:
            is_co2 = primary_sub == "co2"
            substrates[primary_sub] = SubstrateConfig(
                initial_mm=substrate_initial_mm if not is_co2 else 5.0,
                bulk_mm=substrate_bulk_mm if not is_co2 else 5.0,
                carbon_per_mol=1 if is_co2 else 6,
            )

    if not species_list:
        raise ValueError("species_genomes must contain at least one entry")

    patch = PatchConfig(
        name="env",
        kind="chemostat",
        width=width,
        height=height,
        flow_rate=flow_rate,
        initial_biomass={sp.name: initial_biomass for sp in species_list},
        substrates=substrates,
    )

    cfg = EcosystemConfig(
        ticks=ticks,
        species=species_list,
        patches=[patch],
        gem_driven=True,
    )
    return Ecosystem(cfg)
