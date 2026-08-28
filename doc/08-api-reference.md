# API Reference

> **2026-08-28 — Legacy import paths updated** for the doc/36 plugin re-layout (flat `helixlang.X` -> `helixlang.core.*`/`helixlang.plugins.runtime.*`).

> Quick reference for the core HelixLang Python API. Organized by module, listing dataclasses, function signatures, and key parameters.

---

## Table of Contents

1. [codon_table — codon table](#1-codon_table--codon-table)
2. [central_dogma — central dogma](#2-central_dogma--central-dogma)
3. [metabolism — metabolic FBA](#3-metabolism--metabolic-fba)
4. [protein_structure — protein structure](#4-protein_structure--protein-structure)
5. [crispr — CRISPR gene editing](#5-crispr--crispr-gene-editing)
6. [epigenetics — epigenetics](#6-epigenetics--epigenetics)
7. [evolution — evolution engine](#7-evolution--evolution-engine)
8. [grn — gene regulatory network](#8-grn--gene-regulatory-network)
9. [units — calibration registry](#9-units--calibration-registry)
10. [reaction_diffusion — reaction-diffusion](#10-reaction_diffusion--reaction-diffusion)
11. [lsystem — L-system](#11-lsystem--l-system)
12. [stochastic — promoter noise & SSA](#12-stochastic--two-state-promoter-noise)
13. [environment — diffusing nutrient fields](#13-environment--diffusing-nutrient-fields)

---

## 1. codon_table — Codon Table

```python
from helixlang.core.codon_table import (
    Op,                    # enum: all opcodes
    STANDARD_TABLE,        # dict[str, Op]: standard codon→opcode
    MITO_VERTEBRATE_TABLE,  # mitochondrial table
    CILIATE_TABLE,         # ciliate table
    TABLES,                # dict[str, dict]: all translation tables
    get_table(name),       # get a named translation table
    wobble(codon),         # wobble rule at the degenerate position
)
```

---

## 2. central_dogma — Central Dogma

### Dataclasses

```python
@dataclass
class Transcript:
    sequence: str               # full mRNA sequence (T replaced by U)
    cds: str                    # coding sequence (AUG...UAA/UAG/UGA)
    utr5: str                   # 5'UTR
    utr3: str                   # 3'UTR
    poly_a_tail: str            # poly-A tail
    half_life_minutes: float    # half-life (default 5.0)
    elongation_time_s: float    # transcription elongation time
    promoter_strength: float    # promoter strength [0,1]
    initiation_frequency_per_min: float
    has_terminator: bool        # whether a terminator was detected

@dataclass
class TranslationResult:
    protein: str                # protein sequence
    elongation_time: float      # translation elongation time
    codon_rates: list[float]   # translation rate of each codon
    stop_codon: str             # stop codon
    stop_efficiency: float      # stop efficiency
    rbs_found: bool             # whether an RBS was detected
    rbs_sequence: str           # RBS sequence
```

### Functions

```python
transcribe(dna: str, promoter_strength: float = 1.0,
           transcription_factors: dict[str, float] | None = None
           ) -> Transcript

translate(transcript: Transcript) -> TranslationResult

calculate_mrna_level(transcript: Transcript, time_minutes: float,
                     degradation_rate: float | None = None) -> float
    # returns the mRNA level as a molecule count:
    # level × PROTEINS_PER_MRNA_LIFETIME (the number of proteins each
    # mRNA is expected to produce over its lifetime).

coupled_transcription_translation(dna: str,
                                  promoter_strength: float = 1.0
                                  ) -> dict
```

### Constants

```python
TRANSCRIPTION_ELONGATION_RATE_NT_PER_S = 50.0   # Proshkin 2010
TRANSLATION_ELONGATION_RATE_AA_PER_S   = 20.0   # Ingolia 2009
MRNA_HALF_LIFE_MEDIAN_MIN              = 5.0    # Bernstein 2002
E_COLI_POLY_A_TAIL_LENGTH              = 15
MAX_TRNA_ABUNDANCE                     = 3500   # CTG
MAX_INITIATION_FREQUENCY_PER_MIN       = 10.0
COUPLING_OFFSET_NT                     = 30     # transcription-translation coupling offset (Miller 1972)
PROTEINS_PER_MRNA_LIFETIME             = 100.0  # proteins per mRNA per lifetime (molecule-count scaling)
```

---

## 3. metabolism — Metabolic FBA

### Dataclasses

```python
@dataclass(slots=True)
class Reaction:
    id: str
    name: str
    stoichiometry: dict[str, float]  # {metabolite: coefficient}, negative=reactant, positive=product
    lower_bound: float = 0.0           # 0=irreversible
    upper_bound: float = 1000.0
    subsystem: str = "other"           # glycolysis/tca/ppp/fermentation/exchange/biomass
```

### MetabolicModel

```python
class MetabolicModel:
    reactions: dict[str, Reaction]
    metabolites: set[str]
    biomass_reaction: str | None

    def add_reaction(self, reaction: Reaction) -> None
    def set_biomass(self, reaction_id: str) -> None
    def get_stoichiometry_matrix(self) -> tuple[list[str], list[str], list[list[float]]]
        # → (metabolites, reactions, S_matrix)
```

### FluxBalanceAnalysis

```python
class FluxBalanceAnalysis:
    def __init__(self, model: MetabolicModel)
    def set_uptake(self, metabolite: str, flux: float) -> None
    def solve(self, objective: str = "biomass", maximize: bool = True
              ) -> dict[str, float]
        # → {reaction_id: flux_value}
    def analyze(self) -> dict
        # → {biomass_yield, glucose_uptake, byproduct_secretion,
        #    key_fluxes, biomass_per_glucose, subsystem_fluxes, ...}
```

### simplex

```python
def simplex(c: list[float],            # objective coefficients
            A: list[list[float]],        # constraint matrix
            b: list[float],              # constraint right-hand sides
            bounds: list[tuple[float, float]],  # variable bounds
            maximize: bool = True,
            max_iter: int = 10000
            ) -> dict
    # → {"status": "optimal"/"infeasible"/"unbounded",
    #    "x": list[float], "objective": float}
```

### Dynamic Flux Balance Analysis (dFBA)

Dynamic batch-culture simulation (Mahadevan et al. 2002, static
optimization approach). Each step sets the glucose-uptake LP bound from
the external substrate via Michaelis-Menten kinetics, solves the
instantaneous FBA, and integrates the batch ODEs with forward Euler:

```
v_glc(t) = v_max · S(t) / (Ks + S(t))
dX/dt = μ·X        dS/dt = −v_glc·X        dP/dt = v_secret·X
```

```python
@dataclass(slots=True)
class DynamicFBAConfig:
    dt_h: float = 0.25                       # integration step (h)
    initial_biomass_gdw: float = 0.05
    initial_glucose_mm: float = 10.0
    initial_acetate_mm: float = 0.0
    max_glucose_uptake: float = DEFAULT_GLC_UPTAKE
    glucose_half_saturation_mm: float = 0.1 # Ks (mM)
    biomass_per_mmol: float = _BIOMASS_MW    # gDW/mmol biomass flux
    min_biomass: float = 1e-9                # growth floor for stop

class DynamicFluxBalance:
    def __init__(self,
                 model: MetabolicModel | None = None,
                 config: DynamicFBAConfig | None = None,
                 fba: FluxBalanceAnalysis | None = None)
    def reset(self) -> None
    def set_state(self, biomass_gdw=None, glucose_mm=None,
                  acetate_mm=None) -> None
    def uptake_bound(self, glucose_mm: float) -> float
    def step(self, dt_h: float | None = None) -> dict[str, float]
        # → {"time", "biomass", "glucose", "growth_rate",
        #    "glucose_uptake", <byproduct pools>}
    def run(self, duration_h: float | None = None,
            max_steps: int = 100000) -> list[dict[str, float]]
    @property
    def growth_rate(self) -> float
    def last(self) -> dict[str, float]
    def update_from_environment(self, environment, x=None, y=None) -> None
        # → batch glucose from the environment field at (x, y)
    def apply_to_environment(self, environment, x=None, y=None) -> None
        # → deposit accumulated acetate into the environment field

    # instance state
    time_h: float; biomass_gdw: float; glucose_mm: float
    byproducts_mm: dict[str, float]   # lactate/acetate/co2 pools
    history: list[dict[str, float]]
```

Byproduct exchanges are discovered from the model (lactate/acetate/CO₂).
The reduced 37-reaction core has no glyoxylate shunt, so overflow acetate
is not re-consumed — the fermentative phase and glucose-exhaustion arrest
of the classic diauxic shift are reproduced; a model with the shunt
consumes acetate automatically. When glucose is exhausted, `run()` stops
growth at the `min_biomass` floor.

### Prebuilt Model

```python
ECOLI_CORE_MODEL: MetabolicModel   # 37-reaction E. coli core metabolism
```

### Constants

```python
DEFAULT_UPPER_BOUND  = 1000.0
DEFAULT_LOWER_BOUND  = -1000.0
DEFAULT_GLC_UPTAKE   = 10.0        # mmol/gDW/h
ATP_MAINTENANCE_FLUX = 8.39        # mmol/gDW/h (Orth 2010)
```

---

## 4. protein_structure — Protein Structure

### Dataclasses

```python
@dataclass(slots=True)
class SecondaryStructureSegment:
    start: int          # 1-indexed
    end: int
    ss_type: str        # "H"/"E"/"T"/"C"
    score: float
    sequence: str

@dataclass(slots=True)
class TransmembraneHelix:
    start: int; end: int; length: int
    mean_hydropathy: float; sequence: str

@dataclass(slots=True)
class DisorderRegion:
    start: int; end: int; length: int
    mean_hydropathy: float; sequence: str

@dataclass
class ProteinStructureReport:
    sequence: str
    length: int
    secondary_structure: str          # per-residue SS state "HE TC..."
    ss_segments: list[SecondaryStructureSegment]
    helix_fraction: float
    sheet_fraction: float
    turn_fraction: float
    coil_fraction: float
    hydropathy_profile: list[float]
    mean_hydropathy: float
    transmembrane_helices: list[TransmembraneHelix]
    disorder_regions: list[DisorderRegion]
    disorder_fraction: float
    is_membrane_protein: bool
    gravy: float
    summary: str
    def to_dict(self) -> dict
```

### Functions

```python
predict_secondary(sequence: str
                  ) -> tuple[str, list[SecondaryStructureSegment]]
    # → (ss_string, segments)

hydropathy_profile(sequence: str, window: int = 9) -> list[float]

gravy(sequence: str) -> float

predict_transmembrane(sequence: str,
                      window: int = 19,
                      threshold: float = 1.6,
                      min_length: int = 18,
                      max_length: int = 30,
                      extension_threshold: float = 0.8
                      ) -> list[TransmembraneHelix]

predict_disorder(sequence: str,
                 window: int = 30,
                 hydropathy_max: float = -0.5,
                 charge_threshold: float = 0.2
                 ) -> list[DisorderRegion]

predict_structure(sequence: str) -> ProteinStructureReport
```

### Constants

```python
HELIX_NUCLEATION_LENGTH    = 6
SHEET_NUCLEATION_LENGTH    = 3
KD_WINDOW_SIZE             = 9
TM_MIN_LENGTH              = 18
TM_MAX_LENGTH              = 30
TM_HYDROPATHY_THRESHOLD     = 1.6
TM_WINDOW_SIZE             = 19
TM_EXTENSION_THRESHOLD      = 0.8
DISORDER_WINDOW_SIZE       = 30
DISORDER_HYDROPATHY_MAX    = -0.5
DISORDER_CHARGE_THRESHOLD   = 0.2
```

### Data Tables

```python
CHOU_FASMAN_TABLE: dict[str, tuple[float, float, float, str]]
    # aa → (P_helix, P_sheet, P_turn, label)

KYTE_DOOLITTLE_SCALE: dict[str, float]
    # aa → hydropathy value [-4.5, +4.5]

HELIX_FORMERS: set[str]   # {"A","L","M","E","Q","K","H"}
SHEET_FORMERS: set[str]   # {"V","I","Y","F","W","T","C"}
TURN_FORMERS: set[str]    # {"N","D","G","S","P"}
HELIX_BREAKERS: set[str]  # {"P","G","N","D","S"}
```

---

## 5. crispr — CRISPR Gene Editing

### Dataclasses

```python
@dataclass(slots=True)
class GuideRNA:
    spacer: str           # 20nt spacer sequence
    pam: str              # PAM sequence
    cas_variant: str      # "SpCas9"/"SaCas9"/"Cas12a"
    target_dna: str       # target DNA
    cut_position: int     # cut position

@dataclass(slots=True)
class EditResult:
    edit_type: str        # "NHEJ"/"HDR"/"no_edit"
    edited_dna: str       # edited DNA
    indel_size: int       # indel size
    edit_position: int

@dataclass(slots=True)
class OffTargetSite:
    position: int
    mismatches: int
    sequence: str
    score: float
```

### Functions

```python
find_pam_sites(dna: str, cas_variant: str = "SpCas9"
               ) -> list[tuple[int, str]]
    # → [(position, pam_sequence), ...]

design_guide(target_dna: str, cas_variant: str = "SpCas9",
             position: int = 0, mode: str = "nearest") -> GuideRNA
    # mode: "nearest" (PAM closest to position, default) |
    #       "best" (max Rule Set 2 on-target score over all PAM sites)

on_target_score(guide: GuideRNA,
                model: str = "doench2016",
                method: str | None = None) -> float
    # → [0.0, 1.0]
    # method: "doench_2016" (Rule Set 2, default) | "legacy" (simplified)

off_target_score(guide: GuideRNA, genome: str,
                 max_mismatches: int = 4
                 ) -> list[OffTargetSite]

cut_dna(dna: str, guide: GuideRNA,
        repair: str = "NHEJ",
        rng: random.Random | None = None
        ) -> EditResult

edit_gene(dna: str, target_position: int, new_sequence: str,
          method: str = "HDR"
          ) -> str
    # → edited_dna
```

### Configuration

```python
CAS_VARIANTS: dict[str, dict]
    # "SpCas9":  {pam: "NGG",     spacer_length: 20, ...}
    # "SaCas9":  {pam: "NNGRRT",  spacer_length: 21, ...}
    # "Cas12a":  {pam: "TTTV",    spacer_length: 23, ...}

NHEJ_INDEL_SPECTRUM: dict[str, float]   # indel type → frequency
HDR_EFFICIENCY: dict[str, float]          # {"typical": 0.05, "high": 0.10, ...}
```

---

## 6. epigenetics — Epigenetics

### Dataclasses

```python
@dataclass(slots=True)
class MethylationState:
    positions: dict[int, float]    # position → methylation probability [0,1]
    methylase: str                   # "dam"/"dcm"/"cpg"/"custom"
    total_sites: int
    methylated_sites: int

@dataclass(slots=True)
class HistoneMark:
    position: int
    mark: str        # "H3K4me3"/"H3K27me3"/"H3K36me3"/"H3K9me3"/"H3K27ac"
    level: float      # 0-1

@dataclass(slots=True)
class ChromatinState:
    methylation: MethylationState
    histone_marks: list[HistoneMark]
    chromatin_accessibility: dict[int, float]
    expression_modifier: dict[str, float]
```

### Functions

```python
find_dam_sites(dna: str) -> list[int]      # GATC sites
find_dcm_sites(dna: str) -> list[int]      # CCWGG sites
find_cpg_sites(dna: str) -> list[int]      # CG sites
find_cpg_islands(dna: str, min_length: int = 200,
                 gc_min: float = 0.55,
                 oe_min: float = 0.65
                 ) -> list[tuple[int, int]]

methylate_dna(dna: str, positions: list[int],
              methylase: str = "custom",
              level: float = 1.0
              ) -> MethylationState

add_histone_marks(positions: list[int], marks: list[str],
                   levels: list[float] | None = None
                   ) -> list[HistoneMark]

calculate_accessibility(methylation: MethylationState,
                        histone_marks: list[HistoneMark]
                        ) -> dict[int, float]

calculate_expression_modifier(methylation: MethylationState,
                              histone_marks: list[HistoneMark],
                              gene_positions: dict[str, tuple[int, int]]
                              ) -> dict[str, float]
```

### Configuration

```python
HISTONE_MARK_TYPES: dict[str, dict]
    # "H3K4me3":  {effect: "activating",      score: +0.5}
    # "H3K27me3": {effect: "repressing",      score: -0.7}
    # "H3K36me3": {effect: "elongation",      score: +0.3}
    # "H3K9me3":  {effect: "heterochromatin",  score: -0.9}
    # "H3K27ac":  {effect: "activating",       score: +0.6}
```

---

## 7. evolution — Evolution Engine

### Dataclasses

```python
@dataclass
class EvolutionConfig:
    substitution_rate: float = 2.2e-10
    indel_rate: float = 4.5e-11
    transition_transversion_ratio: float = 2.0
    population_size: int = 1000
    generations: int = 100
    recombination_rate: float = 0.01
    seed: int | None = None

@dataclass
class Individual:
    dna: str
    fitness: float = 0.0
    mutations: list = field(default_factory=list)
    generation: int = 0
```

### Functions

```python
mutate(dna: str, config: EvolutionConfig,
       rng: random.Random | None = None
       ) -> str
    # → mutated_dna

mutate_batch(dna_list: list[str], config: EvolutionConfig
             ) -> list[str]

select(population: list[Individual],
       population_size: int,
       rng: random.Random | None = None
       ) -> list[Individual]
    # Wright-Fisher fitness-proportional sampling

recombine(parent1: str, parent2: str,
          rng: random.Random | None = None
          ) -> tuple[str, str]
    # homologous recombination (single crossover point)

calculate_fitness(dna: str, mode: str = "hamming",
                 target: str | None = None
                 ) -> float
    # mode: "hamming"/"cai"/"gc"/"custom"

fitness_landscape(dna: str, landscape: dict[int, float]
                  ) -> float

dnds_ratio(reference: str, query: str
           ) -> tuple[float, float]
    # → (dN, dS) nonsynonymous/synonymous substitution rates
```

### Constants

```python
E_COLI_SUBSTITUTION_RATE     = 2.2e-10
E_COLI_INDEL_RATE            = 4.5e-11
TRANSITION_TRANSVERSION_RATIO = 2.0
E_COLI_NE     = 1.3e8     # effective population size
HUMAN_NE      = 1e4
DROSOPHILA_NE = 1e6
ARABIDOPSIS_NE = 4e5
```

---

## 8. grn — Gene Regulatory Network

```python
from helixlang.plugins.runtime.grn import GRN, decay_from_half_life_ticks

class GRN:
    DECAY ≈ 0.994           # universal decay from the 110-min protein half-life
                            # decay_from_half_life_ticks(110) (Mosteller 1980,
                            # Helbig 2011); genes without an explicit decay=
                            # default to this.

    def __init__(self, noise_enabled: bool = False,
                 noise_seed: int | None = None)
        # noise_enabled: per-gene two-state (telegraph) intrinsic noise,
        # zero-mean Fano-scaled; deterministic default keeps the mean
        # trajectory unchanged

    def add_gene(self, name: str, threshold: float,
                 initial_level: float = 0.0,
                 decay: float | None = None,
                 hill_n: float | None = None,
                 kd: float | None = None,
                 noise: TelegraphPromoter | None = None) -> None
    def add_edge(self, source: str, target: str, weight: float) -> None
    def step(self) -> list[str]
        # → names of genes triggered this tick (level > 0.5)
    def set_level(self, name: str, level: float) -> None

decay_from_half_life_ticks(half_life_ticks: float) -> float
    # per-tick decay coefficient for a given protein half-life
```

Noise model (see §stochastic): `noise=` is a `TelegraphPromoter` whose
stationary variance `Fano·mean/expression_scale` (normalized units) is
applied as zero-mean additive noise, so the deterministic mean is
preserved and existing tests stay green. With `noise_enabled=True` and
no per-gene promoter, a default constitutive-noise promoter is used.

---

## 9. units — Physical Unit System

```python
from helixlang.core.units import (
    TIME_TICK_MIN, TIME_TICK_S, LATTICE_SPACING_UM,
    ATP_PER_GLUCOSE, PROTEIN_HALF_LIFE_MEDIAN_TICKS,
    AI2_DIFFUSION_UM2_S, DIFFUSION_DT_S,
    ticks_to_min,
    diffusion_to_lattice, diffusion_lattice_to_dx,
    decay_from_half_life_ticks, decay_to_half_life_ticks,
)

TIME_TICK_MIN              = 1.0     # 1 tick = 1 minute (Neidhardt 1996)
TIME_TICK_S                = 60.0
LATTICE_SPACING_UM         = 10.0    # lattice site edge
ATP_PER_GLUCOSE            = 38      # ATP per glucose (Alberts)
PROTEIN_HALF_LIFE_MEDIAN_TICKS = 110.0   # E. coli median protein half-life
AI2_DIFFUSION_UM2_S        = 100.0   # AI-2 diffusion coefficient (Miller & Bassler 2001)
DIFFUSION_DT_S             = 60.0    # diffusion time step per tick

ticks_to_min(ticks: float) -> float
diffusion_to_lattice(D_um2_s, dt_s, dx_um) -> float
diffusion_lattice_to_dx(D_um2_s, dt_s, D_lattice) -> float
decay_from_half_life_ticks(half_life_ticks) -> float
decay_to_half_life_ticks(decay) -> float
```

Physical units are always on (no `units=` / `calibrated=` switch): energy counts
are ATP molecules (cell defaults in `cell.py`: newborn `1e9`, division
`1.8e9`; population defaults in `population.py`: quorum `10.0` µM AI-2,
diffusion `100.0` µm²/s, emission `2.0` µM/tick), signals are µM, one tick is
one minute. The legacy `CALIBRATED` registry and its conversion functions
(`energy_to_atp`, `signal_to_um`) were removed.

---

## 10. reaction_diffusion — Reaction-Diffusion

```python
from helixlang.plugins.runtime.reaction_diffusion import (
    GrayScott, PRESETS,
)

class GrayScott:
    def __init__(self, size: int = 64,
                 F: float = 0.035, k: float = 0.065,
                 Du: float = 0.16, Dv: float = 0.08)
    def step(self, n_steps: int = 1) -> None
    def get_state(self) -> tuple[np.ndarray, np.ndarray]
        # → (u, v) concentration fields

PRESETS: dict[str, dict]   # 14 classic parameter sets (Pearson 1993)
```

---

## 11. lsystem — L-System

```python
from helixlang.plugins.runtime.lsystem import LSystem

class LSystem:
    def __init__(self, axiom: str,
                 rules: dict[str, str],
                 angle: float = 25.0,
                 step: float = 1.0)
    def iterate(self, n: int) -> str
        # → rewritten string
    def render(self, n: int) -> list[tuple[float, float, float]]
        # → [(x, y, angle), ...] path points
```

---

## 12. stochastic — Two-State Promoter Noise

```python
from helixlang.plugins.runtime.stochastic import (
    telegraph_fano_factor, TelegraphPromoter,
    fano_to_noise_std, gillespie_telegraph,
)

telegraph_fano_factor(k_on, k_off, burst_size, degradation_rate) -> float
    # steady-state Fano factor (variance/mean) of the two-state
    # (telegraph) promoter; == 1 in the constitutive/Poisson limit
    # (Jones 2014; Rijal 2025)

@dataclass(frozen=True, slots=True)
class TelegraphPromoter:
    k_on: float          # OFF -> ON rate (1/min)
    k_off: float         # ON -> OFF rate (1/min)
    burst_size: float    # mean transcripts per ON interval
    degradation_rate: float = 0.14   # 5-min mRNA half-life rate (Bernstein 2002)
    expression_scale: float = 100.0  # copy number at level = 1.0

    @property
    def transcription_rate(self) -> float   # r = b * k_off
    @property
    def on_fraction(self) -> float           # k_on/(k_on+k_off)
    def fano_factor(self) -> float

fano_to_noise_std(fano, mean, decay, expression_scale=100.0) -> float
    # zero-mean noise std that reproduces steady-state Fano factor in the
    # discrete-time AR(1) GRN update (deterministic mean preserved)

gillespie_telegraph(k_on, k_off, burst_size, degradation_rate,
                    t_max, n_replicates=2000, seed=None) -> dict
    # exact continuous-time Gillespie SSA; → {"mean", "variance", "fano"}
    # of mRNA counts at t_max across independent runs
```

This module backs `grn.py`'s optional intrinsic noise and the
`PopulationConfig.noise_enabled`/`noise_seed` toggle for per-cell GRNs
in `population.py`; it is stdlib-only (no numpy).

---

## 13. environment — Diffusing Nutrient Fields

Extracellular medium with physical diffusion coefficients and Monod
uptake; the diffusion scheme is the same flux-conservative sub-stepped
5-point Laplacian used for the AI-2 field (`units.py` conversion,
`D_lattice ≤ 0.25` per sub-step, zero-flux boundaries).

```python
from helixlang.plugins.runtime.environment import (
    GLUCOSE_DIFFUSION_UM2_S, OXYGEN_DIFFUSION_UM2_S, ACETATE_DIFFUSION_UM2_S,
    GLUCOSE_HALF_SATURATION_MM, OXYGEN_HALF_SATURATION_MM,
    BULK_GLUCOSE_MM, BULK_OXYGEN_MM, SITE_VOLUME_L,
    monod_uptake, michaelis_menten_rate,
    molecules_per_site, atp_yield,
    ConcentrationField, EnvironmentConfig, Environment,
)

GLUCOSE_DIFFUSION_UM2_S = 600.0     # Stewart 2003; CRC Handbook
OXYGEN_DIFFUSION_UM2_S  = 2500.0    # CRC Handbook
ACETATE_DIFFUSION_UM2_S = 1200.0    # CRC Handbook (small organic acid)
GLUCOSE_HALF_SATURATION_MM = 0.1    # Ks, Kovárová-Kovar & Egli 1998
OXYGEN_HALF_SATURATION_MM  = 0.05
BULK_GLUCOSE_MM = 1.0               # rich-medium glucose-equivalent
BULK_OXYGEN_MM  = 0.21              # air-saturated water at 25 °C
SITE_VOLUME_L   = 1e-12             # (10 µm)^3

monod_uptake(v_max, substrate_concentration, half_saturation) -> float
    # v_max·S/(Ks+S)   (Monod 1949; Kovárová-Kovar & Egli 1998)
michaelis_menten_rate(v_max, substrate_concentration, km) -> float
    # alias of monod_uptake (Michaelis & Menten 1913)
molecules_per_site(concentration_mm) -> float
    # ≈6.02e8 molecules at 1 mM in a (10 µm)^3 site
atp_yield(glucose_molecules) -> float
    # ×38 ATP/glucose (Alberts)

class ConcentrationField:
    def __init__(self, name, width, height,
                 diffusion_um2_s, initial_concentration=0.0)
    def get(self, x, y) -> float          # mM at (x, y)
    def set(self, x, y, value) -> None
    def add(self, x, y, amount) -> None
    def deplete(self, x, y, amount) -> float   # returns actually removed
    def snapshot(self) -> list[list[float]]    # grid [y][x]
    def diffuse(self) -> None                  # one tick, sub-stepped
    def total_mm(self) -> float                # mM × sites

@dataclass(slots=True)
class EnvironmentConfig:
    width: int = 100
    height: int = 100
    flow_rate: float = 0.0              # chemostat per-tick volume fraction
    bulk_glucose_mm: float = BULK_GLUCOSE_MM
    bulk_oxygen_mm: float = BULK_OXYGEN_MM
    glucose_diffusion_um2_s: float = GLUCOSE_DIFFUSION_UM2_S
    oxygen_diffusion_um2_s: float = OXYGEN_DIFFUSION_UM2_S
    glucose_initial_mm: float = BULK_GLUCOSE_MM
    oxygen_initial_mm: float = BULK_OXYGEN_MM

class Environment:
    def __init__(self, config=EnvironmentConfig())
    # fields: glucose, oxygen (ConcentrationField), fields: dict[str, ...]
    def add_field(self, name, field) -> None
    def get_field(self, name) -> ConcentrationField
    def step(self) -> None               # diffuse + chemostat refresh
    def substrate_at(self, x, y, name="glucose") -> float
    def local_uptake(self, x, y, name="glucose",
                     half_saturation=None, v_max=1.0) -> float
```

`DynamicFluxBalance` (see §3) couples to the environment through
`update_from_environment`/`apply_to_environment`, depositing overflow
acetate into a `"acetate"` field created on first use.

---

## Error Hierarchy

```python
from helixlang.core.errors import (
    HelixError,            # base class
    LexError,              # lexical error
    ParseError,            # syntax error
    SemanticError,         # semantic error
    CompileError,          # compilation error
    RegulationError,       # regulatory-network error
    RuntimeHelixError,     # runtime error
)
```
