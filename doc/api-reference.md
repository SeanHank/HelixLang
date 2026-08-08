# API Reference

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
9. [reaction_diffusion — reaction-diffusion](#9-reaction_diffusion--reaction-diffusion)
10. [lsystem — L-system](#10-lsystem--l-system)

---

## 1. codon_table — Codon Table

```python
from helixlang.codon_table import (
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
                     degradation_rate: float | None = None
                     ) -> float

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
COUPLING_OFFSET_NT                     = 35     # transcription-translation offset
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

### Prebuilt Model

```python
ECOLI_CORE_MODEL: MetabolicModel   # ~24-reaction E. coli core metabolism
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
from helixlang.grn import GRN, RegulatoryLink

class GRN:
    def add_gene(self, name: str, basal_rate: float = 0.1) -> None
    def add_link(self, source: str, target: str,
                 strength: float, k: float = 1.0) -> None
    def step(self, dt: float = 0.1) -> dict[str, float]
        # → {gene: concentration}
    def get_state(self) -> dict[str, float]
```

---

## 9. reaction_diffusion — Reaction-Diffusion

```python
from helixlang.reaction_diffusion import (
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

## 10. lsystem — L-System

```python
from helixlang.lsystem import LSystem

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

## Error Hierarchy

```python
from helixlang.errors import (
    HelixError,            # base class
    LexError,              # lexical error
    ParseError,            # syntax error
    SemanticError,         # semantic error
    CompileError,          # compilation error
    RegulationError,       # regulatory-network error
    RuntimeHelixError,     # runtime error
)
```
