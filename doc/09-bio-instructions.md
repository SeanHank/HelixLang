# Bio Instruction Guide

> Quick reference for the `.helix` source file annotation syntax + a guide to calling the Python bio modules.

---

## Table of Contents

1. [.helix annotation syntax](#1-helix-annotation-syntax)
2. [Genes and promoters](#2-genes-and-promoters)
3. [Gene regulatory network](#3-gene-regulatory-network)
4. [Morphogenesis (L-system)](#4-morphogenesis-l-system)
5. [Reaction-diffusion field](#5-reaction-diffusion-field)
6. [Run configuration](#6-run-configuration)
7. [Bio module Python API calls](#7-bio-module-python-api-calls)
8. [Complete example patterns](#8-complete-example-patterns)

---

## 1. .helix Annotation Syntax

HelixLang source files use `#` annotation blocks to describe gene structure, regulatory relationships, and simulation parameters. DNA triplets are separated by spaces.

### Supported Annotations

| Annotation | Purpose | Example |
|---|---|---|
| `#promoter` | define a promoter | `#promoter name=p1 strength=0.8` |
| `#gene` ... `#end` | define a gene (including the DNA sequence) | `#gene name=geneA promoter=p1` |
| `#regulate` | regulatory relationship | `#regulate geneA -> geneB strength=+0.5` |
| `#lsystem` | L-system morphogenesis | `#lsystem name=plant axiom=F rules=0:F->F[+F]F` |
| `#field` | reaction-diffusion field parameters | `#field size=32 F=0.035 k=0.065` |
| `#config` | run configuration | `#config ticks=100 output=stdout` |
| `#media` | growth medium (sim backends) | `#media nutrient=GLC concentration=10.0` |
| `#enzyme` | bind a gene to an FBA reaction | `#enzyme gene=glk reaction=HEX1` |
| `#metabolite` | initialise an intracellular pool | `#metabolite name=glc__D init=0.5` |
| `#sim` | long-tail backend hook | `#sim kind=spatial_dfba length=32` |

### Basic Structure

```
# comment line (starts with # but is not an annotation keyword)

#promoter name=<name> strength=<-1.0~1.0>

#gene name=<gene name> promoter=<promoter name>
ATG GCT GGT TAA
#end

#config ticks=<number of ticks> output=<output format>
```

`#config` keys are `ticks`, `output`, `table`, `ops_per_tick`, `react_steps`,
`use_central_dogma`, `species`, and `backend` (see `doc/02-language-spec.md`
§3.6). The classic runtime runs on **physical units** end-to-end: energy
counts are ATP molecules, signals are µM, diffusion is µm²/s, and one tick
is one minute. GRN decay defaults to the 110-min protein half-life
(≈ 0.994/tick), quorum fires at 10 µM AI-2, and the division threshold is
reachable in ~20 rich-medium minutes. See `doc/04-simulation-model.md` §6.3 and
`doc/02-language-spec.md` §3.6.

### DNA Triplet Rules

- each codon is 3 bases (A/T/G/C)
- `ATG` = start codon (START)
- `TAA` / `TAG` / `TGA` = stop codons (STOP)
- intermediate codons are mapped to opcodes by the translation table
- codons are separated by spaces

---

## 2. Genes and Promoters

### Promoters

```
#promoter name=p_strong strength=1.0    # strong promoter
#promoter name=p_weak   strength=0.1    # weak promoter
#promoter name=p_const  strength=-0.5   # negative value=constitutive expression
```

**Strength semantics**:
- `1.0`: maximum transcription initiation frequency
- `0.0`: fully silent
- `-0.5`: constitutive (unaffected by regulatory factors)
- range `[-1.0, 1.0]`

### Genes

```
#gene name=lacZ promoter=p_lac
ATG GCT GCT GCT TAA
#end
```

- `name`: gene name (optional; auto-named when omitted)
- `promoter`: associated promoter (optional)
- gene body = the codon stream between the `#gene` line and `#end`

### Bare DNA

DNA without a `#gene` is automatically wrapped as an anonymous gene:

```
ATG GCT GTA TAA
```

equivalent to:

```
#gene name=__anon_0
ATG GCT GTA TAA
#end
```

---

## 3. Gene Regulatory Network

### Regulatory Relationships

```
#regulate <source> -> <target> strength=<-1.0~+1.0>
```

- **positive value** = activation (promotes target expression)
- **negative value** = repression (suppresses target expression)
- the source/target can be a promoter or a gene

### Example: lac Operon

```
#promoter name=p_lac  strength=0.5
#promoter name=p_lacI strength=-0.5

#gene name=lacZ promoter=p_lac
ATG GCT GCT GCT TAA
#end

#gene name=lacI promoter=p_lacI
ATG GCT GCT TAA
#end

#regulate p_lacI -> lacI strength=+0.8
#regulate lacI   -> p_lac strength=-0.9
#regulate p_lac  -> lacZ strength=+0.9

#config ticks=20 output=csv
```

**Regulation logic**:
1. `p_lacI` constitutively expresses lacI (a repressor protein)
2. lacI represses `p_lac` (strength=-0.9)
3. `p_lac` is repressed → lacZ is not expressed
4. if lacI is inactivated by the inducer → `p_lac` is derepressed → lacZ is expressed

---

## 4. Morphogenesis (L-System)

```
#lsystem name=<name>
         axiom=<axiom>
         rules=<rules>
         angle=<angle>
         step=<step>
```

### Rule Syntax

```
rules=0:F->F[+F]F[-F]F;1:X->FX
```

- format: `<iteration>:<symbol>-><production>;<iteration>:<symbol>-><production>`
- multiple rules are separated by `;`
- multiple symbols are separated by `,`

### Example: Plant Growth

```
#lsystem name=plant
         axiom=F
         rules=0:F->F[+F]F[-F]F
         angle=25
         step=1.0
```

### Symbol Meanings

| Symbol | Action |
|---|---|
| `F` | move forward one step and draw |
| `f` | move forward one step without drawing |
| `+` | turn left |
| `-` | turn right |
| `[` | push stack (save the current state) |
| `]` | pop stack (restore the state) |
| `X`, `Y` | auxiliary symbols (no drawing) |

---

## 5. Reaction-Diffusion Field

```
#field size=<grid size>
       F=<feed rate>
       k=<kill rate>
       Du=<U diffusion rate>
       Dv=<V diffusion rate>
```

### Example: Turing Pattern

```
#field size=32 F=0.035 k=0.065 Du=0.16 Dv=0.08

#config ticks=100 output=stdout react_steps=2
```

### Preset Parameters

| Name | F | k | Pattern |
|---|---|---|---|
| Pearson α | 0.014 | 0.045 | spots |
| Pearson β | 0.020 | 0.050 | stripes |
| Pearson γ | 0.035 | 0.065 | holes |
| Solitions | 0.030 | 0.062 | solitons |
| Mazes | 0.029 | 0.057 | mazes |

---

## 6. Run Configuration

```
#config ticks=<number of ticks>
        output=<output format>
        table=<translation table>
        ops_per_tick=<instructions per tick>
        react_steps=<reaction-diffusion sub-steps>
        backend=<classic|whole_cell|population|fba|calibration|benchmark>
```

### Parameter Descriptions

| Parameter | Default | Description |
|---|---|---|
| `ticks` | 100 | number of simulation ticks |
| `output` | `stdout` | output format: `stdout` / `csv` / `png` / `json` |
| `table` | `standard` | translation table: `standard` / `mito_vertebrate` / `ciliate` |
| `ops_per_tick` | 64 | number of instructions executed per tick |
| `react_steps` | 1 | reaction-diffusion sub-steps per tick |
| `backend` | `classic` | runtime selector; see §6.1 (sim backends) |

### 6.1 Simulation backends

Setting `#config backend` switches the run from the classic bytecode VM to
the quantitative simulation library (`helixlang.sim_runtime`). The same
`#gene` / `#promoter` / `#regulate` declarations plus `#media` / `#enzyme`
/ `#metabolite` and `#config` sim keys drive the chosen simulator:

| Backend | What runs | `ticks` means |
|---|---|---|
| `classic` | the bytecode VM (default; unchanged) | ticks |
| `whole_cell` | `VirtualCell` — cell cycle, adder size control, protein maturation, enzyme-constrained FBA | minutes |
| `population` | `CellPopulation3D` — shared `Environment`, per-cell dFBA, colony observables | ticks |
| `fba` | `FluxBalanceAnalysis` (static) or `DynamicFluxBalance` (`dynfba=true`) | steps |
| `calibration` | recover the four hidden whole-cell parameters from synthetic observables | — |
| `benchmark` | the 4-gate calibrate→predict whole-cell benchmark | — |

Sim configuration uses `#config` keys that mirror the target dataclass
fields — e.g. `division_rule=adder`, `population_size=2000`, `dfba=true`,
`fba_model=core dynfba=true` — plus `seed=` for determinism and
`output=` for column selection. Example (population colony with per-cell
dFBA):

```
#media nutrient=GLC concentration=10.0 diffusion_um2_s=300
#media nutrient=O2  concentration=0.25 diffusion_um2_s=1600
#config backend=population
#config population_size=2000 grid_width=64 grid_height=64
#config dfba=true dfba_dt_h=0.1 dfba_oxygen_max_uptake=20.0
#config signaling=true signal_diffusion=0.3 signal_threshold=20.0
#config ticks=60
#config output=alive_count,core_oxygen_mm,edge_oxygen_mm,core_acetate_mm,edge_acetate_mm
```

Full key tables and coercion rules are in `doc/12-helix-language-wiring.md`
§6.2–6.3; new examples run as
`helixlang examples/32_colony_dfba.helix --csv`.

### Variable Translation Table Example

```
#gene name=morpheus
ATG TGA GCT TAA
#end

#config ticks=1 output=stdout table=standard
# standard table: TGA = Stop → ORF terminates immediately

#config ticks=1 output=stdout table=mito_vertebrate
# mitochondrial table: TGA = Trp → continues until TAA terminates
```

---

## 7. Bio Module Python API Calls

HelixLang's biological function modules (central dogma, metabolism, protein structure, CRISPR, epigenetics, evolution) are callable through the Python API. The metabolism module is additionally reachable from `.helix` source via `#config backend=fba` (§6.1) with `#media` / `#enzyme` annotations; the remaining modules below are Python-first.

### Central Dogma

```python
from helixlang.central_dogma import transcribe, translate, coupled_transcription_translation

dna = "ATG" + "CTG" * 100 + "TAA"
result = coupled_transcription_translation(dna, promoter_strength=1.0)
print(result["protein"])
print(result["mrna_steady_state"])
```

### Metabolic FBA

```python
from helixlang.metabolism import ECOLI_CORE_MODEL, FluxBalanceAnalysis

fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
fba.set_uptake("GLC", 10.0)
fluxes = fba.solve()
report = fba.analyze()
print(f"Biomass yield: {report['biomass_yield']:.4f}")
```

The same static solve (plus `DynamicFluxBalance` batches) runs in-language as:

```
#media nutrient=GLC concentration=10.0
#config backend=fba
#config fba_model=core output=BIOMASS,EX_glc,growth_rate_per_hour
#config dynfba=true fba_dt_h=0.25 fba_oxygen_max=20.0 fba_steps=32
```

### Protein Structure Prediction

```python
from helixlang.protein_structure import predict_structure

report = predict_structure("MKVLAACDEFGHIKLMNPQRSTVWY" * 3)
print(report.summary)
print(f"Helix: {report.helix_fraction:.1%}")
print(f"TM helices: {len(report.transmembrane_helices)}")
```

### CRISPR Gene Editing

```python
from helixlang.crispr import design_guide, cut_dna, on_target_score

target = "ATCGATCGATCGATCGATCGGATC"  # contains an NGG PAM
guide = design_guide(target, cas_variant="SpCas9")
score = on_target_score(guide)
result = cut_dna(target, guide, repair="NHEJ")
print(f"Edit type: {result.edit_type}")
```

### Epigenetics

```python
from helixlang.epigenetics import find_cpg_sites, methylate_dna, calculate_expression_modifier

dna = "ATCGCGATCGCGATC"
sites = find_cpg_sites(dna)
meth = methylate_dna(dna, sites, methylase="cpg", level=0.8)
mod = calculate_expression_modifier(meth, [], {"gene1": (0, 10)})
print(f"Expression modifier: {mod['gene1']:.3f}")
```

### Evolution Simulation

```python
from helixlang.evolution import EvolutionConfig, Population, mutate

config = EvolutionConfig(mutation_rate=1e-5, population_size=100)
pop = Population(
    initial_dna="ATG" + "CTG" * 50 + "TAA",
    config=config,
    target_dna="ATG" + "CTG" * 50 + "TAA",
    fitness_method="hamming",
)
pop.evolve(generations=50)
stats = pop.get_generation_stats()[-1]
print(f"Best fitness: {stats['max_fitness']:.4f}")
```

---

## 8. Complete Example Patterns

### Pattern 1: Gene Regulation (GRN)

```
# gene regulatory network: bistable switch
#promoter name=pA strength=-0.5
#promoter name=pB strength=-0.5

#gene name=geneA promoter=pA
ATG GCT GCT GCT GCT TAA
#end

#gene name=geneB promoter=pB
ATG GGT GGT GGT GGT TAA
#end

#regulate geneA -> pB strength=-0.9
#regulate geneB -> pA strength=-0.9

#config ticks=50 output=csv
```

### Pattern 2: Morphogenesis

```
# L-system + gene expression driving growth
#promoter name=p_grow strength=-0.3
#gene name=grow promoter=p_grow
ATG CTC TAA
#end

#regulate p_grow -> grow strength=+0.6

#lsystem name=plant axiom=F rules=0:F->F[+F]F[-F]F angle=25 step=1.0

#config ticks=5 output=stdout
```

### Pattern 3: Reaction-Diffusion

```
# Turing pattern formation
#promoter name=p_pigment strength=-0.4
#gene name=pigment promoter=p_pigment
ATG GAT GAT GAA TAA
#end

#regulate p_pigment -> pigment strength=+0.5

#field size=32 F=0.035 k=0.065 Du=0.16 Dv=0.08

#config ticks=100 output=stdout react_steps=2
```

### Pattern 4: Variable Translation Table

```
# the same DNA expresses differently under different translation tables
#gene name=morpheus
ATG TGA GCT TAA
#end

#config ticks=1 output=stdout table=mito
```

### Pattern 5: CRISPR Editing Concept

```
# CRISPR concept example: DNA + annotations describing the edit target
# this example shows the DNA context of a CRISPR edit
# the actual editing is performed via the Python API (helixlang.crispr)

#gene name=target_gene
ATG GCT GCT GCT GCT GCT GCT TAA
#end

#config ticks=1 output=stdout
```

---

## Codon to opcode Mapping Quick Reference

| Codon | Amino acid | Opcode (standard table) |
|---|---|---|
| ATG | Met (M) | OP_START |
| TAA | Stop | OP_HALT |
| TAG | Stop | OP_HALT |
| TGA | Stop | OP_HALT (standard) / Trp (mitochondrial) |
| GCT | Ala (A) | OP_BUILD (structural) |
| GGT | Gly (G) | OP_BUILD (membrane) |
| GTA | Val (V) | OP_MOVE |
| GAT | Asp (D) | OP_SIGNAL |
| GAA | Glu (E) | OP_SIGNAL |
| CTC | Leu (L) | OP_GROW |

> See `STANDARD_TABLE` in `src/helixlang/codon_table.py` for the complete mapping.

---

## 9. Bio Instruction Annotations (P0-1.1 Language Extension)

HelixLang supports writing bio instruction annotations directly in `.helix` source code, executed automatically by the VM's central-dogma pipeline (`use_central_dogma=true`) on every tick.

### Supported Instructions

| Annotation | Description | Required fields | Optional fields |
|---|---|---|---|
| `#crispr` | CRISPR-Cas gene editing | `target` | `position`, `new_sequence`, `cas` |
| `#evolve` | evolutionary mutation (one generation) | `target` | `mutation_rate`, `indel_rate` |
| `#methylate` | DNA methylation (represses expression) | `target` | `methylase` (dam/dcm/cpg) |
| `#histone` | histone modification | `target` | `mark` (H3K4me3/H3K27me3/...) |
| `#transcribe` | force transcription of the target gene | `target` | — |
| `#translate` | force translation of the target gene | `target` | — |
| `#quorum` | quorum sensing activation | `target` | `threshold`, `activate` |

### Type Annotations (P0-1.3 Type System)

| Annotation | Description | Format |
|---|---|---|
| `#type` | type annotation | `#type gene_name=TypeName` |

Supported types: `Protein`, `Signal`, `Float`, `Int`, `Bool`, `String`, `Gene`, `Record`, `Any`

### Central Dogma Pipeline Configuration (P0-1.2)

Set `use_central_dogma=true` in `#config` to enable the central-dogma pipeline:

```
#config ticks=100 use_central_dogma=true species=ecoli
```

After enabling, each tick executes:
1. process bio instructions (`#crispr` / `#evolve` / `#methylate` / ...)
2. coupled transcription-translation (DNA → mRNA → protein, based on Proshkin 2010 / Ingolia 2009)
3. morphology update and GRN feedback
4. snapshot output

`species` supports: `ecoli` (default) / `yeast` / `human`, affecting codon usage frequencies and tRNA abundance.

### Complete Example

```
#config ticks=50 use_central_dogma=true species=ecoli
#promoter name=lacp strength=0.8
#gene name=lacZ promoter=lacp
ATG GAT CAA ACG TTT GAA AGC GAT CCG GTG AAA GCG TAA
#end
#type lacZ=Protein
#evolve target=lacZ mutation_rate=0.01
#methylate target=lacZ methylase=dam
```

### Type Checking

Enabling type checking verifies symbol-reference integrity:

```python
from helixlang.lexer import Lexer
from helixlang.parser import Parser

tokens = list(Lexer(src).tokens())
prog = Parser(tokens, enable_type_check=True).parse()
# raises ParseError when type checking fails
```
