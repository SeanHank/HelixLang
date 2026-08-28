# Bio Modules in Detail

> **2026-08-28 — Legacy import paths updated** for the doc/36 plugin re-layout (flat `helixlang.X` -> `helixlang.core.*`/`helixlang.plugins.runtime.*`).

> HelixLang's biological function modules are implemented based on real literature data, covering six major domains: the central dogma, metabolic networks, protein structure, gene editing, epigenetics, and evolution.

---

## Table of Contents

1. [Central dogma (central_dogma)](#1-central-dogma-central_dogma)
2. [Metabolic FBA (metabolism)](#2-metabolic-fba-metabolism)
3. [Protein structure (protein_structure)](#3-protein-structure-protein_structure)
4. [CRISPR gene editing (crispr)](#4-crispr-gene-editing-crispr)
5. [Epigenetics (epigenetics)](#5-epigenetics-epigenetics)
6. [Evolution engine (evolution)](#6-evolution-engine-evolution)
7. [Real biological data (bio_data)](#7-real-biological-data-bio_data)

---

## 1. Central Dogma (central_dogma)

**Module file**: `src/helixlang/plugins/runtime/central_dogma.py`

Models the complete information flow from DNA to protein: transcription → translation → mRNA degradation.

### Data Sources

| Parameter | Value | Literature |
|---|---|---|
| Transcription elongation rate | 50 nt/s | Proshkin 2010 Nature 458:507 |
| Translation elongation rate | 20 aa/s | Ingolia 2009 Science 324:218 |
| mRNA half-life | 5 min | Bernstein 2002 J Bacteriol 184:6477 |
| poly-A tail length | 15 nt | Mohanty & Kushner 2006 |
| tRNA abundance | full coverage of 64 codons | Dong 1996 J Mol Biol 260:649 |

### Core Components

```
transcribe(dna) → Transcript           # transcription: DNA → mRNA
translate(transcript) → TranslationResult  # translation: mRNA → protein
calculate_mrna_level(transcript, t) → float  # mRNA degradation kinetics
coupled_transcription_translation(dna) → dict  # coupled model
```

### Usage Example

```python
from helixlang.plugins.runtime.central_dogma import transcribe, translate

# transcription
dna = "ATGGCTGGTTAA"          # ATG=M, GCT=A, GGT=G, TAA=stop
transcript = transcribe(dna)
print(transcript.cds)          # "AUGGCUUAA"
print(transcript.elongation_time_s)  # ~0.24s (12nt / 50nt/s)

# translation
result = translate(transcript)
print(result.protein)          # "MAG"
print(result.elongation_time)  # ~0.15s (3aa / 20aa/s)
```

### mRNA Degradation Kinetics

mRNA concentration follows first-order kinetics:

```
[mRNA](t) = [mRNA]_ss × (1 - e^(-k·t))
```

where `k = ln(2) / half_life`, `[mRNA]_ss = synthesis_rate / k`.

---

## 2. Metabolic FBA (metabolism)

**Module file**: `src/helixlang/plugins/runtime/metabolism.py`

Flux Balance Analysis: uses linear programming to solve for optimal metabolic fluxes.

### Data Sources

- E. coli core metabolic model: Orth 2010 Mol Syst Biol 6:390 (iJO1366 simplified)
- 37 reactions covering glycolysis, TCA, PPP, fermentation, biomass
- Pure-Python two-phase simplex method (no scipy dependency)
- dFBA batch kinetics: Mahadevan 2002 Biophys J 83(3):1331–1340 (static optimization approach)

### Core Components

| Component | Description |
|---|---|
| `Reaction` | a single biochemical reaction (id/stoichiometry/bounds/subsystem) |
| `MetabolicModel` | metabolic network (set of reactions + stoichiometry matrix) |
| `ECOLI_CORE_MODEL` | prebuilt E. coli core model |
| `FluxBalanceAnalysis` | FBA solver |
| `simplex()` | pure-Python simplex solver |
| `DynamicFBAConfig` | batch-culture parameters (dt, initial biomass/glucose/acetate, Ks, μ conversion) |
| `DynamicFluxBalance` | dynamic FBA: MM uptake bound per step + forward-Euler batch integration |

### Usage Example

```python
from helixlang.plugins.runtime.metabolism import ECOLI_CORE_MODEL, FluxBalanceAnalysis

fba = FluxBalanceAnalysis(ECOLI_CORE_MODEL)
fba.set_uptake("GLC", 10.0)       # glucose uptake 10 mmol/gDW/h
fluxes = fba.solve(objective="biomass")
report = fba.analyze()

print(f"Biomass: {report['biomass_yield']:.4f}")
print(f"Glucose uptake: {report['glucose_uptake']:.2f}")
print(f"CO2: {report['byproduct_secretion']['co2']:.2f}")
```

### Dynamic Flux Balance Analysis (dFBA)

Batch-culture simulation (Mahadevan 2002). Each step sets the glucose
uptake bound from the external substrate via Monod/Michaelis-Menten,
solves the instantaneous FBA LP, and integrates the batch ODEs with
forward Euler:

```
v_glc(t) = v_max·S(t)/(Ks+S(t))     dX/dt = μ·X
dS/dt = −v_glc·X                     dP/dt = v_secret·X
```

```python
from helixlang.plugins.runtime.metabolism import DynamicFluxBalance, DynamicFBAConfig

d = DynamicFluxBalance(config=DynamicFBAConfig(dt_h=0.25,
                                               initial_glucose_mm=10.0))
hist = d.run()                       # until glucose exhausted / growth stalls
d.last()                             # → {time, biomass, glucose, growth_rate,
                                     #    glucose_uptake, acetate, lactate, co2}
d.growth_rate                        # latest specific growth rate (1/h)
```

`update_from_environment(env)`/`apply_to_environment(env)` couple the
batch pools to `helixlang.plugins.runtime.environment` fields (glucose in, overflow
acetate out — see §environment coupling under `population.py`). The
reduced 37-reaction core has no glyoxylate shunt, so overflow acetate is
not re-consumed: the fermentative phase and glucose-exhaustion arrest of
the classic diauxic shift are reproduced; a full model with the shunt
consumes acetate automatically once glucose is gone.

### Metabolic Network Structure

```
Exchange reactions:  EX_glc, EX_lac, EX_ac, EX_co2, EX_biomass
Glycolysis:    GLK → PGI → PFK → FBA → TPI → GAPD → PGK → PGM → ENO → PYK
TCA:       PDH → CS → ACONT → ICDH → AKGDH → SUCDHi → MDH (+ PPC)
PPP:       G6PDH → PGD → RPI
Fermentation:      LDH, PTA_ACK
Respiration:      NADH_OX (P/O=1.5), FADH2_OX (P/O=0.5)
Biomass:    BIOMASS, ATPM (maintenance)
```

### Mass Balance Constraints

Steady-state condition: `S · v = 0` (production rate = consumption rate for each metabolite)

---

## 3. Protein Structure (protein_structure)

**Module file**: `src/helixlang/plugins/runtime/protein_structure.py`

Pure-Python protein structure prediction: secondary structure + transmembrane helices + intrinsically disordered regions.

### Data Sources

| Method | Data | Literature |
|---|---|---|
| Chou-Fasman secondary structure | 20 aa propensity table | Chou & Fasman 1978 Adv Enzymol 47:45 |
| Kyte-Doolittle hydropathy | 20 aa hydropathy values | Kyte & Doolittle 1982 J Mol Biol 157:105 |
| Transmembrane helix prediction | two-threshold method | Krogh 2001 J Mol Biol 305:567 |
| Intrinsic disorder prediction | charge-hydropathy | Dunker 2001, Uversky 2000 |

### Core Functions

| Function | Purpose |
|---|---|
| `predict_secondary(seq)` | Chou-Fasman secondary structure prediction |
| `hydropathy_profile(seq)` | Kyte-Doolittle hydropathy profile |
| `gravy(seq)` | GRAVY (overall hydropathy) |
| `predict_transmembrane(seq)` | transmembrane helix prediction (two-threshold method) |
| `predict_disorder(seq)` | intrinsically disordered region prediction |
| `predict_structure(seq)` | complete structure prediction report |

### Usage Example

```python
from helixlang.plugins.runtime.protein_structure import predict_structure

report = predict_structure("MKVLAACDEFGHIKLMNPQRSTVWY" * 3)
print(report.summary)
# length=72 | helix=22.2% | sheet=15.3% | turn=12.5% | coil=50.0%
# | GRAVY=-1.22 | TM=0 | disorder=0.0%
print(report.helix_fraction)
print(report.transmembrane_helices)
print(report.disorder_regions)
```

### Chou-Fasman Algorithm

1. **Helix nucleation**: ≥6 consecutive helix formers (P_a > 1.0)
2. **Helix extension**: extend at both ends while the 4-residue window average P_a ≥ 1.0
3. **Sheet nucleation**: ≥3 consecutive sheet formers (P_b > 1.0)
4. **Turn recognition**: a 4-residue fragment with average P_turn > 1.0 and containing Pro/Gly
5. **Pro forced break**: Pro positions are never marked H

### Transmembrane Two-Threshold Prediction

- **Extension threshold** (0.8): consecutive segments ≥ this threshold are candidates
- **Nucleation threshold** (1.6): candidate segments with a peak ≥ this threshold are confirmed as TM
- **Length constraint**: 18 ≤ L ≤ 30 (allows +5 flanks)

---

## 4. CRISPR Gene Editing (crispr)

**Module file**: `src/helixlang/plugins/runtime/crispr.py`

CRISPR-Cas gene editing model: sgRNA design + cleavage + off-target prediction + editing outcome.

### Data Sources

| Parameter | Value | Literature |
|---|---|---|
| SpCas9 PAM | NGG | Jinek 2012 Science 337:816 |
| SaCas9 PAM | NNGRRT | Ran 2015 Nature 526:113 |
| Cas12a PAM | TTTV | Zetsche 2015 Cell 163:759 |
| On-target efficiency | Doench 2016 model | Doench 2016 Nat Biotechnol 34:184 |
| Off-target score | Hsu 2013 model | Hsu 2013 Nat Biotechnol 31:827 |
| NHEJ indel spectrum | 7 indel classes | Paixão 2022 Nat Commun 13 |
| HDR efficiency | 1-10% | Heyer 2010 |

### Core Functions

| Function | Purpose |
|---|---|
| `find_pam_sites(dna, cas)` | search for PAM sites |
| `design_guide(target, cas)` | design sgRNA |
| `on_target_score(guide)` | on-target efficiency score |
| `off_target_score(guide, genome)` | off-target score |
| `cut_dna(dna, guide)` | simulate Cas cleavage |
| `edit_gene(dna, pos, new_seq)` | gene editing (NHEJ/HDR) |

### Usage Example

```python
from helixlang.plugins.runtime.crispr import design_guide, cut_dna, edit_gene

target = "ATCGATCGATCGATCGATCGGATC"  # contains an NGG PAM
guide = design_guide(target, cas_variant="SpCas9")
print(f"Guide spacer: {guide.spacer}")
print(f"On-target score: {on_target_score(guide):.3f}")

# simulate cleavage
cut_result = cut_dna(target, guide, repair="NHEJ")
print(f"Edit type: {cut_result.edit_type}")

# HDR precise editing
edited = edit_gene(target, target_position=10, new_sequence="GGGG")
```

---

## 5. Epigenetics (epigenetics)

**Module file**: `src/helixlang/plugins/runtime/epigenetics.py`

DNA methylation + histone modification models.

### Data Sources

| Modification | Site | Literature |
|---|---|---|
| Dam methylation | GATC | Marinus 1973 MGG 115:248 |
| Dcm methylation | CCWGG | Marinus 1984 |
| CpG methylation | CG | Bird 2002 Cell 109:1 |
| H3K4me3 | active promoters | +0.5 |
| H3K27me3 | Polycomb repression | -0.7 |
| H3K9me3 | heterochromatin | -0.9 |
| CpG islands | GC>55%, >200bp, o/e>0.65 | Takai 2002 |

### Core Functions

| Function | Purpose |
|---|---|
| `find_dam_sites(dna)` | search for GATC sites |
| `find_dcm_sites(dna)` | search for CCWGG sites |
| `find_cpg_sites(dna)` | search for CpG sites |
| `find_cpg_islands(dna)` | identify CpG islands |
| `methylate_dna(...)` | DNA methylation |
| `add_histone_marks(...)` | add histone modifications |
| `calculate_accessibility(...)` | chromatin accessibility |
| `calculate_expression_modifier(...)` | gene expression modifier |

### Usage Example

```python
from helixlang.plugins.runtime.epigenetics import (
    find_cpg_sites, methylate_dna, add_histone_marks,
    calculate_expression_modifier,
)

dna = "ATCGCGATCGCGATCGCGAT"
sites = find_cpg_sites(dna)
meth = methylate_dna(dna, sites, methylase="cpg", level=0.8)
marks = add_histone_marks(positions=[5, 10], marks=["H3K4me3", "H3K27me3"])
mod = calculate_expression_modifier(meth, marks, gene_positions={"gene1": (0, 10)})
```

---

## 6. Evolution Engine (evolution)

**Module file**: `src/helixlang/plugins/runtime/evolution.py`

Wright-Fisher evolution model with mutation + selection + drift + recombination.

### Data Sources

| Parameter | Value | Literature |
|---|---|---|
| E. coli substitution rate | 2.2e-10 /nt/gen | Lee 2012 Nature 489:527 |
| E. coli indel rate | 4.5e-11 /nt/gen | Lee 2012 |
| transition:transversion | 2:1 | Stoltzfus 2009 |
| E. coli Ne | 1.3e8 | Hartl & Clark 2007 |
| human Ne | 1e4 | Hartl & Clark 2007 |

### Core Components

| Component | Description |
|---|---|
| `EvolutionConfig` | evolution parameters (mutation rate/population size/selection coefficient) |
| `Individual` | an individual (DNA + fitness + mutation history) |
| `Population` | Wright-Fisher population |
| `mutate(dna, config)` | introduce substitution/indel |
| `select(population)` | natural selection (fitness-proportional sampling) |
| `recombine(p1, p2)` | homologous recombination |
| `calculate_fitness(dna, mode)` | fitness calculation |
| `dnds_ratio(ref, query)` | dN/dS |

### Usage Example

```python
from helixlang.plugins.runtime.evolution import EvolutionConfig, Population, mutate

config = EvolutionConfig(
    mutation_rate=2.2e-10,
    population_size=1000,
    generations=100,
)
pop = Population(
    initial_dna="ATG" + "CTG" * 100 + "TAA",
    config=config,
    target_dna="ATG" + "CTG" * 100 + "TAA",
    fitness_method="hamming",
)
pop.evolve(generations=100)
print(f"Best fitness: {pop.get_generation_stats()[-1]['max_fitness']:.4f}")
```

---

## 7. Real Biological Data (bio_data)

**Module file**: `src/helixlang/plugins/runtime/bio_data.py`

Centrally manages real biological data, replacing fabricated parameters.

### Data Contents

| Dataset | Description |
|---|---|
| `ECOLI_CODON_USAGE` | E. coli K-12 MG1655 codon frequencies |
| `YEAST_CODON_USAGE` | S. cerevisiae codon frequencies |
| `HUMAN_CODON_USAGE` | H. sapiens codon frequencies |
| `YEAST_TRNA_ABUNDANCE` | S. cerevisiae tRNA gene copy numbers (Chan & Lowe 2009) |
| `HUMAN_TRNA_ABUNDANCE` | H. sapiens tRNA abundance (Chan & Lowe 2009; Dittmar 2006) |
| `get_species_trna(species)` | get the species tRNA abundance table (ecoli/yeast/human) |
| PCR error rate | Saiki 1988, Potapov 2017 |
| Sequencing platform error rate | Ceze 2019 Nat Rev Genet |
| DNA synthesis error rate | Filges 2021 |
| DNA storage density | Goldman 2013, Erlich 2017 |
| DNA decay | Allentoft 2012, Grass 2015 |

### Usage Example

```python
from helixlang.plugins.runtime.bio_data import ECOLI_CODON_USAGE, HUMAN_CODON_USAGE, get_species_trna

# E. coli's most-used Leu codon
print(ECOLI_CODON_USAGE["CTG"])  # ('L', 48.4, 0.47)

# human prefers GC-ending codons
print(HUMAN_CODON_USAGE["CTG"])  # ('L', 39.8, 0.40)

# multi-species tRNA abundance (for translation rate calculation)
trna_ecoli = get_species_trna("ecoli")   # CTG → 3500 (Dong 1996)
trna_yeast = get_species_trna("yeast")   # CTG → 14 (Chan & Lowe 2009)
trna_human = get_species_trna("human")   # CTG → 20 (Dittmar 2006)
```

---

## Inter-Module Dependencies

```
bio_data ←── central_dogma (codon frequency tables)
         ←── evolution (CAI fitness)

central_dogma ←── cell (cellular simulation uses transcription/translation)

metabolism ←── environment (dFBA: glucose bound from field, acetate back into field)
metabolism (independent core, includes its own simplex solver)

population ←── environment (Monod uptake, diffusion, CROMICS crowding D)
           ←── grn (per-cell GRN) + vm (per-cell bytecode)
           ←── stochastic (telegraph noise toggle)
           ←── metabolism (DynamicFluxBalance coupling)

protein_structure (independent, pure-algorithm module)

crispr ←── bio_data (codon mapping)

epigenetics (independent, includes methylation/histone models)
```

All bio modules are pure-Python implementations with no external dependencies such as numpy/scipy (numpy is optional in evolution and population for accelerating population operations; when missing it automatically degrades to pure Python).
