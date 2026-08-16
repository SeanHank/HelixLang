# From One Genome (ATCG) to Life and Environmental Change: Full-Pipeline Simulation

## Current-State Audit and Next-Step Design

Status: **assessment + design, Phases A–D landed** (2026-08). This document
is an audit of how far the
codebase already goes toward the end-to-end vision — *starting from the
complete genetic code (ATCG) of an organism and simulating its entire life and
the change of its environment* — and a concrete, gated plan for the remaining
gap. Facts below are anchored to `src/helixlang/` file:line references and to
the literature citations already used across the project (full list in §10).
The design plan in §5 is grounded in the multi-species / multi-population /
multi-environment modeling literature (mapping table in §5.2).

Target version: 2026.8.2 (45 `.helix` examples, 72 test files, quality gates
`ruff check src tests` + `mypy` + `pytest --cov-fail-under=80`).

Landing (Phases A–D): `apps/ecosystem.py` `Species`/`Patch`/`Ecosystem` spine
(`#sim kind=ecosystem`, `#species`, `#patch`), Levins metapopulation (L7),
source–sink, CENTURY litter/SOM pools, closed C/N budgets, Q10/DAMM
temperature dependence, diurnal/seasonal `ScalarField` forcing, photoautotrophy
with O₂/CO₂ drawdown, community FBA, event-driven `Scheduler` fast-forward,
the invasion-fitness evolution loop (`run_generations`), and the population
DBTL data loop (`apps/population_dbtl.py`, `#sim kind=population_dbtl`).
Phase-C genome completeness: GFF3 chromosome import (`load_chromosome`/
`Chromosome`, minus-strand + multi-segment CDS), a RegulonDB regulatory-map
import (`tf_map="regulondb"` + `parse_regulondb`), and a replicon model
(`RepliconSpec`, chromosome oriC/terC fork dosage + constant-copy plasmids).
Gate examples 41–45 (`examples/41_two_species_crossfeeding.helix` …
`44_diauxie_complete.helix`, `45_population_dbtl.helix`)
run in the examples audit; regression tests live in `tests/test_ecosystem.py`
(39 tests), `tests/test_population_dbtl.py` (6 tests),
`tests/test_whole_cell_scale.py` / `tests/test_genome_scale.py` /
`tests/test_virtual_cell.py` (GFF3, RegulonDB, replicon sections), plus the
Phase-B environment tests in `tests/test_environment.py` and the population
split-rate test in `tests/test_population.py`.

---

## 1. Vision and scope

HelixLang is built on the thesis **“DNA is the source, codons are the
mnemonics, the ribosome is the VM, and the cell is the runtime.”** The next
frontier is to close the loop that runs *back* from the environment to the
genome:

```
  complete genome (ATCG)
        │ ① DNA structure / codecs / CRISPR
        ▼
  gene expression (transcription · translation · folding)
        │ ② central dogma
        ▼
  regulatory state (GRN · epigenetics · noise)
        │ ③ regulation
        ▼
  metabolism (FBA / dFBA / enzyme capacity)
        │ ④ metabolism
        ▼
  cell physiology (energy · volume · cell cycle · division · death)
        │ ⑤ single-cell physiology
        ▼
  multicellular spatial structure (colony · 3D · mechanics · flow)
        │ ⑥ population structure
        ▼
  environment (nutrients · diffusion · flow · temperature · light · chemistry)
        │ ⑦ environment
        ▼
  ecosystem (multi-species interaction · food web · biogeochemistry)
        │ ⑧ ecology
        ▼
  selection pressure → genome change (mutation · recombination · drift)
        │ ⑨ evolution
        ▼
  environment change → next generation            (feedback, ⑩)
```

The current codebase implements ①–⑦ plus ⑨ in *isolated, high-fidelity
segments*; ⑧ does not exist; and the ⑩ feedback arm exists in only one
special-case module. This document audits each segment and designs the
integration that makes the full loop runnable from a single entry point.

---

## 2. Current-state audit (what exists today)

### 2.1 ① Genome (ATCG) and DNA structure

| Capability | Where | Facts |
|---|---|---|
| DNA ↔ HelixLang source | `biocodec.py:395` (`dna_to_helix`), `:463` (`helix_to_dna`) | any ATCG string round-trips to `.helix` source and back |
| DNA storage codecs | `dna_codec.py:603/656` (Goldman), `:848/953` (Erlich), `pcr_amplify:1106`, `synthesize_dna:1171`, `sequence_dna:1253`, `decay_dna:1304` | real storage/watermark lifecycle, incl. physical decay |
| CRISPR-Cas editing | `crispr.py`: `find_pam_sites:133`, `design_guide:211`, `on_target_score:385`, `off_target_score:660`, `PAMIndex:739`, `cut_dna:992`, `edit_gene:1055` | PAM search, gRNA design, on/off-target scoring, in-silico editing |
| Codon tables / translation bias | `codon_table.py:95/149/157` (standard, mito, ciliate); `bio_data.py` codon usage; `#sim kind=codon_usage` | "same DNA, different organism" via ISA variants |
| Whole-genome-scale GRN | `apps/genome_scale.py`: `build_genome:168` (default `DEFAULT_GENES=4300`, E. coli MG1655 ORF scale), `GenomeColony:310`, `knock_out:384`, `fba_biomass:396`, `expression_gated_biomass:425` | synthetic 4300-gene template, ~10⁴ sparse regulatory edges (Martínez-Antonio & Collado-Vides 2003; Martínez-Antonio et al. 2008), RegulonDB-scale hubs (`MASTER_REGULATORS = crp, fis, lrp, hns`) |
| Whole-cell FASTA import | `apps/whole_cell_scale.py`: `load_genome:129` (dict **or FASTA**, `>gene` records), `random_genome:177` | real ATCG text in → re-encoded CDS + RBS → transcribable genome |
| Essentiality screen | `apps/whole_cell_scale.py`: `predict_essentiality:243`, `essentiality_screen:288`, `single_gene_ko_protocol:375` | Feist 2007 / EcoCyc (Gerdes 2003) protocol; every core gene the reduced 37-reaction model represents faithfully matches EcoCyc glucose-minimal labels |
| DNA ↔ physiology data flow | `omics.py`: `ExpressionMatrix:66`, `expression_to_grn_states:273`, `expression_to_fba_bounds:338` | expression matrices drive GRN states and FBA bounds |
| Standards interop | `interop.py`: `sbml_to_model:79`, `sbol3_dumps:199`, `sbol3_loads:284` | SBML in, SBOL3 out |

**Segment verdict**: the strongest segment. Real ATCG can enter, be edited,
be stored, be translated, and drive a genome-scale network — **but** the
import path is CDS-oriented (per-gene FASTA), there is no chromosome/replicon
structure (origin/terminus, multiple replicons, plasmids) at the genome level,
and the 4300-gene regulatory template is *synthetic*, not a real imported
regulatory map.

### 2.2 ② Gene expression (central dogma)

| Capability | Where | Facts |
|---|---|---|
| Transcription/translation | `central_dogma.py`: `transcribe:401`, `translate:612`, `calculate_mrna_level:750`, `coupled_transcription_translation:798`, `ProteinPool:283` | real cited rates: elongation 50 nt/s (Proshkin 2010), translation 20 aa/s (Ingolia 2009), mRNA half-life median 5 min (Bernstein 2002), tRNA abundances (Dong 1996) |
| VM wiring | `vm.py:584` (`_transcribe_translate`); `#config use_central_dogma` (`ast_nodes.py:112`) | central dogma is an opt-in bytecode backend; the default path is GRN-triggered abstract proteins |
| Protein structure prediction | `protein_structure.py`: `predict_secondary:468`, `predict_secondary_gor:6611`, `predict_transmembrane:6761`, `predict_disorder:10000`, `predict_structure:10114` | secondary structure / TM helices / disorder from sequence |

**Segment verdict**: physically calibrated and tested, but expression is
mostly *per-gene and intracellular*; there is no protein secretion/import
machinery, no molecular transport between cells, and no post-translational
signalling beyond the AI-2 quorum field.

### 2.3 ③ Regulation

| Capability | Where | Facts |
|---|---|---|
| GRN | `grn.py`: `GRN:188`, `step:262`, `integrate_grn:545`, `decay_from_half_life_ticks:77`; sparse genome-scale path `sparse_grn.py`: `SparseGRN:84`, `step:249`, `step_budgeted:285` | per-cell independent GRNs (deep-copied at division); 4300-gene shared-matrix path with an active-gene budget (`DEFAULT_ACTIVE_BUDGET=512`, ~10% of genes active per Martínez-Antonio et al. 2008) |
| Noise / stochasticity | `stochastic.py`: `TelegraphPromoter:72`, `telegraph_fano_factor:42`, `gillespie_telegraph:146`; `fate_analysis.py` switching | Peccoud & Ycart 1995 two-state promoters; noise reproduces scalar statistics at genome scale (gate 5) |
| Epigenetics | `epigenetics.py`: `find_cpg_islands:152`, `methylate_dna:241`, `calculate_accessibility:382`, `calculate_expression_modifier:423` | methylation + histone access modifier on expression |
| Cell-fate analysis | `apps/fate_analysis.py`: `bistability_scan`, `switching_rate`, `critical_slowing_down` | Gardner 2000 toggle switch; Scheffer 2009 early-warning signals |

**Segment verdict**: rich. The main gap is *developmental*: nothing turns a
shared genome into distinct, self-organised cell types within one body —
roles are either uniform (all cells identical) or hard-coded externally
(consortium roles, `apps/consortium.py`).

### 2.4 ④ Metabolism

| Capability | Where | Facts |
|---|---|---|
| Static FBA | `metabolism.py`: `FluxBalanceAnalysis:1064`, `solve:1225`, `ECOLI_CORE_MODEL:342` | 37-reaction E. coli core (Orth 2010); SBML import (`interop.py:79`) |
| Dynamic FBA | `metabolism.py`: `DynamicFluxBalance:1443` | Mahadevan 2002 static-optimization dFBA; Michaelis–Menten uptake bound; acetate/CO₂ by-products; diauxie phase 1 |
| Enzyme capacity / metabolite pools | `metabolism.py`: `EnzymeCapacity:907`, `MetabolitePool:965`, `MetabolicProxy:1777` | Phase-4 GECKO-style kcat scaling (Sanchez 2017) |
| Per-cell dFBA in populations | `population.py`: `_step_dfba_metabolism:1390`, `_step_dfba_shared_batch:1527`, `_sync_acetate:1616` | one LP per cell, or one shared LP per site (surfin-FBA style); acetate cross-feeding within the clone (`acetate_switch`) |
| 1D spatial dFBA | `apps/spatial_dfba.py`: `SpatialDFBA:95` | plug-flow biofilm/reactor; **self-documented limitation**: no glyoxylate shunt, so classic acetate re-consumption cannot occur (`spatial_dfba.py:20-23`) |

**Segment verdict**: solid but small by design (37 reactions). The reduced
core cannot reproduce a true full metabolic network, lacks the glyoxylate
shunt (no diauxie phase 2), and has no phototrophy/chemolithotrophy — which
matters when the environment becomes the driver.

### 2.5 ⑤ Single-cell physiology

| Capability | Where | Facts |
|---|---|---|
| Integrated virtual cell | `virtual_cell.py`: `VirtualCell:301`, `step:660`; `encode_gene:147` | four-layer coupling (GRN → central dogma → FBA → energy budget), Karr-2012 style; 1 step = 1 min |
| Cell cycle & chromosome timing | `virtual_cell.py`: `_advance_replication:422`, `_divide_replication:458`, `_wants_division:633` | Cooper–Helmstetter C/D-phase alignment |
| Volume / adder control | `virtual_cell.py` + `cell.py:82` (`membrane_permeability`), `cell_body.py` | birth-volume + fixed increment adder (Taheri-Araghi 2015); real physical units |
| Folding / QC / turnover | `virtual_cell.py` (Phase 3, Balchin 2016) | cotranslational folding fraction, chaperone-mediated folding |
| Energy accounting | `cell.py:27` (newborn ≈ 10⁹ ATP), `virtual_cell.py` maintenance ≈ 2.5×10⁷ ATP/min (Orth 2010), `units.py` | ATP-molecule unit system; 1 tick = 1 min, site = 10 µm |

**Segment verdict**: the physical completeness roadmap of
`doc/15-whole-cell-realism.md` is implemented and gated (Phases 1–5). What is
missing at this layer is *intercellular physiology*: there is no secretion/
uptake of proteins, no inter-cell molecular flux, and no tissue/organ
abstraction above the single cell.

### 2.6 ⑥ Multicellular spatial structure

| Capability | Where | Facts |
|---|---|---|
| Population engine | `population.py`: `CellPopulation:570`, `step:747` (5-phase tick), `CellPopulation3D:2044` | per-cell GRN + DNA VM, division, signalling, mechanics, metabolism |
| Spatial mechanics | `population.py`: `_apply_mechanics:1159`, contact/rod mechanics `:1293`, 3D 26-neighbour `CellPopulation3D` | shoving/force/contact; rod-cell geometry (Design 6 L3) |
| Fluid mechanics | `flow.py`: `FlowField:53`, `channel_poiseuille:104`/`_3d:296`; `apps/lattice_boltzmann.py` (D2Q9), `lattice_boltzmann_3d.py` (D3Q19) | analytic Poiseuille + self-consistent LBM with cells as no-slip obstacles (Guo 2002 body force, Ladd 1994 momentum exchange); 2D 10⁴ sites ≈ 1.68 ms/tick, 3D 100×100×50 ≈ 205 ms/tick |
| Morphogenesis | `reaction_diffusion.py`: `GrayScott:27` (Pearson 1993); `apps/morphogen_gradient.py`: `MorphogenGradient:138` (Wolpert 1969; Basu 2005); `lsystem.py`; `morphology_3d.py`; `CellPopulation3D.to_lsystem3d:2432` | Turing patterns, French-flag positional information, L-system morphology export |
| dFBA spatial stratification | `population.py`: `colony_observables:1848`, `dfba_stratification:1892` | core/edge O₂ and acetate stratification in colonies |

**Segment verdict**: impressive but **single-species by construction** — every
cell is a clone of one `.helix`/one genome; there is no `species`/`strain`
field on `PopulationCell` (`population.py:266`). “Multi-species” in the
repository today means *codon-usage species* (`examples/12_multi_species.helix`
scores E. coli / yeast / human optimal-codon encodings of the same protein),
not coexisting organisms. A population also owns a *single* lattice
(`population.py:570`): there is no habitat structure, no metapopulation.

### 2.7 ⑦ Environment

| Capability | Where | Facts |
|---|---|---|
| Concentration fields | `environment.py`: `ConcentrationField:189` (2D), `ConcentrationField3D:518` | Fick diffusion (5-point/7-point Laplacian, Neumann), upwind advection, chemostat replenishment, CROMICS crowding |
| Uptake kinetics | `environment.py`: `monod_uptake:89`, `michaelis_menten_rate:118`, `atp_yield:146` (38 ATP/glucose), `molecules_per_site:133` | Monod/M-M saturation; GLC D=600, O₂ 2500, acetate 1200 µm²/s; Ks 0.1 / 0.05 mM |
| Media declaration | `sim_runtime.py:1758` (`#media` → shared substrate fields; `_environment:1524`) | arbitrary nutrient fields registered at parse time |
| Crowding | `population.py`: `_crowded_diffuse:875`, `crowding_diffusion_factor` (`environment.py:159`) | CROMICS volume-fraction slowdown, critical 0.14 |

**Segment verdict**: diffusion/advection/uptake for 2–3 small molecules is
real. **There is no temperature, no light, no pH, no toxin field, no
nitrogen/phosphorus chemistry, no diurnal/seasonal/climate forcing anywhere**
(searches for `circadian/diurnal/season/climate/photoperiod` in the runtime
return nothing; temperature exists only in DNA-storage decay,
`dna_codec.decay_dna:1304`).

### 2.8 ⑧ Ecosystem / multi-species

**Nothing implements this layer today.** The closest components:

- `apps/consortium.py`: `ConsortiumSimulator:147` — producer/sensor/actuator
  roles with density-dependent consensus (You 2004) and ratio control
  (Mee & Wang 2012). Roles are **hard-coded**, uniform fields, no species
  identity, no material exchange except signal.
- `examples/12_multi_species.helix`: codon-usage species, not ecology.
- `population.py` acetate cross-feeding (`acetate_switch`): intra-clone
  syntrophy, the only metabolite-exchange between cells.
- `fate_analysis.py` mentions “resource competition” only as a metaphor.

No predation, no competition, no parasitism/phage, no producer–consumer–
decomposer chain, no food web, no Lotka–Volterra-style population dynamics,
no decomposition of dead biomass back into nutrients, no nitrogen/phosphorus
cycles, no carbon-cycle closure, no habitat/metapopulation structure.

### 2.9 ⑨ Evolution and the feedback arm

| Capability | Where | Facts |
|---|---|---|
| Sequence-level evolution | `evolution.py`: `mutate:170`, `mutate_batch:301`, `select:366`, `recombine:438`, `Population:1003`/`step:1115`, `dnds_ratio` | real mutation spectrum: E. coli ~2.2×10⁻¹⁰ nt/gen (Lee 2012), genome 10⁻³/gen (Drake 1991), ts/tv ≈ 3:1, indels; Wright–Fisher |
| Digital organisms | `apps/digital_evolution.py`: `DigitalEvolution:176` | Avida-style instruction genomes; Eigen error-threshold collapse |
| ML-guided directed evolution | `apps/protein_evolution.py`: `guided_directed_evolution:239`; `protein_fitness.py`: `BLOSUMOracle:174`, `ESM2Oracle:203` | DBTL loop on GB1 (Wu 2016 DMS), BLOSUM62/ESM-2 oracles |
| **Spatial evolution closed loop** | `apps/spatial_evolution.py`: `SpatialEvolutionConfig:56` (default 10 gen × 10 pop × 25 colonisation ticks × 40 inner cells, `:88-103`) | the **only** module that closes ①→⑥→⑦→⑨→①: outer loop mutates real DNA, recompiles, inner loop scores each genotype as a spatial colonizer (fitness = colony radius × core survival − cost, Bosshard 2020); `#sim kind=spatial_evolution` (`examples/40_spatial_evolution.helix`) |
| Calibration/benchmark closure | `apps/whole_cell_calibration.py` (`WholeCellCalibration:70`, 4 params), `population_calibration.py` (3 dFBA params, 3-probe identifiability), `omics_calibration.py` (`OmicsCalibrationBenchmark:287`, VCC-2025 perturb-seq protocol), `virtual_cell_bench.py` (`VirtualCellBench:90`, calibrate-then-predict) | shared `fit_parameters` framework (`virtual_cell.py:746`), inverse-variance weighting (Karr DREAM8) |

**Segment verdict**: the sequence-level engine and the calibration framework
are production-grade; **but** `CellPopulation.evolve` (`population.py:1764`)
only runs *N* ticks, it does not mutate DNA; calibration is one-shot and
offline; and the environment→selection→genome feedback exists in exactly one
place (`spatial_evolution`) with a single fitness axis (expansion speed).

### 2.10 Orchestration and entry points

| Capability | Where | Facts |
|---|---|---|
| Language wiring | `parser.py`: `_parse_genome:356`, `_parse_sim`; `sim_runtime.py`: `run:293` | `#config backend` (6 top-level: classic/whole_cell/population/fba/calibration/benchmark, `sim_runtime.py:127`) + `#sim kind` (17 registered kinds, `sim_runtime.py:1486-1504`) |
| CLI | `cli.py`: `main:41`, `_run_sim:253` | `--backend`, `--json`, `--csv`, table output |
| Web | `server.py`: `/api/sim/run:152` | all backends reachable over HTTP |

---

## 3. What already closes the loop today

A fair reading of the audit: **two half-loops exist, and one full loop.**

1. **Genome → phenotype (closed).** ATCG → codon table → bytecode → GRN/
   central dogma → FBA → energy → division/death is a complete, physically
   calibrated, gated pipeline (`examples/01`–`40` cover every hop).
2. **Phenotype → population → environment (closed).** Colony growth on a
   diffusing/advecting substrate field with dFBA stratification, mechanics,
   and (in Design 6) self-consistent fluid flow.
3. **Genome → environment → selection → genome (closed once, narrowly).**
   `spatial_evolution` + `examples/40_spatial_evolution.helix` is the sole
   end-to-end loop, and it deliberately uses a *single-trait* fitness proxy.

Everything between “clone in a dish” and “ecosystem + environmental change”
is absent, which is exactly the frontier this document plans.

---

## 4. Gap analysis (audit vs. vision)

| # | Gap | Severity | Evidence |
|---|---|---|---|
| G1 | **No ecology/multi-species layer** — single clone per simulation; no species identity, predation, competition, symbiosis, food web, decomposition | Critical | `PopulationCell` has no species field (`population.py:266`); `examples/12` is codon usage, not ecology |
| G2 | **No environmental dynamics** — temperature, light, pH, diurnal/seasonal/climate forcing, and climate–biosphere feedback | Critical | no such fields in `environment.py`; no `circadian/season` in runtime |
| G3 | **No full multi-generation loop on a real genome** — `Population.evolve` doesn’t mutate; calibration is offline; only `spatial_evolution` closes the loop, at 10 generations on a single fitness axis | High | `population.py:1764`; `spatial_evolution.py` |
| G4 | **No intermediate organism scale** — cell↔colony only; no development of distinct cell types from a shared genome, no tissue/organ abstraction | High | consortium roles hard-coded (`apps/consortium.py`) |
| G5 | **Genome import is CDS-oriented, not chromosome-oriented** — no replicon/origin/terminus, no plasmids, synthetic 4300-gene map instead of a real regulatory map | Medium | `whole_cell_scale.load_genome:129`; `genome_scale.build_genome:168` |
| G6 | **Reduced metabolic core** — 37 reactions, no glyoxylate shunt (no diauxie phase 2), no phototrophy/chemolithotrophy | Medium | `spatial_dfba.py:20-23` |
| G7 | **Scale mismatch** — 1 tick = 1 min; ecological timescales (~year) need ≈5×10⁵ ticks, no event-driven fast-forward | High | `units.py` |
| G8 | **No ecosystem analytics** — no species abundance/energy-flow/trophic-efficiency outputs | Medium | only Shannon diversity (`population.py`) |
| G9 | **No cross-layer data loop** — designer/calibration/evolution outputs don’t feed each other | Medium | `synbio_designer.py`, `whole_cell_calibration.py` are one-shot |
| G10 | **No habitat/metapopulation structure** — one patch per run; no multiple populations of one species connected by dispersal (source–sink, drift), no multi-environment heterogeneity | Medium | `CellPopulation` owns one lattice (`population.py:570`); `environment.py` has one field grid |

---

## 5. Next-step implementation plan

### 5.1 Design principles (inherit from `doc/15`)

1. **Backward compatibility**: every new capability lands behind new classes,
   config flags, and parser annotations; defaults reproduce today’s
   behaviour. Public API never breaks.
2. **One spine, many backends**: the integrated loop is one new runtime
   entry (`#sim kind=ecosystem`); every existing backend stays reachable.
3. **Physically anchored**: each new term has a literature citation and real
   units (tick = min, site = 10 µm, ATP counts, µM).
4. **Gated by tests**: each phase ships tests that pass `ruff`, `mypy`,
   and the pytest coverage gate; CI notes that scipy/numpy are optional
   extras, so no test may *require* them to merely pass (performance gates
   are environment-aware, see §5.7 of the earlier fix).
5. **Methodologically anchored**: every new design element names the
   modeling method it implements (agent-based particle model, multi-level
   optimization, first-order pool decomposition, eco-evolutionary feedback)
   and a citation; §5.2 is the mapping, §10 the bibliography.

### 5.2 Literature foundation (method → design-element mapping)

The three words in this document’s brief — **multi-species, multi-population,
multi-environment** — correspond to three bodies of modeling literature. The
mapping below is the contract the phases implement:

| # | Design element | Method from literature | Citation | Lands in |
|---|---|---|---|---|
| L1 | Species identity; cells as discrete particles in continuous space; shoving/force mechanics | individual-based (agent-based) biofilm modeling, the BacSim → iDynoMiCS lineage | Lardon et al. 2011; Kreft et al. 1998 | A1 (mechanics already in `population.py:1159/1293`) |
| L2 | Condition-dependent metabolic switching with a switching cost | iDynoMiCS denitrifier case study: cost of fast pathway switching → optimal response time per environmental-fluctuation frequency; biodiversity maximal in biofilms and at intermediate frequency | Lardon et al. 2011 | A3, B2 |
| L3 | Community-level goal + member-level fitness (syntrophy, competition) | OptCom multi-level optimization: inner species LPs, outer community objective | Zomorrodi & Maranas 2012 | A3 (second engine beside the agent loop) |
| L4 | Spectrum of community-modeling approaches (super-individual ODE ↔ individual-based ↔ stoichiometric), and why to integrate | methodological review of microbial-community dynamics modeling | Song et al. 2014 | A2 scheduler, §5.6 |
| L5 | Coexistence by neutral drift vs. by niche/resource partitioning | unified neutral theory (Hubbell); neutral-vs-niche comparison framework | Hubbell 2001; Biol. Philos. 2024 | A3 |
| L6 | Predation; analytic validation target | classic Lotka–Volterra; conserved quantity `V = δx−γ ln x + βy−α ln y`, period `T≈2π/√(αγ)` near the centre | Volterra 1926; Lotka 1925; Hsu 1983 | A3 gate |
| L7 | Metapopulation: extinction–colonization, source–sink patches, dispersal | Levins metapopulation; Pulliam source–sink | Levins 1969; Pulliam 1988 | A2 (new) |
| L8 | Eco-evolutionary feedbacks; trait-based selection on continuous traits | adaptive dynamics: invasion fitness, mutation–selection, evolutionary suicide/trap/rescue | Ferrière & Legendre 2013 | A4 |
| L9 | Environmental drivers of selection; trait trade-offs (growth vs. tolerance, growth vs. yield) | trait-based ecosystem modeling under global change | Fisher 2021 | A4, B4 |
| L10 | Temperature/moisture response of growth and decay | Dual Arrhenius–Michaelis–Menten (Q10/Arrhenius rate modifiers) | Saifuddin et al. 2021 | B2, B3 |
| L11 | Decomposition → nutrient pools (carbon closure) | CENTURY pool structure: structural/metabolic litter + active/slow/passive SOM, first-order decay × T/moisture/clay; N-flow tied to C-flow via C:N ratios; linear analysis → closed-form equilibria | Parton et al. 1987; Bolker et al. 1998 | B3 |
| L12 | Daily N-cycle gases (nitrification/denitrification, CH₄ oxidation) | DAYCENT daily-time-step submodels | Parton et al. 1994 | B3 |
| L13 | Reference implementation of pool-decay ODEs in code | SoilR package (`CenturyModel`, 7 pools, per-week k) | Sierra et al. 2012 | B3 (validation target) |

### 5.3 Phase A — Ecosystem spine: multi-species + multi-population + environment↔genome feedback (priority)

**Goal**: make the ⑧ layer exist and close loop ①→⑦→⑧→⑨→① from a single
entry point, covering *coexisting species*, *multiple populations connected
by dispersal*, and *multiple habitats in one run* (G1, G10).

**A1. Species identity in the population engine.**
Add `species`/`strain` to `PopulationConfig` (`population.py:131`) and
`PopulationCell` (`population.py:266`). Multiple `.helix` programs (or
`#genome` templates) may coexist in one `CellPopulation`; each species owns a
row-subset of the shared sparse-GRN matrix (`sparse_grn.py`) and a species-
local `ECOLI_CORE_*` or dFBA configuration. Cells stay **individual agents in
continuous space** exactly as in the iDynoMiCS particle model (L1): the
existing shoving/force mechanics (`population.py:1159/1293`) is unchanged and
simply applies between species. Gate: a two-species culture grows/dies per
its own parameters while sharing one environment.

**A2. `apps/ecosystem.py` — the integrated runtime, with patches.**
New `EcosystemConfig`, `Species` (genome + programme + trait weights),
`Patch` (a habitat: its own `CellPopulation` subset + local fields), and
`Ecosystem` (owns per-species populations, the patches, the environment, and
the evolution loop), wired as `#sim kind=ecosystem` with a new parser
annotation `#species name=... genome=...` (extend `parser.py`’s annotation
dispatch, as done for `#genome` at `parser.py:356`).  The genotype may be a
`genome=` field or a multi-line DNA code block (analogous to `#gene`):
the block DNA is concatenated into the species genome — e.g.
`examples/42_ecosystem_evolution.helix`.

- **Multi-population / metapopulation (L7).** A `Patch` is a spatial unit
  (chemostat, biofilm, soil column). Populations of the *same* species across
  patches are linked by dispersal: patch-level extinction–colonization
  (Levins 1969) and source–sink flow (Pulliam 1988) are emergent from
  per-cell migration (cells move/are carried along a defined dispersal edge,
  at a `dispersal_rate` per patch pair). No ODE metapopulation is imposed —
  it *falls out* of the agent loop; the Levins equilibrium
  `p* = 1 − e/m` (extinction rate e, colonization rate m) is the analytic
  validation target.
- **Multi-environment (G10).** Each `Patch` configures its own
  `ConcentrationField` + `ScalarField` stack (Phase B), so a single run can
  hold a light-saturated water column, an anoxic sediment, and an aerobic
  surface patch — environmental heterogeneity becomes the substrate for
  niche segregation (L5) and for species-specific stress selection (L8/L9).
- The scheduler is *multi-scale*: `step()` advances ticks, but
  **fast-forwards** quiescent epochs (no cell above a change threshold →
  advance to the next scheduled event), addressing G7. Gate: a year-scale
  run terminates in seconds and is tick-reproducible; a two-patch
  metapopulation matches the Levins equilibrium and a source–sink pair
  maintains the sink at the predicted immigration-sustained abundance.

**A3. Inter-species interactions.**
- *Competition (L5)*: shared-substrate Monod uptake already exists; add
  per-species uptake coefficients so one substrate can be partitioned
  (niche separation) — and expose the neutral limit (identical coefficients,
  pure drift + dispersal, Hubbell 2001) as a configurable regime, so
  coexistence tests can span neutral→niche.
- *Cross-feeding / syntrophy (L3)*: generalize the intra-clone `acetate_switch`
  (`population.py:1616`) to a per-species substrate-exchange table (species A
  exhausts X, species B consumes the waste X’).
- *Community FBA (L3)*: alongside the agent loop, an **OptCom-style
  second engine** — inner level: one dFBA LP per species (reusing the
  shared-batch solver `population.py:1527`); outer level: a community
  objective (maximize total community biomass, or a species-weighted goal).
  This resolves the community metabolic trade-off (syntrophy vs.
  competition) that a pure per-cell loop leaves implicit. The agent loop and
  the community-FBA level are reconciled each tick by using the outer
  solution to set per-species uptake/deposit bounds. Gate: a two-member
  syntrophy reproduces the OptCom-predicted flux split.
- *Predation / death (L6)*: a consumption term (cells as substrate) with
  Monod handling — validate the resulting dynamics against analytic
  Lotka–Volterra for a two-species predator–prey pair (limit-cycle period
  vs. `2π/√(αγ)` near the centre; conserved quantity on closed orbits).
- *Metabolic switching with cost (L2)*: introduce a per-species switching
  cost (delay/ATP penalty for flipping between growth modes, e.g.
  glucose↔acetate, oxic↔anoxic). Gate: the iDynoMiCS prediction — at
  intermediate environmental-fluctuation frequency, *slow* switchers
  outcompete fast ones, and biodiversity is higher in spatially
  heterogeneous (biofilm) patches than in homogeneous (chemostat) ones.
- *Decomposition (L11)*: dead biomass → dissolved nutrients via a
  pool-structured decay (B3), closing the carbon/nitrogen loop at
  population level.
Gate: tests with coexistence (niche separation), neutral drift, competitive
exclusion, predator–prey limit cycles vs. the analytic solution, and the
switching-cost biodiversity result.

**A4. Generalize the feedback loop.**
Promote the `spatial_evolution` pattern (outer: `evolution.mutate/recombine`;
inner: spatial fitness on `CellPopulation3D`) into `Ecosystem` with
**multi-trait fitness** (growth rate, stress tolerance, secreted-metabolite
production, switching cost) — a trait axis per the adaptive-dynamics and
trait-based frameworks (L8, L9) — so selection is no longer single-axis.
Fitness is computed from *environment-dependent* phenotypes, so a change in
the environment changes selection — the explicit niche-construction arm of
the vision. Because traits are continuous (an uptake coefficient, a Q10, a
switching cost) the outer loop implements **invasion-fitness style
selection**: a mutant genotype invades when its long-term growth rate in the
resident community+environment exceeds the resident’s; repeated invasion →
substitution sequences are the simulated adaptive dynamics (Ferrière &
Legendre 2013), and the loop is instrumented to *detect* evolutionary
suicide/trap/rescue signatures (population collapses following trait change;
recovery via mutation under a changed environment). Gate:
`examples/41_two_species_crossfeeding.helix` and
`examples/42_ecosystem_evolution.helix` demonstrate a genome that changes
*in response to* an environment it also changes; an evolutionary-rescue test
shows extinction avoided by mutation when the environment shifts.

### 5.4 Phase B — Environmental dynamics (temperature, light, chemistry, time)

**Goal**: give the environment its own physics and its own timescales (G2, G6).

**B1. New environment fields, per patch.** Generalize `ConcentrationField`
(`environment.py:189`) behind a `ScalarField` base with kinds `temperature`,
`light`, `pH`, plus a generic `toxin`. Each `Patch` (A2) declares its own
field stack. Add per-tick forcing: diurnal sine for light/temperature,
seasonal envelope, and climate tables (time → value) so different habitats
in one run see different drivers.

**B2. Temperature-dependence of biology (L10).** Introduce a
Q10/Arrhenius rate modifier applied to uptake (`environment.py:89`),
division, decay, and decomposition, per species (thermophile vs. mesophile
parameterization), following the Dual Arrhenius–Michaelis–Menten (DAMM)
structure used in coupled C/N models: a Michaelis–Menten substrate term
times an Arrhenius temperature term times a moisture term. Gate: growth-rate
vs. temperature matches the literature curve used for calibration, and the
sealed-microcosm temperature sensitivity matches the DAMM reference.

**B3. Biogeochemistry (L11–L13).**
- *Decomposition pools (CENTURY structure)*: replace the Phase-A placeholder
  “dead biomass → nutrients” with a real pool model — structural vs.
  metabolic litter partitioned by a `lignin:N` proxy, feeding
  active → slow → passive SOM pools with first-order decay constants
  modulated by temperature (B2), moisture, and texture (clay) factors
  (Parton et al. 1987). Use the SoilR `CenturyModel` parameter set
  (k/week: STR.surface 0.076, MET 0.28, STR.below 0.094, MET.below 0.35,
  ACT 0.14, SLW 0.0038, PAS 0.00013; Sierra et al. 2012) as the numerical
  reference, and the Bolker–Pacala–Parton linear reduction (Bolker et al.
  1998) as the closed-form equilibrium check.
- *Nitrogen/phosphorus cycles*: tie N-flow to C-flow through C:N ratios with
  mineralization/immobilization (CENTURY), and add
  nitrification/denitrification N-gas fluxes + CH₄ oxidation as `#media`-
  declarable nutrient cycles following DAYCENT (Parton et al. 1994);
  phosphorus uptake as an additional limiting field. C budget closes through
  respiration/decay; N budget closes through fixation + denitrification.
- *Photoautotrophy*: add a light-gated uptake mode (B1’s light field) as the
  first step toward phototrophs/chemolithotrophs beyond the 37-reaction
  heterotrophic core (G6). Gate: a sealed microcosm test where biomass +
  CO₂ + O₂ + N₍fixation→denitrification₎ balance is conserved to numerical
  tolerance; a litter-decay test reproduces CENTURY pool turnover to within
  a stated tolerance.

**B4. Climate–biosphere feedback (L9).** Allow biomass to modify local
fields (O₂/CO₂ drawdown, metabolic-heat option, evapotranspiration-style
water draw in the soil patch), i.e., environment change that is *driven by*
the organisms. Species are parameterized by *functional traits* (light-use,
uptake Q10, growth yield, stress tolerance) rather than ad hoc constants, so
trait trade-offs (Fisher 2021) — e.g. fast growth vs. tolerance — generate
the coexistence/selection axes that Phases A3–A4 consume. Gate:
`examples/43_diurnal_microcosm.helix` shows daytime photosynthesis-style O₂
supersaturation and night-time sag in the water patch and the matching
anoxic-sediment response.

### 5.5 Phase C — Genome completeness (real genomes, full structure)

**Goal**: weaken G5 by making the *whole* genome (not a gene list) the input.

**C1. Chromosome-oriented import.** Extend `load_genome`
(`whole_cell_scale.py:129`) to accept full-chromosome FASTA + a GFF/annotation
table (promoters, terminators, operons); keep the existing CDS path intact.
Add an optional real regulatory map import behind `tf_map="regulondb"`
(`genome_scale.py:168`), replacing the synthetic attachment with RegulonDB-
derived edges while keeping the sparse CSR machinery. — *Landed:*
`parse_gff3`/`load_chromosome`/`Chromosome` (`whole_cell_scale.py`) — CDS
extraction by coordinate with minus-strand reverse-complement and multi-row
`Parent` merge; and `tf_map="regulondb"` (`genome_scale.py`) —
`parse_regulondb` (RegulonDB network TSV) + the curated
`REGULONDB_DEMO_EDGES` subset replace the Barabási–Albert attachment on the
same CSR template (crp top hub, ±weights); tests in `tests/test_whole_cell_scale.py`
and `tests/test_genome_scale.py`.

**C2. Replicon structure.** Model genotype as chromosome (oriC/terC) + zero or
more plasmids; reuse the existing Cooper–Helmstetter timing
(`virtual_cell.py:422`) so copy-number, and therefore expression level, is
replicon-aware. — *Landed:* `RepliconSpec` + `VirtualCellConfig.replicons`/
`gene_replicons` (`virtual_cell.py`); chromosome genes keep fork-driven
Cooper–Helmstetter dosage while plasmid genes carry a constant base copy
number (immune to forks and division halving); transcription scales with the
replicon copy → a 20-copy pBR322 gene expresses 20× (gene-dosage effect).
Language wiring: `#config sim replicons=pBR322:20` + `#gene ... replicon=pBR322`.
Tests: `tests/test_virtual_cell.py` (replicon section), `tests/test_sim_runtime.py`.

**C3. Metabolic-core expansion.** Add the glyoxylate shunt (removing the
`spatial_dfba.py:20-23` limitation → true diauxie phase 2) and a first
iJO1366-derived subsystem beyond the 37-reaction core. Gate:
`examples/44_diauxie_complete.helix` shows glucose → acetate → acetate
re-consumption; an imported small-bacterium genome runs
ATCG→phenotype→selection end to end; essentiality on an imported genome
matches EcoCyc within the previously validated tolerance.

### 5.6 Phase D — Cross-scale performance and the data loop

**Goal**: make Phases A–C usable (G7, G8, G9) and give the ecosystem layer
its own analytics.

**D1. Event-driven scheduler.** Ship the fast-forward engine of A2 as a
general `Scheduler` (next-event time advance) usable by any backend, with
deterministic tick-equivalence tests. The multi-scale philosophy follows the
community-modeling-review guidance that agent-based, ODE, and stoichiometric
layers should coexist rather than substitute (L4): the scheduler decides,
per patch, whether to advance agent-by-agent or to fast-forward a
pool-kinetics sub-step (CENTURY-style decay needs ~1/week updates, not
minute ticks).

**D2. Performance.** Per-species vectorized metabolism and shared-batch dFBA
across species (`population.py:1527`); keep the LBM paths. Add a year-scale
benchmark to `tests/test_benchmark.py` with environment-aware timing bounds.

**D3. Data loop.** Wire `synbio_designer` output DNA → `evolution` →
`Ecosystem` → observation → next round of calibration (a population-scale
DBTL), reusing `fit_parameters` (`virtual_cell.py:746`). Gate:
`apps/population_dbtl.py` + test proving a designed strain is improved by the
loop. — *Landed:* `apps/population_dbtl.py` (design→build→test→learn rounds,
elitist; round-0 population seeded from a `synbio_designer` cassette;
growth-trait surrogate via `fit_parameters`; unsaturated-trait biasing);
`#sim kind=population_dbtl` in `sim_runtime.py`; `examples/45_population_dbtl.helix`
improves the designed strain 0.077 → 0.133 (≈1.7×); `tests/test_population_dbtl.py`.

**D4. Ecosystem analytics.** New `SimResult` columns/rows: per-species
abundance (per patch and total), Shannon diversity (exists), biomass turnover,
**energy flow** (net primary production → consumption → decomposition),
**trophic-efficiency** estimates (consumer-biomass/producer-biomass per
link), and biomass↔field balance per element (C, N, P). A neutral-drift
diagnostic (per-species abundance variance vs. the Hubbell null) is emitted
so a run can be labelled *niche-structured* vs. *drift-dominated* (L5). Gate:
the `ecosystem` backend emits these for `--csv`/`--json` in `cli.py`.

### 5.7 Method inventory (what the phases add, mapped to literature)

| New code (indicative) | Implements | Citation |
|---|---|---|
| `species` field on `PopulationConfig`/`PopulationCell` | particle-based species identity | Lardon 2011 (L1) |
| `Ecosystem` + `Patch` + dispersal | metapopulation / multi-habitat / source–sink | Levins 1969; Pulliam 1988 (L7) |
| `CommunityFBA` (OptCom-style inner/outer) | community-level metabolic goal | Zomorrodi & Maranas 2012 (L3) |
| `switching_cost` trait + fluctuation-forcing | condition-dependent metabolic switching | Lardon 2011 (L2) |
| `ScalarField` kinds (T, light, pH, toxin) | environmental drivers | Fisher 2021 (L9) |
| Q10/Arrhenius rate modifier | temperature-dependence of biology | Saifuddin 2021 (L10) |
| `DecompositionPoolModel` (CENTURY-style) | litter + SOM pools, C/N coupling | Parton 1987; Bolker 1998; Sierra 2012 (L11, L13) |
| N-gas + CH₄ submodel | daily biogeochemical cycling | Parton 1994 (L12) |
| invasion-fitness outer loop + rescue detector | eco-evolutionary feedbacks | Ferrière & Legendre 2013 (L8) |
| ecosystem analytics + neutral/null diagnostic | abundance, energy flow, trophic efficiency | Hubbell 2001 (L5) |

---

## 6. Acceptance gates and tests

All phases inherit the project gates:
`ruff check src tests` · `mypy` (src only, `mypy.ini`) ·
`pytest --cov=helixlang --cov-fail-under=80`.

| Phase | Hard gates |
|---|---|
| A | `tests/test_ecosystem.py` (coexistence, competitive exclusion, neutral drift, predator–prey vs. analytic Lotka–Volterra, switching-cost biodiversity); two-species shared-matrix state; metapopulation vs. Levins equilibrium + source–sink test; year-scale fast-forward reproducibility; `examples/41`, `examples/42` compile and run in the examples audit (`tests/test_end_to_end.py`) |
| B | field-forcing unit tests; Q10 curve gate (DAMM reference); CENTURY pool-turnover test (SoilR parameters); sealed-microcosm C/N conservation test; `examples/43` runs |
| C | chromosome/GFF import round-trip; replicon copy-number effect; glyoxylate-shunt diauxie test; essentiality parity on imported genome; `examples/44` runs |
| D | scheduler tick-equivalence test; year-scale benchmark with environment-aware bound; population-DBTL test; ecosystem analytics columns asserted; neutral/null-labelling diagnostic test |

Notes for CI (learned from the scipy-optional environment): any test that
requires numpy/scipy either runs under their guards or computes the same
quantity without them; performance assertions bind to the code path actually
executed.

---

## 7. Phasing summary

| Phase | Theme | New code (indicative) | New backends/annotations | Examples | Priority |
|---|---|---|---|---|---|
| A | Ecosystem spine + multi-species + metapopulation + feedback | `apps/ecosystem.py` (Species/Patch/Ecosystem), species fields in `population.py`, OptCom-style community FBA | `#sim kind=ecosystem`, `#species`, `#patch` | 41, 42 | ★★★ |
| B | Environment dynamics + biogeochemistry | `environment.py` `ScalarField` kinds, Q10/DAMM, CENTURY pools, N/P cycles, light-gated uptake | `#media` cycle declarations, per-patch field stack | 43 | ★★★ |
| C | Genome completeness | chromosome/GFF import, replicon model, glyoxylate shunt, real regulatory map | `tf_map="regulondb"`, replicon config | 44 | ★★ |
| D | Scale + data loop + ecosystem analytics | `Scheduler`, per-species vectorized metabolism, population DBTL, trophic/energy-flow outputs | — | 45 (DBTL) | ★★ |

---

## 8. What “done” looks like (acceptance narrative)

A user can write:

```helix
#species name=producer genome=phototroph.helix
#species name=consumer genome=heterotroph.helix
#species name=decomposer genome=soil_chemotroph.helix
#patch name=water light=day cycle=diurnal
#patch name=sediment light=none anoxic=true
#patch name=soil temperature=seasonal
#config backend=ecosystem
#config ticks=525600            # one year, fast-forwarded
#sim kind=ecosystem generations=200
#sim seed=7
```

and get back, per generation and per patch, per-species abundance, biomass
turnover, energy flow, trophic efficiency, the O₂/CO₂/temperature
trajectories, the fitness axes, the neutral-vs-niche label, and the changed
genomes — i.e., **a single run in which complete ATCG genomes, through their
own life processes, change a multi-patch environment, and that changed
environment selects the next generation’s genomes.** Every hop of that loop
already exists in the codebase today except the ecology layer and the
environmental dynamics; this plan adds those two layers (grounded in the
methods of §5.2) and the integration spine to run the loop end to end.

---

## 9. Relationship to earlier documents

- `doc/15-whole-cell-realism.md` — the physical-completeness roadmap for the
  single cell (implemented); Phases A–D of this document build *on top* of
  that layer.
- `doc/18-programmable-cell-population-simulation.md` — the population
  roadmap (§13 Design 1–6); Phase A generalizes Design 1’s dual loop and
  Design 5’s genome-scale colony to multiple species; Design 6’s LBM is
  reused unchanged in Phase D.
- `doc/10-frontier-biology-analysis.md` — capability mapping and SOTA
  benchmark; the ecology/environment phases extend that mapping into the
  ecosystem scale.
- `doc/00-overview.md` — navigation table is updated with this document.

---

## 10. References

### Already used in the audit (kept)

- Karr et al. 2012, Cell 150:389 — whole-cell model (M. genitalium).
- Bosshard et al. 2020, BMC Genomics 21:232 — range-expansion fitness.
- Orth et al. 2010, Mol Syst Biol 6:390 / Feist et al. 2007 — E. coli core.
- Mahadevan et al. 2002, Biophysical J 83:1331 — static-optimization dFBA.
- Lee et al. 2012, Genome Res 22:885 — E. coli mutation rate; Drake 1991.
- Martínez-Antonio & Collado-Vides 2003; Martínez-Antonio et al. 2008.
- Proshkin 2010; Ingolia 2009; Bernstein 2002; Dong 1996 — central dogma.
- Wolpert 1969; Basu et al. 2005 — morphogen gradients.
- You et al. 2004 (PNAS) — quorum consensus; Mee & Wang 2012 — ratio control.
- Pearson 1993 / Gray & Scott — Turing patterns; Guo 2002; Ladd 1994 — LBM.
- Peccoud & Ycart 1995; Scheffer et al. 2009 — noise & critical transitions.
- Taheri-Araghi 2015 (adder); Balchin 2016 (folding); Sanchez 2017 (GECKO).
- Gerdes et al. 2003 — EcoCyc essentiality; Wu et al. 2016 — GB1 DMS.
- Neidhardt 1996 — physical units; Karr DREAM8 — inverse-variance weighting;
  VCC 2025 / Macklin et al. 2020 — calibrate-then-predict benchmarks.

### Added for the ecosystem/environment plan (§5.2)

- Lardon, L. A., Merkey, B. V., Martins, S., Dötsch, A., Picioreanu, C.,
  Kreft, J.-U. & Smets, B. F. 2011. *iDynoMiCS: next-generation
  individual-based modelling of biofilms.* Environmental Microbiology
  13(9):2416–2434. DOI:10.1111/j.1462-2920.2011.02414.x — particle-based
  agents, pressure-field mechanics, stochastic chemostat, condition-dependent
  metabolic switching (denitrifier case study).
- Kreft, J.-U., Booth, G. & Wimpenny, J. W. T. 1998. *BacSim, a simulator for
  individual-based modelling of bacterial colony growth.* Microbiology
  144:3275–3287 — predecessor; shoving algorithm.
- Zomorrodi, A. R. & Maranas, C. D. 2012. *OptCom: a multi-level
  optimization framework for the metabolic modeling and analysis of microbial
  communities.* PLoS Computational Biology 8(2):e1002363 — inner species-level
  LPs + outer community-level objective.
- Song, H.-S., Cannon, W. R., Beliaev, A. S. & Konopka, A. 2014.
  *Mathematical modeling of microbial community dynamics: a methodological
  review.* Processes 2:711–752 — super-individual → individual-based
  spectrum; integration guidance.
- Hubbell, S. P. 2001. *The Unified Neutral Theory of Biodiversity and
  Biogeography.* Princeton University Press — neutral drift, immigration.
- (2024) *Neutral and niche theory in community ecology: a framework for
  comparing model realism.* Biology & Philosophy 39 — neutral-vs-niche
  comparison framework.
- Lotka, A. J. 1925. *Elements of Physical Biology.* Williams & Wilkins;
  Volterra, V. 1926. *Variazioni e fluttuazioni del numero d'individui in
  specie animali conviventi.* Mem. R. Accad. Naz. Lincei 2:31–113 —
  predator–prey equations and the conserved quantity.
- Hsu, S.-B. 1983. *A remark on the period of the periodic solution in the
  Lotka–Volterra system.* J. Math. Anal. Appl. 95:144–148 — period as a
  strictly increasing function of the energy level.
- Levins, R. 1969. *Some demographic and genetic consequences of
  environmental heterogeneity for biological control.* Bull. Entomol. Soc.
  Am. 15:237–240 — metapopulation extinction–colonization.
- Pulliam, H. R. 1988. *Sources, sinks, and population regulation.* American
  Naturalist 132:652–661 — source–sink dynamics.
- Ferrière, R. & Legendre, S. 2013. *Eco-evolutionary feedbacks, adaptive
  dynamics and evolutionary rescue theory.* Phil. Trans. R. Soc. B
  368:20120081 — invasion fitness, evolutionary suicide/trap/rescue.
- Fisher, R. A. 2021. *Trait-based modeling of terrestrial ecosystems:
  advances and challenges under global change.* Current Climate Change
  Reports 7:1–14 — functional-trait parameterization, trade-offs.
- Saifuddin, M. et al. 2021. *Identifying data needed to reduce parameter
  uncertainty in coupled carbon and nitrogen models.* JGR Biogeosciences
  126 — DAMM-MCNiP: Dual Arrhenius–Michaelis–Menten microbial C/N
  physiology, enzyme kinetics, temperature/moisture response.
- Parton, W. J., Schimel, D. S., Cole, C. V. & Ojima, D. S. 1987. *Analysis
  of factors controlling soil organic matter levels in Great Plains
  grasslands.* Soil Sci. Soc. Am. J. 51:1173–1179 — CENTURY pool structure.
- Parton, W. J., Ojima, D. S., Cole, C. V. & Schimel, D. S. 1994. *A general
  model for soil organic matter dynamics: sensitivity to litter chemistry,
  texture and management.* — DAYCENT daily-time-step version:
  nitrification/denitrification N-gas fluxes, CH₄ oxidation.
- Bolker, B. M., Pacala, S. W. & Parton, W. J. 1998. *Linear analysis of soil
  decomposition: insights from the Century model.* Ecological Applications
  8:425–439 — linear reduction, closed-form equilibria.
- Sierra, C. A., Müller, M. & Trumbore, S. E. 2012. *Modeling organic carbon
  dynamics in heterogeneous soils with the SoilR package.* Geosci. Model Dev.
  5:1045–1060 — reference `CenturyModel` (7 pools, per-week k) for
  numerical validation.
