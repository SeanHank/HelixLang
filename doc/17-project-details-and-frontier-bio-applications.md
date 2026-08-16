# HelixLang Project Details and Frontier Bio-Applications

This document collects the design and implementation sections of the HelixLang walkthrough:
what happens when a program is compiled and run (§3), the software architecture (§4),
worked examples (§5), how the toolset maps onto frontier biology problems (§8), and the
per-problem solution designs that were delivered (§9). Each section is self-contained.
Writing date: 2026-08 · Baseline: 2134 tests, ≈89% coverage, `ruff` + `mypy` clean
(revision 2026.8.2).

## Table of Contents

3. [What happens when you compile and run once](#3-what-happens-when-you-compile-and-run-once)
4. [Software architecture overview](#4-software-architecture-overview)
5. [Understanding it with real examples](#5-understanding-it-with-real-examples)
8. [How HelixLang maps onto these problems](#8-how-helixlang-maps-onto-these-problems)
9. [Solution designs (problem by problem)](#9-solution-designs-problem-by-problem)

---

## 3. What happens when you compile and run once

### 3.1 Flow diagram

```
  .helix source
  #gene name=hello
  ATG GCT GGT TAA
  #end
       │
       ▼ Lexer (slices DNA into codons every 3 bases, also recognizes #... comment blocks)
  codon stream + comment markers
       │
       ▼ Parser (splits START..STOP into individual "genes")
  AST (Program → Gene → Codon)
       │
       ▼ Semantic (semantic checks: reading frame, regulatory edges, symbol table)
       │
       ▼ Compiler (looks up the genetic code table: codon → opcode)
  ATG→OP_START  GCT→BUILD_PROTEIN  GGT→BUILD_MEMBRANE  TAA→OP_HALT
       │
       ▼ Bytecode (bytecode + constant pool + line-number table)
  0x10 0x01 0x02 0x00 …
       │
       ▼ CellVM (stack VM = "ribosome")
  ┌─────────────┬──────────┬──────────────────────┬─────────────────────┐
  │ GRN reg     │ L-system │ reaction-diffusion   │ cell state          │
  │ (conc→switch) │ (morpho) │ (Turing patterns)    │ (energy/protein/pos)│
  └─────────────┴──────────┴──────────────────────┴─────────────────────┘
       │
       ▼ Observable output
  cell trajectories (conc/energy/position), morphology (L-system strings / images),
  behavior logs (move/signal/divide)
```

### 3.2 Plain-language analogy

Think of the whole process as "**building a car out of LEGO that can drive itself**":

1. **You write the instruction book** (`.helix` source): you specify the type and order of
   every brick (the codons).
2. **A translator reads the book** (the compiler): it converts "brick 3 is a red 2×2"
   into step-by-step instructions a machine understands.
3. **A little robot executes it** (the VM/ribosome): it follows the steps one by one, and
   can change its mind in real time based on "is there an obstacle ahead?" (GRN
   concentrations).
4. **The result**: a car that runs (a cell's behavior) — not only is it assembled, it
   moves, reproduces, and can signal to other cars.

### 3.3 Binary artifacts: a program you can "put in a file" and take with you

What you saw above is the "source → VM" pipeline. On top of that, HelixLang can package the
compilation result into a **`.helixc` binary file** (analogous to Java's `.class` /
Python's `.pyc`):

- `helixlang app.helix --compile -o app.helixc`: packages **program + bytecode + genetic
  code table** into a single self-contained file. The format is typed, length-checked and
  versioned — it is **not** `pickle`, and loading never executes code from the file;
- `helixlang app.helixc`: runs the binary directly, behaving exactly like running the
  source;
- `helixlang app.helixc --disassemble`: disassembles to inspect the bytecode, with
  breakpoint debugging;
- `helixlang app.helixc --decompile -o app.helix`: **decompiles back to source**. When the
  original source was embedded at compile time it can be restored byte for byte;
  `--compare` runs the binary path and the source path once each and compares trajectories,
  guaranteeing "two entry points, one behavior".

In plain words: **your "gene program" can now be distributed, archived and audited like a
deliverable binary artifact** — a key step in HelixLang's move toward engineering practice.

---

## 4. Software architecture overview

The whole codebase lives under `src/helixlang/`, roughly in four layers (module name =
file name):

### 4.1 Compiler toolchain (9 modules)

| Module | Responsibility |
|---|---|
| `codon_table.py` | 64-codon → opcode mapping table (standard / mitochondrial / ciliate tables) |
| `lexer.py` | Tokenizes DNA into 3-base codons, also recognizing `#gene` comment blocks |
| `parser.py` | Recursive-descent parser assembling the codon stream into an AST |
| `ast_nodes.py` | AST node definitions (Program / Gene / Promoter…) |
| `semantic.py` | Semantic checks (reading frame, regulatory edges, symbol table) |
| `compiler.py` | AST → bytecode (opcodes + operands + line numbers) |
| `bytecode.py` / `disassembler.py` | Bytecode data structures / disassembler |
| `hxbc.py` | `.helixc` binary artifact codec: versioned container (program + bytecode + codon table), supporting compile / decompile / round-trip comparison |
| `type_system.py` | Type checker |

### 4.2 Simulation runtime (the "cell" layer)

| Module | Responsibility |
|---|---|
| `vm.py` / `cell.py` | Stack VM (ribosome) + single-cell state |
| `grn.py` | Gene regulatory networks: sigmoid/Hill dynamics, half-life decay, optional noise, ODE solver |
| `stochastic.py` | Stochastic gene expression: two-state (telegraph) promoters, Fano factor, Gillespie SSA |
| `lsystem.py` / `reaction_diffusion.py` | L-system morphogenesis / Gray-Scott reaction-diffusion fields (Turing patterns) |
| `population.py` | Multicellular populations: each cell programmable (own GRN + bytecode), shoving mechanics, lineage tracking; 3D path is numpy-batch vectorized (packed arrays + `[z][y][x]` scattering, near-linear scaling to 10⁵ cells) |
| `environment.py` | Glucose / oxygen / AI-2 diffusion fields, Monod uptake, chemostat, CROMICS crowding diffusion |
| `morphology_3d.py` | 3D populations + 3D diffusion + LSystem3D morphology export |
| `vectorized.py` | numpy evaluation of GRNs across tens of thousands of cells at once (speed-up) |
| `units.py` | Physical units carried end-to-end (1 tick = 1 min, µM, µm²/s, ATP molecule counts) |

### 4.3 Bio function modules

| Module | Responsibility |
|---|---|
| `central_dogma.py` | Central dogma: transcription/translation kinetics, mRNA half-life |
| `metabolism.py` | Flux balance analysis (FBA) + dynamic FBA (dFBA, models diauxic growth / glucose depletion) |
| `protein_structure.py` | Secondary-structure prediction (Chou-Fasman, GOR IV), disorder regions (IUPred) |
| `protein_fitness.py` | Protein fitness oracles: BLOSUM62 conservation + ESM-2 protein language model (optional) |
| `crispr.py` | CRISPR-Cas editing: guide-RNA design, off-target prediction (Doench 2016) |
| `epigenetics.py` | CpG methylation, histone modification |
| `evolution.py` | Evolution engine: Wright–Fisher selection, mutation, drift, dN/dS |
| `omics.py` | Spatial omics: expression matrices → GRN/FBA state, spatial maps, heterogeneity metrics (ARI) |
| `virtual_cell.py` | Virtual cell: GRN → central dogma → metabolism → energy budget integrated in one cell-cycle model; adds four physical layers — cell-cycle timing (Cooper–Helmstetter), adder size control, chaperone-mediated folding, enzyme-capacity-constrained metabolism; `fit_parameters` supports inverse-variance-weighted multi-observable fitting |
| `interop.py` | Standard-format interchange: SBML import, SBOL3 export/import |

### 4.4 Application layer and surroundings

`apps/` currently holds 18 directly runnable application modules (the delivery points for
the §9 solution designs, the three extension directions, and the §13 designs of
`doc/18-programmable-cell-population-simulation.md`):

| Module | Responsibility |
|---|---|
| `consortium.py` | Synthetic consortia: consensus voting (You 2004 style) + ratio control |
| `morphogen_gradient.py` | Morphogenesis gradients: source/host cells differentiate by concentration threshold (Basu 2005 style) |
| `digital_evolution.py` | Digital evolution: DNA programs driven by natural selection in a population (Avida paradigm) |
| `spatial_evolution.py` | Spatial range-expansion evolution: dual-loop adaptation on the lattice (doc/18 §13 Design 1) |
| `synbio_automation.py` | Cello-style automation: truth table → logic gates → DNA → SBOL3 → predicted dynamics |
| `synbio_designer.py` | Expression cassette / vector design |
| `dna_storage.py` | DNA storage scenario decisions: coding benchmarks + per-GB cost comparison |
| `lattice_boltzmann.py` | Lattice-Boltzmann D2Q9 BGK flow solver (doc/18 §13 Design 6, Level 2) |
| `lattice_boltzmann_3d.py` | Lattice-Boltzmann D3Q19 BGK flow solver, 3D (doc/18 §13 Design 6, Level 2 3D) |
| `spatial_dfba.py` | Spatial dynamic FBA biofilms: 1-D gradients + diffusion-coupled dFBA batches |
| `fate_analysis.py` | Cell-fate decisions: bistable-switch bifurcation + stochastic switching + critical slowing down |
| `protein_evolution.py` | ML-guided directed evolution: ESM-2/BLOSUM oracle loop vs random baseline |
| `virtual_cell_bench.py` | Virtual cell "calibrate → predict" loop benchmark (Virtual Cell Challenge protocol) |
| `omics_calibration.py` | Omics-level calibration: CRISPRi PerturbSeq noise model + VCC-style fold-change fitting (weighted) |
| `population_calibration.py` | Population-level mixed-observable calibration: fits colony dFBA parameters (doc/18 §13 Design 4) |
| `genome_scale.py` | Genome-scale GRN builder + colony closure: ~4300-ORF per-cell GRN + FBA gating (doc/18 §13 Design 5) |
| `whole_cell_scale.py` | Whole-genome scale: FASTA genome loading + KO→FBA gene essentiality screening |
| `whole_cell_calibration.py` | Whole-cell parameter calibration loop: invert adder/k_fold/enzyme_scale/maintenance from mixed observables (growth curve + protein abundance + cell size + biomass flux) (Karr 2012 DREAM8 weighted fitting, aligned with Virtual Cell Challenge 2025) |

- `dna_codec.py`: Goldman rotated-key encoding (≈5.05 trits/byte) and Erlich DNA Fountain
  (≈1.57 bit/nt, near the Shannon limit).
- `server.py` / `web/`: Flask REST API + browser visualization.
- `cli.py`: command-line entry point.
- Supporting tooling: a PyCharm plugin (LSP-based) can breakpoint-debug `.helix` programs.

---

## 5. Understanding it with real examples

`examples/` contains 40 `.helix` example programs (plus 5 directly runnable Python API
scripts, see §5.4); 39 of them ship with `.helixc` binary artifacts (example 40 is the
freshly added spatial-evolution file). A few representative ones:

### 5.1 Hello, DNA (minimal program)

```helix
#gene name=hello
ATG GCT GGT TAA   # START -> BUILD_PROTEIN -> BUILD_MEMBRANE -> HALT
#end
```

When run, the VM executes 4 instructions: start → build one protein → build a membrane →
halt.

### 5.2 The lac operon (a real negative-feedback loop)

```helix
#gene name=lacI promoter=p_lacI    # constitutively expresses the lacI repressor
ATG GCT GCT TAA
#end
#regulate lacI -> p_lac strength=-0.9   # lacI represses p_lac
```

This is the core logic of the *E. coli* lactose operon: the repressor protein inhibits the
promoter, forming negative feedback. The GRN actually computes the concentration over
time.

### 5.3 Turing patterns (reaction-diffusion)

```helix
#field size=32 F=0.035 k=0.065 Du=0.16 Dv=0.08
#config ticks=100 react_steps=2
```

The Gray-Scott reaction-diffusion equations automatically generate leopard/spotted Turing
patterns — using the parameter sets measured in the literature.

### 5.4 More complex systems (population + environment + frontier layer)

- `20_diauxic_growth.helix`: *E. coli* two-phase growth (glucose first, then starvation),
  simulated with dFBA.
- `21_quorum_circuit.helix`: quorum-sensing circuit — a few cells release signal molecules;
  once the concentration is high enough, **the whole population** flips behavior.
- `22_pattern_synthesis.helix`: programmable cells self-organize spatial patterns on a
  shared diffusion field.

Frontier layer (delivery examples for Designs 1–9):

- `23_evolve_signal.helix`: digital evolution — DNA programs are driven by natural
  selection in a population, evolving "emit as much signal as possible" behavior.
- `24_spatial_diauxie.helix`: spatial dFBA — the grid has a glucose-rich and a
  glucose-poor region, and the population self-organizes into "glycolytic fast-growth zone
  / respiratory slow-growth zone".
- `25_morphogen_gradient.helix`: morphogen gradient — source cells secrete a morphogen,
  host cells differentiate by concentration threshold into a concentric cell-type map.
- `26_cello_workflow.helix`: Cello-style design automation — the one-stop loop
  truth table → logic gates → DNA → SBOL3 → predicted dynamics.
- `27_codec_benchmark.helix`: DNA-storage coding benchmark — Fountain / Goldman /
  Reed-Solomon cost-robustness comparison under a realistic error model.
- `28_fate_analysis.helix` (with `fate_analysis_workflow.py`): cell-fate decisions —
  bistable-switch bifurcation scan + stochastic switching + critical slowing down.
- `29_directed_evolution.helix` (with `directed_evolution_workflow.py`): ML-guided
  directed evolution — zero-shot oracle-guided multi-round GB1 sampling, aligned with
  EVOLVEpro.
- `30_virtual_cell.helix` (with `virtual_cell_workflow.py`): virtual cell
  "calibrate → predict" loop benchmark (Virtual Cell Challenge 2025 protocol).

**Latest 2026-08 deliverables (examples 31–40) — whole-cell realism layer, dFBA
deepening, and the population roadmap**:

- `31_whole_cell_adder.helix`: all four whole-cell layers on — **adder size control of
  division** (birth volume + fixed increment), **Cooper–Helmstetter chromosome replication
  timing** (C/D phases aligned to doubling time), **chaperone-mediated folding**
  (`frac_cotranslational_fold`), **enzyme-capacity-constrained metabolism**; runs from
  plain `#config` declarations.
- `32_colony_dfba.helix`: **per-cell dFBA for a 2000-cell colony** — oxygen is consumed at
  the colony edge, the core switches to fermentative metabolism, outputting the
  "core/edge acetate, core/edge oxygen" spatial stratification.
- `33_fba_diauxie.helix`: pure **FBA batch diauxie curve** — single sugar source + oxygen
  cap triggering overflow metabolism, directly emitting full time–biomass–glucose–growth
  rate trajectories.
- `34_whole_cell_calibration.helix`: one-click **calibration loop** — inverts the hidden
  `enzyme_scale / maintenance / adder / k_fold` parameters from noisy mixed observables
  (`backend=calibration`).
- `35_acetate_switch.helix`: **overflow-metabolism switch** — oxygen-cap-triggered acetate
  accumulation plus secondary consumption (relying on glycerol/ethanol/acetate
  dissimilatory pathways), emitting the full metabolic-flux switching trajectory over time.
- `36_population_calibration.helix`: **population-level calibration loop**
  (doc/18-programmable-cell-population-simulation.md §13 Design 4) — reads only colony-level
  mixed observables (growth rate, core/edge O₂ and acetate, mean/newborn cell energy,
  colony size) and inverts the three free parameters
  `dfba_oxygen_max_uptake / dfba_energy_scale / division_threshold`
  (`#sim kind=population_calibration`).
- `37_genome_colony.helix`: **genome-scale GRN colony**
  (doc/18-programmable-cell-population-simulation.md §13 Design 5) — `#genome
  source=synth-4300` builds one shared sparse 4338-gene template in a single pass, and
  each cell's expression is one row of the shared matrix; expression-driven dFBA gating
  scales the "essential-gene knockout → biomass → 0" closure from single cells to a whole
  population (`#sim` outputs `triggered_genes`).
- `38_flow_biofilm.helix`: **rod-cell colony inside an LBM microfluidic channel**
  (doc/18-programmable-cell-population-simulation.md §13 Design 6 Level 2+3) — `#sim
  lbm=true` recomputes the flow field around rod obstacles every tick with the D2Q9
  solver; the local flow drifts rods downstream by Stokes drag and Hertzian contacts keep
  rods non-overlapping; glucose/O₂ are advected with the same flow field, so colony-edge
  O₂ stays above core O₂ (nutrient boundary layer).
- `39_lbm3d_biofilm.helix`: **3D LBM flow through a colony** (doc/18 §13 Design 6 Level 2
  3D) — `#sim lbm_3d=true` solves Navier–Stokes on the D3Q19 stencil over a depth-volume
  (requires `grid_depth > 1`); the dense biofilm acts as no-slip obstacles and the
  refreshed `FlowField3D` drives 3D cell drift (x/y/z) plus substrate advection.
- `40_spatial_evolution.helix`: **spatial range-expansion evolution** (doc/18 §13 Design 1;
  Bosshard et al. 2020, BMC Genomics 21:232) — the "large helix example" of the
  population roadmap. `#sim kind=spatial_evolution` mutates DNA genotypes with the real
  mutation spectrum, recompiles them, and scores each as a spatial colonizer on the
  32×32 lattice (80-cell inner colonies); fitness is
  `colony_radius_sites × core_survival − metabolic_cost`, and truncation selection feeds
  the next generation. Mean fitness roughly doubles in the first generations (fast
  colonizers fix) before plateauing at the mutation-selection balance
  (`helixlang examples/40_spatial_evolution.helix`, ~25 s, deterministic `seed=42`).

**Frontier-delivered Python API applications** (delivered as directly runnable Python
rather than `.helix` — the three extension directions, the workflow scripts paired with
examples 28–30, and the D3Q19 LBM tooling):

- `apps/omics_calibration.py`: CRISPRi PerturbSeq calibration — synthesizes noisy omics
  data and uses `fit_parameters` (inverse-variance weighted) to recover model parameters;
  benchmark improvement 0.858, corr 0.998, DE 1.0.
- `apps/whole_cell_scale.py`: whole-genome essentiality screening — loads a FASTA genome,
  knocks out genes one at a time → re-solves FBA → judges whether growth fails; 19 core
  metabolic genes are 100% consistent with EcoCyc essentiality on glucose minimal medium.
- `fate_analysis_workflow.py` / `directed_evolution_workflow.py` /
  `virtual_cell_workflow.py`: the Python-side loops paired with examples 28/29/30
  (bifurcation scan + stochastic switching, oracle-guided directed evolution,
  calibrate→predict benchmark).
- `d3q19_lbm_pressure_channel.py` / `d3q19_lbm_bench.py`: D3Q19 pressure-driven channel
  workflow and 100×100×50 benchmark (205 ms/tick), the LBM-3D counterpart of example 39.
- `CellPopulation3D` numpy large-scale path: near-linear scaling for 10⁴–10⁵ cells
  (10⁵ full steps including diffusion ≈ 0.9 s/tick); the vectorized metabolism path is
  bit-for-bit identical to the pure-Python path.

---

## 8. How HelixLang maps onto these problems

The eight frontier problems mapped onto HelixLang's existing capabilities. Three states:
**✅ ready now** (module exists, just write an example), **🔧 one step away** (needs a
little new wiring code), **🕐 long term** (needs groundwork first).

| # | Frontier problem | Existing HelixLang building blocks | State |
|---|---|---|---|
| P1 | Synthetic-consortium distributed control / ratio control | `population.py` (each cell programmable + AI-2 diffusion + 10 µM quorum sensing + lineage tracking), `evolution.py` | ✅ Delivered (`apps/consortium.py` + example 21) |
| P2 | Spatial pattern formation / synthetic morphogenesis | Gray-Scott reaction-diffusion + `#morphogen` declarative wiring + 2D/3D diffusion + L-system | ✅ Delivered (`apps/morphogen_gradient.py` + examples 22/25) |
| P3 | Stochastic expression and cell fate | `stochastic.py` (telegraph + Gillespie SSA) + `grn.py` bistable switch + ODE solver | ✅ Delivered (`apps/fate_analysis.py`: bifurcation scan + stochastic switching + critical slowing down) |
| P4 | Virtual cell | `virtual_cell.py` (GRN→central dogma→FBA→energy budget + cell-cycle/adder/folding/enzyme-capacity four layers) + `fit_parameters` + `apps/whole_cell_calibration.py` mixed-observable calibration loop | ✅ Delivered (Phases 1–5 fully implemented, examples 31/34) |
| P5 | ML-guided directed evolution | `protein_fitness.py` (BLOSUM62 + ESM2Oracle) + `evolution.py` (Wright–Fisher) | ✅ Delivered (`apps/protein_evolution.py`: oracle-guided loop vs random baseline) |
| P6 | Dynamic metabolism (dFBA) | `metabolism.py` (`DynamicFluxBalance`: Monod boundaries + forward Euler + byproducts) | ✅ Delivered (`apps/spatial_dfba.py` + example 24, wired into the population loop) |
| P7 | DNA data storage | `dna_codec.py` (Goldman + Fountain + Reed-Solomon + error models) | ✅ Delivered (`apps/dna_storage.py`: coding benchmarks + scenario cost decisions) |
| P8 | Biofilm spatial ecology | 2D/3D populations + nutrient fields + shoving mechanics + CROMICS crowding diffusion | ✅ Delivered (`CellPopulation3D` + `ConcentrationField3D` + `shoving`/`force` mechanics + CROMICS crowding diffusion, 56 3D tests green; vectorized large-scale 3D is a later extension) |

**Core conclusion**: HelixLang's most distinctive value is that it packs "**DNA compiler +
GRN + diffusion + quorum sensing + evolvable genotypes + morphogenesis**" into a single
runtime. This lands exactly at the intersection of P1/P2/P3/P6 — and **no open-source tool
today** has all four capabilities at once (see `doc/10-frontier-biology-analysis.md` §3 and
the SOTA tool comparison).

---

## 9. Solution designs (problem by problem)

Every design below lands in concrete modules and files and obeys the project's three iron
rules: **zero external dependencies in the core (stdlib-first), physical units carried
end-to-end, tests/ruff/mypy all green**.

### Design 1: Consensus decisions and ratio control in synthetic consortia (P1)

**Goal**: verify in silico the You 2004-style "a few flip the whole group" consensus
circuit, and 2026-Nature-style "single founder auto-differentiates + ratio control"
devices (`s41586-026-10259-3`).

**How**:

1. **Already usable**: `examples/21_quorum_circuit.helix` already demonstrates
   "producer strain + sensor strain + quorum-sensing switch".
2. **New example** (`examples/`): write a "two-group differentiation" program — one founder
   cell splits into two subgroups at a specified tick, the subgroups get their roles
   hardcoded via `#regulate`, and `population.py`'s lineage tracking is used to tally how
   the two-group ratio evolves over time.
3. **New small module** (`apps/consortium.py`): encapsulate a "consortium task"
   abstraction — each group specifies `producer / sensor / actuator` roles, signal-channel
   and target ratio; auto-generate and run the corresponding `.helix` source.
4. **Verify**: add `tests/test_consortium.py`, asserting that "after the sensor strain
   passes the 10 µM signal threshold, the whole population flips its expression state
   within N ticks" (reproduces consensus decisions) and that the ratio-control steady-state
   error is <5%.

**Value**: "multi-strain division of labor raises yield" (cellulose degradation etc.) in
bioprocesses requires knowing *what the ratio will drift to* — this design turns the
2024–2026 consortium-control literature into repeatable digital experiments.

### Design 2: Programmable morphogenesis and the "gradient → gene → behavior" loop (P2)

**Goal**: turn the developmental-biology main line "morphogen gradient decides cell fate"
into a programmable, evolvable pipeline.

**How**:

1. **Already usable**: `#morphogen gene=<name> channel=U|V gain=<value>` already feeds the
   diffusion-field concentration directly into any gene's GRN input
   (`tests/test_g9.py` verifies this).
2. **New example** (delivered: `examples/25_morphogen_gradient.helix` +
   `apps/morphogen_gradient.py`): synthesize a Basu 2005-style "source–host" gradient —
   source cells keep secreting a morphogen, host cells enter different expression states
   by concentration threshold, producing concentric cell-type maps on 2D/3D grids.
3. **Enhancement**: add an optional `threshold` attribute to genes in `grn.py` so the GRN
   output forms a sharp "differentiation boundary" at the threshold (closer to the
   switch-like developmental response than a pure sigmoid).
4. **Verify**: add `tests/test_morphogen_gradient.py` (delivered), asserting that "from
   source to host, expression state shows ≥2 discrete zones along the radius", and rerun
   it once in 3D (`CellPopulation3D`).

**Value**: this is a necessary step toward "synthetic organs/organoids", and the physical
foundation of P3 (evolution).

### Design 3: Digital evolution — DNA programs driven by natural selection (P3 escalation)

**Goal**: string together "genotype (a stretch of DNA) → compile → run in a population /
environment → compute fitness → mutate + select" into one complete evolutionary loop
(Avida-style, but with real codons + real physical simulation).

**How**:

1. **Wiring** (the most critical "one step away" in the whole project): add
   `digital_evolution.py` under `apps/`:
   - compile random/mutated DNA to bytecode with `Compiler` (mutation happens directly on
     the DNA string, using the *real* mutation spectrum from `evolution.py`, not random
     scrambling);
   - run the compiled programmable cells for several ticks with `population.py`;
   - define **fitness** from the VM's trace (e.g. `OP_SIGNAL` count, energy balance,
     whether it divides);
   - select the next generation with `evolution.py`'s Wright–Fisher.
2. **New example**: `examples/23_evolve_signal.helix` — evolve "emit as much signal as
   possible" behavior.
3. **Verify**: add `tests/test_digital_evolution.py`: after G generations the population's
   mean fitness rises monotonically, and the "signal-emitting genotype" frequency is
   significantly above the random-drift baseline (run a no-selection control for
   comparison).

**Value**: directly lands the digital-evolution frontier problem of "in-silico evolution of
genotype→phenotype mappings", and is a differentiating capability unique to HelixLang
(Avida cannot do real codons + morphology).

### Design 4: Cello-style "Boolean logic → DNA" design-automation loop (P4 escalation)

**Goal**: a user writes a truth table (e.g. "output C if and only if input A and not B"),
and the tool automatically emits: DNA sequence + predicted dynamics + an SBOL3 file the wet
lab can use.

**How**:

1. **Already usable**: `apps/synbio_automation.py` implements truth table → logic-gate
   library allocation → DNA sequence → SBOL3 export; `interop.py` parses SBOL3 both ways;
   `central_dogma.py` predicts expression dynamics.
2. **New** (delivered): string `synbio_designer.py` (expression cassette/vector design)
   together with `synbio_automation.py` to emit a one-stop report from "logic gate" all
   the way to "complete plasmid + simulated time curves" — `examples/26_cello_workflow.helix`
   demonstrates the full loop.
3. **Verify**: use the NOT/NAND/XOR circuits published by Cello 2.0 as gold standards and
   assert the simulated truth table matches the target (`tests/test_synbio_automation.py`
   already has 30 cases to extend).

**Value**: this is synthetic biology's "EDA tool" dream (like electronic design automation
in the chip industry); SBOL3 export makes designs interoperable with the SynBioHub/Cello/
LOICA ecosystem.

### Design 5: DNA data storage for applications (P7)

**Goal**: turn the already-mature encoders into a research tool that "answers literature
questions": compare the **cost–robustness trade-off** of different encodings (Goldman vs
Fountain vs Reed-Solomon) under a realistic error model.

**How**:

1. **Already usable**: `dna_codec.py` supports the two mainstream encodings +
   Reed-Solomon error correction + simulated PCR error injection.
2. **New small tool** (`apps/dna_storage.py` extension): `benchmark_codecs()` — following
   the Nat Commun 2026 methodology, sweep error/loss rates at fixed coding rates
   (0.5/1.0/1.5 bit/nt) and output a "per-GB cost, tolerated-loss range, decode time"
   comparison table.
3. **Verify**: add `tests/test_dna_codec_bench.py`, asserting that the literature
   conclusion "loss tolerance rises as coding rate falls" holds in simulation (e.g. the
   0.50 bit/nt Fountain tolerates ~60% sequence loss, consistent with the literature).

**Value**: pushes "what DNA storage actually is" from proof-of-concept to a
**scenario-decision tool** — exactly the direction the 2026 literature calls for ("which
data scenario DNA storage fits best" is still undecided).

### Design 6: Coupling dynamic metabolism with nutrient fields (P6)

**Goal**: make dFBA run not just "in a single reactor" but **wired into the population
loop**: every cell feeds from a shared glucose/O₂ diffusion field, solves FBA with local
concentrations, and excretes byproducts back into the field — forming real "spatial
competition".

**How**:

1. **Already usable**: `environment.py`'s `ConcentrationField` (Monod uptake, depletion,
   chemostat), `metabolism.py`'s `DynamicFluxBalance`, and `population.py`'s programmable
   populations.
2. **Wiring**: add an optional `dFBA` mode to `population.py`'s tick loop — when
   `config.metabolism == "dFBA"`, each cell's energy settlement is done by
   `DynamicFluxBalance.update_from_environment` + `apply_to_environment`, replacing the
   constant `ENERGY_INTAKE_PER_STEP`.
3. **New example**: `examples/24_spatial_diauxie.helix` — the grid has two regions: a
   glucose-rich region (fast growth, acetate overflow) and a glucose-poor region
   (stalled), letting the population self-organize into "glycolytic fast-growth zone /
   respiratory slow-growth zone".
4. **Verify**: add `tests/test_population_dfba.py`: assert total glucose conservation
   (uptake = biomass-growth carbon equivalent + byproducts) and that the population stops
   dividing in the poor region.

**Value**: this is the metabolic-side engine of P8 (biofilm), and the basis for
MiMICS-2024-style "omics-guided metabolic state".

### Design 7: Noise-driven cell-fate decision research scaffolding (P3 deepening)

**Goal**: turn "bistable switch + telegraph noise + bifurcation analysis" into a ready-made
research tool that directly answers "at a given parameter set, what fraction of isogenic
cells will switch fate".

**How**:

1. **Already usable**: `grn.py`'s bistable switch (Gardner 2000 model), `stochastic.py`'s
   `telegraph_fano_factor` / `gillespie_telegraph`.
2. **New** (`apps/fate_analysis.py`, delivered):
   - `bistability_scan()`: grid-root-find + bisection refinement on the fixed-point map
     `a* = S(-w·S(-w·a*))`, classifying stable/unstable by the local map slope
     (|slope|<1), directly plotting the bifurcation diagram — reproducing the saddle-node
     giving birth to two fate branches at w≈5.5 (w=5 monostable, w=6 bistable;
     Gardner 2000);
   - `switching_rate()`: run N stochastic trajectories with two-state-promoter
     Fano-matched Gaussian noise (Peccoud–Ycart 1995) and tally fate-switch rates,
     supporting an optional shared-translation-resource-pool term `1/(1+res·(a+b))`
     (aligned with Goetz 2025 "resource competition induces stochastic switching": the
     stronger the resource, the higher the switching rate);
   - `critical_slowing_down()`: the lag-1 autocorrelation of a single long near-critical
     trajectory → 1 (Scheffer 2009 early-warning signal, i.e. the observational proxy for
     "noise amplification near criticality").
3. **Example**: `examples/28_fate_analysis.helix` (annotated), with the directly runnable
   Python script `examples/fate_analysis_workflow.py`.
4. **Verify**: `tests/test_fate_analysis.py` (10 cases): assert the saddle-node location
   (w=5 monostable, w=6 bistable), resource competition amplifies switching (res=0 →
   0.007, res=0.5 → 0.443), and near-critical autocorrelation monotonically approaches 1
   (0.608 → 0.908).

**Value**: turns noise from "disturbance" into "design freedom" — the frontier claim of the
2025 synthetic-circuit literature.

### Design 8: ML-guided directed evolution, the "digital DBTL" loop (P5)

**Goal**: without wet experiments, first run the "ESM-2 scoring → mutant library →
simulate/score → next round" loop in silico as a pre-screen before the wet lab.

**How**:

1. **Already usable**: `protein_fitness.py`'s `ESM2Oracle` (zero-shot pseudo-likelihood
   scoring) + `BLOSUMOracle` + `rank_variants`; `evolution.py`'s Wright–Fisher and
   mutation.
2. **New** (`apps/protein_evolution.py`, delivered): `guided_directed_evolution()` — each
   round scores a single-residue mutant library against wild-type with a zero-shot oracle
   (`make_oracle()`: ESM-2 pseudo-likelihood, auto-falling back to BLOSUM62 when missing),
   selects Top-K on the weighted-BLOSUM62 GB1 chemical-tolerance landscape (interface-window
   weighted, Wu 2016), and the best becomes the next round's parent; compared from the same
   start against a no-oracle random-sampling baseline (aligned with EVOLVEpro /
   MULTI-evolve / FSFP's DBTL loop).
3. **Example**: `examples/29_directed_evolution.helix` (annotated), with the directly
   runnable Python script `examples/directed_evolution_workflow.py`.
4. **Verify**: `tests/test_protein_evolution.py` (12 cases): on GB1, assert "oracle-guided
   multi-round sampling cumulative best fitness significantly beats the random baseline"
   (guided +0.207 vs random +0.039), and oracle predictions vs landscape Spearman rank
   correlation matches the literature (ρ≈0.97, ProteinGym metric).

**Value**: upgrades P5 from "there is a scorer" to "a reproducible guided-evolution
protocol", with the core being zero-external-dependency (auto-fallback to BLOSUM62 when
ESM-2 is missing).

### Design 9: Virtual cell parameter calibration and standard benchmarks (P4 delivery)

**Goal**: stop relying only on literature constants for `virtual_cell.py` — instead **fit
the model to experimental data with `fit_parameters`**, and submit repeatable benchmarks the
Virtual Cell Challenge way. At the same time, complete the virtual cell's "physical realism
layer" — cell-cycle timing, size control, protein folding, enzyme-capacity constraints — so
what gets calibrated is **a cell with physical meaning**, not just an energy-balance toy.

**How**:

1. **Already usable**: `virtual_cell.py`'s `VirtualCell`, `fit_parameters`,
   `run_biofilm_benchmark`, `perturbation_response`, `encode_gene`.
2. **New example** (delivered): `examples/30_virtual_cell.helix` (annotated `.helix` form,
   plus the Python API script `examples/virtual_cell_workflow.py`): feed a time-series
   dataset (e.g. "doubling time + several protein-concentration trajectories") and it
   automatically does random search + coordinate refinement for parameter fitting, outputs
   the calibrated cell model, then predicts "the growth curve after knocking out a gene".
3. **New benchmark entry point**: `apps/virtual_cell_bench.py` (`VirtualCellBench` +
   `run_virtual_cell_benchmark`) exports "calibrate → predict" results as a standard table
   for comparison with tools like iDynoMiCS 2.0.
4. **New physical realism layer** (delivered, corresponding to Phases 1–5 of
   `15-whole-cell-realism.md`):
   - **Phase 1** cell cycle: `replication_mode=cooper_helmstetter`, C-phase replication +
     D-phase segregation, replication origins scheduled by doubling time (overlapping
     replication allowed);
   - **Phase 2** size control: `division_rule=adder`, divide only when volume exceeds
     "birth volume + Δ" (Δ = `adder_volume_um3`, with optional noise);
   - **Phase 3** folding: `protein_maturation_mode=chaperone`, chaperone-mediated folding
     with balanced fold fraction k_fold, deducting folding ATP;
   - **Phase 4** enzyme capacity: `enzyme_capacity=true`, GECKO-style kcat scaling
     constraining each enzyme-catalyzed reaction flux;
   - **Phase 5** calibration loop: `apps/whole_cell_calibration.py` — inverse-variance
     weighted fitting from mixed observables (growth curve + protein abundance + cell size
     + biomass flux) inverting the four hidden parameters
     `adder_volume_um3 / k_fold / enzyme_scale / maintenance_atp_per_min`;
     `tests/test_whole_cell_calibration.py`'s end-to-end loop case asserts all four
     parameters are recovered within 10% tolerance.
5. **Verify**: `tests/test_virtual_cell.py` (existing 17 cases) extended with the
   "calibrate–predict" closed-loop case `test_calibration_prediction_closed_loop`: fit the
   true parameters from synthetic data (recovers biomass_to_atp = 5.000e6, relative error
   0.0%), then predict an independent condition (GLC 20 mM/60 min) whose energy matches
   ground truth exactly (284,982,602 ATP, identical alive/protein counts), `passed=True`.

**Value**: whole-cell modeling (Virtual Cell Challenge 2025) explicitly demands
"standardized benchmarks + calibratability" — this is HelixLang's decisive leap from
"teaching toy" to "trustworthy research tool"; examples 31/34 let you reproduce "a cell
with a physical realism layer + parameter inversion" with one command.
