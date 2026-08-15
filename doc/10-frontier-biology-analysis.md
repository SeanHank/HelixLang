# Frontier Biology Analysis and Upgrade Plan

> Analyzes which frontier biological problems HelixLang can already attack with its
> current runtime, benchmarks the project against the state of the art in
> computational biology, and lays out a tiered upgrade plan to close the gaps.
>
> Status: **B1–B10 implemented** (T1.1–T1.5, T2.1–T2.7, T3.1–T3.5 landed in `src/`,
> `apps/`, and `examples/`). G9 declarative couplings resolved. Scope decisions are
> for the maintainer.
>
> Date: 2026-08 · Baseline: **1766 tests passing**, coverage **89%**, `ruff` + `mypy` clean
> (`/opt/anaconda3/envs/helix/bin/python`). Runtime runs on physical units end-to-end
> (1 tick = 1 min, ATP molecule counts, µM signals, µm²/s diffusion — see `units.py`).

---

## Executive Summary

HelixLang's unique positioning — **"DNA as program"** (codon→opcode compilation +
GRN + L-system + Gray-Scott + multicellular population + metabolic FBA + evolution
engine) — already lets it tackle several classes of frontier biological problems:

1. **Population-level logic and consensus (quorum sensing) circuits in synthetic
   consortia** — doable now (`population.py` already has the AI-2 diffusion field +
   quorum sensing + lineage tracking);
2. **Spatial pattern formation and synthetic morphogen gradients** (Turing patterns
   + morphogens + gene feedback) — doable now;
3. **In-silico evolution of the genotype→phenotype mapping (evo-devo / digital
   evolution)** — one step away (needs `evolution.py`, the compilation pipeline,
   and the VM/population runtime wired together);
4. **Synthetic biology design automation (compile-to-DNA, Cello-style)** — one step away;
5. **DNA storage and in-vivo molecular recording** — essentially complete
   (Goldman/Erlich encodings + CRISPR);
6. **Dynamic metabolic modeling (dFBA) / diauxic growth** — needs a dynamic layer
   on top of the static FBA;
7. **Noise-driven cell-fate decisions / whole-cell (virtual cell) modeling /
   ML-guided directed evolution** — mid- to long-term.

The main bottleneck was an **architectural split**: `vm.CellVM` (single-cell,
programmable) and `population.CellPopulation` (multi-cell, not programmable) were
two disconnected runtimes. Every frontier application that needs "multicellular ×
programmed gene circuits × spatial environment" was blocked by this split. **Batch
B1 (§7) closed the split** — population cells are now programmable (per-cell GRN +
bytecode under an ops budget, shared diffusing fields). The tiered plan (§5) builds
the frontier applications on top of that foundation, with a verification strategy
(§6) and numbered implementation batches (§7).

---

## 1. Document Structure (How to Read This Document)

| Section | Content |
|---|---|
| §2 | Current capability → frontier problem mapping (what it can already do, readiness) |
| §3 | Benchmark against state-of-the-art tools (iDynoMiCS 2.0, NUFEB, MiMICS, Cello, whole-cell, PLM-directed evolution, stochastic GRN) |
| §4 | Gap analysis (what must improve, and why each gap blocks frontier work) |
| §5 | Tiered upgrade plan (concrete, per-module) |
| §6 | Verification strategy (tests / benchmarks / quality gates) |
| §7 | Implementation batches (numbered roadmap) |
| §8 | New literature references added by this analysis |
| §9 | Appendix: capability inventory summary |

Terminology: **frontier** here means problems actively worked on in the 2024–2026
computational-biology literature (whole-cell/virtual-cell models, individual-based
biofilm simulators, dynamic flux balance, stochastic gene expression, synthetic
design automation, ML-guided directed evolution, spatial-omics-guided multi-scale
models).

---

## 2. Current Capability → Frontier Problem Mapping

Readiness legend: **Now** = buildable today with existing modules (possibly an
example/glue only); **Near** = needs one or two focused modules; **Later** =
needs the foundation tiers first.

| # | Frontier problem | Current building blocks | Readiness | Key gap |
|---|---|---|---|---|
| P1 | **Synthetic quorum/consensus circuits & population-level logic** (a fraction of cells flips the whole colony; population control, You et al. 2004) | `population.py`: AI-2 diffusion (sub-stepped, D≈60), quorum threshold 10 µM, up to 10⁴ agents, lineage + Shannon diversity | **Now** | ✅ resolved (B1): per-cell GRN + bytecode in `population.py`; see `examples/21_quorum_circuit.helix` |
| P2 | **Spatiotemporal pattern formation by cell–cell signaling + morphogens** (synthetic morphogen gradients, Turing pattern synthesis, Basu et al. 2005; Payne et al. 2013 coupled oscillators) | Gray-Scott field (`reaction_diffusion.py`, Pearson presets) + `OP_SIGNAL`/`OP_EMIT_MORPHOGEN` + morphogen→gene feedback + L-system morphology | **Now** (2D) | ✅ largely resolved (B1): population cells execute the VM path on shared fields; see `examples/22_pattern_synthesis.helix`. G9 removed the hard-coded `"pigment"` coupling: `#morphogen gene=<name> channel=U\|V gain=<float>` wires any channel to any gene declaratively (legacy pigment fallback preserved) |
| P3 | **In-silico evolution of GRN programs** (evo-devo: evolve DNA/genotype for a target phenotype; Avida/Tierra line, Lenski et al. 2003) | `evolution.py` (Wright–Fisher, mutate, dN/dS) + `compiler.py`/`vm.py` (DNA→bytecode→behavior) | **Near** | ✅ per-cell programs now exist (B1, `population.py`); remaining: `mutate()` not wired to recompile+resimulate a `.helix` program, fitness not connected to VM/population traces |
| P4 | **Synthetic circuit design automation (compile-to-DNA, Cello-style)** | Codon table as an ISA; `synbio_designer.py` (cassette/vector, CAI, GenBank); `central_dogma.py` (predicted expression) | **Near** | No Boolean-logic → DNA front end, no SBOL3 export, no part-library characterization model |
| P5 | **DNA storage & in-vivo molecular recording** | `dna_codec.py` (Goldman + Erlich fountain, Reed-Solomon, error models), `crispr.py`, `apps/dna_storage.py` | **Now** | In-vivo recording backend (`target=in_vivo_crispr`) not wired |
| P6 | **Dynamic metabolic modeling (dFBA) & diauxie / metabolic switching** | `metabolism.py` (static FBA, 37-rxn E. coli core, pure-Python simplex, iJO1366 optional via cobrapy); diauxic growth example | **Near** | ✅ resolved (T2.1/B6): `DynamicFluxBalance` + environment coupling; see `tests/test_dFBA.py`, `examples/20_diauxic_growth.helix`. Remaining: per-cell dFBA in the population loop |
| P7 | **Noise-driven cell-fate decisions & stochastic switching** (bistability + noise, bet-hedging; Goetz et al. 2025; Xue et al. 2024) | `grn.py` toggle switch (Gardner 2000), Hill cooperativity, physical decay | **Later** | ✅ resolved (T1.4/B4): telegraph noise + Gillespie SSA in `stochastic.py`, optional in `grn.py`; see `tests/test_stochastic.py`. Remaining: cell-fate decision studies on top |
| P8 | **ML-guided directed evolution & protein fitness oracles** (EVOLVEpro 2025, MULTI-evolve 2025, FSFP 2024) | `evolution.py` Wright–Fisher + `protein_structure.py` (GOR IV, IUPred) | **Later** | No protein language model (ESM) integration; fitness = user-supplied heuristics |
| P9 | **Whole-cell / virtual-cell modeling** (Karr et al. 2012; Bunne et al. 2024 AIVC; Virtual Cell Challenge 2025) | `central_dogma.py` + `grn.py` + `metabolism.py` + `cell.py` (all physical units, cited) | **Later** | No integrated cell-cycle/budgets, no parameter estimation, no omics calibration |
| P10 | **Spatial microbial community / biofilm structure** (nutrient gradients, crowding, heterogeneity; iDynoMiCS 2.0 2024, CROMICS) | `population.py` 2D lattice + signal diffusion; agent-based binary fission | **Later** | ✅ nutrient/O₂ fields (T1.2), volume exclusion/crowding + mechanics (T1.3/B3), crowding-dependent D (T2.6) implemented; remaining: 3D |

**Readiness verdict:** P1, P2, P5 are buildable today; P3, P4, P6 are one or two
modules away; P7–P10 require the foundation tiers. The project's highest-value
sweet spot is **P1+P2+P3 in combination** — no other open tool unifies "DNA-as-
program compiler + GRN + diffusion + quorum + evolvable genotypes + morphology".

---

## 3. Benchmark Against the State of the Art

| Area | State of the art | What SOTA does | HelixLang today | Gap |
|---|---|---|---|---|
| Multi-agent microbial simulation | iDynoMiCS 2.0 (2024), NUFEB (2019, LAMMPS-based) | 10⁶–10⁷ agents in 3D, arbitrary kinetics, force-based mechanics, fluid coupling | 10⁴ agents, 2D lattice, no mechanics, no nutrients | Scale, 3D, mechanics, environment |
| Metabolic community models | MiMICS (2024): omics-guided GENRE + HAL ABM; CROMICS: crowding-dependent diffusion + TFA | Per-agent metabolic state, nutrient reaction–diffusion, transcriptomics-guided switching | Static per-tick FBA, flat energy cash flow, no substrate field | dFBA coupling, environment, omics-guided states |
| Dynamic metabolism | dAMN (2025), COSMIC-dFBA (2024), COBRApy dFBA (Mahadevan 2002 SOA) | dFBA with dynamic uptake bounds, ML surrogate fluxes, lag-phase modeling, genome-scale (iML1515) | Static FBA (LP biomass max), 37-reaction core | Dynamic bounds, substrate integration, scale |
| GRN / circuit simulation | GRN_modeler (2025), COPASI, Tellurium; differentiable Gillespie (2025) | Deterministic ODE + stochastic SSA, parameter estimation, spatial simulation | Discrete-time recurrence, deterministic, no SSA | ODE/SSA solvers, noise, bifurcation tooling |
| Genetic design automation | Cello 2.0 (Nat Protoc 2022), LOICA (SBOL3), CELLM (LLM+Cello 2025) | Boolean Verilog → DNA sequence → predicted dynamics; SBOL ecosystem | Codon-table ISA + cassette designer, GenBank export, no SBOL/Boolean front end | SBOL3 export, gate/part libraries, logic front end |
| Protein engineering | EVOLVEpro (2025), MULTI-evolve (2025), FSFP (2024) | PLM (ESM) zero-/few-shot fitness oracles guiding rounds of mutagenesis | Heuristic structure predictors (GOR IV, IUPred, simplified TMHMM) | PLM fitness oracle, closed-loop DBTL glue |
| Whole-cell / virtual cell | Karr et al. 2012 *M. genitalium*; AIVC proposals (Bunne 2024); Virtual Cell Challenge (2025) | Integrated submodels + data calibration + perturbation prediction | Physical-unit cell state + central dogma + FBA (uncoupled) | Model integration, parameter estimation, validation data |
| Evolution | Avida (2003), digital evo-devo, MLDE | Evolve genotypes/programs for target functions with selection | Wright–Fisher sequence evolution, dN/dS | Evolution wired to phenotype (VM trace) fitness |

**Positioning note:** HelixLang is not competing with iDynoMiCS/NUFEB on raw
multi-agent scale, nor with COBRApy on genome-scale metabolism, nor with Cello on
production design automation. Its *differentiator* — the codon→opcode compiler with
real genetic-code tables, physical units, and a unified GRN/morphology/signaling
runtime — targets the **synthetic-biology research niche**: "compile a circuit,
put it in cells, let them communicate and evolve in space". The upgrade plan below
is scoped to make that niche *scientifically credible* (stochasticity, environment,
mechanics) rather than to chase SOTA scale.

---

## 4. Gap Analysis

Each gap: **Current** → **Gap** → **Why it blocks frontier work** → **Related literature**.

### G1 — Two disconnected runtimes (THE critical architectural gap)
- **Current:** `vm.CellVM` is a single programmable cell (genome, GRN, Gray-Scott field, ops quota); `population.CellPopulation` is multi-cell (≤10⁴, AI-2 diffusion, quorum, lineage) but carries **no genome, no GRN, no bytecode**. They share neither code paths nor state.
- **Gap:** No way to run "programmed cells in a spatial population".
- **Blocks:** P1 (programmable quorum circuits), P2 (signaling-driven patterning), P3 (evolution of multicellular behavior), P10. It is the prerequisite for almost everything else.
- **Literature:** iDynoMiCS 2.0 modules (agent properties assembled from orthogonal modules, 2024); MiMICS per-agent metabolic states (2024); NUFEB "biological + chemical + physical" integration (2019); `06-engineering-design.md` §6.2 already sketches "OP_DIVIDE creates daughters; GRN per cell; shared fields" — this is the documented multicellular extension.

### G2 — Fully deterministic core; no stochastic gene expression
- **Current:** GRN is a discrete-time recurrence (`level' = clamp(decay·level + (1−decay)·sigmoid(...))`); Gray-Scott, cell dynamics, and L-systems are deterministic (seeded RNG only in division partitioning and bio instructions).
- **Gap:** No intrinsic/extrinsic noise, no low-copy-number effects, no telegraph (two-state promoter) model, no Gillespie SSA.
- **Blocks:** P7 (noise-driven fate decisions, stochastic switching, bet-hedging), realistic variability in synthetic circuits (noise amplifies near bifurcations — resource-competition bistability, Goetz 2025).
- **Literature:** differentiable Gillespie (Rijal 2025, eLife); two-state promoter (Jones 2014); LNA analysis of GRN motifs near criticality (bioRxiv 2025); logic-incorporated GRN fate-decision (Xue 2024, eLife).

### G3 — Static FBA; no nutrient/O₂ environment; flat energy cash flow
- **Current:** `metabolism.py` solves steady-state FBA (LP) over a 37-reaction E. coli core; the population runtime instead uses flat `ENERGY_INTAKE_PER_STEP`/`METABOLIC_COST_PER_STEP` constants; no substrate depletion, no competition, no carrying capacity.
- **Gap:** No dynamic flux balance, no nutrient fields (glucose/O₂), no Monod uptake, no per-cell metabolic state.
- **Blocks:** P6 (diauxie, metabolic switching), P10 (nutrient-gradient-driven heterogeneity — the central mechanism of biofilm niches), realistic fitness for evolution.
- **Literature:** Mahadevan 2002 (dFBA SOA); dAMN 2025 (neural-mechanistic dFBA, iML1515, lag phase); COSMIC-dFBA 2024; MiMICS 2024 (transcriptome-guided metabolic states in biofilms); CROMICS (crowding modifies effective diffusion).

### G4 — No cell mechanics: stacking, no exclusion/adhesion/crowding
- **Current:** multiple cells may occupy the same lattice site; no shoving, no volume, no EPS, no mechanical repulsion/adhesion; crowding effects absent.
- **Gap:** A cell-sorting/interference framework and crowding-dependent diffusion.
- **Blocks:** P10 — the literature shows crowding alters results above ~14% volume fraction (CROMICS 2021); mechanical stress anisotropy drives nematic ordering in confined biofilms (Soft Matter 2024); iDynoMiCS 2.0 offers force-based mechanics.
- **Literature:** CROMICS (PLoS Comput Biol 2021); Li et al. 2024 Soft Matter (stress anisotropy → nematic ordering); iDynoMiCS 2.0 2024.

### G5 — 2D runtime only
- **Current:** cell movement, diffusion, signaling are 2D; `morphology_3d.py` is offline L-system geometry only.
- **Gap:** 3D diffusion/neighborhoods; 3D colony growth.
- **Blocks:** quantitative comparison with NUFEB/iDynoMiCS 2.0 (3D), realistic biofilm volume packing.
- **Literature:** NUFEB 2019 (3D, 10⁷ agents); iDynoMiCS 2.0 2024 (3D).

### G6 — Discrete-time GRN, fixed 1-min tick; no continuous-time kinetics
- **Current:** GRN is a discrete recurrence at 1-tick (1-min) resolution; no intra-tick integration, no ODE/SDE solvers, no bifurcation analysis tooling.
- **Gap:** Optional RK45/SSA solvers so the same circuit can be studied at continuous time and compared to COPASI/Tellurium benchmarks.
- **Blocks:** P7 (noise/criticality studies need continuous-time reference), P9.
- **Literature:** GRN_modeler 2025 (COPASI + SimBiology solvers, deterministic+stochastic); LNA framework 2025.

### G7 — Heuristic protein structure; no ML/protein-language-model fitness
- **Current:** Chou-Fasman, GOR IV/DSSP, simplified TMHMM, faithful IUPred2A port. No ESM/AlphaFold integration; evolution fitness is user-supplied.
- **Gap:** A pluggable fitness oracle (zero-/few-shot PLM) for ML-guided directed evolution.
- **Blocks:** P8.
- **Literature:** EVOLVEpro (Jiang 2025, Science 387:eadr6006); MULTI-evolve (Tran 2025, Science aea1820); FSFP (Nat Commun 2024, 15:5566); PLM directed-evolution reviews (Cell Systems 2025).

### G8 — No omics/data integration or exchange standards (SBML/SBOL)
- **Current:** no importer/exporter for SBML, SBOL, or expression datasets; parameters are literature-cited constants, not calibrated to data.
- **Gap:** SBOL3 export (design automation), SBML import (validate against BioModels), expression-matrix → GRN/FBA-bound mapping (MiMICS-style), parameter estimation.
- **Blocks:** P4 (interop with Cello/SynBioHub ecosystem), P9/P10 (data calibration, spatial-omics guidance).
- **Literature:** LOICA 2022 (SBOL3); design-automation review (BMC Bioinformatics Data 2024); MiMICS 2024; whole-cell data-integration review (FEBS J 2024).

### G9 — Hard-coded couplings and closed feedback loops
- **Current:** VM morphogen feedback targets a gene literally named `"pigment"`; `OP_GROW_LSYSTEM` always uses L-system #1, rule set 0; `OP_DIVIDE` in the VM only halves energy; `OP_CALL_GENE` has a 2-bit operand limit (mitigated by `call_target=`).
- **Gap:** Declarative morphogen→gene wiring; operand/address-space expansion; real division.
- **Blocks:** P2 generality, P3 evolvability, expressiveness.
- **Literature:** none needed (internal correctness/design debt).
- **Status:** ✅ resolved (B9, `tests/test_g9.py`): the parser accepts `#morphogen gene=<name> channel=U|V gain=<float>` (defaults V, 0.1) into `Program.morphogen_feedback`; the VM wires the named channel's local concentration × gain into the named gene's GRN level (clamped to [0,1]), falling back to the legacy `pigment`+V coupling only when no declaration exists. `OP_DIVIDE` now performs real division via `CellVM._divide()`: halves parent energy, then spawns a daughter `Cell` in `vm.daughters` (inherits proteins/slots/color, name `<parent>-d<n>`, fresh age/divisions; no spawn when energy too low). `OP_CALL_GENE`'s `call_target=` back-patching is verified against a fifth gene (g4) whose u16 offset lies outside the wobble 0..3 range; an unknown target raises `CompileError`.

### G10 — Performance and scale ceilings
- **Current:** VM dispatch ≈ 770 ns/op (~1.3M ops/s); pure-Python Gray-Scott fallback ~0.14–0.18 µs/cell; `trace` accumulates every snapshot (O(ticks) memory); population metabolism vectorizes only via optional numpy.
- **Gap:** For 10⁴–10⁶ cells × per-cell GRN, need vectorized across-cell GRN, grid-based diffusion, snapshot streaming/downsampling, optional C/numba/numba-less parallel hot paths.
- **Blocks:** P10 scale, whole-population evolution studies.
- **Literature:** `doc/13-performance-report.md` §4 items 4–5 (open items); NUFEB parallelization lessons.

### G11 — No validation/calibration pipeline
- **Current:** 1766 tests validate internal consistency and literature anchors, but there is no pipeline to fit model parameters to experimental data or run standardized benchmarks (BM2/BM3 biofilms, Virtual Cell Challenge style).
- **Gap:** Calibration harness + standardized benchmark cases.
- **Blocks:** P9/P10 credibility; the virtual-cell field explicitly demands shared benchmarks (Virtual Cell Challenge 2025).
- **Literature:** Karr 2012 (validation methodology); Virtual Cell Challenge (Cell 2025); IWA biofilm Benchmark Problem 3 (used by iDynoMiCS 2.0 / NUFEB).

---

## 5. Tiered Upgrade Plan

Respects the project constraints: **stdlib-first core** (numpy/scipy/cobrapy/biopython
are optional extras), **physical units on**, **≥90% coverage / ruff / mypy clean**.
Each item lists the modules touched and the frontier problems it unlocks.

### Tier 1 — Foundation (unblocks most frontier work)

**T1.1 Unify the multicellular runtime** (`vm.py`, `population.py`, `cell.py`) — ✅ implemented
- Give `PopulationCell` a `program` (bytecode chunk), a per-cell `GRN`, and a protein/energy budget; give `CellVM`-style dispatch per cell under an ops budget.
- `OP_DIVIDE` actually spawns a daughter (existing `divide_cell()` binary fission); `OP_SIGNAL` writes into the shared AI-2 field (already exists in population path); `OP_EMIT_MORPHOGEN` into the shared Gray-Scott field.
- Share the tick loop: metabolism → field diffusion → per-cell GRN → per-cell dispatch → quorum → division.
- **Unlocks:** P1, P2, P3, P10. **Note:** `06-engineering-design.md` §6.2 is the existing design anchor.
- **Status:** `population.py` — `_build_program_grn` (promoter→nodes, regulations→edges), `_push_gene_frame` / `_execute_cell` (full bytecode dispatch incl. move/signal/feed/divide/die/build/bind/call/jump/stack), `_step_programs` under `config.ops_per_tick`; each cell deep-copies the template GRN so daughter state stays isolated. Verified in `tests/test_population_advances.py` (signal, build, move, divide, per-cell GRN isolation).

**T1.2 Environment fields: nutrients + O₂** (`population.py`, new `environment.py`) — ✅ implemented
- Add diffusing glucose/O₂ fields (same sub-stepped explicit scheme as AI-2, `units.py` conversion).
- Replace flat `ENERGY_INTAKE_PER_STEP` with Monod uptake `vmax·S/(Ks+S)` scaled by energy yield (38 ATP/glucose); add per-field depletion and spatial competition.
- **Unlocks:** P6, P10. **Literature:** Mahadevan 2002; CROMICS 2021.
- **Status:** new `environment.py` — `ConcentrationField` (µm²/s→lattice conversion, flux-conservative sub-stepping, deplete/add/snapshot), `monod_uptake`/`michaelis_menten_rate`, `molecules_per_site` (≈6.02e8 glucose/site at 1 mM), `atp_yield`, `Environment` (+ chemostat `flow_rate`). Population metabolism is Monod-scaled and depletes the field. Verified in `tests/test_environment.py` (mass conservation, 4Dt variance, flow steady state) and `tests/test_population_advances.py`.

**T1.3 Lattice exclusion + shoving** (`population.py`) — ✅ implemented
- One cell per site (or a shoving rule that moves neighbors); track biomass/volume; crowding-dependent diffusion reduction (CROMICS: above ~14% volume fraction).
- **Unlocks:** P10. **Literature:** CROMICS 2021; iDynoMiCS 2.0 2024.
- **Status:** `CELL_SLOT_COUNT = 256`, `mechanics` ∈ {None, "shoving", "force"}; `_apply_mechanics` (shoving = nearest empty neighbor, force = least-crowded neighbor), `get_volume_fractions`, CROMICS `1−φ` crowding factor at `CELL_VOLUME_FRACTION = 1.5e-3` per cell. Verified in `tests/test_population_advances.py`.

**T1.4 Stochastic gene expression** (`grn.py`, new `stochastic.py`) — ✅ implemented
- Add per-cell intrinsic noise: telegraph (two-state promoter) + extrinsic noise; `noise` seed parameter, deterministic default preserved (tests stay stable).
- Optional Gillespie SSA (`stochastic.py`, stdlib or optional-numpy) for small copy-number systems.
- **Unlocks:** P7. **Literature:** Rijal 2025 (differentiable Gillespie); Jones 2014 (two-state); Goetz 2025 (resource-competition noise).
- **Status:** new `stochastic.py` — `telegraph_fano_factor`, `TelegraphPromoter` (k_on/k_off/burst_size/degradation_rate/expression_scale), `fano_to_noise_std`, `gillespie_telegraph` (SSA); `grn.py` gains `GRN(noise_enabled, noise_seed)` + `GeneNode.noise`, zero-mean Fano-scaled noise keeping the deterministic mean unchanged. Verified in `tests/test_stochastic.py` (SSA vs analytic Fano, Poisson limit, seed reproducibility).

**T1.5 Trace streaming / downsampling** (`vm.py`, CLI) — ✅ implemented
- Snapshot every k-th tick or stream to file (documented open item in `13-performance-report.md` §4).
- **Unlocks:** P3 (evolution across many generations needs memory-bounded traces).
- **Status:** `PopulationConfig.trace_streaming` appends per-cell snapshots (id/x/y/alive/energy/proteins/gene levels) each tick, off by default. Verified in `tests/test_population_advances.py`.

### Tier 2 — Frontier applications

**T2.1 dFBA coupling** (`metabolism.py`, `population.py`) — ✅ implemented
- Wrap the FBA solver with dynamic uptake bounds from the local nutrient field (Mahadevan SOA); per-cell biomass → growth rate; acetate/overflow secretion back into the field.
- **Unlocks:** P6, P10. **Literature:** Mahadevan 2002; COSMIC-dFBA 2024; dAMN 2025.
- **Status:** `metabolism.py` — `DynamicFBAConfig` + `DynamicFluxBalance`: per-step MM glucose bound `v_max·S/(Ks+S)`, forward-Euler integration `dX/dt = μ·X`, `dS/dt = −v_glc·X`, byproduct (CO₂/acetate/lactate) tracking, glucose-exhaustion growth arrest, `run`/`reset`/`set_state`, plus `Environment` coupling (`update_from_environment`, `apply_to_environment`). Verified in `tests/test_dFBA.py` (log-linear growth, mass-balance closure, MM bound, environment wiring). Note: the reduced 37-rxn core model has no glyoxylate shunt, so overflow acetate is not re-consumed — diauxie's first phase and arrest are reproduced, documented in the class docstring.

**T2.2 GRN ODE/intra-tick solvers** (`grn.py`) — ✅ implemented
- Optional scipy RK45 integration of the same equations between ticks; keep the discrete recurrence as the default.
- **Unlocks:** P7, validation against COPASI/Tellurium. **Literature:** GRN_modeler 2025.
- **Status:** `grn.py` — `integrate_ode` (scipy `RK45`, optional-dependency fallback to fixed-step Euler), `_dopri5`, `_resample` to the discrete 1-min grid, `ContinuousGRNResult`, and `integrate_grn` (per-node activation reuse of `_activation_raw`). Verified in `tests/test_grn_ode.py` (10 tests): continuous vs discrete convergence, monotone steps, error tolerance, optional-scipy fallback.

**T2.3 Synthetic design automation: Boolean → DNA + SBOL3** (`apps/synbio_automation.py`, `compiler.py`, `central_dogma.py`) — ✅ implemented
- Boolean/logic netlist front end → gate assignment from a characterized part library → DNA via the codon table → simulated dynamics via `central_dogma.py` (Cello-like workflow).
- Add **SBOL3 export** so designs interop with Cello/LOICA/SynBioHub.
- **Unlocks:** P4. **Literature:** Cello 2.0 (Nat Protoc 2022); LOICA (SBOL3); CELLM 2025.
- **Status:** new `apps/synbio_automation.py` — truth-table → minterm/`_binary_reduce` balanced >2-fan-in decomposition → part library (`CharacterizedGate` NOT kd=0.1, others kd=0.25) → DNA codon sequence → SBOL3 text export; `central_dogma.py` compiles the parts to a protein-GRN. Verified in `tests/test_synbio_automation.py` (30 tests: truth tables, gate reduction, SBOL3 export, DNA round-trip).

**T2.4 PLM fitness oracles** (`evolution.py`, optional `ai` extra) — ✅ implemented
- Pluggable fitness: zero-shot ESM-2 pseudo-likelihood or few-shot MLP (LoRA) on top of `protein_structure.py` features; keep heuristic fitness as default.
- **Unlocks:** P8. **Literature:** EVOLVEpro 2025; FSFP 2024.
- **Status:** `protein_fitness.py` — BLOSUM62 matrix, `ESM2Oracle` (optional-dependency `transformers`; zero-vector fallback when absent), `oracle_score`, `rank_variants`; `evolution.calculate_fitness(method="oracle")`. Verified in `tests/test_protein_fitness.py` (16 tests: conservation scoring, ranking, ESM fallback).

**T2.5 Quorum-circuit + patterning example library** (`examples/`) — ✅ implemented
- Consensus detector, population control (You 2004 style), synthetic Turing pattern synthesis, coupled oscillator synchronization (Payne 2013 style) — as `.helix` programs exercising T1.1–T1.5.
- **Unlocks:** demonstrates P1/P2 to users and reviewers.
- **Status:** `examples/20_diauxic_growth.helix` (environment-coupled Monod growth + batch termination), `examples/21_quorum_circuit.helix` (production + sensing population with quorum switch, signal-feedback growth), `examples/22_pattern_synthesis.helix` (build/move shoving self-organization with diffusing fields). All compile and run under the CLI smoke test; API assertions in `tests/test_examples.py`.

**T2.6 Crowding-dependent diffusion** (`population.py`) — ✅ implemented
- Effective diffusion `D_eff = D·f(volume_fraction)` (scaled-particle theory style, CROMICS).
- **Unlocks:** P10 quantitative realism.
- **Status:** CROMICS flux-conservative scheme in `environment.py` (`_cromics_step`, `D·(1−φ)` factor) and `population.py` `_apply_cromics`; effective diffusion falls off with occupied-neighbor fraction while still conserving mass. Verified in `tests/test_environment.py`.

**T2.7 3D population extension** (`population.py`, `morphology_3d.py`) — ✅ implemented
- 3D Laplacian for diffusion; z-axis neighborhoods; reuse LSystem3D for morphology output.
- **Unlocks:** P10, SOTA comparability. **Literature:** NUFEB 2019.
- **Status:** new `morphology_3d.py` — `CellPopulation3D` (z, seeding, `_emit_signal` hook), `ConcentrationField3D` with `_laplacian_step_3d` and 6/26-connectivity `_neighbors_3d`, sub-step cap `_MAX_SUBSTEP_D_3D`; LSystem3D export. Verified in `tests/test_morphology_3d.py` (23 tests: diffusion conservation/decay, 3D connectivity, seeding bounds).

### Tier 3 — Long-term frontier

**T3.1 Spatial-omics-guided models (MiMICS-style)** (new `omics.py`) — ✅ implemented
- Importers for expression matrices; map per-cell expression states → distinct FBA bound sets / GRN parameter sets, so simulated heterogeneity is compared against spatial transcriptomics (Par-seqFISH-scale).
- **Unlocks:** P10/P9 data-calibrated heterogeneity. **Literature:** MiMICS 2024; ASM MMBR 2024 review.
- **Status:** new `omics.py` — `ExpressionMatrix`, `read_expression_matrix`, `expression_to_grn_states`, `build_state_grn`, `expression_to_fba_bounds`, `apply_fba_bounds`, `SpatialAtlas`, `adjusted_rand_index`, `compare_heterogeneity`. Verified in `tests/test_omics.py` (23 tests).

**T3.2 Genome-scale dFBA** (`metabolism.py`, optional cobrapy) — ✅ implemented
- iML1515 loadable already; add dynamic bounds + per-agent proxy (surrogate fluxes to stay fast).
- **Unlocks:** P9 metabolic side. **Literature:** Monk 2017 (iML1515); dAMN 2025.
- **Status:** `metabolism.py` — `DynamicFluxBalance.bound_override` (dynamic growth-linked uptake, hook before step), plus `MetabolicProxy` surrogate (`_poly_features` polynomial features + lstsq fit, nearest-neighbor fallback, non-negative irreversible fluxes, unknown-metabolite `ValueError`). Verified in `tests/test_metabolism.py` (58 regression + 12 new).

**T3.3 Scale to 10⁶+ agents** (all runtime) — ✅ implemented
- Vectorized across-cell GRN step (numpy), grid diffusion, population sorting; snapshot streaming; optional numba/C hot paths; parallel grid decomposition (NUFEB lesson).
- **Unlocks:** P10 scale studies.
- **Status:** new `vectorized.py` — `VectorizedGRN` (numpy across-cell activation, matches scalar GRN), `sort_cells` (stable key sort), `iter_snapshots` (interval/downsampling), `optional_jit` decorator. Verified in `tests/test_vectorized.py` (11 tests).

**T3.4 Virtual-cell integration & validation benchmark** (`virtual_cell.py`, `central_dogma.py`, `grn.py`, `metabolism.py`) — ✅ implemented
- Integrate central dogma + GRN + metabolism into one cell-cycle budget model; add a parameter-estimation harness; publish standardized benchmark cases (biofilm BM3-style; perturbation-response-style).
- **Unlocks:** P9. **Literature:** Karr 2012; Virtual Cell Challenge 2025.
- **Status:** new `virtual_cell.py` — `VirtualCellConfig`/`VirtualCell` (GRN → trigger → transcription/translation ATP draw → FBA biomass → energy budget → maintenance/division/death), `encode_gene` (ECOLI codon-usage table; leading-M start-codon handling), `fit_parameters` (random-search + two-stage refinement: full-box grid scan at doubling resolution, then parabolic-interpolation polish; exact-fit k=2.0, sse=0), `run_biofilm_benchmark`, `perturbation_response`. Verified in `tests/test_virtual_cell.py` (16 tests).

**T3.5 SBML/SBOL interop** (`interop.py`, `apps/synbio_automation.py`) — ✅ implemented
- SBML import for validation against BioModels; SBOL import to consume designs from the ecosystem.
- **Unlocks:** P4/P9 ecosystem credibility.
- **Status:** `interop.py` (stdlib-only, gap G8) — `sbml_to_model`/`load_sbml` (SBML L3V1 core → `MetabolicModel`, BioModels-compatible, solvable without cobrapy) and `sbol3_dumps`/`sbol3_loads` (SBOL3 RDF/XML round-trip for ComponentDefinition/Sequence/roles); `apps/synbio_automation.py` emits SBOL3 via the design workflow. Verified in `tests/test_interop.py` (16 tests) and `tests/test_synbio_automation.py`.

---

## 6. Verification Strategy

Each tier ships with tests + benchmarks; coverage/ruff/mypy gates stay enforced.

| Tier item | Tests | Benchmark/validation target | Status |
|---|---|---|---|
| T1.1 unified runtime | population cells execute bytecode; `OP_DIVIDE` spawns daughters; GRN runs per cell with shared fields | run 10³ programmed cells × 10³ ticks within budget | ✅ `test_population_advances.py` |
| T1.2 nutrient fields | Monod uptake reproduces exponential growth then saturation; diauxie on glucose→acetate | match dFBA reference curve (Mahadevan SOA case) | ✅ `test_environment.py`, `test_dFBA.py` |
| T1.3 exclusion/shoving | no two cells per site after shove; crowding reduces D (CROMICS threshold) | compare density profile vs iDynoMiCS 2.0 BM3-style case | ✅ `test_population_advances.py` |
| T1.4 stochasticity | telegraph model reproduces analytic Fano factor for two-state promoter; deterministic default unchanged (existing 1590 tests still green) | mean/noise match Jones 2014 promoter data | ✅ `test_stochastic.py` |
| T1.5 trace streaming | downsampled trace identical at sampled ticks | memory O(ticks/k) | ✅ `test_population_advances.py` |
| T2.1 dFBA | dynamic biomass/glucose/acetate curves match COBRApy SOA reference | R² ≥ 0.9 on a batch case | ✅ `test_dFBA.py` (log-linear growth, mass-balance closure, MM bound, environment wiring) |
| T2.3 design automation | Boolean spec → DNA → simulated truth table matches target (Cello-style circuits: NOT, NAND, XOR) | compare against Cello 2.0 published circuits | ✅ `test_synbio_automation.py` |
| T2.4 PLM fitness | PLM oracle outranks random/Blosum heuristic on a DMS benchmark subset | Spearman vs ProteinGym where feasible | ✅ `test_protein_fitness.py` |
| T3.4 virtual cell | integrated cell completes division cycle under budget; perturbation response qualitative | Karr-style validation checklist | ✅ `test_virtual_cell.py` |
| T3.5 SBML/SBOL | SBML L3V1 import solves in the core FBA; SBOL3 payload round-trips | validate BioModels export; Cello/SynBioHub-style SBOL3 | ✅ `test_interop.py`, `test_synbio_automation.py` |
| G9 declarative couplings | `#morphogen` wires declared channel→gene (V/U), legacy pigment fallback intact; OP_DIVIDE spawns a daughter cell; call_target reaches 5th gene, unknown target rejected | no external benchmark (design debt) | ✅ `test_g9.py` |

Regression gates: every existing example must keep running; new examples added under
`examples/` must compile + run in the CI smoke test; `pytest --cov=helixlang
--cov-fail-under=80`, `ruff check src tests`, `mypy` stay green.

---

## 7. Implementation Batches (roadmap)

| Batch | Scope | Depends on | Target |
|---|---|---|---|
| B1 | T1.1 unified multicellular runtime | — | multi-cell programmable runtime | ✅ shipped |
| B2 | T1.2 nutrient/O₂ fields + Monod | B1 | environment, competition | ✅ shipped |
| B3 | T1.3 exclusion + shoving + crowding D | B1, B2 | realistic packing | ✅ shipped |
| B4 | T1.4 stochastic gene expression (telegraph + optional Gillespie) | B1 | noise-driven fate studies | ✅ shipped |
| B5 | T1.5 trace streaming; T2.5 example library (quorum circuits, pattern synthesis, consensus) | B1–B4 | proof-of-frontier demos | ✅ shipped (`examples/20–22`) |
| B6 | T2.1 dFBA coupling | B2 | dynamic metabolism | ✅ shipped |
| B7 | T2.2 GRN ODE solvers; T2.3 design automation + SBOL3 | B4 | Cello-like workflow | ✅ shipped |
| B8 | T2.4 PLM fitness oracles; T2.7 3D population | B6, B7 | ML-guided evolution; 3D | ✅ shipped |
| B9 | T3.1 omics import; T3.2 genome-scale dFBA; T3.3 scaling | B6, B8 | data-calibrated, large-scale | ✅ shipped |
| B10 | T3.4 virtual-cell integration + benchmarks; T3.5 SBML/SBOL interop | B7, B9 | whole-cell credibility | ✅ shipped |

Suggested order: **B1 → B2 → B3 → B4 → B5** (a coherent, demoable milestone),
then B6–B7 in parallel, then B8–B10 as stretch goals.
**B1–B10 are complete** (T2.3 in `apps/synbio_automation.py`, T3.1–T3.4 in
`omics.py`/`vectorized.py`/`virtual_cell.py`, T3.5 in `interop.py`).

---

## 8. New Literature References (this analysis)

**Virtual cell / whole-cell**
- Karr et al. (2012) *A whole-cell computational model predicts phenotype from genotype*, Cell 150:389–401. DOI:10.1016/j.cell.2012.05.044
- Bunne et al. (2024) *How to build the virtual cell with AI*, arXiv:2409.11654 (also Cell 187(25):7045–7063).
- Virtual Cell Challenge (Arc Institute) (2025), Cell. DOI:10.1016/j.cell.2025.06.021
- *Grow AI virtual cells: three data pillars and closed-loop learning*, Cell Research 2025. DOI:10.1038/s41422-025-01101-y
- Szigeti et al. (2018) *A blueprint for human whole-cell modeling*, Curr Opin Syst Biol.
- *Data integration strategies for whole-cell modeling* (FEBS Letters/PMC11042497), 2024.

**Multi-agent / biofilm simulators**
- Cockx et al. (2024) *iDynoMiCS 2.0*, PLoS Comput Biol 20(2):e1011303. DOI:10.1371/journal.pcbi.1011303
- Li et al. (2019) *NUFEB*, PLoS Comput Biol 15(12):e1007125. DOI:10.1371/journal.pcbi.1007125
- Li et al. (2024) *Stress-anisotropy-driven nematic ordering in growing biofilms*, Soft Matter 20:3401. DOI:10.1039/D3SM01535A
- *Resolving spatiotemporal dynamics in bacterial multicellular populations*, Microbiol Mol Biol Rev 2024. DOI:10.1128/mmbr.00138-24

**Multi-scale / omics-guided / crowding**
- Walsh et al. (2024) *MiMICS: spatial-transcriptome-guided multi-scale framework*, PLoS Comput Biol 20(4):e1012031. DOI:10.1371/journal.pcbi.1012031
- Bauer et al. (2021) *CROMICS*, PLoS Comput Biol 17(9):e1009158. DOI:10.1371/journal.pcbi.1009158

**dFBA / dynamic metabolism**
- Mahadevan, Edwards, Palsson (2002) *Dynamic flux balance analysis of diauxic growth in E. coli*, Biophys J 83(3):1331–1340.
- Gopalakrishnan et al. (2024) *COSMIC-dFBA*, PLoS Comput Biol.
- Mehrtens et al. / Brsynth (2025) *dAMN: genome-scale neural-mechanistic dFBA* (iML1515), Zenodo DOI:10.5281/zenodo.17908125.
- Negahban et al. (2025) *Hybrid DFBA-PLS*, Bioprocess Biosyst Eng 48:841–856. DOI:10.1007/s00449-025-03147-z

**Stochastic gene expression / GRN**
- Rijal et al. (2025) *A differentiable Gillespie algorithm*, eLife 13:RP103877. DOI:10.7554/eLife.103877
- Jones et al. (2014) *Promoter architecture dictates cell-to-cell variability*, PLoS Comput Biol.
- Goetz, Zhang, Wang, Tian (2025) *Resource-competition-driven bistability and stochastic switching*, PLoS Comput Biol 21(4):e1012931.
- Xue et al. (2024) *A logic-incorporated GRN deciphers principles in cell fate decisions*, eLife 13:e88742.
- GRN_modeler (2025), Mol Syst Biol. DOI:10.1038/s44320-025-00148-8

**Design automation / SBOL**
- Nielsen et al. (2016) *Genetic circuit design automation*, Science 352:aac7341.
- Jones et al. (2022) *Cello 2.0*, Nat Protoc 17:1097–1113. DOI:10.1038/s41596-021-00675-2
- LOICA (2022), ACS Synth Biol (SBOL3).
- CELLM (2025), ACS Synth Biol (LLM + Cello).

**PLM-guided directed evolution**
- Jiang et al. (2025) *EVOLVEpro*, Science 387:eadr6006.
- Tran et al. (2025) *MULTI-evolve*, Science 388:aea1820.
- Li et al. (2024) *FSFP*, Nat Commun 15:5566. DOI:10.1038/s41467-024-49798-6
- *Evaluation of MLDE across combinatorial landscapes*, Cell Systems 2025.

**Genome-scale metabolism**
- Monk et al. (2017) *iML1515*, Nat Biotechnol 35:904–908.

---

## 9. Appendix: Capability Inventory Summary

| Module | Current capability | Readiness |
|---|---|---|
| `compiler.py`/`codon_table.py` | Codon→opcode ISA, wobble operands, 3 genetic-code tables, call-gene back-patching | Mature |
| `vm.py` | Single-cell stack VM + GRN dispatch + Gray-Scott + L-system ops; deterministic | Mature (single-cell) |
| `grn.py` | Discrete-time sigmoid/Hill recurrence, physical decay ≈0.994/tick; optional telegraph-promoter intrinsic noise (`noise_enabled`/`noise_seed`) | Mature; noise optional |
| `population.py` | ≤10⁴ agents, 2D, AI-2 diffusion (sub-stepped, D≈60), quorum 10 µM, lineage+diversity; programmable cells (per-cell GRN + bytecode), Monod environment metabolism, shoving/force mechanics, CROMICS crowding, trace streaming | Mature (programmable) |
| `environment.py` | Glucose/O₂/AI-2 `ConcentrationField` diffusion (flux-conservative sub-steps), Monod/MM uptake, chemostat flow, CROMICS crowding D, molecule↔µM conversion | New (tested) |
| `reaction_diffusion.py` | Gray-Scott (Pearson presets), numpy/pure-Python backends | Mature (2D) |
| `metabolism.py` | Static FBA, 37-rxn E. coli core, pure-Python simplex; dynamic FBA (Mahadevan SOA: MM bounds, Euler integration, byproducts, growth arrest); iJO1366 via cobrapy | Static + dynamic (tested) |
| `stochastic.py` | Two-state (telegraph) promoter Fano factor, telegraph simulator, Gillespie SSA for small copy-number systems | New (tested) |
| `evolution.py` | Wright–Fisher, mutation/selection/drift/recombination, dN/dS | Not wired to VM/population |
| `central_dogma.py` | Transcription/translation kinetics, mRNA half-life, molecule-count scaling | Mature |
| `crispr.py` | Doench-2016-reduced on-target, Hsu-2013 off-target, indel spectrum | Research-grade (not clinical) |
| `protein_structure.py` | Chou-Fasman, GOR IV/DSSP, simplified TMHMM, IUPred2A port | Heuristic |
| `epigenetics.py` | Dam/Dcm/CpG methylation + histone marks (heuristic coefficients) | Heuristic |
| `dna_codec.py` / `apps/dna_storage.py` | Goldman + Erlich fountain, Reed-Solomon, error channels, lifecycle sim | Mature |
| `apps/synbio_designer.py` | Cassette/vector design, CAI back-translation, GenBank/FASTA export | Needs SBOL3 + logic front end |
| `apps/synbio_automation.py` | Truth-table → gate reduction → part library → DNA → SBOL3 export (Cello-like) | New (tested) |
| `interop.py` | SBML L3V1 import → `MetabolicModel` (BioModels-compatible, no cobrapy); SBOL3 RDF/XML round-trip | New (tested) |
| `protein_fitness.py` | BLOSUM62 conservation, ESM-2 oracle (optional transformers), variant ranking | New (tested) |
| `morphology_3d.py` | 3D population + 3D concentration field (6/26-connectivity Laplacian), LSystem3D | New (tested) |
| `omics.py` | Expression matrices → GRN/FBA states, spatial atlas, heterogeneity metrics (ARI) | New (tested) |
| `vectorized.py` | Across-cell numpy GRN, stable sort, snapshot iteration, optional jit | New (tested) |
| `virtual_cell.py` | Cell-cycle budget model (GRN→central dogma→FBA), gene encoding, parameter fitting, perturbation response | New (tested) |
| `units.py` | Physical units always on (min/µM/µm²/s/ATP) | Mature |

---

## Change log

- **2026-08-12** — Initial version. Frontier mapping (§2), SOTA benchmark (§3), gap
  analysis (§4), tiered plan (§5), verification (§6), batches (§7), new references (§8).
  No code changes; proposal only.
- **2026-08-12** — Implementation sync. T1.1–T1.5, T2.1, T2.5, T2.6 marked ✅ implemented
  with `Status:` pointers (modules + tests); verification table gains a Status column;
  B1–B6 marked shipped in §7. New modules `environment.py` and `stochastic.py` added to
  the §9 capability inventory; `grn.py`, `population.py`, `metabolism.py` rows updated.
  Documents the implemented environment-coupled dFBA layer and the CROMICS
  flux-conservative crowding diffusion.
- **2026-08-12** — Frontier sync. T2.2, T2.3, T2.4, T2.7 and T3.1–T3.4 marked ✅ implemented
  (new modules `apps/synbio_automation.py`, `protein_fitness.py`, `morphology_3d.py`,
  `omics.py`, `vectorized.py`, `virtual_cell.py`; `grn.py` ODE solvers; `metabolism.py`
  dynamic bounds + surrogate proxy) with `Status:` test pointers; verification table
  rows T2.3/T2.4/T3.4 flipped to ✅; B7–B9 marked shipped in §7 (B10 partially shipped,
  only T3.5 open). G9 resolved: `#morphogen gene=<name> channel=U|V gain=<float>`
  declarative morphogen→gene wiring (legacy `pigment` fallback preserved), real
  `OP_DIVIDE` division via `CellVM._divide()`/`vm.daughters`, and `OP_CALL_GENE`
  `call_target=` back-patching verified against a 5th gene with unknown-target
  `CompileError`. Verified in `tests/test_g9.py` (16 tests) and the G9 row added to the
  §6 verification table. Full suite: 1766 passed; coverage 89% (gate 80%).
- **2026-08-12** — T3.5 sync. T3.5 marked ✅ implemented (`interop.py`: SBML L3V1
  import → `MetabolicModel`, SBOL3 `sbol3_dumps`/`sbol3_loads`; `tests/test_interop.py`
  16 tests); B10 flipped to ✅ shipped in §7; §6 gains a T3.5 verification row; `interop.py`
  added to the §9 inventory; header status line now reads **B1–B10 implemented**.
