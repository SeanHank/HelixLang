# HelixLang Programmable Cell Population Simulation: the anatomy of a tick and frontier applications

> This document goes deep into `src/helixlang/population.py`, taking "tens of thousands of
> programmable cells growing, dividing, signaling, feeding, competing and being evolved on a
> 2D/3D grid" apart **tick by tick**, then landing it on 2024–2026 frontier biology problems
> and showing "what this simulation can answer and how to use it".
>
> It is the **technical deep-dive** of
> `doc/17-project-details-and-frontier-bio-applications.md`: 17 explains "what it is and
> what it can do", this document explains "**how it is done — what happens in every cell, every
> tick, every instruction**".
>
> Intended reader: someone who has read 17 (or already has a mental model of multi-agent /
> spatial simulation). Every claim is anchored to specific files and line numbers so you can
> verify against the source. Writing date: 2026-08 · Baseline: 2134 tests, ≈89% coverage,
> `ruff` + `mypy` clean (revision 2026.8.2).

---

## Table of Contents

1. [Overview: one tick, five forces](#1-overview-one-tick-five-forces)
2. [Stage 0: metabolism — energy in/out, Monod feeding, dFBA, and vectorization](#2-stage-0-metabolism--energy-inout-monod-feeding-dfba-and-vectorization)
3. [Stage 1: program execution — every cell runs its own DNA bytecode](#3-stage-1-program-execution--every-cell-runs-its-own-dna-bytecode)
4. [Stage 2: signal diffusion — physical diffusion onto a lattice](#4-stage-2-signal-diffusion--physical-diffusion-onto-a-lattice)
5. [Stage 3: quorum sensing, division, and death](#5-stage-3-quorum-sensing-division-and-death)
6. [Stage 4: spatial mechanics — shoving and force repulsion](#6-stage-4-spatial-mechanics--shoving-and-force-repulsion)
7. [Stage 5: environment refresh and statistics output](#7-stage-5-environment-refresh-and-statistics-output)
8. [The third dimension: CellPopulation3D](#8-the-third-dimension-cellpopulation3d)
9. [The evolution engine: Wright–Fisher and the division of labor between populations](#9-the-evolution-engine-wrightfisher-and-the-division-of-labor-between-populations)
13. [Improvement plans (next steps)](#13-improvement-plans-next-steps)

---

## 1. Overview: one tick, five forces

`CellPopulation` (`src/helixlang/population.py:504`) is the center of everything. It holds
three things:

- **The cell list** `self.cells`: each element is a `PopulationCell` (`population.py:218`) —
  energy, coordinates `(x, y, z)`, protein dict, GRN, bytecode VM state
  (`vm_ip / vm_stack / vm_frames`), emitted signal amount, volume `volume_um3`, a lazy dFBA
  batch `dfba`, etc.;
- **The signal field** `self.signal_field`: a `grid_height × grid_width` 2D concentration
  field (AI-2 signal, in µM);
- **The environment object** `self.config.environment`: optional, holding glucose/oxygen
  diffusion fields (`environment.py`).

One tick (`CellPopulation.step`, `population.py:624`) advances five things in a fixed order:

```
         ┌─────────────────────────── CellPopulation.step() ───────────────────────────┐
 tick ──► ① metabolism  energy += intake − maintenance; Monod coupling with the env or per-cell dFBA;
         │                judge death (numpy path auto-chosen for >100 cells)
         ▼
        ② program execution  per cell: GRN step fires genes → push call frame → run ≤ ops_per_tick
         │                    bytecode instructions (OP_MOVE / OP_SIGNAL / OP_DIVIDE / OP_DIE / OP_FEED ...)
         ▼
        ③ signal diffusion   signal field 5-point Laplacian substep diffusion
         │                    (optional CROMICS crowding factor) + per-cell quorum sensing
         │                    (concentration ≥ threshold → set quorum flag)
         ▼
        ④ division / death  energy ≥ division_threshold → binary division (capped by max_size);
         │                   energy ≤ death_threshold → removal; program-controlled division by OP_DIVIDE flag
         ▼
        ⑤ spatial mechanics  shoving (push out to nearest empty site) or force (shift toward sparsest
         │                    neighbor) + environment refresh (diffusion + chemostat flow) + trace streaming snapshots
         ▼
      statistics get_statistics(): alive count / division rate / death rate / Shannon diversity / age distribution
```

Core principle (throughout the whole codebase):

> **Physical units stay online the whole way.** 1 tick = 1 minute
> (`DIFFUSION_DT_S = 60.0`, `units.py:56`), each lattice site = 10 µm
> (`LATTICE_SPACING_UM`, `units.py:42`), signal concentration in µM, energy is ATP molecule
> counts. Every coefficient that "looks like a magic number on the lattice" is obtained by
> `units.diffusion_to_lattice`, which converts physical diffusion coefficients (µm²/s) into
> lattice substep coefficients before integration, so the analytic and numerical solutions
> match.

Each stage below is broken out. Every stage's "seam" has a corresponding `self._step_*`
method in `step`, making it unit-testable.

---

## 2. Stage 0: metabolism — energy in/out, Monod feeding, dFBA, and vectorization

### 2.1 Automatic selection between two code paths

The entry point of `step` (`population.py:644`):

```python
if (_HAS_NUMPY and alive_before > 100 and self.config.environment is None):
    metabolized, deaths = self._step_vectorized_metabolism()   # numpy batch
else:
    metabolized, deaths = self._step_metabolism_python()       # pure Python
```

- **Pure Python** (`_step_metabolism_python`, `population.py:1073`): a per-cell loop, the
  most direct to read;
- **numpy batch** (`_step_vectorized_metabolism`, `population.py:1230`): switched to
  automatically when alive cells > 100 and there is no environment field. Energy/age/
  coordinates/signal amounts are copied into arrays, incremented/decremented wholesale,
  signals scattered into the field with `np.add.at`, and death decided once with a boolean
  mask. The 10⁴–10⁵ cell regime stays near-linear because of this path (see §8).

### 2.2 Pure-Python path: three energy models

Every alive cell does at least three things per tick: `age += 1`, settle energy per the
model, judge death.

**Model A: constant rich medium** (when `config.environment is None`)

```python
cell.energy += config.energy_intake - config.metabolic_cost
```

That is `5.0e7 − 1.0e7` ATP/tick (`ENERGY_INTAKE_PER_STEP` / `METABOLIC_COST_PER_STEP`,
`population.py:82,85`). The division threshold is `1.8e9` ATP and newborn cells start with
`1.0e9` (`POPULATION_CELL_INITIAL_ENERGY`, `population.py:87`), so in rich medium the net is
`+4.0e7`/tick, and **a division is saved up in about 20 ticks**.

**Model B: environment Monod coupling** (when `environment` exists, `population.py:1086-1095`)

```python
local_s = env.glucose.get(cell.x, cell.y)
factor   = monod_uptake(1.0, local_s, config.glucose_half_saturation_mm)  # s/(Ks+s)
intake   = config.energy_intake * factor
demand   = config.max_glucose_uptake_mm * factor
env.glucose.deplete(cell.x, cell.y, demand)
```

- Monod saturation: `factor = s / (Ks + s)`, with `Ks` defaulting to **0.1 mM**
  (`glucose_half_saturation_mm`, following the glucose-growth saturation constant of
  Kovářová-Kovar & Egli 1998);
- **The cell and the field are coupled both ways**: the local sugar concentration scales
  intake, and the cell in turn deducts the `demand` from the field. That is "feeding" —
  tens of thousands of cells competing for sugar on the same grid, where the rich eat fast
  and the poor go hungry.

**Model C: per-cell dynamic FBA** (`dfba_enabled=true` routes to `_step_dfba_metabolism`,
`population.py:1107`)

This is the most hard-core path, and the core of the biofilm-stratification experiment
(example 32):

```python
# Each cell lazily loads its own dFBA batch (deepcopy of the E. coli core model)
dfba = DynamicFluxBalance(model=copy.deepcopy(ECOLI_CORE_MODEL), config=...)
dfba.update_from_environment(env, cell.x, cell.y)   # read local glucose
self._apply_dfba_oxygen_cap(dfba, env, cell.x, cell.y)  # cap respiratory flux by local O2
dfba.step(dt_h)                                     # solve one FBA LP, integrate dt_h hours
consumed = ...; env.glucose.deplete(...)            # deduct sugar
self._deplete_dfba_oxygen(dfba, env, cell.x, cell.y, dt_h)  # deduct oxygen
dfba.apply_to_environment(env, cell.x, cell.y)      # excrete acetate back into the field
cell.energy += dfba.growth_rate * dt_h * config.dfba_energy_scale  # growth rate → ATP
```

Mechanistic points:

- Every cell solves **its own** LP (not one shared solution), and the model is `deepcopy`'d
  so cells' respiratory caps never crosstalk;
- **The oxygen cap** (`_apply_dfba_oxygen_cap`, `population.py:1182`): local O2 yields a
  respiratory cap `v_o2 = v_max·O2/(Ks+O2)` via Monod, which then caps the flux upper
  bounds of `NADH_OX` and `FADH2_OX` at `2·v_o2` (2 reducing equivalents per O2 molecule).
  Once peripheral cells have grabbed all the oxygen, core cells' LP is forced onto the
  **fermentative route** and overflows acetate — that is the mechanism behind colony
  "core/edge metabolic stratification", aligned with Cockx 2024 and COSMIC-dFBA 2024;
- The FBA biomass reaction of dFBA already carries a maintenance term (ATPM), so the
  constant `METABOLIC_COST_PER_STEP` is **not** deducted here, otherwise it would be double
  billing (`population.py:1136-1138`).

### 2.3 Judging death

All three models converge: `energy <= death_threshold` (default `0.0`) → `cell.alive =
False`, counted into `deaths`. Dead cells are skipped in the program and division stages and
are eventually removed from the list.

---

## 3. Stage 1: program execution — every cell runs its own DNA bytecode

This is what makes HelixLang's "programmable cells" different from an ordinary cellular
automaton (like Game of Life): **every cell carries a GRN + bytecode compiled from the same
stretch of DNA, yet they execute independently, each with its own expression state.**

### 3.1 One-time compilation of the template GRN

In `CellPopulation.__init__` (`population.py:535-545`): when `config.program` is given,
`_build_program_grn` (`population.py:575`) compiles the HelixLang program into a single
**template GRN** — promoters/genes become GRN nodes (negative strength = constitutive
expression), and `#regulate` edges become regulatory edges. Each alive cell then
`deepcopy`s the template into its own `cell.grn`. The result:

- all cells share one "genome", but each has an independent gene-concentration trajectory
  (`deepcopy` gives each cell its own expression state);
-   `config.noise_enabled` plugs the GRN's telegraph promoter noise (Fano-factor-matched
  Gaussian, Peccoud & Ycart 1995) into every copy — stochastic gene expression is the
  mechanistic basis of "same genome, different fates" (see `apps/fate_analysis.py` and
  example 28; to keep inter-cell noise uncorrelated, give each cell its own
  `noise_seed`).

### 3.2 GRN firing → call frame → bytecode quota

`_step_programs` (`population.py:772`):

```python
for gene in grn.step():                 # GRN updates concentrations once per tick, returns just-fired genes
    self._push_gene_frame(cell, gene)   # look up gene_offsets, push return address, jump to gene entry
self._execute_cell(cell, self.config.ops_per_tick)   # run ≤ quota instructions
```

- `_push_gene_frame` (`population.py:798`): when the GRN says "this gene fired", push the
  current `vm_ip` onto `vm_frames` and jump `vm_ip` to that gene's entry offset in the
  bytecode — **a gene is a function called by an event**;
- `_execute_cell` (`population.py:811`): a `while quota > 0 and frames:` interpreter loop
  with `quota = ops_per_tick` (default 100). Unfinished frames are **suspended** and resume
  next tick — modeling "translating an mRNA takes time, and a cell does a finite amount of
  work at every moment".

### 3.3 Key instruction table (`population.py:847-1003`)

| Instruction | Effect | Biological analogy |
|---|---|---|
| `OP_BUILD_PROTEIN kind` | `proteins[kind] += 1` | Make a protein |
| `OP_BUILD_MEMBRANE v` | set `membrane_permeability` (0–255) | tune membrane permeability |
| `OP_MOVE d` | step in `DIRECTIONS[d%4]`, deduct `MOVE_ENERGY_COST`(1e7) | chemotaxis / migration |
| `OP_SIGNAL ch` | `signal_emitted += 0.5·(1+ch)`; local signal field `+= min(1.0, 0.5·(1+ch))` | release AI-2 |
| `OP_DIVIDE` | `flag_divide = True` (with `program_controlled_division`) | active division |
| `OP_DIE` | `alive = False` | programmed death |
| `OP_FEED` | `energy += 1e8·permeability/255` | permeability decides feeding |
| `OP_READ_MEM / OP_WRITE_MEM` | read/write 256 `slots` | cell memory / state |
| `OP_REGULATE mode` | rewrite edge weights / add edges / adjust promoter strength (the dynamic version of `#regulate`) | gene regulation |
| `OP_BUILD_PIGMENT` etc. | change color | phenotype markers |

Note the detail of `OP_MOVE`: it only moves when the target site is in bounds, and **the
move itself costs energy** (`MOVE_ENERGY_COST = 1e7`, `cell.py:31`). So "wandering"
cells are eliminated by natural selection — movement has a metabolic cost, and that is the
economics of spatial competition.

### 3.4 Program-controlled death and division

The division test in `step` (`population.py:672-674`) distinguishes two modes:

- default: divide whenever `cell.energy >= division_threshold` (energy-economy driven);
- `program_controlled_division=true`: divide only when `cell.flag_divide` (i.e. an
  `OP_DIVIDE` was executed), still subject to the energy threshold. So "when to divide"
  becomes a **program decision** instead of a hardcoded threshold — a program may `OP_DIVIDE`
  only when it is advantageous for the population.

---

## 4. Stage 2: signal diffusion — physical diffusion onto a lattice

### 4.1 From physical coefficients to lattice coefficients

`_diffuse` (`population.py:707`):

```python
D_lattice = diffusion_to_lattice(config.signal_diffusion, DIFFUSION_DT_S, LATTICE_SPACING_UM)
```

`signal_diffusion` is the **physical** diffusion coefficient (default AI-2 ≈ 100 µm²/s,
`units.py:53`); `diffusion_to_lattice` converts it to a dimensionless lattice coefficient.
It is then split into substeps by `MAX_SUBSTEP_D_LATTICE = 0.25` (`population.py:101`) —
the stability ceiling of the explicit 5-point Laplacian scheme is 1/4, and the substeps
guarantee the numerics **never blow up**, with the analytic Gaussian broadening matching the
physical D.

### 4.2 Five-point Laplacian + Neumann boundaries

`signal_diffusion_step` (`population.py:383`): `∂c/∂t = D∇²c`, five-point template; the
numpy version uses `np.pad(mode="edge")` for zero-flux (Neumann) boundaries — concentrations
at the boundary stay put and signals never "leak out" of the grid.

### 4.3 CROMICS crowding diffusion: signals travel slower where cells are dense

With `crowding_enabled=true`, `_crowded_diffuse` (`population.py:746`) is used:

```python
fracs    = self.get_volume_fractions()          # volume fraction per site φ = 1.5e-3 × cell count
factors  = [[crowding_diffusion_factor(min(0.999, φ)) for φ in row] ...]
# D_eff(x,y) = D_lattice · crowding_diffusion_factor(φ)
```

- `crowding_diffusion_factor` (`environment.py:158`) follows the CROMICS model
  (Angeles-Martinez & Hatzimanikatis 2021): effective diffusion decreases with volume
  fraction — **the more crowded the cells, the harder signals/nutrients diffuse**. This
  "crowding feedback" is a major cause of the gradients inside biofilms;
- The implementation `_crowded_laplacian_step` (`population.py:426`) uses the
  **flux-conserving** Fick form `∂c/∂t = div(D grad c)`: the flux on each edge takes the
  arithmetic mean of the two sites' coefficients, and fluxes cancel head-to-tail, so
  **mass is strictly conserved** (with uniform D it degenerates to the standard five-point
  scheme).

### 4.4 Where the signal comes from

Each cell emits `SIGNAL_EMISSION_PER_STEP = 2.0` per tick by default
(`population.py:90`), added to its own site by `_emit_signal` (`population.py:1220`). The
numpy path scatters it all at once with `np.add.at` (`population.py:1277-1283`), avoiding a
per-site Python loop.

---

## 5. Stage 3: quorum sensing, division, and death

### 5.1 Quorum sensing

Every cell calls `quorum_sensing` (`population.py:489`) after reading the signal
concentration of its own site:

```python
if signal_concentration >= threshold:
    cell.proteins["quorum"] = max(cell.proteins.get("quorum", 0.0), 1.0)
    return True
```

The threshold defaults to **10 µM** (`QUORUM_SIGNAL_THRESHOLD`, `population.py:79`). The
effect of `quorum_sensing` is to set `cell.proteins["quorum"]` to 1.0 — an observable
"quorum triggered" flag/product that both the program bytecode and external inspection code
can read and flip behavior on. This is a You-2004-style **density switch**: a few cells emit
first; once the concentration crosses the threshold with population density, the whole group
flips together. Example 21 is exactly this mechanism — at `population_size=2` the signal is
diffused below threshold (off), at `population_size=81` the concentration crosses 20 µM
(on), and `run_consortium_quorum` judges whether consensus is reached by reading
`cells[0].proteins["quorum"]` (`apps/consortium.py:418-461`).

### 5.2 Division

```python
divisions_allowed = max(0, config.max_size - len(metabolized))   # hard cap (default 10000)
wants_division = cell.flag_divide if program_controls
                 else cell.energy >= config.division_threshold
if wants_division and cell.energy >= config.division_threshold and divisions_done < divisions_allowed:
    a, b = divide_cell(cell, config, self.rng)
```

`divide_cell` does binary division: the two daughter cells each inherit half/randomized
energy and proteins (keeping `divide_cell`'s random protein-partition semantics), one stays
in place and one is offset to a neighboring site; the daughter with `id == -1` gets a global
unique id in `_assign_ids`, keeping `parent_id` — this is the data source for lineage
tracking / Shannon diversity. `max_size` (default 10⁴) is the carrying-capacity ceiling:
beyond it, no more division, simulating the **environment's carrying capacity** and
preventing a site explosion.

### 5.3 Death

The program stage's `OP_DIE` sets `alive=False` directly (counted as `prog_deaths`);
energy-exhaustion death in the metabolism stage is counted as `deaths`; both accumulate into
`self._last_deaths` and the cumulative `self._total_deaths`.

---

## 6. Stage 4: spatial mechanics — shoving and force repulsion

`_apply_mechanics` (`population.py:1007`) handles multi-cell overlap on one site after
division. Two modes:

**shoving** (Lardon 2011, iDynoMiCS style): overcrowded cells are "shoved" to the nearest
empty neighboring site. Simple, at most one cell per site, good for fast startup.

**force** (Cockx 2024, the lattice version of iDynoMiCS 2.0's force balance): for every
cell in an overcrowded site, find a neighbor with the lowest `occupancy` score in the 8
neighborhood and shift there (only move when `score < occ[c.y][c.x]`). This is a discrete
implementation of "density-gradient repulsion" — cells always drift away from the most
crowded direction, so colonies spread into circles rather than piling into a column. With
the default `mechanics=None`, **overlap is not handled** (pure lattice model) — the choice
of model is itself an experimental variable.

`get_volume_fractions` (`population.py:737`) feeds CROMICS crowding diffusion as well:
`φ = CELL_VOLUME_FRACTION × cell count`, starting at 1.5e-3 per site.

---

## 7. Stage 5: environment refresh and statistics output

### 7.1 Environment: diffusion + chemostat flow

`Environment.step` (`environment.py:522`): for each field (glucose / oxygen / custom),
first diffuse one physical substep, then pull every site's concentration toward the bulk
concentration by `flow_rate` — `flow_rate > 0` is a **chemostat/biofilm flow**
(`_replenish`, `environment.py:531`), and `flow_rate = 0` is a closed batch.
`ConcentrationField.diffuse` (`environment.py:255`) performs the same physical → lattice
coefficient conversion and stable substeps, sharing the same numerics as the signal field.

### 7.2 Statistics: `get_statistics` (`population.py:1323`)

Outputs `PopulationStatistics` (`population.py:260`) plus an age distribution:

- `population_size` (cumulative), `alive_count`, `dead_count`;
- `division_rate = last_divisions / step_start_alive`, `death_rate` likewise;
- `diversity_index`: **Shannon diversity** grouped by `parent_id` — the more dispersed the
  lineages, the higher the value;
- `age_distribution`: `{age: count}`, for plotting age pyramids.

Two more observation ports: `get_grid()` (cell count per site) and `get_signal_field()`
(signal-field snapshot), plus per-tick per-cell snapshots when `trace_streaming` is on
(`_append_trace`, `population.py:1052`).

### 7.3 Colony-level observables (dFBA populations only)

`colony_observables` (`population.py:1382`) outputs BM3 / iDynoMiCS-style quantities for
per-cell dFBA colonies (example 32): per-cell doubling time `ln2/μ`, birth-volume
distribution (the age=0 adder distribution), volume-weighted growth rate, **radial density
profile** (`radial_density`, 10 ring bands by distance from the colony center), and colony
radius. This is the quantitative outlet for "core/edge" stratification — overlay the rings
on the oxygen/acetate fields and you get the spatial evidence for "core hypoxia →
fermentative acid production".

---

## 8. The third dimension: CellPopulation3D

`CellPopulation3D` (`population.py:1578`) inherits all the 2D stages, replacing only the
spatial bookkeeping:

- **Signal field** becomes a `[z][y][x]` volume diffused with a 7-point Laplacian
  (`_diffuse`, `population.py:1770`), Neumann boundaries + substeps
  (`_MAX_SUBSTEP_D_3D = 0.15 < 1/6` keeps it stable); under numpy the whole volume runs all
  substeps in the array and is converted back to a list only at the end, avoiding
  per-substep round-trip copies;
- **Mechanics** extends to the 26 neighborhood (`_apply_mechanics_3d`,
  `population.py:1710`): shoving shoves to the nearest empty face-neighbor, force scans all
  26 neighbors for the sparsest site;
- **Division** adds a z-offset (`_divide_3d`, `population.py:1807`): the two daughters split
  ±1 along z;
- **Observation**: `neighbors_3d` exposes the 6/26 connected neighborhoods;
  `to_lsystem3d` (`population.py:1880`) encodes the colony's occupied voxels into
  `LSystem3D` turtle geometry (one `F` marker per occupied site), ready for rendering —
  NUFEB-style 3D spatial output.

**Performance**: numpy batch metabolism/diffusion + `np.add.at` scattering scale
near-linearly to 10⁴–10⁵ cells — the full 10⁵-cell step (including diffusion) is ≈ 0.9
s/tick, matching the individual-cell scale of NUFEB 2019 / iDynoMiCS 2.0; the vectorized and
pure-Python paths are bit-for-bit numerically identical (guaranteed by tests).

---

## 9. The evolution engine: Wright–Fisher and the division of labor between populations

There are **two** "populations" in the project, easy to confuse — the code was renamed on
purpose for this reason:

| Class | Location | Role |
|---|---|---|
| `CellPopulation` (alias `Population`, `population.py:1929`, with deprecation warning) | `population.py:504` | **Spatial** multicellular: grid, energy, programs, signals, mechanics |
| `EvolutionaryPopulation` | `evolution.py:1003` | **Genetic** population: mutation/selection/drift/recombination of DNA sequences |

Their relationship is "two nested layers":

- The outer layer `EvolutionaryPopulation.step` (`evolution.py:1115`) runs the
  Wright–Fisher three steps:
  1. **Mutation**: each individual mutates its DNA at `mutation_rate` (E. coli
     2.2e-10/base/generation), `indel_rate`, transition:transversion ≈ 3:1
     (`mutate`, `evolution.py:170`; >100 individuals automatically use the numpy batch
     `mutate_batch`, `evolution.py:301`);
  2. **Selection + drift**: sample N individuals with replacement proportional to fitness
     (`select`, `evolution.py:366`);
  3. **Recombination**: when `recombination_rate > 0`, random pairing and crossover
     (`recombine`, `evolution.py:438`).
- The inner layer `CellPopulation.evolve` (`population.py:1302`) merely "runs `step()` for
  several generations" — it does not touch DNA variation, only spatial-behavior evolution.

**The bridge between the two layers** is `apps/digital_evolution.py` (example 23): in
`DigitalEvolution.step` (`digital_evolution.py:244`), each digital organism's "genome" is a
string of instructions that is first mutated by a realistic spectrum (`mutate_genome`), then
selected by Wright–Fisher on "this program's score on the target behavior" (`fitness_of`).
That closes the Avida-style loop "genotype (instruction string) → program → behavior →
fitness → next generation", and it uses a real mutation spectrum rather than random
scrambling. Wiring DNA compilation all the way into `CellPopulation` (mutated DNA →
compiler → programmable cells running spatial behavior → behavior defines fitness) is §13
Design 1.

---

## 13. Improvement plans (next steps)

This integrates the full-stack advanced scenario and the original boundary limitations into
one executable plan, following the delivery pattern of
`doc/17-project-details-and-frontier-bio-applications.md` §9 (`apps/` module + tests +
`.helix` example, gates all green). Every plan gives
**goal → how → deliverables → verification**; priority is marked with ★.

### 13.0 Starting point: current limitations (each limitation maps to one plan below)

Honest boundary-drawing, to avoid overpromising:

| Current limitation | Corresponding plan |
|---|---|
| `#sim` has no backend that chains "evolution + spatial population": mutated DNA cannot be compiled into `CellPopulation` to run behavior | Plan 1 |
| `dfba_enabled` solves one LP per cell; 10⁵ scale is the heavy path (example 32 is 2000 cells); large-scale 3D can only use constant metabolism + the vectorized path | Plan 2 |
| The core model lacks the glyoxylate shunt, so acetate cannot be reused as a second carbon source; there is no classical diauxic second phase (`spatial_dfba.py:17-23` already notes this) | Plan 3 |
| `virtual_cell.py::fit_parameters` (inverse-variance weighted) only fits single-cell budgets, not wired to `colony_observables` | Plan 4 |
| Each cell's GRN comes from one `.helix` program (a few dozen genes), not a whole-genome ~4300-gene schedule; transcription/translation happen only on fired genes | Plan 5 |
| Chemostat flow is just a linear pull of concentrations toward bulk (`_replenish`); cells are point particles + occupied sites, with no fluid mechanics, no continuous deformation/contact mechanics | Plan 6 (Level 1→3) |

### Plan 1: Full spatial-evolution closure + a large helix example (★★★)

**Goal**: turn the 10-layer full-stack scenario into one runnable `.helix` showcase —
"DNA programs on a 3D grid are driven by natural selection to evolve a strain that survives
in the oxygen-poor colony core and lives in quorum-sensing symbiosis". This is the biggest
remaining gap of the evolution line, and it is HelixLang's unique differentiating capability
(real codons + physical space + evolvable genotypes).

**How**:

1. **New `apps/spatial_evolution.py`**: a `SpatialEvolution` class running a two-level loop —
   - Outer (genetic layer): reuse `evolution.py`'s real mutation spectrum to mutate program
     DNA (`mutate`: 2.2e-10/base/generation, indels, transition:transversion ≈ 3:1,
     `evolution.py:170`); `SemanticAnalyzer` checks the reading frame (variants with
     premature STOP are eliminated outright), and `Compiler` recompiles;
   - Inner (spatial layer): each generation, compile `variants_per_gen` mutated programs,
     each into a `CellPopulation3D` (`dfba=true` + `crowding=true` + `mechanics=shove` +
     `signaling=true` + `noise_enabled=true`) and run `ticks=T`;
   - **Fitness** = `colony radius × core survival − metabolic cost` (computed from
     `colony_observables`' radial/oxygen/acetate profiles; configurable to any phenotype);
   - **Feedback**: `select()` (`evolution.py:366`) samples with replacement by fitness,
     optionally `recombine()`.
2. **Register `#sim kind=spatial_evolution` in `sim_runtime.py`**: reuse every key of
   `_build_population_config` (`population_size/grid_*/dfba_*/mechanics/crowding/...`,
   `sim_runtime.py:1448`); the outer evolution parameters go through `#sim`
   (`generations/variants_per_gen/mutation_rate/fitness/...`); registration copies the
   `kind=digital_evolution` pattern (`sim_runtime.py:694-732`).
3. **Deliver the big example**: once landed, place it at `examples/35_spatial_evolution.helix`
   and add the `.helixc` (`hxbc.py --compile`); the example list in
   `doc/17-project-details-and-frontier-bio-applications.md` grows by one (see the delivery
   note below for where the slot actually went).
   Design draft — the 10 layers of the full-stack scenario map one-to-one onto this file:

```helix
# Example 35 (design draft): full-spatial evolution — the 10-layer full-stack scenario
# #sim kind=spatial_evolution backend is the deliverable of §13 Plan 1 of this document.
# The backend has since landed as apps/spatial_evolution.py, and the plan shipped as
# examples/40_spatial_evolution.helix (the "35" slot went to
# examples/35_acetate_switch.helix) — see the delivery note below.

#promoter name=p_housekeeping strength=-0.4
#promoter name=p_quorum        strength=0.7

#gene name=adhE promoter=p_housekeeping
ATG GCT GGT GCT TAA
#end

#gene name=reporter promoter=p_quorum
ATG GCT GCT TAA
#end

#regulate p_housekeeping -> adhE strength=+0.8
#regulate p_quorum -> reporter strength=+0.9

#quorum target=reporter threshold=10.0

#media nutrient=GLC concentration=10.0 diffusion_um2_s=300
#media nutrient=O2  concentration=0.25 diffusion_um2_s=1600

#config backend=population
#config population_size=2000 grid_width=48 grid_height=48 grid_depth=12
#config division_threshold=1.8e9 death_threshold=0.0
#config signaling=true signal_diffusion=0.3 signal_threshold=10.0
#config crowding=true mechanics=shove
#config dfba=true dfba_dt_h=0.1
#config dfba_glucose_half_saturation_mm=1.0
#config dfba_oxygen_max_uptake=20.0 dfba_oxygen_half_saturation_mm=0.002
#config noise_enabled=true noise_seed=7
#config trace_streaming=true
#config ticks=60

#sim kind=spatial_evolution generations=30 variants_per_gen=6
#sim mutation_rate=0.01 indel_rate=0.001 transition_transversion_ratio=2.0
#sim selection_coefficient=0.2 recombination_rate=0.0
#sim fitness=colony_radius_x_core_survival_minus_cost
#sim output=generation,mean_fitness,max_fitness,core_acetate_mm,edge_oxygen_mm,core_oxygen_mm
#sim seed=7
```

   The ten layers map onto this file as follows: ① genotype layer = `#sim
   kind=spatial_evolution` + `#sim mutation_rate/...`, ② compilation layer =
   `SemanticAnalyzer`+`Compiler` (outer-loop recompilation), ③ spatial layer =
   `backend=population` + `grid_depth=12` + `mechanics=shove` + `crowding=true`, ④
   environment layer = the `#media GLC/O2` fields, ⑤ metabolism layer = `dfba=true` +
   oxygen-cap keys, ⑥ communication layer = `#quorum` + `signaling=true` (AI-2
   diffusion), ⑦ expression layer = `noise_enabled=true` (`noise_seed=7` shared noise
   source), ⑧ observation layer = `trace_streaming` +
   `output=core_acetate_mm,edge_oxygen_mm,...`, ⑨ fitness =
   `fitness=colony_radius_x_core_survival_minus_cost`, ⑩ feedback = the `generations=30`
   loop + `select()`.

**Verification**: new `tests/test_spatial_evolution.py` —
   1. evolved-group mean fitness rises monotonically across generations and is
      significantly above the drift control with `selection_enabled=false` (same control
      method as `test_digital_evolution.py`);
   2. the selected strain's colony satisfies "core acetate > edge acetate and edge O2 >
      core O2" (`dfba_stratification`);
   3. convergence check: the dominant genotype fixes at specified codon sites.

**Value**: closes the "genotype → compile → spatial behavior → fitness → selection" loop
entirely with real codons and physical units — the moat that sets HelixLang apart from
Tierra/Avida/NetLogo.

> **Delivery status (2026-08)**: delivered as `apps/spatial_evolution.py` (`SpatialEvolution`),
> a dual-loop implementation — the outer loop mutates DNA genotypes with `evolution.py`'s
> real spectrum (2.2e-10/base/generation, indels, transition:transversion ≈ 3:1), checks the
> reading frame and recompiles via `SemanticAnalyzer`+`Compiler`; the inner loop scores each
> genotype as a spatial colonizer on `CellPopulation3D` (`signaling`/`mechanics=shoving` on)
> with the fitness proxy `colony_radius_sites * core_survival - metabolic_cost` (Bosshard et
> al. 2020, BMC Genomics 21:232). `#sim kind=spatial_evolution` is registered in
> `sim_runtime.py` (`_run_spatial_evolution`, keys incl.
> `generations/population_size/genome_length/substitution_rate/indel_rate/.../signaling`),
> and `tests/test_spatial_evolution.py` (6 cases) is green. The large example shipped as
> **`examples/40_spatial_evolution.helix`** (+ `.helixc`): the 10-layer full-stack scenario
> as a runnable file — `#sim kind=spatial_evolution generations=12 population_size=10`
> seeds 80-cell inner colonies on a 32×32 lattice; mean fitness roughly doubles in the
> first generations (fast colonizers fix) then plateaus at the mutation-selection balance
> (`run: helixlang examples/40_spatial_evolution.helix`, ~25 s). The `35` slot went to
> `examples/35_acetate_switch.helix`, so the draft above reads as `40` in the shipped file.

### Plan 2: Vectorized dFBA with shared batches (★★★)

**Goal**: turn `dfba_enabled`'s per-cell LP from a pure-Python loop into batched solving,
pushing toward 10⁴–10⁵ cells.

**How**:

1. `apps/spatial_dfba.py` already demonstrates "vectorize by sharing reactors across a grid
   row" (`spatial_dfba.py:154`); port the same idea into `_step_dfba_metabolism`
   (`population.py:1107`) — cells with identical environment state on the same grid share a
   batch, and only sites with different O2 caps get separately capped;
2. keep `dfba_energy_scale` semantics unchanged; the batch result is bit-for-bit identical
   to the per-cell path within acceptable tolerance (reusing the existing conservation
   assertions);
3. optional switch `#sim dfba_shared=true` toggles between the old and new paths for easy
   comparison.

**Verification**: extend `tests/test_population_dfba.py` with a large-scale case (10⁴ cells,
60 ticks), asserting runtime drops ≥10× while glucose/oxygen conservation and energy
conservation still hold.

### Plan 3: Acetate reuse and diauxic second growth (★★)

**Goal**: add the classical second diauxie phase that the core model lacks without a
glyoxylate shunt — the acetate piled up in the colony core as a second carbon source.

**How**: give `DynamicFluxBalance` (`metabolism.py`) an optional "acetate uptake +
glyoxylate shunt" reaction set (`aceA/aceB` mode), activated by `#media nutrient=AC`;
coordinate with Plan 2's product field.

**Verification**: add to `tests/test_population_dfba.py` the assertion "biomass still rises
(using acetate) after glucose is exhausted", and reproduce the literature's diauxic growth
curve on a 1-D diffusion strip (matching the spatial-diauxie recipe of example 24).

### Plan 4: Population-level mixed-observable calibration (★★)

**Goal**: invert population parameters from real colony observations (radial density /
oxygen / acetate profiles + doubling times).

**How**: extend `virtual_cell.py::fit_parameters`'s inverse-variance weighting to
`colony_observables` (`population.py:1382`) — swap the observation dict for
`{"radial_density", "edge_oxygen_mm", "doubling_times_h", ...}` and the SSE for a weighted
multi-objective.

**Verification**: extend `tests/test_whole_cell_calibration.py`: recover the three
population parameters `dfba_oxygen_max_uptake / dfba_energy_scale / division_threshold`
from synthetic colony data within <10% tolerance.

### Plan 5: Whole-genome GRN — ~4300-gene scheduling per cell (★★★)

**Goal**: scale each cell's GRN from "one `.helix` program (a few dozen genes)" up to E.
coli MG1655 scale (~4300 ORFs, ~10⁴ sparse regulatory edges, RegulonDB order of magnitude),
keeping the runtime at minutes for 10⁴ cells × 60 ticks. Current anchor:
`CellPopulation._build_program_grn` (`population.py:574`) only compiles the `.helix`
program's promoter/gene/regulation declarations into a GRN; `VirtualCell`
(`virtual_cell.py`) and `apps/whole_cell_scale.py` (500-gene sparse GRN + hub + FBA
essentiality screen) already have the whole-genome skeleton, and this plan wires them into
the spatial population.

**Design principle**: layered addition, not replacement — the whole-genome path only turns
on with `#sim genome=...`; the default `.helix` few-dozen-gene path and all existing tests
stay untouched. A whole-genome cell = a background genome GRN (~4300 nodes, mostly silent) + the
engineered `.helix` program (overlapping nodes that read/write the background), both
evaluated in the same sparse matrix.

**How** (four subtasks, each independently deliverable):

1. **`#genome` language wiring (parsing layer)**: add the `#genome source=...` directive;
   its fields merge into `Program.sim_extensions` (the same open extension point as `#sim`,
   `sim_runtime.py:1394`/wiring.md §8.6):
   - `source`: `ecoli-mg1655` (built-in deterministic synthetic genome, b-number named) |
     `synth-4300` | FASTA/GenBank file path;
   - `tf_map`: `regulon` (literature-seeded TF network) | `random` (fixed-seed scale-free) |
     `off`;
   - `grn_mode`: `sparse` (default, CSR matrix) | `full` (dense, tests with small gene
     counts only);
   - `active_gene_budget`: active-gene budget per cell per tick (default 512).
   The `population` mapping in `sim_runtime.py` (`_build_population_config`, line 1448)
   gains a `genome` key, and `_build_program_grn` gains the `_build_genome_grn` branch.

2. **Sparse GRN core (`grn.py` + `vectorized.py`)**: the dense `W ∈ R^{4300×4300}` would be
   148 MB per cell — impossible; it must be sparse. Add `SparseGRN`: ~10⁴ edges stored in
   CSR/CSC (optional scipy; without scipy, fall back to `GRN._incoming`'s target→edges
   index, `grn.py:206`). Evaluation reuses `VectorizedGRN.step`'s matrix semantics
   (`vectorized.py:174`), replacing `@ W.T` with the sparse matmul
   `levels(N,G) @ W.T(sparse) → inputs(N,G)`; without scipy, write a numpy CSR matmul
   (accumulate per-row nonzeros: full population per tick ≈ NNZ × N_cells = 10⁴ edges ×
   10⁴ cells ≈ 10⁸ flops — negligible; under the active budget, only active source columns
   are computed, even less).
   **The active-gene budget is the performance key**: E. coli only significantly
   transcribes ~10% of its genes at any moment; maintain a per-cell active mask (level >
   0.01, or an incoming source is active) and only update the active genes' incoming edges;
   silent genes simply do `level' = decay * level`. Bit-for-bit identical to the scalar
   `GRN.step` (`grn.py:262`) (reusing the existing VectorizedGRN cross-validation test
   pattern).

3. **Row-matrix cell state (the key architectural change)**: today each cell deep-copies the
   template GRN (`population.py:543`); 4300 nodes × 10⁴ cells = 4.3e7 objects, and both
   copying and traversal are infeasible. Change to **shared template, row-matrix state**:
   the background GRN's `W/thresholds/decay` are built once (immutable); per-cell expression
   state is a single block `levels: np.ndarray (N_cells, G)`; one numpy call steps all
   cells, then the fired engineered genes (level > 0.5, `vectorized.py:197`) are connected
   to the `.helix` layer by `triggered`. The engineered layer stays a small GRN with
   per-cell independent expression state as today.

4. **Wire into the spatial population + FBA closure**: `PopulationCell` gains
   `genome_levels` (row index) and `engineered_grn`; `_step_programs` (around
   `population.py:652`) first matrix-steps the background genome, then connects fired
   metabolic genes into dFBA — `whole_cell_scale.py` already has "gene→reaction" gating
   (`ko_model`/`predict_essentiality`, `whole_cell_scale.py:225/243`), and this plan
   generalizes that mapping (`ECOLI_CORE_GENE_REACTIONS`) to "expression-triggered →
   reaction bound open/closed": metabolic genes with expression > 0.5 open their reaction
   upper bounds, otherwise the lower bounds stay. That closes the "whole-genome GRN →
   metabolic phenotype" loop, where a knocked-out gene is a never-fired silent node.
   Transcription/translation costs are charged only for fired genes, reusing `VirtualCell`'s
   per-gene mrna/protein pools (`virtual_cell.py:337`) and `central_dogma.py`'s cost
   functions.

**Deliverables**:
- `apps/genome_scale.py`: `SparseGRN` builder + gene→reaction wiring + `GenomeCell`
  (a 4300-gene lightweight cell: numpy row state + sparse protein dict + FBA reference).
- `#genome` parsing and the `sim_runtime` mapping.
- Large example `examples/37_genome_colony.helix` (design draft below) + `.helixc`.
- `tests/test_genome_scale.py`.

**Example 37 design draft**:
```helix
#genome source=synth-4300 tf_map=regulon grn_mode=sparse active_gene_budget=512
#genome seed=7
#config backend=population
#config population_size=256 grid_width=32 grid_height=32
#config dfba=true dfba_dt_h=0.1
#config signaling=true signal_threshold=10.0
#config crowding=true mechanics=force
#config ticks=12
#media nutrient=GLC concentration=10.0 diffusion_um2_s=300
#media nutrient=O2  concentration=0.25 diffusion_um2_s=1600
#sim output=population_size,alive,triggered_genes,core_acetate_mm,edge_oxygen_mm
```
   (the 4300-gene background comes from `#genome`; engineered `.helix` genes can still be
   layered on with `#gene/#regulate`. At this scale 12 ticks ≈ 9 s; scaling up to the
   original draft's 5000 cells/64×64/60 ticks would drag a single tick to ~5 s because the
   dFBA LP costs ~2 ms/cell/tick, so the example uses the 256-cell tier to demonstrate the
   closure, with the performance gate held by `test_genome_scale.py`'s 10⁴-cell GRN single
   step < 1 s.)

**Verification** (acceptance criteria):
1. **Bit-for-bit identity**: `SparseGRN.step` and scalar `GRN.step` produce identical
   levels on a 10⁴-random-edge network (rng-aligned; the existing
   `test_vectorized.py` pattern).
2. **Essentiality closure**: set the essential genes of
   `ECOLI_CORE_ESSENTIALITY_REFERENCE` (`whole_cell_scale.py:60`) to never fire (silent
   nodes) → FBA biomass → 0; non-essential genes untouched → normal growth.
   Direction B already provides a 20/20 baseline (`apps/whole_cell_scale.py`).
3. **Performance gate**: 4300 genes, ~10⁴ sparse edges, 10⁴ cells, 60 ticks,
   `grn_mode=sparse` + active budget → < 1 s/tick (contrast: the dense path's 148 MB/cell
   infeasibility).
4. **Network structure**: with `tf_map=regulon`, the master regulators (crp/fis/lrp/hns)
   rank top-4 by out-degree; a scale-free degree-distribution power-law fit holds.
5. **Noise fidelity**: `noise_enabled` still reproduces the telegraph Fano factor on 4300
   genes (the per-gene noise vectorization of `grn.py:281` as a single draw).

**Priority**: ★★★ — connecting "real-codon whole-genome" semantics to the spatial population
is the genotype-side upgrade path of Plan 1; the skeleton (`whole_cell_scale` +
`VectorizedGRN` + FBA essentiality) is already in place, with the changes concentrated in
`grn.py`/`apps/`.

### Plan 6: Fluid mechanics and continuous cell morphology (★★★)

**Goal**: add two fidelity tiers while keeping the lightweight default path: (1) **fluid
mechanics** — replace the chemostat flow's pointwise linear pull
(`Environment._replenish`, `environment.py:531`) with transport by a real velocity field;
(2) **cell morphology** — replace "point particles + occupied sites" with rod-shaped cells
(spherocylinders) + contact/fluid-drag mechanics. Target applications: biofilm
microfluidic shear, nutrient boundary layers, colony wall-attached stratification (aligned
with iDynoMiCS 2.0's force-based method and NUFEB's LAMMPS force-based multicellular, see
`doc/10` P10/B10).

**Design principle**: a fidelity ladder; default Level 0 = current behavior (the existing
tests stay untouched), with `#sim flow=...`/`#config cell_shape=...` enabling each tier:

- **Level 0** (today): occupied sites + shoving/force (`population.py:1007`/`1710`) +
  Fickian diffusion + `_replenish`.
- **Level 1**: **prescribed flow** — analytic velocity field u(x,y) for a channel
  Poiseuille profile; the concentration field becomes advection-diffusion; cells drift with
  the field (simple drag).
- **Level 2**: **LBM self-consistent flow** — D2Q9 (2D) / D3Q19 (3D) lattice-Boltzmann,
  with occupied sites as bounce-back obstacles so flow routes around colonies; the existing
  CROMICS crowding factor `1-phi` (`environment.py:158`) is reused directly as the local
  permeability.
- **Level 3**: **rod cells + contact mechanics** — continuous coordinates + rod/rod contact
  (Hertzian elasticity) + Stokes drag; division elongates first, then breaks
  (replacing `divide_cell`'s 8-neighborhood offset, `population.py:318`).

**How**:

1. **Level 1 — analytic flow + advective transport (independent deliverable, lowest risk)**:
   - new `flow.py`: `FlowField` (`channel_poiseuille(direction, mean_velocity_um_s)`,
     `stagnant()`), returning pointwise velocities `u,v`;
   - `ConcentrationField.diffuse` (near `environment.py:225`) gains `advect`: upwind
     differencing + the existing Laplacian, with the CFL-constrained substeps reusing the
     existing `diffusion_to_lattice` stable-substep pattern; `Environment.step`
     (`environment.py:522`) advects first, then diffuses, then `_replenish` (`_replenish`
     stays as the chemostat boundary for Levels 0/1);
   - cell drift: when `config.flow` is set, move cells by `drift = u*dt` before
     `_apply_mechanics` (this level keeps lattice coordinates).

2. **Level 2 — LBM (2D first, 3D later)**:
   - new `apps/lattice_boltzmann.py`: D2Q9 BGK; state `f[9]`/site, occupied sites as
     bounce-back (`_occupancy`'s `[y][x]` mask used directly as the obstacle); Poiseuille
     source/constant-pressure-difference at the inlet, anti-bounce-back at the outlet;
   - field coupling: the local velocity u feeds Level 1's advect step; cell forces =
     the obstacle momentum-exchange rate (bounce-back momentum loss), used as a one-step
     replacement for `force` mechanics;
   - performance: 100×100×9 ≈ 9e4 floats/tick, numpy single-step in milliseconds; 3D
     100³×19 under an optional switch.
   - **The deferred 3D part (Level 2, full D3Q19 design)**: make good on the "2D first, 3D
     later" promise as a deliverable design — with `grid_depth>1`, solve the W×H×D 3D
     channel flow with `LatticeBoltzmann3D` from the new file
     `apps/lattice_boltzmann_3d.py`, occupied volume `[z][y][x]` as bounce-back obstacles,
     and the 3D velocity field driving 3D advection and cell drift; the metabolism/
     program/signal/division skeleton all reuse `CellPopulation3D.step` (`population.py:2331`
     already reserves the `_step_lbm` hook). Specifically:
     - **State and templates**: `f[19][z][y][x]`; the D3Q19 velocity set = rest + 6 axial
       directions (±x,±y,±z) + 12 diagonal directions (4 each in the xy/xz/yz planes), with
       weights `(1/3, 1/18×6, 1/36×12)` and `cs²=1/3`; `OPPOSITE` uses the `e_opp = -e_i`
       index table (9 pairs shared by collision and bounce-back). Capacity: 100×100×50 ≈
       9.5e6 doubles ≈ 76 MB, 100³ ≈ 152 MB — memory-safe.
     - **Streaming (precomputed index tables, no `np.roll`)**: `np.roll` wraps opposite
       edges into each other; 2D already special-cases this by rolling only x and slicing y
       (`lattice_boltzmann.py:247-254`); 3D instead precomputes source/target slice triples
       for each direction and does one whole-block slice assignment per direction,
       `f[i][t_z,t_y,t_x] = f[i][s_z,s_y,s_x]` (`t = s + e_i`); diagonal directions are the
       same single memmove (numpy is safe with overlapping buffers — 2D's
       `f[i][1:] = f[i][:-1]` is the same mechanism), with no wrapping and no per-point
       Python loop; the x axis is ring-connected or solid-walled per `periodic_x`.
     - **Collision and body force**: BGK relaxation applies only to fluid nodes;
       `equilibrium_3d` generalizes the D2Q9 equilibrium as `cu = 3(e_x u + e_y v + e_z w)`
       with kinetic term `u²+v²+w²`; Guo-2002 body force generalized to D3Q19 (cs²=1/3,
       expanded component-wise, applied to fluid nodes), with `body_force=(F,0,0)` driving a
       periodic channel (the 3D version of `_apply_body_force`).
     - **Boundaries (BFL 0.8 by-lattice equilibrium scheme)**: the inlet plane x=0
       overwrites all 19 distributions with the local equilibrium `(rho_in, profile(y,z),
       0, 0)`; the outlet plane x=W-1 likewise with `rho_out`; `profile` supports a 2D
       parabolic inlet profile. The four walls y=0/H-1, z=0/D-1 are solid planes using
       mid-grid bounce-back (no-slip); in a closed box the x columns are solid too. The 2D
       "overwrite entire inlet/outlet columns with equilibrium" is the degenerate case of
       this scheme — both share one equilibrium kernel.
     - **Forces**: the Ladd momentum-exchange method generalized to the 19 directions
       (dimension-by-dimension generalization of the 2D loop at
       `lattice_boltzmann.py:263`), giving each solid node `fx,fy,fz`; the colony's net
       force = downstream drag, and a symmetric obstacle's transverse component should net
       ≈ 0 (acceptance 7d).
     - **Population/field coupling**: the occupied volume `[z][y][x]` is laid out from the
       alive cells' `(x,y,z)` (3D rasterization of Level-3 rods is a later extension); add
       `flow_field3d(substeps) -> FlowField3D` (u,v,w, `[z][y][x]`, sites/tick) and
       `_spread_velocity_3d` (solid nodes take the 26-neighborhood fluid-velocity mean,
       iterated 4 rounds — the 3D version of `_spread_velocity`); `CellPopulation3D._step_lbm_3d`
       mirrors the 2D `_step_lbm` (`population.py:1252`): build the mask → run `lbm_substeps`
       → refresh `config.flow` and `config.environment.set_flow`. Slot reuse:
       `config.lbm`/`config.flow` dispatch 2D/3D by `isinstance` (`_step_lbm` returns
       immediately for non-`LatticeBoltzmann`, `population.py:1266`, so there is no
       conflict); `Environment.set_flow` takes the 3D advection branch when it receives a
       `FlowField3D`.
     - **3D advection**: `ConcentrationField3D` gains `advect_3d(flow3d)` — the conserved
       flux scheme of `_upwind_step` (`environment.py:295`) extended to 3D with z-face
       fluxes; the CFL-substep pattern is unchanged; `Environment.step` runs advect_3d
       before diffuse when the flow is 3D.
     - **Performance gate**: one tick = 19 whole-block slice moves + fluid-node BGK, all
       numpy planar ops, no nested Python loops. 100×100×50 single-core target < 2 s/tick;
       streaming/collision parallelize naturally by z-slice with `lbm_threads=N`
       (`multiprocessing.Pool`, workers holding contiguous z-segments and exchanging one
       halo layer) targeting < 0.5 s/tick; the existing 10⁴-cell 2D Level-2 full loop
       < 1.5 s/tick gate must not regress.
     - **Language wiring** (reusing existing open keys, `_build_pop_lbm` at
       `sim_runtime.py:1649` gains the `lbm_3d` branch):
       ```helix
       #sim grid_depth=50
       #sim lbm_3d=true relaxation_omega=1.2 lbm_substeps=1
       #sim lbm_inlet_density=1.001 lbm_outlet_density=0.999   # or body_force
       ```
       `lbm_3d` is mutually exclusive with `flow`/`lbm` (conflict raises `SimConfigError`,
       reusing the mutual-exclusion check pattern of `sim_runtime.py:1563`) and requires
       `grid_depth>1`; `relaxation_omega`/`lbm_inlet_density`/`lbm_outlet_density`/
       `lbm_substeps` share the same keys as 2D.

3. **Level 3 — rod cells + contact (continuous deformation mechanics)**:
   - new `cell_body.py`: `CellBody` (float center coordinates + axial angle + length +
     diameter); `PopulationCell` gains `body: CellBody | None`, replacing the integer
     lattice coordinates when set;
   - contact: rod/rod nearest-point distance test (r ≥ R_i + R_j), normal elastic repulsion
     (Hertzian, parameter `contact_stiffness`); cell-wall repulsion;
   - drag: Stokes drag (velocity = force / 6πμr, μ optional config);
   - division: `divide_cell` (`population.py:306`) becomes "energy-threshold trigger →
     elongate along the axis → the mother rod breaks at its midpoint into two half-length
     rods with a random ±ε axis tilt", mating the existing adder volume control
     (`virtual_cell.py`'s `volume_um3`/size control);
   - fluid coupling: Level-2 LBM local velocity drives the drag; one-way coupling first
     (flow drags cells, cells do not alter flow), two-way coupling later.

**Language wiring** (all via the open `#sim`/`#config` keys, the `sim_runtime.py:1394`
extension point):
```helix
#sim flow=channel_poiseuille direction=E mean_velocity_um_s=50
#sim lbm=true relaxation_omega=1.5
#sim grid_depth=50
#sim lbm_3d=true relaxation_omega=1.2 lbm_substeps=1   # D3Q19, mutually exclusive with flow/lbm
#config cell_shape=rod length_um=2.0 diameter_um=1.0
#config mechanics=contact contact_stiffness=1.0e3   # or keep shove/force
```

**Deliverables**: `flow.py` (`FlowField` + advection `advect`), `apps/lattice_boltzmann.py`
(LBM-D2Q9), `cell_body.py` (rods + contact + drag), the `sim_runtime` mapping, the large
example `examples/38_flow_biofilm.helix` (design draft: Poiseuille flow + LBM + rod-cell
wall-attached colony) + `.helixc`, `tests/test_flow.py`,
`tests/test_lattice_boltzmann.py`, `tests/test_cell_body.py`; the deferred 3D part:
`apps/lattice_boltzmann_3d.py` (D3Q19), `flow.py` gains `FlowField3D`/`channel_poiseuille_3d`,
`ConcentrationField3D.advect_3d`, `tests/test_lattice_boltzmann_3d.py`,
`benchmarks/bench_lbm3d.py`, examples `examples/d3q19_lbm_pressure_channel.py` +
`examples/d3q19_lbm_bench.py`; end-to-end wiring: `sim_runtime._build_pop_lbm` (`lbm_3d`
mutual-exclusion check), `CellPopulation3D._step_lbm_3d`/`_drift_cells_3d`, example
`examples/39_lbm3d_biofilm.helix`.

**Verification** (acceptance criteria):
1. **Poiseuille analytic solution**: the 2D channel u(y) parabolic peak = 1.5× the mean,
   <1% error vs the analytic solution (`test_lattice_boltzmann`).
2. **Mass conservation**: total LBM density drifts <1e-6 relative over 10³ steps; no
   leakage at bounce-back obstacles.
3. **Obstacle flow reduction**: a channel with one obstacle's flux ≈ unobstructed × the
   cross-section ratio (compare streamline plots).
4. **Nutrient boundary layer**: steady flow + Monod uptake at the colony surface → a
   concentration boundary layer forms at the surface, "edge O2 > core O2" still holds with a
   steeper gradient (cf. the biofilm-stratification recipe, example 32).
5. **Rods do not overlap**: two oppositely growing rods' overlap → 0 under contact forces;
   post-division halves do not penetrate.
6. **Regression gate**: the Level-0 default path is bit-for-bit unchanged and all existing
   tests stay green; performance gate — the 10⁴-cell Level-2 full loop < 1.5 s/tick.
7. **3D gate (the deferred Level-2 part)**: (a) steady u(y,z) in a square duct vs the 3D
   rectangular-duct series solution (Boussinesq, including the effective duct-width
   correction — `_duct_profile`'s `y'=(y+0.5)/H` sampling places the no-slip plane half a
   lattice spacing outside the sites, exactly the effective wall position of full-node
   bounce-back, consistent with the 2D gate-1 handling): pointwise correlation > 0.99, u=0
   at all four walls, interior peak/mean ≈ 2.096 (vs 2D's 1.5 — corner viscosity cuts deeper
   in 3D), relative error < 1% (21×21×21, ~1500 steps to converge); (b) closed box: total
   density drift < 1e-6 over 10³ steps, no leakage through the bounce-back volume obstacle;
   (c) 3D channel with one obstacle: x-flux conserved along the flow (continuity), flow
   detours around the obstacle (u=0 inside), the accelerated gap recovers downstream — the
   same continuity properties as the 2D gate 3; (d) the colony's Ladd 3D net force points
   downstream, a symmetric obstacle's transverse component ≈ 0; (e) performance gate
   100×100×50 < 2 s/tick (single core) / < 0.5 s/tick (`lbm_threads`).

**Priority**: ★★★ — Levels 1→2→3 delivered tier by tier, each tier with its own gate; Level 1
is purely additive and concentrated in `flow.py`+`environment.py` (lowest risk, delivered
first); the deferred 3D part (D3Q19) is ★★, honoring the Level-2 promise and decoupled from
Level 3 — landing it only needs the new solver and the wiring (`CellPopulation3D` +
`ConcentrationField3D` skeletons already exist).

> **Delivery status (2026-08)**: Levels 1→2→3 all delivered, gates all green.
> - Level 1: `flow.py` (`FlowField`/`channel_poiseuille`/`stagnant` + advection `advect`) +
>   `environment.py` advection-diffusion; `tests/test_flow.py` (15) all green.
> - Level 2: `apps/lattice_boltzmann.py` D2Q9 BGK (bounce-back obstacles, Ladd momentum-exchange
>   forces, Guo-2002 body force, constant-density inlet/outlet, `flow_field()` spreading fluid
>   velocity into obstacle sites); `tests/test_lattice_boltzmann.py` (14) all green
>   (Poiseuille peak/mean=1.5, closed-box and obstacle mass conservation, 10⁴-site
>   performance 1.68 ms/tick ≪ the 1.5 s/tick gate).
> - Level 3: `cell_body.py` (spherocylinder volume / nearest-point / rod-rod and rod-wall
>   Hertzian contact + Stokes drag + Jacobi relaxation + mid-point division);
>   `population.py` gains `body`, `_step_lbm`, `_apply_contact_mechanics`, and the `divide_cell`
>   rod branch; `sim_runtime.py` wires `#sim lbm=true`/`flow=`, `#config cell_shape=rod
>   mechanics=contact`. `tests/test_cell_body.py` (22) all green (rods non-overlapping,
>   post-division halves non-penetrating, LBM flow drags rods downstream, invalid combos
>   error out).
> - Large example `examples/38_flow_biofilm.helix` + `.helixc`: LBM microchannel + rod-colony
>   contact mechanics + dFBA nutrient boundary layer (core O2 < edge O2).
> - The deferred 3D part of Level 2 (D3Q19): `apps/lattice_boltzmann_3d.py` (D3Q19, BFL 0.8
>   inlet/outlet, Ladd-force neighbor `np.roll` ring fix — wall rows self-exclude,
>   wall resistance / body force = 1.0000), `flow.py` gains `FlowField3D`/
>   `channel_poiseuille_3d`/`stagnant_3d`, `environment.py` `advect_3d` + the `set_flow`/`step`
>   3D branches, `tests/test_lattice_boltzmann_3d.py` (16) all green (gate 1 peak/mean
>   2.0794→2.096 relative error 0.79%, pointwise correlation 0.9977, gates 2/3 closed-box and
>   obstacle, Ladd drag downstream), `benchmarks/bench_lbm3d.py` (100×100×50 = 205 ms/tick ≪
>   the 2 s/tick gate), examples `examples/d3q19_lbm_pressure_channel.py` +
>   `d3q19_lbm_bench.py`.
> - End-to-end wiring (`#sim lbm_3d=true`): `sim_runtime.py` `_build_pop_lbm` 3D branch +
>   three-way mutual exclusion of `lbm_3d`/`flow`/`lbm` + `grid_depth>1` check, `_seed_cells`
>   3D volumetric seeding, `population.py` `PopulationConfig.flow3d`,
>   `CellPopulation3D._step_lbm` (D3Q19 dispatch) + `_step_lbm_3d` (obstacle mask → advance →
>   publish `config.flow3d`) + `_drift_cells_3d` (x/y/z drift), example
>   `examples/39_lbm3d_biofilm.helix`; `tests/test_sim_runtime.py` (3) +
>   `tests/test_population_3d.py` (3) all green, full regression 2134 passed.
