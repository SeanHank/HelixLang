# Wiring the Simulation Library into the Helix Language

*Design document.* How to make every Python simulation feature reachable from
`.helix` source — turning the language from a bytecode DSL into the front end of
the whole simulation stack, without breaking the existing compiler pipeline.

Status: **implemented (W-1 … W-5 complete, Aug 2026).** The full gate is green:
`mypy src` (0 issues, 56 files), `ruff check src tests` (clean), and
`pytest --cov=helixlang` (1972 passed, coverage 89.2 % ≥ 80 %). `backend=classic`
remains the default and is bit-identical. §18 is the coverage audit of which
Python features are reachable from `.helix` today, §19 is the remaining backlog
and next-step plan.

---

## 0. Implementation status

| Phase | Scope | Status |
|---|---|---|
| W-1 | parser: `Config.sim`, `backend`, `#media`/`#enzyme`/`#metabolite`; golden bit-compat tests | **done** (`ast_nodes.py`, `parser.py`) |
| W-2 | `sim_runtime.py` (`whole_cell`, `fba`), CLI `--backend`/`--json`, examples 31/33 | **done** (`sim_runtime.py`, `cli.py`) |
| W-3 | `population` backend, example 32 | **done** |
| W-4 | `calibration`/`benchmark` backends, `/api/sim/run`, example 34, long-tail `#sim` hook (`spatial_dfba`) | **done** (`server.py`) |
| W-5 | docs: `language-spec.md`, `bio-instructions.md`, README + drift fixes | **done** |

Public surface delivered:

- `#config backend = classic | whole_cell | population | fba | calibration |
  benchmark` (classic default, bit-identical).
- Structural annotations `#media` / `#enzyme` / `#metabolite` (inert with a
  warning under `classic`).
- `#config sim` keys (typed coercion, §6.2–6.3) + `seed=` determinism.
- CLI `--backend`/`--json`; server `POST /api/sim/run`.
- `#sim` long-tail hook; `kind=spatial_dfba` is the first registered backend.

---

## 1. Executive summary

HelixLang contains **two runtimes**:

1. the **classic runtime** (`lexer → parser → compiler → CellVM`): executes
   `.helix` programs as bytecode inside a small GRN + reaction-diffusion cell;
2. the **simulation library** (`VirtualCell`, `CellPopulation3D`, FBA/dFBA,
   the whole-cell apps): a literature-grounded, quantitative simulator.

Before this work the library was Python-API-only (with a few web endpoints).
This document specified a **backend selector** in `#config` plus a small set
of new annotations and config keys, and a single adapter module
(`helixlang/sim_runtime.py`) that maps the parsed program onto whichever
simulator the program asks for — now implemented (W-1…W-5, §0):

```
#config backend = classic | whole_cell | population | fba
                 | calibration | benchmark
```

`classic` stays the default and is **bit-identical** to before. All other
backends reuse the same `#gene` / `#promoter` / `#regulate` declarations and
gain new, typed configuration (§6, §18).

---

## 2. Background: two runtimes today

### 2.1 The classic runtime

`Program` → `Compiler` (codon table → bytecode `Chunk`) → `CellVM`
(`vm.py:471`). The tick loop (`vm.py:556-581`) advances GRN + bytecode
(`ops_per_tick` quota), flushes L-system / reaction-diffusion morphology, and
optionally runs the central-dogma pipeline (`#config use_central_dogma=true`).

This runtime is the documented language (`doc/language-spec.md`,
`doc/bio-instructions.md`). It is intentionally minimal: ~30 opcodes, one GRN
per cell, no physical units, no metabolism.

### 2.2 The simulation library

Separate, Python-first modules:

- `virtual_cell.py` — `VirtualCell` + `VirtualCellConfig`: GRN → central dogma
  → FBA energy budget, cell-cycle phase / Cooper–Helmstetter replication
  (Phase 1), volume + adder size control (Phase 2), protein maturation / QC
  (Phase 3), enzyme-constrained FBA + pools (Phase 4).
- `population.py` — `CellPopulation3D`: programmable per-cell programs, shared
  `Environment`, per-cell dFBA, colony observables, stratification (Phase 5).
- `metabolism.py` — `FluxBalanceAnalysis`, `DynamicFluxBalance`,
  `EnzymeCapacity`, `MetabolitePool`.
- `apps/` — `whole_cell_calibration.py`, `virtual_cell_bench.py`,
  `spatial_dfba.py`, `consortium.py`, `omics_calibration.py`, … (13 apps).
- Supporting: `environment.py`, `central_dogma.py`, `grn.py`, `units.py`,
  `omics.py`, `protein_structure.py`, `morphology_3d.py`, `vectorized.py`.

### 2.3 Why they were separate

The classic runtime is a **teaching tool / DSL runtime** (deterministic
bytecode, morphogenesis). The simulation library is a **quantitative model**
(with physical units, RNG seeds, FBA solvers). They were never wired together —
examples 10/16/20/24/30 described the sim runs only in "via the Python API"
comments. §0 documents how the backend selector now bridges them; §18.3 lists
the remaining Python-only features.

---

## 3. Audit: what `.helix` exposes today

### 3.1 Annotation surface (`parser.py`)

| Kind | Fields | Runtime target |
|---|---|---|
| `#promoter` | `name`, `strength` | GRN (classic) |
| `#gene` | `name`, `promoter`, **any extra `field=value`** stored verbatim | GRN + bytecode |
| `#regulate` | `source -> target`, `strength` | GRN edge |
| `#lsystem` | `name`, `axiom`, `angle`, `step`, `rules` | L-system morphology |
| `#field` | `size`, `F`, `k`, `Du`, `Dv` | Gray–Scott |
| `#morphogen` | `gene`, `channel`, `gain` | GRN↔field feedback (undocumented in spec) |
| `#config` | see §3.2 | run control |
| `#type` | `symbol=Type` pairs | inert unless `enable_type_check` |
| `#crispr` | `target=`, `position`, `new_sequence`, `cas` | `crispr.edit_gene` |
| `#evolve` | `target=`, `mutation_rate`, `indel_rate` | `evolution.mutate` |
| `#methylate` | `target=`, `methylase` | `epigenetics.methylate_dna` |
| `#histone` | `target=`, `mark` | `epigenetics` marks |
| `#transcribe` / `#translate` | `target=` | VM-internal (central-dogma path only) |
| `#quorum` | `target=`, `threshold`, `activate` | reads AI-2 field |

The 7 bio instructions only dispatch when `use_central_dogma=true`
(`vm.py:564`); in the default path they are parsed but inert
(`language-spec.md` §6.5).

### 3.2 `#config` keys (`parser.py:229-246`)

`ticks`, `output`, `table`, `ops_per_tick`, `react_steps`,
`use_central_dogma`, `species`. **All other keys are silently dropped**
(evidence: `examples/26`, `27`, `24` pass `library=`, `payload=`, `length=`,
`inlet=`, `diffusion_um2_s=` with no effect). `output` and `table` are also
effectively dead in the CLI (`--csv`/`--png` flags replace `output`; the
translation table is chosen by `--table`).

### 3.3 CLI surface (`cli.py`)

Positional `source`, `--table`, `--disassemble`, `--debug`, `--csv`, `--png`,
`--ticks`, `--serve`/`--port`/`--host`, `--encode-dna`, `--decode-dna`,
`--pcr-cycles`. The CLI imports only the compiler pipeline + `dna_codec` +
`server`; **it cannot run `VirtualCell`, `CellPopulation3D`, FBA, or any app**.

### 3.4 Reachability matrix (simulation library → language)

> Design-time baseline (pre-W-1). The implemented state is §18 — most "No"
> rows below are now wired via `#config backend` / `#sim`.

| Simulation feature | Triggerable from `.helix`? | How today |
|---|---|---|
| GRN regulation | **Yes** (core) | `#gene/#promoter/#regulate` |
| 2D L-system | **Yes** | `#lsystem` |
| Gray–Scott + morphogen feedback | **Yes** | `#field` + `#morphogen` |
| Central dogma (transcribe/translate) | **Partial** | bio instructions, central-dogma path |
| Quorum signal gate | **Partial** | `#quorum` (field threshold only) |
| CRISPR / per-gene mutation / methylation / histones | **Yes** | bio instructions |
| FBA / enzyme-constrained FBA | **No** | `metabolism.py`, Python only |
| Dynamic FBA / dFBA batches | **No** | `DynamicFluxBalance`, Python only |
| `VirtualCell` (cell cycle, adder, maturation, enzyme caps) | **No** | `virtual_cell.py`, Python only |
| `CellPopulation3D` (colony, per-cell programs, dFBA) | **No** | `population.py`, Python only (driver e.g. `run_consortium_quorum`) |
| Environment / media fields | **No** | `environment.py`, Python only |
| Whole-cell calibration / benchmark | **No** | `apps/whole_cell_calibration.py`, `apps/virtual_cell_bench.py` |
| Spatial dFBA, omics, protein structure/fitness, 3D morphology, vectorized, other apps | **No** | Python API / web endpoints only |

### 3.5 Documentation drift found by the audit

- `#morphogen` is implemented (`parser.py:205`) but absent from
  `language-spec.md` and `bio-instructions.md`.
- `bio-instructions.md` §1 documents `#config grid_width/grid_height` — never
  implemented.
- `bio-instructions.md` §6 defaults (`ticks=1`, `ops_per_tick=100`) differ from
  code (`100`, `64`); the `table` value `mito` is `mito_vertebrate` in code;
  its codon-table quick reference mislabels opcodes.
- `language-spec.md` §3.6 documents `output=stdout|png|csv|none`, but `output`
  is not consumed by the CLI.

> **Resolved in W-5 (§15)**: `#morphogen` is now in `language-spec.md` §3.9;
> `grid_width/grid_height` was removed (replaced by the real population keys);
> `ticks`/`ops_per_tick`/`table=mito_vertebrate` were corrected in
> `bio-instructions.md` §6; the `output=` semantics were redefined as sim
> column selection (§6.7) and are consumed by the CLI.

---

## 4. Design goals and non-goals

**Goals**

1. Every simulation feature in §3.4's "No" rows becomes reachable from
   `.helix` source **or** a first-class CLI/`#config` entry point.
2. `backend=classic` (default) is **bit-identical** to today's behaviour — the
   existing 30 examples and 1,946-test baseline stay green.
3. The language surface stays small and compositional: reuse `#gene` /
   `#promoter` / `#regulate`; add only typed configuration and two structural
   annotations.
4. Determinism: RNG (`#config seed=`) makes every sim backend reproducible.
5. The adapter is one module (`sim_runtime.py`); it never touches the classic
   bytecode pipeline.

**Non-goals**

- Rewriting the classic VM or its bytecode format.
- Making the classic runtime physics-aware.
- Wiring the long-tail Python-only features (omics, protein structure, 3D
  morphology, vectorized, apps) behind bespoke annotations *now* — they get a
  forward-compatible extension point (§8.6) and a documented backlog.

---

## 5. Architecture: the backend selector

```
                .helix source
                      │  Lexer → Parser
                      ▼
                 Program (AST)
                      │  #config backend=?
              ┌───────┼──────────────┬───────────────┐
     classic  ▼            │             │              │
   Compiler→CellVM   sim_runtime.py   (dispatch)        │
   (unchanged)            │             │              │
              ┌───────────┼─────────────┼──────────────┤
              ▼           ▼             ▼              ▼
          whole_cell    population     fba        calibration/
          (VirtualCell) (CellPop3D)  (dFBA)       benchmark
              │           │             │              │
              └───────► history / colony observables / flux tables / scores
                            │
                            ▼
                     CLI: --csv / --json / --png ; Server: /api/sim/*
```

The parser keeps producing a single `Program`. A new function
`sim_runtime.run(program) -> SimResult` decides the backend from
`program.config.backend`, builds the appropriate simulator, runs it, and
returns a uniform result (history records, observables, or score tables) that
the CLI/server render.

---

## 6. Language surface changes

### 6.1 `#config backend = ...`

| Value | Simulator | Meaning |
|---|---|---|
| `classic` (default) | `CellVM` | today's bytecode runtime, unchanged |
| `whole_cell` | `VirtualCell` | Phases 1–4: cell cycle, adder, maturation, enzyme caps |
| `population` | `CellPopulation3D` | Phase 5: per-cell program + environment + dFBA colony |
| `fba` | `FluxBalanceAnalysis` / `DynamicFluxBalance` | standalone metabolism batch |
| `calibration` | `apps/whole_cell_calibration.py` | recover hidden parameters from mixed observables |
| `benchmark` | `apps/virtual_cell_bench.py` | run the 4-gate whole-cell benchmark |

`backend` may also be overridden from the CLI with `--backend` (overrides the
source's choice; useful for CI).

### 6.2 `#config` simulation parameters

Every key below is collected into `Program.config.sim` (a typed dict; §7.1)
and applied by the backend adapter. **Names mirror the target dataclass fields
so the mapping is self-documenting**; a few friendlier aliases are noted.

#### Whole-cell domain (`VirtualCellConfig`, `virtual_cell.py:173`)

| `#config` key | Type | Target field | Notes |
|---|---|---|---|
| `division_rule` | `energy\|adder` | `division_rule` | |
| `division_energy` | float | `division_energy` | ATP threshold |
| `adder_volume_um3` | float | `adder_volume_um3` | adder Δ (default 1.6) |
| `adder_noise_std` | float | `adder_noise_std` | relative Gaussian noise (σ≈0.1–0.2) |
| `volume_init_um3` | float | `volume_init_um3` | newborn volume |
| `biomass_to_volume_pg_per_min` | float | `biomass_to_volume_pg_per_min` | volume model coupling |
| `cell_density_dry_pg_um3` | float | `cell_density_dry_pg_um3` | ρ (0.15) |
| `surface_scaling` | bool | `surface_scaling` | S/V uptake scaling |
| `surface_exponent` | float | `surface_exponent` | default 2/3 |
| `replication_mode` | `flat\|cooper_helmstetter` | `replication_mode` | Phase 1 |
| `c_period_min` / `d_period_min` / `doubling_time_min` | float | same names | Cooper–Helmstetter timing |
| `chromosome_map` | `gene=coord,...` | `chromosome_map` | coord ∈ [0,1] |
| `energy_init` | float | `energy_init` | ATP budget |
| `maintenance_atp_per_min` | float | `maintenance_atp_per_min` | |
| `biomass_to_atp` | float | `biomass_to_atp` | energy per flux |
| `transcription_atp_per_nt` / `translation_atp_per_aa` | float | same | expression cost |
| `protein_yield_per_mrna` | float | `protein_yield_per_mrna` | |
| `minutes_per_step` | float | `minutes_per_step` | default 1.0 min |
| `enzyme_capacity` | bool | `enzyme_capacity_enabled` | Phase 4 MOMENT caps |
| `enzyme_scale` | float | `enzyme_scale` | kcat rescale |
| `protein_mass_fraction` | float | `protein_mass_fraction` | sMOMENT pool row |
| `metabolite_pools` | bool | `metabolite_pools_enabled` | pool ODE integration |
| `protein_maturation_mode` | `instant\|chaperone` | `protein_maturation_mode` | Phase 3 |
| `frac_cotranslational_fold` | float | `frac_cotranslational_fold` | |
| `folding_atp_per_protein` | float | `folding_atp_per_protein` | |
| `k_fold` | float | `fold_rate_per_min` | derived via `_fold_rate_from_k_fold(k_fold, misfold_rate_per_min)` |
| `misfold_rate_per_min` / `aggregation_rate_per_min` / `degraded_rate_per_min` / `protein_half_life_min` | float | same | QC rates |

#### Population domain (`PopulationConfig`, `population.py:186`)

| `#config` key | Type | Target field | Notes |
|---|---|---|---|
| `population_size` | int | `max_size` | |
| `grid_width` / `grid_height` / `grid_depth` | int | same | lattice |
| `division_threshold` / `death_threshold` | float | same | energy gates |
| `signaling` | bool | `signaling_enabled` | AI-2 quorum field |
| `signal_diffusion` / `signal_threshold` | float | same | µm²/s, µM |
| `crowding` | bool | `crowding_enabled` | CROMICS factor |
| `mechanics` | `none\|shove\|...` | `mechanics` | spatial mechanics |
| `noise_enabled` / `noise_seed` | bool/int | same | GRN noise |
| `trace_streaming` | bool | `trace_streaming` | per-cell snapshots |
| `dfba` | bool | `dfba_enabled` | per-cell dFBA |
| `dfba_dt_h` / `dfba_energy_scale` / `dfba_initial_biomass_gdw` | float | same | |
| `dfba_glucose_half_saturation_mm` / `dfba_oxygen_max_uptake` / `dfba_oxygen_half_saturation_mm` | float | same | |

#### FBA domain (`backend=fba`)

| `#config` key | Type | Meaning |
|---|---|---|
| `fba_model` | `core\|<path>` | `ECOLI_CORE_MODEL` or an SBML/JSON model path |
| `dynfba` | bool | use `DynamicFluxBalance` batch instead of static solve |
| `fba_dt_h` | float | dFBA integration step (hours) |
| `fba_glucose_mm` | float | batch glucose (mM) |
| `fba_oxygen_max` | float | respiratory O₂ cap |
| `fba_steps` | int | dFBA iterations |

#### Common

| `#config` key | Type | Meaning |
|---|---|---|
| `seed` | int\|none | RNG seed (adder noise, GRN noise, population noise, calibration) |
| `ticks` | int | already exists; means *minutes* for `whole_cell`, *steps* for `fba`, ticks for `classic`/`population` |
| `output` | comma list | column selection for `--csv` (§6.7) |

### 6.3 Type coercion rules

`Config.sim` stores raw strings; the adapter coerces per the declared type
(§6.2). Rules:

- `bool`: `true|1|yes` / `false|0|no` (case-insensitive) — same convention as
  `use_central_dogma` today.
- `int` / `float`: `int(x)` / `float(x)`; overflow/malformed values raise
  `SimConfigError` naming the key.
- `dict`: comma-separated `key=value` pairs (e.g. `chromosome_map=gltA=0.1,zwf=0.7`).
- enums (`division_rule`, `replication_mode`, `protein_maturation_mode`,
  `backend`, `mechanics`): exact lowercase match, otherwise `SimConfigError`.
- `none`: the literal `none`.

### 6.4 `#media` annotation

Declares the growth medium. Repeatable.

```
#media nutrient=GLC  concentration=10.0
#media nutrient=O2   concentration=0.25  diffusion_um2_s=1600
```

- `nutrient` (required): metabolite id (e.g. `GLC`, `O2`, `AC`).
- `concentration` (required): medium concentration. For `whole_cell`/`fba`
  this sets the FBA uptake bound (`VirtualCellConfig.uptake` /
  `DynamicFBAConfig`); for `population` it initialises the shared
  `Environment` field.
- `diffusion_um2_s` (optional): Fick diffusion of the field (population only).
- Units documented in `doc/bio-instructions.md` (mM for concentrations; the
  FBA exchange bound is the numeric value, matching today's `uptake={"GLC":10}`).

### 6.5 `#enzyme` annotation

Binds a gene to an FBA reaction for enzyme-constrained metabolism. Repeatable.

```
#enzyme gene=gltA reaction=CS kcat=2800
```

- `gene` (required): must match a `#gene` name.
- `reaction` (required): reaction id in the model.
- `kcat` (optional): overrides `ECOLI_CORE_KCAT` for that reaction.
- When `#config enzyme_capacity=true` and no `#enzyme` is given, the adapter
  falls back to the default `ECOLI_CORE_GENE_REACTIONS` / `ECOLI_CORE_KCAT`
  tables (`metabolism.py`).

### 6.6 `#metabolite` annotation

Initialises intracellular pools (requires `#config metabolite_pools=true`).

```
#metabolite name=glc__D init=0.5
```

### 6.7 `output=` column selection

In sim backends, `#config output=` selects CSV columns instead of the dead
legacy list:

```
#config backend=whole_cell output=energy,volume_um3,biomass_flux,divisions,phase
```

Available columns (whole-cell history keys, `virtual_cell.py:687-723`):
`age, energy, alive, divisions, mass, volume_um3, volume_birth_um3,
added_volume_um3, biomass_flux, dna_copy_number, phase, proteins,
proteins_unfolded, proteins_misfolded, proteins_degraded, proteins_aggregated,
metabolite_pools, overflow_secretion, folding_atp_cost`.

Population columns come from `colony_observables` (`population.py:1382`);
`fba` reports the flux table (`FluxBalanceAnalysis.solution`);
`calibration`/`benchmark` report their score dicts.

### 6.8 Seeds and determinism

`#config seed=N` is threaded to `VirtualCellConfig.seed`, GRN/population noise
seeds, and the calibration `fit_seed`. Same source + same seed ⇒ identical
output (verified by tests, §13).

---

## 7. AST and parser changes

### 7.1 `Config.sim`

Extend `Config` (`ast_nodes.py:75`) with two fields:

- `backend: str = "classic"`
- `sim: dict[str, str] = field(default_factory=dict)` — every `#config` key
  *not* consumed by the classic pipeline, preserved verbatim.

`_parse_config` (`parser.py:229`) changes: keep the 7 existing reads exactly;
then store the remaining fields into `prog.config.sim`. The classic pipeline
never reads `sim`, so classic behaviour is untouched. Unknown keys that used to
be silently dropped now survive to the sim backends (and only warn in
`classic`).

### 7.2 New annotation handlers

Add to the parser dispatch table (`parser.py:60-69`) and to
`BIO_INSTRUCTION_KINDS`-style registries:

- `_parse_media` → `Program.media: list[MediaDecl]` (`MediaDecl(nutrient,
  concentration, diffusion_um2_s)`).
- `_parse_enzyme` → `Program.enzymes: list[EnzymeDecl]`.
- `_parse_metabolite` → `Program.pools: list[PoolDecl]`.

These are **structural declarations** (like `#field`), not bio instructions:
they are consumed by the backend adapter, never by the classic VM. In
`classic` mode they are ignored (with a warning), preserving compatibility.

The existing `#gene` extra-field passthrough (`Gene.fields`, `parser.py:144`)
already gives us per-gene hooks with **no parser work**:

- `chromosome=0.3` → `chromosome_map` entry (Phase 1);
- `threshold=0.5` → `GRN.add_gene(name, threshold)`;
- `initial_level=1.0` → initial GRN level (mirrors the calibration's wiring).

### 7.3 Backward-compatibility guarantees

1. `backend` defaults to `classic`; absent `backend` → today's pipeline.
2. The 7 classic `#config` keys parse exactly as before.
3. `#media` / `#enzyme` / `#metabolite` are inert (warned) under `classic`.
4. All existing examples and the 1,946-test baseline must stay green — the
   parser change is additive only (a new `sim` dict, never reordered tokens).

---

## 8. The sim runtime adapter (`sim_runtime.py`)

New module `src/helixlang/sim_runtime.py`. Public API:

```python
run(program: Program) -> SimResult            # dispatch on config.backend
SimResult = HistoryResult | FluxResult | ColonyResult | ScoreResult
```

### 8.1 Dispatch

```python
BACKENDS = {"classic": None, "whole_cell": ..., "population": ...,
            "fba": ..., "calibration": ..., "benchmark": ...}
```

`classic` returns `None` (the CLI keeps the existing compile→VM path).

### 8.2 `whole_cell` mapping (`annotation → VirtualCell`)

1. **genome**: `{g.name: "".join(g.codons) for g in program.genes}` — the
   `.helix` codon stream is literal DNA, exactly what `VirtualCell` consumes.
2. **GRN**: `GRN(noise_enabled=...)`; for each `#gene` → `add_gene(name,
   threshold)` (default threshold 0.5); for each `#regulate` →
   `add_edge(source, target, strength)`; promoter strengths applied via
   `#promoter`; initial levels from `initial_level=` / default 0.0. This
   mirrors the wiring used by `whole_cell_calibration.py`.
3. **config**: `VirtualCellConfig` built from §6.2 keys + `#media` → `uptake`;
   `k_fold` → `fold_rate_per_min` via `_fold_rate_from_k_fold`; `#enzyme` →
   `EnzymeCapacity` overrides; `#metabolite` → pool inits; `seed` threaded.
4. **run**: `cell.run(config.ticks)` (minutes); history → `HistoryResult` with
   the `output=` columns.
5. Invalid combos (e.g. `division_rule=adder` with `surface_scaling` off is
   fine; `replication_mode=cooper_helmstetter` without a `chromosome_map` is
   allowed — absent genes default to origin-proximal per
   `virtual_cell.py:194-196`) → `SimConfigError` with the offending key.

### 8.3 `fba` mapping

1. Model: `fba_model=core` → `ECOLI_CORE_MODEL`; a path → `load_model`.
2. Static: `FluxBalanceAnalysis.solve()` with `#media` uptake bounds.
3. Dynamic: `DynamicFluxBalance` batch from `fba_*` keys; returns
   `FluxResult` (flux vector / batch ODE trace — the diauxie curve that
   `examples/20` and `24` describe but cannot run today).
4. `#enzyme` kcat overrides apply when `enzyme_capacity` caps are requested.

### 8.4 `population` mapping

1. Build `Program` + `chunk` as today (the existing `PopulationConfig.program /
   chunk` wiring used by `apps/consortium.py`).
2. `Environment` from `#media` fields (concentration + diffusion).
3. `PopulationConfig` from §6.2 keys; `dfba` key ⇒ per-cell dFBA on the shared
   fields (O₂ cap drives acetate overflow stratification).
4. Run; return `colony_observables()` + optional `trace_streaming` records.

### 8.5 `calibration` / `benchmark` mapping

- `backend=calibration`: build the `WholeCellCalibration` genome from
  `#gene` declarations (falling back to `DEFAULT_GENOME` when the program
  declares none), honour `seed`/`n_samples`-style `#config sim` keys, run
  `run_whole_cell_calibration`, return the fitted parameters + `passed`.
- `backend=benchmark`: run `run_whole_cell_benchmark()` and return the four
  gate scores.

Both require no genes — a pure-config `.helix` program is enough:
`#config backend=benchmark`.

### 8.6 Extension point for the long tail

A `Program.sim_extensions: dict[str, dict[str, str]]` collected from an
open `#sim key=value` annotation reserves a generic, forward-compatible hook.
Phase W-4 of the roadmap uses it for `#sim kind=omics_calibration`-style
backends; it is inert until a backend registers it. This keeps the surface
closed today and open tomorrow without a parser redesign.

---

## 9. CLI and server exposure

**CLI** (`cli.py`):

- `--backend <name>`: overrides `#config backend`.
- `--csv`: works for every backend (sim backends emit the `output=` columns;
  `calibration`/`benchmark` emit their score dicts as CSV).
- `--json`: new; machine-readable `SimResult` for sim backends.
- `--png` stays classic-only (morphology).

**Server** (`server.py`): one endpoint

`POST /api/sim/run` — parses source, honours `#config backend`, returns the
`SimResult` payload (history / flux / observables / scores) as JSON. Keeps the
existing `/api/*` endpoints untouched.

---

## 10. Examples

Two kinds of `.helix` source illustrate the wiring:

- **§10.1 — rewritten examples**: existing examples (10, 16, 20, 21, 24, 30)
  that today only *describe* a Python-API run in a comment. After the wiring,
  the comment block is deleted and the run happens in-language.
- **§10.2 — new examples**: examples 31–34, which exercise the new surface
  end-to-end.

All examples run as `helixlang examples/NN_*.helix --csv` (or with
`--backend` overriding the source's `#config backend`). The 24 classic
examples are untouched.

### 10.1 Rewritten examples (before → after)

| Example | Today | Rewritten to |
|---|---|---|
| `10_metabolism_fba.helix` | FBA shown as a Python comment | `backend=fba` + `#enzyme` + `#media` |
| `16_population_dynamics.helix` | population run shown as a Python comment | `backend=population` + `#media` |
| `20_diauxic_growth.helix` | dFBA batch shown as a Python comment | `backend=fba` + `dynfba=true` |
| `21_quorum_circuit.helix` | quorum colony driven from Python | `backend=population`, `signaling=true` |
| `24_spatial_diauxie.helix` | spatial dFBA shown as a Python comment | `#sim kind=spatial_dfba` (long-tail hook, §8.6) |
| `30_virtual_cell.helix` | calibrate→predict shown as a Python comment | `backend=benchmark` |

Each "after" is the complete file; the deleted Python comment block is shown
as the "before".

#### 10.1.1 `10_metabolism_fba.helix` → `backend=fba`

**Before**: lines 35–50 of the file are a dead Python block
(`FluxBalanceAnalysis(...).solve()` + prints). The `#gene`s are declared but
only the classic VM reads them.

**After** — the same glycolysis genes, now bound to model reactions; the
uptake bound and the solve are config:

```
# Example 10: Metabolism - flux balance analysis (FBA)
# The glycolysis enzymes are declared once and bound to model reactions with
# #enzyme; backend=fba turns them into enzyme-constrained FBA instead of
# bytecode. #media sets the glucose uptake bound. The Python solve() block
# that used to live here is gone - the run IS the program.

#promoter name=p_glc strength=-0.3

#gene name=glk promoter=p_glc
ATG GCT GCT GCT GCT TAA
#end

#gene name=pgi promoter=p_glc
ATG GGT GGT GGT GGT TAA
#end

#gene name=pyk promoter=p_glc
ATG GCT GGT GCT GGT TAA
#end

#regulate p_glc -> glk strength=+0.8
#regulate p_glc -> pgi strength=+0.7
#regulate p_glc -> pyk strength=+0.6

#enzyme gene=glk reaction=HEX1
#enzyme gene=pgi reaction=PGI
#enzyme gene=pyk reaction=PYK

#media nutrient=GLC concentration=10.0

#config backend=fba
#config fba_model=core
#config output=BIOMASS,EX_glc,EX_ac,EX_lac,growth_rate_per_hour
```

#### 10.1.2 `16_population_dynamics.helix` → `backend=population`

**Before**: the `Population` + `PopulationConfig(max_size=1000,
division_threshold=200.0, death_threshold=0.0, signal_diffusion=0.4,
signal_threshold=5.0)` block is a comment.

**After** — the population config moves into `#config` (names mirror
`PopulationConfig`, §6.2); the `#quorum` bio-instruction stays as the
per-cell gate:

```
# Example 16: Population dynamics - multicellular population simulation
# Producer cells secrete the AI signal; the responder's reporter turns on
# once the shared field crosses the quorum threshold. Everything below runs
# in-language on backend=population; the Python evolve() loop is gone.

#promoter name=p_auto strength=-0.4
#promoter name=p_quorum strength=0.7

#gene name=ai_synth promoter=p_auto
ATG GCT GCT GCT GCT TAA
#end

#gene name=reporter promoter=p_quorum
ATG GGT GGT GGT GGT TAA
#end

#regulate p_auto -> ai_synth strength=+0.8
#regulate p_quorum -> reporter strength=+0.9

#quorum target=reporter threshold=5.0

#media nutrient=GLC concentration=10.0

#config backend=population
#config population_size=1000 grid_width=32 grid_height=32
#config division_threshold=200.0 death_threshold=0.0
#config signaling=true signal_diffusion=0.4 signal_threshold=5.0
#config ticks=30
#config output=alive_count,avg_energy,diversity_index
```

#### 10.1.3 `20_diauxic_growth.helix` → `backend=fba` + `dynfba=true`

**Before**: the two-phase switch and the `DynamicFluxBalance` batch
(`dt_h=0.25, initial_biomass_gdw=0.05, initial_glucose_mm=10.0,
max_glucose_uptake=10.0`, 8 h) are comments.

**After** — the lac-operon GRN stays as the intracellular declaration; the
batch that Monod–Jacob–Monod described now runs from config:

```
# Example 20: Diauxic growth - the lac operon and catabolite repression
# The lacI/crp/lacZ regulatory layers are declared below; the two-phase
# expression switch and the Mahadevan et al. 2002 dFBA batch (which the
# old file could only describe in a comment) now run on backend=fba.

#promoter name=p_lac strength=0.6
#promoter name=p_lacI strength=-0.5
#promoter name=p_crp strength=-0.5

#gene name=lacI promoter=p_lacI
ATG GCT GCT TAA
#end

#gene name=crp promoter=p_crp
ATG GCT GCT TAA
#end

#gene name=lacZ promoter=p_lac
ATG GCT GCT GCT TAA
#end

#regulate p_lacI -> lacI strength=+0.8
#regulate p_crp -> crp strength=+0.8
#regulate lacI -> p_lac strength=-0.9
#regulate crp -> p_lac strength=+0.9

#media nutrient=GLC concentration=10.0

#config backend=fba
#config fba_model=core dynfba=true
#config fba_dt_h=0.25 fba_oxygen_max=20.0 fba_steps=32
#config output=time_h,biomass,glucose,co2,growth_rate
```

#### 10.1.4 `21_quorum_circuit.helix` → `backend=population`, `signaling=true`

**Before**: the density sweep (`sparse=build(2)` vs `dense=build(81)`) and the
two 30-step runs are a Python block with its own `PopulationConfig`.

**After** — one program per density: run once with `population_size=2` (OFF)
and once with `population_size=81` (ON); the `#gene` signal opcode (`TCA`)
still releases AI-2 into the shared field, and `#config signaling=true` is the
switch that used to be `PopulationConfig(signaling_enabled=...)`:

```
# Example 21: Quorum circuit - programmable cell-cell communication
# Every cell runs the signal gene (TCA opcode releases AI-2); past the
# density-dependent threshold the reporter fires. Run twice - with
# population_size=2 the field stays below threshold (OFF), with 81 it
# crosses it (ON). The old Python build()/step() block is now config.

#promoter name=p_signal strength=-0.4
#promoter name=p_reporter strength=0.7

#gene name=signal promoter=p_signal
ATG TCA TAA
#end

#gene name=reporter promoter=p_reporter
ATG GCT GCT TAA
#end

#regulate p_signal -> signal strength=+0.8
#regulate p_reporter -> reporter strength=+0.9

#config backend=population
#config population_size=81 grid_width=24 grid_height=24
#config signaling=true signal_diffusion=0.3 signal_threshold=20.0
#config division_threshold=1e9
#config trace_streaming=true
#config ticks=30
#config output=alive_count,avg_energy,diversity_index
```

The quorum switch itself is per-cell (`proteins["quorum"]`); with
`trace_streaming=true` the per-cell traces are part of the `SimResult`, so the
"OFF at 2 cells / ON at 81 cells" assertion from the old Python block is
reproduced by scanning the traces — no Python needed.

#### 10.1.5 `24_spatial_diauxie.helix` → `#sim` long-tail hook

**Before**: the `SpatialDFBA(SpatialDFBAConfig(length=32, ...)).run(120)` block
is a comment. Spatial dFBA is a long-tail app, so it is exposed through the
generic extension point (§8.6) rather than a first-class backend:

**After** — every `SpatialDFBAConfig` field maps 1:1 to a `#sim` key:

```
# Example 24: Spatial dFBA - substrate-gradient colony growth
# A one-dimensional strip of dynamic-FBA batches coupled by glucose
# diffusion. The app that used to be Python-only now runs through the #sim
# extension point (kind=spatial_dfba, backend=fba).

#gene name=glucose_uptake
ATG GCT GGT GTA TAA
#end

#config backend=fba
#sim kind=spatial_dfba length=32
#sim inlet_glucose_mm=5.0 initial_glucose_mm=5.0
#sim initial_biomass_gdw=0.05 max_biomass_gdw=2.0
#sim glucose_diffusion_um2_s=2.0 steps=120
#sim output=depletion_front,co2_overflow
```

#### 10.1.6 `30_virtual_cell.helix` → `backend=benchmark`

**Before**: the whole calibrate→predict workflow is a comment block using
`VirtualCellBench(VirtualCellBenchConfig(...)).calibrate()` /
`run_prediction(...)`. The `#gene` in the file is inert.

**After** — the `VirtualCellBenchConfig` fields become `#config sim` keys
(truth + calibration + prediction split). No Python:

```
# Example 30: virtual-cell calibration -> prediction workflow
# The "calibrate then predict" protocol of the Virtual Cell Challenge 2025
# runs on backend=benchmark: fit the hidden biomass-to-ATP constant on the
# calibration condition, predict the harder one, report the four gates.
# The Python VirtualCellBench block that used to live here is gone.

#config backend=benchmark
#config truth_biomass_to_atp=5e6 truth_maintenance_atp_per_min=2.5e7
#config calibration_uptake=GLC=10.0 calibration_minutes=20
#config prediction_uptake=GLC=20.0 prediction_minutes=60
#config n_samples=150 fit_seed=0
#config output=scores,passed,all_passed
```

Config keys for `benchmark`/`calibration` mirror the run functions' signatures
(`VirtualCellBenchConfig` and `run_whole_cell_calibration`,
`apps/virtual_cell_bench.py:73` / `apps/whole_cell_calibration.py:422`); the
dict-typed keys (`calibration_uptake`) use the §6.3 comma/equals syntax.

### 10.2 New examples

- `31_whole_cell_adder.helix` — a `VirtualCell` built entirely from
  annotations (Phases 1–4).
- `32_colony_dfba.helix` — per-cell dFBA colony with metabolic stratification
  (Phase 5).
- `33_fba_diauxie.helix` — the batch dFBA diauxie trace, pure-config.
- `34_whole_cell_calibration.helix` — parameter recovery under adder noise.

#### 10.2.1 `31_whole_cell_adder.helix`

Two genes, one auto-activation edge, a glucose medium and a handful of
`#config` keys — the `VirtualCell` wiring that `whole_cell_calibration.py`
does in Python, expressed as source. `chromosome=` / `initial_level=` reuse the
existing `#gene` field passthrough (§7.2); `#config enzyme_capacity=true`
activates Phase-4 MOMENT caps via the default enzyme tables.

```
# Example 31: Whole-cell - adder size control, Cooper-Helmstetter
# replication and chaperone-mediated folding (NEW; Phases 1-4)
# A VirtualCell runs from in-language declarations only. The adder rule
# (division at birth_volume + adder_volume_um3) is #config; replication is
# timed on the chromosome map; folding is chaperone-assisted.

#promoter name=p_constitutive strength=-0.4

#gene name=gltA promoter=p_constitutive chromosome=0.1
ATG GCT GGT GCT TAA
#end

#gene name=zwf promoter=p_constitutive chromosome=0.7 initial_level=1.0
ATG GCT GGT GCT TAA
#end

#regulate gltA -> zwf strength=+0.6

#media nutrient=GLC concentration=10.0

#config backend=whole_cell
#config division_rule=adder adder_volume_um3=1.6 adder_noise_std=0.1
#config replication_mode=cooper_helmstetter doubling_time_min=30
#config c_period_min=20 d_period_min=10
#config protein_maturation_mode=chaperone frac_cotranslational_fold=0.7
#config enzyme_capacity=true
#config seed=0
#config ticks=120
#config output=energy,volume_um3,added_volume_um3,divisions,phase,proteins
```

#### 10.2.2 `32_colony_dfba.helix`

A 64×64 lattice of 2000 cells running one housekeeping program on a shared
glucose/oxygen `Environment`. Per-cell dFBA (`dfba=true`) lets the oxygen
gradient split the colony into a fermentative core and a respiratory edge —
read off with the `dfba_stratification` columns.

```
# Example 32: Colony - per-cell dFBA metabolic stratification (NEW; Phase 5)
# 2000 cells grow on shared glucose/oxygen fields; each cell re-solves a
# per-cell FBA every step. Oxygen is consumed at the edge, so the core turns
# fermentative - observable as the core/edge acetate split.

#promoter name=p_housekeeping strength=-0.4

#gene name=adhE promoter=p_housekeeping
ATG GCT GGT GCT TAA
#end

#media nutrient=GLC concentration=10.0 diffusion_um2_s=300
#media nutrient=O2 concentration=0.25 diffusion_um2_s=1600

#config backend=population
#config population_size=2000 grid_width=64 grid_height=64
#config dfba=true dfba_dt_h=0.1
#config dfba_glucose_half_saturation_mm=1.0
#config dfba_oxygen_max_uptake=20.0 dfba_oxygen_half_saturation_mm=0.002
#config signaling=true signal_diffusion=0.3 signal_threshold=20.0
#config mechanics=shove crowding=true
#config seed=0
#config ticks=60
#config output=alive_count,diversity_index,core_cell_count,edge_cell_count,core_oxygen_mm,edge_oxygen_mm,core_acetate_mm,edge_acetate_mm
```

#### 10.2.3 `33_fba_diauxie.helix`

Example 20 describes this curve but could not run it. A pure-config program —
no `#gene` needed — reproduces the Mahadevan et al. 2002 batch: exponential
growth on glucose, overflow CO2, growth arrest at exhaustion.

```
# Example 33: FBA - batch dFBA diauxie trace (NEW)
# backend=fba with dynfba=true runs the DynamicFluxBalance batch that
# examples/20 and 24 could only describe in comments. Glucose is the single
# #media nutrient; the O2 cap drives overflow. No genes required.

#media nutrient=GLC concentration=10.0

#config backend=fba
#config fba_model=core dynfba=true
#config fba_dt_h=0.25 fba_oxygen_max=20.0 fba_steps=32
#config output=time_h,biomass,glucose,co2,growth_rate
```

#### 10.2.4 `34_whole_cell_calibration.helix`

Recovering hidden parameters from synthetic observables — the workflow
example 30 performs as a *benchmark* — here as a *fit*: `n_samples` global
search + refinement, with `adder_noise_std>0` averaged over `n_cells=4`
independent cells (the √n noise reduction of the Phase-5 closure).

```
# Example 34: Whole-cell calibration (NEW)
# Recovers the hidden enzyme_scale/maintenance/adder/k_fold parameters from
# synthetic observables. adder_noise_std>0 makes each generation's added
# volume noisy, so n_cells independent cells back every observable vector.

#config backend=calibration
#config fit_seed=1 n_samples=60 refine_rounds=2
#config adder_noise_std=0.1 n_cells=4
#config output=best,sse,n_samples
```

Config keys map directly onto `run_whole_cell_calibration` arguments
(`apps/whole_cell_calibration.py:422`): `minutes`, `n_samples`,
`refine_rounds`, `fit_seed`, `adder_noise_std`, `n_cells`.

---

## 11. Determinism, observability, performance

- **Determinism**: `seed` drives all RNG (`VirtualCellConfig.seed`,
  GRN noise, population `noise_seed`, calibration `fit_seed`). Tests assert
  same-source + same-seed ⇒ identical output.
- **Observability**: full history keys (§6.7) surface Phase 1–5 state
  (phase, dna_copy_number, added_volume_um3, pools, overflow) — the state the
  classic trace never exposed.
- **Performance**: sim backends are Python (like today's library); no new
  hot loops. `classic` performance is unchanged by definition.

---

## 12. Backward compatibility and migration

- `backend` defaults to `classic`; the classic pipeline, bytecode format,
  CLI flags and web endpoints are untouched.
- The 24 classic examples stay unchanged and green. The 6 rewritten examples
  (§10.1) change *during* their phase (W-2/W-3/W-4) and are re-gated there;
  until then they run exactly as they do today.
- `#media`/`#enzyme`/`#metabolite` in a `classic` program produce a **warning**
  (not an error), matching today's lenient handling of unknown keys.
- No public Python API is broken: `sim_runtime.py` is purely additive.

---

## 13. Test plan and verification gates

**W-1 (parser)**
- `test_config_sim_collects_unknown_keys` — extra `#config` keys land in
  `Config.sim`; the 7 classic keys still parse.
- `test_media_enzyme_metabolite_parsed` — new annotations produce AST nodes.
- `test_classic_bitcompat` — all existing examples produce the same classic
  trace as before the parser change (golden traces).

**W-2 (adapter: whole_cell + fba)**
- `test_whole_cell_maps_annotations` — `#gene/#promoter/#regulate` →
  correct `VirtualCell` genome/GRN/config (incl. `k_fold`→`fold_rate`,
  `#media`→uptake, `#enzyme`→kcat).
- `test_whole_cell_adder_history` — `division_rule=adder` emits
  `added_volume_um3` and ≥1 division in `ticks` minutes.
- `test_fba_backend_fluxes` — static solve + dFBA batch (diauxie) produce
  non-empty flux tables.
- `test_sim_config_error` — malformed enum/float raises `SimConfigError`.

**W-3 (population)**
- `test_population_backend_colony` — `dfba=true` colony returns
  `colony_observables` with metabolic stratification (centre vs edge).
- `test_population_determinism` — same seed ⇒ identical observables.

**W-4 (calibration/benchmark + server)**
- `test_calibration_backend_recovers_parameters` — `backend=calibration`
  recovers the four hidden parameters within tolerance.
- `test_benchmark_backend_gates` — `backend=benchmark` returns all four gates
  passing.
- `test_api_sim_run` — `/api/sim/run` round-trips a whole-cell program.

**Exit gate per phase**: `mypy src` + `ruff check src tests` +
`pytest --cov=helixlang` (fail_under=80) — the existing project gates.

---

## 14. Implementation roadmap

| Phase | Scope | Files | Gate | Status |
|---|---|---|---|---|
| W-1 | parser: `Config.sim`, `backend`, `#media`/`#enzyme`/`#metabolite`; golden bit-compat tests | `ast_nodes.py`, `parser.py`, `tests/` | existing suite green | **done** |
| W-2 | `sim_runtime.py` (`whole_cell`, `fba`), CLI `--backend`/`--json`, examples 31/33 | `sim_runtime.py` (new), `cli.py`, `examples/` | W-1 + adapter tests | **done** |
| W-3 | `population` backend, examples 32 | `sim_runtime.py`, `examples/` | W-2 + colony tests | **done** |
| W-4 | `calibration`/`benchmark` backends, `/api/sim/run`, example 34, long-tail `#sim` hook | `sim_runtime.py`, `server.py`, `examples/` | W-3 + endpoint tests | **done** |
| W-5 | docs: this surface into `language-spec.md`/`bio-instructions.md` + fix §3.5 drift; README | `doc/*`, `README.md` | — | **done** |
| W-6 | long tail (§8.6, §19): register remaining apps behind `#sim`; rewire stub examples | `sim_runtime.py`, `parser.py`, `examples/` | W-5 + app tests | **done** |

W-1…W-5 each landed behind the project's quantitative gates; `classic` stayed
the default until the final gate, so every phase was merge-safe. W-6 shipped
the §19 backlog: 14 `#sim kind=` backends + example rewrites, all green behind
`mypy src`, `ruff`, and `pytest --cov` ≥ 80 %.

---

## 15. Documentation updates

In addition to documenting the new surface:

- `doc/language-spec.md` — add `#config backend` + sim keys (§3.6), new
  annotations (§3.9 `#media`/`#enzyme`/`#metabolite`), runtime semantics for
  sim backends (§6.8), and document `#morphogen` (drift).
- `doc/bio-instructions.md` — sim backend section; fix drift: remove
  `grid_width/grid_height` (replaced by real population keys), correct
  `ticks`/`ops_per_tick` defaults, `mito_vertebrate`, codon-table labels.
- `README.md` — module map row for `sim_runtime.py`; a "simulation backends"
  note in Highlights; docs table row for this document.

---

## 16. Open questions and risks

1. **GRN semantics mismatch** — **resolved in W-2**: classic `strength∈[−1,1]`
   edges map to `GRN.add_edge(source, target, strength)`; per-gene
   `threshold=` / `initial_level=` are honoured via the `#gene` field
   passthrough (`parser.py:144`), mirroring `whole_cell_calibration.py`.
2. **Units** — **deferred (open)**: `#media` concentrations are pass-through
   numerics; a future `units.py`-typed surface (mM ↔ FBA bound) is a follow-up,
   not part of W-1…W-5.
3. **Performance** — whole-cell history at `ticks` minutes is O(ticks) records;
   large `population_size` runs are the library's existing cost. Verified in
   W-3 (2000-cell colony ≈ seconds).
4. **Scope creep** — **resolved in W-6**: the long tail (§8.6, §19) is fully
   registered behind `#sim`; the 14 W-6 kinds share one dispatch dict and the
   parser hook, so new apps register with one line each.

---

## 17. Code anchors

| Concern | Location |
|---|---|
| Parser annotation dispatch | `parser.py:60-69` |
| `#config` parsing | `parser.py:229-246` |
| `Config` dataclass | `ast_nodes.py:75-86` |
| `#gene` extra-field passthrough | `parser.py:144-145` |
| VM central-dogma branch | `vm.py:556-581` |
| `VirtualCellConfig` | `virtual_cell.py:173-262` |
| `VirtualCell.__init__` | `virtual_cell.py:322` |
| GRN construction API | `grn.py:188-246` (`add_gene`, `add_edge`) |
| `PopulationConfig` | `population.py:186-215` |
| Colony observables / stratification | `population.py:1382`, `population.py:1426` |
| FBA / dFBA / enzyme caps / pools | `metabolism.py` (`FluxBalanceAnalysis`, `DynamicFluxBalance`, `EnzymeCapacity`, `MetabolitePool`) |
| Whole-cell calibration | `apps/whole_cell_calibration.py` |
| 4-gate benchmark | `apps/virtual_cell_bench.py:284` |
| CLI | `cli.py` |
| Server | `server.py` |

---

## 18. Coverage audit: what runs in-language today (Aug 2026)

Every `examples/*.helix` file was run end-to-end
(`helixlang examples/NN_*.helix --json`, and `--disassemble` for classic).
"Wired" means the feature executes from `.helix` source with no Python driver.

### 18.1 Wired — classic backend (bytecode VM, default)

| Feature | Annotation / config | Examples |
|---|---|---|
| GRN (promoter/gene/regulation) | `#promoter` `#gene` `#regulate` | 01–05, 11–12, 14–15, 17–19, 22, 25 |
| L-system 2D morphology | `#lsystem` | 03, 15 (2D turtle only) |
| Gray–Scott reaction-diffusion | `#field` | 04 |
| Central dogma steps | `#transcribe` / `#translate` + `use_central_dogma=true` | 09 |
| CRISPR gene editing | `#crispr` | 06 |
| Evolution (mutation/selection) | `#evolve` | 07 |
| Epigenetics | `#methylate` / `#histone` | 08 |
| Quorum signal gate | `#quorum` | 16 (per-cell gate) |
| Species / translation table | `#config species` / CLI `--table` | 05, 12 |
| DNA storage codec | CLI `--encode-dna` / `--decode-dna` | 01, 13 |
| `#morphogen` feedback | `#morphogen` | implemented (`parser.py:205`) but no example uses it; documented in `language-spec.md` §6.5 |

### 18.2 Wired — simulation backends (`sim_runtime.py`)

All of the following run in-language and are verified green:

| Feature | Backend / annotation | Examples |
|---|---|---|
| Static FBA | `#config backend=fba` | 10 |
| Dynamic FBA / diauxic batch | `backend=fba` + `dynfba=true` | 20, 33 |
| Spatial dFBA biofilm | `#sim kind=spatial_dfba` | 24 |
| Whole-cell `VirtualCell` (Phases 1–4) | `backend=whole_cell` | 31 |
| Cell population colony (Phase 5) | `backend=population` | 16, 21, 22, 32 |
| Whole-cell parameter calibration | `backend=calibration` | 34 |
| 4-gate whole-cell benchmark | `backend=benchmark` | 30 |
| Media / shared environment fields | `#media` | 10, 16, 20, 31–33 |
| Enzyme-constrained FBA | `#enzyme` | 10 |
| Metabolite pools | `#metabolite` + `metabolite_pools=true` | coerced in `sim_runtime`; **no example yet** |
| Deterministic seeds | `#config seed=` | 31, 32, 34 |

**W-6 long-tail `#sim kind=` backends** (all wired, verified green):

| Feature | Backend / annotation | Examples |
|---|---|---|
| Consortium quorum voting | `#sim kind=consortium` | — |
| Digital evolution (Avida) | `#sim kind=digital_evolution` | 23 |
| Stochastic gene expression | `#sim kind=stochastic` | — |
| DNA-storage codec benchmark | `#sim kind=codec_benchmark` | 27 |
| SynBio cassette designer | `#sim kind=synbio_design` | 14 |
| Protein fitness oracles | `#sim kind=protein_fitness` | — |
| Morphogen gradient / French flag | `#sim kind=morphogen_gradient` | 25 |
| Protein structure prediction | `#sim kind=protein_structure` | 11 |
| Fate analysis (toggle bistability) | `#sim kind=fate_analysis` | 28 |
| ML-guided directed evolution (GB1) | `#sim kind=directed_evolution` | 29 |
| 3D morphology (`LSystem3D`) | `#sim kind=3d_morphology` | 15 |
| Omics / omics calibration | `#sim kind=omics_calibration` | — |
| Cello closed-loop automation | `#sim kind=cello_workflow` | 26 |
| Codon usage / CAI analysis | `#sim kind=codon_usage` | 12 |

The W-6 dispatch lives in `_SIM_BACKENDS` (`sim_runtime.py`) and takes
precedence over the first-class backend — including the classic default — so
`#config backend=fba` is only a neutral placeholder on those examples.

### 18.3 NOT wired — Python-only, `.helix` file is a stub

**Resolved in W-6.** Every feature formerly listed here (protein structure,
codon usage, codec benchmark, synBio designer, 3D morphology, digital
evolution, morphogen gradient, Cello workflow, pattern synthesis, fate
analysis, directed evolution, omics calibration, consortium, protein fitness,
stochastic) is now reachable from `.helix` source through the backends in
§18.2. The interop surface (`interop.py` SBML/SBOL) and the internal
numpy-vectorized path remain Python-import surfaces by design (not language
features).

### 18.4 Wired in the adapter but not yet exercised by an example

- `#metabolite` + `metabolite_pools=true` (whole-cell pools).
- `#morphogen` (classic GRN↔field feedback).
- CLI `--backend` override and `--json` output (covered by tests
  `tests/test_sim_runtime.py`, `tests/test_api_sim_run.py`).

`mechanics=force` population runs are now exercised by example 22
(`pattern_synthesis`), and the `#sim kind=` column selection (`#sim output=`)
is covered in `tests/test_sim_runtime.py`.

---

## 19. Next steps: wiring the remaining Python surface

**Completed in W-6.** The long tail registered every §18.3 feature behind the
`#sim` extension point — one `kind=...` per app, dispatched through the
`_SIM_BACKENDS` dict in `sim_runtime.run()`, no parser changes. The plan
below is retained as the historical record of what shipped:

**Tier 1 — low effort, immediate win (1–2 backends)**
1. **`#sim kind=consortium`** — reuses the `population` backend
   (`run_consortium_quorum`); a shared-pool community with `#media` fields.
2. **`#sim kind=digital_evolution`** — `DigitalEvolutionConfig` mirrors to
   `#config` keys; returns fitness curves as a `ScoreResult`.
3. **Convert example 22** (`pattern_synthesis`) to `backend=population`
   `mechanics=force` — zero new code, kills one stub.
4. **`#sim kind=stochastic`** — telegraph-promoter Fano / Gillespie SSA trace;
   keys map 1:1 to `stochastic.py` signatures.

**Tier 2 — small dedicated backends**
5. **`#sim kind=codec_benchmark`** (27) and **`#sim kind=synbio_design`** (14) —
   both are pure-config report generators (no genes required), mirroring how
   `backend=benchmark` handles example 30.
6. **`#sim kind=protein_fitness`** — BLOSUM62 oracle is dependency-free;
   ESM-2 path stays optional (`fitness_mode=blosum|esm`).

**Tier 3 — richer runtimes**
7. **`#sim kind=morphogen_gradient`** (25) — add a 1-D source+diffusion+decay
   strip to `sim_runtime`, reusing `morphogen_gradient.py` untouched.
8. **`#sim kind=protein_structure`** (11) — `predict_structure(seq)` is already
   CPU-light; expose the report's summary fields as `ScoreResult`.
9. **`#sim kind=fate_analysis`** (28) — three sub-modes
   (`bistability_scan` / `switching_rate` / `critical_slowing_down`).
10. **`#sim kind=directed_evolution`** (29) — `rounds/library_size/top_k` keys;
    oracle guard-rail (fall back to BLOSUM when transformers is absent).
11. **`#sim kind=3d_morphology`** (15) — `LSystem3D` preset + generation count;
    emit mesh stats. Distinct from the 2D `#lsystem` renderer.
12. **`#sim kind=omics_calibration`** — heaviest; needs matrix inputs, so it
    stays last and may take JSON payloads rather than `#config` keys.

**Each tier** lands with: a stub-example rewrite (delete the Python comment
block), a `tests/test_sim_runtime.py` case, and the §14 gate
(`mypy src`, `ruff`, `pytest --cov` ≥ 80 %). All twelve shipped in W-6 behind
the same gate; remaining loose ends are §18.4 (`#metabolite` example,
`#morphogen` example) and the Python-import-only interop surface.
