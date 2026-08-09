# Gameplay-Unit Calibration Plan — From Toy Design to Physical Biology

> Goal: turn the remaining **gameplay units** (arbitrary, dimensionless energy / signal / threshold
> constants that the previous upgrade rounds only *documented*) into **physically grounded,
> literature-cited calibration targets**, without breaking the language, the VM semantics, the
> public API, or the green test suite.
>
> This plan is the follow-on to `production-upgrade.md` §4.10: Batch 10 registered the magic
> numbers as named constants and added honest `UNITS` disclaimers. This plan goes one step
> further — it defines **what physical quantity each gameplay unit should mean**, derives a
> consistent unit system, and proposes opt-in `calibrated=` re-parameterization backed by primary
> literature.
>
> Baseline: **1471 tests passing**, coverage **90%**, `ruff` + `mypy` clean
> (`/opt/anaconda3/envs/helix/bin/python`).

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Audit Ledger — The Gameplay-Unit Catalog](#2-audit-ledger--the-gameplay-unit-catalog)
3. [The Unit System Design](#3-the-unit-system-design)
4. [Physical Calibration Targets (Primary Literature)](#4-physical-calibration-targets-primary-literature)
5. [Tiered Implementation Plan](#5-tiered-implementation-plan)
6. [Compatibility and API Preservation](#6-compatibility-and-api-preservation)
7. [Verification Strategy](#7-verification-strategy)
8. [Implementation Batches](#8-implementation-batches)

---

## 1. Background and Motivation

HelixLang's *language* modules (codon tables, DNA codec, CRISPR, metabolism, evolution,
central dogma) are anchored in real, cited constants. But the *runtime simulation* — the part
that actually "runs life" — still uses **gameplay units**: a dimensionless cellular-automaton
budget that is internally consistent but physically unattached.

Example: a cell starts with `energy = 100`, loses `1` per move, gains `10` per `feed`,
divides at `200`, and senses quorum at a signal of `5.0`. Nothing wrong *in play* — the numbers
are tuned so examples behave nicely. But none of these numbers means anything physical, and the
previous upgrade rounds (production-upgrade.md Batch 10) deliberately stopped at *documenting*
them ("gameplay units, not Joules") rather than *calibrating* them.

**Why calibrate now?**
- **Cross-module consistency.** The FBA module already computes real ATP maintenance fluxes
  (Orth 2010, 8.39 mmol/gDW/h) and the central-dogma module uses real half-lives and elongation
  rates — yet `Cell.energy` floats in its own arbitrary world. Calibration couples the two so a
  simulated cell's energy budget is consistent with its own metabolome.
- **Measurable claims.** With a unit system, "a cell doubles every ~20 ticks" becomes
  "a cell doubles every ~20 minutes (rich medium, 37 °C)", and "quorum at signal 5.0" becomes
  "quorum at ~10 µM AI-2 (Xavier & Bassler 2003)".
- **Predictive tests.** Physical anchoring turns the test suite into a *validation* suite
  (does a starved cell die on schedule? does the doubling time match Neidhardt?).

**Design constraints** (unchanged from production-upgrade.md §5):
1. Do not change existing functionality — every default stays, every documented API keeps working.
2. Keep all tests green (1471 baseline) plus ruff + mypy.
3. Stay dependency-light: stdlib first; numpy only where already optional.
4. New behavior is **opt-in** behind `calibrated=` / `units=` kwargs.
5. Docstrings stay honest — each calibrated module updates its `UNITS` disclaimer to name the
   physical mapping now in force.

---

## 2. Audit Ledger — The Gameplay-Unit Catalog

Line numbers are exact as of this document. Every entry is a **gameplay unit**: dimensionless
and/or not tied to a cited physical measurement. Entries already cited as physical are marked
and **not** targets.

### 2.1 `cell.py` — single-cell energy budget

| Constant | Value | Meaning | Physical counterpart |
|---|---|---|---|
| `INITIAL_CELL_ENERGY` (L27) | `100` | starting energy budget | ATP pool / biomass budget |
| `CELL_PROTEIN_SLOT_COUNT` (L29) | `256` | protein species capacity | symbolic capacity (E. coli ≈ 4,300 genes) |
| `MOVE_ENERGY_COST` (L31) | `1` | energy per step | chemotaxis / flagellar motor ATP cost |
| `FEED_ENERGY_AMOUNT` (L33) | `10` | nutrient intake per `feed` | glucose uptake → ATP yield |
| `MIN_DIVISION_ENERGY` (L35) | `2` | minimum energy to divide | growth-threshold → biomass accumulation |
| `DEFAULT_CELL_COLOR` (L37) | `(255,255,255)` | display only | none (presentation) |
| `MAX_MEMBRANE_PERMEABILITY` (L39) | `255` | permeability scale | porin / transporter density |
| `DEFAULT_MEMBRANE_PERMEABILITY` (L43) | `255` | fully permeable | high-porin rich-medium default |
| `divide()` halving (L120–126) | `//2` | symmetric division | equal biomass partition |

### 2.2 `population.py` — multicellular lattice budget

| Constant | Value | Meaning | Physical counterpart |
|---|---|---|---|
| `DEFAULT_MAX_POPULATION_SIZE` (L43) | `10000` | lattice capacity | none (numerical) |
| `DEFAULT_GRID_WIDTH/HEIGHT` (L45–46) | `100` | lattice size | biofilm patch size (µm scale) |
| `DIVISION_ENERGY_THRESHOLD` (L48) | `200.0` | energy to divide | doubling-time biomass target |
| `DEATH_ENERGY_THRESHOLD` (L50) | `0.0` | energy → death | starvation death |
| `SIGNAL_DIFFUSION_COEFFICIENT` (L53) | `0.1` | on-lattice diffusion | D_phys → lattice conversion |
| `QUORUM_SIGNAL_THRESHOLD` (L56) | `5.0` | quorum trigger | ~10 µM AI-2 (Xavier & Bassler 2003) |
| `METABOLIC_COST_PER_STEP` (L58) | `1.0` | maintenance cost | ATP maintenance flux (Orth 2010) |
| `ENERGY_INTAKE_PER_STEP` (L60) | `5.0` | rich-medium intake | glucose uptake rate |
| `POPULATION_CELL_INITIAL_ENERGY` (L62) | `100.0` | newborn energy | newborn biomass |
| signal emission `+1.0` (L385–387, 434) | `1.0` | per-cell-per-step signal | AI-2 secretion flux |

### 2.3 `vm.py` — runtime opcode semantics

| Constant | Value | Meaning | Physical counterpart |
|---|---|---|---|
| `REGULATE_EDGE_WEIGHT` (L45) | `1.0` | runtime rewired edge weight | effector concentration increment (nM) |
| `BIND_LEVEL_BOOST` (L52) | `0.5` | protein–DNA binding boost | TF occupancy fold-change (McClure 1985) |
| `EMIT_MORPHOGEN_SCALE` (L57) | `256` | morphogen emission divisor | morphogen dose injected into field |
| `SIGNAL_EMISSION_AMOUNT` (L64) | `0.25` | autoinducer per `OP_SIGNAL` | µM AI-2 per secretion event |
| `OP_FEED` literal `10` (L158) | `10` | **hardcoded, not the named constant** | should be `FEED_ENERGY_AMOUNT` |
| `OP_BUILD_PIGMENT` color (L134) | `(200,50,50)` | display only | none |
| `_handle_quorum` threshold default (L414) | `5.0` | duplicated quorum threshold | should reference a shared constant |
| central-dogma `ribosome_density=0.1` (L586) | `0.1` | ribosomes per 100 nt | ribosome loading (real: ~1 per 100 nt) |
| central-dogma protein factor `*0.1` (L593) | `0.1` | mRNA → protein yield | proteins per mRNA lifetime (real: ~10²–10³) |
| central-dogma GRN feedback `*0.01` (L600) | `0.01` | protein → gene level | translation-coupled regulation gain |
| `_feedback` morphogen gain `v*0.1` (L704) | `0.1` | field V → pigment activation | morphogen → transcription occupancy |
| `_get_promoter_strength` default `0.5` (L606) | `0.5` | constitutive expression | constitutive promoter reference level |
| `_call_gene` / frame cap `256` (L636, 646) | `256` | scheduler guard | none (numerical) |

### 2.4 `grn.py` — regulatory kinetics

| Constant | Value | Meaning | Physical counterpart |
|---|---|---|---|
| `DECAY = 0.7` (L104) | `0.7` | legacy universal decay | protein half-life (level halves every ~2 ticks) |
| `level > 0.5` trigger (L178) | `0.5` | expression trigger | expression above half-maximal |
| `GeneNode.threshold` default (L81) | `0.5` | sigmoid midpoint | threshold − input = 0 |
| optional `hill_n` / `kd` (L84–85) | — | Hill kinetics | real Kd (lacI ~0.1 nM, Oehler 1990) |

### 2.5 `config` / geometry (document, don't calibrate)

| Constant | Value | Meaning | Status |
|---|---|---|---|
| `Config.ticks` (ast_nodes L64) | `100` | sim duration | OK |
| `Config.ops_per_tick` (L67) | `64` | op budget per tick | scheduler budget; tie to translation capacity (opt-in) |
| `Config.react_steps` (L68) | `1` | field steps per `OP_REACT` | OK |
| `LSystem` angle/step (lsystem L20) | `25.0° / 1.0` | turtle geometry | geometric, not a biological constant |
| `LSystem3D` angle/step (morphology_3d L97) | `22.5° / 1.0` | 3D turtle geometry | geometric |

### 2.6 Already-physical (NO ACTION)

`central_dogma.py` elongation/half-life rates (Proshkin 2010, Ingolia 2009, Bernstein 2002),
`metabolism.py` fluxes (Orth 2010), `evolution.py` rates (Lee 2012, Drake 1991), `dna_codec.py`
error rates (Saiki 1988, Potapov 2017, Filges 2021), `reaction_diffusion.py` Gray-Scott presets
(Pearson 1993), `crispr.py` tables (Doench 2016, Hsu 2013), `bio_data.py` codon/tRNA tables
(CUTG/Kazusa, Dong 1996). The Gray-Scott F/k/Du/Dv are **dimensionless model parameters by
definition** (Pearson 1993) — they generate patterns, they do not need physical calibration.

---

## 3. The Unit System Design

The core deliverable of this plan is a **consistent unit system** plus a calibration registry,
so every gameplay constant has a physical meaning and a conversion factor. Four base axes
suffice; everything else derives from them.

### 3.1 Base axes

| Axis | Unit | Anchor | Basis |
|---|---|---|---|
| **Time** | `TIME_TICK_MIN = 1.0` (1 tick = 1 minute) | E. coli doubling ~20 min rich medium | Neidhardt 1996; matches `central_dogma.calculate_mrna_level(time_step_min=5.0)` and the GRN half-life model (E. coli protein half-life median ~110 min ⇒ decay ≈ 0.994) |
| **Space** | `LATTICE_SPACING_UM` (lattice site edge, µm) | cell-scale lattice | default proposal 10 µm (a 100×100 grid = 1 mm biofilm patch); must be declared before diffusion calibration |
| **Energy** | `ENERGY_UNIT` = **10⁷ ATP molecules** (≈ 1 min of maintenance) | 8.39 mmol ATP/gDW/h × 0.3 pg DW/cell × (60 min)⁻¹ ≈ 2.5×10⁷ ATP/min | Orth 2010 + E. coli dry mass ~0.3 pg |
| **Concentration** | `SIGNAL_UNIT` = **2 µM per lattice unit** | quorum threshold: 5.0 lattice units = 10 µM AI-2 | Xavier & Bassler 2003 |

### 3.2 Derived conversions

```
energy_to_atp(e)       = e × 10⁷                     # energy unit → ATP molecules
ticks_to_min(t)        = t × TIME_TICK_MIN           # trivial while TIME_TICK_MIN = 1
diffusion_to_lattice(D_phys_um2_s, dt_s, dx_um) =
    D_phys_um2_s × dt_s / dx_um**2                   # dimensionless on-lattice D
signal_to_um(s)        = s × SIGNAL_UNIT             # lattice units → µM
```

**The diffusion inconsistency (worked example).** The current `SIGNAL_DIFFUSION_COEFFICIENT =
0.1` is "dimensionless on-lattice". For a small molecule such as AI-2, D_phys ≈ 10⁻⁶ cm²/s.
Converting to µm²/s (1 cm² = 10⁸ µm²): 10⁻⁶ cm²/s = **100 µm²/s**. With `dt = 60 s` and
`dx = 10 µm`:

```
D_lattice = 100 µm²/s × 60 s / 100 µm² = 60        # calibrated value
```

The legacy `0.1` therefore corresponds to a **very coarse lattice**: solving
`0.1 = 100 × 60 / dx²` gives `dx ≈ 245 µm` — i.e. each lattice site currently represents a
*colony-scale* patch, not a single cell. The calibration registry must therefore carry the
`(dx, dt)` pair explicitly so the physical meaning of `D_lattice` is unambiguous. This is the
single most important correction the plan makes: the same number means different physics
depending on the declared lattice scale.

### 3.3 The calibration registry (`helixlang/units.py`)

A new stdlib-only module holding the base axes and a `CALIBRATED` table mapping **every** audit
entry in §2 to `(physical_value, unit, citation, conversion_fn, legacy_default)`:

```python
# helixlang/units.py  (new module — stdlib only)
TIME_TICK_MIN      = 1.0                 # 1 tick = 1 min (Neidhardt 1996)
LATTICE_SPACING_UM = 10.0                # default lattice edge; declare, don't hide
ENERGY_UNIT_ATP    = 1.0e7               # ≈ 1 min of maintenance ATP (Orth 2010)
SIGNAL_UNIT_UM     = 2.0                 # 1 lattice unit = 2 µM (Xavier & Bassler 2003)
ATP_PER_GLUCOSE    = 38                  # textbook aerobic yield (Alberts)

def diffusion_to_lattice(D_um2_s: float, dt_s: float, dx_um: float) -> float: ...
def energy_to_atp(energy: float) -> float: ...
def signal_to_um(signal: float) -> float: ...

CALIBRATED = {
    # name                     physical value           unit        citation
    "cell.INITIAL_CELL_ENERGY": (100.0,                 "energy",   "Orth 2010"),
    "population.DIVISION_ENERGY_THRESHOLD": (180.0,     "energy",   "Neidhardt 1996"),
    ...
}
```

`CALIBRATED` is the single source of truth for both the `calibrated=` kwargs and the `--units`
CLI flag; every value carries its citation so future updates are a one-line edit.

---

## 4. Physical Calibration Targets (Primary Literature)

All targets are already cited somewhere in the codebase (so they are trusted, re-usable anchors)
plus the two textbook anchors below.

| Ref | Work | DOI / Access | Use |
|---|---|---|---|
| R1 | Orth et al. 2011, *A comprehensive genome-scale reconstruction of E. coli metabolism (iJO1366)*, Mol Syst Biol 7:535 | 10.1038/msb.2011.65 | ATP maintenance 8.39 mmol/gDW/h; energy-unit anchor (already in `metabolism.py`) |
| R2 | Neidhardt 1996, *Escherichia coli and Salmonella* (rich medium growth) | textbook | E. coli doubling ~20 min at 37 °C rich medium; time-axis anchor |
| R3 | Xavier & Bassler 2003, quorum-sensing AI-2 uptake/processing, Genes Dev 17:971 | 10.1101/gad.1099803 | AI-2 ~10 µM threshold; concentration-axis anchor (already cited in `population.py`) |
| R4 | Oehler et al. 1990, lacI repressor binding, EMBO J 9:973 | 10.1002/j.1460-2075.1990.tb08199.x | lacI Kd ~0.1 nM; Hill kd anchor (already cited in `grn.py`) |
| R5 | Mosteller 1980 / Helbig 2011, E. coli protein half-life median ~110 min | 10.1016/S0021-9258(19)85713-9 / 10.1002/pmic.201000335 | per-gene decay anchor (already used by `decay_from_half_life_ticks`) |
| R6 | Berg & von Hippel 1987, *Selection of DNA binding sites*, PNAS 84:7827 | 10.1073/pnas.84.22.7827 | `OP_BIND` specificity/occupancy grounding |
| R7 | McClure 1985, *Mechanism and control of transcription initiation in prokaryotes*, Annu Rev Biochem 54:171 | 10.1146/annurev.bi.54.070185.001131 | activator fold-change / `BIND_LEVEL_BOOST` |
| R8 | Ingolia 2009, *Genome-wide analysis in vivo of translation*, Science 324:218 | 10.1126/science.1168978 | 20 aa/s elongation; translation-capacity anchor for `ops_per_tick` |
| R9 | Miller & Bassler 2001, *Quorum sensing in bacteria*, Annu Rev Microbiol 55:165 | 10.1146/annurev.micro.55.1.165 | `OP_SIGNAL` autoinducer secretion model |
| R10 | Bernstein et al. 2002, mRNA decay, J Bacteriol 184:6477 | 10.1128/JB.184.23.6477-6488.2002 | mRNA half-life; `*0.1` yield calibration |
| R11 | Alberts et al., *Molecular Biology of the Cell* | textbook | ATP yield ~38 ATP/glucose (aerobic); cell dry mass ~0.3 pg |
| R12 | O'Toole et al. 2000, *Biofilm formation as microbial development*, Annu Rev Microbiol 54:49 | 10.1146/annurev.micro.54.1.49 | biofilm-density → grid-scale rationale (already cited in `population.py`) |

---

## 5. Tiered Implementation Plan

### Tier 0 — this document (DONE when this file ships)

The unit system (§3), the catalog (§2), and the calibration targets (§4) are defined. Nothing
in `src/` changes.

### Tier 1 — `helixlang/units.py` + calibration registry (highest value)

- **Code**: new stdlib-only module `src/helixlang/units.py` per §3.3: base-axis constants,
  `diffusion_to_lattice` / `energy_to_atp` / `signal_to_um` conversion functions, and the
  `CALIBRATED` table carrying `(physical_value, unit, citation, conversion_fn, legacy_default)`
  for every §2 row.
- **Expose** in `helixlang/__init__.py` (`__all__`).
- **No behavior change** anywhere: `CALIBRATED` is data, not execution.
- **Tests**: table invariants (every §2 constant name resolves to the right module namespace;
  every entry has a citation and a `legacy_default` matching the current value; conversions are
  dimensionally consistent — e.g. `signal_to_um(5.0) == 10.0`, `energy_to_atp(1.0) == 1e7`).

### Tier 2 — opt-in physical re-parameterization (`calibrated=` / `units=`)

Each module gains a `calibrated: bool = False` (or `units: str = "gameplay"`) kwarg. Default
(`False`) reproduces today exactly; `True` activates the §4-derived values. Batches are ordered
by dependency (units.py first, then consumers).

#### 5.1 `cell.py` — energy → ATP budget

- `Cell(..., energy: int = INITIAL_CELL_ENERGY)` unchanged; add `Cell(..., calibrated=False)`.
- When calibrated: `INITIAL_CELL_ENERGY` kept at the same *count* but reinterpreted as
  `10⁹ ATP` (100 × 10⁷); `FEED_ENERGY_AMOUNT` derived from glucose uptake → ATP yield:
  `ENERGY_INTAKE_PER_STEP` (5 units = 5 × 10⁷ ATP/min) ≈ 0.1% of a division budget — consistent
  with the ~20-tick doubling cycle; `MOVE_ENERGY_COST` mapped to flagellar motor ATP
  (order 10³–10⁴ ATP per flagellar revolution) documented, still 1 energy unit for play.
- `OP_FEED` (vm.py L158) must use `FEED_ENERGY_AMOUNT` instead of the hardcoded `10` —
  a **correctness fix** independent of calibration (default value identical, tests unaffected).
- **Tests**: calibrated `Cell` reaches the calibrated division threshold in ~20 ticks under
  rich-medium intake; starvation (no `feed`) dies within `INITIAL_CELL_ENERGY / cost` ticks;
  `calibrated=False` equals today's behavior (existing `tests/test_cell.py` 52 tests).

#### 5.2 `population.py` — doubling time + quorum + diffusion

- `PopulationConfig(..., calibrated=False)`.
- Calibrated `DIVISION_ENERGY_THRESHOLD`: derived so that a newborn cell at
  `POPULATION_CELL_INITIAL_ENERGY` with `ENERGY_INTAKE_PER_STEP − METABOLIC_COST_PER_STEP`
  net flux reaches the threshold in **20 ticks** (rich-medium doubling time, R2). Worked number
  from §3: net +4/tick ⇒ threshold − initial = 80 ⇒ threshold 180 (initial 100). The current
  default 200 gives ~25 ticks — same order; calibration pins it to 20.
- Calibrated `QUORUM_SIGNAL_THRESHOLD`: unchanged count (5.0) but now *declared* = 10 µM via
  `SIGNAL_UNIT_UM` (R3), and `signal_emitted`/field units documented as µM.
- Calibrated `SIGNAL_DIFFUSION_COEFFICIENT`: computed from D_phys ≈ 100 µm²/s (AI-2 small
  molecule), `dt=60 s`, declared `dx=LATTICE_SPACING_UM` via `diffusion_to_lattice` ⇒ ~60 at
  dx=10 µm (see §3.2). The legacy 0.1 stays as the gameplay default; the registry records that
  it implicitly assumes dx ≈ 245 µm.
- **Tests**: doubling time — a homogeneous clonal population under calibrated params has
  `population_size × 2` after 20 ticks ± tolerance; quorum — an emitting-cell cluster above the
  (µM) threshold activates the quorum gene, below it does not; diffusion — a point-source
  field's spread at tick t matches the analytical 2D-Gaussian width for the calibrated D;
  `calibrated=False` preserves all 27 `tests/test_population.py` tests.

#### 5.3 `vm.py` — runtime constants + coupling gains

- `REGULATE_EDGE_WEIGHT` / `BIND_LEVEL_BOOST` / `EMIT_MORPHOGEN_SCALE` /
  `SIGNAL_EMISSION_AMOUNT` stay as gameplay defaults but gain calibrated values in `CALIBRATED`
  (edge weight → nM effector increment consistent with Hill `kd=`; signal emission → µM per
  event such that ~5 adjacent emitters cross the 10 µM threshold — the current 0.25/1.0 × 5.0
  ≈ 10 µM mapping is already close).
- Promote the **uncited coupling constants to named, cited constants** (a Tier-2 correctness
  change even when counts stay identical): `RIBO_SOME_DENSITY_PER_100NT = 0.1`,
  `PROTEIN_YIELD_PER_MRNA_AA = 0.1`, `PROTEIN_TO_GRN_GAIN = 0.01`,
  `MORPHOGEN_TO_GRN_GAIN = 0.1`, `CONSTITUTIVE_PROMOTER_STRENGTH = 0.5`. Register each in
  `CALIBRATED` with a citation (R10 for the yield, R6/R7 for the gains) and a `legacy_default`
  equal to today's value — **no numerical change in default mode**.
- **Tests**: constants resolve and equal today's literals (`vm._feedback` uses the named
  constant, not `0.1`); calibrated mode changes only the registered values; existing
  `tests/test_vm.py` (incl. the 8 `TestRegulateBind` and signal tests) pass unchanged.

#### 5.4 `grn.py` — decay from half-life by default (opt-in)

- `GRN` gains `calibrated: bool = False`; when `True`, genes with no explicit `decay=` default
  to `decay_from_half_life_ticks(110)` (E. coli median protein half-life, R5) instead of the
  legacy `DECAY = 0.7`.
- **Tests**: under calibrated mode a single-gene GRN with no edges decays to half level at
  ~110 ticks (within tolerance); `calibrated=False` keeps the legacy 0.7 (existing
  `tests/test_grn.py` 12 tests unchanged).

#### 5.5 `central_dogma.py` — couple to the energy axis (optional)

- Document + optionally expose a `units=` on `calculate_mrna_level` so protein yield can be
  expressed in molecules (R10: ~10²–10³ proteins per mRNA lifetime) instead of the arbitrary
  `0.1` factor. **Low priority** — the elongation/half-life constants are already real; this
  only relabels the coupling constant introduced in vm.py §5.3.

### Tier 3 — `#config units=real` end-to-end (future)

- A language-level `units=real` config that activates calibrated defaults across `Cell`,
  `PopulationConfig`, `GRN`, and the VM in one switch; snapshots/CSV gain a `unit` metadata row
  (energy in ATP, signal in µM, field in µm); the web frontend displays the physical units.
- Output wire format for CSV/PNG is stable (same column order; metadata added, not reordered).
- Explicitly out of scope until Tier 1+2 land and the validation suite (§7) is green.

---

## 6. Compatibility and API Preservation

1. **Defaults never change.** Every `calibrated=False` / `units="gameplay"` default reproduces
   today's numbers bit-for-bit; the 1471-test baseline must pass without modification.
2. **New behavior is opt-in** behind `calibrated=` kwargs or `#config units=real`.
3. **No new hard dependencies.** `units.py` is stdlib-only; numpy stays optional.
4. **Constants become canonical.** Hardcoded literals that duplicate a named constant
   (`vm.py:158` `feed(10)`, `vm.py:414` quorum `5.0`) are replaced by the constants with
   *identical* default values — a correctness/consistency fix, not a behavior change.
5. **Docstrings stay honest.** Each module's `UNITS` note is updated to name the physical
   mapping when calibrated mode is on; the disclaimers remain accurate for gameplay mode.

---

## 7. Verification Strategy

Every batch, under `/opt/anaconda3/envs/helix/bin/python`:

```bash
ruff check src tests                     # lint gate
mypy                                     # type gate
python -m pytest -q                      # full suite: 1471 baseline
python -m pytest -q --cov=helixlang --cov-fail-under=80
```

Per-batch additions (the "validation suite" — these are the *point* of calibration):

| Batch | Validation test | Checks |
|---|---|---|
| T1 units.py | conversion dimension checks | `signal_to_um(5.0)==10.0`, `energy_to_atp(1.0)==1e7`, `diffusion_to_lattice(100,60,10)≈60`; every `CALIBRATED` name resolves; `legacy_default` == current constant |
| T2.1 cell | doubling/starve cycle | calibrated cell divides in ~20 ticks rich medium; starved cell dies on schedule; `calibrated=False` bit-identical |
| T2.2 population | doubling time / quorum / diffusion | clonal population doubles in ~20 ticks; µM quorum threshold flips a cluster but not an isolated cell; Gaussian point-source spread matches D |
| T2.3 vm | constant canonicalization + calibrated values | named constants replace literals with equal values; calibrated mode changes only registered entries |
| T2.4 grn | half-life decay | calibrated single-gene level reaches 0.5× at ~110 ticks; legacy 0.7 preserved |
| T2.5 central_dogma | yield mapping | protein yield in molecules consistent with mRNA lifetime (R10) when `units=` on |
| T3 | end-to-end | `#config units=real` runs all 16 `examples/*.helix` with physical-unit metadata; same traces modulo documented gains |

Coverage gate stays ≥ 80% (new registry + constants add coverage, not reduce it).

---

## 8. Implementation Batches

Status ledger for §5. Every batch is verified under `/opt/anaconda3/envs/helix/bin/python`
with the gates of §7. Totals after batches 1–8 land: **1517 passing**, coverage **90.4%**,
ruff + mypy clean. Validation suite: `tests/test_units.py` + calibrated sections in
`tests/{test_cell,test_grn,test_population,test_vm,test_central_dogma,test_semantic,
test_parser,test_server,test_end_to_end}.py`.

| Batch | Section | Scope | Status |
|---|---|---|---|
| 1 | §5.0 / Tier 0 | This document (unit system + catalog + targets) | **DONE** |
| 2 | §5.1 Tier 1 | `helixlang/units.py` + `CALIBRATED` registry + `__all__` export | **DONE** |
| 3 | §5.2 Tier 2 | `cell.py` `calibrated=` + `OP_FEED` constant fix | **DONE** |
| 4 | §5.3 Tier 2 | `population.py` `calibrated=` (threshold, quorum µM, diffusion) | **DONE** |
| 5 | §5.4 Tier 2 | `vm.py` named coupling constants + calibrated runtime values | **DONE** |
| 6 | §5.5 Tier 2 | `grn.py` half-life decay default when calibrated | **DONE** |
| 7 | §5.6 Tier 2 | `central_dogma.py` `units=` yield mapping (optional) | **DONE** |
| 8 | §5.7 Tier 3 | `#config units=real` + output metadata + frontend units | **DONE** |
| 9 | Docs | update `language-spec.md`, `bio-instructions.md`, `simulation-model.md`, `api-reference.md` with the unit system and `units=` | **DONE** |

**Pending follow-ups (pre-existing, orthogonal)**: production-upgrade.md §4.2 Erlich/legacy
Goldman; §4.4 TMHMM-style TM; §4.5 codon-dependent `stop_efficiency`; the remaining
parameterized-but-functional opcode operands (`OP_FEED <src>`, `OP_DIVIDE <mode>`, `OP_DIE`,
`OP_DIFFUSE`, `OP_REACT`).

---

## Appendix — Quick reference: gameplay → physical (post-Tier-2, calibrated mode)

| Gameplay quantity | Default (gameplay) | Calibrated meaning | Citation |
|---|---|---|---|
| 1 tick | — | 1 minute | Neidhardt 1996 |
| 1 energy unit | — | 10⁷ ATP molecules | Orth 2010 + Alberts |
| division threshold | 200 | reachable in ~20 ticks rich medium | Neidhardt 1996 |
| quorum signal 5.0 | — | 10 µM AI-2 | Xavier & Bassler 2003 |
| diffusion 0.1 | — | D_phys ≈ 100 µm²/s at dx=245 µm; calibrated mode recomputes at declared dx | §3.2 |
| GRN decay 0.7 | — | half-life 110 min ⇒ decay ≈ 0.994 | Mosteller 1980, Helbig 2011 |
| lacI repression | — | Kd ~0.1 nM | Oehler 1990 |
| translation 20 aa/s | — | ops_per_tick capacity reference | Ingolia 2009 |
