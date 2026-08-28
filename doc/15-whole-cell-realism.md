# Toward a Physically Complete Virtual Cell: Design Roadmap

Status: **implemented (Phases 1–5) and gated**. Every proposal below has
landed on a concrete module behind backward-compatible flags whose defaults
reproduce the original behaviour; each phase ships tests and passes the
project quality gates (`mypy src`, `ruff check src tests`,
`pytest --cov=helixlang`). §12 tracks exactly what each
phase shipped and where.

---

## 1. Goal

Take the current integrated lifecycle model (`VirtualCell`, Karr-2012 style
four-layer coupling: gene regulatory network + central dogma + flux-balance
metabolism + energy budget) and, in five literature-anchored phases, close the
gaps that separate it from a "complete, physically realistic" virtual cell:

1. **cell-cycle phasing and chromosome replication timing**,
2. **volume growth and size control (adder)**, with real physical units,
3. **protein maturation, folding, quality control and turnover**,
4. **enzyme-constrained metabolism and intracellular metabolite-pool dynamics**,
5. **multicellular coupling and a real-data calibration/benchmark closure**.

The invariant throughout: **never break backward compatibility of the public
API**. New realism is added behind new classes, configuration flags and
extension points, and the defaults keep the pre-Phase-1 1,879-test green
baseline behaviour intact.

---

## 2. AS-IS: what the code simulates today

### 2.1 The integrated cell

`helixlang.plugins.runtime.virtual_cell.VirtualCell` (`virtual_cell.py:120`) couples four
layers. Each `step()` advances exactly **one minute**
(`VirtualCellConfig.minutes_per_step = 1.0`):

1. advance the GRN (`grn.py`) and collect the genes that cross their
   activation threshold;
2. for each triggered gene, `_express()` transcribes + translates the DNA,
   crediting `proteins`/`mrna` and debiting ATP
   (`transcription_atp_per_nt`, `translation_atp_per_aa`);
3. solve the FBA biomass reaction (`FluxBalanceAnalysis` over
   `ECOLI_CORE_MODEL`) and credit energy `flux * biomass_to_atp`;
4. pay basal maintenance (`maintenance_atp_per_min`);
5. grow mass `mass += flux * 0.01`; divide at `energy >= division_energy`
   (energy halved, `divisions += 1`); die at `energy <= 0`.

History entries carry `age, energy, alive, divisions, mass, biomass_flux,
proteins, triggered`.

### 2.2 Physical unit anchors (`units.py`)

- Time: `1 tick = 1 min` (Neidhardt 1996).
- Space: lattice site edge `10 µm`; a 100×100 grid = 1 mm biofilm patch.
- Energy: **ATP molecule counts**; newborn cell ≈ 10⁹ ATP; maintenance
  ≈ 2.5×10⁷ ATP/min (Orth 2010; Alberts).
- Concentration: µM (Xavier & Bassler 2003, AI-2 threshold ≈ 10 µM).
- Conversion helpers: `diffusion_to_lattice`, `decay_from_half_life_ticks`.

### 2.3 Central dogma (`central_dogma.py`)

Real, cited rates: transcription elongation 50 nt/s (Proshkin 2010),
translation 20 aa/s (Ingolia 2009), mRNA half-life median 5 min
(Bernstein 2002), tRNA abundance tables (Dong 1996).

### 2.4 Metabolism (`metabolism.py`)

- Static FBA: `Reaction`, `MetabolicModel`, `ECOLI_CORE_MODEL`
  (37-reaction E. coli core), `FluxBalanceAnalysis.solve()`.
- Dynamic FBA: `DynamicFluxBalance` + `DynamicFBAConfig`
  (Mahadevan 2002 static-optimization dFBA: Michaelis–Menten glucose-uptake
  bound, forward-Euler batch ODEs, acetate/CO₂ by-products, diauxic phase 1),
  couplable to the `Environment` via `update_from_environment` /
  `apply_to_environment`.

### 2.5 Calibration and essentiality (delivered)

- `fit_parameters` (`virtual_cell.py:225`) supports inverse-variance weighted
  multi-observation fitting; `apps/omics_calibration.py` recovers perturbed
  coupling parameters to <1% error on noisy CRISPRi-PerturbSeq-style data
  (Virtual Cell Challenge 2025 protocol).
- `apps/whole_cell_scale.py`: `build_whole_cell`, `ko_model`,
  `predict_essentiality`, `essentiality_screen`, `WholeCellBenchmark`.
  19 core metabolic genes match the EcoCyc glucose-minimal-media essentiality
  reference 100% on the faithfully representable subset (Feist 2007;
  deviations documented).

### 2.6 Population scale (`population.py`)

`CellPopulation` (2D lattice) and `CellPopulation3D` (numpy-vectorized,
`_diffuse` keeps arrays across substeps): ~10⁵ cells at ≈ 0.9 s/tick,
near-linear scaling 2504 → 24480 → 114965 cells/s (iDynoMiCS 2.0 / NUFEB
scale class).

---

## 3. Gap analysis: what "complete realism" still lacks

| # | Gap | Consequence today | Literature anchor |
|---|-----|-------------------|-------------------|
| G1 | No cell-cycle phasing, no scheduled chromosome replication | all genes exist in single copy for the whole life; no gene-dosage wave | Cooper & Helmstetter 1968 (C/D periods); Karr 2012 |
| G2 | `mass` is dimensionless; no volume, no surface/volume scaling | no size-control law, no adder, no correct uptake scaling | Taheri-Araghi 2015; Jun 2018 |
| G3 | No protein folding / QC / turnover | every transcript instantly yields functional protein | Balchin 2016; Mosteller 1980 (half-life anchor already in `units.py`) |
| G4 | FBA is steady-state; no enzyme capacity, no intracellular pools | fluxes cannot be limited by protein abundance; no pool dynamics / overflow | Beg 2007 (MOMENT); Sánchez 2017 (GECKO); O'Brien 2013 (ME-model) |
| G5 | Calibration is omics-only; no standardized whole-cell benchmark | no reproducible "predict phenotype from genotype" score | Virtual Cell Challenge 2025 |
| G6 | Per-cell metabolism is uncoupled from the spatial environment | no metabolic heterogeneity in biofilms | Cockx 2024 (iDynoMiCS 2.0); COSMIC-dFBA 2024 |

The rest of this document is a phased design to close G1–G6.

---

## 4. Phase 1 — Cell-cycle phase and chromosome replication timing

### 4.1 Objective (closes G1)

Introduce a temporal cell-cycle state machine and scheduled chromosome
replication so that DNA copy number (gene dosage) rises during the
replication period and doubles just before division — the minimal
Cooper–Helmstetter pattern every whole-cell model must reproduce.

### 4.2 Literature

- **Cooper & Helmstetter 1968**: for a doubling time τ, chromosome
  replication takes a C period and division follows D minutes after
  termination; with growth faster than the C period, replication overlaps
  (multifork). τ = 20 min rich medium → C ≈ 40 min, D ≈ 20 min are the
  canonical E. coli values.
- **Karr 2012**: the M. genitalium whole-cell model schedules replication
  of each chromosome and lets transcript/protein abundance scale with copy
  number — the correct integration pattern for our layer model.

### 4.3 Design

New module-level `CellCyclePhase` (``B_GAP, C_PERIOD, D_PERIOD, DIVISION``)
and per-cell state in `VirtualCell`:

```
VirtualCell.phase            : CellCyclePhase      # current phase
VirtualCell.phase_progress   : float               # minutes into the phase
VirtualCell.dna_copy_number  : dict[str, int]      # per-gene copy number
VirtualCell.replication_fork : float               # 0..1 completion, or per-origin
VirtualCellConfig.replication_mode : "flat" | "cooper_helmstetter"
```

Semantics (mode `"cooper_helmstetter"`):

- On division the cell re-enters `B_GAP`; the origin fires when
  `age >= C + D - (τ - gap)` per Cooper–Helmstetter timing, i.e. replication
  must terminate ≥ D minutes before the next division.
- As the fork progresses, `dna_copy_number[gene]` steps 1 → 2 (→ 4 with
  multifork) according to each gene's coordinate on the chromosome map.
- `_express()` in `virtual_cell.py` scales transcription by the current
  copy number: `dna[gene] * copy` before elongation cost, giving a
  **gene-dosage wave** whose peak trails the fork.

Landing module: `virtual_cell.py` (state + scheduling), `grn.py` (unchanged),
`central_dogma.transcribe` (new optional `copy_number` argument).

### 4.4 Verification

- Invariant test: replication always terminates ≥ `D` minutes before
  division; the copy-number trajectory is monotone non-decreasing between
  divisions.
- Dosage test: an origin-proximal gene's `mrna`/`proteins` peak measurably
  above a terminus-proximal gene in rich medium.
- Compatibility test: `replication_mode="flat"` (default) reproduces the
  current single-copy behaviour bit-for-bit on the existing suite.

---

## 5. Phase 2 — Volume growth and size control (adder)

### 5.1 Objective (closes G2)

Give the cell a **physical volume** in µm³, grow it from biomass flux, and
implement the **adder size-control law** so population-level size
distributions and birth-size/additions match experiment.

### 5.2 Literature

- **Taheri-Araghi 2015**: E. coli follows the *adder* — division is triggered
  when the cell adds a constant volume Δ since birth, independent of birth
  size; this alone reproduces the observed narrow size distributions and
  weak size-homeostasis slope.
- **Jun 2018**: review of size control ("sizer vs timer vs adder");
  the adder is the correct minimal model for rod-shaped bacteria.

### 5.3 Design

- Replace the dimensionless `mass` growth with a volume model in `units.py`
  units: `volume_um3` from biomass flux via a dry-mass density ρ
  (E. coli ≈ 0.28 pg/µm³ wet, ≈0.15 pg/µm³ dry; Anchored in a new
  `UNITS_CELL_DENSITY` constant).
- Surface-to-volume scaling: uptake bounds scale with
  `surface_um2 ~ volume_um3^(2/3)` (rod/sphere geometry), so
  `_metabolism()` becomes `flux(S/V)` instead of constant flux.
- Division rule: divide when `volume_um3 - volume_birth_um3 >= adder_volume_um3`
  (`adder` mode), replacing the energy-threshold-only rule; keep the
  energy-threshold rule available as `division_rule="energy"` (default, today)
  and `division_rule="adder"`.
- `population.py`: `CellPopulation3D` cells carry `volume_um3`; lattice
  occupancy checks use the real radius from volume.

Landing module: `virtual_cell.py` (volume state + adder), `units.py` (density
constants), `metabolism.py` (S/V-aware uptake), `population.py` (per-cell
volume).

### 5.4 Verification

- Adder test: simulate a cohort in `adder` mode; fit
  `birth_size ~ added_size` linear regression → slope ≈ 0
  (Taheri-Araghi 2015 reports ≈ −0.1…0); report the slope in history.
- Scaling test: doubling time is invariant to initial size (size homeostasis).
- Volume conservation: at division, `volume_um3` halves with the energy
  budget; total population volume stays smooth across divisions.
- Population distribution: CV of the newborn size distribution
  (literature ≈ 0.1) matched to within a tolerance.

---

## 6. Phase 3 — Protein maturation, folding, quality control, turnover

### 6.1 Objective (closes G3)

Stop treating every transcript as instantly functional protein. Model
co-translational folding, chaperone-assisted maturation, misfolding/
aggregation, and degradation so that **functional protein abundance lags and
damps expression**, with the correct half-life and an explicit ATP cost for QC.

### 6.2 Literature

- **Balchin 2016**: chaperone systems (GroEL/GroES, trigger factor, DnaK/DnaJ)
  drive ~30–40% of cytosolic E. coli proteins to fold; misfolding competes
  with aggregation and degradation.
- **units.py already anchors** `PROTEIN_HALF_LIFE_MEDIAN_TICKS = 110`
  (Mosteller 1980; Helbig 2011) — the turnover anchor to use.

### 6.3 Design

New `ProteinPool` state in `central_dogma.py` (per gene):

```
ProteinPool(unfolded, folded, misfolded, degraded, decay_per_tick)
```

- On translation, a fraction `frac_cotranslational_fold` folds immediately;
  the rest enter `unfolded` and fold with first-order rate `k_fold` at an
  ATP cost (GroEL cycle ≈ 1 ATP per subunit-turnover step, order ~10¹–10²
  ATP per protein — configurable `folding_atp_per_protein`).
- Misfolding competes with folding (`k_misfold`); misfolded protein is
  degraded (Lon/Clp) or, under stress, forms aggregates that are
  `aggregated` (inert) rather than removed.
- `_express()` and the GRN trigger count only the **folded** pool; history
  adds `proteins_unfolded`, `proteins_misfolded`, `proteins_degraded`.
- New `VirtualCellConfig` flags: `protein_maturation_mode =
  "instant" | "chaperone"` (default `"instant"` preserves today's behaviour).

Landing module: `central_dogma.py` (ProteinPool, maturation model), `units.py`
(ATP/QC constants), `virtual_cell.py` (folded-only coupling).

### 6.4 Verification

- Steady-state test: fraction folded converges to the
  `k_fold/(k_fold+k_misfold)` equilibrium; degradation removes protein with
  the configured half-life.
- Dosage interplay test: in Phase-1 mode, the folded-pool peak lags the
  mRNA peak (translation + folding delay).
- Stress test: raising `k_misfold` (heat-shock-like) raises `aggregated`
  fraction and cost; total ATP expenditure stays within budget.
- Compatibility: `"instant"` mode byte-identical to the current suite.

---

## 7. Phase 4 — Enzyme-constrained metabolism and intracellular pools

### 7.1 Objective (closes G4)

Make metabolism respond to the proteome: FBA fluxes become bounded by the
abundance of the enzymes that carry them, and intracellular metabolite pools
integrate over time instead of being a steady-state snapshot.

### 7.2 Literature

- **Beg 2007 (MOMENT)**: enzyme-constrained FBA adds capacity constraints
  `Σ kcat·E ≤ total protein`; captures flux redirection and the "metabolic
  burden" of enzyme expression.
- **Sánchez 2017 (GECKO)**: genome-scale enzyme-constrained model of
  *S. cerevisiae*; predicts overflow metabolism from proteome allocation.
- **O'Brien 2013 (ME-models)**: coupled metabolic + expression model; the
  direct precedent for coupling `proteins` (our Phase-3 pool) to flux.

### 7.3 Design

- Extend `FluxBalanceAnalysis` with optional enzyme-capacity rows:
  `FluxBalanceAnalysis(model, enzyme_capacity=...)` where each reaction gets
  `v_i ≤ kcat_i · E_i` and `E_i` is read from the cell's folded `ProteinPool`
  (Phase 3), mapped gene → reaction via the existing
  `ECOLI_CORE_GENE_REACTIONS` table used by `ko_model`.
- Add `MetabolitePool` (in `metabolism.py`): per-metabolite ODE integration
  `d[P]/dt = Σ v_in − Σ v_out − dilution·[P]` with forward Euler, sharing the
  `DynamicFBAConfig` style (`dt_h`, `min_biomass`).
- Rewire `VirtualCell._metabolism()`:
  1. protein-limited bounds from `folded` pools → solve FBA → fluxes;
  2. integrate pools for one tick; 3. pool feedback (e.g. substrate inhibition,
     allosteric) can clamp exchange bounds — the same hook
   `DynamicFluxBalance.bound_override` already provides.
- Overflow: when respiratory capacity is saturated by the enzyme constraint,
  the FBA naturally redirects to acetate overflow — make it a testable
  prediction (diauxie phase 1 at the cell level, not only the batch level).

Landing module: `metabolism.py` (`MetabolitePool`, enzyme-capacity rows),
`apps/whole_cell_scale.py` (gene→reaction reuse), `virtual_cell.py`.

### 7.4 Verification

- Enzyme-limitation test: overexpressing a bottleneck enzyme raises its
  reaction flux; knocking it out collapses it (consistent with
  `predict_essentiality`).
- Overflow test: at high glucose the constrained model secretes acetate
  while the unconstrained one does not (Sánchez 2017 prediction).
- Pool dynamics test: internal pools reach steady state; dilution matches
  the growth rate; pools feed back into bounds within one tick.
- Backward-compat: with no `enzyme_capacity` and no pools, `solve()` results
  are unchanged.

---

## 8. Phase 5 — Multicellular coupling and real-data calibration closure

### 8.1 Objective (closes G5, G6)

Bind Phases 1–4 into the population scale and close the loop with data:
per-cell dFBA in a shared environment, standardized benchmarks, and a
calibration pipeline that can ingest experimental observables.

### 8.2 Literature

- **Cockx 2024 (iDynoMiCS 2.0)**: per-agent metabolic state in a shared
  environment field; the composition pattern to follow.
- **COSMIC-dFBA 2024**: dFBA applied per agent in biofilm geometry.
- **Virtual Cell Challenge 2025**: the field explicitly demands shared
  benchmarks + calibratability ("Turing test for the virtual cell").

### 8.3 Design

- **Per-cell dFBA in `CellPopulation3D`**: each cell owns a
  `DynamicFluxBalance`; `Environment` fields (glucose, acetate, O₂) are read
  via `update_from_environment` and written back via
  `apply_to_environment` per tick — the coupling pair already exists.
- **Population observables**: doubling-time distribution, birth-size
  distribution (Phase 2 adder), volume-weighted growth rate, colony radial
  growth rate (BM3-style density profile).
- **Calibration closure**: extend `apps/omics_calibration.py` into
  `apps/whole_cell_calibration.py` — a single entry point that fits
  Phases 1–4 parameters (`adder_volume`, `k_fold`, `kcat` scaling,
  `maintenance_atp_per_min`) against mixed observables (growth curves, size
  distributions, essentiality, protein abundances) using the existing
  inverse-variance-weighted `fit_parameters`.
- **Benchmark harness**: `apps/virtual_cell_bench.py` grows into a
  `whole_cell_benchmark` protocol runner: ① essentiality accuracy
  (already ≥ 0.95), ② doubling-time fidelity on the Mahadevan batch curve,
  ③ adder-slope ≈ 0, ④ BM3-style density profile vs iDynoMiCS 2.0.

Landing module: `apps/whole_cell_calibration.py`, `population.py`,
`environment.py`, `apps/virtual_cell_bench.py`.

### 8.4 Verification

- Batch-vs-steady test: the population-level dFBA reproduces the
  Mahadevan 2002 diauxie reference curve at the colony edge.
- Heterogeneity test: cells near the colony centre run at lower glucose and
  slower growth than the edge (metabolic stratification).
- End-to-end test: a single `run_whole_cell_benchmark()` call returns all
  four scores and `passed` only when each gate holds.

---

## 9. Sequencing and exit criteria per phase

| Phase | Primary module(s) | Entry test | Exit gate |
|-------|-------------------|------------|-----------|
| 1 | `virtual_cell.py` | replication-termination invariant | dosage wave + flat-mode bit-compat |
| 2 | `virtual_cell.py`, `units.py`, `population.py` | adder slope ≈ 0 | size-distribution CV + S/V scaling |
| 3 | `central_dogma.py` | folded-pool equilibrium | half-life match + instant-mode bit-compat |
| 4 | `metabolism.py` | enzyme-limited flux | overflow prediction + solve() bit-compat |
| 5 | `apps/*`, `population.py`, `environment.py` | batch diauxie match | 4-gate whole-cell benchmark |

Gates repeat the project's established pattern: quantitative, literature
anchored, with a compatibility path (`mode="..."` flags) so every phase is
mergable without breaking the current baseline.

---

## 10. Design principles (applied to every phase)

1. **Physical units from `units.py`** — no new dimensionless quantities;
   new constants land there with citations (as today: Proshkin 2010,
   Ingolia 2009, Bernstein 2002, Dong 1996, Neidhardt 1996, Orth 2010).
2. **API continuity** — new behaviour behind flags/enum modes whose default
   reproduces today's results; no signature breakage.
3. **Literature anchors in code** — every new constant is documented with its
   source in the docstring, mirroring `central_dogma.py`/`units.py`.
4. **Vectorization budget** — per-cell state stays numpy-friendly
   (`CellPopulation3D` pattern); the 10⁵-cell/tick budget
   (≈ 0.9 s/tick) is a hard constraint for Phases 2 and 5.
5. **Data closure** — every phase ships a fitting hook (`fit_parameters`
   weights API) and a benchmark score so realism is always measured, not
   asserted.

---

## 11. References

1. Karr, J. R. et al. *A whole-cell computational model predicts phenotype
   from genotype.* Cell 150:389–401 (2012). DOI:10.1016/j.cell.2012.05.044
2. Cooper, S. & Helmstetter, C. E. *Chromosome replication and the division
   cycle of Escherichia coli B/r.* J Mol Biol 31:519–540 (1968).
3. Taheri-Araghi, S. et al. *Cell-size control and homeostasis in bacteria.*
   Curr Biol 25:385–391 (2015). DOI:10.1016/j.cub.2014.12.009
4. Jun, S., Si, F., Pugatch, R. & Scott, M. *Fundamental principles in
   bacterial physiology—history, recent progress and the future with focus
   on cell size control.* Rep Prog Phys 81:056601 (2018).
5. Balchin, D., Hayer-Hartl, M. & Hartl, F. U. *In vivo aspects of protein
   folding and quality control.* Science 353:aac4354 (2016).
6. Beg, Q. K. et al. *Intracellular crowding defines the mode and sequence
   of substrate uptake by Escherichia coli and constrains its metabolic
   activity.* PNAS 104:12663–12668 (2007).
7. Sánchez, B. J. et al. *Improving the phenotype predictions of a yeast
   genome-scale metabolic model by incorporating enzymatic constraints.*
   Mol Syst Biol 13:935 (2017).
8. O'Brien, E. J., Lerman, J. A., Chang, R. L., Hyduke, D. R. & Palsson, B. Ø.
   *Genome-scale models of metabolism and gene expression extend and refine
   growth phenotype prediction.* Mol Syst Biol 9:693 (2013).
9. Mahadevan, R., Edwards, J. S. & Palsson, B. Ø. *Dynamic flux balance
   analysis of diauxic growth in E. coli.* Biophys J 83:1331–1340 (2002).
10. Cockx, B. et al. *iDynoMiCS 2.0.* PLoS Comput Biol 20(2):e1011303 (2024).
11. *Virtual Cell Challenge: toward a Turing test for the virtual cell.*
    Cell (2025). DOI:10.1016/j.cell.2025.06.021
12. Proshkin, S., Rahmouni, A. R., Mironov, A. & Nudler, E. Science
    328:504–508 (2010); Ingolia, N. T. et al. Science 324:218–223 (2009);
    Bernstein, J. A. et al. J Bacteriol 184:6477 (2002);
    Dong, H. et al. J Mol Biol 260:649 (1996);
    Neidhardt, F. C. et al. *Escherichia coli and Salmonella* (1996);
    Orth, J. D. et al. Mol Syst Biol 6:369 (2010).
    (Already cited in `central_dogma.py`/`units.py`.)

---

## 12. Implementation status (delivered)

Each phase below lists what actually shipped. Compatibility note: every
design flag keeps its `"flat"`/`"instant"`/`"energy"` default (or `False`),
so running the pre-Phase-1 configuration reproduces the original behaviour
bit-for-bit — verified by the "bit-compat" tests listed per phase.

### 12.1 Phase 1 — cell cycle and chromosome replication (G1)

- `VirtualCellConfig.replication_mode: "flat" | "cooper_helmstetter"`
  (`virtual_cell.py:273`); `VirtualCell.phase`, `dna_copy_number`,
  `replication_forks`, `replication_fork` property, `_advance_replication`
  (scheduled origin firing, C/D timing), `_divide_replication` (fork
  propagation + multifork via `MAX_DNA_COPY_NUMBER`).
- Gene-dosage coupling: `_express()` scales transcription by
  `dna_copy_number[gene]`; `central_dogma.transcribe` accepts
  `copy_number`.
- Tests: replication-termination invariant, dosage wave (origin-proximal
  vs terminus-proximal peaks), `"flat"` bit-compat on the whole suite.

### 12.2 Phase 2 — volume growth and size control (G2)

- Physical volume: `volume_init_um3` (= newborn 1.6 µm³),
  `volume_birth_um3`, dry-mass density
  `cell_density_dry_pg_um3` (`UNITS_CELL_DENSITY_DRY_PG_UM3`);
  growth `volume_um3 += flux·biomass_to_volume_pg_per_min / ρ` per minute.
- Adder: `division_rule="energy" | "adder"`; `adder_volume_um3` = 1.6 µm³;
  `adder_noise_std` draws one Gaussian threshold per generation
  (`_draw_adder_threshold`, Taheri-Araghi 2015 σ≈0.1–0.2); history records
  `added_volume_um3`.
- `surface_scaling` (uptake ∝ `volume^(2/3)`), `cell_radius_um` + real-size
  occupancy in `population.py`.
- Bench gate: adder slope ≈ 0.16 (reference ≤ 0.2; Taheri-Araghi ≈ −0.1…0).

### 12.3 Phase 3 — protein maturation and QC (G3)

- `ProteinPool` (unfolded/folded/misfolded/degraded/aggregated) in
  `central_dogma.py:283`; `protein_maturation_mode: "instant" | "chaperone"`
  (`"instant"` default = bit-compat), chaperone mode with
  `folding_atp_per_protein`, `misfold_rate_per_min`, `fold_rate_per_min`,
  `aggregation_rate_per_min`, `degraded_rate_per_min`;
  `_fold_rate_from_k_fold` maps the folding equilibrium `k_fold`.
- History adds `proteins_unfolded/misfolded/degraded/aggregated`.

### 12.4 Phase 4 — enzyme-constrained metabolism and pools (G4)

- `enzyme_capacity_enabled` + `enzyme_scale` (kcat scale): per-reaction
  capacity `v_i ≤ kcat·E_i`, enzyme levels from folded pools via the
  `ECOLI_CORE_GENE_REACTIONS` table (reused from `ko_model`).
- `MetabolitePool` / `MetabolitePoolConfig` (`metabolism.py:947`): per-
  metabolite ODE integration with dilution; overflow (acetate) emerges when
  respiration is enzyme-limited.
- Calibration anchor: truth `enzyme_scale = 1e4`.

### 12.5 Phase 5 — population coupling and calibration closure (G5, G6)

- **Per-cell dFBA in `population.py`**: `PopulationCell.dfba`,
  `_step_dfba_metabolism` — deep-copied core model, respiration capped by
  O₂ (`NADH_OX`/`FADH2_OX` ≤ 2·v_o2 Monod), energy credited at
  `growth_rate·dt·dfba_energy_scale`; environment coupling through
  `update_from_environment` / `apply_to_environment`.
- **Colony observables**: `colony_observables` (doubling-time distribution,
  birth-size distribution, volume-weighted growth, colony radial growth,
  density profile) and `dfba_stratification` (centre-vs-edge metabolic
  heterogeneity).
- **Calibration closure** — `apps/whole_cell_calibration.py`:
  `WholeCellCalibration` fits the four hidden parameters (adder, k_fold,
  enzyme_scale, maintenance) from a mixed observable vector (energy/volume/
  biomass curves + protein abundances + per-division added volumes +
  division count). Joint 4-D fitting stalls in correlated valleys, so the
  fit runs as two nearly separable 2-D stages (growth stage, then
  size/folding stage). `adder_noise_std > 0` is handled by population
  averaging (`n_cells`) over independent cells, verified down to σ = 0.1.
- **Benchmark harness** — `apps/virtual_cell_bench.py::run_whole_cell_benchmark`
  (one call, all four gates): ① essentiality 19/19 (≥ 0.95),
  ② batch doubling-time fidelity (0.547 h vs 0.5 h reference, rel. 0.094),
  ③ adder slope (0.163 ≤ 0.2), ④ BM3-style radial density profile
  (inner 1.23 / outer 0.63 ≥ 0.5).
- **Performance**: `central_dogma._find_rho_independent_terminator` is
  memoized (`@lru_cache(maxsize=4096)`); the 28 shared enzyme-gene
  sequences make this the hot path of repeated transcription (5000 cached
  calls ≈ 0.1 ms vs 330 ms uncached).

### 12.6 Gate history

| Gate | Result |
|------|--------|
| Phase 4 (Phases 1–4 complete) | Tests pass, mypy + ruff clean |
| Phase 5 (full suite, this doc) | Tests pass, coverage ≥80%, EXIT 0 (mypy + ruff clean) |
