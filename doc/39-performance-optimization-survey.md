# doc/39 — Performance Optimization Survey: Realism-Preserving Speedups Across the Stack

> **Status:** implemented (O1–O12) · 2026-09-01 · baseline 2026.8.5 · O11 (JAX BatchRuntime) guarded by `has_jax` extra — **live-verified with jax 0.4.38** (JAX engine selected, bit-exact parity; numpy fallback also bit-correct); O12 (Rust/PyO3 simplex) landed — **9.9–65.8×** over pure-Python (measured)
>
> **Depends on:** doc/13 (performance report), doc/37 (validity/performance decoupling), doc/36 (plugin architecture), doc/31 (frontier design)
>
> **Goal:** Inventory every performance-critical path in the project and rank optimization
> opportunities that preserve — and where possible increase — simulation realism, across
> the full stack (pure Python → numpy → Cython/numba → optional JAX/GPU → serving).
> Companion to doc/40, which targets immune-model realism; the two plans share the same
> hot-loop budget and must land in lockstep.

---

## 1 — Executive Summary

doc/13 established a strong baseline for the **Helix language / CellVM / GRN** path:

| Metric | doc/13 measured value |
|---|---|
| Compile speed | ~470–710k codons/s, linear in gene count (~3 ms regardless of N) |
| VM execution | ~1.3 M ops/s (~770 ns/op), ~20k ticks/s |
| GRN update | O(N+E): 128-node cycle ~30k steps/s; dense 4096-edge ~6k steps/s (≈25× vs baseline) |
| Memory | ~0.33 MiB peak for 16-gene × 200-tick compile+run |
| Example wall-time | 04_turing_pattern 110 ms → ~12.6 ms; total examples ~16 ms |

The **human-simulation plugin** (`helixlang/plugins/human/`) and the **runtime service layer**
(`helixlang/sim_runtime/`, `helixlang/_accel/`) are the current hotspots. They mix:
forward-Euler PBPK sub-stepping, per-patient two-phase simplex LP solves, per-drug
per-step `solve_ivp` ODE solves, and a primarily pure-Python numerical layer. These are
exactly the components whose *realism* doc/37, doc/31 and doc/40 want to raise, so the
optimization here is deliberately framed as "math-equivalent or math-better", never
"math-worse".

Validation gate: **85/85 validation goldens, SHA256-verified** (`validation/report.md`,
README §Validation). Every optimization below is gated on either bit-identical golden
outputs or a documented regeneration path through the doc/37 Biological-Accuracy framework.

---

## 2 — Baseline & Hotspot Inventory

### 2.1 Virtual Patient PBPK loop (dominant cost in human runs)

- `virtual_patient.py:2228-2230` — `_EULER_SAFETY = 0.4`, `_MAX_SUBSTEPS = 2000` clamp.
- `virtual_patient.py:2400` `_euler_step`; `virtual_patient.py:2470-2471` — per-tick
  `max_substep_h = _EULER_SAFETY / self._max_rate_constant(...)`, then
  `n_substeps = min(2000, ceil(dt_h / max_substep_h))`.
- Consequences: (1) reaction rates are recomputed inside `_max_rate_constant` on every
  tick even when concentrations are flat; (2) a fixed Euler clamp enforces tiny substeps
  for stiff absorption/elimination regardless of whether the local Jacobian needs them;
  (3) the 2000-substep ceiling is a silent accuracy cap (stiffest cases take fewer
  substeps than the stability criterion demands).

**Realism note:** the clamp is a *stability* measure, not a *biology* measure. An adaptive
solver (same equations, local-error control) is strictly more realistic: it removes the
ceiling-induced truncation error without changing the model.

### 2.2 Metabolism LP (simplex) solver

- `plugins/runtime/metabolism.py:414` `_simplex_max` (pure Python); `:495`
  `_simplex_max_numpy` (NumPy-vectorized, Bland's rule, identical algorithm — see docstring
  `:502-510`); dispatch at `:786-830` / `:873-878`.
- The numpy path exists and is algorithm-identical; it is a default-selection and
  coverage question, not a rewrite question.

### 2.3 GRN per-gene activation

- `plugins/runtime/grn.py:262` `step` (sigmoid/Hill, optional telegraph noise) vs `:297`
  `step_accel` (threshold-only subset that loads `_accel`, `prefer=` native>numpy>python).
- Hill-coefficient and telegraph paths still run the interpreted Python branch.

### 2.4 Legacy human-simulation ODE path

- `plugins/human/simulation.py:363-371` — `scipy.integrate.solve_ivp` (RK45,
  `rtol=1e-6`, `atol=1e-9`) invoked **per drug per time step**, re-importing/chasing
  `scipy` at line 53-57. Pure-Python Euler fallback otherwise.

### 2.5 `_accel` pure-Python fallbacks

- `_accel/` provides `grn_step`, `simplex`, `diffusion` with native → numpy → python
  resolution (`HAS_NATIVE` census in the package `__init__` / `home`). Fallbacks exist so a
  pure-wheel install stays fully functional; they are functionally correct, not fast.

### 2.6 Service / serving layer

- `server/app.py` — single-process, no request-level cache; every `compute(...)` call
  re-runs lex → parse → semantic → compile → execute with no memoization. Whole-cell and
  virtual-patient runs are the same hot path as the CLI.

### 2.7 Human-plugin downstream (doc/40 co-tenant)

- `plugins/human/immune.py:75` `CytokinePool.step` and `:138` `ImmuneCellPopulation.step`
  — forward Euler, rate constants recomputed each call
  (`k = ln2/half_life` on lines 77-80, loop-invariant). `CRPDriver.step` at `:298`.
- `virtual_patient.py:1321` calls `self._immune.step(dt_h)`; `:1348` `_crp_driver.step(...)`.

---

## 3 — Optimization Catalog

Key to **realism impact**: ⚪ math-equivalent (bit-identical or numerically identical → goldens
must not change); 🟩 math-better (same equations, tighter/adaptive integration, more
faithful numerics → plausibly *raises* fidelity; goldens regenerated via doc/37 and
re-validated); 🟪 realism-better (same equations but more scenarios/variability reachable →
outright new realism, e.g. virtual populations).

**Imp.** column: ✅ merged · 🟨 partial · ⬜ not started (as of 2026-08-31, release 2026.8.5).

| # | Hotspot | Change | Expected gain | Realism | Effort | Risk | Imp. |
|---|---------|--------|---------------|---------|--------|------|------|
| O1 | CytokinePool/cell ODE `immune.py:75-99,138-176` | Hoist rate constants out of `step`; batch with numpy internally | 2–6× | ⚪ | S | L | ✅ |
| O2 | Same hot loops, cohort level | Vectorize across N virtual patients simultaneously (numpy arrays over patients) | 5–20× for n=100 pop | 🟪 (enables virtual-population variability) | M | M | ✅ `cohort_immune_step` bit-identical + `run_cohort` multiprocessing runner (O9) |
| O3 | `virtual_patient.py:2400-2471` Euler clamp | Replace `_EULER_SAFETY` clamp w/ adaptive stepping (local-error-controlled `solve_ivp`/`LowLevelCallable` patient-level); keep `_MAX_SUBSTEPS` as hard safety only | 3–10× on stiff runs + removes truncation | 🟩 | M | M | ✅ recursive step-doubling `_integrate_slot` (`_RTOL`/`_ATOL`/`_MIN_STEP_H`) |
| O4 | `simulation.py:363-371` per-drug solve_ivp | Batch all drugs into one system; use `vectorized=True` + pre-imported scipy object | 2–5× | 🟩 (consistent cross-drug coupling) | M | L | ✅ `_PBPKEngine.advance_batch` (one block-diagonal `solve_ivp(vectorized=True)`); verified vs per-engine ≈1e-5 |
| O5 | `metabolism.py` simplex dispatch | Default to `_simplex_max_numpy` when numpy present; Cython the core pivots; pure-python retained as guaranteed fallback | 3–20× | ⚪ | S–M | L | ✅ already defaults to numpy simplex; pure-python fallback kept |
| O6 | `grn.py:262` Hill/telegraph Python branch | Add `_accel` Hill path (extend `step_accel` beyond threshold-only); keep telegraph as optional sim feature | 3–8× | ⚪ | S–M | L | ✅ `_accel/grn_step` `step_mixed` (python/numpy/numba) + `grn_step_mixed`; verified vs `GRN.step` |
| O7 | `_accel` fallbacks (grn_step/simplex/diffusion) | Widen native coverage; add numba-jit experimental path | 5–30× on pure-wheel installs | ⚪ | M | M | ✅ numba + numpy for all three hot paths, fidelity bit-identical: simplex **23.1×**, diffusion **5.1×** vs pure-python; grn_step single-tick numba/numpy ≈ **parity** (real GRN win is cohort vectorization, O2/O10) — measured 2026-09-01 |
| O8 | `server/app.py` | LRU memoization keyed on (source, seed, config-hash/genetics); process the memoized compile only | 10–100× on repeated cohort sweeps | 🟪 (cheap sensitivity sweeps) | S | L | ✅ `_PIPELINE_CACHE` (64 MiB) via `_pipeline_cached`/`_pipeline_fresh` |
| O9 | cohort/validation runs | `joblib`/`multiprocessing` map across patients & goldens (embarrassingly parallel) | N× practical | 🟪 (larger ensembles) | S | L | ✅ `immune.run_cohort` (spawn-pool slabs, bit-identical) + `validation/run_all.py --parallel` |
| O10 | adaptive-ODE backbone for doc/40 | When doc/40 lands Friberg/IFN/complement, author them as a vectorized scipy system, not per-hour Euler | headroom for +tens of ODEs | 🟩 | M | M | ✅ vectorized cohort steps for adaptive (`cohort_adaptive_step`, + PD-1 checkpoint), complement (`cohort_complement_step`), tissue/blood (`cohort_tissue_blood_step`) — each bit-identical to its scalar path (doc/39 §5.1) |
| O11 | (optional) JAX BatchRuntime | doc/37 §5 batch runtime across patients/GPU; guarded by `has_jax` extra | 10–100× cohort | 🟪 | L | H | ✅ `BatchRuntime` `_JAXEngine` + `_make_engine` jax→numpy transparent fallback (device availability is a *performance*, never *fidelity*, property — doc/37 §3); `has_jax` extra added (`jax>=0.4,<1`). **Live-verified 2026-09-01 with jax 0.4.38**: `_JAXEngine` selected (`active_backend=jax`) and traces bit-identical to the numpy engine and the single runtime; numpy-fallback guard (jax absent) also verified bit-correct. GPU/CUDA and cross-device 10–100× gains not measured (CPU-only host) |
| O12 | (optional) Rust/PyO3 | Port hottest kernels after profiling shows steady-state winners | 10–50× | ⚪ | L | H | ✅ Rust/PyO3 simplex `impl_rust` (`_accel`, abi3), landed 2026-09-01 — **9.9×/29×/65.8×** over pure-Python (20/40/60-size LPs, measured), also beats numba (1.4–1.8×); **bit-identical** to `impl_python` (status+basis+full tableau, max diff 0.0). Loader now prefers it under `native`/`HELIX_ACCEL=rust`; built via `python -m helixlang._accel.build` (cargo only, no maturin); source crate ships at `simplex/rust/` |

Priorities: **O3, O5, O6, O8** first (low-risk, standalone); **O1/O2/O10** co-scheduled
with doc/40 Phase A (they touch the same file); **O4, O7, O9** opportunistic; **O11/O12**
stretch, only if profiling justifies.

> **Implemented 2026-08-31 (release 2026.8.5):** O1 (CytokinePool hoisting), O3 (adaptive
> PBPK step-doubling), O5 (numpy simplex default, already present), O6 (mixed Hill/sigmoid
> `_accel` path), O8 (server LRU), O2 (cohort-vectorized `cohort_immune_step`, verified
> **bit-identical** to the scalar per-model path — a strict acceleration with zero behavior
> change), O4 (batched multi-drug `advance_batch`, verified vs per-engine ≈1e-5), and O9
> (`immune.run_cohort` multiprocessing slabs, bit-identical);
> `validation/run_all.py` gained `--parallel`. **O7 complete 2026-09-01**: numba + numpy
> coverage for all three `_accel` hot paths (`grn_step`/`simplex`/`diffusion`), each verified
> bit-identical to its `impl_python` reference. Measured on arm64 (numba 0.61.0), speedup vs
> pure-python: simplex **23.1×** (`run`, 40×40), diffusion **5.1×** (`step`, 64×64 field),
> grn_step single-tick ≈ **0.8–1.2× (parity — the per-tick op is not GRN's bottleneck; the
> cohort-level numpy vectorization O2/O10 is the real accelerator)**. **O10 complete 2026-08-31**:
> vectorized cohort backbones for the doc/40 modules — `adaptive.cohort_adaptive_step`
> (incl. PD-1 checkpoint), `complement.cohort_complement_step`,
> `tissue_blood.cohort_tissue_blood_step` — each verified
> **bit-identical** to its scalar path (`tests/test_adaptive_immunity.py`, `test_complement_g6.py`).
> Verification:
> `tests/test_perf_cohort.py` (O2 equivalence + O3 determinism + O9 runner),
> `tests/test_human_simulation.py` (O4 batch vs per-engine); green under ruff/mypy, the
> human-plugin test suite, and the boundary census.
>
> **O11 complete 2026-09-01**: `BatchRuntime`'s engine already had a `_JAXEngine` plus a
> `_make_engine` jax→numpy transparent fallback (doc/37 §3 — device availability is a
> *performance*, never *fidelity*, property); `pyproject.toml` gained the `has_jax` extra
> (`jax>=0.4,<1`). **Live-verified with jax 0.4.38** (isolated venv, CPU): with jax present
> `backend="jax"` selects the real JAX engine (`active_backend=jax`) and its traces are
> **bit-identical** to the numpy engine and to `IRRuntime.run`; with jax absent it falls
> back to numpy, still bit-identical (`tests/test_ir.py::test_backend_selection_and_fallback`).
>
> **O12 complete 2026-09-01**: per the directive, O12 landed *only because* measured Rust
> beat pure-Python. A Rust/PyO3 (`abi3`) port of the hottest steady-state winner — the
> simplex pivot (doc/39 O5/O7) — ships as `_accel/simplex/impl_rust.abi3.so`, wired into the
> `_accel` loader's native tier (`impl_rust`; also `HELIX_ACCEL=rust`), with the crate source
> at `_accel/simplex/rust/` and a cargo step in `python -m helixlang._accel.build`. Measured
> on arm64: simplex `run` is **9.9×/29.3×/65.8×** faster than pure-Python at 20/40/60-size
> LPs, and also beats numba (1.4–1.8×); it is **bit-identical** to `impl_python` (status +
> basis + every tableau cell, max diff 0.0). `tests/test_accel_foundation.py` now
> validates the Rust backend's native-resolution and bit-exactness. (Had Rust not been
> faster, O12 would have been marked *no performance improvement* instead of landed.)


---

## 4 — Cross-Stack Options (language/technology survey)

| Stack | Where it fits | Packaging | Realism neutral? |
|---|---|---|---|
| Pure Python (status quo) | correctness-first fallbacks, non-hot paths | universal wheel | yes (must keep) |
| numpy vectors | batch cohorts (O2,O10), simplex (O5), arrays of patient states | `fast` extra (already exists) | yes — bit-identical |
| Cython (in-repo `_accel`) | already shipping native dual-wheel; extend to immune/ODE/simplex | `HAS_NATIVE` dual wheel (doc/36) | yes — tested bit-identical in doc/13/37 |
| numba (opt-in) | jit hot kernels without rebuild; pure-wheel fallback keeps determinism | optional extra | yes, if float paths identical |
| scipy adaptive ODE | replace Euler clamps + per-drug solve_ivp (O3, O4, O10) | already a dependency | 🟩 math-better |
| JAX (opt-in) | BatchRuntime virtual-population sweeps, GPU | optional extra | 🟩 if seeded determinism preserved |
| Rust/C via PyO3/nanobind | last-resort hot kernels if profiling demands | native dual wheel | yes |

Constraint from doc/36/doc/37: native must never be required — pure-Python fallbacks stay
functional on pure installs; determinism policy (seeded, SHA256 goldens) governs which paths
may reshape arithmetic. Every algorithmic change that touches arithmetic goes through
`validation/report.md` regeneration + doc/37 Biological-Accuracy review instead of silent
golden rewrites.

---

## 5 — Realism-Preservation & Determinism Checklist

1. ⚪ changes must reproduce goldens bit-identically (`--exact`/goldens path), else they are
   not merged.
2. 🟩 math-better changes: re-run `validation/run_all.py` → document diff → clear ANY changed
   golden column through doc/37 accuracy review AND a spot-check that the changed value moved
   *toward* the published reference (e.g. LPS time-course, PK curves).
3. Seeding: `rng` usage is explicit in the plugin layer; cohort vectorization must not change
   per-patient seed streams (map seed → patient index deterministically, not RNG-global).
4. `HAS_NATIVE` census and `prefer=` plumbing stay: no native-only divergence in results.
5. Each phase lands with a profiling before/after entry appended to doc/13's harness (or its
   `bench/` port), not ad-hoc timings.

---

## 6 — Phased Plan

- **Phase 0 — Profile (0.5 wk).** cProfile + py-spy flamegraphs over: whole-cell example,
  1-patient virtual run, 100-patient cohort, 1 validation harness run. Publish the table of
  §2 hot-spots with measured shares. No behavior change.
- **Phase 1 — Low-risk code motion (1–1.5 wk).** O1 hoisting, O5 default numpy simplex,
  O6 `_accel` Hill path, O8 server LRU. Gate: 75/75 identical, examples timings unchanged or
  better, no API change. ✅ **Complete 2026-08-31** (O1 CytokinePool hoisting, O5 already
  numpy-default, O6 mixed Hill `step_mixed`, O8 server LRU; all gate-green).
- **Phase 2 — Adaptive & vectorized ODE (2–3 wk).** O3 `virtual_patient` adaptive stepping,
  O4 batched `simulation.py`, O2/O10 cohort-vectorized immune/cell ODEs **in lockstep with
  doc/40 Phase A**. Gate: spot-checked LPS/neutropenia trains move toward published
  references; deterministic across seeds; fallbacks active on pure wheels.
  ✅ **O3 complete** (recursive step-doubling `_integrate_slot`); **O2 kernel + runner complete**
  (`cohort_immune_step` bit-identical to scalar, numpy `+ fast`; `run_cohort` multiprocessing);
  **O4 complete** (`advance_batch`, verified ≈1e-5 vs per-engine); **O10 complete 2026-08-31**
  (vectorized `cohort_adaptive_step` incl. checkpoint, `cohort_complement_step`,
  `cohort_tissue_blood_step`, each bit-identical to its scalar path).
- **Phase 3 — Parallel ensembles (1 wk).** O9 cohort/validation multiprocessing; cohort
  defaults raised from n→n×k where k affordable. Gate: 75/75, cohort wall-time linear-ish
  speedup. ✅ **O9 complete** (`run_cohort` spawn-pool slabs, bit-identical; `run_all.py
  --parallel` for goldens; `run_cohort` verified == scalar in `test_perf_cohort.py`).
- **Phase 4 — Compiler-accelerated kernels (2–3 wk).** O7 native coverage expansion, optional
  numba path behind extra; stretch: O11 JAX BatchRuntime prototype, O12 profiling-gated PyO3.
  Gate: pure-wheel parity + native speed wins measured in doc/13 harness.
  ✅ **Complete 2026-09-01.** O7 (numba + numpy for grn_step/simplex/diffusion, each
  bit-identical — simplex **23.1×**, diffusion **5.1×**, grn single-tick ≈ parity),
  O11 (JAX `BatchRuntime` `_JAXEngine` + transparent jax→numpy fallback via `[has_jax]`
  extra, live-verified bit-identical with jax 0.4.38), and O12 (Rust/PyO3 `impl_rust`
  simplex, **9.9×/29.3×/65.8×** vs pure-Python and bit-identical, `_accel/simplex/rust/`).

**Total ~7–9 wk part-time; doc/40 Phase A–C interleaved at Phase 2 rendezvous points.**

---

## 7 — References

- doc/13 — performance-report (compile/VM/GRN/Gray-Scott numbers quoted in §1)
- doc/37 — biological-validity-performance-decoupling (validation 85/85, doc/37 §5 batch runtime)
- doc/36 — plugin-architecture (dual-wheel `HAS_NATIVE` packaging)
- doc/31 — frontier-virtual-patient-design (§2.4 immune ABM survey; §4.5 performance budget: "ABM adds minutes per run at tissue-agent counts × population n; acceptable for n≥100 if agents capped ≈10³–10⁴")
- doc/33 — 100-percent-completion (validation lineage)
- `validation/report.md` — 75/75 SHA256 goldens
- `doc/40-human-immune-realism.md` — the realism plan this doc budgets for