# doc/42 — Production Readiness: 100% Biologically Realistic Simulation via the Helix Language (Gap Assessment & Remediation Plan)

> **Status:** Open — assessment complete 2026-09-01 · remediation plan proposed, **not yet implemented**
>
> **Depends on:** doc/34 (architectural plan), doc/37 (validity framework), doc/39 (performance budget + O1–O12), doc/40 (human immune realism Phases A–H), doc/41 (validation-level taxonomy + extensible parser), doc/38 (compiler modernization + plugin platform), doc/04 (simulation model), doc/27-32 (virtual patient / disease / pharmacology)
>
> **Objective:** Deep audit of the entire codebase against the goal "**production-usable, 100%-realistic biological simulation authored in the Helix language, with production-usable performance**" — focusing on the human plugin while checking every other subsystem. This doc records (a) what is already production-grade, (b) the exact gaps that block the goal, ranked, and (c) a phased remediation design. It is intentionally a *gap register + plan*, not a build.

---

## 1 — Executive Summary

HelixLang is an unusually mature codebase on the **engineering/DSL-infrastructure** axis: a
real parse→AST→semantic→IR→bytecode→VM pipeline, differential-incremental recompile
(incr "JIT"), a plugin registry (`PluginProvider` + `GrammarDescriptor`), physical
dimensional analysis (doc/41 Ring 1–3), model provenance, CI with lint/mypy/pytest/coverage
gates, cross-OS wheel builds, and a release gate. The **human plugin** is correspondingly
large (~24k LOC across 40 modules) and its immune subsystem is genuinely literature-grounded
(doc/40, L3 benchmarks), while oncology/recovery/disease E2E tests exercise real behavioral
paths.

But measured against the stated goal, the project fails on **three structural axes**:

1. **Model authorship — the "Helix language" cannot author biology.** The `.helix` DSL is a
   *parameter-annotation grammar*: `#person`/`#drug`/`#disease`/`#gene`/`#config` set named
   parameters on pre-existing Python model classes. A user **cannot express an original ODE,
   an original tissue geometry, a novel rule, or new physiology** in the language. Every
   biological model is hardcoded in `plugins/human/*.py`. This is the single biggest blocker
   to "simulate 100%-realistic biology **via the Helix language**."

2. **Validation depth — the human stack is unvalidated.** Only **13/82 benchmarks are
   L3+** (literature/experimental); **59/82 (72%) are L0 functional/import/smoke** checks.
   The flagship human PK/PD/DDI/virtual-patient benchmarks (28–34, 57–67) are **L0**, not L3.
   There is **no L5 clinical tier at all** (0 benchmarks). The maths are coded and internally
   consistent and behaviorally plausible, but are **not pinned to clinical datasets or real
   PK/PD curves**, so "100% real" is an over-claim unsupported by evidence.

3. **Performance is uneven.** The immune/cohort kernels are properly numpy-vectorized
   (O2/O4/O9/O10), and native `_accel/` backends exist (Rust simplex 9.9–65.8×, Cython, GRN,
   diffusion) — but **the human plugin never imports `_accel`**. Its PBPK path regressed in
   efficiency (doc/39 O3 step-doubling is *layered on* the old fixed Euler grid and deep-copies
   dicts per slot), its dFBA LP solves do not use the native simplex backend, its
   stochastic-ensemble layer is fully scalar, and it has **no cohort-level parallelism** for
   the full `VirtualPatient`/`HumanSimulation` orchestrators.

Additional production gaps: the cardiology plugin is a grammar demo with **no sim backend**;
`interop` is only SBML/SBOL converters (no CellML/PhysiCell/virtual-tissue); no bundled
`helix.plugin.toml` manifests exist; the README/`__init__.py` "100% real" claim conflicts with
`DISCLAIMER.md`'s explicit "not faithful reproduction" hedge — a documented-reliability
mismatch that must be reconciled; and the deepest realism holes are cardiovascular, gas
exchange, renal filtration, thermoregulation, and cross-system coupling, all of which are
currently coarse proxies or absent.

This doc mandates a **rebasing of claims onto evidence**: either raise validation to match the
"100% real" claim or lower the claim; and a **language expressiveness program** so biology can
actually be authored in Helix. The remediation is scoped into Phases A–E with gates below.

---

## 2 — What Is Already Production-Grade (do not regress)

These are verified strengths and are treated as stable foundations:

- **Compiler/core pipeline** (`core/parser.py`→`ast_nodes.py`→`semantic`→`IR`→`hxbc`/`vm`),
  incremental recompile (`core/incr.py`), physical `Quantity`/`DimInferencer` dimensional
  checking (doc/41 Rings 1–3 + benchmark 76).
- **Plugin architecture**: `PluginProvider`, `GrammarDescriptor`/`FieldSpec` extensible
  grammar registry; cardiology `#cardiac_cycle` proves zero-parser-edit grammar extension.
- **Validation harness** (`validation/schema.py` L0–L5 taxonomy, `run_all.py`, goldens,
  provenance): well-built backbone; currently **82/82** goldens (doc/40 Phase E).
- **Human immune subsystem** (doc/40 Phases A–H): innate + adaptive + complement +
  tissue-vs-blood + spatial ABM + virtual population, **L3 literature-validated**, with
  numpy-vectorized cohort kernels and a multiprocessing `run_cohort`.
- **Oncology/recovery/virtual-patient E2E behavior**: post-treatment recovery, CYP2D6
  PM exposure bias, warfarin–amiodarone DDI → INR rise, microbiome sulfasalazine activation,
  cisplatin WBC nadir/creatinine rise — these are real behavioral tests of the biology code.
- **GEM reconstruction/import** with COBRApy reference paths (closest to production among
  non-human plugins).
- **Packaging/CI/release**: wheels, coverage gate, doc/41 CI repair with offline fallback.

---

## 3 — Gap Register (ranked)

Priority: **P0** blocks the goal outright · **P1** major · **P2** moderate. Each row is
evidence-anchored (file:line) and cross-checked against prior doc claims.

| # | Pri | Gap | Evidence | Severity |
|---|-----|-----|----------|----------|
| RT-1 | **P0** | **Helix cannot author novel bio models** — DSL is parameter-annotation over Python-hardcoded models; no user ODE/expressions/functions/control-flow in the language | `type_system.py`, `grammar_handlers.py`, `sim_runtime/backends/pipelines.py` (`#sim kind=` bespoke `_run_*`); ~24k LOC in `plugins/human/*.py` | HIGH |
| VD-1 | **P0** | **Human stack unvalidated** — only 13/82 benchmarks L3+; 59/82 (72%) L0 smoke; human PK/PD/DDI/tox/virtual-patient are L0; no L5 clinical tier (0) | `validation/report.md` (L0×59·L1×2·L2×8·L3×7·L4×6·L5×0); `benchmarks/{28-34,57-67}` | HIGH |
| VD-2 | **P0** | **"100% real" over-claim vs `DISCLAIMER.md`** — README + `human/__init__.py:34` claim every parameter anchored to literature, while the disclaimer says outputs "not faithful reproductions" | `README.md:159`, `plugins/human/__init__.py:34`, `DISCLAIMER.md` | HIGH (doc risk) |
| PF-1 | **P1** | **PBPK/ODE efficiency regression** — O3 step-doubling layered on the fixed `_EULER_SAFETY`/`_MAX_SUBSTEPS` outer grid; each slot deep-copies full state dict twice (coarse+fine), more on recursion → ≥3–10× work vs plain Euler | `virtual_patient.py:2569-2602,2635-2642` (`_snapshot_state`/`_restore_state`) | HIGH |
| PF-2 | **P1** | **dFBA LP not wired to native simplex** — per-hour LP solve per patient in pure numpy; `_accel/simplex` (Rust 9.9–65.8×) not integrated | `runtime/metabolism.py` (`simplex` is a Python module); `_accel/simplex/` exists unused by dFBA | HIGH |
| PF-3 | **P1** | **No cohort-level parallelism for full orchestrators** — multiprocessing only in immune `run_cohort`; `VirtualPatient.run()`/`HumanSimulation.run()` single-core, GIL-bound | `virtual_patient.py:1096`, `simulation.py:775` | HIGH (n≥100) |
| PF-4 | **P1** | Per-hour drug-key string normalization + fresh dict churn in the main loop (≥15 call sites) | `virtual_patient.py:1096-1541` | MED-HIGH |
| RT-2 | **P1** | **Type system can't typecheck user-authored math** — no expression language; typed diagnostics but no LSP/rich fixes | `type_system.py`, `semantic.py` | HIGH (ties RT-1) |
| RT-3 | **P1** | **No real JIT to native code** — "JIT" is incremental bytecode recompile; no LLVM/numba/AOT; native accel is Cython kernels | `core/incr.py`, `core/vm.py`, `_accel/` | MED |
| RT-4 | **P1** | **No bundled `helix.plugin.toml` manifests** — §6.8 prov/requires/native contract untested; plugins declared as Python `PluginProvider` only | `core/manifest.py`, `plugins/*/__init__.py` | MED |
| RT-5 | **P1** | **cardiology is a grammar demo only** — `#cardiac_cycle` validates but runs no cardio sim | `plugins/cardiology/__init__.py` | MED-HIGH |
| RL-1 | **P1** | **Cardiovascular coupling absent** — PBPK venous-equilibrated, no lung/arterial compartment; `CardiovascularODE` (disease_ode_models) and `VitalsModel` (clinical_output) compute BP/HR from disjoint states; no baroreflex/contractility/Starling | `pharmacokinetics.py:11-19`, `disease_ode_models.py:76-123,96`, `clinical_output.py:709-835` | HIGH |
| RL-2 | **P1** | **No gas exchange / O₂ delivery** — no alveolar-arterial transfer, Hb-O₂ dissociation, dead space, V/Q; SpO₂ is anemia-only linear drop; RR is bicarbonate-only | `clinical_output.py:813-833`, `disease_ode_models.py:879-918` | HIGH |
| RL-3 | **P1** | **No renal filtration/clearance + acid-base/electrolyte balance** — eGFR-slope CKD only; 3 divergent CKD impls; electrolytes are target-tracking not mass-balanced; no Henderson–Hasselbalch | `renal_model.py`, `disease_ode_models.py:725-769`, `clinical_output.py:846-868` | HIGH |
| RL-4 | **P1** | **No thermoregulation** — temp is only CRP-fever/thyroid scaling; no heat balance, set-point, ambient, circadian | `clinical_output.py:739-740`, `virtual_patient.py:1718,1817` | HIGH |
| RL-5 | **P1** | **Cross-system coupling is point-wise, not a closed loop** — disease→organ→PK clearance→perfusion→disease not bi-directional; FBA/7 clearance fixed while organs fail | `pharmacokinetics.py:81-87`, `organ_crosstalk.py` | HIGH |
| VD-3 | **P1** | **10 benchmarks level-tagged L2–L4 fail their own schema gate** (missing `reference.doi`/`golden_hash`/`experimental_comparison`) | `report.md` §Level-Gate Warnings; `benchmarks/*/benchmark.yaml` | MED |
| VD-4 | **P1** | Human tests assert **qualitative trends** not quantitative fidelity to measured datasets; no clinical-data-pinned tests | `tests/test_human_*.py`, `test_doc33_*.py` | MED-HIGH |
| RT-6 | **P2** | `interop` is only SBML/SBOL — no CellML/PhysiCell/virtual-tissue/external model integration | `interop/__init__.py` | MED |
| RT-7 | **P2** | No language stdlib / user functions / control flow; no LSP-style diagnostics | `doc/02-language-spec.md`, `core/*.py` | MED |
| PF-5 | **P2** | `solve_sde_ensemble` fully scalar single-threaded (500 trajectories, Python gauss loops) | `stochastic_ode.py:98,136-162` | MED |
| PF-6 | **P2** | spatial ABM agent-loop churn + O(n_apc×n_tcell) contact detection (numpy path partially vectorized) | `spatial_abm.py:255-337` | MED |
| RL-6 | **P2** | Deep tissue realism: no intra-organ gradients, PBPK partition ratios default to 1.0, no interstitial/intercellular volumes, no allometric/BSA organ scaling | `pharmacokinetics.py:79`, `physiology.py:60-133` | MED |
| RL-7 | **P2** | Sex/age depth shallow (70kg-male reference physiology), no pediatric/geriatric pharmacology, no menstrual/tanner/pregnancy staging | `physiology.py:245-249`, `phenotype.py` | MED |
| RL-8 | **P2** | Default-on uncertainty absent — stochastic/ensemble and Bayesian/4D-Var are optional post-processing, not the main loop; results are point estimates without credible intervals | `stochastic_ode.py`, `bayesian_fitter.py`, `virtual_4dvar.py` | MED |
| RL-9 | **P2** | Circadian limited to HPA cortisol; no circadian drive on temp/HR/BP; no sleep/wake | `endocrine.py:155-201` | LOW |
| RL-10 | **P2** | Microbiome uses arbitrary relative units (km_um/vmax_relative), not tissue/GEM-driven; no gut-diversity dynamics | `microbiome.py` | LOW |

---

## 4 — Remediation Plan (Phases)

Each phase has a concrete gate. This plan is **proposed**; implement per phase.

### Phase A — Reconcile claims with evidence (P0, smallest effort, highest integrity value)

**Problems:** VD-1 (unvalidated human), VD-2 (over-claim), VD-3 (gate breakage).

**Fix:**
1. **Re-scope the "100% real" claim.** Edit `plugins/human/__init__.py:34` and `README.md`
   so language/claims match `DISCLAIMER.md`: "internally consistent, literature-influenced
   models; validation status per benchmark level; **not** clinical decision support". Add a
   per-module validation-status matrix to the human plugin README/docstring.
2. **Close VD-3**: for the 10 level-gate-violating benchmarks either add the required
   `reference.doi`/`golden_hash`/`experimental_comparison` or downgrade their `level:` so the
   taxonomy is honest. Make `level_gate_violations` a CI **failure**, not a warning.
3. **Raise the human PK/PD/DDI/virtual-patient benchmarks from L0 to L3/L4** using published
   reference curves (e.g. Rowland–Tozer, published PBPK PK curves, published DDI interaction
   ratios, published dose–response). This is the first concrete step to making the flagship
   unvalidated surface actually validated.

**Gate:** `validation/report.md` shows 0 level-gate warnings; human PK/PD/DDI assertions carry
`reference.doi`; README removes unsupported "100% real" wording (keeps a scoped, evidenced
claim).

---

### Phase B — Press the human plugin toward physiological realism (P1, RL-1..RL-8)

**Problems:** cardiovascular/gas-exchange/renal/thermoregulation/coupling + deep tissue + sex
age + uncertainty.

**Fix (in dependency order):**
1. **Unify a single shared hemodynamic+fluid state** feeding PBPK perfusion, vitals, and
   disease models — resolve the three divergent renal/CKD implementations and the two disjoint
   CV models (RL-1, RL-3). Introduce a closed-loop circulatory core: CO driven by
   preload/afterload/contractility/Starling + a simple baroreflex/autonomic gain loop.
2. **Add a gas-exchange layer** (RL-2): alveolar-arterial O₂/CO₂ transfer with an
   oxygen-hemoglobin dissociation curve feeding real SpO₂, and respiratory-rate drive from
   PₐCO₂/PₐO₂/acid-base rather than bicarbonate-only rules.
3. **Rename/extend the "renal" layer to a filtration/clearance model** (RL-3): GFR-driven
   solute clearance, tubular secretion/reabsorption, creatinine turnover input, and an
   acid-base buffer (Henderson–Hasselbalch) reconciling CO₂/bicarbonate.
4. **Thermoregulation model** (RL-4): heat-production/heat-loss balance with a hypothalamic
   set-point and circadian + ambient coupling; fever becomes a *perturbation* of the set-point
   (from CRP/cytokines), not a direct temperature assignment.
5. **Bidirectional cross-system coupling** (RL-5): tie disease → organ function → renal/hepatic
   clearance → organ perfusion → disease, so a failing organ measurably changes drug exposure.
6. **Deep-tissue/sex-age/uncertainty** (RL-6/7/8) as lower-priority follow-ons: populate PBPK
   partition ratios, allometric organ scaling, pediatric/geriatric maturation, and make the
   stochastic/4D-Var ensemble and credible-interval reporting a default reporting option.

**Gate:** new doc/40-style L3/L4 benchmarks for CV/gas-exchange/renal/thermoregulation assert
mechanistic outputs against literature; a coupled organ-failure scenario measurably changes PK
clearance; vitals/electrolytes are mass/flow-balanced (not target-tracking).

---

### Phase C — Restore and extend performance (P1/P2, PF-1..PF-6)

**Problems:** PBPK efficiency regression, dFBA not wired to native simplex, no cohort
parallelism, per-hour churn, scalar stochastic + spatial ensembles.

**Fix:**
1. **Fix the O3 layering (PF-1):** remove the redundant outer `_EULER_SAFETY`/`_MAX_SUBSTEPS`
   fixed grid in `advance()` and let `_integrate_slot`'s local-error control alone govern
   resolution; replace the per-slot dict `_snapshot_state`/`_restore_state` deep-copies with a
   flat-array state (copy once, cheap).
2. **Wire `_accel/simplex` into the dFBA LP (PF-2)** and batch the per-hour LP over the cohort;
   apply the same to any per-patient LP solves. This is the largest algorithmic lever.
3. **Add cohort-level parallelism (PF-3)** to `VirtualPatient.run()`/`HumanSimulation.run()`
   (multiprocessing over patients, mirroring immune `run_cohort`), once the scalar hot spots are
   cut.
4. **Hoist invariant per-hour work (PF-4):** compute the drug-key normalization and `cyp_profile`
   once at `__init__` (the accepted O1 hoisting idiom).
5. **Vectorize + parallelize `solve_sde_ensemble` (PF-5)** and **bench/vectorize the spatial ABM
   agent loops + contact detection (PF-6)** ahead of n→10³–10⁴.

**Gate:** 30-day `VirtualPatient.run()` shows a measured ≥2–3× speedup over the O3 baseline with
bit-identical accepted-state results (golden-verifiable); a 100-patient cohort scales with core
count; goldens stay 82/82.

---

### Phase D — Make Helix a language that authors biology (P0, RT-1/2/3/7)

**Problems:** cannot author novel bio models; type system can't check user math; no stdlib /
functions / control flow.

**Fix:**
1. **Minimal user-authored model surface in the DSL:** add `#model`/`#species`/`#reaction` (or
   an SBML-import-backed model body) that lets users define ODE rate laws and parameters in the
   language and compile them into the human/ecosystem integrator — rather than only tuning
   Python-builtins. This is the minimum that makes "author biology in Helix" true.
2. **Expression + dimension checking for user math:** extend the doc/41 Ring 3 runtime guard and
   the `DimInferencer`/`semantic` layer to typecheck user-authored rule equations at
   compile time (units across species/reaction rates).
3. **Stdlib + functions/control-flow (RT-7):** a small math stdlib and user functions/loops so a
   novel computation is scriptable in-language.
4. **Native-code JIT (RT-3)** as a later milestone: compile user-authored model equations to
   numpy/numba or the existing `_accel` kernel-stub ABI, giving language-authored models the same
   native path as the built-ins.

**Gate:** an end-to-end `.helix` example defines an original 1–2-compartment ODE (or SBML model)
in-language, unit-checks it, runs it, and produces a golden-verifiable output — proving language
authorship, not just parameterization.

---

### Phase E — Close the remaining production surface (RT-4/5/6)

**Problems:** no bundled plugin manifests; cardiology demo has no sim; interop limited to
SBML/SBOL.

**Fix:**
1. **Ship at least one bundled `helix.plugin.toml`** (e.g. cardiology, or a new small plugin)
   exercising the §6.8 prov/requires/native manifest contract so it is actually tested.
2. **Give cardiology a real sim backend** (closed-loop CV is Phase B RL-1's core; reuse it) so
   `#cardiac_cycle` drives hemodynamic output instead of only validating a period.
3. **Extend `interop`** with CellML and a virtual-tissue/PhysiCell-style spatial exchange format
   to broaden model interop.

**Gate:** a plugin shipped via manifest builds, validates, and runs end-to-end in CI; cardiology
benchmark ≥ L2.

---

## 5 — Validation / Acceptance Summary

| Phase | Primary gate | Level target |
|-------|--------------|--------------|
| A | 0 level-gate warnings; human PK/PD/DDI carry `reference.doi`; README claims match disclaimer | L3/L4 on human core |
| B | New L3/L4 CV/gas/renal/thermo benchmarks; coupled organ-failure changes PK | L3/L4 |
| C | ≥2–3× PBPK speedup, cohort scales, goldens stay 82/82 | — |
| D | End-to-end user-authored ODE in `.helix`, unit-checked, golden output | L0→L3 as authored |
| E | Manifest-shipped plugin runs in CI; cardiology ≥L2 | L2+ |

The critical judgement: **the deterministic-engineering foundation is strong, but
"100%-realistic-biology-via-the-Helix-language" is currently blocked by (a) the language not
being able to author biology, (b) the human stack being unvalidated, and (c) the documented
"100% real" claim exceeding the evidence.** Phases A–D address these in increasing effort; Phase
A alone materially improves the honest reliability of the project.

---

## 6 — Appendix: Priority decision rationale

- **P0/P1 split** reflects "blocks the stated goal" (P0) vs "major, must fix for production
  realism/perf" (P1). RT-1 and VD-1/VD-2 are P0 because no amount of downstream polish makes the
  goal true while the language cannot author models and the human stack is unvalidated.
- **RL-1..RL-5 are P1 not P0** because the human plugin already simulates many realistic
  behaviors; these close the *physiological* fidelity holes in dependency order, but even a
  perfectly-coupled CV/gas/renal/thermo core remains "unvalidated" without VD-1's evidence base —
  hence Phase B depends on Phase A's honesty rebase.
- **PF-1** is the most cost-effective performance fix (pure code motion, no numerics change) and
  is therefore the first perf item; PF-2 is the largest algorithmic gain but requires wiring a
  backend.
