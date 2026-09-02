# doc/37 — Biological Validity, Performance Optimization, Decoupling Verification, and Helix-IR-First-Class Pipeline

> **Status:** IMPLEMENTED (2026-08-28)
>
> **Depends on:** doc/13 (performance report), doc/34 (architectural plan), doc/36 (plugin architecture)
>
> **Goal:** Establish complete biological validity infrastructure (Helix Model vs measured
> data), resolve performance bottlenecks across the full stack, formally verify
> that the scientific simulation layer and the language interpreter/compiler remain
> cleanly decoupled, and make Helix IR a true first-class abstraction:
> **helix language → typed biological IR → optimization → bytecode → multiple
> runtimes (CPU, GPU/vector, …)**.

---

## 1 — Executive Summary

This document covers four interconnected objectives:

1. **Biological Validity** — A framework for comparing HelixLang simulation output
   against published experimental and computational reference data, including
   out-of-scope detection, parameter fitting, uncertainty quantification, and
   replication verification.

2. **Performance Optimization** — Integration of the C dispatch kernel into the
   main CellVM execution loop, snapshot memory downsampling, and a comprehensive
   profiling harness. Target: ≤3× of native C throughput for the full VM pipeline.

3. **Decoupling Verification** — Automated AST-level and import-level tests that
   enforce the doc/36 architecture: core never imports plugins at module level;
   the registry is the sole bridge; sim_runtime depends on core via public types only.

4. **Helix IR as a First-Class Abstraction** (§5) — A typed biological IR between
   the AST and the runtimes, with optimization and multiple execution backends
   (classic CellVM + C kernel, portable IRRuntime, numpy/JAX BatchRuntime).

---

## 2 — Part A: Biological Validity Framework

### 2.1 Problem Statement

Current validation (doc/00–36, `validation/`) has 67 benchmarks, but only ~8 perform
true quantitative comparison against published data. The remaining ~59 are boolean
checks (does the API exist? does it import?). There is no systematic framework for:

- Detecting when simulation input falls outside the reference data domain
- Fitting model parameters to minimize error against experimental data
- Quantifying uncertainty in simulation predictions
- Verifying that repeated runs reproduce the same results

### 2.2 Architecture

```
bio_validity.py (src/helixlang/plugins/runtime/)
├── OutOfScopeDetector     — input range validation against reference bounds
├── ParameterFitter        — scipy.optimize-based parameter fitting
├── UncertaintyQuantifier  — bootstrap / Monte Carlo uncertainty
├── ReplicationVerifier    — multi-seed reproducibility checks
├── BioAccuracyReport      — aggregated accuracy assessment
└── BioAccuracySuite       — orchestrated full-chain validation
```

### 2.3 Components

#### 2.3.1 OutOfScopeDetector

Compares simulation input parameters against known reference ranges to detect
when the model extrapolates beyond its validated domain.

- Loads reference parameter ranges from `validation/references/`
- Checks each parameter against `[min, max]` bounds from literature
- Returns a scope report: which parameters are in-scope, which are out-of-scope
- Severity: `SAFE` (within 1σ), `WARNING` (1σ–2σ), `OUT_OF_SCOPE` (>2σ)

**Reference data sources:**
- BRENDA enzyme kinetics (kcat, Km ranges)
- BiGG metabolic models (flux bounds, growth rates)
- Published GRN parameters (Hill coefficients, Kd values)
- E. coli physiological ranges (generation time, cell volume, protein count)

#### 2.3.2 ParameterFitter

Fits HelixLang model parameters to minimize discrepancy with experimental data.

- Uses `scipy.optimize.minimize` (L-BFGS-B) with bounds from literature
- Objective function: weighted sum of squared relative errors
- Supports fixing subsets of parameters (partial fitting)
- Returns fitted parameters, residual error, and convergence status

**Supported fitting targets:**
- FBA growth rate vs measured growth rate
- GRN oscillation period vs measured period
- Enzyme kinetics (kcat/Km) vs BRENDA values
- Flux distributions vs fluxomics data

#### 2.3.3 UncertaintyQuantifier

Estimates confidence intervals on simulation predictions.

- **Bootstrap**: resample reference data with replacement, refit, collect predictions
- **Monte Carlo**: sample parameters from prior distributions, run forward model
- Returns mean prediction, 95% CI, standard deviation, coefficient of variation

#### 2.3.4 ReplicationVerifier

Verifies that simulation results are deterministic given the same seed.

- Runs the same program N times with the same seed
- Compares all output traces for bit-exact or ε-exact match
- Reports any non-determinism (expected: zero for seeded RNG)
- Cross-backend verification: Python vs numpy vs C backend outputs match

#### 2.3.5 BioAccuracyReport

Aggregated accuracy report comparing HelixLang against all available references:

```python
@dataclass
class BioAccuracyReport:
    benchmark_id: str
    scope: OutOfScopeReport          # are inputs in the validated domain?
    fit: FitResult                   # parameter fitting result
    uncertainty: UncertaintyResult   # CI on predictions
    replication: ReplicationResult   # reproducibility status
    overall_accuracy: float          # 0.0–1.0 composite score
    status: str                      # PASS / WARN / FAIL
```

### 2.4 Evidence Chain Integration

Each biological validity check produces an `EvidenceChain` (extending `validation/schema.py`)
with the full Reference → Expected → Actual → Error → Reproducibility chain, plus
the new `OutOfScopeReport` and `UncertaintyResult` fields.

---

## 3 — Part B: Performance Optimization

### 3.1 Current Bottleneck Analysis (from doc/13)

| Bottleneck | Current | Target | Status |
|---|---|---|---|
| Compile pipeline | ~470k codons/s | linear | DONE (doc/13) |
| GRN step | O(N+E) via `_incoming` | constant | DONE (doc/13) |
| Gray-Scott | numpy 0.007 μs/cell | DONE | DONE (doc/13) |
| **VM dispatch** | **~770 ns/op (Python)** | **≤50 ns/op (C)** | **THIS DOC** |
| **Snapshot memory** | **O(ticks) unbounded** | **O(1) via downsampling** | **THIS DOC** |
| **GRN step_accel** | **sigmoid only** | **Hill + noise** | **THIS DOC** |

### 3.2 VM Dispatch Integration

The C dispatch kernel (`_accel/dispatch/impl_cext.c`) already achieves ~50 ns/op
(15× faster than Python). Integration into CellVM:

1. **`_execute_pending` hot path**: Replace the Python `match`-dispatch loop with
   a call to `_accel.dispatch.backend.run_quota()` for the fast path.

2. **Fallback to Python dispatch**: Opcodes not handled by the C kernel
   (bio-instructions, GRN step, morphology) fall back to the Python dispatcher.

3. **Integration strategy**:
   - Extract the contiguous bytecode segment for the current tick's quota
   - Call `run_quota(code, constants, quota=ops_per_tick)`
   - Map the remaining stack state back to `vm.stack`
   - Handle any bio-opcodes by scanning the executed segment

**Implementation**: `core/performance.py` provides `accelerated_execute_pending()`.

### 3.3 Snapshot Downsampling

The trace list grows by one dict per tick, consuming O(ticks) memory. For long
simulations (10k+ ticks), this becomes the dominant memory cost.

**Strategy:**
- Configurable `snapshot_interval` (default: 1 = every tick)
- When `snapshot_interval > 1`, only append every Nth tick
- Always include tick 0 and the final tick
- For `max_ticks > 1000`, auto-set `snapshot_interval = max(1, max_ticks // 500)`

**Implementation**: Modified `_snapshot()` in `core/vm.py`.

### 3.4 GRN step_accel Extension

Extend `GRN.step_accel()` to support:
- Hill kinetics (vectorized via numpy)
- Telegraph noise (Monte Carlo perturbation on top of vectorized step)

This eliminates the Python fallback for noisy/Hill GRNs. **Status:** shipped —
`step_accel` layers the same per-node two-state-promoter perturbation as `step()`
over the kernel mean on the same RNG, so noisy and Hill graphs advance through
the accelerated kernel with results identical to the scalar path (verified
draw-for-draw in `tests/test_accel_foundation.py`).

### 3.5 Profiling Harness

A comprehensive profiling script (`benchmarks/bench_profile.py`) that:
- Profiles the full VM pipeline with `cProfile` + `line_profiler`
- Measures per-component time (compile, GRN, dispatch, snapshot)
- Compares Python vs accelerated execution
- Reports memory usage via `tracemalloc`
- Outputs a structured JSON report

---

## 4 — Part C: Decoupling Verification

### 4.1 Architecture Contract (from doc/36)

```
Layer 1: core/       — zero scientific deps, <50ms import
Layer 2: plugins/    — lazy-loaded, registered via plugin_registry
Layer 3: sim_runtime — integration adapter, imports core types + plugins
```

**Rules:**
- R1: `core/` must NEVER import `plugins/` or `sim_runtime/` at module level
- R2: `core/vm.py` may import plugins inside method bodies (lazy pattern)
- R3: `sim_runtime/` may import `core/` public types only
- R4: The registry (`plugin_registry.py`) is the sole bridge between core and plugins
- R5: No silent fallbacks — all fidelity degradation requires explicit opt-in

### 4.2 Verification Tests

`tests/test_decoupling.py` enforces:

1. **AST-level import scan**: Parse all `core/*.py` files, check that no
   `import helixlang.plugins.*` or `import helixlang.sim_runtime.*` appears
   at module level (outside function/method/class bodies).

2. **Runtime import injection test**: Import `helixlang.core` in a fresh
   subprocess with no scientific extras installed; verify it succeeds in <50ms
   and does not trigger any numpy/scipy/cobra imports.

3. **Registry-only bridge test**: Verify that the only core module mentioning
   plugin names is `plugin_registry.py`.

4. **Silent fallback linter**: Run `find_silent_fallbacks src --fail` and
   verify zero violations (already in CI).

5. **sim_runtime direction test**: Verify sim_runtime imports from core
   are limited to public types (Program, Compiler, SimConfigError, etc.).

---

## 5 — Part D: Helix IR as a True First-Class Abstraction

### 5.1 Goal

Make the compiler/runtime split a genuine **IR pipeline** with Helix IR as a
first-class, typed, portable artifact — the single object every backend
consumes — instead of an incidental intermediate inside `compiler.py`.  The
target chain (part of doc/34's architectural roadmap, fully implemented here):

```
helix language (DNA codons)                    ← high-level .helix source
        │  Lexer → Parser → SemanticAnalyzer   (AST, unchanged)
        ▼
         typed biological IR  (HLIR v1)        ← ir.py   + ir_serialize.py
        │  IRBuilder                           ← ir_builder.py
        ▼
         optimization  (opt-in)                ← ir_opt.py  fold / dead / unreachable
        │
        ▼
         bytecode   (existing Chunk ABI)       ← ir_lower.py  (byte-identical default)
        │
        ├──► CPU  runtime #1   CellVM + C dispatch kernel   (--runtime classic)
        ├──► CPU  runtime #2   IRRuntime: portable typed-IR interpreter
        │                      (--runtime ir)
        └──► GPU/vector runtime  BatchRuntime: numpy/JAX banked-stack engine
                                 (--runtime batch [--batch-backend numpy|jax])
```

One program source → one typed IR → one lowering → *any* runtime.  The IR is
**inspectable** (`--ir-text`, `--dump-ir`, HLIR JSON), **optimizable** without
touching the bytecode ABI, and **re-targetable**: new runtimes consume the same
artifact.  Doc/36's R5 holds by construction: every non-default backend is
parity-tested (benchmarks 71–73), so a runtime switch is a *performance* switch,
never a *fidelity* switch.

### 5.2 Why the IR is first-class (design decisions)

- **Stack-machine form, not register/SSA.** The IR mirrors the bytecode VM's
  operand stack exactly (same pop order, same underflow semantics), so lowering
  is a mechanical re-encoding and the C dispatch kernel / CellVM behaviour is
  preserved bit-for-bit.
- **Typed.** Every instruction that pushes a value carries an `IRType`
  annotation (`NUM / I64 / F64 / BOOL / GENE / mRNA / PROTEIN / METAB /
  SIGNAL / ENERGY / VOID`); binary numeric ops promote via `promote_numeric`.
  Types are inspectable and enable type-aware future passes; they never alter
  execution semantics.
- **Semantic operands, not positions.** `OP_PUSH_CONST` carries the constant
  *literal* (the lowerer re-inserts it into the constant pool); `OP_CALL_GENE`
  carries the *resolved target gene name* (honouring `call_target=<name>`,
  falling back to `wobble % n` exactly like the legacy emitter).  The IR is
  stable across pool/offset layout changes.
- **Explicit purity.** `PURE_OPS` plus per-instruction `pop_effect() /
  push_effect()` / `net_effect()` make optimization decisions *checkable*, not
  just implemented.

### 5.3 Implementation path analysis

| Stage | Mechanism | File(s) |
|---|---|---|
| language → IR | codon-table mapping, wobble operands, CALL_GENE resolution, config/call_target/lsystem/use-directive snapshots | `core/ir_builder.py` |
| IR core types | `IRProgram` / `IRFunction` / `IRInst`, `IRType`, `IR_VERSION`, purity tables | `core/ir.py` |
| optimization | constant folding, dead-code elimination, unreachable-suffix removal; **only ever folds within pure windows** (a simulated stack depth refuses underflow windows, so degenerate programs are left loud instead of *rewritten to be wrong*); idempotent, per-gene | `core/ir_opt.py` |
| IR → bytecode | per-gene ORF emission, HALT guard, inter-gene jump barrier, CALL_GENE back-patching, plugin/lsystem/gene-name constants; `lower(ir)` == legacy bytecode with `optimize=False` | `core/ir_lower.py` |
| serialization | HLIR JSON with embedded `IR_VERSION`; too-new payloads raise `IRFormatError` | `core/ir_serialize.py` |
| CPU runtimes | `IRRuntime(CellVM)`: executes the IR directly (flat gene-ORF ip, shared `BioInstructionDispatcher`, delta-proteins, snapshot machinery); `CellVM` + `performance.accelerated_execute_pending` on lowered chunks | `core/ir_runtime.py`, `core/performance.py` |
| GPU/vector runtime | `BatchRuntime`: N virtual cells, numpy/JAX banked-stack engine, vectorised pure-op cohorts, scalar fallback for effects, per-cell quotas, `StackDepthError` on underflow | `core/ir_batch_runtime.py` |
| compiler facade | `compile()` == `compile_ir(optimize=False)[1]`; `build_ir` hook | `core/compiler.py` |
| tooling | `--ir-text`, `--dump-ir PATH`, `--optimize`, `--runtime {classic,ir,batch}`, `--batch-n`, `--batch-backend` | `cli.py` |
| verification | builder / lowering / optimizer / serializer / parity / fallback tests (39) + validation benchmarks 71–73 | `tests/test_ir.py`, `validation/benchmarks/71-73/` |

**Why optimization is opt-in:** folding removes pure instructions, which can
shift the `ops_per_tick` quota boundary crossed inside a gene ORF within a
tick.  The classic VM and golden validation therefore default to the
byte-identical `optimize=False` path; `--optimize` and the IR runtimes opt in
explicitly.

### 5.4 Multiple runtimes (CPU, GPU/vector, and the extension point)

- **`classic`** — CellVM + C dispatch kernel on the lowered chunk.  Default;
  goldens unchanged.
- **`ir`** — portable typed-IR interpreter (`IRRuntime`): no chunk decoding,
  `ip` is an instruction index into the flattened gene ORFs; parity-tested
  against `classic`.
- **`batch`** — banked-stack engine over N virtual cells (`--batch-n`), engine
  selected by `--batch-backend numpy|jax` (graceful numpy fallback when JAX is
  absent).  Pure-op cohorts execute vectorised; effect/call/halt opcodes drop to
  the shared scalar dispatcher.  Trace-parity-tested cell-for-cell.
- **extension point** — a new runtime only needs `IRProgram` + `Program`;
  the interface contract is "trace-identical to `IRRuntime`, verified by
  benchmark 72's pattern".

---

## 6 — Implementation Files

| File | Purpose |
|---|---|
| `src/helixlang/plugins/runtime/bio_validity.py` | Biological validity framework |
| `src/helixlang/core/performance.py` | Performance optimization integration |
| `src/helixlang/core/vm.py` (modified) | Snapshot downsampling + accel dispatch |
| `src/helixlang/plugins/runtime/grn.py` (modified) | Extended step_accel with Hill/noise |
| `tests/test_bio_validity.py` | Biological validity unit tests |
| `tests/test_performance.py` | Performance optimization tests |
| `tests/test_decoupling.py` | Decoupling verification tests |
| `benchmarks/bench_profile.py` | Comprehensive profiling harness |
| `validation/benchmarks/68_bio_validity/` | Biological validity benchmark |
| `validation/benchmarks/69_performance_benchmark/` | Performance optimization benchmark |
| `validation/benchmarks/70_decoupling_verify/` | Decoupling verification benchmark |
| `src/helixlang/core/ir.py` | Typed biological IR core (`IRProgram`/`IRInst`/`IRType`, purity tables) |
| `src/helixlang/core/ir_builder.py` | AST → typed IR (CALL_GENE resolution, config snapshot) |
| `src/helixlang/core/ir_opt.py` | fold / dead / unreachable passes (pure-window, idempotent) |
| `src/helixlang/core/ir_lower.py` | typed IR → bytecode `Chunk` (byte-identical default) |
| `src/helixlang/core/ir_runtime.py` | Portable typed-IR CPU runtime (`IRRuntime(CellVM)`) |
| `src/helixlang/core/ir_batch_runtime.py` | numpy/JAX vector runtime over N virtual cells |
| `src/helixlang/core/ir_serialize.py` | HLIR JSON round-trip (`IRFormatError` version guard) |
| `src/helixlang/core/compiler.py` (modified) | IR pipeline facade: `compile_ir` / `build_ir` |
| `tests/test_ir.py` | IR builder/lowerer/optimizer/serializer/runtime parity tests |
| `validation/benchmarks/71_ir_roundtrip/` | IR round-trip + optimizer correctness benchmark |
| `validation/benchmarks/72_batch_runtime_parity/` | numpy/JAX batch runtime parity benchmark |
| `validation/benchmarks/73_ir_serialization/` | HLIR serialization robustness benchmark |

---

## 7 — Acceptance Criteria

### 6.1 Biological Validity
- [x] `OutOfScopeDetector` correctly identifies parameters outside literature ranges
- [x] `ParameterFitter` reduces error by ≥30% vs unoptimized parameters
- [x] `UncertaintyQuantifier` produces valid 95% CIs containing the reference value
- [x] `ReplicationVerifier` confirms bit-exact reproducibility across 10 runs
- [x] `BioAccuracyReport` aggregates all checks into a single pass/fail/warn status
- [x] Validation benchmark 68 passes with all checks green

### 6.2 Performance
- [x] C dispatch kernel integrated into CellVM (≥5× speedup on dispatch-heavy programs)
- [x] Snapshot downsampling reduces memory by ≥50% for 1000+ tick simulations
- [x] Profiling harness produces structured JSON report
- [x] Validation benchmark 69 passes

### 6.3 Decoupling
- [x] AST-level import scan finds zero module-level plugin imports in core/
- [x] `import helixlang.core` succeeds without scientific extras
- [x] Registry is the sole bridge (only `plugin_registry.py` references plugin names)
- [x] Validation benchmark 70 passes with all coupling checks green

### 7.4 Helix IR (Part D)
- [x] `Compiler.compile()` (IR path, `optimize=False`) is byte-identical to the legacy
      bytecode ABI (golden chunk regression in benchmark 71)
- [x] IRBuilder emits one typed IR function per gene; `PUSH_CONST` carries the literal,
      `CALL_GENE` the resolved target name; config/call_target/lsystem/use metadata snapshotted
- [x] `IROpt` folds only within pure windows (refuses underflow windows), is idempotent,
      and never removes effect boundaries (semantic parity proven in benchmark 71)
- [x] HLIR JSON round-trips with full fidelity; future versions raise `IRFormatError`
- [x] `IRRuntime` (portable CPU) and `BatchRuntime` (numpy/JAX vector CPU/GPU-class) parity
      tests pass; benchmark 72 verifies every batch cell is trace-identical to the IR VM
      and that numpy/JAX engines agree
- [x] CLI exposes `--ir-text`, `--dump-ir`, `--optimize`, `--runtime {classic,ir,batch}`,
      `--batch-n`, `--batch-backend`; default behaviour (`classic`) is unchanged
- [x] Validation benchmarks 71–73 pass (IR round-trip, batch runtime parity, serialization)
