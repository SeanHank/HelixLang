# HelixLang Performance Report

> Status: measured with `benchmarks/bench_helix.py` — a pure-stdlib harness that
> isolates and times each stage of the language pipeline. Reproducible on any
> machine (see §7). This report is an **analysis**, not a claim about absolute
> numbers: absolute throughput varies with hardware; the *scaling behavior* and
> *bottleneck structure* are the durable findings.
>
> **2026-08 update.** All three §4 bottlenecks have been **fixed** in this build:
> `Compiler` (linear compile), `GRN.step` (O(N+E)), and `GrayScott.step`
> (scratch double-buffer + optional `numpy` backend, bit-identical output).
> Before/after measurements appear inline below.

## 1. Executive summary

| Question | Answer |
|---|---|
| Compile speed | ~470–710k codons/s; **no longer grows with gene count** (linear scaling restored) |
| VM execution speed | ~**1.3 M ops/s** (~770 ns/op), ~20k ticks/s |
| GRN update | **O(N+E)** now: 128-node cycle ~30k steps/s, dense 4096-edge ~6k steps/s (was 25× slower, now 4.7×) |
| Reaction–diffusion | numpy backend **0.007–0.056 μs/cell** (pure-Python fallback ~0.18 μs/cell, was ~0.30) |
| Memory | ~**0.33 MiB** peak for a 16-gene × 200-tick compile+run |
| Where the time goes | `04_turing_pattern.helix` dropped **110 ms → ~12.6 ms**; total example wall-time **114 ms → ~16 ms** |

**Verdict.** HelixLang is fast enough for interactive use and for simulation
experiments at the shipped scale: **every** shipped example now completes in
**≤ 13 ms**. The three scaling problems identified in the previous report are
resolved; the remaining cost structure is flat and predictable:

1. **Compiler** — O(genes × chunk) eliminated; constant-total-codons compile
   times are now ~3 ms regardless of gene count (was 6.4 → 12.4 → 24.2 ms).
2. **GRN** — incoming-edge index makes `step()` O(N+E); a 4096-edge GRN is now
   ~25× faster than before (165 vs 4 119 μs/step).
3. **Gray-Scott** — full-array copies removed and a `numpy` backend added; the
   dominant hotspot is 8–40× faster depending on grid size.

## 2. Methodology

* **Workloads.** Synthetic programs (N genes × M codons, fixed shapes, seeded
  RNG) plus all 16 shipped `examples/*.helix`.
* **Measurement.** `time.perf_counter`, **best-of-5 repeats**, each run with the
  GC disabled during timing (allocations still occur; only periodic sweeps are
  excluded). A warm-up pass precedes each measurement. Per-stage isolation is
  achieved by timing each stage against the *output of the previous stage*
  (lex on source, parse on pre-lexed tokens, semantic on the pre-built AST,
  compile on the pre-built AST), so the timings reflect that stage alone.
* **Instruction counting.** A `CountingVM` subclass wraps the dispatcher to count
  dispatched instructions exactly.
* **Gray-Scott backend.** When `numpy` is installed (the optional `fast` extra)
  the vectorized backend runs; otherwise the pure-Python scratch double-buffer
  runs. Both paths are verified bit-identical to the pre-optimization algorithm
  (see `tests/test_reaction_diffusion.py`).
* **Environment.** Apple M4 Pro, 24 GB RAM, macOS (Darwin), CPython 3.11.15.
  Raw numbers are reproduced in the JSON block in §6.

## 3. Results

### 3.1 Compile pipeline

| Genes | Codons | lex | parse | semantic | compile | total | codons/s |
|---|---|---|---|---|---|---|---|
| 1 | 16 | 0.02 ms | 0.01 ms | 0.00 ms | 0.01 ms | 0.03 ms | 471 k |
| 4 | 64 | 0.06 ms | 0.02 ms | 0.00 ms | 0.02 ms | 0.11 ms | 599 k |
| 16 | 256 | 0.25 ms | 0.08 ms | 0.00 ms | 0.08 ms | 0.41 ms | 621 k |
| 64 | 1024 | 1.06 ms | 0.35 ms | 0.00 ms | 0.34 ms | 1.74 ms | 587 k |
| 16 | 1024 | 0.89 ms | 0.28 ms | 0.00 ms | 0.28 ms | 1.44 ms | 710 k |
| 64 | 4096 | 3.75 ms | 1.11 ms | 0.00 ms | 1.17 ms | 6.04 ms | 678 k |

Observations:

* **Lex + parse dominate small programs**; semantic analysis is negligible
  (< 2% at every size).
* **Compile no longer scales with gene count.** The constant-total-codons
  experiment — which previously grew 6.4 → 12.4 → 24.2 ms (doubling per gene
  doubling) — is now flat:

  | Genes × codons/gene | total codons | compile time (before → after) |
  |---|---|---|
  | 32 × 64 | 2048 | 6.36 ms → **2.91 ms** |
  | 64 × 32 | 2048 | 12.36 ms → **3.14 ms** |
  | 128 × 16 | 2048 | 24.15 ms → **3.56 ms** |

  Root cause was `Compiler._ends_with_halt` (compiler.py:88), which rescanned
  the entire emitted chunk from byte 0 for *every* gene (O(genes × chunk)).
  `_compile_orf` now returns the ip of the last emitted instruction and the
  HALT check is a single byte comparison (`_last_op_is_halt`), restoring linear
  compile scaling. Codons/s roughly **4.5×** at 64×4096 (147k → 678k).

### 3.2 CellVM execution

| Ticks | wall | ticks/s | executed ops | ops/s |
|---|---|---|---|---|
| 100 | 4.8 ms | 20.8 k | 6 400 | 1.33 M |
| 1000 | 48.7 ms | 20.5 k | 64 000 | 1.31 M |
| 5000 | 242.7 ms | 20.6 k | 320 000 | 1.32 M |

Observations:

* **Throughput is flat** from 100 to 5000 ticks (~20.5–20.8k ticks/s, ~1.3M ops/s):
  per-op cost ~**770 ns**. This is a healthy sign — no hidden super-linear term in
  the execution loop.
* **Snapshot overhead is ~2%** of a run (measured by disabling `_snapshot`),
  so the tick loop is dominated by GRN + dispatch + cell bookkeeping, not tracing.
* Per-op cost is dominated by Python-level dispatch (`match` on `Op` IntEnum +
  per-op `_read_u8`/handler calls). At ~770 ns/op this is *expected* for a pure
  Python interpreter and is acceptable for the simulator use case; a future
  bytecode loop in C (or `numba`/vectorized dispatch) is the only path to a
  step-change improvement.

### 3.3 GRN `step()` scaling

| Nodes | Edges | steps/s | μs/step | μs/node |
|---|---|---|---|---|
| 8 | 8 | 459.7 k | 2.2 | 0.27 |
| 32 | 32 | 118.4 k | 8.4 | 0.26 |
| 128 | 128 | 29.5 k | 33.8 | 0.26 |

| N=128 | E | μs/step (before → after) |
|---|---|---|
| cycle | 128 | 160 → **35** |
| dense | 4096 | 4 119 → **165** |

Observations:

* **Per-node cost is now constant** (~0.26 μs/node) regardless of graph size —
  the previous super-linear blow-up is gone. Adding 32× more edges now costs only
  **4.7×** more per step (was **25.7×**), consistent with O(N+E).
* `GRN.step()` now reads a **target → incoming-edges index** (`_incoming`, grn.py)
  built and kept in sync by `add_edge`/`add_gene`. In-place weight updates
  (`OP_REGULATE`) mutate the same `Edge` objects, so the index stays valid; a
  cheap `len(edges)` guard rebuilds it if external code ever appends to `edges`
  directly. Dense 4096-edge GRNs are ~25× faster.

### 3.4 Reaction–diffusion (Gray-Scott)

numpy backend (the `fast` extra; the pure-Python fallback is shown below):

| Field | μs/step | μs/cell (before → after) |
|---|---|---|
| 16×16 | 14.2 | 0.245 → **0.056** |
| 32×32 | 19.1 | 0.282 → **0.019** |
| 64×64 | 41.1 | 0.294 → **0.010** |
| 128×128 | 121.8 | 0.305 → **0.007** |

* Two changes: (a) the per-step full-array copies (`[row[:] for row in self.u]`)
  are replaced by a **scratch double-buffer** that only copies the O(n) border
  cells; (b) when numpy is installed the whole update is **vectorized**
  (`np.clip`, sliced Laplacians). Both backends produce **bit-identical** output
  to the original algorithm (regression-tested in `tests/test_reaction_diffusion.py`).
* Even the pure-Python fallback (no numpy) is ~2× faster — ~0.14–0.18 μs/cell vs
  ~0.30 before — because the array copies are gone:

  | Field | μs/step | μs/cell |
  |---|---|---|
  | 16×16 | 35.4 | 0.138 |
  | 32×32 | 166.7 | 0.163 |
  | 64×64 | 724.0 | 0.177 |
* **`04_turing_pattern.helix`** (100 ticks × `react_steps=2` × two `GAT` codons
  = 400 field steps) dropped from **110.3 ms to ~12.6 ms** — the largest single
  win in the language runtime.

### 3.5 Full compile+run of the shipped examples

| Example | ticks | wall |
|---|---|---|
| `01_hello_dna.helix` | 1 | 0.06 ms |
| `02_lac_operon.helix` | 20 | 0.23 ms |
| `03_plant_growth.helix` | 5 | 0.09 ms |
| `04_turing_pattern.helix` | 100 | **12.57 ms** |
| `05_table_switch.helix` | 1 | 0.07 ms |
| `06_crispr_edit.helix` | 1 | 0.21 ms |
| `07_evolution.helix` | 1 | 0.22 ms |
| `08_epigenetics.helix` | 1 | 0.25 ms |
| `09_central_dogma_pipeline.helix` | 20 | 0.72 ms |
| `10_metabolism_fba.helix` | 20 | 0.46 ms |
| `11_protein_structure.helix` | 1 | 0.21 ms |
| `12_multi_species.helix` | 1 | 0.25 ms |
| `13_dna_storage.helix` | 1 | 0.17 ms |
| `14_synbio_designer.helix` | 1 | 0.21 ms |
| `15_3d_morphology.helix` | 5 | 0.19 ms |
| `16_population_dynamics.helix` | 30 | 0.41 ms |

Sum of all 16: **~16.3 ms** (was ~114 ms), of which `04` alone is 12.6 ms (77%).

### 3.6 Memory

| Workload | Peak |
|---|---|
| 16 genes × 64 codons, 200 ticks (compile + run, tracemalloc) | **0.33 MiB** (344 344 B), 200 snapshots |

Memory use is minimal for simulator-scale programs. One structural caveat: the VM
retains **every snapshot in `trace`** (`vm.run` returns the full trace), so memory
grows linearly with `ticks`. For long runs this should be streamed or downsampled.

## 3.7 Cross-stack hot-loop matrix (doc/36 Phase 5)

The `_accel` hot loops (VM `dispatch`, `grn_step`, `simplex`, `diffusion`) are
each backed by multiple **equivalent-fidelity** implementations — swapping
backend is a deliberate *speed*-only switch that never changes numerics
(doc/36 §3ξ.5). This matrix measures every backend present on this interpreter
at the same workload, using the same public loading path production code uses
(`python benchmarks/bench_accel_matrix.py`, best-of-5, GC disabled, warm-up):

| kernel | workload | impl_cext | impl_cython | impl_numba | impl_numpy | impl_python |
|---|---|---|---|---|---|---|
| `dispatch` | length=128 | **0.50 us** | n/a | n/a | n/a | 7.38 us |
| `grn_step` | N=256, E=4096 | **23.08 us** | 66.54 us | n/a | 156.00 us | 125.08 us |
| `simplex` | 40×60 | n/a | **78.50 us** | n/a | 79.83 us | 84.37 us |
| `diffusion` | 64×64 | n/a | n/a | **5.38 us** | 40.58 us | 3.898 ms |

Observations:

* The compiled/native stack is the only path to a step-change for CPU-bound
  kernels: `impl_cext` is **~15×** faster than the pure-Python reference on VM
  dispatch and ~5.4× on `grn_step`; numba `diffusion` is **~725×** faster than
  the Python loop.
* The selection order `native,numpy,python` (doc/36 §4.2) maps a declared
  fidelity level to its fastest present impl; because the numerics are
  identical, these speed-ups are pure wins with no change to results.
* One caveat for the matrix: it reflects *this* build's compiled set
  (`impl_cext` for dispatch/grn, `impl_cython` for grn/simplex, `impl_numba`
  for diffusion). On a py-only wheel the `impl_python` column is the floor;
  installing the native wheel raises that floor per the table above.

## 4. Bottleneck analysis

All three §2026 bottlenecks are **resolved** in this build. Ranking by the
measured improvement:

| # | Hot spot | Location | Cost model | Status |
|---|---|---|---|---|
| 1 | `GrayScott.step` array copies + Python loop | `reaction_diffusion.py` | O(cells) × 0.3 μs | **FIXED** — scratch double-buffer + numpy backend; ~0.007–0.056 μs/cell |
| 2 | `GRN.step` full-edge scan per node | `grn.py:143-147` | O(N·E) | **FIXED** — per-target incoming-edge index → O(N+E) |
| 3 | `Compiler._ends_with_halt` chunk rescan | `compiler.py:88-98` | O(genes × size) | **FIXED** — last instruction tracked at emission; O(1) HALT check |
| 4 | VM per-op dispatch overhead | `vm.py` `dispatch()` | ~770 ns/op | open — acceptable; C/numba loop is the only step change |
| 5 | Snapshot accumulation in `trace` | `vm.py:708-725` | O(ticks) memory | open — stream/downsample snapshots |

The three fixes are pure algorithmic/constant-factor changes that do **not** alter
language semantics: they were made under the existing §5 compatibility rules
(§10 of the production-upgrade plan) without touching the public API or breaking
legacy behavior. Full-suite regressions pass.

## 5. Recommendations

1. **Done — linear compiler.** `_last_op_is_halt` replaced the per-gene chunk
   rescan; constant-total-codons compile time is flat (~3 ms).
2. **Done — indexed GRN.** `_incoming` keeps `step()` at O(N+E); update it in
   `add_edge`/`add_gene` (and it self-heals on direct `edges` mutation).
3. **Done — fast Gray-Scott.** Scratch double-buffer + numpy backend; output is
   bit-identical to the original. If numpy is unwanted at runtime, the pure-Python
   path is still ~2× faster than before.
4. **Remaining — stream or downsample snapshots** for long runs (e.g. every k-th
   tick) to bound memory growth; expose it via `#config`.
5. **Re-run `benchmarks/bench_helix.py`** on target hardware to re-baseline — the
   absolute numbers above are machine-specific, but the scaling columns
   (§3.1/3.3/3.4) should stay flat.

## 6. Reproducibility

Re-run the full matrix on this machine:

```bash
python benchmarks/bench_helix.py            # full matrix (takes ~1-2 min)
python benchmarks/bench_helix.py --fast     # reduced matrix
python benchmarks/bench_helix.py --json out.json
```

The harness is pure standard library (no pytest-benchmark). The raw best-of-N
measurements behind every table above (JSON) follow.

```json
{
  "platform": {
    "python": "3.11.15",
    "implementation": "CPython",
    "machine": "arm64",
    "system": "Darwin",
    "node": "Admins-MacBook-Pro.local",
    "processor": "arm"
  },
  "compile_matrix": [
    {
      "genes": 1,
      "codons": 16,
      "lex": 1.8625054508447647e-05,
      "parse": 7.847324013710022e-06,
      "semantic": 7.500251134236654e-07,
      "compile": 6.777932867407799e-06,
      "total": 3.400033650298913e-05,
      "codons_per_s": 470583.57785939454
    },
    {
      "genes": 4,
      "codons": 64,
      "lex": 6.373599171638489e-05,
      "parse": 2.2125042354067166e-05,
      "semantic": 9.719903270403545e-07,
      "compile": 2.0055643593271572e-05,
      "total": 0.00010688866799076398,
      "codons_per_s": 598753.836145943
    },
    {
      "genes": 16,
      "codons": 256,
      "lex": 0.0002540693773577611,
      "parse": 8.130570252736409e-05,
      "semantic": 1.7220154404640198e-06,
      "compile": 7.519428618252277e-05,
      "total": 0.00041229138150811195,
      "codons_per_s": 620920.0858470119
    },
    {
      "genes": 64,
      "codons": 1024,
      "lex": 0.0010575416963547468,
      "parse": 0.00034711137413978577,
      "semantic": 4.680672039588292e-06,
      "compile": 0.0003354862953225772,
      "total": 0.001744820037856698,
      "codons_per_s": 586880.0092746877
    },
    {
      "genes": 16,
      "codons": 1024,
      "lex": 0.000885277676085631,
      "parse": 0.000277152673030893,
      "semantic": 1.66667935748895e-06,
      "compile": 0.000278666615486145,
      "total": 0.0014427636439601579,
      "codons_per_s": 709748.9628926897
    },
    {
      "genes": 64,
      "codons": 4096,
      "lex": 0.003754124976694584,
      "parse": 0.0011138193464527528,
      "semantic": 4.749977961182594e-06,
      "compile": 0.0011657499708235264,
      "total": 0.006038444271932046,
      "codons_per_s": 678320.4109441013
    }
  ],
  "vm": [
    {
      "ticks": 100,
      "wall_s": 0.004808333003893495,
      "ticks_per_s": 20797.228461303763,
      "executed_ops": 6400,
      "ops_per_s": 1331022.6215234408
    },
    {
      "ticks": 1000,
      "wall_s": 0.04868383286520839,
      "ticks_per_s": 20540.6998822117,
      "executed_ops": 64000,
      "ops_per_s": 1314604.7924615487
    },
    {
      "ticks": 5000,
      "wall_s": 0.24265112495049834,
      "ticks_per_s": 20605.715308428993,
      "executed_ops": 320000,
      "ops_per_s": 1318765.7797394556
    }
  ],
  "grn": [
    {
      "nodes": 8,
      "edges": 8,
      "steps_per_s": 459734.8220791283,
      "step_s": 2.1751669701188804e-06
    },
    {
      "nodes": 32,
      "edges": 32,
      "steps_per_s": 118431.95493297528,
      "step_s": 8.443667087703943e-06
    },
    {
      "nodes": 128,
      "edges": 128,
      "steps_per_s": 29544.024549998776,
      "step_s": 3.384779207408428e-05
    }
  ],
  "gray_scott": [
    {
      "grid": "16x16",
      "step_s": 1.4214601833373308e-05,
      "cell_s": 5.5525788411614485e-08
    },
    {
      "grid": "32x32",
      "step_s": 1.9143742974847555e-05,
      "cell_s": 1.8695061498874566e-08
    },
    {
      "grid": "64x64",
      "step_s": 4.1145796421915294e-05,
      "cell_s": 1.0045360454569164e-08
    },
    {
      "grid": "128x128",
      "step_s": 0.00012181875063106417,
      "cell_s": 7.435226478946788e-09
    }
  ],
  "examples": [
    {
      "example": "01_hello_dna.helix",
      "wall_s": 5.587493069469929e-05,
      "ticks": 1
    },
    {
      "example": "02_lac_operon.helix",
      "wall_s": 0.00022808299399912357,
      "ticks": 20
    },
    {
      "example": "03_plant_growth.helix",
      "wall_s": 9.241700172424316e-05,
      "ticks": 5
    },
    {
      "example": "04_turing_pattern.helix",
      "wall_s": 0.01257062517106533,
      "ticks": 100
    },
    {
      "example": "05_table_switch.helix",
      "wall_s": 6.508408114314079e-05,
      "ticks": 1
    },
    {
      "example": "06_crispr_edit.helix",
      "wall_s": 0.0002062499988824129,
      "ticks": 1
    },
    {
      "example": "07_evolution.helix",
      "wall_s": 0.00021970784291625023,
      "ticks": 1
    },
    {
      "example": "08_epigenetics.helix",
      "wall_s": 0.0002516659442335367,
      "ticks": 1
    },
    {
      "example": "09_central_dogma_pipeline.helix",
      "wall_s": 0.000721583841368556,
      "ticks": 20
    },
    {
      "example": "10_metabolism_fba.helix",
      "wall_s": 0.0004613748751580715,
      "ticks": 20
    },
    {
      "example": "11_protein_structure.helix",
      "wall_s": 0.0002128342166543007,
      "ticks": 1
    },
    {
      "example": "12_multi_species.helix",
      "wall_s": 0.0002533751539885998,
      "ticks": 1
    },
    {
      "example": "13_dna_storage.helix",
      "wall_s": 0.00016970792785286903,
      "ticks": 1
    },
    {
      "example": "14_synbio_designer.helix",
      "wall_s": 0.0002057079691439867,
      "ticks": 1
    },
    {
      "example": "15_3d_morphology.helix",
      "wall_s": 0.00018833298236131668,
      "ticks": 5
    },
    {
      "example": "16_population_dynamics.helix",
      "wall_s": 0.0004135000053793192,
      "ticks": 30
    }
  ],
  "memory": {
    "peak_bytes": 344344,
    "peak_mib": 0.32839202880859375,
    "snapshots": 200
  }
}
```

Raw best-of-5 measurements behind the §3.7 cross-stack matrix
(`benchmarks/bench_accel_matrix.py`, seconds per unit of work):

```json
{
  "dispatch": [
    {
      "kernel": "dispatch",
      "workload": "length=128",
      "backends": {
        "impl_cext": 5.00003807246685e-07,
        "impl_python": 7.3750270530581474e-06
      }
    }
  ],
  "grn_step": [
    {
      "kernel": "grn_step",
      "workload": "N=256,E=4096",
      "backends": {
        "impl_cext": 2.3083004634827375e-05,
        "impl_cython": 6.654200842604041e-05,
        "impl_numpy": 0.00015600002370774746,
        "impl_python": 0.00012508401414379478
      }
    }
  ],
  "simplex": [
    {
      "kernel": "simplex",
      "workload": "40x60",
      "backends": {
        "impl_cython": 7.850001566112041e-05,
        "impl_numpy": 7.983302930369973e-05,
        "impl_python": 8.43749730847776e-05
      }
    }
  ],
  "diffusion": [
    {
      "kernel": "diffusion",
      "workload": "64x64",
      "backends": {
        "impl_numpy": 4.0583021473139524e-05,
        "impl_numba": 5.375011824071407e-06,
        "impl_python": 0.0038980419631116092
      }
    }
  ]
}
```

## 7. Where to find the harness

* `benchmarks/bench_helix.py` — the measurement harness (this report's numbers;
  includes the Gray-Scott per-cell section added with the numpy backend).
* `benchmarks/bench_accel_matrix.py` — the cross-stack hot-loop matrix (§3.7);
  its raw best-of-5 JSON is reproduced at the end of §6.
* `tests/test_reaction_diffusion.py` — regression tests pinning both Gray-Scott
  backends to the original algorithm (bit-identical).
* `tests/test_vm_fuzz.py` — Phase 5 fuzz parity harness: seeds the native
  dispatch kernel against the pure-Python reference over randomized bytecode
  (including truncated/unknown opcodes) and asserts byte-identical results (or
  identical typed exceptions). This is what caught the impl_cext out-of-bounds
  read on a trailing `PUSH_CONST` (fixed; see §5.5/§11 of doc/36).
* `tests/test_benchmark.py` — the existing pytest-benchmark regression suite for
  the biological modules (FBA, CRISPR, evolution, protein structure); it is
  skipped without `pytest-benchmark`. The new harness complements it by covering
  the **language pipeline itself** without any third-party dependency.
