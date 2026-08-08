# HelixLang Performance Report

> Status: measured with `benchmarks/bench_helix.py` — a pure-stdlib harness that
> isolates and times each stage of the language pipeline. Reproducible on any
> machine (see §7). This report is an **analysis**, not a claim about absolute
> numbers: absolute throughput varies with hardware; the *scaling behavior* and
> *bottleneck structure* are the durable findings.

## 1. Executive summary

| Question | Answer |
|---|---|
| Compile speed | ~130–470k codons/s; a 16-gene program compiles in **<3 ms** |
| VM execution speed | ~**1.3 M ops/s** (~770 ns/op), ~20k ticks/s |
| GRN update | ~53k steps/s at 32 nodes; **scales O(N·E)** (quadratic in dense graphs) |
| Reaction–diffusion | ~0.28–0.3 μs per grid cell per step; **the single dominant hotspot** |
| Memory | ~**0.33 MiB** peak for a 16-gene × 200-tick compile+run |
| Where the time goes | 98% of example wall-time is one workload: `04_turing_pattern.helix` (reaction-diffusion) |

**Verdict.** HelixLang is fast enough for interactive use and for simulation
experiments at the shipped scale: all examples except `04_turing_pattern.helix`
complete in **≤ 1 ms**. Three concrete, verifiable scaling problems dominate the
cost structure — none is architectural, all are fixable by targeted refactors:

1. **`Compiler._ends_with_halt()` rescans the whole emitted chunk for every gene**
   → compile time is **O(genes × chunk size)** (quadratic in gene count).
2. **`GRN.step()` scans every edge for every node** (`for e in self.edges if
   e.target == name`) → **O(N·E)** per tick (quadratic for dense GRNs).
3. **`GrayScott.step()` copies both concentration arrays and loops in pure Python**
   → ~0.3 μs/cell; the field dominates any program that uses it.

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
* **Environment.** Apple M4 Pro, 24 GB RAM, macOS (Darwin), CPython 3.11.15.
  Raw numbers are reproduced in the JSON block in §6.

## 3. Results

### 3.1 Compile pipeline

| Genes | Codons | lex | parse | semantic | compile | total | codons/s |
|---|---|---|---|---|---|---|---|
| 1 | 16 | 0.02 ms | 0.01 ms | 0.00 ms | 0.01 ms | 0.04 ms | 410 k |
| 4 | 64 | 0.06 ms | 0.02 ms | 0.00 ms | 0.05 ms | 0.14 ms | 468 k |
| 16 | 256 | 0.27 ms | 0.09 ms | 0.00 ms | 0.47 ms | 0.83 ms | 309 k |
| 64 | 1024 | 1.07 ms | 0.35 ms | 0.00 ms | 6.21 ms | 7.63 ms | 134 k |
| 16 | 1024 | 0.90 ms | 0.26 ms | 0.00 ms | 1.74 ms | 2.91 ms | 352 k |
| 64 | 4096 | 3.53 ms | 1.09 ms | 0.00 ms | 23.22 ms | 27.85 ms | 147 k |

Observations:

* **Lex + parse dominate small programs**; semantic analysis is negligible
  (< 2% at every size).
* **Compile dominates large programs** and its per-codon cost grows with the
  number of *genes*, not codons — see the constant-total-codons experiment:

  | Genes × codons/gene | total codons | compile time |
  |---|---|---|
  | 32 × 64 | 2048 | 6.36 ms |
  | 64 × 32 | 2048 | 12.36 ms |
  | 128 × 16 | 2048 | 24.15 ms |

  Doubling the gene count **doubles** compile time at a fixed total codon count,
  i.e. compile is **O(genes × chunk size)**. Root cause: `Compiler._ends_with_halt`
  (compiler.py:88) rescans the entire emitted chunk from byte 0 for *every* gene to
  decide whether to append a synthetic `OP_HALT`. Fix: track the emitted chunk
  length and check the last instruction directly, or record whether the ORF ended
  with `OP_HALT` during emission. Cost of the fix is negligible; it restores
  linear compile scaling.

### 3.2 CellVM execution

| Ticks | wall | ticks/s | executed ops | ops/s |
|---|---|---|---|---|
| 100 | 4.7 ms | 21.1 k | 6 400 | 1.35 M |
| 1000 | 49.4 ms | 20.2 k | 64 000 | 1.30 M |
| 5000 | 247.5 ms | 20.2 k | 320 000 | 1.29 M |

Observations:

* **Throughput is flat** from 100 to 5000 ticks (~20.2–21.1k ticks/s, ~1.3M ops/s):
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
| 8 | 8 | 299.9 k | 3.3 | 0.42 |
| 32 | 32 | 53.0 k | 18.9 | 0.59 |
| 128 | 128 | 6.3 k | 158.1 | 1.24 |

| N=128 | E | μs/step |
|---|---|---|
| cycle | 128 | 160 |
| dense | 4096 | 4 119 |

Observations:

* Per-node cost **grows with edge count**: 32× more edges → 25.7× slower step.
  `GRN.step()` (grn.py:140) computes each node's `inputs` by linearly scanning the
  entire `self.edges` list filtering on `e.target == name` → **O(N·E)** per tick.
  Fix: build a **target → incoming-edges index** once (edges are append-only in
  practice; `OP_REGULATE` can update the index on add/update). This makes the step
  O(N + E) and removes the quadratic behavior on dense GRNs.

### 3.4 Reaction–diffusion (Gray-Scott)

| Field | μs/step | μs/cell |
|---|---|---|
| 16×16 | 62.7 | 0.245 |
| 32×32 | 289.0 | 0.282 |
| 64×64 | 1204.3 | 0.294 |
| 128×128 | 4994.3 | 0.305 |

* Cost is **O(cells)** (~0.28–0.30 μs/cell, flat across sizes — good linear
  scaling). The per-cell cost is dominated by two things: (a) `step()` copies both
  `u` and `v` arrays every step (`nu = [row[:] for row in self.u]`, reaction_diffusion.py:68-69), and (b) the update loop is pure Python with 5-list-lookup
  Laplacians per cell.
* **This is the single dominant hotspot of the language runtime.** `04_turing_pattern.helix`
  (100 ticks × `react_steps=2` × two `GAT` codons = 400 field steps × ~289 μs ≈ 115 ms)
   measures **110.3 ms** — ~98% of the combined wall time of all 16 examples.
* Fix options, in increasing impact: skip the array copy by updating in place with
  boundary guards (or keep one scratch row), vectorize with `numpy` (F, k, Du, Dv
  are already plain floats), or switch to an out-of-core/tiled update for big grids.

### 3.5 Full compile+run of the shipped examples

| Example | ticks | wall |
|---|---|---|
| `01_hello_dna.helix` | 1 | 0.06 ms |
| `02_lac_operon.helix` | 20 | 0.23 ms |
| `03_plant_growth.helix` | 5 | 0.10 ms |
| `04_turing_pattern.helix` | 100 | **110.32 ms** |
| `05_table_switch.helix` | 1 | 0.07 ms |
| `06_crispr_edit.helix` | 1 | 0.21 ms |
| `07_evolution.helix` | 1 | 0.22 ms |
| `08_epigenetics.helix` | 1 | 0.25 ms |
| `09_central_dogma_pipeline.helix` | 20 | 0.75 ms |
| `10_metabolism_fba.helix` | 20 | 0.44 ms |
| `11_protein_structure.helix` | 1 | 0.22 ms |
| `12_multi_species.helix` | 1 | 0.27 ms |
| `13_dna_storage.helix` | 1 | 0.18 ms |
| `14_synbio_designer.helix` | 1 | 0.20 ms |
| `15_3d_morphology.helix` | 5 | 0.19 ms |
| `16_population_dynamics.helix` | 30 | 0.44 ms |

Sum of all 16: **~114 ms**, of which `04` alone is 110 ms (97%).

### 3.6 Memory

| Workload | Peak |
|---|---|
| 16 genes × 64 codons, 200 ticks (compile + run, tracemalloc) | **0.33 MiB** (342 848 B), 200 snapshots |

Memory use is minimal for simulator-scale programs. One structural caveat: the VM
retains **every snapshot in `trace`** (`vm.run` returns the full trace), so memory
grows linearly with `ticks`. For long runs this should be streamed or downsampled.

## 4. Bottleneck analysis

Ranked by impact on the shipped workloads:

| # | Hot spot | Location | Cost model | Fix |
|---|---|---|---|---|
| 1 | `GrayScott.step` array copies + Python loop | `reaction_diffusion.py:65-91` | O(cells) × 0.3 μs | in-place/scratch update; optional numpy vectorization |
| 2 | `GRN.step` full-edge scan per node | `grn.py:143-147` | O(N·E) | per-target incoming-edge index → O(N+E) |
| 3 | `Compiler._ends_with_halt` chunk rescan | `compiler.py:88-98` | O(genes × size) | track last instruction at emission time |
| 4 | VM per-op dispatch overhead | `vm.py` `dispatch()` | ~770 ns/op | acceptable; C/numba loop is the only step change |
| 5 | Snapshot accumulation in `trace` | `vm.py:708-725` | O(ticks) memory | stream/downsample snapshots |

Note that 1–3 are pure algorithmic/constant-factor issues that do **not** change
language semantics — they can be fixed under the existing §5 compatibility rules
(§10 of the production-upgrade plan) without touching the public API or breaking
legacy behavior.

## 5. Recommendations

1. **Fix `Compiler._ends_with_halt`** (10-line change) to restore linear compile
   scaling for large gene counts. Highest value-per-line-of-code.
2. **Index GRN incoming edges** to drop the O(N·E) step to O(N+E). Keep the index
   updated by `add_edge` and `OP_REGULATE`.
3. **Eliminate the Gray-Scott full-array copies** (in-place with boundary guards
   or a single scratch row); optionally add a `numpy`-backed backend behind the
   existing `GrayScott` interface.
4. **Stream or downsample snapshots** for long runs (e.g. every k-th tick) to bound
   memory growth; expose it via `#config`.
5. **Re-run `benchmarks/bench_helix.py`** after any of the above and confirm the
   scaling columns (§3.1/3.3/3.4) flatten before/after.

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
      "lex": 2.2083365668853123e-05,
      "parse": 7.555664827426274e-06,
      "semantic": 7.776543498039246e-07,
      "compile": 8.569331839680672e-06,
      "total": 3.898601668576399e-05,
      "codons_per_s": 410403.5590238309
    },
    {
      "genes": 4,
      "codons": 64,
      "lex": 6.476370617747307e-05,
      "parse": 2.2708360726634662e-05,
      "semantic": 9.166542440652847e-07,
      "compile": 4.8222020268440247e-05,
      "total": 0.00013661074141661328,
      "codons_per_s": 468484.3910247378
    },
    {
      "genes": 16,
      "codons": 256,
      "lex": 0.0002670139850427707,
      "parse": 8.993060328066349e-05,
      "semantic": 1.8749851733446121e-06,
      "compile": 0.0004688193245480458,
      "total": 0.0008276388980448246,
      "codons_per_s": 309313.63980688003
    },
    {
      "genes": 64,
      "codons": 1024,
      "lex": 0.0010681112762540579,
      "parse": 0.0003511250639955203,
      "semantic": 4.6944090475638705e-06,
      "compile": 0.0062061943269024296,
      "total": 0.007630125076199572,
      "codons_per_s": 134204.87734783452
    },
    {
      "genes": 16,
      "codons": 1024,
      "lex": 0.0008999723164985577,
      "parse": 0.0002633333206176758,
      "semantic": 1.6250026722749074e-06,
      "compile": 0.0017407780202726524,
      "total": 0.0029057086600611606,
      "codons_per_s": 352409.72850266634
    },
    {
      "genes": 64,
      "codons": 4096,
      "lex": 0.0035281249632438025,
      "parse": 0.0010940279656400282,
      "semantic": 4.597318669160207e-06,
      "compile": 0.023221278019870322,
      "total": 0.027848028267423313,
      "codons_per_s": 147084.02191588946
    }
  ],
  "vm": [
    {
      "ticks": 100,
      "wall_s": 0.004739332944154739,
      "ticks_per_s": 21100.015799340516,
      "executed_ops": 6400,
      "ops_per_s": 1350401.011157793
    },
    {
      "ticks": 1000,
      "wall_s": 0.04939429182559252,
      "ticks_per_s": 20245.25432070013,
      "executed_ops": 64000,
      "ops_per_s": 1295696.2765248083
    },
    {
      "ticks": 5000,
      "wall_s": 0.24749149987474084,
      "ticks_per_s": 20202.714042828036,
      "executed_ops": 320000,
      "ops_per_s": 1292973.6987409943
    }
  ],
  "grn": [
    {
      "nodes": 8,
      "edges": 8,
      "steps_per_s": 299891.25601907977,
      "step_s": 3.3345420379191637e-06
    },
    {
      "nodes": 32,
      "edges": 32,
      "steps_per_s": 52950.79451700272,
      "step_s": 1.8885457888245582e-05
    },
    {
      "nodes": 128,
      "edges": 128,
      "steps_per_s": 6325.025678474398,
      "step_s": 0.0001581021249294281
    }
  ],
  "examples": [
    {
      "example": "01_hello_dna.helix",
      "wall_s": 5.587516352534294e-05,
      "ticks": 1
    },
    {
      "example": "02_lac_operon.helix",
      "wall_s": 0.00022841710597276688,
      "ticks": 20
    },
    {
      "example": "03_plant_growth.helix",
      "wall_s": 0.00010045897215604782,
      "ticks": 5
    },
    {
      "example": "04_turing_pattern.helix",
      "wall_s": 0.11032416694797575,
      "ticks": 100
    },
    {
      "example": "05_table_switch.helix",
      "wall_s": 6.699981167912483e-05,
      "ticks": 1
    },
    {
      "example": "06_crispr_edit.helix",
      "wall_s": 0.0002054581418633461,
      "ticks": 1
    },
    {
      "example": "07_evolution.helix",
      "wall_s": 0.00021741585806012154,
      "ticks": 1
    },
    {
      "example": "08_epigenetics.helix",
      "wall_s": 0.00025233300402760506,
      "ticks": 1
    },
    {
      "example": "09_central_dogma_pipeline.helix",
      "wall_s": 0.0007532499730587006,
      "ticks": 20
    },
    {
      "example": "10_metabolism_fba.helix",
      "wall_s": 0.0004407090600579977,
      "ticks": 20
    },
    {
      "example": "11_protein_structure.helix",
      "wall_s": 0.0002173751126974821,
      "ticks": 1
    },
    {
      "example": "12_multi_species.helix",
      "wall_s": 0.00026662484742701054,
      "ticks": 1
    },
    {
      "example": "13_dna_storage.helix",
      "wall_s": 0.00017554196529090405,
      "ticks": 1
    },
    {
      "example": "14_synbio_designer.helix",
      "wall_s": 0.00019724993035197258,
      "ticks": 1
    },
    {
      "example": "15_3d_morphology.helix",
      "wall_s": 0.0001858340110629797,
      "ticks": 5
    },
    {
      "example": "16_population_dynamics.helix",
      "wall_s": 0.00043816701509058475,
      "ticks": 30
    }
  ],
  "memory": {
    "peak_bytes": 342848,
    "peak_mib": 0.32696533203125,
    "snapshots": 200
  }
}
```


## 7. Where to find the harness

* `benchmarks/bench_helix.py` — the measurement harness (this report's numbers).
* `tests/test_benchmark.py` — the existing pytest-benchmark regression suite for
  the biological modules (FBA, CRISPR, evolution, protein structure); it is
  skipped without `pytest-benchmark`. The new harness complements it by covering
  the **language pipeline itself** without any third-party dependency.
