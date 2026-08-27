# doc/36 — Performance Optimization & Code Structure Refactoring Plan

> **Status:** DRAFT
>
> **Depends on:** doc/13 (performance report), doc/34 (architectural improvement plan)
>
> **Goal:** Systematically address Python performance bottlenecks by delegating
> compute-critical paths to native extensions (C/Rust), while preserving the
> zero-dependency core and pure-Python fallbacks.

---

## 1 — Executive Summary

HelixLang's pure-Python VM dispatch loop runs at ~770 ns/op (~1.3 M ops/s).
For single-cell simulations this is adequate, but population-scale (N=1000+
cells) and genome-scale FBA (2700+ reactions) hit Python's interpreter overhead
as a hard ceiling. This document proposes a phased plan to:

1. **Identify** the 4 hot paths that account for >90% of wall-clock time
2. **Implement** native C extensions for each, with pure-Python fallbacks
3. **Restructure** the codebase to cleanly separate "hot inner loops" from
   "orchestration logic" so native code is a drop-in acceleration, not a rewrite

---

## 2 — Performance Bottleneck Inventory

### 2.1 Current Baseline (doc/13, measured 2026-08-26)

| Metric | Value | Bottleneck |
|--------|-------|-----------|
| VM dispatch | ~770 ns/op | Python `match` on `Op` IntEnum |
| GRN step (128 nodes) | ~33 us/step | Pure Python edge traversal |
| Simplex (E. coli core) | ~50 ms | Python list-of-lists iteration |
| Population (1000 cells) | ~800 ms/tick | Per-cell bytecode dispatch x N |
| Gray-Scott (128x128) | ~122 us/step | numpy (already optimized) |
| Compile throughput | ~710k codons/s | Already fast enough |

### 2.2 The 4 Critical Hot Paths

| Priority | Path | File:Line | Current Cost | Why Python Hurts |
|----------|------|-----------|-------------|-----------------|
| **P0** | VM opcode dispatch | `vm.py:742-769` | 770 ns/op | `match` on IntEnum + `_read_u8`/`_read_u16` per instruction |
| **P0** | Population per-cell dispatch | `population.py:1004-1178` | 770 ns/op x N | Same dispatch loop duplicated for N cells |
| **P1** | GRN edge accumulation | `grn.py:262-295` | 0.26 us/node | Python dict iteration over edges per node |
| **P1** | Simplex pivot inner loop | `metabolism.py:414-492` | O(rows x vars) | Python list-of-lists; numpy path helps but still Python-controlled |

### 2.3 What Already Has numpy Acceleration

| Module | Pure Python | numpy Path | Status |
|--------|-------------|-----------|--------|
| `reaction_diffusion.py` | `_step_py()` | `_step_numpy()` | Done |
| `metabolism.py` | `_simplex_max()` | `_simplex_max_numpy()` + scipy | Done |
| `population.py` | `_step_metabolism_python()` | `_step_vectorized_metabolism()` | Done |
| `environment.py` | list loops | numpy diffusion | Done |
| `evolution.py` | pure Python | numpy batch mutation | Done |
| `vm.py` | **NO numpy path** | **NONE** | **Needs native** |
| `grn.py` | **NO numpy path** | **NONE** | **Needs native** |
| `population.py` (dispatch) | **NO numpy path** | **NONE** | **Needs native** |

---

## 3 — Architecture: Hot Loop Isolation Pattern

The key design principle: **isolate hot inner loops into self-contained modules
that can be replaced with native code without touching orchestration logic.**

```
+----------------------------------------------+
|  Orchestration Layer (pure Python)            |
|  vm.py, population.py, server.py, CLI        |
|  - tick loop, GRN wiring, snapshot, debug    |
|  - calls into hot_loop.py for dispatch       |
+---------------------+------------------------+
                      |
          +-----------v-----------+
          |  Hot Loop Module       |
          |  _dispatch.py          |
          |  - dispatch_table      |
          |  - execute_chunk()     |
          |  - read_u8/u16()       |
          +-----------+-----------+
                      |
     +----------------+----------------+
     v                v                v
+----------+  +------------+  +------------+
| pure     |  | numpy      |  | native C   |
| Python   |  | batch      |  | extension  |
| fallback |  | (future)   |  | (future)   |
+----------+  +------------+  +------------+
```

### 3.1 Module Decomposition

| Current Module | Extracted Hot Loop | New Module | Contents |
|---------------|-------------------|-----------|---------|
| `vm.py` | `_execute_pending` + `_dispatch` | `helixlang/_dispatch.py` | Op dispatch table, read_u8/u16, frame push/pop |
| `grn.py` | `step()` inner loop | `helixlang/_grn_step.py` | Edge accumulation, level update, decay |
| `metabolism.py` | `_simplex_max` inner loop | `helixlang/_simplex.py` | Pivot selection, ratio test, tableau update |
| `population.py` | `_execute_cell` inner loop | `helixlang/_pop_dispatch.py` | Per-cell bytecode execution |

Each new module exports a **pure-Python implementation** with the same signature.
Future native extensions replace only these modules.

---

## 4 — Phase 1: Dispatch Table Refactoring (Weeks 1-2)

### 4.1 Current State

`vm.py` has a 200-line `match` statement in `BioInstructionDispatcher.dispatch()`
that handles 50+ opcodes. This is called ~1.3M times/second and is the single
hottest loop.

### 4.2 Proposed Refactoring

Extract into `helixlang/_dispatch.py`:

```python
# _dispatch.py — hot loop module
def dispatch(op: int, vm: 'CellVM') -> None:
    """Dispatch a single opcode. This is the hot inner loop."""
    _DISPATCH_TABLE[op](vm)

# Computed dispatch table (avoids match overhead)
_DISPATCH_TABLE: list[Callable] = [None] * 256
_DISPATCH_TABLE[Op.OP_PUSH_CONST] = _op_push_const
_DISPATCH_TABLE[Op.OP_CALL_GENE] = _op_call_gene
# ... etc
```

**Benefits:**
- Table lookup is O(1) vs match cascade
- Native extension can replace `_dispatch.py` entirely
- Population backend reuses the same dispatch table

### 4.3 Migration Steps

1. Create `helixlang/_dispatch.py` with pure-Python dispatch table
2. Update `vm.py` to import and call `_dispatch.dispatch(op, self)`
3. Update `population.py` to use the same dispatch table
4. Verify all tests pass
5. Benchmark: expect 10-20% improvement from table dispatch alone

---

## 5 — Phase 2: C Extension for VM Dispatch (Weeks 3-6)

### 5.1 Technology Choice: C with Python C API

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **C + CPython API** | Maximum control, no runtime deps, ~3x faster than Cython | Manual reference counting | **Selected** |
| Cython | Easier to write, `.pyx` syntax | Runtime dep, less control | Rejected |
| Rust + PyO3 | Memory safety, modern toolchain | New ecosystem, larger binary | Future option |
| C++ + pybind11 | RAII, templates | Heavier ABI, C++ runtime | Rejected |
| Numba JIT | Zero code change | JIT warmup, no control flow opt | Rejected |

### 5.2 C Extension Architecture

```
helixlang/
  _vm_core.c          # C implementation of dispatch loop
  _vm_core.h          # Shared types (Chunk, Frame, Cell)
  _vm_core_module.c   # Python module definition
  vm.py               # Python fallback + imports _vm_core if available
```

### 5.3 Key C Functions

```c
// _vm_core.c
typedef struct {
    uint8_t *code;
    int code_len;
    uint16_t *gene_offsets;
} VmChunk;

typedef struct {
    int return_ip;
    char gene_name[64];
} VmFrame;

// The hot loop — replacing 200 lines of Python match
int vm_execute_pending(VmChunk *chunk, double *stack, int *stack_len,
                       VmFrame *frames, int *frames_len,
                       int ops_per_tick) {
    int quota = ops_per_tick;
    while (*frames_len > 0 && quota > 0) {
        uint8_t op = chunk->code[ip++];
        switch (op) {
            case OP_PUSH_CONST: { /* ... */ break; }
            case OP_CALL_GENE: { /* ... */ break; }
            // ... 50 cases
            default: break;
        }
        quota--;
    }
    return 0;
}
```

### 5.4 Expected Performance

| Metric | Current | With C Extension | Speedup |
|--------|---------|-----------------|---------|
| VM dispatch | 770 ns/op | ~50-80 ns/op | **10-15x** |
| Population (1000 cells) | 800 ms/tick | ~80-120 ms/tick | **7-10x** |
| Total 16 examples | 16.3 ms | ~3-5 ms | **3-5x** |

### 5.5 Build System Changes

```toml
# pyproject.toml — add ext_modules
[tool.setuptools.ext-modules]
ext_modules = [
    {name = "helixlang._vm_core", sources = ["src/helixlang/_vm_core.c"]}
]
```

Alternative: migrate to `scikit-build-core` for CMake-based builds if the
C extension grows complex.

### 5.6 Fallback Strategy

```python
# vm.py — graceful degradation
try:
    from helixlang._vm_core import dispatch as _native_dispatch
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False
    from helixlang._dispatch import dispatch as _py_dispatch

def _dispatch(self, op: Op) -> None:
    if _HAS_NATIVE:
        _native_dispatch(op, self)
    else:
        _py_dispatch(op, self)
```

---

## 6 — Phase 3: GRN Step Acceleration (Weeks 5-8)

### 6.1 Current GRN Step

```python
# grn.py:262-295
def step(self) -> list[str]:
    triggered = []
    for name, node in self.nodes.items():
        accumulation = 0.0
        for edge in self.edges.get(name, []):
            accumulation += edge.weight * self.nodes[edge.source].level
        # ... update level, check threshold
```

### 6.2 Proposed: Sparse Matrix-Vector Multiply

For GRNs with >100 nodes, the edge traversal becomes a sparse matrix-vector
product: `levels += W @ levels`. This is a natural fit for:

1. **numpy `sparse`** — if scipy is available (dev dependency)
2. **C extension** — a lightweight sparse matvec for the production path
3. **Existing `SparseGRN`** — already does this for genome-scale models; extend
   the pattern to all GRN sizes

### 6.3 Implementation

```python
# _grn_step.py
import numpy as np

def grn_step_matrix(nodes: dict, edges: dict, decay: float) -> list[str]:
    """GRN step via sparse matrix multiply (numpy path)."""
    names = list(nodes.keys())
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}
    # Build sparse weight matrix
    row, col, data = [], [], []
    for target, edge_list in edges.items():
        for edge in edge_list:
            row.append(idx[target])
            col.append(idx[edge.source])
            data.append(edge.weight)
    W = np.zeros((n, n))
    W[row, col] = data
    levels = np.array([nodes[name].level for name in names])
    # Matrix multiply + decay
    new_levels = levels + W @ levels - decay * levels
    new_levels = np.clip(new_levels, 0.0, 1.0)
    # Check thresholds
    triggered = []
    for i, name in enumerate(names):
        nodes[name].level = float(new_levels[i])
        if new_levels[i] > nodes[name].threshold:
            triggered.append(name)
    return triggered
```

### 6.4 Expected Performance

| GRN Size | Current | numpy matvec | Speedup |
|----------|---------|-------------|---------|
| 128 nodes | 33 us | ~5 us | **6x** |
| 1000 nodes | ~260 us | ~15 us | **17x** |
| 5000 nodes | ~1.3 ms | ~40 us | **32x** |

---

## 7 — Phase 4: Simplex Solver Native Extension (Weeks 7-10)

### 7.1 Current State

The simplex solver in `metabolism.py` has two paths:
- Pure Python: `_simplex_max()` — list-of-lists, ~50 ms for E. coli core
- numpy: `_simplex_max_numpy()` — ndarray ops, ~5 ms for E. coli core
- scipy: `linprog()` — for models > 500 reactions, ~2 ms

### 7.2 Proposed: C Simplex Core

A C implementation of the revised simplex method would:
- Eliminate scipy dependency for genome-scale models
- Achieve ~0.5 ms for E. coli core (100x faster than pure Python)
- Enable real-time dFBA with 1000+ time steps

### 7.3 File Layout

```
helixlang/
  _simplex.c           # C simplex implementation
  _simplex.h           # API header
  _simplex_module.c    # Python module definition
  metabolism.py        # imports _simplex or falls back to Python
```

### 7.4 Expected Performance

| Model | Reactions | Pure Python | numpy | C Extension | Speedup vs Python |
|-------|-----------|-------------|-------|-------------|-------------------|
| E. coli core | 95 | 50 ms | 5 ms | ~0.5 ms | **100x** |
| iML1515 | 2712 | 15 s | 200 ms | ~20 ms | **750x** |
| Human generic | 10000+ | N/A | >5 s | ~200 ms | **N/A** |

---

## 8 — Phase 5: Population Backend Unification (Weeks 9-12)

### 8.1 Current Problem

`population.py` duplicates the entire VM dispatch loop (~200 lines) for
per-cell execution. This is P0 priority because population simulations
multiply the dispatch cost by N (number of cells).

### 8.2 Proposed: Shared Dispatch Core

```
+-------------------+     +-------------------+
|   vm.py           |     |  population.py    |
|   (single cell)   |     |  (N cells)        |
+--------+----------+     +--------+----------+
         |                         |
         v                         v
+--------+-------------------------+----------+
|              _dispatch.py                    |
|  dispatch(op, vm) — single shared impl      |
+--------+------------------------------------+
         |
    +----+----+
    v         v
+-------+ +--------+
| Python| | C ext  |
+-------+ +--------+
```

### 8.3 Implementation Steps

1. Extract `_dispatch.py` from `vm.py` (Phase 1)
2. Refactor `population.py._execute_cell()` to call `_dispatch.dispatch()`
3. When C extension is available, both paths get the same speedup
4. Population cell state (ip, frames, stack) becomes a C struct for cache locality

### 8.4 Expected Performance

| Population Size | Current | After Unification | With C Extension |
|----------------|---------|-------------------|-----------------|
| 100 cells | 80 ms/tick | 70 ms/tick | ~10 ms/tick |
| 1000 cells | 800 ms/tick | 650 ms/tick | ~80 ms/tick |
| 10000 cells | 8 s/tick | 6.5 s/tick | ~800 ms/tick |

---

## 9 — Implementation Roadmap

### Quarter 1 (Weeks 1-6)

| Week | Task | Deliverable | Risk |
|------|------|-------------|------|
| 1-2 | Extract `_dispatch.py` with dispatch table | All tests pass, 10-20% speedup | Low |
| 3-4 | Write `_vm_core.c` dispatch loop | C extension compiles, tests pass | Medium |
| 5-6 | Integrate C extension with fallback | Benchmarks show 10x speedup | Medium |

### Quarter 2 (Weeks 7-12)

| Week | Task | Deliverable | Risk |
|------|------|-------------|------|
| 7-8 | GRN step numpy matrix path | 6x speedup for large GRNs | Low |
| 9-10 | C simplex solver | 100x speedup for FBA | High |
| 11-12 | Population unification | Shared dispatch, 7-10x speedup | Medium |

### Quarter 3+ (Weeks 13+)

| Task | Priority | Notes |
|------|----------|-------|
| Rust + PyO3 rewrite of `_vm_core` | Medium | If C maintenance burden is too high |
| SIMD dispatch (AVX2/NEON) | Low | For batch operations on stacks |
| WebAssembly compilation | Low | Browser-based HelixLang execution |
| GPU population simulation | Low | CUDA/OpenCL for 100k+ cells |

---

## 10 — Code Structure Changes

### 10.1 New File Layout

```
src/helixlang/
  # Existing (orchestration layer)
  vm.py                    # Modified: imports _dispatch
  population.py            # Modified: imports _dispatch
  grn.py                   # Modified: imports _grn_step
  metabolism.py            # Modified: imports _simplex
  compiler.py              # Unchanged
  lexer.py                 # Unchanged
  parser.py                # Unchanged
  server.py                # Unchanged
  
  # New (hot loop modules — pure Python implementations)
  _dispatch.py             # NEW: opcode dispatch table
  _grn_step.py             # NEW: GRN step implementations
  _simplex.py              # NEW: simplex solver implementations
  _pop_dispatch.py         # NEW: population cell dispatch
  
  # New (C extension sources — not distributed in sdist)
  _vm_core.c               # NEW: C VM dispatch loop
  _vm_core.h               # NEW: shared C types
  _vm_core_module.c        # NEW: Python module definition
  _simplex_core.c          # NEW: C simplex implementation
```

### 10.2 Import Pattern

Every hot loop module follows the same pattern:

```python
# Any hot loop module
try:
    from helixlang._native_xyz import hot_function
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False
    from helixlang._py_xyz import hot_function
```

### 10.3 Testing Strategy

1. **All existing tests run against pure-Python fallback** (CI always)
2. **Same tests run against native extension** (CI with build step)
3. **Performance regression tests** added to `validation/benchmarks/`
4. **Fuzz testing** for C extension edge cases (malformed bytecode, etc.)

---

## 11 — Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| C extension segfaults on bad bytecode | High | Bounds checking in C; fuzz testing |
| C extension breaks on Python version upgrade | High | Pin CPython ABI; CI matrix tests |
| Maintenance burden of C code | Medium | Keep C code < 1000 lines; pure Python fallback always works |
| Build complexity on Windows | Medium | Use `scikit-build-core` + GitHub Actions Windows runner |
| C extension slower than expected | Low | Profile with `perf`; optimize hot cases first |

---

## 12 — Success Criteria

| Metric | Current | 30-day | 90-day | 180-day |
|--------|---------|--------|--------|---------|
| VM dispatch speed | 770 ns/op | 500 ns/op (table) | 80 ns/op (C) | 50 ns/op (optimized C) |
| Population 1000 cells | 800 ms/tick | 700 ms | 80 ms | 50 ms |
| FBA E. coli core | 50 ms | 50 ms | 0.5 ms | 0.3 ms |
| GRN 1000 nodes | 260 us | 200 us (table) | 15 us (numpy) | 10 us (C) |
| Native extension lines | 0 | 0 | ~800 | ~1200 |
| Pure Python fallback | 100% | 100% | 100% | 100% |
