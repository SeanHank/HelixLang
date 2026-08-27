# doc/36 — Architecture: Minimal Core + Plugin System + Native Acceleration

> **Status:** DRAFT
>
> **Depends on:** doc/12 (language wiring), doc/13 (performance report), doc/34 (architectural plan)
>
> **Goal:** Reduce the core to a minimal semantic compiler + VM (16 files, zero
> external deps, <50 ms import). All scientific modules become lazy-loaded plugins.
> Compute-critical hot paths within plugins are delegated to native C extensions
> with pure-Python fallbacks.

---

## 1 — Executive Summary

HelixLang currently has 55 files in `src/helixlang/` with every scientific
module imported eagerly. This causes:

1. **Slow imports** (~800 ms) — numpy, scipy, rdkit, matplotlib all loaded at startup
2. **No separation of concerns** — compiler, VM, and 50+ scientific modules tangled
3. **Cannot extend without modifying core** — new backends require editing `sim_runtime.py`
4. **Performance ceiling** — pure-Python VM dispatch at ~770 ns/op, GRN at ~33 us/step

This document proposes two coupled transformations:

**A. Plugin Architecture** — Core shrinks to 16 files (compiler + VM). Scientific modules
become plugins under `helixlang/plugins/`, loaded on-demand via a new `use` DSL statement.

**B. Native Acceleration** — The 4 hottest inner loops (VM dispatch, population dispatch,
GRN step, simplex pivot) are extracted into hot-loop modules replaceable by C extensions.

---

## 2 — Part A: Plugin Architecture

### 2.1 The Three Layers

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Application / CLI                             │
│  cli.py, server.py, __main__.py                        │
│  - parses user intent, dispatches to backends           │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Plugin Runtime                                │
│  plugin_registry.py — discovers + lazy-loads plugins    │
│  - no scientific code lives here                        │
│  - only the interface contract + loader                 │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Core Compiler + VM                            │
│  lexer → parser → AST → semantic → compiler → bytecode → VM│
│  - zero scientific dependencies                         │
│  - no numpy, no scipy, no rdkit                         │
│  - pure stdlib only                                     │
└─────────────────────────────────────────────────────────┘

    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  GRN     │ │  FBA     │ │  PK/PD   │ │  Pop     │
    │  plugin  │ │  plugin  │ │  plugin  │ │  plugin  │
    └──────────┘ └──────────┘ └──────────┘ └──────────┘
         ▲             ▲             ▲             ▲
         └─────────────┴─────────────┴─────────────┘
                   loaded on demand by
                   plugin_registry.py
```

### 2.2 Core = Minimum Viable Compiler

```
src/helixlang/
  # --- Layer 1: Core (zero external deps) ---
  __init__.py          # version string only
  __main__.py          # entry point
  ast_nodes.py         # AST node dataclasses
  lexer.py             # tokenizer
  parser.py            # recursive-descent parser
  semantic.py          # semantic analysis
  bytecode.py          # bytecode compiler + opcodes
  vm.py                # bytecode VM (dispatch loop)
  hxbc.py              # helix bytecode format
  type_system.py       # type inference (core)
  disassembler.py      # bytecode disassembly
  errors.py            # error types
  units.py             # unit system (pure math)
  provenance.py        # provenance tracking (dict-based)
  plugin_registry.py   # plugin discovery + lazy loader
  cli.py               # CLI entry (calls plugin_registry)
  server.py            # HTTP server (calls plugin_registry)
```

**17 files. Zero scientific dependencies. Imports in <50 ms.**

### 2.3 Plugin Interface Contract

```python
# helixlang/plugin_registry.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import importlib
from typing import Any, Callable

@dataclass
class PluginMeta:
    """Metadata about a plugin."""
    name: str                    # e.g. "grn", "fba", "population"
    version: str                 # e.g. "1.0.0"
    description: str
    keywords: list[str] = field(default_factory=list)
    extra_deps: list[str] = field(default_factory=list)

class HelixPlugin(ABC):
    """Base class all HelixLang plugins must implement."""

    @abstractmethod
    def meta(self) -> PluginMeta:
        """Return plugin metadata."""

    @abstractmethod
    def register(self, registry: 'PluginRegistry') -> None:
        """Register this plugin's backends, keywords, and handlers."""

    def on_load(self) -> None:
        """Called after register(). Optional setup."""

    def on_unload(self) -> None:
        """Called when plugin is unloaded. Optional cleanup."""


class PluginRegistry:
    """Discovers, loads, and dispatches to plugins.

    The ONLY module that knows about scientific plugins.
    Everything else (CLI, server, VM) goes through the registry.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, Any] = {}
        self._backends: dict[str, Callable] = {}
        self._keywords: dict[str, str] = {}
        self._loaded: set[str] = set()

    def discover(self) -> list[str]:
        """Scan helixlang.plugins.* for plugin names (does NOT load)."""
        ...

    def load(self, name: str) -> Any:
        """Lazy-load a plugin by name. Idempotent."""
        if name in self._loaded:
            return self._plugins[name]
        mod = importlib.import_module(f"helixlang.plugins.{name}")
        plugin = mod.PLUGIN
        plugin.register(self)
        plugin.on_load()
        self._plugins[name] = plugin
        self._loaded.add(name)
        return plugin

    def load_keyword(self, keyword: str) -> Any:
        """When helix source contains '#grn' or '#fba', load that plugin."""
        if keyword in self._keywords:
            return self.load(self._keywords[keyword])
        raise KeyError(f"Unknown helix keyword: #{keyword}")

    def register_backend(self, name: str, factory: Callable) -> None:
        self._backends[name] = factory

    def register_keyword(self, keyword: str, plugin_name: str) -> None:
        self._keywords[keyword] = plugin_name

    def get_backend(self, name: str) -> Callable:
        if name not in self._backends:
            for pname in self._plugins:
                p = self._plugins[pname]
                if hasattr(p, 'provides_backend') and p.provides_backend(name):
                    self.load(pname)
                    break
        return self._backends[name]
```

### 2.4 Helix DSL: `use` Statement

```helix
use grn
use fba "ecoli_core"
use population
use pk --optional
```

**Compilation flow:**
1. Lexer tokenizes `use grn`, `use fba`, `#gene`, `#regulate`, `#config`
2. Parser builds AST with `UseStatement(keyword="grn")` node
3. Semantic analyzer verifies plugins are registered
4. Compiler emits `OP_USE_PLUGIN "grn"`, `OP_USE_PLUGIN "fba"`
5. VM executes: calls `registry.load_keyword("grn")` → imports `plugins/grn`

### 2.5 Plugin Directory Layout

```
src/helixlang/plugins/
  __init__.py
  grn/
    __init__.py         # exports PLUGIN = GRNPlugin()
    grn_backend.py      # GRN simulation logic
  fba/
    __init__.py         # exports PLUGIN = FBAPlugin()
    fba_backend.py      # FBA / dFBA / flux balance
  population/
    __init__.py         # exports PLUGIN = PopulationPlugin()
    population_backend.py
  pk/
    __init__.py         # exports PLUGIN = PKPlugin()
    pk_backend.py       # pharmacokinetics / pharmacodynamics
  disease/
    __init__.py
    disease_ode.py
  annotation/
    __init__.py
    blast_backend.py
    kegg_backend.py
  gem/
    __init__.py
    gem_backend.py
  human/
    __init__.py
    human_backend.py
  apps/
    __init__.py
    (all app backends)
```

### 2.6 Example Plugin: GRN

```python
# helixlang/plugins/grn/__init__.py
from helixlang.plugin_registry import HelixPlugin, PluginMeta

class GRNPlugin(HelixPlugin):
    def meta(self) -> PluginMeta:
        return PluginMeta(
            name="grn", version="1.0.0",
            description="Gene Regulatory Network simulation",
            keywords=["grn", "regulate"],
            extra_deps=["numpy"],
        )

    def register(self, registry) -> None:
        registry.register_keyword("grn", "grn")
        registry.register_keyword("regulate", "grn")
        registry.register_backend("grn", self._create_grn)

    def _create_grn(self, config: dict):
        from helixlang.plugins.grn.grn_backend import GRNBackend
        return GRNBackend(config)

PLUGIN = GRNPlugin()
```

### 2.7 Dependency Isolation

```toml
# pyproject.toml
[project]
dependencies = []   # core: zero scientific deps

[project.optional-dependencies]
grn = ["numpy"]
fba = ["numpy", "scipy"]
population = []
pk = ["numpy"]
disease = ["numpy", "scipy"]
gem = ["numpy", "cobrapy"]
annotation = ["numpy", "rdkit"]
human = ["numpy", "scipy"]
apps = ["numpy", "matplotlib"]
all = ["helixlang[grn,fba,pk,disease,gem,annotation,human,apps]"]
dev = ["helixlang[all]", "pytest", "ruff", "mypy"]
```

```bash
pip install helixlang          # core only (~2 MB)
pip install helixlang[grn]     # with GRN support
pip install helixlang[all]     # everything (~150 MB)
```

---

## 3 — Part B: Native Acceleration

### 3.1 Performance Baseline (doc/13, measured 2026-08-26)

| Metric | Value | Bottleneck |
|--------|-------|-----------|
| VM dispatch | ~770 ns/op | Python `match` on `Op` IntEnum |
| GRN step (128 nodes) | ~33 us/step | Pure Python edge traversal |
| Simplex (E. coli core) | ~50 ms | Python list-of-lists iteration |
| Population (1000 cells) | ~800 ms/tick | Per-cell bytecode dispatch x N |
| Gray-Scott (128x128) | ~122 us/step | numpy (already optimized) |
| Compile throughput | ~710k codons/s | Already fast enough |

### 3.2 The 4 Critical Hot Paths

| Priority | Path | Current Cost | Why Python Hurts |
|----------|------|-------------|-----------------|
| **P0** | VM opcode dispatch | 770 ns/op | `match` on IntEnum + `_read_u8`/`_read_u16` per instruction |
| **P0** | Population per-cell dispatch | 770 ns/op x N | Same dispatch loop duplicated for N cells |
| **P1** | GRN edge accumulation | 0.26 us/node | Python dict iteration over edges per node |
| **P1** | Simplex pivot inner loop | O(rows x vars) | Python list-of-lists; numpy path helps but still Python-controlled |

### 3.3 What Already Has numpy Acceleration

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

### 3.4 Hot Loop Isolation Pattern

**Design principle:** Isolate hot inner loops into self-contained modules
that can be replaced with native code without touching orchestration logic.

```
+----------------------------------------------+
|  Orchestration Layer (pure Python)            |
|  vm.py, population.py, server.py, CLI        |
|  - tick loop, GRN wiring, snapshot, debug    |
+---------------------+------------------------+
                      |
          +-----------v-----------+
          |  Hot Loop Module       |
          |  _dispatch.py          |
          |  - dispatch_table      |
          |  - execute_chunk()     |
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

### 3.5 Module Decomposition

| Current Module | Extracted Hot Loop | New Module | Contents |
|---------------|-------------------|-----------|---------|
| `vm.py` | `_execute_pending` + `_dispatch` | `helixlang/_dispatch.py` | Op dispatch table, read_u8/u16, frame push/pop |
| `grn.py` | `step()` inner loop | `helixlang/_grn_step.py` | Edge accumulation, level update, decay |
| `metabolism.py` | `_simplex_max` inner loop | `helixlang/_simplex.py` | Pivot selection, ratio test, tableau update |
| `population.py` | `_execute_cell` inner loop | `helixlang/_pop_dispatch.py` | Per-cell bytecode execution |

---

## 4 — Native Extension: VM Dispatch (P0)

### 4.1 Technology Choice

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **C + CPython API** | Maximum control, no runtime deps, ~3x faster than Cython | Manual reference counting | **Selected** |
| Cython | Easier to write, `.pyx` syntax | Runtime dep, less control | Rejected |
| Rust + PyO3 | Memory safety, modern toolchain | New ecosystem, larger binary | Future option |
| C++ + pybind11 | RAII, templates | Heavier ABI, C++ runtime | Rejected |
| Numba JIT | Zero code change | JIT warmup, no control flow opt | Rejected |

### 4.2 C Extension Architecture

```
helixlang/
  _vm_core.c          # C implementation of dispatch loop
  _vm_core.h          # Shared types (Chunk, Frame, Cell)
  _vm_core_module.c   # Python module definition
  vm.py               # Python fallback + imports _vm_core if available
```

### 4.3 Key C Function

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

### 4.4 Expected Performance

| Metric | Current | With C Extension | Speedup |
|--------|---------|-----------------|---------|
| VM dispatch | 770 ns/op | ~50-80 ns/op | **10-15x** |
| Population (1000 cells) | 800 ms/tick | ~80-120 ms/tick | **7-10x** |
| Total 16 examples | 16.3 ms | ~3-5 ms | **3-5x** |

### 4.5 Fallback Strategy

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

## 5 — Native Extension: GRN Step (P1)

### 5.1 Current GRN Step

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

### 5.2 Proposed: Sparse Matrix-Vector Multiply

```python
# _grn_step.py
import numpy as np

def grn_step_matrix(nodes: dict, edges: dict, decay: float) -> list[str]:
    """GRN step via sparse matrix multiply (numpy path)."""
    names = list(nodes.keys())
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}
    row, col, data = [], [], []
    for target, edge_list in edges.items():
        for edge in edge_list:
            row.append(idx[target])
            col.append(idx[edge.source])
            data.append(edge.weight)
    W = np.zeros((n, n))
    W[row, col] = data
    levels = np.array([nodes[name].level for name in names])
    new_levels = levels + W @ levels - decay * levels
    new_levels = np.clip(new_levels, 0.0, 1.0)
    triggered = []
    for i, name in enumerate(names):
        nodes[name].level = float(new_levels[i])
        if new_levels[i] > nodes[name].threshold:
            triggered.append(name)
    return triggered
```

### 5.3 Expected Performance

| GRN Size | Current | numpy matvec | Speedup |
|----------|---------|-------------|---------|
| 128 nodes | 33 us | ~5 us | **6x** |
| 1000 nodes | ~260 us | ~15 us | **17x** |
| 5000 nodes | ~1.3 ms | ~40 us | **32x** |

---

## 6 — Native Extension: Simplex Solver (P1)

### 6.1 Current State

- Pure Python: `_simplex_max()` — list-of-lists, ~50 ms for E. coli core
- numpy: `_simplex_max_numpy()` — ndarray ops, ~5 ms for E. coli core
- scipy: `linprog()` — for models > 500 reactions, ~2 ms

### 6.2 Proposed: C Simplex Core

```
helixlang/
  _simplex.c           # C simplex implementation
  _simplex.h           # API header
  _simplex_module.c    # Python module definition
  metabolism.py        # imports _simplex or falls back to Python
```

### 6.3 Expected Performance

| Model | Reactions | Pure Python | numpy | C Extension | Speedup vs Python |
|-------|-----------|-------------|-------|-------------|-------------------|
| E. coli core | 95 | 50 ms | 5 ms | ~0.5 ms | **100x** |
| iML1515 | 2712 | 15 s | 200 ms | ~20 ms | **750x** |
| Human generic | 10000+ | N/A | >5 s | ~200 ms | **N/A** |

---

## 7 — Population Backend Unification

### 7.1 Problem

`population.py` duplicates the entire VM dispatch loop (~200 lines) for
per-cell execution. Population simulations multiply the dispatch cost by N.

### 7.2 Proposed: Shared Dispatch Core

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

### 7.3 Expected Performance

| Population Size | Current | After Unification | With C Extension |
|----------------|---------|-------------------|-----------------|
| 100 cells | 80 ms/tick | 70 ms/tick | ~10 ms/tick |
| 1000 cells | 800 ms/tick | 650 ms/tick | ~80 ms/tick |
| 10000 cells | 8 s/tick | 6.5 s/tick | ~800 ms/tick |

---

## 8 — Implementation Roadmap

### Phase 1: Plugin Interface + Core Extraction (Weeks 1-2)

| Task | Files | Risk |
|------|-------|------|
| Create `plugin_registry.py` | `src/helixlang/plugin_registry.py` | Low |
| Create `helixlang/plugins/` directory | `src/helixlang/plugins/__init__.py` | Low |
| Add `OP_USE_PLUGIN` to bytecode | `bytecode.py`, `vm.py` | Low |
| Add `use` statement to lexer/parser | `lexer.py`, `parser.py`, `ast_nodes.py` | Low |
| Extract `_dispatch.py` with dispatch table | `src/helixlang/_dispatch.py` | Low |
| Write plugin interface tests | `tests/test_plugin_registry.py` | Low |

### Phase 2: First Plugin + Dispatch Table (Weeks 3-4)

| Task | Files | Risk |
|------|-------|------|
| Migrate GRN → `plugins/grn/` | `plugins/grn/__init__.py`, `grn_backend.py` | Low |
| Update `sim_runtime.py` to use registry for GRN | `sim_runtime.py` | Medium |
| Verify all GRN tests pass | `tests/test_grn.py` | Low |
| Benchmark dispatch table: expect 10-20% speedup | `validation/benchmarks/` | Low |

### Phase 3: C Extension + Remaining Plugins (Weeks 5-10)

| Week | Task | Deliverable | Risk |
|------|------|-------------|------|
| 5-6 | Write `_vm_core.c` dispatch loop | C extension compiles, tests pass | Medium |
| 5-6 | Migrate FBA → `plugins/fba/` | `plugins/fba/` | Medium |
| 7-8 | Integrate C extension with fallback | Benchmarks show 10x speedup | Medium |
| 7-8 | GRN step numpy matrix path | 6x speedup for large GRNs | Low |
| 9-10 | C simplex solver | 100x speedup for FBA | High |
| 9-10 | Migrate remaining plugins (pk, disease, gem, human, apps) | `plugins/*/` | Medium |

### Phase 4: Unification + Cleanup (Weeks 11-14)

| Week | Task | Deliverable | Risk |
|------|------|-------------|------|
| 11-12 | Population backend unification | Shared dispatch, 7-10x speedup | Medium |
| 13-14 | Remove all eager scientific imports from `sim_runtime.py` | Import time <50ms | Medium |
| 13-14 | Update `__init__.py` to only export core API | Clean public API | Low |

### Future (Week 15+)

| Task | Priority | Notes |
|------|----------|-------|
| Rust + PyO3 rewrite of `_vm_core` | Medium | If C maintenance burden is too high |
| SIMD dispatch (AVX2/NEON) | Low | For batch operations on stacks |
| WebAssembly compilation | Low | Browser-based HelixLang execution |
| GPU population simulation | Low | Apple MPS for 100k+ cells |

---

## 9 — File Layout: Final State

```
src/helixlang/
  # --- Core (17 files, zero external deps) ---
  __init__.py
  __main__.py
  ast_nodes.py
  lexer.py
  parser.py
  semantic.py
  bytecode.py
  vm.py
  hxbc.py
  type_system.py
  disassembler.py
  errors.py
  units.py
  provenance.py
  plugin_registry.py   # NEW
  cli.py
  server.py

  # --- Hot Loop Modules (pure Python implementations) ---
  _dispatch.py         # NEW: opcode dispatch table
  _grn_step.py         # NEW: GRN step implementations
  _simplex.py          # NEW: simplex solver implementations
  _pop_dispatch.py     # NEW: population cell dispatch

  # --- C Extension Sources (not distributed in sdist) ---
  _vm_core.c           # NEW: C VM dispatch loop
  _vm_core.h           # NEW: shared C types
  _vm_core_module.c    # NEW: Python module definition
  _simplex_core.c      # NEW: C simplex implementation

  # --- Plugins (lazy-loaded, each has own deps) ---
  plugins/
    __init__.py
    grn/
    fba/
    population/
    pk/
    disease/
    annotation/
    gem/
    human/
    apps/
```

---

## 10 — Import Time Impact

| Metric | Current | After Plugin Arch |
|--------|---------|-------------------|
| `import helixlang` time | ~800 ms | <50 ms |
| `pip install helixlang` size | ~150 MB (with numpy) | ~2 MB (core only) |
| `pip install helixlang[all]` size | ~150 MB | ~150 MB (same) |

---

## 11 — Success Criteria

| Metric | Current | 30-day | 90-day | 180-day |
|--------|---------|--------|--------|---------|
| Core files | 55 (all mixed) | 17 (core) + plugins | 17 + plugins + C | 17 + plugins + C |
| Core import time | ~800 ms | <50 ms | <50 ms | <50 ms |
| Core external deps | numpy, scipy, rdkit... | 0 | 0 | 0 |
| VM dispatch speed | 770 ns/op | 500 ns/op (table) | 80 ns/op (C) | 50 ns/op (opt C) |
| Population 1000 cells | 800 ms/tick | 700 ms | 80 ms | 50 ms |
| FBA E. coli core | 50 ms | 50 ms | 0.5 ms | 0.3 ms |
| GRN 1000 nodes | 260 us | 200 us (table) | 15 us (numpy) | 10 us (C) |
| Native extension lines | 0 | 0 | ~800 | ~1200 |
| Pure Python fallback | 100% | 100% | 100% | 100% |

---

## 12 — Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Plugin load order issues | Medium | Registry discovers plugins at startup, loads on demand |
| `use` keyword conflicts | Low | `use` is new, no existing helix code uses it |
| Performance overhead of lazy loading | Low | Plugin loaded once, cached in registry |
| C extension segfaults on bad bytecode | High | Bounds checking in C; fuzz testing |
| C extension breaks on Python version upgrade | High | Pin CPython ABI; CI matrix tests |
| Maintenance burden of C code | Medium | Keep C code < 1000 lines; pure Python fallback always works |
| Testing complexity | Medium | Each plugin has its own test suite; core tests stay simple |
