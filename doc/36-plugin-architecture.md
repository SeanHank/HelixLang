# doc/36 — Full Architecture Restructure: Minimal Core + Plugin System + Multi-Stack Native Acceleration

> **2026-08-28 — Rust/PyO3 removed.** The Rust/PyO3 accelerated backend (impl_pyomod, the pyomod/ Cargo crate, hl_dispatch, maturin) has been deleted; only pure-Python and C/Cython/numpy native backends remain. See §5.

> **Status:** DRAFT (replaces previous draft, 2026-08-27)
>
> **Depends on:** doc/12 (language wiring), doc/13 (performance report), doc/34 (architectural plan)
>
> **Goal:** **Complete rebuild** of the project file structure with **no
> backward compatibility** for the old layout. Reduce the core to a **minimal
> semantic compiler** (lexer → parser → AST → semantic → bytecode → VM) with
> **zero scientific dependencies**. Every scientific/simulation module becomes a
> **lazy-loaded plugin** pulled into the helix source only when a `use` statement
> (or a `#keyword`) requires it. Compute-critical hot paths inside plugins are
> delegated to **native accelerators** (C/CPython API, Cython, Numba,
> numpy) with pure-Python reference implementations — specified in detail per
> technology. Concurrently, every silent fallback in the current codebase is
> converted to an explicit error or a declared opt-in (§3ξ).

---

## 1 — Executive Summary

HelixLang currently ships **135 Python files** under `src/helixlang/`
(46 top-level core/scientific files + 89 across subpackages), all eagerly
imported by `__init__.py`. This causes:

1. **Slow imports** (~800 ms) — numpy, scipy, rdkit, matplotlib, cobra, torch all
   pulled at startup via `__init__.py`'s monolithic `__all__`.
2. **No separation of concerns** — the compiler/VM are tangled with 50+ scientific
   modules in a single namespace.
3. **Cannot extend without touching core** — every new backend required editing
   `sim_runtime.py`.
4. **Hard-coded coupling** — DSL `#keywords` (`#gene`, `#drug`, `#person`,
   `#qsp_binding`, ...) are resolved by bespoke import chains rather than a
   uniform plugin registry.
5. **Performance ceiling** — pure-Python VM dispatch at ~770 ns/op is the
   bottleneck for every cell in population simulations.

This document specifies a **clean-slate reorganization**:

**A. Minimal Core** — a self-contained semantic compiler + bytecode VM in
`src/helixlang/core/` (~16 files, **zero external deps**, <50 ms import).

**B. Plugin System** — every scientific capability moves to
`src/helixlang/plugins/<name>/`, each a self-contained unit exposing the
`HelixPlugin` contract. Plugins are loaded lazily, driven by a new `use`
statement in the helix DSL plus auto-detection from `#keywords`.

**C. Native Acceleration** — the hottest inner loops (VM opcode dispatch,
GRN step, simplex/flux pivot, population cell dispatch, diffusion kernels) are
extracted into isolated hot-loop modules. Each hot loop is implemented across **4
technology stacks** (C extension, Cython, Numba) with a shared numpy
batch path and a pure-Python reference. Each stack is specified with file layout,
API, build hooks, expected speedup, and backend-selection rules. **In parallel,
the restructure converts every silent fallback in the current codebase into an
explicit, typed error (or a declared opt-in)** so the system never silently
computes at lower fidelity than the author intended (§3ξ).

> **Compatibility note:** This is a **breaking rewrite of the public layout**.
> The old `helixlang.<module>` import surface and the monolith `__init__.py`
> exports are intentionally removed, and **no legacy import shims are shipped** —
> the 6 root shim packages and the `_legacy_reexports` opt-in bridge were fully
> removed (2026-08-28): the default namespace is completely clean, and all code
> imports from the canonical `helixlang.core.*` / `helixlang.plugins.*` paths.
> The helix `.helix` source language itself is preserved and remains forward
> compatible — only the Python packaging/layout changes.

---

## 2 — Part A: Minimal Semantic Compiler Core

### 2.1 Design Decision: What Stays in Core

The core must be able to do one thing well: **compile and run a helix program's
language semantics** — the genetic DSL, bytecode, and VM — with NO scientific
modeling capability of its own. Anything that imports numpy/scipy/rdkit/cobra/
torch/matplotlib is **excluded** from core by rule.

**Core responsibilities (language only):**
- Lexing / parsing / AST of the DSL (genes, traits, `#end`, `use`, control flow).
- Type inference and semantic analysis.
- Bytecode emission + freeze (ABI `OPCODE_VERSION`).
- Bytecode disassembly + VM interpretation for **pure-language ops**.
- Unit system, provenance tracking, error types.
- Plugin registry (the *only* module that knows plugins exist).

**Core forbidden deps:** numpy, scipy, rdkit, cobra, torch, matplotlib, biopython,
reedsolo, flask, esm. Only the Python standard library is allowed.

### 2.2 Final Core Layout

```
src/helixlang/
  __init__.py            # re-exports core public API only; version string
  core/                  # ← Layer 1: the minimal semantic compiler + VM
    __init__.py
    lexer.py             # tokenizer (DSL keywords, #keywords, use)
    parser.py            # recursive-descent parser
    ast_nodes.py         # AST dataclasses
    semantic.py          # semantic analysis + type inference
    bytecode.py          # opcodes + compiler + ABI freeze (OPCODE_VERSION)
    vm.py                # dispatch loop (pure-Python reference; native via _accel)
    hxbc.py              # helix bytecode container format (serialize/deserialize)
    type_system.py       # static type system
    disassembler.py      # bytecode → text
    units.py             # unit conversion + constants (pure math)
    provenance.py        # provenance tracking (dict/JSON based)
    errors.py            # HelixError hierarchy
    plugin_registry.py   # discovery + lazy loader + #keyword→backend map
    use_stmt.py          # canonicalizes `use X` → PluginSpec
  plugins/               # ← Layer 2: all scientific modules (lazy)
    __init__.py          # empty, or entry-point discovery hook
    grn/
    fba/
    pk/
    disease/
    annotation/
    gem/
    human/
    apps/
  cli.py                 # Layer 3: CLI (thin, calls core + registry)
  __main__.py            # entry: python -m helixlang
```

**16 core files. Zero scientific dependencies. `import helixlang` < 50 ms.**

### 2.3 Core Public API (what `__init__.py` exports)

Only language/runtime primitives + the registry surface:

```python
# src/helixlang/__init__.py
from helixlang.core.version import __version__
from helixlang.core.errors import (HelixError, LexError, ParseError,
    SemanticError, CompileError, RuntimeHelixError, RegulationError)
from helixlang.core.parser import parse_program
from helixlang.core.bytecode import compile_program, OPCODE_VERSION
from helixlang.core.vm import VM, Program
from helixlang.core.units import (TIME_TICK_MIN, ticks_to_min,
    decay_from_half_life_ticks, ...)
from helixlang.core.plugin_registry import PluginRegistry, HelixPlugin, PluginMeta

# registry singleton (does NOT import any plugin)
registry = PluginRegistry()
registry.discover(entry_points=True)  # lazy: discovers names only
```

Everything scientific (`GRN`, `CellPopulation`, `FluxBalanceAnalysis`,
`VirtualPatient`, ...) is **removed from the root namespace**. It is reachable only
through `registry`, or by importing a plugin directly.

---

## 3 — Part B: The Plugin System

### 3.1 Plugin Contract

```python
# src/helixlang/core/plugin_registry.py
# Layer 1 — the ONLY file that knows plugins exist. Zero science.

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import importlib.metadata, importlib
from typing import Any, Callable

@dataclass
class PluginMeta:
    name: str                       # "grn"
    version: str                    # "1.0.0"
    description: str
    keywords: list[str] = field(default_factory=list)   # ["grn","regulate"]
    extra_deps: list[str] = field(default_factory=list) # ["numpy"]
    provides: list[str] = field(default_factory=list)   # backend names
    native: bool = False            # whether a native accel is bundled

class HelixPlugin(ABC):
    @abstractmethod
    def meta(self) -> PluginMeta: ...
    @abstractmethod
    def register(self, registry: "PluginRegistry") -> None: ...
    def on_load(self) -> None: ...      # optional setup
    def on_unload(self) -> None: ...    # optional cleanup

class PluginRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, PluginMeta] = {}   # name → meta (known, unloaded)
        self._plugins: dict[str, HelixPlugin] = {}
        self._backends: dict[str, Callable] = {}
        self._keywords: dict[str, str] = {}       # "#gene" → "grn"
        self._loaded: set[str] = set()

    # Discovery: build a NAME→PLUGIN map WITHOUT importing plugin modules.
    # Sources: (a) local plugins/ dir by convention,
    #          (b) installed packages exposing an entry point group.
    def discover(self, entry_points: bool = False) -> list[str]: ...

    def load(self, name: str) -> HelixPlugin:         # idempotent lazy load
    def unload(self, name: str) -> None:
    def resolve(self, spec: "PluginSpec") -> HelixPlugin:  # if dep missing → nice error
    def load_keyword(self, keyword: str) -> HelixPlugin:   # "#gene" → grn
    def register_backend(self, name: str, factory: Callable) -> None:
    def get_backend(self, name: str) -> Callable:
```

### 3.2 Helix DSL: the `use` Statement

New grammar token `use` plus existing `#keyword`s both route through the registry.

```helix
# Pure-language program (no plugins needed):
gene EGFP { promotes: }           # compiles with core only

# Scientific program — pull in plugins as needed:
use grn                # numpy-backed GRN
use fba "ecoli_core"   # FBA, with a model alias
use population         # population simulation
use pk --optional      # load only if deps available

# Or rely on #keyword auto-detection:
#gene g1              → registry.load_keyword("gene")
#drug cisplatin       → registry.load_keyword("drug")
#person pt1           → registry.load_keyword("person")
#qsp_binding ...      → registry.load_keyword("qsp_binding")
```

**Compilation flow for `use`:**
1. Lexer emits `USE` token + a string operand (plugin name) and optional flags.
2. Parser builds `UseStmt(name=..., alias=..., optional=...)`.
3. Semantic analyzer validates the name is **known** to the registry
   (`registry.discover()` was run at CLI startup) — unknown = `SemanticError`.
4. Bytecode compiler emits `OP_USE_PLUGIN "grn"`.
5. VM executes `OP_USE_PLUGIN`: calls `registry.resolve(spec)`; if `--optional`
   and deps are missing, it degrades to a no-op marking the plugin unavailable;
   otherwise a missing plugin aborts with a clear `PluginMissingError`.

**Auto-detection of `#keywords`:** during semantic analysis, each `#keyword` in
the AST is looked up in `registry._keywords`. If found, the semantic analyzer
emits the corresponding `OP_USE_PLUGIN` **before** the first use of that keyword.
This guarantees plugin backends are present before the VM reaches a
backend-dependent op.

### 3.3 Plugin Directory Layout

```
src/helixlang/plugins/
  __init__.py                       # EMPTY (prevents implicit package import of children)
  grn/           __init__.py, backend.py, native.py
  fba/           __init__.py, backend.py, simplex.py, native.py
  pk/            __init__.py, backend.py, pk_ode.py, pd_ode.py
  disease/       __init__.py, backend.py, disease_ode.py
  annotation/    __init__.py, backend.py, blast.py, kegg.py, ec_mapping.py
  gem/           __init__.py, backend.py, gapfill.py, sbml.py, community.py
  human/         __init__.py, backend.py, virtual_patient.py, drug.py, ...
  apps/          __init__.py, backend.py, digital_evolution.py, ...
```

Each plugin's `__init__.py` is trivial:

```python
# src/helixlang/plugins/grn/__init__.py
from helixlang.core.plugin_registry import HelixPlugin, PluginMeta

class GRNPlugin(HelixPlugin):
    def meta(self) -> PluginMeta:
        return PluginMeta(name="grn", version="1.0.0",
                          description="Gene regulatory network simulation",
                          keywords=["gene", "regulate"], extra_deps=["numpy"],
                          provides=["grn"], native=True)
    def register(self, registry) -> None:
        for kw in ["gene", "regulate"]:
            registry.register_keyword(kw, "grn")
        registry.register_backend("grn", lambda cfg: self._new_backend(cfg))
    def _new_backend(self, cfg):
        from helixlang.plugins.grn.backend import GRNBackend
        return GRNBackend(cfg, native=self._native)
    def _native(self):
        from helixlang.plugins.grn import native   # optional C .so
        return native if native.available else None

PLUGIN = GRNPlugin()
```

### 3.4 Dependency Isolation

```toml
# pyproject.toml (reorganized)
[project]
dependencies = []                       # core only → ~2 MB install
requires-python = ">=3.11,<3.12"

[project.optional-dependencies]
# Core tooling
dev  = ["pytest>=7.4,<9", "pytest-cov>=4.1,<6", "pytest-xdist>=3.5,<4",
        "pytest-benchmark>=4.0,<5"]
# One extra per plugin so users install only what they run:
grn        = ["numpy>=1.26,<3"]
fba        = ["numpy>=1.26,<3", "scipy>=1.12,<2"]
pk         = ["numpy>=1.26,<3", "scipy>=1.12,<2"]
disease    = ["numpy>=1.26,<3", "scipy>=1.12,<2"]
annotation = ["numpy>=1.26,<3", "biopython>=1.83,<2", "reedsolo>=1.7,<2"]
gem        = ["numpy>=1.26,<3", "cobra>=0.26,<1"]
human      = ["numpy>=1.26,<3", "scipy>=1.12,<2", "rdkit>=2026.3,<2027"]
apps       = ["numpy>=1.26,<3", "matplotlib>=3.8,<4"]
web        = ["flask>=3.0,<4"]
ml         = ["numpy>=1.26,<3", "esm>=2.0", "torch>=2.0"]
all        = ["helixlang[grn,fba,pk,disease,annotation,gem,human,apps,web,ml]"]
# Native builds (Cython/C). `native` is built-at-install when toolchain exists,
# else the pure-Python/numpy path is used automatically.
native     = ["cython>=3.0", "setuptools>=68", "wheel", "build"]
```

```bash
pip install helixlang            # core only (~2 MB)
pip install helixlang[grn]       # + GRN
pip install helixlang[all]       # everything
pip install helixlang[all,native]  # everything + compiled C/Cython accel
```

### 3.5 Import-Time Model

- `registry.discover()` at CLI startup builds a `name→meta` map by scanning the
  local `plugins/` package + installed entry-points. It **does not import** any
  plugin module.
- The first `use`/`#keyword` triggers `registry.load(name)` → import that one
  plugin and its deps. Subsequent uses are cache hits.
- Core import stays <50 ms; a helix program importing only the language never
  touches numpy.

---

## 3ξ — Cross-Cutting Mandate: Replace Every Silent Fallback with an Explicit Error

> **Applies to ALL restructuring.** During the rebuild, **every** silently
> degrading path in the current codebase becomes an **explicit, deterministic
> error** (or an explicitly-declared opt-out — never an implicit one). "It
> probably works-ish without X" is no longer acceptable; the program must either
> fail loudly with a precise diagnostic, or the author must state up-front
> "I accept reduced functionality" via an explicit declaration.

### 3ξ.1 Catalog of current silent fallbacks (removed during restructure)

| # | Current behavior | Location | New behavior during restructure |
|---|------------------|----------|----------------------------------|
| F1 | `numpy` missing → GRN/population silently use pure-Python path | `grn.py`, `population.py`, `evolution.py` | `ImportError` with actionable message; or explicit `Conf[fast]` opt-in |
| F2 | `scipy` missing → ODE integrator silently drops to fixed-step Euler | `grn.py` (`integrate_ode`) | Explicit `MissingOptionalError` unless `--approx-euler` acknowledged |
| F3 | `cobra` missing → FBA silently falls back to built-in core model | `metabolism.py`, `gem/*` | explicit `ModelMissingError` listing the required `[gem]` extra + canonical `import path` |
| F4 | `rdkit` missing → SMILES parsing silently limited to MW | `human/drug.py` | explicit `PluginMissingError` for the `human`/`molecular` capability |
| F5 | `esm`/`torch` missing → structure prediction silently falls back to Chou-Fasman/physics | `protein_structure_predictor.py`, `kinetics/*` | explicit error; a documented `--low-fidelity` flag is the *only* way to opt in |
| F6 | `matplotlib` missing → PNG output silently replaced by PPM | `apps/*`, `viz` | explicit error; alternative writer chosen only via explicit output-format directive |
| F7 | unknown `#keyword` silently ignored / dropped | lexer → semantic | `SemanticError: unknown keyword #x` (semantic analyzer must know every keyword) |
| F8 | `#keyword` without its plugin installed → silently no-op | plugin resolution | `PluginMissingError` naming the plugin + the extra to install |
| F9 | bytecode ABI mismatch → silently-incompatible run | `hxbc.py` | hard `ABIVersionError` at load; never a wrong-result run |
| F10 | `OP_USE_PLUGIN`/`use` for unregistered plugin → no-op | `vm.py` | `PluginMissingError` (absent dependency) — fails the run, not degrades |
| F11 | stack underflow in VM → silently ignored ("prototype robustness") | `vm.py` | `RuntimeHelixError` (matches doc/06 production note); device==None reads throw |
| F12 | GRN node referenced but absent → `KeyError` panic or silent default | `grn.py` | explicit `UnknownNodeError` with node name + closest matches |

### 3ξ.2 Error model and rationale

- **New error hierarchy** (all subclass `HelixError`, defined in `core/errors.py`):

  ```
  HelixError
   ├─ LexError / ParseError / SemanticError / CompileError
   ├─ RuntimeHelixError
   │    ├─ StackUnderflowError
   │    └─ UnknownNodeError
   ├─ PluginError
   │    ├─ PluginMissingError      # not installed / not discoverable
   │    ├─ PluginDependencyError   # installed but its optional dep is absent
   │    └─ PluginConflictError     # two plugins claim the same backend/keyword
   ├─ ModelMissingError            # FBA/GEM model data absent
   ├─ ABIVersionError              # bytecode ABI mismatch
   └─ NativeBackendError           # compiled impl failed to load/passed bad data
  ```

- **Why:** silent fallbacks mask real problems — a user thinks they got the
  accurate model but actually got a low-fidelity proxy. That is a **scientific
  correctness bug**, not a convenience. Making it explicit guarantees a
  reproducible, trustworthy result (consistent with the project's Build→Proof
  mandate).
- **Every fallback now REQUIRES a user-visible declaration** to trigger, NEVER an
  implicit environment condition. Each new error message must contain:
  1. Missing component name,
  2. the exact `pip install helixlang[<extra>]` command that fixes it,
  3. (where a degraded mode exists) the explicit flag/`use`-declaration to opt in.

### 3ξ.3 Explicit opt-in for genuinely-optional fidelity

Reduced-fidelity modes are retained only where the experimental design *wants*
them, and only via an explicit declaration — never silently:

```helix
use fba "ecoli_core"          # full model required; ModelMissingError if absent
use pk --approx-euler          # explicitly accepts fixed-step integration
use human --low-fidelity       # explicitly accepts MW-only SMILES
use grn --pure-python          # explicitly skips native/numpy acceleration
```

`--pure-python` / `--approx-euler` / `--low-fidelity` are **documented,
versioned capability flags**. Omitting them when the high-fidelity path is
unavailable is an error, not a warning.

### 3ξ.4 Enforcement in the build

- **Static check:** a `find_silent_fallbacks` lint passes over `core/` and
  `plugins/` during CI, flagging `except ImportError: pass`,
  bare `except:`, and `if not <dep>:`-style degradation branches (they must call
  `raise MissingOptionalError(...)` or carry an explicit opt-in guard).
- **Test rule:** every plugin ships a test asserting that **removing its core
  optional dep raises PluginMissingError/PluginDependencyError** (not a silent
  fallback).
- **Runtime:** the loader never silently swaps backends; a failed native load
  raises `NativeBackendError` unless the program declared `--pure-python`.
  (This deliberately reverses the "auto-fallback ladder" language in §§4/5 below —
  see 3ξ.5.)

### 3ξ.5 Interaction with the native acceleration fallback ladder (§§4–5)

The earlier reduction described a *silent* "native > numba > numpy > python"
ladder. **That ladder is inverted for mandatory fidelity**: the default is
`--pure-python`-declared or explicit-backend-selected. Specifically:

- Default behavior: **no silent degradation**. If a program needs numpy/high-fi
  behavior and the backend is absent, it errors with the fix command.
- The *only* automatic selection is the **exact same algorithms, same accuracy**,
  differing solely in *implementation speed* (e.g. Python vs Cython vs C for the
  same GRN step math). This is a pure performance switch, not a fidelity switch,
  so it may auto-select without changing results.
- Any switch that changes *algorithm/fidelity* (numpy→pure-python GRN that
  changes numerics, scipy RK45→Euler, cobra→core-model, rdkit→MW-only,
  esm→Chou-Fasman) **must** be gated behind an explicit opt-in flag and default
  to an error.
- Therefore §4's loader is reworded: it selects among **equivalent-fidelity**
  implementations of the *same* math, and the fallback ladder only applies within
  that equivalence class. Crossing an equivalence-class boundary requires an
  explicit declaration (see 3ξ.3), enforced by `find_silent_fallbacks`.

### 3ξ.6 Migration checklist (folded into the roadmap, §10)

- [ ] Audit all modules for the F1–F12 patterns; replace each with an explicit
      error or an explicit opt-in guard.
- [ ] Implement the `PluginError`/`ModelMissingError`/`ABIVersionError`/
      `NativeBackendError` hierarchy in `core/errors.py`.
- [ ] Add the `--approx-euler`, `--low-fidelity`, `--pure-python` capability
      flags to the `use` grammar + their `OP_USE_PLUGIN` encoding.
- [ ] Implement `UnknownKeywordError` in the semantic analyzer (F7).
- [ ] Add `find_silent_fallbacks` lint + per-plugin missing-dep tests (3ξ.4).
- [ ] Add a `+missing_dep` fixture test matrix so CI loads each plugin both with
      and without its optional dependencies.
- [ ] Update validation/report.md to record the active backend + fidelity level
      so the evidence chain states exactly what was computed (no ambiguity).

---

## 4 — Part C: Native Acceleration — Overview

### 4.1 The 4 Identified Hot Paths + Diffusion

| Path | Module (current) | Current Cost | Why Python Hurts |
|------|------------------|-------------|------------------|
| **P0** VM opcode dispatch | `vm.py` | ~770 ns/op | `match` on `Op` + `_read_u8/_read_u16` per instr |
| **P0** Population per-cell dispatch | `population.py` | 770 ns/op × N cells | dispatch duplicated per cell |
| **P1** GRN edge accumulation | `grn.py` | ~260 us / 1000 nodes | Python dict edge iteration |
| **P1** Simplex/FBA pivot | `metabolism.py` | ~50 ms (E. coli core) | Python list-of-lists |
| **P1** Reaction–diffusion kernel | `reaction_diffusion.py` | ~122 us/step (numpy) | numpy already helps; C can add more |

### 4.2 Unified Hot-Loop Isolation Pattern

Every hot loop is extracted into a **standalone module** in an `_accel/`
namespace with a **uniform loader** that picks the fastest available
implementation at import time:

```
src/helixlang/_accel/
  __init__.py            # scanner: picks native/Cython × numpy/Python
  _loaders.py            # shared "try native → try numpy → python" resolver
  dispatch/
    __init__.py
    backend.py           # dispatch table API
    impl_python.py       # pure-Python reference
    impl_numpy.py        # numpy batch (where applicable)
    impl_cython.pyx      # Cython
    impl_cext.c          # CPython C API
    impl_numba.py        # Numba njit (optional, JIT)
  grn_step/
    __init__.py  backend.py  impl_python.py  impl_numpy.py
    impl_cython.pyx  impl_cext.c  impl_numba.py
  simplex/
    __init__.py  backend.py  impl_python.py  impl_numpy.py
    impl_cython.pyx  impl_cext.c  impl_numba.py
  pop_dispatch/
    __init__.py  backend.py  impl_python.py  impl_numpy.py  impl_cext.c
  diffusion/
    __init__.py  backend.py  impl_python.py  impl_numpy.py  impl_cext.c
```

**Loader contract (uniform across all stacks) — equivalena-fidelity only:**

```python
# src/helixlang/_accel/_loaders.py
import importlib, os

# ONLY picks among implementations of the SAME algorithm with IDENTICAL
# numerics (speed-only switch). It NEVER silently crosses a fidelity boundary
# (e.g. numpy-GRN → pure-python-GRN-that-changes-rounding, scipy-RK45 → Euler,
# cobra → built-in model, rdkit → MW-only, esm → Chou-Fasman). Crossing such a
# boundary must be declared explicitly by the caller (e.g. a `use` capability
# flag, §3ξ.3) — otherwise the caller raises NativeBackendError /
# PluginMissingError and the run FAILS rather than degrades.
#
# If the requested backend is unavailable, this module raises, it does NOT
# reselect. Program authors choose fidelity via explicit `use` flags; this
# selector only maps a chosen fidelity level to its fastest available impl.

def choose_backend(pkg: str, prefer: str | None = None) -> str:
    """Return the backend module name for a hot-loop package.
    Priority (configurable via env HELIX_ACCEL): native > numba > numpy > python.
    'native' = any compiled impl (cython/cext) present on disk.
    Fidelity is fixed by the caller; this only picks speed.
    """
    order = (prefer or os.environ.get("HELIX_ACCEL", "native")).split(",")
    for tag in order:
        if tag == "native":
            for impl in ("impl_cext", "impl_cython"):
                if _importable(pkg, impl): return impl
        else:
            mod = f"{pkg}.impl_{tag}"
            if _importable(pkg, f"impl_{tag}"): return mod
    # If the *chosen fidelity* impl is absent, error loudly — no silent swap.
    raise NativeBackendError(
        f"No implementation of {pkg} for requested backend '{order[-1]}'. "
        f"Rebuild with `pip install helixlang[native]` or declare `--pure-python`."
    )

def load_hot(pkg: str) -> Any:
    name = choose_backend(pkg)
    return importlib.import_module(f"{pkg}.{name}")
```

Each hot-loop backend exposes the **same callable signature**, so orchestration
code never changes when swapping stacks.

#### 4.2.1 Build hooks

Compiled backends (`impl_cext.c`, `impl_cython.pyx`) are built
by an optional build backend. On a machine with a compiler, `pip install
helixlang[native]` compiles them into `helixlang/_accel/*/build/*.so`. When the
toolchain is absent, the `.so` is missing and the loader **raises
`NativeBackendError`** (or, if the program declared `use ... --pure-python`,
selects the pure-Python impl explicitly). CI builds a "native" wheel; PyPI ships
a "py" wheel.

```toml
# pyproject.toml build hooks (setuptools + optional extension build)
[tool.setuptools]
ext-modules = [
  "helixlang._accel.dispatch.impl_cext = helixlang._accel.dispatch:build_cext",
  "helixlang._accel.grn_step.impl_cython = ...",
  ...
]
```

---

## 5 — Per-Technology Deep-Dives

This section specifies **each of the 3 technology stacks** in detail for the
**VM dispatch** hot path (the P0 that everything else depends on), then notes
the per-path adaptations. All three stacks coexist behind the loader; you pick one
per install.

### 5.1 Stack 1 — C + CPython C API

**Best for:** maximum control, no runtime dep, lowest overhead (`METH_*` slots,
direct stack access). Chosen as the **primary native path**.

#### 5.1.1 File layout

```
src/helixlang/_accel/dispatch/
  impl_cext.c           # C dispatch loop + module
  setup_build.py        # setuptools Extension build script
  test_cext.c           # (optional) C-level smoke test
```

#### 5.1.2 Data model (shared with Python VM)

The VM keeps bytecode as a contiguous `bytes` buffer and a `double[]` operand
stack. Exposure to C via a small struct + method table (avoids boxing every op in
a Python object):

```c
/* impl_cext.c */
#include <Python.h>
#include <stdint.h>

typedef struct {
    const uint8_t *code;      /* bytecode buffer             */
    Py_ssize_t      code_len;
    double         *stack;     /* operand stack               */
    Py_ssize_t     *sp;        /* stack pointer              */
    /* gene-frame table (call targets) */
    const uint32_t *gene_table;
    Py_ssize_t      gene_count;
    /* opcode quota for tick-based execution */
    Py_ssize_t      quota;
} VmCtx;

/* One function runs a whole quota of ops in C → ~10-15x fewer Python
   transitions than calling dispatch once per op. */
static PyObject* vm_run_tick(PyObject *self, PyObject *args) {
    Py_buffer code; VmCtx ctx; double stack[1024]; Py_ssize_t sp = 0;
    if (!PyArg_ParseTuple(args, "y*n", &code, &ctx.quota)) return NULL;
    ctx.code = code.buf; ctx.code_len = code.len; ctx.stack = stack; ctx.sp = &sp;
    int rc = vm_exec(&ctx);                 /* hot switch loop in C */
    if (rc) { PyErr_SetString(PyExc_RuntimeError, "vm_exec failed"); return NULL; }
    /* copy sp back to Python VM object, return number of ops consumed */
    return Py_BuildValue("n", ctx.quota);
}

/* ---- the hot loop: compiled switch on contiguous uint8 opcodes ---- */
static inline int vm_exec(VmCtx *c) {
    Py_ssize_t ip = 0;
    while (ip < c->code_len && c->quota > 0) {
        uint8_t op = c->code[ip++];
        switch (op) {
            case OP_PUSH_CONST:  c->stack[(*c->sp)++] = _read_f64(c->code, &ip); break;
            case OP_ADD:  c->stack[*c->sp - 2] += c->stack[*c->sp - 1]; (*c->sp)--; break;
            case OP_CALL_GENE:  { uint32_t idx = _read_u32(c->code, &ip);
                                   /* push gene frame, set ip = gene_table[idx] */
                                   break; }
            /* ~54 language opcodes total */
            case OP_USE_PLUGIN:  { /* callback into registry; return special code */ }
            default: return -1;
        }
        c->quota--;
    }
    return 0;
}
```

#### 5.1.3 Type-specialized variant (further speedup)

For population use-cases where every cell runs identical code, an optional
**precompiled chunk** specialization compiles the bytecode to a C function pointer
`double(*)(const double*, double*)` per-program. This removes even the
`switch`/`quota` overhead — turning per-op dispatch into straight-line native code.

#### 5.1.4 Expected performance

| Metric | Python | C (switch) | C (specialized) |
|--------|--------|-----------|----------------|
| VM dispatch | 770 ns/op | ~55 ns/op | ~20 ns/op |
| Speedup | 1× | **14×** | **~38×** |
| Pop 1000 cells/tick | 800 ms | ~75 ms | ~30 ms |

#### 5.1.5 Fallback (explicit, never silent)

If `impl_cext.so` is absent (no compiler), the loader does **not** silently
reselect: it raises `NativeBackendError` with the rebuild command, unless the
program declared `use ... --pure-python` (in which case the pure-Python impl is
chosen explicitly and recorded in provenance). Because C/Cython/C implement the
*identical* GRN/simplex math, the speed-only selection may be automatic without
changing fidelity; any *fidelity* reduction requires the explicit capability flag
(§3ξ.3). GIL is held by default; optional `Py_BEGIN_ALLOW_THREADS` region around
pure-computation zones for parallelism (disabled for stack-sensitive ops).

---

### 5.2 Stack 2 — Cython

**Best for:** fastest authoring of compiled hot loops with minimal boilerplate;
near-C speed with Python-level ergonomics. Great for the **GRN step**, **simplex
pivot**, and **diffusion** kernels (vectorized math is easy to express).

#### 5.2.1 File layout + `.pyx`

```python
# src/helixlang/_accel/grn_step/impl_cython.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp

cpdef list grn_step_edge(double[::1] levels, cnp.int32_t[::1] edges_src,
                         cnp.int32_t[::1] edges_dst, double[::1] weights,
                         double decay, double[::1] thresholds,
                         int n_nodes):
    """Cython GRN step: edge accumulation as array-of-COO, fully typed.
    Returns list of triggered node indices."""
    cdef int e, i, src, dst
    cdef double acc
    cdef cnp.int32_t[:] trig = np.zeros(n_nodes, dtype=np.int32)
    cdef int nt = 0
    cdef double[:] new = np.copy(levels)
    # accumulation loop (types fixed → C speed)
    for e in range(edges_src.shape[0]):
        src = edges_src[e]; dst = edges_dst[e]
        new[dst] += weights[e] * levels[src]
    # decay + clip + threshold
    for i in range(n_nodes):
        new[i] = (new[i] - decay * levels[i])
        if new[i] < 0: new[i] = 0.0
        elif new[i] > 1: new[i] = 1.0
        if new[i] > thresholds[i]:
            trig[nt] = i; nt += 1
    # copy back
    for i in range(n_nodes):
        levels[i] = new[i]
    return [int(trig[t]) for t in range(nt)]
```

#### 5.2.2 Build

```toml
[tool.setuptools.ext-modules]
helixlang._accel.grn_step.impl_cython = "helixlang._accel.grn_step:build_cython"
```

`build_cython()` runs `cythonize("impl_cython.pyx")` and returns a
`setuptools.Extension`. Compile-to-wheel is enabled *only* under the `[native]`
extra; otherwise a stub raising `ImportError` makes the loader fall back.

#### 5.2.3 Expected performance

| Path | Python | Cython | Speedup |
|------|--------|--------|---------|
| GRN 1000 nodes | ~260 us | ~8 us | **~32×** |
| Simplex E. coli | ~50 ms | ~0.6 ms | **~85×** |
| Diffusion 128×128 | ~122 us | ~15 us | **~8×** |

#### 5.2.4 Notes
- `boundscheck=False` etc. require the **caller to guarantee** valid indices —
  the Python wrapper validates once before the hot loop.
- Cython is a **build-time** dep only; the compiled `.so` has **no Cython runtime**
  dependency at run time.

---

### 5.4 Stack 4 — Numba (JIT)

**Best for:** **zero host-build-required** acceleration on machines where numpy
is already present — you just decorate the existing numpy implementation. Not a
replacement for C in standalone-wheel scenarios (requires LLVM at runtime),
hence tagged **optional**.

#### 5.4.1 File layout + `njit`

```python
# src/helixlang/_accel/diffusion/impl_numba.py
import numpy as np
try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:                      # pragma: no cover
    HAVE_NUMBA = False

if HAVE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _step(u, v, Du, Dv, f, k, dt, dx2, dy2):
        n, m = u.shape
        nu = np.empty_like(u); nv = np.empty_like(v)
        for i in prange(1, n-1):
            for j in range(1, m-1):
                lapl_u = (u[i+1,j]+u[i-1,j]+u[i,j+1]+u[i,j-1]-4*u[i,j])/dx2
                lapl_v = (v[i+1,j]+v[i-1,j]+v[i,j+1]+v[i,j-1]-4*v[i,j])/dy2
                uvv = u[i,j]*v[i,j]*v[i,j]
                nu[i,j] = u[i,j] + dt*(Du*lapl_u - uvv + f*(1-u[i,j]))
                nv[i,j] = v[i,j] + dt*(Dv*lapl_v + uvv - (f+k)*v[i,j])
        nu[0,:]=nu[-1,:]=nu[:,0]=nu[:,-1]=0; nv[0,:]=nv[-1,:]=nv[:,0]=nv[:,-1]=0
        return nu, nv
    def step(u,v,Du,Dv,f,k,dt,dx2,dy2):
        return _step(np.ascontiguousarray(u), np.ascontiguousarray(v),
                     Du,Dv,f,k,dt,dx2,dy2)
else:
    def step(*a, **k):
        from .impl_python import step as _p
        return _p(*a, **k)
```

#### 5.4.2 Expected performance

| Path | numpy | Numba | Speedup vs numpy |
|------|-------|-------|------------------|
| Diffusion 128×128 | ~122 us | ~25 us | ~5× |
| GRN 5000 nodes | ~40 us | ~10 us | ~4× |

#### 5.4.3 Notes
- `cache=True` persists compiled code across runs, so the ~0.5–2 s JIT warmup is
  paid once.
- Numba requires LLVM at import; if `numba` isn't installed, `HAVE_NUMBA` is
  False and JIT acceleration is **unavailable — this is an explicit
  `NativeBackendError`** unless `use ... --pure-python` was declared. Loader
  ordering (**native > numba > numpy > python**) picks speed among
  equivalent-fidelity impls only; it never silently swaps a fidelity class (Numba
  only wins when no compiled `.so` exists *and* the chosen fidelity permits it).

---

### 5.5 Cross-cutting rules for all stacks

1. **Same public signature per backend** — orchestration never branches on stack.
2. **Validate once in Python, hot-loop in native** — index/type checks happen
   outside the loop for `boundscheck=False`/C safety.
3. **Determinism preserved** — every backend is a pure function over explicit
   inputs (no hidden RNG state); the `validation/` golden SHA256 suite must pass
   identically (within float tolerance) on all *equivalent-fidelity* stacks.
4. **No silent degradation (per §3ξ)** — a broken `.so` (ABI mismatch, Python
   upgrade) raises `NativeBackendError` at load, listing the rebuild command,
   rather than degrading to numpy/Python at first run. A `use ... --pure-python`
   declaration is the only route to a lower-fidelity impl — and it is recorded,
   not implicit.
5. **Env override** `HELIX_ACCEL=python` forces the pure path for debugging —
   still explicit (it is set by the operator), and still logged in provenance.

---

## 6 — Per-Hot-Path Stack Assignments (recommended matrix)

| Hot path | Python ref | numpy | **C** | **Cython** | **Numba** |
|----------|-----------|-------|-------|-----------|----------|
| VM dispatch (P0) | ✔ | — | **primary** | ✔ | —(control flow) |
| Pop cell dispatch (P0) | ✔ | batch | **primary** (reuse) | ✔ | — |
| GRN step (P1) | ✔ | matvec | ✔ | **primary** | ✔ |
| Simplex pivot (P1) | ✔ | ndarray | ✔ | **primary** | ✔ |
| Diffusion kernel (P1) | ✔ | ✔ (exists) | ✔ | ✔ | **primary** (no build) |

---

## 7 — Migration of Existing Modules into Plugins

| Old module | New plugin | Backend layout | Keywords |
|-----------|-----------|----------------|----------|
| `grn.py`, `sparse_grn.py` | `plugins/grn/` | `backend.py` (GRN, SparseGRN) | `#gene`, `#regulate` |
| `metabolism.py`, `central_dogma.py`, `codon_table.py` | `plugins/fba/` | `backend.py` (FBA, dFBA), `simplex.py` | `#media`, `#metabolite`, `#reaction`, `#sim` |
| `pharmacokinetics.py`, `pharmacodynamics.py`, `ddi.py`, `drug.py` | `plugins/pk/` | `backend.py`, `pk_ode.py`, `pd_ode.py` | `#drug`, `#pd_effect`, `#person` |
| `disease.py`, `disease_ode_models.py`, `recovery.py` | `plugins/disease/` | `backend.py`, `disease_ode.py` | `#disease`, `#qsp_binding`, `#tumor_biopsy` |
| `annotation/*.py` | `plugins/annotation/` | `backend.py`, `blast.py`, `kegg.py`, `ec_mapping.py` | `#gene_id`, `#enzyme` |
| `gem/*.py` | `plugins/gem/` | `backend.py`, `gapfill.py`, `sbml.py`, `community.py` | `#gem`, `#genome` |
| `human/*.py` (36 files) | `plugins/human/` | `backend.py`, `virtual_patient.py`, `drug.py`, ... | `#person`, `#drug`, `#immune_config`, `#endocrine_config` |
| `apps/*.py` (23 files) | `plugins/apps/` | `backend.py`, `digital_evolution.py`, ... | `#evolve`, `#lsystem`, `#crispr`, `#morphogen`, `#species` |
| `population.py`, `evolution.py`, `stochastic.py` | `plugins/apps/` (or `plugins/population/`) | `backend.py` | `#evolve` |
| `reaction_diffusion.py`, `environment.py`, `flow.py`, `morphology_3d.py` | `plugins/apps/` | `backend.py` | `#field`, `#patch`, `#morphogen` |
| `protein_*`, `kinetics/*`, `omics/*` | `plugins/ml/` | `backend.py` | `#type` |

---

## 8 — File Layout: Final State

```
src/helixlang/
  __init__.py                        # core public API only (<50 ms)
  __main__.py
  cli.py
  core/                              # 16 files, ZERO external deps
    __init__.py lexer.py parser.py ast_nodes.py semantic.py
    bytecode.py vm.py hxbc.py type_system.py disassembler.py
    units.py provenance.py errors.py plugin_registry.py use_stmt.py
  _accel/                            # hot-loop implementations (all stacks)
    __init__.py _loaders.py
    dispatch/   (backend.py, impl_python.py, impl_numpy.py,
                 impl_cython.pyx, impl_cext.c, impl_numba.py)
    grn_step/   (backend.py, impl_python.py, impl_numpy.py,
                 impl_cython.pyx, impl_cext.c, impl_numba.py)
    simplex/    (...)
    pop_dispatch/(...)
    diffusion/  (...)
  plugins/                           # lazy, per-plugin deps
    __init__.py (empty)
    grn/ fba/ pk/ disease/ annotation/ gem/ human/ apps/ ml/
  data/
```

**Everything outside `core/` is optional and none of it is imported by default.**

---

## 9 — Performance Targets (measured via validation suite)

| Metric | Current | Core-only | With native (C/Cython) |
|--------|---------|-----------|------------------------|
| `import helixlang` | ~800 ms | **<50 ms** | <50 ms |
| Core install size | ~150 MB | **~2 MB** | ~2 MB + `.so` |
| VM dispatch | 770 ns/op | 770 ns/op (py) | **~55 ns/op** |
| Pop 1000 cells | 800 ms/tick | — | **~75 ms/tick** |
| FBA E. coli core | 50 ms | — | **~0.6 ms** |
| GRN 1000 nodes | 260 us | — | **~8 us** |
| Diffusion 128×128 | 122 us | — | **~15-25 us** |

The validation suite (`validation/`, 67 benchmarks) is **stack-agnostic**: it runs
against whichever backend the loader chose and records the backend in the result
JSON provenance.

---

## 10 — Implementation Roadmap

### Phase 1 — Core extraction + registry (Weeks 1-2)
1. Create `core/` with the 16 files; move compiler/VM/language files verbatim.
2. Write `plugin_registry.py` + `use_stmt.py`.
3. Add `use` token to lexer/parser; `OP_USE_PLUGIN` to bytecode + VM (incl. the
   `--pure-python` / `--approx-euler` / `--low-fidelity` capability flags, §3ξ.3).
4. Implement the explicit error hierarchy in `core/errors.py` (`PluginError`,
   `ModelMissingError`, `ABIVersionError`, `NativeBackendError`, ... §3ξ.2).
5. Trim `__init__.py` to core API; confirm `import helixlang` <50 ms.
6. Add `tests/test_plugin_registry.py`, `tests/test_use_stmt.py`, and the
   `find_silent_fallbacks` CI lint (§3ξ.4).

### Phase 2 — Dispatch table + first plugins + fallback purge (Weeks 3-4)
1. Extract `_accel/dispatch/`; pure-Python backend only (matches current VM).
2. Migrate `grn.py` → `plugins/grn/`; wire `#gene` keyword.
3. Migrate `metabolism.py` → `plugins/fba/`; wire `#media`/`#sim`.
4. Add `impl_numpy.py` for GRN step (matvec).
5. **Audit grn.py/metabolism.py/population.py for F1–F12 silent fallbacks and
   convert each to an explicit error or declared opt-in** (incl. replacing
   stack-underflow-ignore ≠ F11 and unknown-keyword-drop ≠ F7).
6. Add per-plugin missing-dep tests: load each plugin with and without its core
   extra → assert `PluginDependencyError`/`ModelMissingError`, never a silent run.
7. Update `validation/` so benchmarks import through the registry and record
   backend + fidelity in provenance; all 67 pass.

### Phase 3 — Native backends (Weeks 5-10)
1. Cython `impl_cython.pyx` for GRN step + simplex (build under `[native]`).
2. C `impl_cext.c` VM dispatch (P0) + population dispatch.
3. Numba `impl_numba.py` diffusion + GRN.
4. Loader with `HELIX_ACCEL` override + `NativeBackendError` on unavailable
   backend (§4.2) + determinism checks at equivalent fidelity.
5. Confirm §3ξ: absent native ⇒ explicit error or declared `--pure-python`,
   never a transparent swap.

### Phase 4 — Full plugin migration + packaging (Weeks 11-14)
1. Migrate `human/*` (36), `apps/*` (23), `annotation/*`, `gem/*`, `kinetics/*`,
   `omics/*` into plugins — **converting every remaining silent fallback
   (rdkit, cobra, esm, torch, matplotlib — F3–F6) to explicit errors/opt-ins**.
2. Rework `pyproject.toml` extras; drop monolithic `__init__` exports.
3. Ship dual wheels (py + native) from CI.
4. Update README/CONTRIBUTING/doc/* to the new layout + the no-silent-fallback
   policy.

### Phase 5 — Hardening (Week 15+)
- Fuzz the native VM against randomized bytecode.
- Benchmark matrix across all stacks; publish in doc/13.
- Re-run `find_silent_fallbacks` across the final tree → 0 hits; confirm each
  reduced-fidelity path is reachable only via a declared, provenance-recorded
  capability flag.
- ~~Optional `_legacy_reexports.py` shim~~ — **superseded (2026-08-28):** the
  shim and the six root shim packages were removed entirely; the default
  namespace is clean by design.

---

## 11 — Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Breaking the public `helixlang.*` import surface | High | Documented breaking change; no backwards-compat shim shipped — the default namespace is intentionally clean |
| Plugin load order / keyword collisions | Medium | Single registry; discover-then-load; `#keyword` asserted once |
| Native ABI breakage on Python upgrade | High | `NativeBackendError` at load (explicit, §3ξ); CI builds each stack; `--pure-python` opt-in documented |
| C segfault on malicious bytecode | High | Bounds-check in wrapper; fuzz tests; specialized chunks disabled for untrusted input |
| Build toolchain absent on user machine | Medium | Ship prebuilt native wheel + py wheel; explicit error (not silent fallback) with rebuild command |
| Numba JIT warmup / LLVM runtime | Low | `cache=True`; optional extra; absence is an explicit `NativeBackendError`, never silent |
| Missing optional dep at runtime does a wrong-fidelity computation | High | §3ξ mandate: every degradation is an explicit error or a declared opt-in; `find_silent_fallbacks` CI lint gates it |
| Maintenance burden (3 stacks) | Medium | Keep 1 primary native (C/Cython); shared tests per backend |

---

## 12 — Success Criteria

| Criteria | Definition |
|----------|-----------|
| Minimal core | Core = compiler+VM+registry only; `import helixlang` < 50 ms; zero non-stdlib deps |
| Plugin isolation | No scientific import reachable from `core/`; each plugin importable alone |
| Lazy loading | Running a pure-language program never imports numpy |
| Hot-loop parity | Equivalent-fidelity backends pass the same validation suite within float tolerance |
| Speedup | ≥10× on VM dispatch, ≥30× GRN, ≥80× simplex vs pure Python |
| Backward-compat shim | **None — removed (2026-08-28).** The root shim packages and `_legacy_reexports` were deleted; the default namespace is clean, no legacy paths resolve |
| **No silent fallbacks** | `find_silent_fallbacks` lint passes; every missing dep/ABI/backend raises a typed `HelixError` with a fix command; zero `except ImportError: pass` / bare `except:` / `if not <dep>` degradation branches remain in `core/` or `plugins/` |
| Explicit opt-in only | Every reduced-fidelity path is reachable ONLY via a declared `use` capability flag (`--pure-python`, `--approx-euler`, `--low-fidelity`), recorded in provenance |
| Provenance records fidelity | Every computed result's evidence chain states the active backend + fidelity level (no ambiguous "default" computation) |

---

## 13 — Implementation Status

Live-as-built tracker (updated per milestone; roadmap items left unchecked in
§10 are not yet complete).

### Phase 1 — Core extraction + registry — **DONE (2026-08-27)**
- [x] `src/helixlang/core/` with registry + use-statement + explicit error
  hierarchy (`Plugin*Error`, `ABIVersionError`, `UnknownKeywordError`, ...).
- [x] `use` token wired through lexer/parser/semantic; `OP_USE_PLUGIN` (0x14)
  emitted by the compiler and dispatched by the VM; VM activates the plugin
  backend on entry (independent of gene entry point).
- [x] ABI version (`OPCODE_VERSION=1`) embedded in the `CHNK` section; decode
  raises `ABIVersionError` on mismatch.
- [x] `find_silent_fallbacks` linter (F1/F2/F12; readable `except` detail),
  wired as a pre-commit hook and a hard CI job over `core/` + `_accel/` plus an
  informational full-tree audit.
- [x] `test_core_restructure.py`, `test_accel_foundation.py`,
  `test_plugins.py` green.

### Phase 2 — Dispatch table + first plugins — **DONE (2026-08-27)**
- [x] `_accel/` (dispatch/ + grn_step/) with `impl_python`/`impl_numpy` stacks.
- [x] `plugins/` package created; `plugins/grn/` bundled plugin (keywords
  `gene`/`regulate`, `extra="grn"`, numpy check, `PLUGIN` provider).
- [x] `Registry.discover()` lazy bundled-import; `get_registry()` auto-discovers;
  activated-flag waive generalised to *any* declared capability flag the plugin
  honours (`--pure-python` waives missing numpy for the equivalent-fidelity GRN
  ref; §3ξ.3/§3ξ.5).
- [x] Per-plugin missing-dep tests (`tests/test_plugins.py`): absent numpy
  raises `PluginDependencyError` unless explicitly opted in.
- [x] Migrate `metabolism.py` → `plugins/fba/` (`#media`/`#sim`): bundled
  provider with numpy (hard) + cobra (SBML-import, `--low-fidelity` opt-in)
  checks; `#use fba` activates the `FluxBalanceAnalysis` backend through the
  registry; `media`/`sim` keywords route to `fba`.
- [x] Record backend + fidelity in provenance (§3ξ.6 / roadmap §10.2.7):
  `Registry.fidelity()`/`active()` accessors; `build_provenance` gains an
  optional `fidelity` field (absent by default, so benchmark-16 goldens stay
  stable); new `provenance_from_registry()` records active plugin backends +
  fidelity class for registry-imported benchmarks.

### Phase 3 — Native backends — **DONE (2026-08-27)**
- [x] Loader with `HELIX_ACCEL` override + `NativeBackendError` on absent
  backend (§4.2 item 5): `_accel/_loaders.py` (`choose_backend`/`load_hot`/
  `backend_for`, priority `native > numba > numpy > python`, env override) +
  `NativeBackendError` in `core/errors.py`.
- [x] Determinism at equivalent fidelity (§4.2/§5.5 item 5): tests assert the
  numpy / python GRN hot-loop kernels produce the same triggering and levels
  within float tolerance, and that a full `step_accel` run is deterministic-
  equivalent when switching `HELIX_ACCEL` between backends.
- [x] No-silent-swap contract (§3ξ.3 item 6): absent native (no `.so`) raises
  `NativeBackendError` even through the consumer (`grn_step.backend.step`); the
  explicit `--pure-python` declaration selects the pure-Python impl and never
  errors.
- [x] Numba JIT stack for the GRN step hot loop (P1, part of item 3):
  `grn_step/impl_numba.py` (optional, lazy import; never selected by default;
  raises   `NativeBackendError` if numba absent).  Numba-byte-equivalent to the
  reference within float tolerance; selectable via `HELIX_ACCEL=numba`.
- [x] Cython `impl_cython.pyx` for the GRN step hot loop (item 1, P1) — built
  into `grn_step/*.so` by `python -m helixlang._accel.build`; loader picks it
  under the `native` tag.
- [x] C `impl_cext.c` for the GRN step hot loop (item 2, P1 boundary — the
  Cython/C/cext native path) — same in-place build; numerically byte-identical
  to the reference (verified against python/numpy/cython).
- [x] In-place native build backend (`_accel/build.py`; `NativeBackendError.
  rebuild_cmd` = `python -m helixlang._accel.build`): compiles `.pyx` via
  Cython + `.c` via setuptools into the source tree, guarded so default `pip
  install` stays pure-Python (no compiler required) and an absent toolchain
  yields a clear hint + nonzero exit.
- [x] Native parity/determinism tests: `step_accel` is deterministic-equivalent
  across `cext`/`cython`/`numpy`/`python` (exact trace, levels ≤ ~1e-16);
  tests auto-skip per interpreter when the compiled `.so` isn't loadable.
- [x] Numba diffusion stack (item 3, P1): new `_accel/diffusion/` hot-loop package
  with `impl_python`, `impl_numpy` and `impl_numba` stacks mirroring
  `reaction_diffusion.GrayScott` (5-point Laplacian, U·V² reaction, clamp,
  borders preserved).  Byte-identical across all three (max diff 0.0); numba is
  optional/lazy, raises `NativeBackendError` if absent; guarded equivalence +
  determinism tests (`test_accel_foundation.py`).  Self-contained — production
  call sites untouched, FBA/golden paths safe.
- [x] Cython `impl_cython.pyx` for the **simplex** hot loop (item 1): new
  `_accel/simplex/` package (`impl_python`, `impl_numpy`, `impl_cython.pyx`),
  byte-identical port of `_simplex_max` (Bland's rule, reduced-cost entering
  test, ratio test with smallest-index tie-break, in-place pivot
  normalization + elimination).  python/numpy/Cython all produce the same
  status, final basis and a byte-identical tableau (max diff 0.0); guarded
  equivalence + native-loader tests.  Self-contained — production
  `metabolism.simplex()` call sites untouched, so FBA/golden paths stay safe.
- [x] C `impl_cext.c` for **VM dispatch (P0)** + **population dispatch** (item
  2): native backend for `_accel/dispatch/` (`run_quota` single-cell + `run_many`
  batch per-cell population dispatch on the shared bytecode+kernel), byte-
  identical to `impl_python` (same opcodes, stack, IEEE-754, quota accounting
  incl. HALT=0 ops, unhandled-op `NotImplementedError`, empty-pop `IndexError`).
  Built by `python -m helixlang._accel.build`; guarded parity/determinism tests.

Phase 3 (GRN Cython/C, simplex Cython, VM+population C, numba diffusion) is now
**DONE** in its entirety.

### Phase 4 — Silent-fallback conversion + packaging — **IN PROGRESS (2026-08-27)**
- [x] Shared reduced-fidelity opt-in helper `core/fidelity.py`: module-level
  `opt_in(flag, allow=, env=)` / `require(...)` gate; checks the explicit
  parameter, then `Registry.has_capability(flag)`, then the documented
  `HELIX_ALLOW_LOW_FIDELITY` env var.  Passing the gate lets a genuine reduced-
  fidelity path run; otherwise the caller raises the typed error (§3ξ.3).
- [x] `gem/community._solve_single`, `gem/ecgem._solve_growth`/`_solve_fluxes`:
  FBA solver crashes no longer silently read as 0.0 / `{}` — raise
  `ModelMissingError` unless `--low-fidelity` is declared.
- [x] `human/gem_human.load_from_sbml`: a failed human-GEM load no longer
  silently substitutes the E. coli core model — raises `ModelMissingError`
  unless `--low-fidelity` (dead-in-package branch; sibling modules present).
- [x] `human/simulation._load_base_model`: a failed load of a user-requested
  `base_model_path` no longer falls back to E. coli core — raises
  `ModelMissingError` unless `--low-fidelity`; the no-path core default (the
  documented default engine) is retained.
- [x] `vm.py` GEM re-solve paths: a failed FBA re-solve no longer leaves a stale
  `_growth_rate_gem` — raises `ModelMissingError` unless `--low-fidelity`.
- [x] `human/virtual_patient`: 10 lazy-import guards (`_import_doc32` +
  `_ensure_hematology`) no longer silently skip bundled `human/*` modules —
  re-raise an actionable `ImportError` naming the module + reinstall hint
  (dead-in-package branches; all sibling modules ship in the wheel).
- [x] `kinetics/sequence_predictor.get_esm2_embedding`: missing ESM-2 no longer
  silently substitutes AA-composition features — raises `PluginDependencyError`
  (`[ml]` extra) unless `--low-fidelity`.
- [x] `sim_runtime._run_population` GEM auto-attach: a failed requested-GEM
  pipeline no longer silently degrades to the E. coli core proxy — raises
  `ModelMissingError` unless `--low-fidelity`.  Documented-benign paths left
  intact: per-species Monod default during population seeding and PK-inference
  → default values (both explicit, non-wrong-result design; §10 triage).
- [x] `protein_structure_predictor`: ESM3 absence error message now carries the
  `pip install 'helixlang[ml]'` fix + the `--low-fidelity` opt-in hint; stale
  "silently falls back to Chou-Fasman" docstring corrected (the function
  already raises, so this is a message/accuracy fix, not a behavior add).
- [x] `pyproject.toml` extras reworked to the per-plugin `== module-name` scheme
  (backward-compatible): `grn, fba, pk, disease, annotation, gem, human, apps,
  web, ml` + `all`, plus retained `dev, fast, viz, bio, native` (§3.4 / roadmap
  extras lines).
- [x] Plugin migration of `human/*`, `apps/*`, `annotation/*`, `kinetics/*`,
  `omics/*` into `plugins/` + dropping monolithic `__init__` exports — all six
  packages migrated into `plugins/{annotation,apps,gem,human,kinetics,omics}/`
  with a single registry (`core/registry.py`), `use`-statement activation, and
  per-plugin extras. The six root-level shim packages
  (`src/helixlang/{human,apps,annotation,gem,kinetics,omics}/`) that once
  aliased `plugins.*` have been **fully removed** (2026-08-28): the default
  namespace is clean and code imports `helixlang.plugins.*` directly.
- [x] Dual-wheel (py + native) CI shipping — `ci.yml` gains a `build-wheels` job
  (pure-Python wheel + native wheel via `HELIX_BUILD_NATIVE=1` with
  `--no-isolation`, uploading `dist/`); the release job builds sdist + py wheel
  + native wheel. `release.py` `build()` now emits sdist + both wheels.
- [x] README/CONTRIBUTING/doc updated to the plugin layout + the
  no-silent-fallback policy.

Phase 4 (full plugin migration + dual-wheel shipping) is now **DONE — the
no-silent-fallback policy is enforced across `core/` + `_accel/` + the full
tree**. Tests run in the project env (3.11 + Cython for
native builds) and the canonical 3.13 test env; 67/67 validation benchmarks
remain green.

### Phase 5 — Hardening — **DONE (2026-08-27)**
- [x] Fuzz the native VM against randomized bytecode — `tests/test_vm_fuzz.py`
  (46 tests, batch + single-cell quotas, opcode/operand/constant fuzzing; cext
  vs numpy vs python parity). The fuzzer **caught a real C out-of-bounds
  segfault** on a bare trailing `OP_PUSH_CONST` (read past the code buffer) in
  `_accel/dispatch/impl_cext.c`; fixed with a bounds check that raises
  `IndexError` (parity with `impl_python`), rebuilt in-place.
- [x] Benchmark matrix across all stacks; publish in doc/13 — new
  `benchmarks/bench_accel_matrix.py` runs dispatch / grn_step / simplex /
  diffusion hot loops across cext / cython / numba / numpy / python with
  `--json` output; published as doc/13 §3.7 (table + §6 raw JSON + §7 harness
  list). Highlights: dispatch cext 0.50 µs vs python 7.38 µs (≈15×); simplex
  cython 78.5 µs vs python 84.4 µs; diffusion numba 5.4 µs vs python 3.9 ms.
- [x] **Removed all legacy import shims (2026-08-28):** the six root shim
  packages (`src/helixlang/{human,apps,annotation,gem,kinetics,omics}/`) and the
  opt-in `_legacy_reexports.py` bridge (plus `tests/test_legacy_reexports.py`)
  were deleted outright. The default namespace is now completely clean — the
  legacy top-level plugin import paths no longer resolve at all, and all code
  uses `helixlang.plugins.*`. The custom `build_py` prune for the bridge and the
  `packages.find` shim exclusions in `pyproject.toml` were removed accordingly.
- [x] Re-run `find_silent_fallbacks` across the final tree → 0 hits — the F4
  string-assignment heuristic is gated by a `# SILENTBENIGN` marker
  (`core/find_silent_fallbacks.py`); all 33 known-benign sites annotated with a
  reason. CI now runs the full-tree audit as a **hard gate** with `--fail`
  (removed `continue-on-error`). `find_silent_fallbacks src --fail` exits 0
  ("No silent-fallback patterns found").
