# 38 — Compiler Modernization & Plugin Platform

> Source-of-truth · LanguageConfig · Grammar Registry · Plugin API ·
> Observability · Artifact versioning · Type/Unit systems · Engine split · Fuzzing
>
> Status: **plan (pending implementation)** · 2026-08-29 · baseline `2026.8.5`

This document turns twelve architecture goals into a concrete, codebase-anchored
analysis and a phased implementation plan.  Every claim below cites the current
file/line so the reader can verify the state before changing it.

## 0 — The twelve goals

| # | Goal | Doc § |
|---|------|-------|
| 1 | DNA is the single source of truth → incremental JIT (edit → invalidate gene → recompile → patch function) | §4 |
| 2 | A single `LanguageConfig` (codon table / stop codons / start codons / translation semantics) shared by Parser, Compiler, VM | §5 |
| 3 | Grammar Registry (core + annotation + plugin-provided syntax); no monolithic domain parser | §6 |
| 4 | True plugin architecture (Plugin API, manifest, AST/IR extension contracts, Backend interface, capabilities, serialization ABI) | §6 |
| 5 | `performance.py` `ops_per_sec` must be a real statistic (today it is hardcoded 0) | §2.1 |
| 6 | `accel_used` must be an observation, not the requested flag (`use_accel` never reaches the VM) | §2.2 |
| 7 | `release.py` gains pyproject.toml version sync + drift detection | §2.3 |
| 8 | Strict `.helixc` semantic versioning: Language Spec, AST Schema, Simulation Semantics, Reference Data versions | §2.4 |
| 9 | Type system: inference, unification, constant solving, effect typing, unit typing | §8 |
| 10 | Unit system `Quantity[T, unit]` + dimensional metadata in IR | §9 |
| 11 | Split `sim_runtime/_engine.py` into Engine / Scheduler / Backend / State / Provenance / Snapshot | §10 |
| 12 | Fuzz: Lexer, Parser, SemanticAnalyzer, HXBC loader, bytecode interpreter | §11 |

## 1 — Audit: current state per goal (verified)

| File:line | What is there today |
|---|---|
| `core/performance.py:202-271` | `VMProfiler`; **:224** `result.accel_used = use_accel` (the *requested* flag, not observed); **:261** `result.ops_per_sec = 0.0` (hardcoded); `ops_executed` (**:174**) is never assigned |
| `core/performance.py:64-132` | `accelerated_execute_pending(vm)` — C-dispatch acceleration *exists* but is **never called by any VM**; only `tests/test_performance.py:109-116` monkeypatches it into `_execute_pending` |
| `core/vm.py:771-798` | `CellVM._execute_pending` is a pure-Python loop; decrements `quota` but counts **no** ops; unknown opcodes are silently skipped (`:787-793`, `continue`) |
| `core/ir_runtime.py:106-127` | `IRRuntime._execute_pending` — same, no op counter, no accel path |
| `benchmarks/bench_profile.py:70` | prints `acceleration: native C` from the *requested* flag → misleading today |
| `validation/benchmarks/69_performance_benchmark/run.py` | consumes `VMProfiler`; currently cannot assert `ops_per_sec > 0` |
| `release.py:130-149` | `sync_version` **already** rewrites `pyproject.toml` (`^version = "..."`), `core/version.py`, `server/app.py`, plus the `bytecode.py` "Frozen as of" comment; verification covers the same 3 files |
| `core/hxbc.py:80-83` | `MAGIC=HLXC`, `FORMAT_VERSION=1`; sections PROG / CHNK / SRC / EOF |
| `core/hxbc.py:794-797, 821-826` | OPCODE_VERSION embedded and enforced (`ABIVersionError`) |
| `core/hxbc.py:905, 924-936` | header carries format version + `table_id`; table ids guarded by `_check_table_names()` `:1321-1328` |
| `core/hxbc.py:514, 687` | `#config table` string stored in program body |
| `core/ir.py:44-47, 210` | `IR_VERSION = 1`, embedded by `core/ir_serialize.py:50`, reader rejects newer payloads (`:82-85`) |
| `core/provenance.py:45-105` | build/registry provenance (helix version + python + optional dep versions + source hash) |
| `core/codon_table.py:16-92` | `Op` IntEnum + `OP_OPERAND_BYTES`; `:99-170` STANDARD / MITO_VERTEBRATE / CILIATE tables + `TABLES` |
| `core/codon_table.py:185-197` | `stop_codons_from_table(table)` (codons → `OP_HALT`) |
| `core/parser.py:50-56` | `Parser(stop_codons=None)` **defaults to a literal `{"TAA","TAG","TGA"}`** independent of any table |
| `core/parser.py:66-95` | monolithic annotation→handler dict (27 keywords) |
| `core/parser.py:97-104` | `BIO_INSTRUCTION_KINDS` + `UnknownKeywordError` for unknown `#keyword` |
| `core/parser.py:106-147` | special-cased inline DNA block after `#gem` |
| `core/parser.py:1070` | `parse_source(source, stop_codons=None)` |
| `core/compiler.py:33-34` | `Compiler(table: dict[str, Op] = STANDARD_TABLE)` |
| `core/hxbc.py:1216` | `_ANNOTATION_PREFIX_MAP` — decompile round-trip of `#annotation` forms |
| `sim_runtime/_engine.py:142-182` | module-level `run()` dispatcher: `_SIM_BACKENDS[kind]` (**:150-153**) + hardcoded `elif name == ...` chain (**:156-170**) |
| `sim_runtime/_engine.py` | ~35 top-level `_run_*` functions, 4,029 lines, no Engine/Scheduler/Backend/State classes |
| `core/plugin_registry.py:48-76` | `NativeBackend` + `PluginProvider` (name / extra / keywords / capability_flags / checks / load) |
| `core/plugin_registry.py:78+` | `Registry` with lazy `activate` and conflict detection; `_BUNDLED_PLUGINS` |
| `core/type_system.py` | `HelixType` enum + `SymbolTable` + `TypeChecker.check/infer` (231 lines) — **no unification, no type variables, no effects, no units** |
| `core/parser.py:920-948` | `#type name=...` annotation parsing; `Program.type_annotations` persisted by `hxbc.py:707-711` |
| `core/units.py` | physical constants + conversion helpers only — **no `Quantity`, no dimension typing** |
| `tests/test_vm_fuzz.py` | fuzzes only the native dispatch kernel (impl_python vs impl_cext parity, 300 trials) — nothing fuzzes lexer/parser/semantic/hxbc-loader/interpreter |

## 2 — Quick wins: observability and release hygiene (#5, #6, #7, #8-lite)

These four are small, low-risk, and immediately verifiable.  They should land
first; they also give the rest of this document real numbers to measure against.

### 2.1 — `ops_per_sec` becomes a true statistic (#5)

**Problem.** `VMProfileResult.ops_executed` is never written and
`ops_per_sec` is a literal `0.0` (`core/performance.py:261`).

**Fix.**

1. Count dispatch in `CellVM`: add `self.ops_executed: int = 0`, increment once
   per `_dispatch` in `_execute_pending` (`core/vm.py:797`), and expose the same
   counter on `IRRuntime._execute_pending` (`core/ir_runtime.py:126`).
2. In the accelerated path (`accelerated_execute_pending`, `performance.py:88-132`),
   add `ops_consumed` from every `run_quota` return plus the Python-dispatched ops
   to the same counter.
3. In `VMProfiler.profile`: replace `result.ops_per_sec = 0.0` with
   `result.ops_executed = vm.ops_executed` and
   `ops_per_sec = ops_executed / (vm_run_time_ms/1000)`.
4. `ir_batch_runtime.py` (`:230`) shares the quota semantics; mirror the counter.

**Acceptance.** `VMProfiler.profile(...).ops_per_sec > 0` for a program that does
work, and `ops_executed` equals the sum of ops in `trace`-covered ticks; a unit
test asserts the pure-Python counter matches `accelerated_execute_pending` counters
for a byte-identical run (extend `tests/test_performance.py`).

### 2.2 — `accel_used` becomes an observation (#6)

**Problem.** `use_accel` is a parameter to `VMProfiler.profile` that never reaches
the VM.  `accelerated_execute_pending` is unwired dead code outside a test
monkeypatch (`tests/test_performance.py:112`), so "accel used" is fabricated.

**Fix.**

1. Give the VM a real observation: `self.accel_native_ops: int = 0` on both VMs.
   `accelerated_execute_pending` records how many ops were actually committed by
   `run_quota`.
2. Wire VM → accel: in `CellVM`, gate the accelerated tick loop behind
   `program.config.use_accel` (new config flag, default on) and a
   `use_accel` attribute passed at construction.  The VM calls
   `accelerated_execute_pending(self)` before the pure fallback — this is what
   doc/37 §3.4 already describes and what `tests/test_performance.py:95-120`
   approximates via monkeypatch.
3. `VMProfiler.profile`: `result.accel_used = (accel_native_ops > 0)` — an
   observation of what actually happened, not the request.  Expose
   `accel_ops` too.
4. Fix `benchmarks/bench_profile.py:70` to print from the observed flag and add
   `accel_ops` to its JSON report.

**Acceptance.** With native dispatch built, a fully-arithmetic program reports
`accel_used=True` and `accel_ops==ops_executed`; the identical program with
`use_accel=False` reports `accel_used=False` and byte-identical traces (the
existing parity contract of `tests/test_vm_fuzz.py`).

### 2.3 — `release.py` version gate (#7)

**Problem.** `sync_version` already writes `pyproject.toml`, but the docstring
promises `__init__.py` (it actually writes `core/version.py`), verification only
covers the same 3 files, and nothing stops version drift in other metadata.

**Fix.**

1. Make `pyproject.toml` the **single source of truth**: `core/version.py` already
   re-exports through `helixlang/__init__.py` — keep that chain; assert the
   `__init__` re-export matches `core/version.py`.
2. Add a `--check-versions` mode (non-mutating) that freshens after sync:
   every file that embeds the version (pyproject.toml, version.py, app.py,
   bytecode.py comment, README, README_PYPI, CONTRIBUTING) must read the same
   value; mismatch → abort the release before the gate sequence.
3. Correct the docstring (sync targets, not `__init__.py`).

**Acceptance.** A deliberate off-by-one injected into one metadata file fails the
new gate; all six doc/version-bearing files stay in lockstep across a full
`release.py <version>` run.

### 2.4 — strict `.helixc` semantic manifest (#8, minimal slice)

**Problem.** The binary header carries format version + `OPCODE_VERSION` + table id
(`hxbc.py:80-83, 794-826, 905`)— all *ABI* versions.  There is no notion of
**language spec**, **AST schema**, **simulation semantics**, or **reference data**
version, so an artifact compiled from an old-but-compatible grammar and one from a
semantics-changing release are indistinguishable.

**Fix (minimal, phase-1).**

1. Define 4 constants in `core/version.py`:
   `LANGUAGE_SPEC_VERSION`, `AST_SCHEMA_VERSION`, `SIMULATION_SEMANTICS_VERSION`,
   `REFERENCE_DATA_VERSION` (each a monotone integer or `"YYYY.M.D"`).
2. Encode them in a new `.helixc` section (e.g. `SECT_META`) written by
   `_encode_program`/`dumps_program`; `loads_program` gains
   `require_compatible(reference_data=...)` semantics:
   - `LANGUAGE_SPEC` / `AST_SCHEMA` newer than the loader → hard error
     (`ABIVersionError`-family), mirroring the existing policy at `hxbc.py:821-826`.
   - `SIMULATION_SEMANTICS` mismatch → strict `SemanticVersionError` (never a wrong
     result), while older-but-compatible → explicit warning.
   - `REFERENCE_DATA` mismatch → warning with the recorded data-set id (see §9).
3. Extend `ArtifactInfo` (`hxbc.py:874`) to surface the four versions for
   `helixc info`.

**Acceptance.** A `dumps_program`→`loads_program` round trip preserves all four
versions; a payload with a bumped `LANGUAGE_SPEC_VERSION` is refused; reference
data drift is reported, not silent (per doc/36 F7 "never silently").

## 3 — DNA as the single source of truth (goal 1)

**Current state.**

- The DSL is DNA-first: `Parser` decodes `#gene` ORFs as DNA codons
  (`parser.py`, `_parse_gene`), `codon_table` maps codons→opcodes, and
  `hxbc.decompile` reconstructs a faithful `.helix` text (§round-trip is already a
  test fixture).  So DNA **is** currently the truth for the *initial* build.
- The IR pipeline (`ir_builder` → `ir_lower` → `ir_opt` → `ir_runtime`, IR_VERSION)
  compiles a whole program once, in one shot.  There is no per-gene unit of
  compilation, no dependency graph, and no invalidation.

**Gap.** Nothing watches the DNA for changes.  "CRISPR edit" surfaces only as a
runtime bio-instruction (`test_crispr.py`), never as a compile-time delta.  Editing
one gene means recompiling the whole program.

**Design — incremental JIT.**

```
DNA source (only truth) ─► Parser ─► Program (AST)
                                        │
                     GeneDependencyGraph: gene → {regulations referencing it,
                                        │       calls, IR inst index ranges}
                                        ▼
                       whole-program IRProgram + gene→IR-range map
                                        │
                    edit detected (hash of gene DNA block) ── invalidation set
                                        ▼
            re-lower only invalidated genes → new IR range → patch IRProgram
                                        ▼
                                  IRRuntime (already split by gene ORF, doc/37)
```

1. **`GeneDependencyGraph`** (`core/ir_builder.py` or new `core/incr.py`):
   - nodes = genes (by ORF hash), plus the `{promoter, target}` edges from
     `Regulation` and `OP_CALL_GENE` edges;
   - a unit test asserts an edit to gene *g* invalidates exactly `{g} ∪
     {regulations targetting g} ∪ {callers}` (the DAG's closure).
2. **Gene-block hashing**: hash each ORF's codon block (and its `#gene` header
   fields) as the cache key, stored in an `IncrementalCache` alongside the
   artifact (`hxbc` SRC section already stores the source for hashing at
   `provenance.py:26`).
3. **Patch interface**: `IRProgram.patch_gene(name, new_ir)` that re-ranges
   `OP_CALL_GENE` operands, updates gene offset tables, and invalidates cached
   lowered/bytecode chunks that referenced the edited gene.
4. **CLI**: `helixc run --watch <file>` (or a `#crispr`-triggered recompile hook in
   the server) recomputes only the closure.

**Acceptance.** `edit one gene → compile` is proportional to the closure size, not
the program; byte-identical program behavior before/after an unrelated edit (the
whole-program result is invariant when the edit touches no reachable gene); a
differential test runs a *logically identical* edited program through incremental
vs. full compile and requires identical traces.

## 4 — `LanguageConfig` (goal 2)

**Problem.** grammatical truth is fragmentary:

- `codon_table.TABLES` is the canonical codon→opcode map (good);
- stop codons are *derived* per-consumer: `stop_codons_from_table`
  (`codon_table.py:185`), reused by hxbc (`:1047`) and re-exported via
  `plugins/runtime/seq_utils.py:21`, which is what `cli.py:40` and
  `server/app.py:35` import;
- but `Parser` defaults to a *hardcoded literal* `{"TAA","TAG","TGA"}`
  (`parser.py:56`) independent of the table, while `parse_source` exposes a
  `stop_codons` knob;
- start codons never exist as a set (only "codons mapping to `OP_START`": ATG
  standard, ATG/ATA mitochondrial);
- amino-acid translation tables are **duplicated**: `plugins/annotation/sequences.py:8`
  and `plugins/apps/full_pipeline.py:170` each define their own `_CODON_TABLE`.

**Design — one object, four consumers.**

```python
@dataclass(frozen=True)
class LanguageConfig:
    table_name: str                       # "standard" | "mito_vertebrate" | "ciliate"
    codon_to_op: Mapping[str, Op]         # get_table(table_name) — canonical
    stop_codons: frozenset[str]           # derived: {codon | op == OP_HALT}
    start_codons: frozenset[str]          # derived: {codon | op == OP_START}
    translation: Mapping[str, str]        # codon → amino acid (NCBI), single source
```

- `codon_table` owns the *derived properties*: `start_codons_from_table`,
  `translation_table_from_ncbi(codon_to_op)` — **derivation from one map kills all
  duplication** (the two duplicated AA tables are replaced by lookup).
- `Parser`, `Compiler`, and both VMs take a `LanguageConfig` (default
  `LanguageConfig(table_name="standard")`) instead of individual `stop_codons` /
  `table` parameters; `Program.config.table` stays the wire format string and
  resolves into a `LanguageConfig` at build time.
- The VM never re-derives: the compiled chunk embeds the resolved config id
  (already partially true — `hxbc` stores `cfg.table` `:514`).

**Acceptance.** A property test builds identical `Program`s for all three tables
via `parse_source` with no `stop_codons` argument; a config whose table is
`ciliate` yields `TAA→OP_EMIT_MORPHOGEN` consistently in lexer, parser, chunk, and
VM, and the two plugin AA tables in §4 are deleted in favor of
`codon_table`.

## 5 — Grammar Registry (goal 3)

**Problem.** `Parser.parse` hardcodes a 27-entry annotation dispatch dict
(`parser.py:66-95`), a separate `BIO_INSTRUCTION_KINDS` branch (`:97-99`), a
special-cased `#gem` inline-DNA block (`:106-147`), and `hxbc` maintains a parallel
`_ANNOTATION_PREFIX_MAP` (`hxbc.py:1216`) to decompile those same annotations.
Adding any domain schema ("gene therapy vector", "synthetic promoter library")
means editing the parser.

**Design — registry-driven parsing and decompilation.**

```python
@dataclass(frozen=True)
class AnnotationGrammar:
    keyword: str
    parse: ParserMethod            # (parser, ProgramBuilder) -> None
    validate: Validator | None     # runs in semantic phase
    decompile: Decompiler | None   # extension section -> annotations text
    extension: ASTExtension | None  # section owned by this grammar (§6.3)
    core: bool = True              # core grammars always present
```

- `GrammarRegistry` (new `core/grammar_registry.py`): core grammars registered at
  import (the 27 existing + bio instructions); `register(grammar)` appends a
  plugin grammar; **conflict on keyword → `PluginConflictError`** (mirroring
  `plugin_registry.py` doc/36 F7).
- `Parser` consults the registry instead of the literal dict; the `#gem` inline-DNA
  special case becomes a property of the `gem` grammar (its `parse` consumes
  GENE_ID/CODON tokens until `#end`), removing the `parser.py:106-147` branch.
- `hxbc.decompile` walks the same registry to write annotations back (replacing
  `_ANNOTATION_PREFIX_MAP`), so a plugin grammar is *automatically* round-trippable
  through `.helixc` if it declares `extension_keys`.

**Acceptance.** Registering a test grammar `#vector gene=... plasmid=...` makes
parse/compile/decompile/load work without touching `parser.py` or `hxbc.py`;
keyword collision raises `PluginConflictError`; `helixc info` lists registered
grammars.

### 5.1 — End state after doc/41 (descriptor grammars, `#use` activation)

doc/41 (Item 3 + 4) converged this goal with the plugin platform: the parser now
knows *zero* `#keyword` beyond dispatch plumbing.  Data annotations are declared
as **grammar descriptors** and compiled by the registry:

- **`GrammarDescriptor`** (in `core/grammar_registry.py`) declares `keyword`,
  `fields: tuple[FieldSpec, …]`, `allow_extra`, `body` (`"fields"` | `"raw"`),
  `target` (`"section"` | `"sim_extensions"`), plus optional `parse`/`validate`/
  `decompile` hooks and an `owner`.  `FieldSpec` carries `type`, `required`,
  `default`, and `unit` (consumed by the doc/41 Item 5 static unit layer).
- **`compile_descriptor`** turns a descriptor into an `AnnotationGrammar`: a
  generic field-collector parse hook (coercion + required + default per
  `FieldSpec`) or, for `body="raw"`, the plugin-supplied callable that drives the
  parser through the protected **`Parser.token_hooks`** surface
  (`advance`/`peek`/`expect`/`collect_fields`).
- **`#use` activation** (`requires_use`): a plugin's grammar is *inert* until the
  program declares `#use <plugin>`; without it the keyword is the hard
  `UnknownKeywordError` — never a silent passthrough.  `Registry.register` installs
  a provider's `grammars` slot and wires `provider_for_keyword` to the one shared
  table, so backend-plugin ↔ language-plugin is one mechanism (doc/41 §4.2–4.3).
- **`#quantity name=X expr=A+B`** is built the same way (a core descriptor whose
  semantic hook is the doc/41 Item 5 `DimInferencer`).

A brand-new `#keyword` now lands with zero edits to `parser.py`/`lexer.py`
(demo: `helixlang.plugins.cardiology` → `#cardiac_cycle period=0.8 …`, with a
semantic validator and a `.helixc` round-trip).

## 6 — True plugin architecture (goal 4)

This section is the formal spec for pluginization and is normative.  It defines
the **Plugin API**, **Plugin manifest**, **AST extension contract**, **IR
extension contract**, **Backend interface**, **Capabilities**, and
**Serialization ABI**, then a migration that removes the two structural leaks the
goal names: the free-form `sim_extensions` dict and the backend-specific dispatch
(`_SIM_BACKENDS` + the `elif backend ==` chain).

### 6.1 — Why plugins are not plugins today (verified leaks)

`core/plugin_registry.py` already gives lazy discovery, capability flags,
dependency checks and conflict detection — the *activation layer*.  What a
third-party plugin cannot do without patching core:

1. **Backends are core, not plugins.** `sim_runtime/_engine.run` (`_engine.py:142-182`)
   dispatches through a hardcoded `elif name == ...` chain (`:156-170`, 8 names)
   plus `_SIM_BACKENDS[kind]` (`:150-153`, **21 entries at `:3641-3662`**).  Both
   tables are module-level dicts of private `_run_*` functions: 30+ backends can
   only exist by editing `_engine.py`.
2. **Program state leaks through core fields.** Every backend reads
   `program.sim_extensions` directly (~20 sites, `_engine.py:468…3917`), which is a
   `dict[str, Any]` on the core `Program` node (`core/ast_nodes.py:189`), written
   from the parser in ~15 places (`parser.py:138, 223-232, 496, 517-519, 566,
   650-665, 701, 727, 743, 760-774, 918`) and by the CLI (`cli.py:256-262`), and
   encoded free-form by `hxbc` (`hxbc.py:570-571, 766-778, 1216-1286`).  Any
   plugin feature is therefore a string key agreement across parser→AST→binary→
   engine — a "plugin" cannot ship one without a core change.
3. **Plugins import core internals.** `plugins/runtime/population.py:32-34`
   imports `Program` and `Chunk` from `helixlang.core.*` (and
   `opcode_semantics` at `:1010`); `plugins/apps/consortium.py:428-432` re-runs
   lexer/parser/compiler inside the plugin.  There is no import boundary, so
   nothing stops a plugin from depending on private core fields.
4. **Serialization is not namespaced.** `_decode_ext` (`hxbc.py:765-778`) reads a
   flat value dict; a plugin cannot version or namespace its own bytes.

**Design rule.** After this section, the *only* sanctioned way for a plugin to do
anything (parse a keyword, extend the AST/IR, run a backend, serialize) is
through the public **`helixlang.api.*`** surface defined in §6.2, the contracts in
§6.3–6.5, the capability rules in §6.6, and the manifest in §6.8.  Doc/36 §8
remains the normative contract; this section formalises it and makes it
enforceable.

### 6.2 — Plugin API: the public import surface

New internal-to-package namespace `src/helixlang/api/` (public, frozen): plugins
import **only** from here (plus `helixlang.core.errors`), never from
`helixlang.core.ast_nodes` / `helixlang.core.parser` / `helixlang.sim_runtime._engine`.

| Module | Provides |
|--------|----------|
| `helixlang.api.registry` | `PluginProvider`, `NativeBackend`, `Registry` (re-export, typed) |
| `helixlang.api.grammar` | `AnnotationGrammar` (the §5 registry entry type) |
| `helixlang.api.ast` | `ASTExtension`, `ProgramView`, `ProgramBuilder`, `SectionField` |
| `helixlang.api.ir` | `IRExtension`, IR inst kind namespace + ABI policy |
| `helixlang.api.backend` | `Backend` (ABC), `RunRequest`, `EffectiveConfig`, `SimResult`, `BackendRegistry` |
| `helixlang.api.capabilities` | `Capability` (`id`, `summary`, `reduces_fidelity`) |
| `helixlang.api.errors` | re-export of the typed error family (`PluginError`, `BioError`, `ModelMissingError`, `SimConfigError`, …) |
| `helixlang.api.language` | `LanguageConfig` (goal #2) + the public constants of `core.codon_table` and `core.units` — so plugins stop importing `helixlang.core.{codon_table,units,...}` |

**Import-boundary enforcement (mechanical, not by convention).** Extend the
existing source-audit tool `core/find_silent_fallbacks.py` (same family, designed
for exactly this) with `find_core_imports.py`: scan every
`helixlang/plugins/**` and `sim_runtime/backends/**` module and reject any root
import outside the allowlist `{helixlang.api, helixlang.core.errors, stdlib}`.
Wired into the release gates (§12) and as `tests/test_plugin_purity.py` that runs
the scanner over the wheel's installed files.  Known compliant exceptions are
listed in the scanner config *and* migrated (§6.8 `plugins/runtime/population.py`)
so the allowlist ends with zero exemptions.

### 6.3 — AST extension contract (replaces `sim_extensions`)

**Contract shape (`helixlang.api.ast`):**

```python
@dataclass(frozen=True)
class SectionField:
    key: str
    type: FieldType           # INT | FLOAT | BOOL | STR | LIST | MAP — coerced, typed

@dataclass(frozen=True)
class ASTExtension:
    id: str                              # e.g. "human_profile", "ecosystem", "gem"
    grammars: tuple[str, ...]            # #person, #disease, ... (must be in GrammarRegistry §5)
    parse: Callable[[ProgramBuilder, ASTExtension], None]
    validate: Validator | None
    fields: tuple[SectionField, ...]     # the only keys this extension may consume
    decompile: Decompiler | None
    abi_version: int                     # must match manifest abi_version
```

- `Program.sim_extensions: dict[str, Any]` is **replaced** by
  `Program.extensions: PluginExtensions`, a read-only namespace with one typed
  attribute per `ASTExtension.id`
  (`program.extensions.human_profile.drugs` instead of
  `sim_extensions["drugs"]`).  `ProgramBuilder` only accepts writes for declared
  fields; an unknown key is a hard `UnknownKeywordError`-family error (doc/36 F7 —
  never a silent ignore).
- Every `#sim key=value`/`#person ...`/`#gem ...` statement now targets the owning
  extension's section, resolved through the **GrammarRegistry** (§5).  Core keeps
  exactly one tiny section: `kind`, `output`, and the `#config` keys (all typed).
- CLI writes move to the builder: `cli.py:256-262` (`gem_dynamic`/`gem_duration`/
  `gem_dt`) become `builder.extension("gem").set(...)`, never a dict poke.
- Backends receive a read-only `ProgramView` (`api.ast`): they can *read* typed
  extension sections but cannot reach the raw `Program`, `Config`, or
  `sim_extensions` dict (the view exposes a whitelisted surface — kills the
  `Program`/`Chunk` import in `plugins/runtime/population.py:32-34`).

**Migration map** (owner plugin ← keys):

| Owner | Keys currently in `sim_extensions` / `config.sim` |
|-------|---------------------------------------------------|
| core | `kind`, `output`, `table`, `ticks`, `ops_per_tick`, `react_steps`, `backend`, `use_central_dogma`, `species` → typed `EffectiveConfig` |
| `human` | `person_*`, `trait_*`, `disease_*`, `disease_genes`, `disease_metabolites`, `drugs`, `pd_effects`, `qsp_bindings`, `endocrine_configs`, `immune_configs`, `tumor_biopsy`, `genes` (genotype) |
| `ecosystem` | `genome`, `genome_*`, `species.<name>.*`, `patch.<name>.*` |
| `gem` | `gem_*`, `gem_inline_genes`, `gem_inline_genome`, `gem_dynamic`, `gem_duration`, `gem_dt` |
| `population` | `mechanics`, `lbm`, and the population `#sim` keys merged at `_engine.py:3703` |
| `fba` | dFBA/FBA `#sim` keys (dissolved from `{**config.sim, **sim_extensions}`) |
| per-backend | each of the 21 `#sim kind=…` blocks' owning keys (§6.5) |

### 6.4 — IR extension contract

```python
# helixlang.api.ir
@dataclass(frozen=True)
class IRExtension:
    id: str
    kinds: tuple[str, ...]          # new IR inst kinds, namespaced "<plugin>.<inst>"
    build: Callable[[ProgramBuilder, IRProgram], IRInst | None]   # IRBuilder hook
    execute: Callable[[IRRuntime, IRInst], None]                  # IRRuntime arm
    operand_schema: OperandSchema   # how its operands serialize
    abi_version: int
```

- Mirrors the existing `OP_USE_PLUGIN` opt-in (`codon_table.py:24-27`):
  compiler-emitted plugin insts are *never* generated without an explicit `use`;
  an IR kind that is not registered at load is refused (never silently skipped).
- `IRRuntime` dispatch consults a per-instruction extension map; the
  `CellVM`/bytecode path is unaffected unless the extension also registers an
  opcode (out of scope — IR extensions are IR-first).

### 6.5 — Backend interface & BackendRegistry (kills the dispatch leaks)

```python
# helixlang.api.backend
class Backend(ABC):
    id: str                          # "population", "human", ...
    kinds: tuple[str, ...]           # aliases usable as `#sim kind=...`
    @abstractmethod
    def run(self, req: RunRequest) -> SimResult: ...

class RunRequest:
    program: ProgramView             # read-only, typed view of extensions
    config: EffectiveConfig          # merged #config + core section
    registry: Registry
    seed: int | None
    source: str | None

class BackendRegistry:
    def register(self, backend: Backend) -> None      # PluginConflictError on id clash
    def resolve(self, *, backend: str | None, kind: str | None) -> Backend
```

- `_engine.run` (142-182) shrinks to:
  `backend = backend_registry.resolve(backend=req.config.backend, kind=req.kind)`
  then `backend.run(req)`; the `elif` chain (`:156-170`) and `_SIM_BACKENDS`
  (`:3641-3662`) are **deleted**, not deprecated.
- **Migration table** — every current backend becomes a `Backend` subclass moved to
  `sim_runtime/backends/<id>.py` (core backends are core-*bundled plugins*,
  registered like third-party ones):

| Group | Current form | Becomes |
|-------|--------------|---------|
| config backends | `elif name == "whole_cell"|"population"|"fba"|"calibration"|"benchmark"|"gem"|"ecosystem"` (`_engine.py:156-170`) | `Backend` subclasses, `.id` = name |
| kind backends | `_SIM_BACKENDS` 21 entries (`:3641-3662`) | `Backend` subclasses with `kinds=("human",)`, etc. |
| legacy | `_run_human_simulation` / `_run_human_simulation_legacy` (`:3022-3087`) | one `human` backend with two run modes |
| long-tail | `#sim kind=spatial_dfba` etc. | `kinds` alias resolved in the same registry |

- `cli.py:262, 531` `if program.sim_extensions.get("kind") in _SIM_BACKENDS`
  becomes `registry.resolve(kind=...) is not None` — the CLI stops importing the
  private table.

### 6.6 — Capabilities

Extends `use_stmt.KNOWN_FLAGS` / `PluginProvider.capability_flags` from 3 string
flags to declared, typed capabilities:

```python
# helixlang.api.capabilities
@dataclass(frozen=True)
class Capability:
    id: str               # "--pure-python", "--lower-fidelity", or plugin-defined
    summary: str
    reduces_fidelity: bool
```

- A plugin **declares** its capabilities in the manifest (§6.6).  `#use <plugin>
  --flag` remains the *only* way to activate them; a flag that no provider claims
  is a hard `UseError` (as today for unknown flags).  `reduces_fidelity=True`
  drives the existing fidelity record (`plugin_registry.Registry.fidelity`,
  `:169-188`) so provenance stays honest.
- A capability the core does not know costs nothing: the registry stores it by id
  and only checks claims; the manifest supplies the description for
  `helixc plugin info`.

### 6.7 — Serialization ABI

- New `.helixc` PROG tag `0x0E PLUGIN_EXT`:
  `(plugin_id: str, abi_version: u32, payload: bytes)` where `payload` is the
  plugin's own section bytes, encoded/decoded by the manifest's declared codec.
  Sections are namespaced by plugin so no two plugins can collide.
- **Load policy** (same spirit as `OPCODE_VERSION` at `hxbc.py:821-826`):
  plugin absent or `abi_version` mismatched → typed
  `PluginBinaryError`/`ABIVersionError` (never a wrong-result run); missing
  optional plugin with materialized data → `PluginMissingError` with the pip hint,
  unless the payload is empty.
- Plugins that only add *grammars* (no AST/IR records) need no ABI —
  `grammars`-only entries are stateless.
- **Legacy compat window**: the old free-form `sim_extensions` TAG stays decodable
  *read-only* for two releases, then refused, so `.helixc` artifacts compiled
  pre-migration still load (doc/36 §3ξ: announce, then hard error — never silent).

### 6.8 — Manifest (`helix.plugin.toml`)

```toml
# helixlang/plugins/human/helix.plugin.toml
name         = "human"
version      = "1.0.0"          # plugin release version (not pip)
entry_point  = "helixlang.plugins.human"      # module exporting PLUGIN
abi_version  = 1

[provides]
grammars = ["person", "trait", "disease", "disease_gene", "disease_metabolite",
            "drug", "pd_effect", "qsp_binding", "endocrine_config",
            "immune_config", "tumor_biopsy"]
ast    = ["human_profile"]
ir     = []
backends = ["human", "human_virtual_patient"]

[capabilities]
flags = ["--low-fidelity"]

[requires]
pip = ["numpy>=1.24", "pandas", "rdkit"]       # → PluginDependencyError extras

[native]                                        # optional
module = "helixlang._accel.human_step"
rebuild = "python -m helixlang._accel.build"
```

- Parseable with **no plugin import** (stdlib `tomllib`): `helixc plugin list`,
  dependency resolution and conflict checks run cold (doc/36 §3.4/§3.5).
- `Registry.register` (now in `api.registry`) enforces: manifest `ast`/`ir`/
  `grammars`/`backends` ids must match the registered `ASTExtension`/`IRExtension`/
  `AnnotationGrammar`/`Backend` ids; `abi_version` must match; drift is a
  `PluginConflictError`.  Any plugin may still ship only a manifest + `load` if it
  needs nothing core-visible.

### 6.9 — Migration: existing plugins to public API only

| Step | Action | Removes |
|------|--------|---------|
| E1 | Ship `helixlang.api.*` + manifest parser + `Backend`/`ASTExtension`/`IRExtension` types | — |
| E2 | Replace `sim_extensions` writes in `parser.py` with builder→extension sections; typed `Program.extensions`; hxbc `PLUGIN_EXT` tag | free-form dict growth; `_ANNOTATION_PREFIX_MAP` (`hxbc.py:1216`) consolidated into `ASTExtension.decompile` |
| E3 | Move 8+21 backends into `sim_runtime/backends/*` as `Backend` subclasses; delete `elif` chain + `_SIM_BACKENDS`; `run()` → `BackendRegistry.resolve` | backend-specific dispatch |
| E4 | Migrate `_engine.py` sim_extensions readers (`~20 sites`) and `_coerce.py:164` to `ProgramView` sections; `plugins/runtime/population.py` and `apps/consortium.py` to `api.*` only | plugin→core-internal imports |
| E5 | Enable `find_core_imports.py` in gates + `tests/test_plugin_purity.py`; drop legacy `sim_extensions` TAG after 2 releases | residual leaks |

**Acceptance.**

1. `test_plugin_purity.py` passes with **zero exemptions** — no bundled plugin
   imports outside `helixlang.api.*` / `helixlang.core.errors` / stdlib, and the
   strings `sim_extensions` and `_SIM_BACKENDS` appear nowhere under
   `src/helixlang/` except hxbc's legacy-decode path.
2. An out-of-tree `TestPlugin` with a manifest, one grammar, one AST extension,
   one IR kind and one backend runs end-to-end
   parse → compile → (bytecode | IR) → result via `run --use plugin`; its absence
   raises `PluginMissingError`, never a silent skip; `helixc plugin list` lists it
   without importing it.
3. Every `validation/benchmarks/*` golden result is unchanged under the new
   backend registry (parity `run()` contract, §9), and provenance keeps recording
   fidelity exactly as `Registry.fidelity` does today.

## 7 — Type system (goal 9)

**Current state.** `type_system.py` is a typing *skeleton*: `HelixType` enum +
`SymbolTable` + `TypeChecker.check`/`infer` (register promoters/genes, check gene
references, infer literals).  Parser stores `Program.type_annotations` (`#type`),
semantic validates symbol names.  There is no inference variable, no unification,
no constraints, and the checker handles only a handful of declarations.

**Design — staged, DSL-shaped (not full Hindley–Milner).**

1. **`Type` hierarchy** replacing the plain enum: `TypeVar(name)` is introduced so
   inference can produce fresh variables; a small **`Unifier`**
   (Robinson-style, `unify(t1, t2) -> subst`) that resolves variable chains and
   reduces to `Schema` generalization.
2. **Constraint solving**: `#type` annotations become pre-instantiated schemas;
   `infer` on expressions/instructions generates constraints; `solve` performs
   `occurs-check` and **constant solving** — a bound like `#type mygene=Protein` or
   a promoter annotated `Protein` must be solved to a *single* ground type; an
   unsatisfiable system is a `SemanticError` naming the offending symbol (doc/36 F7
   naming rule).
3. **Effect typing**: a `BioEffect` lattice `{pure, side_effect, quota_boundary}`
   (per opcode family in `_dispatch`, see vm `_bio_handlers` + `OP_*`); the checker
   rejects side-effecting ops inside declared-pure regions and records.
4. **Unit typing (tying into §9)**: `Float<mol>` vs `Float<μM>`; arithmetic requires
   compatible dimensions at check time; `#type` annotation syntax extended to
   `concentration=Float<μM>`.
5. Checker becomes **lossless on accept**: an accepted program's fresh variables
   are all resolved to ground types in the final substitution (the "constant
   solving" acceptance criterion).

**Acceptance.** A corpus of typed programs checks: well-typed passes; type-mismatch
(yet symbol-exists) fails with a precise message; inference resolves a program with
zero annotations to full ground types; effect check catches a side-effecting read
in a pure block; `table`/time variables infer `Float<min>` consistent with
`unify`.

## 8 — Unit system & dimensional safety (goal 10)

**Current state.** `core/units.py` holds *anchored constants* (ticks ↔ minutes,
µm, µM, gDW, ATP molecules) and stateless converters
(`ticks_to_min`, `diffusion_to_lattice`, …).  Dimensions live only in names:
`volume_um3`, `c_period_min`, `maintenance_atp_per_min` (`_engine.py:185-196`).
Nothing prevents `minutes_since_birth + volume_um3`.

**Design.**

1. **`Quantity[T, unit]`** (new `core/dimensions.py`, stdlib-only like `units.py`):
   a `Dimension` is a tuple of 7 SI exponents; `Quantity` carries `(value, dim)`;
   `+`/`-` demand equal dims, `*`/`/` compose exponents; conversions table between
   named units (`min`, `s`, `µm`, `µM`, `mol`, `gDW`, …) with the anchors in
   `units.py` as the conversion basis.
2. **IR dimensional metadata**: `IRInst` gains optional `dim: Dimension | None`
   (op literals that are physical quantities); `ir_serialize` round-trips it with
   a `DIM` tag, gated by `LANGUAGE_SPEC_VERSION` (new metadata = version bump, see
   §2.4).  The runtime stays dimension-free (metadata-only flow like source maps),
   so `CellVM`/`IRRuntime` behavior is unchanged.
3. **Unit-typed compile errors**: `SemanticAnalyzer` gets a unit pass that checks
   config fields (`ops_per_tick` dimensionless, ticks integer) and expression-level
   quantity math, converting known fields declared in `_engine.py` ID lists to
   inferred dimensions.
4. `validation/benchmarks` get new unit fixtures whose cross-unit arithmetic must
   fail to compile, plus a conversion table whose `minutes==seconds` holds exactly.

**Acceptance.** `Quantity(5, min) + Quantity(7, min)` is valid;
`Quantity(5, min) + Quantity(7, μm3)` fails at compile time; a program that adds a
concentration to a volume fails `semantic.check` with the dimension tree in the
message; IR round-trip preserves dims; runtime trace is bit-identical with the
metadata stripped (differential test).

## 9 — Engine split (goal 11)

**Current state.** `sim_runtime/_engine.py` is a 4,029-line module of top-level
`_run_*` functions behind one `run()` dispatcher
(`_engine.py:142-182`) with attached provenance
(`provenance.build_provenance`, `:174-181`), and VM-side snapshot machinery
(`SnapshotDownsampler`, `performance.py`) already separated.

**Design — five collaborating classes (thin refactor, no behavior change in phase 1):**

| Class | Responsibility | Becomes home of |
|-------|---------------|-----------------|
| `Engine` | orchestration: config/lifecycle, chooses backend | `run()` glue `_engine.py:142-182` |
| `Scheduler` | tick loop, quotas, batch backends | `ops_per_tick` loop, `ir_batch_runtime`, `_batch_*` helpers (drains legacy `_run_*`) |
| `Backend` (ABC) | one `SimResult`-producing pipeline per backend | each `_run_*` as a `Backend` subclass (moves into `sim_runtime/backends/*`) |
| `State` | per-simulation mutable state + deep-reset | `_seed_cells`, config-built state objects (`_build_population_config`, …) |
| `Provenance` | record & attach (already exists) | `_engine.py:174-181` `build_provenance` (kept) |
| `Snapshot` | bounded trace (already exists) | `SnapshotDownsampler`, satisfy in `Scheduler` |

- Refactor as a *pure move* first: register the extracted backends in the
  `BackendRegistry` of §6.5 (single dispatch table), and assert the
  `#sim kind=` / `backend=` resolution produces identical results to today
  (a golden diff test over the existing `validation/benchmarks`).
- §6.5 and §9 share the `Backend`/`BackendRegistry` types from `helixlang.api.backend`;
  the engine split lands on top of the plugin migration (E3), not before it.
- Keep `sim_runtime/__init__.py` exporting the legacy `run()` for the CLI.

**Acceptance.** Every `validation` benchmark passes via the new architecture with
a build-flag parity test (`run(legacy) == run(new)` per backend on the benchmark
programs); file-level cyclomatic/size signal: no `_run_*` function remains in
`_engine.py`.

## 10 — Fuzzing (goal 12)

**Current state.** `tests/test_vm_fuzz.py` only exercises the *native dispatch
kernel* (impl_python vs impl_cext parity, unknown-op rejection, determinism).  The
frontend and interpreter are fuzz-exposed nowhere.

**Design — one file per stage, shared invariants.**

1. **`tests/test_fuzz_frontend.py`** (Lexer, then Parser/Semantic behind it):
   - random token/DNA/annotation text drawn from a skewed alphabet (valid codons,
     `#keyword`, `=`, `->`, garbage bytes, unterminated blocks, deep nesting,
     CRLF, backslash continuations);
   - **invariant A**: every input either produces a `LexError`/`ParseError`/
     `SemanticError`/`UnknownKeywordError` **or** a `Program` — never an untyped
     `Exception` or hang; inputs are bounded (max length, seeding like
     `test_vm_fuzz.py`'s `random.Random(seed)` 40-seed parametrization).
   - **invariant B**: determinism — same seed, same tokens, same program.
   - **invariant C** (round-trip): for every `Program` that parses + passes
     semantic, `hxbc.decompile(Program) → parse_source → Program'` re-parses to an
     equivalent AST (decompile is canonical — the two-plugin AA-table extraction
     from §5 makes the same guarantee at the grammar level).
2. **`tests/test_fuzz_hxbc.py`** (loader): a valid artifact, then bit-level
   corruption — truncation at every offset, flipped bytes in header/body,
   `MAGIC`/version/table_id/`OPCODE_VERSION` mutations — must always terminate
   with a typed `BinaryError`/`ABIVersionError`/`BinaryFormatError` (never a raw
   `IndexError` cross module boundary or crash; the reader contract at
   `hxbc.py:821-826` and the `_MAX_*` guards `:117-118` are the baseline).
3. **`tests/test_fuzz_interp.py`** (interpreter): random *valid* chunks (from the
   compiler, not the dispatch subset) with random constant pools run under
   `CellVM` and `IRRuntime`; invariant = **determinism** + **typed** runtime errors
   only (`RuntimeHelixError` family), and parity between `CellVM` and `IRRuntime`
   where supported.  This is where the `vm.py:787-793` silent-skip of unknown
   opcodes is pinned: either kept-but-instrumented (counter) or hardened to raise
   under a `--strict-dispatch` flag made the default — decide in review, then
   assert it in the fuzz test.
4. Tie into the `use_accel` observer (§2.2): accelerate a random chunk fuzz set
   and assert `accel_used`/`accel_ops` never exceed `ops_executed` and results
   match pure Python.

**Acceptance.** Each fuzz suite runs ≥ 1000 seeded trials in CI with the invariant
assertions above; a regression corpus stores any seed that ever triggered a bug.

## 11 — Phasing

| Phase | Goals | Scope | Risk | Gate |
|-------|-------|-------|------|------|
| **A** | #5 #6 #7 #8-lite | counters, accel wiring, release gate, semantic manifest | very low | full `release.py` |
| **B** | #2 #3 | `LanguageConfig`, `GrammarRegistry` (parser + hxbc refactor) | low-medium (decompile golden tests) | full `release.py` |
| **C** | #1 | incremental JIT on the gene DAG | medium (IR patching) | full `release.py` + new diff bench |
| **D** | #9 #10 | type inference + units + IR dim metadata (behind `LANGUAGE_SPEC_VERSION`) | medium-high | full `release.py` |
| **E** | #4 #11 | plugin platform (§6.2–6.8) + engine split — split into E1 (api/manifest + hxbc PLUGIN_EXT tag) → E2 (AST sections replace `sim_extensions`) → E3 (BackendRegistry; delete `_SIM_BACKENDS` + `elif` chain) → E4 (migrate plugins to `api.*`) → E5 (purity gate + legacy deprecation) | high (cross-cutting) | full `release.py` + `test_plugin_purity` + parity suite |

Dependencies: A before D (units need IR dims versioned), B before E (plugin
grammars need the registry — E1 waits for §5 grammar entries, E2 waits for
§6.3 field typing), C isolated. Phase E has an internal hard order E1→E2→E3→E4→E5
(tagged sections before the compile step, registry before the backend move).
Fuzzing (#12) decomposes by phase: frontend fuzzers land with A (they exercise the
same code), HXBC fuzzers with the manifesto
sections as they land.

## 12 — Validation wiring & release impact

- Every phase ends with the count-synced benchmark set: any new
  `validation/benchmarks/NN_*/run.py` updates README/README_PYPI/CONTRIBUTING
  totals (73/73 today) and re-runs `release.py <version>` as the acceptance
  command (with the 3.11 project env active, or `PYTHON=<path-to-3.11-python>`).
- New core modules join the mypy/ruff gates (`disallow_untyped_defs=True` — new
  classes must be fully typed); doc/36/37 invariants (typed errors only, no silent
  fallback, explicit opt-in) are checked by the fuzz suites before they are merged.
- The `4`+1 semantic versions (§2.4) are recorded inside every artifact from phase
  A onward, so phase D's IR dim metadata ride a monotonically increasing
  `LANGUAGE_SPEC_VERSION` without breaking phase-A loaders.
- **Docs to keep in sync** when E lands: doc/36 (contract), doc/12 §7.1/§8.6 and
  doc/02 §`#sim` (the `sim_extensions` "open extension point" wording), doc/20
  (`gem_*` fields), doc/33/29/27 (`tumor_biopsy`/human `sim_extensions` keys),
  and wiring.md — every place that describes `Program.sim_extensions` moves to the
  `Program.extensions.<id>` section language of §6.3.
- **Purity gate** (E5): `find_core_imports.py` runs in the release sequence with
  zero exemptions and blocks on any `sim_extensions` / `_SIM_BACKENDS` reference
  outside the legacy hxbc decode path.