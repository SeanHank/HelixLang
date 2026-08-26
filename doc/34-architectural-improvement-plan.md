# Doc 34 — Architectural Improvement Plan

**Author perspective: from Build Mode → Proof Mode**

Date: 2026-08-26
Scope: src/helixlang + tests + doc + examples
Audience: Author (Sean) and future contributors

---

## 1. Current State Assessment

### 1.1 What exists

| Metric | Value | Notes |
|--------|-------|-------|
| Source modules | 126 (.py) | 35 human/, 22 apps/, 14 gem/, 42 root |
| Total Python LOC | 80,699 | Across all src/helixlang/ |
| Test files | 106 | |
| Total test cases | 3,062 | All passing, 81% coverage |
| Example files | 59 | .helix source |
| Documentation files | 35 | doc/00 through doc/34 |
| Version | 2026.8.4 | pyproject.toml |
| Circular imports | 0 | Clean |
| Mutable global state | 1 | `_DEBUG_SESSIONS` in server.py (thread-safe via lock) |
| Random references | 438 | Across codebase |
| Heaviest dependency | human/ (104 imports) | Most coupled subsystem |
| Validation benchmarks | 10 | validation/benchmarks/01-10 |
| Bytecode ABI version | v1 | OPCODE_VERSION in bytecode.py |

### 1.2 What's proven vs. what's claimed

| Level | Definition | Current status |
|-------|-----------|----------------|
| **A — Implemented** | Code exists and runs | All 126 modules |
| **B — Validated** | Tested against reference dataset/model | E. coli iML1515 FBA (growth 0.821), Synechocystis iJN678 (0.292), codon translation, lac operon |
| **C — Literature-informed** | Parameters from published sources | PBPK organ volumes, CYP star alleles, disease ODE parameters |
| **D — Predictive** | Demonstrated on held-out data | None yet |

### 1.3 Where the risk is

The project has **126 modules** but only **~15 validated benchmarks**. The gap between Implemented (A) and Validated (B) is the primary credibility risk.

Specific high-risk boundaries:
- `human/virtual_patient.py` (2,372 LOC) — most complex single file, 104 import dependencies
- `human/drug.py` (1,161 LOC) — RDKit-dependent, SMILES parsing edge cases
- `sim_runtime.py` — 36 backend dispatch paths, most likely to have implicit state
- `apps/` (22 modules) — pipeline glue, least tested individually

---

## 2. P0: Core Stability (Week 1-2)

### 2.1 Bytecode ABI freeze

**Goal**: Make bytecode a stable boundary that can outlive the Python implementation.

**Files to audit**:
- `src/helixlang/bytecode.py` — instruction set definition
- `src/helixlang/hxbc.py` — binary serialization
- `src/helixlang/vm.py` — execution engine
- `doc/11-helixc-binary-format.md` — format specification

**Action items**:
1. Freeze instruction opcodes — add `OPCODE_VERSION = 1` constant
2. Freeze bytecode header format — document magic bytes, version field
3. Add `bytecode_roundtrip` test: compile → serialize → deserialize → execute → compare
4. Write `spec/bytecode-abi.md` — formal ABI document (not just description)
5. Add `--check-bytecode-version` flag to CLI

### 2.2 VM semantic stability

**Goal**: Same source + same seed + same backend = same result.

**Files to audit**:
- `src/helixlang/vm.py` — instruction dispatch
- `src/helixlang/cell.py` — cell state management
- `src/helixlang/cell_body.py` — physical cell model
- `src/helixlang/central_dogma.py` — transcription/translation

**Action items**:
1. Audit all `random` calls in VM path — ensure every RNG is seeded from master seed
2. Audit all `time.time()` / `datetime.now()` calls — replace with simulation clock
3. Audit all module-level mutable state — `_DEBUG_SESSIONS` in server.py is the only one found; add `_STATE_LOCK` or document thread-safety assumptions
4. Add `test_determinism_audit.py` — run each backend 3 times with same seed, compare outputs byte-for-byte
5. Write `spec/vm-semantics.md` — formalize: instruction semantics, memory model, RNG behavior, error model

### 2.3 Global state elimination

**Current findings**:
- `_DEBUG_SESSIONS` in `server.py:55` — mutable dict, protected by `_get_debug_lock()` threading.Lock
- `_DEBUG_LOCK` in `server.py:54` — lazily initialized, `# STATE: global (lazily initialized)`
- DNA codec trie (`_BASE_IDX`, `_DNA_BIN`, `_BIN_DNA`, `_TRANSITIONS`, `_TRANSVERSIONS` in `dna_codec.py`) — computed at import, immutable, annotated `# STATE: global`
- Codon tables (`STANDARD_TABLE`, `WOBBLE_BITS`, `TABLES`, etc. in `codon_table.py`) — immutable, annotated `# STATE: global`
- Population units (`UNITS` in `population.py`) — immutable, annotated `# STATE: global`
- Evolution lookups (`_TRANSITIONS`, `_TRANSVERSIONS`, `_CODON_MUTATIONS` in `evolution.py`) — immutable, annotated `# STATE: global`
- `_TS_TRANSITIONS` in `evolution.py:776` — plain `set` (mutable), never mutated at runtime, annotated `# STATE: global (mutable set, never mutated at runtime)`

**Action items**:
1. ~~`server.py`: Replace `_DEBUG_SESSIONS` with `threading.Lock`-protected dict or document single-threaded assumption~~ ✓ Done
2. ~~Add `# STATE: global` comments to all module-level mutable state for discoverability~~ ✓ Done
3. ~~Write `tests/test_global_state.py` — import every module, verify no unexpected mutations~~ ✓ Done

---

## 3. P1: Scientific Validation Framework (Week 2-3)

### 3.1 Validation directory structure

Create `validation/` at repo root:

```
validation/
├── README.md                    # How to run, what each benchmark validates
├── run_all.sh                   # Single command to run full suite
├── benchmarks/
│   ├── 01_codon_translation/
│   │   ├── input.helix
│   │   ├── reference.json       # Expected codon→amino acid mapping
│   │   ├── run.py
│   │   └── GOLDEN.sha256
│   ├── 02_lac_operon/
│   ├── 03_ecoli_fba/
│   ├── 04_iml1515_fba/
│   ├── 05_in678_photoauto/
│   ├── 06_dfba_diauxic/
│   ├── 07_grn_repressilator/
│   ├── 08_population_dynamics/
│   ├── 09_reaction_diffusion/
│   ├── 10_whole_cell/
│   ├── 11_digital_evolution/
│   ├── 12_crispr_edit/
│   ├── 13_gem_reconstruction/
│   ├── 14_ecosystem_competition/
│   ├── 15_viral_phage/
│   ├── 16_human_diabetes/
│   ├── 17_human_cisplatin/
│   ├── 18_genotype_cyp2d6/
│   ├── 19_ddi_warfarin_amiodarone/
│   └── 20_end_to_end_pipeline/
├── results/                     # Generated after run
├── references/                  # External reference datasets
│   ├── iML1515_fluxes.json
│   ├── iJN678_growth.json
│   └── ...
└── report.md                    # Auto-generated comparison
```

### 3.2 Benchmark specification

Each benchmark must include:

```yaml
# benchmark.yaml
id: 03_ecoli_fba
name: E. coli FBA growth rate
layer: metabolism          # language | runtime | metabolism | grn | whole_cell | human
reference: Orth et al. 2010, E. coli iML1515
reference_doi: 10.1016/j.bpj.2010.12.3721
input: ecoli_model.xml
parameters:
  backend: fba
  seed: 42
  organism: ecoli
expected:
  metric: growth_rate
  value: 0.877
  tolerance: 0.05
  unit: h^-1
helixlang_result: null        # filled after run
error: null                   # filled after run
runtime_seconds: null         # filled after run
helix_version: null           # filled after run
```

### 3.3 Priority benchmarks (implement first)

| # | Benchmark | Validates | Reference | Expected |
|---|-----------|-----------|-----------|----------|
| 1 | Codon translation | Language semantics | Standard genetic code | 64→20 mapping |
| 2 | lac operon | GRN regulation |Jakob & Monod 1961 | Biphasic growth |
| 3 | E. coli FBA | Core metabolism | Orth 2010 | μ = 0.877 h⁻¹ |
| 4 | iML1515 | Genome-scale | Monk et al. 2017 | μ = 0.821 h⁻¹ |
| 5 | iJN678 photoauto | Photoautotrophy | Knoop 2010 | μ = 0.292 h⁻¹ |
| 6 | dFBA diauxic | Dynamic metabolism | Goncalves 2006 | Diauxic shift |
| 7 | Repressilator | Synthetic GRN | Elowitz 2000 | Oscillation τ ≈ 150 min |
| 8 | Whole-cell | Integrated physiology | Karr 2012 | Cell cycle ≈ 100 min |
| 9 | CRISPR edit | Gene editing | — | Knockout efficiency |
| 10 | Ecosystem competition | Multi-species | Lotka-Volterra | Competitive exclusion |

### 3.4 Result provenance schema

Every simulation result should carry:

```json
{
  "helix_version": "2026.8.4",
  "source_hash": "sha256:abc123...",
  "seed": 42,
  "backend": "fba",
  "parameters": {
    "organism": "ecoli",
    "model": "iML1515",
    "objective": "BIOMASS_Ecoli"
  },
  "dependencies": {
    "python": "3.11.15",
    "numpy": "2.3.4",
    "rdkit": "2026.03.5"
  },
  "timestamp": "2026-08-26T20:00:00Z",
  "runtime_seconds": 0.42
}
```

**Files to modify**:
- `src/helixlang/sim_runtime.py` — attach provenance to `SimulationResult`
- Add `src/helixlang/provenance.py` — helper to build provenance dict

---

## 4. P2: Product Identity & User Entry (Week 3-4)

### 4.1 Three-layer architecture declaration

Formalize the project structure in code and documentation:

```
Layer 1: Helix Language
├── lexer.py, parser.py, ast_nodes.py, semantic.py
├── compiler.py, bytecode.py, vm.py
├── type_system.py, flow.py
└── spec/bytecode-abi.md, spec/language-spec.md

Layer 2: Biological Runtime
├── cell.py, cell_body.py, central_dogma.py
├── grn.py, sparse_grn.py, metabolism.py
├── environment.py, population.py
├── codon_table.py, bio_data.py
└── spec/vm-semantics.md

Layer 3: Scientific Applications
├── human/          → virtual patient, pharmacology
├── gem/            → GEM reconstruction, FBA
├── apps/           → pipelines, synbio, consortium
├── kinetics/       → enzyme kinetics
├── omics/          → expression, spatial
└── ecosystem/      → (future, currently in apps/)
```

**Files to modify**:
- `src/helixlang/__init__.py` — add module docstring declaring layers
- `doc/00-overview.md` — restructure with Layer 1/2/3 framing

### 4.2 README rewrite: 5-minute proof

Replace the current "feature catalog" first screen with:

```markdown
# HelixLang

**A programming language for executable biological models.**

## What is HelixLang?

DNA is source code. HelixLang compiles it into executable biological simulations.

## 30-second install

```bash
pip install helixlang
```

## 60-second hello world

```helix
#gene lacI
ATG AAA TAT ACC GCT TCA CCG GAT AAA ACG ...
#end

#gene lacZ
ATG ACC ATG ATT ACG CCA AAG CAT AAA TAA ...
#end

#regulate lacI -> lacZ strength=-0.8

#config ticks=100
```

```bash
helix run examples/02_lac_operon.helix --json
```

Output: gene expression levels, protein concentrations, metabolic state.

## Architecture

```
HelixLang source (.helix)
        ↓
    Lexer → Parser → AST → Semantic → Compiler
        ↓
    Bytecode (.helixc)
        ↓
    VM → Biological State
        ↓
  ┌─────┼─────┐
  ↓     ↓     ↓
Cell  Genome  Ecosystem
  ↓     ↓     ↓
  └─────┼─────┘
        ↓
  Reproducible Result
```

## Benchmarks

22 reproducible biological benchmarks covering language semantics,
GRN regulation, metabolism, whole-cell physiology, and ecosystem dynamics.

[View benchmarks →](validation/)

## Documentation

[Language Spec](doc/02-language-spec.md) ·
[Compiler Design](doc/03-compiler-design.md) ·
[API Reference](doc/08-api-reference.md) ·
[Examples](examples/)
```

### 4.3 Killer demo selection

**Current demo**: Virtual Patient (too heavy, implies medical credibility)

**Recommended demo**: lac operon → phenotype

```helix
#gene lacI
ATG AAA TAT ACC GCT TCA CCG GAT AAA ACG GTG AAT GAA ACC GGT AAC CGG CGC ATT CAG CGC ACC ...
#end

#gene lacZ  
ATG ACC ATG ATT ACG CCA AAG CAT AAA TAA GCT TAC GCT ACG TTC AGC TTC ACC ACT AAA GGT GTA ...
#end

#regulate lacI -> lacZ strength=-0.8

#config backend=classic ticks=200
#media glucose=10 lactose=0

#run
```

Run → observe biphasic growth (glucose consumed first, then lactose via lac operon induction).

This demonstrates the full chain: **DNA → language → compiler → bytecode → VM → biological phenotype**.

---

## 5. P3: Module Hygiene (Week 4-5)

### 5.1 Modules to merge

| Current | Merge into | Reason |
|---------|-----------|--------|
| `protein_structure.py` + `protein_structure_predictor.py` | `protein_structure.py` | Redundant — predictor wraps structure |
| `stochastic.py` + `vectorized.py` | `stochastic.py` | Both are SSA variants |
| `hxbc.py` + `bytecode.py` | Keep separate but document relationship | Serialization vs. definition |

### 5.2 Modules to split

| Current | Split into | Reason |
|---------|-----------|--------|
| `virtual_patient.py` (2,372 LOC) | `virtual_patient.py` (core loop) + `patient_labs.py` (lab panels) + `patient_vitals.py` (vitals) | Too large, multiple responsibilities |
| `drug.py` (1,161 LOC) | `drug.py` (data model) + `drug_parser.py` (SMILES/RDKit) + `drug_adme.py` (ADME inference) | RDKit coupling should be isolated |
| `sim_runtime.py` | `sim_runtime.py` (orchestrator) + `backend_dispatch.py` (36 backends) | Dispatch logic is independent |

### 5.3 Modules to delete (or deprecate)

| Module | Reason |
|--------|--------|
| `server.py` `_DEBUG_SESSIONS` | Global mutable state; consider if web server is core or peripheral |
| `disassembler.py` | If bytecode ABI is frozen, disassembler should be a separate tool |

### 5.4 Files requiring immediate attention

| File | LOC | Risk | Action |
|------|-----|------|--------|
| `human/virtual_patient.py` | 2,372 | Highest complexity, 104 deps | Split into 3 modules |
| `human/drug.py` | 1,161 | RDKit coupling, SMILES edge cases | Isolate RDKit behind interface |
| `sim_runtime.py` | ~800 | 36 backend dispatch, implicit state | Extract dispatch table |
| `server.py` | ~300 | Mutable global `_DEBUG_SESSIONS` | Add lock or document single-thread |
| `vm.py` | ~600 | Core execution, determinism critical | Audit RNG, add determinism tests |
| `cell.py` | ~500 | Cell state, shared across backends | Audit mutation paths |

---

## 6. Specific Technical Debt Items

### 6.1 Determinism debt

| Location | Issue | Fix |
|----------|-------|-----|
| `vm.py` — any `random.randint` without seed | Non-deterministic | Route through seeded RNG |
| `evolution.py` — `random.random()` | Non-deterministic | Accept `rng` parameter |
| `stochastic.py` — module-level RNG | Global state | Accept `rng` parameter |
| `population.py` — `np.random` | Non-deterministic | Accept `rng` parameter |
| Any `time.time()` in simulation path | Wall-clock dependency | Replace with sim clock |

### 6.2 Testing debt

| Area | Current | Needed |
|------|---------|--------|
| Determinism tests | 4 (FBA, GRN, VM, stochastic) | 1 per backend (22+) |
| Golden output tests | 10 (benchmarks 01-10) | 1 per canonical example (59+) |
| Regression tests | Implicit | Explicit with pinned hashes |
| Edge case coverage | Partial | SMILES parsing, empty input, overflow |
| Concurrency tests | 0 | 1 for server.py global state |

### 6.3 Documentation debt

| Doc | Status | Action |
|-----|--------|--------|
| `doc/02-language-spec.md` | Descriptive | Upgrade to prescriptive ("shall") |
| `doc/11-helixc-binary-format.md` | Descriptive | Add byte-level format table |
| Bytecode ABI | Formal | `spec/bytecode-abi.md` created, v1 frozen |
| VM semantics | Formal | `spec/vm-semantics.md` created |
| Validation results | Generated | `validation/report.md` with 10 benchmarks |
| Release checklist | Created | `doc/35-release-checklist.md` |

---

## 7. Recommended 30-day Plan

### Week 1: Core stability audit
- [x] Add `OPCODE_VERSION = 1` to `bytecode.py`
- [x] Create `spec/bytecode-abi.md` with byte-level format
- [x] Create `spec/vm-semantics.md` with instruction semantics
- [x] Audit `vm.py` for unseeded RNG — all seeded via `random.Random(0)`
- [x] Audit `server.py` `_DEBUG_SESSIONS` for thread safety — `_get_debug_lock()` lazy-init
- [x] Add `tests/test_determinism_audit.py`
- [x] Add `--check-bytecode-version` CLI flag

### Week 2: Validation framework
- [x] Create `validation/` directory structure
- [x] Implement benchmarks 1-5 (codon, lac operon, E. coli FBA, iML1515, iJN678)
- [x] Add provenance schema to `SimResult`
- [x] Write `validation/run_all.sh`

### Week 3: Validation expansion + provenance
- [x] Implement benchmarks 6-10 (dFBA, repressilator, population, reaction-diffusion, whole-cell)
- [x] Add `src/helixlang/provenance.py`
- [x] Attach provenance to all simulation outputs
- [x] Generate first `validation/report.md`

### Week 4: Product identity
- [x] Rewrite README first screen (30s install → 60s demo → architecture → benchmarks)
- [ ] Move virtual patient / pharmacology / DNA storage to "Applications" section
- [x] Add Layer 1/2/3 declarations to `__init__.py` and `doc/00-overview.md`
- [x] Write `doc/35-release-checklist.md` for v0.1 release

---

## 8. Release Strategy

**Release criteria**:
1. Bytecode ABI frozen and documented
2. VM semantics documented and tested
3. 10+ benchmarks passing with golden outputs
4. Provenance attached to all simulation results
5. README rewritten with 5-minute proof
6. Zero `warn_unused_ignores` mypy errors in CI
7. All random sources audited and seeded

**Tagline**: "DNA is source code. HelixLang compiles it."

---

## 9. What NOT to do

- Do NOT add new simulation backends until benchmarks 1-10 pass
- Do NOT add new disease models until existing ones have validation data
- Do NOT add new pharmacology features until PBPK is validated against reference
- Do NOT add new example files until existing 59 have golden outputs
- Do NOT add new documentation until existing 34 docs are cross-referenced
- Do NOT add new test cases until determinism tests cover all backends

---

## 10. Success Metrics

| Metric | Current | 30-day target | 90-day target |
|--------|---------|---------------|---------------|
| Validated benchmarks | 10 (explicit, golden) | 10 (explicit, golden) | 22 (all backends) |
| Determinism tests | 1 per backend (4 backends) | 1 per backend | 3 per backend |
| Provenance coverage | 100% (SimResult) | 100% (core backends) | 100% |
| README first-screen time-to-understand | <2 min | <2 min | <1 min |
| Bytecode ABI version | v1 | v1 | v1 |
| External references | 5 (benchmark papers) | 5 | 10+ |
| `# type: ignore` comments | 0 (CI clean) | 0 | 0 |
| Module LOC (max single file) | 2,372 | <1,500 | <1,000 |
| Module-level mutable state documented | 100% | 100% | 100% |
| `--check-bytecode-version` CLI | ✓ | ✓ | ✓ |
