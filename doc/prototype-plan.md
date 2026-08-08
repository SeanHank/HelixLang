# HelixLang Prototype Implementation Plan and Validation

> This document lists the specific milestones, validation cases, test matrix, and future expansion roadmap for the prototype implementation. The prototype goal is to **run four example categories end to end**: Hello DNA / lac operon / plant growth / Turing pattern.

---

## 1. Implementation Milestones

| Milestone | Content | Acceptance |
|---|---|---|
| **M1 Bytecode + VM skeleton** | `bytecode.py` + `vm.py` + `cell.py`, hand-construct chunk to run OP_START/OP_BUILD_PROTEIN/OP_HALT | console prints "protein synthesized" |
| **M2 Codon table + Compiler** | `codon_table.py` (standard table) + `compiler.py`, generate chunk from codon list | disassembler output matches the source |
| **M3 Lexer + Parser** | `lexer.py` + `parser.py` + `semantic.py`, build AST from `.helix` file | parse example 01 without errors |
| **M4 Disassembler** | `disassembler.py` full disassembly with codon→opcode annotations | output format per doc 03 |
| **M5 GRN integration** | `grn.py` + `vm.call_gene` dispatch, run toggle switch | toggle switch bistability confirmed |
| **M6 L-system** | `lsystem.py` + `OP_GROW_LSYSTEM` | example 03 generates plant topology |
| **M7 Reaction-diffusion** | `reaction_diffusion.py` + `OP_REACT`/`OP_DIFFUSE`/`OP_EMIT_MORPHOGEN` | example 04 generates spots/stripes |
| **M8 CLI + tests** | `cli.py` + `tests/test_*.py` end to end | `pytest` all green |

---

## 2. Validation Cases

### 2.1 Example 1: Hello DNA (minimal smoke test)

**Goal**: validate the full Lex→Parse→Compile→VM pipeline.

```
#gene name=hello
ATG GCT GGT TAA
#end
#config ticks=1 output=stdout
```

**Expected**:
- Disassembler outputs `OP_START OP_BUILD_PROTEIN(0) OP_BUILD_MEMBRANE(1) OP_HALT`
- After VM execution `cell.proteins = {0: 1.0, 1: 1.0}`
- stdout: `tick=0 protein[0]=1.0 protein[1]=1.0 energy=100`

### 2.2 Example 2: lac operon (GRN validation)

**Goal**: validate GRN sigmoid dispatch and promoter thresholds.

```
#promoter name=p_lac strength=0.5
#gene name=lacZ
ATG GCT GCT GCT TAA
#end
#gene name=lacY
ATG GGT GGT GGT TAA
#end
#gene name=lacI
ATG GCT GCT TAA
#end
#regulate lacI -> p_lac strength=-0.8
#regulate p_lac -> lacZ strength=+0.9
#regulate p_lac -> lacY strength=+0.9
#config ticks=20 output=csv
```

**Expected**:
- Initially `lacI` is highly expressed → represses `p_lac` → `lacZ/lacY` weakly expressed
- Simulated addition of inducer (set `lacI.level=0`) → `p_lac` derepressed → `lacZ/lacY` rise
- The CSV trajectory shows a clear "induction-expression" switch curve

### 2.3 Example 3: Plant growth (L-system)

**Goal**: validate L-system morphogenesis.

```
#gene name=grow
ATG CTC TAA
#end
#promoter name=p_grow strength=0.3
#regulate p_grow -> grow strength=+0.6
#config ticks=5 output=png,stdout
#lsystem axiom=F rules=0:F->F[+F]F[-F]F angle=25
```

**Expected**:
- Each tick the `grow` gene is triggered once → `OP_GROW_LSYSTEM rules=0` iterates once
- After 5 ticks the L-system string has iterated 5 times, turtle interpretation yields a branching tree
- PNG output shows a plant-like topology

### 2.4 Example 4: Turing pattern (reaction-diffusion)

**Goal**: validate Gray-Scott reaction-diffusion.

```
#gene name=pigment
ATG GAT GAT GAA TAA
#end
#promoter name=p_pigment strength=0.4
#regulate p_pigment -> pigment strength=+0.5
#config ticks=200 output=png react_steps=2
#field size=32 F=0.035 k=0.065
```

**Expected**:
- Each tick the `pigment` gene is triggered → 2 `OP_REACT` steps
- After 200 ticks field `V` shows the classic Pearson 1993 spot (mitosis) pattern
- PNG output shows a spotted Turing pattern

### 2.5 Example 5: Variable translation table (ISA switching)

**Goal**: validate `--table` switching.

```
#gene name=morpheus
ATG TGA TAA
#end
#config ticks=1
```

**Expected**:
- `--table=standard`: TGA=OP_HALT → ORF terminates immediately, only OP_START then HALT
- `--table=mito_vertebrate`: TGA=OP_BUILD_PIGMENT → ORF continues until TAA and produces pigment

---

## 3. Test Matrix

### 3.1 Unit tests (`tests/`)

| File | Coverage |
|---|---|
| `test_lexer.py` | DNA splitting, annotation tokens, line numbers, codon indices, case sensitivity |
| `test_parser.py` | ORF recognition, annotation block nesting, error recovery |
| `test_codon_table.py` | 64 codon mapping, variable table switching, third-base wobble position |
| `test_compiler.py` | codon→opcode, constant pool, HALT completion |
| `test_disassembler.py` | disassembly format, codon annotations |
| `test_vm.py` | stack operations, frame calls, tick quota |
| `test_grn.py` | sigmoid update, toggle switch bistability, decay |
| `test_lsystem.py` | axiom iteration, turtle interpretation, branch stack |
| `test_reaction_diffusion.py` | Gray-Scott single step, energy conservation check |

### 3.2 End-to-end tests

| File | Cases |
|---|---|
| `test_end_to_end.py` | runs all 4 examples + translation table switching, asserts key invariants |

### 3.3 Invariant assertions

- **Energy conservation**: `Σ(cell.energy) + Σ(action energy costs) == initial energy` (when no feed).
- **Protein mass conservation**: `Σ(proteins) - Σ(degraded) == Σ(synthesized) - Σ(consumed)`.
- **GRN convergence**: the toggle switch stabilizes to (1,0) or (0,1) within 20 ticks with no external input.
- **L-system length**: the string length after the n-th iteration satisfies the rewriting-system recurrence.
- **Reaction-diffusion stability**: total U+V decreases when there are no source terms (consumed by the k term).

---

## 4. Performance Baseline

Prototype target (single-core Python 3.13):

| Workload | Target |
|---|---|
| Lex + Parse 1 KB `.helix` | < 50 ms |
| Compile 100 genes | < 100 ms |
| VM 1000 ticks single cell + GRN | < 1 s |
| 32×32 reaction-diffusion 200 steps | < 5 s |
| L-system 7 iterations (≤10k chars) | < 200 ms |

If not met, move to the optimization phase (see 5.1).

---

## 5. Expansion Roadmap

### 5.1 Short term (performance)

- Port reaction-diffusion to numpy (vectorized Laplacian)
- Replace VM dispatch `match/case` with `dict[int, callable]` and micro-benchmark the comparison
- Accumulate L-system turtle point sequences with numpy arrays

### 5.2 Mid term (language capability)

- **Control-flow instructions**: `OP_JUMP` / `OP_JUMP_IF_ZERO` synthesized by the compiler (the codon table uses `OP_NOP`-series placeholders)
- **Multicellular**: `OP_DIVIDE` creates daughter cells, VM maintains `list[Cell]`, shared signal field
- **Evolution frontend**: apply point mutations/recombination to `.helix` genomes, run genetic algorithms to optimize target phenotypes

### 5.3 Long term (compilation backend)

- **MLIR dialect**: define the four-layer dialect `helix.dna` / `helix.gene` / `helix.morph` / `helix.sim`, lower to LLVM
- **Physical DNA output**: with `target=synth_dna`, invoke the Church/Goldman/Erlich encoders to generate synthesizable oligo sequences
- **CRISPR in-vivo writing**: with `target=in_vivo_crispr`, compile to CRISPR-Cas guide RNA design

### 5.4 Toolchain

- **Lark migration**: when grammar complexity rises (nested annotations, expressions), move the Lexer/Parser to Lark (LALR mode)
- **tree-sitter integration**: provide syntax highlighting and incremental parsing for IDEs
- **Jupyter kernel**: `%helix_run` magic command to run simulations directly in notebooks

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| 64 codons are insufficient to cover all desired semantics | expand 4× via the third-base wobble position, introduce `OP_NOP` extension slots if needed |
| GRN dispatch deadlock/starvation | tick quota + fair round-robin + decay ensure all genes get a chance |
| Reaction-diffusion numerical explosion | clamp U,V to [0,1] each step; use sub-step dt=1.0 |
| L-system string exponential growth | limit the maximum number of iterations and string length |
| Bytecode offset patch ordering issues | two-pass compilation: first emit placeholders, then backfill `OP_CALL_GENE` offsets |

---

## 7. Acceptance Checklist (Definition of Done)

- [ ] `python -m helixlang examples/01_hello_dna.helix` outputs protein synthesis log
- [ ] `python -m helixlang examples/02_lac_operon.helix --csv > trace.csv` generates a trajectory
- [ ] `python -m helixlang examples/03_plant_growth.helix --png` generates a morphology PNG
- [ ] `python -m helixlang examples/04_turing_pattern.helix --png` generates a Turing pattern
- [ ] `python -m helixlang examples/05_table_switch.helix --table=mito_vertebrate` behaves differently
- [ ] `python -m helixlang --disassemble examples/01_hello_dna.helix` outputs disassembly
- [ ] `pytest tests/` all green
- [ ] Design docs and implementation are consistent (disassembly format, opcode encoding, codon tables)

---

## 8. Implementation Status

The prototype implementation is complete, including the following files (see `src/helixlang/`):

- `codon_table.py` — three translation tables (standard/mito_vertebrate/ciliate) + WobbleBits
- `lexer.py` — dual-mode scanner
- `ast_nodes.py` — AST dataclasses
- `parser.py` — recursive-descent Parser
- `bytecode.py` — Opcode constants + Chunk
- `compiler.py` — Program → Chunk
- `disassembler.py` — disassembly
- `grn.py` — sigmoid GRN
- `lsystem.py` — L-system + turtle
- `reaction_diffusion.py` — Gray-Scott
- `cell.py` — Cell state
- `vm.py` — CellVM
- `cli.py` — command-line entry point

Test coverage (see `tests/`):
- Unit: lexer / parser / codon_table / compiler / disassembler / vm / grn / lsystem / reaction_diffusion
- End-to-end: 4 examples + translation table switching

See the project-root README for detailed usage (run `python -m helixlang --help`).
