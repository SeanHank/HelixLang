# HelixLang Project Overview

> A domain-specific language (DSL) whose source is biological genetic material, whose intermediate representation is binary bytecode, and whose execution semantics simulate biological behavior/morphology/properties — along with its compiler.

---

## 1. Project Vision

HelixLang abstracts "biological genetic information processing" into a **programmable, compilable, executable** computational model:

- **Source code** is a DNA-style sequence (bases `A C G T`) accompanied by structured annotation blocks (promoters, genes, terminators, etc.).
- **Compiler** splits the DNA sequence into triplets (codons) and maps them through a codon table to binary bytecode (opcode + operand).
- **Virtual machine** is a "cell simulator": with a stack-based bytecode VM at its core, wired to a gene regulatory network (GRN), an L-system morphogenesis generator, and a reaction-diffusion morphogenesis field; it executes bytecode tick by tick and produces **observable biological behavior/morphology/properties**.

In one sentence: **DNA is the source, codons are the mnemonics, the ribosome is the VM, and the cell is the runtime.**

---

## 2. Design Goals

| Goal | Description |
|---|---|
| **Biological isomorphism** | Language primitives correspond one-to-one to biological entities: codons, amino acids, genes, promoters, operons, regulatory factors, proteins. |
| **Compilable** | Provides a complete Lex → Parse → AST → IR → Bytecode → VM pipeline, and the output can be disassembled. |
| **Executable** | Bytecode drives a cell simulator, outputting morphology (L-system/reaction-diffusion), behavior (move/signal), and properties (concentration/energy). |
| **Degeneracy as aliasing** | 64 codons map to ~30 logical opcodes; **the third-position degeneracy acts as an operand modifier / parameter gear**, isomorphic to biological degeneracy. |
| **Turing-complete** | Formally has jumps, loops, and procedure calls; theoretically guaranteed by Păun splicing systems + universal computation. |
| **Zero external dependencies** | The prototype is implemented with the pure Python standard library, making it easy to teach, embed, and extend. |

---

## 3. Overall Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                 HelixLang source program (.helix)                  │
│  #gene promoter=strong                                             │
│  ATG GCT GGT TAA    # DNA triplet stream + annotation blocks       │
│  #end                                                              │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ Lexer (split codons by 3 bases + annotation tokens)
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                                Token stream                                │
│  CODON(ATG) CODON(GCT) CODON(GGT) STOP(TAA) ANNOT(gene, promoter=strong)   │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │ Parser (group ORFs: START..STOP is one gene)
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                                AST                                 │
│  Program(                                                          │
│    Gene(name, promoter, ORF: [Codon, Codon, ...], terminator))     │
│  )                                                                 │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ Semantic analysis (reading frame, pairing, symbol table)
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│          Compiler: codon table decoding (codon → opcode)           │
│   ATG→OP_START  GCT→OP_BUILD(protein, kind=structural)             │
│   GGT→OP_BUILD(membrane)    TAA→OP_HALT                            │
└──────────────────────────────┬─────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│       Bytecode Chunk (opcode + constant pool + line numbers)       │
│   0x10 0x01 0x02 0x00 ...   (OP_START, OP_BUILD, OP_HALT, ...)     │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ Disassembler (debuggable)
                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  Cell Virtual Machine (tick-based simulator)                                         │
│   ┌────────────┐  ┌──────────────────┐  ┌─────────────┐  ┌───────────────────────┐   │
│   │  Ribosome  │  │       GRN        │  │  L-system  │  │  Reaction-Diffusion    │   │
│   │    (VM)    │←→│(regulatory graph)│→→│ (morphology)│  │ (Turing pattern field)│   │
│   └────────────┘  └──────────────────┘  └─────────────┘  └───────────────────────┘   │
│     bytecode exec  concentration→gene switch  growth iteration  pattern formation    │
└──────────────────────────────────────────────────────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  Observable output                                                                   │
│   • Cell state trajectory (protein concentration, energy, position)                  │
│   • Morphology (L-system string / reaction-diffusion PNG)                            │
│   • Behavior log (move / signal / divide)                                            │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Innovations

1. **Codons as mnemonics**: The 64 codons serve directly as assembly-level mnemonics, with semantics defined by the codon table. This is absent from other ALife languages (Tierra, Avida, NetLogo), which use raw binary or hand-crafted mnemonics.
2. **Degeneracy as aliasing + modifier bits**: Synonymous codons (e.g., all `GCN` encode Ala) map to aliases of the same opcode, and the third base acts as a parameter gear (0/1/2/3) — preserving biological fact while exploiting the 64 = 4 × 16 instruction space.
3. **Variable translation table = ISA variant**: The standard table, mitochondrial table, etc. can be switched as compilation parameters, achieving "the same DNA expressed differently in different cellular environments".
4. **Multimodal execution**: A single VM simultaneously drives the GRN (regulatory logic), L-system (topological morphology), and reaction-diffusion (continuous patterning) — bringing the "genotype → phenotype" mapping down to the bytecode layer.
5. **Zero-dependency pure Python**: The prototype can run as a single file, making it easy to embed in Jupyter teaching and bioinformatics pipelines.

---

## 5. Document Navigation

| Document | Content |
|---|---|
| [00-overview.md](./overview.md) | This document: project vision, architecture, navigation |
| [01-references.md](./references.md) | Academic literature review (DNA computing / ALife / compilers) |
| [02-language-spec.md](./language-spec.md) | Language spec: alphabet, syntax, codon table, data types |
| [03-compiler-design.md](./compiler-design.md) | Compiler pipeline, AST, bytecode, VM design |
| [04-simulation-model.md](./simulation-model.md) | Cell simulation: GRN, L-system, reaction-diffusion, tick loop |
| [05-prototype-plan.md](./prototype-plan.md) | Prototype implementation plan, verification cases, future extensions |
| [06-engineering-design.md](./engineering-design.md) | Engineering design: module decoupling, error handling, extension points |
| [07-bio-modules.md](./bio-modules.md) | Bio modules in detail: central dogma, metabolic FBA, protein structure, CRISPR, epigenetics, evolution |
| [08-api-reference.md](./api-reference.md) | API reference: core data classes, function signatures, parameter docs |
| [09-bio-instructions.md](./bio-instructions.md) | Bio instruction guide: .helix annotation syntax, bio operation usage |

---

## 6. Positioning vs. Existing Systems

| System | Source form | Output | HelixLang difference |
|---|---|---|---|
| Tierra / Avida | Numeric instructions (0/1 strings) | Digital organism populations | HelixLang uses real codon semantics + a morphology layer |
| NetLogo | Logo-like | Multi-agent | HelixLang is compiled + cell-level semantics |
| L-system | String rewriting | Plant morphology | HelixLang emits L-system as a growth opcode |
| Grammatical Evolution | BNF + chromosome | Arbitrary programs | HelixLang internalizes the GE idea as codon→opcode |
| CPPN-NEAT | Neural network weights | Morphology/texture | HelixLang uses GRN rather than NN for development |

HelixLang's uniqueness lies in **being the first to make "codon table → bytecode opcode" the core of its compiler**, and in unifying multiple ALife output modalities (GRN + L-system + reaction-diffusion) under a single bytecode VM.

---

## 7. Project Structure

HelixLang adopts a **16-core-module** architecture in three layers: the compiler toolchain (8 modules), the simulation runtime (4 modules), and the bio function modules (4 modules). It also includes data/tool modules and a CLI entry point.

```
HelixLang/
├── doc/                              # Design docs (00-09)
│   ├── 00-overview.md                # Project overview
│   ├── 01-references.md             # Literature review
│   ├── 02-language-spec.md           # Language spec
│   ├── 03-compiler-design.md         # Compiler design
│   ├── 04-simulation-model.md        # Simulation model
│   ├── 05-prototype-plan.md           # Prototype plan
│   ├── 06-engineering-design.md      # Engineering design
│   ├── 07-bio-modules.md             # Bio modules in detail
│   ├── 08-api-reference.md           # API reference
│   └── 09-bio-instructions.md        # Bio instruction guide
├── src/helixlang/                    # Compiler and runtime implementation
│   ├── __init__.py                   # Package exports
│   │
│   │  ===== Compiler toolchain (8 modules) =====
│   ├── codon_table.py                # (1) 64-codon → opcode mapping
│   ├── lexer.py                      # (2) DNA splitter + annotation tokens
│   ├── parser.py                     # (3) Recursive-descent parser → AST
│   ├── ast_nodes.py                  # (4) AST node definitions
│   ├── compiler.py                   # (5) AST → bytecode chunk
│   ├── bytecode.py                   # (6) Chunk / Op / constant pool
│   ├── disassembler.py               # (7) Bytecode disassembly
│   ├── vm.py                         # (8) Stack VM (ribosome)
│   │
│   │  ===== Simulation runtime (4 modules) =====
│   ├── cell.py                       # (9) Cell state + tick loop
│   ├── grn.py                        # (10) Gene regulatory network
│   ├── lsystem.py                    # (11) L-system morphogenesis
│   ├── reaction_diffusion.py         # (12) Gray-Scott reaction-diffusion
│   │
│   │  ===== Bio function modules (4 core modules) =====
│   ├── central_dogma.py              # (13) Central dogma: transcription + translation + degradation
│   ├── metabolism.py                 # (14) Metabolic network FBA: flux balance analysis
│   ├── protein_structure.py          # (15) Protein structure prediction: secondary structure + TM + disorder
│   ├── crispr.py                     # (16) CRISPR-Cas gene editing model
│   │
│   │  ===== Data / tools / entry points =====
│   ├── bio_data.py                   # Real biological data (codon frequencies, mutation rates, etc.)
│   ├── epigenetics.py                # Epigenetics: DNA methylation + histone modification
│   ├── evolution.py                  # Evolution engine: mutation + selection + drift
│   ├── population.py                 # Population dynamics
│   ├── morphology_3d.py              # 3D morphogenesis
│   ├── dna_codec.py                  # DNA codec (storage/watermark)
│   ├── biocodec.py                   # Bio codec utilities
│   ├── type_system.py                # Type system
│   ├── semantic.py                   # Semantic analysis
│   ├── errors.py                     # Exception hierarchy
│   ├── debugger.py                   # Debugger
│   ├── server.py                     # HTTP API server
│   ├── cli.py                        # CLI entry point
│   ├── apps/                         # Application layer
│   │   ├── dna_storage.py            # DNA storage app
│   │   └── synbio_designer.py        # Synthetic biology designer
│   └── web/                          # Web frontend
│       ├── index.html
│       └── labs.html
├── examples/                         # Example programs (.helix)
│   ├── 01_hello_dna.helix            # Minimal smoke test
│   ├── 02_lac_operon.helix           # lac operon GRN
│   ├── 03_plant_growth.helix         # L-system plant growth
│   ├── 04_turing_pattern.helix       # Reaction-diffusion Turing pattern
│   ├── 05_table_switch.helix         # Switchable translation table
│   ├── 06_crispr_edit.helix          # CRISPR gene editing
│   ├── 07_evolution.helix            # Evolution simulation
│   └── 08_epigenetics.helix          # Epigenetic regulation
└── tests/                            # Verification tests (~30 test files)
    ├── test_codon_table.py
    ├── test_lexer.py
    ├── test_parser.py
    ├── test_compiler.py
    ├── test_vm.py
    ├── test_cell.py
    ├── test_grn.py
    ├── test_central_dogma.py
    ├── test_metabolism.py
    ├── test_protein_structure.py
    ├── test_crispr.py
    ├── test_epigenetics.py
    ├── test_evolution.py
    └── ...
```
